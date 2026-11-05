"""Headless workspace model: immutable config snapshots, history, anchors."""
from __future__ import annotations

import copy
import json
import math
from typing import Any

from util.config_schema import ValidationIssue, validate_config

_HISTORY_LIMIT = 100


def _deep(v: Any) -> Any:
    return copy.deepcopy(v)


def _nonfinite_or_unsupported(v: Any) -> bool:
    if isinstance(v, float):
        return math.isnan(v) or math.isinf(v)
    if isinstance(v, dict):
        return any(_nonfinite_or_unsupported(x) for x in v.values())
    if isinstance(v, (list, tuple)):
        return any(_nonfinite_or_unsupported(x) for x in v)
    if isinstance(v, (str, int, bool)) or v is None:
        return False
    return True  # non-JSON-serializable type


def _draft_type_issues(base: Any, current: Any, path: str = "") -> list[ValidationIssue]:
    issues = []
    if isinstance(base, dict):
        if not isinstance(current, dict):
            issues.append(ValidationIssue(path, "expected a mapping; repair the retained YAML draft"))
        else:
            for key in base.keys() & current.keys():
                issues.extend(_draft_type_issues(base[key], current[key], f"{path}.{key}" if path else str(key)))
    elif isinstance(base, bool) and not isinstance(current, bool):
        issues.append(ValidationIssue(path, "expected a boolean, not raw invalid editor text"))
    elif isinstance(base, (int, float)) and not isinstance(base, bool):
        if isinstance(current, bool) or not isinstance(current, (int, float)):
            issues.append(ValidationIssue(path, "expected a number, not raw invalid editor text"))
    elif isinstance(base, (list, tuple)):
        if not isinstance(current, (list, tuple)):
            issues.append(ValidationIssue(path, "expected a list; repair the retained YAML draft"))
        else:
            for index, (old, new) in enumerate(zip(base, current)):
                issues.extend(_draft_type_issues(old, new, f"{path}[{index}]"))
    return issues


def _finite_coefficient(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


class WorkspaceModel:
    def __init__(self, config: dict, preset_path: str | None = None,
                 session_id: str = "") -> None:
        self._cfg = _deep(dict(config))
        self._preset_path = preset_path
        self._preset_snapshot = _deep(dict(config))
        self._saved_snapshot = _deep(dict(config))
        self._source: dict | None = None
        self._saved_source: dict | None = None
        self._revision = 0
        self._generation = 0
        self._undo: list[dict] = []
        self._redo: list[dict] = []
        self._session_id = session_id
        self._edit_group: str | None = None
        self._group_recorded = False

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def config(self) -> dict:
        return _deep(self._cfg)

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def source(self) -> dict | None:
        return _deep(self._source)

    @property
    def preset_path(self) -> str | None:
        return self._preset_path

    @property
    def preset_snapshot(self) -> dict:
        return _deep(self._preset_snapshot)

    @property
    def saved_source(self) -> dict | None:
        return _deep(self._saved_source)

    @property
    def saved_snapshot(self) -> dict:
        return _deep(self._saved_snapshot)

    @property
    def is_dirty(self) -> bool:
        return (self._cfg != self._saved_snapshot
                or self._source != self._saved_source)

    @property
    def modified_paths(self) -> set[str]:
        return _diff_paths(self._preset_snapshot, self._cfg)

    def _snapshot(self) -> dict:
        return {
            "config": _deep(self._cfg),
            "preset_path": self._preset_path,
            "preset_snapshot": _deep(self._preset_snapshot),
            "source": _deep(self._source),
        }

    def _push_undo(self, label: str) -> None:
        if self._edit_group is not None:
            if self._group_recorded:
                return
            self._group_recorded = True
        self._undo.append({"state": self._snapshot(), "label": label})
        if len(self._undo) > _HISTORY_LIMIT:
            self._undo.pop(0)
        self._redo.clear()

    def begin_edit_group(self, key: str) -> None:
        if self._edit_group != key:
            self._edit_group = key
            self._group_recorded = False

    def end_edit_group(self, key: str | None = None) -> None:
        if key is None or key == self._edit_group:
            self._edit_group = None
            self._group_recorded = False

    def edit_many(self, values: dict[str, Any], label: str = "Edit fields") -> None:
        candidate = self.config
        for path, value in values.items():
            node = candidate
            parts = path.split(".")
            for part in parts[:-1]:
                if part not in node:
                    node[part] = {}
                if not isinstance(node[part], dict):
                    raise ValueError(f"{path}: parent {part} is not a mapping")
                node = node[part]
            node[parts[-1]] = _deep(value)
        if candidate != self._cfg:
            self.replace_config(candidate, label=label)

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self._cfg
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return _deep(node)

    def edit(self, path: str, value: Any, label: str = "Edit") -> None:
        try:
            current = self.get(path, _MISSING)
        except KeyError:
            current = _MISSING
        if current is not _MISSING and current == value:
            return  # no-op equal edit: no history, no revision
        self._push_undo(label)
        node = self._cfg
        parts = path.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _deep(value)
        self._revision += 1

    def reset_parameter(self, path: str, label: str = "Reset") -> None:
        if path not in _flat(self._preset_snapshot):
            self._push_undo(label)
            node = self._cfg
            parts = path.split(".")
            for part in parts[:-1]:
                node = node.get(part, {})
            if isinstance(node, dict):
                node.pop(parts[-1], None)
            self._revision += 1
            return
        base = self._preset_snapshot
        for part in path.split("."):
            base = base[part]
        self.edit(path, base, label)

    def reset_module(self, section: str, label: str = "Reset module") -> None:
        if section not in self._preset_snapshot:
            return
        self.edit(section, self._preset_snapshot[section], label)

    def reset_all(self, label: str = "Reset all") -> None:
        self.replace_config(self._preset_snapshot, label=label)

    def replace_config(self, config: dict, preset_path: str | None = None,
                       as_preset: bool = False, label: str = "Replace") -> None:
        self.end_edit_group()
        self._push_undo(label)
        self._cfg = _deep(dict(config))
        if preset_path is not None:
            self._preset_path = preset_path
            self._preset_snapshot = _deep(dict(config))
        elif as_preset:
            self._preset_snapshot = _deep(dict(config))
        self._revision += 1

    def _apply_state(self, state: dict) -> None:
        src_changed = state["source"] != self._source
        self._cfg = _deep(state["config"])
        self._preset_path = state["preset_path"]
        self._preset_snapshot = _deep(state["preset_snapshot"])
        self._source = _deep(state["source"])
        self._revision += 1
        if src_changed:
            self._generation += 1

    def undo(self) -> bool:
        self.end_edit_group()
        if not self._undo:
            return False
        item = self._undo.pop()
        self._redo.append({"state": self._snapshot(), "label": item["label"]})
        self._apply_state(item["state"])
        return True

    def redo(self) -> bool:
        self.end_edit_group()
        if not self._redo:
            return False
        item = self._redo.pop()
        self._undo.append({"state": self._snapshot(), "label": item["label"]})
        self._apply_state(item["state"])
        return True

    def mark_saved(self) -> None:
        self.end_edit_group()
        self._saved_snapshot = _deep(self._cfg)
        self._saved_source = _deep(self._source)

    def adopt_source(self, record: dict, overrides: dict | None = None,
                     label: str = "Import") -> None:
        self.end_edit_group()
        self._push_undo(label)
        if overrides and isinstance(self._cfg.get("sensor_info"), dict):
            self._cfg["sensor_info"].update(_deep(overrides))
        self._source = _deep(record)
        self._generation += 1
        self._revision += 1

    def restore_saved_state(self, *, config: dict, preset_snapshot: dict,
                            preset_path: str | None, saved_snapshot: dict,
                            source: dict | None, saved_source: dict | None,
                            generation: int | None = None,
                            revision: int | None = None,
                            session_id: str | None = None) -> None:
        self.end_edit_group()
        self._cfg = _deep(dict(config))
        self._preset_snapshot = _deep(dict(preset_snapshot))
        self._preset_path = preset_path
        self._saved_snapshot = _deep(dict(saved_snapshot))
        self._source = _deep(source)
        self._saved_source = _deep(saved_source)
        if session_id is not None:
            self._session_id = session_id
        if generation is not None:
            self._generation = max(self._generation + 1, generation + 1)
        else:
            self._generation += 1
        if revision is not None:
            self._revision = max(self._revision + 1, revision + 1)
        else:
            self._revision += 1
        self._undo.clear()
        self._redo.clear()

    def new_session(self, config: dict, session_id: str,
                    preset_path: str | None = None) -> None:
        self.end_edit_group()
        self._cfg = _deep(dict(config))
        self._preset_path = preset_path
        self._preset_snapshot = _deep(dict(config))
        self._saved_snapshot = _deep(dict(config))
        self._saved_source = None
        self._undo.clear()
        self._redo.clear()
        self._generation += 1
        self._revision += 1
        self._source = None
        self._session_id = session_id

    def set_source(self, record: dict) -> None:
        self._source = _deep(record)
        self._generation += 1
        self._revision += 1

    def validation_issues(self) -> list:
        try:
            issues = validate_config(self._cfg)
        except Exception as exc:
            return [ValidationIssue(path="config", message=str(exc),
                                    severity="error")]
        reported = {i.path for i in issues if i.severity == "error"}
        for issue in _draft_type_issues(self._preset_snapshot, self._cfg):
            if issue.path not in reported:
                issues.append(issue)
                reported.add(issue.path)
        ccm = self._cfg.get("color_correction_matrix")
        if isinstance(ccm, dict):
            for name in ("corrected_red", "corrected_green", "corrected_blue"):
                if name not in ccm:
                    continue
                values = ccm[name]
                path = f"color_correction_matrix.{name}"
                valid = isinstance(values, (list, tuple)) and len(values) == 3 and all(
                    _finite_coefficient(v) for v in values)
                if not valid and path not in reported:
                    issues.append(ValidationIssue(path, "CCM rows require three finite numeric coefficients"))
        if _nonfinite_or_unsupported(self._cfg):
            issues = list(issues)
            issues.append(ValidationIssue(
                path="config",
                message="non-finite or non-JSON-serializable value",
                severity="error"))
        return issues

    def can_process(self) -> bool:
        return not any(i.severity == "error" for i in self.validation_issues())

    def config_json_ready(self) -> bool:
        if _nonfinite_or_unsupported(self._cfg):
            return False
        try:
            json.dumps(self._cfg, allow_nan=False)
        except (TypeError, ValueError):
            return False
        return self.can_process()


_MISSING = object()


def _flat(d: Any, path: str = "") -> set[str]:
    out: set[str] = set()
    if isinstance(d, dict):
        for k, v in d.items():
            p = f"{path}.{k}" if path else k
            out.add(p)
            out |= _flat(v, p)
    return out


def _diff_paths(base: dict | None, cur: dict) -> set[str]:
    out: set[str] = set()
    if base is None:
        return set(cur)
    def walk(b: Any, c: Any, path: str) -> None:
        if isinstance(b, dict) and isinstance(c, dict):
            for k in b.keys() | c.keys():
                walk(b.get(k), c.get(k), f"{path}.{k}" if path else k)
        elif b != c:
            out.add(path)
    walk(base, cur, "")
    return out
