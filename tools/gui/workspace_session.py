"""Workspace session bundles: atomic workspace.json save/load."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

from tools.gui.session import read_last_session_id
from util.config_merge import load_merged_yaml, pipeline_config_paths
from util.config_schema import normalize_config

BUNDLE_NAME = "workspace.json"
BUNDLE_VERSION = 1
_LAST_POINTER = "last_session.json"


class WorkspaceSession:
    def __init__(self, directory: Path, session_id: str) -> None:
        self.directory = Path(directory)
        self.session_id = session_id

    @property
    def workspace_path(self) -> Path:
        return self.directory / BUNDLE_NAME

    @classmethod
    def create(cls, root: Path, config: dict,
               preset_path: str | None = None) -> tuple["WorkspaceSession", Any]:
        from .model import WorkspaceModel
        model = WorkspaceModel(config, preset_path=preset_path,
                               session_id="")
        if not model.config_json_ready():
            raise ValueError("config is not JSON-serializable or invalid")
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        directory = root / uuid.uuid4().hex
        directory.mkdir()
        session = cls(directory, directory.name)
        model._session_id = session.session_id
        session._write_bundle(
            {"version": BUNDLE_VERSION,
             "config": model.config,
             "preset_snapshot": model.preset_snapshot,
             "preset_path": preset_path,
             "saved_snapshot": model.config,
             "source": None,
             "saved_source": None,
             "ui_state": {},
             "snapshots": {}})
        model.mark_saved()
        return session, model

    @classmethod
    def open(cls, directory: Path) -> tuple["WorkspaceSession", Any, dict, dict, list[str]]:
        from .model import WorkspaceModel
        directory = Path(directory).resolve()
        session = cls(directory, directory.name)
        warnings: list[str] = []
        bundle_path = directory / BUNDLE_NAME
        if bundle_path.is_file():
            payload = _read_bundle(bundle_path)
            model, ui_state, snaps = _model_from_payload(
                payload, directory, warnings)
            return session, model, ui_state, snaps, warnings
        if (directory / "config.yml").is_file():
            model, ui_state, snaps = _model_from_legacy(directory, warnings)
            return session, model, ui_state, snaps, warnings
        if (directory / "session.json").is_file():
            raise ValueError(
                f"{directory}: legacy session.json found without config.yml")
        raise ValueError(f"{directory}: no workspace.json or legacy session files")

    def _write_bundle(self, payload: dict) -> None:
        text = json.dumps(payload, allow_nan=False)
        fd, tmp = tempfile.mkstemp(dir=str(self.directory),
                                   prefix=".workspace-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.workspace_path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _rel_source(self, record: dict | None, field: str) -> dict | None:
        if record is None:
            return None
        if not isinstance(record, dict):
            raise ValueError(f"{field}: source record must be an object")
        path = record.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError(f"{field}: source path must be a nonempty string")
        resolved = Path(path).resolve()
        if not _within(resolved, self.directory.resolve()):
            raise ValueError(f"{field}: source path escapes session dir")
        out = dict(record)
        out["path"] = str(resolved.relative_to(self.directory.resolve()))
        vm = record.get("view_manifest")
        if vm is not None:
            if not isinstance(vm, str) or not vm:
                raise ValueError(f"{field}: view_manifest malformed")
            rv = Path(vm).resolve()
            if not _within(rv, self.directory.resolve()):
                raise ValueError(f"{field}: view_manifest escapes session dir")
            out["view_manifest"] = str(
                rv.relative_to(self.directory.resolve()))
        return out

    def save(self, model, ui_state: dict, snapshots: dict) -> None:
        if not model.config_json_ready():
            raise ValueError("config is not JSON-serializable or invalid")
        if not isinstance(ui_state, dict):
            raise ValueError("ui_state must be an object")
        if not isinstance(snapshots, dict):
            raise ValueError("snapshots must be an object")
        snaps = _relativize_snapshots(snapshots, self.directory)
        payload = {
            "version": BUNDLE_VERSION,
            "config": model.config,
            "preset_snapshot": model.preset_snapshot,
            "preset_path": model.preset_path,
            "saved_snapshot": model.config,
            "source": self._rel_source(model.source, "source"),
            "saved_source": self._rel_source(
                model.source, "saved_source"),
            "ui_state": dict(ui_state),
            "snapshots": snaps,
        }
        self._write_bundle(payload)
        model.mark_saved()

    def save_last_pointer(self, root: Path) -> None:
        _write_last_pointer(Path(root), self.session_id)


def _write_last_pointer(root: Path, session_id: str) -> None:
    if not session_id or Path(session_id).name != session_id:
        raise ValueError("session id must be a nonempty basename")
    root.mkdir(parents=True, exist_ok=True)
    text = json.dumps({"session_id": session_id}, allow_nan=False)
    fd, tmp = tempfile.mkstemp(dir=str(root), prefix=".last-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, root / _LAST_POINTER)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_last_workspace(root: Path) -> Path | None:
    root = Path(root)
    pointer = root / _LAST_POINTER
    if not pointer.is_file():
        return None
    try:
        data = json.loads(pointer.read_text(encoding="utf-8"),
                          parse_constant=_reject_const)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{_LAST_POINTER}: unreadable: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{_LAST_POINTER}: must be an object")
    sid = data.get("session_id")
    if (not isinstance(sid, str) or not sid
            or Path(sid).name != sid or sid.startswith(".")):
        raise ValueError(f"{_LAST_POINTER}: invalid session id")
    directory = (root / sid).resolve()
    if not _within(directory, root.resolve()):
        raise ValueError(f"{_LAST_POINTER}: session id escapes session root")
    if not ((directory / BUNDLE_NAME).is_file()
            or (directory / "config.yml").is_file()):
        raise ValueError(f"{directory}: last session has no usable bundle")
    return directory


def _reject_const(value: str) -> Any:
    raise ValueError(f"non-finite literal {value}")


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
        return True
    except ValueError:
        return False


def _read_bundle(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"),
                             parse_constant=_reject_const)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path.name}: unreadable bundle: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name}: bundle must be an object")
    if payload.get("version") is not BUNDLE_VERSION:
        raise ValueError(
            f"{path.name}: unsupported bundle version {payload.get('version')!r}")
    for key in ("config", "preset_snapshot", "saved_snapshot"):
        if not isinstance(payload.get(key), dict):
            raise ValueError(f"{path.name}: missing object field {key!r}")
    for key in ("ui_state", "snapshots"):
        if key in payload and not isinstance(payload[key], dict):
            raise ValueError(f"{path.name}: field {key!r} must be an object")
    if "preset_path" in payload and payload["preset_path"] is not None \
            and not isinstance(payload["preset_path"], str):
        raise ValueError(f"{path.name}: preset_path must be a string or null")
    for key in ("source", "saved_source"):
        if key in payload and payload[key] is not None \
                and not isinstance(payload[key], dict):
            raise ValueError(f"{path.name}: {key!r} must be an object or null")
    return payload


def _resolve_within(directory: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field}: path must be a nonempty string")
    p = (directory / value).resolve()
    if not _within(p, directory):
        raise ValueError(f"{field}: path escapes session dir: {value!r}")
    return p


def _resolve_source(record: dict | None, directory: Path,
                    warnings: list[str], field: str) -> dict | None:
    if record is None:
        return None
    resolved = _resolve_within(directory, record.get("path"), f"{field}.path")
    out = dict(record)
    out["path"] = str(resolved)
    if not resolved.is_file():
        out["missing"] = True
        warnings.append(f"{field}: staged source missing; choose a new source")
    vm = record.get("view_manifest")
    if vm is not None:
        rv = _resolve_within(directory, vm, f"{field}.view_manifest")
        out["view_manifest"] = str(rv)
        if not (rv / "result.json").is_file():
            out["view_manifest_missing"] = True
            warnings.append(f"{field}: view manifest missing result.json")
    return out


def _model_from_payload(payload: dict, directory: Path,
                        warnings: list[str]) -> tuple[Any, dict, dict]:
    from .model import WorkspaceModel
    model = WorkspaceModel(payload["config"],
                           preset_path=payload.get("preset_path"),
                           session_id=directory.name)
    source = _resolve_source(payload.get("source"), directory, warnings,
                             "source")
    saved_source = _resolve_source(payload.get("saved_source"), directory,
                                   warnings, "saved_source")
    snapshots = payload.get("snapshots") or {}
    resolved_snaps: dict[str, Any] = {}
    for name, entry in snapshots.items():
        if entry is None:
            resolved_snaps[name] = None
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"snapshots.{name}: must be an object or null")
        if entry.get("manifest") is not None:
            resolved = _resolve_within(directory, entry["manifest"],
                                       f"snapshots.{name}.manifest")
            entry = dict(entry)
            entry["manifest"] = str(resolved)
            if not (resolved / "result.json").is_file():
                entry["missing"] = True
                warnings.append(f"snapshot {name}: result.json missing")
        resolved_snaps[name] = entry
    model.restore_saved_state(
        config=payload["config"],
        preset_snapshot=payload["preset_snapshot"],
        preset_path=payload.get("preset_path"),
        saved_snapshot=payload["saved_snapshot"],
        source=source, saved_source=saved_source)
    ui_state = payload.get("ui_state")
    if ui_state is None:
        ui_state = {}
    return model, ui_state, resolved_snaps


def _relativize_snapshots(snapshots: dict, directory: Path) -> dict:
    out: dict[str, Any] = {}
    for name, entry in snapshots.items():
        if entry is None:
            out[name] = None
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"snapshots.{name}: must be an object or null")
        if entry.get("manifest") is not None:
            if not isinstance(entry["manifest"], str) or not entry["manifest"]:
                raise ValueError(f"snapshots.{name}.manifest malformed")
            p = Path(entry["manifest"]).resolve()
            if not _within(p, directory.resolve()):
                raise ValueError(
                    f"snapshots.{name}.manifest escapes session dir")
            entry = dict(entry)
            entry["manifest"] = str(
                p.relative_to(directory.resolve()))
        out[name] = entry
    return out


def _fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_input_filename(value: Any) -> str | None:
    if value is None:
        return "input.raw"
    if not isinstance(value, str) or not value:
        raise ValueError("legacy input_filename must be a nonempty basename")
    if "\\" in value or "/" in value or value.startswith("."):
        raise ValueError("legacy input_filename must be a basename")
    return value


def _model_from_legacy(directory: Path, warnings: list[str]) -> tuple[Any, dict, dict]:
    from .model import WorkspaceModel
    cfg_path = directory / "config.yml"
    if not cfg_path.is_file():
        raise ValueError(f"{directory}: legacy session missing config.yml")
    config, _notices = normalize_config(load_merged_yaml([cfg_path]))
    state: dict = {}
    state_path = directory / "session.json"
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"),
                               parse_constant=_reject_const)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"legacy session.json unreadable: {exc}") from exc
        if not isinstance(state, dict):
            raise ValueError("legacy session.json must be an object")
    input_name = _validate_input_filename(state.get("input_filename"))
    source = None
    staged = (directory / input_name).resolve() if input_name else None
    if staged is not None:
        if not _within(staged, directory):
            raise ValueError("legacy source escapes session dir")
        original = state.get("original_source_path")
        if original is not None and not isinstance(original, str):
            raise ValueError("original_source_path malformed")
        if staged.is_file():
            source = {"path": str(staged),
                      "original_path": original or input_name,
                      "fingerprint": _fingerprint(staged),
                      "size": staged.stat().st_size,
                      "format_overrides": {}}
        else:
            warnings.append("legacy staged source missing")
            source = {"path": str(staged),
                      "original_path": original or input_name,
                      "fingerprint": "", "size": 0, "missing": True,
                      "format_overrides": {}}
    model = WorkspaceModel(config, session_id=directory.name)
    preset_path = state.get("last_preset_path")
    preset_snapshot = model.preset_snapshot
    if isinstance(preset_path, str) and preset_path:
        try:
            preset_cfg, _n = normalize_config(
                load_merged_yaml(pipeline_config_paths(preset_path)))
            preset_snapshot = preset_cfg
        except Exception:
            warnings.append(
                "legacy preset could not be loaded; using config as anchor")
    ui_state = {k: v for k, v in state.items()
                if k in ("zoom", "pan", "xlim", "ylim", "roi")}
    model.restore_saved_state(
        config=model.config,
        preset_snapshot=preset_snapshot,
        preset_path=preset_path if isinstance(preset_path, str) else None,
        saved_snapshot=model.config,
        source=source, saved_source=source)
    return model, ui_state, {}
