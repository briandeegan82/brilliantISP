"""Single shared configuration object mirroring camera YAML."""

from __future__ import annotations

import copy
from typing import Any, Sequence

PathKey = str | Sequence[str]


def _split(path: PathKey) -> tuple[str, ...]:
    if isinstance(path, str):
        return tuple(p for p in path.split(".") if p)
    return tuple(path)


def get_path(config: dict[str, Any], path: PathKey, default: Any = None) -> Any:
    cur: Any = config
    for key in _split(path):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def set_path(config: dict[str, Any], path: PathKey, value: Any) -> None:
    keys = _split(path)
    if not keys:
        raise ValueError("empty path")
    cur: dict[str, Any] = config
    for key in keys[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[keys[-1]] = value


def _values_equal(a: Any, b: Any) -> bool:
    return a == b


def collect_dirty_paths(current: dict[str, Any], snapshot: dict[str, Any]) -> list[str]:
    dirty: list[str] = []

    def walk(cur: Any, snap: Any, prefix: str) -> None:
        if isinstance(cur, dict) and isinstance(snap, dict):
            keys = set(cur) | set(snap)
            for k in sorted(keys):
                path = f"{prefix}.{k}" if prefix else k
                if k not in cur:
                    dirty.append(path)
                    continue
                if k not in snap:
                    dirty.append(path)
                    continue
                walk(cur[k], snap[k], path)
            return
        if not _values_equal(cur, snap):
            dirty.append(prefix or ".")

    walk(current, snapshot, "")
    return dirty


class SharedConfig:
    """In-memory YAML-shaped config + preset snapshot for change tracking."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = copy.deepcopy(config) if config else {}
        self.preset_snapshot: dict[str, Any] = copy.deepcopy(self.config)
        self.preset_path: str | None = None

    def load_from(self, config: dict[str, Any], *, preset_path: str | None = None) -> None:
        self.config = copy.deepcopy(config)
        self.preset_snapshot = copy.deepcopy(config)
        self.preset_path = preset_path

    def get(self, path: PathKey, default: Any = None) -> Any:
        return get_path(self.config, path, default)

    def set(self, path: PathKey, value: Any) -> None:
        set_path(self.config, path, value)

    def dirty_paths(self) -> list[str]:
        return collect_dirty_paths(self.config, self.preset_snapshot)

    def is_dirty(self, path: PathKey | None = None) -> bool:
        dirty = self.dirty_paths()
        if path is None:
            return bool(dirty)
        prefix = ".".join(_split(path))
        return any(d == prefix or d.startswith(prefix + ".") for d in dirty)

    def reset_parameter(self, path: PathKey) -> None:
        keys = _split(path)
        snap = get_path(self.preset_snapshot, keys, default=None)
        if snap is None and get_path(self.preset_snapshot, keys[:-1] if keys else []) is None:
            parent = get_path(self.config, keys[:-1]) if len(keys) > 1 else self.config
            if isinstance(parent, dict) and keys[-1] in parent:
                del parent[keys[-1]]
            return
        set_path(self.config, keys, copy.deepcopy(snap))

    def reset_module(self, section: str) -> None:
        if section in self.preset_snapshot:
            self.config[section] = copy.deepcopy(self.preset_snapshot[section])
        elif section in self.config:
            del self.config[section]

    def reset_all(self) -> None:
        self.config = copy.deepcopy(self.preset_snapshot)

    def apply_ae(self, enabled: bool) -> None:
        set_path(self.config, "auto_exposure.is_enable", bool(enabled))

    def apply_awb(self, enabled: bool) -> None:
        set_path(self.config, "auto_white_balance.is_enable", bool(enabled))

    def set_digital_gain_index(self, index: int) -> None:
        set_path(self.config, "digital_gain.current_gain", int(index))
