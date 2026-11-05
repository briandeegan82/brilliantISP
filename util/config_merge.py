"""
Deep-merge YAML pipeline configs: later files override earlier ones (per-key, recursive for dicts).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import yaml

StrPath = str | Path
ConfigPathArg = StrPath | list[StrPath] | tuple[StrPath, ...]

_PIPELINE_BASE_NAME = "base_hdr.yml"


def pipeline_config_paths(config_path: StrPath) -> list[Path]:
    """
    Resolve paths for ``BrilliantISP.load_config`` / ``load_merged_yaml``.

    Filenames ending in ``_cam.yml`` are merged **after** ``config/base_hdr.yml``.
    All other paths load as a single file (standalone full config or temp export).

    A **bare** filename (e.g. ``svs_cam.yml``) is looked up under ``config/``.
    Paths with a directory part are ``resolve()``'d from the current working directory.
    """
    p = Path(config_path).expanduser()
    cfg_root = Path(__file__).resolve().parent.parent / "config"
    if p.is_absolute():
        resolved = p
    elif p.parent == Path("."):
        resolved = (cfg_root / p.name).resolve()
    else:
        resolved = p.resolve()
    base = cfg_root / _PIPELINE_BASE_NAME
    if resolved.name.endswith("_cam.yml") and base.is_file() and resolved.is_file():
        return [base, resolved]
    return [resolved]


def normalize_config_paths(config_path: ConfigPathArg) -> list[Path]:
    """Single path or non-empty list/tuple of paths."""
    if isinstance(config_path, (str, Path)):
        return [Path(config_path)]
    if len(config_path) == 0:
        raise ValueError("config_path sequence must not be empty")
    return [Path(p) for p in config_path]


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """
    Merge overlay into base, returning a completely independent deep copy.

    Dict values are merged recursively; lists and scalars are replaced entirely
    by overlay when the key is present in overlay. Neither input is mutated and
    no nested dict/list in the result is shared with either input.
    """
    out: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(
                cast(Mapping[str, Any], out[key]),
                cast(Mapping[str, Any], value),
            )
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml_mapping(path: StrPath) -> dict[str, Any]:
    """Load a YAML file; root must be a mapping. Empty / null file yields {}."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"{p}: root YAML value must be a mapping (dict), got {type(data).__name__}")
    return data


def _resolve_extends_entries(raw: Mapping[str, Any], path: Path) -> list[Path]:
    if "extends" not in raw:
        return []
    extends = raw["extends"]
    if extends is None:
        raise ValueError(
            f"{path}: 'extends' must be a path string or a non-empty list of path strings"
        )
    if isinstance(extends, (str, Path)):
        entries: list[Any] = [extends]
    elif (
        isinstance(extends, (list, tuple))
        and len(extends) > 0
        and all(isinstance(e, (str, Path)) for e in extends)
    ):
        entries = list(extends)
    else:
        raise ValueError(
            f"{path}: 'extends' must be a path string or a non-empty list of path strings"
        )
    resolved: list[Path] = []
    for entry in entries:
        if isinstance(entry, str) and not entry.strip():
            raise ValueError(
                f"{path}: 'extends' entries must be non-empty path strings"
            )
        base = Path(entry).expanduser()
        if not base.is_absolute():
            base = path.parent / base
        resolved.append(base)
    return resolved


def _load_yaml_with_inheritance(path: StrPath, chain: tuple[Path, ...]) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if p in chain:
        cycle = " -> ".join(str(c) for c in (*chain, p))
        raise ValueError(f"config extends cycle detected: {cycle}")
    trail = " -> ".join(str(c) for c in (*chain, p))
    try:
        raw = load_yaml_mapping(p)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{exc} (extends chain: {trail})") from exc
    except (yaml.YAMLError, TypeError) as exc:
        raise ValueError(f"extends chain {trail}: {exc}") from exc
    try:
        bases = _resolve_extends_entries(raw, p)
    except ValueError as exc:
        raise ValueError(f"extends chain {trail}: {exc}") from exc
    merged: dict[str, Any] = {}
    for base in bases:
        merged = deep_merge(merged, _load_yaml_with_inheritance(base, (*chain, p)))
    child = dict(raw)
    child.pop("extends", None)
    return deep_merge(merged, child)


def load_merged_yaml(paths: Sequence[StrPath]) -> dict[str, Any]:
    """
    Load and deep-merge YAML documents in order. Same rules as ``deep_merge``.

    Each document may declare a top-level ``extends`` entry: a path string or a
    non-empty list of path strings, resolved relative to the referring file.
    Inherited documents are merged in listed order before the child document;
    ``extends`` itself is removed from the effective config. Inheritance cycles
    raise ``ValueError`` naming the path chain.

    Example:

        load_merged_yaml(pipeline_config_paths("config/svs_cam.yml"))
    """
    if len(paths) == 0:
        raise ValueError("load_merged_yaml requires at least one path")
    merged: dict[str, Any] = {}
    for p in paths:
        merged = deep_merge(merged, _load_yaml_with_inheritance(p, ()))
    return merged


def format_config_source(paths: Sequence[StrPath]) -> str:
    """Human-readable label for errors/logging (single path is unchanged)."""
    parts = [str(Path(p)) for p in paths]
    return parts[0] if len(parts) == 1 else " | ".join(parts)
