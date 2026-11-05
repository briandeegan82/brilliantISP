from __future__ import annotations

import math
from typing import Any

import yaml

from util.config_schema import ABSENT, spec_map

SPECS = spec_map()


def field_kind(path: str, value: Any) -> str:
    spec = SPECS.get(path)
    if spec is not None:
        return spec.value_type
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return "str" if isinstance(value, str) else "yaml"


def parse_draft(path: str, raw: str, original: Any) -> Any:
    kind = field_kind(path, original)
    text = raw.strip()
    if kind in ("int", "integer", "float", "number"):
        if not text and (original is None or path.endswith("row_stride_bytes")):
            return None
        try:
            if kind in ("int", "integer"):
                return int(text, 16) if text.lower().startswith(("0x", "-0x", "+0x")) else int(text, 10)
            value = float(text)
            return value if math.isfinite(value) else raw
        except ValueError:
            return raw
    if kind == "bool":
        if text.lower() in ("true", "false"):
            return text.lower() == "true"
        return raw
    if kind in ("str", "string", "enum"):
        return raw
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def visible_fields(section: str, config: dict) -> list[tuple[str, Any, bool]]:
    block = config.get(section, ABSENT)
    if block is ABSENT:
        return [(p, s.default, False) for p, s in SPECS.items() if p.startswith(section + ".") and s.has_default]
    if not isinstance(block, dict):
        return [(section, block, True)]
    result: dict[str, tuple[Any, bool]] = {}

    def walk(values: dict, prefix: str) -> None:
        for key, value in values.items():
            path = f"{prefix}.{key}"
            if isinstance(value, dict) and value:
                walk(value, path)
            else:
                result[path] = (value, True)
    walk(block, section)
    for path, spec in SPECS.items():
        if path.startswith(section + ".") and spec.has_default and path not in result:
            parts = path.split(".")
            parent: Any = config
            for part in parts[:-1]:
                parent = parent.get(part, {}) if isinstance(parent, dict) else None
            if isinstance(parent, dict):
                result[path] = (spec.default, False)
    if section == "tone_mapping" and config.get("schema_version") == 2:
        from modules.tone_mapping.registry import OPERATOR_PARAMETERS
        operator = block.get("operator")
        params = OPERATOR_PARAMETERS.get(operator, {}) if isinstance(operator, str) else {}
        for name, value in params.items():
            result.setdefault(f"tone_mapping.parameters.{name}", (value, False))
    if section in ("aces", "aces_integer", "hable", "hable_integer", "reinhard_integer", "integer_tmo"):
        from tools.gui.legacy_calibration import LEGACY_IGNORED_TM_FIELDS
        result = {p: v for p, v in result.items() if p not in LEGACY_IGNORED_TM_FIELDS and p != "aces.variant"}
    result = {p: v for p, v in result.items() if not (
        isinstance(v[0], dict) and any(other.startswith(p + ".") for other in result))}
    return [(path, value, present) for path, (value, present) in result.items()]
