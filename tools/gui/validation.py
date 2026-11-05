"""Pre-Process validation for the calibration GUI.

Compatibility wrapper: the implementation lives in
``util.config_schema.validate_config`` so GUI and future CLI/migration tooling
share one contract. ``can_process`` blocks only on ``severity == "error"``
issues; warnings are returned to the caller but do not block.
"""

from __future__ import annotations

from typing import Any, Mapping

from util.config_schema import VALID_ENDIAN_TYPES as VALID_ENDIAN
from util.config_schema import ValidationIssue, validate_config

__all__ = ["ValidationIssue", "validate_config", "can_process", "VALID_ENDIAN"]


def can_process(
    config: Mapping[str, Any],
    *,
    has_raw: bool,
) -> tuple[bool, list[ValidationIssue]]:
    issues = validate_config(config, require_raw=True, has_raw=has_raw)
    return (not any(i.severity == "error" for i in issues), issues)


def parse_numeric_editor(raw: str, original: Any) -> int | float | str:
    """Strict GUI numeric parsing: integers stay int only when the parsed
    value is integral, fractional input stays float, and non-numeric text is
    returned unchanged so the shared validator can report it — never
    truncated or silently reset to a prior value.
    """
    text = str(raw).strip()
    try:
        value = float(text)
    except (TypeError, ValueError):
        return raw
    if isinstance(original, bool):
        return raw
    if isinstance(original, int) and value.is_integer():
        return int(value)
    return value
