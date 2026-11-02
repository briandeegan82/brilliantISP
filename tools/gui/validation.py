"""Pre-Process validation for the calibration GUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tools.gui.format_map import BAYER_PATTERNS

VALID_ENDIAN = ("ieee-le", "ieee-be")


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def validate_config(
    config: dict[str, Any],
    *,
    require_raw: bool = False,
    has_raw: bool = False,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    sensor = config.get("sensor_info") or {}
    crop = config.get("crop") or {}
    wb = config.get("white_balance") or {}
    dg = config.get("digital_gain") or {}

    width = _as_number(sensor.get("width"))
    height = _as_number(sensor.get("height"))
    if width is None or width <= 0:
        issues.append(ValidationIssue("sensor_info.width", "width must be > 0"))
    if height is None or height <= 0:
        issues.append(ValidationIssue("sensor_info.height", "height must be > 0"))

    if bool(crop.get("is_enable")) and width and height and width > 0 and height > 0:
        x0 = _as_number(crop.get("crop_x_start")) or 0.0
        y0 = _as_number(crop.get("crop_y_start")) or 0.0
        nw = _as_number(crop.get("new_width"))
        nh = _as_number(crop.get("new_height"))
        if nw is None or nw <= 0 or nh is None or nh <= 0:
            issues.append(ValidationIssue("crop.new_width", "crop region must have positive size"))
        elif x0 < 0 or y0 < 0 or (x0 + nw) > width or (y0 + nh) > height:
            issues.append(ValidationIssue("crop", "crop region must lie inside the sensor frame"))

    pattern = str(sensor.get("bayer_pattern", "")).lower()
    if pattern not in BAYER_PATTERNS:
        issues.append(
            ValidationIssue(
                "sensor_info.bayer_pattern",
                f"bayer_pattern must be one of {', '.join(BAYER_PATTERNS)}",
            )
        )

    bit_depth = _as_number(sensor.get("bit_depth"))
    if bit_depth is None or bit_depth <= 0 or bit_depth > 32:
        issues.append(ValidationIssue("sensor_info.bit_depth", "bit_depth must be a positive integer ≤ 32"))

    endian = str(sensor.get("endian_type", "")).lower()
    if endian not in VALID_ENDIAN:
        issues.append(
            ValidationIssue(
                "sensor_info.endian_type",
                "endian_type must be ieee-le or ieee-be",
            )
        )

    r_gain = _as_number(wb.get("r_gain"))
    b_gain = _as_number(wb.get("b_gain"))
    if r_gain is None or r_gain <= 0:
        issues.append(ValidationIssue("white_balance.r_gain", "WB r_gain must be > 0"))
    if b_gain is None or b_gain <= 0:
        issues.append(ValidationIssue("white_balance.b_gain", "WB b_gain must be > 0"))

    gain_array = dg.get("gain_array") if isinstance(dg.get("gain_array"), list) else []
    idx = _as_number(dg.get("current_gain"))
    if idx is None:
        issues.append(ValidationIssue("digital_gain.current_gain", "current_gain is required"))
    else:
        i = int(idx)
        if not gain_array or i < 0 or i >= len(gain_array):
            issues.append(
                ValidationIssue(
                    "digital_gain.current_gain",
                    "current_gain index is out of range for gain_array",
                )
            )
        else:
            gval = _as_number(gain_array[i])
            if gval is None or gval <= 0:
                issues.append(ValidationIssue("digital_gain.gain_array", "digital gain must be > 0"))

    if require_raw and not has_raw:
        issues.append(ValidationIssue("input.raw", "Load a test image before Process"))

    return issues


def can_process(
    config: dict[str, Any],
    *,
    has_raw: bool,
) -> tuple[bool, list[ValidationIssue]]:
    issues = validate_config(config, require_raw=True, has_raw=has_raw)
    return (len(issues) == 0, issues)
