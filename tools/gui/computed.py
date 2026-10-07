"""Read-only format statistics derived from shared YAML state."""

from __future__ import annotations

from typing import Any


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def bytes_per_sample(bit_depth: int, data_format: str | None) -> int:
    fmt = (data_format or "").lower()
    if "uint8" in fmt or fmt in ("raw8", "u8"):
        return 1
    if "uint32" in fmt or fmt in ("raw32", "u32"):
        return 4
    if "uint16" in fmt or fmt in ("raw16", "u16"):
        return 2
    if bit_depth <= 8:
        return 1
    if bit_depth <= 16:
        return 2
    if bit_depth <= 24:
        return 3
    return 4


def compute_format_stats(
    config: dict[str, Any],
    *,
    header_bytes: int = 0,
) -> dict[str, Any]:
    sensor = config.get("sensor_info") or {}
    crop = config.get("crop") or {}
    scale = config.get("scale") or {}

    width = _as_int(sensor.get("width"))
    height = _as_int(sensor.get("height"))
    bit_depth = _as_int(sensor.get("bit_depth"), 16)
    data_format = sensor.get("data_format")

    crop_on = bool(crop.get("is_enable"))
    if crop_on:
        eff_w = _as_int(crop.get("new_width"), width)
        eff_h = _as_int(crop.get("new_height"), height)
    else:
        eff_w, eff_h = width, height

    sensor_pixels = max(width, 0) * max(height, 0)
    active = max(eff_w, 0) * max(eff_h, 0)
    if crop_on and sensor_pixels > 0:
        crop_pct = 100.0 * (1.0 - (active / sensor_pixels))
    else:
        crop_pct = 0.0

    bpp = bytes_per_sample(bit_depth, str(data_format) if data_format else None)
    estimated = max(0, int(header_bytes)) + sensor_pixels * bpp

    if bool(scale.get("is_enable")):
        out_w = _as_int(scale.get("new_width"), eff_w)
        out_h = _as_int(scale.get("new_height"), eff_h)
    else:
        out_w, out_h = eff_w, eff_h

    return {
        "effective_width": eff_w,
        "effective_height": eff_h,
        "active_pixel_count": active,
        "crop_percentage": crop_pct,
        "estimated_raw_size": estimated,
        "output_width": out_w,
        "output_height": out_h,
        "bytes_per_sample": bpp,
    }
