"""Read-only format statistics derived from shared YAML state."""

from __future__ import annotations

from typing import Any

from util.raw_decode import RawFormat, RawFormatError, storage_bytes_per_sample


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def bytes_per_sample(bit_depth: int, data_format: str | None) -> int:
    try:
        return storage_bytes_per_sample(data_format, bit_depth)
    except RawFormatError:
        return 0


def compute_format_stats(
    config: dict[str, Any],
    *,
    header_bytes: int = 0,
) -> dict[str, Any]:
    sensor = config.get("sensor_info") or {}
    crop = config.get("crop") or {}
    scale = config.get("scale") or {}

    format_error = ""
    width = height = bit_depth = 0
    bpp = 0
    row_bytes = 0
    estimated = 0
    try:
        probe = dict(sensor)
        probe.setdefault("bayer_pattern", "rggb")
        probe.setdefault("bit_depth", 16)
        probe.setdefault("header_bytes", header_bytes)
        fmt = RawFormat.from_sensor_info(probe)
        width = fmt.width
        height = fmt.height
        bit_depth = fmt.bit_depth
        bpp = storage_bytes_per_sample(fmt.storage_format, bit_depth)
        row_bytes = (
            fmt.row_stride_bytes
            if fmt.row_stride_bytes is not None
            else width * bpp
        )
        estimated = fmt.header_bytes + height * row_bytes
    except (RawFormatError, ValueError, TypeError, OverflowError) as exc:
        format_error = f"invalid RAW layout: {exc}"
        bpp = 0
        estimated = 0

    embedded_top = _as_int(sensor.get("embedded_rows_top"))
    embedded_bottom = _as_int(sensor.get("embedded_rows_bottom"))
    active_rows = max(0, height - embedded_top - embedded_bottom)

    crop_on = bool(crop.get("is_enable"))
    if crop_on:
        eff_w = _as_int(crop.get("new_width"), width)
        eff_h = _as_int(crop.get("new_height"), active_rows)
    else:
        eff_w, eff_h = width, active_rows

    sensor_pixels = max(width, 0) * max(active_rows, 0)
    active = max(eff_w, 0) * max(eff_h, 0)
    if crop_on and sensor_pixels > 0:
        crop_pct = 100.0 * (1.0 - (active / sensor_pixels))
    else:
        crop_pct = 0.0

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
        "format_error": format_error,
    }
