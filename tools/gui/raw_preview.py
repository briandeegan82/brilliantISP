"""Quick Bayer/RAW mosaic preview for the calibration GUI (not the full ISP)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tools.gui.format_map import BAYER_PATTERNS
from util.raw_io import apply_data_alignment, apply_raw_flips, flips_from_sensor_info, read_raw_payload

_CHANNEL_INDEX = {"r": 0, "g": 1, "b": 2}


def _dtype_for(endian: str, bit_depth: int, data_format: str) -> np.dtype:
    fmt = (data_format or "").lower()
    et = (endian or "").lower()
    little = "le" in et or "little" in et
    if "uint8" in fmt or bit_depth <= 8:
        return np.dtype(np.uint8)
    if "uint32" in fmt or bit_depth > 16:
        return np.dtype("<u4" if little else ">u4")
    return np.dtype("<u2" if little else ">u2")


def load_bayer_array(
    path: str | Path,
    *,
    width: int,
    height: int,
    endian: str = "ieee-be",
    bit_depth: int = 16,
    data_format: str = "uint16",
    data_alignment: str = "LSB",
    manual_bit_shift: int | None = None,
    header_bytes: int = 0,
    horizontal_flip: bool = False,
    vertical_flip: bool = False,
) -> np.ndarray:
    payload = read_raw_payload(path)
    raw = payload.data
    skip = max(0, int(header_bytes))
    if payload.is_pgm:
        skip = 0
        if payload.pgm_width:
            width = int(payload.pgm_width)
        if payload.pgm_height:
            height = int(payload.pgm_height)
    payload_bytes = raw[skip:]
    dtype = _dtype_for(endian, bit_depth, data_format)
    need = int(width) * int(height) * dtype.itemsize
    if len(payload_bytes) < need:
        raise ValueError(f"RAW too small for {width}x{height} {dtype}: need {need} bytes, got {len(payload_bytes)}")
    arr = np.frombuffer(payload_bytes[:need], dtype=dtype).reshape((int(height), int(width)))
    arr = apply_data_alignment(np.asarray(arr), bit_depth, data_alignment, manual_bit_shift)
    return apply_raw_flips(arr, horizontal_flip, vertical_flip)


def mosaic_preview(bayer: np.ndarray, pattern: str = "rggb") -> np.ndarray:
    """Colorize CFA sites and stretch to 8-bit RGB for display."""
    pat = str(pattern).lower()
    if pat not in BAYER_PATTERNS:
        pat = "rggb"
    letters = pat.upper()
    img = bayer.astype(np.float32)
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return np.zeros((*img.shape, 3), dtype=np.uint8)
    lo = float(np.percentile(finite, 1.0))
    hi = float(np.percentile(finite, 99.5))
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    rgb = np.zeros((*img.shape, 3), dtype=np.float32)
    for y in range(2):
        for x in range(2):
            ch = _CHANNEL_INDEX[letters[y * 2 + x].lower()]
            rgb[y::2, x::2, ch] = norm[y::2, x::2]
    return (rgb * 255.0).astype(np.uint8)


def preview_from_config(path: str | Path, config: dict[str, Any], header_bytes: int = 0) -> np.ndarray:
    sensor = config.get("sensor_info") or {}
    hflip, vflip = flips_from_sensor_info(sensor)
    manual_shift = sensor.get("manual_bit_shift")
    manual_shift = int(manual_shift) if manual_shift is not None else None
    bayer = load_bayer_array(
        path,
        width=int(sensor.get("width") or 0),
        height=int(sensor.get("height") or 0),
        endian=str(sensor.get("endian_type") or "ieee-le"),
        bit_depth=int(sensor.get("bit_depth") or 16),
        data_format=str(sensor.get("data_format") or "uint16"),
        data_alignment=str(sensor.get("data_alignment") or sensor.get("bit_alignment") or "LSB"),
        manual_bit_shift=manual_shift,
        header_bytes=header_bytes,
        horizontal_flip=hflip,
        vertical_flip=vflip,
    )
    return mosaic_preview(bayer, str(sensor.get("bayer_pattern") or "rggb"))
