"""Quick Bayer/RAW mosaic preview for the calibration GUI (not the full ISP).

The preview decodes the configured input through the shared
``util.raw_decode`` contract (storage/alignment/embedded rows/flips) and
shows the frame BEFORE the pipeline's own crop — it is not a full-output
parity check.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tools.gui.format_map import BAYER_PATTERNS
from util.raw_decode import RawFormat, decode_raw

_CHANNEL_INDEX = {"r": 0, "g": 1, "b": 2}


def load_bayer_array(
    path: str | Path,
    *,
    width: int,
    height: int,
    endian: str = "ieee-be",
    bit_depth: int = 16,
    data_format: str = "uint16",
    storage_format: str | None = None,
    source_format: str = "auto",
    bayer_pattern: str = "rggb",
    data_alignment: str = "LSB",
    manual_bit_shift: int | None = None,
    header_bytes: int = 0,
    row_stride_bytes: int | None = None,
    word_order: str = "low_first",
    embedded_rows_top: int = 0,
    embedded_rows_bottom: int = 0,
    horizontal_flip: bool = False,
    vertical_flip: bool = False,
) -> np.ndarray:
    sensor: dict[str, Any] = {
        "width": width,
        "height": height,
        "bit_depth": bit_depth,
        "bayer_pattern": bayer_pattern,
        "endian_type": endian,
        "data_alignment": data_alignment,
        "manual_bit_shift": manual_bit_shift,
        "header_bytes": header_bytes,
        "row_stride_bytes": row_stride_bytes,
        "word_order": word_order,
        "horizontal_flip": horizontal_flip,
        "vertical_flip": vertical_flip,
        "embedded_rows_top": embedded_rows_top,
        "embedded_rows_bottom": embedded_rows_bottom,
        "source_format": source_format,
    }
    if storage_format is not None:
        sensor["storage_format"] = storage_format
    else:
        sensor["data_format"] = data_format
    fmt = RawFormat.from_sensor_info(sensor)
    return decode_raw(path, fmt).pixels


def mosaic_preview(bayer: np.ndarray, pattern: str = "rggb") -> np.ndarray:
    """Colorize CFA sites and stretch to 8-bit RGB for display."""
    pat = str(pattern).lower()
    if pat not in BAYER_PATTERNS:
        raise ValueError(f"unknown bayer_pattern {pattern!r}")
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


def preview_from_config(
    path: str | Path,
    config: dict[str, Any],
    header_bytes: int | None = None,
) -> np.ndarray:
    """Decode a source preview BEFORE the pipeline's own crop stage.

    Returns the aligned/trimmed/flipped input mosaic — not a processed
    output preview. Uses the same ``RawFormat.from_sensor_info`` contract
    as the pipeline, so invalid fields fail identically.
    """
    sensor = config.get("sensor_info")
    if not isinstance(sensor, dict):
        raise ValueError("config must contain a sensor_info mapping")
    fmt_sensor = dict(sensor)
    if header_bytes is not None:
        fmt_sensor["header_bytes"] = header_bytes
    fmt = RawFormat.from_sensor_info(fmt_sensor)
    decoded = decode_raw(path, fmt)
    return mosaic_preview(decoded.pixels, decoded.bayer_pattern)
