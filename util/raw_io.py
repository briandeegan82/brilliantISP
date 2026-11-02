"""RAW / PGM payload helpers used by the pipeline and GUIs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RawPayload:
    data: bytes
    header_bytes: int = 0
    pgm_width: int | None = None
    pgm_height: int | None = None
    pgm_maxval: int | None = None
    is_pgm: bool = False


def _skip_ws_and_comments(data: bytes, i: int) -> int:
    n = len(data)
    while i < n:
        ch = data[i]
        if ch in (9, 10, 13, 32):  # \t \n \r space
            i += 1
            continue
        if ch == 35:  # '#'
            i += 1
            while i < n and data[i] not in (10, 13):
                i += 1
            continue
        break
    return i


def _read_ascii_token(data: bytes, i: int) -> tuple[str, int]:
    i = _skip_ws_and_comments(data, i)
    n = len(data)
    start = i
    while i < n and data[i] not in (9, 10, 13, 32, 35):
        i += 1
    if start == i:
        raise ValueError("Truncated PGM header")
    return data[start:i].decode("ascii"), i


def parse_pgm_header(data: bytes) -> tuple[str, int, int, int, int] | None:
    """
    Parse a Netpbm PGM header.

    Returns (magic, width, height, maxval, payload_offset) or None if not PGM.
    """
    if len(data) < 3 or data[:2] not in (b"P2", b"P5"):
        return None
    if data[2] not in (9, 10, 13, 32, 35):
        return None
    magic = data[:2].decode("ascii")
    i = 2
    width_s, i = _read_ascii_token(data, i)
    height_s, i = _read_ascii_token(data, i)
    maxval_s, i = _read_ascii_token(data, i)
    # After maxval, consume exactly one separator so binary raster bytes that
    # happen to look like whitespace are not eaten.
    if i >= len(data):
        raise ValueError("Truncated PGM header after maxval")
    i += 1
    try:
        width = int(width_s)
        height = int(height_s)
        maxval = int(maxval_s)
    except ValueError as exc:
        raise ValueError("Invalid PGM header numbers") from exc
    if width <= 0 or height <= 0 or maxval <= 0 or maxval > 65535:
        raise ValueError(f"Invalid PGM geometry/maxval: {width}x{height} maxval={maxval}")
    return magic, width, height, maxval, i


def _p2_payload_bytes(text: bytes, width: int, height: int, maxval: int) -> bytes:
    values: list[int] = []
    i = 0
    n = len(text)
    while i < n:
        i = _skip_ws_and_comments(text, i)
        if i >= n:
            break
        token, i = _read_ascii_token(text, i)
        values.append(int(token))
    need = width * height
    if len(values) < need:
        raise ValueError(f"P2 PGM too short: need {need} samples, got {len(values)}")
    arr = np.asarray(values[:need], dtype=np.uint16 if maxval > 255 else np.uint8)
    if maxval > 255:
        # Netpbm 16-bit raster is big-endian.
        return arr.astype(">u2").tobytes()
    return arr.astype(np.uint8).tobytes()


def read_raw_payload(path: str | Path) -> RawPayload:
    """Read file bytes, stripping a PGM header when present."""
    path = Path(path)
    blob = path.read_bytes()
    parsed = parse_pgm_header(blob)
    if parsed is None:
        return RawPayload(data=blob)
    magic, width, height, maxval, offset = parsed
    raster = blob[offset:]
    if magic == "P2":
        data = _p2_payload_bytes(raster, width, height, maxval)
    else:
        sample_bytes = 2 if maxval > 255 else 1
        need = width * height * sample_bytes
        if len(raster) < need:
            raise ValueError(f"P5 PGM raster too small for {width}x{height}: need {need} bytes, got {len(raster)}")
        data = raster[:need]
    return RawPayload(
        data=data,
        header_bytes=offset,
        pgm_width=width,
        pgm_height=height,
        pgm_maxval=maxval,
        is_pgm=True,
    )


def apply_raw_flips(image: np.ndarray, horizontal: bool, vertical: bool) -> np.ndarray:
    """Mirror a loaded Bayer / RAW plane. Bayer phase is not rewritten."""
    out = image
    if horizontal:
        out = np.fliplr(out)
    if vertical:
        out = np.flipud(out)
    if horizontal or vertical:
        return np.ascontiguousarray(out)
    return out


def flips_from_sensor_info(sensor: dict[str, Any] | None) -> tuple[bool, bool]:
    sensor = sensor or {}
    return bool(sensor.get("horizontal_flip", False)), bool(sensor.get("vertical_flip", False))


def apply_data_alignment(
    image: np.ndarray,
    bit_depth: int,
    alignment: str | None = "LSB",
    manual_shift: int | None = None,
) -> np.ndarray:
    """Unpack MSB- or LSB-aligned samples into the low ``bit_depth`` bits.

    Many sensors store N-bit samples left-justified in a wider container
    (e.g. 12-bit in uint16 → values are ``code << (16 - 12)``). MSB alignment
    right-shifts by ``container_bits - bit_depth``. LSB leaves values unchanged.

    Args:
        image: Input image array
        bit_depth: Actual bit depth of the sensor data
        alignment: "MSB" (auto-shift), "LSB" (no shift), or "MANUAL" (use manual_shift)
        manual_shift: Number of bits to right-shift (only used when alignment="MANUAL")
    """
    align = str(alignment or "LSB").strip().upper()

    # MANUAL mode: use the explicitly specified shift amount
    if align == "MANUAL":
        if manual_shift is None or manual_shift <= 0:
            return image
        shift = int(manual_shift)
        depth = int(bit_depth)
        out_dtype = image.dtype if image.dtype.itemsize * 8 <= 16 and depth <= 16 else np.uint32
        mask = (1 << depth) - 1
        return ((image.astype(np.uint32) >> shift) & mask).astype(out_dtype)

    # LSB mode: no shift
    if align in ("", "LSB", "LSBFIRST", "RIGHT"):
        return image

    # MSB mode: auto-calculate shift based on container size
    if align not in ("MSB", "MSBFIRST", "LEFT"):
        raise ValueError(f"Unknown data_alignment {alignment!r}; expected MSB, LSB, or MANUAL")

    depth = int(bit_depth)
    if depth <= 0:
        return image
    container_bits = int(image.dtype.itemsize) * 8
    shift = container_bits - depth
    if shift <= 0:
        return image
    # Keep a wide enough dtype so shifted codes are not truncated early.
    out_dtype = image.dtype if container_bits <= 16 and depth <= 16 else np.uint32
    mask = (1 << depth) - 1
    return ((image.astype(np.uint32) >> shift) & mask).astype(out_dtype)


def alignment_from_sensor_info(sensor: dict[str, Any] | None) -> str:
    sensor = sensor or {}
    return str(sensor.get("data_alignment") or sensor.get("bit_alignment") or "LSB")
