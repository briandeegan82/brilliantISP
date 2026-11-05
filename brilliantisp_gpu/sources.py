"""CPU-only source decoding: arrays and RAW paths -> DecodedSource.

Pure loader for the engine/microbatch (future prefetch); no torch imports.
Mirrors the qualified engine._load_source CPU portion exactly: arrays are
already in processing coordinates (no flips/alignment/row trim); paths go
through the shared RawFormat/decode_raw declaration decoder.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from util.raw_decode import RawFormat, decode_raw


@dataclass(frozen=True)
class DecodedSource:
    pixels: np.ndarray
    sensor_info: dict[str, Any]
    warnings: tuple[str, ...]
    in_file: str
    out_file: str
    source_identity: str


def decode_source(source, *, sensor_info, platform, run_id) -> DecodedSource:
    """CPU decode of one frame source into processing coordinates.

    ``source`` is a decoded 2-D Bayer ndarray (exact declared geometry,
    already processing coordinates) or a RAW file path decoded through the
    declared RawFormat. Parent sensor_info/platform mappings are never
    mutated.
    """
    sensor = copy.deepcopy(dict(sensor_info))
    warnings: list[str] = []
    if isinstance(source, np.ndarray):
        frame = np.array(source, copy=True)
        if frame.ndim != 2:
            raise ValueError(
                f"array source must be a 2-D Bayer frame, got shape {frame.shape}")
        if frame.dtype.kind != "u" or frame.dtype.itemsize not in (1, 2, 4):
            raise ValueError(
                f"array source dtype {frame.dtype} is not supported; "
                "decoded Bayer input must be uint8/uint16/uint32")
        frame = frame.astype(frame.dtype.newbyteorder("="), copy=False)
        expected = (int(sensor["height"]), int(sensor["width"]))
        if frame.shape != expected:
            raise ValueError(
                f"array shape {frame.shape} does not match declared "
                f"sensor_info geometry {expected[1]}x{expected[0]}")
        return DecodedSource(
            pixels=frame,
            sensor_info=sensor,
            warnings=(),
            in_file="array_source",
            out_file="array_source",
            source_identity=f"array:{run_id}",
        )

    path = Path(source).expanduser().resolve()
    fmt = RawFormat.from_sensor_info(sensor)
    decoded = decode_raw(str(path), fmt)
    for warning in decoded.warnings:
        if warning not in warnings:
            warnings.append(warning)
    sensor["width"] = decoded.width
    sensor["height"] = decoded.height
    sensor["bayer_pattern"] = decoded.bayer_pattern
    short_names = platform.get("short_output_names", False)
    in_file = path.stem
    out_file = in_file if short_names else "Out_" + in_file
    return DecodedSource(
        pixels=np.array(decoded.pixels, copy=True),
        sensor_info=sensor,
        warnings=tuple(warnings),
        in_file=in_file,
        out_file=out_file,
        source_identity=str(path),
    )
