"""Shared declared-scale luma helpers for native plane processing.

``normalize_plane`` maps a real 2-D input into float64 ``0..1`` using an
explicit declared ``scale`` (never inferred from dtype or content);
``restore_plane`` maps normalized output back to the same declared units
with a checked float32 cast. ``replace_luminance`` substitutes a new Y
plane into normalized float RGB, preserving Cb/Cr with no gamut clip.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _require_scale(scale: Any) -> float:
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise ValueError(f"scale must be a positive finite number, got {scale!r}")
    try:
        out = float(scale)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"scale must be a positive finite number, got {scale!r}") from exc
    if not math.isfinite(out) or out <= 0:
        raise ValueError(f"scale must be a positive finite number, got {scale!r}")
    return out


def normalize_plane(values: Any, *, scale: float) -> np.ndarray:
    """Real 2-D input -> float64 clipped to [0, scale] / scale."""
    s = _require_scale(scale)
    arr = np.asarray(values)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError(f"luma plane must be a nonempty 2-D array, got {arr.shape}")
    if arr.dtype.kind not in "fiu":
        raise ValueError(f"luma plane must be real numeric, got {arr.dtype}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("luma plane must be finite")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        clipped = np.clip(arr, 0.0, s).astype(np.float64)
    return clipped / s


def restore_plane(normalized: Any, *, scale: float) -> np.ndarray:
    """Normalized float output -> checked float32 in declared units."""
    s = _require_scale(scale)
    arr = np.asarray(normalized)
    if arr.ndim != 2 or not arr.size or arr.dtype.kind not in "fiu" or not np.all(np.isfinite(arr)):
        raise ValueError("luma restore requires a nonempty finite real 2-D plane")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        out = arr.astype(np.float64) * s
    if not np.all(np.isfinite(out)):
        raise ValueError("luma restore produced nonfinite output")
    from util.native_array import as_float32
    return as_float32(out, name="luma_restore")


def replace_luminance(
    rgb: Any, new_y: Any, *, matrix: str = "bt709"
) -> np.ndarray:
    """Substitute ``new_y`` into normalized float RGB, preserving chroma.

    ``rgb + (new_y - luminance(rgb))`` per channel; no gamut clipping.
    """
    from util.color import luminance
    from util.native_array import as_float32

    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"rgb must be HWC3, got {arr.shape}")
    if arr.dtype.kind != "f":
        raise ValueError(f"rgb must be a floating-point array, got {arr.dtype}")
    f = arr.astype(np.float64)
    if not np.all(np.isfinite(f)):
        raise ValueError("rgb must be finite")
    source_y = np.asarray(new_y)
    if source_y.dtype.kind not in "fiu":
        raise ValueError("new_y must be real numeric")
    y = source_y.astype(np.float64)
    if y.shape != f.shape[:2] or not np.all(np.isfinite(y)):
        raise ValueError("new_y must be finite with the same spatial shape")
    with np.errstate(over="ignore", invalid="ignore"):
        out = f + (y - luminance(f, matrix=matrix, dtype=np.float64))[..., None]
    return as_float32(out, name="replace_luminance")
