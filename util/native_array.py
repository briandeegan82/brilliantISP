"""Shared checked float32 boundary for native linear compute paths.

The cast is verified: finite computed input AND finite float32 output are
required, so a finite float64 value above the float32 range raises a named
ValueError instead of silently returning inf. Uses the same bounded
errstate pattern as ``util.color._finish``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def as_float32(values: Any, *, name: str = "array") -> np.ndarray:
    """Cast a real numeric array to float32, rejecting nonfinite results.

    Bool/complex/object inputs are rejected; computed nonfinite output and
    overflow in the float32 cast are named ``ValueError``.
    """
    arr = np.asarray(values)
    if arr.dtype.kind not in "fiu" or arr.size == 0:
        raise ValueError(f"{name} must be a nonempty real numeric array")
    if arr.dtype.kind == "f" and not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} produced nonfinite output")
    if arr.dtype == np.dtype(np.float32):
        return arr
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        try:
            out = arr.astype(np.float32, copy=False)
        except (OverflowError, FloatingPointError) as exc:
            raise ValueError(
                f"{name} output overflows float32 (inf would result)"
            ) from exc
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} output overflows float32 (inf would result)")
    return out
