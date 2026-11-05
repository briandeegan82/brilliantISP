"""Emulated reference log primitives matching the host libc float32 profile.

reference_log10_float32 reproduces NumPy/glibc float32 log10 bit-for-bit
on positive normal float32 inputs using the glibc e_logf table polynomial
and the Sun fdlibm split-constant log10 reduction.
"""

from __future__ import annotations

import numpy as np

from ..types import require_torch
from .reference_math_data import (LOGF_OFFSET, LOGF_LN2, LOGF_POLY,
                                  LOGF_TABLE, LOG10_SPLIT_HI,
                                  LOG10_SPLIT_IV, LOG10_SPLIT_LO)

__license__ = "LGPL-2.1-or-later"

_F32_TINY = float(np.float32(2.0 ** -126))


def _logf_mantissa(x):
    """glibc e_logf natural log on float32 mantissa domain [0.5, 2)."""
    torch = require_torch()
    table = torch.as_tensor(LOGF_TABLE, dtype=torch.float64,
                            device=x.device)
    b = x.contiguous().view(torch.int32).to(torch.int64)
    d = b - LOGF_OFFSET
    i = (d >> 19) & 15
    k = d >> 23
    z = (b - (k << 23)).to(torch.int32).view(torch.float32).to(torch.float64)
    r = z * table[i, 0] - 1.0
    base = table[i, 1] + k.to(torch.float64) * LOGF_LN2
    r2 = r * r
    p = LOGF_POLY[1] * r + LOGF_POLY[2]
    p = LOGF_POLY[0] * r2 + p
    return (p * r2 + (base + r)).to(torch.float32)


def reference_log10_float32(x):
    """Bit-exact NumPy float32 log10 on positive normal float32 input."""
    torch = require_torch()
    arr = torch.as_tensor(x)
    if arr.dtype != torch.float32:
        raise ValueError(
            f"reference_log10_float32 expects float32 input, got {arr.dtype}")
    if not arr.numel():
        raise ValueError("reference_log10_float32 input must be nonempty")
    if not bool(torch.isfinite(arr).all()):
        raise ValueError("reference_log10_float32 input must be finite")
    if not bool((arr >= _F32_TINY).all()):
        raise ValueError(
            "reference_log10_float32 requires positive normal float32")
    bits = arr.contiguous().view(torch.int32).to(torch.int64)
    e = (bits >> 23) - 127
    flag = (e < 0).to(torch.int64)
    mantissa_bits = (bits & 0x7FFFFF) | ((127 - flag) << 23)
    u = mantissa_bits.to(torch.int32).view(torch.float32)
    y = (e + flag).to(torch.float32)
    ln = _logf_mantissa(u)
    return ((y * LOG10_SPLIT_LO) + (LOG10_SPLIT_IV * ln)) \
        + (y * LOG10_SPLIT_HI)
