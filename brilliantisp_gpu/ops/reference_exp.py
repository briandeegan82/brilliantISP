"""NumPy 1.26.4 float32 ``exp`` reference profile (pure torch, same device).

Bit-exact port of NumPy's AVX2/FMA3 ``simd_exp_FLOAT`` path
(``loops_exponent_log.dispatch.c.src``) — see ``LICENSE.numpy`` for the
upstream BSD-3-Clause license. Constants and algorithm follow the frozen
NumPy 1.26.4 source; no CPU transfer or fallback is performed — every op
runs on the input tensor's device.
"""

from __future__ import annotations

from ..types import require_torch

__license__ = "BSD-3-Clause"

_XMAX = 88.72283935546875
_XMIN = -103.97208404541015625


def _fma32(a, b, c):
    """Exact fused-multiply-add in float32 via float64 intermediates."""
    torch = require_torch()
    return (a.to(torch.float64) * b.to(torch.float64)
            + c.to(torch.float64)).to(torch.float32)


def reference_exp_float32(x):
    """Bit-exact NumPy ``exp`` for a float32 tensor, same device/shape."""
    torch = require_torch()
    if not isinstance(x, torch.Tensor):
        raise TypeError(
            f"reference_exp_float32 requires a torch.Tensor, got {type(x).__name__}")
    if x.dtype != torch.float32:
        raise ValueError(
            f"reference_exp_float32 requires float32 input, got {x.dtype}")
    if x.numel() == 0:
        raise ValueError("reference_exp_float32 requires a nonempty tensor")
    arr = x.contiguous()
    f32 = lambda v: torch.as_tensor(v, dtype=torch.float32, device=arr.device)

    magic = f32(12582912.0)
    log2e = f32(1.4426950408889634)
    c1 = f32(-6.93145752e-1)
    c2 = f32(-1.42860677e-6)
    P = [f32(v) for v in (
        9.999999999980870924916e-01, 7.257664613233124478488e-01,
        2.473615434895520810817e-01, 5.114512081637298353406e-02,
        6.757896990527504603057e-03, 5.082762527590693718096e-04)]
    Q = [f32(1.0), f32(-2.742335390411667452936e-01),
         f32(2.159509375685829852307e-02)]

    nan_mask = torch.isnan(arr)
    over_mask = arr >= f32(_XMAX)
    under_mask = arr <= f32(_XMIN)
    safe = torch.where(nan_mask | over_mask | under_mask,
                       f32(0.0), arr)

    q = (safe * log2e + magic) - magic
    r = _fma32(q, c1, safe)
    r = _fma32(q, c2, r)

    p = _fma32(P[5], r, P[4])
    p = _fma32(p, r, P[3])
    p = _fma32(p, r, P[2])
    p = _fma32(p, r, P[1])
    p = _fma32(p, r, P[0])
    d = _fma32(Q[2], r, Q[1])
    d = _fma32(d, r, Q[0])
    poly = p / d

    n = q.to(torch.int32)
    neg125 = torch.as_tensor(-125, dtype=torch.int32, device=arr.device)
    safe_n = torch.clamp(n, min=-125)
    scaled = (poly.view(torch.int32) + (safe_n << 23)).view(torch.float32)
    den_shift = torch.clamp(neg125 - n, min=0)
    den = (torch.ones_like(den_shift) << den_shift).to(torch.float32)
    result = scaled / den

    result = torch.where(over_mask, f32(float("inf")), result)
    result = torch.where(under_mask, f32(0.0), result)
    result = torch.where(nan_mask, f32(float("nan")), result)
    return result.reshape(x.shape).clone()
