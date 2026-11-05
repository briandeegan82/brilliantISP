"""Runtime-compiled reference argsort and fused FMA projection.

``reference_argsort`` reproduces NumPy 1.26.4's scalar ``aquicksort_``
(introsort with aheapsort fallback) bit-for-bit on finite float64 keys,
including tie order. ``fused_project`` evaluates one correctly-rounded
``fma(r2,u2, fma(r1,u1, r0*u0))`` per row. Both run entirely on device —
keys and indices never cross to host beyond scalar sizes/status.
"""

from __future__ import annotations

from ..types import GpuError, GpuUnavailableError, require_torch
from . import rtc

_DEPTH_KEEP = -(2**31)


def _check_keys(keys):
    torch = require_torch()
    if not isinstance(keys, torch.Tensor):
        raise GpuError(f"reference_argsort expects a torch tensor, got {type(keys).__name__}")
    if not keys.is_cuda:
        raise GpuError("reference_argsort requires a CUDA/HIP device tensor (no CPU sort)")
    if keys.dtype != torch.float64:
        raise GpuError(f"reference_argsort requires float64 keys, got {keys.dtype}")
    if keys.ndim != 1:
        raise GpuError(f"reference_argsort requires a 1-D tensor, got {tuple(keys.shape)}")
    if not bool(torch.isfinite(keys).all()):
        raise GpuError("reference_argsort requires finite keys")
    return keys.contiguous()


def _get_funcs(device):
    return rtc.get_module(device, rtc.load_source("reference_sort.cu"))


def reference_argsort(keys, *, _force_depth=None):
    """int64 argsort indices matching np.argsort(kind='quicksort') on the
    NumPy 1.26.4 scalar finite-key path, including all tie orders.

    ``_force_depth`` is an internal test seam: when given, the introsort
    recursion-depth counter is overridden (e.g. -1 forces the aheapsort
    fallback immediately); the public API does not expose depth control.
    """
    torch = require_torch()
    keys = _check_keys(keys)
    n = keys.numel()
    device = keys.device
    if n == 0:
        return torch.empty(0, dtype=torch.int64, device=device)
    if n == 1:
        return torch.arange(1, dtype=torch.int64, device=device)
    idx = torch.empty(n, dtype=torch.int64, device=device)
    depth = _DEPTH_KEEP if _force_depth is None else int(_force_depth)
    with _get_funcs(device) as funcs:
        rtc.launch(device, funcs("ref_init_index"), (n + 255) // 256, 256,
                   [("ptr", idx.data_ptr()), ("i64", n)], keepalive=(keys, idx))
        rtc.launch(device, funcs("ref_argsort"), 1, 1,
                   [("ptr", keys.data_ptr()), ("ptr", idx.data_ptr()),
                    ("i64", n), ("i32", depth)], keepalive=(keys, idx))
    return idx


def fused_project(rows, unit):
    """out[i] = fma(r2,u2, fma(r1,u1, r0*u0)) — one correctly-rounded
    double fma chain per row, matching the frozen CPU projections."""
    torch = require_torch()
    rows = torch.as_tensor(rows)
    unit = torch.as_tensor(unit)
    if not isinstance(rows, torch.Tensor) or not rows.is_cuda:
        raise GpuError("fused_project requires CUDA/HIP device tensors (no CPU path)")
    if not unit.is_cuda:
        raise GpuError("fused_project requires CUDA/HIP device tensors (no CPU path)")
    if rows.device != unit.device:
        raise GpuError(
            f"fused_project requires rows and unit on the same device, got {rows.device} / {unit.device}"
        )
    if rows.dtype != torch.float64 or unit.dtype != torch.float64:
        raise GpuError("fused_project requires float64 tensors")
    if rows.ndim != 2 or rows.shape[1] != 3 or unit.shape != (3,):
        raise GpuError(
            f"fused_project expects (N,3) rows and (3,) unit, got {tuple(rows.shape)} / {tuple(unit.shape)}"
        )
    if not bool(torch.isfinite(rows).all()) or not bool(torch.isfinite(unit).all()):
        raise GpuError("fused_project requires finite values")
    rows = rows.contiguous()
    unit = unit.contiguous()
    n = rows.shape[0]
    out = torch.empty(n, dtype=torch.float64, device=rows.device)
    if n == 0:
        return out
    with _get_funcs(rows.device) as funcs:
        rtc.launch(rows.device, funcs("ref_fused_project"), (n + 255) // 256, 256,
                   [("ptr", rows.data_ptr()), ("ptr", unit.data_ptr()),
                    ("i64", n), ("ptr", out.data_ptr())],
                   keepalive=(rows, unit, out))
    return out
