"""float32 INTER_AREA downsample via portable CUDA/HIP RTC kernels.

Replicates OpenCV 4.11 INTER_AREA arithmetic exactly: CSR tap tables,
integer fast path (balanced 2x2 / groups-of-4 accumulation), general
horizontal-then-vertical weighted sums. GPU-only, no CPU fallback.
"""

from __future__ import annotations

import math

import numpy as np

from ..types import GpuError, require_torch
from . import rtc

_DBL_EPS = float(np.finfo(np.float64).eps)


def _tables(n_in, n_out):
    scale = 1.0 / (n_out / n_in)
    indices = []
    weights = []
    offsets = [0]
    for o in range(n_out):
        lo = o * scale
        hi = lo + scale
        width = min(scale, n_in - lo)
        left = math.ceil(lo)
        right = min(math.floor(hi), n_in - 1)
        left = min(left, right)
        if left - lo > 1e-3:
            indices.append(left - 1)
            weights.append(np.float32((left - lo) / width))
        for i in range(left, right):
            indices.append(i)
            weights.append(np.float32(1.0 / width))
        if hi - right > 1e-3:
            indices.append(right)
            weights.append(
                np.float32(min(min(hi - right, 1.0), width) / width))
        offsets.append(len(indices))
    return (np.asarray(offsets, np.int64), np.asarray(indices, np.int64),
            np.asarray(weights, np.float32))


def _funcs(device):
    return rtc.get_module(
        device, rtc.load_source("resize_area_float32.cu"),
        func_names=("area_f32_fast", "area_f32_generic"))


def area_float32(x, out_h, out_w):
    """INTER_AREA downsample on the last two axes of a CUDA/HIP f32 tensor."""
    torch = require_torch()
    arr = torch.as_tensor(x)
    if not isinstance(arr, torch.Tensor) or not arr.is_cuda:
        raise GpuError(
            "area_float32 requires CUDA/HIP device tensors (no CPU path)")
    if arr.dtype != torch.float32:
        raise GpuError(
            f"area_float32 requires float32 tensors, got {arr.dtype}")
    if arr.ndim < 2 or not arr.numel():
        raise GpuError(
            f"area_float32 expects nonempty tensors with >=2 dims, "
            f"got {tuple(arr.shape)}")
    if not bool(torch.isfinite(arr).all()):
        raise GpuError("area_float32 requires finite values")
    for name, v in (("out_h", out_h), ("out_w", out_w)):
        if isinstance(v, bool) or not isinstance(
                v, (int, np.integer)) or int(v) != v or v <= 0:
            raise ValueError(f"area_float32 {name} must be a positive integer")
    out_h, out_w = int(out_h), int(out_w)
    h, w = arr.shape[-2], arr.shape[-1]
    if out_h > h or out_w > w:
        raise ValueError(
            f"area_float32 only downsamples: target {(out_h, out_w)} "
            f"exceeds input {(h, w)}")
    src = arr.contiguous()
    if (out_h, out_w) == (h, w):
        return src.clone()
    lead = src.shape[:-2]
    batch = int(np.prod(lead)) if lead else 1
    flat = src.reshape(batch, h, w)
    out = torch.empty((batch, out_h, out_w), dtype=torch.float32,
                      device=src.device)
    n_out = batch * out_h * out_w
    grid = (n_out + 255) // 256
    sy = 1.0 / (out_h / h)
    sx = 1.0 / (out_w / w)
    iy, ix = round(sy), round(sx)
    keep = (flat, out)
    with _funcs(src.device) as funcs:
        if abs(sy - iy) < _DBL_EPS and abs(sx - ix) < _DBL_EPS:
            norm = torch.as_tensor(
                np.float32(1.0 / (iy * ix)), device=src.device)
            keep = keep + (norm,)
            rtc.launch(src.device, funcs("area_f32_fast"), grid, 256,
                       [("ptr", flat.data_ptr()), ("ptr", out.data_ptr()),
                        ("i64", batch), ("i64", h), ("i64", w),
                        ("i64", out_h), ("i64", out_w),
                        ("i64", iy), ("i64", ix),
                        ("ptr", norm.data_ptr())],
                       keepalive=keep)
        else:
            yo, yi, yw = _tables(h, out_h)
            xo, xi, xw = _tables(w, out_w)
            dev = src.device
            yo = torch.as_tensor(yo, device=dev)
            yi = torch.as_tensor(yi, device=dev)
            yw = torch.as_tensor(yw, device=dev)
            xo = torch.as_tensor(xo, device=dev)
            xi = torch.as_tensor(xi, device=dev)
            xw = torch.as_tensor(xw, device=dev)
            keep = keep + (yo, yi, yw, xo, xi, xw)
            rtc.launch(src.device, funcs("area_f32_generic"), grid, 256,
                       [("ptr", flat.data_ptr()), ("ptr", out.data_ptr()),
                        ("i64", batch), ("i64", h), ("i64", w),
                        ("i64", out_h), ("i64", out_w),
                        ("ptr", yo.data_ptr()), ("ptr", yi.data_ptr()),
                        ("ptr", yw.data_ptr()),
                        ("ptr", xo.data_ptr()), ("ptr", xi.data_ptr()),
                        ("ptr", xw.data_ptr())],
                       keepalive=keep)
    return out.reshape(*lead, out_h, out_w)
