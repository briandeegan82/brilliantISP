"""GPU native Scale block: hardware + non-hardware 2-D scaling.

Mirrors modules.scale Scale.compute_linear: checked-f32 HWC3/BHWC3,
per-frame Scale2D workflow (OpenCV-ratio NN, fused-lerp bilinear,
hardware recipe integer-reduce/crop/rational steps).
"""

from __future__ import annotations

import math

import numpy as np

from ..types import GpuError, require_torch
from ..ops.array import checked_float32
from ..ops.scale_geometry import HARDWARE_RECIPES


def _centers(n_in, n_out, torch, device):
    return (torch.arange(n_out, dtype=torch.float64, device=device) + 0.5) \
        * (n_in / n_out) - 0.5


def _coords(n_in, n_out, torch, device):
    c = _centers(n_in, n_out, torch, device).clamp(0.0, float(n_in - 1))
    i0 = torch.floor(c).to(torch.int64)
    i1 = torch.clamp(i0 + 1, max=n_in - 1)
    frac = (c - i0.to(torch.float64)).to(torch.float32)
    return i0, i1, frac


def _axis_linear(x, axis, out_n, fused=True):
    torch = require_torch()
    n = x.shape[axis]
    if n == out_n:
        return x
    i0, i1, frac = _coords(n, out_n, torch, x.device)
    a = x.index_select(axis, i0)
    b = x.index_select(axis, i1)
    delta = b - a
    shape = [1] * x.ndim
    shape[axis] = out_n
    f = frac.view(shape)
    if fused:
        return (a.to(torch.float64)
                + delta.to(torch.float64) * f.to(torch.float64)
                ).to(torch.float32)
    return a + delta * f


def _axis_linear_generic(x, axis, out_n):
    torch = require_torch()
    n = x.shape[axis]
    if n == out_n:
        return x
    c = ((torch.arange(out_n, dtype=torch.float64, device=x.device) + 0.5)
         * (1.0 / (out_n / n)) - 0.5).to(torch.float32)
    i0 = torch.floor(c).to(torch.int64)
    frac = c - i0.to(torch.float32)
    if axis in (-1, x.ndim - 1):
        frac = torch.where((i0 < 0) | (i0 >= n - 1),
                           torch.zeros_like(frac), frac)
    i1 = torch.clamp(i0 + 1, 0, n - 1)
    i0 = i0.clamp(0, n - 1)
    a = x.index_select(axis, i0)
    b = x.index_select(axis, i1)
    shape = [1] * x.ndim
    shape[axis] = out_n
    f = frac.view(shape)
    return a * (1.0 - f) + b * f


def _linear_generic_f32(x, out_h, out_w):
    x = _axis_linear_generic(x, -1, out_w)
    return _axis_linear_generic(x, -2, out_h)


def _linear_native_f32(x, out_h, out_w):
    torch = require_torch()
    if x.shape[-2] == 1 or x.shape[-1] == 1:
        return _linear_generic_f32(x, out_h, out_w)
    fused = _axis_linear(_axis_linear(x, -1, out_w, fused=True), -2, out_h,
                         fused=True)
    n_in_w, n_in_h = x.shape[-1], x.shape[-2]
    cx = _centers(n_in_w, out_w, torch, x.device)
    cy = _centers(n_in_h, out_h, torch, x.device)
    xi = (cx >= 0.0) & (cx < float(n_in_w - 1))
    count = int(xi.sum())
    if count == 0:
        return fused
    first = int(xi.to(torch.int64).argmax())
    tail_start = first + (count // 4) * 4
    if tail_start >= out_w:
        return fused
    xtail = xi & (torch.arange(out_w, device=x.device) >= tail_start)
    if not bool(xtail.any()):
        return fused
    yin = (cy >= 0.0) & (cy < float(n_in_h - 1))
    plain = _axis_linear(_axis_linear(x, -1, out_w, fused=False), -2, out_h,
                         fused=False)
    mask = yin[:, None] & xtail[None, :]
    return torch.where(mask, plain, fused)


def _axis_nn(x, axis, out_n):
    torch = require_torch()
    n = x.shape[axis]
    scale = 1.0 / (out_n / n)
    idx = torch.floor(
        torch.arange(out_n, dtype=torch.float64, device=x.device) * scale
    ).to(torch.int64).clamp(0, n - 1)
    return x.index_select(axis, idx)


def _nn_native(x, out_h, out_w):
    return _axis_nn(_axis_nn(x, -1, out_w), -2, out_h)


def _axis_box(x, axis, target):
    torch = require_torch()
    n = x.shape[axis]
    box = int(math.ceil(1.0 / (target / n)))
    cnt = (n - box) // box + 1
    f = x.to(torch.float64)
    sl = [slice(None)] * x.ndim
    w = torch.as_tensor(1.0 / box, dtype=torch.float64, device=x.device)
    acc = None
    for u in range(box):
        sl[axis] = slice(u, None, box)
        part = f[tuple(sl)].narrow(axis, 0, cnt) * w
        acc = part if acc is None else acc + part
    return acc.to(torch.float32)


def _box_reduce(x, target_h, target_w):
    torch = require_torch()
    bh = int(math.ceil(1.0 / (target_h / x.shape[-2])))
    bw = int(math.ceil(1.0 / (target_w / x.shape[-1])))
    nh = (x.shape[-2] - bh) // bh + 1
    nw = (x.shape[-1] - bw) // bw + 1
    f = x.to(torch.float64)
    w_t = torch.as_tensor(1.0 / (bh * bw), dtype=torch.float64, device=x.device)
    acc = torch.zeros((*x.shape[:-2], nh, nw), dtype=torch.float64, device=x.device)
    for u in range(bh):
        for v in range(bw):
            acc = acc + f[..., u::bh, v::bw][..., :nh, :nw] * w_t
    return acc.to(torch.float32)


def _axis_nn_down(x, axis, target):
    torch = require_torch()
    n = x.shape[axis]
    k = int(n / target)
    cnt = (n - k) // k + 1
    sl = [slice(None)] * x.ndim
    sl[axis] = slice(0, None, k)
    return x[tuple(sl)].narrow(axis, 0, cnt).to(torch.float32)


def _crop(x, rows, cols):
    if rows:
        if rows % 2 == 0:
            x = x[..., rows // 2:-rows // 2, :]
        else:
            x = x[..., :-1, :]
    if cols:
        if cols % 2 == 0:
            x = x[..., cols // 2:-cols // 2]
        else:
            x = x[..., :-1]
    return x


def _rational_axis(x, axis, fact, up_method, down_method):
    torch = require_torch()
    n, d = fact
    if up_method == "Bilinear":
        targets = list(x.shape[-2:])
        targets[axis % x.ndim - x.ndim + 2] = n * x.shape[axis]
        x = _linear_native_f32(x, targets[0], targets[1])
    else:
        x = _axis_nn(x, axis, n * x.shape[axis])
    cur = x.shape[axis]
    if axis in (-2, x.ndim - 2):
        tgt = int(np.round(cur / d))
    else:
        tgt = int(np.round(cur // d))
    if down_method == "Bilinear":
        return _axis_box(x, axis, tgt)
    return _axis_nn_down(x, axis, tgt)


def _scale2d(planes, recipe, up_method, down_method):
    f_h, c_h, r_h = recipe[0]
    f_w, c_w, r_w = recipe[1]
    if f_h > 1 or f_w > 1:
        planes = _box_reduce(
            planes, planes.shape[-2] // f_h, planes.shape[-1] // f_w)
    planes = _crop(planes, c_h, c_w)
    if r_h:
        planes = _rational_axis(planes, -2, r_h, up_method, down_method)
    if r_w:
        planes = _rational_axis(planes, -1, r_w, up_method, down_method)
    return planes


def scale(rgb, sensor_info, params):
    """HWC3/BHWC3 -> checked f32 target geometry, per-frame."""
    torch = require_torch()
    if not params.get("is_enable"):
        return checked_float32(torch.as_tensor(rgb), name="scale")
    arr = checked_float32(torch.as_tensor(rgb), name="scale input")
    if arr.ndim not in (3, 4) or arr.shape[-1] != 3 or not arr.numel():
        raise ValueError(f"scale input must be HWC3 or BHWC3, got {tuple(arr.shape)}")
    if not bool(torch.isfinite(arr).all()):
        raise ValueError("scale input must be finite")
    h, w = arr.shape[-3], arr.shape[-2]
    if int(sensor_info["height"]) != h or int(sensor_info["width"]) != w:
        raise ValueError(
            f"scale sensor {sensor_info['height']}x{sensor_info['width']} "
            f"does not match input {h}x{w}")
    for name in ("new_height", "new_width"):
        value = params[name]
        if isinstance(value, bool) or not isinstance(
                value, (int, float, np.integer, np.floating)) \
                or not np.isfinite(value) or value <= 0 or int(value) != value:
            raise ValueError(f"scale.{name} must be a positive integer")
    new_h, new_w = int(params["new_height"]), int(params["new_width"])
    planes = arr.movedim(-1, -3).reshape(-1, h, w)
    if params.get("is_hardware"):
        recipe = HARDWARE_RECIPES.get((h, w), {}).get((new_h, new_w))
        out = planes if recipe is None else _scale2d(
            planes, recipe,
            params.get("upscale_method"), params.get("downscale_method"))
    elif params.get("algorithm") == "Bilinear":
        out = _linear_native_f32(planes, new_h, new_w)
    else:
        out = _nn_native(planes, new_h, new_w)
    if out.shape[-2:] != (new_h, new_w):
        raise ValueError(
            f"scale output {tuple(out.shape[-2:])} does not match "
            f"declared {(new_h, new_w)}")
    out = out.reshape(*arr.shape[:-3], 3, new_h, new_w).movedim(-3, -1)
    return checked_float32(out, name="scale")
