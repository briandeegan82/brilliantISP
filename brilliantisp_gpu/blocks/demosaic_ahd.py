"""GPU AHD demosaic (basic + optimized variants).

Mirrors modules.demosaic.ahd_demosaic: directional green + color
difference R/B, homogeneity decision or smooth alpha blend, and
percentile-outlier median artifact removal per channel/frame.
"""

from __future__ import annotations

import math

from ..ops.reference_exp import reference_exp_float32
from ..types import require_torch
from ..ops.borders import pad
from .demosaic_directional import (_k, _conv32, _H3, _D3,
                                   _color_difference_rgb)

_H5 = ((0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (.25, .5, 0, .5, .25),
       (0, 0, 0, 0, 0), (0, 0, 0, 0, 0))
_DC5 = ((.125, 0, 0, 0, .125), (0, .25, 0, .25, 0), (0, 0, 0, 0, 0),
        (0, .25, 0, .25, 0), (.125, 0, 0, 0, .125))
_GRAD1X3 = ((-1.0, 0.0, 1.0),)


def _ones_div(window, torch, dev):
    return torch.ones((window, window), dtype=torch.float32, device=dev) \
        / torch.as_tensor(float(window * window), dtype=torch.float32,
                          device=dev)


def _median3(plane):
    torch = require_torch()
    ext = pad(plane, 1, mode="reflect")
    h, w = plane.shape[-2], plane.shape[-1]
    taps = []
    for dy in range(3):
        for dx in range(3):
            taps.append(ext[..., dy:dy + h, dx:dx + w])
    return torch.median(torch.stack(taps, dim=-1), dim=-1).values


def _percentile_threshold(plane, q):
    torch = require_torch()
    b = plane.shape[0]
    flat = plane.reshape(b, -1)
    n = flat.shape[1]
    vals, _ = torch.sort(flat, dim=1)
    rank = (n - 1) * (q / 100.0)
    lo = int(math.floor(rank))
    hi = min(lo + 1, n - 1)
    gamma = rank - lo
    a = vals[:, lo]
    c = vals[:, hi]
    delta = c - a
    if gamma < 0.5:
        thr = a.to(torch.float64) + delta.to(torch.float64) * gamma
    else:
        thr = c.to(torch.float64) - delta.to(torch.float64) * (1.0 - gamma)
    return thr.to(torch.float32).reshape(b, 1, 1)


def _basic_decision(rgb_h, rgb_v):
    torch = require_torch()
    diff_h = (rgb_h[..., 0] - rgb_h[..., 1]).abs() \
        + (rgb_h[..., 1] - rgb_h[..., 2]).abs()
    diff_v = (rgb_v[..., 0] - rgb_v[..., 1]).abs() \
        + (rgb_v[..., 1] - rgb_v[..., 2]).abs()
    k5 = _ones_div(5, torch, rgb_h.device)
    mean_h = _conv32(diff_h, k5, torch)
    mean_v = _conv32(diff_v, k5, torch)
    var_h = _conv32((diff_h - mean_h) ** 2, k5, torch)
    var_v = _conv32((diff_v - mean_v) ** 2, k5, torch)
    return var_h < var_v


def _opt_alpha(rgb_h, rgb_v):
    torch = require_torch()
    dev = rgb_h.device
    eps = torch.as_tensor(1e-6, dtype=torch.float32, device=dev)
    diff_h = (rgb_h[..., 0] - rgb_h[..., 1]).abs() \
        + (rgb_h[..., 1] - rgb_h[..., 2]).abs()
    diff_v = (rgb_v[..., 0] - rgb_v[..., 1]).abs() \
        + (rgb_v[..., 1] - rgb_v[..., 2]).abs()
    gk = _k(_GRAD1X3, torch, dev)
    grad_h = torch.zeros_like(diff_h)
    grad_v = torch.zeros_like(diff_v)
    for c in range(3):
        grad_h = grad_h + _conv32(rgb_h[..., c], gk, torch).abs()
        grad_v = grad_v + _conv32(rgb_v[..., c], gk, torch).abs()
    k7 = _ones_div(7, torch, dev)
    var_h = _conv32(diff_h ** 2, k7, torch)
    var_v = _conv32(diff_v ** 2, k7, torch)
    metric_h = 0.4 * var_h + 0.3 * diff_h + 0.3 * grad_h
    metric_v = 0.4 * var_v + 0.3 * diff_v + 0.3 * grad_v
    ratio = metric_h / (metric_v + eps)
    t = 5.0 * (ratio - 1.0)
    one = torch.ones_like(t)
    return one / (one + reference_exp_float32(t))


def _artifact(blended, *, optimized):
    torch = require_torch()
    dev = blended.device
    q = 98.0 if optimized else 99.0
    a, c_coef = (0.8, 0.2) if optimized else (0.7, 0.3)
    window = 5 if optimized else 3
    k = torch.ones((window, window), dtype=torch.float64, device=dev) \
        / torch.as_tensor(float(window * window), dtype=torch.float64,
                          device=dev)
    result = []
    for c in range(3):
        plane = blended[..., c]
        local = _conv32(plane, k, torch)
        diff = (plane - local).abs()
        threshold = _percentile_threshold(diff, q)
        outlier = diff > threshold
        med = _median3(plane)
        result.append(torch.where(outlier, a * plane + c_coef * med, plane))
    return torch.stack(result, dim=-1)


def _enhanced_rb(raw, g, mask_r, mask_g, mask_b):
    torch = require_torch()
    b = raw.shape[0]
    dev = raw.device
    h5 = _k(_H5, torch, dev)
    v5 = h5.T
    dc5 = _k(_DC5, torch, dev)
    channels = []
    for mask_c, mask_other in ((mask_r, mask_b), (mask_b, mask_r)):
        mc = mask_c.to(torch.float32)[None].expand(b, -1, -1)
        diff = (raw - g) * mc
        dh = _conv32(diff, h5, torch)
        dv = _conv32(diff, v5, torch)
        dd = _conv32(diff, dc5, torch)
        has_h = _conv32(mc, h5 * 2.0, torch) > 0.3
        c = torch.where(mask_g[None].expand(b, -1, -1),
                        torch.where(has_h, g + dh, g + dv),
                        raw * mc)
        c = torch.where(mask_other[None].expand(b, -1, -1), g + dd, c)
        channels.append(c)
    return torch.stack((channels[0], g, channels[1]), dim=-1)


def ahd(raw, mask_r, mask_g, mask_b, *, optimized=False):
    """BHW f32 raw + HW bool masks -> BHWC3 f32."""
    torch = require_torch()
    b = raw.shape[0]
    dev = raw.device
    mg = mask_g.to(torch.float32)[None].expand(b, -1, -1)
    mg_bool = mask_g[None].expand(b, -1, -1)
    known_g = raw * mg
    if optimized:
        h5 = _k(_H5, torch, dev)
        v5 = h5.T
        g_h = torch.where(
            mg_bool, known_g,
            _conv32(known_g, h5, torch)
            + 0.5 * (raw - _conv32(raw, h5, torch)))
        g_v = torch.where(
            mg_bool, known_g,
            _conv32(known_g, v5, torch)
            + 0.5 * (raw - _conv32(raw, v5, torch)))
        rgb_h = _enhanced_rb(raw, g_h, mask_r, mask_g, mask_b)
        rgb_v = _enhanced_rb(raw, g_v, mask_r, mask_g, mask_b)
        alpha = _opt_alpha(rgb_h, rgb_v)[..., None]
        blended = alpha * rgb_h + (1.0 - alpha) * rgb_v
    else:
        h3 = _k(_H3, torch, dev)
        v3 = h3.T
        g_h = torch.where(mg_bool, known_g, _conv32(known_g, h3, torch))
        g_v = torch.where(mg_bool, known_g, _conv32(known_g, v3, torch))
        rgb_h = _color_difference_rgb(raw, g_h, mask_r, mask_g, mask_b)
        rgb_v = _color_difference_rgb(raw, g_v, mask_r, mask_g, mask_b)
        blended = torch.where(
            _basic_decision(rgb_h, rgb_v)[..., None], rgb_h, rgb_v)
    return _artifact(blended, optimized=optimized)
