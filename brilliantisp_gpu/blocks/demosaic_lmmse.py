"""GPU LMMSE demosaic (basic / optimized / fast variants).

Mirrors modules.demosaic.lmmse_demosaic: directional green init,
local-mean/variance LMMSE (Wiener) refinement, window sizes 5/7/3.
"""

from __future__ import annotations

from ..types import require_torch
from .demosaic_directional import _k, _conv32, _H3, _GRAD3H, _PPG_GH5


def _ones(window, torch, dev):
    return torch.ones((window, window), dtype=torch.float32, device=dev)


def _local_stats(channel, mask, window, *, fast=False):
    torch = require_torch()
    eps = torch.as_tensor(1e-6, dtype=torch.float32, device=channel.device)
    if fast:
        k = _ones(3, torch, channel.device)
        count = _conv32(mask, k, torch)
        mean = _conv32(channel * mask, k, torch) / (count + eps)
        mean_sq = _conv32((channel ** 2) * mask, k, torch) / (count + eps)
        var = torch.clamp_min(mean_sq - mean ** 2, 0.0)
        return mean, var
    k = _ones(window, torch, channel.device) / torch.as_tensor(
        float(window * window), dtype=torch.float32, device=channel.device)
    count = _conv32(mask, k, torch)
    mean = _conv32(channel * mask, k, torch) / (count + eps)
    var = _conv32(((channel - mean) ** 2) * mask, k, torch) / (count + eps)
    return mean, var


def _lmmse_estimate(target, tm, reference, rm, window, *, fast=False,
                    confidence=False):
    torch = require_torch()
    dev = target.device
    eps = torch.as_tensor(1e-6, dtype=torch.float32, device=dev)
    mean_t, var_t = _local_stats(target, tm, window, fast=fast)
    mean_r, var_r = _local_stats(reference, rm, window, fast=fast)
    both = tm * rm
    if fast:
        k = _ones(3, torch, dev)
        product = (target * reference) * both
        mean_prod = _conv32(product, k, torch) / (_conv32(both, k, torch) + eps)
        cov = mean_prod - mean_t * mean_r
    else:
        k = _ones(window, torch, dev) / torch.as_tensor(
            float(window * window), dtype=torch.float32, device=dev)
        d1 = (target - mean_t) * both
        d2 = (reference - mean_r) * both
        cov = _conv32(d1 * d2, k, torch) / (_conv32(both, k, torch) + eps)
    gain = cov / (var_r + eps)
    estimate = mean_t + gain * (reference - mean_r)
    conf = None
    if confidence:
        one = torch.ones_like(var_t)
        conf = one / (one + var_t / (var_r + eps))
    return estimate, conf


def lmmse(raw, mask_r, mask_g, mask_b, *, variant="basic"):
    """BHW f32 raw + HW bool masks -> BHWC3 f32."""
    torch = require_torch()
    b = raw.shape[0]
    dev = raw.device
    eps = torch.as_tensor(1e-6, dtype=torch.float32, device=dev)
    h3 = _k(_H3, torch, dev)
    v3 = h3.T
    mg = mask_g.to(torch.float32)[None].expand(b, -1, -1)
    mr = mask_r[None].expand(b, -1, -1)
    mb = mask_b[None].expand(b, -1, -1)
    mrf = mr.to(torch.float32)
    mbf = mb.to(torch.float32)
    known = raw * mg
    gh = _conv32(known, h3, torch)
    gv = _conv32(known, v3, torch)
    fast = variant == "fast"
    if variant == "optimized":
        window = 7
        gh5 = _k(_PPG_GH5, torch, dev)
        gv5 = gh5.T
        grad_h = _conv32(raw, gh5, torch).abs()
        grad_v = _conv32(raw, gv5, torch).abs()
        direction = torch.where(grad_h < grad_v, gh, gv)
        ratio = grad_h / (grad_v + eps)
        similar = (ratio > 0.7) & (ratio < 1.43)
        direction = torch.where(similar, (gh + gv) * 0.5, direction)
        temp = known + (1.0 - mg) * direction
        est_r, conf_r = _lmmse_estimate(temp, mg, raw * mrf, mrf, window,
                                        confidence=True)
        est_b, conf_b = _lmmse_estimate(temp, mg, raw * mbf, mbf, window,
                                        confidence=True)
        one = torch.ones_like(conf_r)
        g = torch.where(mr, conf_r * est_r + (one - conf_r) * direction,
                        known)
        g = torch.where(mb, conf_b * est_b + (one - conf_b) * direction, g)
    else:
        window = 3 if fast else 5
        g3h = _k(_GRAD3H, torch, dev)
        grad_h = _conv32(raw, g3h, torch).abs()
        grad_v = _conv32(raw, g3h.T, torch).abs()
        direction = torch.where(grad_h < grad_v, gh, gv)
        combined = known + (1.0 - mg) * direction
        if fast:
            g = combined
        else:
            est_r, _ = _lmmse_estimate(combined, mg, raw * mrf, mrf, window)
            est_b, _ = _lmmse_estimate(combined, mg, raw * mbf, mbf, window)
            g = torch.where(mr, est_r, known)
            g = torch.where(mb, est_b, g)
    full = torch.ones_like(mg)
    channels = []
    for mask_c in (mask_r, mask_b):
        mc = mask_c.to(torch.float32)[None].expand(b, -1, -1)
        known_c = raw * mc
        est, _ = _lmmse_estimate(known_c, mc, g, full, window, fast=fast)
        channels.append(known_c + (1.0 - mc) * est)
    return torch.stack((channels[0], g, channels[1]), dim=-1)
