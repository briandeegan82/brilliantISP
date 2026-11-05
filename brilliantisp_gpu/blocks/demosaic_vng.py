"""GPU VNG demosaic (basic gradient-selection + optimized variants).

Mirrors modules.demosaic.vng_demosaic: basic interpolates each unknown
color site by the mean of same-color samples from up to three selected
physical direction pairs; optimized uses masked convolutions only.
"""

from __future__ import annotations

from ..types import require_torch
from ..ops.borders import pad
from .demosaic_directional import _k, _conv32, _H3, _D3, _CROSS3

_PAIRS = (((-1, 0), (1, 0)), ((-1, 1), (1, -1)),
          ((0, -1), (0, 1)), ((-1, -1), (1, 1)))


def _selected_mean(a_slots, b_slots, count):
    torch = require_torch()
    dev = a_slots[0].device
    ab = a_slots[0] + b_slots[0]
    cd = a_slots[1] + b_slots[1]
    ef = a_slots[2] + b_slots[2]
    s1 = ((a_slots[0] + b_slots[0]) + a_slots[0]) + b_slots[0]
    s2 = (ab + cd) + (ab + cd)
    s3 = ((ab + cd) + (ef + ab)) + a_slots[1]
    s3 = (s3 + b_slots[1]) + a_slots[2]
    s3 = s3 + b_slots[2]
    zero = torch.zeros_like(s1)
    mean = torch.where(
        count == 1, s1 / torch.as_tensor(4.0, dtype=s1.dtype, device=dev),
        zero)
    mean = torch.where(
        count == 2, s2 / torch.as_tensor(8.0, dtype=s1.dtype, device=dev),
        mean)
    mean = torch.where(
        count == 3, s3 / torch.as_tensor(12.0, dtype=s1.dtype, device=dev),
        mean)
    return mean


def _vng_basic(raw, mask_r, mask_g, mask_b):
    torch = require_torch()
    b, h, w = raw.shape
    dev = raw.device
    inf = torch.as_tensor(float("inf"), dtype=torch.float64, device=dev)
    two = torch.as_tensor(2.0, dtype=torch.float32, device=dev)
    raw_pad = pad(raw, 1, mode="mirror")
    channels = []
    for mask_c in (mask_r, mask_g, mask_b):
        unknown = (~mask_c)[None].expand(b, -1, -1)
        mc_pad = pad(mask_c.to(torch.float32)[None].expand(b, -1, -1),
                     1, mode="mirror")
        grads = []
        samples = []
        present = []
        for (d1y, d1x), (d2y, d2x) in _PAIRS:
            a = raw_pad[:, 1 + d1y:1 + d1y + h, 1 + d1x:1 + d1x + w]
            c = raw_pad[:, 1 + d2y:1 + d2y + h, 1 + d2x:1 + d2x + w]
            pa = mc_pad[:, 1 + d1y:1 + d1y + h, 1 + d1x:1 + d1x + w] > 0
            pc = mc_pad[:, 1 + d2y:1 + d2y + h, 1 + d2x:1 + d2x + w] > 0
            p = pa & pc & unknown
            mean = (a + c) / two
            da = a - mean
            dc = c - mean
            var = (da * da + dc * dc) / two
            grad = (torch.sqrt(var) + (c - a).abs()).to(torch.float64)
            grads.append(torch.where(
                p & torch.isfinite(grad), grad, inf))
            samples.append((a, c))
            present.append(p)
        min_grad = torch.minimum(
            torch.minimum(grads[0], grads[1]),
            torch.minimum(grads[2], grads[3]))
        threshold = torch.where(torch.isfinite(min_grad), min_grad * 1.5,
                                torch.zeros_like(min_grad))
        a_slots = [torch.zeros_like(raw) for _ in range(3)]
        b_slots = [torch.zeros_like(raw) for _ in range(3)]
        rank = torch.zeros(b, h, w, dtype=torch.int64, device=dev)
        for d, ((a, c), p) in enumerate(zip(samples, present)):
            sel = (p & torch.isfinite(grads[d])
                   & (grads[d] <= threshold))
            for j in range(3):
                a_slots[j] = torch.where(sel & (rank == j), a, a_slots[j])
                b_slots[j] = torch.where(sel & (rank == j), c, b_slots[j])
            rank = rank + sel.to(torch.int64)
        mean = _selected_mean(a_slots, b_slots, rank)
        channels.append(torch.where(mask_c[None].expand(b, -1, -1), raw, mean))
    return torch.stack(channels, dim=-1)


def _vng_optimized(raw, mask_r, mask_g, mask_b):
    torch = require_torch()
    b = raw.shape[0]
    dev = raw.device
    h3 = _k(_H3, torch, dev)
    v3 = h3.T
    d3 = _k(_D3, torch, dev)
    cross3 = _k(_CROSS3, torch, dev)
    mg = mask_g.to(torch.float32)[None].expand(b, -1, -1)
    known_g = raw * mg
    g = known_g + (1.0 - mg) * _conv32(known_g, cross3, torch)
    channels = []
    for mask_c, mask_other in ((mask_r, mask_b), (mask_b, mask_r)):
        mc = mask_c.to(torch.float32)[None].expand(b, -1, -1)
        src = raw * mc
        ch = _conv32(src, h3, torch)
        cv = _conv32(src, v3, torch)
        cd = _conv32(src, d3, torch)
        has_h = _conv32(mc, h3 * 2.0, torch) > 0
        c = torch.where(mask_g[None].expand(b, -1, -1),
                        torch.where(has_h, ch, cv), src)
        c = torch.where(mask_other[None].expand(b, -1, -1), cd, c)
        channels.append(c)
    return torch.stack((channels[0], g, channels[1]), dim=-1)


def vng(raw, mask_r, mask_g, mask_b, *, optimized=False):
    """BHW f32 raw + HW bool masks -> BHWC3 f32."""
    if optimized:
        return _vng_optimized(raw, mask_r, mask_g, mask_b)
    return _vng_basic(raw, mask_r, mask_g, mask_b)
