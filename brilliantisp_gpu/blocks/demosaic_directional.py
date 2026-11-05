"""GPU Hamilton-Adams demosaic (basic + optimized variants).

Mirrors modules.demosaic.hamilton_adams_demosaic: scipy 'reflect'
(edge-repeated) convolutions cast to float32 after every filter,
directional green interpolation, color-difference R/B reconstruction.
Internal BHW float32 -> BHWC3 float32; callers restore known CFA sites.
"""

from __future__ import annotations

from ..types import require_torch
from ..ops.filters import convolve

_H3 = ((0, 0, 0), (.5, 0, .5), (0, 0, 0))
_D3 = ((.25, 0, .25), (0, 0, 0), (.25, 0, .25))
_HC5 = ((0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (-.25, 0, .5, 0, -.25),
        (0, 0, 0, 0, 0), (0, 0, 0, 0, 0))
_HI5 = ((0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, .5, 0, .5, 0),
        (0, 0, 0, 0, 0), (0, 0, 0, 0, 0))
_GH5 = ((0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (1, 0, -2, 0, 1),
        (0, 0, 0, 0, 0), (0, 0, 0, 0, 0))
_CROSS3 = ((0, .25, 0), (.25, 0, .25), (0, .25, 0))
_GRAD3H = ((0, 0, 0), (1, -2, 1), (0, 0, 0))
_PPG_GH5 = ((0, 0, 0, 0, 0), (0, 0, 1, 0, 0), (1, -2, 0, -2, 1),
            (0, 0, 1, 0, 0), (0, 0, 0, 0, 0))


def _k(rows, torch, device):
    return torch.as_tensor(rows, dtype=torch.float32, device=device)


def _conv32(x, kernel, torch):
    return convolve(x, kernel, mode="reflect").to(torch.float32)


def hamilton_adams(raw, mask_r, mask_g, mask_b, *, optimized=False):
    """BHW f32 raw + HW bool masks -> BHWC3 f32."""
    torch = require_torch()
    b = raw.shape[0]
    dev = raw.device
    eps = torch.as_tensor(1e-6, dtype=torch.float32, device=dev)
    h3 = _k(_H3, torch, dev)
    v3 = h3.T
    d3 = _k(_D3, torch, dev)
    mg = mask_g.to(torch.float32)[None].expand(b, -1, -1)
    known_g = raw * mg
    if optimized:
        h_i = _k(_HI5, torch, dev)
        v_i = h_i.T
        h_g = _k(_GH5, torch, dev)
        v_g = h_g.T
        gh = _conv32(known_g, h_i, torch)
        gv = _conv32(known_g, v_i, torch)
        grad_h = _conv32(raw, h_g, torch).abs()
        grad_v = _conv32(raw, v_g, torch).abs()
        eh = gh + 0.5 * (raw - _conv32(raw, h_i, torch))
        ev = gv + 0.5 * (raw - _conv32(raw, v_i, torch))
        interp = torch.where(grad_h < grad_v, eh, ev)
        ratio = grad_h / (grad_v + eps)
        similar = (ratio > 0.8) & (ratio < 1.25)
        interp = torch.where(similar, (eh + ev) * 0.5, interp)
        g = known_g + (1.0 - mg) * interp
    else:
        hc5 = _k(_HC5, torch, dev)
        vc5 = hc5.T
        gh = _conv32(known_g, h3, torch)
        gv = _conv32(known_g, v3, torch)
        g = known_g
        for mask_c in (mask_r, mask_b):
            mc = mask_c.to(torch.float32)[None].expand(b, -1, -1)
            plane = raw * mc
            ch = _conv32(plane, hc5, torch)
            cv = _conv32(plane, vc5, torch)
            eh = gh + ch
            ev = gv + cv
            ah = ch.abs()
            av = cv.abs()
            interp = torch.where(ah <= av, eh, ev)
            ratio = ah / (av + eps)
            similar = (ratio > 1 / 1.2) & (ratio < 1.2)
            interp = torch.where(similar, (eh + ev) * 0.5, interp)
            g = torch.where(mask_c[None].expand(b, -1, -1), interp, g)
        g = torch.clamp_min(g, 0.0)
    return _color_difference_rgb(raw, g, mask_r, mask_g, mask_b)


def _color_difference_rgb(raw, g, mask_r, mask_g, mask_b):
    torch = require_torch()
    b = raw.shape[0]
    h3 = _k(_H3, torch, raw.device)
    v3 = h3.T
    d3 = _k(_D3, torch, raw.device)
    channels = []
    for mask_c, mask_other in ((mask_r, mask_b), (mask_b, mask_r)):
        mc = mask_c.to(torch.float32)[None].expand(b, -1, -1)
        diff = (raw - g) * mc
        dh = _conv32(diff, h3, torch)
        dv = _conv32(diff, v3, torch)
        dd = _conv32(diff, d3, torch)
        has_h = _conv32(mc, h3 * 2.0, torch) > 0.5
        c = torch.where(mask_g[None].expand(b, -1, -1),
                        torch.where(has_h, g + dh, g + dv),
                        raw * mc)
        c = torch.where(mask_other[None].expand(b, -1, -1), g + dd, c)
        channels.append(c)
    return torch.stack((channels[0], g, channels[1]), dim=-1)


def ppg(raw, mask_r, mask_g, mask_b, *, optimized=False):
    """BHW f32 raw + HW bool masks -> BHWC3 f32."""
    torch = require_torch()
    b = raw.shape[0]
    dev = raw.device
    eps = torch.as_tensor(1e-6, dtype=torch.float32, device=dev)
    h3 = _k(_H3, torch, dev)
    v3 = h3.T
    cross3 = _k(_CROSS3, torch, dev)
    d3 = _k(_D3, torch, dev)
    mg = mask_g.to(torch.float32)[None].expand(b, -1, -1)
    mr = mask_r[None].expand(b, -1, -1)
    mb = mask_b[None].expand(b, -1, -1)
    known_g = raw * mg
    gh = _conv32(known_g, h3, torch)
    gv = _conv32(known_g, v3, torch)
    if optimized:
        gh5 = _k(_PPG_GH5, torch, dev)
        gv5 = gh5.T
        grad_h = _conv32(raw, gh5, torch).abs()
        grad_v = _conv32(raw, gv5, torch).abs()
        eh = gh + 0.5 * (raw - _conv32(raw, h3, torch))
        ev = gv + 0.5 * (raw - _conv32(raw, v3, torch))
        interp = torch.where(grad_h < grad_v, eh, ev)
        ratio = grad_h / (grad_v + eps)
        similar = (ratio > 0.75) & (ratio < 1.33)
        interp = torch.where(similar, (eh + ev) * 0.5, interp)
        g = known_g + (1.0 - mg) * interp
        for i in range(2):
            rdiff = (raw - g) * mr.to(torch.float32)
            bdiff = (raw - g) * mb.to(torch.float32)
            rc = _conv32(rdiff, cross3, torch)
            rd = _conv32(rdiff, d3, torch)
            bc = _conv32(bdiff, cross3, torch)
            bd = _conv32(bdiff, d3, torch)
            corr_r = -0.25 * (bc + bd)
            corr_b = -0.25 * (rc + rd)
            damp = 0.5 / (i + 1)
            g = torch.where(mr, g + damp * corr_r, g)
            g = torch.where(mb, g + damp * corr_b, g)
    else:
        grad_h = _conv32(raw, _k(_GRAD3H, torch, dev), torch).abs()
        grad_v = _conv32(raw, _k(_GRAD3H, torch, dev).T, torch).abs()
        interp = torch.where(grad_h < grad_v, gh, gv)
        ratio = grad_h / (grad_v + eps)
        similar = (ratio > 0.8) & (ratio < 1.25)
        interp = torch.where(similar, (gh + gv) * 0.5, interp)
        g = known_g + (1.0 - mg) * interp
        g0 = g
        rdiff = (raw - g0) * mr.to(torch.float32)
        bdiff = (raw - g0) * mb.to(torch.float32)
        ri = _conv32(rdiff, cross3, torch)
        bi = _conv32(bdiff, cross3, torch)
        g = torch.where(mr, g0 + (-0.5 * bi), g0)
        g = torch.where(mb, g + (-0.5 * ri), g)
    return _color_difference_rgb(raw, g, mask_r, mask_g, mask_b)
