"""GPU bilinear + Malvar-He-Cutler demosaic (65535-code float32 RGB, no clip).

Mirrors BilinearDemosaic.apply_bilinear, malvar_he_cutler.MAL, and
Demosaic.compute_linear: scipy 'mirror' convolutions with f64
accumulation cast back to f32 at num/den before division (bilinear);
signal.correlate2d f32 row grouping (4 taps + remainder, primitive f32,
no FMA) for MAL; periodic 2x2-tile denominators when H,W>=2; zero-edge
neighbour presence; exact restore of known CFA sites. Remaining
variants are named unported.
"""

from __future__ import annotations

from .demosaic_directional import hamilton_adams, ppg
from .demosaic_vng import vng
from .demosaic_lmmse import lmmse
from .demosaic_ahd import ahd
from ..types import GpuError, require_torch
from ..ops.array import checked_float32
from ..ops.borders import pad
from ..ops.filters import convolve
from .raw import _layout


def _interp(channel, mask, kernel, torch):
    """num/den f32 division; scipy mirror mode; tile fast path for H,W>=2."""
    num = convolve(channel, kernel, mode="mirror").to(torch.float32)
    h, w = channel.shape[-2:]
    out = torch.zeros_like(num)
    if h >= 2 and w >= 2:
        tile = mask[:, :2, :2].to(torch.float32)
        den = convolve(tile, kernel, mode="mirror").to(torch.float32)
        for dy in range(2):
            for dx in range(2):
                d = den[0, dy, dx]
                view_n = num[:, dy::2, dx::2]
                view_o = out[:, dy::2, dx::2]
                d_safe = torch.where(d > 0, d, torch.ones_like(d))
                view_o[:] = torch.where(d > 0, view_n / d_safe,
                                        torch.zeros_like(view_o))
    else:
        den = convolve(mask.to(torch.float32), kernel, mode="mirror").to(torch.float32)
        d_safe = torch.where(den > 0, den, torch.ones_like(den))
        out = torch.where(den > 0, num / d_safe, out)
    return out


def _masks(shape, sensor, torch, device):
    layout = _layout(sensor)
    b, h, w = shape
    masks = {}
    for site in range(4):
        row, col = divmod(site, 2)
        m = torch.zeros(h, w, dtype=torch.bool, device=device)
        m[row::2, col::2] = True
        masks[layout[site]] = m
    mask_r = masks[0]
    mask_g = masks[1] | masks[2]
    mask_b = masks[3]
    return mask_r, mask_g, mask_b


def _bilinear(arr, mask_r, mask_g, mask_b, torch):
    b, h, w = arr.shape
    mr = mask_r.to(torch.float32)[None]
    mg = mask_g.to(torch.float32)[None]
    mb = mask_b.to(torch.float32)[None]
    r0 = arr * mr
    g0 = arr * mg
    b0 = arr * mb
    cross = torch.as_tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                            dtype=torch.float32, device=arr.device)
    horiz = torch.as_tensor([[0, 0, 0], [1, 0, 1], [0, 0, 0]],
                            dtype=torch.float32, device=arr.device) * 0.5
    vert = torch.as_tensor([[0, 1, 0], [0, 0, 0], [0, 1, 0]],
                           dtype=torch.float32, device=arr.device) * 0.5
    diag = torch.as_tensor([[1, 0, 1], [0, 0, 0], [1, 0, 1]],
                           dtype=torch.float32, device=arr.device) * 0.25
    g_at_rb = _interp(g0, mg, cross, torch)
    g = g0 + (1.0 - mg) * g_at_rb
    has_r_h = torch.zeros_like(mask_r)
    has_r_h[:, 1:] |= mask_r[:, :-1]
    has_r_h[:, :-1] |= mask_r[:, 1:]
    has_b_h = torch.zeros_like(mask_b)
    has_b_h[:, 1:] |= mask_b[:, :-1]
    has_b_h[:, :-1] |= mask_b[:, 1:]
    r_h = _interp(r0, mr, horiz, torch)
    r_v = _interp(r0, mr, vert, torch)
    r_d = _interp(r0, mr, diag, torch)
    b_h = _interp(b0, mb, horiz, torch)
    b_v = _interp(b0, mb, vert, torch)
    b_d = _interp(b0, mb, diag, torch)
    r = r0.clone()
    at_g = mask_g[None].expand(b, -1, -1)
    at_b = mask_b[None].expand(b, -1, -1)
    at_r = mask_r[None].expand(b, -1, -1)
    has_r_h_b = has_r_h[None].expand(b, -1, -1)
    has_b_h_b = has_b_h[None].expand(b, -1, -1)
    sel = at_g & has_r_h_b
    r[sel] = r_h[sel]
    sel = at_g & ~has_r_h_b
    r[sel] = r_v[sel]
    r[at_b] = r_d[at_b]
    bb = b0.clone()
    sel = at_g & has_b_h_b
    bb[sel] = b_h[sel]
    sel = at_g & ~has_b_h_b
    bb[sel] = b_v[sel]
    bb[at_r] = b_d[at_r]
    return torch.stack((r, g, bb), dim=-1)


def _malvar_correlate(raw, kernel, torch):
    """signal.correlate2d 'symm' f32 row grouping: (((w0x0+w1x1)+w2x2)+w3x3),
    then += w4*x4, five kernel rows ascending, primitive f32, no FMA."""
    k = torch.as_tensor(kernel, dtype=torch.float32, device=raw.device)
    if k.shape != (5, 5):
        raise GpuError(f"_malvar_correlate requires a 5x5 kernel, got {tuple(k.shape)}")
    h, w = raw.shape[-2:]
    ext = pad(raw, 2, "reflect")
    out = torch.zeros_like(raw)
    for i in range(5):
        acc = None
        for j in range(4):
            p = k[i, j] * ext[..., i:i + h, j:j + w]
            if j == 0:
                acc = p
            else:
                acc = acc + p
        out = out + acc + (k[i, 4] * ext[..., i:i + h, 4:4 + w])
    return out


def _malvar(arr, mask_r, mask_g, mask_b, torch):
    g_at_rb = torch.as_tensor(
        [[0, 0, -1, 0, 0], [0, 0, 2, 0, 0], [-1, 2, 4, 2, -1],
         [0, 0, 2, 0, 0], [0, 0, -1, 0, 0]],
        dtype=torch.float32, device=arr.device) * 0.125
    h_k = torch.as_tensor(
        [[0, 0, 0.5, 0, 0], [0, -1, 0, -1, 0], [-1, 4, 5, 4, -1],
         [0, -1, 0, -1, 0], [0, 0, 0.5, 0, 0]],
        dtype=torch.float32, device=arr.device) * 0.125
    v_k = h_k.t().contiguous()
    d_k = torch.as_tensor(
        [[0, 0, -1.5, 0, 0], [0, 2, 0, 2, 0], [-1.5, 0, 6, 0, -1.5],
         [0, 2, 0, 2, 0], [0, 0, -1.5, 0, 0]],
        dtype=torch.float32, device=arr.device) * 0.125
    r = arr * mask_r.to(arr.dtype)[None]
    g = arr * mask_g.to(arr.dtype)[None]
    b = arr * mask_b.to(arr.dtype)[None]
    g_corr = _malvar_correlate(arr, g_at_rb, torch)
    g = torch.where((mask_r | mask_b)[None], g_corr, g)
    h_corr = _malvar_correlate(arr, h_k, torch)
    v_corr = _malvar_correlate(arr, v_k, torch)
    d_corr = _malvar_correlate(arr, d_k, torch)
    r_rows = mask_r.any(dim=1)[:, None]
    r_cols = mask_r.any(dim=0)[None, :]
    b_rows = mask_b.any(dim=1)[:, None]
    b_cols = mask_b.any(dim=0)[None, :]
    rr_bc = r_rows & b_cols
    br_rc = b_rows & r_cols
    bb_bc = b_rows & b_cols
    rr_rc = r_rows & r_cols
    r = torch.where(rr_bc[None], h_corr, r)
    r = torch.where(br_rc[None], v_corr, r)
    r = torch.where(bb_bc[None], d_corr, r)
    b = torch.where(br_rc[None], h_corr, b)
    b = torch.where(rr_bc[None], v_corr, b)
    b = torch.where(rr_rc[None], d_corr, b)
    return torch.stack((r, g, b), dim=-1)


def demosaic(image, sensor: dict, params: dict):
    """BHW code tensor -> BHWC3 float32 65535-domain RGB."""
    torch = require_torch()
    algo = params.get("algorithm", "malvar")
    if algo not in ("bilinear", "malvar", "hamilton_adams", "hamilton_adams_opt",
            "ppg", "ppg_opt", "vng", "vng_opt",
            "lmmse", "lmmse_opt", "lmmse_fast", "ahd", "ahd_opt"):
        raise ValueError(f"Unknown demosaic algorithm: {algo}")
    raw_arr = torch.as_tensor(image)
    if raw_arr.ndim != 3 or not raw_arr.numel():
        raise GpuError(f"demosaic expects nonempty BHW input, got {tuple(raw_arr.shape)}")
    arr = checked_float32(raw_arr, name="demosaic")
    b, h, w = arr.shape
    mask_r, mask_g, mask_b = _masks(arr.shape, sensor, torch, arr.device)
    if algo == "bilinear":
        out = _bilinear(arr, mask_r, mask_g, mask_b, torch)
    elif algo == "malvar":
        out = _malvar(arr, mask_r, mask_g, mask_b, torch)
    elif algo in ("hamilton_adams", "hamilton_adams_opt"):
        out = hamilton_adams(arr, mask_r, mask_g, mask_b,
                             optimized=(algo == "hamilton_adams_opt"))
    elif algo in ("ppg", "ppg_opt"):
        out = ppg(arr, mask_r, mask_g, mask_b,
                  optimized=(algo == "ppg_opt"))
    elif algo in ("vng", "vng_opt"):
        out = vng(arr, mask_r, mask_g, mask_b,
                  optimized=(algo == "vng_opt"))
    elif algo in ("lmmse", "lmmse_opt", "lmmse_fast"):
        variant = {"lmmse": "basic", "lmmse_opt": "optimized",
                   "lmmse_fast": "fast"}[algo]
        out = lmmse(arr, mask_r, mask_g, mask_b, variant=variant)
    else:
        out = ahd(arr, mask_r, mask_g, mask_b,
                  optimized=(algo == "ahd_opt"))
    out = checked_float32(out, name=f"demosaic {algo}")
    src = arr
    out[..., 0][mask_r.expand(b, -1, -1)] = src[mask_r.expand(b, -1, -1)]
    out[..., 1][mask_g.expand(b, -1, -1)] = src[mask_g.expand(b, -1, -1)]
    out[..., 2][mask_b.expand(b, -1, -1)] = src[mask_b.expand(b, -1, -1)]
    return out
