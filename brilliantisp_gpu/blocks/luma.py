"""GPU luma-domain sharpen and non-local means blocks.

Mirrors sharpen.unsharp_masking.apply_luma and
noise_reduction_2d.non_local_means.apply_luma on HW/BHW planes.
"""

from __future__ import annotations

import math

from ..types import GpuError, require_torch
from ..ops.borders import border_indices, pad
from ..ops.filters import convolve, gaussian_filter
from ..ops.luma import normalize_plane, restore_plane


def _nonneg_number(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"{name} must be a finite nonnegative number, got {value!r}")
    return float(value)


def sharpen_luma(luma, *, scale, sigma, strength):
    """clip(y + strength*(y - gaussian(y)), 0, 1) in declared units."""
    torch = require_torch()
    sig = _nonneg_number("sigma", sigma)
    st = _nonneg_number("strength", strength)
    arr = torch.as_tensor(luma)
    if arr.ndim not in (2, 3) or not arr.numel():
        raise GpuError(f"sharpen_luma expects HW or BHW input, got {tuple(arr.shape)}")
    batched = arr.ndim == 3
    frames = arr if batched else arr[None]
    out = torch.empty(frames.shape, dtype=torch.float32, device=arr.device)
    for i in range(frames.shape[0]):
        y = normalize_plane(frames[i], scale=scale)
        smooth = gaussian_filter(y, sig, mode="reflect", truncate=4)
        sharp = torch.clamp(y + st * (y - smooth), 0.0, 1.0)
        out[i] = restore_plane(sharp, scale=scale)
    return out if batched else out[0]


def non_local_means(luma, *, scale, window_size, patch_size, strength):
    """Reflected search-window NLM with f64 uniform-filter distances."""
    torch = require_torch()
    for name, value in (("window_size", window_size), ("patch_size", patch_size)):
        if isinstance(value, bool) or int(value) != value or int(value) <= 0 or int(value) % 2 == 0:
            raise ValueError(f"{name} must be a positive odd integer, got {value!r}")
    st = _nonneg_number("strength", strength)
    arr = torch.as_tensor(luma)
    if arr.ndim not in (2, 3) or not arr.numel():
        raise GpuError(f"non_local_means expects HW or BHW input, got {tuple(arr.shape)}")
    batched = arr.ndim == 3
    win, patch = int(window_size), int(patch_size)
    radius = win // 2
    s2 = st * st
    frames = arr if batched else arr[None]
    out = torch.empty(frames.shape, dtype=torch.float32, device=arr.device)
    ones_h = torch.ones((patch, 1), dtype=torch.float64, device=arr.device)
    ones_w = torch.ones((1, patch), dtype=torch.float64, device=arr.device)
    dev_patch = torch.as_tensor(float(patch), dtype=torch.float64, device=arr.device)
    for i in range(frames.shape[0]):
        y = normalize_plane(frames[i], scale=scale)
        h, w = y.shape
        ext = pad(y, radius, mode="mirror")
        num = torch.zeros_like(y)
        den = torch.zeros_like(y)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                shifted = ext[radius + dy:radius + dy + h, radius + dx:radius + dx + w]
                diff2 = (y - shifted) ** 2
                dist = convolve(diff2, ones_h, mode="mirror") / dev_patch
                dist = convolve(dist, ones_w, mode="mirror") / dev_patch
                if s2 == 0:
                    weight = (dist == 0).to(torch.float64)
                else:
                    weight = torch.exp(-dist / torch.as_tensor(
                        s2, dtype=torch.float64, device=arr.device))
                num = num + weight * shifted
                den = den + weight
        out[i] = restore_plane(num / den, scale=scale)
    return out if batched else out[0]


def _asym_pad(y, before, after, axis):
    torch = require_torch()
    n = y.shape[axis]
    r = max(before, after)
    idx = border_indices(n, r, "mirror", device=y.device)
    idx = idx[r - before:r + n + after]
    return torch.index_select(y, axis, idx)


def clahe_luma(luma, *, scale, wind, clip_limit, bins=4096):
    """Native CLAHE over HW/BHW planes; independent frames."""
    torch = require_torch()
    try:
        wind_i = int(wind)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"wind must be a positive integer, got {wind!r}") from exc
    if isinstance(wind, bool) or wind_i != wind or wind_i <= 0:
        raise ValueError(f"wind must be a positive integer, got {wind!r}")
    try:
        cl = float(clip_limit)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"clip_limit must be a finite nonnegative number, got {clip_limit!r}") from exc
    if isinstance(clip_limit, bool) or not math.isfinite(cl) or cl < 0:
        raise ValueError(f"clip_limit must be a finite nonnegative number, got {clip_limit!r}")
    try:
        bins_i = int(bins)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"bins must be one of 256 or 4096, got {bins!r}") from exc
    if isinstance(bins, bool) or bins_i != bins or bins_i not in (256, 4096):
        raise ValueError(f"bins must be one of 256 or 4096, got {bins!r}")
    wind, bins = wind_i, bins_i
    arr = torch.as_tensor(luma)
    if arr.ndim not in (2, 3) or not arr.numel():
        raise GpuError(f"clahe_luma expects HW or BHW input, got {tuple(arr.shape)}")
    batched = arr.ndim == 3
    frames = arr if batched else arr[None]
    out = torch.empty(frames.shape, dtype=torch.float32, device=arr.device)
    dev = arr.device
    identity = torch.linspace(0.0, 1.0, bins, dtype=torch.float64, device=dev)
    area = float(wind * wind)
    cap = cl * area / bins
    prior = 256.0 / bins
    d_bins = torch.as_tensor(float(bins), dtype=torch.float64, device=dev)
    d_area = torch.as_tensor(area + 256.0, dtype=torch.float64, device=dev)
    for i in range(frames.shape[0]):
        y = normalize_plane(frames[i], scale=scale)
        if float(y.min()) == float(y.max()):
            out[i] = restore_plane(y, scale=scale)
            continue
        h, w = y.shape
        rowpad = (-h) % wind
        colpad = (-w) % wind
        top, bottom = rowpad // 2, rowpad - rowpad // 2
        left, right = colpad // 2, colpad - colpad // 2
        yp = y
        if rowpad:
            yp = _asym_pad(yp, top, bottom, 0)
        if colpad:
            yp = _asym_pad(yp, left, right, 1)
        ph, pw = yp.shape
        th, tw = ph // wind, pw // wind
        ntiles = th * tw
        flat = yp.reshape(th, wind, tw, wind).permute(0, 2, 1, 3).reshape(ntiles, wind * wind)
        hist_idx = torch.floor(
            torch.clamp(flat * (bins - 1) + 0.5, 0.0, float(bins - 1))).to(torch.int64)
        counts = torch.zeros((ntiles, bins), dtype=torch.int64, device=dev)
        counts.scatter_add_(1, hist_idx, torch.ones_like(hist_idx))
        counts_f = counts.to(torch.float64)
        constant_tile = (flat.min(dim=1).values == flat.max(dim=1).values)
        kept = torch.minimum(counts_f, torch.as_tensor(cap, dtype=torch.float64, device=dev))
        excess = area - kept.sum(dim=1)
        lut = torch.cumsum(
            kept + (excess[:, None] / d_bins) + prior, dim=1) / d_area
        lut = torch.where(constant_tile[:, None], identity.expand(ntiles, bins), lut)
        lut = lut.reshape(th, tw, bins)
        cy = torch.clamp((torch.arange(ph, dtype=torch.float64, device=dev) + 0.5)
                         / torch.as_tensor(float(wind), dtype=torch.float64, device=dev) - 0.5,
                         0.0, float(th - 1))
        cx = torch.clamp((torch.arange(pw, dtype=torch.float64, device=dev) + 0.5)
                         / torch.as_tensor(float(wind), dtype=torch.float64, device=dev) - 0.5,
                         0.0, float(tw - 1))
        yi = torch.floor(cy).to(torch.int64)
        xi = torch.floor(cx).to(torch.int64)
        yj = torch.clamp(yi + 1, max=th - 1)
        xj = torch.clamp(xi + 1, max=tw - 1)
        fy = torch.clamp(cy - yi.to(torch.float64), 0.0, 1.0)
        fx = torch.clamp(cx - xi.to(torch.float64), 0.0, 1.0)
        pos = torch.clamp(yp * (bins - 1), 0.0, float(bins - 1))
        ilo = torch.floor(pos).to(torch.int64)
        ihi = torch.clamp(ilo + 1, max=bins - 1)
        frac = pos - ilo.to(torch.float64)
        acc = torch.zeros((ph, pw), dtype=torch.float64, device=dev)
        for ty, wy_ in ((yi, 1.0 - fy), (yj, fy)):
            for tx, wx_ in ((xi, 1.0 - fx), (xj, fx)):
                w_ = wy_[:, None] * wx_[None, :]
                lo = lut[ty[:, None].expand(ph, pw), tx[None, :].expand(ph, pw), ilo]
                hi = lut[ty[:, None].expand(ph, pw), tx[None, :].expand(ph, pw), ihi]
                acc = acc + w_ * (lo + frac * (hi - lo))
        cropped = acc[top:ph - bottom, left:pw - right] if (rowpad or colpad) else acc
        out[i] = restore_plane(cropped, scale=scale)
    return out if batched else out[0]
