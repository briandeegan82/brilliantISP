"""Correlation/convolution over the last two axes via explicit border maps.

Accumulation is float64 (precision-first); callers apply checked float32
boundaries at their declared reference points. Loops are bounded over
kernel taps, never over pixels.
"""

from __future__ import annotations

import math

from ..types import require_torch
from .borders import _normalize_mode


def _fold_positions(positions, n: int, mode: str):
    torch = require_torch()
    if mode == "constant":
        outside = (positions < 0) | (positions >= n)
        return positions.clamp(0, n - 1), outside
    if mode == "nearest" or n == 1:
        return positions.clamp(0, n - 1), None
    if mode == "wrap":
        return positions % n, None
    if mode == "reflect":
        period = 2 * n
        t = positions % period
        return torch.where(t < n, t, period - 1 - t), None
    period = 2 * n - 2
    t = positions % period
    return torch.where(t < n, t, period - t), None


def _correlate_origin(x, kernel, ry: int, rx: int, mode: str, cval: float):
    torch = require_torch()
    arr = torch.as_tensor(x)
    w = torch.as_tensor(kernel, dtype=torch.float64, device=arr.device)
    if arr.ndim < 2:
        raise ValueError(f"correlate expects (...,H,W) input, got {arr.shape}")
    mode = _normalize_mode(mode)
    h, wdt = arr.shape[-2], arr.shape[-1]
    kh, kw = w.shape
    src = arr.to(torch.float64)
    acc = torch.zeros(arr.shape[:-2] + (h, wdt), dtype=torch.float64, device=arr.device)
    base_r = torch.arange(h, dtype=torch.int64, device=arr.device)
    base_c = torch.arange(wdt, dtype=torch.int64, device=arr.device)
    for ky in range(kh):
        ridx, rmask = _fold_positions(base_r + (ky - ry), h, mode)
        for kx in range(kw):
            cidx, cmask = _fold_positions(base_c + (kx - rx), wdt, mode)
            tap = src[..., ridx, :][..., :, cidx]
            weight = w[ky, kx]
            if rmask is not None or cmask is not None:
                inside = torch.ones(h, wdt, dtype=torch.bool, device=arr.device)
                if rmask is not None:
                    inside &= ~rmask[:, None]
                if cmask is not None:
                    inside &= ~cmask[None, :]
                acc += torch.where(inside, tap, cval) * weight
            else:
                acc += tap * weight
    return acc


def correlate(x, kernel, mode: str = "reflect", cval: float = 0.0):
    """Cross-correlation over the last two axes, same-size output.

    ``mode`` takes the scipy-style names ('reflect', 'mirror'/'symm',
    'nearest', 'constant', 'wrap'); 'constant' adds ``cval`` for
    out-of-frame taps exactly like scipy.ndimage.
    """
    torch = require_torch()
    w = torch.as_tensor(kernel)
    if w.ndim != 2:
        raise ValueError(f"correlate kernel must be 2-D, got {w.shape}")
    return _correlate_origin(x, w, w.shape[0] // 2, w.shape[1] // 2, mode, cval)


def convolve(x, kernel, mode: str = "reflect", cval: float = 0.0):
    """Convolution over the last two axes, matching scipy.ndimage.convolve.

    The kernel is flipped; even-sized axes additionally shift the origin
    by one so the asymmetric even-kernel placement is not silently wrong.
    """
    torch = require_torch()
    w = torch.as_tensor(kernel)
    if w.ndim != 2:
        raise ValueError(f"convolve kernel must be 2-D, got {w.shape}")
    kh, kw = w.shape
    ry = kh // 2 - (1 if kh % 2 == 0 else 0)
    rx = kw // 2 - (1 if kw % 2 == 0 else 0)
    return _correlate_origin(x, w.flip(-1, -2), ry, rx, mode, cval)


def gaussian_kernel1d(sigma: float, truncate: float = 4.0, *, device=None):
    """Normalized 1-D Gaussian kernel matching scipy.ndimage.gaussian_filter."""
    torch = require_torch()
    if not math.isfinite(sigma) or sigma < 0 or not math.isfinite(truncate) or truncate <= 0:
        raise ValueError(f"sigma must be finite nonnegative and truncate finite positive, got {sigma!r}/{truncate!r}")
    if sigma == 0:
        return torch.ones(1, dtype=torch.float64, device=device)
    lw = int(truncate * sigma + 0.5)
    x = torch.arange(-lw, lw + 1, dtype=torch.float64, device=device)
    weights = torch.exp(-0.5 * (x / sigma) ** 2)
    return weights / weights.sum()


def gaussian_filter(x, sigma: float, mode: str = "reflect", truncate: float = 4.0):
    """Separable Gaussian filter over the last two axes, float64 result.

    sigma == 0 (or a radius collapsing to a single tap) is the reference
    identity, exactly as scipy.ndimage.gaussian_filter.
    """
    torch = require_torch()
    arr = torch.as_tensor(x)
    if arr.ndim < 2:
        raise ValueError(f"gaussian_filter expects (...,H,W) input, got {arr.shape}")
    if sigma == 0:
        return arr.to(torch.float64)
    k = gaussian_kernel1d(sigma, truncate, device=arr.device)
    r = k.numel() // 2
    tmp = correlate(arr, k[None, :], mode=mode)
    return correlate(tmp, k[:, None], mode=mode)
