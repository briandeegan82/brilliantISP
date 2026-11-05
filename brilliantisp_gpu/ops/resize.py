"""Exact resize helpers matching OpenCV coordinate conventions.

INTER_NEAREST: source index floor(dst * scale), clamped to the frame.
INTER_LINEAR: half-pixel centers ``(dst + 0.5) * scale - 0.5`` clamped to
the frame, linear blend. INTER_AREA: each destination pixel averages the
source rectangle it covers with per-sample overlap weights, in both
directions (upscale cells overlap one or two source samples).

All math is float64 over the last two axes; batch/leading dims pass
through. Area resize gathers at most ``ceil(scale) + 1`` taps per output
position rather than materializing a dense n_in x n_out weight matrix.
"""

from __future__ import annotations

import math

from ..types import require_torch


def _checked_input(x):
    torch = require_torch()
    arr = torch.as_tensor(x)
    if arr.ndim < 2 or arr.numel() == 0:
        raise ValueError(f"resize expects a nonempty (...,H,W) input, got {arr.shape}")
    return arr


def _check(out_h: int, out_w: int) -> None:
    for name, v in (("out_h", out_h), ("out_w", out_w)):
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError(f"{name} must be a positive integer, got {v!r}")


def nearest(x, out_h: int, out_w: int):
    torch = require_torch()
    arr = _checked_input(x)
    _check(out_h, out_w)
    h, w = arr.shape[-2], arr.shape[-1]
    rows = torch.clamp(
        (torch.arange(out_h, dtype=torch.float64, device=arr.device) * (1.0 / (out_h / h))).to(torch.int64),
        0, h - 1)
    cols = torch.clamp(
        (torch.arange(out_w, dtype=torch.float64, device=arr.device) * (1.0 / (out_w / w))).to(torch.int64),
        0, w - 1)
    return arr[..., rows, :][..., :, cols]


def _linear_weights(n_in: int, n_out: int, device):
    torch = require_torch()
    scale = n_in / n_out
    center = (torch.arange(n_out, dtype=torch.float64, device=device) + 0.5) * scale - 0.5
    center = center.clamp(0.0, n_in - 1.0)
    i0 = torch.floor(center).to(torch.int64)
    i1 = torch.clamp(i0 + 1, max=n_in - 1)
    frac = center - i0.to(torch.float64)
    return i0, i1, frac


def bilinear(x, out_h: int, out_w: int):
    torch = require_torch()
    arr = _checked_input(x)
    _check(out_h, out_w)
    h, w = arr.shape[-2], arr.shape[-1]
    r0, r1, rf = _linear_weights(h, out_h, arr.device)
    c0, c1, cf = _linear_weights(w, out_w, arr.device)
    src = arr.to(torch.float64)
    top = src[..., r0, :][..., :, c0] * (1 - cf) + src[..., r0, :][..., :, c1] * cf
    bot = src[..., r1, :][..., :, c0] * (1 - cf) + src[..., r1, :][..., :, c1] * cf
    return top * (1 - rf)[:, None] + bot * rf[:, None]


def _area_axis(x, n_in: int, n_out: int, axis: int):
    """Bounded overlap-average along one axis via per-tap gathers.

    Each output cell covers ``[o*s, (o+1)*s)`` of the source; at most
    ``floor(s) + 2`` source samples can overlap, so the reduction loops
    over that bounded tap count instead of an n_in x n_out dense matrix.
    """
    torch = require_torch()
    scale = n_in / n_out
    dst = torch.arange(n_out, dtype=torch.float64, device=x.device)
    lo = dst * scale
    hi = lo + scale
    base = torch.floor(lo).to(torch.int64)
    max_taps = int(math.floor(scale)) + 2
    out = torch.zeros(x.movedim(axis, -1).shape[:-1] + (n_out,),
                      dtype=torch.float64, device=x.device)
    moved = x.movedim(axis, -1)
    for k in range(max_taps):
        raw_idx = base + k
        in_bounds = (raw_idx >= 0) & (raw_idx < n_in)
        idx = raw_idx.clamp(0, n_in - 1)
        weight = torch.clamp(
            torch.minimum(hi, idx.to(torch.float64) + 1.0)
            - torch.maximum(lo, idx.to(torch.float64)),
            min=0.0,
        ) * in_bounds.to(torch.float64)
        gathered = moved.index_select(-1, idx.reshape(-1)).reshape(
            moved.shape[:-1] + (n_out,))
        out += gathered * weight
    return (out / torch.as_tensor(
        scale, dtype=torch.float64, device=x.device)).movedim(-1, axis)


def area(x, out_h: int, out_w: int):
    torch = require_torch()
    arr = _checked_input(x)
    _check(out_h, out_w)
    h, w = arr.shape[-2], arr.shape[-1]
    src = arr.to(torch.float64)
    if out_h != h:
        src = _area_axis(src, h, out_h, -2)
    if out_w != w:
        src = _area_axis(src, w, out_w, -1)
    return src
