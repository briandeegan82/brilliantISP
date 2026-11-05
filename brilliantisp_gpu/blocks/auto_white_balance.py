"""GPU auto white balance measurement (gray_world/norm_2/pca).

Mirrors AutoWhiteBalance.compute_gains sample flow: validate CFA +
controls before sampling, widen to float64 before the green-pair
average, reject rows outside declared 2**hdr thresholds or nonfinite,
then per-algorithm channel means (PCA via the settled on-device
reference path). Fallback pairs mirror _awb_fallback exactly. Torch is
lazy; nothing except small per-frame control scalars crosses to host.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..types import GpuError, require_torch
from .raw import _layout


@dataclass(frozen=True)
class AwbMeasurement:
    """Per-frame AWB outcome: gains (Bx2 f64 device), statuses, counts."""

    gains: object
    statuses: tuple
    sample_counts: tuple


def _fallback(provided, manual, status):
    """Mirror AutoWhiteBalance._awb_fallback including zero-allowed pairs."""
    def _valid(pair):
        try:
            if not isinstance(pair, (tuple, list, np.ndarray)) or len(pair) != 2:
                return False
            if any(isinstance(v, (bool, str)) for v in pair):
                return False
            a, b = float(pair[0]), float(pair[1])
        except (TypeError, ValueError, IndexError, KeyError):
            return False
        return math.isfinite(a) and math.isfinite(b) and a >= 0 and b >= 0
    if _valid(provided):
        return float(provided[0]), float(provided[1])
    if _valid(manual):
        return float(manual[0]), float(manual[1])
    return 1.0, 1.0


def _controls(sensor, params):
    """Validate CFA + under/over pct + algorithm before any sampling."""
    bayer = str(sensor["bayer_pattern"]).strip().lower()
    if bayer not in ("rggb", "bggr", "grbg", "gbrg"):
        raise ValueError(f"unknown CFA pattern {bayer!r}")
    for name in ("underexposed_percentage", "overexposed_percentage"):
        v = params[name]
        if isinstance(v, bool) or not isinstance(v, (int, float)) \
                or not math.isfinite(v) or not 0 <= v <= 100:
            raise ValueError(f"{name} must be finite in 0..100")
    algo = "grey_world" if params["algorithm"] == "gray_world" else params["algorithm"]
    if algo == "pca":
        try:
            pct = float(params["percentage"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"pca percentage must be in (0,100], got {params.get('percentage')!r}"
            ) from exc
        if not math.isfinite(pct) or pct <= 0 or pct > 100:
            raise ValueError(f"pca percentage must be in (0,100], got {pct!r}")
    elif algo not in ("grey_world", "norm_2"):
        raise ValueError(f"unknown AWB algorithm {params['algorithm']!r}")
    return algo


def _rows(frame, sensor, params, input_gain, torch):
    """(r,g,b) Nx3 float64 rows after per-row rejection, on device."""
    raw = frame.to(torch.float64)
    if raw.ndim != 2 or raw.numel() == 0:
        return torch.empty(0, 3, dtype=torch.float64, device=frame.device)
    layout = _layout(sensor)
    ch = {}
    for site in range(4):
        row, col = divmod(site, 2)
        ch[layout[site]] = raw[row::2, col::2]
    mh = min(c.shape[0] for c in ch.values())
    mw = min(c.shape[1] for c in ch.values())
    if mh <= 0 or mw <= 0:
        return torch.empty(0, 3, dtype=torch.float64, device=frame.device)
    r = ch[0][:mh, :mw]
    g = (ch[1][:mh, :mw] + ch[2][:mh, :mw]) / 2.0
    b = ch[3][:mh, :mw]
    rgb = torch.stack((r, g, b), dim=-1).reshape(-1, 3)
    rgb = rgb / torch.as_tensor(input_gain, dtype=torch.float64, device=frame.device)
    base = 2.0 ** int(sensor["hdr_bit_depth"])
    approx = base / 100
    low = params["underexposed_percentage"] * approx
    high = base - params["overexposed_percentage"] * approx
    bad = (rgb < low) | (rgb > high)
    rows = rgb[~bad.any(dim=1)]
    rows = rows[torch.isfinite(rows).all(dim=1)]
    return rows


def _estimate(rows, algo, params):
    """Estimated channel means on device; ValueError marks data degeneracy."""
    torch = require_torch()
    means = rows.mean(dim=0)
    if not bool(torch.isfinite(means).all()) or bool((means <= 0).any()):
        raise ValueError("AWB channel means are degenerate")
    if algo == "norm_2":
        return torch.sqrt((rows * rows).mean(dim=0))
    if algo == "pca":
        norm = float(torch.linalg.norm(means))
        if not math.isfinite(norm) or norm == 0.0:
            raise ValueError("AWB channel means are degenerate")
        from .statistics import pca_means
        return pca_means(rows, float(params["percentage"]))
    return means


def auto_white_balance(image, sensor: dict, params: dict, white_balance: dict,
                       *, input_gains: tuple | None = None,
                       fallback_gains: tuple | None = None) -> AwbMeasurement:
    torch = require_torch()
    arr = torch.as_tensor(image)
    if arr.ndim != 3 or arr.shape[0] < 1:
        raise GpuError(f"auto_white_balance expects nonempty BHW input, got {tuple(arr.shape)}")
    b = arr.shape[0]
    manual = (white_balance.get("r_gain"), white_balance.get("b_gain"))
    if not params.get("is_enable", True):
        gains = [_fallback(None, manual, "disabled") for _ in range(b)]
        return AwbMeasurement(
            torch.as_tensor(gains, dtype=torch.float64, device=arr.device),
            ("disabled",) * b, (0,) * b)
    if input_gains is None:
        input_gains = (1.0,) * b
    if fallback_gains is None:
        fallback_gains = (None,) * b
    if len(input_gains) != b or len(fallback_gains) != b:
        raise GpuError("input_gains/fallback_gains length must match batch")
    algo = _controls(sensor, params)
    out_gains, statuses, counts = [], [], []
    for i in range(b):
        ig = input_gains[i]
        if isinstance(ig, (bool, np.bool_)) or not isinstance(
                ig, (int, float, np.integer, np.floating)) \
                or not 0 < float(ig) <= float(np.finfo(np.float64).max):
            raise ValueError("AWB input_gain must be finite and positive")
        rows = _rows(arr[i], sensor, params, float(ig), torch)
        counts.append(int(rows.shape[0]))
        if rows.shape[0] == 0:
            out_gains.append(_fallback(fallback_gains[i], manual, "fallback_no_samples"))
            statuses.append("fallback_no_samples")
            continue
        try:
            means = _estimate(rows, algo, params)
            if not bool(torch.isfinite(means).all()) or bool((means <= 0).any()):
                raise ValueError("AWB channel means are degenerate")
            rgain = float(means[1] / means[0])
            bgain = float(means[1] / means[2])
            if not (math.isfinite(rgain) and math.isfinite(bgain)):
                raise ValueError("AWB channel means are degenerate")
        except ValueError:
            out_gains.append(_fallback(fallback_gains[i], manual,
                                       "fallback_ill_conditioned"))
            statuses.append("fallback_ill_conditioned")
            continue
        rg = float(np.clip(rgain, 0.25, 4.0))
        bg = float(np.clip(bgain, 0.25, 4.0))
        out_gains.append((rg, bg))
        statuses.append("estimated")
    return AwbMeasurement(
        torch.as_tensor(out_gains, dtype=torch.float64, device=arr.device),
        tuple(statuses), tuple(counts))
