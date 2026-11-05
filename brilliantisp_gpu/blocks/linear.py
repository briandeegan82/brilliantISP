"""GPU digital gain, lens shading correction, and white balance.

Mirrors modules/digital_gain, lens_shading_correction and
white_balance_optimized compute_linear semantics: float32 output, no code
clip, float64 product internals, per-site CFA application. Torch is lazy.
"""

from __future__ import annotations

import numbers

import numpy as np

from ..types import GpuError, RawTensor, require_torch
from ..ops.array import checked_float32
from .raw import _check_image, _layout


def digital_gain(image: RawTensor, sensor: dict, params: dict, *,
                 gain_indices: tuple[int, ...] | None = None,
                 ae_feedback: tuple[int | None, ...] | None = None):
    """codes->float32 after configured gain multiply; returns (f32 BHW, indices)."""
    torch = require_torch()
    codes = _check_image(image, sensor)
    b = codes.shape[0]
    if gain_indices is None:
        gain_indices = (params.get("current_gain", 0),) * b
    if len(gain_indices) != b:
        raise ValueError(f"gain_indices length {len(gain_indices)} != batch {b}")
    for v in gain_indices:
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Integral):
            raise ValueError(f"gain index must be an integer, got {v!r}")
    gain_indices = tuple(int(v) for v in gain_indices)
    if ae_feedback is None:
        ae_feedback = (params.get("ae_feedback"),) * b
    if len(ae_feedback) != b:
        raise ValueError(f"ae_feedback length {len(ae_feedback)} != batch {b}")
    gains = params.get("gain_array")
    if not params.get("is_enable", True):
        return checked_float32(codes, name="digital_gain"), tuple(gain_indices)
    if gains is None:
        raise ValueError("digital_gain requires gain_array when enabled")
    n = len(gains)
    is_auto = bool(params.get("is_auto", False))
    step_mode = params.get("exposure_correction_mode", "step") != "direct"
    out_idx = []
    gvals = []
    for i in range(b):
        idx = gain_indices[i]
        if idx < 0 or idx >= n:
            raise ValueError(f"gain index {idx} out of bounds for {n} gains")
        fb = ae_feedback[i]
        if is_auto and step_mode and fb is not None:
            if fb < 0:
                idx = min(n - 1, idx + 1)
            elif fb > 0:
                idx = max(0, idx - 1)
        out_idx.append(idx)
        gvals.append(float(gains[idx]))
    g = torch.as_tensor(gvals, dtype=torch.float64, device=codes.device)[:, None, None]
    out = g * codes.to(torch.float64)
    return checked_float32(out, name="digital_gain"), tuple(out_idx)


def lens_shading_correction(image, sensor: dict, params: dict):
    """f32 radial polynomial gain per CFA site; disabled -> f32 identity."""
    torch = require_torch()
    arr = checked_float32(image, name="lens_shading_correction")
    if arr.ndim != 3:
        raise GpuError(f"LSC expects BHW input, got {tuple(arr.shape)}")
    if not params.get("is_enable", True):
        return arr
    h, w = arr.shape[-2:]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    max_dist = float(np.sqrt(cy * cy + cx * cx))
    out = arr.clone()
    layout = _layout(sensor)
    coeffs = (
        (float(params.get("r_k1", 0.0)), float(params.get("r_k2", 0.0))),
        (float(params.get("gr_k1", 0.0)), float(params.get("gr_k2", 0.0))),
        (float(params.get("gb_k1", 0.0)), float(params.get("gb_k2", 0.0))),
        (float(params.get("b_k1", 0.0)), float(params.get("b_k2", 0.0))),
    )
    if max_dist < 1e-6:
        return out
    y = torch.arange(h, dtype=torch.float32, device=arr.device)[:, None]
    x = torch.arange(w, dtype=torch.float32, device=arr.device)[None, :]
    denom = torch.as_tensor(max_dist ** 2, dtype=torch.float32, device=arr.device)
    r_sq = ((y - cy) ** 2 + (x - cx) ** 2) / denom
    def _term(k, base):
        dt = np.result_type(np.dtype("float32"), k)
        return k * base.to(torch.float64 if dt == np.float64 else torch.float32)

    for site in range(4):
        k1, k2 = coeffs[layout[site]]
        gain = torch.clamp(1.0 + _term(k1, r_sq) + _term(k2, r_sq * r_sq), min=0.1).to(torch.float32)
        out[:, site // 2::2, site % 2::2] = out[:, site // 2::2, site % 2::2] * gain[site // 2::2, site % 2::2]
    return checked_float32(out, name="lens_shading_correction")


def white_balance(image, sensor: dict, params: dict, awb_gains):
    """f32 code output; float64 gain multiply on R/B CFA sites only."""
    torch = require_torch()
    arr = torch.as_tensor(image)
    if arr.ndim != 3:
        raise GpuError(f"white_balance expects BHW input, got {tuple(arr.shape)}")
    if not params.get("is_enable", True):
        return checked_float32(arr, name="white_balance")
    b = arr.shape[0]
    raw = arr.to(torch.float64, copy=True)
    layout = _layout(sensor)
    if params.get("is_auto", False):
        g = torch.as_tensor(awb_gains, dtype=torch.float64, device=arr.device)
        if g.shape != (b, 2):
            raise GpuError(f"awb_gains must be (B,2), got {tuple(g.shape)}")
    else:
        g = torch.as_tensor(
            [[float(params.get("r_gain", 1.0)), float(params.get("b_gain", 1.0))]] * b,
            dtype=torch.float64, device=arr.device)
    for site in range(4):
        ch = layout[site]
        if ch in (0, 3):
            factor = g[:, 0 if ch == 0 else 1][:, None, None]
            raw[:, site // 2::2, site % 2::2] = raw[:, site // 2::2, site % 2::2] * factor
    return checked_float32(raw, name="white_balance")
