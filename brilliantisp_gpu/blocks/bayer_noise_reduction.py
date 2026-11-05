"""GPU Bayer noise reduction (fast joint bilateral filter).

Mirrors modules/bayer_noise_reduction compute_linear + JBF.apply_jbf
(preserve_float) + fast_joint_bilateral_filter: float32 conversion first,
code-range normalization, 5x5 G interpolation via scipy 'reflect'
convolution, per-channel JBF with bounded shifted-tap accumulation, no
code clipping. Torch is lazy.
"""

from __future__ import annotations

import numpy as np

from ..ops.reference_exp import reference_exp_float32
from ..types import GpuError, require_torch
from ..ops.array import checked_float32
from ..ops.borders import pad
from ..ops.filters import convolve
from .raw import _layout


_G_KERNEL = np.array(
    [[0, 0, -1, 0, 0], [0, 0, 2, 0, 0], [-1, 2, 4, 2, -1],
     [0, 0, 2, 0, 0], [0, 0, -1, 0, 0]],
    dtype=np.float32)
_G_KERNEL = _G_KERNEL / np.sum(_G_KERNEL)


def _gauss_kern_raw(kern: int, std_dev: float, stride: int) -> np.ndarray:
    k = np.zeros((kern, kern), dtype=np.float32)
    for i in range(kern):
        for j in range(kern):
            k[i, j] = np.exp(
                -((stride * (i - ((kern - 1) / 2))) ** 2
                  + (stride * (j - ((kern - 1) / 2))) ** 2)
                / (2 * (std_dev ** 2)))
    return k / np.sum(k)


def _corrected_window(size: int) -> int:
    if size <= 0:
        return 3
    if size % 2 == 0:
        return size + 1
    return int(size)


def _fast_jbf(in_img, guide_img, window: int, stddev_s: float, stddev_r: float,
              stride: int):
    """Shifted-tap joint bilateral accumulation, mirrors fast_joint_bilateral_filter."""
    torch = require_torch()
    spatial_kern = _corrected_window(window)
    s_kern = torch.as_tensor(
        _gauss_kern_raw(spatial_kern, stddev_s, stride), device=in_img.device)
    arm = (spatial_kern - 1) // 2
    in_ext = pad(in_img, arm, "mirror")
    guide_ext = pad(guide_img, arm, "mirror")
    h, w = in_img.shape[-2:]
    denom = 2 * stddev_r ** 2
    norm_np = np.dtype("float64") if in_img.dtype == torch.float64 else np.dtype("float32")
    div_dt = torch.float64 if np.result_type(
        norm_np, denom) == np.float64 else torch.float32
    div = torch.as_tensor(denom, dtype=div_dt, device=in_img.device)
    norm_fact = torch.zeros_like(in_img, dtype=torch.float64)
    sum_filt_out = torch.zeros_like(in_img, dtype=torch.float64)
    for i in range(spatial_kern):
        for j in range(spatial_kern):
            in_shift = in_ext[..., i:i + h, j:j + w]
            guide_shift = guide_ext[..., i:i + h, j:j + w]
            diff = guide_img - guide_shift
            arg = -(diff * diff).to(div_dt) / div
            exp_arg = (reference_exp_float32(arg)
                       if div_dt == torch.float32 else torch.exp(arg))
            weight = s_kern[i, j] * exp_arg
            norm_fact += weight
            sum_filt_out += weight * in_shift
    return sum_filt_out / norm_fact


def bayer_noise_reduction(image, sensor: dict, params: dict):
    """f32 code-domain BNR on BHW input; disabled/tiny -> f32 identity."""
    torch = require_torch()
    arr = checked_float32(image, name="bayer_noise_reduction")
    if arr.ndim != 3 or not arr.numel():
        raise GpuError(f"BNR expects nonempty BHW input, got {tuple(arr.shape)}")
    if not params.get("is_enable", True):
        return arr
    b, h, w = arr.shape
    if h < 2 or w < 2:
        return arr
    ph, pw = h + (h % 2), w + (w % 2)
    src = arr
    if ph != h:
        src = torch.cat((src, src[..., -2:-1, :].to(torch.float64)), dim=-2)
    if pw != w:
        src = torch.cat((src, src[..., :, -2:-1].to(torch.float64)), dim=-1)
    bits = int(sensor.get("hdr_bit_depth", sensor["bit_depth"]))
    code_max = 2 ** bits - 1
    norm_dt = torch.float64 if np.result_type(
        np.dtype("float32"), code_max) == np.float64 else torch.float32
    in_img = src.to(norm_dt) / torch.as_tensor(
        code_max, dtype=norm_dt, device=arr.device)
    layout = _layout(sensor)
    g_conv = convolve(in_img, torch.as_tensor(_G_KERNEL, device=arr.device),
                      mode="reflect").to(norm_dt)
    interp_g = in_img.clone()
    r_site = layout.index(0)
    b_site = layout.index(3)
    ry, rx = divmod(r_site, 2)
    by, bx = divmod(b_site, 2)
    interp_g[:, ry::2, rx::2] = g_conv[:, ry::2, rx::2]
    interp_g[:, by::2, bx::2] = g_conv[:, by::2, bx::2]
    filt_size = int(params["filter_window"])
    sub_size = int((filt_size + 1) / 2)
    out_r = _fast_jbf(
        in_img[:, ry::2, rx::2].contiguous(), g_conv[:, ry::2, rx::2].contiguous(),
        sub_size, float(params["r_std_dev_s"]), float(params["r_std_dev_r"]), 2)
    out_b = _fast_jbf(
        in_img[:, by::2, bx::2].contiguous(), g_conv[:, by::2, bx::2].contiguous(),
        sub_size, float(params["b_std_dev_s"]), float(params["b_std_dev_r"]), 2)
    out_g = _fast_jbf(
        interp_g, interp_g, filt_size, float(params["g_std_dev_s"]),
        float(params["g_std_dev_r"]), 1)
    joined = out_g.clone()
    joined[:, ry::2, rx::2] = out_r
    joined[:, by::2, bx::2] = out_b
    codes = joined * code_max
    return checked_float32(codes[..., :h, :w], name="bayer_noise_reduction")
