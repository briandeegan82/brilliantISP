"""GPU native output quantization and sRGB8 preview.

Mirrors util.native_output.quantize_output/preview_from_native over HWC3
or BHWC3 inputs: no OETF during quantization, signed float-linear TIFF
preserved, physical uint8/16 code/packed tensors, preview derived from
actual native pixels. Torch is lazy.
"""

from __future__ import annotations

from ..types import GpuError, require_torch
from ..ops.array import checked_float32
from ..ops.color import (
    codes_to_ycbcr, encode_transfer, decode_transfer,
    quantize_rgb, rgb_to_ycbcr, ycbcr_to_rgb, ycbcr_to_codes,
    pack_ycbcr422, unpack_ycbcr422)


def _rgb_source(values, torch):
    arr = torch.as_tensor(values)
    if arr.ndim not in (3, 4) or arr.shape[-1] != 3 or not arr.numel():
        raise GpuError(f"output requires a nonempty floating-point HWC3/BHWC3 array, got {tuple(arr.shape)}")
    if arr.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("output requires a floating-point RGB array")
    if not bool(torch.isfinite(arr).all()):
        raise ValueError("output RGB must be finite")
    return arr


def quantize_output(encoded_rgb, spec):
    """Pre-container pixels; never applies an OETF."""
    torch = require_torch()
    rgb = _rgb_source(encoded_rgb, torch)
    if spec.model == "rgb":
        if spec.floating_linear:
            return checked_float32(rgb, name="linear_float_output")
        return quantize_rgb(rgb, bit_depth=spec.bit_depth)
    bounded = rgb.clamp(0.0, 1.0)
    ycbcr = rgb_to_ycbcr(bounded, matrix=spec.matrix, dtype="float64")
    codes = ycbcr_to_codes(ycbcr, bit_depth=spec.bit_depth, range_mode=spec.range_mode)
    if spec.layout == "422":
        return pack_ycbcr422(codes, packing=spec.packing)
    return codes


def _native_shape(native, spec, torch):
    arr = torch.as_tensor(native)
    if spec.model == "ycbcr" and spec.layout == "422":
        if arr.ndim not in (2, 3) or not arr.numel() or arr.shape[-1] % 4:
            raise ValueError("native 4:2:2 requires nonempty (...,H,2W) data with even pixel width")
        height, width = arr.shape[-2], arr.shape[-1] // 2
    else:
        if arr.ndim not in (3, 4) or arr.shape[-1] != 3 or not arr.numel():
            raise ValueError("native RGB/4:4:4 requires a nonempty (...,H,W,3) array")
        height, width = arr.shape[-3], arr.shape[-2]
    if spec.floating_linear:
        if arr.dtype != torch.float32:
            raise ValueError("native dtype does not match the declared output representation")
        if not bool(torch.isfinite(arr).all()):
            raise ValueError("native floating-point output must be finite")
    else:
        expected = torch.uint8 if spec.bit_depth == 8 else torch.uint16
        if arr.dtype != expected:
            raise ValueError("native dtype does not match the declared output representation")
    return height, width


def preview_from_native(native, spec):
    """Actual native pixels -> explicit sRGB8 uint8."""
    torch = require_torch()
    arr = torch.as_tensor(native)
    _native_shape(arr, spec, torch)
    if (spec.model == "rgb" and spec.bit_depth == 8
            and spec.transfer == "srgb" and arr.dtype == torch.uint8):
        return arr.clone()
    if spec.model == "rgb":
        encoded = arr.to(torch.float64)
        if not spec.floating_linear:
            encoded = encoded / torch.as_tensor(2 ** spec.bit_depth - 1, dtype=torch.float64, device=arr.device)
    else:
        codes = unpack_ycbcr422(arr, packing=spec.packing) if spec.layout == "422" else arr
        ycbcr = codes_to_ycbcr(codes, bit_depth=spec.bit_depth, range_mode=spec.range_mode, dtype="float64")
        encoded = ycbcr_to_rgb(ycbcr, matrix=spec.matrix, dtype="float64")
    if spec.transfer == "srgb":
        return quantize_rgb(encoded, bit_depth=8)
    linear = decode_transfer(encoded.clamp(0, 1), spec.transfer, gamma=spec.gamma, dtype="float64")
    srgb = encode_transfer(linear.clamp(0, 1), "srgb", dtype="float64")
    return quantize_rgb(srgb, bit_depth=8)
