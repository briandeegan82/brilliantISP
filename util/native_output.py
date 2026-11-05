from __future__ import annotations

import numpy as np

from util.color import (
    codes_to_ycbcr, decode_transfer, encode_transfer, pack_ycbcr422,
    quantize_rgb, rgb_to_ycbcr, unpack_ycbcr422, ycbcr_to_codes, ycbcr_to_rgb,
)
from util.native_array import as_float32
from util.output_contract import OutputSpec
from util.pipeline_types import FrameMetadata


def _rgb_source(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values)
    if data.ndim != 3 or data.shape[-1] != 3 or not data.size or data.dtype.kind != "f":
        raise ValueError("output requires a nonempty floating-point HWC3 RGB array")
    if not np.all(np.isfinite(data)):
        raise ValueError("output RGB must be finite")
    return data


def quantize_output(encoded_rgb: np.ndarray, spec: OutputSpec) -> np.ndarray:
    """Format pre-container pixels; input already owns its declared transfer.

    This function never applies an OETF. Float TIFF pixels remain signed
    normalized linear values. JPEG's lossy container compression is not
    simulated by the native pixel array or its preview.
    """
    rgb = _rgb_source(encoded_rgb)
    if spec.model == "rgb":
        return as_float32(rgb, name="linear_float_output") if spec.floating_linear else quantize_rgb(rgb, bit_depth=spec.bit_depth)
    bounded = np.clip(rgb, 0, 1)
    ycbcr = rgb_to_ycbcr(bounded, matrix=spec.matrix, dtype=np.float64)
    codes = ycbcr_to_codes(ycbcr, bit_depth=spec.bit_depth, range_mode=spec.range_mode)
    native = pack_ycbcr422(codes, packing=spec.packing) if spec.layout == "422" else codes
    return native.astype("<u2", copy=False) if spec.bit_depth == 16 else native


def _native_shape(native: np.ndarray, spec: OutputSpec) -> tuple[int, int]:
    data = np.asarray(native)
    if spec.model == "ycbcr" and spec.layout == "422":
        if data.ndim != 2 or not data.size or data.shape[1] % 4:
            raise ValueError("native 4:2:2 requires nonempty Hx(2W) data with even pixel width")
        height, width = data.shape[0], data.shape[1] // 2
    else:
        if data.ndim != 3 or data.shape[-1] != 3 or not data.size:
            raise ValueError("native RGB/4:4:4 requires a nonempty HWC3 array")
        height, width = data.shape[:2]
    kind = "f" if spec.floating_linear else "u"
    if data.dtype.kind != kind or data.dtype.itemsize * 8 != spec.bit_depth:
        raise ValueError("native dtype does not match the declared output representation")
    if kind == "f" and not np.all(np.isfinite(data)):
        raise ValueError("native floating-point output must be finite")
    return height, width


def preview_from_native(native: np.ndarray, spec: OutputSpec) -> np.ndarray:
    """Convert actual quantized/packed pixels to explicit sRGB8 for display."""
    _native_shape(native, spec)
    if (spec.model == "rgb" and spec.bit_depth == 8
            and spec.transfer == "srgb" and native.dtype == np.uint8):
        return np.array(native, copy=True)
    if spec.model == "rgb":
        encoded = np.asarray(native, dtype=np.float64)
        if not spec.floating_linear:
            encoded /= 2**spec.bit_depth - 1
    else:
        codes = unpack_ycbcr422(native, packing=spec.packing) if spec.layout == "422" else native
        ycbcr = codes_to_ycbcr(codes, bit_depth=spec.bit_depth, range_mode=spec.range_mode, dtype=np.float64)
        encoded = ycbcr_to_rgb(ycbcr, matrix=spec.matrix, dtype=np.float64)
    if spec.transfer == "srgb":
        return quantize_rgb(encoded, bit_depth=8)
    linear = decode_transfer(np.clip(encoded, 0, 1), spec.transfer, gamma=spec.gamma, dtype=np.float64)
    srgb = encode_transfer(np.clip(linear, 0, 1), "srgb", dtype=np.float64)
    return quantize_rgb(srgb, bit_depth=8)


def output_metadata(
    native: np.ndarray, spec: OutputSpec, *, source_identity: str, tone_mapped: bool,
) -> FrameMetadata:
    height, width = _native_shape(native, spec)
    if spec.model == "ycbcr":
        domain = "encoded_ycbcr"
    elif spec.transfer != "linear":
        domain = "display_encoded_rgb"
    else:
        domain = "display_linear_rgb" if tone_mapped else "scene_linear_rgb"
    if spec.floating_linear:
        limits = ((0.0, 1.0),) * 3
    elif spec.model == "ycbcr" and spec.range_mode == "limited":
        scale = 2**(spec.bit_depth - 8)
        limits = ((16.0 * scale, 235.0 * scale), (16.0 * scale, 240.0 * scale), (16.0 * scale, 240.0 * scale))
    else:
        limits = ((0.0, float(2**spec.bit_depth - 1)),) * 3
    return FrameMetadata(
        width=width, height=height, cfa_pattern=None,
        storage_bit_depth=native.dtype.itemsize * 8,
        code_bit_depth=None if spec.floating_linear else spec.bit_depth,
        signal_domain=domain, transfer=spec.transfer, source_identity=source_identity,
        value_units="normalized" if spec.floating_linear else "codes",
        color_model=spec.model, color_space=spec.color_space,
        matrix=spec.matrix if spec.model == "ycbcr" else None,
        range_mode=spec.range_mode,
        layout=spec.layout if spec.model == "ycbcr" else None,
        packing=spec.packing if spec.model == "ycbcr" and spec.layout == "422" else None,
        byte_order="little" if spec.model == "ycbcr" and spec.bit_depth == 16 else None,
        format=spec.format, gamma=spec.gamma if spec.transfer == "power" else None,
        nominal_range=limits[0], component_ranges=limits,
        reference_white=limits[0][1], allows_signed_headroom=spec.floating_linear,
    )
