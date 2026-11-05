from __future__ import annotations

import math

import numpy as np
import pytest

from util.native_output import output_metadata, preview_from_native, quantize_output
from util.output_contract import OutputSpec


def srgb(value: float) -> float:
    return 12.92 * value if value <= 0.0031308 else 1.055 * value**(1 / 2.4) - 0.055


@pytest.mark.parametrize("bits", (8, 16))
def test_quantization_does_not_apply_another_transfer(bits: int) -> None:
    spec = OutputSpec.from_mapping({"bit_depth": bits})
    source = np.array([[[0, 0.25, 1], [0.5, 0.75, 1.1]]], np.float64)
    native = quantize_output(source, spec)
    maximum = 2**bits - 1
    expected = np.floor(np.clip(source, 0, 1) * maximum + 0.5)
    np.testing.assert_array_equal(native, expected)
    assert native.dtype == (np.uint8 if bits == 8 else np.uint16)
    metadata = output_metadata(native, spec, source_identity="test", tone_mapped=False)
    assert metadata.value_units == "codes"
    assert metadata.code_bit_depth == bits
    assert metadata.nominal_range == (0, maximum)
    assert metadata.transfer == "srgb"


def test_float_tiff_is_normalized_signed_linear_rgb_not_encoded_codes() -> None:
    spec = OutputSpec.from_mapping({"format": "tiff", "transfer": "linear", "bit_depth": 32})
    source = np.array([[[-0.125, 0.5, 2]]], np.float32)
    original = source.copy()
    native = quantize_output(source, spec)
    np.testing.assert_array_equal(native, source)
    np.testing.assert_array_equal(preview_from_native(native, spec), [[[0, 188, 255]]])
    metadata = output_metadata(native, spec, source_identity="test", tone_mapped=False)
    assert metadata.signal_domain == "scene_linear_rgb"
    assert metadata.storage_bit_depth == 32
    assert metadata.code_bit_depth is None
    assert metadata.value_units == "normalized"
    assert metadata.reference_white == 1
    assert metadata.allows_signed_headroom
    np.testing.assert_array_equal(source, original)
    mapped = output_metadata(native, spec, source_identity="test", tone_mapped=True)
    assert mapped.signal_domain == "display_linear_rgb"


@pytest.mark.parametrize("transfer", ("linear", "power", "rec709"))
def test_preview_inverts_actual_output_transfer_then_encodes_srgb(transfer: str) -> None:
    spec = OutputSpec.from_mapping({"transfer": transfer, "gamma": 2.4} if transfer == "power" else {"transfer": transfer})
    native = np.full((1, 1, 3), 128, np.uint8)
    encoded = 128 / 255
    if transfer == "linear":
        linear = encoded
    elif transfer == "power":
        linear = encoded**2.4
    else:
        linear = ((encoded + 0.099) / 1.099)**(1 / 0.45)
    expected = math.floor(srgb(linear) * 255 + 0.5)
    np.testing.assert_array_equal(preview_from_native(native, spec), np.full((1, 1, 3), expected, np.uint8))


def test_srgb8_preview_is_the_actual_quantized_output_not_original_float() -> None:
    spec = OutputSpec()
    source = np.array([[[0.1, 0.5, 0.9]]], np.float64)
    native = quantize_output(source, spec)
    np.testing.assert_array_equal(preview_from_native(native, spec), native)


@pytest.mark.parametrize("matrix,expected", (
    ("bt709", [[16, 128, 128], [235, 128, 128], [63, 102, 240], [173, 42, 26], [32, 240, 118]]),
    ("bt601", [[16, 128, 128], [235, 128, 128], [81, 90, 240], [145, 54, 34], [41, 240, 110]]),
))
def test_ycbcr_black_white_and_primaries_are_standard_limited_codes(matrix: str, expected: list) -> None:
    spec = OutputSpec.from_mapping({"model": "ycbcr", "matrix": matrix})
    encoded = np.array([[[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]], np.float64)
    native = quantize_output(encoded, spec)
    np.testing.assert_array_equal(native[0], expected)
    neutral = np.full((1, 1, 3), 0.25, np.float64)
    np.testing.assert_array_equal(quantize_output(neutral, spec), [[[71, 128, 128]]])


@pytest.mark.parametrize("packing,expected", (
    ("yuyv", [[63, 171, 32, 179]]),
    ("uyvy", [[171, 63, 179, 32]]),
))
def test_packed_output_has_averaged_chroma_and_logical_pixel_dimensions(packing: str, expected: list) -> None:
    spec = OutputSpec.from_mapping({"model": "ycbcr", "layout": "422", "packing": packing})
    source = np.array([[[1, 0, 0], [0, 0, 1]]], np.float64)
    native = quantize_output(source, spec)
    np.testing.assert_array_equal(native, expected)
    metadata = output_metadata(native, spec, source_identity="test", tone_mapped=True)
    assert (metadata.height, metadata.width) == (1, 2)
    assert metadata.layout == "422"
    assert metadata.packing == packing
    assert metadata.signal_domain == "encoded_ycbcr"
    assert metadata.range_mode == "limited"
    assert metadata.component_ranges == ((16, 235), (16, 240), (16, 240))


def test_yuv16_is_little_endian_and_metadata_is_not_uint8() -> None:
    spec = OutputSpec.from_mapping({"model": "ycbcr", "bit_depth": 16, "layout": "422"})
    source = np.array([[[0, 0, 0], [1, 1, 1]]], np.float32)
    native = quantize_output(source, spec)
    np.testing.assert_array_equal(native, [[4096, 32768, 60160, 32768]])
    assert native.dtype == np.dtype("<u2")
    assert native.tobytes() == b"\x00\x10\x00\x80\x00\xeb\x00\x80"
    metadata = output_metadata(native, spec, source_identity="test", tone_mapped=False)
    assert metadata.byte_order == "little"
    assert metadata.storage_bit_depth == metadata.code_bit_depth == 16
    assert metadata.reference_white == 60160
    assert metadata.component_ranges == ((4096, 60160), (4096, 61440), (4096, 61440))
    np.testing.assert_array_equal(preview_from_native(native, spec), [[[0, 0, 0], [255, 255, 255]]])


def test_422_preview_reconstructs_the_packed_chroma_not_unpacked_source() -> None:
    spec = OutputSpec.from_mapping({"model": "ycbcr", "layout": "422", "transfer": "srgb"})
    source = np.array([[[1, 0, 0], [0, 0, 1]]], np.float64)
    native = quantize_output(source, spec)
    preview = preview_from_native(native, spec)
    assert not np.array_equal(preview, np.array([[[255, 0, 0], [0, 0, 255]]], np.uint8))
    assert preview.dtype == np.uint8
    assert preview.shape == source.shape


@pytest.mark.parametrize("data", (
    np.zeros((1, 1, 3), np.uint16),
    np.zeros((1, 1), np.uint8),
    np.zeros((0, 1, 3), np.uint8),
    np.zeros((1, 1, 3), np.float32),
))
def test_native_dtype_and_shape_are_not_guessed(data: np.ndarray) -> None:
    with pytest.raises(ValueError):
        preview_from_native(data, OutputSpec())


def test_float_overflow_nonfinite_and_odd_422_are_not_silently_exported() -> None:
    floating = OutputSpec.from_mapping({"format": "tiff", "transfer": "linear", "bit_depth": 32})
    with np.errstate(all="raise"):
        with pytest.raises(ValueError):
            quantize_output(np.full((1, 1, 3), 1e100), floating)
    with pytest.raises(ValueError):
        quantize_output(np.full((1, 1, 3), np.nan), OutputSpec())
    with pytest.raises(ValueError):
        quantize_output(np.zeros((1, 3, 3), np.float32), OutputSpec.from_mapping({"model": "ycbcr", "layout": "422"}))
