import math

import numpy as np
import pytest

from util.color import (
    adjust_saturation,
    codes_to_ycbcr,
    decode_transfer,
    encode_transfer,
    luminance,
    pack_ycbcr422,
    quantize_rgb,
    rgb_to_ycbcr,
    unpack_ycbcr422,
    ycbcr_to_codes,
    ycbcr_to_rgb,
)


@pytest.mark.parametrize("transfer", ("linear", "srgb", "rec709", "power"))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_transfer_known_vectors_and_inverse(transfer: str, dtype) -> None:
    source = np.array([0, 0.001, 0.0031308, 0.018, 0.18, 0.5, 1], np.float64)
    if transfer == "linear":
        expected = source
    elif transfer == "srgb":
        expected = np.array([
            12.92 * x if x <= 0.0031308 else 1.055 * math.pow(x, 1 / 2.4) - 0.055
            for x in source
        ])
    elif transfer == "rec709":
        expected = np.array([
            4.5 * x if x < 0.018 else 1.099 * math.pow(x, 0.45) - 0.099
            for x in source
        ])
    else:
        expected = np.array([math.pow(x, 1 / 2.2) for x in source])
    original = source.copy()
    encoded = encode_transfer(source, transfer, dtype=dtype)
    assert encoded.dtype == dtype
    np.testing.assert_allclose(encoded, expected, rtol=0, atol=3e-7 if dtype == np.float32 else 5e-15)
    decoded = decode_transfer(encoded, transfer, dtype=dtype)
    np.testing.assert_allclose(decoded, source, rtol=0, atol=4e-7 if dtype == np.float32 else 5e-9)
    np.testing.assert_array_equal(source, original)


def test_transfer_breakpoints_and_linear_headroom() -> None:
    points = np.array([0.0031308 - 1e-12, 0.0031308, 0.0031308 + 1e-12])
    expected = np.array([
        12.92 * points[0], 12.92 * points[1], 1.055 * points[2] ** (1 / 2.4) - 0.055,
    ])
    np.testing.assert_allclose(encode_transfer(points, "srgb", dtype=np.float64), expected, atol=1e-15, rtol=0)
    encoded = np.array([0.04045 - 1e-12, 0.04045, 0.04045 + 1e-12])
    expected_inverse = np.array([
        encoded[0] / 12.92, encoded[1] / 12.92, ((encoded[2] + 0.055) / 1.055) ** 2.4,
    ])
    np.testing.assert_allclose(decode_transfer(encoded, "srgb", dtype=np.float64), expected_inverse, atol=1e-15, rtol=0)
    linear = np.array([-0.125, 0.0, 1.0, 3.0], np.float32)
    np.testing.assert_array_equal(encode_transfer(linear, "linear"), linear)
    np.testing.assert_array_equal(decode_transfer(linear, "linear"), linear)


def test_power_gamma_controls_and_ignored_gamma() -> None:
    source = np.array([0.0, 0.25, 1.0])
    np.testing.assert_allclose(encode_transfer(source, "power", gamma=2, dtype=np.float64), [0, 0.5, 1])
    np.testing.assert_allclose(decode_transfer(source, "power", gamma=2, dtype=np.float64), [0, 0.0625, 1])
    assert encode_transfer(source, "srgb", gamma=float("nan")).dtype == np.float32
    for value in (True, 0, -1, float("nan"), float("inf"), 10**400):
        with pytest.raises(ValueError):
            encode_transfer(source, "power", gamma=value)
    with pytest.raises(ValueError):
        encode_transfer(np.array([-0.1]), "power")
    with pytest.raises(ValueError):
        decode_transfer(np.array([-0.1]), "power")


@pytest.mark.parametrize("matrix,red_y,blue_y", (("bt601", 0.299, 0.114), ("bt709", 0.2126, 0.0722)))
def test_rgb_ycbcr_primaries_and_exact_float_roundtrip(matrix: str, red_y: float, blue_y: float) -> None:
    source = np.array([[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64)
    green_y = 1 - red_y - blue_y
    expected = np.array([
        [0, 0.5, 0.5], [1, 0.5, 0.5],
        [red_y, 0.5 - red_y / (2 * (1 - blue_y)), 1],
        [green_y, 0.5 - green_y / (2 * (1 - blue_y)), 0.5 - green_y / (2 * (1 - red_y))],
        [blue_y, 1, 0.5 - blue_y / (2 * (1 - red_y))],
    ])
    converted = rgb_to_ycbcr(source, matrix=matrix, dtype=np.float64)
    np.testing.assert_allclose(converted, expected, rtol=0, atol=5e-15)
    np.testing.assert_allclose(ycbcr_to_rgb(converted, matrix=matrix, dtype=np.float64), source, atol=5e-15, rtol=0)
    np.testing.assert_allclose(luminance(source, matrix=matrix, dtype=np.float64), expected[:, 0], atol=5e-15, rtol=0)


def test_saturation_preserves_luma_without_clipping_headroom() -> None:
    source = np.array([[[-0.1, 0.5, 1.5], [2, 1, 0]]], np.float32)
    before = source.copy()
    np.testing.assert_array_equal(adjust_saturation(source, 1), source)
    grey = adjust_saturation(source, 0)
    np.testing.assert_allclose(grey[..., 0], grey[..., 1], rtol=0, atol=0)
    np.testing.assert_allclose(grey[..., 1], grey[..., 2], rtol=0, atol=0)
    enhanced = adjust_saturation(source, 1.5)
    assert enhanced.min() < 0 and enhanced.max() > 1
    np.testing.assert_allclose(luminance(enhanced), luminance(source), rtol=0, atol=3e-7)
    np.testing.assert_array_equal(source, before)


@pytest.mark.parametrize("bits,dtype", ((8, np.uint8), (16, np.uint16)))
def test_output_quantization_is_only_declared_range_and_rounds_half_up(bits: int, dtype) -> None:
    maximum = 2**bits - 1
    source = np.array([-1, 0, 0.5 / maximum, 1.5 / maximum, 0.5, 1, 2], np.float64)
    out = quantize_rgb(source, bit_depth=bits)
    assert out.dtype == dtype
    np.testing.assert_array_equal(out, [0, 0, 1, 2, (maximum + 1) // 2, maximum, maximum])


@pytest.mark.parametrize("bits,dtype", ((8, np.uint8), (16, np.uint16)))
@pytest.mark.parametrize("range_mode", ("full", "limited"))
def test_ycbcr_codes_black_white_and_neutral_chroma(bits: int, dtype, range_mode: str) -> None:
    neutral = np.array([[0, 0.5, 0.5], [1, 0.5, 0.5]], np.float64)
    codes = ycbcr_to_codes(neutral, bit_depth=bits, range_mode=range_mode)
    assert codes.dtype == dtype
    center = 2 ** (bits - 1)
    shift = 2 ** (bits - 8)
    black, white = (0, 2**bits - 1) if range_mode == "full" else (16 * shift, 235 * shift)
    np.testing.assert_array_equal(codes, [[black, center, center], [white, center, center]])
    np.testing.assert_allclose(
        codes_to_ycbcr(codes, bit_depth=bits, range_mode=range_mode, dtype=np.float64),
        neutral, rtol=0, atol=1e-15,
    )


@pytest.mark.parametrize("matrix", ("bt601", "bt709"))
@pytest.mark.parametrize("range_mode", ("full", "limited"))
@pytest.mark.parametrize("bits", (8, 16))
def test_code_roundtrip_with_bounded_quantization(matrix: str, range_mode: str, bits: int) -> None:
    source = np.random.default_rng(57).random((17, 19, 3))
    normalized = rgb_to_ycbcr(source, matrix=matrix, dtype=np.float64)
    codes = ycbcr_to_codes(normalized, bit_depth=bits, range_mode=range_mode)
    decoded = codes_to_ycbcr(codes, bit_depth=bits, range_mode=range_mode, dtype=np.float64)
    restored = ycbcr_to_rgb(decoded, matrix=matrix, dtype=np.float64)
    np.testing.assert_allclose(restored, source, rtol=0, atol=3 / (2**bits - 1))


@pytest.mark.parametrize("packing", ("yuyv", "uyvy"))
@pytest.mark.parametrize("dtype", (np.uint8, np.uint16))
def test_ycbcr422_width_layout_and_chroma_pair_average(packing: str, dtype) -> None:
    source = np.array([[[10, 20, 30], [11, 23, 35], [12, 40, 50], [13, 44, 56]]], dtype=dtype)
    expected = [10, 22, 11, 33, 12, 42, 13, 53] if packing == "yuyv" else [22, 10, 33, 11, 42, 12, 53, 13]
    packed = pack_ycbcr422(source, packing=packing)
    assert packed.shape == (1, 8) and packed.dtype == dtype
    np.testing.assert_array_equal(packed[0], expected)
    restored = unpack_ycbcr422(packed, packing=packing)
    np.testing.assert_array_equal(restored[..., 0], source[..., 0])
    np.testing.assert_array_equal(restored[..., 1], [[22, 22, 42, 42]])
    np.testing.assert_array_equal(restored[..., 2], [[33, 33, 53, 53]])
    with pytest.raises(ValueError):
        pack_ycbcr422(source[:, :3], packing=packing)
    with pytest.raises(ValueError):
        unpack_ycbcr422(packed[:, :6], packing=packing)


def test_color_helpers_reject_invalid_requests_and_nonfinite_values() -> None:
    rgb = np.ones((2, 2, 3), np.float32)
    for operation in (encode_transfer, decode_transfer):
        with pytest.raises(ValueError):
            operation(rgb, "bogus")
        with pytest.raises(ValueError):
            operation(np.array([np.nan]), "linear")
    for operation in (rgb_to_ycbcr, ycbcr_to_rgb, luminance):
        with pytest.raises(ValueError):
            operation(rgb, matrix="bogus")
        with pytest.raises(ValueError):
            operation(np.ones((2, 2)))
    with pytest.raises(ValueError):
        adjust_saturation(rgb, True)
    with pytest.raises(ValueError):
        adjust_saturation(rgb, -1)
    with pytest.raises(ValueError):
        quantize_rgb(rgb, bit_depth=12)
    with pytest.raises(ValueError):
        quantize_rgb(np.array([np.inf]), bit_depth=8)
    with pytest.raises(ValueError):
        ycbcr_to_codes(rgb, bit_depth=8, range_mode="bogus")
    with pytest.raises(ValueError):
        codes_to_ycbcr(np.ones((1, 1, 3), np.uint16) * 256, bit_depth=8)
    with pytest.raises(ValueError):
        pack_ycbcr422(np.ones((1, 2, 3), np.float32))


@pytest.mark.parametrize("operation", (encode_transfer, decode_transfer))
def test_negative_signed_integer_power_input_is_rejected(operation) -> None:
    with pytest.raises(ValueError):
        operation(np.array([-1, 0, 1], np.int64), "power")


@pytest.mark.parametrize("transfer", ("srgb", "rec709"))
def test_transfer_does_not_evaluate_overflowing_unused_linear_branch(transfer: str) -> None:
    with np.errstate(over="raise", invalid="raise"):
        result = encode_transfer(np.array([0.0, 1e308]), transfer, dtype=np.float64)
    assert result[0] == 0 and np.all(np.isfinite(result))


def test_identity_saturation_keeps_small_channel_next_to_large_channel() -> None:
    source = np.array([[[1e-30, 1e30, 0]]], np.float32)
    np.testing.assert_array_equal(adjust_saturation(source, 1), source)


def test_float_helpers_reject_output_not_representable_in_requested_dtype() -> None:
    huge_rgb = np.full((1, 1, 3), 1e40, np.float64)
    operations = (
        lambda: encode_transfer(np.array([1e40]), "linear"),
        lambda: decode_transfer(np.array([1e40]), "linear"),
        lambda: luminance(huge_rgb),
        lambda: adjust_saturation(huge_rgb, 1),
        lambda: rgb_to_ycbcr(huge_rgb),
        lambda: ycbcr_to_rgb(huge_rgb),
    )
    for operation in operations:
        with pytest.raises(ValueError):
            operation()
    with pytest.raises(ValueError):
        ycbcr_to_rgb(np.array([[0, 1e308, 1e308]]), dtype=np.float64)


def test_packed_images_require_positive_height() -> None:
    with pytest.raises(ValueError):
        pack_ycbcr422(np.empty((0, 2, 3), np.uint8))
    with pytest.raises(ValueError):
        unpack_ycbcr422(np.empty((0, 4), np.uint16))


def test_identity_saturation_still_rejects_unknown_matrix() -> None:
    with pytest.raises(ValueError):
        adjust_saturation(np.ones((1, 1, 3)), 1, matrix="bogus")


@pytest.mark.parametrize("matrix,range_mode,expected", (
    ("bt601", "full", [[76, 85, 255], [150, 44, 21], [29, 255, 107]]),
    ("bt601", "limited", [[81, 90, 240], [145, 54, 34], [41, 240, 110]]),
    ("bt709", "full", [[54, 99, 255], [182, 30, 12], [18, 255, 116]]),
    ("bt709", "limited", [[63, 102, 240], [173, 42, 26], [32, 240, 118]]),
))
def test_known_eight_bit_primary_codes(matrix: str, range_mode: str, expected) -> None:
    colors = np.eye(3, dtype=np.float64)
    ycbcr = rgb_to_ycbcr(colors, matrix=matrix, dtype=np.float64)
    np.testing.assert_array_equal(
        ycbcr_to_codes(ycbcr, bit_depth=8, range_mode=range_mode), expected,
    )


def test_sixteen_bit_codes_are_not_scaled_eight_bit_quantization() -> None:
    colors = np.eye(3, dtype=np.float64)
    ycbcr = rgb_to_ycbcr(colors, matrix="bt601", dtype=np.float64)
    np.testing.assert_array_equal(
        ycbcr_to_codes(ycbcr, bit_depth=16, range_mode="limited"),
        [[20859, 23092, 61440], [37006, 13772, 8759], [10487, 61440, 28105]],
    )
    ramp = np.linspace(0, 1, 4097)
    neutral = np.stack((ramp, np.full_like(ramp, 0.5), np.full_like(ramp, 0.5)), axis=-1)
    coded = ycbcr_to_codes(neutral, bit_depth=16, range_mode="full")
    assert np.unique(coded[:, 0]).size == 4097


def test_quantizers_clip_finite_values_before_narrowing_float_type() -> None:
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("platform has no wider-than-float64 long double")
    values = np.array([np.longdouble("-1e400"), np.longdouble("1e400")])
    with np.errstate(over="raise", invalid="raise"):
        np.testing.assert_array_equal(quantize_rgb(values, bit_depth=8), [0, 255])


@pytest.mark.parametrize("transfer", ("srgb", "rec709", "power"))
def test_transfer_unrepresentable_result_raises_value_error(transfer: str) -> None:
    with np.errstate(over="raise", invalid="raise"):
        with pytest.raises(ValueError):
            decode_transfer(np.array([1e308]), transfer, dtype=np.float64)


def test_big_endian_uint16_packing_preserves_word_values() -> None:
    source = np.array([[[1, 65535, 65535], [2, 65535, 65535]]], dtype=">u2")
    packed = pack_ycbcr422(source)
    np.testing.assert_array_equal(packed, [[1, 65535, 2, 65535]])
    assert packed.dtype == source.dtype
    np.testing.assert_array_equal(unpack_ycbcr422(packed), source)
