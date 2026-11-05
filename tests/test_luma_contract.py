from __future__ import annotations

import math

import numpy as np
import pytest

from modules.ldci.clahe import apply_luma as clahe
from modules.noise_reduction_2d.non_local_means import apply_luma as nlm
from modules.sharpen.unsharp_masking import apply_luma as sharpen
from util.color import rgb_to_ycbcr
from util.luma import replace_luminance


def clipped_cdf(value: float, counts: dict[int, int], area: int, ratio: float) -> float:
    bins = 4096
    cap = ratio * area / bins
    kept = {index: min(count, cap) for index, count in counts.items()}
    base = (area - sum(kept.values()) + 256) / bins

    def lookup(index: int) -> float:
        return (base * (index + 1) + sum(count for k, count in kept.items() if k <= index)) / (area + 256)

    coordinate = value * (bins - 1)
    lo = int(math.floor(coordinate))
    hi = min(lo + 1, bins - 1)
    return lookup(lo) + (coordinate - lo) * (lookup(hi) - lookup(lo))


def mirror(index: int, size: int) -> int:
    if size == 1:
        return 0
    position = index % (2 * (size - 1))
    return position if position < size else 2 * (size - 1) - position


def nlm_oracle(source: np.ndarray, strength: float, window: int, patch: int) -> np.ndarray:
    image = source.astype(np.float64)
    height, width = image.shape
    result = np.empty_like(image)
    radius, patch_radius = window // 2, patch // 2
    for y in range(height):
        for x in range(width):
            numerator, denominator = 0.0, 0.0
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    distance = 0.0
                    for py in range(-patch_radius, patch_radius + 1):
                        for px in range(-patch_radius, patch_radius + 1):
                            r, c = mirror(y + py, height), mirror(x + px, width)
                            a = image[r, c]
                            b = image[mirror(r + dy, height), mirror(c + dx, width)]
                            distance += (a - b)**2
                    distance /= patch**2
                    weight = math.exp(-distance / strength**2)
                    numerator += weight * image[mirror(y + dy, height), mirror(x + dx, width)]
                    denominator += weight
            result[y, x] = numerator / denominator
    return result


@pytest.mark.parametrize("shape", ((1, 1), (3, 5), (9, 7)))
@pytest.mark.parametrize("level", (0.0, 0.2, 1.0))
def test_constant_small_and_partial_clahe_tiles_are_identity(shape: tuple[int, int], level: float) -> None:
    source = np.full(shape, level, np.float32)
    actual = clahe(source, scale=1, wind=64, clip_limit=2)
    np.testing.assert_array_equal(actual, source)
    assert actual.dtype == np.float32


def test_clahe_retains_total_prior_mass_and_interpolates_intensity() -> None:
    source = np.array([[0, 0.25], [0.75, 1]], np.float32)
    expected = np.array([[clipped_cdf(float(value), {0: 1, 1024: 1, 3071: 1, 4095: 1}, 4, 2)
                          for value in row] for row in source])
    actual = clahe(source, scale=1, wind=2, clip_limit=2)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)


def test_clahe_spatial_weights_are_normalized_between_tile_centers() -> None:
    source = np.array([[0, 0.25, 0.75, 1], [0, 0.25, 0.75, 1]], np.float32)
    left, right = {0: 2, 1024: 2}, {3071: 2, 4095: 2}
    expected = np.empty((2, 4))
    for x, weight in enumerate((0, 0.25, 0.75, 1)):
        value = float(source[0, x])
        expected[:, x] = (1 - weight) * clipped_cdf(value, left, 4, 2) + weight * clipped_cdf(value, right, 4, 2)
    np.testing.assert_allclose(clahe(source, scale=1, wind=2, clip_limit=2), expected, rtol=0, atol=1e-7)


def test_clahe_constant_detection_does_not_discard_sub_bin_contrast() -> None:
    source = np.array([[0.2, 0.2000001], [0.2, 0.2000001]], np.float32)
    actual = clahe(source, scale=1, wind=2, clip_limit=2)
    assert actual[0, 0] != actual[0, 1]


def test_luma_scale_is_explicit_not_inferred_from_dtype_or_peak() -> None:
    codes = np.array([[0, 16384], [49151, 65535]], np.uint16)
    normalized = codes.astype(np.float64) / 65535
    result = clahe(codes, scale=65535, wind=2, clip_limit=2)
    reference = clahe(normalized, scale=1, wind=2, clip_limit=2)
    np.testing.assert_allclose(result / 65535, reference, rtol=0, atol=1e-7)
    assert result.dtype == np.float32


def test_nlm_includes_the_full_black_white_distance_endpoint() -> None:
    source = np.array([[0, 1]], np.float32)
    left = 2 / (math.e + 2)
    np.testing.assert_allclose(
        nlm(source, scale=1, window_size=3, patch_size=1, strength=1),
        [[left, 1 - left]], rtol=0, atol=1e-7,
    )


@pytest.mark.parametrize("patch", (1, 3))
def test_nlm_float_patch_distances_match_independent_reflected_loop(patch: int) -> None:
    source = np.array([[0, 0.1, 0.4], [0.8, 0.2, 0.3], [1, 0.75, 0.5]], np.float32)
    original = source.copy()
    expected = nlm_oracle(source, 0.2, 3, patch)
    result = nlm(source, scale=1, window_size=3, patch_size=patch, strength=0.2)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-7)
    np.testing.assert_array_equal(source, original)
    assert result.dtype == np.float32


@pytest.mark.parametrize("strength", (0, 1e-300))
def test_nlm_zero_strength_limit_has_self_weight_without_nan(strength: float) -> None:
    source = np.array([[0, 0.2, 0.75, 1]], np.float32)
    with np.errstate(all="raise"):
        actual = nlm(source, scale=1, window_size=3, patch_size=1, strength=strength)
    np.testing.assert_array_equal(actual, source)


def test_nlm_reduces_flat_field_noise_without_destroying_high_contrast_edge() -> None:
    rng = np.random.default_rng(422)
    source = (0.5 + rng.normal(0, 0.01, (16, 16))).astype(np.float32)
    result = nlm(source, scale=1, window_size=3, patch_size=3, strength=0.05)
    assert np.mean((result.astype(np.float64) - 0.5)**2) < np.mean((source.astype(np.float64) - 0.5)**2)
    edge = np.zeros((5, 9), np.float32)
    edge[:, 4:] = 1
    np.testing.assert_array_equal(nlm(edge, scale=1, window_size=3, patch_size=1, strength=0.01), edge)


def test_gaussian_sharpen_uses_declared_normalized_scale_and_reflected_edges() -> None:
    source = np.full((3, 5), 0.5, np.float32)
    source[1, 2] = 0.75
    sigma, strength, radius = 0.75, 0.5, 3
    kernel = [math.exp(-0.5 * (i / sigma)**2) for i in range(-radius, radius + 1)]
    total = sum(kernel)
    kernel = [value / total for value in kernel]

    def reflect(index: int, size: int) -> int:
        position = index % (2 * size)
        return position if position < size else 2 * size - 1 - position

    expected = np.empty(source.shape)
    for y in range(3):
        for x in range(5):
            smooth = sum(
                kernel[dy + radius] * kernel[dx + radius] * float(source[reflect(y + dy, 3), reflect(x + dx, 5)])
                for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)
            )
            expected[y, x] = min(max(float(source[y, x]) + strength * (float(source[y, x]) - smooth), 0), 1)
    np.testing.assert_allclose(sharpen(source, scale=1, sigma=sigma, strength=strength), expected, rtol=0, atol=1e-7)


def test_luma_recombination_preserves_chroma_without_gamut_clipping() -> None:
    rgb = np.array([[[0.2, 0.4, 0.6], [-0.1, 0.3, 1.5]]], np.float32)
    before = rgb_to_ycbcr(rgb, matrix="bt709", dtype=np.float64)
    target = before[..., 0] + 0.1
    result = replace_luminance(rgb, target, matrix="bt709")
    after = rgb_to_ycbcr(result, matrix="bt709", dtype=np.float64)
    np.testing.assert_allclose(after[..., 0], target, rtol=0, atol=1e-7)
    np.testing.assert_allclose(after[..., 1:], before[..., 1:], rtol=0, atol=1e-7)
    assert result[0, 1, 2] > 1


@pytest.mark.parametrize("function,kwargs", (
    (clahe, {"wind": 2, "clip_limit": 2}),
    (nlm, {"window_size": 3, "patch_size": 3, "strength": 0.1}),
    (sharpen, {"sigma": 1, "strength": 0.5}),
))
def test_luma_planes_reject_empty_nonfinite_and_undeclared_scale(function, kwargs: dict) -> None:
    for data in (np.empty((0, 2)), np.array([[float("nan")]]), np.zeros((2, 2, 3)), np.array([[True]])):
        with pytest.raises(ValueError):
            function(data, scale=1, **kwargs)
    for scale in (0, -1, True, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="scale"):
            function(np.ones((2, 2)), scale=scale, **kwargs)
