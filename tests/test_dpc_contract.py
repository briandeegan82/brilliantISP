from __future__ import annotations

import numpy as np
import pytest

from typing import cast

from modules.dead_pixel_correction.dynamic_dpc import DynamicDPC
from modules.dead_pixel_correction.dynamic_dpc_numba_optimized import DynamicDPCNumbaOptimized
from util.isp_types import DeadPixelCorrectionConfig, SensorInfo


def oracle(image: np.ndarray, threshold: float, bits: int) -> np.ndarray:
    height, width = image.shape
    maximum = 2**bits - 1

    def mirror(index: int, size: int) -> int:
        if size == 1:
            return 0
        position = index % (2 * (size - 1))
        return position if position < size else 2 * (size - 1) - position

    def sample(y: int, x: int) -> float:
        return float(image[mirror(y, height), mirror(x, width)])

    directions = (((-2, 0), (2, 0)), ((0, -2), (0, 2)),
                  ((-2, -2), (2, 2)), ((-2, 2), (2, -2)))
    result = np.empty(image.shape, np.uint16 if bits <= 16 else np.uint32)
    for y in range(height):
        for x in range(width):
            center = float(image[y, x])
            pairs = [(sample(y + a[0], x + a[1]), sample(y + b[0], x + b[1]))
                     for a, b in directions]
            neighbors = [value for pair in pairs for value in pair]
            outside = center < min(neighbors) or center > max(neighbors)
            if outside and all(abs(center - value) > threshold for value in neighbors):
                gradients = [abs(2 * center - a - b) for a, b in pairs]
                index = min(range(4), key=lambda k: gradients[k])
                center = sum(pairs[index]) / 2
            result[y, x] = int(min(max(center, 0), maximum))
    return result


def sensor(image: np.ndarray, bits: int, pattern: str) -> SensorInfo:
    return cast(SensorInfo, {
        "height": image.shape[0], "width": image.shape[1],
        "bit_depth": bits, "hdr_bit_depth": 32,
        "bayer_pattern": pattern,
    })


PARAMS = cast(DeadPixelCorrectionConfig, {
    "is_enable": True, "is_save": False, "is_debug": False, "dp_threshold": 8,
})


@pytest.mark.parametrize("pattern", ("rggb", "bggr", "grbg", "gbrg"))
@pytest.mark.parametrize("shape", ((1, 1), (1, 7), (2, 3), (5, 7), (8, 10)))
def test_cpu_and_optimized_preparation_match_independent_reflected_oracle(pattern: str, shape: tuple[int, int]) -> None:
    rng = np.random.default_rng(421)
    image = rng.integers(0, 4096, size=shape, dtype=np.uint16)
    expected = oracle(image, 8, 12)
    cpu = DynamicDPC(image.copy(), sensor(image, 12, pattern), PARAMS).dynamic_dpc()
    optimized = DynamicDPCNumbaOptimized(image.copy(), sensor(image, 12, pattern), PARAMS)
    optimized.use_numba = False
    actual = optimized.dynamic_dpc()
    np.testing.assert_array_equal(cpu, expected)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("site", ((0, 0), (0, 6), (4, 0), (4, 6), (2, 2)))
def test_hot_pixel_uses_only_same_color_and_never_opposite_edge(site: tuple[int, int]) -> None:
    image = np.full((5, 7), 100, np.uint16)
    image[0::2, 1::2] = 1000
    image[1::2, 0::2] = 1000
    image[1::2, 1::2] = 10
    image[site] = 1000
    expected = oracle(image, 8, 12)
    assert expected[site] == 100
    dpc = DynamicDPCNumbaOptimized(image, sensor(image, 12, "rggb"), PARAMS)
    dpc.use_numba = False
    actual, mask = dpc.dynamic_dpc(return_mask=True)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(mask.astype(bool), actual != image)


@pytest.mark.parametrize("backend", ("reference", "optimized"))
def test_unsigned_32_bit_differences_do_not_wrap_and_source_is_not_modified(backend: str) -> None:
    image = np.array([
        [0, 2**32 - 1, 7, 2**32 - 2, 0],
        [2**31, 1, 2**31 + 1, 2, 2**31],
        [2**32 - 1, 4, 0, 5, 2**32 - 1],
        [2**31 + 2, 3, 2**31 - 1, 1, 2**31 + 2],
        [0, 2**32 - 1, 7, 2**32 - 2, 0],
    ], np.uint32)
    original = image.copy()
    expected = oracle(image, 8, 32)
    cls = DynamicDPC if backend == "reference" else DynamicDPCNumbaOptimized
    dpc = cls(image, sensor(image, 32, "bggr"), PARAMS)
    if isinstance(dpc, DynamicDPCNumbaOptimized):
        dpc.use_numba = False
    np.testing.assert_array_equal(dpc.dynamic_dpc(), expected)
    np.testing.assert_array_equal(image, original)


def test_pre_decompanding_range_uses_sensor_depth_not_hdr_target_depth() -> None:
    image = np.full((5, 7), 100, np.uint16)
    image[2, 2] = 4095
    for cls in (DynamicDPC, DynamicDPCNumbaOptimized):
        dpc = cls(image, sensor(image, 12, "rggb"), PARAMS)
        if isinstance(dpc, DynamicDPCNumbaOptimized):
            dpc.use_numba = False
        result = cast(np.ndarray, dpc.dynamic_dpc())
        assert result.dtype == np.uint16
        np.testing.assert_array_equal(result, oracle(image, 8, 12))


def test_numba_correction_matches_the_same_oracle_when_available() -> None:
    import modules.dead_pixel_correction.dynamic_dpc_numba_optimized as implementation

    if not implementation.NUMBA_AVAILABLE:
        pytest.skip("Numba is optional")
    rng = np.random.default_rng(422)
    image = rng.integers(0, 65536, size=(7, 9), dtype=np.uint16)
    dpc = DynamicDPCNumbaOptimized(image, sensor(image, 16, "grbg"), PARAMS)
    dpc.use_numba = True
    np.testing.assert_array_equal(dpc.dynamic_dpc(), oracle(image, 8, 16))
