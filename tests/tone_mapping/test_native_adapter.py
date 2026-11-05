from __future__ import annotations

import numpy as np
import pytest

from modules.tone_mapping.native import map_native
from modules.tone_mapping.normalization import FrameStats
from modules.tone_mapping.registry import ToneMappingError, ToneMappingSpec


def reinhard(x: np.ndarray) -> np.ndarray:
    return x / (x + 0.25)


def test_disabled_returns_same_object_without_work() -> None:
    spec = ToneMappingSpec("bogus", backend="bogus", enabled=False)
    src = np.array([[np.nan]], np.float32)
    assert map_native(src, config=spec) is src


@pytest.mark.parametrize("backend,atol", (("float", 3e-7), ("lut", 2 / 65535), ("fixed", 2 / 65535)))
def test_bayer_reinhard_known_values(backend: str, atol: float) -> None:
    src = np.array([[0.0, 0.25, 1.0, 2.0]], np.float64)
    out = map_native(src, config=ToneMappingSpec("reinhard", backend=backend))
    expected = reinhard(np.array([[0.0, 0.25, 1.0, 2.0]]))
    np.testing.assert_allclose(out, expected, rtol=0, atol=atol)
    assert out.dtype == np.float32


@pytest.mark.parametrize("backend", ("float", "lut", "fixed"))
def test_rgb_hdr_and_zero_luma(backend: str) -> None:
    src = np.array([[[2.0, 0.0, 0.0]]], np.float64)
    out = map_native(src, config=ToneMappingSpec("reinhard", backend=backend))
    assert out[0, 0, 0] > 1.0
    assert np.all(np.isfinite(out))
    black = np.zeros((2, 2, 3), np.float64)
    np.testing.assert_array_equal(
        map_native(black, config=ToneMappingSpec("reinhard", backend=backend)), 0.0,
    )


def test_rgb_saturation_zero_maps_all_channels_by_luma() -> None:
    from util.color import adjust_saturation

    src = np.array([[[2.0, 0.0, 0.0]]], np.float64)
    spec = ToneMappingSpec("reinhard")
    out = map_native(src, config=spec)
    grey = adjust_saturation(out, 0.0)
    mapped_y = 0.4252 / (0.4252 + 0.25)
    np.testing.assert_allclose(grey[0, 0], mapped_y, rtol=0, atol=3e-7)
    np.testing.assert_allclose(grey[..., 0], grey[..., 1], rtol=0, atol=1e-7)
    np.testing.assert_allclose(grey[..., 1], grey[..., 2], rtol=0, atol=1e-7)


def test_rgb_near_black_luma_maps_to_black() -> None:
    src = np.array([[[1e-15, 0.0, 0.0]]], np.float64)
    out = map_native(src, config=ToneMappingSpec("reinhard"))
    np.testing.assert_array_equal(out, 0.0)


def test_tiny_per_frame_fixed_retains_contrast() -> None:
    src = np.array([[1e-9, 2e-9, 3e-9]], np.float64)
    spec = ToneMappingSpec("reinhard", backend="fixed", normalization="per_frame")
    fixed = map_native(src, config=spec)
    flt = map_native(src, config=ToneMappingSpec("reinhard", backend="float", normalization="per_frame"))
    np.testing.assert_allclose(fixed, flt, rtol=0, atol=2 / 65535)


def test_fixed_saturated_coordinate_and_uncapped_overflow() -> None:
    ok = map_native(
        np.array([[256.0]], np.float64),
        config=ToneMappingSpec("reinhard", backend="fixed"),
    )
    assert np.isfinite(ok[0, 0])
    with pytest.raises(ToneMappingError):
        map_native(
            np.array([[512.0]], np.float64),
            config=ToneMappingSpec("reinhard", backend="fixed", exposure_ev=-32),
        )


def test_invalid_inputs_rejected() -> None:
    spec = ToneMappingSpec("reinhard")
    with pytest.raises(ToneMappingError):
        map_native(np.zeros((2, 2), np.uint16), config=spec)
    with pytest.raises(ToneMappingError):
        map_native(np.full((2, 2), np.nan, np.float64), config=spec)
    with pytest.raises(ToneMappingError):
        map_native(np.zeros((0, 0), np.float64), config=spec)
    with pytest.raises(ToneMappingError):
        map_native(np.zeros((2, 2, 2), np.float64), config=spec)
