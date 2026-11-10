"""Tests for the max_rgb / max_blend RGB norms and adaptive white."""

# pylint: disable=missing-function-docstring
import numpy as np
import pytest

from modules.tone_mapping import adaptive_white as aw
from modules.tone_mapping.native import map_native, map_native_max_blend, map_native_max_rgb
from modules.tone_mapping.registry import ToneMappingError, ToneMappingSpec
from util.color import luminance


@pytest.mark.parametrize("operator", ("reinhard", "hable", "aces_fitted"))
def test_max_rgb_keeps_saturated_channels_in_range_and_preserves_hue(operator):
    spec = ToneMappingSpec(operator, backend="float")
    rgb = np.array([[[0.02, 0.05, 0.9], [0.4, 0.3, 0.05], [0.3, 0.3, 0.3]]], dtype=np.float32) * 8.0
    out = np.asarray(map_native_max_rgb(rgb, config=spec), dtype=np.float64)
    lum_out = np.asarray(map_native(rgb, config=spec), dtype=np.float64)
    assert out.max() <= 1.0 + 1e-6
    assert lum_out[0, 0].max() > out[0, 0].max()
    ratio = out / rgb
    assert np.allclose(ratio, ratio[..., :1], rtol=1e-5)
    assert np.all(np.diff(out[0, 2]) == 0)


def test_max_rgb_black_and_validation():
    spec = ToneMappingSpec("reinhard", backend="float")
    out = np.asarray(map_native_max_rgb(np.zeros((2, 2, 3), np.float32), config=spec))
    assert np.all(out == 0)
    with pytest.raises(ToneMappingError):
        map_native_max_rgb(np.zeros((2, 2), np.float32), config=spec)


C2_COLOURS = {
    "red": (1.0, 0.12, 0.08),
    "sodium": (1.0, 0.55, 0.15),
    "sky": (0.45, 0.70, 1.0),
    "pale_sky": (0.75, 0.85, 1.0),
    "green": (0.2, 1.0, 0.3),
    "yellow": (1.0, 0.9, 0.1),
    "grey": (1.0, 1.0, 1.0),
}


@pytest.mark.parametrize("operator", ("reinhard", "hable", "aces_fitted"))
@pytest.mark.parametrize("k", (0.0, 0.5, 0.7, 0.85, 1.0))
@pytest.mark.parametrize("desat", (None, (0.8, 0.97)))
def test_max_blend_output_luma_is_monotone_in_exposure(operator, k, desat):
    """C2: fixed scene colour, exposure swept over 4.5 decades; no output-luma reversal > 0.1%."""

    spec = ToneMappingSpec(operator, backend="float")
    s = np.logspace(-3, 1.5, 4000)
    for name, c in C2_COLOURS.items():
        img = (s[:, None] * np.array(c))[None].astype(np.float32)
        y = luminance(
            np.asarray(map_native_max_blend(img, config=spec, k=k, desat=desat), np.float64)
        )[0]
        peak = np.maximum.accumulate(y)
        drop = float(np.max((peak - y) / np.maximum(peak, 1e-9)))
        assert drop <= 1e-3, (name, drop)
        assert y[-1] > y[0]


def test_max_blend_endpoints_and_midtone_luma():

    spec = ToneMappingSpec("reinhard", backend="float")
    hot = (
        np.array([[[0.02, 0.05, 0.9], [0.4, 0.3, 0.05], [0.10, 0.12, 0.08]]], dtype=np.float32)
        * 4.0
    )
    assert np.allclose(
        map_native_max_blend(hot, config=spec, k=1.0),
        map_native_max_rgb(hot, config=spec),
        atol=1e-6,
    )
    assert np.allclose(
        map_native_max_blend(hot, config=spec, k=0.0), map_native(hot, config=spec), atol=1e-6
    )
    # mildly coloured midtone with k*max <= Y follows luminance,
    # and a grey-preserving WB step keeps its luma
    a = np.array([[[0.10, 0.12, 0.08]]], dtype=np.float32)
    b = a * np.array([1.0, 1.0, 1.15], dtype=np.float32)
    b = b * float(luminance(a.astype(np.float64)).sum() / luminance(b.astype(np.float64)).sum())
    ya = luminance(np.asarray(map_native_max_blend(a, config=spec, k=0.7), np.float64)).sum()
    yb = luminance(np.asarray(map_native_max_blend(b, config=spec, k=0.7), np.float64)).sum()
    assert abs(ya - yb) < 1e-6


def test_max_blend_luma_matches_ae_model():
    """Smooth AE models output luma as f(Y*r)/r, r = max(1, k*max/Y) (model_rgb_norm)."""

    spec = ToneMappingSpec("reinhard", backend="float")
    rng = np.random.default_rng(0)
    rgb = rng.uniform(0.0, 2.0, (8, 8, 3)) * rng.uniform(0.01, 1.0, (8, 8, 1))
    k = 0.7
    y = luminance(rgb.astype(np.float64))
    r = np.maximum(1.0, k * rgb.max(-1) / y)
    model = np.asarray(map_native(y * r, config=spec), np.float64) / r
    out = luminance(
        np.asarray(map_native_max_blend(rgb.astype(np.float32), config=spec, k=k), np.float64)
    )
    assert np.allclose(out, model, rtol=1e-5, atol=1e-6)


def test_max_blend_near_white_desat_and_validation():

    spec = ToneMappingSpec("reinhard", backend="float")
    lamp = np.array([[[40.0, 20.0, 2.0]]], dtype=np.float32)
    out = np.asarray(map_native_max_blend(lamp, config=spec, k=0.7, desat=(0.5, 0.9)), np.float64)
    plain = np.asarray(map_native_max_blend(lamp, config=spec, k=0.7), np.float64)
    assert np.ptp(out) < 1e-6 and out.max() <= 1.0 + 1e-6
    assert abs(luminance(out).sum() - luminance(plain).sum()) < 1e-6
    with pytest.raises(ToneMappingError):
        map_native_max_blend(lamp, config=spec, k=1.5)
    with pytest.raises(ToneMappingError):
        map_native_max_blend(lamp, config=spec, desat=(0.9, 0.5))


def test_adaptive_white_desired_and_filter():

    cfg = {
        "percentile": 99.0,
        "max_white": 4.0,
        "time_constant_s": 1.0,
        "frame_rate": 10.0,
        "meter_stride": 1,
    }
    img = np.full((10, 10, 3), 0.2)
    assert aw.desired_white(img, cfg) == 1.0
    img[:5] = 9.0
    assert aw.desired_white(img, cfg) == 4.0
    w = aw.next_white(1.0, 4.0, cfg)
    assert 1.0 < w < 4.0
    assert aw.next_white(None, 2.0, cfg) == 2.0
    with pytest.raises(ValueError):
        aw.next_white(1.0, 2.0, {"max_white": 0.5})
