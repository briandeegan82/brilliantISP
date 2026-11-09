import numpy as np
import pytest

from modules.tone_mapping.native import map_native, map_native_max_rgb
from modules.tone_mapping.registry import ToneMappingError, ToneMappingSpec


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


def test_hybrid_midtones_follow_luminance_and_highlights_stay_in_range():
    from modules.tone_mapping.native import map_native_hybrid

    spec = ToneMappingSpec("reinhard", backend="float")
    mid = np.array([[[0.10, 0.12, 0.08], [0.05, 0.06, 0.09]]], dtype=np.float32)
    assert np.allclose(map_native_hybrid(mid, config=spec), map_native(mid, config=spec), atol=1e-6)
    hot = np.array([[[0.02, 0.05, 0.9], [0.4, 0.3, 0.05]]], dtype=np.float32) * 8.0
    out = np.asarray(map_native_hybrid(hot, config=spec), dtype=np.float64)
    assert out.max() <= 1.0 + 1e-6
    assert np.allclose(out, map_native_max_rgb(hot, config=spec), atol=1e-6)


def test_hybrid_wb_step_on_grey_midtone_keeps_luma():
    from modules.tone_mapping.native import map_native_hybrid
    from util.color import luminance

    spec = ToneMappingSpec("reinhard", backend="float")
    a = np.array([[[0.10, 0.12, 0.08]]], dtype=np.float32)
    b = a * np.array([1.0, 1.0, 1.15], dtype=np.float32)
    b = b * float(luminance(a.astype(np.float64)).sum() / luminance(b.astype(np.float64)).sum())
    ya = luminance(np.asarray(map_native_hybrid(a, config=spec), np.float64)).sum()
    yb = luminance(np.asarray(map_native_hybrid(b, config=spec), np.float64)).sum()
    assert abs(ya - yb) < 1e-6


def test_hybrid_near_white_desat_and_validation():
    from modules.tone_mapping.native import map_native_hybrid

    spec = ToneMappingSpec("reinhard", backend="float")
    lamp = np.array([[[40.0, 20.0, 2.0]]], dtype=np.float32)
    out = np.asarray(map_native_hybrid(lamp, config=spec, desat=(0.5, 0.9)), np.float64)
    assert np.ptp(out) < 1e-6 and out.max() <= 1.0 + 1e-6
    from util.color import luminance
    plain = np.asarray(map_native_hybrid(lamp, config=spec), np.float64)
    assert abs(luminance(out).sum() - luminance(plain).sum()) < 1e-6
    with pytest.raises(ToneMappingError):
        map_native_hybrid(lamp, config=spec, blend=(1.0, 0.5))


def test_adaptive_white_desired_and_filter():
    from modules.tone_mapping import adaptive_white as aw

    cfg = {"percentile": 99.0, "max_white": 4.0, "time_constant_s": 1.0, "frame_rate": 10.0, "meter_stride": 1}
    img = np.full((10, 10, 3), 0.2)
    assert aw.desired_white(img, cfg) == 1.0
    img[:5] = 9.0
    assert aw.desired_white(img, cfg) == 4.0
    w = aw.next_white(1.0, 4.0, cfg)
    assert 1.0 < w < 4.0
    assert aw.next_white(None, 2.0, cfg) == 2.0
    with pytest.raises(ValueError):
        aw.next_white(1.0, 2.0, {"max_white": 0.5})
