from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pytest

from modules.auto_exposure.auto_exposure import AutoExposure
from modules.auto_white_balance.auto_white_balance import AutoWhiteBalance
from util.isp_types import AutoExposureConfig, GenericConfig, SensorInfo, WhiteBalanceConfig


def sensor(pattern: str = "rggb", bits: int = 12, hdr: int = 32) -> SensorInfo:
    return cast(SensorInfo, {
        "height": 8, "width": 8, "bit_depth": bits, "hdr_bit_depth": hdr,
        "bayer_pattern": pattern, "output_bit_depth": 8,
        "pipeline_rgb_bit_depth": 16,
    })


def awb_cfg(algorithm: str = "grey_world") -> GenericConfig:
    return cast(GenericConfig, {
        "is_enable": True, "is_debug": False, "algorithm": algorithm,
        "percentage": 20, "underexposed_percentage": 0.0,
        "overexposed_percentage": 0.0,
    })


def wbc(r: Any = 1.2, b: Any = 0.8) -> WhiteBalanceConfig:
    return cast(WhiteBalanceConfig, {"r_gain": r, "b_gain": b, "is_enable": True, "is_save": False})


@pytest.mark.parametrize("pattern", ("rggb", "bggr", "grbg", "gbrg"))
@pytest.mark.parametrize("algorithm", ("grey_world", "norm_2", "pca"))
def test_awb_unsigned_green_and_known_gains(pattern, algorithm) -> None:
    img = np.full((6, 6), 2**31, np.uint32)
    awb = AutoWhiteBalance(img, sensor(pattern), awb_cfg(algorithm), wbc())
    awb.parm_awb = dict(awb.parm_awb)
    # R half of G/B: expected gains (2,1)
    if pattern == "rggb":
        img[0::2, 0::2] = 2**30
    elif pattern == "bggr":
        img[1::2, 1::2] = 2**30
    elif pattern == "grbg":
        img[0::2, 1::2] = 2**30
    else:
        img[1::2, 0::2] = 2**30
    awb2 = AutoWhiteBalance(img, sensor(pattern), awb_cfg(algorithm), wbc())
    r, b = awb2.compute_gains()
    np.testing.assert_allclose((r, b), (2.0, 1.0), rtol=0, atol=1e-6)
    assert awb2.last_status == "estimated" and awb2.sample_count > 0


@pytest.mark.parametrize("pattern", ("rggb", "bggr", "grbg", "gbrg"))
@pytest.mark.parametrize("algorithm", ("grey_world", "norm_2", "pca"))
@pytest.mark.parametrize("input_gain", (0.25, 1.0, 8.0))
def test_awb_compensated_gain_preserves_samples_and_illuminant(pattern, algorithm, input_gain) -> None:
    image = np.full((8, 8), 1000, np.float32)
    blue_row, blue_col = divmod(pattern.index("b"), 2)
    image[blue_row::2, blue_col::2] = 2000
    image[blue_row, blue_col] = 4095
    params = {**awb_cfg(algorithm), "underexposed_percentage": 0.05, "overexposed_percentage": 0.05}
    awb = AutoWhiteBalance(image * input_gain, sensor(pattern, hdr=12), params, wbc())
    result = awb.compute_gains(input_gain=input_gain)
    assert awb.last_status == "estimated"
    assert awb.sample_count == 15
    np.testing.assert_allclose(result, (1.0, 0.5), rtol=0, atol=1e-7)


@pytest.mark.parametrize("input_gain", (True, 0, -1, None, "1", math.nan, math.inf))
def test_awb_invalid_active_input_gain(input_gain) -> None:
    image = np.full((8, 8), 1000, np.float32)
    awb = AutoWhiteBalance(image, sensor(hdr=12), awb_cfg(), wbc())
    with pytest.raises(ValueError, match="input_gain"):
        awb.compute_gains(input_gain=input_gain)
    awb.enable = False
    assert awb.compute_gains(input_gain=input_gain) == (1.2, 0.8)


def test_awb_odd_frame_and_fallback_paths() -> None:
    img = np.full((5, 7), 100, np.uint16)
    out = AutoWhiteBalance(img, sensor(), awb_cfg(), wbc()).compute_gains()
    assert np.all(np.isfinite(out))
    tiny = np.zeros((1, 1), np.uint16)
    awb = AutoWhiteBalance(tiny, sensor(), awb_cfg(), wbc())
    out = awb.compute_gains(fallback_gains=(1.6, 0.9))
    assert out == (1.6, 0.9) and awb.last_status == "fallback_no_samples"
    dis = AutoWhiteBalance(img, sensor(), cast(GenericConfig, {**awb_cfg(), "is_enable": False}), wbc(1.2, 0.8))
    assert dis.compute_gains(fallback_gains=(9.0, 9.0)) == (1.2, 0.8)
    bad_manual = AutoWhiteBalance(tiny, sensor(), awb_cfg(), wbc(-1.0, -1.0))
    assert bad_manual.compute_gains() == (1.0, 1.0)
    zero_manual = AutoWhiteBalance(tiny, sensor(), awb_cfg(), wbc(0.0, 8.0))
    assert zero_manual.compute_gains() == (0.0, 8.0)
    unknown = AutoWhiteBalance(img, sensor(), awb_cfg("bogus"), wbc())
    with pytest.raises(ValueError):
        unknown.compute_gains()


def ae_cfg(center=0.5, skew=0.9) -> AutoExposureConfig:
    return cast(AutoExposureConfig, {
        "is_enable": True, "is_debug": False,
        "center_illuminance": center, "histogram_skewness": skew,
    })


def test_ae_native_domain_meter_and_gain() -> None:
    img = np.full((2, 3, 3), 0.18, np.float32)
    ae = AutoExposure(img, sensor(), ae_cfg(0.5, 0.9), meter_domain="scene_linear_rgb")
    fb = ae.get_exposure_feedback()
    assert ae.last_meter_average == pytest.approx(0.18, abs=1e-7)
    assert fb in (-1, 0, 1)
    idx = ae.suggest_direct_gain_index(0, [1.0, 2.0, 4.0])
    assert idx == 1
    ae5 = AutoExposure(
        np.full((2, 2, 3), 0.5, np.float32), sensor(), ae_cfg(0.5, 0.5),
        meter_domain="scene_linear_rgb")
    assert ae5.get_exposure_feedback() == 0
    hdr = np.full((4, 4, 3), 2.0, np.float32)
    ae6 = AutoExposure(hdr, sensor(), cast(AutoExposureConfig, {**ae_cfg(0.5, 0.9), "target_luminance": 2.5}), meter_domain="scene_linear_rgb")
    ae6.get_exposure_feedback()
    assert ae6.last_meter_average == pytest.approx(2.0)
    assert ae6.suggest_direct_gain_index(0, [1.0, 2.0, 4.0]) == 0
    empty = AutoExposure(np.zeros((0, 0, 3), np.float32), sensor(), ae_cfg(), meter_domain="scene_linear_rgb")
    assert empty.get_exposure_feedback() is None
    black = AutoExposure(np.zeros((2, 2, 3), np.float32), sensor(), ae_cfg(), meter_domain="scene_linear_rgb")
    assert black.get_exposure_feedback() is None
    with pytest.raises(ValueError):
        AutoExposure(np.full((2, 2, 3), np.nan, np.float32), sensor(), ae_cfg(), meter_domain="scene_linear_rgb").get_exposure_feedback()
    orig = img.copy()
    ae7 = AutoExposure(img, sensor(), ae_cfg(), meter_domain="scene_linear_rgb")
    ae7.get_exposure_feedback()
    np.testing.assert_array_equal(img, orig)


def test_ae_native_one_two_pixel_finite_under_raise() -> None:
    for n in (1, 2):
        img = np.full((1, n, 3), 0.5, np.float32)
        with np.errstate(all="raise"):
            out = AutoExposure(img, sensor(), ae_cfg(), meter_domain="scene_linear_rgb").get_exposure_feedback()
        assert out in (-1, 0, 1)


def test_awb_pca_lead_golden_and_execute_match() -> None:
    rows = np.array([[1,5,1],[10,1,1],[1,1,10],[20,10,2],
                     [2,20,5],[3,4,20],[9,3,1],[1,2,9]], np.float64)
    raw = np.zeros((2, 16), np.uint32)
    for i in range(8):
        raw[0, 2 * i] = rows[i, 0]
        raw[0, 2 * i + 1] = rows[i, 1]
        raw[1, 2 * i] = rows[i, 1]
        raw[1, 2 * i + 1] = rows[i, 2]
    parm = awb_cfg("pca")
    parm = cast(GenericConfig, dict(parm))
    parm["percentage"] = 12.5
    awb = AutoWhiteBalance(raw, sensor("rggb", hdr=32), parm, wbc())
    r, b = awb.compute_gains()
    d = rows.mean(0)
    d /= np.linalg.norm(d)
    order = np.argsort(rows @ d)
    selected = rows[order[[0, -1]]]
    w, v = np.linalg.eigh(selected.T @ selected)
    a = np.abs(v[:, int(np.argmax(w))])
    expected = np.clip([a[1] / a[0], a[1] / a[2]], 0.25, 4.0)
    np.testing.assert_allclose((r, b), expected, rtol=0, atol=1e-6)
    # legacy execute() uses the same guarded estimation path
    legacy = awb.execute()
    np.testing.assert_allclose(legacy, expected, rtol=0, atol=1e-6)


def test_awb_nan_row_filtered_and_unknown_on_empty() -> None:
    img = np.full((4, 4), 1000, np.float64)
    img[0, 0] = np.nan
    awb = AutoWhiteBalance(img, sensor(), awb_cfg(), wbc())
    r, b = awb.compute_gains()
    assert np.isfinite(r) and np.isfinite(b)
    assert awb.sample_count > 0
    unknown = AutoWhiteBalance(np.zeros((2, 2)), sensor(), awb_cfg("bogus"), wbc())
    with pytest.raises(ValueError):
        unknown.compute_gains()
    malformed = AutoWhiteBalance(np.zeros((2, 2)), sensor(), awb_cfg(), wbc())
    assert malformed.compute_gains(fallback_gains=cast(Any, "bad")) == (1.2, 0.8)


def test_ae_native_center_and_target_are_independent() -> None:
    img = np.full((4, 4, 3), 0.25, np.float32)
    cfg = ae_cfg(0.1, 0.9)
    cfg = cast(AutoExposureConfig, cast(Any, dict(cfg, target_luminance=0.9)))
    ae = AutoExposure(img, sensor(), cfg, meter_domain="scene_linear_rgb")
    fb = ae.get_exposure_feedback()
    assert fb == 1
    assert ae.suggest_direct_gain_index(0, [1.0, 2.0, 4.0]) == 2
    black = AutoExposure(np.zeros((2, 2, 3), np.float32), sensor(), cfg,
                         meter_domain="scene_linear_rgb")
    assert black.get_exposure_feedback() is None
    assert black.last_meter_average == 0.0
    assert black.suggest_direct_gain_index(0, [1.0, 2.0]) == 0
    assert black.get_luminance_histogram_skewness(np.empty((0, 0)), 0.5) == 0.0
    with pytest.raises(ValueError):
        AutoExposure(np.zeros((2, 2), np.float32), sensor(), cfg,
                     meter_domain="scene_linear_rgb").get_exposure_feedback()
    bad_target = cast(AutoExposureConfig, cast(Any, dict(cfg, target_luminance=np.nan)))
    with pytest.raises(ValueError):
        AutoExposure(img, sensor(), bad_target,
                     meter_domain="scene_linear_rgb").get_exposure_feedback()


def test_replace_luminance_rejects_nonfloat_rgb() -> None:
    from util.luma import replace_luminance
    y = np.zeros((1, 1), np.float64)
    for bad in (np.zeros((1, 1, 3), np.uint8), np.zeros((1, 1, 3), bool),
                np.zeros((1, 1, 3), np.complex128)):
        with pytest.raises(ValueError):
            replace_luminance(bad, y)


@pytest.mark.parametrize("algorithm", ("grey_world", "norm_2", "pca"))
def test_zero_signal_has_no_fake_estimate_in_native_or_compat_awb(algorithm: str) -> None:
    awb = AutoWhiteBalance(np.zeros((4, 4), np.uint32), sensor(), awb_cfg(algorithm), wbc())
    with np.errstate(all="raise"):
        assert awb.compute_gains() == (1.2, 0.8)
        np.testing.assert_array_equal(awb.execute(), (1.2, 0.8))
    assert awb.last_status == "fallback_ill_conditioned"


def test_invalid_manual_awb_gains_are_not_cast_before_safe_fallback() -> None:
    awb = AutoWhiteBalance(np.zeros((1, 1), np.uint16), sensor(), awb_cfg(), wbc(None, "bad"))
    assert awb.compute_gains() == (1, 1)
    assert awb.compute_gains(fallback_gains=(1.5, 0.75)) == (1.5, 0.75)


@pytest.mark.parametrize("shape", ((0, 0, 1), (0, 0), (0, 0, 4)))
def test_native_ae_rejects_malformed_empty_shapes(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="HWC3"):
        AutoExposure(np.zeros(shape), sensor(), ae_cfg(), meter_domain="scene_linear_rgb").get_exposure_feedback()


def test_native_ae_invalid_controls_are_not_hidden_by_no_signal() -> None:
    config = cast(AutoExposureConfig, {**ae_cfg(), "target_luminance": float("nan")})
    for shape in ((0, 0, 3), (2, 2, 3)):
        with pytest.raises(ValueError, match="target"):
            AutoExposure(np.zeros(shape), sensor(), config, meter_domain="scene_linear_rgb").get_exposure_feedback()
