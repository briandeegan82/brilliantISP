from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from modules.auto_exposure.auto_exposure import AutoExposure
from modules.auto_white_balance.auto_white_balance import AutoWhiteBalance
from util.isp_types import AutoExposureConfig, GenericConfig, SensorInfo, WhiteBalanceConfig
from util.metering_roi import roi_config, weight_map


def sensor() -> SensorInfo:
    return cast(SensorInfo, {
        "height": 8, "width": 8, "bit_depth": 12, "hdr_bit_depth": 32,
        "bayer_pattern": "rggb", "output_bit_depth": 8, "pipeline_rgb_bit_depth": 16,
    })


def ae_cfg(roi: Any = None) -> AutoExposureConfig:
    cfg: dict[str, Any] = {"is_enable": True, "is_debug": False, "center_illuminance": 0.18,
                           "histogram_skewness": 0.9, "target_luminance": 0.18}
    if roi is not None:
        cfg["roi"] = roi
    return cast(AutoExposureConfig, cfg)


def awb_cfg(roi: Any = None) -> GenericConfig:
    cfg: dict[str, Any] = {"is_enable": True, "is_debug": False, "algorithm": "grey_world",
                           "percentage": 20, "underexposed_percentage": 0.0, "overexposed_percentage": 0.0}
    if roi is not None:
        cfg["roi"] = roi
    return cast(GenericConfig, cfg)


def wbc() -> WhiteBalanceConfig:
    return cast(WhiteBalanceConfig, {"r_gain": 1.0, "b_gain": 1.0, "is_enable": True, "is_save": False})


LOWER_HALF = {"is_enable": True, "include_rect": [0.0, 0.5, 1.0, 1.0]}


def test_inactive_roi_returns_none() -> None:
    assert roi_config(None) is None
    assert roi_config({"is_enable": True}) is None
    assert roi_config({"roi": {"is_enable": False, "ellipse": "ignored"}}) is None
    assert weight_map((4, 4), None) is None


def test_geometry_ellipse_rects_polygon_and_zones() -> None:
    full = weight_map((4, 4), {"is_enable": True})
    assert full is not None and np.all(full == 1.0) and not full.flags.writeable
    ell = weight_map((10, 10), {"is_enable": True, "ellipse": {"cx": 0.5, "cy": 0.5, "rx": 0.3, "ry": 0.3}})
    assert ell is not None and ell[5, 5] == 1 and ell[0, 0] == 0 and ell[5, 0] == 0
    rect = weight_map((4, 4), {"is_enable": True, "exclude_rects": [[0.0, 0.0, 0.5, 0.5]]})
    assert rect is not None and rect[:2, :2].sum() == 0 and rect[2:, :].sum() == 8 and rect[:2, 2:].sum() == 4
    # Triangle over the lower-left corner (ego-body style boundary).
    tri = weight_map((4, 4), {"is_enable": True, "exclude_polygons": [[[0, 1], [0, 0.5], [0.5, 1]]]})
    assert tri is not None and tri[3, 0] == 0 and tri[0, 0] == 1 and tri[3, 3] == 1
    zones = weight_map((4, 4), {"is_enable": True, "zone_weights": [[1, 1], [3, 3]]})
    assert zones is not None and np.all(zones[:2] == 1) and np.all(zones[2:] == 3)


@pytest.mark.parametrize("roi, msg", [
    ({"is_enable": True, "bogus": 1}, "unknown roi keys"),
    ({"is_enable": True, "include_rect": [0.5, 0, 0.4, 1]}, "include_rect"),
    ({"is_enable": True, "ellipse": {"cx": 0.5, "cy": 0.5, "rx": 0, "ry": 1}}, "positive"),
    ({"is_enable": True, "exclude_rects": [[0, 0, 1, 1]]}, "selects no pixels"),
    ({"is_enable": True, "zone_weights": [[0, 0]]}, "at least one positive"),
    ({"is_enable": True, "zone_weights": [[1], [1, 2]]}, "rectangular"),
    ({"is_enable": True, "exclude_polygons": [[[0, 0], [1, 1]]]}, "at least 3"),
    ({"is_enable": True, "include_rect": [0, 0, True, 1]}, "number"),
])
def test_invalid_roi_raises(roi, msg) -> None:
    with pytest.raises(ValueError, match=msg):
        weight_map((4, 4), roi)


def test_roi_is_enable_must_be_bool() -> None:
    with pytest.raises(ValueError, match="bool"):
        roi_config({"roi": {"is_enable": "yes"}})


def _sky_road_rgb() -> np.ndarray:
    img = np.full((8, 8, 3), 0.05, np.float64)
    img[:4] = 0.9  # bright sky in the top half
    return img


def test_native_ae_meters_only_roi() -> None:
    img = _sky_road_rgb()
    full = AutoExposure(img, sensor(), ae_cfg(), meter_domain="scene_linear_rgb")
    full.get_exposure_feedback()
    roi = AutoExposure(img, sensor(), ae_cfg(LOWER_HALF), meter_domain="scene_linear_rgb")
    roi.get_exposure_feedback()
    assert full.last_meter_average == pytest.approx(0.475)
    assert roi.last_meter_average == pytest.approx(0.05)
    assert full.sample_count == 64 and roi.sample_count == 32
    # More gain is requested when only the dark road is metered.
    gains = [1.0, 2.0, 4.0, 8.0]
    assert roi.suggest_direct_gain_index(0, gains) > full.suggest_direct_gain_index(0, gains)


def test_native_ae_zone_weights_are_weighted_mean() -> None:
    img = _sky_road_rgb()
    ae = AutoExposure(img, sensor(), ae_cfg({"is_enable": True, "zone_weights": [[1], [3]]}),
                      meter_domain="scene_linear_rgb")
    ae.get_exposure_feedback()
    assert ae.last_meter_average == pytest.approx((0.9 * 1 + 0.05 * 3) / 4)


def test_legacy_ae_meters_only_roi() -> None:
    img = (_sky_road_rgb() * 65535).astype(np.uint16)
    full = AutoExposure(img.copy(), sensor(), ae_cfg())
    full.execute()
    roi = AutoExposure(img.copy(), sensor(), ae_cfg(LOWER_HALF))
    roi.execute()
    assert full.last_meter_average is not None and roi.last_meter_average is not None
    assert roi.last_meter_average == pytest.approx(int(0.05 * 65535) >> 8, abs=1)
    assert roi.last_meter_average < full.last_meter_average


def test_skewness_unchanged_without_weights() -> None:
    rng = np.random.default_rng(0)
    y = rng.random((16, 16))
    ae = AutoExposure(np.zeros((1, 1, 3)), sensor(), ae_cfg())
    base = ae.get_luminance_histogram_skewness(y, 0.3)
    ones = ae.get_luminance_histogram_skewness(y, 0.3, weights=np.ones_like(y))
    assert ones == pytest.approx(base, rel=1e-12)


def test_awb_samples_only_roi() -> None:
    # RGGB 8x8 raw: top half lit by a reddish source, bottom half neutral grey.
    raw = np.full((8, 8), 1000.0)
    raw[0:4:2, 0::2] = 4000.0  # R sites in the top half
    full = AutoWhiteBalance(raw, sensor(), awb_cfg(), wbc()).compute_gains()
    roi_awb = AutoWhiteBalance(raw, sensor(), awb_cfg(LOWER_HALF), wbc())
    roi = roi_awb.compute_gains()
    assert roi == pytest.approx((1.0, 1.0))
    assert full[0] < 1.0
    assert roi_awb.sample_count == 8
