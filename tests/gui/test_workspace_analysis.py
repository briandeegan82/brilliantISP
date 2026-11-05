from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from tools.gui.analysis import analyze_rgb8, analyze_values, native_pixel, native_values, normalize_roi, sample_tone_curve
from tools.gui.comparison import ComparisonIdentity, comparison_identity, compatibility_reason, compose_rgb8, difference_statistics


def test_rgb_histogram_roi_and_units() -> None:
    rgb = np.array([[[0, 10, 255], [255, 20, 0]], [[128, 30, 64], [64, 40, 128]]], np.uint8)
    all_stats = analyze_rgb8(rgb)
    assert all_stats["pixel_count"] == 4
    assert [c["mean"] for c in all_stats["components"]] == [111.75, 25, 111.75]
    assert all_stats["any_display_zero_pixels"] == 2
    assert all_stats["any_display_max_pixels"] == 2
    assert all_stats["domain"] == "display sRGB8 codes"
    assert sum(all_stats["encoded_luma"]["histogram"]) == 4
    for c in all_stats["components"]:
        assert sum(c["histogram"]) == 4
    roi = analyze_rgb8(rgb, (1, 0, 2, 2))
    assert roi["roi"] == [1, 0, 2, 2] and roi["pixel_count"] == 2
    assert [c["mean"] for c in roi["components"]] == [159.5, 30, 64]
    assert roi["components"][0]["histogram"][255] == 1
    assert roi["components"][0]["histogram"][64] == 1
    assert roi["components"][1]["population_std"] == 10
    assert roi["components"][2]["population_std"] == 64
    assert roi["components"][0]["population_std"] == 95.5
    luma = [255 * .2126 + 20 * .7152, 64 * .2126 + 40 * .7152 + 128 * .0722]
    ordered = sorted(luma)
    expected_low = ordered[0] + .01 * (ordered[1] - ordered[0])
    expected_high = ordered[0] + .99 * (ordered[1] - ordered[0])
    assert roi["encoded_luma"]["percentile_1"] == pytest.approx(expected_low, abs=1e-12)
    assert roi["encoded_luma"]["percentile_99"] == pytest.approx(expected_high, abs=1e-12)


def test_headroom_statistics_are_not_discarded_or_named_sensor_dr() -> None:
    data = np.array([[-.5, 0], [1, 2]], np.float32)
    stats = analyze_values(data, nominal_ranges=((0, 1),))
    c = stats["components"][0]
    assert c["below_nominal"] == c["above_nominal"] == 1
    assert c["at_nominal_low"] == c["at_nominal_high"] == 1
    assert sum(c["histogram"]) == 2
    assert c["mean"] == .625
    assert c["minimum"] == -.5 and c["maximum"] == 2
    assert "not a measurement of sensor saturation" in stats["limit_note"]


@pytest.mark.parametrize("bad", ((3, 0, 2, 2), (0, 0, 0, 1), (8, 0, 9, 1), (0, 0, 1.5, 2), (False, 0, 2, 2)))
def test_roi_rejects_empty_inverted_unrelated_or_noninteger(bad) -> None:
    with pytest.raises(ValueError):
        normalize_roi(bad, 4, 3)


def test_roi_intersects_half_open_pixels() -> None:
    assert normalize_roi((-5, -8, 3, 2), 4, 3) == (0, 0, 3, 2)
    assert normalize_roi(None, 4, 3) == (0, 0, 4, 3)


@pytest.mark.parametrize("packing,packed", (("yuyv", [10, 128, 20, 240]), ("uyvy", [128, 10, 240, 20])))
def test_native_picker_uses_full_pixel_coordinates_in_both_packed_orders(packing, packed) -> None:
    meta = {"width": 2, "height": 1, "color_model": "ycbcr", "layout": "422",
            "packing": packing, "value_units": "codes", "signal_domain": "encoded_ycbcr"}
    data = np.array([packed], np.uint16)
    first, second = native_pixel(data, meta, 0, 0), native_pixel(data, meta, 1, 0)
    assert first["values"] == [10, 128, 240]
    assert second["values"] == [20, 128, 240]
    assert second["component_names"] == ["Y", "Cb", "Cr"]
    assert second["dtype"] == "uint16"
    assert native_values(data, meta).tolist() == [[[10, 128, 240], [20, 128, 240]]]
    with pytest.raises(ValueError):
        native_pixel(data, meta, 2, 0)


def test_picker_keeps_float_signed_headroom_and_native16_codes() -> None:
    meta = {"width": 1, "height": 1, "color_model": "rgb", "value_units": "normalized",
            "signal_domain": "scene_linear_rgb", "transfer": "linear"}
    sample = native_pixel(np.array([[[-.25, 1.5, 2]]], np.float32), meta, 0, 0)
    assert sample["values"] == [-.25, 1.5, 2]
    assert sample["domain"] == "scene_linear_rgb"
    sample = native_pixel(np.array([[[0, 32768, 65535]]], np.uint16), meta, 0, 0)
    assert sample["values"] == [0, 32768, 65535]


def test_comparison_does_not_wrap_uint8_and_roi_counts_are_pixels() -> None:
    a = np.array([[[255, 0, 0], [0, 0, 0]]], np.uint8)
    b = np.array([[[0, 0, 0], [0, 3, 4]]], np.uint8)
    identity = ComparisonIdentity("sha", "geometry", "display_srgb8", a.shape)
    metrics = difference_statistics(a, b, identity_a=identity, identity_b=identity)
    assert metrics["changed_pixels"] == 2
    assert metrics["changed_components"] == 3
    assert metrics["maximum_absolute"] == 255
    assert metrics["mean_absolute"] == 262 / 6
    assert metrics["rmse"] == math.sqrt((255**2 + 3**2 + 4**2) / 6)
    roi = difference_statistics(a, b, identity_a=identity, identity_b=identity, roi=(1, 0, 2, 1))
    assert roi["pixel_count"] == roi["changed_pixels"] == 1
    assert roi["changed_components"] == 2
    assert roi["rmse"] == math.sqrt(25 / 3)
    wipe = compose_rgb8(a, b, identity_a=identity, identity_b=identity, mode="wipe", fraction=.5)
    assert wipe.tolist() == [[[255, 0, 0], [0, 3, 4]]]
    diff = compose_rgb8(a, b, identity_a=identity, identity_b=identity, mode="difference")
    assert diff.tolist() == [[[255, 0, 0], [0, 3, 4]]]


def test_comparison_rejects_unregistered_sources_geometry_and_domains() -> None:
    identity = ComparisonIdentity("sha", "geometry", "display_srgb8", (2, 4, 3))
    data = np.zeros(identity.shape, np.uint8)
    for altered in (
        dataclasses.replace(identity, source_fingerprint="another"),
        dataclasses.replace(identity, geometry="different_crop_same_shape"),
        dataclasses.replace(identity, domain="raw_mosaic_visualization"),
    ):
        assert compatibility_reason(identity, altered) is not None
        with pytest.raises(ValueError):
            difference_statistics(data, data, identity_a=identity, identity_b=altered)


def test_geometry_identity_includes_decode_crop_and_scale_not_color_tuning() -> None:
    cfg = {"sensor_info": {"width": 4, "height": 2, "bayer_pattern": "rggb", "bit_depth": 12},
           "crop": {"is_enable": False}, "scale": {"is_enable": False}}
    base = comparison_identity("sha", cfg, (2, 4, 3))
    cfg["white_balance"] = {"r_gain": 2}
    assert comparison_identity("sha", cfg, (2, 4, 3)) == base
    cfg["crop"] = {"is_enable": True, "left": 1}
    assert comparison_identity("sha", cfg, (2, 4, 3)).geometry != base.geometry
    cfg["crop"] = {"is_enable": False}
    cfg["sensor_info"]["horizontal_flip"] = True
    assert comparison_identity("sha", cfg, (2, 4, 3)).geometry != base.geometry


def test_curve_uses_native_mapper_and_marks_ramp_local_normalization() -> None:
    cfg = {
        "schema_version": 2, "sensor_info": {"bit_depth": 12, "hdr_bit_depth": 24},
        "tone_mapping": {"is_enable": True, "operator": "reinhard", "backend": "float",
                         "normalization": "declared_range", "normalize_output": False,
                         "parameters": {"knee": .25, "strength": 1}},
        "output": {"format": "png", "model": "rgb", "bit_depth": 8, "transfer": "srgb"},
    }
    curve = sample_tone_curve(cfg, samples=3)
    assert curve["x"] == [0, .5, 1]
    expected = [0, 2 / 3, .8]
    np.testing.assert_allclose(curve["display_linear"], expected, rtol=1e-7, atol=0)
    assert curve["after_transfer"][0] == 0
    assert "Ramp-derived" in curve["caveat"] and "Durand" in curve["caveat"]
