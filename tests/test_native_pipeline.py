from __future__ import annotations

import copy
import numpy as np
import pytest
import yaml

from util.pipeline_runner import process_frame
from pathlib import Path
from util.pipeline_types import ProcessingOptions


REPO = Path(__file__).resolve().parents[1]


def synth_config() -> dict:
    merged = yaml.safe_load((REPO / "config" / "SVS_cam.yml").read_text())
    merged["sensor_info"].update(height=8, width=10, bit_depth=12, hdr_bit_depth=24,
                                 bayer_pattern="rggb")
    merged["sensor_info"]["repair_zero_green"] = False
    for key in ("digital_gain", "lens_shading_correction", "bayer_noise_reduction",
                "black_level_correction", "white_balance", "auto_white_balance",
                "color_correction_matrix", "color_saturation_enhancement", "ldci",
                "sharpen", "2d_noise_reduction", "scale", "companding",
                "dead_pixel_correction", "oecf", "crop", "auto_exposure",
                "gamma_correction", "color_space_conversion",
                "rgb_conversion", "color_saturation_enhancement",
                "tone_mapping", "yuv_conversion_format"):
        if isinstance(merged.get(key), dict):
            merged[key]["is_enable"] = False
    merged["platform"].update(render_3a=False, plot_histograms=False,
                              debug_enabled=False, disable_progress_bar=True)
    merged["demosaic"]["is_enable"] = True
    merged["demosaic"]["algorithm"] = "malvar"
    merged["output"] = {"model": "rgb", "transfer": "linear",
                        "format": "tiff", "bit_depth": 32}
    merged["schema_version"] = 2
    return merged


def test_native_linear_tiff_1024_codes() -> None:
    src = np.full((8, 10), 1024, np.uint16)
    out = process_frame(src, synth_config(),
                        options=ProcessingOptions(save_outputs=False))
    assert out.native_image.dtype == np.float32
    np.testing.assert_allclose(out.native_image, 1024.0 / 4095.0, rtol=0, atol=2e-6)
    assert out.native_metadata.storage_bit_depth == 32
    assert out.native_metadata.code_bit_depth is None
    assert out.native_metadata.reference_white == pytest.approx(1.0)
    assert not out.artifacts


def test_native_srgb_png8_single_oetf() -> None:
    config = synth_config()
    config["output"] = {"model": "rgb", "transfer": "srgb", "format": "png", "bit_depth": 8}
    src = np.full((8, 10), 1024, np.uint16)
    out = process_frame(src, config, options=ProcessingOptions(save_outputs=False))
    np.testing.assert_array_equal(out.native_image, 137)
    np.testing.assert_array_equal(out.preview_rgb8, 137)


def test_native_ccm_active_preserves_negative_headroom() -> None:
    config = synth_config()
    config["color_correction_matrix"]["is_enable"] = True
    config["color_correction_matrix"]["corrected_red"] = [-1.0, 0.0, 0.0]
    config["color_correction_matrix"]["corrected_green"] = [0.0, 1.0, 0.0]
    config["color_correction_matrix"]["corrected_blue"] = [0.0, 0.0, 2.0]
    x = 1024.0 / 4095.0
    src = np.full((8, 10), 1024, np.uint16)
    out = process_frame(src, config, options=ProcessingOptions(save_outputs=False))
    np.testing.assert_allclose(out.native_image[0, 0], [-x, x, 2 * x], rtol=0, atol=2e-6)


def test_native_reinhard_ae_meter_linear_tap() -> None:
    config = synth_config()
    config["tone_mapping"]["is_enable"] = True
    config["tone_mapping"]["operator"] = "reinhard"
    config["tone_mapping"]["backend"] = "float"
    config["auto_exposure"]["is_enable"] = True
    config["digital_gain"]["is_auto"] = False
    x = 1024.0 / 4095.0
    src = np.full((8, 10), 1024, np.uint16)
    out = process_frame(src, config, options=ProcessingOptions(save_outputs=False))
    np.testing.assert_allclose(out.native_image, x / (x + 0.25), rtol=0, atol=2e-6)
    assert out.feedback["ae_meter_average"] == pytest.approx(x, abs=1e-6)


def test_native_awb_estimated_feedback_manual_gains_apply() -> None:
    config = synth_config()
    config["auto_white_balance"]["is_enable"] = True
    config["white_balance"]["is_enable"] = True
    config["white_balance"]["is_auto"] = False
    config["white_balance"]["r_gain"] = 2.0
    config["white_balance"]["b_gain"] = 0.5
    src = np.full((8, 10), 1024, np.uint16)
    out = process_frame(src, config, options=ProcessingOptions(save_outputs=False))
    assert out.feedback["awb_status"] == "estimated"
    x = 1024.0 / 4095.0
    # CFA-site-exact: RGGB (0,0)=R, (0,1)=G, (1,1)=B get manual gains 2/.5.
    np.testing.assert_allclose(out.native_image[0, 0, 0], 2 * x, rtol=0, atol=2e-6)
    np.testing.assert_allclose(out.native_image[0, 1, 1], x, rtol=0, atol=2e-6)
    np.testing.assert_allclose(out.native_image[1, 1, 2], 0.5 * x, rtol=0, atol=2e-6)
    assert config["white_balance"]["r_gain"] == 2.0



def test_ae_controller_restarts_and_manual_config_unchanged() -> None:
    from util.pipeline_types import ProcessingState
    config = synth_config()
    config["white_balance"].update(is_enable=True, is_auto=False, r_gain=2.0, b_gain=0.5)
    config["digital_gain"].update(is_enable=True, is_auto=True,
                                  gain_array=[1.0, 2.0, 4.0], current_gain=0)
    config["auto_exposure"].update(is_enable=True,
                                 exposure_correction_mode="direct",
                                 rerun_from_digital_gain=True,
                                 target_luminance=0.5)
    before = copy.deepcopy(config)
    src = np.full((8, 10), 1024, np.uint16)
    state = ProcessingState()
    out = process_frame(src, config, state=state,
                        options=ProcessingOptions(save_outputs=False))
    assert out.feedback["ae_stop_reason"] == "converged"
    assert out.feedback["digital_gain_index"] == 1
    assert out.feedback["next_digital_gain_index"] == 1
    assert out.feedback["ae_passes"] == 2
    # gain applied once, config untouched
    assert config == before
    x = 1024.0 / 4095.0
    expected_gain = 2.0
    np.testing.assert_allclose(
        out.native_image[0, 0, 0], 2 * x * expected_gain, rtol=0, atol=2e-5)
    # auto state seeded for next frame
    assert state.digital_gain_index == out.feedback["next_digital_gain_index"]
    again = process_frame(src, config, state=state)
    assert again.feedback["digital_gain_index"] == 1
    assert again.feedback["ae_passes"] == 1
    np.testing.assert_array_equal(again.native_image, out.native_image)
    # manual DG ignores state
    config["digital_gain"]["is_auto"] = False
    out2 = process_frame(src, config, state=state,
                         options=ProcessingOptions(save_outputs=False))
    assert out2.feedback["digital_gain_index"] == config["digital_gain"]["current_gain"]
    np.testing.assert_allclose(out2.native_image[0, 0, 0], 2 * x, rtol=0, atol=2e-6)


@pytest.mark.parametrize(("mode", "passes", "expected"), (("direct", 2, 0.5), ("step", 1, 0.25)))
def test_direct_target_is_independent_of_skew_center(mode: str, passes: int, expected: float) -> None:
    config = synth_config()
    config["digital_gain"].update(is_enable=True, is_auto=True, gain_array=[0.25, 0.5], current_gain=0)
    config["auto_exposure"].update(is_enable=True, center_illuminance=0.25, target_luminance=0.5,
                                  exposure_correction_mode=mode, rerun_from_digital_gain=True)
    result = process_frame(np.full((8, 10), 4095, np.uint16), config)
    assert result.feedback["ae_passes"] == passes
    np.testing.assert_array_equal(result.native_image, np.float32(expected))


@pytest.mark.parametrize("algorithm", ("Bilinear", "Nearest_Neighbor"))
@pytest.mark.parametrize(("height", "width"), ((1, 1), (3, 5), (11, 13), (8, 10)))
def test_native_scale_preserves_subcode_float_values(algorithm: str, height: int, width: int) -> None:
    config = synth_config()
    config["platform"]["skip_disabled_modules"] = True
    config["digital_gain"].update(is_enable=True, is_auto=False, gain_array=[0.25], current_gain=0)
    config["scale"].update(is_enable=True, is_hardware=False, new_height=height, new_width=width, algorithm=algorithm)
    original = copy.deepcopy(config)
    result = process_frame(np.full((8, 10), 4095, np.uint16), config,
                           options=ProcessingOptions(capture_stages=("scale", "output")))
    assert config == original
    assert result.native_image.shape == (height, width, 3)
    np.testing.assert_allclose(result.native_image, 0.25, rtol=0, atol=2e-7)
    assert result.captures["scale"].shape == result.native_image.shape
    assert result.captures["output"].dtype == np.float32


@pytest.mark.parametrize(("height", "width"), ((1, 1), (1, 5), (5, 1), (3, 5), (7, 9)))
def test_native_tiny_odd_black_frame(height: int, width: int) -> None:
    config = synth_config()
    config["sensor_info"].update(height=height, width=width, repair_zero_green="auto")
    result = process_frame(np.zeros((height, width), np.uint16), config)
    assert result.native_image.shape == (height, width, 3)
    np.testing.assert_array_equal(result.native_image, 0)


def test_native_black_does_not_commit_fake_awb_estimate() -> None:
    from util.pipeline_types import ProcessingState
    config = synth_config()
    config["auto_white_balance"]["is_enable"] = True
    config["auto_exposure"]["is_enable"] = True
    config["digital_gain"].update(is_enable=True, is_auto=True, gain_array=[1, 2], current_gain=0)
    state = ProcessingState()
    result = process_frame(np.zeros((8, 10), np.uint16), config, state=state)
    assert result.feedback["awb_status"] == "fallback_no_samples"
    assert result.feedback["ae_stop_reason"] == "no_signal"
    assert state.previous_awb_gains is None
    np.testing.assert_array_equal(result.native_image, 0)


@pytest.mark.parametrize("gain_index", (0, 1))
def test_awb_measurement_is_invariant_to_virtual_gain(gain_index: int) -> None:
    config = synth_config()
    config["digital_gain"].update(is_enable=True, is_auto=False, gain_array=[1.0, 8.0], current_gain=gain_index)
    config["auto_white_balance"].update(is_enable=True, algorithm="pca", percentage=20,
                                       underexposed_percentage=0.05, overexposed_percentage=0.05)
    config["white_balance"].update(is_enable=True, is_auto=True)
    source = np.full((8, 10), 1000, np.uint16)
    source[1::2, 1::2] = 2000
    result = process_frame(source, config)
    assert result.feedback["awb_status"] == "estimated"
    assert result.feedback["awb_sample_count"] == 20
    np.testing.assert_allclose(result.feedback["awb_gains"], (1.0, 0.5), rtol=0, atol=1e-7)
    np.testing.assert_allclose(result.native_image, 1000 / 4095 * (1, 8)[gain_index], rtol=0, atol=2e-6)


@pytest.mark.parametrize("raw,target,index,passes", ((100, 0.5, 1, 2), (2048, 0.001, 0, 1)))
def test_native_ae_reports_unreachable_gain_limit(raw: int, target: float, index: int, passes: int) -> None:
    config = synth_config()
    config["digital_gain"].update(is_enable=True, is_auto=True, gain_array=[1.0, 2.0], current_gain=0)
    config["auto_exposure"].update(is_enable=True, exposure_correction_mode="direct", target_luminance=target)
    result = process_frame(np.full((8, 10), raw, np.uint16), config)
    assert result.feedback["ae_stop_reason"] == "gain_limit"
    assert result.feedback["digital_gain_index"] == result.feedback["next_digital_gain_index"] == index
    assert result.feedback["ae_passes"] == passes
    np.testing.assert_allclose(result.native_image, raw / 4095 * (1, 2)[index], rtol=0, atol=2e-6)
