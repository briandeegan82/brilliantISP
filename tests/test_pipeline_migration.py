from __future__ import annotations

import copy

import pytest

from util.output_contract import OutputSpec
from util.pipeline_migration import RETIRED_COLOR_SECTIONS, migrate_pipeline_config


def legacy_config() -> dict:
    return {
        "sensor_info": {"bit_depth": 12, "hdr_bit_depth": 24, "output_bit_depth": 16},
        "platform": {"save_format": "npy", "rgb_output": True},
        "tone_mapping": {"is_enable": True, "operator": "reinhard", "backend": "fixed"},
        "gamma_correction": {"is_enable": True, "curve": "srgb", "gamma": 2.2, "is_save": False},
        "color_space_conversion": {"is_enable": True, "conv_standard": 1},
        "rgb_conversion": {"is_enable": True, "curve": "ignored"},
        "yuv_conversion_format": {"is_enable": False, "conv_type": "422"},
        "auto_exposure": {"is_enable": True, "center_illuminance": 50, "target_luminance": 127.5},
        "digital_gain": {"current_gain": 3, "ae_feedback": -1},
    }


def test_migration_is_a_deep_copy_with_recoverable_legacy_fields() -> None:
    source = legacy_config()
    original = copy.deepcopy(source)
    native, report = migrate_pipeline_config(source)
    assert source == original
    assert native["schema_version"] == 2
    assert OutputSpec.from_mapping(native["output"]).as_config() == native["output"]
    assert native["output"]["format"] == "png"
    assert native["output"]["bit_depth"] == 16
    assert native["output"]["transfer"] == "srgb"
    for name in RETIRED_COLOR_SECTIONS:
        assert name not in native
        assert report["removed_sections"][name] == source[name]
    assert report["removed_fields"] == {
        "sensor_info.output_bit_depth": 16,
        "platform.rgb_output": True, "platform.save_format": "npy",
        "digital_gain.ae_feedback": -1,
    }
    assert native["digital_gain"]["current_gain"] == 3
    assert native["auto_exposure"]["center_illuminance"] == 50 / 255
    assert native["auto_exposure"]["target_luminance"] == 0.5
    assert report["potential_output_changes"]
    report["removed_sections"]["gamma_correction"]["gamma"] = 8
    assert source == original


@pytest.mark.parametrize("bits,maximum", ((1, 1), (4, 15), (8, 255), (12, 255), (16, 255), (24, 255), (32, 255)))
def test_ae_absolute_units_use_declared_legacy_meter_scale(bits: int, maximum: int) -> None:
    source = legacy_config()
    source["sensor_info"]["bit_depth"] = bits
    native, report = migrate_pipeline_config(source)
    assert report["legacy_ae_meter_max"] == maximum
    assert native["auto_exposure"]["center_illuminance"] == 50 / maximum
    assert native["auto_exposure"]["target_luminance"] == 127.5 / maximum


@pytest.mark.parametrize("value", (0.0, 0.18, 0.5, 1.0))
def test_ae_fractional_targets_are_not_divided_again(value: float) -> None:
    source = legacy_config()
    source["auto_exposure"].update(center_illuminance=value, target_luminance=value)
    native, _ = migrate_pipeline_config(source)
    assert native["auto_exposure"]["center_illuminance"] == value
    assert native["auto_exposure"]["target_luminance"] == value


@pytest.mark.parametrize("awb_enabled,old_mode", ((True, False), (False, True), (True, True), (False, False)))
def test_legacy_wb_mode_retains_executed_awb_ownership_without_changing_manual_gains(awb_enabled: bool, old_mode: bool) -> None:
    source = legacy_config()
    source["auto_white_balance"] = {"is_enable": awb_enabled}
    source["white_balance"] = {"is_enable": True, "is_auto": old_mode, "r_gain": 1.25, "b_gain": 0.75}
    original = copy.deepcopy(source)
    native, report = migrate_pipeline_config(source)
    assert native["white_balance"]["is_auto"] is awb_enabled
    assert native["white_balance"]["r_gain"] == 1.25
    assert native["white_balance"]["b_gain"] == 0.75
    assert source == original
    if old_mode != awb_enabled:
        assert report["removed_fields"]["white_balance.is_auto"] is old_mode
        assert any("executed mode" in notice for notice in report["notices"])


def test_native_wb_mode_can_apply_manual_gains_while_awb_only_suggests() -> None:
    native, _ = migrate_pipeline_config(legacy_config())
    native["auto_white_balance"] = {"is_enable": True}
    native["white_balance"] = {"is_enable": True, "is_auto": False, "r_gain": 1.25, "b_gain": 0.75}
    migrated, _ = migrate_pipeline_config(native)
    assert migrated["white_balance"] == native["white_balance"]


def test_legacy_float_output_is_explicitly_changed_to_normalized_linear_tiff() -> None:
    source = legacy_config()
    source["sensor_info"]["output_bit_depth"] = 32
    native, report = migrate_pipeline_config(source)
    spec = OutputSpec.from_mapping(native["output"])
    assert spec.floating_linear
    assert spec.format == "tiff"
    assert spec.transfer == "linear"
    assert any("integer-valued float32" in notice for notice in report["notices"])


def test_legacy_yuv_is_an_output_not_an_intermediate_rgb_round_trip() -> None:
    source = legacy_config()
    source["rgb_conversion"]["is_enable"] = False
    source["yuv_conversion_format"]["is_enable"] = True
    native, _ = migrate_pipeline_config(source)
    output = native["output"]
    assert (output["model"], output["format"], output["layout"]) == ("ycbcr", "yuv", "422")
    assert output["range"] == "limited"
    assert output["packing"] == "yuyv"
    assert output["matrix"] == "bt709"


def test_legacy_yuv_with_disabled_gamma_gets_declared_standard_transfer_and_notice() -> None:
    source = legacy_config()
    source["rgb_conversion"]["is_enable"] = False
    source["gamma_correction"]["is_enable"] = False
    native, report = migrate_pipeline_config(source)
    assert native["output"]["transfer"] == "rec709"
    assert any("linear/mixed-domain YUV" in notice for notice in report["notices"])


def test_output_overrides_are_authoritative_and_not_second_transfers() -> None:
    source = legacy_config()
    source["output"] = {"transfer": "power", "gamma": 2.4, "format": "tiff", "matrix": "bt601"}
    native, _ = migrate_pipeline_config(source)
    assert native["output"]["gamma"] == 2.4
    assert native["output"]["transfer"] == "power"
    assert native["output"]["format"] == "tiff"
    assert native["output"]["matrix"] == "bt601"
    assert "gamma_correction" not in native


def test_old_rgb_conversion_curve_never_becomes_an_oetf() -> None:
    source = legacy_config()
    source["gamma_correction"]["is_enable"] = False
    source["rgb_conversion"]["curve"] = "srgb"
    native, _ = migrate_pipeline_config(source)
    assert native["output"]["transfer"] == "linear"


def test_native_round_trip_does_not_reinterpret_hdr_targets_as_legacy_units() -> None:
    native, _ = migrate_pipeline_config(legacy_config())
    native["auto_exposure"]["target_luminance"] = 2.5
    again, report = migrate_pipeline_config(native)
    assert again == native
    assert again["auto_exposure"]["target_luminance"] == 2.5
    assert report["source_schema_version"] == 2


@pytest.mark.parametrize("section", RETIRED_COLOR_SECTIONS)
def test_active_legacy_color_sections_cannot_silently_compete_with_native_output(section: str) -> None:
    native, _ = migrate_pipeline_config(legacy_config())
    native[section] = {"is_enable": True}
    with pytest.raises(ValueError, match=section):
        migrate_pipeline_config(native)


def test_runtime_copy_can_keep_disabled_invalid_tone_controls_inactive() -> None:
    source = legacy_config()
    source["tone_mapping"] = {"is_enable": False, "operator": "invalid_inactive", "parameters": "inactive"}
    source["auto_exposure"] = {"is_enable": False, "center_illuminance": "inactive"}
    native, report = migrate_pipeline_config(source, canonicalize_tone_mapping=False)
    assert native["tone_mapping"] == source["tone_mapping"]
    assert native["auto_exposure"]["center_illuminance"] == "inactive"
    assert any("inactive auto_exposure" in notice for notice in report["notices"])
    with pytest.raises(ValueError):
        migrate_pipeline_config(source)


@pytest.mark.parametrize("version", (None, False, 0, 3, "2"))
def test_version_declaration_is_not_guessed(version) -> None:
    source = legacy_config()
    source["schema_version"] = version
    with pytest.raises(ValueError, match="schema_version"):
        migrate_pipeline_config(source)


def test_unresolved_inheritance_and_float_yuv_are_not_silently_supported() -> None:
    source = legacy_config()
    source["extends"] = "parent.yml"
    with pytest.raises(ValueError, match="inheritance"):
        migrate_pipeline_config(source)
    del source["extends"]
    source["sensor_info"]["output_bit_depth"] = 32
    source["rgb_conversion"]["is_enable"] = False
    with pytest.raises(ValueError, match="YCbCr"):
        migrate_pipeline_config(source)


@pytest.mark.parametrize("count,expected", ((18, 1.125), (64, 0.32), (0, 0)))
def test_legacy_clahe_clip_counts_become_mean_count_ratios(count: int, expected: float) -> None:
    source = legacy_config()
    source["ldci"] = {"is_enable": True, "wind": 64, "clip_limit": count, "post_gamma": False}
    native, report = migrate_pipeline_config(source)
    assert native["ldci"]["clip_limit"] == expected
    assert native["ldci"]["placement"] == "pre_transfer"
    assert "post_gamma" not in native["ldci"]
    assert report["source_luma"]["ldci"] == source["ldci"]
    source["ldci"]["post_gamma"] = True
    native, _ = migrate_pipeline_config(source)
    assert native["ldci"]["placement"] == "post_transfer"


@pytest.mark.parametrize("weight,expected", ((10, 10 / 255), (0, 1 / 255), (-5, 1 / 255)))
def test_nlm_strength_has_declared_normalized_units_and_odd_geometry(weight: int, expected: float) -> None:
    source = legacy_config()
    source["2d_noise_reduction"] = {"is_enable": True, "wts": weight, "window_size": 4, "patch_size": 4}
    native, report = migrate_pipeline_config(source)
    nr = native["2d_noise_reduction"]
    assert nr["strength"] == expected
    assert "wts" not in nr
    assert (nr["window_size"], nr["patch_size"]) == (5, 5)
    assert report["source_luma"]["2d_noise_reduction"] == source["2d_noise_reduction"]
    again, _ = migrate_pipeline_config(native)
    assert again == native


@pytest.mark.parametrize("section", RETIRED_COLOR_SECTIONS)
def test_retired_color_diagnostic_requests_are_migrated_not_silently_lost(section: str) -> None:
    source = legacy_config()
    source[section]["is_save"] = True
    native, report = migrate_pipeline_config(source)
    assert native["output"]["is_save"]
    assert any("diagnostics consolidate" in notice for notice in report["notices"])
    source["output"] = {"is_save": False}
    native, _ = migrate_pipeline_config(source)
    assert not native["output"]["is_save"]
