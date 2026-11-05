from __future__ import annotations

import copy
import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml
from typing import Any, cast

from brilliant_isp import BrilliantISP
from tools.gui.ui_schema import ENUM_CHOICES
from tools.gui.validation import can_process
from tools.gui.yaml_compat import REQUIRED_CONFIG_KEYS as GUI_REQUIRED
from tools.gui.yaml_compat import load_preset_yaml
from util.pipeline_types import ProcessingOptions
from util.config_schema import (
    ABSENT,
    BLOCK_SPECS,
    PARAMETER_SPECS,
    REQUIRED_CONFIG_KEYS,
    BlockSpec,
    ParameterSpec,
    ValidationIssue,
    normalize_config,
    spec_map,
    validate_config,
)

REPO = Path(__file__).resolve().parents[1]
SVS = REPO / "config" / "SVS_cam.yml"
BASE_HDR = REPO / "config" / "unused" / "base_hdr.yml"


def _cfg() -> dict:
    return load_preset_yaml(SVS)


def test_required_keys_shared_single_source() -> None:
    assert REQUIRED_CONFIG_KEYS is GUI_REQUIRED
    assert REQUIRED_CONFIG_KEYS is BrilliantISP._REQUIRED_CONFIG_KEYS
    assert len(REQUIRED_CONFIG_KEYS) == 24


def test_parameter_spec_frozen_and_defaults_literal() -> None:
    specs = spec_map()
    base = yaml.safe_load(BASE_HDR.read_text())
    assert specs["sensor_info.width"].default == base["sensor_info"]["width"] == 1920
    assert specs["sensor_info.height"].default == base["sensor_info"]["height"] == 1563
    assert specs["sensor_info.bit_depth"].default == 12
    assert specs["white_balance.r_gain"].default == base["white_balance"]["r_gain"] == 1.0
    assert list(specs["digital_gain.gain_array"].default) == base["digital_gain"]["gain_array"]
    assert isinstance(specs["digital_gain.gain_array"].default, tuple)
    assert specs["digital_gain.current_gain"].default == 10
    assert specs["gamma_correction.gamma"].default == 2.2
    assert specs["aces.variant"].default == "fitted"
    spec = specs["sensor_info.width"]
    with pytest.raises(FrozenInstanceError):
        setattr(spec, "default", 0)


def test_collection_defaults_immutable_and_absent_identity() -> None:
    spec = spec_map()["digital_gain.gain_array"]
    with pytest.raises(AttributeError):
        spec.default.append(99)
    copied = copy.deepcopy(spec)
    assert list(copied.default) == list(spec.default)
    absent_spec = spec_map()["auto_exposure.exposure_correction_mode"]
    assert copy.deepcopy(absent_spec).default is ABSENT
    assert copy.copy(absent_spec).default is ABSENT


def test_absent_defaults_marked() -> None:
    specs = spec_map()
    assert specs["auto_exposure.exposure_correction_mode"].default is ABSENT
    assert specs["auto_exposure.exposure_correction_mode"].has_default is False
    assert specs["digital_gain.exposure_correction_mode"].has_default is False
    assert specs["sensor_info.width"].has_default is True


def test_enum_choices_present() -> None:
    specs = spec_map()
    assert specs["sensor_info.bayer_pattern"].choices == ("rggb", "grbg", "gbrg", "bggr")
    assert specs["tone_mapping.tone_mapper"].choices == (
        "reinhard_integer", "aces", "aces_integer", "hable",
        "hable_integer", "durand", "integer_tmo", "hdr_durand",
    )
    assert specs["aces.variant"].choices == ("fitted",)


def test_enum_choices_parity_with_legacy_ui_schema() -> None:
    assert ENUM_CHOICES == {
        "sensor_info.bayer_pattern": ("rggb", "grbg", "gbrg", "bggr"),
        "sensor_info.endian_type": ("ieee-le", "ieee-be"),
        "sensor_info.data_format": ("uint8", "uint16", "uint32", "packed24", "word_pair32"),
        "sensor_info.storage_format": ("uint8", "uint16", "uint32", "packed24", "word_pair32"),
        "sensor_info.source_format": ("auto", "raw", "pgm", "tiff", "rawpy"),
        "sensor_info.word_order": ("low_first", "high_first"),
        "sensor_info.data_alignment": ("LSB", "MSB", "MANUAL"),
        "demosaic.algorithm": (
            "bilinear", "malvar", "vng", "vng_opt", "hamilton_adams", "hamilton_adams_opt",
            "ppg", "ppg_opt", "ahd", "ahd_opt", "lmmse", "lmmse_opt", "lmmse_fast",
        ),
        "tone_mapping.tone_mapper": (
            "reinhard_integer", "aces", "aces_integer", "hable",
            "hable_integer", "durand", "integer_tmo", "hdr_durand",
        ),
        "tone_mapping.operator": ("aces_fitted", "hable", "reinhard", "durand"),
        "tone_mapping.backend": ("float", "lut", "fixed", "cpu"),
        "tone_mapping.placement": ("pre_demosaic", "post_ccm"),
        "tone_mapping.normalization": ("declared_range", "per_frame"),
        "gamma_correction.curve": ("srgb", "gamma"),
        "auto_white_balance.algorithm": ("grey_world", "norm_2", "pca"),
        "yuv_conversion_format.conv_type": ("444", "422"),
        "ldci.placement": ("pre_transfer", "post_transfer"),
        "output.model": ("rgb", "ycbcr"),
        "output.color_space": ("srgb",),
        "output.transfer": ("srgb", "rec709", "power", "linear"),
        "output.format": ("png", "jpeg", "tiff", "yuv"),
        "output.matrix": ("bt601", "bt709"),
        "output.range": ("full", "limited"),
        "output.layout": ("444", "422"),
        "output.packing": ("yuyv", "uyvy"),
        "digital_gain.exposure_correction_mode": ("direct", "step"),
        "auto_exposure.exposure_correction_mode": ("direct", "step"),
        "aces.variant": ("fitted",),
        "scale.algorithm": ("Nearest_Neighbor", "Bilinear"),
        "scale.upscale_method": ("Nearest_Neighbor", "Bilinear"),
        "scale.downscale_method": ("Nearest_Neighbor", "Bilinear"),
        "platform.debug_log_level": ("DEBUG", "INFO", "WARNING", "ERROR"),
        "platform.save_format": ("png", "jpg", "tiff"),
    }


def test_block_specs_cover_sections() -> None:
    ids = {b.block_id for b in BLOCK_SPECS}
    for section in ("sensor_info", "digital_gain", "tone_mapping", "crop"):
        assert section in ids
    for block in BLOCK_SPECS:
        assert isinstance(block, BlockSpec)
        for spec in block.parameters:
            assert any(spec.path.startswith(f"{s}.") for s in block.sections)


def test_normalize_accepts_versionless_and_v1() -> None:
    cfg = _cfg()
    normalized, notices = normalize_config(cfg)
    assert normalized == cfg
    assert normalized is not cfg
    assert any("versionless" in n for n in notices)
    v1 = dict(cfg)
    v1["schema_version"] = 1
    normalized1, notices1 = normalize_config(v1)
    assert normalized1["schema_version"] == 1
    assert not any("versionless" in n for n in notices1)


def test_normalize_does_not_mutate_or_insert_defaults() -> None:
    cfg = _cfg()
    snapshot = copy.deepcopy(cfg)
    normalized, _ = normalize_config(cfg)
    assert cfg == snapshot
    normalized["sensor_info"]["width"] = 1
    assert cfg["sensor_info"]["width"] == snapshot["sensor_info"]["width"]
    minimal = {"sensor_info": {"width": 8}}
    out, _ = normalize_config(minimal)
    assert out == minimal


def test_normalize_rejects_nonmapping_and_v2() -> None:
    with pytest.raises(TypeError):
        normalize_config(cast(Any, [1, 2]))
    # schema_version 2 is the accepted native version; only >2/invalid reject.
    out, _ = normalize_config({"schema_version": 2})
    assert out["schema_version"] == 2
    with pytest.raises(ValueError, match="schema_version"):
        normalize_config({"schema_version": 3})
    with pytest.raises(ValueError, match="schema_version"):
        normalize_config({"schema_version": "x"})
    with pytest.raises(ValueError, match="schema_version"):
        normalize_config({"schema_version": True})


def test_normalize_explicit_null_schema_version_rejected() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        normalize_config({"schema_version": None})


def test_normalize_unknown_keys_retained_with_warning() -> None:
    cfg = {"sensor_info": {"width": 8}, "my_custom_block": {"x": 1}}
    normalized, notices = normalize_config(cfg)
    assert normalized["my_custom_block"] == {"x": 1}
    assert any("my_custom_block" in n for n in notices)


def test_normalize_unused_tmo_knob_warnings() -> None:
    cfg = _cfg()
    cfg["aces"] = {"variant": "canonical", "output_transform": "sRGB",
                   "surround": "dark", "white_point": 11.2}
    cfg["reinhard_integer"] = {"dark_enh": 0.5}
    _, notices = normalize_config(cfg)
    assert any("aces.variant" in n for n in notices)
    assert any("aces.output_transform" in n for n in notices)
    assert any("aces.surround" in n for n in notices)
    assert any("aces.white_point" in n for n in notices)
    assert any("canonical" in n and "not implemented" in n for n in notices)
    assert any("reinhard_integer.dark_enh" in n for n in notices)


BUNDLED_CONFIGS = sorted((REPO / "config").glob("*.yml"))


@pytest.mark.parametrize(
    "preset", BUNDLED_CONFIGS, ids=[p.name for p in BUNDLED_CONFIGS]
)
def test_validate_valid_bundled_configs(preset: Path) -> None:
    cfg = load_preset_yaml(preset)
    errors = [i for i in validate_config(cfg) if i.severity == "error"]
    assert errors == [], f"{preset.name}: {errors}"


@pytest.mark.parametrize(("pwc_update", "blc_linear", "expected_path"), [
    ({"pre_linearization_black_level": 240}, False, "companding.pre_linearization_black_level"),
    ({"pedestal": 16}, False, "companding.pre_linearization_black_level"),
    ({"post_linearization_black_level": 4}, False, "companding.post_linearization_black_level"),
    ({"pre_linearization_black_level": 0}, True, "black_level_correction.is_linear"),
])
def test_blc_pwc_reject_overlapping_black_levels(
    pwc_update: dict, blc_linear: bool, expected_path: str,
) -> None:
    cfg = _cfg()
    cfg["black_level_correction"].update(is_enable=True, is_linear=blc_linear)
    cfg["companding"].update(pre_linearization_black_level=0, post_linearization_black_level=0)
    cfg["companding"].update(pwc_update)
    if "pedestal" in pwc_update:
        cfg["companding"].pop("pre_linearization_black_level")
    assert expected_path in {issue.path for issue in validate_config(cfg) if issue.severity == "error"}
    ok, _ = can_process(cfg, has_raw=True)
    assert not ok


def test_blc_pwc_allow_single_black_owner_and_disabled_stages() -> None:
    cfg = _cfg()
    cfg["black_level_correction"].update(is_enable=True, is_linear=False)
    cfg["companding"].update(pre_linearization_black_level=0, post_linearization_black_level=0)
    assert not any("black_level" in issue.path for issue in validate_config(cfg))
    cfg["companding"]["pedestal"] = 240  # The explicit pre level takes precedence.
    assert not any("black_level" in issue.path for issue in validate_config(cfg))
    cfg["companding"]["pre_linearization_black_level"] = 240
    cfg["black_level_correction"]["is_enable"] = False
    assert not any("black_level" in issue.path for issue in validate_config(cfg))
    cfg["black_level_correction"]["is_enable"] = True
    cfg["companding"]["is_enable"] = False
    assert not any("black_level" in issue.path for issue in validate_config(cfg))


def test_facade_rejects_blc_pwc_conflict_at_load() -> None:
    cfg = _cfg()
    cfg["black_level_correction"].update(is_enable=True, is_linear=False)
    with pytest.raises(ValueError, match="companding.pre_linearization_black_level"):
        BrilliantISP("", cfg, outFileName="", options=ProcessingOptions())


def test_validate_rejects_bool_numeric() -> None:
    cfg = _cfg()
    cfg["sensor_info"]["width"] = True
    cfg["sensor_info"]["bit_depth"] = True
    cfg["white_balance"]["r_gain"] = False
    paths = {i.path for i in validate_config(cfg)}
    assert "sensor_info.width" in paths
    assert "sensor_info.bit_depth" in paths
    assert "white_balance.r_gain" in paths


def test_validate_rejects_fractional_geometry_and_index() -> None:
    cfg = _cfg()
    cfg["sensor_info"]["width"] = 1920.5
    cfg["digital_gain"]["current_gain"] = 10.5
    paths = {i.path for i in validate_config(cfg)}
    assert "sensor_info.width" in paths
    assert "digital_gain.current_gain" in paths
    cfg2 = _cfg()
    cfg2["crop"]["is_enable"] = True
    cfg2["crop"]["crop_x_start"] = 1.5
    paths2 = {i.path for i in validate_config(cfg2)}
    assert "crop.crop_x_start" in paths2


def test_validate_rejects_nonfinite() -> None:
    cfg = _cfg()
    cfg["white_balance"]["b_gain"] = math.nan
    cfg["sensor_info"]["height"] = math.inf
    paths = {i.path for i in validate_config(cfg)}
    assert "white_balance.b_gain" in paths
    assert "sensor_info.height" in paths


def test_validate_malformed_sections_no_crash() -> None:
    issues = validate_config({"sensor_info": [1, 2], "crop": "x",
                              "white_balance": None, "digital_gain": {"gain_array": "nope"}})
    paths = {i.path for i in issues}
    assert "sensor_info" in paths
    assert "crop" in paths
    assert "digital_gain.gain_array" in paths
    issues2 = validate_config(cast(Any, "not a mapping"))
    assert issues2 and issues2[0].path == "config"


def test_validate_rejects_numeric_strings() -> None:
    cfg = _cfg()
    cfg["sensor_info"]["width"] = "1920"
    cfg["sensor_info"]["bit_depth"] = "12"
    cfg["white_balance"]["r_gain"] = "1.5"
    paths = {i.path for i in validate_config(cfg)}
    assert "sensor_info.width" in paths
    assert "sensor_info.bit_depth" in paths
    assert "white_balance.r_gain" in paths


def test_validate_required_section_structure() -> None:
    issues = validate_config({"platform": None, "companding": "x",
                              "crop": []})
    paths = {i.path for i in issues}
    assert "platform" in paths
    assert "companding" in paths
    assert "crop" in paths
    platform_issues = [i for i in issues if i.path == "platform"]
    assert len(platform_issues) == 1


def test_validate_huge_int_no_overflow_crash() -> None:
    cfg = _cfg()
    cfg["sensor_info"]["width"] = 10 ** 400
    paths = {i.path for i in validate_config(cfg)}
    assert "sensor_info.width" in paths


def test_validate_crop_fractional_size_and_origin_rules() -> None:
    cfg = _cfg()
    cfg["crop"]["is_enable"] = True
    cfg["crop"]["crop_x_start"] = 0
    cfg["crop"]["crop_y_start"] = 0
    cfg["crop"]["new_width"] = 100.5
    cfg["crop"]["new_height"] = 100
    assert any(
        i.path == "crop.new_width" for i in validate_config(cfg)
    )
    cfg["crop"]["new_width"] = 100
    cfg["crop"]["crop_x_start"] = None
    paths = {i.path for i in validate_config(cfg)}
    assert "crop.crop_x_start" in paths
    cfg["crop"]["crop_x_start"] = "abc"
    assert "crop.crop_x_start" in {i.path for i in validate_config(cfg)}
    cfg2 = _cfg()
    cfg2["crop"]["is_enable"] = True
    cfg2["crop"].pop("crop_x_start")
    cfg2["crop"].pop("crop_y_start")
    cfg2["crop"]["new_width"] = 100
    cfg2["crop"]["new_height"] = 100
    assert not any(
        i.path.startswith("crop.crop_") for i in validate_config(cfg2)
    )


def test_validate_gain_array_strictly_increasing() -> None:
    cfg = _cfg()
    cfg["digital_gain"]["gain_array"] = [1.0, 1.0, 2.0]
    assert any(
        i.path == "digital_gain.gain_array" for i in validate_config(cfg)
    )
    cfg["digital_gain"]["gain_array"] = [1.0, -2.0, 3.0]
    assert any(
        i.path == "digital_gain.gain_array" for i in validate_config(cfg)
    )
    cfg["digital_gain"]["gain_array"] = [1.0, 2.0, 3.0]
    assert not any(
        i.path == "digital_gain.gain_array" for i in validate_config(cfg)
    )


def test_validate_odd_crop_origin_warns_not_blocks() -> None:
    cfg = _cfg()
    cfg["crop"]["is_enable"] = True
    cfg["crop"]["crop_x_start"] = 41
    cfg["crop"]["crop_y_start"] = 0
    cfg["crop"]["new_width"] = 100
    cfg["crop"]["new_height"] = 100
    issues = validate_config(cfg)
    odd = [i for i in issues if i.path == "crop"]
    assert odd and all(i.severity == "warning" for i in odd)
    ok, blocking = can_process(cfg, has_raw=True)
    assert ok is True
    assert odd[0] in blocking or all(i.severity == "warning" for i in blocking)


def test_validation_issue_default_severity() -> None:
    assert ValidationIssue("p", "m").severity == "error"
