"""prepare_config must mirror BrilliantISP.load_config resolution exactly,
on mappings and path(s), including migration and diagnostic suppression,
without mutating the caller's config."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from brilliant_isp import BrilliantISP
from util.pipeline_types import ProcessingOptions
from util.signal_range import linear_code_range
from util.output_contract import OutputSpec

from brilliantisp_gpu.config import prepare_config

REPO = Path(__file__).resolve().parents[2]
CONFIGS = [REPO / "config" / "AD_cam_2.yml", REPO / "config" / "SVS_cam.yml"]


_DEFAULT = object()


def _facade(config_path, options=_DEFAULT):
    return BrilliantISP(
        data_path=str(REPO / "in_frames"),
        config_path=config_path,
        outFileName="probe",
        options=ProcessingOptions() if options is _DEFAULT else options,
    )


def _compare(prepared, isp, *, options_diagnostics: bool):
    assert prepared.effective_config == isp.c_yaml
    assert prepared.source_config_original == isp.source_config_original
    assert prepared.migration_report == isp.migration_report
    assert sorted(prepared.warnings) == sorted(set(isp.config_notices))
    spec = OutputSpec.from_mapping(isp.c_yaml.get("output"))
    assert prepared.output_spec == spec
    cr = linear_code_range(isp.c_yaml)
    assert prepared.code_range == cr


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_prepare_matches_facade_path(path):
    options = ProcessingOptions()
    prepared = prepare_config(path, options=options)
    isp = _facade(path, options)
    _compare(prepared, isp, options_diagnostics=False)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_prepare_matches_facade_default_options(path):
    prepared = prepare_config(path)
    isp = _facade(path, None)
    _compare(prepared, isp, options_diagnostics=True)


def test_prepare_mapping_does_not_mutate_caller():
    raw = yaml.safe_load((REPO / "config" / "AD_cam_2.yml").read_text())
    snapshot = copy.deepcopy(raw)
    prepared = prepare_config(raw)
    assert raw == snapshot
    assert prepared.effective_config is not raw
    prepared2 = prepare_config(raw)
    assert prepared2.effective_config == prepared.effective_config


def test_prepare_path_list_matches_merged_mapping(tmp_path):
    paths = [REPO / "config" / "AD_cam_2.yml"]
    prepared = prepare_config(paths, options=ProcessingOptions())
    isp = _facade(list(paths))
    _compare(prepared, isp, options_diagnostics=False)


def test_diagnostics_suppression_matches(options_diagnostics=True):
    options = ProcessingOptions(diagnostics=False)
    prepared = prepare_config(REPO / "config" / "AD_cam_2.yml", options=options)
    isp = _facade(REPO / "config" / "AD_cam_2.yml", options)
    for section, values in prepared.effective_config.items():
        if isinstance(values, dict):
            for key in ("is_save", "is_plot_curve", "is_debug"):
                if key in values:
                    assert values[key] is False, (section, key)
    _compare(prepared, isp, options_diagnostics=False)


def test_malformed_mapping_raises():
    with pytest.raises(Exception):
        prepare_config({"sensor_info": 3})
    with pytest.raises(Exception):
        prepare_config(12345)


def test_retired_or_missing_keys_error_matches(tmp_path):
    bad = tmp_path / "bad_cam.yml"
    bad.write_text("platform:\n  filename: f.raw\nsensor_info:\n  bit_depth: 10\n")
    with pytest.raises(KeyError):
        prepare_config(bad)


def test_unknown_config_path_raises():
    with pytest.raises(Exception):
        prepare_config(REPO / "config" / "does_not_exist_cam.yml")


def test_extends_chain_resolves(tmp_path):
    child = tmp_path / "child_cam.yml"
    child.write_text(
        "extends: " + str(REPO / "config" / "AD_cam_2.yml") + "\n"
        "platform:\n  filename: override.raw\n"
    )
    prepared = prepare_config(child)
    base = prepare_config(REPO / "config" / "AD_cam_2.yml")
    assert prepared.effective_config["platform"]["filename"] == "override.raw"
    assert {k: v for k, v in prepared.effective_config.items() if k != "platform"} == \
        {k: v for k, v in base.effective_config.items() if k != "platform"}


def test_extends_cycle_raises(tmp_path):
    a = tmp_path / "a.yml"
    b = tmp_path / "b.yml"
    a.write_text("extends: b.yml\nplatform: {}\n")
    b.write_text("extends: a.yml\nplatform: {}\n")
    with pytest.raises(Exception):
        prepare_config(a)


def test_validation_rejects_invalid_active_values():
    raw = yaml.safe_load((REPO / "config" / "AD_cam_2.yml").read_text())
    raw["sensor_info"]["width"] = -5
    with pytest.raises(ValueError, match="validation"):
        prepare_config(raw)


def test_validation_rejects_schema2_retired_active_section():
    raw = yaml.safe_load((REPO / "config" / "AD_cam_2.yml").read_text())
    raw["schema_version"] = 2
    raw["gamma_correction"] = {"is_enable": True}
    with pytest.raises(ValueError, match="gamma_correction"):
        prepare_config(raw)


def test_validation_can_be_disabled_for_facade_parity():
    raw = yaml.safe_load((REPO / "config" / "AD_cam_2.yml").read_text())
    raw["sensor_info"]["width"] = -5
    prepared = prepare_config(raw, validate=False)
    assert prepared.effective_config["sensor_info"]["width"] == -5


def test_diagnostics_true_with_output_dir(tmp_path):
    options = ProcessingOptions(diagnostics=True, output_dir=tmp_path)
    prepared = prepare_config(REPO / "config" / "AD_cam_2.yml", options=options)
    isp = _facade(REPO / "config" / "AD_cam_2.yml", options)
    platform = prepared.effective_config["platform"]
    assert platform["output_dir"] == str(tmp_path / "module_output")
    assert platform["debug_log_file"] == str(tmp_path / "pipeline.log")
    assert platform["final_output_dir"] == str(tmp_path)
    _compare(prepared, isp, options_diagnostics=True)
