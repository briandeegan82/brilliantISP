from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from brilliant_isp import BrilliantISP
from tools.gui.yaml_compat import load_preset_yaml

REPO = Path(__file__).resolve().parents[1]
CLI = REPO / "tools" / "validate_config.py"
SVS = REPO / "config" / "SVS_cam.yml"
EXPERIMENT = REPO / "config" / "experiment.yml"


def run_cli(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        capture_output=True,
        text=True,
        cwd=cwd or REPO,
    )


def test_cli_valid_config_exit0(tmp_path: Path) -> None:
    result = run_cli("--config", str(SVS))
    assert result.returncode == 0
    assert "Configuration valid" in result.stdout


def test_cli_valid_experiment_extends(tmp_path: Path) -> None:
    result = run_cli("--config", str(EXPERIMENT), "--json")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["valid"] is True
    assert any("experiment.yml" in s for s in payload["config_sources"])
    assert not payload["issues"] or all(
        i["severity"] == "warning" for i in payload["issues"]
    )


def test_cli_invalid_config_exit1(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yml"
    cfg = yaml.safe_load(SVS.read_text())
    cfg["sensor_info"]["width"] = 1920.5
    bad.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    result = run_cli("--config", str(bad), "--json")
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["valid"] is False
    assert any(
        i["path"] == "sensor_info.width" and i["severity"] == "error"
        for i in payload["issues"]
    )
    assert "Traceback" not in result.stderr + result.stdout


def test_cli_rejects_blc_pwc_double_subtraction(tmp_path: Path) -> None:
    bad = tmp_path / "double_black.yml"
    cfg = yaml.safe_load(SVS.read_text())
    cfg["black_level_correction"]["is_enable"] = True
    bad.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    result = run_cli("--config", str(bad), "--json")
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert any(i["path"] == "companding.pre_linearization_black_level"
               for i in payload["issues"])


def test_cli_missing_sections_exit1(tmp_path: Path) -> None:
    sparse = tmp_path / "sparse.yml"
    sparse.write_text("sensor_info:\n  width: 10\n", encoding="utf-8")
    result = run_cli("--config", str(sparse), "--json")
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    missing = {i["path"] for i in payload["issues"]}
    assert "tone_mapping" in missing
    assert "digital_gain" in missing


def test_cli_missing_file_exit2(tmp_path: Path) -> None:
    result = run_cli("--config", str(tmp_path / "nonexistent.yml"), "--json")
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["valid"] is False
    assert "error" in payload
    assert "Traceback" not in result.stderr + result.stdout


def test_cli_malformed_yaml_exit2(tmp_path: Path) -> None:
    broken = tmp_path / "broken.yml"
    broken.write_text("key: [unclosed\n  bad: : :\n", encoding="utf-8")
    result = run_cli("--config", str(broken), "--json")
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["valid"] is False
    assert "error" in payload


def test_cli_null_required_section_exit1(tmp_path: Path) -> None:
    bad = tmp_path / "null_sections.yml"
    cfg = yaml.safe_load(SVS.read_text())
    cfg["companding"] = None
    bad.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    result = run_cli("--config", str(bad), "--json")
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert any(
        i["path"] == "companding" and i["severity"] == "error"
        for i in payload["issues"]
    )


def test_cli_list_platform_section_exit1(tmp_path: Path) -> None:
    bad = tmp_path / "list_platform.yml"
    cfg = yaml.safe_load(SVS.read_text())
    cfg["platform"] = []
    bad.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    result = run_cli("--config", str(bad), "--json")
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert any(
        i["path"] == "platform" and i["severity"] == "error"
        for i in payload["issues"]
    )


def test_cli_schema_v2_sparse_exit1(tmp_path: Path) -> None:
    # schema_version 2 is supported: a sparse v2 source loads and then fails
    # validation (exit 1) because the native 'output' section is missing.
    v2 = tmp_path / "v2.yml"
    v2.write_text("schema_version: 2\ntone_mapping:\n  is_enable: true\n", encoding="utf-8")
    result = run_cli("--config", str(v2), "--json")
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["valid"] is False
    assert "error" not in payload
    assert any(
        i["path"] == "output" and "required" in i["message"]
        for i in payload["issues"]
    )
    result_text = run_cli("--config", str(v2))
    assert result_text.returncode == 1
    assert "output" in result_text.stdout


def test_cli_full_migrated_v2_valid(tmp_path: Path) -> None:
    from tools.migrate_pipeline_config import run_migrate
    migrated = tmp_path / "native.yml"
    run_migrate(["config/SVS_cam.yml"], migrated, tmp_path / "rep.json")
    result = run_cli("--config", str(migrated), "--json")
    assert result.returncode == 0
    assert json.loads(result.stdout)["valid"] is True
    # retired color sections are not required for v2
    cfg = yaml.safe_load(migrated.read_text())
    for retired in ("gamma_correction", "color_space_conversion",
                    "rgb_conversion", "yuv_conversion_format"):
        cfg.pop(retired, None)
    stripped = tmp_path / "stripped.yml"
    stripped.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert run_cli("--config", str(stripped)).returncode == 0


def test_cli_overlay_merge_order_and_sources(tmp_path: Path) -> None:
    overlay = tmp_path / "overlay.yml"
    overlay.write_text("platform:\n  debug_enabled: false\n", encoding="utf-8")
    result = run_cli("--config", str(SVS), "--config", str(overlay), "--json")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["valid"] is True
    assert payload["config_sources"][0].endswith("SVS_cam.yml")
    assert payload["config_sources"][1] == str(overlay.resolve())


def test_cli_runs_from_other_cwd(tmp_path: Path) -> None:
    result = run_cli("--config", str(SVS), cwd=tmp_path)
    assert result.returncode == 0
    assert "Configuration valid" in result.stdout


def test_pipeline_loader_v2_missing_keys_and_null(tmp_path: Path) -> None:
    # sparse v2 loads as a supported version but fails the native required set
    v2 = tmp_path / "v2.yml"
    v2.write_text("schema_version: 2\n", encoding="utf-8")
    with pytest.raises(KeyError, match="missing required keys"):
        BrilliantISP(str(tmp_path), str(v2), outFileName="", output_path=str(tmp_path))
    null = tmp_path / "null.yml"
    null.write_text("schema_version:\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        BrilliantISP(str(tmp_path), str(null), outFileName="", output_path=str(tmp_path))


def test_pipeline_loader_accepts_full_migrated_v2(tmp_path: Path) -> None:
    from tools.migrate_pipeline_config import run_migrate
    migrated = tmp_path / "native.yml"
    run_migrate(["config/SVS_cam.yml"], migrated, tmp_path / "rep.json")
    isp = BrilliantISP(str(tmp_path), str(migrated), outFileName="",
                       output_path=str(tmp_path))
    assert isp.c_yaml is not None
    assert isp.c_yaml.get("schema_version") == 2
    assert isp.c_yaml.get("output") is not None


def test_gui_loader_accepts_v2_rejects_null(tmp_path: Path) -> None:
    v2 = tmp_path / "v2.yml"
    v2.write_text("schema_version: 2\n", encoding="utf-8")
    assert load_preset_yaml(v2)["schema_version"] == 2
    null = tmp_path / "null.yml"
    null.write_text("schema_version:\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_preset_yaml(null)


def test_pipeline_notices_stored_without_processing(tmp_path: Path) -> None:
    cfg = yaml.safe_load(SVS.read_text())
    cfg["aces"] = {"variant": "fitted", "output_transform": "sRGB"}
    cfg_path = tmp_path / "svs_test.yml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    isp = BrilliantISP(str(tmp_path), str(cfg_path), outFileName="", output_path=str(tmp_path))
    assert any("aces.variant" in n for n in isp.config_notices)
    assert any("aces.output_transform" in n for n in isp.config_notices)
    assert not (tmp_path / "module_output").exists()
    assert not list(tmp_path.glob("*.png"))
