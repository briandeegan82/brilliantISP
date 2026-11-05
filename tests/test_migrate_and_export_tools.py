from __future__ import annotations

import json
import os
import subprocess
import sys
import pytest
import yaml
from pathlib import Path

from tools.migrate_pipeline_config import run_migrate, main as migrate_main
from tools.export_boltisp_yaml import export_bolt_yaml
from tools.gui.yaml_compat import load_preset_yaml

REPO = Path(__file__).resolve().parents[1]
SVS = REPO / "config" / "SVS_cam.yml"


def test_migrate_v1_to_v2_exclusive_outputs(tmp_path: Path) -> None:
    out = tmp_path / "native.yml"
    report = tmp_path / "rep.json"
    output, rpt = run_migrate(["config/SVS_cam.yml"], out, report)
    migrated = yaml.safe_load(output.read_text())
    assert migrated["schema_version"] == 2
    assert "output" in migrated
    data = json.loads(rpt.read_text())
    assert data["source_schema_version"] in (None, 1)
    assert data["removed_sections"]
    with pytest.raises(ValueError, match="exists"):
        run_migrate(["config/SVS_cam.yml"], out, report)
    # existing report also refuses even when output is new
    with pytest.raises(ValueError, match="exists"):
        run_migrate(["config/SVS_cam.yml"], tmp_path / "new.yml", report)


def test_migrate_inheritance_and_source_unchanged(tmp_path: Path) -> None:
    before = SVS.read_text()
    base = tmp_path / "base.yml"
    base.write_text(before)
    child = tmp_path / "child.yml"
    child.write_text("extends: base.yml\nsensor_info:\n  width: 64\n")
    out = tmp_path / "out.yml"
    run_migrate([str(child)], out, tmp_path / "r.json")
    assert yaml.safe_load(out.read_text())["sensor_info"]["width"] == 64
    assert SVS.read_text() == before


def test_migrate_rejects_invalid_and_aliases(tmp_path: Path) -> None:
    # invalid source: negative manual WB gain must fail validation
    bad = tmp_path / "bad.yml"
    cfg = yaml.safe_load(SVS.read_text())
    cfg["white_balance"]["r_gain"] = -1.0
    bad.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="valid"):
        run_migrate([str(bad)], tmp_path / "o.yml", tmp_path / "r.json")
    assert not list(tmp_path.glob("o.*")) and not list(tmp_path.glob("r.json"))
    # identical output/report paths rejected
    with pytest.raises(ValueError, match="distinct"):
        run_migrate(["config/SVS_cam.yml"], tmp_path / "same.yml", tmp_path / "same.yml")
    # symlink target rejected
    real = tmp_path / "real.yml"
    real.write_text("x")
    link = tmp_path / "link.yml"
    os.symlink(real, link)
    with pytest.raises(ValueError, match="exists"):
        run_migrate(["config/SVS_cam.yml"], link, tmp_path / "r2.json")
    # missing parent directory rejected (no mkdir)
    with pytest.raises(ValueError, match="directory"):
        run_migrate(["config/SVS_cam.yml"],
                    tmp_path / "no_dir" / "o.yml", tmp_path / "r3.json")


def test_migrate_explicit_v2_input(tmp_path: Path) -> None:
    src = tmp_path / "v2.yml"
    src.write_text("schema_version: 2\n")
    with pytest.raises(ValueError):
        run_migrate([str(src)], tmp_path / "o.yml", tmp_path / "r.json")


def test_bolt_export_rejects_v2(tmp_path: Path) -> None:
    v2 = tmp_path / "v2.yml"
    v2.write_text("schema_version: 2\noutput: {}\n")
    with pytest.raises(ValueError, match="schema_version 2"):
        export_bolt_yaml(v2, tmp_path / "bolt.yml")
    v1 = tmp_path / "v1.yml"
    v1.write_text(yaml.safe_dump(
        {"sensor_info": {"bit_depth": 12}, "tone_mapping": {"tone_mapper": "reinhard"}}))
    export_bolt_yaml(v1, tmp_path / "bolt.yml")
    assert (tmp_path / "bolt.yml").exists()


def test_gui_v2_loads_for_editing(tmp_path: Path) -> None:
    v2 = tmp_path / "v2.yml"
    v2.write_text("schema_version: 2\n")
    assert load_preset_yaml(v2)["schema_version"] == 2


def test_migrate_cli_alias_and_existing_output(tmp_path: Path) -> None:
    out = tmp_path / "n.yml"
    rep = tmp_path / "n_report.json"
    rc = migrate_main(["--config", "config/SVS_cam.yml",
                       "--output", str(out), "--report", str(rep)])
    assert rc == 0 and out.exists() and rep.exists()
    rc = migrate_main(["--config", "config/SVS_cam.yml",
                       "--output", str(out), "--report", str(rep)])
    assert rc == 1


def test_migrate_direct_script_no_pythonpath(tmp_path: Path) -> None:
    out = tmp_path / "direct.yml"
    rep = tmp_path / "direct_report.json"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run(
        [sys.executable, str(REPO / "tools" / "migrate_pipeline_config.py"),
         "--config", str(SVS), "--output", str(out), "--report", str(rep)],
        cwd=REPO, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert out.exists() and rep.exists()
