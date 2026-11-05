from __future__ import annotations

import pytest

from tools.gui.model import WorkspaceModel


def _cfg() -> dict:
    from tests.test_pipeline_runner import _fast_config
    return _fast_config()


def test_config_is_private_deep_copy() -> None:
    cfg = _cfg()
    m = WorkspaceModel(cfg)
    cfg["sensor_info"]["width"] = 999
    assert m.config["sensor_info"]["width"] != 999
    snap = m.config
    snap["sensor_info"]["width"] = 1
    assert m.config["sensor_info"]["width"] != 1


def test_revision_generation_monotonic() -> None:
    m = WorkspaceModel(_cfg(), session_id="s")
    r, g = m.revision, m.generation
    m.edit("crop.is_enable", True)
    m.undo(); m.redo()
    m.set_source({"path": "a.raw", "fingerprint": "f"})
    m.new_session(_cfg(), session_id="s2")
    assert m.revision > r and m.generation > g


def test_noop_equal_edit_no_history() -> None:
    m = WorkspaceModel(_cfg())
    r0 = m.revision
    m.edit("crop.is_enable", m.config["crop"]["is_enable"])
    assert m.revision == r0 and not m._undo


def test_undo_restores_config_and_preset_anchor() -> None:
    cfg = _cfg()
    m = WorkspaceModel(cfg, preset_path="p.yml")
    cfg2 = _cfg(); cfg2["crop"]["is_enable"] = not cfg["crop"]["is_enable"]
    m.replace_config(cfg2, preset_path="q.yml", label="Load")
    m.undo()
    assert m.preset_path == "p.yml"
    assert m.preset_snapshot["crop"]["is_enable"] == cfg["crop"]["is_enable"]
    m.redo()
    assert m.preset_path == "q.yml"


def test_saved_anchor_not_rewound_and_source_dirty() -> None:
    m = WorkspaceModel(_cfg(), preset_path="p")
    m.mark_saved()
    m.adopt_source({"path": "x.raw", "fingerprint": "h"},
                   {"width": 8})
    assert m.is_dirty
    m.undo()
    assert not m.is_dirty  # saved snapshot + saved source restored
    m.mark_saved()
    m.edit("crop.is_enable", True)
    m.undo()
    assert m.is_dirty is False


def test_missing_path_reset_removes_key() -> None:
    m = WorkspaceModel(_cfg(), preset_path="p")
    assert "extra_flag" not in m.config.get("crop", {})
    m.edit("crop.extra_flag", 5)
    assert m.config["crop"]["extra_flag"] == 5
    m.reset_parameter("crop.extra_flag")
    assert "extra_flag" not in m.config["crop"]


def test_history_bounded_and_not_aliased() -> None:
    m = WorkspaceModel(_cfg())
    for i in range(150):
        m.edit("digital_gain.gain", i)
    assert len(m._undo) == 100
    first = m._undo[0]["state"]["config"]
    first["digital_gain"]["gain"] = -1
    assert m.config["digital_gain"]["gain"] == 149


def test_invalid_and_nan_blocking() -> None:
    m = WorkspaceModel(_cfg())
    assert m.can_process() and m.config_json_ready()
    m._cfg["sensor_info"] = "garbage"
    assert not m.can_process()
    assert m.config["sensor_info"] == "garbage"
    m2 = WorkspaceModel(_cfg())
    m2._cfg["sensor_info"]["width"] = float("nan")
    assert not m2.config_json_ready()
    m3 = WorkspaceModel(_cfg())
    m3._cfg["weird"] = object()
    assert not m3.config_json_ready()


def test_modified_paths() -> None:
    m = WorkspaceModel(_cfg(), preset_path="p")
    m.edit("crop.is_enable", True)
    assert "crop.is_enable" in m.modified_paths
