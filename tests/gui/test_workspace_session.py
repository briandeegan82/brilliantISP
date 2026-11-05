from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.gui.model import WorkspaceModel
from tools.gui.workspace_session import (
    BUNDLE_NAME, WorkspaceSession, read_last_workspace)


def _cfg() -> dict:
    from tests.test_pipeline_runner import _fast_config
    return _fast_config()


def test_create_writes_bundle_and_roundtrip(tmp_path) -> None:
    cfg = _cfg()
    session, model = WorkspaceSession.create(tmp_path, cfg)
    assert session.directory.is_dir()
    assert (session.directory / BUNDLE_NAME).is_file()
    payload = json.loads((session.directory / BUNDLE_NAME).read_text())
    assert payload["version"] == 1
    assert payload["config"]["sensor_info"] == cfg["sensor_info"]
    assert model.is_dirty is False
    session2, model2, ui_state, snaps, warns = WorkspaceSession.open(
        session.directory)
    assert model2.config == model.config
    assert warns == []


def test_save_preserves_anchors_and_dirty_source(tmp_path) -> None:
    session, model = WorkspaceSession.create(tmp_path, _cfg())
    job = session.directory / "jobs" / "j1"
    job.mkdir(parents=True)
    src = job / "source.raw"
    src.write_bytes(b"\x01" * 64)
    model.adopt_source({"path": str(src), "fingerprint": "f"}, None)
    assert model.is_dirty  # unsaved source counts
    session.save(model, {"zoom": 2}, {})
    assert not model.is_dirty
    payload = json.loads((session.directory / BUNDLE_NAME).read_text())
    assert payload["saved_source"]["path"] == "jobs/j1/source.raw"


def test_save_external_source_rejected(tmp_path) -> None:
    session, model = WorkspaceSession.create(tmp_path, _cfg())
    outside = tmp_path / "external.raw"
    outside.write_bytes(b"x")
    before = (session.directory / BUNDLE_NAME).read_text()
    model.adopt_source({"path": str(outside)}, None)
    with pytest.raises(ValueError, match="escapes"):
        session.save(model, {}, {})
    assert (session.directory / BUNDLE_NAME).read_text() == before
    assert model.is_dirty  # anchors unchanged


def test_portable_source_roundtrip(tmp_path) -> None:
    session, model = WorkspaceSession.create(tmp_path / "r1", _cfg())
    job = session.directory / "jobs" / "j1"
    job.mkdir(parents=True)
    (job / "source.raw").write_bytes(b"payload")
    model.adopt_source(
        {"path": str(job / "source.raw"), "fingerprint": "f"}, None)
    session.save(model, {}, {})
    import shutil
    (tmp_path / "r2").mkdir()
    moved = tmp_path / "r2" / session.directory.name
    shutil.move(str(session.directory), moved)
    s2, m2, _u, _s, _w = WorkspaceSession.open(moved)
    assert m2.source["path"].startswith(str(moved))
    assert (moved / "jobs" / "j1" / "source.raw").read_bytes() == b"payload"


def test_save_failure_keeps_old_bundle_and_anchors(tmp_path) -> None:
    session, model = WorkspaceSession.create(tmp_path, _cfg())
    original = (session.directory / BUNDLE_NAME).read_text()
    model._cfg["broken"] = object()  # non-serializable
    with pytest.raises(ValueError):
        session.save(model, {}, {})
    assert (session.directory / BUNDLE_NAME).read_text() == original
    # model anchors not advanced
    assert model.is_dirty


def test_atomic_write_failure_preserves_bundle(tmp_path, monkeypatch) -> None:
    session, model = WorkspaceSession.create(tmp_path, _cfg())
    model.edit("digital_gain.gain", 2)
    original = (session.directory / BUNDLE_NAME).read_text()
    saved_before = model.saved_snapshot
    import os
    monkeypatch.setattr(os, "replace",
                        lambda *a: (_ for _ in ()).throw(OSError("io")))
    with pytest.raises(OSError):
        session.save(model, {}, {})
    assert (session.directory / BUNDLE_NAME).read_text() == original
    assert model.saved_snapshot == saved_before
    assert model.is_dirty
    assert not list(session.directory.glob("*.tmp"))


def test_malformed_bundle_raises_not_fallback(tmp_path) -> None:
    session, _m = WorkspaceSession.create(tmp_path, _cfg())
    (session.directory / BUNDLE_NAME).write_text("{not json")
    with pytest.raises(ValueError):
        WorkspaceSession.open(session.directory)
    (session.directory / BUNDLE_NAME).write_text('"string"')
    with pytest.raises(ValueError):
        WorkspaceSession.open(session.directory)
    (session.directory / BUNDLE_NAME).write_text(
        json.dumps({"version": 99, "config": {}, "saved_snapshot": {}}))
    with pytest.raises(ValueError, match="version"):
        WorkspaceSession.open(session.directory)


def test_source_traversal_rejected(tmp_path) -> None:
    session, _m = WorkspaceSession.create(tmp_path, _cfg())
    payload = json.loads((session.directory / BUNDLE_NAME).read_text())
    payload["source"] = {"path": "../../etc/passwd"}
    (session.directory / BUNDLE_NAME).write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="escapes"):
        WorkspaceSession.open(session.directory)


def test_missing_source_warns_and_opens(tmp_path) -> None:
    session, model = WorkspaceSession.create(tmp_path, _cfg())
    payload = json.loads((session.directory / BUNDLE_NAME).read_text())
    payload["source"] = {"path": "source.raw", "fingerprint": "f"}
    (session.directory / BUNDLE_NAME).write_text(json.dumps(payload))
    _s, m2, _ui, _sn, warns = WorkspaceSession.open(session.directory)
    assert any("missing" in w for w in warns)
    assert m2.source["path"].startswith(str(session.directory.resolve()))


def test_snapshot_traversal_and_missing_warn(tmp_path) -> None:
    session, _m = WorkspaceSession.create(tmp_path, _cfg())
    payload = json.loads((session.directory / BUNDLE_NAME).read_text())
    payload["snapshots"] = {"A": {"manifest": "../outside"}}
    (session.directory / BUNDLE_NAME).write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="escapes"):
        WorkspaceSession.open(session.directory)
    jobdir = session.directory / "jobs" / "xyz"
    jobdir.mkdir(parents=True)
    payload["snapshots"] = {"A": {"manifest": "jobs/xyz"}}
    (session.directory / BUNDLE_NAME).write_text(json.dumps(payload))
    _s, _m, _ui, snaps, warns = WorkspaceSession.open(session.directory)
    assert snaps["A"]["missing"] is True
    assert any("missing" in w for w in warns)
    (jobdir / "result.json").write_text("{}")
    _s, _m, _ui, snaps, warns = WorkspaceSession.open(session.directory)
    assert "missing" not in snaps["A"]


def test_legacy_session_fallback(tmp_path) -> None:
    directory = tmp_path / "legacy"
    directory.mkdir()
    cfg = _cfg()
    import yaml
    (directory / "config.yml").write_text(yaml.safe_dump(cfg))
    (directory / "input.raw").write_bytes(b"\x02" * 32)
    (directory / "session.json").write_text(
        json.dumps({"input_filename": "input.raw"}))
    _s, model, _ui, _sn, warns = WorkspaceSession.open(directory)
    assert model.source is not None
    assert model.source["path"].endswith("input.raw")
    assert model.source["fingerprint"]
    # original bytes untouched
    assert (directory / "input.raw").read_bytes() == b"\x02" * 32
    assert warns == []


def test_legacy_basename_validation(tmp_path) -> None:
    import yaml
    directory = tmp_path / "legacy2"
    directory.mkdir()
    (directory / "config.yml").write_text(yaml.safe_dump(_cfg()))
    (directory / "session.json").write_text(
        json.dumps({"input_filename": "../escape.raw"}))
    with pytest.raises(ValueError):
        WorkspaceSession.open(directory)


def test_legacy_missing_config_raises(tmp_path) -> None:
    directory = tmp_path / "legacy3"
    directory.mkdir()
    (directory / "session.json").write_text(
        json.dumps({"input_filename": "a.raw"}))
    with pytest.raises(ValueError, match="config"):
        WorkspaceSession.open(directory)


def test_legacy_default_input_filename(tmp_path) -> None:
    import yaml
    directory = tmp_path / "legacy4"
    directory.mkdir()
    (directory / "config.yml").write_text(yaml.safe_dump(_cfg()))
    (directory / "input.raw").write_bytes(b"\x03" * 16)
    _s, model, _u, _sn, _w = WorkspaceSession.open(directory)
    assert model.source is not None
    assert model.source["path"].endswith("input.raw")


def test_last_pointer_read_then_create(tmp_path) -> None:
    session, model = WorkspaceSession.create(tmp_path, _cfg())
    assert read_last_workspace(tmp_path) is None
    session.save_last_pointer(tmp_path)
    assert read_last_workspace(tmp_path) == session.directory


def test_create_does_not_overwrite_pointer(tmp_path) -> None:
    s1, _m = WorkspaceSession.create(tmp_path, _cfg())
    s1.save_last_pointer(tmp_path)
    s2, _m2 = WorkspaceSession.create(tmp_path, _cfg())
    assert read_last_workspace(tmp_path) == s1.directory


def test_corrupt_last_pointer_raises(tmp_path) -> None:
    (tmp_path / "last_session.json").write_text("{bad")
    with pytest.raises(ValueError):
        read_last_workspace(tmp_path)
    (tmp_path / "last_session.json").write_text('"x"')
    with pytest.raises(ValueError):
        read_last_workspace(tmp_path)
    (tmp_path / "last_session.json").write_text(
        json.dumps({"session_id": "../esc"}))
    with pytest.raises(ValueError):
        read_last_workspace(tmp_path)


def test_nan_literal_bundle_rejected(tmp_path) -> None:
    session, _m = WorkspaceSession.create(tmp_path, _cfg())
    (session.directory / BUNDLE_NAME).write_text(
        '{"version":1,"config":{"a":NaN},"preset_snapshot":{},'
        '"saved_snapshot":{}}')
    with pytest.raises(ValueError):
        WorkspaceSession.open(session.directory)


def test_create_rejects_invalid_config(tmp_path) -> None:
    cfg = _cfg()
    cfg["sensor_info"]["width"] = float("nan")
    with pytest.raises(ValueError):
        WorkspaceSession.create(tmp_path, cfg)
    assert not list(tmp_path.iterdir())


def test_model_restored_counters_monotonic(tmp_path) -> None:
    session, model = WorkspaceSession.create(tmp_path, _cfg())
    model.edit("digital_gain.gain", 2)
    session.save(model, {}, {})
    _s, model2, _u, _sn, _w = WorkspaceSession.open(session.directory)
    model2.edit("digital_gain.gain", 3)
    assert model2.revision > 0
