from __future__ import annotations

import json
from pathlib import Path

import pytest
from typing import cast

from tools.gui.controller import WorkspaceController
from tools.gui.jobs import WorkerManager
from tools.gui.model import WorkspaceModel
from tools.gui.protocol import JobRequest, JobTag, WorkerEvent


class FakeManager(WorkerManager):
    def __init__(self) -> None:
        self.submitted: list[JobRequest] = []
        self.cancelled = 0
        self.events: list[WorkerEvent] = []
        self.terminated = 0
        self._closed = False
        self._stopping = False
        self._active = None
    def submit(self, req):
        if self._active is not None:
            raise RuntimeError("active")
        self.submitted.append(req)
        self._active = req
    def cancel(self): self.cancelled += 1
    def poll(self):
        ev, self.events = self.events, []
        return ev
    def terminate(self):
        self.terminated += 1
        self._active = None
    def finish_active(self): self._active = None
    def request_close(self): self._closed = True


def _make(tmp_path, **kw):
    from tests.test_pipeline_runner import _fast_config
    model = WorkspaceModel(_fast_config(), session_id="s1")
    model.set_source({"path": str(tmp_path / "s.raw"), "fingerprint": "fp"})
    return (WorkspaceController(model, FakeManager(),
                                root_dir=tmp_path / "sessions", **kw),
            model)


def test_aux_priority_after_process(tmp_path) -> None:
    ctl, _m = _make(tmp_path)
    ctl.request_process()
    ctl.request_analysis("m", roi=[0, 0, 4, 4])
    assert len(cast(FakeManager, ctl.manager).submitted) == 1
    assert ctl.active is not None
    ctl._handle(WorkerEvent(ctl.active.tag, "done", "{}"))
    cast(FakeManager, ctl.manager).finish_active()
    ctl.tick()
    assert cast(FakeManager, ctl.manager).submitted[-1].operation == "analysis"


def test_latest_analysis_only_publishes(tmp_path) -> None:
    ctl, _m = _make(tmp_path)
    r1 = ctl.request_analysis("m1")
    assert r1 is not None
    ctl._handle(WorkerEvent(r1.tag, "done",
                            json.dumps({"analysis_path": "a",
                                        "descriptor": {"manifest_dir": "m1"}})))
    cast(FakeManager, ctl.manager).finish_active()
    assert ctl.analysis_result is not None
    assert ctl.analysis_result["descriptor"]["manifest_dir"] == "m1"


def test_old_aux_racing_done_discarded(tmp_path) -> None:
    ctl, _m = _make(tmp_path)
    r1 = ctl.request_analysis("m1")
    assert r1 is not None
    r2 = ctl.request_analysis("m2")
    assert r2 is not None
    assert cast(FakeManager, ctl.manager).cancelled == 1
    ctl._handle(WorkerEvent(r1.tag, "done",
                            json.dumps({"analysis_path": "a"})))
    assert ctl.analysis_result is None
    cast(FakeManager, ctl.manager).finish_active()
    ctl.tick()
    assert ctl.active == r2
    ctl._handle(WorkerEvent(r2.tag, "done",
                            json.dumps({"analysis_path": "b"})))
    cast(FakeManager, ctl.manager).finish_active()
    assert ctl.analysis_result is not None
    assert ctl.analysis_result["request_id"] == r2.tag.request_id
    ctl.context_changed()
    stale = WorkerEvent(r1.tag, "done",
                        json.dumps({"analysis_path": "b"}))
    ctl._handle(stale)
    assert ctl.analysis_result is None
    assert ctl.active is None


def test_analysis_closed_returns_none(tmp_path) -> None:
    ctl, _m = _make(tmp_path)
    ctl.close()
    assert ctl.request_analysis("m") is None


def test_analysis_allowed_with_invalid_cfg(tmp_path) -> None:
    ctl, model, *_ = _make(tmp_path)
    model._cfg["sensor_info"]["width"] = float("nan")
    req = ctl.request_analysis("m")
    assert req is not None and req.config_json == "{}"


def test_full_tag_mismatch_does_not_clear(tmp_path) -> None:
    ctl, _m = _make(tmp_path)
    ctl.request_process()
    assert ctl.active is not None
    wrong = JobTag(request_id=ctl.active.tag.request_id, session_id="s1",
                   generation=9, revision=0, source_fingerprint="fp")
    ctl._handle(WorkerEvent(wrong, "done", "{}"))
    assert ctl.active is not None


def test_auto_no_source_never_submits(tmp_path) -> None:
    from tests.test_pipeline_runner import _fast_config
    model = WorkspaceModel(_fast_config(), session_id="s1")
    t = [0.0]
    ctl = WorkspaceController(model, FakeManager(), auto_default=True,
                              root_dir=tmp_path, clock=lambda: t[0])
    ctl.on_edit()
    t[0] = 10.0
    ctl.tick()
    assert cast(FakeManager, ctl.manager).submitted == []
    assert ctl.status == "source_required"


def test_stop_disables_auto(tmp_path) -> None:
    ctl, _m = _make(tmp_path, auto_default=True)
    ctl.request_process()
    ctl.stop_worker()
    assert ctl.auto_enabled is False


def test_worker_analysis_dispatch(tmp_path, monkeypatch) -> None:
    import tools.gui.worker as worker
    import tools.gui.artifacts as artifacts
    monkeypatch.setattr(artifacts, "analyze_artifact",
                        lambda payload: {"probe": "ok", "payload": payload})
    events = []
    req = JobRequest(
        tag=JobTag(request_id="a1", session_id="s", generation=0,
                   revision=0, source_fingerprint=""),
        operation="analysis", config_json="{}", source_path="",
        output_dir=str(tmp_path / "job"),
        payload_json=json.dumps({"manifest_dir": "m", "stage": "preview",
                                 "roi": None, "comparison_dir": None}))
    class _C:
        def is_set(self): return False
    worker._dispatch(req, _C(), _C(),
                     lambda e, terminal=False: events.append(e))
    done = [e for e in events if e.kind == "done"]
    assert done
    data = json.loads(done[0].payload_json)
    assert Path(data["analysis_path"]).name == "analysis.json"
    assert json.loads(
        (tmp_path / "job" / "analysis.json").read_text())["probe"] == "ok"


def test_manual_pending_beats_aux(tmp_path) -> None:
    ctl, _m = _make(tmp_path)
    ctl.request_process()
    ctl._pending = {"role": "process", "manual": True}
    ctl._pending_at = 0.0
    ctl.request_analysis("m")
    assert ctl.active is not None
    ctl._handle(WorkerEvent(ctl.active.tag, "done", "{}"))
    cast(FakeManager, ctl.manager).finish_active()
    ctl.tick()
    assert cast(FakeManager, ctl.manager).submitted[-1].operation == "process"


def test_real_manager_full_tag_mismatch(tmp_path) -> None:
    import multiprocessing as mp
    import time
    mgr = WorkerManager()
    ctx = mp.get_context("spawn")
    mgr._ensure()
    req = JobRequest(
        tag=JobTag(request_id="r", session_id="s", generation=0,
                   revision=0, source_fingerprint=""),
        operation="process", config_json="{}", source_path="", output_dir="")
    try:
        mgr._active = req
        wrong = JobTag(request_id="r", session_id="s", generation=9,
                       revision=0, source_fingerprint="")
        mgr._evt_q.put(WorkerEvent(wrong, "done", "{}"))
        time.sleep(0.3)
        mgr.poll()
        assert mgr._active is req
    finally:
        mgr.request_close()
        deadline = time.monotonic() + 10
        while not mgr.shutdown_complete and time.monotonic() < deadline:
            mgr.shutdown_step()
            time.sleep(0.05)
        assert mgr.shutdown_complete


def _analysis_req(tmp_path, payload="{}"):
    return JobRequest(
        tag=JobTag(request_id="a", session_id="s", generation=0,
                   revision=0, source_fingerprint=""),
        operation="analysis", config_json="{}", source_path="",
        output_dir=str(tmp_path / "job"), payload_json=payload)


class _Never:
    def is_set(self): return False


class _Always:
    def is_set(self): return True


def test_analysis_existing_jobdir_errors(tmp_path) -> None:
    import tools.gui.worker as worker
    (tmp_path / "job").mkdir()
    sentinel = tmp_path / "job" / "keep.txt"
    sentinel.write_text("keep")
    events = []
    with pytest.raises(FileExistsError):
        worker._dispatch(_analysis_req(tmp_path), _Never(), _Never(),
                         lambda e, terminal=False: events.append(e))
    assert sentinel.read_text() == "keep"


def test_analysis_cancel_maps_cancelled(tmp_path, monkeypatch) -> None:
    import tools.gui.worker as worker
    import tools.gui.artifacts as artifacts
    monkeypatch.setattr(artifacts, "analyze_artifact", lambda p: {"probe": "ok"})
    events = []
    worker._dispatch(_analysis_req(tmp_path), _Always(), _Always(),
                     lambda e, terminal=False: events.append(e))
    assert any(e.kind == "cancelled" for e in events)
    assert not (tmp_path / "job" / "analysis.json").exists()


def test_analysis_stub_error_goes_to_boundary(tmp_path, monkeypatch) -> None:
    import tools.gui.worker as worker
    import tools.gui.artifacts as artifacts
    def boom(payload):
        raise RuntimeError("stub failure")
    monkeypatch.setattr(artifacts, "analyze_artifact", boom)
    events = []
    with pytest.raises(RuntimeError, match="stub failure"):
        worker._dispatch(_analysis_req(tmp_path), _Never(), _Never(),
                         lambda e, terminal=False: events.append(e))


def test_empty_raw_import_fails(tmp_path) -> None:
    import tools.gui.worker as worker
    empty = tmp_path / "e.raw"
    empty.write_bytes(b"")
    events = []
    req = JobRequest(
        tag=JobTag(request_id="i", session_id="s", generation=0,
                   revision=0, source_fingerprint=""),
        operation="import", config_json="{}", source_path=str(empty),
        output_dir=str(tmp_path / "ij"))
    with pytest.raises(ValueError, match="empty"):
        worker._dispatch(req, _Never(), _Never(),
                         lambda e, terminal=False: events.append(e))
    assert not any(e.kind == "done" for e in events)
