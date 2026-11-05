from __future__ import annotations

import json
import time
from pathlib import Path
from typing import cast

import pytest

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


def test_pending_process_beats_render(tmp_path) -> None:
    ctl, _m = _make(tmp_path)
    ctl.request_process()
    ctl.request_render("a", "b", mode="wipe")
    assert len(cast(FakeManager, ctl.manager).submitted) == 1
    assert ctl.active is not None
    ctl._handle(WorkerEvent(ctl.active.tag, "done", "{}"))
    cast(FakeManager, ctl.manager).finish_active()
    ctl.tick()
    assert cast(FakeManager, ctl.manager).submitted[-1].operation == "render"


def test_render_beats_batch(tmp_path) -> None:
    ctl, _m = _make(tmp_path)
    assert ctl.enqueue_batch(label="b1") is not None
    ctl.start_batch()
    ctl.request_render("a", "b", mode="wipe")
    ctl.tick()
    assert cast(FakeManager, ctl.manager).submitted[-1].operation == "render"


def test_old_render_done_cannot_publish(tmp_path) -> None:
    ctl, _m = _make(tmp_path)
    r1 = ctl.request_render("a1", "b", mode="wipe")
    assert r1 is not None and ctl.active_role == "render"
    r2 = ctl.request_render("a2", "b", mode="difference")
    assert r2 is not None
    assert cast(FakeManager, ctl.manager).cancelled >= 1
    ctl._handle(WorkerEvent(
        r1.tag, "done", json.dumps({"render_path": "old"})))
    assert ctl.render_result is None
    cast(FakeManager, ctl.manager).finish_active()
    ctl.tick()
    assert ctl.active is not None
    assert ctl.active.tag.request_id == r2.tag.request_id
    ctl._handle(WorkerEvent(
        r2.tag, "done",
        json.dumps({"render_path": "new",
                    "descriptor": {"mode": "difference"}})))
    assert ctl.render_result is not None
    assert ctl.render_result["request_id"] == r2.tag.request_id


def test_render_allowed_with_invalid_cfg(tmp_path) -> None:
    ctl, model = _make(tmp_path)
    model._cfg["sensor_info"]["width"] = float("nan")
    req = ctl.request_render("a", "b", mode="wipe")
    assert req is not None and req.config_json == "{}"


def test_render_does_not_mutate_model_or_result(tmp_path) -> None:
    ctl, model = _make(tmp_path)
    cfg0 = model.config
    req = ctl.request_render("a", "b", mode="wipe")
    assert req is not None and ctl.active is not None
    ctl._handle(WorkerEvent(
        req.tag, "done",
        json.dumps({"render_path": "r",
                    "descriptor": {"mode": "wipe"}})))
    assert model.config == cfg0
    assert ctl.current_result is None


def test_context_cancel_stop_close_clear_render(tmp_path) -> None:
    ctl, _m = _make(tmp_path)
    req = ctl.request_render("a", "b", mode="wipe")
    assert req is not None
    ctl._handle(WorkerEvent(
        req.tag, "done",
        json.dumps({"render_path": "r", "descriptor": {}})))
    cast(FakeManager, ctl.manager).finish_active()
    ctl.context_changed()
    assert ctl.render_result is None
    assert ctl.latest_render_id is None


def test_render_closed_returns_none(tmp_path) -> None:
    ctl, _m = _make(tmp_path)
    ctl.close()
    assert ctl.request_render("a", "b", mode="wipe") is None


def _render_req(tmp_path, payload="{}"):
    return JobRequest(
        tag=JobTag(request_id="r", session_id="s", generation=0,
                   revision=0, source_fingerprint=""),
        operation="render", config_json="{}", source_path="",
        output_dir=str(tmp_path / "job"), payload_json=payload)


class _Never:
    def is_set(self): return False


class _Always:
    def is_set(self): return True


def test_render_dispatch_exact_payload(tmp_path, monkeypatch) -> None:
    import tools.gui.worker as worker
    import tools.gui.artifacts as artifacts
    seen = {}
    def fake(payload, out_dir):
        seen["payload"] = payload
        seen["out_dir"] = out_dir
        return Path(out_dir) / "wipe.png"
    monkeypatch.setattr(artifacts, "render_comparison", fake)
    events = []
    payload = json.dumps({"a_dir": "A", "b_dir": "B", "mode": "wipe",
                          "fraction": 0.5, "difference_gain": 1.0})
    req = JobRequest(
        tag=JobTag(request_id="r", session_id="s", generation=0,
                   revision=0, source_fingerprint=""),
        operation="render", config_json="{}", source_path="",
        output_dir=str(tmp_path / "job"), payload_json=payload)
    worker._dispatch(req, _Never(), _Never(),
                     lambda e, terminal=False: events.append(e))
    assert seen["payload"] == json.loads(payload)
    assert seen["out_dir"] == str(tmp_path / "job")
    done = [e for e in events if e.kind == "done"]
    assert done
    data = json.loads(done[0].payload_json)
    assert data["render_path"].endswith("wipe.png")
    assert data["descriptor"]["mode"] == "wipe"


def test_render_cancel_maps_cancelled(tmp_path, monkeypatch) -> None:
    import tools.gui.worker as worker
    import tools.gui.artifacts as artifacts
    monkeypatch.setattr(artifacts, "render_comparison",
                        lambda payload, out_dir: Path(out_dir) / "x.png")
    events = []
    worker._dispatch(_render_req(tmp_path), _Always(), _Always(),
                     lambda e, terminal=False: events.append(e))
    assert any(e.kind == "cancelled" for e in events)


def test_render_stub_error_propagates(tmp_path, monkeypatch) -> None:
    import tools.gui.worker as worker
    import tools.gui.artifacts as artifacts
    def boom(payload, out_dir):
        raise RuntimeError("stub render failure")
    monkeypatch.setattr(artifacts, "render_comparison", boom)
    events = []
    with pytest.raises(RuntimeError, match="stub render failure"):
        worker._dispatch(_render_req(tmp_path), _Never(), _Never(),
                         lambda e, terminal=False: events.append(e))


def test_render_stub_creates_own_dir(tmp_path, monkeypatch) -> None:
    import tools.gui.worker as worker
    import tools.gui.artifacts as artifacts
    def fake(payload, out_dir):
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=False)
        (p / "render.json").write_text("{}")
        return p / "render.json"
    monkeypatch.setattr(artifacts, "render_comparison", fake)
    events = []
    req = JobRequest(
        tag=JobTag(request_id="r", session_id="s", generation=0,
                   revision=0, source_fingerprint=""),
        operation="render", config_json="{}", source_path="",
        output_dir=str(tmp_path / "job"), payload_json="{}")
    worker._dispatch(req, _Never(), _Never(),
                     lambda e, terminal=False: events.append(e))
    assert any(e.kind == "done" for e in events)
    assert (tmp_path / "job" / "render.json").is_file()


@pytest.mark.parametrize("action", ["cancel", "stop_worker", "close",
                                    "context_changed"])
def test_render_cleared_on_all_transitions(tmp_path, action) -> None:
    ctl, _m = _make(tmp_path)
    req = ctl.request_render("a", "b", mode="wipe")
    assert req is not None
    getattr(ctl, action)()
    assert ctl._pending_render is None
    assert ctl.latest_render_id is None
    assert ctl.render_result is None


def test_priority_process_then_render_then_aux_then_batch(tmp_path) -> None:
    ctl, _m = _make(tmp_path)
    assert ctl.enqueue_batch(label="b1") is not None
    ctl.start_batch()
    ctl._pending = {"role": "process", "manual": True}
    ctl._pending_at = 0.0
    ctl.request_render("a", "b", mode="wipe")
    ctl.request_analysis("m")
    ctl.tick()
    order = [r.operation
             for r in cast(FakeManager, ctl.manager).submitted]
    assert order == ["process"]
    assert ctl.active is not None
    done = WorkerEvent(ctl.active.tag, "done", "{}")
    ctl._handle(done)
    cast(FakeManager, ctl.manager).finish_active()
    ctl.tick()
    assert ctl.active is not None
    assert ctl.active.operation == "render"
    assert ctl.active is not None
    ctl._handle(WorkerEvent(ctl.active.tag, "done", "{}"))
    cast(FakeManager, ctl.manager).finish_active()
    ctl.tick()
    assert ctl.active is not None
    assert ctl.active.operation == "analysis"
    assert ctl.active is not None
    ctl._handle(WorkerEvent(ctl.active.tag, "done", "{}"))
    cast(FakeManager, ctl.manager).finish_active()
    ctl.tick()
    assert ctl.active is not None
    assert ctl.active.operation == "process"
