from __future__ import annotations

import copy
import json
from typing import cast
from pathlib import Path

import pytest

from tools.gui.controller import WorkspaceController
from tools.gui.jobs import WorkerManager
from tools.gui.model import WorkspaceModel
from tools.gui.protocol import JobRequest, JobTag, WorkerEvent


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0
    def __call__(self) -> float:
        return self.t


class FakeManager(WorkerManager):
    def __init__(self) -> None:
        self.submitted: list[JobRequest] = []
        self.cancelled = 0
        self.events: list[WorkerEvent] = []
        self.terminated = 0
        self.closed = False
        self._closed = False
        self._stopping = False
        self._active = None
    def submit(self, req):
        if self._active is not None:
            raise RuntimeError("a job is already active")
        self.submitted.append(req)
        self._active = req
    def cancel(self): self.cancelled += 1
    def poll(self):
        ev, self.events = self.events, []
        return ev
    def terminate(self):
        self.terminated += 1
        self._active = None
    def finish_active(self):
        self._active = None
    def request_close(self): self.closed = True


def _make(tmp_path, **kw):
    from tests.test_pipeline_runner import _fast_config
    model = WorkspaceModel(_fast_config(), preset_path="p", session_id="s1")
    model.set_source({"path": str(tmp_path / "src.raw"), "fingerprint": "fp"})
    mgr = FakeManager()
    clock = FakeClock()
    return WorkspaceController(model, mgr, clock=clock,
                               root_dir=tmp_path / "sessions", **kw), model, mgr, clock


def _done(ctl):
    """Deliver a done event for the current active request."""
    req = ctl.active
    assert req is not None
    evt = WorkerEvent(req.tag, "done", '{"result_path": "r/result.json"}')
    ctl._handle(evt)
    cast(FakeManager, ctl.manager).finish_active()


def test_active_done_publishes_current_result(tmp_path) -> None:
    ctl, model, mgr, _ = _make(tmp_path)
    assert ctl.request_process()
    _done(ctl)
    assert ctl.current_result is not None
    assert ctl.current_result["result_path"] == "r/result.json"
    assert ctl.current_result["stale"] is False


def test_stale_result_after_edit_discarded(tmp_path) -> None:
    ctl, model, mgr, _ = _make(tmp_path)
    assert ctl.request_process()
    assert ctl.active is not None
    model.edit("crop.is_enable", True)
    _done(ctl)
    assert ctl.current_result is None
    assert ctl.status == "discarded"


def test_late_done_after_cancel_never_publishes(tmp_path) -> None:
    ctl, model, mgr, _ = _make(tmp_path)
    ctl.request_process()
    assert ctl.active is not None
    rid = ctl.active.tag.request_id
    ctl.cancel()
    evt = WorkerEvent(ctl.active.tag, "done", '{"result_path": "x"}')
    ctl._handle(evt)
    cast(FakeManager, ctl.manager).finish_active()
    assert ctl.current_result is None


def test_events_ignored_when_no_active(tmp_path) -> None:
    ctl, model, mgr, _ = _make(tmp_path)
    tag = JobTag(request_id="x", session_id="s1", generation=0,
                 revision=0, source_fingerprint="fp")
    ctl._handle(WorkerEvent(tag, "done", '{"result_path":"r"}'))
    assert ctl.current_result is None


def test_pending_manual_while_busy(tmp_path) -> None:
    ctl, model, mgr, _ = _make(tmp_path)
    ctl.request_process()
    assert ctl.request_process()  # manual queues, doesn't touch active
    assert len(mgr.submitted) == 1
    _done(ctl)
    ctl.tick()
    assert len(mgr.submitted) == 2


def test_import_cancels_and_waits(tmp_path) -> None:
    ctl, model, mgr, _ = _make(tmp_path)
    ctl.request_process()
    ctl.request_import(tmp_path / "n.raw")
    assert mgr.cancelled >= 1
    assert len(mgr.submitted) == 1  # still busy; import pending
    _done(ctl)  # active done (discarded)
    ctl.tick()
    assert mgr.submitted[-1].operation == "import"


def test_import_done_adopts_source(tmp_path) -> None:
    ctl, model, mgr, _ = _make(tmp_path)
    req = ctl.request_import(tmp_path / "n.pgm")
    assert req is not None
    record = {"path": "/j/source.pgm", "original_path": "n.pgm",
              "fingerprint": "h9", "size": 10,
              "format_overrides": {"width": 8}}
    gen0, rev0 = model.generation, model.revision
    ctl._handle(WorkerEvent(req.tag, "done",
                            json.dumps({"source": record})))
    assert model.source is not None
    assert model.source["fingerprint"] == "h9"
    assert model._cfg["sensor_info"]["width"] == 8
    assert model.generation == gen0 + 1 and model.revision > rev0


def test_failed_import_no_model_change(tmp_path) -> None:
    ctl, model, mgr, _ = _make(tmp_path)
    src0, cfg0 = model.source, model.config
    req = ctl.request_import(tmp_path / "n.pgm")
    assert req is not None
    ctl._handle(WorkerEvent(req.tag, "error", '{"error":"x"}'))
    assert model.source == src0 and model.config == cfg0


def test_cancel_stops_auto_and_pauses(tmp_path) -> None:
    ctl, model, mgr, _ = _make(tmp_path, auto_default=True)
    ctl.on_edit()
    ctl.cancel()
    ctl.tick()
    assert mgr.submitted == []
    assert ctl.batch_paused


def test_progress_only_from_current_context(tmp_path) -> None:
    ctl, model, mgr, _ = _make(tmp_path)
    ctl.request_process()
    assert ctl.active is not None
    ctl._handle(WorkerEvent(ctl.active.tag, "progress", '{"stage_id":"x"}'))
    assert len(ctl.progress) == 1
    model.edit("crop.is_enable", True)
    assert ctl.active is not None
    ctl._handle(WorkerEvent(ctl.active.tag, "progress", '{"stage_id":"y"}'))
    assert len(ctl.progress) == 1  # stale progress dropped


def test_batch_requires_start_and_runs_serially(tmp_path) -> None:
    ctl, model, mgr, _ = _make(tmp_path)
    b1 = ctl.enqueue_batch(label="b1")
    b2 = ctl.enqueue_batch(label="b2")
    ctl.tick()
    assert mgr.submitted == []  # paused initially
    ctl.start_batch()
    ctl.tick()
    assert len(mgr.submitted) == 1
    _done(ctl)
    ctl.tick()
    assert len(mgr.submitted) == 2  # second runs after first finishes


def test_batch_frozen_snapshot_and_retry(tmp_path) -> None:
    ctl, model, mgr, _ = _make(tmp_path)
    bid = ctl.enqueue_batch(label="b")
    frozen = copy.deepcopy(ctl.batch[0]["config"])
    model.edit("crop.is_enable", True)
    assert ctl.batch[0]["config"] == frozen  # immutable snapshot
    ctl.start_batch(); ctl.tick()
    req = ctl.active
    assert req is not None
    ctl._handle(WorkerEvent(req.tag, "error", '{"error":"x"}'))
    item = ctl.batch[0]
    assert item["status"] == "error"
    assert ctl.retry_batch(item["batch_id"])
    assert item["request_id"] is None


def test_batch_bounded_unfinished(tmp_path) -> None:
    ctl, model, mgr, _ = _make(tmp_path)
    ids = [ctl.enqueue_batch() for _ in range(20)]
    assert sum(1 for i in ctl.batch if i["status"] == "queued") == 16


def test_context_change_clears_batch_and_retires(tmp_path) -> None:
    ctl, model, mgr, _ = _make(tmp_path)
    ctl.enqueue_batch(); ctl.start_batch(); ctl.tick()
    assert ctl.active is not None
    ctl.context_changed()
    assert ctl.batch == []
    # retiring active can't publish
    assert ctl.active is not None
    evt = WorkerEvent(ctl.active.tag, "done", '{"result_path":"x"}')
    ctl._handle(evt)
    assert ctl.current_result is None


def test_no_feedback_merge(tmp_path) -> None:
    ctl, model, mgr, _ = _make(tmp_path)
    before = model.config
    assert ctl.request_process() and ctl.active is not None
    req = ctl.active
    ctl._handle(WorkerEvent(req.tag, "done",
                            json.dumps({"result_path": "x",
                                        "feedback": {"dg": 9}})))
    assert model.config == before


def test_export_uses_manifest_not_live_cfg(tmp_path) -> None:
    ctl, model, mgr, _ = _make(tmp_path)
    req = ctl.request_export("jobdir", "out", "stem")
    assert req is not None
    assert req.operation == "export"
    assert req.config_json == "{}"
    payload = json.loads(req.payload_json)
    assert payload["manifest_dir"] == "jobdir"


def test_export_while_cfg_invalid_submits_empty(tmp_path) -> None:
    ctl, model, mgr, _ = _make(tmp_path)
    model._cfg["sensor_info"]["width"] = float("nan")
    model._cfg["platform"] = "bad"
    req = ctl.request_export("jobdir", "out", "stem")
    assert req is not None and req.config_json == "{}"
    assert mgr.submitted[-1] is req


def test_auto_pending_invalid_cfg_no_crash(tmp_path) -> None:
    ctl, model, mgr, clock = _make(tmp_path, auto_default=True)
    ctl.on_edit()
    model._cfg["sensor_info"]["width"] = float("nan")
    clock.t = 10.0
    ctl.tick()
    assert ctl.status == "validation_blocked"
    assert mgr.submitted == []


def test_handle_requires_full_tag_match(tmp_path) -> None:
    ctl, model, mgr, _ = _make(tmp_path)
    ctl.request_process()
    assert ctl.active is not None
    wrong = JobTag(request_id=ctl.active.tag.request_id,
                   session_id="s1", generation=99, revision=0,
                   source_fingerprint="fp")
    ctl._handle(WorkerEvent(wrong, "done", '{"result_path":"x"}'))
    assert ctl.active is not None  # not cleared by wrong-generation tag


def test_unknown_event_kind_does_not_clear(tmp_path) -> None:
    ctl, model, mgr, _ = _make(tmp_path)
    ctl.request_process()
    assert ctl.active is not None
    ctl._handle(WorkerEvent(ctl.active.tag, "weird", "{}"))
    assert ctl.active is not None


def test_batch_cancel_racing_done_not_success(tmp_path) -> None:
    ctl, model, mgr, _ = _make(tmp_path)
    ctl.enqueue_batch(label="b")
    ctl.start_batch(); ctl.tick()
    assert ctl.active_role == "batch"
    ctl.cancel()
    req = ctl.active
    assert req is not None
    ctl._handle(WorkerEvent(req.tag, "done", '{"result_path":"x"}'))
    item = ctl.batch[0]
    assert item["status"] != "done"


def test_retry_rejects_over_capacity(tmp_path) -> None:
    ctl, model, mgr, _ = _make(tmp_path)
    for _ in range(16):
        ctl.enqueue_batch()
    done_item = ctl.batch[0]
    done_item["status"] = "done"
    for i in ctl.batch[1:]:
        i["status"] = "queued"
    assert ctl.retry_batch(done_item["batch_id"])


def test_batch_invalid_snapshot_rejected_no_mutation(tmp_path) -> None:
    ctl, model, mgr, _ = _make(tmp_path)
    bad = model.config
    bad["sensor_info"]["width"] = 1.5
    live_before = model.config
    assert ctl.enqueue_batch(config_snapshot=bad) is None
    assert model.config == live_before


def test_restore_monotonic_counters(tmp_path) -> None:
    m = model = WorkspaceModel(__import__('tests.test_pipeline_runner',
        fromlist=['_fast_config'])._fast_config())
    m.edit("crop.is_enable", True)
    r, g = m.revision, m.generation
    m.restore_saved_state(
        config=m.config, preset_snapshot=m.preset_snapshot,
        preset_path=None, saved_snapshot=m.saved_snapshot,
        source=m.source, saved_source=m.saved_source,
        generation=0, revision=0)
    assert m.generation > g and m.revision > r
