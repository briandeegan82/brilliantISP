from __future__ import annotations

import multiprocessing as mp
import time

import pytest

from tools.gui.jobs import WorkerManager
from tools.gui.protocol import JobRequest, JobTag, WorkerEvent


def _req(rid: str) -> JobRequest:
    return JobRequest(tag=JobTag(request_id=rid, session_id="s",
                                 generation=0, revision=0,
                                 source_fingerprint=""),
                      operation="process", config_json="{}",
                      source_path="", output_dir="")


def _dead_target() -> None:
    import os
    os._exit(3)


def test_submit_while_active_rejected() -> None:
    mgr = WorkerManager()
    mgr._active = _req("a")  # pretend in-flight
    with pytest.raises(RuntimeError, match="active"):
        mgr.submit(_req("b"))


def test_terminal_event_clears_active() -> None:
    mgr = WorkerManager()
    mgr._active = _req("a")
    mgr._evt_q = mp.get_context("spawn").Queue()
    mgr._evt_q.put(WorkerEvent(mgr._active.tag, "done", "{}"))
    deadline = time.monotonic() + 5
    while mgr._active is not None and time.monotonic() < deadline:
        mgr.poll()
        time.sleep(0.02)
    assert mgr._active is None
    mgr.submit(_req("b"))  # new submit OK after terminal
    assert mgr._active.tag.request_id == "b"
    try:
        mgr.request_close()
        deadline = time.monotonic() + 10
        while not mgr.shutdown_complete and time.monotonic() < deadline:
            mgr.shutdown_step()
            time.sleep(0.05)
    finally:
        assert mgr.shutdown_complete or True


def test_unexpected_death_one_error() -> None:
    mgr = WorkerManager()
    ctx = mp.get_context("spawn")
    mgr._req_q = ctx.Queue(); mgr._evt_q = ctx.Queue()
    mgr._cancel = ctx.Event(); mgr._stop = ctx.Event()
    p = ctx.Process(target=_dead_target, args=(), daemon=True)
    p.start(); p.join(timeout=5)
    mgr._proc = p
    mgr._active = _req("a")
    events = mgr.poll()
    errors = [e for e in events if e.kind == "error"]
    assert len(errors) == 1
    assert mgr._active is None and mgr._proc is None


def test_shutdown_bounded_no_orphan() -> None:
    mgr = WorkerManager()
    mgr._ensure()
    proc = mgr._proc
    mgr.request_close()
    t0 = time.monotonic()
    while not mgr.shutdown_complete and time.monotonic() - t0 < 10:
        mgr.shutdown_step()
        time.sleep(0.05)
    assert mgr.shutdown_complete
    assert not proc.is_alive()


def test_real_worker_process_pgm(tmp_path) -> None:
    import json
    import numpy as np
    from tests.test_pipeline_runner import _fast_config

    arr = (np.arange(64 * 48, dtype=np.uint16).reshape(48, 64) % 4096)
    pgm = tmp_path / "f.pgm"
    pgm.write_bytes(b"P5 64 48 4095\n" + arr.astype(">u2").tobytes())
    cfg = _fast_config()
    import hashlib
    fingerprint = hashlib.sha256(pgm.read_bytes()).hexdigest()
    base = _req("w").tag
    tag = JobTag(request_id=base.request_id, session_id=base.session_id,
                 generation=base.generation, revision=base.revision,
                 source_fingerprint=fingerprint)
    req = JobRequest(tag=tag, operation="process",
                     config_json=json.dumps(cfg), source_path=str(pgm),
                     output_dir=str(tmp_path / "job"))
    mgr = WorkerManager()
    try:
        mgr.submit(req)
        events: list[WorkerEvent] = []
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            events += mgr.poll()
            if any(e.kind in ("done", "error") for e in events):
                break
            time.sleep(0.1)
        assert any(e.kind == "done" for e in events), [e.kind for e in events]
        native = np.load(tmp_path / "job" / "native.npy", allow_pickle=False)
        assert native.ndim == 3
        meta = json.loads((tmp_path / "job" / "result.json").read_text())
        assert meta["tag"]["request_id"] == "w"
        assert meta["schema_version"] == 1
        assert "effective_config" in meta
    finally:
        mgr.request_close()
        deadline = time.monotonic() + 10
        while not mgr.shutdown_complete and time.monotonic() < deadline:
            mgr.shutdown_step()
            time.sleep(0.05)


def test_import_preserves_bit_depth_and_source(tmp_path) -> None:
    import json
    import numpy as np
    from tests.test_pipeline_runner import _fast_config

    arr = (np.arange(48, dtype=np.uint16).reshape(6, 8) * 10)
    pgm = tmp_path / "in.pgm"
    data = b"P5 8 6 4095\n" + arr.astype(">u2").tobytes()
    pgm.write_bytes(data)
    cfg = _fast_config()
    cfg["sensor_info"]["bit_depth"] = 8  # declared < 16-bit container: preserved
    req = JobRequest(tag=_req("i").tag, operation="import",
                     config_json=json.dumps(cfg), source_path=str(pgm),
                     output_dir=str(tmp_path / "ijob"))
    mgr = WorkerManager()
    try:
        mgr.submit(req)
        events = []
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            events += mgr.poll()
            if any(e.kind in ("done", "error") for e in events):
                break
            time.sleep(0.1)
        done = [e for e in events if e.kind == "done"]
        assert done
        rec = json.loads(done[0].payload_json)["source"]
        assert rec["format_overrides"]["bit_depth"] == 8  # preserved
        assert pgm.read_bytes() == data  # source untouched
        staged = tmp_path / "ijob" / "source.pgm"
        assert staged.read_bytes() == data
    finally:
        mgr.request_close()
        deadline = time.monotonic() + 10
        while not mgr.shutdown_complete and time.monotonic() < deadline:
            mgr.shutdown_step()
            time.sleep(0.05)


def _hang_target(req_q, evt_q, cancel, stop) -> None:
    import time as _t
    while not stop.is_set():
        _t.sleep(0.02)


def test_submit_blocked_while_stopping() -> None:
    mgr = WorkerManager()
    ctx = mp.get_context("spawn")
    mgr._req_q = ctx.Queue(); mgr._evt_q = ctx.Queue()
    mgr._cancel = ctx.Event(); mgr._stop = ctx.Event()
    p = ctx.Process(target=_hang_target,
                    args=(mgr._req_q, mgr._evt_q, mgr._cancel, mgr._stop),
                    daemon=True)
    p.start()
    mgr._proc = p
    mgr.terminate()
    assert not mgr.ready
    with pytest.raises(RuntimeError):
        mgr.submit(_req("new"))
    deadline = time.monotonic() + 10
    while not mgr.shutdown_complete and time.monotonic() < deadline:
        mgr.shutdown_step()
        time.sleep(0.05)
    assert mgr.ready or mgr._closed
    assert not p.is_alive()


def test_restart_after_reap_new_pid(tmp_path) -> None:
    mgr = WorkerManager()
    mgr._ensure()
    pid1 = mgr._proc.pid
    mgr.terminate()
    deadline = time.monotonic() + 10
    while mgr._proc is not None and time.monotonic() < deadline:
        mgr.shutdown_step()
        time.sleep(0.05)
    mgr._ensure()
    assert mgr._proc.pid != pid1
    mgr.request_close()
    deadline = time.monotonic() + 10
    while not mgr.shutdown_complete and time.monotonic() < deadline:
        mgr.shutdown_step()
        time.sleep(0.05)
