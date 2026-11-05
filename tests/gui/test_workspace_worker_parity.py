from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import time

import cv2
import numpy as np
import yaml

from tools.gui.jobs import WorkerManager
from tools.gui.protocol import JobRequest, JobTag, WorkerEvent
from util.pipeline_runner import process_frame
from util.pipeline_types import ProcessingOptions


def _terminal(manager: WorkerManager, request: JobRequest) -> WorkerEvent:
    manager.submit(request)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        for event in manager.poll():
            if event.tag == request.tag and event.kind in ("done", "error", "cancelled"):
                return event
        time.sleep(.01)
    raise AssertionError("worker did not return a terminal event")


def test_persistent_worker_matches_native_api_and_exports_frozen_spec(tmp_path, monkeypatch) -> None:
    from tests.test_pipeline_runner import _fast_config

    cfg = _fast_config()
    raw = (np.arange(64 * 48, dtype=np.uint16).reshape(48, 64) * 16 % 4096).astype(np.uint16)
    source_bytes = b"P5 64 48 4095\n" + raw.astype(">u2").tobytes()
    source = tmp_path / "first.pgm"
    source.write_bytes(source_bytes)
    fingerprint = hashlib.sha256(source_bytes).hexdigest()
    expected = process_frame(source, cfg, options=ProcessingOptions(capture_stages=("white_balance",)))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
        monkeypatch.setenv(name, "32")
    monkeypatch.setenv("MPLBACKEND", "TkAgg")
    manager = WorkerManager()

    def request(rid: str, operation: str, *, config=None, payload=None, output=None, path=None) -> JobRequest:
        return JobRequest(
            JobTag(rid, "session", 1, 1, fingerprint), operation,
            json.dumps(cfg if config is None else config),
            str(source if path is None else path), str(tmp_path / rid if output is None else output),
            capture_stages=("white_balance",), payload_json=json.dumps(payload or {}),
        )

    try:
        event = _terminal(manager, request("render", "process"))
        assert event.kind == "done", event.payload_json
        directory = tmp_path / "render"
        metadata = json.loads((directory / "result.json").read_text())
        np.testing.assert_array_equal(np.load(directory / "native.npy", allow_pickle=False), expected.native_image)
        np.testing.assert_array_equal(np.load(directory / "preview.npy", allow_pickle=False), expected.preview_rgb8)
        np.testing.assert_array_equal(
            np.load(directory / "capture_white_balance.npy", allow_pickle=False),
            expected.captures["white_balance"],
        )
        assert metadata["feedback"] == json.loads(json.dumps(expected.feedback))
        assert metadata["native_metadata"] == json.loads(json.dumps(asdict(expected.native_metadata)))
        runtime = metadata["runtime"]
        assert set(runtime["thread_limits"].values()) == {"1"}
        assert runtime["matplotlib_backend_environment"] == "Agg"
        assert runtime["opencv_threads"] == 1 and runtime["opencv_opencl"] is False
        original_pid = manager._proc.pid
        export = request("export", "export", config={"output": {"format": "tiff", "bit_depth": 32}},
                         payload={"manifest_dir": str(directory), "dest_dir": str(tmp_path / "exported"), "stem": "original"})
        event = _terminal(manager, export)
        assert event.kind == "done", event.payload_json
        assert manager._proc.pid == original_pid
        image = cv2.imread(str(tmp_path / "exported" / "original.png"), cv2.IMREAD_UNCHANGED)
        assert image is not None
        np.testing.assert_array_equal(image[..., ::-1], expected.native_image)
        exported_cfg = yaml.safe_load((tmp_path / "exported" / "original_effective.yaml").read_text())
        assert exported_cfg == metadata["effective_config"]
        assert json.loads((tmp_path / "exported" / "original.json").read_text()) == metadata["native_metadata"]
        event = _terminal(manager, request("import_one", "import"))
        assert event.kind == "done", event.payload_json
        one = json.loads(event.payload_json)["source"]
        event = _terminal(manager, request("import_two", "import"))
        assert event.kind == "done", event.payload_json
        two = json.loads(event.payload_json)["source"]
        assert one["path"] != two["path"]
        assert one["fingerprint"] == two["fingerprint"] == fingerprint
        assert manager._proc.pid == original_pid
        from pathlib import Path
        assert Path(one["path"]).read_bytes() == Path(two["path"]).read_bytes() == source_bytes
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}
        event = _terminal(manager, request("duplicate", "process", output=directory))
        assert event.kind == "error"
        assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()} == before
        assert source.read_bytes() == source_bytes
    finally:
        manager.request_close()
        deadline = time.monotonic() + 10
        while not manager.shutdown_complete and time.monotonic() < deadline:
            manager.shutdown_step()
            time.sleep(.01)
        assert manager.shutdown_complete
