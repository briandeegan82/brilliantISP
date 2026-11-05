from __future__ import annotations

import hashlib
import threading
from types import SimpleNamespace

import pytest

from tools.gui.protocol import JobRequest, JobTag
from tools.gui.worker import _run_process, _verified_source


def _request(tmp_path) -> JobRequest:
    source = tmp_path / "frame.raw"
    source.write_bytes(b"original")
    return JobRequest(
        JobTag("source-check", "session", 1, 0, hashlib.sha256(b"original").hexdigest()),
        "process", "{}", str(source), str(tmp_path / "job"),
    )


def test_worker_verifies_staged_bytes_not_only_recorded_identity(tmp_path) -> None:
    request = _request(tmp_path)
    signature = _verified_source(request)
    assert len(signature) == 5
    assert _verified_source(request) == signature
    (tmp_path / "frame.raw").write_bytes(b"modified")
    with pytest.raises(ValueError, match="Staged source changed"):
        _verified_source(request)


def test_process_rejects_source_change_before_artifact_publication(tmp_path, monkeypatch) -> None:
    request = _request(tmp_path)

    def process(*args, **kwargs):
        (tmp_path / "frame.raw").write_bytes(b"modified")
        return SimpleNamespace()

    monkeypatch.setattr("util.pipeline_runner.process_frame", process)
    with pytest.raises(ValueError, match="Staged source changed"):
        _run_process(request, {}, threading.Event(), lambda *args: None)
    assert not list((tmp_path / "job").glob("*.npy"))
    assert not (tmp_path / "job" / "result.json").exists()


def test_process_requires_a_content_fingerprint(tmp_path) -> None:
    request = _request(tmp_path)
    request = JobRequest(JobTag("r", "s", 1, 0, "unverified"), request.operation,
                         request.config_json, request.source_path, request.output_dir)
    with pytest.raises(ValueError, match="SHA-256 source fingerprint"):
        _verified_source(request)
