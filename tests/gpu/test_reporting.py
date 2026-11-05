from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from brilliantisp_gpu.reporting import outcome_record, summarize_records
from util.pipeline_types import FrameMetadata


def _attempt(aid, ids, status="success", duration=10, kind=None):
    return dict(
        attempt_id=aid, item_ids=list(ids), batch_size=len(ids), status=status,
        compute_transfer_ns=duration,
        error=None if kind is None else dict(kind=kind, exception_type="Error", message="injected", stage_id=None),
    )


def _record(item_id, attempts, status="success", size=1):
    return dict(
        item_id=item_id, status=status, attempts=attempts, batch_size=size,
        artifacts=[f"{item_id}.tiff"] if status == "success" else [],
    )


def test_reporting_shared_batch_duration_is_not_counted_per_image():
    attempt = _attempt("cohort", ("a", "b", "c", "d"), duration=200)
    records = [_record(i, [attempt], size=4) for i in ("a", "b", "c", "d")]
    summary = summarize_records(records, wall_ns=1000)
    assert summary["items"] == 4
    assert summary["status_counts"] == dict(success=4, failed=0, cancelled=0, blocked=0)
    assert summary["attempts"] == 1
    assert summary["compute_transfer_attempt_ns"] == 200
    assert summary["successful_batches_by_size"] == {"4": 1}
    assert summary["successful_images_by_batch_size"] == {"4": 4}
    assert summary["artifacts"] == 4
    assert summary["exit_code"] == 0


def test_reporting_recursive_oom_splits_and_no_retry_are_distinct():
    root = _attempt("root", ("a", "b", "c", "d"), "failed", 100, "gpu_oom")
    left = _attempt("left", ("a", "b"), "failed", 50, "gpu_oom")
    right = _attempt("right", ("c", "d"), "failed", 51, "gpu_oom")
    records = [
        _record(i, [root, left if i in ("a", "b") else right,
                    _attempt(f"leaf_{i}", (i,), duration=10 + n)])
        for n, i in enumerate(("a", "b", "c", "d"))
    ]
    summary = summarize_records(records, wall_ns=1000)
    assert summary["oom_splits"] == 3
    assert summary["attempts"] == 7
    assert summary["compute_transfer_attempt_ns"] == 247
    assert summary["successful_batches_by_size"] == {"1": 4}
    summary = summarize_records([
        _record(i, [root], status="failed", size=4) for i in ("a", "b", "c", "d")
    ], wall_ns=1000)
    assert summary["attempts"] == 1
    assert summary["oom_splits"] == 0
    assert summary["compute_transfer_attempt_ns"] == 100
    assert summary["exit_code"] == 1


def test_reporting_status_and_interrupt_exit_codes():
    empty = summarize_records([], wall_ns=0)
    assert empty["exit_code"] == empty["items"] == empty["attempts"] == 0
    records = [_record("cancel", [], status="cancelled", size=0),
               _record("blocked", [], status="blocked", size=0)]
    summary = summarize_records(records, wall_ns=20)
    assert summary["status_counts"] == dict(success=0, failed=0, cancelled=1, blocked=1)
    assert summary["exit_code"] == 1
    assert summarize_records(records, wall_ns=20, interrupted=True)["exit_code"] == 130


def test_reporting_refuses_duplicate_items_or_inconsistent_attempts():
    attempt = _attempt("one", ("a", "b"))
    first, second = _record("a", [attempt]), _record("b", [copy.deepcopy(attempt)])
    with pytest.raises(ValueError, match="duplicate"):
        summarize_records([first, first], wall_ns=1)
    second["attempts"][0]["compute_transfer_ns"] = 50
    with pytest.raises(ValueError, match="inconsistent shared attempt"):
        summarize_records([first, second], wall_ns=1)
    for change in (dict(item_ids=["other"]), dict(compute_transfer_ns=-1)):
        bad = copy.deepcopy(first)
        bad["attempts"][0].update(change)
        with pytest.raises(ValueError):
            summarize_records([bad], wall_ns=1)


@dataclass(frozen=True)
class Attempt:
    attempt_id: str
    item_ids: tuple
    batch_size: int
    status: str = "success"
    error: object = None
    compute_transfer_ns: int = 1


def test_reporting_serializes_metadata_without_embedding_pixels():
    config = dict(number=np.float32(1), table=np.array([0, 1], dtype=np.float32))
    item = SimpleNamespace(
        source=np.ones((2, 2), dtype=np.uint16), config=config,
        sequence_id=None, frame_index=None,
    )
    result = SimpleNamespace(
        native_metadata=FrameMetadata(
            width=2, height=2, cfa_pattern=None, storage_bit_depth=8,
            code_bit_depth=8, signal_domain="display_rgb", transfer="srgb",
            source_identity="array:sample",
        ),
        feedback=dict(index=np.int64(7)), warnings=(), migration_report={},
        effective_config=config, artifacts=(Path("one.tiff"),),
        native_image=np.ones((2, 2, 3)), captures=dict(raw=np.ones((2, 2))),
    )
    outcome = SimpleNamespace(
        item_id="sample", status="success", error=None, result=result,
        device=dict(vendor="nvidia"), batch_index=0, batch_size=1,
        attempts=(Attempt("attempt", ("sample",), 1),), metrics=dict(decode_ns=1),
    )
    record = outcome_record(item, outcome, index=0)
    json.dumps(record, allow_nan=False)
    assert "native_image" not in record and "captures" not in record
    assert record["source"] == dict(kind="array", shape=[2, 2], dtype="<u2")
    assert record["effective_config"]["table"] == [0.0, 1.0]
    assert record["feedback"]["index"] == 7
    assert len(record["effective_config_sha256"]) == 64
    initial = record["effective_config_sha256"]
    config["number"] = np.float32(2)
    assert outcome_record(item, outcome, index=0)["effective_config_sha256"] != initial
    outcome.metrics["hidden_pixels"] = np.ones((2, 2))
    with pytest.raises(TypeError, match="pixel arrays"):
        outcome_record(item, outcome, index=0)
