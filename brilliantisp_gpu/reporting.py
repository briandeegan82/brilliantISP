from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .grouping import _value_key


STATUSES = ("success", "failed", "cancelled", "blocked")


def _json_value(value, *, configuration=False):
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _json_value(getattr(value, f.name), configuration=configuration)
                for f in dataclasses.fields(value)}
    if isinstance(value, np.ndarray):
        if not configuration:
            raise TypeError("pixel arrays must not be embedded in batch reports")
        return _json_value(value.tolist(), configuration=True)
    if isinstance(value, np.generic):
        return _json_value(value.item(), configuration=configuration)
    if isinstance(value, dict):
        return {str(k): _json_value(v, configuration=configuration) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v, configuration=configuration) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": value.hex()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported report value: {type(value).__name__}")


def outcome_record(item, outcome, *, index):
    if outcome.status not in STATUSES:
        raise ValueError(f"unknown outcome status: {outcome.status}")
    source = (
        {"kind": "array", "shape": list(item.source.shape), "dtype": item.source.dtype.str}
        if isinstance(item.source, np.ndarray)
        else {"kind": "path", "path": str(item.source)}
    )
    config_spec = (
        {"kind": "mapping"} if isinstance(item.config, dict)
        else _json_value(item.config)
    )
    record = {
        "schema_version": 1, "index": index, "item_id": outcome.item_id,
        "sequence_id": item.sequence_id, "frame_index": item.frame_index,
        "source": source, "config_spec": config_spec, "status": outcome.status,
        "error": _json_value(outcome.error), "device": _json_value(outcome.device),
        "batch_index": outcome.batch_index, "batch_size": outcome.batch_size,
        "attempts": _json_value(outcome.attempts), "metrics": _json_value(outcome.metrics),
        "metadata": None, "feedback": None, "warnings": [], "migration_report": {},
        "effective_config": None, "effective_config_sha256": None, "artifacts": [],
    }
    if outcome.result is not None:
        result = outcome.result
        record.update(
            metadata=_json_value(result.native_metadata),
            feedback=_json_value(result.feedback),
            warnings=_json_value(result.warnings),
            migration_report=_json_value(result.migration_report),
            effective_config=_json_value(result.effective_config, configuration=True),
            artifacts=_json_value(result.artifacts),
        )
        try:
            canonical = json.dumps(_value_key(result.effective_config), separators=(",", ":"))
        except TypeError:
            record["warnings"].append("effective configuration identity is unavailable")
        else:
            record["effective_config_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return record


def summarize_records(records, *, wall_ns, interrupted=False):
    if isinstance(wall_ns, bool) or not isinstance(wall_ns, int) or wall_ns < 0:
        raise ValueError("wall_ns must be a nonnegative integer")
    counts = dict.fromkeys(STATUSES, 0)
    attempts, split_ids, artifacts, seen = {}, set(), set(), set()
    success_images_by_size = {}
    for record in records:
        item_id, status = record["item_id"], record["status"]
        if item_id in seen:
            raise ValueError(f"duplicate reported item: {item_id}")
        if status not in counts:
            raise ValueError(f"unknown outcome status: {status}")
        seen.add(item_id)
        counts[status] += 1
        artifacts.update(record["artifacts"])
        if status == "success":
            size = str(record["batch_size"])
            success_images_by_size[size] = success_images_by_size.get(size, 0) + 1
        chain = record["attempts"]
        for i, attempt in enumerate(chain):
            aid = attempt["attempt_id"]
            members = set(attempt["item_ids"])
            if item_id not in members or len(members) != attempt["batch_size"]:
                raise ValueError(f"inconsistent attempt members: {aid}")
            duration = attempt["compute_transfer_ns"]
            if isinstance(duration, bool) or not isinstance(duration, int) or duration < 0:
                raise ValueError(f"invalid attempt duration: {aid}")
            if attempt["status"] not in ("success", "failed", "cancelled"):
                raise ValueError(f"unknown attempt status: {aid}")
            if aid in attempts and attempts[aid] != attempt:
                raise ValueError(f"inconsistent shared attempt: {aid}")
            attempts[aid] = attempt
            error = attempt["error"]
            if attempt["status"] == "failed" and error and error["kind"] == "gpu_oom":
                if any(set(child["item_ids"]) < members for child in chain[i + 1:]):
                    split_ids.add(aid)
    successful_batches_by_size = {}
    for attempt in attempts.values():
        if attempt["status"] == "success":
            size = str(attempt["batch_size"])
            successful_batches_by_size[size] = successful_batches_by_size.get(size, 0) + 1
    return {
        "schema_version": 1, "items": len(seen), "status_counts": counts,
        "attempts": len(attempts), "oom_splits": len(split_ids),
        "successful_batches_by_size": successful_batches_by_size,
        "successful_images_by_batch_size": success_images_by_size,
        "compute_transfer_attempt_ns": sum(a["compute_transfer_ns"] for a in attempts.values()),
        "artifacts": len(artifacts), "wall_ns": wall_ns, "interrupted": bool(interrupted),
        "exit_code": 130 if interrupted else 1 if any(counts[s] for s in STATUSES[1:]) else 0,
    }
