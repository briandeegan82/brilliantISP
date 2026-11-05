from __future__ import annotations

import argparse
from contextlib import contextmanager
import dataclasses
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from tools.gpu_validation.contracts import (
    CODE_FLOAT_STAGES, FINAL_FLOAT, FINAL_INTEGER, STATISTICS, stage_gate,
)
from tools.gpu_validation.metrics import compare_arrays
from tools.gpu_validation.pca import load_array
from tools.gpu_validation.reference import (
    file_identity, gpu_and_harness_sources, jsonable, save_array, save_source_files,
    source_hashes, write_json,
)


def _record(value: Any, directory: Path, prefix: str) -> dict[str, Any]:
    if isinstance(value, np.ndarray):
        return {"array": save_array(directory / f"{prefix}.npy", value)}
    if isinstance(value, tuple):
        return {"members": [_record(v, directory, f"{prefix}_{i}") for i, v in enumerate(value)]}
    return {"value": jsonable(value)}


@contextmanager
def observe_gpu_stages(output: Path) -> Iterator[list[dict[str, Any]]]:
    from brilliantisp_gpu.engine import GpuEngine

    original = GpuEngine._run_stage
    records: list[dict[str, Any]] = []
    occurrences: dict[tuple[str, int, int], int] = {}

    def observed(engine: Any, stage_id: str, thunk: Any, ae_pass: int = 0) -> Any:
        value = original(engine, stage_id, thunk, ae_pass=ae_pass)
        key = (stage_id, engine.pass_index, ae_pass)
        occurrence = occurrences.get(key, 0)
        occurrences[key] = occurrence + 1
        host_value = engine._host_result(value, frame_index=0)
        record = {
            "stage_id": stage_id, "pass_index": engine.pass_index,
            "ae_pass": ae_pass, "occurrence": occurrence,
            **_record(host_value, output, f"{len(records):04d}_{stage_id}"),
        }
        records.append(record)
        return value

    GpuEngine._run_stage = observed
    try:
        yield records
    finally:
        GpuEngine._run_stage = original


def _array_comparison(expected: np.ndarray, actual: np.ndarray, gate, *, scale=1.0, data_range=None):
    try:
        return compare_arrays(expected, actual, gate, scale=scale, data_range=data_range)
    except ValueError as exc:
        return {"passed": False, "error": str(exc), "gate": gate.name}


def _compare_record(expected, actual, reference_dir, output_dir, stage, code_max):
    if "array" in expected and "array" in actual:
        left = load_array(reference_dir, expected["array"])
        right = load_array(output_dir, actual["array"])
        gate = stage_gate(stage, logical_integer=left.dtype.kind in "ui")
        scale = code_max if stage in CODE_FLOAT_STAGES else 1.0
        return _array_comparison(left, right, gate, scale=scale)
    if "members" in expected and "members" in actual:
        if len(expected["members"]) != len(actual["members"]):
            return {"passed": False, "error": "stage tuple length mismatch"}
        members = [
            _compare_record(e, a, reference_dir, output_dir, stage, code_max)
            for e, a in zip(expected["members"], actual["members"])
        ]
        return {"passed": all(m["passed"] for m in members), "members": members}
    if "value" in expected and "value" in actual:
        left, right = expected["value"], actual["value"]
        if stage == "auto_white_balance" and isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return _array_comparison(np.array([left], np.float64), np.array([right], np.float64), STATISTICS)
        return {"passed": type(left) is type(right) and left == right,
                "reference": left, "actual": right}
    return {"passed": False, "error": "stage result structure mismatch"}


def validate_reference(reference: Path, output: Path, device: str) -> dict[str, Any]:
    from brilliantisp_gpu.api import process_frame_gpu
    from brilliantisp_gpu.device import device_properties, resolve_device
    from brilliantisp_gpu.types import GpuOptions
    from util.pipeline_types import ProcessingOptions

    manifest_path = reference / "manifest.json"
    frozen = json.loads(manifest_path.read_text())
    if frozen["source_hashes"] != source_hashes():
        raise RuntimeError("CPU source identity differs from frozen reference")
    if frozen["source"] != file_identity(Path(frozen["source"]["path"])):
        raise RuntimeError("RAW input identity differs from frozen reference")
    implementations = gpu_and_harness_sources()
    requested = sorted({
        s["stage_id"] for s in frozen["stages"]
        if "array" in s or any("array" in part for part in s.get("members", []))
    })
    options = ProcessingOptions(run_id="cpu_reference", capture_stages=tuple(requested))
    output.mkdir(parents=True, exist_ok=False)
    with observe_gpu_stages(output) as actual_stages:
        result = process_frame_gpu(
            Path(frozen["source"]["path"]), frozen["source_config"],
            options=options, gpu=GpuOptions(device=device),
        )
    if result.artifacts:
        raise AssertionError("compute-only GPU API wrote pipeline artifacts")
    native = load_array(reference, frozen["native"])
    preview = load_array(reference, frozen["preview"])
    native_range = float(2**frozen["metadata"]["code_bit_depth"] - 1) if native.dtype.kind == "u" else 1.0
    comparisons = {
        "native": _array_comparison(native, result.native_image, FINAL_INTEGER if native.dtype.kind == "u" else FINAL_FLOAT,
                                    data_range=native_range),
        "preview": _array_comparison(preview, result.preview_rgb8, FINAL_INTEGER, data_range=255.0),
        "metadata": {"passed": frozen["metadata"] == jsonable(result.native_metadata)},
        "effective_config": {"passed": frozen["effective_config"] == jsonable(result.effective_config)},
        "migration_report": {"passed": frozen["migration_report"] == jsonable(result.migration_report)},
        "warnings": {"passed": frozen["warnings"] == jsonable(result.warnings)},
    }
    key = lambda s: (s["stage_id"], s["pass_index"], s["ae_pass"], s["occurrence"])
    expected_keys, actual_keys = [key(s) for s in frozen["stages"]], [key(s) for s in actual_stages]
    comparisons["stage_order"] = {"passed": expected_keys == actual_keys,
                                   "reference": expected_keys, "actual": actual_keys}
    stage_results = []
    for expected, actual in zip(frozen["stages"], actual_stages):
        if key(expected) == key(actual):
            stage_results.append({
                "key": key(expected),
                **_compare_record(expected, actual, reference, output, expected["stage_id"],
                                  frozen["feedback"]["linear_reference_code_max"]),
            })
    comparisons["stages"] = {
        "passed": expected_keys == actual_keys and all(s["passed"] for s in stage_results),
        "results": stage_results,
    }
    actual_feedback = jsonable(result.feedback)
    feedback = {}
    for name in frozen["feedback"].keys() | actual_feedback.keys():
        if name not in frozen["feedback"] or name not in actual_feedback:
            feedback[name] = {"passed": False, "error": "missing feedback field"}
            continue
        left, right = frozen["feedback"][name], actual_feedback[name]
        if name in ("awb_gains", "ae_meter_average") and left is not None and right is not None:
            feedback[name] = _array_comparison(np.atleast_1d(left).astype(np.float64),
                                              np.atleast_1d(right).astype(np.float64), STATISTICS)
        else:
            feedback[name] = {"passed": type(left) is type(right) and left == right,
                              "reference": left, "actual": right}
    comparisons["feedback"] = {"passed": all(v["passed"] for v in feedback.values()), "fields": feedback}
    if implementations != gpu_and_harness_sources() or frozen["source_hashes"] != source_hashes():
        raise RuntimeError("implementation changed during validation")
    saved = {
        "native": save_array(output / "native.npy", result.native_image),
        "preview": save_array(output / "preview.npy", result.preview_rgb8),
    }
    save_source_files(output / "source", implementations)
    record = {
        "schema_version": 1, "passed": all(v["passed"] for v in comparisons.values()),
        "source_reference": file_identity(manifest_path),
        "implementation_hashes": implementations,
        "gpu": device_properties(resolve_device(device)),
        "comparisons": comparisons, "stages": actual_stages, "arrays": saved,
        "metadata": jsonable(result.native_metadata), "feedback": actual_feedback,
        "effective_config": result.effective_config, "warnings": result.warnings,
        "migration_report": result.migration_report,
        "timings": [dataclasses.asdict(event) for event in result.timings],
    }
    write_json(output / "report.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = validate_reference(args.reference, args.output, args.device)
    print(json.dumps({"passed": result["passed"], "report": str(args.output / "report.json")}))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
