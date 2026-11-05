from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import uuid

import numpy as np

from brilliant_isp import BrilliantISP
from brilliantisp_gpu.api import process_frame_gpu
from brilliantisp_gpu.engine import GpuEngine
from brilliantisp_gpu.types import GpuOptions
from tools.gpu_validation.contracts import (
    CODE_FLOAT_STAGES, FINAL_FLOAT, FINAL_INTEGER, STATISTICS, stage_gate,
)
from tools.gpu_validation.metrics import compare_arrays
from tools.gpu_validation.reference import (
    environment, file_identity, gpu_and_harness_sources, jsonable, save_array,
    save_source_files, source_hashes, write_json,
)
from tools.gpu_validation.workloads import compare_feedback
from util.pipeline_runner import process_frame
from util.pipeline_types import ProcessingOptions


def _array_identity(value):
    data = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": list(data.shape), "dtype": data.dtype.str}).encode())
    digest.update(memoryview(data).cast("B"))
    return {"shape": list(data.shape), "dtype": data.dtype.str,
            "content_sha256": digest.hexdigest()}


def _value_identity(value):
    if isinstance(value, np.ndarray):
        return {"array": _array_identity(value)}
    if isinstance(value, tuple):
        return {"members": [_value_identity(part) for part in value]}
    return {"value": jsonable(value)}


class MemoryReference:
    def __init__(self):
        self.records = []
        self.blobs = {}

    def capture(self, value):
        if isinstance(value, np.ndarray):
            descriptor = _array_identity(value)
            key = descriptor["content_sha256"]
            if key not in self.blobs:
                stream = io.BytesIO()
                np.savez_compressed(stream, pixels=np.ascontiguousarray(value))
                self.blobs[key] = stream.getvalue()
            return {"array": descriptor}
        if isinstance(value, tuple):
            return {"members": [self.capture(part) for part in value]}
        return {"value": jsonable(value)}

    def read(self, descriptor):
        with np.load(io.BytesIO(self.blobs[descriptor["content_sha256"]])) as archive:
            data = archive["pixels"]
        if _array_identity(data) != descriptor:
            raise RuntimeError("immutable CPU stage capture identity changed")
        return data

    def retain_failure(self, root):
        root.mkdir(parents=True, exist_ok=False)
        for key, contents in self.blobs.items():
            with (root / f"{key}.npz").open("xb") as stream:
                stream.write(contents)
        write_json(root / "stages.json", self.records)


def _compare_value(reference, actual, memory, stage, code_max):
    if "array" in reference and isinstance(actual, np.ndarray):
        expected = memory.read(reference["array"])
        gate = stage_gate(stage, logical_integer=expected.dtype.kind in "ui")
        scale = code_max if stage in CODE_FLOAT_STAGES else 1.0
        try:
            return compare_arrays(expected, actual, gate, scale=scale)
        except ValueError as exc:
            return {"passed": False, "gate": gate.name, "error": str(exc)}
    if "members" in reference and isinstance(actual, tuple):
        if len(reference["members"]) != len(actual):
            return {"passed": False, "error": "stage tuple length mismatch"}
        members = [
            _compare_value(left, right, memory, stage, code_max)
            for left, right in zip(reference["members"], actual)
        ]
        return {"passed": all(part["passed"] for part in members), "members": members}
    if "value" in reference:
        left, right = reference["value"], jsonable(actual)
        if stage == "auto_white_balance" and isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return compare_arrays(np.array([left], np.float64),
                                  np.array([right], np.float64), STATISTICS)
        return {"passed": type(left) is type(right) and left == right,
                "reference": left, "actual": right}
    return {"passed": False, "error": "stage result structure mismatch"}


def _stage_key(stage, engine, ae_pass, occurrences):
    key = (stage, engine.pass_index, ae_pass)
    occurrence = occurrences.get(key, 0)
    occurrences[key] = occurrence + 1
    return [stage, engine.pass_index, ae_pass, occurrence]


@contextmanager
def observe_cpu(memory):
    original = BrilliantISP._run_stage
    occurrences = {}

    def observed(engine, stage, thunk, ae_pass=0):
        value = original(engine, stage, thunk, ae_pass=ae_pass)
        memory.records.append({
            "key": _stage_key(stage, engine, ae_pass, occurrences),
            "value": memory.capture(value),
        })
        return value

    BrilliantISP._run_stage = observed
    try:
        yield
    finally:
        BrilliantISP._run_stage = original


@contextmanager
def observe_gpu(memory, code_max, results):
    original = GpuEngine._run_stage
    occurrences = {}

    def observed(engine, stage, thunk, ae_pass=0):
        value = original(engine, stage, thunk, ae_pass=ae_pass)
        host = engine._host_result(value, frame_index=0)
        key = _stage_key(stage, engine, ae_pass, occurrences)
        index = len(results)
        if index >= len(memory.records) or memory.records[index]["key"] != key:
            comparison = {"passed": False, "error": "stage ordering mismatch"}
        else:
            comparison = _compare_value(
                memory.records[index]["value"], host, memory, stage, code_max,
            )
        results.append({"key": key, "candidate": _value_identity(host), **comparison})
        return value

    GpuEngine._run_stage = observed
    try:
        yield
    finally:
        GpuEngine._run_stage = original


def qualify_case(case, directory: Path, device: str):
    directory.mkdir(parents=True, exist_ok=False)
    source, config = Path(case["source"]), case["config"]
    source_identity = file_identity(source)
    for identity in case.get("input_identities", ()):
        if file_identity(Path(identity["path"])) != identity:
            raise RuntimeError(f"declared qualification input changed: {identity['path']}")
    cpu_hashes, implementation = source_hashes(), gpu_and_harness_sources()
    config_before = json.dumps(jsonable(config), sort_keys=True, allow_nan=False)
    options = ProcessingOptions(run_id="qualified_reference")
    memory = MemoryReference()
    write_json(directory / "case.json", case)
    with observe_cpu(memory):
        expected = process_frame(source, config, options=options)
    if expected.artifacts:
        raise RuntimeError("compute-only CPU capture wrote artifacts")
    cpu_arrays = {
        "native": save_array(directory / "cpu_native.npy", expected.native_image),
        "preview": save_array(directory / "cpu_preview.npy", expected.preview_rgb8),
    }
    write_json(directory / "cpu_reference.json", {
        "source": source_identity, "source_config": config,
        "source_hashes": cpu_hashes, "arrays": cpu_arrays,
        "metadata": expected.native_metadata, "effective_config": expected.effective_config,
        "migration_report": expected.migration_report, "feedback": expected.feedback,
        "warnings": expected.warnings, "stages": memory.records,
        "stage_retention": "Immutable compressed memory captures for this comparison; descriptors and all gates retained. Failed captures are archived separately.",
    })
    results = []
    try:
        with observe_gpu(memory, expected.feedback["linear_reference_code_max"], results):
            actual = process_frame_gpu(
                source, config, options=options, gpu=GpuOptions(device=device),
            )
    except Exception as exc:
        retained = Path("/dev/shm") / ("brilliantisp_failed_stages_" + uuid.uuid4().hex[:12])
        memory.retain_failure(retained)
        write_json(directory / "gpu_failure.json", {
            "error_type": type(exc).__name__, "error": str(exc),
            "stages": results, "cpu_stage_archive": str(retained),
        })
        raise
    gate = FINAL_INTEGER if expected.native_image.dtype.kind == "u" else FINAL_FLOAT
    data_range = float(2**expected.native_metadata.code_bit_depth - 1) if gate is FINAL_INTEGER else 1.0
    native = compare_arrays(expected.native_image, actual.native_image, gate, data_range=data_range)
    preview = compare_arrays(expected.preview_rgb8, actual.preview_rgb8,
                             FINAL_INTEGER, data_range=255.0)
    feedback = compare_feedback(expected.feedback, actual.feedback)
    comparisons = {
        "native": native["passed"], "preview": preview["passed"],
        "metadata": expected.native_metadata == actual.native_metadata,
        "effective_config": jsonable(expected.effective_config) == jsonable(actual.effective_config),
        "migration_report": expected.migration_report == actual.migration_report,
        "warnings": expected.warnings == actual.warnings,
        "feedback": feedback["passed"],
        "stages": len(memory.records) == len(results) and all(row["passed"] for row in results),
        "no_artifacts": not actual.artifacts,
    }
    if source_identity != file_identity(source):
        raise RuntimeError("input changed during qualification")
    for identity in case.get("input_identities", ()):
        if file_identity(Path(identity["path"])) != identity:
            raise RuntimeError(f"declared qualification input changed: {identity['path']}")
    if config_before != json.dumps(jsonable(config), sort_keys=True, allow_nan=False):
        raise RuntimeError("caller configuration mutated during qualification")
    if cpu_hashes != source_hashes() or implementation != gpu_and_harness_sources():
        raise RuntimeError("implementation changed during qualification")
    delta = actual.native_image.astype(np.float64) - expected.native_image.astype(np.float64)
    record = {
        "id": case["id"], "passed": all(comparisons.values()),
        "comparisons": comparisons, "native": native, "preview": preview, "feedback": feedback,
        "cpu_arrays": cpu_arrays,
        "gpu_arrays": {
            "native": save_array(directory / "gpu_native.npy", actual.native_image),
            "preview": save_array(directory / "gpu_preview.npy", actual.preview_rgb8),
        },
        "error_arrays": {
            "signed": save_array(directory / "signed_error.npy", delta),
            "absolute": save_array(directory / "absolute_error.npy", np.abs(delta)),
        },
        "stages": results, "metadata": actual.native_metadata,
        "effective_config": actual.effective_config, "warnings": actual.warnings,
        "source": source_identity, "cpu_source_hashes": cpu_hashes,
        "implementation_hashes": implementation,
    }
    if not record["passed"]:
        retained = Path("/dev/shm") / ("brilliantisp_failed_stages_" + uuid.uuid4().hex[:12])
        memory.retain_failure(retained)
        record["cpu_stage_archive"] = str(retained)
    write_json(directory / "report.json", record)
    return record


def run_cases(cases, output: Path, device: str):
    output.mkdir(parents=True, exist_ok=False)
    cpu, implementation = source_hashes(), gpu_and_harness_sources()
    save_source_files(output / "source" / "cpu", cpu)
    save_source_files(output / "source" / "gpu_and_harness", implementation)
    write_json(output / "run_manifest.json", {
        "cases": cases, "cpu_source_hashes": cpu,
        "implementation_hashes": implementation, "environment": environment(),
        "retention": "Final arrays, native error arrays, stage identities and all numerical gates persisted. Additional successful stage payloads live only as immutable compressed memory buffers during each comparison. Previously saved stage archives are untouched.",
    })
    rows = []
    for case in cases:
        print("QUALIFY", case["id"], flush=True)
        result = qualify_case(case, output / case["id"], device)
        row = {
            "id": case["id"], "passed": result["passed"], "native": result["native"],
            "preview": result["preview"],
            "report": str(output / case["id"] / "report.json"),
        }
        rows.append(row)
        write_json(output / f"partial_results_{len(rows):03d}.json", rows)
        print(json.dumps(jsonable(row)), flush=True)
        if not result["passed"]:
            raise RuntimeError(f"qualification failed: {case['id']}")
    if cpu != source_hashes() or implementation != gpu_and_harness_sources():
        raise RuntimeError("implementation changed during qualification group")
    write_json(output / "results.json", rows)
    return rows


def revalidate_frozen_case(reference: Path, output: Path, device: str):
    from tools.gpu_validation.pca import load_array

    frozen_path = reference / "cpu_reference.json"
    frozen = json.loads(frozen_path.read_text())
    previous_path = reference / "report.json"
    previous = json.loads(previous_path.read_text())
    case = json.loads((reference / "case.json").read_text())
    if source_hashes() != frozen["source_hashes"]:
        raise RuntimeError("frozen CPU reference implementation changed")
    identities = [frozen["source"], *case.get("input_identities", ())]
    for identity in identities:
        if file_identity(Path(identity["path"])) != identity:
            raise RuntimeError(f"frozen input changed: {identity['path']}")
    if jsonable(case["config"]) != frozen["source_config"]:
        raise RuntimeError("frozen case/config binding changed")
    archive_root = Path(previous["cpu_stage_archive"])
    memory = MemoryReference()
    memory.records = frozen["stages"]

    def restore(value):
        if "array" in value:
            descriptor = value["array"]
            key = descriptor["content_sha256"]
            if key not in memory.blobs:
                memory.blobs[key] = (archive_root / f"{key}.npz").read_bytes()
                memory.read(descriptor)
        for part in value.get("members", ()):
            restore(part)

    for row in memory.records:
        restore(row["value"])
    native_expected = load_array(reference, frozen["arrays"]["native"])
    preview_expected = load_array(reference, frozen["arrays"]["preview"])
    if _array_identity(native_expected)["content_sha256"] != frozen["arrays"]["native"]["content_sha256"]:
        raise RuntimeError("frozen native array identity changed")
    if _array_identity(preview_expected)["content_sha256"] != frozen["arrays"]["preview"]["content_sha256"]:
        raise RuntimeError("frozen preview array identity changed")
    cpu_hashes, implementation = source_hashes(), gpu_and_harness_sources()
    output.mkdir(parents=True, exist_ok=False)
    save_source_files(output / "source" / "gpu_and_harness", implementation)
    write_json(output / "reference_binding.json", {
        "cpu_reference": file_identity(frozen_path), "previous_report": file_identity(previous_path),
        "case": file_identity(reference / "case.json"), "cpu_stage_archive": str(archive_root),
        "cpu_arrays_root": str(reference), "cpu_arrays": frozen["arrays"],
        "cpu_execution": "Not rerun. Immutable captured stage bytes and final arrays reloaded.",
    })
    results = []
    with observe_gpu(memory, frozen["feedback"]["linear_reference_code_max"], results):
        actual = process_frame_gpu(
            Path(frozen["source"]["path"]), frozen["source_config"],
            options=ProcessingOptions(run_id="qualified_reference"),
            gpu=GpuOptions(device=device),
        )
    gate = FINAL_INTEGER if native_expected.dtype.kind == "u" else FINAL_FLOAT
    data_range = float(2**frozen["metadata"]["code_bit_depth"] - 1) if gate is FINAL_INTEGER else 1.0
    native = compare_arrays(native_expected, actual.native_image, gate, data_range=data_range)
    preview = compare_arrays(preview_expected, actual.preview_rgb8, FINAL_INTEGER, data_range=255.0)
    feedback = compare_feedback(frozen["feedback"], actual.feedback)
    comparisons = {
        "native": native["passed"], "preview": preview["passed"],
        "metadata": frozen["metadata"] == jsonable(actual.native_metadata),
        "effective_config": frozen["effective_config"] == jsonable(actual.effective_config),
        "migration_report": frozen["migration_report"] == jsonable(actual.migration_report),
        "warnings": frozen["warnings"] == jsonable(actual.warnings),
        "feedback": feedback["passed"],
        "stages": len(memory.records) == len(results) and all(row["passed"] for row in results),
        "no_artifacts": not actual.artifacts,
    }
    if cpu_hashes != source_hashes() or implementation != gpu_and_harness_sources():
        raise RuntimeError("implementation changed during frozen revalidation")
    for identity in identities:
        if file_identity(Path(identity["path"])) != identity:
            raise RuntimeError(f"frozen input changed: {identity['path']}")
    delta = actual.native_image.astype(np.float64) - native_expected.astype(np.float64)
    record = {
        "id": case["id"], "passed": all(comparisons.values()),
        "comparisons": comparisons, "native": native, "preview": preview, "feedback": feedback,
        "source_reference": file_identity(frozen_path),
        "previous_report": file_identity(previous_path),
        "cpu_arrays_root": str(reference), "cpu_arrays": frozen["arrays"],
        "cpu_stage_archive": str(archive_root),
        "gpu_arrays": {
            "native": save_array(output / "gpu_native.npy", actual.native_image),
            "preview": save_array(output / "gpu_preview.npy", actual.preview_rgb8),
        },
        "error_arrays": {
            "signed": save_array(output / "signed_error.npy", delta),
            "absolute": save_array(output / "absolute_error.npy", np.abs(delta)),
        },
        "stages": results, "metadata": actual.native_metadata,
        "effective_config": actual.effective_config, "warnings": actual.warnings,
        "source": frozen["source"], "cpu_source_hashes": cpu_hashes,
        "implementation_hashes": implementation,
    }
    write_json(output / "report.json", record)
    return record
