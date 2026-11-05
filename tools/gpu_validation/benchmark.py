from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import resource
import subprocess
import time
import uuid

import numpy as np
import yaml

from brilliantisp_gpu.batch import process_batch
from brilliantisp_gpu.config import prepare_config
from brilliantisp_gpu.device import gpu_scope, resolve_device
from brilliantisp_gpu.engine import GpuEngine
from brilliantisp_gpu.grouping import _IO_KEYS, group_key
from brilliantisp_gpu.memory import plan_memory
from brilliantisp_gpu.microbatch import _front_cohort, _render_cohort
from brilliantisp_gpu.reporting import outcome_record, summarize_records
from brilliantisp_gpu.sources import decode_source
from brilliantisp_gpu.types import BatchItem, GpuOptions, RawTensor, require_torch
from modules.tone_mapping.registry import resolve_tone_mapping
from tools.gpu_validation.contracts import FINAL_FLOAT, FINAL_INTEGER
from tools.gpu_validation.metrics import compare_arrays
from tools.gpu_validation.pca import load_array
from tools.gpu_validation.qualification import _array_identity
from tools.gpu_validation.reference import (
    ROOT, environment, file_identity, gpu_and_harness_sources, jsonable,
    save_source_files, source_hashes, write_json,
)
from tools.gpu_validation.workloads import compare_feedback
from util.native_output import output_metadata
from util.pipeline_types import ProcessingOptions


FULL_REPORTS = {
    "fv": "fv_ad_post_bnr_28b557d5/fv/report.json",
    "ad": "fv_ad_post_bnr_28b557d5/ad/report.json",
    "sdr_colorchecker": "sdr_exp_fix_de876dc9/report.json",
    "triton_sdr_tiff": "triton_qualification_17739e72/triton_sdr_tiff/report.json",
    "triton_hdr24_overlay": "triton_qualification_17739e72/triton_hdr24_overlay/report.json",
}
LAYOUTS = {
    "repeated_fv_16": ("fv",) * 16,
    "repeated_ad_16": ("ad",) * 16,
    "repeated_sdr_16": ("sdr_colorchecker",) * 16,
    "diverse_5": tuple(FULL_REPORTS),
}
MODES = ("device_resident_compute", "array_to_result", "file_to_result", "file_to_file")


@dataclass
class Case:
    identifier: str
    source: Path
    config: dict
    frozen: dict
    native: np.ndarray
    preview: np.ndarray
    decoded: object
    prepared: object
    config_file: Path
    key: str
    identities: list


def _config_identity(config):
    result = copy.deepcopy(jsonable(config))
    platform = result.get("platform", {})
    for name in _IO_KEYS:
        platform.pop(name, None)
    return result


def _check_frozen_array(array, descriptor):
    if _array_identity(array)["content_sha256"] != descriptor["content_sha256"]:
        raise RuntimeError("frozen benchmark reference array changed")


def load_cases(scope, identifiers, directory):
    result = {}
    roi_root = ROOT / "tmp/gpu_validation/m5/real_roi_workloads_1053aebc"
    roi = json.loads((roi_root / "datasets.json").read_text())["datasets"] if scope == "roi512" else []
    for identifier in dict.fromkeys(identifiers):
        if scope == "full":
            report_path = ROOT / "tmp/gpu_validation/m5" / FULL_REPORTS[identifier]
            report = json.loads(report_path.read_text())
            if not report["passed"]:
                raise RuntimeError("performance requires qualified full-frame output")
            binding = report["source_reference"]
            reference_path = Path(binding["path"])
            if file_identity(reference_path) != binding:
                raise RuntimeError("full-frame reference manifest changed")
            frozen = json.loads(reference_path.read_text())
            reference_root = reference_path.parent
            source = Path(frozen["source"]["path"])
            config = frozen["source_config"]
            identities = [file_identity(report_path), binding, frozen["source"]]
            if source_hashes() != frozen["source_hashes"]:
                raise RuntimeError("CPU reference source changed")
        else:
            entry = next(row for row in roi if row["id"] == identifier)
            source = Path(entry["raw"]["path"])
            config = yaml.safe_load(Path(entry["config"]["path"]).read_text())
            reference_path = Path(entry["cpu_reference"]["path"])
            reference_root = reference_path.parent
            frozen = json.loads(reference_path.read_text())
            identities = [entry["raw"], entry["config"], entry["pixels"],
                          entry["cpu_reference"], entry["original_source"], *entry["original_configs"]]
        for identity in identities:
            if file_identity(Path(identity["path"])) != identity:
                raise RuntimeError(f"benchmark input changed: {identity['path']}")
        native = load_array(reference_root, frozen["native"])
        preview = load_array(reference_root, frozen["preview"])
        _check_frozen_array(native, frozen["native"])
        _check_frozen_array(preview, frozen["preview"])
        prepared = prepare_config(config, options=ProcessingOptions())
        sensor, platform = prepared.effective_config["sensor_info"], prepared.effective_config["platform"]
        decoded = decode_source(source, sensor_info=sensor, platform=platform, run_id="benchmark_setup")
        if decoded.pixels.shape != (int(sensor["height"]), int(sensor["width"])) \
                or decoded.sensor_info["bayer_pattern"] != sensor["bayer_pattern"]:
            raise ValueError("this benchmark requires matching declared and decoded geometry/CFA for array residency")
        config_file = directory / f"{identifier}_merged.yml"
        with config_file.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(jsonable(config), stream, sort_keys=False)
        key = group_key(prepared.effective_config, shape=decoded.pixels.shape,
                        cfa=decoded.sensor_info["bayer_pattern"],
                        storage_dtype=decoded.pixels.dtype.name)
        result[identifier] = Case(
            identifier, source, config, frozen, native, preview, decoded,
            prepared, config_file, key, identities,
        )
    return result


def validate_pixels(case, native, preview):
    comparisons = {}
    for name, expected, actual in (
        ("native", case.native, native), ("preview", case.preview, preview),
    ):
        descriptor = case.frozen[name]
        identity = _array_identity(actual)
        if identity["content_sha256"] == descriptor["content_sha256"]:
            comparison = {"passed": True, "exact": True}
        else:
            gate = FINAL_INTEGER if expected.dtype.kind == "u" else FINAL_FLOAT
            maximum = float(2**case.frozen["metadata"]["code_bit_depth"] - 1) \
                if name == "native" and gate is FINAL_INTEGER else 255.0 if name == "preview" else 1.0
            try:
                comparison = compare_arrays(expected, actual, gate, data_range=maximum)
            except ValueError as exc:
                comparison = {"passed": False, "error": str(exc), "gate": gate.name}
        comparisons[name] = {**comparison, "identity": identity}
    return comparisons


def validate_result(case, native, preview, metadata, feedback, config, migration, warnings, mode):
    checks = validate_pixels(case, native, preview)
    actual_metadata = jsonable(metadata)
    expected_metadata = case.frozen["metadata"]
    if mode in ("device_resident_compute", "array_to_result"):
        actual_metadata["source_identity"] = expected_metadata["source_identity"]
    checks.update(
        metadata={"passed": actual_metadata == expected_metadata},
        feedback=compare_feedback(case.frozen["feedback"], feedback),
        effective_config={"passed": _config_identity(config) == _config_identity(case.frozen["effective_config"])},
        migration_report={"passed": jsonable(migration) == case.frozen["migration_report"]},
        warnings={"passed": jsonable(warnings) == case.frozen["warnings"]},
    )
    return {"passed": all(value["passed"] for value in checks.values()), "checks": checks}


def summarize_samples(samples, *, images, megapixels):
    measured = [row for row in samples if row["phase"] == "measured"]
    phases = [row["phase"] for row in samples]
    if phases != ["cold", "warmup", "warmup"] + ["measured"] * 5 \
            or any(not row["passed"] for row in samples):
        raise ValueError("benchmark summary requires one cold, two warmup, five measured samples and parity on every run")
    if isinstance(images, bool) or not isinstance(images, int) or images <= 0 \
            or not np.isfinite(megapixels) or megapixels <= 0:
        raise ValueError("benchmark workload size must be positive and finite")
    nanoseconds = np.asarray([row["wall_ns"] for row in measured], np.float64)
    if not np.all(np.isfinite(nanoseconds)) or np.any(nanoseconds <= 0):
        raise ValueError("benchmark durations must be finite and positive")
    median_ns = float(np.median(nanoseconds))
    return {
        "measured_repetitions": 5, "images_per_workload": images,
        "megapixels_per_workload": megapixels,
        "workload_wall_ns": {
            "median": median_ns, "mean": float(nanoseconds.mean()),
            "min": float(nanoseconds.min()), "max": float(nanoseconds.max()),
            "std_population": float(nanoseconds.std()),
            "p05": float(np.percentile(nanoseconds, 5)),
            "p95": float(np.percentile(nanoseconds, 95)),
        },
        "images_per_second_at_median_workload_wall": images * 1e9 / median_ns,
        "megapixels_per_second_at_median_workload_wall": megapixels * 1e9 / median_ns,
        "per_image_wall_equivalent_ns": median_ns / images,
        "latency_definition": "Whole fixed workload wall latency; per-image value is a throughput equivalent, not individual queue/service latency.",
    }


def thermal_context():
    try:
        process = subprocess.run([
            "nvidia-smi", "--query-gpu=name,driver_version,temperature.gpu,power.draw,clocks.sm,clocks.mem",
            "--format=csv,noheader,nounits",
        ], capture_output=True, text=True, timeout=10, check=False)
        return {"returncode": process.returncode, "stdout": process.stdout, "stderr": process.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"unavailable": str(exc)}


def resident_pool(cases, gpu):
    pool = {}
    torch = require_torch()
    device = resolve_device(gpu.device)
    with torch.cuda.device(device), gpu_scope(device):
        for name, case in cases.items():
            engine = GpuEngine(case.prepared, ProcessingOptions(), None, gpu)
            engine._prepare_execution()
            engine._configure_source(case.decoded)
            pool[name] = engine._upload_source(case.decoded)
    return pool


def prepare_resident_run(cases, names, pool, gpu):
    torch = require_torch()
    contexts = []
    for index, name in enumerate(names):
        case = cases[name]
        context = GpuEngine(case.prepared, ProcessingOptions(run_id=f"benchmark_device_{index}"), None, gpu)
        context._prepare_execution()
        context._configure_source(case.decoded)
        context._upload_dtype = case.decoded.pixels.dtype.name
        contexts.append(context)
    groups, index = [], 0
    while index < len(names):
        stop = index + 1
        while stop < len(names) and cases[names[stop]].key == cases[names[index]].key:
            stop += 1
        free, total = torch.cuda.mem_get_info(contexts[index].device)
        plan = plan_memory(
            shape=cases[names[index]].decoded.pixels.shape,
            algorithm=contexts[index].parm_dem["algorithm"], gpu=gpu,
            free_bytes=free, total_bytes=total, available_count=stop - index,
        )
        end = index + plan.group_size
        groups.append((index, end, jsonable(plan)))
        index = end
    return contexts, groups


def run_resident(names, pool, contexts, groups):
    torch = require_torch()
    outputs = []
    device = contexts[0].device
    with torch.cuda.device(device), gpu_scope(device):
        for start, end, _ in groups:
            cohort = tuple(contexts[start:end])
            codes = torch.cat([pool[name].codes for name in names[start:end]], dim=0)
            for context in cohort:
                context.pass_index += 1
            front = _front_cohort(cohort, RawTensor(codes, cohort[0]._upload_dtype))
            del codes
            linears = [
                context._compute_linear(RawTensor(front.codes[i:i + 1], front.storage_dtype))
                for i, context in enumerate(cohort)
            ]
            del front
            resolved = resolve_tone_mapping(cohort[0].c_yaml)
            native, preview, _ = _render_cohort(cohort, linears, resolved, cohort[0].prepared.output_spec)
            del linears
            for i, context in enumerate(cohort):
                outputs.append((context, native[i:i + 1], preview[i:i + 1], resolved))
    return outputs


def profile_boundary(cases, names, pool, gpu, mode, leaf):
    from brilliantisp_gpu.blocks import output as output_blocks

    torch = require_torch()
    selected = resolve_device(gpu.device)
    profiled_gpu = replace(gpu, profile_stages=True)
    uploads, previews, downloads = [], [], []
    upload_original, host_original = GpuEngine._upload_source, GpuEngine._host_result
    preview_original = output_blocks.preview_from_native

    def upload(context, decoded):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream(selected))
        result = upload_original(context, decoded)
        end.record(torch.cuda.current_stream(selected))
        uploads.append((start, end, int(decoded.pixels.nbytes)))
        return result

    def host(context, value, *, frame_index=0):
        start = time.perf_counter_ns()
        result = host_original(context, value, frame_index=frame_index)
        downloads.append({"host_wall_ns": time.perf_counter_ns() - start,
                          "shape": list(result.shape) if isinstance(result, np.ndarray) else None})
        return result

    def preview(native, spec):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream(selected))
        result = preview_original(native, spec)
        end.record(torch.cuda.current_stream(selected))
        previews.append((start, end, int(native.shape[0])))
        return result

    if mode == "device_resident_compute":
        contexts, groups = prepare_resident_run(cases, names, pool, profiled_gpu)
    else:
        items = [
            BatchItem(
                f"{index:03d}_{name}",
                cases[name].decoded.pixels if mode == "array_to_result" else cases[name].source,
                cases[name].config if mode == "array_to_result" else cases[name].config_file,
                output_stem=f"{index:03d}_{name}",
            )
            for index, name in enumerate(names)
        ]
        options = ProcessingOptions(run_id="benchmark_profile_probe",
                                    save_outputs=mode == "file_to_file", output_dir=leaf)
    GpuEngine._upload_source = upload
    GpuEngine._host_result = host
    output_blocks.preview_from_native = preview
    try:
        torch.cuda.synchronize(selected)
        start = time.perf_counter_ns()
        if mode == "device_resident_compute":
            outputs = run_resident(names, pool, contexts, groups)
        else:
            outcomes = list(process_batch(iter(items), gpu=profiled_gpu, options=options))
        torch.cuda.synchronize(selected)
        wall = time.perf_counter_ns() - start
    finally:
        GpuEngine._upload_source = upload_original
        GpuEngine._host_result = host_original
        output_blocks.preview_from_native = preview_original
    checks, artifacts, timings, metrics = [], [], [], {}
    if mode == "device_resident_compute":
        for name, (context, native_tensor, preview_tensor, resolved) in zip(names, outputs):
            native, rgb = context._host_result(native_tensor), context._host_result(preview_tensor)
            metadata = output_metadata(
                native, context.prepared.output_spec, source_identity=context.source_identity,
                tone_mapped=bool(resolved.spec.enabled))
            checks.append(validate_result(
                cases[name], native, rgb, metadata, context.last_feedback, context.c_yaml,
                context.prepared.migration_report, tuple(context.config_notices), mode))
            timings.append(jsonable(context.stage_events))
        metrics["actual_batches"] = [end - start for start, end, _ in groups]
    else:
        if len(outcomes) != len(names):
            raise RuntimeError("profiling probe result count mismatch")
        records = [outcome_record(item, outcome, index=i)
                   for i, (item, outcome) in enumerate(zip(items, outcomes))]
        metrics.update(
            cpu_prepare_decode_sum_ns=sum(row["metrics"]["decode_ns"] for row in records),
            commit_sum_ns=sum(row["metrics"]["commit_ns"] for row in records),
            summary=summarize_records(records, wall_ns=wall),
        )
        for name, outcome in zip(names, outcomes):
            if outcome.status != "success":
                checks.append({"passed": False, "status": outcome.status,
                               "error": jsonable(outcome.error)})
                continue
            actual = outcome.result
            checks.append(validate_result(
                cases[name], actual.native_image, actual.preview_rgb8, actual.native_metadata,
                actual.feedback, actual.effective_config, actual.migration_report, actual.warnings, mode))
            timings.append(jsonable(actual.timings))
            for path in actual.artifacts:
                path = Path(path)
                if not path.resolve().is_relative_to(leaf.resolve()):
                    raise RuntimeError("probe output escaped this run")
                artifacts.append(file_identity(path))
    return {
        "passed": len(checks) == len(names) and all(row["passed"] for row in checks),
        "wall_ns": wall, "checks": checks, "metrics": metrics, "artifacts": artifacts,
        "upload_and_device_conversion_stream_ns": [a.elapsed_time(b) * 1e6 for a, b, _ in uploads],
        "source_upload_bytes": [size for _, _, size in uploads],
        "preview_generation_stream_ns": [a.elapsed_time(b) * 1e6 for a, b, _ in previews],
        "preview_batch_sizes": [size for _, _, size in previews],
        "bulk_output_host_result_calls": downloads,
        "synchronized_stage_events_per_image": timings,
        "methodology": {
            "instrumented_extra_run_not_used_for_headlines": True,
            "profile_stages": True,
            "upload": "CUDA stream interval includes upload plus int64 device conversion and host gaps",
            "download": "Host perf_counter_ns includes bulk D2H, dtype handling, and owned CPU result copy",
            "preview": "CUDA stream interval around preview_from_native, excluding later D2H",
            "stage_events": "Synchronized host stage wall intervals, not kernel-active time. Cohort stage intervals are shared across images; do not sum the image axis.",
            "overlap": "CPU prep may overlap GPU work; boundaries and stage intervals are nonadditive",
            "resident_exclusions": "Resident input setup uploads and untimed output-validation downloads are excluded",
        },
    }


def worker(output, scope, layout, mode, cap, device, *, profile_breakdown=False):
    output.mkdir(parents=True, exist_ok=False)
    torch = require_torch()
    selected = resolve_device(device)
    gpu = GpuOptions(device=device, max_batch_size=4 if cap == "auto" else int(cap))
    cpu_sources, implementations = source_hashes(), gpu_and_harness_sources()
    names = LAYOUTS[layout]
    cases = load_cases(scope, names, output)
    pool = resident_pool(cases, gpu) if mode == "device_resident_compute" else None
    ram = Path("/dev/shm") / ("brilliantisp_benchmark_" + uuid.uuid4().hex[:12])
    ram.mkdir()
    megapixels = sum(cases[name].decoded.pixels.size / 1e6 for name in names)
    start_thermal = thermal_context()
    manifest = {
        "scope": scope, "layout": layout, "names": names, "mode": mode, "requested_cap": cap,
        "gpu_options": gpu, "device": str(selected), "environment": environment(),
        "cpu_sources": cpu_sources, "implementation_hashes": implementations,
        "cases": {name: {"source": file_identity(case.source), "config": case.config,
                          "frozen_metadata": case.frozen["metadata"],
                          "native": case.frozen["native"], "preview": case.frozen["preview"],
                          "bindings": case.identities} for name, case in cases.items()},
        "cpu_threads": {"os_logical": os.cpu_count(), "torch": torch.get_num_threads(),
                        "torch_interop": torch.get_num_interop_threads(),
                        "environment": {key: os.environ.get(key) for key in
                                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}},
        "io": {"output_medium": "tmpfs /dev/shm; not physical disk throughput",
               "output_parent": str(ram), "input_cache_policy": "OS page cache not flushed",
               "durability": "API write/close semantics; no additional fsync"},
        "timing": {"cold": 1, "warmup": 2, "measured": 5,
                   "setup": "CUDA initialization/config preparation/resident input setup excluded from cold graph interval",
                   "reporting_checksums_parity": "outside timed intervals",
                   "diagnostics": False, "profile_stages": False},
        "thermal_start": start_thermal,
    }
    write_json(output / "manifest.json", manifest)
    samples = []
    for run_index, phase in enumerate(("cold", "warmup", "warmup") + ("measured",) * 5):
        leaf = ram / f"{run_index:02d}_{phase}"
        if mode == "device_resident_compute":
            contexts, groups = prepare_resident_run(cases, names, pool, gpu)
            items = None
        else:
            items = [
                BatchItem(
                    f"{index:03d}_{name}",
                    cases[name].decoded.pixels if mode == "array_to_result" else cases[name].source,
                    cases[name].config if mode == "array_to_result" else cases[name].config_file,
                    output_stem=f"{index:03d}_{name}",
                )
                for index, name in enumerate(names)
            ]
            options = ProcessingOptions(run_id=f"benchmark_{run_index}",
                                        save_outputs=mode == "file_to_file", output_dir=leaf)
        torch.cuda.synchronize(selected)
        torch.cuda.reset_peak_memory_stats(selected)
        start_event = torch.cuda.Event(enable_timing=True) if mode == "device_resident_compute" else None
        end_event = torch.cuda.Event(enable_timing=True) if start_event is not None else None
        t0 = time.perf_counter_ns()
        if start_event is not None:
            start_event.record(torch.cuda.current_stream(selected))
            outputs = run_resident(names, pool, contexts, groups)
            end_event.record(torch.cuda.current_stream(selected))
        else:
            outcomes = list(process_batch(iter(items), gpu=gpu, options=options))
        torch.cuda.synchronize(selected)
        elapsed = time.perf_counter_ns() - t0
        stream_ns = start_event.elapsed_time(end_event) * 1e6 if start_event is not None else None
        peaks = {"allocated_bytes": torch.cuda.max_memory_allocated(selected),
                 "reserved_bytes": torch.cuda.max_memory_reserved(selected),
                 "driver_free_total_bytes": list(torch.cuda.mem_get_info(selected)),
                 "host_process_lifetime_ru_maxrss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
        checks, artifacts, metadata = [], [], {}
        if mode == "device_resident_compute":
            for name, (context, native_tensor, preview_tensor, resolved) in zip(names, outputs):
                native = context._host_result(native_tensor)
                preview = context._host_result(preview_tensor)
                native_metadata = output_metadata(
                    native, context.prepared.output_spec, source_identity=context.source_identity,
                    tone_mapped=bool(resolved.spec.enabled))
                checks.append(validate_result(
                    cases[name], native, preview, native_metadata, context.last_feedback,
                    context.c_yaml, context.prepared.migration_report, tuple(context.config_notices), mode))
            metadata = {"actual_batches": [end - start for start, end, _ in groups],
                        "memory_plans": [plan for _, _, plan in groups], "oom_splits": 0}
            del outputs, contexts
        else:
            records = [outcome_record(item, outcome, index=i)
                       for i, (item, outcome) in enumerate(zip(items, outcomes))]
            aggregate = summarize_records(records, wall_ns=elapsed)
            if len(outcomes) != len(names) or any(o.item_id != item.item_id for item, o in zip(items, outcomes)):
                raise RuntimeError("benchmark result count/order mismatch")
            for name, outcome in zip(names, outcomes):
                if outcome.status != "success":
                    checks.append({"passed": False, "status": outcome.status, "error": jsonable(outcome.error)})
                    continue
                actual = outcome.result
                checks.append(validate_result(
                    cases[name], actual.native_image, actual.preview_rgb8, actual.native_metadata,
                    actual.feedback, actual.effective_config, actual.migration_report, actual.warnings, mode))
                for path in actual.artifacts:
                    path = Path(path)
                    if not path.resolve().is_relative_to(leaf.resolve()):
                        raise RuntimeError("benchmark output artifact escaped this run")
                    artifacts.append(file_identity(path))
            metadata = {
                "summary": aggregate,
                "items": [{key: record[key] for key in ("item_id", "status", "batch_size", "attempts", "metrics")}
                          for record in records],
                "breakdown": {
                    "cpu_prepare_decode_sum_ns": sum(r["metrics"].get("decode_ns", 0) for r in records),
                    "compute_upload_preview_download_aggregate_ns": aggregate["compute_transfer_attempt_ns"],
                    "separate_upload_ns": None, "separate_download_ns": None, "separate_preview_ns": None,
                    "separation_limit": "Uninstrumented public boundary combines these operations; aggregate is not pure GPU compute. CPU prep may overlap and sums are nonadditive.",
                },
            }
            del outcomes, records
        sample = {
            "phase": phase, "run_index": run_index, "wall_ns": elapsed,
            "device_stream_elapsed_ns": stream_ns,
            "device_interval_definition": "Resident stream elapsed includes host control/launch gaps; not SM-active time." if stream_ns is not None else None,
            "passed": len(checks) == len(names) and all(c["passed"] for c in checks),
            "checks": checks, "memory": peaks, "batch": metadata, "artifacts": artifacts,
            "output_artifacts_retention": "Only this run's newly created benchmark files removed after identities saved; qualification references remain untouched.",
        }
        write_json(output / f"sample_{run_index:02d}_{phase}.json", sample)
        samples.append(sample)
        for artifact in artifacts:
            Path(artifact["path"]).unlink()
        if not sample["passed"]:
            write_json(output / "failure.json", sample)
            raise RuntimeError("benchmark parity/status gate failed; no throughput summary")
        if mode != "file_to_file" and leaf.exists():
            raise RuntimeError("compute-only benchmark wrote files")
    for case in cases.values():
        for identity in case.identities:
            if file_identity(Path(identity["path"])) != identity:
                raise RuntimeError("benchmark input changed during timing")
    if cpu_sources != source_hashes() or implementations != gpu_and_harness_sources():
        raise RuntimeError("benchmark implementation changed during timing")
    probe = profile_boundary(cases, names, pool, gpu, mode, ram / "profile_probe") if profile_breakdown else None
    if probe is not None:
        write_json(output / "profile_probe.json", probe)
        for artifact in probe["artifacts"]:
            Path(artifact["path"]).unlink()
    if cpu_sources != source_hashes() or implementations != gpu_and_harness_sources():
        raise RuntimeError("benchmark implementation changed during profiling")
    summary = {
        "passed": probe is None or probe["passed"], "headline_parity_passed": True,
        "scope": scope, "layout": layout, "mode": mode, "requested_cap": cap,
        **summarize_samples(samples, images=len(names), megapixels=megapixels),
        "samples": [file_identity(output / f"sample_{i:02d}_{row['phase']}.json")
                    for i, row in enumerate(samples)],
        "thermal_end": thermal_context(),
        "profile_probe": file_identity(output / "profile_probe.json") if probe is not None else None,
        "profile_probe_passed": probe["passed"] if probe is not None else None,
    }
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scope", choices=("full", "roi512"), default="full")
    parser.add_argument("--layout", choices=tuple(LAYOUTS), default="repeated_fv_16")
    parser.add_argument("--mode", choices=MODES, default="array_to_result")
    parser.add_argument("--cap", choices=("1", "2", "4", "auto"), default="1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--profile-breakdown", action="store_true")
    args = parser.parse_args()
    summary = worker(args.output.resolve(), args.scope, args.layout, args.mode, args.cap, args.device,
                     profile_breakdown=args.profile_breakdown)
    print(json.dumps(summary), flush=True)
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
