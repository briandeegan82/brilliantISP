from __future__ import annotations

import argparse
import copy
import cProfile
import dataclasses
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(image) -> str:
    if not image.flags.c_contiguous:
        raise ValueError("benchmark checksums require contiguous output arrays")
    return hashlib.sha256(memoryview(image).cast("B")).hexdigest()


def engine_hashes(root: Path) -> dict[str, str]:
    files = [root / "brilliant_isp.py"]
    for directory in ("modules", "util"):
        files.extend((root / directory).rglob("*.py"))
    return {str(path.relative_to(root)): file_hash(path) for path in sorted(files)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fresh-process native CPU timing; manifest supplies exact cases and references.")
    parser.add_argument("--engine-root", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--mode", choices=("array", "raw", "check", "export"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if not sys.platform.startswith("linux"):
        parser.error("this benchmark records Linux ru_maxrss units")
    if args.runs < 1 or args.warmups < 0:
        parser.error("runs must be positive; warmups must be nonnegative")
    if not args.output.parent.is_dir() or args.output.exists():
        parser.error("output parent must exist and output must be new")
    engine_root = args.engine_root.resolve()
    source_hashes_before = engine_hashes(engine_root)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(engine_root))
    start = time.perf_counter_ns()
    import cv2
    import numpy as np
    import yaml
    import brilliant_isp
    from util.pipeline_runner import process_frame
    from util.pipeline_types import ProcessingOptions
    from util.raw_decode import decode_raw, RawFormat
    import_ns = time.perf_counter_ns() - start
    if Path(brilliant_isp.__file__).resolve().parent != engine_root:
        raise RuntimeError("engine import did not use the requested source tree")
    cv2.ocl.setUseOpenCL(False)
    cv2.setNumThreads(1)
    cases = json.loads(args.cases.read_text())
    case = cases["cases"][args.case]
    config_path = Path(case["config"])
    raw_path = Path(case["raw"])
    if file_hash(config_path) != case["config_sha256"] or file_hash(raw_path) != case["raw_sha256"]:
        raise RuntimeError("case input identity mismatch")
    config = yaml.safe_load(config_path.read_text())
    original = copy.deepcopy(config)
    start = time.perf_counter_ns()
    decoded = decode_raw(raw_path, RawFormat.from_sensor_info(config["sensor_info"]))
    decode_once_ns = time.perf_counter_ns() - start
    pixels = decoded.pixels
    pixels.setflags(write=False)
    source_hash_before = array_hash(pixels)
    array_config = copy.deepcopy(config)
    array_config["sensor_info"].update(
        width=decoded.width, height=decoded.height, bayer_pattern=decoded.bayer_pattern)
    captures = tuple(case["captures"]) if args.mode == "check" else ()
    forbidden = args.output.with_suffix(".must_not_exist")
    if forbidden.exists():
        raise RuntimeError("compute-only sentinel path already exists")
    options = ProcessingOptions(output_dir=forbidden, capture_stages=captures)
    if args.mode == "export":
        export_dir = args.output.with_suffix(".exports")
        options = ProcessingOptions(
            output_dir=export_dir,
            save_outputs=True, diagnostics=True)
        if export_dir.exists():
            raise RuntimeError("export directory must be new")
    source = pixels if args.mode == "array" else raw_path
    selected_config = array_config if args.mode == "array" else config
    selected_original = copy.deepcopy(selected_config)

    def run():
        gc.collect()
        begin = time.perf_counter_ns()
        result = process_frame(source, selected_config, options=options)
        elapsed = time.perf_counter_ns() - begin
        peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        hashes = {"native": array_hash(result.native_image), "preview": array_hash(result.preview_rgb8)}
        if hashes != case["output_sha256"]:
            raise AssertionError(f"output regression: {hashes}")
        if json.loads(json.dumps(result.feedback)) != case["feedback"]:
            raise AssertionError(f"feedback regression: {result.feedback}")
        metadata = dataclasses.asdict(result.native_metadata)
        metadata.pop("source_identity")
        if json.loads(json.dumps(metadata)) != case["metadata"]:
            raise AssertionError("output metadata changed")
        if selected_config != selected_original or config != original:
            raise AssertionError("input config mutated")
        if args.mode != "export" and (result.artifacts or forbidden.exists()):
            raise AssertionError("compute-only request wrote artifacts")
        for identifier, expected in case["captures"].items():
            if args.mode == "check":
                actual = result.captures[identifier]
                if str(actual.dtype) != expected["dtype"] or list(actual.shape) != expected["shape"] or array_hash(actual) != expected["sha256"]:
                    raise AssertionError(f"stage regression: {identifier}")
        events = [dataclasses.asdict(event) for event in result.timings]
        if any(event["status"] != "completed" for event in events):
            raise AssertionError("non-completed stage in successful result")
        return {
            "wall_ns": elapsed, "peak_rss_kib_before_validation": peak_kib,
            "stage_events": events, "output_sha256": hashes,
            "feedback": result.feedback, "artifact_count": len(result.artifacts),
        }

    cold = run()
    warmups = [] if args.mode == "check" else [run() for _ in range(args.warmups)]
    samples = [] if args.mode == "check" else [run() for _ in range(args.runs)]
    peak_before_profile = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if args.profile:
        profile_path = args.output.with_suffix(".prof")
        if profile_path.exists():
            raise RuntimeError("profile path must be new")
        profiler = cProfile.Profile()
        profiler.runcall(run)
        profiler.dump_stats(str(profile_path))
    if array_hash(pixels) != source_hash_before:
        raise AssertionError("decoded input mutated")
    if engine_hashes(engine_root) != source_hashes_before or file_hash(raw_path) != case["raw_sha256"]:
        raise AssertionError("engine or RAW input changed during measurement")
    versions = {}
    for package in ("numpy", "scipy", "opencv-python", "numba", "tifffile", "PyYAML"):
        versions[package] = importlib.metadata.version(package)
    thread_names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
    record = {
        "schema_version": 1,
        "case": args.case, "mode": args.mode,
        "scope": "Independent-frame public process_frame call; wall includes validation/construction/result preparation; no state carryover. Array mode excludes RAW decode; raw mode includes it. GC, checksums, assertions and cProfile are outside recorded wall times.",
        "engine_root": str(engine_root), "harness_sha256": file_hash(Path(__file__)),
        "engine_source_sha256": source_hashes_before,
        "cases_sha256": file_hash(args.cases),
        "environment": {
            "python": sys.version, "platform": platform.platform(), "versions": versions,
            "threads": {key: os.environ.get(key) for key in thread_names},
            "opencv_threads": cv2.getNumThreads(), "opencl": cv2.ocl.useOpenCL(),
            "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "cpu_model": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), platform.processor()),
        },
        "import_ns": import_ns, "decode_once_setup_ns": decode_once_ns,
        "cold": cold, "warmups": warmups, "samples": samples,
        "peak_rss_kib_before_optional_profile": peak_before_profile,
        "memory_scope": "Linux process high-water RSS, including imports, one decoded input, first-call/JIT, warmups and untimed checksum validation; not per-stage allocation or total machine memory.",
    }
    with args.output.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    print(json.dumps({"output": str(args.output), "case": args.case, "mode": args.mode,
                      "samples": len(samples), "all_output_hashes_match": True}))


if __name__ == "__main__":
    main()
