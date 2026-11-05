from __future__ import annotations

import argparse
from contextlib import contextmanager
import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Iterator, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE_TREES = ("modules", "util", "config", "tests")
VERSIONS = ("numpy", "scipy", "opencv-python", "numba", "tifffile", "PyYAML", "torch")


def file_identity(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ):
        raise RuntimeError(f"file changed while hashing: {path}")
    return {"path": str(path), "bytes": before.st_size, "sha256": digest.hexdigest()}


def source_paths(root: Path = ROOT) -> tuple[Path, ...]:
    files = {root / "brilliant_isp.py", root / "requirements.txt"}
    files.update(p for p in root.glob("isp_pipeline*.py")
                 if p.name != "isp_pipeline_gpu_batch.py")
    if (root / "requirements-dev.txt").exists():
        files.add(root / "requirements-dev.txt")
    for tree in SOURCE_TREES:
        files.update(
            p for p in (root / tree).rglob("*")
            if p.is_file() and p.suffix in (".py", ".yml", ".yaml")
            and "__pycache__" not in p.parts
            and not (tree == "tests" and "gpu" in p.relative_to(root / tree).parts)
            and not (tree == "tests" and p.name.startswith("test_gpu_"))
        )
    return tuple(sorted(files))


def source_hashes(root: Path = ROOT) -> dict[str, str]:
    return {str(p.relative_to(root)): file_identity(p)["sha256"] for p in source_paths(root)}


def gpu_and_harness_sources(root: Path = ROOT) -> dict[str, str]:
    files = set()
    launcher = root / "isp_pipeline_gpu_batch.py"
    if launcher.is_file():
        files.add(launcher)
    for relative in ("brilliantisp_gpu", "tools/gpu_validation", "tests/gpu"):
        files.update(
            p for p in (root / relative).rglob("*")
            if p.is_file() and p.suffix in (".py", ".cu", ".json", ".txt", ".npz")
            and "__pycache__" not in p.parts
        )
    files.update((root / "tests").glob("test_gpu_validation_*.py"))
    return {str(p.relative_to(root)): file_identity(p)["sha256"] for p in sorted(files)}


def save_source_files(output: Path, identities: Mapping[str, str], root: Path = ROOT) -> None:
    for relative, expected in identities.items():
        source = root / relative
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError(f"source changed during evidence capture: {relative}")
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)


def jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported reference value: {type(value).__name__}")


def write_json(path: Path, value: Any) -> None:
    text = json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text + "\n")


def environment() -> dict[str, Any]:
    versions = {}
    for name in VERSIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "versions": versions}


def snapshot(output: Path, root: Path = ROOT) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    before = source_hashes(root)
    for relative, expected in before.items():
        source = root / relative
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError(f"source changed during snapshot: {relative}")
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
    git = {}
    for key, args in (
        ("head", ("rev-parse", "HEAD")),
        ("status", ("status", "--porcelain=v1", "--untracked-files=normal")),
        ("diff", ("diff", "--binary", "HEAD")),
    ):
        git[key] = subprocess.run(
            ("git", *args), cwd=root, check=True, capture_output=True, text=True
        ).stdout
    after = source_hashes(root)
    if before != after:
        raise RuntimeError("CPU sources changed during snapshot")
    record = {"schema_version": 1, "root": str(root), "sources": before, "git": git,
              "environment": environment(), "harness": file_identity(Path(__file__))}
    write_json(output / "manifest.json", record)
    return record


def save_array(path: Path, values: np.ndarray) -> dict[str, Any]:
    image = np.asarray(values)
    if image.dtype.kind not in "buif" or not image.size:
        raise ValueError(f"reference capture is not a nonempty real array: {path.name}")
    if image.dtype.kind == "f" and not np.all(np.isfinite(image)):
        raise ValueError(f"reference capture is not finite: {path.name}")
    data = np.ascontiguousarray(image)
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": list(data.shape), "dtype": data.dtype.str}).encode())
    digest.update(memoryview(data).cast("B"))
    content_hash = digest.hexdigest()
    target = path.parent / f"array_{content_hash}.npz"
    if not target.exists():
        with target.open("xb") as stream:
            np.savez_compressed(stream, pixels=data)
    return {"file": target.name, "label": path.name, "shape": list(data.shape),
            "dtype": data.dtype.str, "content_sha256": content_hash,
            "sha256": file_identity(target)["sha256"]}


@contextmanager
def observe_cpu_stages(output: Path) -> Iterator[list[dict[str, Any]]]:
    from brilliant_isp import BrilliantISP

    original = BrilliantISP._run_stage
    records: list[dict[str, Any]] = []
    occurrences: dict[tuple[str, int, int], int] = {}

    def observed(engine: Any, stage_id: str, thunk: Any, ae_pass: int = 0) -> Any:
        value = original(engine, stage_id, thunk, ae_pass=ae_pass)
        key = (stage_id, engine.pass_index, ae_pass)
        occurrence = occurrences.get(key, 0)
        occurrences[key] = occurrence + 1
        record: dict[str, Any] = {
            "stage_id": stage_id, "pass_index": engine.pass_index,
            "ae_pass": ae_pass, "occurrence": occurrence,
        }
        prefix = f"{len(records):04d}_{stage_id}_p{engine.pass_index}_ae{ae_pass}"
        if isinstance(value, np.ndarray):
            record["array"] = save_array(output / f"{prefix}.npy", value)
        elif isinstance(value, tuple):
            members = []
            for i, part in enumerate(value):
                members.append(
                    {"array": save_array(output / f"{prefix}_{i}.npy", part)}
                    if isinstance(part, np.ndarray) else {"value": jsonable(part)}
                )
            record["members"] = members
        else:
            record["value"] = jsonable(value)
        records.append(record)
        return value

    BrilliantISP._run_stage = observed
    try:
        yield records
    finally:
        BrilliantISP._run_stage = original


def capture_reference(source: Path, config_paths: list[Path], output: Path) -> dict[str, Any]:
    from util.config_merge import load_merged_yaml, pipeline_config_paths
    from util.pipeline_runner import process_frame
    from util.pipeline_types import ProcessingOptions

    output.mkdir(parents=True, exist_ok=False)
    before = source_hashes()
    chain = pipeline_config_paths(config_paths[0]) if len(config_paths) == 1 else config_paths
    source_id = file_identity(source)
    config_ids = [file_identity(p) for p in chain]
    config = load_merged_yaml(chain)
    config_before = json.dumps(jsonable(config), sort_keys=True, allow_nan=False)
    with observe_cpu_stages(output) as stages:
        result = process_frame(source, config, options=ProcessingOptions(run_id="cpu_reference"))
    if result.artifacts:
        raise AssertionError("compute-only CPU reference wrote pipeline artifacts")
    if json.dumps(jsonable(config), sort_keys=True, allow_nan=False) != config_before:
        raise AssertionError("CPU processing mutated source configuration")
    if before != source_hashes() or source_id != file_identity(source):
        raise RuntimeError("CPU source or input changed while capturing reference")
    if config_ids != [file_identity(p) for p in chain]:
        raise RuntimeError("configuration changed while capturing reference")
    record = {
        "schema_version": 1, "source": source_id, "config_chain": config_ids,
        "source_hashes": before, "environment": environment(),
        "harness": file_identity(Path(__file__)), "source_config": config,
        "native": save_array(output / "native.npy", result.native_image),
        "preview": save_array(output / "preview.npy", result.preview_rgb8),
        "metadata": result.native_metadata, "feedback": result.feedback,
        "effective_config": result.effective_config, "migration_report": result.migration_report,
        "warnings": result.warnings, "stages": stages,
    }
    write_json(output / "manifest.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    snap = commands.add_parser("snapshot")
    snap.add_argument("--output", type=Path, required=True)
    capture = commands.add_parser("capture")
    capture.add_argument("--input", type=Path, required=True)
    capture.add_argument("--config", type=Path, action="append", required=True)
    capture.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("MPLBACKEND", "Agg")
    if args.command == "snapshot":
        snapshot(args.output)
    else:
        capture_reference(args.input, args.config, args.output)
    print(args.output / "manifest.json")


if __name__ == "__main__":
    main()
