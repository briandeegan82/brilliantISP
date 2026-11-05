from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import tifffile
import yaml

from tools.gpu_validation.contracts import FINAL_FLOAT
from tools.gpu_validation.metrics import compare_arrays
from util.pipeline_runner import process_frame
from test_microbatch import _config, _sources


REPO = Path(__file__).resolve().parents[2]
LAUNCHER = REPO / "isp_pipeline_gpu_batch.py"


@pytest.fixture
def cli_case(tmp_path):
    cfg = _config()
    cfg["platform"]["short_output_names"] = False
    cfg["sensor_info"].update(
        source_format="raw", storage_format="uint16",
        endian_type="ieee-le", data_alignment="LSB",
        horizontal_flip=False, vertical_flip=False, header_bytes=0,
        row_stride_bytes=None, embedded_rows_top=0, embedded_rows_bottom=0,
    )
    preset = tmp_path / "preset.yml"
    preset.write_text(yaml.safe_dump(cfg))
    paths = []
    for i, source in enumerate(_sources(2)):
        path = tmp_path / f"frame_{i}.raw"
        path.write_bytes(source.astype("<u2").tobytes())
        paths.append(path)
    return tmp_path, preset, paths


def _run(args, *, blocked=False):
    if blocked:
        code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch','tkinter','PyQt5','PyQt6','PySide6','dearpygui'):
            raise AssertionError('forbidden import: '+fullname)
sys.meta_path.insert(0, Block())
from brilliantisp_gpu.cli import main
raise SystemExit(main(sys.argv[1:]))
"""
        command = [sys.executable, "-c", code]
    else:
        command = [sys.executable, str(LAUNCHER)]
    return subprocess.run(
        command + list(map(str, args)), cwd=REPO, capture_output=True, text=True,
        env={**os.environ, "DISPLAY": "", "MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1"},
    )


def _manifest(root, rows):
    path = root / "batch.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def _read_report(path):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    return records, json.loads(path.with_suffix(".summary.json").read_text())


def test_cli_manifest_relative_paths_and_saved_native_roundtrip(cuda_device, cli_case):
    root, preset, paths = cli_case
    manifest = _manifest(root, [
        dict(id=f"frame_{i}", input=path.name, config=preset.name)
        for i, path in enumerate(paths)
    ])
    report, output = root / "report.jsonl", root / "output"
    original = {p: p.read_bytes() for p in [preset, *paths]}
    result = _run([
        "--manifest", manifest, "--output", output, "--report", report,
        "--device", cuda_device, "--batch-size", "2",
    ])
    assert result.returncode == 0, result.stdout + result.stderr
    records, summary = _read_report(report)
    assert [r["item_id"] for r in records] == ["frame_0", "frame_1"]
    assert [r["status"] for r in records] == ["success", "success"]
    assert summary["status_counts"] == dict(success=2, failed=0, cancelled=0, blocked=0)
    assert summary["successful_batches_by_size"] == {"2": 1}
    for path, record in zip(paths, records):
        native_path, = [Path(p) for p in record["artifacts"] if Path(p).suffix == ".tiff"]
        actual = tifffile.imread(native_path)
        expected = process_frame(path, yaml.safe_load(preset.read_text())).native_image
        assert compare_arrays(expected, actual, FINAL_FLOAT)["passed"]
        assert record["metadata"]["source_identity"] == str(path)
        assert "native_image" not in record and "captures" not in record
    assert all(path.read_bytes() == data for path, data in original.items())


@pytest.mark.parametrize("invalid", (
    "unknown", "duplicate", "traversal", "windows", "index", "boolean", "empty_config",
))
def test_cli_rejects_malformed_manifests_before_files_or_gpu(cli_case, invalid):
    root, preset, paths = cli_case
    rows = [dict(id="first", input=paths[0].name, config=preset.name)]
    if invalid == "unknown":
        rows[0]["surprise"] = True
    elif invalid == "duplicate":
        rows.append(dict(rows[0]))
    elif invalid in ("traversal", "windows"):
        rows[0]["output_subdir"] = "../escape" if invalid == "traversal" else r"C:\escape"
    elif invalid == "boolean":
        rows[0].update(sequence_id="s", frame_index=True)
    elif invalid == "empty_config":
        rows[0]["config"] = ""
    else:
        rows[0].update(sequence_id="s", frame_index=2)
        rows.append(dict(id="second", input=paths[1].name, config=preset.name,
                         sequence_id="s", frame_index=1))
    manifest = _manifest(root, rows)
    output, report = root / "must_not_exist", root / "report.jsonl"
    result = _run(["--manifest", manifest, "--output", output, "--report", report], blocked=True)
    assert result.returncode == 2, result.stdout + result.stderr
    assert not output.exists() and not report.exists()


@pytest.mark.parametrize("fail_fast", (False, True))
def test_cli_runtime_failure_and_fail_fast_reporting(cuda_device, cli_case, fail_fast):
    root, preset, paths = cli_case
    manifest = _manifest(root, [
        dict(id="missing", input="absent.raw", config=preset.name),
        dict(id="valid", input=paths[1].name, config=preset.name),
    ])
    report, output = root / "report.jsonl", root / "output"
    args = ["--manifest", manifest, "--output", output, "--report", report,
            "--device", cuda_device]
    if fail_fast:
        args.append("--fail-fast")
    result = _run(args)
    assert result.returncode == 1, result.stdout + result.stderr
    records, summary = _read_report(report)
    assert [r["status"] for r in records] == ["failed", "blocked" if fail_fast else "success"]
    assert summary["items"] == 2 and summary["exit_code"] == 1
    assert bool(list(output.rglob("*.tiff"))) != fail_fast


def test_cli_existing_report_is_preserved_without_gpu(cli_case):
    root, preset, paths = cli_case
    manifest = _manifest(root, [dict(id="one", input=paths[0].name, config=preset.name)])
    report = root / "report.jsonl"
    report.write_bytes(b"existing report\n")
    result = _run(["--manifest", manifest, "--output", root / "output",
                   "--report", report], blocked=True)
    assert result.returncode == 2, result.stdout + result.stderr
    assert report.read_bytes() == b"existing report\n"
    assert not list((root / "output").rglob("*.tiff"))


def test_cli_help_and_empty_folder_are_headless_and_torch_optional(cli_case):
    root, preset, paths = cli_case
    result = _run(["--help"], blocked=True)
    assert result.returncode == 0, result.stdout + result.stderr
    empty = root / "empty"
    empty.mkdir()
    report = root / "report.jsonl"
    result = _run(["--input", empty, "--config", preset, "--output", root / "output",
                   "--report", report], blocked=True)
    assert result.returncode == 0, result.stdout + result.stderr
    records, summary = _read_report(report)
    assert records == [] and summary["items"] == 0 and summary["exit_code"] == 0


def test_cli_recursive_folder_preserves_subdirs_and_excludes_outputs(cuda_device, cli_case):
    root, preset, paths = cli_case
    inputs = root / "inputs"
    nested = inputs / "nested"
    output = inputs / "gpu_out"
    nested.mkdir(parents=True)
    output.mkdir()
    (inputs / "same.raw").write_bytes(paths[0].read_bytes())
    (nested / "same.raw").write_bytes(paths[1].read_bytes())
    sentinel = output / "Out_same.tiff"
    sentinel.write_bytes(b"pre-existing output must survive")
    report = root / "report.jsonl"
    result = _run([
        "--input", inputs, "--recursive", "--config", preset,
        "--output", output, "--report", report, "--device", cuda_device,
    ])
    assert result.returncode == 0, result.stdout + result.stderr
    records, summary = _read_report(report)
    assert [r["item_id"] for r in records] == ["nested/same.raw", "same.raw"]
    parents = {
        Path(p).parent for r in records for p in r["artifacts"] if Path(p).suffix == ".tiff"
    }
    assert parents == {output, output / "nested"}
    assert sentinel.read_bytes() == b"pre-existing output must survive"
    assert summary["status_counts"]["success"] == 2


@pytest.mark.parametrize("extra", (
    ("--batch-size", "0"), ("--batch-size", "8", "--max-batch-size", "4"),
    ("--device", "cpu"),
))
def test_cli_invalid_options_are_invocation_errors_without_gpu(cli_case, extra):
    root, preset, paths = cli_case
    manifest = _manifest(root, [dict(id="one", input=paths[0].name, config=preset.name)])
    output = root / "output"
    report = root / "report.jsonl"
    result = _run(["--manifest", manifest, "--output", output, "--report", report,
                   *extra], blocked=True)
    assert result.returncode == 2, result.stdout + result.stderr
    assert not output.exists() and not report.exists()


def test_cli_interrupt_reports_all_unprocessed_items_as_cancelled(cli_case, monkeypatch):
    import brilliantisp_gpu.cli as cli

    root, preset, paths = cli_case
    manifest = _manifest(root, [
        dict(id=f"frame_{i}", input=path.name, config=preset.name)
        for i, path in enumerate(paths)
    ])
    report, output = root / "report.jsonl", root / "output"

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt
        yield None

    monkeypatch.setattr(cli, "process_batch", interrupted)
    assert cli.main(["--manifest", str(manifest), "--output", str(output),
                     "--report", str(report)]) == 130
    records, summary = _read_report(report)
    assert [r["status"] for r in records] == ["cancelled", "cancelled"]
    assert summary["interrupted"] and summary["exit_code"] == 130
    assert not list(output.rglob("*.tiff"))


def test_cli_default_report_creates_new_output_root(cuda_device, cli_case):
    root, preset, paths = cli_case
    output = root / "new_output"
    result = _run([
        "--input", root, "--config", preset, "--output", output,
        "--device", cuda_device,
    ])
    assert result.returncode == 0, result.stdout + result.stderr
    reports = list(output.glob("gpu_batch_*.jsonl"))
    assert len(reports) == 1
    records, summary = _read_report(reports[0])
    assert len(records) == 2 and summary["status_counts"]["success"] == 2


def test_cli_output_subdirectory_cannot_escape_through_symlink(cli_case):
    root, preset, paths = cli_case
    output, outside = root / "output", root / "outside"
    output.mkdir()
    outside.mkdir()
    (output / "escape").symlink_to(outside, target_is_directory=True)
    manifest = _manifest(root, [
        dict(id="one", input=paths[0].name, config=preset.name, output_subdir="escape"),
    ])
    report = root / "report.jsonl"
    result = _run([
        "--manifest", manifest, "--output", output, "--report", report,
    ], blocked=True)
    assert result.returncode == 2, result.stdout + result.stderr
    assert not report.exists() and not list(outside.iterdir())


@pytest.mark.parametrize("mode", ("folder", "manifest", "manifest_relative"))
def test_cli_preserves_single_path_versus_explicit_list_semantics(cli_case, monkeypatch, mode):
    import brilliantisp_gpu.cli as cli
    from brilliantisp_gpu.types import BatchError, BatchOutcome

    root, preset, paths = cli_case
    seen = []

    def probe(items, **kwargs):
        for item in items:
            seen.append(item.config)
            yield BatchOutcome(
                item.item_id, "failed", None,
                BatchError("processing", "Probe", "configuration shape probe"),
                {}, -1, 0, (), {},
            )

    monkeypatch.setattr(cli, "process_batch", probe)
    if mode == "folder":
        args = ["--input", str(root), "--config", str(preset)]
    else:
        manifest = _manifest(root, [
            dict(id="single", input=paths[0].name, config=preset.name),
            dict(id="list", input=paths[1].name, config=[preset.name]),
        ])
        if mode == "manifest_relative":
            monkeypatch.chdir(root)
            args = ["--manifest", manifest.name]
        else:
            args = ["--manifest", str(manifest)]
    code = cli.main([
        *args, "--output", str(root / "output"), "--report", str(root / "report.jsonl"),
    ])
    assert code == 1 and len(seen) == 2
    assert isinstance(seen[0], (str, Path))
    assert Path(seen[0]).is_absolute()
    if mode != "folder":
        assert isinstance(seen[1], list) and len(seen[1]) == 1
    else:
        assert all(isinstance(config, (str, Path)) for config in seen)


def test_cli_report_pair_reservation_is_exclusive_and_rolls_back_own_file(
    cli_case, monkeypatch,
):
    import brilliantisp_gpu.cli as cli

    root, preset, paths = cli_case
    manifest = _manifest(root, [dict(id="one", input=paths[0].name, config=preset.name)])
    report = root / "report.jsonl"
    summary = report.with_suffix(".summary.json")
    original = Path.open
    raced = []

    def competing_open(path, mode="r", *args, **kwargs):
        if path == summary and mode == "x":
            with original(path, "w") as stream:
                stream.write("competing summary\n")
            raced.append(True)
            raise FileExistsError("injected competing reservation")
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", competing_open)
    with pytest.raises(SystemExit) as exc:
        cli.main([
            "--manifest", str(manifest), "--output", str(root / "output"),
            "--report", str(report),
        ])
    assert exc.value.code == 2 and raced
    assert not report.exists()
    assert summary.read_text() == "competing summary\n"


@pytest.mark.parametrize("failure", ("foreign_id", "early_end", "early_end_failfast"))
def test_cli_unexpected_stream_failure_does_not_lose_first_pending_item(
    cli_case, monkeypatch, failure,
):
    import brilliantisp_gpu.cli as cli
    from brilliantisp_gpu.types import BatchError, BatchOutcome

    root, preset, paths = cli_case
    manifest = _manifest(root, [
        dict(id=f"frame_{i}", input=path.name, config=preset.name)
        for i, path in enumerate(paths)
    ])
    report = root / "report.jsonl"

    def broken(*args, **kwargs):
        if failure.startswith("early_end"):
            return
        yield BatchOutcome(
            "foreign_id", "failed", None,
            BatchError("processing", "Probe", "unexpected identity"),
            {}, -1, 0, (), {},
        )

    monkeypatch.setattr(cli, "process_batch", broken)
    assert cli.main([
        "--manifest", str(manifest), "--output", str(root / "output"),
        "--report", str(report),
        *(["--fail-fast"] if failure == "early_end_failfast" else []),
    ]) == 1
    records, summary = _read_report(report)
    assert [r["item_id"] for r in records] == ["frame_0", "frame_1"]
    assert [r["status"] for r in records] == ["failed", "blocked"]
    assert summary["items"] == 2 and summary["exit_code"] == 1


def test_cli_invalid_manifest_encoding_is_an_invocation_error(cli_case):
    root, preset, paths = cli_case
    manifest = root / "invalid.jsonl"
    manifest.write_bytes(b"\xff\xfe")
    output, report = root / "output", root / "report.jsonl"
    result = _run([
        "--manifest", manifest, "--output", output, "--report", report,
    ], blocked=True)
    assert result.returncode == 2, result.stdout + result.stderr
    assert not output.exists() and not report.exists()


def test_cli_output_root_file_is_rejected_before_report_creation(cli_case):
    root, preset, paths = cli_case
    manifest = _manifest(root, [dict(id="one", input=paths[0].name, config=preset.name)])
    output, report = root / "output", root / "report.jsonl"
    output.write_bytes(b"existing non-directory")
    result = _run([
        "--manifest", manifest, "--output", output, "--report", report,
    ], blocked=True)
    assert result.returncode == 2, result.stdout + result.stderr
    assert output.read_bytes() == b"existing non-directory" and not report.exists()


@pytest.mark.parametrize("signal", ("normal", "interrupt"))
def test_cli_report_write_failure_does_not_retry_or_leak_handles(
    cli_case, monkeypatch, capsys, signal,
):
    import brilliantisp_gpu.cli as cli
    from brilliantisp_gpu.types import BatchError, BatchOutcome

    root, preset, paths = cli_case
    manifest = _manifest(root, [dict(id="one", input=paths[0].name, config=preset.name)])
    report = root / "report.jsonl"
    original, wrapped = Path.open, []

    class BrokenSink:
        def __init__(self, stream):
            self.stream, self.writes = stream, 0

        def write(self, text):
            self.writes += 1
            raise OSError("injected report write failure")

        def __getattr__(self, name):
            return getattr(self.stream, name)

    def open_sink(path, mode="r", *args, **kwargs):
        stream = original(path, mode, *args, **kwargs)
        if path == report and mode == "x":
            stream = BrokenSink(stream)
            wrapped.append(stream)
        return stream

    def probe(items, **kwargs):
        if signal == "interrupt":
            raise KeyboardInterrupt
        for item in items:
            yield BatchOutcome(
                item.item_id, "failed", None,
                BatchError("input", "Probe", "synthetic input failure"),
                {}, -1, 0, (), {},
            )

    monkeypatch.setattr(Path, "open", open_sink)
    monkeypatch.setattr(cli, "process_batch", probe)
    assert cli.main([
        "--manifest", str(manifest), "--output", str(root / "output"),
        "--report", str(report),
    ]) == (130 if signal == "interrupt" else 1)
    assert wrapped and wrapped[0].writes == 1 and wrapped[0].stream.closed
    assert "injected report write failure" in capsys.readouterr().err
