"""Headless folder/manifest batch CLI over the qualified process_batch API.

``main(argv)`` returns a process exit code: 0 all-success, 1 any
failed/blocked/cancelled, 130 interrupt, 2 structural/invocation errors
(argparse). Imports ``process_batch`` at module scope; torch is resolved
lazily only when a nonempty compute set exists.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections import deque
from contextlib import closing
from pathlib import Path
import time
import uuid
from threading import Event

from util.pipeline_types import ProcessingOptions

from .batch import process_batch
from .reporting import outcome_record, summarize_records
from .types import (
    BatchError, BatchItem, BatchOutcome, GpuOptions)

_ALLOWED_ROW_KEYS = frozenset(
    ("id", "input", "config", "output_subdir", "sequence_id", "frame_index"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="isp_pipeline_gpu_batch",
        description="Headless GPU batch pipeline (folder or JSONL manifest).")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--config", nargs="+", required=False, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--sequence", action="store_true")
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--memory-fraction", type=float, default=0.70)
    parser.add_argument("--profile-stages", action="store_true")
    parser.add_argument("--no-split-on-oom", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    return parser


def _invoke_error(parser, message):
    parser.error(message)


def _gpu_options(args, parser) -> GpuOptions:
    batch_raw = str(args.batch_size)
    if batch_raw == "auto":
        batch = args.max_batch_size
    else:
        try:
            batch = int(batch_raw)
        except ValueError:
            _invoke_error(parser, "--batch-size must be 'auto' or a positive integer")
        if batch < 1:
            _invoke_error(parser, "--batch-size must be a positive integer")
        if batch > args.max_batch_size:
            _invoke_error(parser,
                          "--batch-size exceeds --max-batch-size")
    try:
        return GpuOptions(
            device=str(args.device),
            max_batch_size=min(batch, args.max_batch_size),
            memory_fraction=float(args.memory_fraction),
            prefetch=int(args.prefetch),
            profile_stages=bool(args.profile_stages),
            split_on_oom=not args.no_split_on_oom,
        )
    except (TypeError, ValueError) as exc:
        _invoke_error(parser, str(exc))


def _check_output_subdir(value):
    if not isinstance(value, str) or not value:
        raise ValueError("output_subdir must be a nonempty string")
    if "\x00" in value:
        raise ValueError("output_subdir contains NUL")
    if "\\" in value or ":" in value:
        raise ValueError("output_subdir must be a POSIX-style relative path")
    path = Path(value)
    if path.is_absolute() or path.drive or path.anchor not in ("",):
        raise ValueError("output_subdir must be relative")
    if any(part == ".." for part in path.parts):
        raise ValueError("output_subdir must not contain '..'")
    return path


def _nonempty_path_str(value, name, line_no, parser):
    if not isinstance(value, str) or not value or "\x00" in value:
        _invoke_error(
            parser, f"manifest line {line_no}: {name} must be a "
            "nonempty path string")
    return value


def _parse_manifest(path: Path, parser):
    """Structural prevalidation of a JSONL manifest (no GPU/FS writes)."""
    seen = set()
    sequence_last = {}
    rows = []
    try:
        lines = path.read_text().splitlines()
    except UnicodeDecodeError as exc:
        _invoke_error(parser, f"manifest {path} is not valid UTF-8: {exc}")
    except OSError as exc:
        _invoke_error(parser, f"cannot read manifest {path}: {exc}")
    base = path.parent.resolve()
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            _invoke_error(
                parser, f"manifest line {line_no} is not a JSON object: {exc}")
        if not isinstance(row, dict):
            _invoke_error(parser, f"manifest line {line_no} is not an object")
        unknown = set(row) - _ALLOWED_ROW_KEYS
        if unknown:
            _invoke_error(
                parser, f"manifest line {line_no} has unknown keys "
                f"{sorted(unknown)}")
        for required in ("id", "input", "config"):
            if required not in row:
                _invoke_error(
                    parser, f"manifest line {line_no} missing {required!r}")
        item_id = row["id"]
        if not isinstance(item_id, str) or not item_id or "\x00" in item_id:
            _invoke_error(
                parser, f"manifest line {line_no}: id must be nonempty")
        if item_id in seen:
            _invoke_error(
                parser, f"manifest line {line_no}: duplicate item id "
                f"{item_id!r}")
        seen.add(item_id)
        source = _nonempty_path_str(row["input"], "input", line_no, parser)
        config = row["config"]
        if isinstance(config, str):
            _nonempty_path_str(config, "config", line_no, parser)
            config_spec: object = (
                config if Path(config).is_absolute()
                else str(base / config))
        elif isinstance(config, list):
            if not config:
                _invoke_error(
                    parser, f"manifest line {line_no}: config list must be "
                    "nonempty")
            config_spec = [
                c if Path(c).is_absolute() else str(base / c)
                for c in (
                    _nonempty_path_str(c, "config", line_no, parser)
                    for c in config)]
        else:
            _invoke_error(
                parser, f"manifest line {line_no}: config must be a path "
                "string or nonempty list of paths")
        subdir = None
        if "output_subdir" in row:
            try:
                subdir = _check_output_subdir(row["output_subdir"])
            except ValueError as exc:
                _invoke_error(parser, f"manifest line {line_no}: {exc}")
        sequence_id = row.get("sequence_id")
        frame_index = row.get("frame_index")
        if sequence_id is not None:
            if not isinstance(sequence_id, str) or not sequence_id:
                _invoke_error(
                    parser, f"manifest line {line_no}: sequence_id must be "
                    "a nonempty string")
            if isinstance(frame_index, bool) or not isinstance(
                    frame_index, int) or frame_index < 0:
                _invoke_error(
                    parser, f"manifest line {line_no}: frame_index must be "
                    "a nonnegative integer")
            previous = sequence_last.get(sequence_id)
            if previous is not None and frame_index <= previous:
                _invoke_error(
                    parser, f"manifest line {line_no}: sequence frame_index "
                    "must be strictly increasing")
            sequence_last[sequence_id] = frame_index
        elif frame_index is not None:
            _invoke_error(
                parser, f"manifest line {line_no}: frame_index requires "
                "sequence_id")
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = base / source_path
        try:
            item = BatchItem(
                item_id=item_id, source=source_path, config=config_spec,
                sequence_id=sequence_id, frame_index=frame_index)
        except (TypeError, ValueError) as exc:
            _invoke_error(parser, f"manifest line {line_no}: {exc}")
        rows.append((item, subdir))
    return rows


def _input_for_bayer(path: Path) -> bool:
    """Static decoder metadata check: Bayer RAW extensions only."""
    try:
        from util.raw_decode import resolve_source_format
        resolve_source_format(path, "auto")
        return True
    except Exception:
        return False


def _scan_folder(root: Path, recursive: bool, parser):
    """Lexically-sorted candidate frame paths (relative names as ids)."""
    if not root.is_dir():
        _invoke_error(parser, f"input folder {root} does not exist")
    iterator = root.rglob("*") if recursive else root.glob("*")
    candidates = sorted(
        (p for p in iterator if p.is_file() and _input_for_bayer(p)),
        key=lambda p: p.relative_to(root).as_posix())
    return candidates


def _reserve_pair(report_path: Path, summary_path: Path, parser):
    """Exclusive creation of the report/summary pair; returns open handles."""
    if report_path.exists() or summary_path.exists():
        _invoke_error(
            parser, f"report target already exists: {report_path}")
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        report_fh = report_path.open("x", encoding="utf-8")
    except (FileExistsError, OSError) as exc:
        _invoke_error(
            parser, f"cannot reserve report {report_path}: {exc}")
    try:
        summary_fh = summary_path.open("x", encoding="utf-8")
    except (FileExistsError, OSError) as exc:
        report_fh.close()
        try:
            report_path.unlink()
        except OSError:
            pass
        _invoke_error(
            parser, f"cannot reserve summary {summary_path}: {exc}")
    return report_fh, summary_fh


class _ReportWriteError(Exception):
    """Internal marker: the report sink failed; never retried."""


def _synth_outcome(item_id, kind, message, device):
    status = {"blocked": "blocked", "cancelled": "cancelled"}.get(
        kind, "failed")
    return BatchOutcome(
        item_id=item_id,
        status=status,
        result=None,
        error=BatchError(kind, "Synthesized", message),
        device=device,
        batch_index=-1, batch_size=0, attempts=(),
        metrics={
            "decode_ns": 0, "compute_transfer_group_ns": 0, "commit_ns": 0,
            "profile_stages": False, "estimated_peak_bytes": 0,
            "selection_budget_bytes": 0,
            "single_frame_over_estimate": False, "free_bytes": 0,
            "total_bytes": 0, "planned_group_size": 0,
        })


def _resolve_outdir(output_root, subdir, parser):
    out_dir = output_root / subdir if subdir is not None else output_root
    try:
        resolved_root = output_root.resolve()
        resolved_dir = out_dir.resolve()
    except OSError as exc:
        _invoke_error(parser, f"cannot resolve output path {out_dir}: {exc}")
    if resolved_dir != resolved_root and not resolved_dir.is_relative_to(
            resolved_root):
        _invoke_error(
            parser, f"output_subdir escapes --output root: {subdir}")
    return out_dir


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)

    if args.manifest is not None and args.config is not None:
        _invoke_error(parser, "--config is only valid with --input")
    if args.manifest is not None and (args.recursive or args.sequence):
        _invoke_error(
            parser, "--recursive/--sequence are only valid with --input")
    if args.input is not None and args.config is None:
        _invoke_error(parser, "--input requires --config")

    gpu_options = _gpu_options(args, parser)

    output_root = args.output
    report_path = args.report or (
        output_root / f"gpu_batch_{uuid.uuid4().hex}.jsonl")
    summary_path = report_path.with_suffix(".summary.json")

    token = Event()
    options = ProcessingOptions(
        run_id=f"gpu_batch_{uuid.uuid4().hex}", save_outputs=True, diagnostics=bool(
            args.diagnostics), output_dir=output_root,
        cancellation_token=token)

    if args.manifest is not None:
        rows = _parse_manifest(args.manifest, parser)
        items = [
            dataclasses.replace(
                item, output_dir=_resolve_outdir(output_root, subdir, parser))
            for item, subdir in rows]
    else:
        folder = args.input.resolve()
        out_resolved = output_root.resolve()
        if out_resolved == folder or out_resolved in folder.parents:
            _invoke_error(
                parser, "--output must not equal or contain --input")
        candidates = _scan_folder(folder, bool(args.recursive), parser)
        candidates = [
            p for p in candidates
            if out_resolved != p.resolve() and out_resolved not in
            p.resolve().parents]
        config_args = [str(Path(c).resolve()) for c in args.config]
        config_spec = config_args[0] if len(config_args) == 1 else config_args
        items = []
        for index, path in enumerate(candidates):
            relative = path.relative_to(folder).as_posix()
            sub = path.parent.relative_to(folder)
            sub = None if str(sub) == "." else sub
            item = BatchItem(
                item_id=relative,
                source=path,
                config=config_spec,
                sequence_id="batch" if args.sequence else None,
                frame_index=index if args.sequence else None,
                output_dir=_resolve_outdir(output_root, sub, parser),
            )
            items.append(item)

    if output_root.exists() and not output_root.is_dir():
        _invoke_error(parser, f"--output {output_root} exists and is not a directory")
    report_fh, summary_fh = _reserve_pair(report_path, summary_path, parser)

    records = []
    start = time.perf_counter_ns()
    interrupted = False
    device = {"requested": gpu_options.device, "vendor": None}
    pending_ids = deque(item.item_id for item in items)
    item_by_id = {item.item_id: item for item in items}
    stream = None

    def record_one(item, outcome):
        record = outcome_record(item, outcome, index=len(records))
        line = json.dumps(record, allow_nan=False) + "\n"
        try:
            report_fh.write(line)
            report_fh.flush()
        except Exception as exc:
            raise _ReportWriteError(
                f"report write failure: {exc}") from exc
        records.append(record)

    def remaining(kind, message, *, first_kind=None):
        first = True
        while pending_ids:
            pid = pending_ids[0]
            synth = _synth_outcome(
                pid, first_kind if first and first_kind else kind, message, device)
            record_one(item_by_id[pid], synth)
            pending_ids.popleft()
            first = False

    with closing(report_fh), closing(summary_fh):
        try:
            try:
                if items:
                    stream = process_batch(iter(items), gpu=gpu_options,
                                           options=options)
                    for outcome in stream:
                        if not pending_ids or outcome.item_id != pending_ids[0]:
                            raise RuntimeError(
                                f"unexpected outcome for {outcome.item_id!r}")
                        record_one(item_by_id[outcome.item_id], outcome)
                        pending_ids.popleft()
                        device = outcome.device
                        if outcome.status != "success" and args.fail_fast:
                            stream.close()
                            remaining("blocked", "fail-fast: skipped after earlier failure")
                            break
                    if pending_ids:
                        raise RuntimeError(
                            "batch stream ended before all items were processed")
            except KeyboardInterrupt:
                token.set()
                interrupted = True
                if stream is not None:
                    stream.close()
                remaining("cancelled", "interrupted")
            except _ReportWriteError:
                raise
            except Exception as exc:
                if stream is not None:
                    stream.close()
                if not pending_ids:
                    raise
                remaining(
                    "blocked", f"batch aborted: {exc}",
                    first_kind="io" if isinstance(exc, OSError) else "processing")
            wall_ns = time.perf_counter_ns() - start
            summary = summarize_records(records, wall_ns=wall_ns,
                                        interrupted=interrupted)
            summary_fh.write(json.dumps(summary, allow_nan=False) + "\n")
            summary_fh.flush()
        except KeyboardInterrupt:
            token.set()
            print("interrupted while writing batch reports", file=sys.stderr)
            return 130
        except Exception as exc:
            print(f"batch reporting failed: {exc}", file=sys.stderr)
            return 130 if interrupted else 1
        finally:
            if stream is not None:
                stream.close()
    print(json.dumps({
        "report": str(report_path),
        "summary": str(summary_path),
        "status_counts": summary["status_counts"],
    }))
    return summary["exit_code"]
