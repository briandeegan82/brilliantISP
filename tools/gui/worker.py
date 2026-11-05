"""Spawn worker entry point and job handlers (no Tk, no GUI imports)."""
from __future__ import annotations


def worker_main(req_q, evt_q, cancel_event, stop_event) -> None:
    import os
    os.environ["MPLBACKEND"] = "Agg"
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[var] = "1"
    import cv2  # noqa: F401
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    import json
    import queue as _queue
    from .protocol import JobRequest, WorkerEvent

    def emit(evt: WorkerEvent, terminal: bool = False) -> None:
        while True:
            try:
                if terminal:
                    evt_q.put(evt, timeout=0.5)
                else:
                    evt_q.put_nowait(evt)
                return
            except _queue.Full:
                if terminal and not stop_event.is_set():
                    continue
                return

    while not stop_event.is_set():
        try:
            req: JobRequest = req_q.get(timeout=0.1)
        except _queue.Empty:
            continue
        try:
            _dispatch(req, cancel_event, stop_event, emit)
        except Exception:
            import traceback
            from util.pipeline_types import ProcessingCancelled
            try:
                emit(WorkerEvent(req.tag, "error", json.dumps(
                    {"error": traceback.format_exc(limit=8)})),
                    terminal=True)
            except Exception:
                pass


def _dispatch(req, cancel_event, stop_event, emit) -> None:
    import json
    from .protocol import WorkerEvent

    cfg = json.loads(req.config_json) if req.config_json else {}
    try:
        if req.operation == "process":
            _run_process(req, cfg, cancel_event, emit)
        elif req.operation == "import":
            _run_import(req, cfg, cancel_event, emit)
        elif req.operation == "export":
            _run_export(req, cancel_event, emit)
        elif req.operation == "analysis":
            _run_analysis(req, cancel_event, emit)
        elif req.operation == "render":
            _run_render(req, cancel_event, emit)
        else:
            emit(WorkerEvent(req.tag, "error",
                             json.dumps({"error": f"unknown op {req.operation!r}"})),
                 terminal=True)
    except _cancelled():
        emit(WorkerEvent(req.tag, "cancelled", "{}"), terminal=True)


def _cancelled():
    from util.pipeline_types import ProcessingCancelled
    return ProcessingCancelled


def _check(cancel_event) -> None:
    if cancel_event.is_set():
        raise _cancelled()()


def _verified_source(req) -> tuple[int, int, int, int, int]:
    import hashlib
    from pathlib import Path

    expected = req.tag.source_fingerprint
    if len(expected) != 64 or any(value not in "0123456789abcdef" for value in expected):
        raise ValueError("Processing requires a SHA-256 source fingerprint; reload the source.")
    path = Path(req.source_path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    signature = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if signature(before) != signature(after) or digest.hexdigest() != expected:
        raise ValueError("Staged source changed; reload the source before processing.")
    return signature(after)


def _run_process(req, cfg, cancel_event, emit) -> None:
    import dataclasses
    import json
    import os
    from pathlib import Path

    import cv2
    import numpy as np
    from .protocol import WorkerEvent
    from util.pipeline_runner import process_frame
    from util.pipeline_types import ProcessingOptions, StageEvent

    _check(cancel_event)
    source_signature = _verified_source(req)
    out_dir = Path(req.output_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    def progress(ev: StageEvent) -> None:
        try:
            emit(WorkerEvent(req.tag, "progress", json.dumps({
                "stage_id": ev.stage_id, "status": ev.status,
                "pass_index": ev.pass_index, "ae_pass": ev.ae_pass,
                "elapsed_ns": ev.elapsed_ns})))
        except Exception:
            pass

    result = process_frame(
        req.source_path, cfg,
        options=ProcessingOptions(
            run_id=req.tag.request_id,
            capture_stages=req.capture_stages,
            save_outputs=False,
            diagnostics=False,
            output_dir=out_dir,
            cancellation_token=cancel_event,
            progress_callback=progress))
    _check(cancel_event)
    if _verified_source(req) != source_signature:
        raise ValueError("Staged source changed during processing; reload the source.")

    def save_npy(name: str, arr) -> str:
        p = out_dir / name
        tmp = out_dir / (name + ".part")
        with open(tmp, "xb") as fh:
            np.save(fh, arr, allow_pickle=False)
        tmp.replace(p)
        return name

    paths = [save_npy("native.npy", result.native_image),
             save_npy("preview.npy", result.preview_rgb8)]
    captures = []
    for name, arr in result.captures.items():
        fname = f"capture_{name}.npy"
        captures.append({"stage": name, "path": fname})
        paths.append(save_npy(fname, arr))
    _check(cancel_event)
    meta = {
        "schema_version": 1,
        "tag": dataclasses.asdict(req.tag),
        "request_config": cfg,
        "source_fingerprint": req.tag.source_fingerprint,
        "files": {"native": "native.npy", "preview": "preview.npy",
                  "captures": captures},
        "native_metadata": dataclasses.asdict(result.native_metadata),
        "effective_config": result.effective_config,
        "feedback": result.feedback,
        "warnings": list(result.warnings),
        "timings": [dataclasses.asdict(t) for t in result.timings],
        "migration_report": result.migration_report,
        "runtime": {
            "pid": os.getpid(), "matplotlib_backend_environment": os.environ.get("MPLBACKEND"),
            "thread_limits": {
                v: os.environ.get(v) for v in (
                    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
                )
            },
            "opencv_threads": cv2.getNumThreads(), "opencv_opencl": cv2.ocl.useOpenCL(),
        },
    }
    tmp = out_dir / "result.json.part"
    tmp.write_text(json.dumps(meta, allow_nan=False))
    tmp.replace(out_dir / "result.json")
    _check(cancel_event)
    emit(WorkerEvent(req.tag, "done",
                     json.dumps({"result_path": str(out_dir / "result.json")})),
         terminal=True)


def _run_import(req, cfg, cancel_event, emit) -> None:
    import hashlib
    import json
    import shutil
    from pathlib import Path

    from .protocol import WorkerEvent
    from util.raw_io import read_raw_payload

    src = Path(req.source_path)
    suffix = src.suffix.lower() or ".raw"
    _check(cancel_event)
    out_dir = Path(req.output_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    dst = out_dir / f"source{suffix}"
    before = src.stat()
    _check(cancel_event)
    with open(src, "rb") as fin, open(dst, "xb") as fout:
        shutil.copyfileobj(fin, fout)
    after = src.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("source changed during import copy")
    _check(cancel_event)
    h = hashlib.sha256()
    head = bytearray()
    with open(dst, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            if len(head) < 65536:
                head += chunk[: 65536 - len(head)]
    fingerprint = h.hexdigest()
    _check(cancel_event)
    overrides: dict = {}
    if suffix == ".pgm":
        pgm = read_raw_payload(dst)
        if not pgm.is_pgm:
            raise ValueError(f"{src.name}: not a PGM container")
        width, height, maxval = pgm.pgm_width, pgm.pgm_height, pgm.pgm_maxval
        if width is None or height is None or maxval is None:
            raise ValueError(f"{src.name}: missing PGM geometry")
        container_bits = 16 if maxval > 255 else 8
        sensor = {}
        if isinstance(cfg, dict):
            si = cfg.get("sensor_info")
            if isinstance(si, dict):
                sensor = si
        declared = sensor.get("bit_depth")
        if (isinstance(declared, int) and not isinstance(declared, bool)
                and 0 < declared <= container_bits):
            bit_depth = declared
        else:
            bit_depth = max(1, maxval.bit_length())
        overrides = {
            "width": width, "height": height,
            "bit_depth": bit_depth,
            "storage_format": f"uint{container_bits}",
            "data_format": f"uint{container_bits}",
            "endian_type": "ieee-be", "header_bytes": 0,
            "source_format": "auto",
            "row_stride_bytes": None,
        }
        if bit_depth != declared:
            overrides["data_alignment"] = "LSB"
    record = {
        "path": str(dst), "original_path": str(src),
        "fingerprint": fingerprint, "size": dst.stat().st_size,
        "format_overrides": overrides,
    }
    if dst.stat().st_size == 0:
        raise ValueError("empty source file")
    try:
        import copy as _copy
        import dataclasses
        from tools.gui.artifacts import write_raw_artifact
        prospective = _copy.deepcopy(cfg) if isinstance(cfg, dict) else {}
        si = prospective.get("sensor_info")
        if isinstance(si, dict) and overrides:
            si.update(overrides)
        manifest = write_raw_artifact(
            dst, output_dir=out_dir / "raw_view", config=prospective,
            source_fingerprint=fingerprint,
            tag=dataclasses.asdict(req.tag))
        record["view_manifest"] = str(Path(manifest).parent)
    except Exception as exc:
        record["preview_error"] = str(exc)
    _check(cancel_event)
    emit(WorkerEvent(req.tag, "done", json.dumps({"source": record})),
         terminal=True)


def _run_export(req, cancel_event, emit) -> None:
    import dataclasses
    import json
    from pathlib import Path

    import numpy as np
    from .protocol import WorkerEvent
    from util.output_contract import OutputSpec
    from util.output_writer import write_output
    from util.pipeline_types import FrameMetadata

    payload = json.loads(req.payload_json)
    manifest_dir = Path(payload["manifest_dir"])
    _check(cancel_event)
    result_meta = json.loads((manifest_dir / "result.json").read_text())
    spec = OutputSpec.from_mapping(result_meta["effective_config"]["output"])
    metadata = FrameMetadata(**result_meta["native_metadata"])
    native = np.load(manifest_dir / "native.npy", allow_pickle=False)
    _check(cancel_event)
    artifacts = write_output(
        native, spec, metadata,
        output_dir=Path(payload["dest_dir"]),
        stem=payload["stem"],
        effective_config=result_meta["effective_config"],
        migration_report=result_meta.get("migration_report", {}))
    _check(cancel_event)
    emit(WorkerEvent(req.tag, "done",
                     json.dumps({"artifacts": [str(p) for p in artifacts]})),
         terminal=True)


def _run_analysis(req, cancel_event, emit) -> None:
    import json
    from pathlib import Path

    from .protocol import WorkerEvent
    from tools.gui.artifacts import analyze_artifact

    _check(cancel_event)
    out_dir = Path(req.output_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    payload = json.loads(req.payload_json)
    data = analyze_artifact(payload)
    _check(cancel_event)
    tmp = out_dir / "analysis.json.part"
    with tmp.open("x", encoding="utf-8") as fh:
        json.dump(data, fh, allow_nan=False)
    tmp.replace(out_dir / "analysis.json")
    _check(cancel_event)
    emit(WorkerEvent(req.tag, "done", json.dumps(
        {"analysis_path": str(out_dir / "analysis.json"),
         "descriptor": payload})), terminal=True)


def _run_render(req, cancel_event, emit) -> None:
    import json

    from .protocol import WorkerEvent
    from tools.gui.artifacts import render_comparison

    _check(cancel_event)
    payload = json.loads(req.payload_json)
    path = render_comparison(payload, req.output_dir)
    _check(cancel_event)
    emit(WorkerEvent(req.tag, "done", json.dumps(
        {"render_path": str(path), "descriptor": payload},
        allow_nan=False)), terminal=True)
