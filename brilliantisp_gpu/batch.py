"""Bounded batch scheduler over the qualified microbatch core.

``process_batch`` accepts a lazy item stream on the main thread, does CPU
config/decode work in bounded windows (optionally on one prefetch worker),
groups consecutive compatible independent frames into real batched cohorts
via ``execute_group``, isolates failures, and defers all IO to the commit
phase. Sequence items always run B=1 with private per-sequence state.
"""

from __future__ import annotations

import copy
import dataclasses
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

import numpy as np

from util.pipeline_types import (
    ProcessingCancelled, ProcessingOptions, ProcessingState)

from .api import _commit_state
from .config import prepare_config
from .engine import GpuEngine
from .grouping import group_key
from .memory import plan_memory
from .microbatch import commit_frame, execute_group
from .sources import decode_source
from .types import (
    BatchAttempt, BatchError, BatchItem, BatchOutcome, GpuError,
    GpuOptions, require_torch)


@dataclass
class _Accepted:
    """Main-thread frozen snapshot of one accepted BatchItem."""
    item: BatchItem
    config: Any
    source: Any
    options: ProcessingOptions


@dataclass
class _Loaded:
    """CPU-prepared work item: prepared config + decoded source."""
    accepted: _Accepted
    prepared: Any = None
    decoded: Any = None
    key: Any = None
    coalescible: bool = True
    error: BatchError = None
    decode_ns: int = 0


def _error(exc, kind, stage_id=None):
    return BatchError(kind, type(exc).__name__, str(exc), stage_id)


def _accept(item, base_options) -> _Accepted:
    """Snapshot a just-pulled item before the next pull."""
    opts = dataclasses.replace(
        base_options,
        run_id=f"{base_options.run_id}:{item.item_id}",
        output_dir=item.output_dir
        if item.output_dir is not None else base_options.output_dir,
    )
    cfg = copy.deepcopy(item.config)
    if isinstance(item.source, np.ndarray):
        source = np.array(item.source, copy=True)
    else:
        source = item.source
    return _Accepted(item=item, config=cfg, source=source, options=opts)


def _prepare_one(accepted, profile_stages=False) -> _Loaded:
    """CPU-only prepare_config + decode_source for one item."""
    t0 = time.perf_counter_ns()
    item = accepted.item
    try:
        if (accepted.options.save_outputs or accepted.options.diagnostics) \
                and accepted.options.output_dir is None:
            raise ValueError(
                "save_outputs/diagnostics requires an explicit output_dir")
        prepared = prepare_config(accepted.config,
                                  options=accepted.options)
        decoded = decode_source(
            accepted.source,
            sensor_info=prepared.effective_config["sensor_info"],
            platform=prepared.effective_config["platform"],
            run_id=accepted.options.run_id)
        accepted.source = decoded.pixels
        if item.output_stem is not None:
            decoded = dataclasses.replace(decoded,
                                          out_file=item.output_stem)
        sensor = decoded.sensor_info
        loaded = _Loaded(accepted=accepted, prepared=prepared,
                         decoded=decoded)
        try:
            loaded.key = group_key(
                prepared.effective_config,
                shape=(int(sensor["height"]), int(sensor["width"])),
                cfa=str(sensor["bayer_pattern"]),
                storage_dtype=str(decoded.pixels.dtype.name),
                capture_stages=tuple(accepted.options.capture_stages),
                diagnostics=bool(accepted.options.diagnostics),
                profile_stages=bool(profile_stages))
        except TypeError:
            loaded.coalescible = False
    except Exception as exc:
        return _Loaded(accepted=accepted, error=_error(exc, "input"),
                       decode_ns=time.perf_counter_ns() - t0)
    loaded.decode_ns = time.perf_counter_ns() - t0
    return loaded


def _prepare_window(accepteds, profile_stages=False, stop=None):
    records = []
    for accepted in accepteds:
        if stop is not None and stop.is_set():
            break
        records.append(_prepare_one(accepted, profile_stages))
    return records


_CREDIT_STALL = object()


def _loaded_stream(items, base_options, gpu, credits):
    """Bounded lazy stream of _Loaded records; main-thread acceptance."""
    maximum = gpu.max_batch_size
    it = iter(items)
    seen = set()
    last_index = {}
    source_done = False

    def accept_window():
        nonlocal source_done
        window = []
        while len(window) < maximum and credits["remaining"] > 0:
            try:
                raw = next(it)
            except StopIteration:
                source_done = True
                break
            if not isinstance(raw, BatchItem):
                raise TypeError(
                    f"batch items must be BatchItem, got {type(raw).__name__}")
            if raw.item_id in seen:
                raise ValueError(
                    f"duplicate item_id {raw.item_id!r} in batch stream")
            seen.add(raw.item_id)
            if raw.sequence_id is not None:
                previous = last_index.get(raw.sequence_id)
                if previous is not None and raw.frame_index <= previous:
                    raise ValueError(
                        "sequence frame_index must be strictly increasing")
                last_index[raw.sequence_id] = raw.frame_index
            window.append(_accept(raw, base_options))
            credits["remaining"] -= 1
        if not window and not source_done:
            return _CREDIT_STALL
        return window

    stop = Event()
    if gpu.prefetch <= 0:
        try:
            while True:
                window = accept_window()
                if window is _CREDIT_STALL:
                    raise RuntimeError("batch admission stall")
                if not window:
                    return
                for record in _prepare_window(
                        window, gpu.profile_stages, stop):
                    yield record
        finally:
            stop.set()
        return

    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = deque()
        try:
            while True:
                while len(pending) <= gpu.prefetch:
                    window = accept_window()
                    if window is _CREDIT_STALL:
                        break
                    if not window:
                        break
                    pending.append(executor.submit(
                        _prepare_window, window, gpu.profile_stages, stop))
                if not pending:
                    if not source_done:
                        raise RuntimeError("batch admission stall")
                    return
                future = pending.popleft()
                for record in future.result():
                    yield record
        finally:
            stop.set()
            for future in pending:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)


def _oom(exc, torch):
    if torch is None:
        return False
    kinds = []
    oom = getattr(torch, "OutOfMemoryError", None)
    if oom is not None:
        kinds.append(oom)
    cuda_oom = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
    if cuda_oom is not None:
        kinds.append(cuda_oom)
    return bool(kinds) and isinstance(exc, tuple(kinds))


def _failed_stage(*engines):
    for engine in reversed(engines):
        for event in reversed(engine.stage_events):
            if event.status == "failed":
                return event.stage_id
    return None


def process_batch(items: Iterable[BatchItem], *, gpu=None, options=None
                  ) -> Iterator[BatchOutcome]:
    """Bounded scheduler: real batched cohorts, deferred IO, per-item order."""
    options = dataclasses.replace(
        options if options is not None else ProcessingOptions())
    gpu = gpu if gpu is not None else GpuOptions()

    credits = {"remaining": (gpu.prefetch + 1) * gpu.max_batch_size}
    seq_states: dict[str, ProcessingState] = {}
    seq_blocked: set[str] = set()
    attempt_ids = iter(range(1, 1 << 62))
    batch_counter = 0
    device_info: dict | None = None
    torch_cache = {"torch": None, "device": None}

    def _torch():
        if torch_cache["torch"] is None:
            torch_cache["torch"] = require_torch()
        return torch_cache["torch"]

    def _device():
        if torch_cache["device"] is None:
            from .device import resolve_device
            torch_cache["device"] = resolve_device(gpu.device)
        return torch_cache["device"]

    def _device_info():
        nonlocal device_info
        if device_info is not None:
            return device_info
        from .device import device_properties
        props = device_properties(_device())
        device_info = dict(props)
        device_info["requested"] = gpu.device
        return device_info

    def _metrics(loaded, plan, elapsed, commit_ns):
        record = {
            "decode_ns": int(loaded.decode_ns),
            "compute_transfer_group_ns": int(elapsed),
            "commit_ns": int(commit_ns),
            "profile_stages": bool(gpu.profile_stages),
            "estimated_peak_bytes": 0,
            "selection_budget_bytes": 0,
            "single_frame_over_estimate": False,
            "free_bytes": 0,
            "total_bytes": 0,
            "planned_group_size": 0,
        }
        if plan is not None:
            record.update(
                estimated_peak_bytes=int(plan.estimated_peak_bytes),
                selection_budget_bytes=int(plan.selection_budget_bytes),
                single_frame_over_estimate=bool(
                    plan.single_frame_over_estimate),
                free_bytes=int(plan.free_bytes),
                total_bytes=int(plan.total_bytes),
                planned_group_size=int(plan.group_size))
        return record

    def _outcome(item, status, result, error, batch_index, batch_size,
                 attempts, metrics):
        credits["remaining"] += 1
        return BatchOutcome(
            item_id=item.item_id, status=status, result=result, error=error,
            device=dict(device_info) if device_info is not None
            else {"requested": gpu.device, "vendor": None},
            batch_index=batch_index, batch_size=batch_size,
            attempts=tuple(attempts), metrics=metrics)

    def _attempt_id():
        return next(attempt_ids)

    def _run_cohort(group, batch_index, seq_hint_copies):
        """One execute_group attempt; returns (pendings, error, attempt)."""
        t0 = time.perf_counter_ns()
        engines = []
        torch = None
        try:
            torch = _torch()
            engines = []
            for loaded in group:
                hint = seq_hint_copies.get(loaded.accepted.item.item_id)
                engines.append(GpuEngine(
                    loaded.prepared, loaded.accepted.options, hint, gpu))
            decoded = tuple(l.decoded for l in group)
            device = _device()
            torch.cuda.synchronize(device)
            t0 = time.perf_counter_ns()
            pending = execute_group(tuple(engines), decoded)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter_ns() - t0
            return pending, None, BatchAttempt(
                _attempt_id(),
                tuple(l.accepted.item.item_id for l in group),
                len(group), "success", None, elapsed)
        except ProcessingCancelled as exc:
            elapsed = time.perf_counter_ns() - t0
            error = _error(exc, "cancelled",
                           stage_id=_failed_stage(*engines))
            return None, error, BatchAttempt(
                _attempt_id(),
                tuple(l.accepted.item.item_id for l in group),
                len(group), "cancelled", error, elapsed)
        except Exception as exc:
            elapsed = time.perf_counter_ns() - t0
            from .types import GpuUnavailableError
            if _oom(exc, torch):
                kind = "gpu_oom"
            elif isinstance(exc, GpuUnavailableError):
                kind = "gpu_unavailable"
            else:
                kind = "processing"
            error = _error(exc, kind, stage_id=_failed_stage(*engines))
            return None, error, BatchAttempt(
                _attempt_id(),
                tuple(l.accepted.item.item_id for l in group),
                len(group), "failed", error, elapsed)

    def _compute(group, batch_index, seq_hint_copies):
        """Compute a cohort with OOM split/isolation; returns per-item
        (pending|error, attempts) maps. Never writes or commits."""
        attempts_by_item = {l.accepted.item.item_id: [] for l in group}
        results: dict = {}

        def record(loaded, attempt):
            attempts_by_item[loaded.accepted.item.item_id].append(attempt)

        def recurse(sub):
            pending, error, attempt = _run_cohort(sub, batch_index,
                                                  seq_hint_copies)
            for loaded in sub:
                record(loaded, attempt)
            if error is None:
                for loaded, frame in zip(sub, pending):
                    results[loaded.accepted.item.item_id] = frame
                return
            if error.kind == "cancelled":
                for loaded in sub:
                    results[loaded.accepted.item.item_id] = error
                return
            if error.kind == "gpu_oom":
                device = torch_cache.get("device")
                if device is not None:
                    with torch_cache["torch"].cuda.device(device):
                        torch_cache["torch"].cuda.empty_cache()
            if error.kind == "gpu_oom" and gpu.split_on_oom \
                    and len(sub) > 1:
                half = len(sub) // 2
                recurse(sub[:half])
                recurse(sub[half:])
                return
            if error.kind == "gpu_oom" and len(sub) == 1:
                results[sub[0].accepted.item.item_id] = error
                return
            if error.kind == "gpu_oom":
                for loaded in sub:
                    results[loaded.accepted.item.item_id] = error
                return
            if len(sub) > 1:
                for loaded in sub:
                    recurse_solo(loaded)
                return
            results[sub[0].accepted.item.item_id] = error

        def recurse_solo(loaded):
            pending, error, attempt = _run_cohort(
                (loaded,), batch_index, seq_hint_copies)
            record(loaded, attempt)
            if error is None:
                results[loaded.accepted.item.item_id] = pending[0]
            elif error.kind == "cancelled":
                results[loaded.accepted.item.item_id] = error
            else:
                results[loaded.accepted.item.item_id] = error

        recurse(group)
        return results, attempts_by_item

    buffer: deque = deque()
    stream = _loaded_stream(items, options, gpu, credits)
    stream_done = False

    def _fill(minimum=1):
        nonlocal stream_done
        while not stream_done and len(buffer) < minimum:
            try:
                buffer.append(next(stream))
            except StopIteration:
                stream_done = True

    cancelled_all = False
    try:
        while True:
            _fill()
            if not buffer:
                return
            loaded = buffer.popleft()
            item = loaded.accepted.item
            seq = item.sequence_id
            if seq is not None and seq in seq_blocked:
                yield _outcome(item, "blocked", None, BatchError(
                    "blocked", "BlockedSequence",
                    f"sequence {seq!r} is blocked by an earlier failure"),
                    -1, 0, (), _metrics(loaded, None, 0, 0))
                continue
            if cancelled_all or (
                    options.cancellation_token is not None
                    and options.cancellation_token.is_set()):
                cancelled_all = True
                if seq is not None:
                    seq_blocked.add(seq)
                yield _outcome(item, "cancelled", None, BatchError(
                    "cancelled", "ProcessingCancelled",
                    "processing cancelled"), -1, 0, (),
                    _metrics(loaded, None, 0, 0))
                continue
            if loaded.error is not None:
                if seq is not None:
                    seq_blocked.add(seq)
                yield _outcome(item, "failed", None, loaded.error, -1, 0,
                               (), _metrics(loaded, None, 0, 0))
                continue

            group = [loaded]
            if seq is None and loaded.coalescible:
                while len(group) < gpu.max_batch_size:
                    _fill()
                    if not buffer:
                        break
                    nxt = buffer[0]
                    if (nxt.error is None
                            and nxt.accepted.item.sequence_id is None
                            and nxt.coalescible
                            and nxt.key == loaded.key):
                        group.append(buffer.popleft())
                    else:
                        break

            plan = None
            try:
                torch = _torch()
                device = _device()
                free_bytes, total_bytes = torch.cuda.mem_get_info(
                    device.index)
                sensor = loaded.decoded.sensor_info
                algorithm = str(loaded.prepared.effective_config
                                .get("demosaic", {})
                                .get("algorithm", "malvar"))
                plan = plan_memory(
                    shape=(int(sensor["height"]), int(sensor["width"])),
                    algorithm=algorithm, gpu=gpu, free_bytes=free_bytes,
                    total_bytes=total_bytes,
                    available_count=len(group))
                if plan.group_size and len(group) > plan.group_size:
                    extra = group[plan.group_size:]
                    group = group[:plan.group_size]
                    for leftover in reversed(extra):
                        buffer.appendleft(leftover)
                _device_info()
            except Exception as exc:
                from .types import GpuUnavailableError
                kind = ("gpu_unavailable"
                        if isinstance(exc, GpuUnavailableError)
                        else "processing")
                for member in group:
                    m_item = member.accepted.item
                    if m_item.sequence_id is not None:
                        seq_blocked.add(m_item.sequence_id)
                    yield _outcome(m_item, "failed", None,
                                   _error(exc, kind), -1, 0, (),
                                   _metrics(member, plan, 0, 0))
                continue

            batch_index = batch_counter
            batch_counter += 1
            hint_copies = {}
            for member in group:
                member_seq = member.accepted.item.sequence_id
                if member_seq is not None:
                    private = seq_states.setdefault(
                        member_seq, ProcessingState())
                    hint_copies[member.accepted.item.item_id] = (
                        copy.deepcopy(private))
            results, attempts_by_item = _compute(
                group, batch_index, hint_copies)

            for member in group:
                m_item = member.accepted.item
                m_seq = m_item.sequence_id
                entry = results[m_item.item_id]
                m_attempts = attempts_by_item[m_item.item_id]
                m_metrics = _metrics(
                    member, plan,
                    m_attempts[-1].compute_transfer_ns if m_attempts else 0,
                    0)
                if isinstance(entry, BatchError):
                    if entry.kind == "cancelled":
                        cancelled_all = True
                        if m_seq is not None:
                            seq_blocked.add(m_seq)
                        yield _outcome(m_item, "cancelled", None, entry,
                                       batch_index,
                                       m_attempts[-1].batch_size
                                       if m_attempts else 0,
                                       m_attempts, m_metrics)
                        continue
                    if m_seq is not None:
                        seq_blocked.add(m_seq)
                    yield _outcome(m_item, "failed", None, entry,
                                   batch_index,
                                   m_attempts[-1].batch_size
                                   if m_attempts else 0,
                                   m_attempts, m_metrics)
                    continue
                if (options.cancellation_token is not None
                        and options.cancellation_token.is_set()):
                    cancelled_all = True
                    if m_seq is not None:
                        seq_blocked.add(m_seq)
                    yield _outcome(m_item, "cancelled", None, BatchError(
                        "cancelled", "ProcessingCancelled",
                        "processing cancelled"), batch_index,
                        m_attempts[-1].batch_size if m_attempts else 0,
                        m_attempts, m_metrics)
                    continue
                commit_t0 = time.perf_counter_ns()
                result = None
                commit_error = None
                try:
                    result = commit_frame(entry)
                except ProcessingCancelled:
                    commit_error = BatchError(
                        "cancelled", "ProcessingCancelled",
                        "processing cancelled")
                except Exception as exc:
                    commit_error = _error(exc, "io")
                commit_ns = time.perf_counter_ns() - commit_t0
                m_metrics["commit_ns"] = int(commit_ns)
                if commit_error is None and (
                        options.cancellation_token is not None
                        and options.cancellation_token.is_set()):
                    commit_error = BatchError(
                        "cancelled", "ProcessingCancelled",
                        "processing cancelled")
                if commit_error is not None:
                    if commit_error.kind == "cancelled":
                        cancelled_all = True
                    if m_seq is not None:
                        seq_blocked.add(m_seq)
                    status = ("cancelled" if commit_error.kind == "cancelled"
                              else "failed")
                    yield _outcome(
                        m_item, status, None, commit_error,
                        batch_index,
                        m_attempts[-1].batch_size if m_attempts else 0,
                        m_attempts, m_metrics)
                    continue
                if m_seq is not None:
                    _commit_state(seq_states[m_seq], result)
                yield _outcome(
                    m_item, "success", result, None, batch_index,
                    m_attempts[-1].batch_size if m_attempts else 0,
                    m_attempts, m_metrics)
    finally:
        if hasattr(stream, "close"):
            stream.close()
