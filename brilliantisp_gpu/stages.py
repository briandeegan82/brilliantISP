"""Cohort stage recorder: one GPU compute, per-context event semantics.

``run_stage`` moves the qualified single-context ``_run_stage`` semantics
into a shared form: started/cancel events fire per context before one GPU
compute; one elapsed measurement is recorded on every context; results are
split per leading batch index only for event metadata/captures.
"""

from __future__ import annotations

import time

import numpy as np

from util.pipeline_types import ProcessingCancelled, StageEvent

from .types import GpuError, RawTensor, require_torch


def _split_image(value, frame_index):
    """Leading-B view of one frame for event metadata/captures."""
    torch = require_torch()
    if isinstance(value, RawTensor):
        return RawTensor(value.codes[frame_index:frame_index + 1],
                         value.storage_dtype)
    if isinstance(value, torch.Tensor):
        return value[frame_index:frame_index + 1]
    if isinstance(value, tuple):
        return tuple(_split_image(v, frame_index) for v in value)
    return value


def run_stage(contexts, stage_id, thunk, *, ae_pass=0, split_batch=False):
    """Run one stage for a cohort of engine contexts (real batched B>1)."""
    torch = require_torch()
    contexts = tuple(contexts)
    if len(contexts) > 1 and not split_batch:
        raise GpuError("cohort stages require split_batch=True")
    started = []
    for ctx in contexts:
        try:
            ctx._check_cancelled()
        except ProcessingCancelled:
            for done in started + [ctx]:
                event = StageEvent(
                    done.run_id, stage_id, "cancelled", elapsed_ns=0,
                    pass_index=done.pass_index, ae_pass=ae_pass,
                    error="cancelled")
                done.stage_events.append(event)
                done._emit(event)
            raise
        ctx._emit(StageEvent(ctx.run_id, stage_id, "started",
                             pass_index=ctx.pass_index, ae_pass=ae_pass))
        started.append(ctx)
    t0 = time.perf_counter_ns()
    try:
        if contexts[0].gpu.profile_stages:
            torch.cuda.synchronize(contexts[0].device)
            t0 = time.perf_counter_ns()
        result = thunk()
        if split_batch:
            if isinstance(result, RawTensor):
                ok = result.codes.ndim == 3 \
                    and result.codes.shape[0] == len(contexts)
            elif isinstance(result, torch.Tensor):
                ok = result.ndim in (3, 4) \
                    and result.shape[0] == len(contexts)
            else:
                ok = False
            if not ok:
                raise GpuError(
                    "split_batch stage must return a leading batch "
                    f"dimension of {len(contexts)}, got "
                    f"{type(result).__name__}")
        if contexts[0].gpu.profile_stages:
            torch.cuda.synchronize(contexts[0].device)
    except ProcessingCancelled:
        elapsed = time.perf_counter_ns() - t0
        for ctx in started:
            event = StageEvent(ctx.run_id, stage_id, "cancelled",
                               elapsed_ns=elapsed, pass_index=ctx.pass_index,
                               ae_pass=ae_pass, error="cancelled")
            ctx.stage_events.append(event)
            ctx._emit(event)
        raise
    except Exception as exc:
        elapsed = time.perf_counter_ns() - t0
        for ctx in started:
            event = StageEvent(ctx.run_id, stage_id, "failed",
                               elapsed_ns=elapsed, pass_index=ctx.pass_index,
                               ae_pass=ae_pass,
                               error=f"{type(exc).__name__}: {exc}")
            ctx.stage_events.append(event)
            ctx._emit(event)
        raise
    elapsed = time.perf_counter_ns() - t0
    for i, ctx in enumerate(started):
        frame_value = _split_image(result, i) if split_batch else result
        shape, dtype = ctx._stage_meta(frame_value)
        event = StageEvent(ctx.run_id, stage_id, "completed",
                           elapsed_ns=elapsed, pass_index=ctx.pass_index,
                           ae_pass=ae_pass, shape=shape, dtype=dtype)
        ctx.stage_events.append(event)
        ctx._emit(event)
        if shape is not None and stage_id in ctx.options.capture_stages:
            captured = np.array(ctx._host_result(
                ctx._stage_image(frame_value), frame_index=0), copy=True)
            captured.setflags(write=False)
            ctx.captures[stage_id] = captured
    return result
