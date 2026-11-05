"""Public GPU entry point; mirrors util.pipeline_runner.process_frame."""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from util.pipeline_types import (
    ProcessingCancelled,
    ProcessingOptions,
    ProcessingState,
    PipelineResult,
)

from .config import prepare_config
from .engine import GpuEngine
from .types import GpuOptions

AE_HISTORY_LIMIT = 32


def _commit_state(state: ProcessingState, result: PipelineResult) -> None:
    """Commit run feedback into a ProcessingState (success-only path)."""
    feedback = result.feedback
    if feedback.get("awb_status") == "estimated":
        state.previous_awb_gains = (
            float(result.feedback["awb_gains"][0]),
            float(result.feedback["awb_gains"][1]),
        )
    eff = result.effective_config["digital_gain"]
    if bool(eff.get("is_enable", True)) and bool(eff.get("is_auto", False)):
        next_idx = feedback.get("next_digital_gain_index")
        state.digital_gain_index = (
            int(next_idx) if isinstance(next_idx, (int, float))
            else int(feedback["digital_gain_index"])
        )
    state.ae_history = (*state.ae_history, feedback["ae_feedback"])[
        -AE_HISTORY_LIMIT:
    ]


def process_frame_gpu(
    source: str | Path | np.ndarray,
    config: str | Path | list | tuple | Mapping[str, Any],
    *,
    state: ProcessingState | None = None,
    options: ProcessingOptions | None = None,
    gpu: GpuOptions | None = None,
) -> PipelineResult:
    """Process one frame on the GPU engine.

    Mirrors ``util.pipeline_runner.process_frame``: ``source`` is a RAW file
    path (shared decoder) or a decoded 2-D Bayer ``np.ndarray`` matching
    ``sensor_info.width``/``height`` exactly. The caller's config mapping and
    state are deep-copied and never mutated; state commits only on complete
    success.
    """
    options = dataclasses.replace(
        options if options is not None else ProcessingOptions()
    )
    if (options.save_outputs or options.diagnostics) and options.output_dir is None:
        raise ValueError(
            "save_outputs/diagnostics requires an explicit output_dir"
        )
    gpu = gpu if gpu is not None else GpuOptions()
    state_in = copy.deepcopy(state) if state is not None else None

    prepared = prepare_config(config, options=options)
    engine = GpuEngine(prepared, options, state_in, gpu)
    result = engine.run(source)

    token = options.cancellation_token
    if token is not None and token.is_set():
        raise ProcessingCancelled("processing cancelled at final boundary")

    if state is not None:
        _commit_state(state, result)

    return result
