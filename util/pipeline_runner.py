"""``process_frame``: single-frame processing entry point for the M2a API.

The engine is the existing ``BrilliantISP`` facade — this runner supplies
in-memory configs, array sources, per-request options/state, and structured
results. Signal domains remain the legacy ones (see ``pipeline_types``);
the unified RAW decoder and native float path are M2b/M4 work.
"""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np

from brilliant_isp import BrilliantISP
from util.config_schema import validate_config
from util.pipeline_types import (
    PipelineResult,
    ProcessingCancelled,
    ProcessingOptions,
    ProcessingState,
)

AE_HISTORY_LIMIT = 32


def _build_engine(
    config: Mapping[str, Any],
    data_path: str,
    options: ProcessingOptions,
    state: ProcessingState | None,
) -> BrilliantISP:
    cfg = copy.deepcopy(dict(config))
    engine = BrilliantISP(
        data_path,
        cfg,
        outFileName="",
        output_path=str(options.output_dir) if options.output_dir else None,
        options=options,
    )
    # State is a read-only input hint; the engine never mutates it. The
    # previous AWB estimate is only a compute_gains fallback, never a manual
    # config substitution; the DG index seeds auto-DG, not manual fields.
    if state is not None:
        engine.processing_state = dataclasses.replace(state)
    return engine


def _run_engine(isp: BrilliantISP, options: ProcessingOptions) -> None:
    if isp.render_3a:
        isp.execute_with_3a_statistics()
    else:
        isp.run_pipeline(visualize_output=options.save_outputs)


def process_frame(
    source: str | Path | np.ndarray,
    config: Mapping[str, Any],
    *,
    state: ProcessingState | None = None,
    options: ProcessingOptions | None = None,
) -> PipelineResult:
    """Process one frame through the existing pipeline.

    ``source`` is a RAW file path (shared decoder) or a decoded 2-D Bayer
    ``np.ndarray`` matching ``sensor_info.width``/``height`` exactly — array
    input performs no file decode, alignment, flips, or embedded-row trim:
    the array is treated as already in processing coordinates with the
    declared CFA; endian variants are normalized to native-endian storage.
    The caller's config mapping is deep-copied and never mutated. Independent
    calls build a fresh engine each time; pass an explicit
    ``ProcessingState`` to thread intentional 3A carryover between frames.
    """
    options = dataclasses.replace(
        options if options is not None else ProcessingOptions()
    )

    errors = [i for i in validate_config(config) if i.severity == "error"]
    if errors:
        raise ValueError(
            "config failed validation: "
            + "; ".join(f"{i.path}: {i.message}" for i in errors)
        )

    if isinstance(source, np.ndarray):
        frame = np.array(source, copy=True)
        if frame.ndim != 2:
            raise ValueError(
                f"array source must be a 2-D Bayer frame, got shape {frame.shape}"
            )
        if frame.dtype.kind != "u" or frame.dtype.itemsize not in (1, 2, 4):
            raise ValueError(
                f"array source dtype {frame.dtype} is not supported; "
                "decoded Bayer input must be uint8/uint16/uint32"
            )
        frame = frame.astype(frame.dtype.newbyteorder("="), copy=False)
        sensor = config.get("sensor_info")
        if not isinstance(sensor, Mapping):
            raise ValueError("config must contain a sensor_info mapping for array input")
        expected = (int(sensor["height"]), int(sensor["width"]))
        if frame.shape != expected:
            raise ValueError(
                f"array shape {frame.shape} does not match declared "
                f"sensor_info geometry {expected[1]}x{expected[0]}"
            )
        isp = _build_engine(config, "", options, state)
        source_identity = f"array:{options.run_id}"
        isp.raw = frame
        isp.in_file = "array_source"
        isp.out_file = "array_source"
        if isp.platform is not None:
            isp.platform["in_file"] = "array_source"
            isp.platform["out_file"] = "array_source"
        _run_engine(isp, options)
    else:
        path = Path(source).expanduser().resolve()
        isp = _build_engine(config, str(path.parent), options, state)
        source_identity = str(path)
        isp.execute(img_path=path.name)

    if isp.last_output_rgb is None or isp.last_output_native is None:
        raise RuntimeError("pipeline produced no output image")
    sensor = isp.sensor_info
    if sensor is None:
        raise RuntimeError("pipeline has no sensor_info")

    metadata = getattr(isp, "last_output_metadata", None)
    if metadata is None:
        raise RuntimeError("pipeline produced no output metadata")
    metadata = dataclasses.replace(metadata, source_identity=source_identity)

    if not isp.last_feedback:
        raise RuntimeError("pipeline produced no processing feedback")
    feedback = dict(isp.last_feedback)

    result = PipelineResult(
        native_image=isp.last_output_native,
        native_metadata=metadata,
        preview_rgb8=isp.last_output_rgb,
        effective_config=cast(
            dict[str, Any], copy.deepcopy(isp.c_yaml) if isp.c_yaml else {}
        ),
        feedback=feedback,
        migration_report=dict(getattr(isp, "migration_report", {})),
        artifacts=tuple(getattr(isp, "last_artifacts", ())),
        timings=tuple(isp.stage_events),
        captures=dict(isp.captures),
        warnings=tuple(isp.config_notices),
    )

    token = options.cancellation_token
    if token is not None and token.is_set():
        raise ProcessingCancelled("processing cancelled at final boundary")

    if state is not None:
        if feedback.get("awb_status") == "estimated":
            state.previous_awb_gains = (
                float(isp.awb_gains[0]),
                float(isp.awb_gains[1]),
            )
        if bool(isp.parm_dga.get("is_enable", True)) and bool(
            isp.parm_dga.get("is_auto", False)
        ):
            next_idx = feedback.get("next_digital_gain_index")
            state.digital_gain_index = (
                int(next_idx) if isinstance(next_idx, (int, float))
                else int(isp.dga_current_gain)
            )
        state.ae_history = (*state.ae_history, isp.ae_feedback)[-AE_HISTORY_LIMIT:]
        if feedback.get("ae_next_gain") is not None:
            state.ae_gain = float(feedback["ae_next_gain"])
        if feedback.get("tm_white_next") is not None:
            state.tone_white = float(feedback["tm_white_next"])

    return result
