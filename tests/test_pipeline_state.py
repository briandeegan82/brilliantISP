from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pytest

from util.pipeline_types import (
    FrameMetadata,
    ProcessingCancelled,
    ProcessingOptions,
    ProcessingState,
    StageEvent,
)


def test_processing_state_defaults_independent() -> None:
    state = ProcessingState()
    assert state.previous_awb_gains is None
    assert state.digital_gain_index is None
    assert state.ae_history == ()


def test_processing_options_defaults_compute_only() -> None:
    options = ProcessingOptions()
    assert options.capture_stages == ()
    assert options.diagnostics is False
    assert options.save_outputs is False
    assert options.output_dir is None
    assert options.cancellation_token is None
    assert options.progress_callback is None
    uuid.UUID(options.run_id)


def test_unique_run_ids() -> None:
    assert ProcessingOptions().run_id != ProcessingOptions().run_id


def test_stage_event_defaults() -> None:
    event = StageEvent(run_id="r", stage_id="crop", status="completed")
    assert event.elapsed_ns == 0
    assert event.pass_index == 0
    assert event.ae_pass == 0
    assert event.shape is None
    assert event.dtype is None
    assert event.error is None


def test_frame_metadata_frozen() -> None:
    meta = FrameMetadata(
        width=4,
        height=2,
        cfa_pattern="rggb",
        storage_bit_depth=16,
        code_bit_depth=12,
        signal_domain="legacy_configured_rgb8",
        transfer="legacy_configured",
        source_identity="array:2x4",
    )
    with pytest.raises(Exception):
        setattr(meta, "width", 5)


def test_processing_cancelled_is_exception() -> None:
    assert issubclass(ProcessingCancelled, Exception)
