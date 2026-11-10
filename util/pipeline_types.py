"""Processing contracts for the M2a pipeline facade and ``process_frame`` API.

These types deliberately describe the *legacy* signal path truthfully:
``signal_domain``/``transfer`` report what the current pipeline actually
produced (``legacy_configured_rgb8`` or ``legacy_ycbcr`` /
``legacy_configured``), not the planned linear/display-domain v2 contracts,
which land with M4. No Tk references live here so worker processes can use
these objects headlessly.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np


class ProcessingCancelled(RuntimeError):
    """Raised when the options' cancellation token is set between stages."""


class CancellationToken(Protocol):
    """Anything with an ``is_set()`` method, such as ``threading.Event``."""
    def is_set(self) -> bool:
        """Return True once cancellation has been requested."""


@dataclass(frozen=True)
class FrameMetadata:
    """Per-frame metadata reported with a processing result."""
    width: int
    height: int
    cfa_pattern: str | None
    storage_bit_depth: int
    code_bit_depth: int | None
    signal_domain: str
    transfer: str
    source_identity: str
    value_units: str = "codes"
    color_model: str | None = None
    color_space: str | None = None
    matrix: str | None = None
    range_mode: str | None = None
    layout: str | None = None
    packing: str | None = None
    byte_order: str | None = None
    format: str | None = None
    gamma: float | None = None
    nominal_range: tuple[float, float] | None = None
    component_ranges: tuple[tuple[float, float], ...] | None = None
    reference_white: float | None = None
    allows_signed_headroom: bool = False


@dataclass
class ProcessingState:
    """Optional carryover for intentional sequence processing.

    Independent frames use the default (no carryover); 3A state is only
    threaded when the caller explicitly passes and reuses a state object.
    """

    previous_awb_gains: tuple[float, float] | None = None
    digital_gain_index: int | None = None
    ae_history: tuple[int | None, ...] = ()
    ae_gain: float | None = None
    tone_white: float | None = None


@dataclass(frozen=True)
class StageEvent:
    """Progress event emitted as a pipeline stage runs."""
    run_id: str
    stage_id: str
    status: str
    elapsed_ns: int = 0
    pass_index: int = 0
    ae_pass: int = 0
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    error: str | None = None


@dataclass
class ProcessingOptions:
    """Per-request processing controls. Defaults are compute-only."""

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    capture_stages: tuple[str, ...] = ()
    diagnostics: bool = False
    save_outputs: bool = False
    output_dir: Path | None = None
    cancellation_token: CancellationToken | None = None
    progress_callback: Callable[[StageEvent], None] | None = None


@dataclass
class PipelineResult:
    """Outputs, metadata and stage events from one ``process_frame`` call."""
    native_image: np.ndarray
    native_metadata: FrameMetadata
    preview_rgb8: np.ndarray
    effective_config: dict[str, Any]
    feedback: dict[str, Any]
    timings: tuple[StageEvent, ...]
    captures: dict[str, np.ndarray]
    warnings: tuple[str, ...] = ()
    migration_report: dict[str, Any] = field(default_factory=dict)
    artifacts: tuple[Path, ...] = ()
