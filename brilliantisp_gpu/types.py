from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, NamedTuple

if TYPE_CHECKING:
    from util.pipeline_types import PipelineResult

import numpy as np


class GpuError(RuntimeError):
    """Base error for GPU engine failures."""


class GpuUnavailableError(GpuError):
    """Torch absent or the requested CUDA/ROCm device is unusable."""


class GpuVendorError(GpuError):
    """The available backend does not match the required vendor."""


class RawTensor(NamedTuple):
    """Unsigned-code BHW int64 tensor plus canonical storage dtype name."""

    codes: Any
    storage_dtype: str  # 'uint8' | 'uint16' | 'uint32'


@dataclass(frozen=True)
class GpuOptions:
    device: str = "cuda:0"
    max_batch_size: int = 4
    memory_fraction: float = 0.70
    prefetch: int = 2
    profile_stages: bool = False
    split_on_oom: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.device, str) or not self.device:
            raise ValueError(f"GpuOptions.device must be a device string, got {self.device!r}")
        if ":" in self.device and self.device.endswith(":"):
            raise ValueError(
                f"GpuOptions.device must be a CUDA device ('cuda[:i]'), got {self.device!r}"
            )
        name, _, index = self.device.partition(":")
        if name != "cuda" or ":" in index:
            raise ValueError(
                f"GpuOptions.device must be a CUDA device ('cuda[:i]'), got {self.device!r}"
            )
        if index:
            if not index.isdigit() or (len(index) > 1 and index.startswith("0")):
                raise ValueError(f"invalid device index in {self.device!r}")
            parsed = int(index)
            if parsed < 0:
                raise ValueError(f"invalid device index in {self.device!r}")
        if isinstance(self.max_batch_size, bool) or not isinstance(self.max_batch_size, int) or self.max_batch_size < 1:
            raise ValueError(f"max_batch_size must be a positive integer, got {self.max_batch_size!r}")
        if isinstance(self.memory_fraction, bool) or not isinstance(self.memory_fraction, (int, float)) or not math.isfinite(float(self.memory_fraction)) or not 0 < float(self.memory_fraction) <= 1:
            raise ValueError(f"memory_fraction must be finite in (0,1], got {self.memory_fraction!r}")
        if isinstance(self.prefetch, bool) or not isinstance(self.prefetch, int) or self.prefetch < 0:
            raise ValueError(f"prefetch must be a nonnegative integer, got {self.prefetch!r}")
        if type(self.profile_stages) is not bool:
            raise ValueError("profile_stages must be a boolean")
        if type(self.split_on_oom) is not bool:
            raise ValueError("split_on_oom must be a boolean")

    @property
    def device_index(self) -> int:
        _, _, index = self.device.partition(":")
        return int(index) if index else 0


def require_torch() -> Any:
    """Import torch lazily; raises GpuUnavailableError when absent."""
    try:
        import torch
    except ImportError as exc:
        raise GpuUnavailableError(
            "the optional GPU engine requires torch (CUDA or ROCm build); "
            "install the backend-appropriate wheel documented for this package"
        ) from exc
    return torch

_ERROR_KINDS = ("input", "processing", "gpu_oom", "gpu_unavailable",
                "io", "cancelled", "blocked")


def _nonempty_str(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True)
class BatchItem:
    """One batch work item: a frame source plus its frame-API config."""

    item_id: str
    source: "str | Path | np.ndarray"
    config: "str | Path | list | tuple | Mapping[str, Any]"
    sequence_id: "str | None" = None
    frame_index: "int | None" = None
    output_dir: "str | Path | None" = None
    output_stem: "str | None" = None

    def __post_init__(self) -> None:
        _nonempty_str(self.item_id, "item_id")
        if self.sequence_id is not None:
            _nonempty_str(self.sequence_id, "sequence_id")
            if isinstance(self.frame_index, bool) or not isinstance(
                    self.frame_index, int) or self.frame_index < 0:
                raise ValueError(
                    "frame_index must be a nonnegative integer when "
                    "sequence_id is set")
        elif self.frame_index is not None:
            raise ValueError("frame_index requires sequence_id")
        if self.output_stem is not None:
            stem = _nonempty_str(self.output_stem, "output_stem")
            if (stem in (".", "..") or "/" in stem or "\\" in stem
                    or "\x00" in stem or ":" in stem
                    or stem != stem.strip()):
                raise ValueError(
                    f"output_stem must be a plain basename, got {stem!r}")
        if not isinstance(self.source, (str, Path)) and not isinstance(
                self.source, np.ndarray):
            raise TypeError(
                f"source must be a path or numpy array, got "
                f"{type(self.source).__name__}")


@dataclass(frozen=True)
class BatchError:
    """Plain structured failure metadata (no exception/traceback/tensors)."""

    kind: str
    exception_type: str
    message: str
    stage_id: "str | None" = None

    def __post_init__(self) -> None:
        if self.kind not in _ERROR_KINDS:
            raise ValueError(f"unknown batch error kind {self.kind!r}")
        _nonempty_str(self.exception_type, "exception_type")
        if not isinstance(self.message, str):
            raise ValueError("message must be a string")


@dataclass(frozen=True)
class BatchAttempt:
    """One compute attempt record; ids are unique across a batch run."""

    attempt_id: int
    item_ids: "tuple[str, ...]"
    batch_size: int
    status: str
    error: "BatchError | None"
    compute_transfer_ns: int

    def __post_init__(self) -> None:
        if self.status not in ("success", "failed", "cancelled"):
            raise ValueError(f"unknown attempt status {self.status!r}")
        if not isinstance(self.item_ids, tuple) or not self.item_ids \
                or not all(isinstance(v, str) and v for v in self.item_ids):
            raise ValueError("item_ids must be a nonempty tuple of strings")
        if len(set(self.item_ids)) != len(self.item_ids):
            raise ValueError("item_ids must be unique within an attempt")
        if isinstance(self.batch_size, bool) or not isinstance(
                self.batch_size, int) or self.batch_size != len(self.item_ids):
            raise ValueError("batch_size must equal len(item_ids)")
        if isinstance(self.compute_transfer_ns, bool) or not isinstance(
                self.compute_transfer_ns, int) \
                or self.compute_transfer_ns < 0:
            raise ValueError("compute_transfer_ns must be a nonnegative int")
        if self.error is not None and not isinstance(self.error, BatchError):
            raise TypeError("error must be a BatchError or None")
        if (self.status == "success") != (self.error is None):
            raise ValueError("success attempts carry no error; others carry one")


@dataclass(frozen=True)
class BatchOutcome:
    """Per-item terminal outcome of process_batch."""

    item_id: str
    status: str
    result: "PipelineResult | None"
    error: "BatchError | None"
    device: "dict[str, Any]"
    batch_index: int
    batch_size: int
    attempts: "tuple[BatchAttempt, ...]"
    metrics: "dict[str, Any]"

    def __post_init__(self) -> None:
        _nonempty_str(self.item_id, "item_id")
        if self.status not in ("success", "failed", "cancelled", "blocked"):
            raise ValueError(f"unknown outcome status {self.status!r}")
        if self.status == "success":
            if self.result is None or self.error is not None:
                raise ValueError("success outcomes carry a result and no error")
        else:
            if self.result is not None or self.error is None:
                raise ValueError(
                    "non-success outcomes carry an error and no result")
        if not isinstance(self.device, dict):
            raise TypeError("device must be a dict")
        if not isinstance(self.metrics, dict):
            raise TypeError("metrics must be a dict")
