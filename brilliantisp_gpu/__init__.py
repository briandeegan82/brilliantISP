"""Additive headless GPU engine package (optional PyTorch dependency).

Importing this package never imports torch. Torch is loaded lazily inside
``brilliantisp_gpu.device``/ops helpers when a device operation is first
requested, so CPU-only installs can use ``types`` and ``config`` freely.
"""

from __future__ import annotations

from .types import (
    BatchAttempt,
    BatchError,
    BatchItem,
    BatchOutcome,
    GpuOptions,
    GpuError,
    GpuUnavailableError,
    GpuVendorError,
)
from .config import PreparedConfig, prepare_config
from .api import process_frame_gpu
from .batch import process_batch

__all__ = [
    "BatchAttempt",
    "BatchError",
    "BatchItem",
    "BatchOutcome",
    "GpuOptions",
    "GpuError",
    "GpuUnavailableError",
    "GpuVendorError",
    "PreparedConfig",
    "prepare_config",
    "process_frame_gpu",
    "process_batch",
]
