"""GPU Durand: explicit CPU-fallback wrapper.

GPU acceleration of the non-equivalent bilateral implementation is not
validated in M3 and is never auto-selected; a ``RuntimeWarning`` is
emitted and the verified CPU path is used. Actual GPU support is a
documented gap, not a parity claim.
"""

from __future__ import annotations

import warnings
from typing import Any, Mapping

import numpy as np

from modules.tone_mapping.durand.hdr_durand_fast import HDRDurandToneMapping


def is_gpu_available() -> bool:
    return False


def should_use_gpu(img_size: tuple[int, int], operation: str) -> bool:
    return False


def to_umat(img: np.ndarray, use_gpu: bool = True) -> np.ndarray:
    return np.asarray(img)


def from_umat(umat_or_array: np.ndarray) -> np.ndarray:
    return np.asarray(umat_or_array)


class HDRDurandToneMappingGPU(HDRDurandToneMapping):
    """CPU-fallback subclass; warns about the unverified-GPU gap on init."""

    def __init__(
        self,
        img: np.ndarray,
        platform: Mapping[str, Any],
        sensor_info: Mapping[str, Any],
        params: Mapping[str, Any],
        input_max: int | None = None,
    ) -> None:
        warnings.warn(
            "HDRDurandToneMappingGPU falls back to the verified CPU path; "
            "GPU acceleration is unverified in M3",
            RuntimeWarning,
            stacklevel=2,
        )
        super().__init__(img, platform, sensor_info, params, input_max)
