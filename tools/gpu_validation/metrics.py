from __future__ import annotations

import math
from typing import Any

import numpy as np

from tools.gpu_validation.contracts import NumericGate


def compare_arrays(
    reference: np.ndarray,
    candidate: np.ndarray,
    gate: NumericGate,
    *,
    scale: float = 1.0,
    data_range: float | None = None,
) -> dict[str, Any]:
    expected, actual = np.asarray(reference), np.asarray(candidate)
    if expected.shape != actual.shape:
        raise ValueError(f"shape mismatch: {expected.shape} != {actual.shape}")
    if expected.dtype != actual.dtype:
        raise ValueError(f"dtype mismatch: {expected.dtype} != {actual.dtype}")
    if not expected.size or expected.dtype.kind not in "buif":
        raise ValueError("comparison requires nonempty real arrays")
    if expected.dtype.kind in "ui" and expected.dtype.itemsize > 4:
        limit = 2**52
        if any(
            int(array.min()) < -limit or int(array.max()) > limit
            for array in (expected, actual)
        ):
            raise ValueError("64-bit integer values exceed the exact float64 difference range")
    if not np.all(np.isfinite(expected)) or not np.all(np.isfinite(actual)):
        raise ValueError("comparison requires finite reference and candidate")
    if isinstance(scale, bool) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("comparison scale must be finite and positive")
    if data_range is not None and (
        isinstance(data_range, bool) or not math.isfinite(data_range) or data_range <= 0
    ):
        raise ValueError("PSNR data range must be finite and positive")
    e = expected.astype(np.float64)
    with np.errstate(over="raise", invalid="raise"):
        delta = actual.astype(np.float64) - e
    absolute = np.abs(delta)
    normalized = absolute / scale
    bound = gate.atol + gate.rtol * np.abs(e / scale)
    outside = normalized > bound
    if gate.max_code_error is not None:
        if expected.dtype.kind != "u" or expected.dtype.itemsize > 4 or scale != 1:
            raise ValueError("integer image gates require unsigned codes at original scale")
        outside = absolute > gate.max_code_error
    maximum = float(absolute.max())
    mean = float(absolute.mean())
    passed = not bool(np.any(outside))
    if gate.mean_code_error is not None:
        passed = passed and mean <= gate.mean_code_error
    rmse = maximum * float(np.sqrt(np.mean((delta / maximum) ** 2))) if maximum else 0.0
    psnr: float | str | None = None
    if data_range is not None:
        psnr = "inf" if rmse == 0 else 20 * math.log10(data_range / rmse)
    per_channel = None
    if delta.ndim == 3 and delta.shape[-1] == 3:
        per_channel = {
            "max_abs": absolute.max(axis=(0, 1)).tolist(),
            "mean_abs": absolute.mean(axis=(0, 1)).tolist(),
            "signed_bias": delta.mean(axis=(0, 1)).tolist(),
        }
    worst = np.unravel_index(int(absolute.argmax()), absolute.shape)
    return {
        "gate": gate.name, "passed": passed, "shape": list(expected.shape),
        "dtype": expected.dtype.str, "scale": scale, "atol": gate.atol, "rtol": gate.rtol,
        "max_code_error": gate.max_code_error, "mean_code_error": gate.mean_code_error,
        "max_abs": maximum, "mean_abs": mean, "rmse": rmse,
        "absolute_error_p50_p95_p99": np.percentile(absolute, (50, 95, 99)).tolist(),
        "mismatched_components": int(np.count_nonzero(delta)),
        "outside_component_bound": int(np.count_nonzero(outside)),
        "fraction_greater_than_one_code": float(np.mean(absolute > 1)),
        "signed_bias": float(delta.mean()), "per_channel": per_channel,
        "worst_index": list(worst), "psnr": psnr, "psnr_data_range": data_range,
    }
