from __future__ import annotations

from dataclasses import dataclass

from .types import GpuOptions


_BASE_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class MemoryPlan:
    group_size: int
    estimated_peak_bytes: int
    selection_budget_bytes: int
    free_bytes: int
    total_bytes: int
    single_frame_over_estimate: bool


def _positive_int(value, name, *, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be a {'nonnegative' if allow_zero else 'positive'} integer")
    return value


def plan_memory(*, shape, algorithm, gpu: GpuOptions, free_bytes, total_bytes,
                available_count):
    """Conservative grouping heuristic, not an allocation guarantee or measurement.

    Even when the estimate exceeds the budget, one frame may be attempted.
    Actual allocation failures must be reported or split by the scheduler.
    """
    if len(shape) != 2:
        raise ValueError("memory planning requires HW geometry")
    height = _positive_int(shape[0], "height")
    width = _positive_int(shape[1], "width")
    free = _positive_int(free_bytes, "free_bytes", allow_zero=True)
    total = _positive_int(total_bytes, "total_bytes")
    count = _positive_int(available_count, "available_count", allow_zero=True)
    if free > total:
        raise ValueError("free_bytes must not exceed total_bytes")
    if not isinstance(algorithm, str):
        raise ValueError("algorithm must be a name")
    name = algorithm.lower()
    bytes_per_pixel = 4096 if name.startswith("ahd") else (
        2048 if name.startswith("lmmse") else 1536 if name.startswith("vng") else 1024
    )
    numerator, denominator = float(gpu.memory_fraction).as_integer_ratio()
    budget = min(free, total) * numerator // denominator
    if not count:
        return MemoryPlan(0, 0, budget, free, total, False)
    per_frame = height * width * bytes_per_pixel
    fits = max(1, (budget - _BASE_BYTES) // per_frame)
    size = min(count, gpu.max_batch_size, fits)
    estimate = _BASE_BYTES + size * per_frame
    return MemoryPlan(size, estimate, budget, free, total,
                      size == 1 and estimate > budget)
