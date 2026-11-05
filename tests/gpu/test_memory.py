from __future__ import annotations

import pytest

from brilliantisp_gpu.memory import plan_memory
from brilliantisp_gpu.types import GpuOptions


MIB = 1024 * 1024


def _plan(**kwargs):
    args = dict(
        shape=(100, 100), algorithm="malvar",
        gpu=GpuOptions(memory_fraction=1.0), free_bytes=100 * MIB,
        total_bytes=200 * MIB, available_count=4,
    )
    args.update(kwargs)
    return plan_memory(**args)


def test_memory_selection_is_bounded_and_estimates_are_explicit():
    plan = _plan()
    assert plan.group_size == 3
    assert plan.selection_budget_bytes == 100 * MIB
    assert plan.estimated_peak_bytes == 64 * MIB + 3 * 10000 * 1024
    assert not plan.single_frame_over_estimate
    assert _plan(gpu=GpuOptions(memory_fraction=1.0, max_batch_size=2)).group_size == 2
    assert _plan(available_count=1).group_size == 1
    assert _plan(algorithm="ahd_opt").group_size == 1
    assert _plan(algorithm="lmmse_fast").group_size == 1


def test_memory_empty_and_single_frame_failure_policy():
    empty = _plan(available_count=0)
    assert empty.group_size == empty.estimated_peak_bytes == 0
    one = _plan(free_bytes=0)
    assert one.group_size == 1
    assert one.single_frame_over_estimate
    assert one.selection_budget_bytes == 0
    huge = _plan(shape=(2185, 3840))
    assert huge.group_size == 1
    assert huge.single_frame_over_estimate


def test_memory_fraction_uses_exact_integer_budget():
    free = 2**60 + 31
    gpu = GpuOptions(memory_fraction=.7)
    n, d = float(gpu.memory_fraction).as_integer_ratio()
    plan = _plan(free_bytes=free, total_bytes=free + 100, gpu=gpu)
    assert plan.selection_budget_bytes == free * n // d


@pytest.mark.parametrize("change", (
    dict(shape=(0, 10)), dict(shape=(True, 10)), dict(free_bytes=-1),
    dict(total_bytes=0), dict(free_bytes=201 * MIB),
    dict(available_count=True), dict(available_count=-1), dict(algorithm=None),
))
def test_memory_invalid_metadata(change):
    with pytest.raises(ValueError):
        _plan(**change)
