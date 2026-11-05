"""Runtime-compiled reference argsort and fused-fma projection tests.

Argsort must equal np.argsort(kind='quicksort') on NumPy 1.26.4's scalar
finite-key path — full index-array equality including every tie order,
not just sorted keys. Real CUDA device only (hardware qualification);
HIP path is implemented but unverified on this machine.
"""

from __future__ import annotations

import numpy as np
import pytest

from brilliantisp_gpu.types import GpuError, GpuUnavailableError

pytestmark = pytest.mark.gpu_hardware


def _argsort(keys, device, **kw):
    import torch
    from brilliantisp_gpu.ops.reference_sort import reference_argsort
    return reference_argsort(torch.as_tensor(keys, dtype=torch.float64, device=device), **kw)


def _cases():
    rng = np.random.RandomState(20240117)
    yield "rand_unique_1000", rng.rand(1000)
    yield "rand_tied_small", rng.randint(0, 4, 30).astype(np.float64)
    yield "rand_tied_1000", rng.randint(0, 7, 1000).astype(np.float64)
    yield "sorted", np.arange(50, dtype=np.float64)
    yield "reversed", np.arange(50, 0, -1, dtype=np.float64)
    yield "all_equal_zeros", np.zeros(40)
    yield "signed_zeros", np.array([0.0, -0.0] * 17)
    yield "equal_groups_straddling", np.concatenate([
        np.full(10, 5.0), np.arange(10, dtype=np.float64) + 1,
        np.full(10, 5.0), np.arange(10, dtype=np.float64) + 6,
    ])
    rng2 = np.random.RandomState(7)
    for n in (0, 1, 2, 15, 16, 17, 30, 31, 32, 127):
        yield f"rand_n{n}", rng2.rand(n)


@pytest.mark.parametrize("name,keys", list(_cases()))
def test_argsort_matches_numpy_quicksort(cuda_device, name, keys):
    expected = np.argsort(keys, kind="quicksort")
    actual = _argsort(keys, cuda_device).cpu().numpy()
    np.testing.assert_array_equal(actual, expected)


def test_argsort_heapsort_fallback_seam(cuda_device):
    """Depth-override seam forces the aheapsort branch; oracle is
    np.argsort(kind='heapsort') (same comparator -> same order here)."""
    rng = np.random.RandomState(99)
    keys = rng.randint(0, 9, 64).astype(np.float64)
    keys = np.concatenate([keys, rng.rand(64)])
    actual = _argsort(keys, cuda_device, _force_depth=-1).cpu().numpy()
    expected = np.argsort(keys, kind="heapsort")
    np.testing.assert_array_equal(actual, expected)


def test_argsort_zero_depth_seam(cuda_device):
    keys = np.random.RandomState(3).rand(40)
    actual = _argsort(keys, cuda_device, _force_depth=0).cpu().numpy()
    expected = np.argsort(keys, kind="quicksort")
    np.testing.assert_array_equal(actual, expected)


def test_argsort_nondefault_stream(cuda_device):
    torch = pytest.importorskip("torch")
    keys = np.random.RandomState(11).rand(33)
    s = torch.cuda.Stream(device=cuda_device)
    with torch.cuda.stream(s):
        actual = _argsort(keys, cuda_device)
    s.synchronize()
    torch.cuda.synchronize()
    np.testing.assert_array_equal(actual.cpu().numpy(), np.argsort(keys, kind="quicksort"))


def test_argsort_rejects_bad_inputs(cuda_device):
    torch = pytest.importorskip("torch")
    from brilliantisp_gpu.ops.reference_sort import reference_argsort
    with pytest.raises(GpuError):
        reference_argsort(torch.ones(5, device="cpu", dtype=torch.float64))
    with pytest.raises(GpuError):
        reference_argsort(torch.ones(5, device=cuda_device, dtype=torch.float32))
    with pytest.raises(GpuError):
        reference_argsort(torch.ones(3, 3, device=cuda_device, dtype=torch.float64))
    bad = torch.ones(4, device=cuda_device, dtype=torch.float64)
    bad[2] = float("nan")
    with pytest.raises(GpuError):
        reference_argsort(bad)


def test_missing_rtc_named_error(cuda_device, monkeypatch):
    torch = pytest.importorskip("torch")
    from brilliantisp_gpu.ops import rtc
    from brilliantisp_gpu.ops.reference_sort import reference_argsort
    monkeypatch.setitem(rtc._libs, "NVRTC", None)
    rtc._module_cache.clear()
    with pytest.raises(GpuUnavailableError):
        reference_argsort(torch.rand(4, device=cuda_device, dtype=torch.float64))
    rtc._libs.clear()


def test_fused_project_matches_cpu(cuda_device):
    """Committed fixture: frozen CPU bit patterns reconstructed via uint64 bits."""
    torch = pytest.importorskip("torch")
    import json
    from pathlib import Path
    from brilliantisp_gpu.ops.reference_sort import fused_project
    fixture = json.loads(
        (Path(__file__).resolve().parent / "fixtures"
         / "pca_projection_cases.json").read_text()
    )
    for case in fixture["cases"]:
        rows = np.tile(np.asarray(case["row_cycle"], np.float64), (case["repeat"], 1))
        unit_bits = np.asarray(case["unit_float64_bits"], np.uint64)
        unit = unit_bits.view(np.float64)
        expected_bits = np.asarray(
            case["projection_float64_bits_per_row_cycle"], np.uint64)
        expected = np.tile(expected_bits.view(np.float64), case["repeat"])
        out = fused_project(
            torch.as_tensor(rows, dtype=torch.float64, device=cuda_device),
            torch.as_tensor(unit, dtype=torch.float64, device=cuda_device),
        ).cpu().numpy()
        np.testing.assert_array_equal(
            out, expected, err_msg=f"fused_project != fixture {case['id']}")
        out_bits = out.view(np.uint64)
        np.testing.assert_array_equal(
            out_bits, np.tile(expected_bits, case["repeat"]))
