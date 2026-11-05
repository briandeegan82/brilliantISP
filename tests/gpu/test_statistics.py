"""pca_means parity vs AutoWhiteBalance._estimate_means(..., 'pca').

Hardware tests run on the real CUDA device only (gpu_hardware marker).
Projections use the runtime-compiled fused-fma kernel; ordering uses the
runtime-compiled reference argsort matching the NumPy 1.26.4 scalar
quicksort finite-key path — the reference profile for this machine
(local CPU lacks AVX512 sorting), not a claim covering every CPU SIMD
build.
"""

from __future__ import annotations

import numpy as np
import pytest

from modules.auto_white_balance.auto_white_balance import AutoWhiteBalance

from tools.gpu_validation.contracts import STATISTICS

from brilliantisp_gpu.blocks.statistics import pca_means, PcaDiagnostics
from brilliantisp_gpu.types import GpuError


def _cpu_result(rows, percentage):
    awb = object.__new__(AutoWhiteBalance)
    awb.parm_awb = {"percentage": percentage}
    return AutoWhiteBalance._estimate_means(awb, rows, "pca")


def _rows(seed, n=200):
    rng = np.random.RandomState(seed)
    return rng.rand(n, 3) * 3000 + 100


@pytest.mark.gpu_hardware
def _compare(rows, percentage, cuda_device, torch, tag):
    expected = _cpu_result(rows, percentage)
    actual, diag = pca_means(
        torch.as_tensor(rows, dtype=torch.float64, device=cuda_device),
        percentage, return_diagnostics=True,
    )
    assert isinstance(diag, PcaDiagnostics)
    for name in ("means", "unit", "projections", "order",
                 "selected_indices", "covariance", "eigenvalues", "eigenvector"):
        t = getattr(diag, name)
        assert t.is_cuda, name
    actual = actual.cpu().numpy()
    np.testing.assert_allclose(
        actual, expected, atol=STATISTICS.atol, rtol=STATISTICS.rtol,
        err_msg=f"pca_means diverged for {tag}",
    )


@pytest.mark.gpu_hardware
def test_pca_random_rows(cuda_device):
    torch = pytest.importorskip("torch")
    _compare(_rows(1), 10.0, cuda_device, torch, "random")


@pytest.mark.gpu_hardware
def test_pca_small_percentage(cuda_device):
    torch = pytest.importorskip("torch")
    _compare(_rows(2, 37), 5.0, cuda_device, torch, "small_pct")


@pytest.mark.gpu_hardware
def test_pca_rank_one(cuda_device):
    torch = pytest.importorskip("torch")
    direction = np.array([2.0, 1.0, 0.5])
    rows = np.outer(np.linspace(0.1, 1.0, 60), direction) * 1000
    _compare(rows, 20.0, cuda_device, torch, "rank_one")


@pytest.mark.gpu_hardware
def test_pca_repeated_eigenvalues(cuda_device):
    torch = pytest.importorskip("torch")
    base = np.array([1000.0, 500.0, 200.0])
    rows = np.tile(base, (50, 1))
    _compare(rows, 30.0, cuda_device, torch, "constant_rows")


@pytest.mark.gpu_hardware
def test_pca_equal_projection_ties_original_fixture(cuda_device):
    torch = pytest.importorskip("torch")
    rows = np.tile(
        np.array([[2.0, 1.0, 0.0], [1.0, 0.0, 2.0], [0.0, 2.0, 1.0]], np.float64),
        (10, 1),
    )
    _compare(rows, 10.0, cuda_device, torch, "ties_original")


@pytest.mark.gpu_hardware
def test_pca_equal_projection_ties_scaled_fixture(cuda_device):
    torch = pytest.importorskip("torch")
    rows = np.tile(
        np.array([[2.0, 1.0, 0.0], [1.0, 0.0, 2.0], [0.0, 2.0, 1.0]], np.float64),
        (10, 1),
    ) * 1000 + 100
    _compare(rows, 10.0, cuda_device, torch, "ties_scaled")


@pytest.mark.gpu_hardware
def test_pca_equal_projection_ties_shuffled(cuda_device):
    """Fixed-seed nonidentical equal-projection rows, shuffled — not the
    curated cyclic fixture."""
    torch = pytest.importorskip("torch")
    base = np.array([[3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0],
                     [2.0, 2.0, 1.0], [1.0, 2.0, 2.0]], np.float64) * 700 + 50
    rng = np.random.RandomState(777)
    rows = base[rng.randint(0, len(base), 90)]
    rng.shuffle(rows)
    _compare(rows, 10.0, cuda_device, torch, "ties_shuffled")


@pytest.mark.gpu_hardware
def test_pca_diagnostics_gpu_tensors(cuda_device):
    torch = pytest.importorskip("torch")
    rows = _rows(5, 120)
    means3, diag = pca_means(
        torch.as_tensor(rows, dtype=torch.float64, device=cuda_device),
        25.0, return_diagnostics=True,
    )
    assert diag.projections.shape == (120,)
    assert diag.order.shape == (120,)
    assert diag.selected_indices.shape == (2 * max(1, int(np.ceil(120 * 0.25))),)
    assert diag.covariance.shape == (3, 3)
    assert means3.shape == (3,)
    np.testing.assert_allclose(
        diag.covariance.cpu().numpy(),
        (rows[diag.selected_indices.cpu().numpy()].T
         @ rows[diag.selected_indices.cpu().numpy()]),
        rtol=1e-10, atol=1e-6,
    )


@pytest.mark.gpu_hardware
def test_pca_rejects_cpu_rows(cuda_device):
    torch = pytest.importorskip("torch")
    with pytest.raises(GpuError):
        pca_means(torch.rand(10, 3, dtype=torch.float64), 10.0)


def test_pca_rejects_cpu_tensor_no_device():
    torch = pytest.importorskip("torch")
    with pytest.raises(GpuError):
        pca_means(torch.rand(10, 3, dtype=torch.float64), 10.0)


def test_pca_bad_inputs(cuda_device):
    torch = pytest.importorskip("torch")
    with pytest.raises(GpuError):
        pca_means(torch.zeros(10, 3, dtype=torch.float64, device=cuda_device), 10.0)
    with pytest.raises(GpuError):
        pca_means(torch.rand(0, 3, device=cuda_device, dtype=torch.float64), 10.0)
    with pytest.raises(GpuError):
        pca_means(torch.rand(10, 3, device=cuda_device, dtype=torch.float64), 0.0)
    with pytest.raises(GpuError):
        pca_means(torch.rand(10, 3, device=cuda_device, dtype=torch.float64), 101.0)
