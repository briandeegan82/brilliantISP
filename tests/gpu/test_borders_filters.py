"""Border index maps and correlation/convolution vs NumPy/SciPy oracles.

Index-map equality is exact; float comparisons use the lead-contract
helper gates. CPU tensors exercise the portable primitive; the 'cuda'
param additionally runs the same checks on the real device.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage, signal

from tools.gpu_validation.contracts import FLOAT64_HELPER, FLOAT32_NORMALIZED

from brilliantisp_gpu.ops import borders, filters

NP_MODES = {
    "reflect": "symmetric",
    "mirror": "reflect",
    "nearest": "edge",
    "wrap": "wrap",
}


def _check(actual, expected, gate):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    np.testing.assert_allclose(
        actual, expected, atol=gate.atol, rtol=gate.rtol,
        err_msg=f"gate {gate.name}",
    )


@pytest.mark.parametrize("n,radius", [(3, 1), (3, 5), (1, 3), (2, 4), (5, 2)])
@pytest.mark.parametrize("mode,np_mode", list(NP_MODES.items()))
def test_border_indices_match_numpy(tensor_device, n, radius, mode, np_mode):
    torch = pytest.importorskip("torch")
    oracle = np.pad(np.arange(n), radius, mode=np_mode)
    idx = borders.border_indices(n, radius, mode, device=tensor_device)
    gathered = idx.cpu().numpy()
    assert gathered.tolist() == oracle.tolist()


@pytest.mark.parametrize("n,radius", [(3, 1), (3, 5), (1, 3)])
def test_border_indices_constant(tensor_device, n, radius):
    positions = np.arange(-radius, n + radius)
    inside = (positions >= 0) & (positions < n)
    expected = np.where(inside, positions, -1)
    idx = borders.border_indices(n, radius, "constant", device=tensor_device)
    assert idx.cpu().numpy().tolist() == expected.tolist()


@pytest.mark.parametrize("mode,np_mode", list(NP_MODES.items()) + [("constant", "constant")])
def test_pad_matches_numpy(tensor_device, mode, np_mode):
    torch = pytest.importorskip("torch")
    x = np.arange(12, dtype=np.float64).reshape(3, 4)
    kwargs = {"constant_values": 7.0} if np_mode == "constant" else {}
    expected = np.pad(x, ((2, 2), (3, 3)), mode=np_mode, **kwargs)
    out = borders.pad(torch.as_tensor(x, device=tensor_device), (2, 3),
                      mode=mode, cval=7.0).cpu().numpy()
    np.testing.assert_array_equal(out, expected)


def _rand(shape, seed):
    return np.random.RandomState(seed).randn(*shape)


SCIPY_MODES = {
    "reflect": "reflect",
    "mirror": "mirror",
    "nearest": "nearest",
    "wrap": "wrap",
}


SEED_BASE = {
    "reflect": 101, "mirror": 102, "nearest": 103, "wrap": 104, "constant": 105,
}


@pytest.mark.parametrize("mode,scipy_mode", list(SCIPY_MODES.items()) + [("constant", "constant")])
@pytest.mark.parametrize("kernel_kind", ["sym", "asym", "even_asym"])
def test_correlate_matches_scipy(tensor_device, mode, scipy_mode, kernel_kind):
    torch = pytest.importorskip("torch")
    seed = SEED_BASE[mode] + {"sym": 0, "asym": 10, "even_asym": 20}[kernel_kind]
    x = _rand((7, 9), seed)
    if kernel_kind == "sym":
        k = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float64) / 16
    elif kernel_kind == "even_asym":
        k = np.array([[1, 0, -2], [3, 1, 5], [0, 2, 0], [-1, 0, 1]], np.float64)
    else:
        k = np.array([[1, 0, -2], [3, 1, 5], [-1, 2, 0]], np.float64)
    cval = 0.5
    expected = ndimage.correlate(x, k, mode=scipy_mode, cval=cval)
    out = filters.correlate(torch.as_tensor(x, device=tensor_device),
                            torch.as_tensor(k, device=tensor_device),
                            mode=mode, cval=cval).cpu().numpy()
    _check(out, expected, FLOAT64_HELPER)


@pytest.mark.parametrize("mode,scipy_mode", list(SCIPY_MODES.items()))
@pytest.mark.parametrize("kernel_kind", ["sym", "asym", "even_asym"])
def test_convolve_matches_scipy(tensor_device, mode, scipy_mode, kernel_kind):
    torch = pytest.importorskip("torch")
    seed = SEED_BASE[mode] + {"sym": 1, "asym": 11, "even_asym": 21}[kernel_kind]
    x = _rand((6, 8), seed)
    if kernel_kind == "sym":
        k = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float64) / 16
    elif kernel_kind == "even_asym":
        k = np.array([[1, 0], [3, 1], [-1, 2]], np.float64)
    else:
        k = np.array([[1, 0, -2], [3, 1, 5], [-1, 2, 0]], np.float64)
    expected = ndimage.convolve(x, k, mode=scipy_mode)
    out = filters.convolve(torch.as_tensor(x, device=tensor_device),
                           torch.as_tensor(k, device=tensor_device),
                           mode=mode).cpu().numpy()
    _check(out, expected, FLOAT64_HELPER)


def test_convolve_even_kernel_impulse_origin(tensor_device):
    """Even asymmetric kernel: scipy origin is shifted, never silently centered."""
    torch = pytest.importorskip("torch")
    x = np.zeros((6, 6)); x[3, 3] = 1.0
    for k in (
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([[1.0, 0.0, -2.0], [3.0, 1.0, 5.0], [0.0, 2.0, 0.0], [-1.0, 0.0, 1.0]]),
        np.array([[2.0, -1.0], [0.0, 3.0], [1.0, 1.0], [-2.0, 0.0]]),
    ):
        expected = ndimage.convolve(x, k, mode="constant")
        out = filters.convolve(torch.as_tensor(x, device=tensor_device),
                               torch.as_tensor(k, device=tensor_device),
                               mode="constant").cpu().numpy()
        _check(out, expected, FLOAT64_HELPER)


def test_correlate_matches_signal_symm(tensor_device):
    torch = pytest.importorskip("torch")
    x = _rand((6, 7), 11)
    k = np.array([[0.2, -0.1, 0.4], [1.0, 0.0, -1.0], [0.3, 0.3, 0.1]], np.float64)
    expected = signal.correlate2d(x, k, mode="same", boundary="symm")
    out = filters.correlate(torch.as_tensor(x, device=tensor_device),
                            torch.as_tensor(k, device=tensor_device),
                            mode="symm").cpu().numpy()
    _check(out, expected, FLOAT64_HELPER)


@pytest.mark.parametrize("sigma,truncate", [(0.8, 4.0), (2.0, 3.0), (1.3, 2.5), (0.0, 4.0), (0.05, 4.0)])
def test_gaussian_matches_scipy(tensor_device, sigma, truncate):
    torch = pytest.importorskip("torch")
    x = _rand((8, 10), 301)
    expected = ndimage.gaussian_filter(x, sigma, mode="reflect", truncate=truncate)
    out = filters.gaussian_filter(
        torch.as_tensor(x, device=tensor_device), sigma,
        mode="reflect", truncate=truncate,
    ).cpu().numpy()
    _check(out, expected, FLOAT64_HELPER)


def test_gaussian_float32_scipy_intermediate(tensor_device):
    """float32 input exercises scipy's float32 intermediate rounding."""
    torch = pytest.importorskip("torch")
    x = _rand((9, 11), 302).astype(np.float32)
    expected = ndimage.gaussian_filter(x, 1.4, mode="reflect", truncate=3.0)
    out = filters.gaussian_filter(
        torch.as_tensor(x, device=tensor_device), 1.4,
        mode="reflect", truncate=3.0,
    ).cpu().numpy()
    _check(out, expected.astype(np.float64), FLOAT32_NORMALIZED)


def test_gaussian_kernel1d_sigma0_identity(tensor_device):
    torch = pytest.importorskip("torch")
    k = filters.gaussian_kernel1d(0.0, device=tensor_device)
    assert k.numel() == 1
    np.testing.assert_array_equal(k.cpu().numpy(), [1.0])
    out = filters.gaussian_filter(
        torch.as_tensor(_rand((5, 5), 304), device=tensor_device), 0.0
    ).cpu().numpy()
    np.testing.assert_array_equal(out, _rand((5, 5), 304))


def test_gaussian_transpose_axes(tensor_device):
    torch = pytest.importorskip("torch")
    x = _rand((8, 10), 303)
    out = filters.gaussian_filter(torch.as_tensor(x, device=tensor_device), 1.2).cpu().numpy()
    out_t = filters.gaussian_filter(torch.as_tensor(x.T, device=tensor_device), 1.2).cpu().numpy()
    _check(out_t, out.T, FLOAT64_HELPER)


def test_float32_helper_precision(tensor_device):
    torch = pytest.importorskip("torch")
    x = _rand((9, 9), 5).astype(np.float32)
    k = np.zeros((3, 3), dtype=np.float32)
    k[1, 1] = 1.0
    out = filters.correlate(torch.as_tensor(x, device=tensor_device),
                            torch.as_tensor(k, device=tensor_device)).cpu().numpy()
    _check(out, x.astype(np.float64), FLOAT32_NORMALIZED)
