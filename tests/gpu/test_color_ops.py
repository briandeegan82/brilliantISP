"""GPU luminance/adjust_saturation vs util.color (seed 9801)."""

from __future__ import annotations

import numpy as np
import pytest

from brilliantisp_gpu.ops.color import luminance, adjust_saturation
from brilliantisp_gpu.types import GpuError
from util.color import luminance as cpu_luma, adjust_saturation as cpu_sat
from tools.gpu_validation.metrics import compare_arrays
from tools.gpu_validation.contracts import FLOAT32_NORMALIZED, FLOAT64_HELPER

SHAPES = [(1, 1), (1, 7), (7, 1), (7, 9), (8, 10)]


def _t(arr, device, dt=None):
    import torch
    return torch.as_tensor(np.asarray(arr) if dt is None else np.asarray(arr, dtype=dt),
                           device=device)


def _check(expected, actual, gate, tag):
    res = compare_arrays(np.asarray(expected), np.asarray(actual), gate, scale=1.0)
    assert res["passed"], (tag, res)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("matrix", ["bt601", "bt709"])
@pytest.mark.parametrize("in_dt", [np.float32, np.float64])
@pytest.mark.parametrize("out_dt", [np.float32, np.float64])
def test_luminance(tensor_device, shape, matrix, in_dt, out_dt):
    import torch
    rng = np.random.RandomState(9801)
    rgb = (rng.rand(2, *shape, 3) * 128 - 32).astype(in_dt)
    expected = cpu_luma(rgb, matrix=matrix, dtype=out_dt)
    image = _t(rgb, tensor_device)
    before = image.clone()
    out = luminance(image, matrix=matrix, dtype=out_dt)
    assert out.dtype == (torch.float64 if out_dt == np.float64 else torch.float32)
    assert out.shape == (2, *shape)
    assert out.device == image.device
    np.testing.assert_array_equal(image.cpu().numpy(), before.cpu().numpy())
    gate = FLOAT32_NORMALIZED if out_dt == np.float32 else FLOAT64_HELPER
    _check(expected, out.cpu().numpy(), gate, (shape, matrix, in_dt, out_dt))


def test_luminance_integer_input(tensor_device):
    rng = np.random.RandomState(9801)
    rgb = rng.randint(0, 4096, size=(2, 8, 10, 3)).astype(np.uint16)
    expected = cpu_luma(rgb, matrix="bt709", dtype=np.float32)
    out = luminance(_t(rgb, tensor_device, np.uint16), matrix="bt709")
    _check(expected, out.cpu().numpy(), FLOAT32_NORMALIZED, "int")


def test_luminance_default_dtype(tensor_device):
    rgb = np.ones((2, 4, 4, 3), np.float32)
    import torch
    assert luminance(_t(rgb, tensor_device)).dtype == torch.float32
    assert luminance(_t(rgb, tensor_device), dtype=None).dtype == torch.float64


@pytest.mark.parametrize("gain", [0.0, 1.0, 0.5, 2.0])
@pytest.mark.parametrize("matrix", ["bt601", "bt709"])
@pytest.mark.parametrize("in_dt", [np.float32, np.float64])
@pytest.mark.parametrize("out_dt", [np.float32, np.float64])
def test_adjust_saturation(tensor_device, gain, matrix, in_dt, out_dt):
    import torch
    rng = np.random.RandomState(9801)
    rgb = (rng.rand(2, 8, 10, 3) * 128 - 32).astype(in_dt)
    expected = cpu_sat(rgb, gain, matrix=matrix, dtype=out_dt)
    image = _t(rgb, tensor_device)
    before = image.clone()
    out = adjust_saturation(image, gain, matrix=matrix, dtype=out_dt)
    assert out.shape == rgb.shape
    assert out.dtype == (torch.float64 if out_dt == np.float64 else torch.float32)
    np.testing.assert_array_equal(image.cpu().numpy(), before.cpu().numpy())
    _check(expected, out.cpu().numpy(),
           FLOAT64_HELPER if out_dt == np.float64 else FLOAT32_NORMALIZED,
           (gain, matrix, in_dt, out_dt))
    if gain == 0.0:
        y = out.cpu().numpy()
        np.testing.assert_array_equal(y[..., 0], y[..., 1], err_msg="gain0 not gray")
    if gain == 1.0:
        np.testing.assert_array_equal(out.cpu().numpy(), expected)


def test_saturation_gain1_f64_no_alias(tensor_device):
    """gain=1 f64 output must not alias the input (CPU _finish copies)."""
    import torch
    image = _t(np.full((2, 4, 4, 3), 0.5, np.float64), tensor_device)
    before = image.clone()
    out = adjust_saturation(image, 1.0, dtype=np.float64)
    out[0, 0, 0, 0] = 999.0
    np.testing.assert_array_equal(image.cpu().numpy(), before.cpu().numpy())


def test_saturation_bad_gain_matrix_dtype(tensor_device):
    import torch
    rgb = _t(np.ones((2, 8, 10, 3), np.float32), tensor_device)
    for bad_gain in (True, "x", float("nan"), -1.0):
        with pytest.raises(ValueError):
            adjust_saturation(rgb, bad_gain)
    with pytest.raises(ValueError):
        luminance(rgb, matrix="nope")
    with pytest.raises(ValueError):
        adjust_saturation(rgb, 1.0, matrix="nope")
    with pytest.raises(ValueError):
        luminance(rgb, dtype=np.int32)
    with pytest.raises(ValueError):
        luminance(rgb, dtype="bogus_dtype")
    with pytest.raises(ValueError):
        luminance(rgb, dtype=torch.int32)
    bad = _t(np.full((2, 8, 10, 3), np.nan), tensor_device)
    with pytest.raises(ValueError):
        luminance(bad)
    with pytest.raises(ValueError):
        adjust_saturation(bad, 1.0)
    with pytest.raises(GpuError):
        luminance(_t(np.ones((0, 8, 10, 3)), tensor_device))
    with pytest.raises(GpuError):
        luminance(_t(np.ones((2, 8, 10)), tensor_device))
    with pytest.raises(GpuError):
        luminance(torch.ones(2, 8, 10, 3, dtype=torch.bool, device=tensor_device))
    with pytest.raises(GpuError):
        luminance(torch.ones(2, 8, 10, 3, dtype=torch.complex64, device=tensor_device))


def test_luminance_headroom_negative(tensor_device):
    rgb = np.array([[[-1.0, 2.0, 0.5], [3.0, -0.5, 1.25]]] * 2, np.float32)[None].repeat(2, 0)
    expected = cpu_luma(rgb, dtype=np.float32)
    out = luminance(_t(rgb, tensor_device))
    np.testing.assert_array_equal(out.cpu().numpy(), expected)
