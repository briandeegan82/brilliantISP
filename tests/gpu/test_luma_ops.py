"""GPU luma helpers vs util.luma (seed 10101)."""

from __future__ import annotations

import numpy as np
import pytest

from brilliantisp_gpu.ops.luma import normalize_plane, restore_plane, replace_luminance
from brilliantisp_gpu.types import GpuError
from util.luma import (normalize_plane as cpu_norm, restore_plane as cpu_restore,
                       replace_luminance as cpu_replace)
from tools.gpu_validation.metrics import compare_arrays
from tools.gpu_validation.contracts import FLOAT32_NORMALIZED, FLOAT64_HELPER

SHAPES = [(1, 1), (1, 7), (7, 1), (7, 9), (8, 10)]


def _t(a, d):
    import torch
    return torch.as_tensor(np.asarray(a), device=d)


def _check(exp, act, gate, tag):
    res = compare_arrays(np.asarray(exp), np.asarray(act), gate, scale=1.0)
    assert res["passed"], (tag, res)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("scale", [1.0, 255.0, 65535.0])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.uint16])
@pytest.mark.parametrize("batched", [True, False])
def test_normalize_restore(tensor_device, shape, scale, dtype, batched):
    rng = np.random.RandomState(10101)
    dims = (2, *shape) if batched else shape
    a = (rng.rand(*dims) * scale * 1.2 - 0.05 * scale).astype(dtype)
    image = _t(a, tensor_device)
    before = image.clone()
    norm = normalize_plane(image, scale=scale)
    assert norm.dtype == __import__("torch").float64
    np.testing.assert_array_equal(image.cpu().numpy(), before.cpu().numpy())
    if batched:
        exp = np.stack([cpu_norm(f, scale=scale) for f in a])
    else:
        exp = cpu_norm(a, scale=scale)
    _check(exp, norm.cpu().numpy(), FLOAT64_HELPER, (shape, scale, dtype))
    out = restore_plane(norm, scale=scale)
    assert out.dtype == __import__("torch").float32
    assert out.shape == image.shape
    exp_r = (np.stack([cpu_restore(f, scale=scale) for f in exp])
             if batched else cpu_restore(exp, scale=scale))
    _check(exp_r, out.cpu().numpy(), FLOAT32_NORMALIZED, (shape, scale, dtype, "r"))


@pytest.mark.parametrize("scale", [1e-40, 1e40, 1.000000000000001])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_normalize_promotion(tensor_device, scale, dtype):
    """Clip dtype follows np.result_type(input, 0.0, scale)."""
    rng = np.random.RandomState(10101)
    a = rng.rand(2, 8, 10).astype(dtype)
    norm = normalize_plane(_t(a, tensor_device), scale=scale)
    exp = np.stack([cpu_norm(f, scale=scale) for f in a])
    _check(exp, norm.cpu().numpy(), FLOAT64_HELPER, (scale, dtype))


def test_plane_validation(tensor_device):
    good = np.ones((2, 8, 10), np.float32)
    for bad_scale in (0, -1, float("nan"), float("inf"), True, "x"):
        with pytest.raises(ValueError):
            normalize_plane(_t(good, tensor_device), scale=bad_scale)
        with pytest.raises(ValueError):
            restore_plane(_t(good, tensor_device), scale=bad_scale)
    with pytest.raises(GpuError):
        normalize_plane(_t(np.ones(8), tensor_device), scale=1.0)
    with pytest.raises(GpuError):
        normalize_plane(_t(np.zeros(0), tensor_device), scale=1.0)
    with pytest.raises(ValueError):
        normalize_plane(_t(np.full((2, 4, 4), np.nan), tensor_device), scale=1.0)
    with pytest.raises(GpuError):
        normalize_plane(_t(np.ones((4, 4), np.complex64), tensor_device), scale=1.0)


def test_restore_overflow(tensor_device):
    huge = np.full((2, 8, 10), 1e30, np.float64)
    with pytest.raises(ValueError):
        restore_plane(_t(huge, tensor_device), scale=1e30)


@pytest.mark.parametrize("matrix", ["bt601", "bt709"])
@pytest.mark.parametrize("batched", [True, False])
def test_replace_luminance(tensor_device, matrix, batched):
    rng = np.random.RandomState(10101)
    dims = (2, 8, 10, 3) if batched else (8, 10, 3)
    rgb = (rng.rand(*dims) * 2.0 - 0.4).astype(np.float64)
    new_y = rng.rand(*(dims[:3] if batched else dims[:2])).astype(np.float64)
    image = _t(rgb, tensor_device)
    before = image.clone()
    out = replace_luminance(image, _t(new_y, tensor_device), matrix=matrix)
    np.testing.assert_array_equal(image.cpu().numpy(), before.cpu().numpy())
    assert out.dtype == __import__("torch").float32
    exp = (np.stack([cpu_replace(rgb[i], new_y[i], matrix=matrix) for i in range(2)])
           if batched else cpu_replace(rgb, new_y, matrix=matrix))
    _check(exp, out.cpu().numpy(), FLOAT32_NORMALIZED, (matrix, batched))
    with pytest.raises(GpuError):
        replace_luminance(_t(np.ones((8, 10), np.float64), tensor_device), _t(new_y[0] if batched else new_y, tensor_device))
    with pytest.raises(ValueError):
        replace_luminance(image, _t(np.ones((9, 9)), tensor_device))


def test_normalize_extra_asserts(tensor_device):
    import torch
    rng = np.random.RandomState(10101)
    a = rng.rand(2, 8, 10).astype(np.float64)
    image = _t(a, tensor_device)
    norm = normalize_plane(image, scale=255.0)
    assert norm.shape == image.shape
    assert norm.device == image.device
    out = restore_plane(norm, scale=255.0)
    assert out.device == image.device
    for bad in (float("inf"), -float("inf")):
        x = a.copy(); x[0, 0, 0] = bad
        with pytest.raises(ValueError):
            normalize_plane(_t(x, tensor_device), scale=1.0)
    with pytest.raises(GpuError):
        normalize_plane(_t(np.ones((4, 4), np.bool_), tensor_device), scale=1.0)


def test_restore_signed_headroom(tensor_device):
    """Signed/above-scale normalized values preserve (no clip in restore)."""
    x = np.array([[-1.0, 0.5, 2.0], [0.0, -0.25, 1.0]], np.float64)[None].repeat(2, 0)
    exp = np.stack([cpu_restore(f, scale=10.0) for f in x])
    out = restore_plane(_t(x, tensor_device), scale=10.0)
    np.testing.assert_array_equal(out.cpu().numpy(), exp)
    assert out[0, 0, 0] == -10.0 and out[0, 0, 2] == 20.0


def test_replace_newy_devices(tensor_device):
    import torch
    rng = np.random.RandomState(10101)
    rgb = rng.rand(2, 8, 10, 3).astype(np.float64)
    new_y = rng.rand(2, 8, 10).astype(np.float64)
    image = _t(rgb, tensor_device)
    for src_y in (new_y, torch.as_tensor(new_y)):
        exp = np.stack([cpu_replace(rgb[i], new_y[i]) for i in range(2)])
        out = replace_luminance(image, src_y)
        assert out.device == image.device
        np.testing.assert_array_equal(image.cpu().numpy(), rgb)
        np.testing.assert_array_equal(np.asarray(src_y.cpu() if hasattr(src_y, "cpu") else src_y), new_y)
        _check(exp, out.cpu().numpy(), FLOAT32_NORMALIZED, type(src_y))


def test_replace_rejects(tensor_device):
    import torch
    rgb = _t(np.ones((2, 8, 10, 3), np.float32), tensor_device)
    good_y = _t(np.ones((2, 8, 10), np.float64), tensor_device)
    with pytest.raises(GpuError):
        replace_luminance(_t(np.ones((8, 10, 3), np.int32), tensor_device), good_y[0])
    with pytest.raises(ValueError):
        replace_luminance(rgb, _t(np.ones((2, 8, 10), np.bool_), tensor_device))
    with pytest.raises(ValueError):
        replace_luminance(rgb, _t(np.full((2, 8, 10), np.inf), tensor_device))
    with pytest.raises(ValueError):
        replace_luminance(rgb, good_y, matrix="bogus")
    huge = _t(np.full((2, 8, 10, 3), 1e39, np.float64), tensor_device)
    with pytest.raises(ValueError):
        replace_luminance(huge, _t(np.full((2, 8, 10), 1e39, np.float64), tensor_device))
