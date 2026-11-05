"""GPU sharpen_luma + non_local_means vs native apply_luma (seed 10101)."""

from __future__ import annotations

import numpy as np
import pytest

from brilliantisp_gpu.blocks.luma import sharpen_luma, non_local_means
from modules.sharpen.unsharp_masking import apply_luma as cpu_sharpen
from modules.noise_reduction_2d.non_local_means import apply_luma as cpu_nlm
from tools.gpu_validation.metrics import compare_arrays
from tools.gpu_validation.contracts import FLOAT32_NORMALIZED

SHAPES = [(1, 1), (1, 7), (7, 1), (7, 9), (8, 10)]


def _t(a, d):
    import torch
    return torch.as_tensor(np.asarray(a), device=d)


def _check(exp, act, scale, tag):
    res = compare_arrays(np.asarray(exp), np.asarray(act),
                         FLOAT32_NORMALIZED, scale=scale)
    assert res["passed"], (tag, res)


def _batched(fn_cpu, fn_gpu, frames, device, scale, tag, **kw):
    image = _t(frames, device)
    before = image.clone()
    out = fn_gpu(image, scale=scale, **kw)
    np.testing.assert_array_equal(image.cpu().numpy(), before.cpu().numpy())
    assert out.dtype == __import__("torch").float32
    assert out.shape == image.shape
    exp = np.stack([fn_cpu(f, scale=scale, **kw) for f in frames]) \
        if frames.ndim == 3 else fn_cpu(frames, scale=scale, **kw)
    _check(exp, out.cpu().numpy(), scale, tag)


def _frames(shape, kind, scale, batched=True):
    rng = np.random.RandomState(10101)
    dims = (2, *shape) if batched else shape
    if kind == "random":
        a = rng.rand(*dims) * scale
    elif kind == "impulse":
        a = np.zeros(dims); a[..., shape[-2] // 2, shape[-1] // 2] = scale
    elif kind == "checker":
        a = ((np.indices((dims[-2], dims[-1])).sum(0) % 2) * scale)
        a = np.broadcast_to(a, dims).copy()
    elif kind == "const":
        a = np.full(dims, 0.5 * scale)
    elif kind == "ramp":
        a = (np.arange(np.prod(dims)).reshape(dims) % int(scale)) * (scale / int(scale))
    return a.astype(np.float32)


@pytest.mark.parametrize("sigma", [0.0, 0.5, 1.3, 2.0])
@pytest.mark.parametrize("strength", [0.0, 0.5, 1.5])
@pytest.mark.parametrize("shape", [(7, 9), (8, 10)])
def test_sharpen(tensor_device, sigma, strength, shape):
    a = _frames(shape, "random", 65535.0)
    _batched(cpu_sharpen, sharpen_luma, a, tensor_device, 65535.0,
             (sigma, strength, shape), sigma=sigma, strength=strength)


@pytest.mark.parametrize("kind", ["impulse", "checker", "const", "ramp"])
def test_sharpen_structured(tensor_device, kind):
    a = _frames((8, 10), kind, 65535.0)
    _batched(cpu_sharpen, sharpen_luma, a, tensor_device, 65535.0,
             kind, sigma=1.0, strength=0.8)


@pytest.mark.parametrize("win", [1, 3, 5])
@pytest.mark.parametrize("patch", [1, 3, 5])
@pytest.mark.parametrize("strength", [0.0, 0.01, 10.0, 255.0, 1e200])
def test_nlm(tensor_device, win, patch, strength):
    a = _frames((7, 9), "random", 65535.0)
    _batched(cpu_nlm, non_local_means, a, tensor_device, 65535.0,
             (win, patch, strength), window_size=win, patch_size=patch,
             strength=strength)


@pytest.mark.parametrize("kind", ["impulse", "checker", "const", "ramp"])
@pytest.mark.parametrize("shape", SHAPES)
def test_nlm_structured(tensor_device, kind, shape):
    a = _frames(shape, kind, 65535.0)
    _batched(cpu_nlm, non_local_means, a, tensor_device, 65535.0,
             (kind, shape), window_size=3, patch_size=3, strength=50.0)


def test_nlm_unbatched(tensor_device):
    a = _frames((8, 10), "random", 255.0, batched=False)
    _batched(cpu_nlm, non_local_means, a, tensor_device, 255.0,
             "unbatched", window_size=3, patch_size=1, strength=10.0)


def test_block_rejects(tensor_device):
    good = _t(np.ones((2, 8, 10), np.float32), tensor_device)
    for name in ("sigma", "strength"):
        kw = dict(sigma=1.0, strength=1.0); kw[name] = -1.0
        with pytest.raises(ValueError):
            sharpen_luma(good, scale=65535.0, **kw)
        kw[name] = float("nan")
        with pytest.raises(ValueError):
            sharpen_luma(good, scale=65535.0, **kw)
    with pytest.raises(ValueError):
        sharpen_luma(good, scale=65535.0, sigma=True, strength=1.0)
    for kw in (dict(window_size=4, patch_size=3, strength=1.0),
               dict(window_size=3, patch_size=4, strength=1.0),
               dict(window_size=0, patch_size=3, strength=1.0),
               dict(window_size=3, patch_size=3, strength=-1.0),
               dict(window_size=True, patch_size=3, strength=1.0),
               dict(window_size=3, patch_size=3, strength=float("nan"))):
        with pytest.raises(ValueError):
            non_local_means(good, scale=65535.0, **kw)
    with pytest.raises(ValueError):
        sharpen_luma(good, scale=0.0, sigma=1.0, strength=1.0)
    with pytest.raises(ValueError):
        non_local_means(good, scale=-1.0, window_size=3, patch_size=3, strength=1.0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.uint16])
@pytest.mark.parametrize("scale", [1.0, 255.0, 65535.0])
@pytest.mark.parametrize("batched", [True, False])
def test_sharpen_dtypes_scales(tensor_device, dtype, scale, batched):
    rng = np.random.RandomState(10101)
    dims = (2, 8, 10) if batched else (8, 10)
    a = (rng.rand(*dims) * scale).astype(dtype)
    _batched(cpu_sharpen, sharpen_luma, a, tensor_device, scale,
             (dtype, scale, batched), sigma=1.0, strength=1.0)


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (7, 1)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sharpen_tiny(tensor_device, shape, dtype):
    rng = np.random.RandomState(10101)
    a = (rng.rand(2, *shape) * 65535).astype(dtype)
    _batched(cpu_sharpen, sharpen_luma, a, tensor_device, 65535.0,
             (shape, dtype), sigma=0.5, strength=0.5)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.uint16])
@pytest.mark.parametrize("scale", [1.0, 255.0, 65535.0])
@pytest.mark.parametrize("batched", [True, False])
@pytest.mark.parametrize("strength", [10.0 / 255.0, 10.0, 255.0])
def test_nlm_dtypes_scales(tensor_device, dtype, scale, batched, strength):
    rng = np.random.RandomState(10101)
    dims = (2, 8, 10) if batched else (8, 10)
    a = (rng.rand(*dims) * scale).astype(dtype)
    _batched(cpu_nlm, non_local_means, a, tensor_device, scale,
             (dtype, scale, batched, strength),
             window_size=3, patch_size=3, strength=strength)


def test_blocks_batch_independence(tensor_device):
    """B2 distinct incl. constant frame, per-frame oracle."""
    rng = np.random.RandomState(10101)
    a = np.empty((2, 8, 10), np.float32)
    a[0] = rng.rand(8, 10) * 65535
    a[1] = 1000.0
    _batched(cpu_sharpen, sharpen_luma, a, tensor_device, 65535.0,
             "batch_sharpen", sigma=1.0, strength=1.2)
    _batched(cpu_nlm, non_local_means, a, tensor_device, 65535.0,
             "batch_nlm", window_size=3, patch_size=3, strength=10.0)
