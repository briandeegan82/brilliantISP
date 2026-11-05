"""GPU clahe_luma vs native modules.ldci.clahe.apply_luma (seed 10201)."""

from __future__ import annotations

import numpy as np
import pytest

from brilliantisp_gpu.blocks.luma import clahe_luma
from brilliantisp_gpu.types import GpuError
from modules.ldci.clahe import apply_luma as cpu_clahe
from tools.gpu_validation.metrics import compare_arrays
from tools.gpu_validation.contracts import FLOAT32_NORMALIZED

SHAPES = [(1, 1), (1, 7), (7, 1), (7, 9), (8, 10), (17, 19)]


def _t(a, d):
    import torch
    return torch.as_tensor(np.asarray(a), device=d)


def _check(exp, act, scale, tag):
    res = compare_arrays(np.asarray(exp), np.asarray(act),
                         FLOAT32_NORMALIZED, scale=scale)
    assert res["passed"], (tag, res)


def _run(frames, device, scale, **kw):
    import torch
    image = _t(frames, device)
    before = image.clone()
    out = clahe_luma(image, scale=scale, **kw)
    np.testing.assert_array_equal(image.cpu().numpy(), before.cpu().numpy())
    assert out.dtype == torch.float32
    assert out.shape == image.shape
    assert out.device == image.device
    exp = (np.stack([cpu_clahe(f, scale=scale, **kw) for f in frames])
           if frames.ndim == 3 else cpu_clahe(frames, scale=scale, **kw))
    _check(exp, out.cpu().numpy(), scale, (frames.shape, kw))


def _frames(shape, kind, scale, seed=10201):
    rng = np.random.RandomState(seed)
    dims = (2, *shape)
    if kind == "random":
        a = rng.rand(*dims) * scale
    elif kind == "const":
        a = np.full(dims, 0.5 * scale)
    elif kind == "ramp":
        a = np.linspace(0, 1, np.prod(dims)).reshape(dims) * scale
    elif kind == "checker":
        base = (np.indices((dims[-2], dims[-1])).sum(0) % 2) * scale
        a = np.broadcast_to(base, dims).copy()
    elif kind == "impulse":
        a = np.zeros(dims); a[..., shape[-2] // 2, shape[-1] // 2] = scale
    elif kind == "mixed_const":
        a = rng.rand(*dims) * scale
        a[0, :shape[0] // 2, :shape[1] // 2] = 0.3 * scale
    return a.astype(np.float32)


@pytest.mark.parametrize("wind", [1, 2, 3, 4, 8, 64])
@pytest.mark.parametrize("shape", [(7, 9), (17, 19)])
def test_clahe_winds(tensor_device, wind, shape):
    a = _frames(shape, "random", 65535.0)
    _run(a, tensor_device, 65535.0, wind=wind, clip_limit=2.0)


@pytest.mark.parametrize("bins", [256, 4096])
@pytest.mark.parametrize("clip", [0.0, 0.5, 2.0, 4.0])
def test_clahe_bins_clip(tensor_device, bins, clip):
    a = _frames((8, 10), "random", 65535.0)
    _run(a, tensor_device, 65535.0, wind=4, clip_limit=clip, bins=bins)


@pytest.mark.parametrize("kind", ["const", "ramp", "checker", "impulse", "mixed_const"])
@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (7, 1), (7, 9), (8, 10), (17, 19)])
def test_clahe_structured(tensor_device, kind, shape):
    a = _frames(shape, kind, 65535.0)
    _run(a, tensor_device, 65535.0, wind=4, clip_limit=2.0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.uint16])
@pytest.mark.parametrize("scale", [1.0, 255.0, 65535.0])
@pytest.mark.parametrize("batched", [True, False])
def test_clahe_dtypes_scales(tensor_device, dtype, scale, batched):
    rng = np.random.RandomState(10201)
    dims = (2, 8, 10) if batched else (8, 10)
    a = (rng.rand(*dims) * scale).astype(dtype)
    _run(a, tensor_device, scale, wind=4, clip_limit=2.0)


def test_clahe_negative_headroom_constant(tensor_device):
    """Constant frame incl. after-normalize edge -> restore(Y) identity."""
    a = np.full((2, 8, 10), -5.0, np.float64)
    _run(a, tensor_device, 65535.0, wind=4, clip_limit=2.0)
    b = np.full((2, 8, 10), 1e5, np.float64)
    _run(b, tensor_device, 65535.0, wind=4, clip_limit=2.0)


@pytest.mark.parametrize("bins", [256, 4096])
def test_clahe_bin_boundary(tensor_device, bins):
    """Half-bin nextafter triplets and endpoints vs oracle."""
    rng = np.random.RandomState(10201)
    a = rng.rand(2, 8, 10)
    lo = np.nextafter(0.5, -np.inf)
    hi = np.nextafter(0.5, np.inf)
    assert lo < 0.5 < hi
    a[0, 0, 0] = lo
    a[0, 0, 1] = 0.5
    a[0, 0, 2] = hi
    a[1, -1, -3] = 0.0
    a[1, -1, -2] = np.nextafter(1.0, -np.inf)
    a[1, -1, -1] = 1.0
    _run(a.astype(np.float64), tensor_device, 1.0, wind=4, clip_limit=2.0,
         bins=bins)


def test_clahe_validation_constant_frames(tensor_device):
    """Controls validated BEFORE constant-frame fast return."""
    const = _t(np.full((2, 8, 10), 100.0, np.float32), tensor_device)
    for kw in (dict(wind=0, clip_limit=1.0), dict(wind=True, clip_limit=1.0),
               dict(wind=-2, clip_limit=1.0), dict(wind=None, clip_limit=1.0),
               dict(wind=float("inf"), clip_limit=1.0),
               dict(wind=float("nan"), clip_limit=1.0),
               dict(wind=4, clip_limit=-1.0),
               dict(wind=4, clip_limit=float("nan")),
               dict(wind=4, clip_limit=float("inf")),
               dict(wind=4, clip_limit=True),
               dict(wind=4, clip_limit=1.0, bins=512),
               dict(wind=4, clip_limit=1.0, bins=None),
               dict(wind=4, clip_limit=1.0, bins=float("inf")),
               dict(wind=4, clip_limit=1.0, bins=True)):
        with pytest.raises(ValueError):
            clahe_luma(const, scale=255.0, **kw)
    out = clahe_luma(const, scale=255.0, wind=4, clip_limit="2.0")
    assert out.shape == const.shape


def test_clahe_rejects(tensor_device):
    good = _t(np.ones((2, 8, 10), np.float32), tensor_device)
    with pytest.raises(ValueError):
        clahe_luma(good, scale=0.0, wind=4, clip_limit=1.0)
    import torch
    with pytest.raises(GpuError):
        clahe_luma(torch.ones(1, 8, 10, 3, device=good.device), scale=1.0,
                   wind=4, clip_limit=1.0)
    with pytest.raises(GpuError):
        clahe_luma(torch.ones(0, 8, 10, device=good.device), scale=1.0,
                   wind=4, clip_limit=1.0)


def test_clahe_neighbor_tiles_distinct(tensor_device):
    """Distinct adjacent-tile CDFs so wrong NE/SE tile index is visible."""
    a = np.zeros((2, 8, 10), np.float64)
    a[:, :, :6] = 0.05 * 65535.0
    a[:, :, 6:] = 0.95 * 65535.0
    a[0, 2, 3] = 0.5 * 65535.0
    _run(a, tensor_device, 65535.0, wind=4, clip_limit=2.0)
