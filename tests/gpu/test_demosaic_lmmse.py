"""GPU LMMSE demosaic vs Demosaic.compute_linear (seed 10801)."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from brilliantisp_gpu.blocks.demosaic import demosaic
from brilliantisp_gpu.types import GpuError
from modules.demosaic.demosaic import Demosaic
from tools.gpu_validation.contracts import FLOAT32_NORMALIZED
from tools.gpu_validation.metrics import compare_arrays

PLATFORM = {
    "in_file": "fixture_9x7", "disable_progress_bar": True,
    "leave_pbar_string": False, "debug_enabled": False,
}
PATTERNS = ("rggb", "bggr", "grbg", "gbrg")
SHAPES = [(1, 1), (1, 7), (7, 1), (2, 2), (7, 9), (8, 10)]
LAYOUTS = {"rggb": (0, 1, 2, 3), "bggr": (3, 2, 1, 0),
           "grbg": (1, 0, 3, 2), "gbrg": (2, 3, 0, 1)}
ALGOS = ("lmmse", "lmmse_opt", "lmmse_fast")


def _sensor(shape, pattern):
    return dict(height=shape[-2], width=shape[-1], bayer_pattern=pattern,
                bit_depth=16, hdr_bit_depth=16)


def _params(algo):
    return dict(is_save=False, algorithm=algo)


def _oracle(frames, sensor, params):
    algo = params["algorithm"]
    return np.stack([
        Demosaic(f.copy(), dict(PLATFORM), sensor,
                 copy.deepcopy(params)).compute_linear(algo)
        for f in frames])


def _site_mask(shape, pattern, out_ch):
    rgb_channels = {0: (0,), 1: (1, 2), 2: (3,)}
    h, w = shape
    m = np.zeros((h, w), bool)
    for site in range(4):
        row, col = divmod(site, 2)
        if LAYOUTS[pattern][site] in rgb_channels[out_ch]:
            m[row::2, col::2] = True
    return m


def _run(frames, sensor, params, device):
    import torch
    image = torch.as_tensor(np.asarray(frames), device=device)
    before = image.clone()
    p_before = copy.deepcopy(params)
    s_before = copy.deepcopy(sensor)
    out = demosaic(image, sensor, params)
    np.testing.assert_array_equal(image.cpu().numpy(), before.cpu().numpy())
    assert params == p_before
    assert sensor == s_before
    assert out.dtype == torch.float32
    assert out.shape == frames.shape + (3,)
    assert out.device == image.device
    for c in range(3):
        m = _site_mask(frames.shape[-2:], sensor["bayer_pattern"], c)
        np.testing.assert_array_equal(
            out[..., c][:, m].cpu().numpy(),
            np.asarray(frames, np.float32)[:, m])
    return out


def _check(exp, act, tag):
    res = compare_arrays(np.asarray(exp), np.asarray(act),
                         FLOAT32_NORMALIZED, scale=65535.0)
    assert res["passed"], (tag, res)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("algo", ALGOS)
def test_lmmse_random(tensor_device, shape, pattern, algo):
    rng = np.random.RandomState(10801)
    frames = (rng.rand(2, *shape) * 65535).astype(np.float32)
    frames[1] *= 0.37
    sensor = _sensor(shape, pattern)
    out = _run(frames, sensor, _params(algo), tensor_device)
    _check(_oracle(frames, sensor, _params(algo)), out.cpu().numpy(),
           (shape, pattern, algo))


@pytest.mark.parametrize("kind", ["zero", "mid", "ramp", "checker", "impulse"])
@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("algo", ALGOS)
def test_lmmse_structured(tensor_device, kind, pattern, algo):
    shape = (7, 9)
    dims = (2, *shape)
    if kind == "zero":
        frames = np.zeros(dims, np.float32)
    elif kind == "mid":
        frames = np.full(dims, 32768.0, np.float32)
    elif kind == "ramp":
        frames = (np.linspace(0, 65535, np.prod(dims))
                  .reshape(dims)).astype(np.float32)
    elif kind == "checker":
        base = (np.indices(shape).sum(0) % 2 * 65535).astype(np.float32)
        frames = np.broadcast_to(base, dims).copy()
    else:
        frames = np.zeros(dims, np.float32)
        frames[:, shape[0] // 2, shape[1] // 2] = 65535.0
    sensor = _sensor(shape, pattern)
    out = _run(frames, sensor, _params(algo), tensor_device)
    _check(_oracle(frames, sensor, _params(algo)), out.cpu().numpy(),
           (kind, pattern, algo))


@pytest.mark.parametrize("amp", [8.0, 128.0])
@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("algo", ALGOS)
def test_lmmse_signed_headroom(tensor_device, amp, pattern, algo):
    rng = np.random.RandomState(10801)
    frames = (rng.rand(2, 8, 10) * amp * 65535
              - amp * 32767).astype(np.float32)
    sensor = _sensor((8, 10), pattern)
    out = _run(frames, sensor, _params(algo), tensor_device)
    _check(_oracle(frames, sensor, _params(algo)), out.cpu().numpy(),
           (amp, pattern, algo))


@pytest.mark.parametrize("algo", ALGOS)
def test_lmmse_f64_input(tensor_device, algo):
    rng = np.random.RandomState(10801)
    frames = (rng.rand(2, 8, 10) * 16777217).astype(np.float64)
    frames[0, 0, 0] = 16777217.0
    sensor = _sensor((8, 10), "rggb")
    out = _run(frames, sensor, _params(algo), tensor_device)
    _check(_oracle(frames, sensor, _params(algo)), out.cpu().numpy(),
           ("f64", algo))


def test_lmmse_batch_independence(tensor_device):
    rng = np.random.RandomState(10801)
    frames = np.empty((2, 7, 9), np.float32)
    frames[0] = rng.rand(7, 9) * 65535
    frames[1] = 1234.0
    sensor = _sensor((7, 9), "rggb")
    for algo in ALGOS:
        out = _run(frames, sensor, _params(algo), tensor_device)
        _check(_oracle(frames, sensor, _params(algo)), out.cpu().numpy(),
               ("batch", algo))


def _cpu_stats(kind, channel, mask):
    """CPU private stats oracle: (mean, var)."""
    m3 = [np.ones_like(mask), np.ones_like(mask), np.ones_like(mask)]
    if kind == "basic":
        from modules.demosaic.lmmse_demosaic import LMMSEDemosaic as C
        obj = C(channel.copy(), m3)
        mean = obj._estimate_local_mean(channel, mask)
        var = obj._estimate_local_variance(channel, mask, mean)
    elif kind == "optimized":
        from modules.demosaic.lmmse_demosaic import LMMSEDemosaicOptimized as C
        obj = C(channel.copy(), m3)
        mean, var = obj._compute_local_statistics(channel, mask)
    else:
        from modules.demosaic.lmmse_demosaic import LMMSEDemosaicFast as C
        obj = C(channel.copy(), m3)
        mean, var = obj._fast_local_mean_var(channel, mask)
    return mean, var


@pytest.mark.parametrize("kind", ["basic", "optimized", "fast"])
@pytest.mark.parametrize("scene", ["random", "constant", "nearly", "zero"])
@pytest.mark.parametrize("mask", ["ones", "r", "zero"])
def test_lmmse_local_stats(tensor_device, kind, scene, mask):
    """_local_stats vs CPU private helpers; gate + exact zero-var mask."""
    from brilliantisp_gpu.blocks.demosaic_lmmse import _local_stats
    import torch
    rng = np.random.RandomState(10801)
    ch = {
        "random": rng.rand(7, 9) * 65535,
        "constant": np.full((7, 9), 30000.0),
        "nearly": np.full((7, 9), 100.0, np.float64),
        "zero": np.zeros((7, 9)),
    }[scene]
    if scene == "nearly":
        ch = ch.astype(np.float32)
        ch[3, 4] = np.nextafter(np.float32(100.0), np.float32(np.inf))
        ch = ch.astype(np.float64)
    m = {
        "ones": np.ones((7, 9)),
        "r": (np.indices((7, 9))[0] % 2 == 0)
        & (np.indices((7, 9))[1] % 2 == 0),
        "zero": np.zeros((7, 9)),
    }[mask].astype(np.float64)
    b = 2
    ch_b = np.stack([ch, ch * 0.5]).astype(np.float32)
    m_b = np.stack([m, m]).astype(np.float32)
    mean, var = _local_stats(
        torch.as_tensor(ch_b, device=tensor_device),
        torch.as_tensor(m_b, device=tensor_device),
        {"basic": 5, "optimized": 7, "fast": 3}[kind],
        fast=(kind == "fast"))
    ch_dev = torch.as_tensor(ch_b, device=tensor_device).device
    assert mean.dtype == torch.float32 and mean.device == ch_dev
    assert var.dtype == torch.float32 and var.device == ch_dev
    cpu_mean, cpu_var = zip(*[_cpu_stats(kind, ch_b[i], m_b[i])
                              for i in range(b)])
    cpu_mean = np.stack(cpu_mean)
    cpu_var = np.stack(cpu_var)
    res = compare_arrays(cpu_mean, mean.cpu().numpy(),
                         FLOAT32_NORMALIZED, scale=65535.0)
    assert res["passed"], ("mean", kind, scene, mask, res)
    res = compare_arrays(cpu_var, var.cpu().numpy(),
                         FLOAT32_NORMALIZED, scale=65535.0 ** 2)
    assert res["passed"], ("var", kind, scene, mask, res)
    if scene in ("constant", "nearly", "zero"):
        np.testing.assert_array_equal(var.cpu().numpy() == 0.0,
                                      cpu_var == 0.0)
