"""GPU AHD demosaic vs Demosaic.compute_linear (seed 10901)."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from brilliantisp_gpu.blocks.demosaic import demosaic
from brilliantisp_gpu.blocks.demosaic_ahd import _opt_alpha
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
ALGOS = ("ahd", "ahd_opt")


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
def test_ahd_random(tensor_device, shape, pattern, algo):
    rng = np.random.RandomState(10901)
    frames = (rng.rand(2, *shape) * 65535).astype(np.float32)
    frames[1] *= 0.37
    sensor = _sensor(shape, pattern)
    out = _run(frames, sensor, _params(algo), tensor_device)
    _check(_oracle(frames, sensor, _params(algo)), out.cpu().numpy(),
           (shape, pattern, algo))


@pytest.mark.parametrize("kind", ["zero", "mid", "ramp", "checker", "impulse"])
@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("algo", ALGOS)
def test_ahd_structured(tensor_device, kind, pattern, algo):
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
def test_ahd_signed_headroom(tensor_device, amp, pattern, algo):
    rng = np.random.RandomState(10901)
    frames = (rng.rand(2, 8, 10) * amp * 65535
              - amp * 32767).astype(np.float32)
    sensor = _sensor((8, 10), pattern)
    out = _run(frames, sensor, _params(algo), tensor_device)
    _check(_oracle(frames, sensor, _params(algo)), out.cpu().numpy(),
           (amp, pattern, algo))


@pytest.mark.parametrize("algo", ALGOS)
def test_ahd_f64_input(tensor_device, algo):
    rng = np.random.RandomState(10901)
    frames = (rng.rand(2, 8, 10) * 16777217).astype(np.float64)
    frames[0, 0, 0] = 16777217.0
    sensor = _sensor((8, 10), "rggb")
    out = _run(frames, sensor, _params(algo), tensor_device)
    _check(_oracle(frames, sensor, _params(algo)), out.cpu().numpy(),
           ("f64", algo))


def test_ahd_batch_independence(tensor_device):
    rng = np.random.RandomState(10901)
    frames = np.empty((2, 7, 9), np.float32)
    frames[0] = rng.rand(7, 9) * 65535
    frames[1] = 1234.0
    sensor = _sensor((7, 9), "rggb")
    for algo in ALGOS:
        out = _run(frames, sensor, _params(algo), tensor_device)
        _check(_oracle(frames, sensor, _params(algo)), out.cpu().numpy(),
               ("batch", algo))


@pytest.mark.parametrize("q", [98.0, 99.0])
@pytest.mark.parametrize("shape", [(1, 1), (1, 2), (1, 10), (2, 13), (3, 17),
                                  (7, 9)])
@pytest.mark.parametrize("scene", ["random", "constant", "impulse", "ties",
                                  "quantized"])
def test_ahd_percentile(tensor_device, q, shape, scene):
    """_percentile_threshold exact vs np.percentile; outlier mask exact."""
    from brilliantisp_gpu.blocks.demosaic_ahd import _percentile_threshold
    import torch
    rng = np.random.RandomState(10901)
    n = shape[0] * shape[1]
    if scene == "random":
        plane = rng.rand(2, *shape) * 65535
    elif scene == "constant":
        plane = np.full((2, *shape), 30000.0)
    elif scene == "impulse":
        plane = np.zeros((2, *shape))
        plane[:, shape[0] // 2, shape[1] // 2] = 65535.0
    elif scene == "ties":
        plane = np.full((2, *shape), 100.0)
        plane[:, 0, 0] = 40000.0
        plane[:, -1, -1] = 200.0
    else:
        plane = np.tile(np.arange(n, dtype=np.float32) + np.float32(1e7),
                        (2, 1)).reshape(2, *shape)
    plane = plane.astype(np.float32)
    thr = _percentile_threshold(
        torch.as_tensor(plane, device=tensor_device), q)
    assert thr.shape == (2, 1, 1) and thr.dtype == torch.float32
    for i in range(2):
        expected = np.float32(np.percentile(plane[i], q))
        np.testing.assert_array_equal(thr[i].cpu().numpy(), expected)
        mask = plane[i] > thr[i].cpu().numpy()
        np.testing.assert_array_equal(
            mask, plane[i] > np.percentile(plane[i], q))


@pytest.mark.parametrize("scene", ["random", "tiny", "constant", "impulse"])
def test_ahd_median3(tensor_device, scene):
    """_median3 exact vs scipy median_filter size 3."""
    from brilliantisp_gpu.blocks.demosaic_ahd import _median3
    from scipy.ndimage import median_filter
    import torch
    rng = np.random.RandomState(10901)
    if scene == "random":
        plane = (rng.rand(2, 8, 10) * 65535).astype(np.float32)
    elif scene == "tiny":
        plane = (rng.rand(2, 1, 7) * 65535).astype(np.float32)
    elif scene == "constant":
        plane = np.full((2, 8, 10), 12345.0, np.float32)
    else:
        plane = np.zeros((2, 8, 10), np.float32)
        plane[:, 0, 0] = 65535.0
        plane[:, -1, -1] = 32768.0
    out = _median3(torch.as_tensor(plane, device=tensor_device))
    for i in range(2):
        np.testing.assert_array_equal(
            out[i].cpu().numpy(), median_filter(plane[i], size=3))


def test_ahd_opt_alpha(tensor_device):
    """_opt_alpha vs CPU private _compute_enhanced_homogeneity oracle."""
    from brilliantisp_gpu.blocks.demosaic_ahd import _opt_alpha
    from modules.demosaic.ahd_demosaic import AHDDemosaicOptimized
    import torch
    rng = np.random.RandomState(10901)
    rgb_h = (rng.rand(2, 7, 9, 3) * 65535).astype(np.float32)
    rgb_v = (rng.rand(2, 7, 9, 3) * 65535).astype(np.float32)
    out = _opt_alpha(torch.as_tensor(rgb_h, device=tensor_device),
                     torch.as_tensor(rgb_v, device=tensor_device))
    assert out.shape == (2, 7, 9)
    assert out.dtype == torch.float32
    assert out.device == torch.as_tensor(rgb_h, device=tensor_device).device
    m3 = [np.ones((7, 9), np.float32) for _ in range(3)]
    dummy = np.zeros((7, 9), np.float32)
    expected = np.stack([
        AHDDemosaicOptimized(dummy.copy(), m3)._compute_enhanced_homogeneity(
            rgb_h[i].copy(), rgb_v[i].copy())
        for i in range(2)])
    res = compare_arrays(expected, out.cpu().numpy(),
                         FLOAT32_NORMALIZED, scale=1.0)
    assert res["passed"], res


@pytest.mark.parametrize("q,vals,expected,wrong", [
    (98.0, [41562748.0, 4713247.0], 40825760.0, 40825756.0),
    (99.0, [7050877.5, 26630378.0], 26434584.0, 26434582.0),
])
def test_ahd_percentile_dtype_regression(tensor_device, q, vals, expected,
                                         wrong):
    """Frozen lead cases: f32-source percentile, not f64-promoted."""
    from brilliantisp_gpu.blocks.demosaic_ahd import _percentile_threshold
    import torch
    plane = np.asarray(vals, dtype=np.float32).reshape(1, 1, 2)
    assert np.float32(np.percentile(plane[0], q)) == np.float32(expected)
    assert np.float32(np.percentile(plane[0].astype(np.float64), q))         == np.float32(wrong)
    thr = _percentile_threshold(
        torch.as_tensor(plane, device=tensor_device), q)
    assert thr.shape == (1, 1, 1) and thr.dtype == torch.float32
    assert thr.device == torch.as_tensor(plane, device=tensor_device).device
    np.testing.assert_array_equal(thr.cpu().numpy(), np.float32(expected))


def test_ahd_basic_decision(tensor_device):
    """_basic_decision matches CPU homogeneity formula exactly (bool)."""
    from brilliantisp_gpu.blocks.demosaic_ahd import _basic_decision
    from scipy.ndimage import convolve
    import torch
    rng = np.random.RandomState(10901)
    rgb_h = (rng.rand(2, 7, 9, 3) * 65535).astype(np.float32)
    rgb_v = (rng.rand(2, 7, 9, 3) * 65535).astype(np.float32)
    out = _basic_decision(torch.as_tensor(rgb_h, device=tensor_device),
                          torch.as_tensor(rgb_v, device=tensor_device))
    assert out.dtype == torch.bool and out.shape == (2, 7, 9)
    k = np.ones((5, 5), np.float32) / 25.0
    expected = []
    for i in range(2):
        dh = (np.abs(rgb_h[i, :, :, 0] - rgb_h[i, :, :, 1])
              + np.abs(rgb_h[i, :, :, 1] - rgb_h[i, :, :, 2])).astype(np.float32)
        dv = (np.abs(rgb_v[i, :, :, 0] - rgb_v[i, :, :, 1])
              + np.abs(rgb_v[i, :, :, 1] - rgb_v[i, :, :, 2])).astype(np.float32)
        vh = convolve((dh - convolve(dh, k, mode="reflect")) ** 2, k,
                      mode="reflect")
        vv = convolve((dv - convolve(dv, k, mode="reflect")) ** 2, k,
                      mode="reflect")
        expected.append(vh < vv)
    np.testing.assert_array_equal(out.cpu().numpy(), np.stack(expected))


@pytest.mark.parametrize("copies", (1, 2))
def test_ahd_opt_frozen_real_alpha(tensor_device, copies):
    import torch
    with np.load(Path(__file__).parent / "fixtures"
                 / "ahd_opt_real_alpha_patch.npz",
                 allow_pickle=False) as frozen:
        rgb_h_np = np.stack([frozen["rgb_h"]] * copies)
        rgb_v_np = np.stack([frozen["rgb_v"]] * copies)
        expected_inner = np.stack(
            [frozen["expected_alpha"][3:-3, 3:-3]] * copies)
    rgb_h = torch.as_tensor(rgb_h_np, device=tensor_device)
    rgb_v = torch.as_tensor(rgb_v_np, device=tensor_device)
    h_before = rgb_h.cpu().numpy().copy()
    v_before = rgb_v.cpu().numpy().copy()
    out = _opt_alpha(rgb_h, rgb_v)
    assert out.dtype == torch.float32
    assert out.shape == (copies, 32, 32)
    assert out.device == rgb_h.device
    np.testing.assert_array_equal(rgb_h.cpu().numpy(), h_before)
    np.testing.assert_array_equal(rgb_v.cpu().numpy(), v_before)
    np.testing.assert_array_equal(
        out[:, 3:-3, 3:-3].cpu().numpy(), expected_inner)
