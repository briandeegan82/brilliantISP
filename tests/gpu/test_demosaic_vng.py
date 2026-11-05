"""GPU VNG demosaic vs Demosaic.compute_linear (seed 10701)."""

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
SHAPES = [(1, 1), (1, 2), (2, 1), (1, 7), (7, 1), (2, 2), (7, 9), (8, 10)]
LAYOUTS = {"rggb": (0, 1, 2, 3), "bggr": (3, 2, 1, 0),
           "grbg": (1, 0, 3, 2), "gbrg": (2, 3, 0, 1)}
ALGOS = ("vng", "vng_opt")


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
def test_vng_random(tensor_device, shape, pattern, algo):
    rng = np.random.RandomState(10701)
    frames = (rng.rand(2, *shape) * 65535).astype(np.float32)
    frames[1] *= 0.37
    sensor = _sensor(shape, pattern)
    out = _run(frames, sensor, _params(algo), tensor_device)
    _check(_oracle(frames, sensor, _params(algo)), out.cpu().numpy(),
           (shape, pattern, algo))


@pytest.mark.parametrize("kind", ["zero", "mid", "ramp", "checker", "impulse"])
@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("algo", ALGOS)
def test_vng_structured(tensor_device, kind, pattern, algo):
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
def test_vng_signed_headroom(tensor_device, amp, pattern, algo):
    rng = np.random.RandomState(10701)
    frames = (rng.rand(2, 8, 10) * amp * 65535
              - amp * 32767).astype(np.float32)
    sensor = _sensor((8, 10), pattern)
    out = _run(frames, sensor, _params(algo), tensor_device)
    _check(_oracle(frames, sensor, _params(algo)), out.cpu().numpy(),
           (amp, pattern, algo))


@pytest.mark.parametrize("algo", ALGOS)
def test_vng_f64_input(tensor_device, algo):
    rng = np.random.RandomState(10701)
    frames = (rng.rand(2, 8, 10) * 16777217).astype(np.float64)
    frames[0, 0, 0] = 16777217.0
    sensor = _sensor((8, 10), "rggb")
    out = _run(frames, sensor, _params(algo), tensor_device)
    _check(_oracle(frames, sensor, _params(algo)), out.cpu().numpy(),
           ("f64", algo))


def test_vng_batch_independence(tensor_device):
    rng = np.random.RandomState(10701)
    frames = np.empty((2, 7, 9), np.float32)
    frames[0] = rng.rand(7, 9) * 65535
    frames[1] = 1234.0
    sensor = _sensor((7, 9), "rggb")
    for algo in ALGOS:
        out = _run(frames, sensor, _params(algo), tensor_device)
        _check(_oracle(frames, sensor, _params(algo)), out.cpu().numpy(),
               ("batch", algo))


def _pack_inputs(a_vals, b_vals, count, device):
    import torch
    a = [torch.as_tensor([[v]], dtype=torch.float32, device=device)
         for v in a_vals]
    b = [torch.as_tensor([[v]], dtype=torch.float32, device=device)
         for v in b_vals]
    while len(a) < 3:
        a.append(torch.zeros_like(a[0]))
    while len(b) < 3:
        b.append(torch.zeros_like(b[0]))
    cnt = torch.full_like(a[0], count, dtype=torch.int64)
    return a, b, cnt


def _expected_mean(a_vals, b_vals, count):
    vals = [np.float32(v) for pair in zip(a_vals, b_vals) for v in pair]
    lst = vals * 2  # ordered selected pairs repeated twice
    return np.mean(np.array(lst, np.float32), dtype=np.float32)


@pytest.mark.parametrize("amp", [1.0, 65535.0, 128.0 * 65535.0])
@pytest.mark.parametrize("count", [1, 2, 3])
def test_vng_selected_mean(tensor_device, count, amp):
    from brilliantisp_gpu.blocks.demosaic_vng import _selected_mean
    rng = np.random.RandomState(10701)
    a_v = (rng.rand(count) * amp * 2 - amp).tolist()
    b_v = (rng.rand(count) * amp * 2 - amp).tolist()
    a_s, b_s, cnt = _pack_inputs(a_v, b_v, count, tensor_device)
    out = _selected_mean(a_s, b_s, cnt)
    assert out.dtype == a_s[0].dtype and out.device == a_s[0].device
    expected = _expected_mean(a_v, b_v, count)
    np.testing.assert_array_equal(out.cpu().numpy(),
                                  np.float32(expected))


def test_vng_selected_mean_adversarial(tensor_device):
    from brilliantisp_gpu.blocks.demosaic_vng import _selected_mean
    a_v = [16777216.0, -16777216.0, 3.0]
    b_v = [1.0, 2.0, -2.0]
    for count in (1, 2, 3):
        a_s, b_s, cnt = _pack_inputs(a_v[:count], b_v[:count], count,
                                     tensor_device)
        out = _selected_mean(a_s, b_s, cnt)
        expected = _expected_mean(a_v[:count], b_v[:count], count)
        np.testing.assert_array_equal(out.cpu().numpy(),
                                      np.float32(expected))


@pytest.mark.parametrize("kind", ["constant", "equal_gradient"])
@pytest.mark.parametrize("algo", ALGOS)
def test_vng_edges(tensor_device, kind, algo):
    """Equal-gradient/constant scenes exercise the <= 1.5*min threshold."""
    rng = np.random.RandomState(10701)
    if kind == "constant":
        frames = np.full((2, 8, 10), 40000.0, np.float32)
    else:
        frames = np.tile(
            np.linspace(0, 65535, 10, dtype=np.float32), (2, 8, 1))
    sensor = _sensor((8, 10), "rggb")
    out = _run(frames, sensor, _params(algo), tensor_device)
    _check(_oracle(frames, sensor, _params(algo)), out.cpu().numpy(),
           (kind, algo))
