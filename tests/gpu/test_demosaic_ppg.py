"""GPU PPG demosaic vs Demosaic.compute_linear (seed 10601)."""

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
ALGOS = ("ppg", "ppg_opt")


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
def test_ppg_random(tensor_device, shape, pattern, algo):
    rng = np.random.RandomState(10601)
    frames = (rng.rand(2, *shape) * 65535).astype(np.float32)
    frames[1] *= 0.37
    sensor = _sensor(shape, pattern)
    out = _run(frames, sensor, _params(algo), tensor_device)
    _check(_oracle(frames, sensor, _params(algo)), out.cpu().numpy(),
           (shape, pattern, algo))


@pytest.mark.parametrize("kind", ["zero", "mid", "ramp", "checker", "impulse"])
@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("algo", ALGOS)
def test_ppg_structured(tensor_device, kind, pattern, algo):
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
def test_ppg_signed_headroom(tensor_device, amp, pattern, algo):
    rng = np.random.RandomState(10601)
    frames = (rng.rand(2, 8, 10) * amp * 65535
              - amp * 32767).astype(np.float32)
    sensor = _sensor((8, 10), pattern)
    out = _run(frames, sensor, _params(algo), tensor_device)
    _check(_oracle(frames, sensor, _params(algo)), out.cpu().numpy(),
           (amp, pattern, algo))


@pytest.mark.parametrize("algo", ALGOS)
def test_ppg_f64_input(tensor_device, algo):
    rng = np.random.RandomState(10601)
    frames = (rng.rand(2, 8, 10) * 16777217).astype(np.float64)
    frames[0, 0, 0] = 16777217.0
    sensor = _sensor((8, 10), "rggb")
    out = _run(frames, sensor, _params(algo), tensor_device)
    _check(_oracle(frames, sensor, _params(algo)), out.cpu().numpy(),
           ("f64", algo))


def test_ppg_batch_independence(tensor_device):
    rng = np.random.RandomState(10601)
    frames = np.empty((2, 7, 9), np.float32)
    frames[0] = rng.rand(7, 9) * 65535
    frames[1] = 1234.0
    sensor = _sensor((7, 9), "rggb")
    for algo in ALGOS:
        out = _run(frames, sensor, _params(algo), tensor_device)
        _check(_oracle(frames, sensor, _params(algo)), out.cpu().numpy(),
               ("batch", algo))


def _ppg_threshold_frame(algo, center, hn, vn):
    """rggb 5x5, center R at (2,2); all other pixels 0."""
    a = np.zeros((2, 5, 5), np.float32)
    a[:, 2, 2] = center
    if algo == "ppg":
        a[:, 2, 1] = a[:, 2, 3] = hn      # green horizontal
        a[:, 1, 2] = a[:, 3, 2] = vn      # green vertical
    else:
        a[:, 2, 1] = a[:, 2, 3] = 200.0   # green horizontal fixed
        a[:, 1, 2] = a[:, 3, 2] = 400.0   # green vertical fixed
        a[:, 2, 0] = a[:, 2, 4] = hn      # same-color red +-2 horizontal
        a[:, 0, 2] = a[:, 4, 2] = vn      # same-color red +-2 vertical
    return a


@pytest.mark.parametrize("algo,center,hn,vn,bound", [
    ("ppg", 100.0, 75.0, 80.0, 1.25),      # upper 50/40 -> 1.25
    ("ppg", 100.0, 80.0, 75.0, 0.8),       # lower 40/50 -> 0.8
    ("ppg_opt", 100.0, 66.5, 650.0, 1.33), # upper 133/100 -> 1.33
    ("ppg_opt", 100.0, 37.5, 650.0, 0.75), # lower 75/100 -> 0.75
])
def test_ppg_threshold_straddle(tensor_device, algo, center, hn, vn, bound):
    """Prev/nominal/next float32 V-probe values around the ratio boundary."""
    vn32 = np.float32(vn)
    probes = {"prev": float(np.nextafter(vn32, np.float32(-np.inf))),
              "nominal": float(vn32),
              "next": float(np.nextafter(vn32, np.float32(np.inf)))}
    sensor = _sensor((5, 5), "rggb")
    f32 = np.float32
    eps = f32(1e-6)
    h_grad = (f32(abs(2 * hn - 200.0)) if algo == "ppg"
              else f32(abs(2 * hn)))
    bound32 = f32(bound)
    ratios = {}
    results = {}
    for name, v in probes.items():
        v_grad = (f32(abs(2 * float(v) - 200.0)) if algo == "ppg"
                  else f32(abs(2 * float(v) - 1200.0)))
        ratios[name] = np.float32(np.float32(h_grad)
                                  / np.float32(v_grad + eps))
        frames = _ppg_threshold_frame(algo, center, hn, v)
        out = _run(frames, sensor, _params(algo), tensor_device)
        _check(_oracle(frames, sensor, _params(algo)), out.cpu().numpy(),
               (algo, name))
        results[name] = out.cpu().numpy()[0, 2, 2]
    assert min(ratios.values()) < bound32 < max(ratios.values())
    assert ratios["nominal"] == bound32
    assert not np.array_equal(results["prev"], results["next"])
