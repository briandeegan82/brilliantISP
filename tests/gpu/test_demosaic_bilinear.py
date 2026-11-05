"""Bilinear demosaic vs CPU Demosaic.compute_linear (seed 9601)."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from brilliantisp_gpu.blocks.demosaic import demosaic
from brilliantisp_gpu.types import GpuError
from modules.demosaic.demosaic import Demosaic
from tools.gpu_validation.metrics import compare_arrays
from tools.gpu_validation.contracts import FLOAT32_NORMALIZED

PATTERNS = ("rggb", "bggr", "grbg", "gbrg")
PLATFORM = {
    "in_file": "fixture_9x7", "disable_progress_bar": True,
    "leave_pbar_string": False, "debug_enabled": False,
}
SHAPES = [(1, 1), (1, 7), (7, 1), (2, 2), (7, 9), (8, 10)]
LAYOUTS = {"rggb": (0, 1, 2, 3), "bggr": (3, 2, 1, 0),
           "grbg": (1, 0, 3, 2), "gbrg": (2, 3, 0, 1)}


def _sensor(shape, pattern):
    return dict(height=shape[-2], width=shape[-1], bayer_pattern=pattern,
                bit_depth=16, hdr_bit_depth=16)


def _params(algo="bilinear"):
    return dict(is_save=False, algorithm=algo)


def _oracle(frame, sensor, params):
    return Demosaic(frame.copy(), dict(PLATFORM), sensor,
                    copy.deepcopy(params)).compute_linear()


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
    out = demosaic(image, sensor, params)
    np.testing.assert_array_equal(image.cpu().numpy(), before.cpu().numpy())
    assert params == p_before
    assert out.dtype == torch.float32
    assert out.shape == frames.shape + (3,)
    assert out.device == image.device
    return out


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("pattern", PATTERNS)
def test_bilinear_random(tensor_device, shape, pattern):
    rng = np.random.RandomState(9601)
    sensor = _sensor(shape, pattern)
    frames = (rng.rand(2, *shape) * 65535).astype(np.float32)
    frames[1] = frames[1] * 0.3 + 100
    params = _params()
    expected = [_oracle(f, sensor, params) for f in frames]
    out = _run(frames, sensor, params, tensor_device)
    res = compare_arrays(np.stack(expected), out.cpu().numpy(),
                         FLOAT32_NORMALIZED, scale=65535.0)
    assert res["passed"], (shape, pattern, res)
    for c in range(3):
        m = _site_mask(shape, pattern, c)
        np.testing.assert_array_equal(
            out.cpu().numpy()[..., c][:, m], frames[:, m],
            err_msg=f"known site {c} not restored")


@pytest.mark.parametrize("scale", [8.0, 128.0])
@pytest.mark.parametrize("pattern", PATTERNS)
def test_bilinear_signed_headroom(tensor_device, scale, pattern):
    rng = np.random.RandomState(9601)
    sensor = _sensor((8, 10), pattern)
    frames = (rng.rand(2, 8, 10) * scale * 65535 - scale * 32767).astype(np.float32)
    params = _params()
    expected = [_oracle(f, sensor, params) for f in frames]
    out = _run(frames, sensor, params, tensor_device)
    res = compare_arrays(np.stack(expected), out.cpu().numpy(),
                         FLOAT32_NORMALIZED, scale=65535.0)
    assert res["passed"], (scale, pattern, res)


def _structured(shape, kind):
    rng = np.random.RandomState(9601)
    h, w = shape
    if kind == "impulse":
        a = np.zeros((2, h, w), np.float32); a[:, h // 2, w // 2] = 65535.0
    elif kind == "checker":
        a = ((np.indices((h, w)).sum(0) % 2) * 65535.0).astype(np.float32)[None].repeat(2, 0)
    elif kind == "zero":
        a = np.zeros((2, h, w), np.float32)
    elif kind == "mid":
        a = np.full((2, h, w), 32768.0, np.float32)
    else:
        a = (np.arange(2 * h * w).reshape(2, h, w) % 65535).astype(np.float32)
    return a


@pytest.mark.parametrize("kind", ["impulse", "checker", "zero", "mid", "ramp"])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("pattern", PATTERNS[:2])
def test_bilinear_structured(tensor_device, kind, shape, pattern):
    sensor = _sensor(shape, pattern)
    frames = _structured(shape, kind)
    params = _params()
    expected = [_oracle(f, sensor, params) for f in frames]
    out = _run(frames, sensor, params, tensor_device)
    res = compare_arrays(np.stack(expected), out.cpu().numpy(),
                         FLOAT32_NORMALIZED, scale=65535.0)
    assert res["passed"], (kind, shape, pattern, res)


def test_bilinear_f64_input(tensor_device):
    """f64 16777217 is not f32-representable; CPU converts to f32 first."""
    import torch
    sensor = _sensor((8, 10), "rggb")
    frames = np.full((2, 8, 10), 16777217.0, np.float64)
    frames[1, ::2, ::2] = 100.0
    params = _params()
    expected = [_oracle(f, sensor, params) for f in frames]
    out = _run(frames, sensor, params, tensor_device)
    np.testing.assert_array_equal(out.cpu().numpy(), np.stack(expected))


def test_bilinear_rejects(tensor_device):
    import torch
    sensor = _sensor((8, 10), "rggb")
    good = torch.ones(2, 8, 10, device=tensor_device)
    with pytest.raises(ValueError):
        demosaic(good, sensor, _params("fourier"))
    bad = good.clone(); bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        demosaic(bad, sensor, _params())
    with pytest.raises(ValueError):
        demosaic(torch.ones(2, 8, 10, dtype=torch.bool, device=tensor_device), sensor, _params())
    with pytest.raises(ValueError):
        demosaic(torch.ones(2, 8, 10, dtype=torch.complex64, device=tensor_device), sensor, _params())
    with pytest.raises(GpuError):
        demosaic(torch.ones(0, 8, 10, device=tensor_device), sensor, _params())
    with pytest.raises(GpuError):
        demosaic(torch.ones(8, 10, device=tensor_device), sensor, _params())
    bad_sensor = _sensor((8, 10), "rrgg")
    with pytest.raises(ValueError):
        demosaic(good, bad_sensor, _params())


@pytest.mark.parametrize("pattern", PATTERNS)
def test_bilinear_tiny_absent_colours(tensor_device, pattern):
    """1-wide/1-high: some channels have no same-colour neighbours —
    CPU quirks (empty interpolants) must match, not idealized RGB."""
    for shape in ((1, 1), (1, 7), (7, 1), (1, 2), (2, 1)):
        sensor = _sensor(shape, pattern)
        rng = np.random.RandomState(9601)
        frames = (rng.rand(2, *shape) * 65535).astype(np.float32)
        params = _params()
        expected = [_oracle(f, sensor, params) for f in frames]
        out = _run(frames, sensor, params, tensor_device)
        res = compare_arrays(np.stack(expected), out.cpu().numpy(),
                             FLOAT32_NORMALIZED, scale=65535.0)
        assert res["passed"], (shape, pattern, res)
