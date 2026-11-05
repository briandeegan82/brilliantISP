"""AE meter + direct gain suggestion vs CPU AutoExposure (seed 9501)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from brilliantisp_gpu.blocks.auto_exposure import (
    AeMeasurement, auto_exposure, suggest_direct_gain_index)
from brilliantisp_gpu.types import GpuError
from modules.auto_exposure.auto_exposure import AutoExposure
from tools.gpu_validation.metrics import compare_arrays
from tools.gpu_validation.contracts import STATISTICS

SHAPES = [(1, 1), (1, 2), (2, 2), (7, 9), (8, 10)]


def _ae_params(**over):
    p = dict(is_enable=True, is_debug=False, center_illuminance=0.5,
             histogram_skewness=0.9, target_luminance=None)
    p.update(over)
    return p


def _oracle(rgb, params):
    ae = AutoExposure(np.asarray(rgb), dict(bit_depth=12, hdr_bit_depth=12),
                      copy.deepcopy(params), meter_domain="scene_linear_rgb")
    fb = ae.get_exposure_feedback()
    return (fb, ae.last_meter_average, ae.sample_count)


def _run(frames, params, device):
    import torch
    image = torch.as_tensor(frames, device=device)
    before = image.clone()
    p_before = copy.deepcopy(params)
    actual = auto_exposure(image, params)
    np.testing.assert_array_equal(image.cpu().numpy(), before.cpu().numpy())
    assert params == p_before
    assert isinstance(actual, AeMeasurement)
    return actual


def _check(actual, expected, tag):
    exp_fb = [e[0] for e in expected]
    exp_avg = [e[1] for e in expected]
    exp_cnt = [e[2] for e in expected]
    assert list(actual.feedbacks) == exp_fb, tag
    assert list(actual.sample_counts) == exp_cnt, tag
    for i, (a, e) in enumerate(zip(actual.meter_averages, exp_avg)):
        if e is None:
            assert a is None, (tag, i)
        else:
            res = compare_arrays(np.float64(e), np.float64(a), STATISTICS, scale=1.0)
            assert res["passed"], (tag, i, e, a, res)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ae_ordinary(tensor_device, shape, dtype):
    rng = np.random.RandomState(9501)
    frames = rng.rand(2, *shape, 3).astype(dtype)
    for intensity in (0.18, 0.5, 2.0):
        data = frames * intensity
        params = _ae_params()
        expected = [_oracle(f, params) for f in data]
        actual = _run(data, params, tensor_device)
        _check(actual, expected, (shape, dtype, intensity))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ae_edge_cases(tensor_device, dtype):
    cases = [
        ("black", np.zeros((2, 8, 10, 3), dtype)),
        ("empty", np.zeros((2, 0, 10, 3), dtype)),
        ("dark", np.full((2, 7, 9, 3), 0.001, dtype)),
        ("saturated", np.full((2, 8, 10, 3), 5.0, dtype)),
        ("headroom", np.full((2, 8, 10, 3), 1.5, dtype)),
        ("negative", np.full((2, 8, 10, 3), -0.5, dtype)),
    ]
    params = _ae_params()
    for tag, frames in cases:
        expected = [_oracle(f, params) for f in frames]
        actual = _run(frames, params, tensor_device)
        _check(actual, expected, (dtype, tag))


def test_ae_center_constant_scale0(tensor_device):
    """All pixels at configured center -> delta 0 -> skew 0 -> fb 0."""
    params = _ae_params(center_illuminance=0.5)
    frames = np.full((2, 8, 10, 3), 0.5, np.float32)
    expected = [_oracle(f, params) for f in frames]
    actual = _run(frames, params, tensor_device)
    _check(actual, expected, "center_const")
    assert actual.feedbacks == (0, 0)
    assert actual.skewness == (0.0, 0.0)


@pytest.mark.parametrize("n", [1, 2])
def test_ae_small_n_fisher(tensor_device, n):
    """B2 x 1xN frames -> sample_counts (N,N), g1=1 for N<=2."""
    params = _ae_params(center_illuminance=0.3)
    frames = np.concatenate(
        [np.full((1, 1, n, 3), 0.4, np.float64),
         np.full((1, 1, n, 3), 0.2, np.float64)])
    expected = [_oracle(f, params) for f in frames]
    actual = _run(frames, params, tensor_device)
    _check(actual, expected, ("n", n))
    assert actual.sample_counts == (n, n)


@pytest.mark.parametrize("dtype,scale", [
    (np.uint16, 65535.0), (np.uint32, 4294967295.0), (np.int32, 4095.0)])
def test_ae_integer_inputs(tensor_device, dtype, scale):
    """Real integer inputs accepted by the 'fiu' domain."""
    rng = np.random.RandomState(9501)
    frames = (rng.rand(2, 8, 10, 3) * scale).astype(dtype)
    params = _ae_params(target_luminance=0.5)
    expected = [_oracle(f, params) for f in frames]
    actual = _run(frames, params, tensor_device)
    _check(actual, expected, ("int", dtype))


def test_ae_bad_params_before_empty(tensor_device):
    """Invalid params reject even when frames are empty."""
    import torch
    frames = np.zeros((2, 0, 10, 3), np.float32)
    for bad in (
        _ae_params(center_illuminance=-1),
        _ae_params(histogram_skewness=float("inf")),
        _ae_params(target_luminance=float("nan")),
    ):
        with pytest.raises(ValueError):
            auto_exposure(torch.as_tensor(frames, device=tensor_device), bad)


def test_ae_disabled_still_meters(tensor_device):
    """is_enable=False still produces meter feedback (engine gates it)."""
    params = _ae_params(is_enable=False)
    frames = np.full((2, 8, 10, 3), 0.4, np.float32)
    expected = [_oracle(f, params) for f in frames]
    actual = _run(frames, params, tensor_device)
    _check(actual, expected, "disabled_meters")


@pytest.mark.parametrize("shape", SHAPES)
def test_ae_skew_sign(tensor_device, shape):
    """Constant frames below/at/above center -> strict skew signs."""
    rng = np.random.RandomState(9501)
    params = _ae_params(center_illuminance=0.5, histogram_skewness=0.2)
    for level in (0.2, 0.5, 0.8):
        frames = np.full((2, *shape, 3), level, np.float32)
        expected = [_oracle(f, params) for f in frames]
        actual = _run(frames, params, tensor_device)
        _check(actual, expected, (shape, level))


def test_ae_rejects(tensor_device):
    import torch
    params = _ae_params()
    frames = np.full((2, 8, 10, 3), 0.4, np.float32)
    for bad_params in (
        _ae_params(center_illuminance=-1),
        _ae_params(center_illuminance=float("nan")),
        _ae_params(histogram_skewness=-0.1),
        _ae_params(histogram_skewness=float("inf")),
        _ae_params(target_luminance=-0.5),
        _ae_params(target_luminance=float("nan")),
    ):
        with pytest.raises(ValueError):
            auto_exposure(torch.as_tensor(frames, device=tensor_device), bad_params)
    bad = torch.as_tensor(frames, device=tensor_device).clone()
    bad[0, 0, 0, 0] = float("nan")
    with pytest.raises(GpuError):
        auto_exposure(bad, params)
    with pytest.raises(GpuError):
        auto_exposure(torch.ones(2, 8, 10, 3, dtype=torch.bool, device=tensor_device), params)
    with pytest.raises(GpuError):
        auto_exposure(torch.ones(2, 8, 10, 3, dtype=torch.complex64, device=tensor_device), params)
    with pytest.raises(GpuError):
        auto_exposure(torch.ones(8, 10, 3, device=tensor_device), params)
    with pytest.raises(GpuError):
        auto_exposure(torch.ones(0, 8, 10, 3, device=tensor_device), params)


def test_ae_frozen_rounding_case(cuda_device):
    """Frozen fixture: SUM/deviceCount mean gives .7/index1 (torch.mean
    gave .7000000000000001/index0)."""
    torch = pytest.importorskip("torch")
    fixture = json.loads((Path(__file__).resolve().parent / "fixtures"
                          / "ae_rounding_cases.json").read_text())
    inp = fixture["input"]
    n_w, n_b = inp["white_pixels"], inp["black_pixels"]
    frame = np.zeros((1, 10, 3), np.float32)
    frame[0, :n_w] = 1.0
    frames = np.stack([frame, frame])
    params = _ae_params(center_illuminance=inp["center_illuminance"],
                        histogram_skewness=inp["histogram_skewness"],
                        target_luminance=inp["target_luminance"])
    actual = auto_exposure(torch.as_tensor(frames, device=cuda_device), params)
    ref = fixture["cpu_reference"]
    assert actual.feedbacks == (ref["feedback"],) * 2
    assert actual.meter_averages[0] == ref["meter_average"]  # exact .7, not .7000000000000001
    idx = suggest_direct_gain_index(inp["current_gain_index"], inp["gain_array"],
                                    actual.meter_averages[0], params)
    assert idx == ref["suggested_gain_index"]
    expected = [_oracle(f, params) for f in frames]
    _check(actual, expected, "frozen")


def test_suggest_direct():
    gains = [1.0, 2.0, 4.0]
    params = _ae_params(target_luminance=0.5)
    assert suggest_direct_gain_index(0, gains, 0.5, params) == 0
    assert suggest_direct_gain_index(0, gains, 0.25, params) == 1   # ratio .5 -> gain2
    assert suggest_direct_gain_index(0, gains, 2.0, params) == 0    # clip to arr.min
    assert suggest_direct_gain_index(0, gains, 0.05, params) == 2   # clip to arr.max
    assert suggest_direct_gain_index(0, gains, None, params) == 0
    assert suggest_direct_gain_index(1, gains, 0.0, params) == 1
    assert suggest_direct_gain_index(1, gains, -1.0, params) == 1
    assert suggest_direct_gain_index(1, [], 0.5, params) == 1
    assert suggest_direct_gain_index(5, gains, 0.25, params) == 2   # gi clipped 5->2, desired clip->4
    # tie: desired exactly midpoint picks FIRST index
    assert suggest_direct_gain_index(0, [1.0, 3.0], 0.5,
                                     _ae_params(target_luminance=1.0)) == 0  # desired2.0 tie->1.0
