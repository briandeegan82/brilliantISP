"""Auto white balance measurement vs CPU compute_gains (seed 9401)."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from brilliantisp_gpu.blocks.auto_white_balance import (
    AwbMeasurement, auto_white_balance)
from brilliantisp_gpu.types import GpuError
from modules.auto_white_balance.auto_white_balance import AutoWhiteBalance
from tools.gpu_validation.metrics import compare_arrays
from tools.gpu_validation.contracts import STATISTICS

PATTERNS = ("rggb", "bggr", "grbg", "gbrg")
SHAPES = [(1, 1), (1, 7), (7, 1), (2, 2), (7, 9), (8, 10)]


def _sensor(shape, pattern, bits):
    return dict(height=shape[-2], width=shape[-1], bayer_pattern=pattern,
                bit_depth=bits, hdr_bit_depth=bits)


def _awb(algo="gray_world", **over):
    p = dict(is_enable=True, is_debug=False, algorithm=algo,
             underexposed_percentage=10, overexposed_percentage=10,
             percentage=25)
    p.update(over)
    return p


_WBC = dict(r_gain=1.5, b_gain=0.75)


def _oracle(frame, sensor, awb, wbc, fb=None, ig=1.0):
    m = AutoWhiteBalance(frame.copy(), sensor, copy.deepcopy(awb), copy.deepcopy(wbc))
    gains = m.compute_gains(fallback_gains=fb, input_gain=ig)
    return gains, m.last_status, m.sample_count


def _check(actual, expected, tag, image=None, params=None, wbc=_WBC):
    assert isinstance(actual, AwbMeasurement)
    import torch
    assert actual.gains.dtype == torch.float64
    assert actual.gains.shape == (len(expected), 2)
    if image is not None:
        assert actual.gains.device == image.device
    exp_g = np.array([e[0] for e in expected])
    exp_s = [e[1] for e in expected]
    exp_c = [e[2] for e in expected]
    assert list(actual.statuses) == exp_s, tag
    assert list(actual.sample_counts) == exp_c, tag
    res = compare_arrays(exp_g, actual.gains.cpu().numpy(), STATISTICS, scale=1.0)
    assert res["passed"], (tag, exp_g, actual.gains.cpu().numpy(), res)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("algo", ["gray_world", "grey_world", "norm_2"])
@pytest.mark.parametrize("bits", [12, 24, 32])
def test_awb_ordinary(tensor_device, shape, pattern, algo, bits):
    rng = np.random.RandomState(9401)
    code_max = 2**bits - 1
    sensor = _sensor(shape, pattern, bits)
    frames = (rng.rand(2, *shape) * code_max).astype(np.float32)
    awb = _awb(algo)
    expected = [_oracle(f, sensor, awb, _WBC) for f in frames]
    import torch
    image = torch.as_tensor(frames, device=tensor_device)
    before = image.clone()
    p_before, w_before = copy.deepcopy(awb), copy.deepcopy(_WBC)
    actual = auto_white_balance(image, sensor, awb, _WBC)
    _check(actual, expected, (shape, pattern, algo, bits), image=image)
    np.testing.assert_array_equal(image.cpu().numpy(), before.cpu().numpy())
    assert awb == p_before and _WBC == w_before
    if shape[0] * shape[1] >= 60:
        assert all(s == "estimated" for s in actual.statuses), (shape, pattern, algo, bits)
        assert all(c > 0 for c in actual.sample_counts)


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("algo", ["gray_world", "norm_2", "pca"])
def test_awb_known_gains_2_1(cuda_device, pattern, algo):
    """uint32 cells G/B=2**31, R=2**30, pct 0 -> estimated (2,1) exactly."""
    torch = pytest.importorskip("torch")
    h, w = 8, 10
    sensor = _sensor((h, w), pattern, 32)
    frame = np.zeros((h, w), np.uint32)
    ch = {"rggb": (0, 1, 2, 3), "bggr": (3, 2, 1, 0),
          "grbg": (1, 0, 3, 2), "gbrg": (2, 3, 0, 1)}[pattern]
    for site in range(4):
        row, col = divmod(site, 2)
        code = {0: 2**30, 1: 2**31, 2: 2**31, 3: 2**31}[ch[site]]
        frame[row::2, col::2] = code
    frames = np.stack([frame, frame])
    awb = _awb(algo, underexposed_percentage=0, overexposed_percentage=0)
    expected = [_oracle(f, sensor, awb, _WBC) for f in frames]
    actual = auto_white_balance(
        torch.as_tensor(frames, device=cuda_device), sensor, awb, _WBC)
    _check(actual, expected, ("known", pattern, algo))
    assert actual.statuses == ("estimated", "estimated")
    np.testing.assert_allclose(actual.gains.cpu().numpy(), [[2.0, 1.0]] * 2,
                               rtol=STATISTICS.rtol, atol=STATISTICS.atol)
    assert actual.sample_counts == ((h // 2) * (w // 2),) * 2


def test_awb_gain_clipping(cuda_device):
    """pct 0 keeps extremes: estimated (4,1) and (.25,.25) literal."""
    torch = pytest.importorskip("torch")
    sensor = _sensor((8, 10), "rggb", 12)
    f1 = np.full((8, 10), 1000.0, np.float32); f1[::2, ::2] = 1.0
    f2 = np.full((8, 10), 100.0, np.float32)
    f2[::2, ::2] = 4000.0; f2[1::2, 1::2] = 4000.0
    frames = np.stack([f1, f2])
    awb = _awb("gray_world", underexposed_percentage=0, overexposed_percentage=0)
    expected = [_oracle(f, sensor, awb, _WBC) for f in frames]
    actual = auto_white_balance(
        torch.as_tensor(frames, device=cuda_device), sensor, awb, _WBC)
    _check(actual, expected, "clip")
    assert actual.statuses == ("estimated", "estimated")
    np.testing.assert_allclose(actual.gains.cpu().numpy(), [[4.0, 1.0], [0.25, 0.25]])


def test_awb_threshold_boundary_exact(tensor_device):
    """pct25 on bits12 -> limits 1024/3072 exact; equality kept."""
    sensor = _sensor((8, 10), "rggb", 12)
    awb = _awb("gray_world", underexposed_percentage=25, overexposed_percentage=25)
    base = np.full((2, 8, 10), 1024.0, np.float64)
    base[0, ::2, ::2] = 3072.0
    expected = [_oracle(f, sensor, awb, _WBC) for f in base]
    import torch
    image = torch.as_tensor(base, device=tensor_device)
    actual = auto_white_balance(image, sensor, awb, _WBC)
    _check(actual, expected, "boundary_equal", image=image)
    just_below = base.copy(); just_below[0, 0, 0] = np.nextafter(1024.0, -np.inf)
    just_above = base.copy(); just_above[1, ::2, ::2] = np.nextafter(3072.0, np.inf)
    for tag, frames in (("below", just_below), ("above", just_above)):
        expected = [_oracle(f, sensor, awb, _WBC) for f in frames]
        actual = auto_white_balance(
            torch.as_tensor(frames, device=tensor_device), sensor, awb, _WBC)
        _check(actual, expected, tag)


def test_awb_nonfinite_rows_filtered(tensor_device):
    """Valid mid-range rows with one NaN and one Inf cell each filtered."""
    sensor = _sensor((8, 10), "rggb", 12)
    frames = np.full((2, 8, 10), 2000.0, np.float32)
    frames[0, 0, 0] = float("nan")
    frames[1, 3, 5] = float("inf")
    awb = _awb("gray_world")
    expected = [_oracle(f, sensor, awb, _WBC) for f in frames]
    import torch
    actual = auto_white_balance(
        torch.as_tensor(frames, device=tensor_device), sensor, awb, _WBC)
    _check(actual, expected, "nonfinite")
    assert actual.sample_counts == tuple(e[2] for e in expected)


def test_awb_empty_tiny_and_zero_means(tensor_device):
    sensor = _sensor((8, 10), "rggb", 12)
    awb = _awb("gray_world")
    import torch
    for shape in ((0, 10), (8, 0)):
        sensor0 = _sensor(shape, "rggb", 12)
        frames = np.zeros((2,) + shape, np.float32)
        expected = [_oracle(f, sensor0, awb, _WBC) for f in frames]
        actual = auto_white_balance(
            torch.as_tensor(frames, device=tensor_device), sensor0, awb, _WBC)
        _check(actual, expected, ("empty", shape))
    awb0 = _awb("gray_world", underexposed_percentage=0, overexposed_percentage=0)
    frames = np.zeros((2, 8, 10), np.float32)
    expected = [_oracle(f, sensor, awb0, _WBC) for f in frames]
    actual = auto_white_balance(
        torch.as_tensor(frames, device=tensor_device), sensor, awb0, _WBC)
    _check(actual, expected, "zero_means")
    assert actual.statuses == ("fallback_ill_conditioned",) * 2


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("shape", [(8, 10), (7, 9), (2, 2), (1, 1)])
def test_awb_pca_cuda(cuda_device, pattern, shape):
    torch = pytest.importorskip("torch")
    rng = np.random.RandomState(9401)
    sensor = _sensor(shape, pattern, 12)
    frames = rng.randint(200, 3500, size=(2,) + shape).astype(np.float32)
    awb = _awb("pca")
    expected = [_oracle(f, sensor, awb, _WBC) for f in frames]
    actual = auto_white_balance(
        torch.as_tensor(frames, device=cuda_device), sensor, awb, _WBC)
    _check(actual, expected, ("pca", pattern, shape))


def test_awb_pca_error_propagates(cuda_device, monkeypatch):
    """A GpuError from pca_means must surface, never a hidden fallback."""
    torch = pytest.importorskip("torch")
    sensor = _sensor((8, 10), "rggb", 12)
    frames = np.full((2, 8, 10), 2000, np.float32)
    from brilliantisp_gpu.blocks import statistics as st

    def boom(rows, pct):
        raise GpuError("kernel unavailable")

    monkeypatch.setattr(st, "pca_means", boom)
    with pytest.raises(GpuError, match="kernel unavailable"):
        auto_white_balance(torch.as_tensor(frames, device=cuda_device),
                           sensor, _awb("pca"), _WBC)


@pytest.mark.parametrize("ig", [0.25, 1.0, 8.0])
def test_awb_input_gain(tensor_device, ig):
    rng = np.random.RandomState(9401)
    sensor = _sensor((8, 10), "grbg", 12)
    frames = rng.randint(100, 4000, size=(2, 8, 10)).astype(np.float32)
    awb = _awb("gray_world")
    expected = [_oracle(f, sensor, awb, _WBC, ig=ig) for f in frames]
    import torch
    actual = auto_white_balance(
        torch.as_tensor(frames, device=tensor_device), sensor, awb, _WBC,
        input_gains=(ig, ig))
    _check(actual, expected, ("ig", ig))


def test_awb_no_samples(tensor_device):
    sensor = _sensor((8, 10), "rggb", 12)
    dark = np.zeros((2, 8, 10), np.float32)
    awb = _awb("gray_world")
    expected = [_oracle(f, sensor, awb, _WBC, fb=(2.0, 0.5)) for f in dark]
    import torch
    actual = auto_white_balance(
        torch.as_tensor(dark, device=tensor_device), sensor, awb, _WBC,
        fallback_gains=((2.0, 0.5), (2.0, 0.5)))
    _check(actual, expected, "no_samples")


def test_awb_fallback_pairs(tensor_device):
    sensor = _sensor((8, 10), "rggb", 12)
    dark = np.zeros((2, 8, 10), np.float32)
    awb = _awb("gray_world")
    cases = [
        ((0.0, 8.0), "zero_allowed_pair"),
        ((9.0, 9.0), "pair_gt4_preserved"),
        ((float("nan"), 1.0), "nan_pair"),
        ("bad", "str_pair"),
        (None, "no_pair_manual"),
    ]
    import torch
    for fb, tag in cases:
        expected = [_oracle(f, sensor, awb, _WBC, fb=fb) for f in dark]
        actual = auto_white_balance(
            torch.as_tensor(dark, device=tensor_device), sensor, awb, _WBC,
            fallback_gains=(fb, fb))
        _check(actual, expected, tag)
    wbc_bad = dict(r_gain=float("nan"), b_gain=1.0)
    expected = [_oracle(f, sensor, awb, wbc_bad, fb=None) for f in dark]
    actual = auto_white_balance(
        torch.as_tensor(dark, device=tensor_device), sensor, awb, wbc_bad)
    _check(actual, expected, "invalid_manual_neutral")


def test_awb_disabled_ignores_controls(tensor_device):
    rng = np.random.RandomState(9401)
    sensor = _sensor((8, 10), "rggb", 12)
    frames = rng.randint(0, 4095, size=(2, 8, 10)).astype(np.float32)
    awb = _awb("nonexistent_algo", is_enable=False)
    expected = [_oracle(f, sensor, awb, _WBC, fb=(9.0, 9.0), ig=-5.0)
                for f in frames]
    import torch
    actual = auto_white_balance(
        torch.as_tensor(frames, device=tensor_device), sensor, awb, _WBC,
        fallback_gains=((9.0, 9.0), (9.0, 9.0)),
        input_gains=(-5.0, -5.0))
    _check(actual, expected, "disabled")
    bad_sensor = _sensor((8, 10), "rrgg", 12)
    expected = [_oracle(f, bad_sensor, awb, _WBC) for f in frames]
    actual = auto_white_balance(
        torch.as_tensor(frames, device=tensor_device), bad_sensor, awb, _WBC,
        input_gains=(1.0, 1.0, 1.0), fallback_gains=((1.0, 1.0),))
    _check(actual, expected, "disabled_bad_cfa_ignored")


def test_awb_validation_rejects(tensor_device):
    import torch
    sensor = _sensor((8, 10), "rggb", 12)
    frames = np.zeros((2, 8, 10), np.float32)
    for awb in (
        _awb("wat"),
        _awb("pca", percentage=0),
        _awb("pca", percentage=101),
        _awb("pca", percentage=float("nan")),
        _awb("gray_world", underexposed_percentage=-1),
        _awb("gray_world", overexposed_percentage=True),
    ):
        with pytest.raises(ValueError):
            auto_white_balance(
                torch.as_tensor(frames, device=tensor_device),
                sensor, awb, _WBC)
    bad_sensor = _sensor((8, 10), "rrgg", 12)
    with pytest.raises(ValueError):
        auto_white_balance(torch.as_tensor(frames, device=tensor_device),
                           bad_sensor, _awb(), _WBC)
    with pytest.raises(ValueError):
        auto_white_balance(torch.as_tensor(frames, device=tensor_device),
                           sensor, _awb(), _WBC, input_gains=(0.0, 1.0))
    with pytest.raises(ValueError):
        auto_white_balance(torch.as_tensor(frames, device=tensor_device),
                           sensor, _awb(), _WBC, input_gains=(True, 1.0))
    with pytest.raises(ValueError):
        auto_white_balance(torch.as_tensor(frames, device=tensor_device),
                           sensor, _awb(), _WBC, input_gains=(float("inf"), 1.0))
