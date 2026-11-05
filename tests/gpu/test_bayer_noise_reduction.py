"""Bayer noise reduction JBF vs cpu_only=True CPU oracle (seed 9301)."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from brilliantisp_gpu.blocks.bayer_noise_reduction import bayer_noise_reduction
from modules.bayer_noise_reduction.bayer_noise_reduction import BayerNoiseReduction
from tools.gpu_validation.metrics import compare_arrays
from tools.gpu_validation.contracts import FLOAT32_NORMALIZED

PATTERNS = ("rggb", "bggr", "grbg", "gbrg")
PLATFORM = {
    "in_file": "fixture_9x7", "disable_progress_bar": True,
    "leave_pbar_string": False, "debug_enabled": False,
}
SHAPES = [(1, 1), (1, 7), (7, 1), (2, 2), (7, 9), (8, 10)]


def _sensor(shape, pattern, bits):
    return dict(height=shape[-2], width=shape[-1], bayer_pattern=pattern,
                bit_depth=bits, hdr_bit_depth=bits)


def _bnr_params(window=5, **over):
    p = dict(is_enable=True, is_save=False, filter_window=window,
             r_std_dev_s=0.8, r_std_dev_r=0.08,
             g_std_dev_s=1.1, g_std_dev_r=0.1,
             b_std_dev_s=1.4, b_std_dev_r=0.2)
    p.update(over)
    return p


def _compare(expected_list, actual_t, bits, tag):
    expected = np.stack(expected_list)
    result = compare_arrays(expected, actual_t.cpu().numpy(),
                            FLOAT32_NORMALIZED, scale=2**bits - 1)
    assert result["passed"], (tag, result)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("bits", [12, 24, 32])
@pytest.mark.parametrize("window", [3, 5])
def test_bnr_exact_batched(tensor_device, shape, pattern, bits, window):
    rng = np.random.RandomState(9301)
    sensor = _sensor((shape[0] + shape[0] % 2, shape[1] + shape[1] % 2),
                     pattern, bits)
    frames = rng.randint(0, 2**min(bits, 20), size=(2,) + shape).astype(np.float32)
    params = _bnr_params(window)
    p_cpu = copy.deepcopy(params)
    expected = [BayerNoiseReduction(f.copy(), sensor, p_cpu, dict(PLATFORM),
                                    cpu_only=True).compute_linear() for f in frames]
    import torch
    image = torch.as_tensor(frames, device=tensor_device)
    before = image.clone()
    p_before = copy.deepcopy(params)
    actual = bayer_noise_reduction(image, sensor, params)
    _compare(expected, actual, bits, (shape, pattern, bits, window))
    np.testing.assert_array_equal(image.cpu().numpy(), before.cpu().numpy())
    assert params == p_before


def _structured(shape, kind):
    rng = np.random.RandomState(9301)
    h, w = shape
    if kind == "impulse":
        a = np.zeros((2, h, w), np.float32); a[:, h // 2, w // 2] = 4000.0
    elif kind == "checker":
        a = (np.indices((h, w)).sum(0) % 2 * 3000.0).astype(np.float32)[None].repeat(2, 0)
    elif kind == "dark":
        a = np.full((2, h, w), 3.0, np.float32)
    elif kind == "saturated":
        a = np.full((2, h, w), 4095.0, np.float32)
    else:
        a = (rng.rand(2, h, w) * 5000 - 500).astype(np.float32)
    return a


@pytest.mark.parametrize("kind", ["impulse", "checker", "dark", "saturated", "signed"])
@pytest.mark.parametrize("pattern", PATTERNS)
def test_bnr_structured(tensor_device, kind, pattern):
    import torch
    sensor = _sensor((8, 10), pattern, 12)
    frames = _structured((8, 10), kind)
    params = _bnr_params(3)
    p_cpu = copy.deepcopy(params)
    expected = [BayerNoiseReduction(f.copy(), sensor, p_cpu, dict(PLATFORM),
                                    cpu_only=True).compute_linear() for f in frames]
    image = torch.as_tensor(frames, device=tensor_device)
    before = image.clone()
    p_before = copy.deepcopy(params)
    actual = bayer_noise_reduction(image, sensor, params)
    assert actual.dtype == torch.float32
    assert actual.shape == (2, 8, 10)
    assert actual.device.type == image.device.type
    _compare(expected, actual, 12, (kind, pattern))
    np.testing.assert_array_equal(image.cpu().numpy(), before.cpu().numpy())
    assert params == p_before


def test_bnr_f64_input_downcast(tensor_device):
    import torch
    rng = np.random.RandomState(9301)
    sensor = _sensor((8, 10), "rggb", 12)
    frames = (rng.rand(2, 8, 10) * 4095).astype(np.float64)
    params = _bnr_params(5)
    p_cpu = copy.deepcopy(params)
    expected = [BayerNoiseReduction(f.copy(), sensor, p_cpu, dict(PLATFORM),
                                    cpu_only=True).compute_linear() for f in frames]
    image = torch.as_tensor(frames, device=tensor_device)
    before = image.clone()
    p_before = copy.deepcopy(params)
    actual = bayer_noise_reduction(image, sensor, params)
    assert actual.dtype == torch.float32
    assert actual.shape == frames.shape
    _compare(expected, actual, 12, "f64")
    np.testing.assert_array_equal(image.cpu().numpy(), before.cpu().numpy())
    assert params == p_before


@pytest.mark.parametrize("bits", [24, 32])
@pytest.mark.parametrize("pattern", PATTERNS)
def test_bnr_full_declared_range(tensor_device, bits, pattern):
    """Full declared-range HDR24/32 codes including explicit 0/max."""
    import torch
    rng = np.random.RandomState(9301)
    code_max = 2**bits - 1
    sensor = _sensor((8, 10), pattern, bits)
    frames = (rng.rand(2, 8, 10) * code_max).astype(np.float32)
    frames[0, 0, 0] = 0.0
    frames[1, -1, -1] = float(code_max)
    params = _bnr_params(5)
    p_cpu = copy.deepcopy(params)
    expected = [BayerNoiseReduction(f.copy(), sensor, p_cpu, dict(PLATFORM),
                                    cpu_only=True).compute_linear() for f in frames]
    image = torch.as_tensor(frames, device=tensor_device)
    before = image.clone()
    p_before = copy.deepcopy(params)
    actual = bayer_noise_reduction(image, sensor, params)
    assert actual.dtype == torch.float32
    assert actual.shape == frames.shape
    _compare(expected, actual, bits, (bits, pattern))
    np.testing.assert_array_equal(image.cpu().numpy(), before.cpu().numpy())
    assert params == p_before


@pytest.mark.parametrize("pattern", PATTERNS[:2])
def test_bnr_huge_sigma_r(tensor_device, pattern):
    """sigma_r=1e20 promotes the division term to f64 on HDR12."""
    import torch
    rng = np.random.RandomState(9301)
    sensor = _sensor((8, 10), pattern, 12)
    frames = (rng.rand(2, 8, 10) * 4095).astype(np.float32)
    params = _bnr_params(5, r_std_dev_r=1e20, g_std_dev_r=1e20,
                         b_std_dev_r=1e20)
    p_cpu = copy.deepcopy(params)
    expected = [BayerNoiseReduction(f.copy(), sensor, p_cpu, dict(PLATFORM),
                                    cpu_only=True).compute_linear() for f in frames]
    image = torch.as_tensor(frames, device=tensor_device)
    actual = bayer_noise_reduction(image, sensor, params)
    assert torch.isfinite(actual).all()
    _compare(expected, actual, 12, (1e20, pattern))


@pytest.mark.parametrize("shape", SHAPES[:3])
def test_bnr_disabled_and_validation(tensor_device, shape):
    import torch
    sensor = _sensor(shape, "rggb", 12)
    frames = np.arange(2 * np.prod(shape)).reshape(2, *shape).astype(np.float32) + 1
    params = _bnr_params(is_enable=False)
    p_cpu = copy.deepcopy(params)
    expected = [BayerNoiseReduction(f.copy(), sensor, p_cpu, dict(PLATFORM),
                                    cpu_only=True).compute_linear() for f in frames]
    actual = bayer_noise_reduction(
        torch.as_tensor(frames, device=tensor_device), sensor, params)
    np.testing.assert_array_equal(actual.cpu().numpy(), np.stack(expected))
    bad = torch.as_tensor(frames, device=tensor_device).clone()
    bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        bayer_noise_reduction(bad, sensor, _bnr_params())


@pytest.mark.parametrize("window", [4, 0])
@pytest.mark.parametrize("pattern", PATTERNS[:2])
def test_bnr_window_corrections(tensor_device, window, pattern):
    rng = np.random.RandomState(9301)
    sensor = _sensor((8, 10), pattern, 12)
    frames = (rng.rand(2, 8, 10) * 4095).astype(np.float32)
    params = _bnr_params(window)
    p_cpu = copy.deepcopy(params)
    with pytest.warns(UserWarning):
        expected = [BayerNoiseReduction(f.copy(), sensor, p_cpu, dict(PLATFORM),
                                        cpu_only=True).compute_linear() for f in frames]
    import torch
    actual = bayer_noise_reduction(
        torch.as_tensor(frames, device=tensor_device), sensor, params)
    _compare(expected, actual, 12, (window, pattern))
