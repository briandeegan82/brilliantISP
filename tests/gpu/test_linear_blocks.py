"""Linear blocks DG/LSC/WB vs CPU compute_linear oracles (seed 9201)."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from brilliantisp_gpu.blocks.linear import (
    digital_gain, lens_shading_correction, white_balance)
from brilliantisp_gpu.types import RawTensor
from modules.digital_gain.digital_gain import DigitalGain
from modules.lens_shading_correction.lens_shading_correction import LensShadingCorrection
from modules.white_balance.white_balance_optimized import WhiteBalanceOptimized
from tools.gpu_validation.metrics import compare_arrays
from tools.gpu_validation.contracts import FLOAT32_NORMALIZED

PATTERNS = ("rggb", "bggr", "grbg", "gbrg")
PLATFORM = {
    "in_file": "fixture_9x7", "disable_progress_bar": True,
    "leave_pbar_string": False, "debug_enabled": False,
}
SHAPES = [(1, 1), (1, 7), (7, 1), (7, 9), (8, 10)]
HDR = 12


def _sensor(shape, pattern, bits=HDR):
    return dict(height=shape[-2], width=shape[-1], bayer_pattern=pattern,
                bit_depth=bits, hdr_bit_depth=bits)


def _codes(raw, device):
    import torch
    return RawTensor(torch.as_tensor(raw.astype(np.int64), device=device),
                     str(raw.dtype.name))


def _t(arr, device):
    import torch
    return torch.as_tensor(np.asarray(arr, np.float32), device=device)


def _dg_params(**over):
    p = dict(is_enable=True, is_save=False, is_debug=False, is_auto=False,
             exposure_correction_mode="step",
             gain_array=[1.0, 1.7, 2.5, 32.0], current_gain=1)
    p.update(over)
    return p


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("pattern", PATTERNS)
def test_dg_exact_manual_and_auto(tensor_device, shape, pattern):
    rng = np.random.RandomState(9201)
    sensor = _sensor(shape, pattern)
    raw = rng.randint(0, 2**16, size=(2,) + shape).astype(np.uint16)
    raw.reshape(-1)[0] = 0
    for mode, fb, idx in (
        ("step", None, 0), ("step", -1, 1), ("step", 1, 3),
        ("step", 0, 2), ("direct", -1, 2), ("manual", -1, 1),
    ):
        params = _dg_params(current_gain=idx, ae_feedback=fb,
                            exposure_correction_mode=mode,
                            is_auto=(mode != "manual"))
        p_cpu = copy.deepcopy(params)
        expected = [DigitalGain(f.copy(), dict(PLATFORM), sensor, p_cpu).compute_linear()
                    for f in raw]
        exp_img = np.stack([e[0] for e in expected])
        exp_idx = tuple(e[1] for e in expected)
        image = _codes(raw, tensor_device)
        before = image.codes.clone()
        params_before = copy.deepcopy(params)
        if mode == "manual":
            actual, ai = digital_gain(image, sensor, params)
        else:
            actual, ai = digital_gain(
                image, sensor, params,
                gain_indices=(idx, idx), ae_feedback=(fb, fb))
        assert ai == exp_idx, (mode, fb, idx)
        np.testing.assert_array_equal(actual.cpu().numpy(), exp_img,
                                      err_msg=f"{mode}/{fb}/{idx}/{shape}/{pattern}")
        np.testing.assert_array_equal(image.codes.cpu().numpy(), before.cpu().numpy())
        assert params == params_before


@pytest.mark.parametrize("pattern", PATTERNS)
def test_dg_uint32_boundaries(tensor_device, pattern):
    rng = np.random.RandomState(9201)
    sensor = _sensor((7, 9), pattern, bits=32)
    raw = rng.randint(0, 2**32, size=(2, 7, 9), dtype=np.int64).astype(np.uint32)
    raw[0, 0, 0], raw[0, 0, 1], raw[1, -1, -1] = 0, 2**31, 2**32 - 1
    params = _dg_params(gain_array=[0.25, 1.0, 1.7, 32.0], current_gain=2)
    p_cpu = copy.deepcopy(params)
    expected = [DigitalGain(f.copy(), dict(PLATFORM), sensor, p_cpu).compute_linear()[0]
                for f in raw]
    actual, idx = digital_gain(_codes(raw, tensor_device), sensor, params)
    assert idx == (2, 2)
    np.testing.assert_array_equal(actual.cpu().numpy(), np.stack(expected))


@pytest.mark.parametrize("shape", SHAPES[:2])
@pytest.mark.parametrize("pattern", PATTERNS)
def test_dg_disabled_and_bounds(tensor_device, shape, pattern):
    sensor = _sensor(shape, pattern)
    raw = np.arange(np.prod(shape) * 2).reshape(2, *shape).astype(np.uint16)
    params = _dg_params(is_enable=False, current_gain=3)
    p_cpu = copy.deepcopy(params)
    expected = [DigitalGain(f.copy(), dict(PLATFORM), sensor, p_cpu).compute_linear()
                for f in raw]
    actual, idx = digital_gain(_codes(raw, tensor_device), sensor, params)
    assert idx == (3, 3)
    np.testing.assert_array_equal(actual.cpu().numpy(),
                                  np.stack([e[0] for e in expected]))
    with pytest.raises(ValueError):
        digital_gain(_codes(raw, tensor_device), sensor,
                     _dg_params(), gain_indices=(99, 0))
    with pytest.raises(ValueError):
        digital_gain(_codes(raw, tensor_device), sensor,
                     _dg_params(), gain_indices=(-1, 0))
    with pytest.raises(ValueError):
        digital_gain(_codes(raw, tensor_device), sensor,
                     _dg_params(), gain_indices=(0,))
    with pytest.raises(ValueError):
        digital_gain(_codes(raw, tensor_device), sensor,
                     _dg_params(), gain_indices=(1.5, 0))
    with pytest.raises(ValueError):
        digital_gain(_codes(raw, tensor_device), sensor,
                     _dg_params(), gain_indices=(True, 0))
    with pytest.raises(ValueError):
        digital_gain(_codes(raw, tensor_device), sensor,
                     _dg_params(current_gain=1.5))


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("pattern", PATTERNS)
def test_lsc_gate_all_coeffs(tensor_device, shape, pattern):
    rng = np.random.RandomState(9201)
    sensor = _sensor(shape, pattern)
    base = (rng.rand(2, *shape) * 4000 - 100).astype(np.float32)
    for coeffs in (
        dict(r_k1=0.4, r_k2=0.05, gr_k1=0.3, gr_k2=-0.02,
             gb_k1=0.2, gb_k2=0.01, b_k1=-0.5, b_k2=-0.3),
        dict(r_k1=0.0, r_k2=0.0, gr_k1=0.0, gr_k2=0.0,
             gb_k1=0.0, gb_k2=0.0, b_k1=0.0, b_k2=0.0),
        dict(r_k1=-2.0, r_k2=-1.0, gr_k1=0.0, gr_k2=0.0,
             gb_k1=0.0, gb_k2=0.0, b_k1=-3.0, b_k2=-2.0),
        dict(r_k1=-1e40, r_k2=0.0, gr_k1=-1e40, gr_k2=-1e40,
             gb_k1=-1e40, gb_k2=0.0, b_k1=-1e40, b_k2=-1e40),
    ):
        params = dict(is_enable=True, is_save=False, **coeffs)
        p_cpu = copy.deepcopy(params)
        expected = [LensShadingCorrection(
            f.copy(), dict(PLATFORM), sensor, p_cpu).compute_linear() for f in base]
        actual = lens_shading_correction(_t(base, tensor_device), sensor, params)
        result = compare_arrays(np.stack(expected), actual.cpu().numpy(),
                                FLOAT32_NORMALIZED, scale=2**HDR - 1)
        assert result["passed"], (coeffs, shape, pattern, result)


def test_lsc_positive_overflow_rejected(tensor_device):
    rng = np.random.RandomState(9201)
    sensor = _sensor((8, 10), "rggb")
    base = (rng.rand(2, 8, 10) * 4000).astype(np.float32)
    params = dict(is_enable=True, is_save=False, r_k1=1e40)
    with pytest.raises(ValueError):
        LensShadingCorrection(
            base[0].copy(), dict(PLATFORM), sensor, copy.deepcopy(params)
        ).compute_linear()
    with pytest.raises(ValueError):
        lens_shading_correction(_t(base, tensor_device), sensor, params)


@pytest.mark.parametrize("pattern", PATTERNS[:2])
def test_lsc_disabled_and_f64_input(tensor_device, pattern):
    rng = np.random.RandomState(9201)
    sensor = _sensor((8, 10), pattern)
    base64 = (rng.rand(2, 8, 10) * 4095).astype(np.float64)
    params = dict(is_enable=False, is_save=False)
    p_cpu = copy.deepcopy(params)
    expected = [LensShadingCorrection(
        f.copy(), dict(PLATFORM), sensor, p_cpu).compute_linear() for f in base64]
    import torch
    image = torch.as_tensor(base64, device=tensor_device)
    before = image.clone()
    actual = lens_shading_correction(image, sensor, params)
    assert actual.dtype == torch.float32
    np.testing.assert_array_equal(actual.cpu().numpy(), np.stack(expected))
    np.testing.assert_array_equal(image.cpu().numpy(), before.cpu().numpy())
    params = dict(is_enable=True, is_save=False, r_k1=0.5, r_k2=0.1)
    p_cpu = copy.deepcopy(params)
    params_before = copy.deepcopy(params)
    expected = [LensShadingCorrection(
        f.copy(), dict(PLATFORM), sensor, p_cpu).compute_linear() for f in base64]
    actual = lens_shading_correction(image, sensor, params)
    np.testing.assert_array_equal(image.cpu().numpy(), before.cpu().numpy())
    assert params == params_before
    result = compare_arrays(np.stack(expected), actual.cpu().numpy(),
                            FLOAT32_NORMALIZED, scale=2**HDR - 1)
    assert result["passed"], result


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("pattern", PATTERNS)
def test_wb_manual_auto_exact(tensor_device, shape, pattern):
    rng = np.random.RandomState(9201)
    sensor = _sensor(shape, pattern)
    base = (rng.rand(2, *shape) * 3000 + 50).astype(np.float32)
    base[0] *= -0.5
    for mode, gains in (
        ("manual", None), ("auto", ((1.314159, 2.71828), (0.5, 4.0))),
    ):
        params = dict(is_enable=True, is_save=False, is_debug=False,
                      is_auto=(mode == "auto"), r_gain=1.314159, b_gain=2.71828)
        p_cpu = copy.deepcopy(params)
        params_before = copy.deepcopy(params)
        expected = [WhiteBalanceOptimized(
            f.copy(), dict(PLATFORM), sensor, p_cpu,
            gains[i] if mode == "auto" else (0.0, 0.0)).compute_linear()
            for i, f in enumerate(base)]
        image = _t(base, tensor_device)
        before = image.clone()
        actual = white_balance(image, sensor, params,
                               gains if mode == "auto" else (0.0, 0.0))
        np.testing.assert_array_equal(actual.cpu().numpy(), np.stack(expected),
                                      err_msg=f"{mode}/{shape}/{pattern}")
        np.testing.assert_array_equal(image.cpu().numpy(), before.cpu().numpy())
        assert params == params_before


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("mode", ["manual", "auto"])
def test_wb_f64_input_unmutated(tensor_device, pattern, mode):
    """Enabled WB on float64 input must not alias/mutate caller data and
    must not prematurely cast to f32 (16777217 is not f32-representable)."""
    import torch
    sensor = _sensor((8, 10), pattern)
    base = np.zeros((2, 8, 10), np.float64)
    base[0] = 16777217.0
    base[1] = np.arange(80).reshape(8, 10).astype(np.float64) * 3 - 100
    gains = ((2.5, 1.0), (1.0, 2.5)) if mode == "auto" else None
    params = dict(is_enable=True, is_save=False, is_debug=False,
                  is_auto=(mode == "auto"), r_gain=2.5, b_gain=1.0)
    p_cpu = copy.deepcopy(params)
    per = gains if mode == "auto" else ((2.5, 1.0), (2.5, 1.0))
    expected = [WhiteBalanceOptimized(
        f.copy(), dict(PLATFORM), sensor, p_cpu, per[i]).compute_linear()
        for i, f in enumerate(base)]
    image = torch.as_tensor(base, device=tensor_device)
    before = image.clone()
    actual = white_balance(image, sensor, params,
                           gains if gains else (2.5, 1.0))
    np.testing.assert_array_equal(image.cpu().numpy(), before.cpu().numpy())
    np.testing.assert_array_equal(actual.cpu().numpy(), np.stack(expected),
                                  err_msg=f"{mode}/{pattern}")


@pytest.mark.parametrize("pattern", PATTERNS[:2])
def test_wb_disabled_and_rejections(tensor_device, pattern):
    rng = np.random.RandomState(9201)
    sensor = _sensor((8, 10), pattern)
    base64 = (rng.rand(2, 8, 10) * 4095).astype(np.float64)
    params = dict(is_enable=False, is_save=False, is_debug=False, is_auto=True)
    p_cpu = copy.deepcopy(params)
    expected = [WhiteBalanceOptimized(
        f.copy(), dict(PLATFORM), sensor, p_cpu, (9.0, 9.0)).compute_linear()
        for f in base64]
    import torch
    actual = white_balance(torch.as_tensor(base64, device=tensor_device),
                           sensor, params, ((9.0, 9.0), (9.0, 9.0)))
    np.testing.assert_array_equal(actual.cpu().numpy(), np.stack(expected))
    bad = torch.as_tensor(base64, device=tensor_device).clone()
    bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        white_balance(bad, sensor, dict(is_enable=True, is_auto=False), (1.0, 1.0))


@pytest.mark.parametrize("gains_device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu_hardware)])
def test_wb_auto_tensor_gains(tensor_device, gains_device, request):
    """Bx2 torch gain tensors (CPU or device) accepted in auto mode."""
    import torch
    if gains_device == "cuda":
        gains_device = request.getfixturevalue("cuda_device")
    sensor = _sensor((8, 10), "rggb")
    base = np.arange(160, dtype=np.float64).reshape(2, 8, 10) * 7.0
    gains = np.array([[1.314159, 2.71828], [0.5, 4.0]])
    params = dict(is_enable=True, is_save=False, is_debug=False, is_auto=True)
    p_cpu = copy.deepcopy(params)
    expected = [WhiteBalanceOptimized(
        f.copy(), dict(PLATFORM), sensor, p_cpu, tuple(gains[i])
    ).compute_linear() for i, f in enumerate(base)]
    g = torch.as_tensor(gains, dtype=torch.float64, device=gains_device)
    actual = white_balance(torch.as_tensor(base, device=tensor_device),
                           sensor, params, g)
    np.testing.assert_array_equal(actual.cpu().numpy(), np.stack(expected))


@pytest.mark.parametrize("bad_shape", [(2, 3), (1, 2), (4, 2)])
def test_wb_auto_gain_shape_rejected(tensor_device, bad_shape):
    import torch
    from brilliantisp_gpu.types import GpuError
    sensor = _sensor((8, 10), "rggb")
    image = torch.zeros(2, 8, 10, device=tensor_device)
    with pytest.raises(GpuError):
        white_balance(image, sensor, dict(is_enable=True, is_auto=True),
                      torch.ones(bad_shape))
