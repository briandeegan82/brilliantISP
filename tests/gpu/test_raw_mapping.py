from __future__ import annotations

import numpy as np
import pytest

from brilliantisp_gpu.blocks.raw import companding, green_zero_fraction, oecf, repair_zero_green_cfa
from brilliantisp_gpu.types import RawTensor
from modules.oecf.oecf import OECF
from modules.pwc_generation.pwc_generation import PiecewiseCurve
from util.bayer_repair import green_zero_fraction as cpu_fraction
from util.bayer_repair import repair_zero_green_cfa as cpu_repair

PATTERNS = ("rggb", "bggr", "grbg", "gbrg")
PLATFORM = {"in_file": "fixture", "debug_enabled": False}


def _device(raw, device):
    import torch

    return RawTensor(torch.as_tensor(raw.astype(np.int64), device=device), raw.dtype.name)


def _pixels(image):
    return image.codes.cpu().numpy().astype(image.storage_dtype)


def _sensor(raw, pattern="rggb", bits=12, hdr=24):
    return dict(height=raw.shape[-2], width=raw.shape[-1], bayer_pattern=pattern,
                bit_depth=bits, hdr_bit_depth=hdr)


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (7, 1), (2, 2), (7, 9), (8, 10)])
@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("dtype", [np.uint16, np.uint32])
@pytest.mark.parametrize("threshold", [0.0, 7.5])
def test_green_repair_and_fraction_exact(tensor_device, shape, pattern, dtype, threshold):
    rng = np.random.RandomState(9101)
    raw = rng.randint(0, 16, (2,) + shape).astype(dtype)
    raw[0, ::2, :] = 0
    raw[1, -1, -1] = np.iinfo(dtype).max
    expected = [cpu_repair(frame, pattern, zero_thresh=threshold) for frame in raw]
    fractions = [cpu_fraction(frame, pattern, zero_thresh=threshold) for frame in raw]
    image = _device(raw, tensor_device)
    before = image.codes.clone()
    result, filled = repair_zero_green_cfa(image, pattern, zero_thresh=threshold)
    np.testing.assert_array_equal(_pixels(result), np.stack([v[0] for v in expected]))
    np.testing.assert_array_equal(filled.cpu().numpy(), [v[1] for v in expected])
    np.testing.assert_array_equal(green_zero_fraction(image, pattern, zero_thresh=threshold).cpu().numpy(), fractions)
    np.testing.assert_array_equal(image.codes.cpu().numpy(), before.cpu().numpy())
    assert result.storage_dtype == np.dtype(dtype).name


@pytest.mark.parametrize("pre,post", [(0, 0), (17, 0), (-3, 11)])
@pytest.mark.parametrize("knots", ["small", "endpoint_65536", "sparse32"])
def test_companding_exact(tensor_device, pre, post, knots):
    if knots == "sparse32":
        values = np.array([0, 1, 17, 18, 2**24 - 1, 2**24, 2**24 + 1,
                           2**31 - 1, 2**31, 2**31 + 1, 2**32 - 2, 2**32 - 1], np.uint32).reshape(3, 4)
        pin, pout = [0, 2**24, 2**31, 2**32], [0, 100000, 1234567891, 2**32]
        bits, hdr = 32, 32
    elif knots == "endpoint_65536":
        values = np.array([0, 1, 2, 255, 4095, 32767, 32768, 32769,
                           65532, 65533, 65534, 65535], np.uint16).reshape(3, 4)
        pin, pout = [0, 32768, 65536], [0, 65535, 2**32]
        bits, hdr = 16, 32
    else:
        values = np.array([0, 1, 2, 15, 16, 17, 127, 128, 129, 254, 255, 65535], np.uint16).reshape(3, 4)
        pin, pout = [1, 16, 128, 256], [1, 200, 1000, 4096]
        bits, hdr = 8, 12
    raw = np.stack((values, values[:, ::-1]))
    params = dict(is_enable=True, is_debug=False, is_save=False,
                  companded_pin=pin, companded_pout=pout, pedestal=999,
                  pre_linearization_black_level=pre, post_linearization_black_level=post)
    sensor = _sensor(raw, bits=bits, hdr=hdr)
    expected = [PiecewiseCurve(frame.copy(), dict(PLATFORM), sensor, params).execute() for frame in raw]
    image = _device(raw, tensor_device)
    result = companding(image, sensor, params)
    np.testing.assert_array_equal(_pixels(result), np.stack(expected))
    np.testing.assert_array_equal(_pixels(image), raw)
    assert result.storage_dtype == "uint32"


def test_companding_legacy_pedestal_and_disabled_dtype(tensor_device):
    raw = np.arange(12, dtype=np.uint16).reshape(1, 3, 4)
    sensor = _sensor(raw, bits=8, hdr=12)
    params = dict(is_enable=True, is_debug=False, is_save=False,
                  companded_pin=[0, 256], companded_pout=[0, 4096], pedestal=3)
    expected = PiecewiseCurve(raw[0].copy(), dict(PLATFORM), sensor, params).execute()
    result = companding(_device(raw, tensor_device), sensor, params)
    np.testing.assert_array_equal(_pixels(result)[0], expected)
    disabled = companding(_device(raw, tensor_device), sensor, {"is_enable": False})
    assert disabled.storage_dtype == "uint32"
    np.testing.assert_array_equal(_pixels(disabled), raw.astype(np.uint32))


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("out_bits", [8, 16, 24, 32])
def test_oecf_per_cfa_lookup_and_fallback(tensor_device, pattern, out_bits):
    raw = np.random.RandomState(9102).randint(0, 16, (2, 7, 9)).astype(np.uint16)
    maximum = 2**out_bits - 1
    lut = np.linspace(0, maximum, 16).astype(np.uint32).tolist()
    params = dict(is_enable=True, is_save=False, r_lut=lut, bl_lut=lut[::-1],
                  gb_lut=[maximum // 2] * 16, output_bit_depth=out_bits)
    sensor = _sensor(raw, pattern)
    expected = [OECF(frame, dict(PLATFORM), sensor, params).execute() for frame in raw]
    image = _device(raw, tensor_device)
    actual = oecf(image, sensor, params)
    np.testing.assert_array_equal(_pixels(actual), np.stack(expected))
    np.testing.assert_array_equal(_pixels(image), raw)
    assert actual.storage_dtype == ("uint16" if out_bits <= 16 else "uint32")


@pytest.mark.parametrize("hdr", [8, 12, 24, 32])
def test_oecf_identity_range_and_disabled(tensor_device, hdr):
    raw = np.array([0, 7, 4096, 2**32 - 1], np.uint32).reshape(1, 2, 2)
    sensor = _sensor(raw, hdr=hdr)
    params = dict(is_enable=True, is_save=False)
    expected = OECF(raw[0], dict(PLATFORM), sensor, params).execute()
    image = _device(raw, tensor_device)
    actual = oecf(image, sensor, params)
    np.testing.assert_array_equal(_pixels(actual)[0], expected)
    assert actual.storage_dtype == expected.dtype.name
    assert oecf(image, sensor, {"is_enable": False}) is image


def test_oecf_skips_empty_cfa_planes(tensor_device):
    raw = np.array([[[0]]], np.uint16)
    params = dict(is_enable=True, is_save=False, r_lut=[31], gr_lut=None,
                  gb_lut=None, bl_lut=None, output_bit_depth=8)
    actual = oecf(_device(raw, tensor_device), _sensor(raw), params)
    np.testing.assert_array_equal(_pixels(actual), [[[31]]])


@pytest.mark.parametrize("lut", [[0, 1.5], [0, float("nan")], [0, -1], [0, 256], []])
def test_oecf_rejects_invalid_lut(tensor_device, lut):
    raw = np.zeros((1, 2, 2), np.uint16)
    with pytest.raises(ValueError):
        oecf(_device(raw, tensor_device), _sensor(raw),
             dict(is_enable=True, r_lut=lut, output_bit_depth=8))
