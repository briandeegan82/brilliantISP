from __future__ import annotations

import copy

import numpy as np
import pytest

from brilliantisp_gpu.blocks.raw import black_level_correction, crop, dead_pixel_correction
from brilliantisp_gpu.types import RawTensor
from modules.black_level_correction.black_level_correction import BlackLevelCorrection
from modules.crop.crop import Crop
from modules.dead_pixel_correction.dead_pixel_correction import DeadPixelCorrection

PATTERNS = ("rggb", "bggr", "grbg", "gbrg")
PLATFORM = {
    "in_file": "fixture_9x7", "disable_progress_bar": True,
    "leave_pbar_string": False, "debug_enabled": False,
}
KINDS = ((np.uint8, 8), (np.uint16, 12), (np.uint16, 16),
         (np.uint32, 24), (np.uint32, 32))


def _sensor(shape, pattern, bits):
    return dict(height=shape[-2], width=shape[-1], bayer_pattern=pattern, bit_depth=bits)


def _device(raw, device):
    import torch

    return RawTensor(torch.as_tensor(raw.astype(np.int64), device=device), raw.dtype.name)


def _pixels(image):
    return image.codes.cpu().numpy().astype(image.storage_dtype)


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("origin", [(1, 0), (0, 1), (1, 1)])
def test_crop_geometry_phase_and_platform(tensor_device, pattern, origin):
    raw = np.arange(2 * 7 * 9, dtype=np.uint16).reshape(2, 7, 9)
    params = dict(is_enable=True, is_debug=False, is_save=False,
                  new_height=5, new_width=6, crop_x_start=origin[0], crop_y_start=origin[1])
    sensor = _sensor(raw.shape, pattern, 12)
    platform = dict(PLATFORM)
    expected_platform = dict(platform)
    expected = []
    for frame in raw:
        expected_sensor = copy.deepcopy(sensor)
        expected.append(Crop(frame, expected_platform, expected_sensor, params).execute())
    image = _device(raw, tensor_device)
    before = image.codes.clone()
    actual = crop(image, sensor, params, platform=platform)
    np.testing.assert_array_equal(_pixels(actual), np.stack(expected))
    np.testing.assert_array_equal(image.codes.cpu().numpy(), before.cpu().numpy())
    assert actual.storage_dtype == "uint16"
    assert sensor == expected_sensor and platform == expected_platform


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (7, 1), (2, 2), (7, 9), (8, 10)])
@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("dtype,bits", KINDS)
@pytest.mark.parametrize("block", ["dpc", "blc", "blc_linear"])
def test_raw_corrections_exact(tensor_device, shape, pattern, dtype, bits, block):
    rng = np.random.RandomState(9001)
    raw = rng.randint(0, int(np.iinfo(dtype).max) + 1, size=(2,) + shape).astype(dtype)
    raw.reshape(-1)[0], raw.reshape(-1)[-1] = 0, np.iinfo(dtype).max
    sensor = _sensor(raw.shape, pattern, bits)
    maximum = 2**bits - 1
    params = dict(
        is_enable=True, is_save=False, is_debug=False, dp_threshold=32,
        is_linear=block == "blc_linear",
        r_offset=1.5, gr_offset=0, gb_offset=-2, b_offset=3.25,
        r_sat=maximum, gr_sat=maximum - 1, gb_sat=maximum - 2, b_sat=maximum - 3,
    )
    cls, fn = (DeadPixelCorrection, dead_pixel_correction) if block == "dpc" else (
        BlackLevelCorrection, black_level_correction)
    if block == "dpc":
        expected = [cls(frame.copy(), sensor, params, dict(PLATFORM)).execute() for frame in raw]
    else:
        expected = [cls(frame.copy(), dict(PLATFORM), sensor, params).execute() for frame in raw]
    image = _device(raw, tensor_device)
    before = image.codes.clone()
    actual = fn(image, sensor, params)
    np.testing.assert_array_equal(_pixels(actual), np.stack(expected))
    np.testing.assert_array_equal(image.codes.cpu().numpy(), before.cpu().numpy())
    assert actual.storage_dtype == expected[0].dtype.name


@pytest.mark.parametrize("dtype,bits", KINDS)
def test_disabled_raw_blocks_preserve_input(tensor_device, dtype, bits):
    raw = np.array([0, np.iinfo(dtype).max], dtype=dtype).reshape(1, 1, 2)
    image = _device(raw, tensor_device)
    sensor = _sensor(raw.shape, "rggb", bits)
    for fn in (crop, dead_pixel_correction, black_level_correction):
        assert fn(image, sensor, {"is_enable": False}) is image
    np.testing.assert_array_equal(_pixels(image), raw)


@pytest.mark.parametrize("threshold,expected", [(25, 32), (24, 7)])
def test_dpc_threshold_boundary(tensor_device, threshold, expected):
    raw = np.full((1, 8, 10), 7, np.uint32)
    raw[0, 4, 4] = 32
    actual = dead_pixel_correction(
        _device(raw, tensor_device), _sensor(raw.shape, "rggb", 32),
        dict(is_enable=True, dp_threshold=threshold),
    )
    assert int(actual.codes[0, 4, 4]) == expected


@pytest.mark.parametrize("field,value", [
    ("new_width", 0), ("new_height", True), ("new_width", 2.5),
    ("crop_x_start", -1), ("crop_y_start", 8), ("crop_x_start", 0.5),
])
def test_crop_rejects_invalid_region(tensor_device, field, value):
    raw = np.zeros((1, 7, 9), np.uint16)
    params = dict(is_enable=True, is_debug=False, is_save=False,
                  new_height=3, new_width=4, crop_x_start=0, crop_y_start=0)
    params[field] = value
    sensor = _sensor(raw.shape, "rggb", 12)
    with pytest.raises(ValueError):
        crop(_device(raw, tensor_device), sensor, params)


@pytest.mark.parametrize("linear,offset,saturation", [
    (False, float("nan"), 4095), (True, 20, 20),
    (True, 20, 19), (True, 0, float("inf")),
])
def test_blc_rejects_invalid_constants(tensor_device, linear, offset, saturation):
    raw = np.zeros((1, 3, 3), np.uint16)
    params = dict(is_enable=True, is_linear=linear)
    params.update({f"{c}_offset": offset for c in ("r", "gr", "gb", "b")})
    params.update({f"{c}_sat": saturation for c in ("r", "gr", "gb", "b")})
    with pytest.raises(ValueError):
        black_level_correction(_device(raw, tensor_device), _sensor(raw.shape, "rggb", 12), params)
