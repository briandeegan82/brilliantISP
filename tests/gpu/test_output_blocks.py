"""GPU quantize_output/preview_from_native vs util.native_output (seed 9901)."""

from __future__ import annotations

import numpy as np
import pytest

from brilliantisp_gpu.blocks.output import quantize_output, preview_from_native
from brilliantisp_gpu.types import GpuError
from util.native_output import quantize_output as cpu_q, preview_from_native as cpu_p
from util.output_contract import OutputSpec
from tools.gpu_validation.metrics import compare_arrays
from tools.gpu_validation.contracts import FLOAT32_NORMALIZED


def _spec(**over):
    m = dict(model="rgb", color_space="srgb", transfer="srgb", gamma=2.2,
             bit_depth=8, format="png", matrix="bt709", range_mode="full",
             layout="444", packing="yuyv", is_save=False)
    m.update(over)
    return OutputSpec(**m)


def _t(a, d):
    import torch
    return torch.as_tensor(np.asarray(a), device=d)


SPECS = [
    _spec(),
    _spec(bit_depth=16, format="tiff"),
    _spec(transfer="rec709", bit_depth=16, format="tiff"),
    _spec(transfer="power", gamma=1.8, bit_depth=8),
    _spec(transfer="linear", bit_depth=16, format="tiff"),
    _spec(transfer="linear", bit_depth=32, format="tiff"),
    _spec(model="ycbcr", format="yuv", bit_depth=8, matrix="bt601"),
    _spec(model="ycbcr", format="yuv", bit_depth=16, range_mode="limited"),
    _spec(model="ycbcr", format="yuv", bit_depth=8, layout="422", packing="yuyv"),
    _spec(model="ycbcr", format="yuv", bit_depth=16, layout="422", packing="uyvy", matrix="bt601", range_mode="limited"),
]
_UINT_DT = {8: np.uint8, 16: np.uint16}


@pytest.mark.parametrize("spec", SPECS, ids=[f"s{i}" for i in range(len(SPECS))])
@pytest.mark.parametrize("batched", [True, False])
def test_quantize_and_preview(tensor_device, spec, batched):
    rng = np.random.RandomState(9901)
    shape = (2, 8, 10, 3) if batched else (8, 10, 3)
    rgb = (rng.rand(*shape) * 1.3 - 0.1).astype(np.float32)
    if spec.floating_linear:
        rgb = (rgb * 128 - 40).astype(np.float32)
    expected = cpu_q(rgb, spec) if not batched else np.stack([cpu_q(f, spec) for f in rgb])
    image = _t(rgb, tensor_device)
    before = image.clone()
    out = quantize_output(image, spec)
    np.testing.assert_array_equal(image.cpu().numpy(), before.cpu().numpy())
    got = out.cpu().numpy()
    assert got.shape == expected.shape
    assert got.dtype == expected.dtype, (spec, got.dtype, expected.dtype)
    assert out.device == image.device
    if expected.dtype.kind == "f":
        res = compare_arrays(expected, got, FLOAT32_NORMALIZED, scale=1.0)
        assert res["passed"], (spec, res)
    else:
        np.testing.assert_array_equal(got, expected)
    expected_preview = (cpu_p(expected, spec) if not batched
                        else np.stack([cpu_p(f, spec) for f in expected]))
    import torch
    prev = preview_from_native(out, spec)
    assert prev.dtype == torch.uint8
    np.testing.assert_array_equal(prev.cpu().numpy(), expected_preview)


def test_preview_srgb8_clones(tensor_device):
    spec = _spec()
    rng = np.random.RandomState(9901)
    rgb = rng.randint(0, 256, size=(2, 8, 10, 3)).astype(np.uint8)
    image = _t(rgb, tensor_device)
    before = image.clone()
    out = preview_from_native(image, spec)
    original = out.clone()
    out[0, 0, 0] = (int(original[0, 0, 0, 0]) + 1) % 256
    np.testing.assert_array_equal(image.cpu().numpy(), before.cpu().numpy())
    assert int(out[0, 0, 0, 0]) != int(original[0, 0, 0, 0])
    exp = np.stack([cpu_p(f, spec) for f in rgb])
    np.testing.assert_array_equal(original.cpu().numpy(), exp)


def test_output_rejects(tensor_device):
    import torch
    spec = _spec()
    with pytest.raises(ValueError):
        quantize_output(_t(np.ones((8, 10, 3), np.int64), tensor_device), spec)
    with pytest.raises(GpuError):
        quantize_output(_t(np.zeros((0, 8, 10, 3), np.float32), tensor_device), spec)
    with pytest.raises(ValueError):
        quantize_output(_t(np.full((8, 10, 3), np.nan, np.float32), tensor_device), spec)
    with pytest.raises(ValueError):
        preview_from_native(_t(np.ones((8, 10, 3), np.float32), tensor_device), spec)
    with pytest.raises(ValueError):
        preview_from_native(_t(np.ones((8, 10, 3), np.uint16), tensor_device), spec)
    spec32 = _spec(transfer="linear", bit_depth=32, format="tiff")
    with pytest.raises(ValueError):
        preview_from_native(_t(np.ones((8, 10, 3), np.float64), tensor_device), spec32)
    spec422 = _spec(model="ycbcr", format="yuv", bit_depth=8, layout="422")
    with pytest.raises(ValueError):
        preview_from_native(_t(np.ones((8, 6), np.uint8), tensor_device), spec422)
    with pytest.raises(ValueError):
        preview_from_native(_t(np.ones((8, 8), np.int64), tensor_device), spec)
