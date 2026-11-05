"""GPU colour_correction_matrix vs ColorCorrectionMatrixOptimized (seed 9801)."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from brilliantisp_gpu.blocks.color import color_correction_matrix
from brilliantisp_gpu.types import GpuError
from modules.color_correction_matrix.color_correction_matrix_optimized import (
    ColorCorrectionMatrixOptimized)
from tools.gpu_validation.metrics import compare_arrays
from tools.gpu_validation.contracts import FLOAT32_NORMALIZED

PLATFORM = {
    "in_file": "fixture_9x7", "disable_progress_bar": True,
    "leave_pbar_string": False, "debug_enabled": False,
}
SHAPES = [(1, 1), (1, 7), (7, 1), (7, 9), (8, 10)]
IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
MATRICES = {
    "identity": IDENTITY,
    "nonsym": [[1.4, -0.3, 0.1], [-0.2, 1.1, 0.05], [0.02, -0.07, 1.3]],
    "negative": [[-1.0, 0.5, 0.0], [0.25, -0.75, 0.1], [0.0, -0.2, -1.0]],
    "zero": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    "mix": [[0.8, 0.15, 0.05], [0.1, 0.85, 0.05], [0.05, 0.1, 0.85]],
}


def _params(mat, **over):
    p = dict(is_enable=True, is_save=False, is_debug=False,
             corrected_red=mat[0], corrected_green=mat[1], corrected_blue=mat[2])
    p.update(over)
    return p


def _sensor():
    return dict(bit_depth=12, hdr_bit_depth=12)


def _oracle(frame, params):
    return ColorCorrectionMatrixOptimized(
        frame.copy(), dict(PLATFORM), _sensor(), copy.deepcopy(params)
    ).compute_linear()


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("mat_name", list(MATRICES))
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ccm(tensor_device, shape, mat_name, dtype):
    rng = np.random.RandomState(9801)
    frames = (rng.rand(2, *shape, 3) * 2.5 - 0.5).astype(dtype)
    params = _params(MATRICES[mat_name])
    expected = [_oracle(f, params) for f in frames]
    import torch
    image = torch.as_tensor(frames, device=tensor_device)
    before = image.clone()
    p_before = copy.deepcopy(params)
    out = color_correction_matrix(image, _sensor(), params)
    assert out.dtype == torch.float32
    assert out.shape == frames.shape
    assert out.device == image.device
    np.testing.assert_array_equal(image.cpu().numpy(), before.cpu().numpy())
    assert params == p_before
    res = compare_arrays(np.stack(expected), out.cpu().numpy(),
                         FLOAT32_NORMALIZED, scale=1.0)
    assert res["passed"], (shape, mat_name, dtype, res)


@pytest.mark.parametrize("dtype", [np.uint16, np.uint32, np.int32])
def test_ccm_integer_input(tensor_device, dtype):
    rng = np.random.RandomState(9801)
    frames = rng.randint(0, 4096, size=(2, 8, 10, 3)).astype(dtype)
    params = _params(MATRICES["mix"])
    expected = [_oracle(f.astype(np.float64), params) for f in frames]
    import torch
    out = color_correction_matrix(
        torch.as_tensor(frames, device=tensor_device), _sensor(), params)
    res = compare_arrays(np.stack(expected), out.cpu().numpy(),
                         FLOAT32_NORMALIZED, scale=1.0)
    assert res["passed"], res


def test_ccm_f64_boundary(tensor_device):
    """f64 16777217*2.5 must not be lost before the product."""
    import torch
    frames = np.full((2, 8, 10, 3), 16777217.0, np.float64)
    params = _params([[2.5, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    expected = [_oracle(f, params) for f in frames]
    out = color_correction_matrix(
        torch.as_tensor(frames, device=tensor_device), _sensor(), params)
    np.testing.assert_array_equal(out.cpu().numpy(), np.stack(expected))


def test_ccm_disabled(tensor_device):
    import torch
    rng = np.random.RandomState(9801)
    frames = rng.rand(2, 8, 10, 3).astype(np.float32)
    params = dict(is_enable=False, is_save=False,
                  corrected_red="garbage", corrected_green=None, corrected_blue=1)
    out = color_correction_matrix(
        torch.as_tensor(frames, device=tensor_device), _sensor(), params)
    assert out.dtype == torch.float32
    np.testing.assert_array_equal(out.cpu().numpy(), frames)


def test_ccm_rejects(tensor_device):
    import torch
    good = torch.ones(2, 8, 10, 3, device=tensor_device)
    for mat in (
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        [[1.0, 0.0, float("nan")], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    ):
        with pytest.raises(ValueError):
            color_correction_matrix(good, _sensor(), _params(mat))
    bad = good.clone(); bad[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        color_correction_matrix(bad, _sensor(), _params(IDENTITY))
    with pytest.raises(ValueError):
        color_correction_matrix(torch.ones(2, 8, 10, 3, device=tensor_device) * 1e300,
                                _sensor(), _params(IDENTITY))
    with pytest.raises(ValueError):
        color_correction_matrix(
            torch.ones(2, 8, 10, 3, dtype=torch.float64, device=tensor_device) * 1e40,
            _sensor(), _params(IDENTITY))
    with pytest.raises(ValueError):
        color_correction_matrix(good, _sensor(), _params(
            [[1e40, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    with pytest.raises(GpuError):
        color_correction_matrix(torch.ones(8, 10, 3, device=tensor_device), _sensor(), _params(IDENTITY))
    with pytest.raises(GpuError):
        color_correction_matrix(torch.ones(0, 8, 10, 3, device=tensor_device), _sensor(), _params(IDENTITY))
    with pytest.raises(GpuError):
        color_correction_matrix(torch.ones(2, 8, 10, 3, dtype=torch.bool, device=tensor_device), _sensor(), _params(IDENTITY))
