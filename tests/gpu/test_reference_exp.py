from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from brilliantisp_gpu.ops.reference_exp import reference_exp_float32
from brilliantisp_gpu.blocks.bayer_noise_reduction import bayer_noise_reduction


FIXTURES = Path(__file__).parent / "fixtures"


def _assert_bits(expected, actual):
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    finite_or_inf = ~np.isnan(expected)
    np.testing.assert_array_equal(
        actual[finite_or_inf].view(np.uint32),
        expected[finite_or_inf].view(np.uint32),
    )


def test_reference_exp_frozen_numpy_profile(tensor_device):
    import torch

    with np.load(FIXTURES / "reference_exp_float32.npz") as data:
        source = torch.as_tensor(data["inputs"], device=tensor_device)
        before = source.clone()
        actual = reference_exp_float32(source)
        assert actual.device == source.device and actual.dtype == torch.float32
        _assert_bits(data["expected"], actual.cpu().numpy())
        torch.testing.assert_close(source, before, rtol=0, atol=0, equal_nan=True)


def test_reference_exp_preserves_strided_shape(tensor_device):
    import torch

    with np.load(FIXTURES / "reference_exp_float32.npz") as data:
        source = torch.as_tensor(
            data["inputs"][:512].reshape(16, 32), device=tensor_device,
        ).T
        assert not source.is_contiguous()
        actual = reference_exp_float32(source)
        assert actual.device == source.device and actual.shape == source.shape
        _assert_bits(data["expected"][:512].reshape(16, 32).T, actual.cpu().numpy())


@pytest.mark.parametrize("invalid", ("dtype", "empty", "not_tensor"))
def test_reference_exp_rejects_invalid_inputs(tensor_device, invalid):
    import torch

    if invalid == "dtype":
        value = torch.ones(2, dtype=torch.float64, device=tensor_device)
    elif invalid == "empty":
        value = torch.empty(0, dtype=torch.float32, device=tensor_device)
    else:
        value = [1.0, 2.0]
    with pytest.raises((TypeError, ValueError)):
        reference_exp_float32(value)


@pytest.mark.parametrize("ae", (0, 1))
def test_real_bnr_exp_regression_frozen_cpu_interior(tensor_device, ae):
    import torch

    metadata = json.loads((FIXTURES / "reference_exp_float32.json").read_text())
    info = metadata["patch_provenance"]
    with np.load(FIXTURES / "bnr_exp_real_patch.npz") as data:
        source = torch.as_tensor(
            data[f"lens_shading_correction_ae{ae}"][None], device=tensor_device,
        )
        actual = bayer_noise_reduction(source, info["sensor"], info["params"])
        assert actual.device == source.device and actual.dtype == torch.float32
        np.testing.assert_array_equal(
            actual[0, 8:-8, 8:-8].cpu().numpy(),
            data[f"bayer_noise_reduction_ae{ae}"][8:-8, 8:-8],
        )
