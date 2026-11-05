"""Emulated reference log10/mantissa primitives vs frozen fixture."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from brilliantisp_gpu.ops.reference_math import (_logf_mantissa,
                                               reference_log10_float32)

FIXTURE = np.load(Path(__file__).parent / "fixtures" / "reference_log10_float32.npz")


def test_logf_mantissa_exact(tensor_device):
    import torch
    u = torch.as_tensor(FIXTURE["mantissa"], device=tensor_device)
    before = u.clone()
    out = _logf_mantissa(u)
    np.testing.assert_array_equal(u.cpu().numpy(), before.cpu().numpy())
    assert out.dtype == torch.float32 and out.shape == u.shape
    assert out.device == u.device
    np.testing.assert_array_equal(out.cpu().numpy(),
                                  FIXTURE["natural_libc"])


def test_reference_log10_exact(tensor_device):
    import torch
    x = torch.as_tensor(FIXTURE["source"], device=tensor_device)
    before = x.clone()
    out = reference_log10_float32(x)
    np.testing.assert_array_equal(x.cpu().numpy(), before.cpu().numpy())
    assert out.dtype == torch.float32 and out.shape == x.shape
    assert out.device == x.device
    np.testing.assert_array_equal(out.cpu().numpy(), FIXTURE["reference"])


def _boundaries():
    f32 = np.float32
    pts = [f32(1e-6),
           np.nextafter(f32(1e-6), f32(-np.inf)),
           np.nextafter(f32(1e-6), f32(np.inf)),
           f32(0.5), np.nextafter(f32(0.5), f32(-np.inf)),
           np.nextafter(f32(0.5), f32(np.inf)),
           f32(1.0), np.nextafter(f32(1.0), f32(-np.inf)),
           np.nextafter(f32(1.0), f32(np.inf)),
           f32(2.0), np.nextafter(f32(2.0), f32(-np.inf)),
           np.nextafter(f32(2.0), f32(np.inf)),
           np.float32(2.0 ** -126),  # smallest normal
           np.nextafter(np.float32(2.0 ** -126), f32(np.inf)),
           np.finfo(np.float32).max,
           np.nextafter(np.finfo(np.float32).max, f32(-np.inf))]
    return np.asarray(pts, np.float32)


def test_reference_log10_boundaries(tensor_device):
    """Boundary points vs NumPy float32 log10 oracle — exact."""
    import torch
    pts = _boundaries()
    expected = np.log10(pts)
    x = torch.as_tensor(pts, device=tensor_device)
    out = reference_log10_float32(x)
    np.testing.assert_array_equal(out.cpu().numpy(), expected)


def test_reference_log10_rejects(tensor_device):
    import torch
    good = torch.ones(4, device=tensor_device)
    for bad in (torch.zeros(2, device=tensor_device),
                -torch.ones(2, device=tensor_device),
                torch.full((2,), np.finfo(np.float32).tiny / 2.0,
                           device=tensor_device),
                torch.full((2,), float("nan"), device=tensor_device),
                torch.full((2,), float("inf"), device=tensor_device),
                torch.empty(0, device=tensor_device),
                torch.ones(4, dtype=torch.float64, device=tensor_device),
                torch.ones(4, dtype=torch.int32, device=tensor_device),
                torch.ones(4, dtype=torch.bool, device=tensor_device)):
        with pytest.raises(ValueError):
            reference_log10_float32(bad)
    out = reference_log10_float32(good)
    assert out.dtype == torch.float32
