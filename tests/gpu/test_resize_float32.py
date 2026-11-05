"""GPU-only INTER_AREA f32 downsample vs frozen refs and cv2 oracle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from brilliantisp_gpu.ops.resize_float32 import area_float32
from brilliantisp_gpu.types import GpuError

FIXTURE = np.load(Path(__file__).parent / "fixtures"
                  / "area_float32_reference.npz")

FIXTURE_CASES = sorted({k[:-6] for k in FIXTURE.files
                        if k.endswith("_input")})


@pytest.mark.parametrize("case", FIXTURE_CASES)
def test_area_frozen(cuda_device, case):
    import torch
    src = FIXTURE[case + "_input"]
    ref = FIXTURE[case + "_reference"]
    image = torch.as_tensor(src, device=cuda_device)
    before = image.clone()
    out = area_float32(image, ref.shape[-2], ref.shape[-1])
    np.testing.assert_array_equal(image.cpu().numpy(), before.cpu().numpy())
    assert out.dtype == torch.float32
    assert out.shape == ref.shape
    assert out.device == image.device
    np.testing.assert_array_equal(out.cpu().numpy(), ref)


def test_area_fullhd_3x3(cuda_device):
    import cv2
    import torch
    rng = np.random.RandomState(11101)
    src = (rng.rand(2, 1080, 1920) * 7.2 - 6).astype(np.float32)
    ref = np.stack([cv2.resize(f, (640, 360),
                               interpolation=cv2.INTER_AREA) for f in src])
    out = area_float32(torch.as_tensor(src, device=cuda_device), 360, 640)
    np.testing.assert_array_equal(out.cpu().numpy(), ref)


def test_area_noncontiguous(cuda_device):
    import cv2
    import torch
    rng = np.random.RandomState(11101)
    src = (rng.rand(7, 9) * 7.2 - 6).astype(np.float32)
    ref = cv2.resize(src, (4, 3), interpolation=cv2.INTER_AREA)
    image = torch.as_tensor(src.T.copy(), device=cuda_device).T
    assert not image.is_contiguous()
    out = area_float32(image, 3, 4)
    np.testing.assert_array_equal(out.cpu().numpy(), ref)


def test_area_leading_dims(cuda_device):
    import cv2
    import torch
    rng = np.random.RandomState(11101)
    src = (rng.rand(2, 2, 8, 10) * 7.2 - 6).astype(np.float32)
    ref = np.stack([cv2.resize(f, (5, 4), interpolation=cv2.INTER_AREA)
                    for f in src.reshape(-1, 8, 10)]).reshape(2, 2, 4, 5)
    out = area_float32(torch.as_tensor(src, device=cuda_device), 4, 5)
    assert out.shape == (2, 2, 4, 5)
    np.testing.assert_array_equal(out.cpu().numpy(), ref)


def test_area_rejects(cuda_device):
    import torch
    good = torch.ones(2, 8, 10, device=cuda_device)
    for bad_h in (0, -1, 4.5, True, None):
        with pytest.raises((ValueError, GpuError)):
            area_float32(good, bad_h, 4)
    with pytest.raises(ValueError):
        area_float32(good, 9, 4)
    with pytest.raises(ValueError):
        area_float32(good, 4, 11)
    with pytest.raises(GpuError):
        area_float32(torch.ones(8, 10), 4, 5)  # cpu tensor
    with pytest.raises(GpuError):
        area_float32(torch.ones(8, 10, dtype=torch.float64,
                                device=cuda_device), 4, 5)
    with pytest.raises(GpuError):
        area_float32(torch.full((8, 10), float("nan"),
                                device=cuda_device), 4, 5)
    with pytest.raises(GpuError):
        area_float32(torch.empty(0, 4, device=cuda_device), 1, 2)


def test_area_coexists_reference_sort(cuda_device):
    """Custom-func area module and default-name sort module coexist."""
    import torch
    from brilliantisp_gpu.ops.reference_sort import reference_argsort
    src = FIXTURE["h17_w19_oh9_ow10_random_input"]
    ref = FIXTURE["h17_w19_oh9_ow10_random_reference"]
    keys = np.arange(97, dtype=np.float64)[::-1].copy()
    expected = np.arange(96, -1, -1, dtype=np.int64)
    for _ in range(2):
        out = area_float32(torch.as_tensor(src, device=cuda_device),
                           ref.shape[-2], ref.shape[-1])
        np.testing.assert_array_equal(out.cpu().numpy(), ref)
        idx = reference_argsort(torch.as_tensor(keys, device=cuda_device))
        np.testing.assert_array_equal(idx.cpu().numpy(), expected)


def test_area_nondefault_stream(cuda_device):
    """Frozen generic case on a custom stream with allocator churn."""
    import torch
    src = FIXTURE["h17_w19_oh9_ow10_random_input"]
    ref = FIXTURE["h17_w19_oh9_ow10_random_reference"]
    s = torch.cuda.Stream(device=cuda_device)
    with torch.cuda.stream(s):
        image = torch.as_tensor(src, device=cuda_device)
        out = area_float32(image, ref.shape[-2], ref.shape[-1])
        churn = [torch.rand(4096, device=cuda_device) for _ in range(8)]
        del churn
        got = out.cpu().numpy()
    s.synchronize()
    np.testing.assert_array_equal(got, ref)
