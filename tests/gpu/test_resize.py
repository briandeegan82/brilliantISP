"""Resize helpers vs the OpenCV oracle on small noninteger scales,
constants and impulses."""

from __future__ import annotations

import numpy as np
import pytest
import cv2

from tools.gpu_validation.contracts import FLOAT32_NORMALIZED

from brilliantisp_gpu.ops import resize


def _check(actual, expected, gate=FLOAT32_NORMALIZED):
    np.testing.assert_allclose(
        np.asarray(actual, np.float64), np.asarray(expected, np.float64),
        atol=gate.atol, rtol=gate.rtol,
    )


CASES = [
    ((5, 7), (3, 4)),
    ((5, 7), (9, 11)),
    ((4, 4), (7, 5)),
    ((8, 6), (4, 3)),
    ((3, 5), (8, 8)),
    ((6, 9), (6, 9)),
]

INTERP = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "area": cv2.INTER_AREA,
}


def _images():
    rng = np.random.RandomState(0)
    yield "random", lambda h, w: rng.randn(h, w)
    yield "constant", lambda h, w: np.full((h, w), 3.25)
    def _impulse(h, w):
        x = np.zeros((h, w))
        x[h // 2, w // 2] = 1.0
        x[0, 0] = -2.0
        return x
    yield "impulse", _impulse


@pytest.mark.parametrize("shape,out", CASES)
@pytest.mark.parametrize("name,interp", list(INTERP.items()))
@pytest.mark.parametrize("img_name,img_fn", list(_images()))
def test_resize_matches_opencv(tensor_device, shape, out, name, interp, img_name, img_fn):
    torch = pytest.importorskip("torch")
    x = img_fn(*shape)
    expected = cv2.resize(x, (out[1], out[0]), interpolation=interp)
    fn = getattr(resize, name)
    actual = fn(torch.as_tensor(x, device=tensor_device), out[0], out[1]).cpu().numpy()
    _check(actual, expected)


@pytest.mark.parametrize("shape,out", CASES[:3])
def test_resize_batch(tensor_device, shape, out):
    torch = pytest.importorskip("torch")
    rng = np.random.RandomState(7)
    x = rng.randn(2, *shape)
    expected = np.stack([
        cv2.resize(x[b], (out[1], out[0]), interpolation=cv2.INTER_LINEAR)
        for b in range(2)
    ])
    actual = resize.bilinear(torch.as_tensor(x, device=tensor_device), out[0], out[1]).cpu().numpy()
    _check(actual, expected)


def test_resize_bad_shape_raises(tensor_device):
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError):
        resize.bilinear(torch.ones(4, 4, device=tensor_device), 0, 4)
    with pytest.raises(ValueError):
        resize.nearest(torch.ones(4, 4, device=tensor_device), 4, -2)
    with pytest.raises(ValueError):
        resize.area(torch.empty(0, 4, device=tensor_device), 2, 2)
    with pytest.raises(ValueError):
        resize.bilinear(torch.ones(5, device=tensor_device), 3, 3)


def test_area_realistic_downscale_bounded(tensor_device):
    """Full-HD -> VGA area downscale must run without a dense weight matrix."""
    torch = pytest.importorskip("torch")
    x = np.random.RandomState(41).randn(2, 1080, 1920)
    actual = resize.area(torch.as_tensor(x, device=tensor_device), 360, 640).cpu().numpy()
    for b in range(2):
        expected = cv2.resize(x[b], (640, 360), interpolation=cv2.INTER_AREA)
        _check(actual[b], expected)
    actual_small = resize.area(torch.as_tensor(x, device=tensor_device), 8, 8)
    assert actual_small.shape == (2, 8, 8)


def test_nearest_reciprocal_ratio_14_to_18(tensor_device):
    """cv2 INTER_NEAREST reciprocal-ratio index order: 14 -> 18, exact."""
    import torch
    src = np.arange(14 * 14, dtype=np.float64).reshape(14, 14)
    expected = cv2.resize(src.astype(np.float32), (18, 18),
                          interpolation=cv2.INTER_NEAREST)
    image = torch.as_tensor(src.astype(np.float32), device=tensor_device)
    before = image.clone()
    out = resize.nearest(image, 18, 18)
    np.testing.assert_array_equal(image.cpu().numpy(), before.cpu().numpy())
    assert out.dtype == torch.float32
    assert out.shape == expected.shape
    assert out.device == image.device
    np.testing.assert_array_equal(out.cpu().numpy(), expected)
