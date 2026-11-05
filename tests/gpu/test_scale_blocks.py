"""GPU scale vs Scale(cpu_only).compute_linear oracle (seed 10301)."""

from __future__ import annotations

import numpy as np
import pytest

from brilliantisp_gpu.blocks.scaling import scale
from modules.scale.scale import Scale
from tools.gpu_validation.metrics import compare_arrays
from tools.gpu_validation.contracts import FLOAT32_NORMALIZED

FIXTURE = np.load("tests/gpu/fixtures/scale_ipp_linear.npz")
TAIL_FIXTURE = np.load("tests/gpu/fixtures/scale_ipp_tail_geometry.npz")


def _t(a, d):
    import torch
    return torch.as_tensor(np.asarray(a), device=d)


def _params(**kw):
    p = dict(is_enable=True, is_save=False, is_debug=False,
             new_height=8, new_width=10, is_hardware=False,
             algorithm="Bilinear", upscale_method="Bilinear",
             downscale_method="Bilinear")
    p.update(kw)
    return p


def _oracle(arr, sensor, params):
    if arr.ndim == 4:
        return np.stack([_oracle(f, sensor, params.copy()) for f in arr])
    return Scale(arr, {}, sensor, params.copy(), 3, cpu_only=True).compute_linear()


def _sensor(h, w):
    return {"height": h, "width": w, "bit_depth": 12, "bayer_pattern": "rggb"}


def _check(exp, act, tag):
    res = compare_arrays(np.asarray(exp), np.asarray(act),
                         FLOAT32_NORMALIZED, scale=1.0)
    assert res["passed"], (tag, res)


def _run(arr, device, sensor, params):
    import torch
    image = _t(arr, device)
    before = image.clone()
    out = scale(image, sensor, params)
    np.testing.assert_array_equal(image.cpu().numpy(), before.cpu().numpy())
    assert out.dtype == torch.float32
    assert out.device == image.device
    exp = _oracle(np.asarray(arr), sensor, params.copy())
    _check(exp, out.cpu().numpy(), (np.asarray(arr).shape, params))
    return out


@pytest.mark.parametrize("batched", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.uint16])
@pytest.mark.parametrize("size", [(8, 10, 4, 5), (7, 9, 14, 18), (12, 16, 6, 8),
                                  (9, 11, 18, 22), (8, 10, 8, 10)])
@pytest.mark.parametrize("algo", ["Bilinear", "Nearest_Neighbor", "bogus"])
def test_nonhardware(tensor_device, batched, dtype, size, algo):
    rng = np.random.RandomState(10301)
    h, w, nh, nw = size
    dims = (2, h, w, 3) if batched else (h, w, 3)
    a = (rng.rand(*dims) * 1.7 - 0.2).astype(dtype)
    _run(a, tensor_device, _sensor(h, w),
         _params(new_height=nh, new_width=nw, algorithm=algo))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("geom", [(9, 11, 7, 13), (12, 16, 10, 24)])
def test_signed_hdr(tensor_device, dtype, geom):
    h, w, nh, nw = geom
    rng = np.random.RandomState(10301)
    a = (rng.rand(2, h, w, 3) * 128 - 64).astype(dtype)
    _run(a, tensor_device, _sensor(h, w),
         _params(new_height=nh, new_width=nw))
    a2 = rng.rand(2, 8, 10, 3).astype(dtype) * 1e-6 - 3e-7
    _run(a2, tensor_device, _sensor(8, 10),
         _params(new_height=12, new_width=8))


def test_frozen_ipp_linear(tensor_device):
    src = np.repeat(FIXTURE["source"][:, :, None], 3, axis=2)
    exp = np.repeat(FIXTURE["reference"][:, :, None], 3, axis=2)
    out = scale(_t(src, tensor_device), _sensor(7, 2581),
                _params(new_height=7, new_width=1920, algorithm="Bilinear"))
    np.testing.assert_array_equal(out.cpu().numpy(), exp.astype(np.float32))


def test_nn_index_regression(tensor_device):
    """cv2 reciprocal-ratio index order: 14 -> 18 exact."""
    rng = np.random.RandomState(10301)
    a = rng.rand(14, 14, 3).astype(np.float32)
    exp = _oracle(a, _sensor(14, 14),
                  _params(new_height=18, new_width=18,
                          algorithm="Nearest_Neighbor"))
    out = scale(_t(a, tensor_device), _sensor(14, 14),
                _params(new_height=18, new_width=18,
                        algorithm="Nearest_Neighbor"))
    np.testing.assert_array_equal(out.cpu().numpy(), exp)


def test_disabled(tensor_device):
    a = np.random.RandomState(10301).rand(2, 8, 10, 3).astype(np.float32)
    out = scale(_t(a, tensor_device), _sensor(0, 0),
                _params(is_enable=False, new_height=None))
    np.testing.assert_array_equal(out.cpu().numpy(), a)


def test_validation(tensor_device):
    import torch
    good = _t(np.ones((2, 8, 10, 3), np.float32), tensor_device)
    s = _sensor(8, 10)
    for bad in (0, -1, 4.5, float("nan"), True, None):
        with pytest.raises(ValueError):
            scale(good, s, _params(new_height=bad))
        with pytest.raises(ValueError):
            scale(good, s, _params(new_width=bad))
    with pytest.raises(ValueError):
        scale(good, _sensor(9, 10), _params())


def test_rejects(tensor_device):
    import torch
    from brilliantisp_gpu.types import GpuError
    good = _t(np.ones((2, 8, 10, 3), np.float32), tensor_device)
    s = _sensor(8, 10)
    with pytest.raises(ValueError):
        scale(torch.ones(8, 10, device=good.device), s, _params())
    with pytest.raises(ValueError):
        scale(_t(np.full((8, 10, 3), np.nan), tensor_device), s, _params())
    with pytest.raises(ValueError):
        scale(_t(np.full((8, 10, 3), 1e40, np.float64), tensor_device),
              s, _params())


# Hardware recipes: all 13 (old,new) pairs in HARDWARE_RECIPES.
from brilliantisp_gpu.ops.scale_geometry import HARDWARE_RECIPES

HW_CASES = [(old, new) for old, table in HARDWARE_RECIPES.items()
            for new in table
            if (old, new) not in (((1944, 2592), (960, 1280)),
                                  ((1944, 2592), (720, 1280)))]


@pytest.mark.parametrize("old,new", HW_CASES)
def test_hardware_pairs(tensor_device, old, new):
    """B1 deterministic structured frame vs CPU oracle; one case at a time."""
    h, w = old
    yy, xx = np.mgrid[0:h, 0:w]
    a = (((yy % 7) * 31 + (xx % 11) * 17 + yy.astype(np.int64) * xx % 1000)
         % 4096).astype(np.float32)[:, :, None].repeat(3, axis=2)
    params = _params(new_height=new[0], new_width=new[1], is_hardware=True)
    exp = _oracle(a, _sensor(h, w), params.copy())
    out = scale(_t(a, tensor_device), _sensor(h, w), params)
    _check(exp, out.cpu().numpy(), (old, new))


ERROR_PAIRS = [((1944, 2592), (960, 1280)), ((1944, 2592), (720, 1280))]


@pytest.mark.parametrize("old,new", ERROR_PAIRS)
def test_hardware_cpu_error_pairs(tensor_device, old, new):
    h, w = old
    a = np.zeros((h, w, 3), np.float32)
    params = _params(new_height=new[0], new_width=new[1], is_hardware=True)
    with pytest.raises(ValueError):
        _oracle(a, _sensor(h, w), params.copy())
    with pytest.raises(ValueError):
        scale(_t(a, tensor_device), _sensor(h, w), params)


def test_hardware_unsupported(tensor_device):
    a = np.ones((8, 10, 3), np.float32)
    p_same = _params(new_height=8, new_width=10, is_hardware=True)
    exp = _oracle(a, _sensor(8, 10), p_same.copy())
    out = scale(_t(a, tensor_device), _sensor(8, 10), p_same)
    _check(exp, out.cpu().numpy(), "hw identity")
    p_diff = _params(new_height=4, new_width=5, is_hardware=True)
    with pytest.raises(ValueError):
        scale(_t(a, tensor_device), _sensor(8, 10), p_diff)


TAIL_CASES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]


@pytest.mark.parametrize("q", TAIL_CASES)
def test_frozen_tail_geometry(tensor_device, q):
    """Frozen IPP scalar-tail captures; exact equality vs reference."""
    src = TAIL_FIXTURE[f"{q}_source"]
    ref = TAIL_FIXTURE[f"{q}_reference"]
    h, w = src.shape[1], src.shape[2]
    nh, nw = ref.shape[1], ref.shape[2]
    out = scale(_t(src, tensor_device), _sensor(h, w),
                _params(new_height=nh, new_width=nw, algorithm="Bilinear"))
    assert out.shape == ref.shape
    np.testing.assert_array_equal(out.cpu().numpy(), ref)


@pytest.mark.parametrize("up,down", [
    ("Nearest_Neighbor", "Nearest_Neighbor"),
    ("Bilinear", "Nearest_Neighbor"),
    ("Nearest_Neighbor", "Bilinear"),
    ("bogus", "bogus"),
])
def test_hardware_method_pairs(tensor_device, up, down):
    """upscale/downscale methods on 1080x1920 -> 720x1280 rational recipe."""
    h, w = 1080, 1920
    yy, xx = np.mgrid[0:h, 0:w]
    a = (((yy % 7) * 31 + (xx % 11) * 17 + yy.astype(np.int64) * xx % 1000)
         % 4096).astype(np.float32)[:, :, None].repeat(3, axis=2)
    params = _params(new_height=720, new_width=1280, is_hardware=True,
                     upscale_method=up, downscale_method=down)
    _run(a, tensor_device, _sensor(h, w), params)


def test_config_immutability(tensor_device):
    import copy
    rng = np.random.RandomState(10301)
    a = rng.rand(2, 8, 10, 3).astype(np.float32)
    sensor = _sensor(8, 10)
    params = _params(new_height=12, new_width=6)
    s0, p0 = copy.deepcopy(sensor), copy.deepcopy(params)
    scale(_t(a, tensor_device), sensor, params)
    assert sensor == s0 and params == p0


@pytest.mark.parametrize("geom", [(1, 1, 7, 9), (1, 7, 9, 11), (7, 1, 11, 9)])
def test_thin_inputs(tensor_device, geom):
    """Width-1/height-1 inputs via generic f32 OpenCV path."""
    h, w, nh, nw = geom
    rng = np.random.RandomState(10301)
    a = (rng.rand(2, h, w, 3) * 128 - 64).astype(np.float32)
    _run(a, tensor_device, _sensor(h, w),
         _params(new_height=nh, new_width=nw))
