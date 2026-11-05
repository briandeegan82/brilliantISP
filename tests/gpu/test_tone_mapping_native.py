"""GPU map_native vs modules.tone_mapping.native.map_native (seed 10001)."""

from __future__ import annotations

import numpy as np
import pytest

from brilliantisp_gpu.blocks.tone_mapping import map_native, _map_fixed_plane_f64
from brilliantisp_gpu.types import GpuError
from modules.tone_mapping.native import map_native as cpu_map_native, _map_plane
from modules.tone_mapping.normalization import FrameStats
from modules.tone_mapping.registry import ToneMappingSpec, ToneMappingError
from tools.gpu_validation.metrics import compare_arrays
from tools.gpu_validation.contracts import FLOAT32_NORMALIZED, FLOAT64_HELPER


def _spec(**kw):
    base = dict(
        operator="reinhard", backend="float", normalization="declared_range",
        exposure_ev=0.0, domain_scale=1.0, normalize_output=False,
        parameters=(), lut_domain=16.0, enabled=True,
    )
    base.update(kw)
    return ToneMappingSpec(**base)


def _t(a, d):
    import torch
    return torch.as_tensor(np.asarray(a), device=d)


def _check(exp, act, tag):
    res = compare_arrays(np.asarray(exp), np.asarray(act),
                         FLOAT32_NORMALIZED, scale=1.0)
    assert res["passed"], (tag, res)


def _run(img, model, spec, device, stats=None):
    import torch
    image = _t(img, device)
    before = image.clone()
    out = map_native(image, config=spec, color_model=model, frame_stats=stats)
    np.testing.assert_array_equal(image.cpu().numpy(), before.cpu().numpy())
    assert out.dtype == torch.float32
    assert out.shape == image.shape
    assert out.device == image.device
    return out.cpu().numpy()


SPEC_GRID = [
    _spec(operator=o, backend=b, normalization=n, normalize_output=norm_out)
    for o, b in (("aces_fitted", "float"), ("aces_fitted", "lut"),
                 ("hable", "float"), ("hable", "lut"),
                 ("reinhard", "float"), ("reinhard", "lut"),
                 ("reinhard", "fixed"))
    for n in ("declared_range", "per_frame")
    for norm_out in (False, True)
] + [
    _spec(operator="reinhard", backend="float", exposure_ev=-2.0),
    _spec(operator="reinhard", backend="float", exposure_ev=2.0),
    _spec(operator="reinhard", backend="float", domain_scale=0.5),
    _spec(operator="aces_fitted", backend="lut", lut_size=256),
    _spec(operator="reinhard", backend="lut", lut_size=65536,
          parameters=(("knee", 0.4), ("strength", 1.3))),
    _spec(operator="hable", backend="float",
          parameters=(("exposure_bias", 1.5), ("white_point", 8.0))),
]


@pytest.mark.parametrize("spec", SPEC_GRID, ids=[f"s{i}" for i in range(len(SPEC_GRID))])
@pytest.mark.parametrize("shape", [(7, 9), (8, 10)])
@pytest.mark.parametrize("in_dt", [np.float32, np.float64])
def test_native_bayer(tensor_device, spec, shape, in_dt):
    rng = np.random.RandomState(10001)
    img = (rng.rand(2, *shape) * 1.5 - 0.1).astype(in_dt)
    expected = np.stack([cpu_map_native(f, config=spec) for f in img])
    out = _run(img, "bayer", spec, tensor_device)
    _check(expected, out, (spec, shape, in_dt))


@pytest.mark.parametrize("spec", SPEC_GRID, ids=[f"s{i}" for i in range(len(SPEC_GRID))])
@pytest.mark.parametrize("in_dt", [np.float32, np.float64])
def test_native_rgb(tensor_device, spec, in_dt):
    rng = np.random.RandomState(10001)
    img = (rng.rand(2, 7, 9, 3) * 1.6 - 0.2).astype(in_dt)
    expected = np.stack([cpu_map_native(f, config=spec) for f in img])
    out = _run(img, "rgb", spec, tensor_device)
    _check(expected, out, (spec, in_dt))


def test_native_unbatched(tensor_device):
    rng = np.random.RandomState(10001)
    spec = _spec()
    img = rng.rand(8, 10).astype(np.float32)
    _check(cpu_map_native(img, config=spec), _run(img, "bayer", spec, tensor_device), "unbatched_bayer")
    img3 = rng.rand(8, 10, 3).astype(np.float32)
    _check(cpu_map_native(img3, config=spec), _run(img3, "rgb", spec, tensor_device), "unbatched_rgb")


@pytest.mark.parametrize("spec", [_spec(normalization="per_frame"),
                                 _spec(normalization="per_frame", operator="aces_fitted")])
def test_native_mixed_batch_stats(tensor_device, spec):
    rng = np.random.RandomState(10001)
    img = np.empty((2, 8, 10), np.float32)
    img[0] = rng.rand(8, 10).astype(np.float32)
    img[1] = 0.4
    expected = np.stack([cpu_map_native(f, config=spec) for f in img])
    out = _run(img, "bayer", spec, tensor_device)
    _check(expected, out, spec)


def test_native_supplied_stats(tensor_device):
    rng = np.random.RandomState(10001)
    img = rng.rand(2, 8, 10).astype(np.float32)
    spec = _spec(normalization="per_frame")
    stats = FrameStats(0.1, 0.9)
    expected = np.stack([cpu_map_native(f, config=spec, frame_stats=stats) for f in img])
    out = _run(img, "bayer", spec, tensor_device, stats=stats)
    _check(expected, out, "stats")
    seq = [FrameStats(0.0, 1.0), FrameStats(0.2, 0.8)]
    expected = np.stack([cpu_map_native(img[i], config=spec, frame_stats=seq[i]) for i in range(2)])
    out = _run(img, "bayer", spec, tensor_device, stats=seq)
    _check(expected, out, "stats_seq")


def test_native_tiny(tensor_device):
    rng = np.random.RandomState(10001)
    spec = _spec()
    for shape in [(1, 1), (1, 3), (3, 1)]:
        img = rng.rand(1, *shape).astype(np.float32)
        expected = np.stack([cpu_map_native(f, config=spec) for f in img])
        out = _run(img, "bayer", spec, tensor_device)
        _check(expected, out, shape)


def test_native_black_and_negative(tensor_device):
    spec = _spec()
    img = np.zeros((2, 8, 10, 3), np.float64)
    img[:, 1::2] = np.array([0.3, 0.3, 0.3])
    img[0, ::2, ::3] = np.array([-0.2, 0.5, 0.5])
    expected = np.stack([cpu_map_native(f, config=spec) for f in img])
    out = _run(img, "rgb", spec, tensor_device)
    _check(expected, out, "black_neg")
    neg = np.full((1, 8, 10, 3), 0.0, np.float64)
    neg[..., 0] = -0.5; neg[..., 1] = 0.8; neg[..., 2] = 0.8
    expected = cpu_map_native(neg[0], config=spec)
    out = _run(neg, "rgb", spec, tensor_device)
    _check(expected[None], out, "neg_comp")


def test_native_y_floor(tensor_device):
    spec = _spec()
    kr = 0.2126
    for val in (1e-13, 1e-12, np.nextafter(1e-12, np.inf), 1e-11):
        img = np.full((1, 2, 2, 3), 0.0, np.float64)
        img[..., 0] = val / kr
        expected = cpu_map_native(img[0], config=spec)
        out = _run(img, "rgb", spec, tensor_device)
        _check(expected[None], out, val)


def test_native_lut_table_once(tensor_device, monkeypatch):
    """LUT prepared once per map_native call, reused across frames."""
    torch = pytest.importorskip("torch")
    import modules.tone_mapping.backends as cb
    rng = np.random.RandomState(10001)
    spec = _spec(operator="aces_fitted", backend="lut", lut_size=256)
    img = rng.rand(2, 8, 10).astype(np.float32)
    expected = np.stack([cpu_map_native(f, config=spec) for f in img])
    from brilliantisp_gpu.blocks import tone_mapping as gm
    calls = []
    real_get = cb._LUT_CACHE.get

    def spy(spec_arg, out_max):
        calls.append(1)
        return real_get(spec_arg, out_max)

    monkeypatch.setattr(gm._LUT_CACHE, "get", spy)
    out = map_native(_t(img, tensor_device), config=spec, color_model="bayer")
    assert len(calls) == 1
    _check(expected, out.cpu().numpy(), "lut_once")


def test_native_disabled_same_object(tensor_device):
    spec = _spec(enabled=False, operator="nonsense")
    import torch
    img = _t(np.zeros((2, 8, 10), np.float32), tensor_device)
    assert map_native(img, config=spec, color_model="bayer") is img
    rgb = _t(np.zeros((8, 10, 3), np.float32), tensor_device)
    assert map_native(rgb, config=spec, color_model="weird") is rgb
    assert map_native(img, config=spec, color_model="bayer",
                    frame_stats=[42, None]) is img


def test_native_rejects(tensor_device):
    import torch
    spec = _spec()
    img = np.ones((2, 8, 10), np.float32)
    with pytest.raises(ToneMappingError):
        map_native(_t(img, tensor_device), config=spec, color_model="bayer",
                   frame_stats=[None])
    with pytest.raises(ValueError):
        map_native(_t(img, tensor_device), config=spec, color_model="weird")
    with pytest.raises(ToneMappingError):
        map_native(_t(np.zeros((0, 8, 10)), tensor_device), config=spec, color_model="bayer")
    with pytest.raises(ToneMappingError):
        map_native(_t(np.ones((2, 8, 10), np.int32), tensor_device), config=spec, color_model="bayer")
    with pytest.raises(ToneMappingError):
        map_native(_t(np.full((2, 8, 10), np.nan), tensor_device), config=spec, color_model="bayer")
    with pytest.raises(ToneMappingError):
        map_native(_t(img, tensor_device), config=spec, color_model="rgb")
    with pytest.raises(ToneMappingError):
        map_native(_t(img, tensor_device), config=spec, color_model="bayer",
                   frame_stats=[42, None])
    with pytest.raises(ToneMappingError):
        map_native(_t(img, tensor_device), config=spec, color_model="bayer",
                   frame_stats=[None, FrameStats(0, 1), FrameStats(0, 1)])


def test_native_stats_span_overflow(tensor_device):
    """FrameStats(-1e308, 1e308) -> span inf -> ValueError like CPU."""
    img = np.full((1, 8, 10), 0.5, np.float32)
    spec = _spec(normalization="per_frame")
    with pytest.raises(ValueError):
        cpu_map_native(img[0], config=spec, frame_stats=FrameStats(-1e308, 1e308))
    with pytest.raises(ValueError):
        map_native(_t(img, tensor_device), config=spec, color_model="bayer",
                   frame_stats=FrameStats(-1e308, 1e308))


def test_native_threshold_underflow(tensor_device):
    """lut_domain/gain -> 0 is unreachable: registry rejects gain/lut_domain
    beyond float64 (same named error as CPU)."""
    with pytest.raises(ToneMappingError):
        _spec(operator="reinhard", backend="float", domain_scale=1e300,
              lut_domain=1e-300, normalize_output=True)


def test_native_huge_rgb_finite(tensor_device):
    """1e300 headroom RGB maps finite, not spurious overflow."""
    spec = _spec(operator="reinhard", backend="float")
    img = np.full((1, 4, 4, 3), 1e300, np.float64)
    expected = cpu_map_native(img[0], config=spec)
    assert np.all(np.isfinite(expected))
    out = _run(img, "rgb", spec, tensor_device)
    _check(expected[None], out, "huge_rgb")


def test_fixed_f64_core_vs_cpu(tensor_device):
    torch = pytest.importorskip("torch")
    rng = np.random.RandomState(10001)
    y = np.maximum((rng.rand(8, 10) * 1.5 - 0.1), 0.0)
    for spec, stats, tag in (
        (_spec(backend="fixed"), None, "declared"),
        (_spec(backend="fixed", normalization="per_frame"), None, "per_frame"),
        (_spec(backend="fixed", normalization="per_frame", normalize_output=True),
         FrameStats(0.1, 0.9), "fractional_stats_norm_out"),
    ):
        expected = np.asarray(_map_plane(y.copy(), spec, stats), np.float64)
        yt = torch.as_tensor(y, device=tensor_device)
        out = _map_fixed_plane_f64(yt, spec, stats)
        assert out.dtype == torch.float64
        res = compare_arrays(expected, out.cpu().numpy(), FLOAT64_HELPER, scale=1.0)
        assert res["passed"], (tag, res)


def test_fixed_coordinate_overflow(tensor_device):
    torch = pytest.importorskip("torch")
    spec = _spec(backend="fixed", lut_domain=1024.0)
    huge = np.full((1, 2, 2), 400.0, np.float64)
    with pytest.raises(ToneMappingError):
        map_native(_t(huge, tensor_device), config=spec, color_model="bayer")
    big = np.full((1, 2, 2), 1e300, np.float64)
    with pytest.raises(ToneMappingError):
        map_native(_t(big, tensor_device), config=spec, color_model="bayer")
