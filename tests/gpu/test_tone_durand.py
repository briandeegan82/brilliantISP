"""GPU Durand plane + map_native durand vs frozen refs & CPU oracle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from brilliantisp_gpu.blocks.durand import _durand_plane, _gaussian_float32
from brilliantisp_gpu.blocks.tone_mapping import map_native
from modules.tone_mapping.native import map_native as cpu_map_native
from modules.tone_mapping.normalization import FrameStats
from modules.tone_mapping.registry import ToneMappingSpec, ToneMappingError
from tools.gpu_validation.metrics import compare_arrays
from tools.gpu_validation.contracts import FLOAT32_NORMALIZED

FIXTURE = np.load(Path(__file__).parent / "fixtures"
                  / "durand_reference.npz")

KINDS = [k[:-7] for k in FIXTURE.files if k.endswith("_source")]
FACTORS = (1, 2, 3, 4, 20)

PARAMS = [
    (.3, .4, 4, 1), (.3, .4, 4, 3),
    (2, .4, 2, 1), (2, .4, 2, 3),
    (5, .4, 4, 1), (5, .4, 4, 3),
    (1.2, .05, 4, 1), (1.2, .05, 4, 3),
    (1.2, 1.2, 4, 1), (1.2, 1.2, 4, 3),
    (1.2, .4, .5, 1), (1.2, .4, .5, 3),
    (1.2, .4, 10, 1), (1.2, .4, 10, 3),
]


def _t(a, d):
    import torch
    return torch.as_tensor(np.asarray(a), device=d)


def _check(exp, act, tag):
    res = compare_arrays(np.asarray(exp), np.asarray(act),
                         FLOAT32_NORMALIZED, scale=1.0)
    assert res["passed"], (tag, res)


def _spec(**kw):
    base = dict(
        operator="durand", backend="cpu", normalization="declared_range",
        exposure_ev=0.0, domain_scale=1.0, normalize_output=False,
        parameters=(), lut_domain=16.0, enabled=True,
    )
    base.update(kw)
    return ToneMappingSpec(**base)


def _run(img, model, spec, device, stats=None):
    import torch, copy
    image = _t(img, device)
    before = image.clone()
    s_before = copy.deepcopy(spec)
    out = map_native(image, config=spec, color_model=model,
                     frame_stats=stats)
    np.testing.assert_array_equal(image.cpu().numpy(), before.cpu().numpy())
    assert spec == s_before
    assert out.dtype == torch.float32
    assert out.shape == image.shape
    assert out.device == image.device
    return out.cpu().numpy()


def _oracle(img, model, spec, stats=None):
    def one(f, s):
        kw = {} if s is None else {"frame_stats": s}
        return cpu_map_native(f.copy(), config=spec, **kw)
    batched = (model == "bayer" and img.ndim == 3) or \
        (model == "rgb" and img.ndim == 4)
    frames = img if batched else img[None]
    seq = list(stats) if isinstance(stats, (list, tuple)) else [stats] * len(frames)
    outs = []
    for i, f in enumerate(frames):
        outs.append(one(f, seq[i]))
    return np.stack(outs) if batched else outs[0]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("factor", FACTORS)
def test_durand_plane_frozen_full(cuda_device, kind, factor):
    """40 frozen full-factor references (sigma 1.2/.4/4)."""
    import torch
    src = FIXTURE[f"{kind}_source"]
    ref = FIXTURE[f"full_factor{factor}_{kind}_reference"]
    image = _t(src, cuda_device)
    before = image.clone()
    out = _durand_plane(image, sigma_space=1.2, sigma_color=.4,
                        contrast_factor=4.0, downsample_factor=factor)
    np.testing.assert_array_equal(image.cpu().numpy(), before.cpu().numpy())
    assert out.dtype == torch.float64
    assert out.shape == src.shape and out.device == image.device
    _check(ref, out.cpu().numpy(), (kind, factor))


@pytest.mark.parametrize("pi", range(len(PARAMS)))
@pytest.mark.parametrize("kind", KINDS)
def test_durand_plane_frozen_params(cuda_device, pi, kind):
    """112 frozen parameter-tuple references."""
    import torch
    sigma, color, contrast, factor = PARAMS[pi]
    src = FIXTURE[f"{kind}_source"]
    ref = FIXTURE[f"params_p{pi}_{kind}_reference"]
    image = _t(src, cuda_device)
    out = _durand_plane(image, sigma_space=sigma, sigma_color=color,
                        contrast_factor=contrast,
                        downsample_factor=factor)
    assert out.dtype == torch.float64 and out.shape == src.shape
    _check(ref, out.cpu().numpy(), (pi, kind))


@pytest.mark.parametrize("model", ["bayer", "rgb"])
@pytest.mark.parametrize("norm", ["declared_range", "per_frame"])
@pytest.mark.parametrize("pi", range(len(PARAMS)))
def test_durand_native_oracle(cuda_device, model, norm, pi):
    """map_native durand vs CPU native per frame, 14 param tuples."""
    rng = np.random.RandomState(11201)
    sigma, color, contrast, factor = PARAMS[pi]
    spec = _spec(normalization=norm, parameters=(
        ("sigma_space", sigma), ("sigma_color", color),
        ("contrast_factor", contrast), ("downsample_factor", factor)))
    dims = (2, 7, 9) if model == "bayer" else (2, 7, 9, 3)
    img = (rng.rand(*dims) * 4 - 0.2).astype(np.float32)
    got = _run(img, model, spec, cuda_device)
    _check(_oracle(img, model, spec), got, (model, norm, pi))


@pytest.mark.parametrize("model", ["bayer", "rgb"])
@pytest.mark.parametrize("scene", ["zero", "constant", "dark", "headroom"])
@pytest.mark.parametrize("factor", [1.0, 3.0])
def test_durand_structured(cuda_device, model, scene, factor):
    """Structured scenes incl. constant fallback & dark levels."""
    spec = _spec(parameters=(("downsample_factor", factor),))
    dims = (2, 7, 9) if model == "bayer" else (2, 7, 9, 3)
    if scene == "zero":
        img = np.zeros(dims, np.float32)
    elif scene == "constant":
        img = np.full(dims, 0.3, np.float32)
    elif scene == "dark":
        img = (np.random.RandomState(11201).rand(*dims) * 1e-12
               ).astype(np.float32)
    else:
        img = (np.random.RandomState(11201).rand(*dims) * 8 - 2
               ).astype(np.float32)
    got = _run(img, model, spec, cuda_device)
    _check(_oracle(img, model, spec), got, (model, scene, factor))


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (7, 1), (2, 2), (7, 9)])
@pytest.mark.parametrize("model", ["bayer", "rgb"])
def test_durand_tiny(cuda_device, shape, model):
    rng = np.random.RandomState(11201)
    spec = _spec()
    dims = shape if model == "bayer" else shape + (3,)
    img = (rng.rand(*dims) * 2).astype(np.float32)
    got = _run(img, model, spec, cuda_device)
    _check(_oracle(img, model, spec), got, (shape, model))


def test_durand_f64_input(cuda_device):
    rng = np.random.RandomState(11201)
    spec = _spec()
    img = (rng.rand(2, 7, 9) * 16777217).astype(np.float64)
    img[0, 0, 0] = 16777217.0
    got = _run(img, "bayer", spec, cuda_device)
    _check(_oracle(img, "bayer", spec), got, "f64")


def test_durand_stats(cuda_device):
    rng = np.random.RandomState(11201)
    img = (rng.rand(2, 7, 9) * 4 - 0.2).astype(np.float32)
    spec = _spec(normalization="per_frame")
    got = _run(img, "bayer", spec, cuda_device,
               stats=FrameStats(0.1, 1.8))
    _check(_oracle(img, "bayer", spec,
                   stats=[FrameStats(0.1, 1.8)] * 2), got, "stats")
    stats = [None, FrameStats(0.1, 1.8)]
    got = _run(img, "bayer", spec, cuda_device, stats=stats)
    _check(_oracle(img, "bayer", spec, stats=stats), got, "stats-seq")


def test_durand_disabled(cuda_device):
    import torch
    img = np.ones((2, 7, 9), np.float32)
    spec = _spec(enabled=False)
    image = _t(img, cuda_device)
    assert map_native(image, config=spec, color_model="bayer") is image
    malformed = torch.ones(7, dtype=torch.complex64, device=cuda_device)
    assert map_native(malformed, config=spec, color_model="bayer") is malformed


def test_durand_rejects(cuda_device):
    import torch
    from brilliantisp_gpu.types import GpuError
    spec = _spec()
    with pytest.raises(ToneMappingError):
        map_native(_t(np.ones((2, 7, 9), np.int32), cuda_device),
                   config=spec, color_model="bayer")
    with pytest.raises(ToneMappingError):
        map_native(_t(np.full((2, 7, 9), np.nan), cuda_device),
                   config=spec, color_model="bayer")
    with pytest.raises(GpuError):
        map_native(torch.ones(2, 7, 9), config=spec, color_model="bayer")
    kw = dict(sigma_space=1.2, sigma_color=.4, contrast_factor=4.0,
              downsample_factor=1)
    with pytest.raises(GpuError):
        _durand_plane(torch.ones(7, 9), **kw)
    for bad in (torch.ones(2, 7, 9, device=cuda_device),
                torch.empty(0, 9, device=cuda_device),
                torch.ones(7, 9, dtype=torch.bool, device=cuda_device),
                torch.ones(7, 9, dtype=torch.int32, device=cuda_device),
                torch.ones(7, 9, dtype=torch.complex64,
                           device=cuda_device)):
        with pytest.raises((ToneMappingError, GpuError)):
            _durand_plane(bad, **kw)
    with pytest.raises(ToneMappingError):
        _durand_plane(torch.full((7, 9), 1e100, dtype=torch.float64,
                                 device=cuda_device), **kw)


@pytest.mark.parametrize("sigma", [0.0, 1e-300, 1e-15,
                                   np.nextafter(np.float64(1e-15),
                                                np.float64(np.inf)),
                                   .3, 1.2, 2.0, 5.0])
@pytest.mark.parametrize("scene", ["random", "constant", "dark"])
def test_durand_gaussian32(cuda_device, sigma, scene):
    """_gaussian_float32 vs scipy gaussian_filter (metadata oracle)."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.RandomState(11201)
    frame = {"random": rng.rand(7, 9) * 4,
             "constant": np.full((7, 9), 0.3),
             "dark": rng.rand(7, 9) * 1e-12}[scene].astype(np.float32)
    out = _gaussian_float32(_t(frame, cuda_device), sigma)
    ref = gaussian_filter(frame, sigma).astype(np.float32)
    _check(ref, out.cpu().numpy(), (sigma, scene))


@pytest.mark.parametrize("factor", [1.0, 3.0])
def test_durand_tiny_sigma_space(cuda_device, factor):
    """sigma_space/factor below gaussian threshold, whole-map oracle."""
    rng = np.random.RandomState(11201)
    spec = _spec(parameters=(("sigma_space", 1e-16),
                            ("downsample_factor", factor)))
    img = (rng.rand(2, 7, 9) * 4 - 0.2).astype(np.float32)
    got = _run(img, "bayer", spec, cuda_device)
    _check(_oracle(img, "bayer", spec), got, factor)
