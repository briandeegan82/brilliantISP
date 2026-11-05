from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from modules.tone_mapping import operators
from modules.tone_mapping.backends import (
    _LUT_CACHE,
    _LutCache,
    _integer_lut,
    _q32_coordinate,
    map_luminance,
    prepare_curve_input,
)
from modules.tone_mapping.local_durand import map_luminance as durand_map
from modules.tone_mapping.normalization import (
    FrameStats,
    frame_stats,
    normalize_integer,
    recombine,
    rgb_luminance,
)
from modules.tone_mapping.registry import (
    ToneMappingError,
    ToneMappingSpec,
    spec_capabilities,
)

M = 2**32 - 1


def spec(**kw) -> ToneMappingSpec:
    kw.setdefault("operator", "reinhard")
    return ToneMappingSpec(**kw)


def test_operators_no_oetf_scalar() -> None:
    x = np.array([0.0, 0.25, 0.5, 1.0, 4.0])
    for fn, kw in (
        (operators.aces_fitted, {}),
        (operators.hable, {}),
        (operators.reinhard, {}),
    ):
        out = fn(x, **kw)
        assert out.shape == x.shape and np.all(np.isfinite(out))
        assert np.all(out >= 0)
    assert operators.aces_fitted(np.array([0.0]))[0] == 0.0
    with pytest.raises(ValueError):
        operators.reinhard(np.array([np.nan]))


def test_spec_valid_and_defaults() -> None:
    s = spec(operator="hable", parameters=())
    assert s.parameters_map() == {"exposure_bias": 2.0, "white_point": 11.2}
    s2 = spec(operator="hable", parameters=(("white_point", 5.0),))
    assert s2.parameters_map()["exposure_bias"] == 2.0
    assert s2.parameters_map()["white_point"] == 5.0
    assert s.parameters == tuple(sorted(s.parameters_map().items()))
    hash(s)


def test_spec_unknown_operator_and_params() -> None:
    with pytest.raises(ToneMappingError):
        spec(operator="bogus")
    with pytest.raises(ToneMappingError):
        spec(operator="hable", parameters=(("nope", 1.0),))
    with pytest.raises(ToneMappingError):
        spec(operator="reinhard", backend="cpu")
    with pytest.raises(ToneMappingError):
        spec(operator="durand", backend="float")
    with pytest.raises(ToneMappingError):
        spec(operator="aces_fitted", parameters=(("x", 1.0),))
    with pytest.raises(ToneMappingError):
        spec(operator="reinhard", parameters=(("knee", 1.0), ("knee", 2.0)))
    with pytest.raises(ToneMappingError):
        spec(operator="reinhard", parameters=(("knee",),))
    with pytest.raises(ToneMappingError):
        spec(operator="reinhard", parameters=((1, 1.0),))


def test_spec_bad_numbers() -> None:
    for bad in (
        dict(exposure_ev=True),
        dict(exposure_ev=float("nan")),
        dict(exposure_ev=float("inf")),
        dict(exposure_ev=33.0),
        dict(exposure_ev=-33.0),
        dict(domain_scale=0.0),
        dict(domain_scale=-1.0),
        dict(lut_domain=0.0),
        dict(lut_size=300),
        dict(lut_size=65537),
        dict(lut_size=1000),
        dict(normalization="magic"),
        dict(backend="cuda"),
        dict(enabled=1),
    ):
        with pytest.raises(ToneMappingError):
            spec(**bad)


def test_spec_unrepresentable_reference() -> None:
    with pytest.raises(ToneMappingError):
        spec(operator="hable", parameters=(("white_point", 5e-324),))
    with pytest.raises(ToneMappingError):
        spec(operator="reinhard",
             parameters=(("knee", 1e-300), ("strength", 1e300)))


def test_downsample_factor_integer_only() -> None:
    with pytest.raises(ToneMappingError):
        spec(operator="durand", backend="cpu",
             parameters=(("downsample_factor", 1.5),))
    with pytest.raises(ToneMappingError):
        spec(operator="durand", backend="cpu",
             parameters=(("downsample_factor", 0.5),))
    s = spec(operator="durand", backend="cpu",
             parameters=(("downsample_factor", 3),))
    assert s.parameters_map()["downsample_factor"] == 3.0


def test_durand_params_representable() -> None:
    with pytest.raises(ToneMappingError):
        spec(operator="durand", backend="cpu",
             parameters=(("sigma_color", 1e-50),))


def test_disabled_bypasses_operator_validation() -> None:
    s = ToneMappingSpec(operator="bogus", backend="bogus", enabled=False,
                        parameters=(("knee", -1.0),))
    assert s.parameters == ()
    s2 = spec(operator="reinhard", enabled=False, exposure_ev=float("nan"))
    assert not s2.enabled


def test_durand_normalize_output_rejected() -> None:
    with pytest.raises(ToneMappingError):
        spec(operator="durand", backend="cpu", normalize_output=True)


def test_capabilities_truthful() -> None:
    caps = spec_capabilities()
    assert caps["aces_fitted"]["backends"] == frozenset({"float", "lut"})
    assert caps["durand"]["local_operator"] is True
    assert caps["reinhard"]["parameters"] == {"knee": 0.25, "strength": 1.0}


def test_float_frame_stats_exact() -> None:
    x = np.array([[0.125, 0.375], [0.5, 0.875]], dtype=np.float64)
    stats = frame_stats(x)
    assert stats.minimum == 0.125 and stats.maximum == 0.875
    assert isinstance(stats.minimum, float)
    s = spec(normalization="per_frame")
    whole = map_luminance(x, input_max=1.0, output_max=65535, config=s)
    roi = x[0:1, 0:2]
    roi_out = map_luminance(roi, input_max=1.0, output_max=255, config=s,
                            frame_stats=stats)
    expected = map_luminance(x, input_max=1.0, output_max=255, config=s)
    np.testing.assert_allclose(roi_out, expected[0:1, 0:2])
    const = np.full((3, 3), 0.5)
    out = map_luminance(const, input_max=1.0, output_max=255, config=s)
    dec = map_luminance(const, input_max=1.0, output_max=255,
                        config=spec(normalization="declared_range"))
    np.testing.assert_array_equal(out, dec)


def test_frame_stats_validation() -> None:
    for bad in (dict(minimum=True, maximum=5), dict(minimum=0.5, maximum=float("nan")),
                dict(minimum=0.5, maximum=float("inf")), dict(minimum=5, maximum=2),
                dict(minimum="x", maximum=5)):
        with pytest.raises(ValueError):
            FrameStats(**cast(Any, bad))
    assert FrameStats(1, 10).minimum == 1
    assert FrameStats(0.125, 0.875).maximum == 0.875


def test_integer_normalization_no_wrap() -> None:
    codes = np.array([[0, 10, 15, 20, 30]], dtype=np.uint16)
    stats = FrameStats(10, 20)
    n, d = normalize_integer(codes, normalization="per_frame",
                             input_max=255, stats=stats)
    np.testing.assert_array_equal(n, [[0, 0, 5, 10, 10]])
    assert d == 10
    np.testing.assert_array_equal(codes, [[0, 10, 15, 20, 30]])
    s = spec(normalization="per_frame", backend="lut")
    out = map_luminance(codes, input_max=255, output_max=255, config=s,
                        frame_stats=stats)
    ref = map_luminance(codes.astype(np.float64), input_max=255,
                        output_max=255,
                        config=spec(normalization="per_frame"), frame_stats=stats)
    np.testing.assert_allclose(out.astype(np.float64), ref, atol=2)
    with pytest.raises(ValueError):
        normalize_integer(codes, normalization="per_frame", input_max=255,
                          stats=FrameStats(0.5, 20))


def test_declared_hdr24_not_guessed() -> None:
    x = np.array([[0, 65535]], dtype=np.uint32)
    s = spec()
    out = map_luminance(x, input_max=2**24 - 1, output_max=65535, config=s)
    assert int(out[0, 1]) < 10000


def test_per_frame_constant_falls_back_declared() -> None:
    x = np.full((4, 4), 100, np.uint16)
    s = spec(normalization="per_frame")
    out = map_luminance(x, input_max=255, output_max=255, config=s)
    expected = map_luminance(x, input_max=255, output_max=255,
                             config=spec(normalization="declared_range"))
    np.testing.assert_array_equal(out, expected)


def test_external_stats_roi_reuse() -> None:
    full = np.arange(16, dtype=np.uint16).reshape(4, 4) * 100
    stats = frame_stats(full)
    roi = full[1:3, 1:3]
    s = spec(normalization="per_frame")
    whole = map_luminance(full, input_max=1500, output_max=255, config=s,
                          frame_stats=stats)
    roi_out = map_luminance(roi, input_max=1500, output_max=255, config=s,
                            frame_stats=stats)
    np.testing.assert_array_equal(roi_out, whole[1:3, 1:3])
    assert stats.minimum == 0


def test_source_not_mutated() -> None:
    x = np.arange(12, dtype=np.uint16).reshape(3, 4)
    before = x.copy()
    map_luminance(x, input_max=255, output_max=255, config=spec())
    np.testing.assert_array_equal(x, before)


def test_q32_endpoints() -> None:
    assert _q32_coordinate(np.array([0], np.uint64), 1, 1.0)[0] == 0
    assert _q32_coordinate(np.array([65535], np.uint64), 65535, 1.0)[0] == M
    assert _q32_coordinate(np.array([M], np.uint64), 2**32 - 1, 1.0)[0] == M
    huge = np.array([0, 1, 5], dtype=np.uint64)
    q = _q32_coordinate(huge, 1, 1e300)
    assert q[0] == 0 and q[1] == M and q[2] == M
    z = _q32_coordinate(huge, 1, 0.0)
    assert np.all(z == 0)
    tiny = _q32_coordinate(np.array([1], np.uint64), 1, 1e-45)
    assert tiny[0] <= 2
    with pytest.raises(ToneMappingError):
        _q32_coordinate(huge, 1, float("nan"))
    with pytest.raises(ToneMappingError):
        _q32_coordinate(huge, 1, -1.0)
    with pytest.raises(ToneMappingError):
        _q32_coordinate(huge, 0, 1.0)
    q2 = _q32_coordinate(np.array([M], np.uint64), 2**32 - 1, 1.0)
    assert q2[0] == M


def test_integer_lut_matches_float() -> None:
    xs = np.array([0, 256, 1024, 4095], dtype=np.uint16)
    s = spec(operator="aces_fitted", backend="lut")
    out = map_luminance(xs, input_max=4095, output_max=65535, config=s)
    assert out.dtype == np.uint16
    s_f = spec(operator="aces_fitted", backend="float")
    ref = map_luminance(xs.astype(np.float64), input_max=4095,
                        output_max=65535, config=s_f)
    np.testing.assert_allclose(out.astype(np.float64), ref, atol=2)


def test_fixed_reinhard_matches_float() -> None:
    xs = np.array([0, 64, 256, 1024, 4095], dtype=np.uint16)
    s = spec(backend="fixed")
    out = map_luminance(xs, input_max=4095, output_max=65535, config=s)
    assert out.dtype == np.uint16
    ref = map_luminance(xs.astype(np.float64), input_max=4095,
                        output_max=65535, config=spec(backend="float"))
    np.testing.assert_allclose(out.astype(np.float64), ref, atol=2)


def test_fixed_gain_above_domain() -> None:
    xs = np.array([0, 1024, 2048, 4095], dtype=np.uint16)
    for norm in (False, True):
        s = spec(backend="fixed", domain_scale=32.0, normalize_output=norm)
        out = map_luminance(xs, input_max=4095, output_max=65535, config=s)
        ref = map_luminance(xs.astype(np.float64), input_max=4095,
                            output_max=65535,
                            config=spec(backend="float", domain_scale=32.0,
                                        normalize_output=norm))
        np.testing.assert_allclose(out.astype(np.float64), ref, atol=2)


def test_fixed_rejects_float_and_out_of_range() -> None:
    s = spec(backend="fixed")
    with pytest.raises(ToneMappingError):
        map_luminance(np.array([0.5]), input_max=4095, output_max=65535, config=s)
    with pytest.raises(ToneMappingError, match="input_max"):
        map_luminance(np.array([5000], np.uint16), input_max=4095,
                      output_max=65535, config=s)


def test_float_headroom_and_negative() -> None:
    out = map_luminance(np.array([2.0, 8.0]), input_max=4, output_max=255,
                        config=spec())
    expected = float(operators.reinhard(np.array([2.0]))[0]) * 255.0
    assert out[1] == pytest.approx(expected)
    out3 = map_luminance(np.array([100.0]), input_max=4, output_max=255,
                         config=spec())
    assert out3[0] == pytest.approx(
        float(operators.reinhard(np.array([16.0]))[0]) * 255.0
    )
    out2 = map_luminance(np.array([-5.0, 1.0]), input_max=4, output_max=255,
                         config=spec())
    assert out2[0] == 0.0


def test_output_scale_contracts() -> None:
    xs = np.array([0, 1000, 4095], dtype=np.uint16)
    out = map_luminance(xs, input_max=4095, output_max=1023, config=spec())
    ref = map_luminance(xs.astype(np.float64), input_max=4095,
                        output_max=1023, config=spec())
    np.testing.assert_allclose(out.astype(np.float64), ref, atol=2)


def test_input_dtype_and_range_enforcement() -> None:
    s = spec(backend="lut")
    with pytest.raises(ToneMappingError):
        map_luminance(np.array([5000], np.uint16), input_max=4095,
                      output_max=65535, config=s)
    with pytest.raises(ToneMappingError):
        map_luminance(np.array([5000], np.uint16), input_max=4095,
                      output_max=65535, config=spec())
    with pytest.raises(ToneMappingError):
        map_luminance(np.array([1], np.uint64), input_max=255,
                      output_max=255, config=s)
    for bad in (np.array([1 + 2j]), np.array([object()]), np.array(["a"]),
                np.array([True]), np.array([], dtype=np.uint16)):
        with pytest.raises(ToneMappingError):
            map_luminance(bad, input_max=255, output_max=255, config=s)
    out = map_luminance(np.array([0.5, 1.0]), input_max=1.0, output_max=255,
                        config=spec())
    assert np.all(np.isfinite(out))
    out2 = map_luminance(np.array([0.25]), input_max=0.5, output_max=255,
                         config=spec())
    assert np.isfinite(out2[0])
    with pytest.raises(ToneMappingError, match="integral input_max"):
        map_luminance(np.array([4], np.uint16), input_max=4.5,
                      output_max=255, config=s)


def test_disabled_identity_no_helpers(monkeypatch) -> None:
    xs = np.array([0, 1000, 4095], dtype=np.uint16)
    s = spec(enabled=False)
    out = map_luminance(xs, input_max=4095, output_max=4095, config=s)
    assert out is xs
    calls = []
    import modules.tone_mapping.backends as backends

    monkeypatch.setattr(backends, "_curve", lambda *a, **k: calls.append(a) or 0)
    monkeypatch.setattr(backends._LUT_CACHE, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(backends, "normalize_integer", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    out2 = map_luminance(xs, input_max=4095, output_max=65535, config=s)
    np.testing.assert_allclose(out2, xs * (65535 / 4095), atol=1)
    assert calls == []


def test_range_validation() -> None:
    with pytest.raises(ToneMappingError):
        map_luminance(np.array([1]), input_max=0, output_max=255, config=spec())
    with pytest.raises(ToneMappingError):
        map_luminance(np.array([1]), input_max=2**32, output_max=255,
                      config=spec())
    with pytest.raises(ToneMappingError):
        map_luminance(np.array([1]), input_max=255, output_max=65536,
                      config=spec())


def test_cache_immutable_bytes() -> None:
    s = spec(backend="lut")
    _LUT_CACHE.clear()
    t1 = _LUT_CACHE.get(s, 65535)
    assert t1.flags.writeable is False
    with pytest.raises((ValueError, AttributeError)):
        t1[0] = 5
    with pytest.raises(ValueError):
        t1.setflags(write=True)


def test_cache_key_variants() -> None:
    _LUT_CACHE.clear()
    base = spec(backend="lut")
    variants = [
        spec(backend="lut", exposure_ev=1.0),
        spec(backend="lut", normalize_output=True),
        spec(backend="lut", lut_size=256),
        spec(backend="lut", domain_scale=4.0),
        spec(backend="lut", normalization="per_frame"),
        spec(backend="lut", parameters=(("knee", 0.5),)),
        spec(operator="hable", backend="lut"),
    ]
    keys = {_LUT_CACHE.key(base, 65535)}
    for v in variants:
        k = _LUT_CACHE.key(v, 65535)
        assert k not in keys
        keys.add(k)
    assert _LUT_CACHE.key(base, 255) != _LUT_CACHE.key(base, 65535)


def test_cache_eviction_budget() -> None:
    cache = _LutCache(budget_bytes=4 * 65536)
    s1 = spec(backend="lut", lut_size=65536)
    s2 = spec(backend="lut", lut_size=65536, exposure_ev=1.0)
    s3 = spec(backend="lut", lut_size=65536, exposure_ev=2.0)
    cache.get(s1, 65535)
    cache.get(s2, 65535)
    cache.get(s1, 65535)
    cache.get(s3, 65535)
    info = cache.info()
    assert info["retained_bytes"] <= info["budget_bytes"]
    assert cache.key(s2, 65535) not in cache._tables
    assert cache.key(s1, 65535) in cache._tables


def test_full_uint32_vectors_all_operators() -> None:
    xs = np.array([0, 1, 65535, 16777215, M], dtype=np.uint32)
    for op in ("reinhard", "hable", "aces_fitted"):
        s = spec(operator=op, backend="lut")
        out = map_luminance(xs, input_max=M, output_max=65535, config=s)
        ref = map_luminance(xs.astype(np.float64), input_max=M,
                            output_max=65535,
                            config=spec(operator=op, backend="float"))
        np.testing.assert_allclose(out.astype(np.float64), ref, atol=2)
        assert out[0] == 0
        assert out.dtype == np.uint16
    s_fixed = spec(backend="fixed")
    out_f = map_luminance(xs, input_max=M, output_max=65535, config=s_fixed)
    ref_f = map_luminance(xs.astype(np.float64), input_max=M, output_max=65535,
                          config=spec())
    np.testing.assert_allclose(out_f.astype(np.float64), ref_f, atol=2)


def _durand_spec(**kw) -> ToneMappingSpec:
    kw.setdefault("operator", "durand")
    kw.setdefault("backend", "cpu")
    return ToneMappingSpec(**kw)


def test_durand_black_and_constant() -> None:
    s = _durand_spec()
    black = np.zeros((5, 5), np.uint16)
    out = durand_map(black, input_max=255, output_max=255, config=s)
    assert np.all(out == 0)
    const = np.full((5, 5), 100, np.uint16)
    out2 = durand_map(const, input_max=255, output_max=255, config=s)
    assert np.all(np.isfinite(out2))


def test_durand_odd_shape_and_validation() -> None:
    rng = np.arange(5 * 7, dtype=np.uint16).reshape(5, 7) * 30 % 255
    out = durand_map(rng, input_max=255, output_max=255, config=_durand_spec())
    assert out.shape == (5, 7)
    with pytest.raises(ToneMappingError):
        durand_map(np.ones((3, 3, 3)), input_max=255, output_max=255,
                   config=_durand_spec())
    with pytest.raises(ToneMappingError):
        durand_map(np.array(5.0), input_max=255, output_max=255,
                   config=_durand_spec())
    with pytest.raises(ToneMappingError):
        durand_map(np.empty((0, 4)), input_max=255, output_max=255,
                   config=_durand_spec())
    with pytest.raises(ValueError):
        durand_map(np.full((4, 4), np.nan), input_max=255, output_max=255,
                   config=_durand_spec())


def test_durand_gain_and_domain_applied() -> None:
    rng = (np.arange(6 * 6, dtype=np.uint16).reshape(6, 6) * 37 % 255).astype(
        np.float64
    )
    captured = []

    def capture(plane, sigma):
        captured.append(plane.copy())
        return plane

    s = _durand_spec(exposure_ev=2.0, domain_scale=3.0)
    durand_map(rng, input_max=255, output_max=255, config=s, gaussian=capture)
    gain = 2.0**2.0 * 3.0
    expected = np.log10(
        np.clip(
            (np.maximum(rng / 255.0, 0.0)) * gain, 0.0, 16.0
        ).astype(np.float32)
        + np.float32(1e-6)
    )
    np.testing.assert_allclose(captured[0], expected, rtol=1e-6)
    o_direct = durand_map(rng, input_max=255, output_max=255, config=s)
    o_dispatched = map_luminance(rng, input_max=255, output_max=255, config=s)
    np.testing.assert_array_equal(o_direct, o_dispatched)


def test_durand_downsample_recipe() -> None:
    rng = (np.arange(9 * 11, dtype=np.uint16).reshape(9, 11) * 37 % 255)
    captured = []

    def capture(plane, sigma):
        captured.append((plane.copy(), sigma))
        return plane

    s = _durand_spec(parameters=(("downsample_factor", 3),))
    out = durand_map(rng, input_max=255, output_max=255, config=s,
                     gaussian=capture)
    assert out.shape == rng.shape
    plane, sigma = captured[0]
    assert plane.shape == (3, 4)
    assert sigma == pytest.approx(2.0 / 3.0)
    import cv2

    log_full = np.log10((rng / 255.0).astype(np.float32) + np.float32(1e-6))
    expected_small = cv2.resize(log_full, (4, 3), interpolation=cv2.INTER_AREA)
    np.testing.assert_allclose(plane, expected_small, rtol=1e-6)
    s1 = _durand_spec()
    durand_map(rng, input_max=255, output_max=255, config=s1,
               gaussian=capture)
    assert captured[-1][0].shape == rng.shape


def test_durand_injected_gaussian_validated() -> None:
    rng = np.arange(4 * 4, dtype=np.uint16).reshape(4, 4) * 10
    with pytest.raises(ValueError, match="shape"):
        durand_map(rng, input_max=255, output_max=255, config=_durand_spec(),
                   gaussian=lambda p, s: np.zeros((2, 2), np.float32))
    with pytest.raises(ValueError, match="nonfinite"):
        durand_map(rng, input_max=255, output_max=255, config=_durand_spec(),
                   gaussian=lambda p, s: np.full(p.shape, np.nan, np.float32))


def test_durand_default_matches_scipy() -> None:
    from scipy.ndimage import gaussian_filter

    impulses = np.zeros((9, 9), np.float32)
    impulses[0, 0] = 1.0
    impulses[0, -1] = 1.0
    impulses[-1, 0] = 1.0
    impulses[-1, -1] = 1.0
    impulses[4, 4] = 1.0
    from modules.tone_mapping.local_durand import _gaussian

    np.testing.assert_allclose(
        _gaussian(impulses, 2.0), gaussian_filter(impulses, 2.0), atol=1e-6
    )


def test_rgb_luminance_integer_oracle() -> None:
    rgb = np.array([[[100, 50, 25], [255, 255, 255], [10, 0, 0]]], np.uint16)
    out = rgb_luminance(rgb)
    expected = np.array([[(2126 * 100 + 7152 * 50 + 722 * 25) / 10000,
                          (2126 * 255 + 7152 * 255 + 722 * 255) / 10000,
                          2126 * 10 / 10000]])
    np.testing.assert_allclose(out, expected, rtol=1e-12)
    big = np.full((2, 2, 3), 65535, np.uint16)
    assert np.isfinite(rgb_luminance(big)).all()
    with pytest.raises(ValueError):
        rgb_luminance(np.ones((2, 2, 3), np.uint64))
    with pytest.raises(ValueError):
        rgb_luminance(np.ones((2, 2), np.uint16))


def test_recombine_floor_clip_zero() -> None:
    rgb = np.array([[[4000, 2000, 1000], [0, 0, 0]]], np.uint16)
    y = rgb_luminance(rgb)
    mapped = np.array([[5000.0, 3.0]])
    out = recombine(mapped, rgb, y, output_max=65535)
    assert out[0, 1].tolist() == [0, 0, 0]
    assert out.dtype == np.uint16
    hot = np.array([[[65535, 65535, 65535]]], np.uint16)
    y2 = rgb_luminance(hot)
    out2 = recombine(np.array([[200000.0]]), hot, y2, output_max=65535)
    assert int(out2.max()) <= 65535
    tiny = np.array([[[300, 200, 100]]], np.uint16)
    y3 = np.array([[1e-300]])
    out3 = recombine(np.array([[65535.0]]), tiny, y3, output_max=65535)
    assert out3[0, 0].tolist() == [0, 0, 0]
    neg = np.array([[[300, 200, 100]]], np.int32)
    out4 = recombine(np.array([[65535.0]]), neg, np.array([[-5.0]]),
                     output_max=65535)
    assert out4[0, 0].tolist() == [0, 0, 0]
    with pytest.raises(ValueError):
        recombine(np.array([[1.0]]), np.zeros((1, 2, 3), np.uint16),
                  np.array([[1.0]]), output_max=65535)
    with pytest.raises(ValueError):
        recombine(np.array([[1.0]]), np.full((1, 1, 3), np.nan),
                  np.array([[1.0]]), output_max=65535)
    with pytest.raises(ValueError):
        recombine(np.array([[np.nan]]), np.zeros((1, 1, 3), np.uint16),
                  np.array([[1.0]]), output_max=65535)
