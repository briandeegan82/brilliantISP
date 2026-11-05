from __future__ import annotations

import numpy as np
import pytest

from modules.tone_mapping.backends import map_luminance
from modules.tone_mapping.local_durand import map_luminance as local_map
from modules.tone_mapping.normalization import FrameStats
from modules.tone_mapping.registry import ToneMappingError, ToneMappingSpec

Q24 = 2**24
U32_MAX = 2**32 - 1


def reinhard_reference(source: np.ndarray, exposure: float = 0, normalized: bool = False) -> np.ndarray:
    gain = 2.0**exposure
    x = np.minimum(np.maximum(source, 0) * gain, 16)
    result = x / (x + 0.25)
    if normalized:
        anchor = min(gain, 16)
        result /= anchor / (anchor + 0.25)
    return np.clip(result, 0, 1) * 65535


@pytest.mark.parametrize("backend", ("float", "lut", "fixed"))
@pytest.mark.parametrize("normalized", (False, True))
@pytest.mark.parametrize("exposure", (-2, 0, 2))
def test_integer_storage_range_is_separate_from_reference_white(
    backend: str, normalized: bool, exposure: int,
) -> None:
    source = np.array([[0, 0.0625, 0.25, 0.5, 1, 2, 8, 16, 32, 255]], np.float64)
    codes = (source * Q24).astype(np.uint32)
    spec = ToneMappingSpec("reinhard", backend=backend, normalize_output=normalized, exposure_ev=exposure)
    out = map_luminance(codes, input_max=U32_MAX, input_reference=Q24, output_max=65535, config=spec)
    np.testing.assert_allclose(
        out, reinhard_reference(source, exposure, normalized), rtol=0,
        atol=1e-10 if backend == "float" else 2,
    )
    if not normalized:
        assert out[0, 5] > out[0, 4]


@pytest.mark.parametrize("operator,backend", (
    ("aces_fitted", "float"), ("aces_fitted", "lut"),
    ("hable", "float"), ("hable", "lut"),
    ("reinhard", "float"), ("reinhard", "lut"), ("reinhard", "fixed"),
))
def test_explicit_reference_equal_to_max_preserves_default_recipe(operator: str, backend: str) -> None:
    source = np.array([[0, 1, 257, 32767, 65535]], np.uint16)
    spec = ToneMappingSpec(operator, backend=backend)
    old_interface = map_luminance(source, input_max=65535, output_max=65535, config=spec)
    explicit = map_luminance(
        source, input_max=65535, input_reference=65535, output_max=65535, config=spec,
    )
    np.testing.assert_array_equal(explicit, old_interface)


@pytest.mark.parametrize("operator", ("aces_fitted", "hable", "reinhard"))
@pytest.mark.parametrize("backend", ("float", "lut"))
def test_reference_rescaling_matches_existing_float_headroom(operator: str, backend: str) -> None:
    normalized = np.array([[0, 0.125, 0.5, 1, 2, 16, 128]], np.float64)
    codes = (normalized * Q24).astype(np.uint32)
    spec = ToneMappingSpec(operator, backend=backend)
    result = map_luminance(
        codes, input_max=U32_MAX, input_reference=Q24, output_max=65535, config=spec,
    )
    reference = map_luminance(normalized, input_max=1, output_max=65535, config=spec)
    np.testing.assert_allclose(result, reference, rtol=0, atol=1e-10 if backend == "float" else 2)


@pytest.mark.parametrize("backend", ("float", "lut", "fixed"))
def test_constant_per_frame_fallback_retains_declared_headroom(backend: str) -> None:
    source = np.full((2, 3), 2 * Q24, np.uint32)
    spec = ToneMappingSpec("reinhard", backend=backend, normalization="per_frame")
    actual = map_luminance(
        source, input_max=U32_MAX, input_reference=Q24, output_max=65535, config=spec,
    )
    np.testing.assert_allclose(actual, reinhard_reference(np.full((2, 3), 2.0)), rtol=0, atol=2)


@pytest.mark.parametrize("backend", ("float", "lut", "fixed"))
def test_supplied_nonconstant_per_frame_bounds_override_declared_reference(backend: str) -> None:
    source = np.array([[0, 10, 20, 30, 40]], np.uint32)
    stats = FrameStats(10, 30)
    spec = ToneMappingSpec("reinhard", backend=backend, normalization="per_frame")
    out = map_luminance(
        source, input_max=100, input_reference=20, output_max=65535, config=spec,
        frame_stats=stats,
    )
    expected_input = np.array([[0, 0, 0.5, 1, 1]])
    np.testing.assert_allclose(out, reinhard_reference(expected_input), rtol=0, atol=2)


@pytest.mark.parametrize("entrypoint", (map_luminance, local_map))
def test_durand_reference_and_physical_bounds_are_independent(entrypoint) -> None:
    source = np.full((5, 7), Q24 // 2, np.uint32)
    spec = ToneMappingSpec("durand", backend="cpu")
    actual = entrypoint(
        source, input_max=U32_MAX, input_reference=Q24, output_max=65535, config=spec,
    )
    np.testing.assert_allclose(actual, 65535 * 0.5, rtol=0, atol=1e-10)
    with pytest.raises(ToneMappingError):
        entrypoint(source, input_max=Q24 // 4, input_reference=Q24 // 8, output_max=65535, config=spec)


@pytest.mark.parametrize("entrypoint", (map_luminance, local_map))
@pytest.mark.parametrize("reference", (0, -1, True, "bad", float("nan"), float("inf"), 256, 10**400))
def test_invalid_reference_is_a_named_error_even_when_disabled(entrypoint, reference) -> None:
    spec = ToneMappingSpec("inactive", backend="inactive", enabled=False)
    with pytest.raises(ToneMappingError, match="input_reference"):
        entrypoint(np.zeros((1, 1), np.uint8), input_max=255, input_reference=reference, output_max=255, config=spec)


@pytest.mark.parametrize("entrypoint", (map_luminance, local_map))
def test_disabled_reference_conversion_needs_no_operator_or_statistics(entrypoint, monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("disabled mapping must not normalize or evaluate a curve")

    monkeypatch.setattr("modules.tone_mapping.backends._curve", forbidden)
    monkeypatch.setattr("modules.tone_mapping.local_durand._durand_plane", forbidden)
    spec = ToneMappingSpec("inactive", backend="inactive", enabled=False)
    source = np.array([[0, 255, 256, 65535]], np.uint16)
    actual = entrypoint(source, input_max=65535, input_reference=255, output_max=255, config=spec)
    np.testing.assert_array_equal(actual, [[0, 255, 255, 255]])
    assert actual is not source
    assert entrypoint(source, input_max=65535, input_reference=65535, output_max=65535, config=spec) is source


@pytest.mark.parametrize("backend", ("lut", "fixed"))
def test_integer_backends_reject_fractional_reference(backend: str) -> None:
    spec = ToneMappingSpec("reinhard", backend=backend)
    with pytest.raises(ToneMappingError, match="input_reference"):
        map_luminance(
            np.array([[1, 2]], np.uint8), input_max=255,
            input_reference=2.5, output_max=65535, config=spec,
        )


def test_float_reference_may_be_fractional_and_does_not_change_source_bounds() -> None:
    source = np.array([[0, 0.25, 0.5, 1, 2]], np.float64)
    actual = map_luminance(
        source, input_max=2, input_reference=0.5, output_max=65535,
        config=ToneMappingSpec("reinhard"),
    )
    np.testing.assert_allclose(actual, reinhard_reference(source / 0.5), rtol=0, atol=1e-10)
    with pytest.raises(ToneMappingError):
        map_luminance(
            np.array([[256]], np.uint16), input_max=255, input_reference=1,
            output_max=65535, config=ToneMappingSpec("reinhard"),
        )


def test_wide_integer_headroom_coordinate_does_not_overflow() -> None:
    source = np.array([[0, 1, Q24, 2 * Q24, U32_MAX]], np.uint32)
    with np.errstate(all="raise"):
        actual = map_luminance(
            source, input_max=U32_MAX, input_reference=Q24, output_max=65535,
            config=ToneMappingSpec("reinhard", backend="fixed"),
        )
    expected = reinhard_reference(source.astype(np.float64) / Q24)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2)
