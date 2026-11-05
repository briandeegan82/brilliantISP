from __future__ import annotations

import copy
from typing import Any, cast

import pytest

from util.output_contract import OutputSpec


def test_default_rgb_output_is_explicit_and_bounded() -> None:
    spec = OutputSpec.from_mapping({})
    assert spec.as_config() == {
        "model": "rgb", "color_space": "srgb", "transfer": "srgb",
        "bit_depth": 8, "format": "png", "matrix": "bt709", "is_save": False,
    }
    assert spec.nominal_range == (0, 255)
    assert spec.component_ranges == ((0, 255),) * 3
    assert not spec.floating_linear
    assert spec.suffix == ".png"


def test_ycbcr_defaults_and_limited_component_ranges() -> None:
    spec = OutputSpec.from_mapping({"model": "ycbcr"})
    assert (spec.transfer, spec.format, spec.matrix) == ("rec709", "yuv", "bt709")
    assert spec.component_ranges == ((16, 235), (16, 240), (16, 240))
    assert spec.as_config()["packing"] == "yuyv"
    assert spec.as_config()["layout"] == "444"
    assert spec.suffix == ".yuv"
    wide = OutputSpec.from_mapping({"model": "ycbcr", "bit_depth": 16, "layout": "422", "packing": "uyvy"})
    assert wide.nominal_range == (0, 65535)
    assert wide.component_ranges == ((4096, 60160), (4096, 61440), (4096, 61440))


def test_float_export_is_normalized_linear_rgb_not_integer_valued_float() -> None:
    spec = OutputSpec.from_mapping({"bit_depth": 32, "format": "tiff", "transfer": "linear"})
    assert spec.floating_linear
    assert spec.nominal_range == (0, 1)
    assert spec.component_ranges == ((0, 1),) * 3
    assert spec.suffix == ".tiff"


@pytest.mark.parametrize("field,value", (
    ("model", "hsv"), ("model", False),
    ("color_space", "acescg"),
    ("transfer", "log"), ("transfer", 1),
    ("bit_depth", True), ("bit_depth", 10), ("bit_depth", 16.0),
    ("format", "npy"), ("matrix", "unknown"),
    ("range", "studio"), ("layout", 422), ("packing", "vyuy"),
    ("is_save", 1), ("is_save", "false"),
))
def test_invalid_output_fields_are_named(field: str, value) -> None:
    with pytest.raises(ValueError, match=field):
        OutputSpec.from_mapping({field: value})


@pytest.mark.parametrize("settings", (
    {"model": "rgb", "format": "yuv"},
    {"model": "rgb", "range": "limited"},
    {"model": "rgb", "layout": "422"},
    {"format": "jpeg", "bit_depth": 16},
    {"format": "png", "bit_depth": 32, "transfer": "linear"},
    {"format": "tiff", "bit_depth": 32, "transfer": "srgb"},
    {"model": "ycbcr", "format": "png"},
    {"model": "ycbcr", "bit_depth": 32, "transfer": "linear"},
    {"model": "ycbcr", "transfer": "linear"},
))
def test_unsupported_combinations_are_not_silently_reinterpreted(settings: dict) -> None:
    with pytest.raises(ValueError):
        OutputSpec.from_mapping(settings)


@pytest.mark.parametrize("gamma", (False, 0, -1, float("nan"), float("inf"), "2.2", 10**400))
def test_power_requires_a_positive_finite_gamma(gamma) -> None:
    with pytest.raises(ValueError, match="gamma"):
        OutputSpec.from_mapping({"transfer": "power", "gamma": gamma})


def test_unused_gamma_is_not_a_second_active_transfer() -> None:
    spec = OutputSpec.from_mapping({"transfer": "srgb", "gamma": "inactive"})
    assert spec.transfer == "srgb"
    assert "gamma" not in spec.as_config()
    power = OutputSpec.from_mapping({"transfer": "power", "gamma": 2.4})
    assert power.as_config()["gamma"] == 2.4


def test_canonical_round_trip_does_not_mutate_source() -> None:
    source = {
        "model": "YCbCr", "transfer": "POWER", "gamma": 2.4, "bit_depth": 16,
        "format": "YUV", "matrix": "BT601", "range": "full",
        "layout": "422", "packing": "UYVY", "is_save": True,
    }
    old = copy.deepcopy(source)
    spec = OutputSpec.from_mapping(source)
    assert OutputSpec.from_mapping(spec.as_config()) == spec
    assert source == old
    assert spec.component_ranges == ((0, 65535),) * 3


def test_unknown_settings_do_not_become_nonfunctional_controls() -> None:
    with pytest.raises(ValueError, match="unknown output settings"):
        OutputSpec.from_mapping({"output_gamma": 2.4})
    with pytest.raises(ValueError, match="mapping"):
        OutputSpec.from_mapping(cast(Any, []))
