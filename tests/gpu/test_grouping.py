from __future__ import annotations

import copy

import numpy as np
import pytest

from brilliantisp_gpu.grouping import group_key


def _config():
    return {
        "platform": {
            "filename": "one.raw", "in_file": "one", "out_file": "Out_one",
            "output_dir": "module", "final_output_dir": "output",
            "debug_log_file": None, "render_3a": False, "skip_disabled_modules": True,
        },
        "sensor_info": dict(height=8, width=10, bit_depth=12, bayer_pattern="rggb"),
        "demosaic": dict(is_enable=True, algorithm="malvar"),
        "output": dict(model="rgb", bit_depth=8, transfer="srgb"),
        "digital_gain": dict(is_enable=True, gain_array=[1.0, 2.0]),
    }


def _key(config, **kwargs):
    args = dict(shape=(8, 10), cfa="rggb", storage_dtype="uint16")
    args.update(kwargs)
    return group_key(config, **args)


def test_group_key_metadata_only_and_nonmutating():
    first = _config()
    before = copy.deepcopy(first)
    second = copy.deepcopy(first)
    second["platform"].update(
        filename="two.raw", in_file="two", out_file="Out_two",
        output_dir="other_module", final_output_dir="other_output",
        debug_log_file="other.log",
    )
    assert _key(first) == _key(second)
    assert first == before
    assert _key(dict(reversed(tuple(first.items())))) == _key(first)


@pytest.mark.parametrize("section,name,value", (
    ("platform", "render_3a", True),
    ("platform", "skip_disabled_modules", False),
    ("sensor_info", "bit_depth", 16),
    ("demosaic", "algorithm", "bilinear"),
    ("output", "bit_depth", 16),
    ("digital_gain", "gain_array", [1.0, 4.0]),
))
def test_group_key_preserves_processing_controls(section, name, value):
    first = _config()
    second = copy.deepcopy(first)
    second[section][name] = value
    assert _key(first) != _key(second)


@pytest.mark.parametrize("change", (
    dict(shape=(7, 10)), dict(cfa="bggr"), dict(storage_dtype="uint32"),
    dict(capture_stages=("demosaic",)), dict(diagnostics=True),
    dict(profile_stages=True),
))
def test_group_key_preserves_geometry_dtype_and_observability(change):
    cfg = _config()
    assert _key(cfg) != _key(cfg, **change)


def test_group_key_preserves_configuration_numeric_types():
    keys = []
    for value in (1, 1.0, True, np.float32(1), np.float64(1)):
        cfg = _config()
        cfg["scalar"] = value
        keys.append(_key(cfg))
    assert len(set(keys)) == len(keys)
    first = _config()
    first["table"] = np.arange(4, dtype=np.float32)
    second = _config()
    second["table"] = np.arange(4, dtype=np.float64)
    assert _key(first) != _key(second)
    second["table"] = first["table"].copy()
    assert _key(first) == _key(second)
    second["table"][0] = 1
    assert _key(first) != _key(second)


def test_group_key_unknown_values_cannot_silently_coalesce():
    cfg = _config()
    cfg["inactive_unknown"] = object()
    with pytest.raises(TypeError, match="unsupported"):
        _key(cfg)


@pytest.mark.parametrize("change", (
    dict(shape=(True, 10)), dict(shape=(0, 10)),
    dict(cfa="rgb"), dict(storage_dtype="int64"),
))
def test_group_key_rejects_invalid_contract(change):
    with pytest.raises(ValueError):
        _key(_config(), **change)
