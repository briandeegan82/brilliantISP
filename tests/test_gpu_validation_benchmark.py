import copy
from types import SimpleNamespace

import numpy as np
import pytest

from tools.gpu_validation.benchmark import (
    _config_identity, summarize_samples, validate_pixels,
)
from tools.gpu_validation.qualification import _array_identity


def _samples():
    phases = ["cold", "warmup", "warmup"] + ["measured"] * 5
    seconds = [99, 50, 20, 1, 2, 3, 4, 5]
    return [{"phase": phase, "passed": True, "wall_ns": second * 10**9}
            for phase, second in zip(phases, seconds)]


def test_benchmark_uses_only_five_measured_workload_times():
    summary = summarize_samples(_samples(), images=16, megapixels=48.)
    values = summary["workload_wall_ns"]
    assert values["median"] == values["mean"] == 3e9
    assert values["p05"] == pytest.approx(1.2e9)
    assert values["p95"] == pytest.approx(4.8e9)
    assert values["std_population"] == pytest.approx(np.sqrt(2) * 1e9)
    assert summary["images_per_second_at_median_workload_wall"] == pytest.approx(16 / 3)
    assert summary["megapixels_per_second_at_median_workload_wall"] == 16
    assert summary["per_image_wall_equivalent_ns"] == 3e9 / 16


def test_benchmark_rejects_incomplete_or_failed_measurement_sets():
    with pytest.raises(ValueError):
        summarize_samples(_samples()[:-1], images=16, megapixels=48.)
    values = _samples()
    values[0]["passed"] = False
    with pytest.raises(ValueError):
        summarize_samples(values, images=16, megapixels=48.)
    with pytest.raises(ValueError):
        summarize_samples(_samples(), images=0, megapixels=48.)


def test_benchmark_config_comparison_removes_only_platform_io():
    config = {"platform": {"filename": "a.raw", "debug_enabled": False},
              "digital_gain": {"current_gain": 2},
              "sensor_info": {"bit_depth": 12}}
    before = copy.deepcopy(config)
    changed = copy.deepcopy(config)
    changed["platform"]["filename"] = "b.raw"
    assert _config_identity(config) == _config_identity(changed)
    changed["digital_gain"]["current_gain"] = 3
    assert _config_identity(config) != _config_identity(changed)
    assert config == before


def test_benchmark_pixel_hash_does_not_hide_dtype_or_scale():
    native = np.zeros((2, 2, 3), np.uint16)
    preview = np.zeros((2, 2, 3), np.uint8)
    case = SimpleNamespace(
        native=native, preview=preview,
        frozen={"native": _array_identity(native), "preview": _array_identity(preview),
                "metadata": {"code_bit_depth": 16}},
    )
    assert all(r["passed"] for r in validate_pixels(case, native.copy(), preview.copy()).values())
    wrong = validate_pixels(case, native.astype(np.uint8), preview)
    assert not wrong["native"]["passed"]
    one_code = native.copy()
    one_code[0, 0, 0] = 1
    assert validate_pixels(case, one_code, preview)["native"]["passed"]
    one_code[0, 0, 0] = 2
    assert not validate_pixels(case, one_code, preview)["native"]["passed"]
