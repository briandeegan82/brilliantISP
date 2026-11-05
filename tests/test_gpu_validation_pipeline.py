from __future__ import annotations

import numpy as np

from tools.gpu_validation.pipeline import _compare_record, _record


def _compare(tmp_path, stage, reference, candidate, code_max=2**24 - 1):
    left, right = tmp_path / "reference", tmp_path / "candidate"
    left.mkdir()
    right.mkdir()
    return _compare_record(
        _record(reference, left, "value"), _record(candidate, right, "value"),
        left, right, stage, code_max,
    )


def test_float_code_comparison_uses_declared_reference_not_observed_peak(tmp_path):
    result = _compare(tmp_path, "white_balance", np.zeros((2, 3), np.float32),
                      np.ones((2, 3), np.float32))
    assert result["passed"] and result["scale"] == 2**24 - 1


def test_raw_integer_stage_is_exact(tmp_path):
    result = _compare(tmp_path, "companding", np.array([1], np.uint32),
                      np.array([2], np.uint32))
    assert not result["passed"] and result["gate"] == "exact"


def test_output_stage_uses_final_integer_mean_gate(tmp_path):
    expected = np.zeros(20, np.uint8)
    actual = expected.copy()
    actual[0] = 1
    result = _compare(tmp_path, "output", expected, actual)
    assert result["passed"] and result["mean_abs"] == 0.05


def test_tuple_image_and_discrete_gain_index_are_both_checked(tmp_path):
    image = np.ones((2, 3), np.float32)
    result = _compare(tmp_path, "digital_gain", (image, 1), (image.copy(), 2))
    assert not result["passed"]
    assert result["members"][0]["passed"] and not result["members"][1]["passed"]


def test_white_balance_control_values_have_statistics_gate(tmp_path):
    result = _compare(tmp_path, "auto_white_balance", (1.0, 2.0), (1.0 + 1e-9, 2.0))
    assert result["passed"] and result["members"][0]["gate"] == "statistics"


def test_exposure_control_types_and_values_are_exact(tmp_path):
    result = _compare(tmp_path, "auto_exposure", 1, True)
    assert not result["passed"]


def test_stage_dtype_mismatch_is_a_failed_comparison(tmp_path):
    result = _compare(tmp_path, "crop", np.array([1], np.uint16), np.array([1], np.uint32))
    assert not result["passed"] and "dtype mismatch" in result["error"]
