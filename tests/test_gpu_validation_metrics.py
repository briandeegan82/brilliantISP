from __future__ import annotations

import numpy as np
import pytest

from tools.gpu_validation.contracts import EXACT, FINAL_INTEGER, FLOAT32_NORMALIZED
from tools.gpu_validation.metrics import compare_arrays
from tools.gpu_validation.reference import file_identity, save_array


def test_integer_subtraction_is_signed_and_psnr_exactness_is_explicit() -> None:
    source = np.array([[0, 65535]], np.uint16)
    actual = np.array([[65535, 0]], np.uint16)
    result = compare_arrays(source, actual, EXACT, data_range=65535)
    assert result["max_abs"] == 65535
    assert result["mean_abs"] == 65535
    assert result["rmse"] == 65535
    assert result["signed_bias"] == 0
    assert result["psnr"] == 0
    assert result["outside_component_bound"] == 2
    same = compare_arrays(source, source, EXACT, data_range=65535)
    assert same["passed"] and same["psnr"] == "inf"


def test_final_code_gates_enforce_both_maximum_and_mean_at_original_depth() -> None:
    reference = np.zeros((2, 5), np.uint16)
    actual = reference.copy()
    actual[0, 0] = 1
    assert compare_arrays(reference, actual, FINAL_INTEGER)["passed"]
    actual[0, 1] = 1
    result = compare_arrays(reference, actual, FINAL_INTEGER)
    assert not result["passed"]
    assert result["outside_component_bound"] == 0
    actual = np.zeros((2, 10), np.uint16)
    actual[0, 0] = 2
    result = compare_arrays(np.zeros_like(actual), actual, FINAL_INTEGER)
    assert result["mean_abs"] == 0.1
    assert not result["passed"] and result["outside_component_bound"] == 1


def test_uint32_endpoint_does_not_wrap_or_drop_low_bits() -> None:
    reference = np.array([0, 2**24 + 1, 2**32 - 1], np.uint32)
    actual = np.array([1, 2**24, 2**32 - 2], np.uint32)
    result = compare_arrays(reference, actual, EXACT)
    assert result["max_abs"] == 1 and result["mean_abs"] == 1
    assert result["mismatched_components"] == 3
    assert result["signed_bias"] == pytest.approx(-1 / 3)


def test_float_gate_is_relative_to_reference_magnitude_and_preserves_headroom() -> None:
    reference = np.array([-2.0, 0.0, 3.0], np.float32)
    actual = reference + np.array([3e-5, 1e-6, -4e-5], np.float32)
    assert compare_arrays(reference, actual, FLOAT32_NORMALIZED)["passed"]
    actual[1] = 3e-6
    result = compare_arrays(reference, actual, FLOAT32_NORMALIZED)
    assert not result["passed"] and result["outside_component_bound"] == 1


def test_code_float_comparison_uses_declared_scale_not_observed_peak() -> None:
    reference = np.array([1000.0], np.float32)
    actual = np.array([1001.0], np.float32)
    assert not compare_arrays(reference, actual, FLOAT32_NORMALIZED)["passed"]
    assert compare_arrays(
        reference, actual, FLOAT32_NORMALIZED, scale=2**24 - 1
    )["passed"]


def test_per_channel_statistics_are_not_channel_averaged() -> None:
    reference = np.zeros((2, 2, 3), np.float32)
    actual = reference + np.array([1.0, -2.0, 3.0], np.float32)
    result = compare_arrays(reference, actual, EXACT)
    assert result["per_channel"]["max_abs"] == [1, 2, 3]
    assert result["per_channel"]["mean_abs"] == [1, 2, 3]
    assert result["per_channel"]["signed_bias"] == [1, -2, 3]
    assert result["worst_index"] == [0, 0, 2]


@pytest.mark.parametrize("value", (np.nan, np.inf, -np.inf))
def test_nonfinite_values_never_pass_comparison(value: float) -> None:
    data = np.array([value], np.float32)
    with pytest.raises(ValueError, match="finite"):
        compare_arrays(data, data, EXACT)


@pytest.mark.parametrize(("reference", "actual", "message"), (
    (np.zeros(1, np.float32), np.zeros(2, np.float32), "shape"),
    (np.zeros(1, np.float32), np.zeros(1, np.float64), "dtype"),
    (np.zeros(0, np.float32), np.zeros(0, np.float32), "nonempty"),
    (np.array([2**63], np.uint64), np.array([2**63 + 1], np.uint64), "integer values"),
))
def test_invalid_comparisons_raise(reference: np.ndarray, actual: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        compare_arrays(reference, actual, EXACT)


@pytest.mark.parametrize("scale", (True, 0, -1, np.inf, np.nan))
def test_invalid_scales_raise(scale: float) -> None:
    with pytest.raises(ValueError, match="scale"):
        compare_arrays(np.zeros(1, np.float32), np.zeros(1, np.float32), EXACT, scale=scale)


def test_integer_code_gates_reject_normalized_proxy() -> None:
    data = np.zeros(1, np.uint16)
    with pytest.raises(ValueError, match="original scale"):
        compare_arrays(data, data, FINAL_INTEGER, scale=65535)


def test_reference_storage_is_lossless_content_addressed_and_deduplicated(tmp_path) -> None:
    source = np.array([[-1.25, 0.0, 3.5], [1e-7, 1.0, 2.0]], np.float32)
    first = save_array(tmp_path / "first.npy", source)
    second = save_array(tmp_path / "second.npy", source.copy())
    assert first["file"] == second["file"]
    assert first["label"] != second["label"]
    assert len(list(tmp_path.iterdir())) == 1
    path = tmp_path / first["file"]
    assert path.suffix == ".npz" and file_identity(path)["sha256"] == first["sha256"]
    with np.load(path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["pixels"], source)
    changed = save_array(tmp_path / "changed.npy", source.astype(np.float64))
    assert changed["file"] != first["file"]


@pytest.mark.parametrize("values", (np.array([]), np.array([np.inf]), np.array([1j])))
def test_invalid_reference_arrays_are_rejected(tmp_path, values) -> None:
    with pytest.raises(ValueError):
        save_array(tmp_path / "invalid.npy", values)
    assert not list(tmp_path.iterdir())


def test_exact_integer_indices_and_bounded_int64_differences() -> None:
    indices = np.array([0, 17, 406776], np.int64)
    assert compare_arrays(indices, indices.copy(), EXACT)["passed"]
    changed = indices.copy()
    changed[-1] -= 1
    result = compare_arrays(indices, changed, EXACT)
    assert not result["passed"] and result["max_abs"] == 1
    endpoints = np.array([-2**52, 2**52], np.int64)
    result = compare_arrays(endpoints, endpoints[::-1], EXACT)
    assert result["max_abs"] == 2**53 and result["signed_bias"] == 0
