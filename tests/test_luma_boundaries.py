from __future__ import annotations

import numpy as np
import pytest

from util.luma import normalize_plane, restore_plane


@pytest.mark.parametrize("value", (float("nan"), float("inf"), -float("inf")))
def test_nonfinite_luma_is_not_clipped_into_valid_looking_pixels(value: float) -> None:
    with np.errstate(all="raise"):
        with pytest.raises(ValueError, match="finite"):
            normalize_plane(np.array([[value, 0.5]]), scale=1)


def test_wide_finite_luma_clips_before_narrowing() -> None:
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("extended float exponent range is unavailable")
    source = np.array([[np.longdouble("1e400"), np.longdouble("-1e400"), 0.5]], np.longdouble)
    with np.errstate(all="raise"):
        np.testing.assert_array_equal(normalize_plane(source, scale=1), [[1, 0, 0.5]])


@pytest.mark.parametrize("data", (
    np.array([[True]], bool), np.array([[1j]], np.complex128),
    np.zeros((0, 0)), np.ones((1, 1, 1)), np.array([[float("inf")]]),
))
def test_restore_does_not_drop_imaginary_data_or_accept_invalid_planes(data: np.ndarray) -> None:
    with pytest.raises(ValueError):
        restore_plane(data, scale=65535)
