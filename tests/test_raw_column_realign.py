"""Tests for sensor_info.realign_columns_from."""

# pylint: disable=missing-function-docstring
import numpy as np
import pytest

from util.raw_decode import RawFormat, RawFormatError, decode_raw_bytes


def _fmt(**kw):
    return RawFormat(
        width=8,
        height=4,
        bayer_pattern="gbrg",
        bit_depth=12,
        storage_format="uint16",
        byte_order="big",
        **kw
    )


def test_realign_shifts_tail_columns_right_and_fills_same_colour():
    raw = np.arange(32, dtype=">u2").reshape(4, 8)
    out = decode_raw_bytes(raw.tobytes(), _fmt(realign_columns_from=5)).pixels
    assert np.array_equal(out[:, :5], raw[:, :5])
    assert np.array_equal(out[:, 5], raw[:, 3])
    assert np.array_equal(out[:, 6:], raw[:, 5:7])


def test_realign_off_by_default_and_validated():
    raw = np.arange(32, dtype=">u2").reshape(4, 8)
    assert np.array_equal(decode_raw_bytes(raw.tobytes(), _fmt()).pixels, raw)
    for bad in (1, 8):
        with pytest.raises(RawFormatError):
            _fmt(realign_columns_from=bad)
