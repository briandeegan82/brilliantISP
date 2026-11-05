from __future__ import annotations

import numpy as np
import pytest

import synthetic_fixtures as sf


def test_constant_and_black_white_frames() -> None:
    frame = sf.constant_frame((4, 6), 1234)
    assert frame.shape == (4, 6)
    assert frame.dtype == np.uint16
    assert np.all(frame == 1234)
    assert np.all(sf.black_frame((3, 3)) == 0)
    assert np.all(sf.white_frame((3, 3)) == sf.UINT16_MAX)


def test_ramp_frame_uint16() -> None:
    ramp = sf.ramp_frame((4, 8), np.uint16)
    assert ramp.dtype == np.uint16
    assert ramp[0, 0] == 0
    assert ramp[-1, -1] == 31
    np.testing.assert_array_equal(ramp.reshape(-1), np.arange(32))


def test_ramp_frame_uint32_spans_24bit() -> None:
    ramp = sf.ramp_frame((16, 16), np.uint32)
    assert ramp.dtype == np.uint32
    assert ramp[0, 0] == 0
    assert ramp[-1, -1] == sf.UINT24_MAX
    assert ramp.max() <= sf.UINT24_MAX
    flat = ramp.reshape(-1).astype(np.uint64)
    assert np.all(np.diff(flat) >= 0)


def test_primaries_and_neutral_rgb() -> None:
    rgb = sf.primaries_rgb((4, 16))
    assert rgb.shape == (4, 16, 3)
    assert tuple(rgb[0, 0]) == (sf.UINT16_MAX, 0, 0)
    assert tuple(rgb[0, 4]) == (0, sf.UINT16_MAX, 0)
    assert tuple(rgb[0, 8]) == (0, 0, sf.UINT16_MAX)
    assert tuple(rgb[0, 15]) == (sf.UINT16_MAX // 2,) * 3
    neutral = sf.neutral_rgb((3, 5), level=1000)
    assert np.all(neutral == 1000)
    assert neutral.shape == (3, 5, 3)


@pytest.mark.parametrize(
    "pattern, r_pos, gr_pos, gb_pos, b_pos",
    [
        ("rggb", (0, 0), (0, 1), (1, 0), (1, 1)),
        ("bggr", (1, 1), (1, 0), (0, 1), (0, 0)),
        ("grbg", (0, 1), (0, 0), (1, 1), (1, 0)),
        ("gbrg", (1, 0), (1, 1), (0, 0), (0, 1)),
    ],
)
def test_rgb_to_bayer_parity_positions(pattern, r_pos, gr_pos, gb_pos, b_pos) -> None:
    rgb = np.zeros((4, 4, 3), dtype=np.uint16)
    rgb[..., 0] = 100
    rgb[..., 1] = 200
    rgb[..., 2] = 300
    bayer = sf.rgb_to_bayer(rgb, pattern)
    assert bayer.shape == (4, 4)
    assert bayer.dtype == np.uint16
    assert bayer[r_pos] == 100
    assert bayer[gr_pos] == 200
    assert bayer[gb_pos] == 200
    assert bayer[b_pos] == 300
    expected = np.empty((4, 4), dtype=np.uint16)
    for r in range(4):
        for c in range(4):
            expected[r, c] = (100, 200, 300)[sf.bayer_channel(pattern, r, c)]
    np.testing.assert_array_equal(bayer, expected)


@pytest.mark.parametrize("pattern", sf.BAYER_PATTERNS)
def test_rgb_to_bayer_neutral_identity(pattern) -> None:
    rgb = sf.neutral_rgb((6, 6), level=777)
    np.testing.assert_array_equal(sf.rgb_to_bayer(rgb, pattern), np.full((6, 6), 777, np.uint16))


def test_rgb_to_bayer_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        sf.rgb_to_bayer(np.zeros((4, 4, 3), np.uint16), "ryyb")
    with pytest.raises(ValueError):
        sf.rgb_to_bayer(np.zeros((4, 4), np.uint16), "rggb")


def test_odd_frame_dimensions() -> None:
    frame = sf.odd_frame()
    assert frame.shape == (5, 7)
    assert frame[0, 0] == 0
    frame = sf.odd_frame((9, 3))
    assert frame.shape == (9, 3)


def test_tall_bayer_frame_2185_rows() -> None:
    frame = sf.tall_bayer_frame()
    assert frame.shape == (2185, 8)
    assert frame.dtype == np.uint16
    for pattern in sf.BAYER_PATTERNS:
        tall = sf.tall_bayer_frame(height=2185, width=4, pattern=pattern)
        assert tall.shape == (2185, 4)
        np.testing.assert_array_equal(tall, sf.ramp_frame((2185, 4)))


def test_hot_dead_pixels() -> None:
    frame = sf.hot_dead_pixels((8, 8), hot=(2, 3), dead=(5, 1))
    assert frame[2, 3] == sf.UINT16_MAX
    assert frame[5, 1] == 0
    masked = frame.copy()
    masked[2, 3] = 8192
    masked[5, 1] = 8192
    assert np.all(masked == 8192)


def test_slanted_edge_rgb_exact() -> None:
    edge = sf.slanted_edge_rgb((4, 8))
    assert edge.shape == (4, 8, 3)
    plane = edge[..., 0]
    expected = np.array(
        [
            [0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1],
        ],
        dtype=np.uint16,
    ) * sf.UINT16_MAX
    np.testing.assert_array_equal(plane, expected)
    np.testing.assert_array_equal(edge[..., 1], plane)
    np.testing.assert_array_equal(edge[..., 2], plane)


def test_checker_frame_exact() -> None:
    board = sf.checker_frame((4, 6), tile=1, low=100, high=200)
    assert board.shape == (4, 6)
    assert board.dtype == np.uint16
    expected = np.array(
        [
            [100, 200, 100, 200, 100, 200],
            [200, 100, 200, 100, 200, 100],
            [100, 200, 100, 200, 100, 200],
            [200, 100, 200, 100, 200, 100],
        ],
        dtype=np.uint16,
    )
    np.testing.assert_array_equal(board, expected)
    tiled = sf.checker_frame((4, 4), tile=2, low=0, high=7)
    np.testing.assert_array_equal(
        tiled,
        np.array(
            [
                [0, 0, 7, 7],
                [0, 0, 7, 7],
                [7, 7, 0, 0],
                [7, 7, 0, 0],
            ],
            dtype=np.uint16,
        ),
    )


def test_noisy_frame_seeded_reproducibility() -> None:
    a = sf.noisy_frame((16, 16), seed=42)
    b = sf.noisy_frame((16, 16), seed=42)
    c = sf.noisy_frame((16, 16), seed=43)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.dtype == np.uint16
    assert a.std() > 0
