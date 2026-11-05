"""Deterministic synthetic image fixtures for pipeline regression tests.

All helpers are pure NumPy, deterministic, and cheap to construct so tests can
assert exact values rather than recorded outputs.  Bayer patterns are named by
the 2x2 cell at the top-left of the frame: ``rggb`` means R at (0, 0), Gr at
(0, 1), Gb at (1, 0), B at (1, 1).
"""

from __future__ import annotations

import numpy as np

BAYER_PATTERNS = ("rggb", "bggr", "grbg", "gbrg")

_PATTERN_LAYOUT = {
    "rggb": ((0, 1), (1, 2)),
    "bggr": ((2, 1), (1, 0)),
    "grbg": ((1, 0), (2, 1)),
    "gbrg": ((1, 2), (0, 1)),
}

UINT16_MAX = np.iinfo(np.uint16).max
UINT24_MAX = (1 << 24) - 1


def constant_frame(shape: tuple[int, int], value: int, dtype=np.uint16) -> np.ndarray:
    """Uniform Bayer/luma frame filled with ``value``."""
    return np.full(shape, value, dtype=dtype)


def black_frame(shape: tuple[int, int], dtype=np.uint16) -> np.ndarray:
    return constant_frame(shape, 0, dtype)


def white_frame(shape: tuple[int, int], dtype=np.uint16) -> np.ndarray:
    return constant_frame(shape, np.iinfo(dtype).max, dtype)


def ramp_frame(
    shape: tuple[int, int],
    dtype: type[np.uint16] | type[np.uint32] = np.uint16,
) -> np.ndarray:
    """Row-major linear ramp ``0 .. count - 1`` clipped to the dtype range.

    For ``np.uint32`` the ramp is rescaled to always span the full 24-bit
    code range ``0 .. 0xFFFFFF`` regardless of frame size (count > 1).
    """
    count = int(shape[0]) * int(shape[1])
    values = np.arange(count, dtype=np.uint64)
    if dtype == np.uint32 and count > 1:
        values = np.floor(values * (np.uint64(UINT24_MAX) / np.uint64(count - 1))).astype(np.uint64)
    info_max = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else UINT24_MAX
    values = np.clip(values, 0, info_max)
    return values.astype(dtype).reshape(shape)


def primaries_rgb(shape: tuple[int, int], dtype=np.uint16) -> np.ndarray:
    """Vertical stripes of saturated R, G, B plus a neutral grey stripe."""
    h, w = shape
    maxv = np.iinfo(dtype).max
    rgb = np.zeros((h, w, 3), dtype=dtype)
    third = max(w // 4, 1)
    rgb[:, :third] = (maxv, 0, 0)
    rgb[:, third:2 * third] = (0, maxv, 0)
    rgb[:, 2 * third:3 * third] = (0, 0, maxv)
    rgb[:, 3 * third:] = (maxv // 2, maxv // 2, maxv // 2)
    return rgb


def neutral_rgb(shape: tuple[int, int], level: int = 32768, dtype=np.uint16) -> np.ndarray:
    """Uniform neutral RGB patch (R == G == B == ``level``)."""
    h, w = shape
    rgb = np.empty((h, w, 3), dtype=dtype)
    rgb[...] = level
    return rgb


def bayer_channel(pattern: str, row: int, col: int) -> int:
    """RGB channel index sampled by ``pattern`` at pixel ``(row, col)``."""
    return _PATTERN_LAYOUT[pattern][row % 2][col % 2]


def rgb_to_bayer(rgb: np.ndarray, pattern: str = "rggb") -> np.ndarray:
    """Mosaic an HxWx3 RGB image into a single-plane CFA frame.

    Each output pixel copies the channel that the pattern samples at that
    parity, so a uniform R stripe reproduces R values only on R sites.
    """
    if pattern not in _PATTERN_LAYOUT:
        raise ValueError(f"unknown Bayer pattern: {pattern}")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must have shape (H, W, 3)")
    h, w = rgb.shape[:2]
    out = np.empty((h, w), dtype=rgb.dtype)
    for parity_r in range(2):
        for parity_c in range(2):
            channel = _PATTERN_LAYOUT[pattern][parity_r][parity_c]
            out[parity_r::2, parity_c::2] = rgb[parity_r::2, parity_c::2, channel]
    return out


def odd_frame(shape: tuple[int, int] = (5, 7), dtype=np.uint16) -> np.ndarray:
    """Small odd-dimension ramp fixture; both extents are odd."""
    assert shape[0] % 2 == 1 and shape[1] % 2 == 1
    return ramp_frame(shape, dtype)


def tall_bayer_frame(height: int = 2185, width: int = 8,
                     pattern: str = "rggb", dtype=np.uint16) -> np.ndarray:
    """Tall narrow mosaic for exercising non-standard row counts cheaply."""
    rgb = ramp_frame((height, width), dtype)
    stacked = np.repeat(rgb[..., None], 3, axis=2)
    return rgb_to_bayer(stacked, pattern)


def hot_dead_pixels(shape: tuple[int, int],
                    background: int = 8192, dtype=np.uint16,
                    hot: tuple[int, int] = (2, 2),
                    dead: tuple[int, int] = (3, 5)) -> np.ndarray:
    """Uniform CFA frame with one saturated (hot) and one zero (dead) pixel."""
    frame = constant_frame(shape, background, dtype)
    frame[hot] = np.iinfo(dtype).max
    frame[dead] = 0
    return frame


def slanted_edge_rgb(shape: tuple[int, int], dtype=np.uint16) -> np.ndarray:
    """Grayscale RGB slanted edge: white where ``col >= w // 2 + row // 2``."""
    h, w = shape
    maxv = np.iinfo(dtype).max
    rows, cols = np.indices((h, w))
    edge = np.where(cols >= w // 2 + rows // 2, maxv, 0).astype(dtype)
    return np.repeat(edge[..., None], 3, axis=2)


def checker_frame(shape: tuple[int, int], tile: int = 1, low: int = 4096,
                  high: int = 61440, dtype=np.uint16) -> np.ndarray:
    """Single-plane checkerboard alternating ``low``/``high`` every ``tile`` px."""
    rows, cols = np.indices(shape)
    board = (rows // tile + cols // tile) % 2
    return np.where(board == 0, low, high).astype(dtype)


def noisy_frame(shape: tuple[int, int], seed: int = 0, mean: float = 8192.0,
                sigma: float = 256.0, dtype=np.uint16) -> np.ndarray:
    """Gaussian noise frame from a seeded generator; reproducible by ``seed``."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(mean, sigma, size=shape)
    info = np.iinfo(dtype)
    return np.clip(np.rint(noise), 0, info.max).astype(dtype)
