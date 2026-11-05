from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import convolve

from modules.demosaic.bilinear_demosaic import BilinearDemosaic
from util.native_array import as_float32
from util.native_output import preview_from_native
from util.output_contract import OutputSpec


def reference_bilinear(raw: np.ndarray, masks: tuple[np.ndarray, ...]) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32)
    r, g, b = (m.astype(np.float32) for m in masks)
    r0, g0, b0 = raw * r, raw * g, raw * b
    cross = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.float32)
    h = np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]], np.float32) * .5
    v = h.T
    d = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], np.float32) * .25

    def interp(values: np.ndarray, mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        numerator = convolve(values, kernel, mode="mirror")
        denominator = convolve(mask, kernel, mode="mirror")
        out = np.zeros_like(numerator)
        np.divide(numerator, denominator, out=out, where=denominator > 0)
        return out

    green = g0 + (1.0 - g) * interp(g0, g, cross)
    has_r = convolve(r, h * 2, mode="constant", cval=0) > 0
    has_b = convolve(b, h * 2, mode="constant", cval=0) > 0
    rh, rv, rd = interp(r0, r, h), interp(r0, r, v), interp(r0, r, d)
    bh, bv, bd = interp(b0, b, h), interp(b0, b, v), interp(b0, b, d)
    red, blue = r0.copy(), b0.copy()
    at_g = g.astype(bool)
    red[at_g & has_r] = rh[at_g & has_r]
    red[at_g & ~has_r] = rv[at_g & ~has_r]
    red[b.astype(bool)] = rd[b.astype(bool)]
    blue[at_g & has_b] = bh[at_g & has_b]
    blue[at_g & ~has_b] = bv[at_g & ~has_b]
    blue[r.astype(bool)] = bd[r.astype(bool)]
    return np.stack((red, green, blue), axis=-1).astype(np.float32)


def masks_for(shape: tuple[int, int], pattern: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masks = {c: np.zeros(shape, dtype=bool) for c in "rgb"}
    for c, (y, x) in zip(pattern, ((0, 0), (0, 1), (1, 0), (1, 1))):
        masks[c][y::2, x::2] = True
    return masks["r"], masks["g"], masks["b"]


@pytest.mark.parametrize("shape", ((1, 1), (1, 7), (8, 1), (2, 2), (2, 3), (3, 2), (3, 3), (5, 9), (12, 16), (17, 13)))
@pytest.mark.parametrize("pattern", ("rggb", "grbg", "gbrg", "bggr"))
@pytest.mark.parametrize("data", ("random_hdr", "constant", "ramp"))
def test_bilinear_exact_reference(shape: tuple[int, int], pattern: str, data: str) -> None:
    if data == "random_hdr":
        raw = (np.random.default_rng(2026).standard_normal(shape) * 800000).astype(np.float32)
    elif data == "constant":
        raw = np.full(shape, 131070, np.float32)
    else:
        raw = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) * np.float32(.375)
    masks = masks_for(shape, pattern)
    raw.setflags(write=False)
    for mask in masks:
        mask.setflags(write=False)
    expected = reference_bilinear(raw, masks)
    actual = BilinearDemosaic(raw, masks).apply_bilinear()
    assert actual.dtype == np.float32
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("kind", ("weighted", "irregular", "empty_channel", "mixed_signed"))
def test_bilinear_general_mask_reference(kind: str) -> None:
    shape = (9, 13)
    raw = np.random.default_rng(99).uniform(-1000, 300000, shape).astype(np.float32)
    r, g, b = (m.astype(np.float32) for m in masks_for(shape, "gbrg"))
    masks = r, g, b
    if kind == "weighted":
        masks[0][:] *= .3
        masks[2][:] *= -.25
    elif kind == "irregular":
        masks[0][3, 5] = .5
        masks[1][7, 9] = 0
    elif kind == "empty_channel":
        masks[2][:] = 0
    else:
        masks[0][1, 2] = -1
    actual = BilinearDemosaic(raw, masks).apply_bilinear()
    assert actual.tobytes() == reference_bilinear(raw, masks).tobytes()


def test_bilinear_periodic_geometry_reduces_dense_convolutions(monkeypatch) -> None:
    from modules.demosaic import bilinear_demosaic as module
    shapes = []
    def traced(values, *args, **kwargs):
        shapes.append(values.shape)
        return convolve(values, *args, **kwargs)
    monkeypatch.setattr(module, "convolve", traced)
    raw = np.ones((9, 13), np.float32)
    demosaic = BilinearDemosaic(raw, masks_for((9, 13), "gbrg"))
    assert demosaic._periodic_mask_tiles() is not None
    demosaic.apply_bilinear()
    assert shapes.count((9, 13)) == 7
    assert shapes.count((2, 2)) == 7


def test_native_float32_identity_and_nonfinite_guards() -> None:
    values = np.arange(80, dtype=np.float32).reshape(8, 10)[::2, ::2]
    values.setflags(write=False)
    assert as_float32(values) is values
    for value in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="nonfinite"):
            as_float32(np.array([value], np.float32), name="probe")
    with pytest.raises(ValueError, match="float32"):
        as_float32(np.array([np.finfo(np.float64).max]), name="probe")
    foreign = np.array([1, 2, 3], dtype=">f4")
    converted = as_float32(foreign)
    assert converted.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(converted, foreign)


def test_srgb8_preview_exact_and_owned() -> None:
    native = np.arange(768, dtype=np.uint16).astype(np.uint8).reshape(16, 16, 3)[:, ::-1]
    native.setflags(write=False)
    spec = OutputSpec.from_mapping({"format": "png", "model": "rgb", "bit_depth": 8, "transfer": "srgb"})
    preview = preview_from_native(native, spec)
    np.testing.assert_array_equal(preview, native)
    assert preview.dtype == np.uint8
    assert not np.shares_memory(preview, native)


def test_srgb8_preview_skips_requantization(monkeypatch) -> None:
    import util.native_output as module
    def unexpected(*args, **kwargs):
        raise AssertionError("sRGB8 preview must not requantize")
    monkeypatch.setattr(module, "quantize_rgb", unexpected)
    spec = OutputSpec.from_mapping({"format": "png", "model": "rgb", "bit_depth": 8, "transfer": "srgb"})
    native = np.full((2, 3, 3), 127, np.uint8)
    np.testing.assert_array_equal(preview_from_native(native, spec), native)
