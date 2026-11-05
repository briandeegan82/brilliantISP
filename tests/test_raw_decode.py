from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
from typing import Any, Literal, cast

from util.raw_decode import (
    DecodedRaw,
    RawFormat,
    RawFormatError,
    canonical_storage_format,
    decode_raw,
    decode_raw_bytes,
    flip_bayer_pattern,
    shift_bayer_pattern,
    storage_bytes_per_sample,
)


def _fmt(**kw: Any) -> RawFormat:
    base: dict[str, Any] = dict(width=4, height=3, bayer_pattern="rggb", bit_depth=12,
                                storage_format="uint16")
    base.update(kw)
    return RawFormat(**base)



@pytest.mark.parametrize("order", ["little", "big"])
def test_packed24_vectors(order: str) -> None:
    samples = [0x000000, 0x000001, 0x123456, 0xABCDEF, 0xFFFFFF]
    blob = b"".join(
        s.to_bytes(3, cast(Literal["little", "big"], order)) for s in samples
    )
    fmt = RawFormat(width=5, height=1, bayer_pattern="rggb", bit_depth=24,
                    storage_format="packed24", byte_order=order)
    dec = decode_raw_bytes(blob, fmt)
    assert dec.pixels.dtype == np.uint32
    np.testing.assert_array_equal(dec.pixels.flatten(), np.array(samples, np.uint32))


def test_uint16_boundaries() -> None:
    samples = [0, 1, 0x8000, 0xFFFF]
    fmt = _fmt(width=4, height=1, bit_depth=16)
    dec = decode_raw_bytes(struct.pack("<4H", *samples), fmt)
    np.testing.assert_array_equal(dec.pixels.flatten(), np.array(samples, np.uint16))


def test_uint32_boundaries_native_endian() -> None:
    samples = [0, 1, 0x80000000, 0xFFFFFFFF]
    for order, dtype_char in (("little", "<4I"), ("big", ">4I")):
        fmt = _fmt(width=4, height=1, bit_depth=32, storage_format="uint32",
                   byte_order=order)
        dec = decode_raw_bytes(struct.pack(dtype_char, *samples), fmt)
        assert dec.pixels.dtype == np.uint32
        np.testing.assert_array_equal(dec.pixels.flatten(),
                                      np.array(samples, np.uint32))


def test_word_pair32_both_orders() -> None:
    for word_order, expected in (
        ("low_first", 0x5678 | (0x1234 << 16)),
        ("high_first", (0x5678 << 16) | 0x1234),
    ):
        fmt = _fmt(width=1, height=1, bit_depth=32,
                   storage_format="word_pair32", word_order=word_order)
        blob = struct.pack("<2H", 0x5678, 0x1234)
        dec = decode_raw_bytes(blob, fmt)
        assert dec.pixels[0, 0] == expected
    fmt_be = _fmt(width=1, height=1, bit_depth=32,
                  storage_format="word_pair32", word_order="low_first")
    dec = decode_raw_bytes(struct.pack("<2H", 0x5678, 0x1234), fmt_be)
    assert dec.pixels[0, 0] == 0x12345678


def test_word_pair32_byte_order() -> None:
    fmt = _fmt(width=1, height=1, bit_depth=32, storage_format="word_pair32",
               byte_order="big", word_order="low_first")
    blob = struct.pack(">2H", 0x5678, 0x1234)
    dec = decode_raw_bytes(blob, fmt)
    assert dec.pixels[0, 0] == 0x12345678


def test_header_and_row_padding() -> None:
    w, h, pad = 3, 2, 2
    stride = w * 2 + pad
    header = b"HDR4HDR4"
    rows = [np.array([10, 20, 30], np.uint16), np.array([40, 50, 60], np.uint16)]
    blob = header + b"".join(
        r.astype("<u2").tobytes() + b"\xAA" * pad for r in rows
    )
    fmt = _fmt(width=w, height=h, header_bytes=8, row_stride_bytes=stride)
    dec = decode_raw_bytes(blob, fmt)
    np.testing.assert_array_equal(dec.pixels, np.stack(rows))


def test_short_and_surplus_rejected() -> None:
    fmt = _fmt(width=2, height=2)
    with pytest.raises(RawFormatError, match="exactly"):
        decode_raw_bytes(b"\x00" * 7, fmt)
    with pytest.raises(RawFormatError, match="exactly"):
        decode_raw_bytes(b"\x00" * 9, fmt)


def test_msb_and_manual_alignment() -> None:
    code = 0x0ABC
    stored = code << 4
    fmt_msb = _fmt(width=1, height=1, bit_depth=12, data_alignment="MSB")
    dec = decode_raw_bytes(struct.pack("<H", stored), fmt_msb)
    assert dec.pixels[0, 0] == code

    fmt_man = _fmt(width=1, height=1, bit_depth=12, data_alignment="MANUAL",
                   manual_bit_shift=4)
    dec = decode_raw_bytes(struct.pack("<H", stored), fmt_man)
    assert dec.pixels[0, 0] == code

    fmt_man0 = _fmt(width=1, height=1, bit_depth=12, data_alignment="MANUAL",
                    manual_bit_shift=0)
    dec = decode_raw_bytes(struct.pack("<H", 0x0ABC), fmt_man0)
    assert dec.pixels[0, 0] == 0x0ABC

    dec = decode_raw_bytes(struct.pack("<H", 0xFFFF), fmt_man0)
    assert dec.pixels[0, 0] == 0x0FFF


def test_msb_12_in_24() -> None:
    code = 0xABC
    fmt = _fmt(width=1, height=1, bit_depth=12, storage_format="packed24",
               data_alignment="MSB")
    blob = (code << 12).to_bytes(3, "little")
    dec = decode_raw_bytes(blob, fmt)
    assert dec.pixels[0, 0] == code


def test_uint8_promoted_to_uint16() -> None:
    fmt = _fmt(width=2, height=1, bit_depth=8, storage_format="uint8")
    dec = decode_raw_bytes(bytes([1, 200]), fmt)
    assert dec.pixels.dtype == np.uint16
    np.testing.assert_array_equal(dec.pixels.flatten(), np.array([1, 200]))



@pytest.mark.parametrize("kw", [
    dict(width=0), dict(width=4.0), dict(width=True),
    dict(height=-1), dict(height=2.5),
    dict(bayer_pattern="gggg"),
    dict(bit_depth=0), dict(bit_depth=20),
    dict(bit_depth=3.0), dict(bit_depth=True),
    dict(byte_order="weird"),
    dict(data_alignment="SIDE"),
    dict(data_alignment="MANUAL", manual_bit_shift=None),
    dict(data_alignment="MANUAL", manual_bit_shift=99),
    dict(header_bytes=-1),
    dict(row_stride_bytes=3),
    dict(row_stride_bytes=4.5),
    dict(word_order="mixed"),
    dict(horizontal_flip=1),
    dict(vertical_flip="yes"),
    dict(embedded_rows_top=-1),
    dict(embedded_rows_top=2, embedded_rows_bottom=1),
    dict(source_format="bmp"),
    dict(storage_format="int8"),
])
def test_invalid_format_fields(kw) -> None:
    with pytest.raises(RawFormatError) as exc:
        _fmt(**kw)
    assert exc.value.field


def test_storage_aliases_and_defaults() -> None:
    assert canonical_storage_format("u8", 8) == "uint8"
    assert canonical_storage_format("raw16", 12) == "uint16"
    assert canonical_storage_format("3byte", 24) == "packed24"
    assert canonical_storage_format("u32", 32) == "uint32"
    assert canonical_storage_format(None, 8) == "uint8"
    assert canonical_storage_format(None, 12) == "uint16"
    assert canonical_storage_format(None, 24) == "uint32"
    assert storage_bytes_per_sample("packed24", 24) == 3
    assert storage_bytes_per_sample("word_pair32", 20) == 4
    with pytest.raises(RawFormatError):
        canonical_storage_format("bogus", 12)


def test_defaulted_storage_warns() -> None:
    fmt = RawFormat.from_sensor_info(
        {"width": 2, "height": 1, "bayer_pattern": "rggb", "bit_depth": 12}
    )
    assert fmt.storage_format == "uint16" and fmt.storage_defaulted
    dec = decode_raw_bytes(struct.pack("<2H", 5, 6), fmt)
    assert any("defaulted" in w for w in dec.warnings)


def test_from_sensor_info_storage_priority() -> None:
    sensor = {"width": 2, "height": 1, "bayer_pattern": "rggb",
              "bit_depth": 12, "data_format": "raw16",
              "storage_format": "packed24", "endian_type": "ieee-be",
              "data_alignment": "MSB"}
    fmt = RawFormat.from_sensor_info(sensor)
    assert fmt.storage_format == "packed24"
    assert fmt.byte_order == "big"
    assert fmt.data_alignment == "MSB"



def _pattern_channels(pattern: str, w: int, h: int) -> np.ndarray:
    grid = {"rggb": ("rg", "gb"), "grbg": ("gr", "bg"),
            "gbrg": ("gb", "rg"), "bggr": ("bg", "gr")}[pattern]
    chars = np.empty((h, w), dtype="<U1")
    for y in range(h):
        for x in range(w):
            chars[y, x] = grid[y % 2][x % 2]
    return chars


@pytest.mark.parametrize("pattern", ["rggb", "grbg", "gbrg", "bggr"])
@pytest.mark.parametrize("w,h", [(5, 3), (4, 4)])
@pytest.mark.parametrize("hf,vf", [(False, False), (True, False),
                                    (False, True), (True, True)])
def test_flip_pattern_oracle(pattern: str, w: int, h: int, hf: bool, vf: bool) -> None:
    channels = _pattern_channels(pattern, w, h)
    flipped = channels
    if hf:
        flipped = np.fliplr(flipped)
    if vf:
        flipped = np.flipud(flipped)
    expected = _pattern_channels(
        flip_bayer_pattern(pattern, w, h, hf, vf), w, h
    )
    np.testing.assert_array_equal(flipped, expected)


@pytest.mark.parametrize("pattern", ["rggb", "grbg", "gbrg", "bggr"])
def test_shift_pattern_oracle(pattern: str) -> None:
    w = h = 8
    channels = _pattern_channels(pattern, w, h)
    for xo in (0, 1, 2, 3):
        for yo in (0, 1, 2, 3):
            sub = channels[yo:, xo:]
            expected = _pattern_channels(
                shift_bayer_pattern(pattern, xo, yo), sub.shape[1], sub.shape[0]
            )
            np.testing.assert_array_equal(sub, expected)


def test_shift_invalid_pattern() -> None:
    with pytest.raises(RawFormatError):
        shift_bayer_pattern("xtrans", 0, 0)



def test_embedded_trim_and_flip_phase() -> None:
    w, h, top, bot = 4, 6, 2, 1
    img = np.arange(w * h, dtype=np.uint16).reshape(h, w)
    fmt = _fmt(width=w, height=h, embedded_rows_top=top,
               embedded_rows_bottom=bot, horizontal_flip=True,
               bit_depth=16)
    dec = decode_raw_bytes(img.astype("<u2").tobytes(), fmt)
    expected_pixels = np.fliplr(img[top:h - bot])
    np.testing.assert_array_equal(dec.pixels, expected_pixels)
    assert (dec.width, dec.height) == (w, h - top - bot)
    expected_pat = flip_bayer_pattern(
        shift_bayer_pattern("rggb", 0, top), w, h - top - bot, True, False
    )
    assert dec.bayer_pattern == expected_pat


def test_decode_does_not_mutate_input() -> None:
    img = np.arange(12, dtype=np.uint16).reshape(3, 4)
    blob = img.astype("<u2").tobytes()
    before = blob[:]
    fmt = _fmt(horizontal_flip=True, vertical_flip=True)
    decode_raw_bytes(blob, fmt)
    assert blob == before



def _write_pgm(path: Path, pixels: np.ndarray, maxval: int,
               separator: bytes = b"\n", comments: bool = False) -> None:
    header = b"P5 "
    if comments:
        header += b"#comment\n"
    header += (f"{pixels.shape[1]} {pixels.shape[0]}".encode() + b"\n"
               + str(maxval).encode() + separator)
    if maxval > 255:
        raster = pixels.astype(">u2").tobytes()
    else:
        raster = pixels.astype(np.uint8).tobytes()
    path.write_bytes(header + raster)


def test_pgm_16bit_always_big_endian(tmp_path: Path) -> None:
    img = np.array([[0, 0x1234], [0xABCD, 0x0001]], np.uint16)
    p = tmp_path / "f.pgm"
    _write_pgm(p, img, 65535)
    fmt = _fmt(width=2, height=2, bit_depth=16, byte_order="little",
               source_format="pgm")
    dec = decode_raw(p, fmt)
    np.testing.assert_array_equal(dec.pixels, img)
    assert dec.container == "pgm"


def test_pgm_8bit(tmp_path: Path) -> None:
    img = np.array([[0, 255], [10, 20]], np.uint8)
    p = tmp_path / "f.pgm"
    _write_pgm(p, img, 255)
    fmt = _fmt(width=2, height=2, bit_depth=8, source_format="pgm")
    dec = decode_raw(p, fmt)
    np.testing.assert_array_equal(dec.pixels, img.astype(np.uint16))


def test_pgm_crlf_and_comments_and_ws_first_byte(tmp_path: Path) -> None:
    img = np.array([[10, 20]], np.uint8)
    p = tmp_path / "f.pgm"
    _write_pgm(p, img, 255, separator=b"\r\n", comments=True)
    fmt = _fmt(width=2, height=1, bit_depth=8, source_format="pgm")
    dec = decode_raw(p, fmt)
    np.testing.assert_array_equal(dec.pixels, img.astype(np.uint16))


def test_p2_range_and_count(tmp_path: Path) -> None:
    p = tmp_path / "f.pgm"
    p.write_bytes(b"P2\n2 1\n255\n0 300\n")
    fmt = _fmt(width=2, height=1, bit_depth=8, source_format="pgm")
    with pytest.raises(ValueError, match="out of range"):
        decode_raw(p, fmt)
    p.write_bytes(b"P2\n2 1\n255\n0 -1\n")
    with pytest.raises(ValueError):
        decode_raw(p, fmt)
    p.write_bytes(b"P2\n2 1\n255\n5\n")
    with pytest.raises(ValueError, match="too short"):
        decode_raw(p, fmt)
    p.write_bytes(b"P2\n2 1\n255\n0 1 2\n")
    with pytest.raises(ValueError, match="surplus"):
        decode_raw(p, fmt)


def test_pgm_geometry_and_stride_conflicts(tmp_path: Path) -> None:
    img = np.zeros((2, 2), np.uint8)
    p = tmp_path / "f.pgm"
    _write_pgm(p, img, 255)
    with pytest.raises(RawFormatError, match="geometry"):
        decode_raw(p, _fmt(width=3, height=2, bit_depth=8,
                           source_format="pgm"))
    with pytest.raises(RawFormatError, match="container"):
        decode_raw(p, _fmt(width=2, height=2, bit_depth=8, header_bytes=4,
                           source_format="pgm"))
    with pytest.raises(RawFormatError, match="cannot fit"):
        decode_raw(p, _fmt(width=2, height=2, bit_depth=16,
                           source_format="pgm"))


def test_auto_source_selection_and_unknown(tmp_path: Path) -> None:
    raw = tmp_path / "f.raw"
    raw.write_bytes(struct.pack("<4H", 1, 2, 3, 4))
    fmt = _fmt(width=2, height=2)
    assert decode_raw(raw, fmt).container == "raw"
    png = tmp_path / "f.png"
    png.write_bytes(b"\x89PNG" + b"\x00" * 20)
    with pytest.raises(RawFormatError, match="source_format"):
        decode_raw(png, fmt)


def test_tiff_2d(tmp_path: Path) -> None:
    tifffile = pytest.importorskip("tifffile")
    img = np.arange(12, dtype=np.uint16).reshape(3, 4)
    p = tmp_path / "f.tiff"
    tifffile.imwrite(p, img)
    fmt = _fmt(bit_depth=16, source_format="tiff")
    dec = decode_raw(p, fmt)
    np.testing.assert_array_equal(dec.pixels, img)
    assert dec.container == "tiff"


def test_tiff_rgb_and_multipage_rejected(tmp_path: Path) -> None:
    tifffile = pytest.importorskip("tifffile")
    rgb = np.zeros((3, 4, 3), np.uint8)
    p = tmp_path / "rgb.tiff"
    tifffile.imwrite(p, rgb)
    fmt = _fmt(bit_depth=8, source_format="tiff")
    with pytest.raises(RawFormatError, match="2-D|shape"):
        decode_raw(p, fmt)
    p2 = tmp_path / "multi.tiff"
    tifffile.imwrite(p2, np.zeros((4, 4), np.uint8))
    tifffile.imwrite(p2, np.ones((4, 4), np.uint8), append=True)
    with pytest.raises(RawFormatError, match="one"):
        decode_raw(p2, _fmt(bit_depth=8, source_format="tiff"))


def test_tiff_signed_float_rejected(tmp_path: Path) -> None:
    tifffile = pytest.importorskip("tifffile")
    p = tmp_path / "f.tiff"
    tifffile.imwrite(p, np.zeros((3, 4), np.float32))
    with pytest.raises(RawFormatError, match="dtype"):
        decode_raw(p, _fmt(bit_depth=8, source_format="tiff"))


class _StubRaw:
    def __init__(self, shape=(3, 4), colors=None, desc=b"RGBG",
                 white_level=4095, raw_pattern=None, dtype=np.uint16):
        self.raw_image_visible = np.arange(
            shape[0] * shape[1], dtype=dtype
        ).reshape(shape)
        if colors is None:
            colors = np.array([[0, 1], [1, 2]], np.uint8)
        self.raw_colors_visible = colors
        if raw_pattern is None:
            raw_pattern = np.zeros((2, 2), np.uint8)
        self.raw_pattern = raw_pattern
        self.color_desc = desc
        self.white_level = white_level
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.closed = True
        return False


def test_rawpy_stub_decode(tmp_path: Path, monkeypatch) -> None:
    import sys
    import types

    stub = _StubRaw()
    module = types.SimpleNamespace(imread=lambda p: stub)
    monkeypatch.setitem(sys.modules, "rawpy", module)
    fmt = _fmt(bit_depth=12, source_format="rawpy")
    dec = decode_raw(tmp_path / "f.dng", fmt)
    assert stub.closed
    assert dec.container == "rawpy"
    assert dec.pixels.shape == (3, 4)
    assert dec.bayer_pattern == "rggb"
    assert not np.shares_memory(dec.pixels, stub.raw_image_visible)


def test_rawpy_g_index3_pattern(tmp_path: Path, monkeypatch) -> None:
    import sys
    import types

    colors = np.array([[0, 3], [3, 2]], np.uint8)
    stub = _StubRaw(colors=colors)
    monkeypatch.setitem(sys.modules, "rawpy",
                        types.SimpleNamespace(imread=lambda p: stub))
    dec = decode_raw(tmp_path / "f.dng",
                     _fmt(bit_depth=12, source_format="rawpy"))
    assert dec.bayer_pattern == "rggb"


def test_rawpy_xtrans_and_mismatch_rejected(tmp_path: Path, monkeypatch) -> None:
    import sys
    import types

    xtrans = _StubRaw(
        colors=np.array([[0, 1], [1, 2]], np.uint8),
        raw_pattern=np.zeros((6, 6), np.uint8),
    )
    monkeypatch.setitem(sys.modules, "rawpy",
                        types.SimpleNamespace(imread=lambda p: xtrans))
    with pytest.raises(RawFormatError):
        decode_raw(tmp_path / "f.raf",
                   _fmt(bit_depth=12, source_format="rawpy"))

    wrong = _StubRaw(shape=(2, 2))
    monkeypatch.setitem(sys.modules, "rawpy",
                        types.SimpleNamespace(imread=lambda p: wrong))
    with pytest.raises(RawFormatError, match="geometry"):
        decode_raw(tmp_path / "f.dng",
                   _fmt(bit_depth=12, source_format="rawpy"))

    depth = _StubRaw(white_level=255)
    monkeypatch.setitem(sys.modules, "rawpy",
                        types.SimpleNamespace(imread=lambda p: depth))
    with pytest.raises(RawFormatError, match="bit_depth"):
        decode_raw(tmp_path / "f.dng",
                   _fmt(bit_depth=12, source_format="rawpy"))


class _NoPatternRaw(_StubRaw):
    def __init__(self):
        super().__init__()
        del self.raw_pattern


def test_rawpy_missing_pattern_and_bad_inputs(tmp_path: Path, monkeypatch) -> None:
    import sys
    import types

    missing = _NoPatternRaw()
    monkeypatch.setitem(sys.modules, "rawpy",
                        types.SimpleNamespace(imread=lambda p: missing))
    with pytest.raises(RawFormatError, match="CFA|XTrans|2x2"):
        decode_raw(tmp_path / "f.dng",
                   _fmt(bit_depth=12, source_format="rawpy"))

    bad = _StubRaw(dtype=cast(Any, np.int16))
    monkeypatch.setitem(sys.modules, "rawpy",
                        types.SimpleNamespace(imread=lambda p: bad))
    with pytest.raises(RawFormatError, match="pixels|dtype"):
        decode_raw(tmp_path / "f.dng",
                   _fmt(bit_depth=12, source_format="rawpy"))

    white = _StubRaw(white_level=1 << 20)
    monkeypatch.setitem(sys.modules, "rawpy",
                        types.SimpleNamespace(imread=lambda p: white))
    with pytest.raises(RawFormatError, match="white level"):
        decode_raw(tmp_path / "f.dng",
                   _fmt(bit_depth=12, source_format="rawpy"))


def test_rawpy_non_lsb_rejected(tmp_path: Path, monkeypatch) -> None:
    import sys
    import types

    stub = _StubRaw()
    monkeypatch.setitem(sys.modules, "rawpy",
                        types.SimpleNamespace(imread=lambda p: stub))
    with pytest.raises(RawFormatError, match="LSB"):
        decode_raw(tmp_path / "f.dng",
                   _fmt(bit_depth=12, source_format="rawpy",
                        data_alignment="MSB"))


def test_raw_file_source_identity(tmp_path: Path) -> None:
    raw = tmp_path / "f.raw"
    raw.write_bytes(struct.pack("<4H", 1, 2, 3, 4))
    dec = decode_raw(raw, _fmt(width=2, height=2))
    assert dec.source_identity == str(raw.resolve())


def test_auto_rawpy_rejects_alignment_before_read(tmp_path: Path, monkeypatch) -> None:
    import sys
    import types

    def unexpected_read(path):
        pytest.fail("invalid rawpy alignment must fail before reading the source")

    monkeypatch.setitem(sys.modules, "rawpy", types.SimpleNamespace(imread=unexpected_read))
    with pytest.raises(RawFormatError, match="LSB"):
        decode_raw(tmp_path / "f.dng", _fmt(data_alignment="MSB"))


def test_from_sensor_info_strict_fields() -> None:
    with pytest.raises(RawFormatError, match="width"):
        RawFormat.from_sensor_info({"height": 1, "bit_depth": 12,
                                    "bayer_pattern": "rggb"})
    with pytest.raises(RawFormatError, match="storage_format"):
        RawFormat.from_sensor_info({"width": 2, "height": 1, "bit_depth": 12,
                                    "bayer_pattern": "rggb",
                                    "storage_format": None})
    with pytest.raises(RawFormatError, match="data_format"):
        RawFormat.from_sensor_info({"width": 2, "height": 1, "bit_depth": 12,
                                    "bayer_pattern": "rggb",
                                    "data_format": ""})
    with pytest.raises(RawFormatError, match="data_alignment"):
        RawFormat.from_sensor_info({"width": 2, "height": 1, "bit_depth": 12,
                                    "bayer_pattern": "rggb",
                                    "data_alignment": None})
    with pytest.raises(RawFormatError, match="bit_depth"):
        RawFormat.from_sensor_info({"width": 2, "height": 1,
                                    "bit_depth": "12",
                                    "bayer_pattern": "rggb"})


def test_helpers_validate_args() -> None:
    with pytest.raises(RawFormatError):
        shift_bayer_pattern("rggb", -1, 0)
    with pytest.raises(RawFormatError):
        shift_bayer_pattern("rggb", cast(Any, 0.5), 0)
    with pytest.raises(RawFormatError):
        shift_bayer_pattern("rggb", cast(Any, True), 0)
    with pytest.raises(RawFormatError):
        flip_bayer_pattern("rggb", 0, 4, False, False)
    with pytest.raises(RawFormatError):
        flip_bayer_pattern("rggb", 4, 4, cast(Any, 1), False)


def test_tiff_big_endian_uint16(tmp_path: Path) -> None:
    tifffile = pytest.importorskip("tifffile")
    img = np.arange(12, dtype=">u2").reshape(3, 4)
    p = tmp_path / "be.tiff"
    tifffile.imwrite(p, img)
    dec = decode_raw(p, _fmt(bit_depth=16, source_format="tiff"))
    np.testing.assert_array_equal(dec.pixels, img)


def test_rawpy_cfa_mismatch_warns_and_wins(tmp_path: Path, monkeypatch) -> None:
    import sys
    import types

    stub = _StubRaw()
    monkeypatch.setitem(sys.modules, "rawpy",
                        types.SimpleNamespace(imread=lambda p: stub))
    dec = decode_raw(tmp_path / "f.dng",
                     _fmt(bit_depth=12, source_format="rawpy",
                          bayer_pattern="bggr"))
    assert dec.bayer_pattern == "rggb"
    assert any("actual" in w for w in dec.warnings)
