"""Explicit Bayer RAW decoding contracts and primitives (M2b).

One strict interpretation of sensor storage: declared geometry, storage
format, byte order, alignment, header and row stride — no file-size
tolerance, dimension factoring, observed-maximum bit-depth inference, or
silent truncation. Container adapters (PGM/TIFF/rawpy) validate against the
same declared contract rather than guessing.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np

BAYER_PATTERNS = ("rggb", "grbg", "gbrg", "bggr")

_GRID = {
    "rggb": (("r", "g"), ("g", "b")),
    "grbg": (("g", "r"), ("b", "g")),
    "gbrg": (("g", "b"), ("r", "g")),
    "bggr": (("b", "g"), ("g", "r")),
}

STORAGE_FORMATS = ("uint8", "uint16", "uint32", "packed24", "word_pair32")

_STORAGE_ALIASES = {
    "u8": "uint8",
    "raw8": "uint8",
    "uint8": "uint8",
    "u16": "uint16",
    "raw16": "uint16",
    "uint16": "uint16",
    "u32": "uint32",
    "raw32": "uint32",
    "uint32": "uint32",
    "u24": "packed24",
    "uint24": "packed24",
    "raw24": "packed24",
    "3byte": "packed24",
    "packed24": "packed24",
    "word_pair32": "word_pair32",
}

_STORAGE_BITS = {
    "uint8": 8,
    "uint16": 16,
    "uint32": 32,
    "packed24": 24,
    "word_pair32": 32,
}

_STORAGE_BYTES = {
    "uint8": 1,
    "uint16": 2,
    "uint32": 4,
    "packed24": 3,
    "word_pair32": 4,
}

_BYTE_ORDER_ALIASES = {
    "little": "little",
    "ieee-le": "little",
    "le": "little",
    "big": "big",
    "ieee-be": "big",
    "be": "big",
}

_WORD_ORDERS = ("low_first", "high_first")
_SOURCE_FORMATS = ("auto", "raw", "pgm", "tiff", "rawpy")
_ALIGNMENTS = ("LSB", "MSB", "MANUAL")

_RAWPY_EXTENSIONS = {
    ".dng", ".nef", ".arw", ".cr2", ".cr3", ".orf", ".rw2", ".raf",
    ".pef", ".srw",
}
_RAW_EXTENSIONS = {".raw", ".bin", ".dat"}
_TIFF_EXTENSIONS = {".tif", ".tiff"}


class RawFormatError(ValueError):
    """Invalid declared format or payload; ``field`` is the failing key."""

    def __init__(self, field: str, message: str) -> None:
        self.field = field
        super().__init__(f"{field}: {message}")


def _require_int(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RawFormatError(name, f"must be an integer, got {value!r}")
    if value < minimum:
        raise RawFormatError(name, f"must be >= {minimum}, got {value}")
    return value


def _require_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise RawFormatError(name, f"must be a boolean, got {value!r}")
    return value


def canonical_storage_format(value: Any, bit_depth: int) -> str:
    """Resolve a declared storage name/alias to its canonical format.

    ``value`` may be None/empty to default from the declared ``bit_depth``
    (uint8 <= 8, uint16 <= 16, uint32 otherwise) — never from file size.
    """
    if value is None or value == "":
        if bit_depth <= 8:
            return "uint8"
        if bit_depth <= 16:
            return "uint16"
        return "uint32"
    key = str(value).strip().lower()
    if key in _STORAGE_ALIASES:
        return _STORAGE_ALIASES[key]
    raise RawFormatError(
        "storage_format",
        f"unknown storage format {value!r}; expected one of {STORAGE_FORMATS} "
        f"or aliases {sorted(_STORAGE_ALIASES)}",
    )


def storage_bytes_per_sample(value: Any, bit_depth: int) -> int:
    """Bytes occupied per stored sample for the given storage format."""
    return _STORAGE_BYTES[canonical_storage_format(value, bit_depth)]


def _pattern_from_grid(grid: tuple[tuple[str, str], tuple[str, str]]) -> str:
    for name, layout in _GRID.items():
        if layout == grid:
            return name
    raise RawFormatError("bayer_pattern", f"unrecognized CFA grid {grid!r}")


def _grid_of(pattern: str) -> tuple[tuple[str, str], tuple[str, str]]:
    key = str(pattern).strip().lower()
    if key not in _GRID:
        raise RawFormatError(
            "bayer_pattern",
            f"must be one of {BAYER_PATTERNS}, got {pattern!r}",
        )
    return _GRID[key]


def shift_bayer_pattern(pattern: str, x_offset: int, y_offset: int) -> str:
    """Effective pattern when sampling starts at ``(x_offset, y_offset)``."""
    _require_int("x_offset", x_offset, 0)
    _require_int("y_offset", y_offset, 0)
    grid = _grid_of(pattern)
    shifted = (
        (grid[y_offset % 2][x_offset % 2], grid[y_offset % 2][(x_offset + 1) % 2]),
        (grid[(y_offset + 1) % 2][x_offset % 2], grid[(y_offset + 1) % 2][(x_offset + 1) % 2]),
    )
    return _pattern_from_grid(shifted)


def flip_bayer_pattern(
    pattern: str,
    width: int,
    height: int,
    horizontal: bool,
    vertical: bool,
) -> str:
    """Effective pattern after mirroring a ``width``x``height`` frame.

    The new (0, 0) site comes from the old (width-1, height-1) parity, so the
    phase offset is ``(width-1) % 2`` horizontally / ``(height-1) % 2``
    vertically.
    """
    _require_int("width", width, 1)
    _require_int("height", height, 1)
    _require_bool("horizontal", horizontal)
    _require_bool("vertical", vertical)
    x_off = (width - 1) % 2 if horizontal else 0
    y_off = (height - 1) % 2 if vertical else 0
    return shift_bayer_pattern(pattern, x_off, y_off)


@dataclass(frozen=True)
class RawFormat:
    width: int
    height: int
    bayer_pattern: str
    bit_depth: int
    storage_format: str = "uint16"
    byte_order: str = "little"
    data_alignment: str = "LSB"
    manual_bit_shift: int | None = None
    header_bytes: int = 0
    row_stride_bytes: int | None = None
    word_order: str = "low_first"
    horizontal_flip: bool = False
    vertical_flip: bool = False
    embedded_rows_top: int = 0
    embedded_rows_bottom: int = 0
    source_format: str = "auto"
    storage_defaulted: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "width", _require_int("width", self.width, 1))
        object.__setattr__(self, "height", _require_int("height", self.height, 1))
        pattern = str(self.bayer_pattern).strip().lower()
        _grid_of(pattern)
        object.__setattr__(self, "bayer_pattern", pattern)
        bit_depth = _require_int("bit_depth", self.bit_depth, 1)
        object.__setattr__(self, "bit_depth", bit_depth)
        if self.storage_format is None or (
            isinstance(self.storage_format, str)
            and self.storage_format.strip() == ""
        ):
            raise RawFormatError(
                "storage_format", "explicit storage_format must not be empty"
            )
        storage = canonical_storage_format(self.storage_format, bit_depth)
        object.__setattr__(self, "storage_format", storage)
        bits = _STORAGE_BITS[storage]
        if bit_depth > bits:
            raise RawFormatError(
                "bit_depth",
                f"{bit_depth}-bit codes cannot fit in {storage} ({bits} bits)",
            )
        order = _BYTE_ORDER_ALIASES.get(str(self.byte_order).strip().lower())
        if order is None:
            raise RawFormatError(
                "byte_order",
                f"must be little/big (or ieee-le/ieee-be), got {self.byte_order!r}",
            )
        object.__setattr__(self, "byte_order", order)
        align = str(self.data_alignment).strip().upper()
        if align not in _ALIGNMENTS:
            raise RawFormatError(
                "data_alignment",
                f"must be one of {_ALIGNMENTS}, got {self.data_alignment!r}",
            )
        object.__setattr__(self, "data_alignment", align)
        if align == "MANUAL":
            if isinstance(self.manual_bit_shift, bool) or not isinstance(
                self.manual_bit_shift, int
            ):
                raise RawFormatError(
                    "manual_bit_shift",
                    "MANUAL data_alignment requires an integer manual_bit_shift",
                )
        if self.manual_bit_shift is not None:
            shift_val = _require_int("manual_bit_shift", self.manual_bit_shift, 0)
            if shift_val > bits - 1:
                raise RawFormatError(
                    "manual_bit_shift",
                    f"must be in 0..{bits - 1} for {storage}, got {self.manual_bit_shift}",
                )
        _require_int("header_bytes", self.header_bytes, 0)
        if self.row_stride_bytes is not None:
            stride = _require_int("row_stride_bytes", self.row_stride_bytes, 0)
            minimum = self.width * _STORAGE_BYTES[storage]
            if stride < minimum:
                raise RawFormatError(
                    "row_stride_bytes",
                    f"{stride} is smaller than one row ({minimum} bytes)",
                )
            object.__setattr__(self, "row_stride_bytes", stride)
        word = str(self.word_order).strip().lower()
        if word not in _WORD_ORDERS:
            raise RawFormatError(
                "word_order", f"must be one of {_WORD_ORDERS}, got {self.word_order!r}"
            )
        object.__setattr__(self, "word_order", word)
        _require_bool("horizontal_flip", self.horizontal_flip)
        _require_bool("vertical_flip", self.vertical_flip)
        top = _require_int("embedded_rows_top", self.embedded_rows_top, 0)
        bottom = _require_int("embedded_rows_bottom", self.embedded_rows_bottom, 0)
        if top + bottom >= self.height:
            raise RawFormatError(
                "embedded_rows_top",
                f"embedded rows {top}+{bottom} leave no active rows in height {self.height}",
            )
        src = str(self.source_format).strip().lower()
        if src not in _SOURCE_FORMATS:
            raise RawFormatError(
                "source_format",
                f"must be one of {_SOURCE_FORMATS}, got {self.source_format!r}",
            )
        object.__setattr__(self, "source_format", src)
        if src in ("pgm", "tiff", "rawpy") and (
            self.header_bytes != 0 or self.row_stride_bytes is not None
        ):
            raise RawFormatError(
                "header_bytes",
                f"{src} containers carry their own layout; header_bytes/"
                "row_stride_bytes must be unset for container input",
            )
        if src == "rawpy" and align != "LSB":
            raise RawFormatError(
                "data_alignment",
                "rawpy containers decode to native samples; only LSB "
                "data_alignment is valid for rawpy input",
            )

    @classmethod
    def from_sensor_info(cls, sensor: Mapping[str, Any]) -> "RawFormat":
        """Build a declared format from a ``sensor_info`` mapping.

        ``storage_format`` takes priority over ``data_format``; when neither
        is present the storage defaults solely from the declared
        ``bit_depth`` (flagged via ``storage_defaulted`` so the decoder can
        surface a warning). ``data_alignment`` takes priority over
        ``bit_alignment``; ``endian_type`` accepts the canonical
        ``ieee-le``/``ieee-be`` spellings as well as ``little``/``big``.
        """
        if not isinstance(sensor, Mapping):
            raise RawFormatError(
                "sensor_info", f"expected a mapping, got {type(sensor).__name__}"
            )
        for key in ("width", "height", "bit_depth", "bayer_pattern"):
            if key not in sensor or sensor[key] is None:
                raise RawFormatError(key, f"sensor_info.{key} is required")
        bit_depth = _require_int("bit_depth", sensor["bit_depth"], 1)
        storage_defaulted = False
        if "storage_format" in sensor:
            storage_value = sensor["storage_format"]
            if storage_value is None or str(storage_value).strip() == "":
                raise RawFormatError(
                    "storage_format", "explicit storage_format must not be empty"
                )
        elif "data_format" in sensor:
            storage_value = sensor["data_format"]
            if storage_value is None or str(storage_value).strip() == "":
                raise RawFormatError(
                    "data_format", "explicit data_format must not be empty"
                )
        else:
            storage_defaulted = True
            storage_value = canonical_storage_format(None, bit_depth)
        if "data_alignment" in sensor:
            alignment = sensor["data_alignment"]
            if alignment is None or str(alignment).strip() == "":
                raise RawFormatError(
                    "data_alignment", "explicit data_alignment must not be empty"
                )
        elif "bit_alignment" in sensor:
            alignment = sensor["bit_alignment"]
            if alignment is None or str(alignment).strip() == "":
                raise RawFormatError(
                    "bit_alignment", "explicit bit_alignment must not be empty"
                )
        else:
            alignment = "LSB"
        return cls(
            width=cast(int, sensor["width"]),
            height=cast(int, sensor["height"]),
            bayer_pattern=sensor["bayer_pattern"],
            bit_depth=bit_depth,
            storage_format=storage_value,
            byte_order=sensor.get("endian_type", "little"),
            data_alignment=alignment,
            manual_bit_shift=sensor.get("manual_bit_shift"),
            header_bytes=sensor.get("header_bytes", 0),
            row_stride_bytes=sensor.get("row_stride_bytes"),
            word_order=sensor.get("word_order", "low_first"),
            horizontal_flip=sensor.get("horizontal_flip", False),
            vertical_flip=sensor.get("vertical_flip", False),
            embedded_rows_top=sensor.get("embedded_rows_top", 0),
            embedded_rows_bottom=sensor.get("embedded_rows_bottom", 0),
            source_format=sensor.get("source_format", "auto"),
            storage_defaulted=storage_defaulted,
        )


@dataclass(frozen=True)
class DecodedRaw:
    pixels: np.ndarray
    width: int
    height: int
    bayer_pattern: str
    storage_bits: int
    code_bit_depth: int
    container: str
    source_identity: str
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _unpack_rows(data: bytes, fmt: RawFormat, offset: int) -> np.ndarray:
    """Decode the raw raster into native-endian samples without copies beyond need."""
    storage = fmt.storage_format
    bpp = _STORAGE_BYTES[storage]
    stride = fmt.row_stride_bytes if fmt.row_stride_bytes is not None else fmt.width * bpp
    expected = offset + fmt.height * stride
    if len(data) != expected:
        raise RawFormatError(
            "data",
            f"payload is {len(data)} bytes but the declared layout requires "
            f"exactly {expected} (header {offset} + {fmt.height} rows x {stride}); "
            "check width/height/header_bytes/row_stride_bytes",
        )
    raster = np.frombuffer(data, dtype=np.uint8, count=len(data) - offset, offset=offset)
    rows = raster.reshape(fmt.height, stride)[:, : fmt.width * bpp]
    rows = np.ascontiguousarray(rows)

    if storage == "uint8":
        return rows.copy()
    if storage == "uint16":
        dtype = "<u2" if fmt.byte_order == "little" else ">u2"
        return rows.view(dtype).copy()
    if storage == "uint32":
        dtype = "<u4" if fmt.byte_order == "little" else ">u4"
        return rows.view(dtype).copy()
    if storage == "packed24":
        trip = rows.reshape(fmt.height, fmt.width, 3).astype(np.uint32)
        if fmt.byte_order == "little":
            return trip[:, :, 0] | (trip[:, :, 1] << 8) | (trip[:, :, 2] << 16)
        return (trip[:, :, 0] << 16) | (trip[:, :, 1] << 8) | trip[:, :, 2]
    if storage == "word_pair32":
        dtype = "<u2" if fmt.byte_order == "little" else ">u2"
        words = rows.view(dtype).reshape(fmt.height, fmt.width, 2).astype(np.uint32)
        if fmt.word_order == "low_first":
            return words[:, :, 0] | (words[:, :, 1] << 16)
        return (words[:, :, 0] << 16) | words[:, :, 1]
    raise RawFormatError("storage_format", f"unsupported storage {storage!r}")


def _apply_alignment(pixels: np.ndarray, fmt: RawFormat, storage_bits: int) -> np.ndarray:
    out_dtype = np.uint16 if storage_bits <= 16 else np.uint32
    align = fmt.data_alignment
    if align == "LSB":
        return pixels.astype(out_dtype)
    mask = (1 << fmt.bit_depth) - 1
    if align == "MSB":
        shift = storage_bits - fmt.bit_depth
        return ((pixels.astype(np.uint32) >> shift) & mask).astype(out_dtype)
    if align == "MANUAL":
        shift = int(fmt.manual_bit_shift or 0)
        return ((pixels.astype(np.uint32) >> shift) & mask).astype(out_dtype)
    raise RawFormatError("data_alignment", f"unsupported alignment {align!r}")


def _finish(
    pixels: np.ndarray,
    fmt: RawFormat,
    storage_bits: int,
    container: str,
    source_identity: str,
    warnings: list[str],
) -> DecodedRaw:
    owned = np.ascontiguousarray(_apply_alignment(pixels, fmt, storage_bits))
    height = fmt.height
    top = fmt.embedded_rows_top
    bottom = fmt.embedded_rows_bottom
    pattern = fmt.bayer_pattern
    if top or bottom:
        owned = owned[top : height - bottom]
        pattern = shift_bayer_pattern(pattern, 0, top)
        height = height - top - bottom
        owned = np.ascontiguousarray(owned)
    if fmt.horizontal_flip or fmt.vertical_flip:
        pattern = flip_bayer_pattern(
            pattern, fmt.width, height, fmt.horizontal_flip, fmt.vertical_flip
        )
        out = owned
        if fmt.horizontal_flip:
            out = np.fliplr(out)
        if fmt.vertical_flip:
            out = np.flipud(out)
        owned = np.ascontiguousarray(out)
    if fmt.storage_defaulted:
        warnings.append(
            "storage_format/data_format absent; defaulted from declared "
            f"bit_depth to {fmt.storage_format} (never from file size)"
        )
    return DecodedRaw(
        pixels=owned,
        width=fmt.width,
        height=height,
        bayer_pattern=pattern,
        storage_bits=storage_bits,
        code_bit_depth=fmt.bit_depth,
        container=container,
        source_identity=source_identity,
        warnings=tuple(warnings),
    )


def decode_raw_bytes(data: bytes | bytearray | memoryview, format: RawFormat) -> DecodedRaw:
    """Decode a headerless RAW buffer against the declared layout.

    The buffer must contain exactly ``header_bytes + height * row_stride``
    bytes (every row including the last carries full stride padding). Short
    or surplus payloads are rejected — no tolerance or truncation.
    """
    payload = bytes(data)
    if format.source_format not in ("auto", "raw"):
        raise RawFormatError(
            "source_format",
            f"decode_raw_bytes only handles raw buffers, got {format.source_format!r}",
        )
    warnings: list[str] = []
    pixels = _unpack_rows(payload, format, format.header_bytes)
    return _finish(
        pixels,
        format,
        _STORAGE_BITS[format.storage_format],
        "raw",
        "bytes",
        warnings,
    )


def _decode_pgm(path: Path, fmt: RawFormat) -> DecodedRaw:
    if fmt.header_bytes or fmt.row_stride_bytes is not None:
        raise RawFormatError(
            "header_bytes",
            "PGM containers carry their own header; header_bytes/row_stride_bytes "
            "must be unset for container input",
        )
    from util.raw_io import read_raw_payload

    payload = read_raw_payload(path)
    if not payload.is_pgm:
        raise RawFormatError("source_format", f"{path}: not a PGM file")
    if payload.pgm_width != fmt.width or payload.pgm_height != fmt.height:
        raise RawFormatError(
            "sensor_info",
            f"PGM geometry {payload.pgm_width}x{payload.pgm_height} does not "
            f"match declared {fmt.width}x{fmt.height}; fix sensor_info",
        )
    container_bits = 16 if (payload.pgm_maxval or 0) > 255 else 8
    if fmt.bit_depth > container_bits:
        raise RawFormatError(
            "bit_depth",
            f"{fmt.bit_depth}-bit codes cannot fit in PGM maxval "
            f"{payload.pgm_maxval} ({container_bits}-bit container)",
        )
    dtype = ">u2" if container_bits == 16 else np.uint8
    pixels = np.frombuffer(payload.data, dtype=dtype).reshape(fmt.height, fmt.width)
    return _finish(
        pixels.copy(),
        fmt,
        container_bits,
        "pgm",
        str(path),
        [],
    )


def _decode_tiff(path: Path, fmt: RawFormat) -> DecodedRaw:
    if fmt.header_bytes or fmt.row_stride_bytes is not None:
        raise RawFormatError(
            "header_bytes",
            "TIFF containers carry their own layout; header_bytes/row_stride_bytes "
            "must be unset for container input",
        )
    try:
        import tifffile as tiff
    except ImportError as exc:
        raise RawFormatError(
            "source_format", "tifffile is required for TIFF input"
        ) from exc
    with tiff.TiffFile(str(path)) as tf:
        pages = tf.pages
        if len(pages) != 1:
            raise RawFormatError(
                "source_format",
                f"{path}: expected exactly one TIFF image, found {len(pages)} pages",
            )
        image = pages[0].asarray()
    if image.ndim != 2:
        raise RawFormatError(
            "source_format",
            f"{path}: expected a single 2-D frame, got shape {image.shape}; "
            "multi-channel TIFF is not Bayer input",
        )
    if image.dtype.kind != "u" or image.dtype.itemsize not in (1, 2, 4):
        raise RawFormatError(
            "source_format",
            f"{path}: unsupported TIFF sample dtype {image.dtype}; "
            "only unsigned 8/16/32-bit frames are accepted",
        )
    if image.shape != (fmt.height, fmt.width):
        raise RawFormatError(
            "sensor_info",
            f"TIFF geometry {image.shape[1]}x{image.shape[0]} does not match "
            f"declared {fmt.width}x{fmt.height}; fix sensor_info",
        )
    container_bits = int(image.dtype.itemsize) * 8
    if fmt.bit_depth > container_bits:
        raise RawFormatError(
            "bit_depth",
            f"{fmt.bit_depth}-bit codes cannot fit in a {container_bits}-bit TIFF",
        )
    return _finish(
        np.ascontiguousarray(image),
        fmt,
        container_bits,
        "tiff",
        str(path),
        [],
    )


def _decode_rawpy(path: Path, fmt: RawFormat) -> DecodedRaw:
    if fmt.header_bytes or fmt.row_stride_bytes is not None:
        raise RawFormatError(
            "header_bytes",
            "rawpy containers carry their own layout; header_bytes/row_stride_bytes "
            "must be unset for container input",
        )
    try:
        import rawpy
    except ImportError as exc:
        raise RawFormatError(
            "source_format", "rawpy is required for this camera format"
        ) from exc
    warnings: list[str] = []
    with rawpy.imread(str(path)) as raw:
        raw_pattern = getattr(raw, "raw_pattern", None)
        if not isinstance(raw_pattern, np.ndarray) or raw_pattern.shape != (2, 2):
            raise RawFormatError(
                "bayer_pattern",
                "rawpy reports a non-2x2 (e.g. XTrans) or missing CFA layout; "
                "such mosaics are not supported",
            )
        pixels = raw.raw_image_visible.copy()
        colors = raw.raw_colors_visible[:2, :2].copy()
        desc = raw.color_desc
        white_level = int(raw.white_level)
    if pixels.ndim != 2 or pixels.dtype.kind != "u" or pixels.dtype.itemsize not in (1, 2, 4):
        raise RawFormatError(
            "source_format",
            f"{path}: rawpy produced unsupported pixels "
            f"(shape {pixels.shape}, dtype {pixels.dtype})",
        )
    container_bits = int(pixels.dtype.itemsize) * 8
    if white_level <= 0 or white_level.bit_length() > container_bits:
        raise RawFormatError(
            "bit_depth",
            f"rawpy white level {white_level} does not fit the "
            f"{container_bits}-bit container",
        )
    if pixels.shape != (fmt.height, fmt.width):
        raise RawFormatError(
            "sensor_info",
            f"visible geometry {pixels.shape[1]}x{pixels.shape[0]} does not "
            f"match declared {fmt.width}x{fmt.height}; fix sensor_info",
        )
    code_bits = white_level.bit_length()
    if fmt.bit_depth != code_bits:
        raise RawFormatError(
            "bit_depth",
            f"declared bit_depth {fmt.bit_depth} does not match rawpy white "
            f"level {white_level} ({code_bits} bits)",
        )
    letters = [chr(desc[c]).lower() for c in colors.flatten()]
    grid: tuple[tuple[str, str], tuple[str, str]] = (
        (letters[0], letters[1]),
        (letters[2], letters[3]),
    )
    try:
        actual = _pattern_from_grid(grid)
    except RawFormatError as exc:
        raise RawFormatError(
            "bayer_pattern",
            f"unsupported CFA layout {grid!r} reported by rawpy",
        ) from exc
    if actual != fmt.bayer_pattern:
        warnings.append(
            f"configured bayer_pattern {fmt.bayer_pattern} differs from "
            f"rawpy-reported {actual}; using the container's actual CFA"
        )
        fmt = dataclasses.replace(fmt, bayer_pattern=actual)
    result = _finish(
        np.ascontiguousarray(pixels),
        fmt,
        container_bits,
        "rawpy",
        str(path),
        warnings,
    )
    return result


def resolve_source_format(path: str | Path, declared: str = "auto") -> str:
    """Resolve the effective source kind for ``path``.

    An explicit ``declared`` kind wins; ``"auto"`` selects by extension only
    (.raw/.bin/.dat/no suffix -> raw, .pgm -> pgm, .tif/.tiff -> tiff,
    recognized camera extensions -> rawpy). Unknown or RGB image extensions
    are rejected with a hint to set ``sensor_info.source_format``
    explicitly — RGB images are not Bayer input.
    """
    if declared != "auto":
        return declared
    ext = Path(path).suffix.lower()
    if ext in _RAW_EXTENSIONS or ext == "":
        return "raw"
    if ext == ".pgm":
        return "pgm"
    if ext in _TIFF_EXTENSIONS:
        return "tiff"
    if ext in _RAWPY_EXTENSIONS:
        return "rawpy"
    raise RawFormatError(
        "source_format",
        f"cannot infer container type from extension {ext or '(none)'!r}; "
        "set sensor_info.source_format explicitly (raw/pgm/tiff/rawpy). "
        "RGB images (.png/.jpg) are not Bayer input",
    )


def decode_raw(path: str | Path, format: RawFormat) -> DecodedRaw:
    """Decode a RAW/PGM/TIFF/camera file through the declared contract."""
    p = Path(path).expanduser().resolve()
    src = resolve_source_format(p, format.source_format)
    format = dataclasses.replace(format, source_format=src)
    if src == "raw":
        data = p.read_bytes()
        return dataclasses.replace(
            decode_raw_bytes(data, format), source_identity=str(p)
        )
    if src == "pgm":
        return _decode_pgm(p, format)
    if src == "tiff":
        return _decode_tiff(p, format)
    if src == "rawpy":
        return _decode_rawpy(p, format)
    raise RawFormatError("source_format", f"unsupported source {src!r}")
