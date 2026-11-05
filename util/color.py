"""Native color helpers: transfers, luma/saturation, YCbCr, code packing.

All float inputs are normalized units (0..1 nominal range), never inferred
from dtype or observed maxima. No implicit clipping except inside final
quantizers/code encoders. Transfers support linear (identity, preserves
negative/headroom), sRGB, Rec.709 (published rounded coefficients — a tiny
coefficient gap exists between the published forward/inverse formulas, so
the inverse is not mathematically exact for arbitrary encoded values in
that gap), and power-law gamma. RGB helpers take any leading dimensions
with last axis = 3. All branches are masked so unused paths never execute
invalid arithmetic.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

_COEFF = {"bt601": (0.299, 0.114), "bt709": (0.2126, 0.0722)}


def _as_real(name: str, value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype.kind not in "fiu" or arr.size == 0:
        raise ValueError(f"{name} must be a nonempty real numeric array")
    if arr.dtype.kind == "f" and not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _nonnegative(name: str, value: Any, allow_negative: bool) -> np.ndarray:
    arr = _as_real(name, value)
    if not allow_negative and float(arr.min()) < 0:
        raise ValueError(f"{name} rejects negative inputs")
    return arr


def _rgb(name: str, value: Any) -> np.ndarray:
    arr = _as_real(name, value)
    if arr.ndim < 1 or arr.shape[-1] != 3:
        raise ValueError(f"{name} requires last axis of size 3, got {arr.shape}")
    return arr


def _dtype(dtype: Any) -> np.dtype:
    try:
        d = np.dtype(dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"dtype must be float32 or float64, got {dtype!r}") from exc
    if d not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(f"dtype must be float32 or float64, got {dtype!r}")
    return d


def _positive_number(name: str, value: Any, minimum: float = 0.0, strict: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}") from exc
    if not math.isfinite(out) or (out <= 0 if strict else out < minimum):
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return out


def _finish(out: np.ndarray, d: np.dtype, name: str) -> np.ndarray:
    """Validate computed output and the requested-dtype cast result."""
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} produced nonfinite output")
    with np.errstate(over="raise", invalid="raise"):
        try:
            cast = out.astype(d)
        except (OverflowError, ValueError, FloatingPointError) as exc:
            raise ValueError(f"{name} output cannot be represented in {d}") from exc
    if not np.all(np.isfinite(cast)):
        raise ValueError(f"{name} output overflows {d} (inf would result)")
    return cast


def _masked_encode_srgb(f: np.ndarray) -> np.ndarray:
    out = np.empty(f.shape, dtype=np.float64)
    lo = f <= 0.0031308
    out[lo] = 12.92 * f[lo]
    hi = ~lo
    out[hi] = 1.055 * np.power(np.maximum(f[hi], 0.0), 1 / 2.4) - 0.055
    return out


def _masked_decode_srgb(e: np.ndarray) -> np.ndarray:
    out = np.empty(e.shape, dtype=np.float64)
    lo = e <= 0.04045
    out[lo] = e[lo] / 12.92
    hi = ~lo
    out[hi] = np.power((e[hi] + 0.055) / 1.055, 2.4)
    return out


def _masked_encode_rec709(f: np.ndarray) -> np.ndarray:
    out = np.empty(f.shape, dtype=np.float64)
    lo = f < 0.018
    out[lo] = 4.5 * f[lo]
    hi = ~lo
    out[hi] = 1.099 * np.power(np.maximum(f[hi], 0.0), 0.45) - 0.099
    return out


def _masked_decode_rec709(e: np.ndarray) -> np.ndarray:
    out = np.empty(e.shape, dtype=np.float64)
    lo = e < 0.081
    out[lo] = e[lo] / 4.5
    hi = ~lo
    out[hi] = np.power((e[hi] + 0.099) / 1.099, 1.0 / 0.45)
    return out


def encode_transfer(
    values: Any,
    transfer: str,
    *,
    gamma: float = 2.2,
    dtype: Any = np.float32,
) -> np.ndarray:
    """Encode normalized linear values to the named transfer domain."""
    d = _dtype(dtype)
    x = _nonnegative("values", values, allow_negative=True)
    if transfer == "linear":
        out = x.astype(np.float64)
    elif transfer in ("srgb", "rec709"):
        with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
            if transfer == "srgb":
                out = _masked_encode_srgb(x.astype(np.float64))
            else:
                out = _masked_encode_rec709(x.astype(np.float64))
    elif transfer == "power":
        g = _positive_number("gamma", gamma)
        f = _nonnegative("values", values, allow_negative=False).astype(np.float64)
        with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
            out = np.zeros(f.shape, dtype=np.float64)
            pos = f > 0
            out[pos] = np.power(f[pos], 1.0 / g)
    else:
        raise ValueError(f"unknown transfer {transfer!r}")
    return _finish(out, d, "encode_transfer")


def decode_transfer(
    values: Any,
    transfer: str,
    *,
    gamma: float = 2.2,
    dtype: Any = np.float32,
) -> np.ndarray:
    """Decode transfer-encoded values back to normalized linear units."""
    d = _dtype(dtype)
    x = _nonnegative("values", values, allow_negative=True)
    if transfer == "linear":
        out = x.astype(np.float64)
    elif transfer in ("srgb", "rec709"):
        with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
            if transfer == "srgb":
                out = _masked_decode_srgb(x.astype(np.float64))
            else:
                out = _masked_decode_rec709(x.astype(np.float64))
    elif transfer == "power":
        g = _positive_number("gamma", gamma)
        e = _nonnegative("values", values, allow_negative=False).astype(np.float64)
        with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
            out = np.zeros(e.shape, dtype=np.float64)
            pos = e > 0
            out[pos] = np.power(e[pos], g)
    else:
        raise ValueError(f"unknown transfer {transfer!r}")
    return _finish(out, d, "decode_transfer")


def _coefficients(matrix: str) -> tuple[float, float, float]:
    key = str(matrix).strip().lower()
    if key not in _COEFF:
        raise ValueError(f"unknown matrix {matrix!r}")
    kr, kb = _COEFF[key]
    return kr, 1.0 - kr - kb, kb


def luminance(rgb: Any, *, matrix: str = "bt709", dtype: Any = np.float32) -> np.ndarray:
    """Kr*R + Kg*G + Kb*B in normalized code units."""
    d = _dtype(dtype)
    arr = _rgb("rgb", rgb)
    kr, kg, kb = _coefficients(matrix)
    f = arr.astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        out = f[..., 0] * kr + f[..., 1] * kg + f[..., 2] * kb
    return _finish(out, d, "luminance")


def adjust_saturation(rgb: Any, gain: float, *, matrix: str = "bt709", dtype: Any = np.float32) -> np.ndarray:
    """L + (RGB - L) * gain; no clipping of negative/headroom."""
    d = _dtype(dtype)
    arr = _rgb("rgb", rgb)
    if isinstance(gain, bool) or not isinstance(gain, (int, float)):
        raise ValueError(f"gain must be a finite nonnegative number, got {gain!r}")
    try:
        g = float(gain)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"gain must be a finite nonnegative number, got {gain!r}") from exc
    if not math.isfinite(g) or g < 0:
        raise ValueError(f"gain must be a finite nonnegative number, got {gain!r}")
    _coefficients(matrix)
    if g == 1.0:
        return _finish(arr.astype(np.float64), d, "adjust_saturation")
    y = luminance(arr, matrix=matrix, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        out = y[..., None] + (arr.astype(np.float64) - y[..., None]) * g
    return _finish(out, d, "adjust_saturation")


def rgb_to_ycbcr(rgb: Any, *, matrix: str = "bt709", dtype: Any = np.float32) -> np.ndarray:
    """Normalized full-range YCbCr: Y=luma, Cb/Cr centered at .5."""
    d = _dtype(dtype)
    arr = _rgb("rgb", rgb)
    kr, kg, kb = _coefficients(matrix)
    f = arr.astype(np.float64)
    y = f[..., 0] * kr + f[..., 1] * kg + f[..., 2] * kb
    cb = 0.5 + (f[..., 2] - y) / (2 * (1 - kb))
    cr = 0.5 + (f[..., 0] - y) / (2 * (1 - kr))
    with np.errstate(over="ignore", invalid="ignore"):
        out = np.stack([y, cb, cr], axis=-1)
    return _finish(out, d, "rgb_to_ycbcr")


def ycbcr_to_rgb(ycbcr: Any, *, matrix: str = "bt709", dtype: Any = np.float32) -> np.ndarray:
    """Inverse of rgb_to_ycbcr; floats are not clipped."""
    d = _dtype(dtype)
    arr = _rgb("ycbcr", ycbcr)
    kr, kg, kb = _coefficients(matrix)
    f = arr.astype(np.float64)
    y, cb, cr = f[..., 0], f[..., 1], f[..., 2]
    with np.errstate(over="ignore", invalid="ignore"):
        r = y + 2 * (1 - kr) * (cr - 0.5)
        b = y + 2 * (1 - kb) * (cb - 0.5)
        g = (y - kr * r - kb * b) / kg
        out = np.stack([r, g, b], axis=-1)
    return _finish(out, d, "ycbcr_to_rgb")


def quantize_rgb(values: Any, *, bit_depth: int) -> np.ndarray:
    """floor(clip(x,0,1)*N + .5) in float64 -> uint8/uint16."""
    if isinstance(bit_depth, bool) or not isinstance(bit_depth, int) or bit_depth not in (8, 16):
        raise ValueError(f"bit_depth must be 8 or 16, got {bit_depth!r}")
    arr = _as_real("values", values)
    n = 2**bit_depth - 1
    out = np.floor(np.clip(arr, 0.0, 1.0).astype(np.float64) * n + 0.5)
    return out.astype(np.uint8 if bit_depth == 8 else np.uint16)


def _ycbcr_code_bits(bit_depth: int, range_mode: str) -> tuple[int, str]:
    if isinstance(bit_depth, bool) or not isinstance(bit_depth, int) or bit_depth not in (8, 16):
        raise ValueError(f"bit_depth must be 8 or 16, got {bit_depth!r}")
    if range_mode not in ("full", "limited"):
        raise ValueError(f"range_mode must be 'full' or 'limited', got {range_mode!r}")
    return bit_depth, range_mode


def ycbcr_to_codes(ycbcr: Any, *, bit_depth: int, range_mode: str) -> np.ndarray:
    """Normalized YCbCr -> integer codes; clip at the code boundary."""
    bits, mode = _ycbcr_code_bits(bit_depth, range_mode)
    arr = _rgb("ycbcr", ycbcr)
    n = 2**bits - 1
    f = np.clip(arr, 0.0, 1.0).astype(np.float64)
    y, cb, cr = f[..., 0], f[..., 1], f[..., 2]
    if mode == "full":
        center = 2.0 ** (bits - 1)
        codes = np.stack([
            y * n,
            center + (cb - 0.5) * n,
            center + (cr - 0.5) * n,
        ], axis=-1)
    else:
        shift = 2.0 ** (bits - 8)
        codes = np.stack([
            (16 + 219 * y) * shift,
            (128 + 224 * (cb - 0.5)) * shift,
            (128 + 224 * (cr - 0.5)) * shift,
        ], axis=-1)
    out = np.floor(np.clip(codes, 0.0, float(n)) + 0.5)
    return out.astype(np.uint8 if bits == 8 else np.uint16)


def codes_to_ycbcr(codes: Any, *, bit_depth: int, range_mode: str = "full", dtype: Any = np.float32) -> np.ndarray:
    """Integer codes -> normalized YCbCr (exact inverse affine, unclipped)."""
    d = _dtype(dtype)
    bits, mode = _ycbcr_code_bits(bit_depth, range_mode)
    arr = np.asarray(codes)
    if arr.ndim < 1 or arr.shape[-1] != 3:
        raise ValueError(f"codes requires last axis of size 3, got {arr.shape}")
    if arr.dtype.kind != "u" or arr.dtype.itemsize not in (1, 2):
        raise ValueError(f"codes must be unsigned uint8/uint16, got {arr.dtype}")
    n = 2**bits - 1
    if int(arr.max()) > n:
        raise ValueError(f"codes exceed declared {bits}-bit range 0..{n}")
    f = arr.astype(np.float64)
    y, cb, cr = f[..., 0], f[..., 1], f[..., 2]
    if mode == "full":
        center = float(2 ** (bits - 1))
        out = np.stack([
            y / n,
            0.5 + (cb - center) / n,
            0.5 + (cr - center) / n,
        ], axis=-1)
    else:
        shift = float(2 ** (bits - 8))
        out = np.stack([
            (y / shift - 16) / 219,
            0.5 + (cb / shift - 128) / 224,
            0.5 + (cr / shift - 128) / 224,
        ], axis=-1)
    return _finish(out, d, "codes_to_ycbcr")


def pack_ycbcr422(ycbcr: Any, *, packing: str = "yuyv") -> np.ndarray:
    """HWC3 uint8/16 YCbCr -> interleaved 4:2:2 rows (H, 2W)."""
    arr = np.asarray(ycbcr)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"pack_ycbcr422 requires an (H,W,3) array, got {arr.shape}")
    if arr.dtype.kind != "u" or arr.dtype.itemsize not in (1, 2):
        raise ValueError(f"pack_ycbcr422 requires uint8/uint16, got {arr.dtype}")
    h, w = arr.shape[0], arr.shape[1]
    if h <= 0 or w <= 0 or w % 2:
        raise ValueError(f"pack_ycbcr422 requires positive H and even W, got {arr.shape[:2]}")
    mode = str(packing).lower()
    if mode not in ("yuyv", "uyvy"):
        raise ValueError(f"unknown packing {packing!r}")
    y0 = arr[:, 0::2, 0]
    y1 = arr[:, 1::2, 0]
    u = ((arr[:, 0::2, 1].astype(np.uint32) + arr[:, 1::2, 1].astype(np.uint32) + 1) // 2).astype(arr.dtype)
    v = ((arr[:, 0::2, 2].astype(np.uint32) + arr[:, 1::2, 2].astype(np.uint32) + 1) // 2).astype(arr.dtype)
    out = np.empty((h, 2 * w), dtype=arr.dtype)
    if mode == "yuyv":
        out[:, 0::4] = y0
        out[:, 1::4] = u
        out[:, 2::4] = y1
        out[:, 3::4] = v
    else:
        out[:, 0::4] = u
        out[:, 1::4] = y0
        out[:, 2::4] = v
        out[:, 3::4] = y1
    return out


def unpack_ycbcr422(packed: Any, *, packing: str = "yuyv") -> np.ndarray:
    """Interleaved 4:2:2 rows -> HWC3 YCbCr with duplicated pair chroma."""
    arr = np.asarray(packed)
    if arr.ndim != 2:
        raise ValueError(f"unpack_ycbcr422 requires a 2-D row array, got {arr.shape}")
    if arr.dtype.kind != "u" or arr.dtype.itemsize not in (1, 2):
        raise ValueError(f"unpack_ycbcr422 requires uint8/uint16, got {arr.dtype}")
    h, rw = arr.shape
    if h <= 0 or rw <= 0 or rw % 4:
        raise ValueError(f"row length must be a positive multiple of 4, got {arr.shape}")
    mode = str(packing).lower()
    if mode not in ("yuyv", "uyvy"):
        raise ValueError(f"unknown packing {packing!r}")
    w = rw // 2
    if mode == "yuyv":
        y0, u, y1, v = arr[:, 0::4], arr[:, 1::4], arr[:, 2::4], arr[:, 3::4]
    else:
        u, y0, v, y1 = arr[:, 0::4], arr[:, 1::4], arr[:, 2::4], arr[:, 3::4]
    out = np.empty((h, w, 3), dtype=arr.dtype)
    out[:, 0::2, 0] = y0
    out[:, 1::2, 0] = y1
    out[:, 0::2, 1] = u
    out[:, 1::2, 1] = u
    out[:, 0::2, 2] = v
    out[:, 1::2, 2] = v
    return out
