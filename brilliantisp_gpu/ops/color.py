"""GPU colour helpers: luminance and saturation in normalized units.

Mirrors util.color.luminance/adjust_saturation: f64 left-associative
coefficient application, optional f32/f64 finish cast with finite checks,
no clipping of negative or headroom values. Torch is lazy.
"""

from __future__ import annotations

import math

import numpy as np

from util.color import _coefficients

from ..types import GpuError, require_torch


def _norm_dtype(dtype, torch):
    if dtype is None:
        return torch.float64
    if isinstance(dtype, torch.dtype):
        if dtype in (torch.float32, torch.float64):
            return dtype
        raise ValueError(f"dtype {dtype} must be float32 or float64")
    try:
        d = np.dtype(dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"dtype {dtype!r} is not a valid dtype") from exc
    if d == np.dtype("float32"):
        return torch.float32
    if d == np.dtype("float64"):
        return torch.float64
    raise ValueError(f"dtype {dtype!r} must be float32 or float64")


def _rgb(name, values, torch):
    arr = torch.as_tensor(values)
    if arr.ndim < 1 or arr.shape[-1] != 3 or not arr.numel():
        raise GpuError(f"{name} requires a nonempty array with last axis 3, got {tuple(arr.shape)}")
    if arr.dtype == torch.bool or arr.is_complex():
        raise GpuError(f"{name} requires real numeric input, got {arr.dtype}")
    if arr.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64,
                         torch.int8, torch.int16, torch.int32, torch.int64,
                         torch.uint8, torch.uint16, torch.uint32):
        raise GpuError(f"{name} requires real numeric input, got {arr.dtype}")
    if not bool(torch.isfinite(arr).all()):
        raise ValueError(f"{name} produced nonfinite input")
    return arr


def _finish(out, d, name):
    torch = require_torch()
    if not bool(torch.isfinite(out).all()):
        raise ValueError(f"{name} produced nonfinite output")
    cast = out.to(d)
    if not bool(torch.isfinite(cast).all()):
        raise ValueError(f"{name} output overflows {d} (inf would result)")
    return cast


def luminance(rgb, *, matrix: str = "bt709", dtype="float32"):
    """Kr*R + Kg*G + Kb*B, f64 left-associative then requested cast."""
    torch = require_torch()
    d = _norm_dtype(dtype, torch)
    arr = _rgb("rgb", rgb, torch)
    kr, kg, kb = _coefficients(matrix)
    f = arr.to(torch.float64)
    out = f[..., 0] * kr + f[..., 1] * kg + f[..., 2] * kb
    return _finish(out, d, "luminance")


def adjust_saturation(rgb, gain, *, matrix: str = "bt709", dtype="float32"):
    """L + (RGB - L) * gain; no clipping."""
    torch = require_torch()
    d = _norm_dtype(dtype, torch)
    arr = _rgb("rgb", rgb, torch)
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
        f = arr.to(torch.float64, copy=True)
        return _finish(f, d, "adjust_saturation")
    y = luminance(arr, matrix=matrix, dtype=np.float64)
    out = y[..., None] + (arr.to(torch.float64) - y[..., None]) * g
    return _finish(out, d, "adjust_saturation")


def _as_real(name, values, torch):
    arr = torch.as_tensor(values)
    if not arr.numel():
        raise ValueError(f"{name} must be a nonempty real numeric array")
    if arr.dtype == torch.bool or arr.is_complex():
        raise ValueError(f"{name} must be a nonempty real numeric array")
    if arr.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64,
                         torch.int8, torch.int16, torch.int32, torch.int64,
                         torch.uint8, torch.uint16, torch.uint32):
        raise ValueError(f"{name} must be a nonempty real numeric array")
    if not bool(torch.isfinite(arr.to(torch.float64)).all()):
        raise ValueError(f"{name} must be a finite real numeric array")
    return arr


def _nonnegative(name, values, torch, allow_negative=True):
    arr = _as_real(name, values, torch)
    if not allow_negative and bool((arr.to(torch.float64) < 0).any()):
        raise ValueError(f"{name} must be nonnegative")
    return arr


def _positive_number(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    try:
        g = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}") from exc
    if not math.isfinite(g) or g <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return g


def _code_bits(bit_depth):
    if isinstance(bit_depth, bool) or not isinstance(bit_depth, int) or bit_depth not in (8, 16):
        raise ValueError(f"bit_depth must be 8 or 16, got {bit_depth!r}")
    return bit_depth


def _uint_dtype(bits, torch):
    return torch.uint8 if bits == 8 else torch.uint16


def encode_transfer(values, transfer: str, *, gamma: float = 2.2, dtype="float32"):
    """Normalized linear -> named transfer domain, f64 compute, no clip."""
    torch = require_torch()
    d = _norm_dtype(dtype, torch)
    x = _nonnegative("values", values, torch, allow_negative=True)
    f = x.to(torch.float64)
    if transfer == "linear":
        out = f.to(torch.float64, copy=True)
    elif transfer == "srgb":
        lo = f <= 0.0031308
        hi = torch.zeros_like(f)
        torch.pow(f.clamp(min=0), 1 / 2.4, out=hi)
        out = torch.where(lo, 12.92 * f, 1.055 * hi - 0.055)
    elif transfer == "rec709":
        lo = f < 0.018
        hi = torch.zeros_like(f)
        torch.pow(f.clamp(min=0), 0.45, out=hi)
        out = torch.where(lo, 4.5 * f, 1.099 * hi - 0.099)
    elif transfer == "power":
        g = _positive_number("gamma", gamma)
        f = _nonnegative("values", values, torch, allow_negative=False).to(torch.float64)
        out = torch.where(f > 0, torch.pow(f.clamp(min=0), 1.0 / g), torch.zeros_like(f))
    else:
        raise ValueError(f"unknown transfer {transfer!r}")
    return _finish(out, d, "encode_transfer")


def decode_transfer(values, transfer: str, *, gamma: float = 2.2, dtype="float32"):
    """Transfer-encoded -> normalized linear, f64 compute, no clip."""
    torch = require_torch()
    d = _norm_dtype(dtype, torch)
    x = _nonnegative("values", values, torch, allow_negative=True)
    e = x.to(torch.float64)
    if transfer == "linear":
        out = e.to(torch.float64, copy=True)
    elif transfer == "srgb":
        lo = e <= 0.04045
        base = (e + 0.055) / torch.as_tensor(1.055, dtype=torch.float64, device=e.device)
        hi = torch.pow(base.clamp(min=0), 2.4)
        out = torch.where(lo, e / torch.as_tensor(12.92, dtype=torch.float64, device=e.device), hi)
    elif transfer == "rec709":
        lo = e < 0.081
        base = (e + 0.099) / torch.as_tensor(1.099, dtype=torch.float64, device=e.device)
        hi = torch.pow(base.clamp(min=0), 1.0 / 0.45)
        out = torch.where(lo, e / torch.as_tensor(4.5, dtype=torch.float64, device=e.device), hi)
    elif transfer == "power":
        g = _positive_number("gamma", gamma)
        e = _nonnegative("values", values, torch, allow_negative=False).to(torch.float64)
        out = torch.where(e > 0, torch.pow(e.clamp(min=0), g), torch.zeros_like(e))
    else:
        raise ValueError(f"unknown transfer {transfer!r}")
    return _finish(out, d, "decode_transfer")


def rgb_to_ycbcr(rgb, *, matrix: str = "bt709", dtype="float32"):
    """Normalized full-range YCbCr centred at .5."""
    torch = require_torch()
    d = _norm_dtype(dtype, torch)
    arr = _rgb("rgb", rgb, torch)
    kr, kg, kb = _coefficients(matrix)
    f = arr.to(torch.float64)
    y = f[..., 0] * kr + f[..., 1] * kg + f[..., 2] * kb
    cb = 0.5 + (f[..., 2] - y) / torch.as_tensor(2 * (1 - kb), dtype=torch.float64, device=arr.device)
    cr = 0.5 + (f[..., 0] - y) / torch.as_tensor(2 * (1 - kr), dtype=torch.float64, device=arr.device)
    out = torch.stack((y, cb, cr), dim=-1)
    return _finish(out, d, "rgb_to_ycbcr")


def ycbcr_to_rgb(ycbcr, *, matrix: str = "bt709", dtype="float32"):
    """Inverse of rgb_to_ycbcr; unclipped."""
    torch = require_torch()
    d = _norm_dtype(dtype, torch)
    arr = _rgb("ycbcr", ycbcr, torch)
    kr, kg, kb = _coefficients(matrix)
    f = arr.to(torch.float64)
    y, cb, cr = f[..., 0], f[..., 1], f[..., 2]
    r = y + 2 * (1 - kr) * (cr - 0.5)
    b = y + 2 * (1 - kb) * (cb - 0.5)
    g = (y - kr * r - kb * b) / torch.as_tensor(kg, dtype=torch.float64, device=arr.device)
    out = torch.stack((r, g, b), dim=-1)
    return _finish(out, d, "ycbcr_to_rgb")


def quantize_rgb(values, *, bit_depth: int):
    """floor(clip(x,0,1)*N + .5) f64 -> physical uint8/uint16."""
    torch = require_torch()
    bits = _code_bits(bit_depth)
    arr = _as_real("values", values, torch)
    n = float(2 ** bits - 1)
    out = torch.floor(arr.to(torch.float64).clamp(0.0, 1.0) * n + 0.5)
    return out.to(_uint_dtype(bits, torch))


def ycbcr_to_codes(ycbcr, *, bit_depth: int, range_mode: str):
    """Normalized YCbCr -> clipped physical uint8/uint16 codes."""
    torch = require_torch()
    bits = _code_bits(bit_depth)
    if range_mode not in ("full", "limited"):
        raise ValueError(f"range_mode must be 'full' or 'limited', got {range_mode!r}")
    arr = _rgb("ycbcr", ycbcr, torch)
    n = float(2 ** bits - 1)
    f = arr.to(torch.float64).clamp(0.0, 1.0)
    y, cb, cr = f[..., 0], f[..., 1], f[..., 2]
    if range_mode == "full":
        center = float(2 ** (bits - 1))
        codes = torch.stack((y * n,
                           center + (cb - 0.5) * n,
                           center + (cr - 0.5) * n), dim=-1)
    else:
        shift = float(2 ** (bits - 8))
        codes = torch.stack(((16 + 219 * y) * shift,
                             (128 + 224 * (cb - 0.5)) * shift,
                             (128 + 224 * (cr - 0.5)) * shift), dim=-1)
    out = torch.floor(codes.clamp(0.0, n) + 0.5)
    return out.to(_uint_dtype(bits, torch))


def codes_to_ycbcr(codes, *, bit_depth: int, range_mode: str = "full", dtype="float32"):
    """uint8/16 codes -> normalized YCbCr (exact inverse affine)."""
    torch = require_torch()
    d = _norm_dtype(dtype, torch)
    bits = _code_bits(bit_depth)
    if range_mode not in ("full", "limited"):
        raise ValueError(f"range_mode must be 'full' or 'limited', got {range_mode!r}")
    arr = torch.as_tensor(codes)
    if arr.ndim < 1 or arr.shape[-1] != 3 or not arr.numel():
        raise ValueError(f"codes requires last axis of size 3, got {tuple(arr.shape)}")
    if arr.dtype not in (torch.uint8, torch.uint16):
        raise ValueError(f"codes must be unsigned uint8/uint16, got {arr.dtype}")
    n = float(2 ** bits - 1)
    if int(arr.to(torch.int64).max()) > 2 ** bits - 1:
        raise ValueError(f"codes exceed declared {bits}-bit range 0..{int(n)}")
    f = arr.to(torch.float64)
    y, cb, cr = f[..., 0], f[..., 1], f[..., 2]
    nd = torch.as_tensor(n, dtype=torch.float64, device=arr.device)
    if range_mode == "full":
        center = float(2 ** (bits - 1))
        out = torch.stack((y / nd,
                           0.5 + (cb - center) / nd,
                           0.5 + (cr - center) / nd), dim=-1)
    else:
        shift = float(2 ** (bits - 8))
        d219 = torch.as_tensor(219.0, dtype=torch.float64, device=arr.device)
        d224 = torch.as_tensor(224.0, dtype=torch.float64, device=arr.device)
        out = torch.stack(((y / shift - 16) / d219,
                           0.5 + (cb / shift - 128) / d224,
                           0.5 + (cr / shift - 128) / d224), dim=-1)
    return _finish(out, d, "codes_to_ycbcr")


def pack_ycbcr422(codes, *, packing: str = "yuyv"):
    """(...,H,W,3) uint8/16 codes -> (...,H,2W) interleaved rows."""
    torch = require_torch()
    arr = torch.as_tensor(codes)
    if arr.ndim < 3 or arr.shape[-1] != 3:
        raise ValueError(f"pack_ycbcr422 requires (...,H,W,3), got {tuple(arr.shape)}")
    if arr.dtype not in (torch.uint8, torch.uint16):
        raise ValueError(f"pack_ycbcr422 requires uint8/uint16, got {arr.dtype}")
    h, w = arr.shape[-3], arr.shape[-2]
    if h <= 0 or w <= 0 or w % 2:
        raise ValueError(f"pack_ycbcr422 requires positive H and even W, got ({h}, {w})")
    mode = str(packing).lower()
    if mode not in ("yuyv", "uyvy"):
        raise ValueError(f"unknown packing {packing!r}")
    y0 = arr[..., 0::2, 0]
    y1 = arr[..., 1::2, 0]
    u = ((arr[..., 0::2, 1].to(torch.int64) + arr[..., 1::2, 1].to(torch.int64) + 1) // 2).to(arr.dtype)
    v = ((arr[..., 0::2, 2].to(torch.int64) + arr[..., 1::2, 2].to(torch.int64) + 1) // 2).to(arr.dtype)
    out = torch.empty((*arr.shape[:-3], h, 2 * w), dtype=arr.dtype, device=arr.device)
    if mode == "yuyv":
        out[..., 0::4] = y0
        out[..., 1::4] = u
        out[..., 2::4] = y1
        out[..., 3::4] = v
    else:
        out[..., 0::4] = u
        out[..., 1::4] = y0
        out[..., 2::4] = v
        out[..., 3::4] = y1
    return out


def unpack_ycbcr422(packed, *, packing: str = "yuyv"):
    """(...,H,2W) interleaved rows -> (...,H,W,3) duplicated pair chroma."""
    torch = require_torch()
    arr = torch.as_tensor(packed)
    if arr.ndim < 2:
        raise ValueError(f"unpack_ycbcr422 requires (...,H,2W), got {tuple(arr.shape)}")
    if arr.dtype not in (torch.uint8, torch.uint16):
        raise ValueError(f"unpack_ycbcr422 requires uint8/uint16, got {arr.dtype}")
    h, rw = arr.shape[-2], arr.shape[-1]
    if h <= 0 or rw <= 0 or rw % 4:
        raise ValueError(f"row length must be a positive multiple of 4, got {(h, rw)}")
    mode = str(packing).lower()
    if mode not in ("yuyv", "uyvy"):
        raise ValueError(f"unknown packing {packing!r}")
    w = rw // 2
    if mode == "yuyv":
        y0, u, y1, v = arr[..., 0::4], arr[..., 1::4], arr[..., 2::4], arr[..., 3::4]
    else:
        u, y0, v, y1 = arr[..., 0::4], arr[..., 1::4], arr[..., 2::4], arr[..., 3::4]
    out = torch.empty((*arr.shape[:-2], h, w, 3), dtype=arr.dtype, device=arr.device)
    out[..., 0::2, 0] = y0
    out[..., 1::2, 0] = y1
    out[..., 0::2, 1] = u
    out[..., 1::2, 1] = u
    out[..., 0::2, 2] = v
    out[..., 1::2, 2] = v
    return out
