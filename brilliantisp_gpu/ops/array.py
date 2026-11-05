from __future__ import annotations

from ..types import require_torch


def checked_float32(values, *, name: str = "array"):
    """Cast a real numeric tensor to float32, rejecting nonfinite results.

    Mirrors util.native_array.as_float32: real numeric, nonempty, finite
    input, finite float32 output (overflow is a named ValueError).
    """
    torch = require_torch()
    arr = torch.as_tensor(values)
    if arr.numel() == 0:
        raise ValueError(f"{name} must be a nonempty real numeric array")
    allowed = _unsigned_torch_dtypes(torch) + (
        torch.float16, torch.bfloat16, torch.float32, torch.float64,
        torch.int8, torch.int16, torch.int32, torch.int64,
    )
    if arr.dtype not in allowed:
        raise ValueError(f"{name} must be a nonempty real numeric array")
    if arr.is_floating_point() and not bool(torch.isfinite(arr).all()):
        raise ValueError(f"{name} produced nonfinite output")
    if arr.dtype == torch.float32:
        return arr
    out = arr.to(torch.float32)
    if not bool(torch.isfinite(out).all()):
        raise ValueError(f"{name} output overflows float32 (inf would result)")
    return out


def _unsigned_torch_dtypes(torch):
    return tuple(
        dt for dt in (
            torch.uint8,
            getattr(torch, "uint16", None),
            getattr(torch, "uint32", None),
        )
        if dt is not None
    )


def codes_to_int64(codes, *, name: str = "codes", maximum: int = 2**32 - 1):
    """Unsigned raw codes stored as int64 tensors (logical uint8/16/32).

    Accepts NumPy unsigned arrays and Torch uint8/16/32; returned int64
    values are the logical unsigned codes, exact in int64 up to 2**63-1.
    """
    torch = require_torch()
    arr = torch.as_tensor(codes)
    if arr.numel() == 0:
        raise ValueError(f"{name} must be a nonempty integer array")
    unsigned = _unsigned_torch_dtypes(torch)
    allowed = unsigned + (torch.int8, torch.int16, torch.int32, torch.int64)
    if arr.dtype not in allowed:
        raise ValueError(f"{name} must be an unsigned-code integer array, got {arr.dtype}")
    arr = arr.to(torch.int64)
    if int(arr.min()) < 0 or int(arr.max()) > maximum:
        raise ValueError(f"{name} codes must lie in 0..{maximum}")
    return arr


def to_device(values, device, *, dtype=None):
    torch = require_torch()
    arr = torch.as_tensor(values)
    if dtype is not None:
        arr = arr.to(dtype)
    return arr.to(device)
