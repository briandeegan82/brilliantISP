"""Exact unsigned 64-bit integer arithmetic on (hi, lo) int64 word pairs.

Torch lacks uint64; each U64 value is stored as two nonnegative int64
words, each <= 2**32-1, representing ``hi * 2**32 + lo`` in 0..2**64-1.
All arithmetic is exact integer work — no double-precision division is
ever used to approximate results.
"""

from __future__ import annotations

from typing import NamedTuple

from ..types import require_torch

_WORD = 2**32 - 1
_MASK16 = 2**16 - 1


class U64(NamedTuple):
    hi: "object"
    lo: "object"


def _check_words(u: U64) -> U64:
    torch = require_torch()
    hi = torch.as_tensor(u.hi)
    lo = torch.as_tensor(u.lo)
    if hi.dtype != torch.int64 or lo.dtype != torch.int64:
        raise ValueError("U64 words must be int64 tensors")
    if hi.numel() and (
        int(hi.min()) < 0 or int(hi.max()) > _WORD
        or int(lo.min()) < 0 or int(lo.max()) > _WORD
    ):
        raise ValueError("U64 words must lie in 0..2**32-1")
    return U64(hi, lo)


def from_nonnegative_int64(t) -> U64:
    torch = require_torch()
    arr = torch.as_tensor(t)
    if arr.dtype == torch.bool or arr.is_complex() or arr.is_floating_point():
        raise ValueError(
            f"from_nonnegative_int64 requires an integer tensor, got {arr.dtype}"
        )
    if arr.dtype != torch.int64:
        arr = arr.to(torch.int64)
    if arr.numel() and int(arr.min()) < 0:
        raise ValueError("from_nonnegative_int64 requires values >= 0")
    return U64(arr >> 32, arr & _WORD)


def from_scalar(value: int, like) -> U64:
    torch = require_torch()
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 2**64 - 1:
        raise ValueError(f"scalar must be an integer in 0..2**64-1, got {value!r}")
    ref = torch.as_tensor(like.hi if isinstance(like, U64) else like)
    hi = torch.full(ref.shape, value >> 32, dtype=torch.int64, device=ref.device)
    lo = torch.full(ref.shape, value & _WORD, dtype=torch.int64, device=ref.device)
    return U64(hi, lo)


def zeros_like(ref) -> U64:
    torch = require_torch()
    r = torch.as_tensor(ref.hi if isinstance(ref, U64) else ref)
    z = torch.zeros(r.shape, dtype=torch.int64, device=r.device)
    return U64(z.clone(), z)


def to_int64(u: U64, *, name: str = "u64") -> "object":
    torch = require_torch()
    u = _check_words(u)
    if u.hi.numel() and int(u.hi.max()) > 2**31 - 1:
        raise ValueError(f"{name} value overflows int64 (>= 2**63)")
    return (u.hi << 32) | u.lo


def add(a: U64, b: U64) -> U64:
    torch = require_torch()
    s = a.lo + b.lo
    lo = s & _WORD
    hi = (a.hi + b.hi + (s >> 32)) & _WORD
    return U64(hi, lo)


def _limbs(word):
    return word & _MASK16, word >> 16


def multiply(a: U64, b: U64) -> U64:
    """Low 64 bits of a*b via 16-bit limbs with signed-int64-safe carries."""
    torch = require_torch()
    a0, a1 = _limbs(a.lo)
    a2, a3 = _limbs(a.hi)
    b0, b1 = _limbs(b.lo)
    b2, b3 = _limbs(b.hi)
    c0 = a0 * b0
    r0 = c0 & _MASK16
    c1 = a0 * b1 + a1 * b0 + (c0 >> 16)
    r1 = c1 & _MASK16
    c2 = a0 * b2 + a1 * b1 + a2 * b0 + (c1 >> 16)
    r2 = c2 & _MASK16
    c3 = a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0 + (c2 >> 16)
    r3 = c3 & _MASK16
    return U64((r2 | (r3 << 16)), (r0 | (r1 << 16)))


def mul_scalar(a: U64, s: int) -> U64:
    torch = require_torch()
    if isinstance(s, bool) or not isinstance(s, int) or not 0 <= s <= 2**64 - 1:
        raise ValueError(f"scalar multiplier must be in 0..2**64-1, got {s!r}")
    other = U64(
        torch.full(a.hi.shape, s >> 32, dtype=torch.int64, device=a.hi.device),
        torch.full(a.hi.shape, s & _WORD, dtype=torch.int64, device=a.hi.device),
    )
    return multiply(a, other)


def _cmp(a: U64, b: U64, op: str):
    torch = require_torch()
    if op == "eq":
        return (a.hi == b.hi) & (a.lo == b.lo)
    if op == "lt":
        return (a.hi < b.hi) | ((a.hi == b.hi) & (a.lo < b.lo))
    if op == "gt":
        return (a.hi > b.hi) | ((a.hi == b.hi) & (a.lo > b.lo))
    raise ValueError(op)


def lt(a: U64, b: U64):
    return _cmp(a, b, "lt")


def gt(a: U64, b: U64):
    return _cmp(a, b, "gt")


def eq(a: U64, b: U64):
    return _cmp(a, b, "eq")


def ge(a: U64, b: U64):
    torch = require_torch()
    return ~lt(a, b)


def le(a: U64, b: U64):
    torch = require_torch()
    return ~gt(a, b)


def where(cond, a: U64, b: U64) -> U64:
    torch = require_torch()
    return U64(torch.where(cond, a.hi, b.hi), torch.where(cond, a.lo, b.lo))


def minimum(a: U64, b: U64) -> U64:
    return where(lt(a, b), a, b)


def divmod_u33(a: U64, divisor) -> tuple[U64, "object"]:
    """Exact long division by a divisor in 1..2**33-1 (int or broadcast tensor).

    Returns ``(quotient U64, remainder int64)``. Runs the fixed 64-step
    binary long division; the running remainder stays below ``2 * divisor``
    (< 2**34) so every step is exact signed int64 work.
    """
    torch = require_torch()
    a = _check_words(a)
    if isinstance(divisor, int):
        if isinstance(divisor, bool) or not 1 <= divisor <= 2**33 - 1:
            raise ValueError(f"divisor must be in 1..2**33-1, got {divisor!r}")
        div = torch.full(a.hi.shape, divisor, dtype=torch.int64, device=a.hi.device)
    else:
        div = torch.as_tensor(divisor)
        if div.dtype == torch.bool or div.is_complex() or div.is_floating_point():
            raise ValueError(
                f"divisor must be an integer tensor in 1..2**33-1, got {div.dtype}"
            )
        if div.dtype != torch.int64:
            div = div.to(torch.int64)
        if div.device != a.hi.device:
            div = div.to(a.hi.device)
        if int(div.min()) < 1 or int(div.max()) > 2**33 - 1:
            raise ValueError("divisor values must lie in 1..2**33-1")
    rem = torch.zeros(torch.broadcast_shapes(a.hi.shape, div.shape),
                      dtype=torch.int64, device=a.hi.device)
    qhi = torch.zeros_like(rem)
    qlo = torch.zeros_like(rem)
    for i in range(63, -1, -1):
        if i >= 32:
            bit = (a.hi >> (i - 32)) & 1
        else:
            bit = (a.lo >> i) & 1
        rem = (rem << 1) | bit
        take = rem >= div
        rem = torch.where(take, rem - div, rem)
        bitset = take.to(torch.int64)
        if i >= 32:
            qhi = qhi + (bitset << (i - 32))
        else:
            qlo = qlo + (bitset << i)
    return U64(qhi, qlo), rem
