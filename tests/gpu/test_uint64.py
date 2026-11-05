"""U64 word-pair arithmetic vs the exact Python int oracle. All checks are
bit-exact, including boundaries, long-division remainders and broadcast."""

from __future__ import annotations

import numpy as np
import pytest

from brilliantisp_gpu.ops import uint64
from brilliantisp_gpu.ops.uint64 import U64

M64 = 2**64
EDGE = [0, 1, 2**31, 2**32 - 1, 2**63, 2**64 - 1]


def _to_int(u):
    return int(uint64.to_int64(u).cpu())


def _from_int(v, device):
    torch = pytest.importorskip("torch")
    return U64(
        torch.full((1,), v >> 32, dtype=torch.int64, device=device),
        torch.full((1,), v & (2**32 - 1), dtype=torch.int64, device=device),
    )


@pytest.mark.parametrize("v", [x for x in EDGE if x <= 2**63 - 1])
def test_roundtrip_int64(tensor_device, v):
    torch = pytest.importorskip("torch")
    t = torch.tensor([v], dtype=torch.int64, device=tensor_device)
    u = uint64.from_nonnegative_int64(t)
    assert _to_int(u) == v


def _u64_value(u):
    return (int(u.hi.cpu()[0]) << 32) | int(u.lo.cpu()[0])


@pytest.mark.parametrize("a", EDGE)
@pytest.mark.parametrize("b", EDGE)
def test_add_mod_2p64(tensor_device, a, b):
    ua, ub = _from_int(a, tensor_device), _from_int(b, tensor_device)
    assert _u64_value(uint64.add(ua, ub)) == (a + b) % M64


@pytest.mark.parametrize("a", EDGE)
@pytest.mark.parametrize("b", EDGE)
def test_multiply_low64(tensor_device, a, b):
    ua, ub = _from_int(a, tensor_device), _from_int(b, tensor_device)
    r = uint64.multiply(ua, ub)
    value = (int(r.hi.cpu()[0]) << 32) | int(r.lo.cpu()[0])
    assert value == (a * b) % M64


@pytest.mark.parametrize("a", EDGE)
@pytest.mark.parametrize("d", [1, 3, 2**16 + 1, 2**32 - 1, 2**32, 2**33 - 1])
def test_divmod_u33_exact(tensor_device, a, d):
    ua = _from_int(a, tensor_device)
    q, rem = uint64.divmod_u33(ua, d)
    qv = (int(q.hi.cpu()[0]) << 32) | int(q.lo.cpu()[0])
    assert qv == a // d
    assert int(rem.cpu()[0]) == a % d


def test_divmod_u33_broadcast(tensor_device):
    torch = pytest.importorskip("torch")
    rng = np.random.RandomState(0)
    vals = rng.randint(0, 2**32, size=(4, 5)).astype(np.int64)
    divisors = np.array([1, 7, 2**20, 2**32 - 1, 2**33 - 1], dtype=np.int64)
    u = uint64.from_nonnegative_int64(torch.as_tensor(vals, device=tensor_device))
    q, rem = uint64.divmod_u33(u, torch.as_tensor(divisors, device=tensor_device)[None, :])
    qv = (q.hi.cpu().numpy() << 32) | q.lo.cpu().numpy()
    assert qv.shape == (4, 5)
    np.testing.assert_array_equal(qv, vals // divisors[None, :])
    np.testing.assert_array_equal(rem.cpu().numpy(), vals % divisors[None, :])


def test_comparisons(tensor_device):
    for a, b in [(0, 1), (2**32 - 1, 2**32), (2**64 - 1, 2**64 - 2), (5, 5)]:
        ua, ub = _from_int(a, tensor_device), _from_int(b, tensor_device)
        assert bool(uint64.lt(ua, ub).cpu()[0]) == (a < b)
        assert bool(uint64.gt(ua, ub).cpu()[0]) == (a > b)
        assert bool(uint64.eq(ua, ub).cpu()[0]) == (a == b)
        assert bool(uint64.ge(ua, ub).cpu()[0]) == (a >= b)
        assert bool(uint64.le(ua, ub).cpu()[0]) == (a <= b)


def test_random_words_exact(tensor_device):
    torch = pytest.importorskip("torch")
    rng = np.random.RandomState(42)
    vals = rng.randint(0, 2**63 - 1, size=(3, 4), dtype=np.int64)
    u = uint64.from_nonnegative_int64(torch.as_tensor(vals, device=tensor_device))
    d = int(rng.randint(1, 2**33 - 1))
    q, rem = uint64.divmod_u33(u, d)
    qv = (q.hi.cpu().numpy() << 32) | q.lo.cpu().numpy()
    np.testing.assert_array_equal(qv, vals // d)
    np.testing.assert_array_equal(rem.cpu().numpy(), vals % d)


def test_to_int64_overflow_raises(tensor_device):
    u = _from_int(2**63, tensor_device)
    with pytest.raises(ValueError):
        uint64.to_int64(u)


def test_divmod_rejects_bad_divisor(tensor_device):
    ua = _from_int(10, tensor_device)
    for d in (0, -1, 2**33):
        with pytest.raises(ValueError):
            uint64.divmod_u33(ua, d)
    with pytest.raises(ValueError):
        uint64.divmod_u33(ua, 2.5)
    with pytest.raises(ValueError):
        uint64.divmod_u33(ua, True)


def test_from_nonnegative_int64_rejects_noninteger(tensor_device):
    torch = pytest.importorskip("torch")
    for t in (
        torch.ones(3, device=tensor_device, dtype=torch.float64),
        torch.ones(3, device=tensor_device, dtype=torch.bool),
        torch.ones(3, device=tensor_device, dtype=torch.complex128),
    ):
        with pytest.raises(ValueError):
            uint64.from_nonnegative_int64(t)


def test_u64_empty_noop(tensor_device):
    torch = pytest.importorskip("torch")
    u = uint64.from_nonnegative_int64(
        torch.empty(0, dtype=torch.int64, device=tensor_device))
    assert uint64.to_int64(u).shape == (0,)
    q, rem = uint64.divmod_u33(u, 7)
    assert q.hi.shape == (0,) and rem.shape == (0,)


def test_checked_float32_unsigned(tensor_device):
    torch = pytest.importorskip("torch")
    from brilliantisp_gpu.ops.array import checked_float32
    for np_dtype, vals in (
        (np.uint16, [0, 1, 65535]),
        (np.uint32, [0, 1, 2**32 - 1]),
    ):
        t = torch.as_tensor(np.array(vals, dtype=np_dtype), device=tensor_device)
        out = checked_float32(t, name="u")
        assert out.dtype == torch.float32
        np.testing.assert_allclose(out.cpu().numpy(), np.array(vals, np.float64),
                                   rtol=1e-6)


def test_checked_float32_rejects_bool_complex(tensor_device):
    torch = pytest.importorskip("torch")
    from brilliantisp_gpu.ops.array import checked_float32
    with pytest.raises(ValueError):
        checked_float32(torch.ones(3, dtype=torch.bool, device=tensor_device))
    with pytest.raises(ValueError):
        checked_float32(torch.ones(3, dtype=torch.complex64, device=tensor_device))
