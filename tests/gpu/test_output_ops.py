"""GPU transfer/YCbCr/quantize/422 ops vs util.color (seed 9901)."""

from __future__ import annotations

import numpy as np
import pytest

from brilliantisp_gpu.ops.color import (
    encode_transfer, decode_transfer, rgb_to_ycbcr, ycbcr_to_rgb,
    quantize_rgb, ycbcr_to_codes, codes_to_ycbcr,
    pack_ycbcr422, unpack_ycbcr422)
from brilliantisp_gpu.types import GpuError
from util import color as cc
from tools.gpu_validation.metrics import compare_arrays
from tools.gpu_validation.contracts import FLOAT32_NORMALIZED, FLOAT64_HELPER

SHAPES = [(1, 1), (1, 7), (7, 1), (7, 9), (8, 10)]
KNEES = {"srgb": (0.0031308, 0.04045), "rec709": (0.018, 0.081)}


def _t(a, d):
    import torch
    return torch.as_tensor(np.asarray(a), device=d)


def _check(exp, act, gate, tag):
    res = compare_arrays(np.asarray(exp), np.asarray(act), gate, scale=1.0)
    assert res["passed"], (tag, res)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("transfer", ["linear", "srgb", "rec709"])
@pytest.mark.parametrize("in_dt", [np.float32, np.float64])
@pytest.mark.parametrize("out_dt", [np.float32, np.float64])
def test_transfer_signed(tensor_device, shape, transfer, in_dt, out_dt):
    rng = np.random.RandomState(9901)
    x = (rng.rand(2, *shape, 3) * 2.5 - 0.5).astype(in_dt)
    gate = FLOAT64_HELPER if out_dt == np.float64 else FLOAT32_NORMALIZED
    e = encode_transfer(_t(x, tensor_device), transfer, dtype=out_dt)
    _check(cc.encode_transfer(x, transfer, dtype=out_dt), e.cpu().numpy(), gate,
           (shape, transfer, in_dt, out_dt))
    dd = decode_transfer(_t(x, tensor_device), transfer, dtype=out_dt)
    _check(cc.decode_transfer(x, transfer, dtype=out_dt), dd.cpu().numpy(), gate,
           (shape, transfer, in_dt, out_dt, "dec"))


@pytest.mark.parametrize("transfer", ["srgb", "rec709"])
@pytest.mark.parametrize("in_dt", [np.float32, np.float64])
def test_transfer_knees(tensor_device, transfer, in_dt):
    """Exact knees on both encode and decode thresholds, +/- np.nextafter."""
    enc_k, dec_k = KNEES[transfer]
    ev, dv = in_dt(enc_k), in_dt(dec_k)
    enc = np.array([np.nextafter(ev, in_dt(-np.inf)), ev, np.nextafter(ev, in_dt(np.inf))],
                   dtype=in_dt)
    dec = np.array([np.nextafter(dv, in_dt(-np.inf)), dv, np.nextafter(dv, in_dt(np.inf))],
                   dtype=in_dt)
    assert enc[0] < enc[1] < enc[2]
    assert dec[0] < dec[1] < dec[2]
    for vals, name in ((enc, "enc"), (dec, "dec")):
        x = np.broadcast_to(vals, (1, vals.size, 3)).copy()
        e = encode_transfer(_t(x, tensor_device), transfer, dtype=np.float64)
        _check(cc.encode_transfer(x, transfer, dtype=np.float64), e.cpu().numpy(),
               FLOAT64_HELPER, (transfer, name, in_dt))
        d = decode_transfer(_t(x, tensor_device), transfer, dtype=np.float64)
        _check(cc.decode_transfer(x, transfer, dtype=np.float64), d.cpu().numpy(),
               FLOAT64_HELPER, (transfer, name, in_dt, "dec"))


def test_transfer_linear_no_alias(tensor_device):
    """linear encode/decode must not alias the f64 input."""
    import torch
    x = _t(np.full((2, 4, 4, 3), 0.5, np.float64), tensor_device)
    before = x.clone()
    for fn in (encode_transfer, decode_transfer):
        out = fn(x, "linear", dtype=np.float64)
        out[0, 0, 0, 0] = 999.0
        np.testing.assert_array_equal(x.cpu().numpy(), before.cpu().numpy())


@pytest.mark.parametrize("gamma", [0.5, 2.2, 4.0])
@pytest.mark.parametrize("out_dt", [np.float32, np.float64])
def test_transfer_power(tensor_device, gamma, out_dt):
    rng = np.random.RandomState(9901)
    x = (rng.rand(2, 8, 10, 3) * 1.5).astype(np.float64)
    gate = FLOAT64_HELPER if out_dt == np.float64 else FLOAT32_NORMALIZED
    out = encode_transfer(_t(x, tensor_device), "power", gamma=gamma, dtype=out_dt)
    _check(cc.encode_transfer(x, "power", gamma=gamma, dtype=out_dt),
           out.cpu().numpy(), gate, (gamma, out_dt))
    dec = decode_transfer(_t(x, tensor_device), "power", gamma=gamma, dtype=out_dt)
    _check(cc.decode_transfer(x, "power", gamma=gamma, dtype=out_dt),
           dec.cpu().numpy(), gate, (gamma, out_dt, "dec"))
    with pytest.raises(ValueError):
        encode_transfer(_t(-x, tensor_device), "power", gamma=gamma)
    with pytest.raises(ValueError):
        encode_transfer(_t(x, tensor_device), "power", gamma=-gamma)
    with pytest.raises(ValueError):
        encode_transfer(_t(x, tensor_device), "power", gamma=True)
    with pytest.raises(ValueError):
        encode_transfer(_t(x, tensor_device), "fourier")
    with pytest.raises(ValueError):
        decode_transfer(_t(x, tensor_device), "fourier")


@pytest.mark.parametrize("matrix", ["bt601", "bt709"])
@pytest.mark.parametrize("out_dt", [np.float32, np.float64])
def test_ycbcr_float(tensor_device, matrix, out_dt):
    rng = np.random.RandomState(9901)
    rgb = (rng.rand(2, 8, 10, 3) * 1.4 - 0.2).astype(np.float32)
    gate = FLOAT64_HELPER if out_dt == np.float64 else FLOAT32_NORMALIZED
    fwd = rgb_to_ycbcr(_t(rgb, tensor_device), matrix=matrix, dtype=out_dt)
    _check(cc.rgb_to_ycbcr(rgb, matrix=matrix, dtype=out_dt), fwd.cpu().numpy(), gate, matrix)
    ycc = cc.rgb_to_ycbcr(rgb, matrix=matrix, dtype=out_dt)
    inv = ycbcr_to_rgb(_t(ycc, tensor_device), matrix=matrix, dtype=out_dt)
    _check(cc.ycbcr_to_rgb(ycc, matrix=matrix, dtype=out_dt),
           inv.cpu().numpy(), gate, (matrix, "inv"))


@pytest.mark.parametrize("bits", [8, 16])
def test_quantize_boundaries(tensor_device, bits):
    import torch
    n = 2**bits - 1
    rng = np.random.RandomState(9901)
    x = rng.rand(2, 8, 10, 3) * 1.4 - 0.2
    x[0, 0, 0] = (np.arange(3) + 0.5) / n
    x[0, 0, 1] = np.nextafter(0.5 / n, 0.0)
    x[1, -1, -1] = (np.arange(3) + 0.5) / n
    exp = cc.quantize_rgb(x, bit_depth=bits)
    out = quantize_rgb(_t(x, tensor_device), bit_depth=bits)
    assert out.dtype == (torch.uint8 if bits == 8 else torch.uint16)
    assert out.cpu().numpy().dtype == exp.dtype
    np.testing.assert_array_equal(out.cpu().numpy(), exp)
    for bad in (4, True, "x"):
        with pytest.raises(ValueError):
            quantize_rgb(_t(x, tensor_device), bit_depth=bad)


def test_nonfinite_rejects(tensor_device):
    x = np.full((2, 4, 4, 3), 0.5, np.float64)
    for v in (float("nan"), float("inf"), -float("inf")):
        bad = x.copy()
        bad[0, 0, 0, 0] = v
        with pytest.raises(ValueError):
            quantize_rgb(_t(bad, tensor_device), bit_depth=8)
        with pytest.raises(ValueError):
            encode_transfer(_t(bad, tensor_device), "srgb")
        with pytest.raises(ValueError):
            decode_transfer(_t(bad, tensor_device), "srgb")


def test_quantize_neutral_and_limited(tensor_device):
    """Neutral center codes + limited-range endpoints, typed unsigned."""
    import torch
    for bits, n, center in ((8, 255, 128), (16, 65535, 32768)):
        shift = 2 ** (bits - 8)
        cases = {
            (0.0, 0.0, 0.0): ("full", (0, 1, 1), "limited", (int(16 * shift),) * 3),
            (0.5, 0.5, 0.5): ("full", (center, center, center), "limited",
                             (int((16 + 109.5) * shift + 0.5), int(128 * shift), int(128 * shift))),
            (1.0, 1.0, 1.0): ("full", (n, n, n), "limited",
                             (int(235 * shift), int(240 * shift), int(240 * shift))),
        }
        for vals, (_, full_lit, _, lim_lit) in cases.items():
            ycc = np.full((2, 2, 2, 3), 0.0, np.float64)
            ycc[:] = vals
            for mode, literal in (("full", full_lit), ("limited", lim_lit)):
                exp = cc.ycbcr_to_codes(ycc, bit_depth=bits, range_mode=mode)
                out = ycbcr_to_codes(_t(ycc, tensor_device), bit_depth=bits, range_mode=mode)
                assert out.dtype == (torch.uint8 if bits == 8 else torch.uint16)
                np.testing.assert_array_equal(out.cpu().numpy(), exp)
                np.testing.assert_array_equal(
                    out.cpu().numpy(), np.broadcast_to(np.asarray(literal, exp.dtype), exp.shape),
                    err_msg=f"{bits} {mode} {vals}")


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("mode", ["full", "limited"])
def test_ycbcr_codes(tensor_device, bits, mode):
    rng = np.random.RandomState(9901)
    dt = np.uint8 if bits == 8 else np.uint16
    ycc = np.stack([(rng.rand(2, 8, 10) * 1.3 - 0.15).astype(np.float64) for _ in range(3)], -1)
    exp = cc.ycbcr_to_codes(ycc, bit_depth=bits, range_mode=mode)
    out = ycbcr_to_codes(_t(ycc, tensor_device), bit_depth=bits, range_mode=mode)
    assert out.cpu().numpy().dtype == exp.dtype
    np.testing.assert_array_equal(out.cpu().numpy(), exp)
    back = codes_to_ycbcr(_t(exp, tensor_device), bit_depth=bits, range_mode=mode, dtype=np.float64)
    _check(cc.codes_to_ycbcr(exp, bit_depth=bits, range_mode=mode, dtype=np.float64),
           back.cpu().numpy(), FLOAT64_HELPER, (bits, mode))
    over = np.zeros((2, 2, 2, 3), np.uint16); over[:] = 65535
    with pytest.raises(ValueError):
        codes_to_ycbcr(_t(over.astype(np.int64), tensor_device),
                       bit_depth=bits, range_mode=mode)
    if bits == 8:
        with pytest.raises(ValueError):
            codes_to_ycbcr(_t(over, tensor_device), bit_depth=bits, range_mode=mode)
    with pytest.raises(ValueError):
        ycbcr_to_codes(_t(ycc, tensor_device), bit_depth=bits, range_mode="weird")


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("packing", ["yuyv", "uyvy"])
@pytest.mark.parametrize("batched", [True, False])
def test_pack422(tensor_device, bits, packing, batched):
    import torch
    rng = np.random.RandomState(9901)
    dt = np.uint8 if bits == 8 else np.uint16
    codes = rng.randint(0, 2**bits - 1, size=(8, 10, 3)).astype(dt)
    codes2 = np.stack([codes, (codes // 2).astype(dt)]) if batched else codes
    exp = (np.stack([cc.pack_ycbcr422(f, packing=packing) for f in codes2])
           if batched else cc.pack_ycbcr422(codes2, packing=packing))
    out = pack_ycbcr422(_t(codes2, tensor_device), packing=packing)
    assert out.dtype == (torch.uint8 if bits == 8 else torch.uint16)
    np.testing.assert_array_equal(out.cpu().numpy(), exp)
    un = unpack_ycbcr422(_t(exp, tensor_device), packing=packing)
    exp_un = (np.stack([cc.unpack_ycbcr422(f, packing=packing) for f in exp])
              if batched else cc.unpack_ycbcr422(exp, packing=packing))
    assert un.dtype == out.dtype
    np.testing.assert_array_equal(un.cpu().numpy(), exp_un)
    with pytest.raises(ValueError):
        pack_ycbcr422(_t(np.zeros((8, 9, 3), dt), tensor_device))
    with pytest.raises(ValueError):
        pack_ycbcr422(_t(np.zeros((8, 10, 3), dt), tensor_device), packing="vvvv")
    with pytest.raises(ValueError):
        pack_ycbcr422(_t(np.zeros((8, 10, 3), np.int64), tensor_device))
    with pytest.raises(ValueError):
        unpack_ycbcr422(_t(np.zeros((8, 6), dt), tensor_device))
    with pytest.raises(ValueError):
        unpack_ycbcr422(_t(np.zeros((8, 8), dt), tensor_device), packing="yvyv")
    with pytest.raises(ValueError):
        unpack_ycbcr422(_t(np.zeros((8, 8), np.int64), tensor_device))


def test_pack422_max_safe(tensor_device):
    import torch
    codes = np.full((4, 8, 3), 65535, np.uint16)
    codes[..., 0] = 32768
    out = pack_ycbcr422(_t(codes, tensor_device))
    assert out.dtype == torch.uint16
    np.testing.assert_array_equal(out.cpu().numpy(), cc.pack_ycbcr422(codes))
    un = unpack_ycbcr422(_t(cc.pack_ycbcr422(codes), tensor_device))
    assert un.dtype == torch.uint16
    np.testing.assert_array_equal(un.cpu().numpy(), cc.unpack_ycbcr422(cc.pack_ycbcr422(codes)))
