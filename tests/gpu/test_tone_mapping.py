"""fixed_reinhard / map_fixed_native vs the CPU private backend oracle.

Integer fixed path must match ``backends._fixed_reinhard`` exactly; the
native float32 bridge uses the approved FLOAT32_NORMALIZED gate.
"""

from __future__ import annotations

import numpy as np
import pytest

from modules.tone_mapping.backends import _fixed_reinhard as cpu_fixed_reinhard
from modules.tone_mapping.native import _map_plane
from modules.tone_mapping.registry import ToneMappingSpec

from tools.gpu_validation.contracts import FLOAT32_NORMALIZED

from brilliantisp_gpu.blocks.tone_mapping import fixed_reinhard, map_fixed_native


def _spec(**kw):
    base = dict(
        operator="reinhard", backend="fixed", normalization="declared_range",
        exposure_ev=0.0, domain_scale=1.0, normalize_output=False,
        parameters=(), lut_domain=16.0, enabled=True,
    )
    base.update(kw)
    return ToneMappingSpec(**base)


SPECS = [
    _spec(),
    _spec(normalize_output=True),
    _spec(normalization="per_frame"),
    _spec(normalization="per_frame", normalize_output=True),
    _spec(exposure_ev=2.5, lut_domain=8.0),
    _spec(exposure_ev=-3.0, normalize_output=True),
    _spec(domain_scale=4.0, parameters=(("knee", 0.5), ("strength", 2.0))),
    _spec(parameters=(("knee", 0.001), ("strength", 8.0)), normalize_output=True),
    _spec(parameters=(("knee", 12.0), ("strength", 0.05))),
    _spec(lut_domain=1.0, exposure_ev=4.0, normalize_output=True),
    _spec(domain_scale=64.0),
]


def _codes(seed, shape, maximum):
    rng = np.random.RandomState(seed)
    return rng.randint(0, maximum + 1, size=shape).astype(np.uint32)


@pytest.mark.parametrize("spec", SPECS, ids=str)
@pytest.mark.parametrize("dtype,shape,seed", [
    (np.uint16, (9, 11), 11), (np.uint32, (9, 11), 12),
])
def test_fixed_reinhard_matches_cpu(tensor_device, spec, dtype, shape, seed):
    torch = pytest.importorskip("torch")
    input_max = int(np.iinfo(dtype).max)
    codes = _codes(seed, shape, input_max).astype(dtype)
    codes[0, 0] = 0
    codes[0, 1] = input_max
    codes[-1, -1] = 1
    expected = cpu_fixed_reinhard(codes, spec, float(input_max), 65535, None)
    actual = fixed_reinhard(
        torch.as_tensor(codes.astype(np.int64), device=tensor_device),
        config=spec, input_max=input_max, output_max=65535,
    ).cpu().numpy()
    np.testing.assert_array_equal(actual, expected)


def test_fixed_reinhard_batch_independent_stats(tensor_device):
    torch = pytest.importorskip("torch")
    spec = _spec(normalization="per_frame", normalize_output=True)
    codes = _codes(9, (3, 7, 9), 4095)
    codes[1] = 2048
    expected = np.stack([
        cpu_fixed_reinhard(codes[b], spec, 4095.0, 65535, None) for b in range(3)
    ])
    actual = fixed_reinhard(
        torch.as_tensor(codes.astype(np.int64), device=tensor_device),
        config=spec, input_max=4095, output_max=65535,
    ).cpu().numpy()
    np.testing.assert_array_equal(actual, expected)


def test_fixed_reinhard_input_reference(tensor_device):
    torch = pytest.importorskip("torch")
    spec = _spec(normalize_output=True)
    codes = _codes(4, (8, 8), 2**32 - 1)
    expected = cpu_fixed_reinhard(
        codes, spec, float(2**32 - 1), 65535, None,
        reference=2**24.0, input_reference=2**24,
    )
    actual = fixed_reinhard(
        torch.as_tensor(codes.astype(np.int64), device=tensor_device),
        config=spec, input_max=2**32 - 1, output_max=65535,
        input_reference=2**24,
    ).cpu().numpy()
    np.testing.assert_array_equal(actual, expected)


def test_fixed_reinhard_rejects_out_of_range(tensor_device):
    torch = pytest.importorskip("torch")
    spec = _spec()
    codes = np.array([[0, 5000]], dtype=np.int64)
    with pytest.raises(Exception):
        fixed_reinhard(torch.as_tensor(codes, device=tensor_device),
                       config=spec, input_max=4095)


NATIVE_SPECS = [
    _spec(),
    _spec(normalize_output=True),
    _spec(normalization="per_frame"),
    _spec(normalization="per_frame", normalize_output=True),
    _spec(exposure_ev=2.0, lut_domain=8.0, normalize_output=True),
    _spec(parameters=(("knee", 0.4), ("strength", 1.5))),
]

NATIVE_CASES = [
    ((7, 9), 21, 1.5, -0.1),
    ((2, 7, 9), 22, 1.5, -0.1),
    ((7, 9), 23, 0.9, 0.0),
]


@pytest.mark.parametrize("spec", NATIVE_SPECS, ids=str)
@pytest.mark.parametrize("shape,seed,scale,offset", NATIVE_CASES)
def test_map_fixed_native_matches_cpu(tensor_device, spec, shape, seed, scale, offset):
    torch = pytest.importorskip("torch")
    rng = np.random.RandomState(seed)
    y = rng.rand(*shape) * scale + offset
    if y.ndim == 2:
        expected = _map_plane(np.clip(y, 0.0, None).astype(np.float64), spec, None)
        np.testing.assert_allclose(
            map_fixed_native(torch.as_tensor(y, device=tensor_device), config=spec).cpu().numpy(),
            expected, atol=FLOAT32_NORMALIZED.atol, rtol=FLOAT32_NORMALIZED.rtol,
        )
    else:
        for b in range(y.shape[0]):
            expected_b = _map_plane(np.clip(y[b], 0.0, None).astype(np.float64), spec, None)
            actual_b = map_fixed_native(
                torch.as_tensor(y[b:b + 1], device=tensor_device), config=spec
            ).cpu().numpy()[0]
            np.testing.assert_allclose(
                actual_b, expected_b,
                atol=FLOAT32_NORMALIZED.atol, rtol=FLOAT32_NORMALIZED.rtol,
            )


def test_map_fixed_native_mixed_bhw_per_frame(tensor_device):
    """Genuine mixed BHW batch: one call, per-frame independent stats."""
    torch = pytest.importorskip("torch")
    spec = _spec(normalization="per_frame", normalize_output=True)
    rng = np.random.RandomState(31)
    y = np.stack([
        rng.rand(6, 8) * 0.3,
        rng.rand(6, 8) * 4.0,
        np.full((6, 8), 0.7),
        rng.rand(6, 8) * 1.5 - 0.2,
    ])
    actual = map_fixed_native(
        torch.as_tensor(y, device=tensor_device), config=spec
    ).cpu().numpy()
    for b in range(y.shape[0]):
        expected_b = _map_plane(np.clip(y[b], 0.0, None).astype(np.float64), spec, None)
        np.testing.assert_allclose(
            actual[b], expected_b,
            atol=FLOAT32_NORMALIZED.atol, rtol=FLOAT32_NORMALIZED.rtol,
            err_msg=f"batch frame {b}",
        )
