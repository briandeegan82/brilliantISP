from __future__ import annotations

import numpy as np
import pytest

from modules.bayer_noise_reduction.bayer_noise_reduction import BayerNoiseReduction
from typing import cast, Any

from modules.color_correction_matrix.color_correction_matrix import ColorCorrectionMatrix
from modules.color_correction_matrix.color_correction_matrix_optimized import ColorCorrectionMatrixOptimized
from modules.demosaic.demosaic import Demosaic
from modules.digital_gain.digital_gain import DigitalGain
from modules.lens_shading_correction.lens_shading_correction import LensShadingCorrection
from modules.white_balance.white_balance_optimized import WhiteBalanceOptimized

ALGORITHMS = (
    "malvar", "bilinear", "vng", "vng_opt", "hamilton_adams",
    "hamilton_adams_opt", "ppg", "ppg_opt", "lmmse", "lmmse_opt",
    "lmmse_fast", "ahd", "ahd_opt",
)
PATTERNS = ("rggb", "grbg", "gbrg", "bggr")
PLATFORM = {
    "in_file": "t", "out_file": "t", "output_dir": "",
    "disable_progress_bar": True, "leave_pbar_string": False,
}


from util.isp_types import (
    BayerNoiseReductionConfig, ColorCorrectionMatrixConfig,
    DemosaicConfig, DigitalGainConfig, LensShadingCorrectionConfig,
    PlatformConfig, SensorInfo, WhiteBalanceConfig,
)

PLATFORM_CFG = cast(PlatformConfig, PLATFORM)


def sensor(h: int, w: int, pattern: str = "rggb", bits: int = 12, hdr: int = 24) -> SensorInfo:
    return cast(SensorInfo, {
        "height": h, "width": w, "bit_depth": bits,
        "hdr_bit_depth": hdr, "bayer_pattern": pattern,
        "output_bit_depth": 8, "pipeline_rgb_bit_depth": 16,
    })


def cfg_demosaic(**kw: Any) -> DemosaicConfig:
    return cast(DemosaicConfig, kw)


def cfg_ccm(**kw: Any) -> ColorCorrectionMatrixConfig:
    return cast(ColorCorrectionMatrixConfig, kw)


def cfg_dga(**kw: Any) -> DigitalGainConfig:
    return cast(DigitalGainConfig, kw)


def cfg_wb(**kw: Any) -> WhiteBalanceConfig:
    return cast(WhiteBalanceConfig, kw)


def cfg_lsc(**kw: Any) -> LensShadingCorrectionConfig:
    return cast(LensShadingCorrectionConfig, kw)


def cfg_bnr(**kw: Any) -> BayerNoiseReductionConfig:
    return cast(BayerNoiseReductionConfig, kw)


def cfa_mask(shape: tuple[int, int], pattern: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    channels = {c: np.zeros(shape, bool) for c in "rgb"}
    for c, (y, x) in zip(pattern, ((0, 0), (0, 1), (1, 0), (1, 1))):
        channels[c][y::2, x::2] = True
    return channels["r"], channels["g"], channels["b"]


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("shape", ((1, 1), (3, 5), (8, 10)))
def test_demosaic_compute_linear_preserves_known_cfa_samples(algorithm, pattern, shape) -> None:
    rng = np.random.default_rng(421)
    src = rng.random(shape, dtype=np.float32) * 65535
    d = Demosaic(src, PLATFORM_CFG, sensor(shape[0], shape[1], pattern), cfg_demosaic(is_save=False))
    out = d.compute_linear(algorithm)
    assert out.dtype == np.float32
    assert out.shape == shape + (3,)
    assert np.all(np.isfinite(out))
    masks = cfa_mask(shape, pattern)
    for c, mask in enumerate(masks):
        np.testing.assert_array_equal(out[..., c][mask], src[mask])


def test_demosaic_malvar_keeps_known_samples_on_impulse() -> None:
    src = np.zeros((9, 9), np.float32)
    src[4, 4] = 65535
    d = Demosaic(src, PLATFORM_CFG, sensor(9, 9, "rggb"), cfg_demosaic(is_save=False))
    out = d.compute_linear("malvar")
    masks = cfa_mask((9, 9), "rggb")
    for c, mask in enumerate(masks):
        np.testing.assert_array_equal(out[..., c][mask], src[mask])
    assert np.all(np.isfinite(out))
    assert float(out.min()) < 0


def test_demosaic_linear_input_validation_and_legacy_default() -> None:
    src = np.ones((4, 4), np.float32)
    d = Demosaic(src, PLATFORM_CFG, sensor(4, 4), cfg_demosaic(is_save=False))
    with pytest.raises(ValueError):
        d.compute_linear("bogus")
    bad = src.copy()
    bad[0, 0] = np.nan
    d2 = Demosaic(bad, PLATFORM_CFG, sensor(4, 4), cfg_demosaic(is_save=False))
    with pytest.raises(ValueError):
        d2.compute_linear()
    empty = Demosaic(np.zeros((0, 0), np.float32), PLATFORM_CFG, sensor(0, 0), cfg_demosaic(is_save=False))
    with pytest.raises(ValueError):
        empty.compute_linear()
    legacy = d.execute("bilinear")
    assert legacy.dtype == np.uint16


def test_demosaic_compute_linear_rejects_nonfinite_kernel_output(monkeypatch) -> None:
    src = np.full((4, 4), 100.0, np.float32)
    d = Demosaic(src, PLATFORM_CFG, sensor(4, 4), cfg_demosaic(is_save=False))
    monkeypatch.setattr(
        d, "_dispatch", lambda algo, *, cpu_only: np.full(src.shape + (3,), 1e40),
    )
    with np.errstate(all="raise"):
        with pytest.raises(ValueError):
            d.compute_linear()


def test_demosaic_compute_linear_is_cpu_only(monkeypatch) -> None:
    monkeypatch.setattr(
        "modules.demosaic.demosaic.CUPY_AVAILABLE", True, raising=False,
    )
    monkeypatch.setattr(
        "modules.demosaic.demosaic.MALCUPY",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("GPU forbidden")),
        raising=False,
    )
    src = np.full((4, 4), 100, np.float32)
    d = Demosaic(src, PLATFORM_CFG, sensor(4, 4), cfg_demosaic(is_save=False))
    out = d.compute_linear("malvar")
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize(
    "cls", (ColorCorrectionMatrix, ColorCorrectionMatrixOptimized)
)
def test_ccm_compute_linear_exact_identity_and_matrix(cls) -> None:
    img = np.array([[[-0.125, 0.5, 2.0]]], np.float32)
    cfg = cfg_ccm(
        is_enable=False, is_save=False,
        corrected_red=[1.0, 0.0, 0.0],
        corrected_green=[0.0, 1.0, 0.0],
        corrected_blue=[0.0, 0.0, 1.0],
    )
    ccm = cls(img, PLATFORM_CFG, sensor(1, 1), cfg)
    np.testing.assert_array_equal(ccm.compute_linear(), img)
    cfg2 = cfg_ccm(
        is_enable=True, is_save=False,
        corrected_red=[1.0, 0.0, 0.0],
        corrected_green=[0.0, 1.0, 0.0],
        corrected_blue=[0.0, 0.0, 1.0],
    )
    ccm2 = cls(img, PLATFORM_CFG, sensor(1, 1), cfg2)
    np.testing.assert_array_equal(ccm2.compute_linear(), img)
    cfg3 = cfg_ccm(
        is_enable=True, is_save=False,
        corrected_red=[1.5, -0.5, 0.0],
        corrected_green=[0.0, 1.0, 0.0],
        corrected_blue=[0.0, 0.0, 1.0],
    )
    ccm3 = cls(img, PLATFORM_CFG, sensor(1, 1), cfg3)
    out = ccm3.compute_linear()
    np.testing.assert_allclose(
        out[0, 0], [-0.4375, 0.5, 2.0], rtol=0, atol=1e-7,
    )
    bad = dict(cfg3)
    bad["corrected_red"] = [np.nan, 0.0, 0.0]
    with pytest.raises(ValueError):
        cls(img, PLATFORM_CFG, sensor(1, 1), cast(ColorCorrectionMatrixConfig, bad)).compute_linear()
    # Malformed matrix must not be read when disabled
    cfg_off = dict(cfg3)
    cfg_off["is_enable"] = False
    cfg_off["corrected_red"] = [np.nan, 0.0, 0.0]
    ident = cls(img, PLATFORM_CFG, sensor(1, 1), cast(ColorCorrectionMatrixConfig, cfg_off))
    np.testing.assert_array_equal(ident.compute_linear(), img)
    # Huge finite coefficient must not silently become inf on the F32 cast
    huge = dict(cfg3)
    huge["corrected_red"] = [1e40, 0.0, 0.0]
    with np.errstate(all="raise"):
        with pytest.raises(ValueError):
            cls(img, PLATFORM_CFG, sensor(1, 1), cast(ColorCorrectionMatrixConfig, huge)).compute_linear()


def test_dg_compute_linear_gain_and_disabled() -> None:
    img = np.array([[0.125, 4095.0]], np.float32)
    parm = cfg_dga(
        is_enable=True, is_save=False, is_debug=False,
        is_auto=False, gain_array=[1.0, 2.0, 4.0],
        current_gain=1, ae_feedback=None,
        exposure_correction_mode="step",
    )
    dg = DigitalGain(img, PLATFORM_CFG, sensor(1, 2), parm)
    out, idx = dg.compute_linear()
    assert idx == 1
    np.testing.assert_allclose(out, [[0.25, 8190.0]], rtol=0, atol=1e-6)
    off = cfg_dga(is_enable=False, is_save=False, is_debug=False,
                  is_auto=False, gain_array=[1.0, 2.0, 4.0],
                  current_gain=1, ae_feedback=None)
    dg2 = DigitalGain(img, PLATFORM_CFG, sensor(1, 2), off)
    huge = cfg_dga(is_enable=True, is_save=False, is_debug=False,
                   is_auto=False, gain_array=[1e40], current_gain=0,
                   ae_feedback=None)
    with np.errstate(all="raise"):
        with pytest.raises(ValueError):
            DigitalGain(img, PLATFORM_CFG, sensor(1, 2), huge).compute_linear()
    out2, _ = dg2.compute_linear()
    np.testing.assert_array_equal(out2, img.astype(np.float32))


@pytest.mark.parametrize("pattern", PATTERNS)
def test_wb_compute_linear_manual_sites(pattern: str) -> None:
    img = np.full((4, 4), 4095.0, np.uint16)
    parm = cfg_wb(
        is_enable=True, is_save=False, is_debug=False,
        is_auto=False, r_gain=2.0, b_gain=0.5,
    )
    wb = WhiteBalanceOptimized(img, PLATFORM_CFG, sensor(4, 4, pattern), parm, (9.0, 9.0))
    out = wb.compute_linear()
    assert out.dtype == np.float32
    masks = cfa_mask((4, 4), pattern)
    np.testing.assert_allclose(out[masks[0]], 8190.0, rtol=0, atol=1e-3)
    np.testing.assert_allclose(out[masks[1]], 4095.0, rtol=0, atol=1e-3)
    np.testing.assert_allclose(out[masks[2]], 2047.5, rtol=0, atol=1e-3)


def test_wb_compute_linear_auto_uses_awb_gains() -> None:
    img = np.full((4, 4), 1000.0, np.uint16)
    parm = cfg_wb(
        is_enable=True, is_save=False, is_debug=False,
        is_auto=True, r_gain=9.0, b_gain=9.0,
    )
    wb = WhiteBalanceOptimized(img, PLATFORM_CFG, sensor(4, 4, "rggb"), parm, (2.0, 0.5))
    huge_img = np.full((4, 4), 4095.0, np.uint16)
    parm_h = cfg_wb(is_enable=True, is_save=False, is_debug=False,
                    is_auto=False, r_gain=1e40, b_gain=1.0)
    with np.errstate(all="raise"):
        with pytest.raises(ValueError):
            WhiteBalanceOptimized(huge_img, PLATFORM_CFG, sensor(4, 4, "rggb"), parm_h, (1.0, 1.0)).compute_linear()
    out = wb.compute_linear()
    masks = cfa_mask((4, 4), "rggb")
    np.testing.assert_allclose(out[masks[0]], 2000.0, rtol=0, atol=1e-2)
    np.testing.assert_allclose(out[masks[2]], 500.0, rtol=0, atol=1e-2)


def test_lsc_compute_linear_zero_and_corner_gain() -> None:
    img = np.full((5, 5), 4095.0, np.uint16)
    zero = cfg_lsc(is_enable=True, is_save=False)
    lsc = LensShadingCorrection(img, PLATFORM_CFG, sensor(5, 5), zero)
    np.testing.assert_allclose(lsc.compute_linear(), img, rtol=0, atol=1e-3)
    corner = cfg_lsc(is_enable=True, is_save=False,
                     r_k1=1.0, gr_k1=1.0, gb_k1=1.0, b_k1=1.0)
    lsc2 = LensShadingCorrection(img, PLATFORM_CFG, sensor(5, 5), corner)
    out = lsc2.compute_linear()
    np.testing.assert_allclose(out[0, 0], 8190.0, rtol=0, atol=1e-2)
    np.testing.assert_allclose(out[2, 2], 4095.0, rtol=0, atol=1e-2)


def test_lsc_no_observed_max_inference() -> None:
    img = np.full((4, 4), 100.0, np.uint16)
    lsc = LensShadingCorrection(img, PLATFORM_CFG, sensor(4, 4, hdr=24), cfg_lsc(is_enable=True, is_save=False))
    assert lsc.output_max == 2**24 - 1


def test_bnr_compute_linear_constant_odd_and_tiny() -> None:
    hdr = 24
    hdr_max = float(2**hdr - 1)
    const = np.full((9, 7), 2 * hdr_max, np.float32)
    parm = cfg_bnr(
        is_enable=True, is_save=False, filter_window=3,
        r_std_dev_s=1.0, r_std_dev_r=1.0,
        g_std_dev_s=1.0, g_std_dev_r=1.0,
        b_std_dev_s=1.0, b_std_dev_r=1.0,
    )
    bnr = BayerNoiseReduction(const, sensor(9, 7, hdr=hdr), parm, PLATFORM_CFG, cpu_only=True)
    out = bnr.compute_linear()
    assert out.shape == const.shape and out.dtype == np.float32
    np.testing.assert_allclose(out, const, rtol=0, atol=1e-6 * hdr_max)
    tiny = np.full((1, 3), 5.0, np.float32)
    bnr2 = BayerNoiseReduction(tiny, sensor(1, 3, hdr=hdr), parm, PLATFORM_CFG, cpu_only=True)
    np.testing.assert_array_equal(bnr2.compute_linear(), tiny)


def test_bnr_compute_linear_is_cpu_only(monkeypatch) -> None:
    monkeypatch.setattr(
        "modules.bayer_noise_reduction.bayer_noise_reduction.GPU_VERSION_AVAILABLE",
        True,
    )
    parm = cfg_bnr(is_enable=True, is_save=False, filter_window=3,
                   r_std_dev_s=1.0, r_std_dev_r=1.0,
                   g_std_dev_s=1.0, g_std_dev_r=1.0,
                   b_std_dev_s=1.0, b_std_dev_r=1.0)
    img = np.full((4, 4), 10.0, np.float32)
    bnr = BayerNoiseReduction(img, sensor(4, 4), parm, PLATFORM_CFG, cpu_only=True)
    assert bnr.use_gpu is False
    out = bnr.compute_linear()
    assert np.all(np.isfinite(out))


def test_clahe_scaled_constant_identity_and_mixed_tile() -> None:
    from modules.ldci.clahe import apply_luma
    src = np.full((4, 4), 13107, np.uint16)
    out = apply_luma(src, scale=65535, wind=64, clip_limit=2)
    np.testing.assert_allclose(out, 13107, rtol=0, atol=1)
    mixed = np.array([[0.2, 0.2, 0.8, 0.8], [0.2, 0.2, 0.8, 0.8]], np.float32)
    out2 = apply_luma(mixed, scale=1, wind=2, clip_limit=2)
    np.testing.assert_allclose(out2, mixed, rtol=0, atol=1e-7)
    big = np.linspace(0.0, 1.0, 129 * 131, dtype=np.float32).reshape(129, 131)
    out3 = apply_luma(big, scale=1, wind=64, clip_limit=2)
    assert out3.shape == big.shape and np.all(np.isfinite(out3))
