from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from modules.color_correction_matrix.color_correction_matrix import ColorCorrectionMatrix
from modules.color_correction_matrix.color_correction_matrix_optimized import ColorCorrectionMatrixOptimized
from modules.dead_pixel_correction.dynamic_dpc import DynamicDPC
from modules.dead_pixel_correction.dynamic_dpc_numba_optimized import DynamicDPCNumbaOptimized
from modules.white_balance.white_balance_optimized import WhiteBalanceOptimized
from util.isp_types import ColorCorrectionMatrixConfig, DeadPixelCorrectionConfig, PlatformConfig, SensorInfo, WhiteBalanceConfig
from util.native_array import as_float32

PLATFORM = cast(PlatformConfig, {})
SENSOR = cast(SensorInfo, {"height": 2, "width": 2, "bit_depth": 12, "hdr_bit_depth": 24, "bayer_pattern": "rggb"})


def test_float32_rounding_underflow_is_not_an_overflow_or_a_nan() -> None:
    with np.errstate(all="raise"):
        np.testing.assert_array_equal(as_float32(np.array([1e-100, -1e-100]), name="tiny"), [0, 0])
        with pytest.raises(ValueError, match="float32"):
            as_float32(np.array([1e100]), name="large")


@pytest.mark.parametrize("cls", (ColorCorrectionMatrix, ColorCorrectionMatrixOptimized))
def test_disabled_ccm_reuses_float32_without_evaluating_matrix(cls) -> None:
    source = np.full((2, 2, 3), 0.5, np.float32)
    config = cast(ColorCorrectionMatrixConfig, {"is_enable": False, "is_save": False, "corrected_red": None})
    assert cls(source, PLATFORM, SENSOR, config).compute_linear() is source


@pytest.mark.parametrize("cls", (ColorCorrectionMatrix, ColorCorrectionMatrixOptimized))
def test_matrix_arithmetic_overflow_has_controlled_value_error(cls) -> None:
    source = np.full((2, 2, 3), 1e30, np.float32)
    config = cast(ColorCorrectionMatrixConfig, {
        "is_enable": True, "is_save": False, "corrected_red": [1e308, 0, 0],
        "corrected_green": [0, 1, 0], "corrected_blue": [0, 0, 1],
    })
    with np.errstate(all="raise"):
        with pytest.raises(ValueError, match="nonfinite"):
            cls(source, PLATFORM, SENSOR, config).compute_linear()


def test_wb_arithmetic_overflow_has_controlled_value_error() -> None:
    source = np.full((2, 2), 1e30, np.float32)
    config = cast(WhiteBalanceConfig, {
        "is_enable": True, "is_auto": False, "is_debug": False, "is_save": False,
        "r_gain": 1e308, "b_gain": 1,
    })
    with np.errstate(all="raise"):
        with pytest.raises(ValueError, match="nonfinite"):
            WhiteBalanceOptimized(source, PLATFORM, SENSOR, config, (1, 1)).compute_linear()


@pytest.mark.parametrize("cls", (DynamicDPC, DynamicDPCNumbaOptimized))
def test_dpc_does_not_reshape_different_geometry_with_the_same_sample_count(cls) -> None:
    source = np.zeros((1, 4), np.uint16)
    config = cast(DeadPixelCorrectionConfig, {"is_enable": True, "is_debug": False, "is_save": False, "dp_threshold": 8})
    with pytest.raises(ValueError, match="shape"):
        cls(source, SENSOR, config).dynamic_dpc()
