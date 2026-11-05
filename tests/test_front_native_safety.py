from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import yaml

from modules.black_level_correction.black_level_correction import BlackLevelCorrection
from modules.oecf.oecf import OECF
from modules.pwc_generation.pwc_generation import PiecewiseCurve
from util.bayer_repair import repair_zero_green_cfa
from util.isp_types import BayerPattern, BlackLevelCorrectionConfig, PlatformConfig, SensorInfo


BASE = yaml.safe_load((Path(__file__).resolve().parents[1] / "config/SVS_cam.yml").read_text())
MAX32 = 2**32 - 1


def sensor(bits: int, pattern: BayerPattern = "rggb") -> SensorInfo:
    result = copy.deepcopy(BASE["sensor_info"])
    result.update(bit_depth=bits, hdr_bit_depth=32, bayer_pattern=pattern, width=2, height=2,
                  data_format="uint32" if bits > 16 else "uint16")
    return cast(SensorInfo, result)


def parameters(section: str, **updates: Any) -> dict[str, Any]:
    result = copy.deepcopy(BASE[section])
    result.update(is_enable=True, is_save=False, is_debug=False)
    result.update(updates)
    return result


def blc(image: np.ndarray, pattern: BayerPattern = "rggb", **updates: Any) -> np.ndarray:
    config = parameters("black_level_correction", **updates)
    return BlackLevelCorrection(
        image, cast(PlatformConfig, BASE["platform"]), sensor(32, pattern),
        cast(BlackLevelCorrectionConfig, config),
    ).apply_blc_parameters()


@pytest.mark.parametrize(("pattern", "expected"), (
    ("rggb", [[MAX32 - 1, MAX32 - 3], [0, 0]]),
    ("bggr", [[MAX32 - 4, MAX32 - 4], [0, 3]]),
    ("grbg", [[MAX32 - 2, MAX32 - 2], [0, 1]]),
    ("gbrg", [[MAX32 - 3, MAX32 - 5], [0, 2]]),
))
def test_blc_uint32_phase_and_subtraction(pattern: BayerPattern, expected: list[list[int]]) -> None:
    source = np.array([[MAX32, MAX32 - 1], [0, 4]], np.uint32)
    with np.errstate(all="raise"):
        actual = blc(source, pattern, is_linear=False, r_offset=1, gr_offset=2, gb_offset=3, b_offset=4)
    assert actual.dtype == np.uint32
    np.testing.assert_array_equal(actual, np.asarray(expected, np.uint32))


def test_blc_uint32_linearization_and_tiny_positive_span() -> None:
    source = np.array([[1, 9], [17, 0]], np.uint32)
    config: dict[str, Any] = dict(is_linear=True, r_offset=1, gr_offset=1, gb_offset=1, b_offset=1,
                                  r_sat=17, gr_sat=17, gb_sat=17, b_sat=17)
    with np.errstate(all="raise"):
        actual = blc(source, **config)
        np.testing.assert_array_equal(actual, np.array([[0, 2147483647], [MAX32, 0]], np.uint32))
        config.update(r_offset=0, gr_offset=0, gb_offset=0, b_offset=0,
                      r_sat=1e-300, gr_sat=1e-300, gb_sat=1e-300, b_sat=1e-300)
        np.testing.assert_array_equal(blc(np.ones((2, 2), np.uint32), **config), MAX32)


@pytest.mark.parametrize("saturation", (0, -1, np.nan, np.inf))
def test_blc_rejects_invalid_active_span(saturation: float) -> None:
    with np.errstate(all="raise"), pytest.raises(ValueError, match="saturation"):
        blc(np.zeros((2, 2), np.uint32), is_linear=True, r_offset=0, r_sat=saturation)


def test_blc_does_not_read_inactive_saturations() -> None:
    actual = blc(np.full((2, 2), 10, np.uint32), is_linear=False,
                 r_offset=1, gr_offset=1, gb_offset=1, b_offset=1, r_sat="unused")
    np.testing.assert_array_equal(actual, 9)


def test_pwc_uint32_endpoint_clipped_before_cast() -> None:
    config = parameters("companding", companded_pin=[0, 3], companded_pout=[0, 2**32],
                        pre_linearization_black_level=0, post_linearization_black_level=0)
    with np.errstate(all="raise"):
        actual = PiecewiseCurve(np.array([[0, 1], [2, 3]], np.uint32), BASE["platform"], sensor(2), config).execute()
    np.testing.assert_array_equal(actual, np.array([[0, 1431655765], [2863311530, MAX32]], np.uint32))


def test_pwc_uint32_input_uses_sparse_knots_without_giant_lut(monkeypatch) -> None:
    config = parameters("companding", companded_pin=[0, MAX32], companded_pout=[0, MAX32],
                        pre_linearization_black_level=0, post_linearization_black_level=0)
    source = np.array([[0, 2**31, MAX32]], np.uint32)
    monkeypatch.setattr(PiecewiseCurve, "generate_decompanding_lut",
                        lambda *a, **k: pytest.fail("32-bit input allocated a dense PWC LUT"))
    with np.errstate(all="raise"):
        actual = PiecewiseCurve(source, BASE["platform"], sensor(32), config).execute()
    np.testing.assert_array_equal(actual, source)


@pytest.mark.parametrize("pins", ([], [0], [0, 0], [1, 0], [0, 1.5], [0, np.inf]))
def test_pwc_invalid_knots_rejected(pins: list[float]) -> None:
    with pytest.raises(ValueError):
        PiecewiseCurve.generate_decompanding_lut(pins, [0] * len(pins))


def test_oecf_direct_uint32_values_and_per_plane_coverage() -> None:
    source = np.array([[3, 0], [1, 0]], np.uint32)
    config = parameters("oecf", output_bit_depth=32, r_lut=[0, 1, 65536, MAX32],
                        gr_lut=[25], gb_lut=[11, 12], bl_lut=[99])
    with np.errstate(all="raise"):
        actual = OECF(source, BASE["platform"], sensor(2), config).apply_oecf()
    assert actual.dtype == np.uint32
    np.testing.assert_array_equal(actual, np.array([[MAX32, 25], [12, 99]], np.uint32))


@pytest.mark.parametrize("values", ([True], [1 + 0j], [-1], [0.5], [np.nan], [np.inf], [2**32]))
def test_oecf_invalid_lut_does_not_wrap(values: list[Any]) -> None:
    config = parameters("oecf", output_bit_depth=32, r_lut=values)
    with np.errstate(all="raise"), pytest.raises(ValueError):
        OECF(np.zeros((2, 2), np.uint32), BASE["platform"], sensor(2), config).apply_oecf()


@pytest.mark.parametrize("bits", (0, 33, 1.5, True, np.nan, np.inf))
def test_oecf_invalid_declared_depth(bits: Any) -> None:
    config = parameters("oecf", output_bit_depth=bits, r_lut=[0])
    with pytest.raises(ValueError, match="output_bit_depth"):
        OECF(np.zeros((2, 2), np.uint32), BASE["platform"], sensor(2), config).apply_oecf()


def test_green_repair_reports_original_odd_sites() -> None:
    source = np.ones((3, 5), np.uint32)
    source[0, 1] = 0
    actual, fraction = repair_zero_green_cfa(source, "rggb")
    assert fraction == pytest.approx(1 / 7)
    np.testing.assert_array_equal(actual, np.ones((3, 5), np.uint32))
    with pytest.raises(ValueError, match="bayer_pattern"):
        repair_zero_green_cfa(np.zeros((1, 1), np.uint32), cast(Any, "invalid"))
