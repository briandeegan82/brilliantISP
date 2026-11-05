from __future__ import annotations

import numpy as np
import pytest
import yaml
from pathlib import Path

from typing import Any, cast

from modules.crop.crop import Crop
from util.isp_types import BayerPattern, CropConfig, PlatformConfig, SensorInfo
from util.raw_decode import shift_bayer_pattern, flip_bayer_pattern

REPO = Path(__file__).resolve().parents[1]


def _cfg(x0: int, y0: int, w: int, h: int) -> CropConfig:
    return {
        "is_enable": True,
        "is_debug": False,
        "is_save": False,
        "crop_x_start": x0,
        "crop_y_start": y0,
        "new_width": w,
        "new_height": h,
    }


def _sensor(pattern: BayerPattern, w: int = 8, h: int = 6) -> SensorInfo:
    return {
        "width": w,
        "height": h,
        "bayer_pattern": pattern,
        "bit_depth": 12,
        "endian_type": "ieee-le",
        "hdr_bit_depth": 12,
        "output_bit_depth": 8,
        "pipeline_rgb_bit_depth": 8,
    }


def _platform() -> PlatformConfig:
    return {
        "in_file": "t", "out_file": "t", "filename": "t",
        "debug_enabled": False, "disable_progress_bar": True,
        "generate_tv": False, "leave_pbar_string": False,
        "output_dir": "", "render_3a": False, "rgb_output": True,
        "save_format": "png", "short_output_names": True,
        "skip_disabled_modules": False,
    }


_PATTERNS = ("rggb", "grbg", "gbrg", "bggr")


def _phase_colors(pattern: str, w: int, h: int) -> np.ndarray:
    grid = {"rggb": ("rg", "gb"), "grbg": ("gr", "bg"),
            "gbrg": ("gb", "rg"), "bggr": ("bg", "gr")}[pattern]
    colors = {"r": 100, "g": 200, "b": 300}
    img = np.empty((h, w), np.uint16)
    for y in range(h):
        for x in range(w):
            img[y, x] = colors[grid[y % 2][x % 2]]
    return img


@pytest.mark.parametrize("pattern", _PATTERNS)
@pytest.mark.parametrize("x0,y0", [(0, 0), (1, 0), (0, 1), (1, 1), (3, 2)])
def test_exact_origins_and_phase(pattern: BayerPattern, x0: int, y0: int) -> None:
    img = _phase_colors(pattern, 8, 6)
    sensor = _sensor(pattern)
    crop = Crop(img.copy(), _platform(), sensor, _cfg(x0, y0, 4, 4))
    out = crop.execute()
    np.testing.assert_array_equal(out, img[y0 : y0 + 4, x0 : x0 + 4])
    assert (sensor["width"], sensor["height"]) == (4, 4)
    assert sensor["bayer_pattern"] == shift_bayer_pattern(pattern, x0, y0)


def test_same_size_origin_zero() -> None:
    img = _phase_colors("rggb", 8, 6)
    sensor = _sensor("rggb")
    crop = Crop(img.copy(), _platform(), sensor, _cfg(0, 0, 8, 6))
    out = crop.execute()
    np.testing.assert_array_equal(out, img)


def test_same_size_invalid_offset_raises() -> None:
    img = _phase_colors("rggb", 8, 6)
    sensor = _sensor("rggb")
    crop = Crop(img.copy(), _platform(), sensor, _cfg(1, 0, 8, 6))
    with pytest.raises(ValueError, match="exceeds"):
        crop.execute()


def test_out_of_bounds_raises_and_metadata_unchanged() -> None:
    img = _phase_colors("rggb", 8, 6)
    sensor = _sensor("rggb")
    before = dict(sensor)
    crop = Crop(img.copy(), _platform(), sensor, _cfg(6, 4, 4, 4))
    with pytest.raises(ValueError):
        crop.execute()
    assert sensor == before


@pytest.mark.parametrize("bad", [
    _cfg(-1, 0, 4, 4),
    _cfg(0, -1, 4, 4),
    _cfg(0, 0, 0, 4),
    _cfg(0, 0, 4, -1),
])
def test_invalid_fields_raise(bad: dict) -> None:
    img = _phase_colors("rggb", 8, 6)
    sensor = _sensor("rggb")
    crop = Crop(img.copy(), _platform(), sensor, cast(Any, bad))
    with pytest.raises(ValueError):
        crop.execute()


def test_fractional_and_bool_origins_raise() -> None:
    img = _phase_colors("rggb", 8, 6)
    for key, val in (("crop_x_start", 1.5), ("crop_y_start", True),
                     ("new_width", "4")):
        sensor = _sensor("rggb")
        cfg = _cfg(0, 0, 4, 4)
        bad_cfg = cast(Any, {**dict(cfg), key: val})
        with pytest.raises(ValueError):
            Crop(img.copy(), _platform(), sensor, bad_cfg).execute()


def test_constructor_does_not_mutate_sensor() -> None:
    img = _phase_colors("rggb", 8, 6)
    sensor = _sensor("rggb")
    before = dict(sensor)
    Crop(img.copy(), _platform(), sensor, _cfg(2, 2, 4, 4))
    assert sensor == before


def test_crop_after_flip_phase() -> None:
    w, h = 8, 6
    sensor = _sensor("gbrg", w, h)
    flipped_pattern = flip_bayer_pattern("gbrg", w, h, True, False)
    img = _phase_colors("gbrg", w, h)[:, ::-1].copy()
    sensor["bayer_pattern"] = cast(BayerPattern, flipped_pattern)
    crop = Crop(img.copy(), _platform(), sensor, _cfg(1, 1, 4, 4))
    out = crop.execute()
    np.testing.assert_array_equal(out, img[1:5, 1:5])
    assert sensor["bayer_pattern"] == shift_bayer_pattern(flipped_pattern, 1, 1)


def test_orig_size_recorded() -> None:
    img = _phase_colors("rggb", 8, 6)
    sensor = _sensor("rggb")
    Crop(img.copy(), _platform(), sensor, _cfg(0, 0, 4, 4)).execute()
    assert "orig_size" in sensor
