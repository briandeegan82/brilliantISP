from __future__ import annotations

import math
import numbers
import re

import numpy as np

from util.raw_decode import shift_bayer_pattern

from ..ops.borders import pad
from ..types import RawTensor, require_torch

_LAYOUTS = {
    "rggb": (0, 1, 2, 3), "bggr": (3, 2, 1, 0),
    "grbg": (1, 0, 3, 2), "gbrg": (2, 3, 0, 1),
}
_PAIR_OFFSETS = (
    ((-2, 0), (2, 0)), ((0, -2), (0, 2)),
    ((-2, -2), (2, 2)), ((-2, 2), (2, -2)),
)


def _check_image(image: RawTensor, sensor=None):
    torch = require_torch()
    if not isinstance(image, RawTensor) or image.storage_dtype not in ("uint8", "uint16", "uint32"):
        raise ValueError("raw image requires a RawTensor with unsigned storage dtype")
    data = image.codes
    if not isinstance(data, torch.Tensor) or data.dtype != torch.int64 or data.ndim != 3 or not data.numel():
        raise ValueError("raw codes must be a nonempty BHW int64 tensor")
    if sensor is not None and data.shape[-2:] != (sensor["height"], sensor["width"]):
        raise ValueError("raw image shape does not match declared sensor geometry")
    return data


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _code_max(sensor):
    bits = _integer(sensor["bit_depth"], "sensor bit depth", 1)
    if bits > 32:
        raise ValueError("sensor bit depth must be in 1..32")
    return 2**bits - 1


def _layout(sensor):
    name = str(sensor["bayer_pattern"]).strip().lower()
    if name not in _LAYOUTS:
        raise ValueError(f"unsupported Bayer pattern {name!r}")
    return _LAYOUTS[name]


def _crop_name(platform, height, width):
    if platform is not None:
        platform["in_file"] = re.sub(r"\d+x\d+", f"{width}x{height}", platform["in_file"])


def crop(image: RawTensor, sensor, params, *, platform=None):
    data = _check_image(image)
    old_h, old_w = data.shape[-2:]
    _crop_name(platform, old_h, old_w)
    if not params["is_enable"]:
        return image
    height = _integer(params["new_height"], "crop new_height", 1)
    width = _integer(params["new_width"], "crop new_width", 1)
    x = _integer(params.get("crop_x_start", 0), "crop crop_x_start")
    y = _integer(params.get("crop_y_start", 0), "crop crop_y_start")
    if y + height > old_h or x + width > old_w:
        raise ValueError(f"crop region ({x}, {y}) {width}x{height} exceeds input {old_w}x{old_h}")
    pattern = shift_bayer_pattern(str(sensor.get("bayer_pattern", "rggb")), x, y)
    if sensor["height"] != height or sensor["width"] != width:
        sensor["orig_size"] = str((old_w, old_h))
    sensor.update(height=height, width=width, bayer_pattern=pattern)
    _crop_name(platform, height, width)
    if height == old_h and width == old_w and x == 0 and y == 0:
        return image
    return RawTensor(data[:, y:y + height, x:x + width], image.storage_dtype)


def dead_pixel_correction(image: RawTensor, sensor, params):
    torch = require_torch()
    data = _check_image(image, sensor)
    if not params["is_enable"]:
        return image
    threshold = float(params["dp_threshold"])
    if not math.isfinite(threshold):
        raise ValueError("dead pixel threshold must be finite")
    maximum = _code_max(sensor)
    f = data.to(torch.float64)
    height, width = data.shape[-2:]
    extended = pad(f, 2, "mirror")
    lo, hi = torch.full_like(f, math.inf), torch.full_like(f, -math.inf)
    detected = torch.ones_like(data, dtype=torch.bool)
    best_gradient = torch.full_like(f, math.inf)
    correction = torch.zeros_like(f)
    for (ay, ax), (by, bx) in _PAIR_OFFSETS:
        a = extended[:, 2 + ay:2 + ay + height, 2 + ax:2 + ax + width]
        b = extended[:, 2 + by:2 + by + height, 2 + bx:2 + bx + width]
        for neighbor in (a, b):
            lo, hi = torch.minimum(lo, neighbor), torch.maximum(hi, neighbor)
            detected &= (f - neighbor).abs() > threshold
        gradient = (2.0 * f - a - b).abs()
        take = gradient < best_gradient
        correction = torch.where(take, (a + b) / 2.0, correction)
        best_gradient = torch.minimum(best_gradient, gradient)
    detected &= (f < lo) | (f > hi)
    result = torch.where(detected, correction, f).clamp(0, maximum).to(torch.int64)
    return RawTensor(result, "uint16" if maximum <= 65535 else "uint32")


def black_level_correction(image: RawTensor, sensor, params):
    torch = require_torch()
    data = _check_image(image, sensor)
    if not params["is_enable"]:
        return image
    channels = ("r", "gr", "gb", "b")
    offsets = np.asarray([params[f"{c}_offset"] for c in channels])
    if offsets.dtype.kind not in "iuf" or not np.all(np.isfinite(offsets)) or np.any(np.abs(offsets) > np.finfo(np.float64).max):
        raise ValueError("black-level offsets must be finite")
    offsets = offsets.astype(np.float64)
    denominators = np.ones(4, np.float64)
    if params["is_linear"]:
        saturations = np.asarray([params[f"{c}_sat"] for c in channels])
        if saturations.dtype.kind not in "iuf" or not np.all(np.isfinite(saturations)) or np.any(np.abs(saturations) > np.finfo(np.float64).max):
            raise ValueError("black-level saturation values must be finite")
        with np.errstate(over="ignore", invalid="ignore"):
            denominators = saturations.astype(np.float64) - offsets
        if not np.all(np.isfinite(denominators)) or np.any(denominators <= 0):
            raise ValueError("black-level saturation minus offset must be finite and positive")
    result = data.to(torch.float64)
    maximum = _code_max(sensor)
    divisors = torch.as_tensor(denominators, dtype=torch.float64, device=data.device)
    for site, channel in enumerate(_layout(sensor)):
        row, col = divmod(site, 2)
        view = result[:, row::2, col::2]
        view -= float(offsets[channel])
        if params["is_linear"]:
            span = float(denominators[channel])
            view[:] = view.clamp(0, span) / divisors[channel] * maximum
    return RawTensor(result.clamp(0, maximum).to(torch.int64), "uint32")


def _green_sites(pattern):
    pattern = str(pattern).strip().lower()
    if pattern not in _LAYOUTS:
        raise ValueError(f"unsupported Bayer pattern {pattern!r}")
    return ((0, 1), (1, 0)) if pattern in ("rggb", "bggr") else ((0, 0), (1, 1))


def green_zero_fraction(image: RawTensor, pattern, *, zero_thresh=0.0):
    torch = require_torch()
    data = _check_image(image)
    count = torch.zeros(data.shape[0], dtype=torch.int64, device=data.device)
    total = 0
    for row, col in _green_sites(pattern):
        plane = data[:, row::2, col::2]
        count += (plane.to(torch.float64) <= zero_thresh).sum(dim=(-1, -2))
        total += plane.shape[-2] * plane.shape[-1]
    if not total:
        return count.to(torch.float64)
    divisor = torch.tensor(total, dtype=torch.float64, device=data.device)
    return count.to(torch.float64) / divisor


def _cell_shift(data, dy=0, dx=0):
    torch = require_torch()
    height, width = data.shape[-2:]
    rows = (torch.arange(height, device=data.device) + dy).clamp(0, height - 1)
    cols = (torch.arange(width, device=data.device) + dx).clamp(0, width - 1)
    return data[:, rows, :][:, :, cols]


def repair_zero_green_cfa(image: RawTensor, pattern, *, zero_thresh=0.0):
    torch = require_torch()
    data = _check_image(image)
    sites = _green_sites(pattern)
    height, width = data.shape[-2:]
    if height < 2 or width < 2:
        return RawTensor(data.clone(), image.storage_dtype), torch.zeros(
            data.shape[0], dtype=torch.float64, device=data.device)
    py, px = height % 2, width % 2
    work = pad(data, (py, px), "mirror")[:, py:py + height + py, px:px + width + px]
    result = work.clone()
    if sites[0] == (0, 1):
        a, b = work[:, 0::2, 0::2], work[:, 1::2, 1::2]
        estimates = (
            torch.div(a + _cell_shift(a, dx=1) + _cell_shift(b, dy=-1) + b, 4, rounding_mode="floor"),
            torch.div(a + _cell_shift(a, dy=1) + _cell_shift(b, dx=-1) + b, 4, rounding_mode="floor"),
        )
    else:
        a, b = work[:, 0::2, 1::2], work[:, 1::2, 0::2]
        estimates = (
            torch.div(_cell_shift(a, dx=-1) + a + _cell_shift(b, dy=-1) + b, 4, rounding_mode="floor"),
            torch.div(a + _cell_shift(a, dy=1) + b + _cell_shift(b, dx=1), 4, rounding_mode="floor"),
        )
    for (row, col), estimate in zip(sites, estimates):
        plane = result[:, row::2, col::2]
        plane[:] = torch.where(plane.to(torch.float64) <= zero_thresh, estimate, plane)
    return (
        RawTensor(result[:, :height, :width].clamp_min(0), image.storage_dtype),
        green_zero_fraction(image, pattern, zero_thresh=zero_thresh),
    )


def _pwc_knots(params):
    pins = np.asarray(params["companded_pin"])
    values = np.asarray(params["companded_pout"])
    if pins.ndim != 1 or values.ndim != 1 or pins.size < 2 or pins.size != values.size or pins.dtype.kind not in "uif" or values.dtype.kind not in "uif":
        raise ValueError("PWC requires matching real 1-D knots with at least two entries")
    if not np.all(np.isfinite(pins)) or not np.all(np.isfinite(values)) or np.any(pins < 0) or np.any(pins > 2**32) or np.any(pins != np.floor(pins)):
        raise ValueError("PWC input knots must be finite integral codes in 0..2**32")
    if np.any(values < 0) or np.any(values > 2**32):
        raise ValueError("PWC output knots must be in 0..2**32")
    pins, values = pins.astype(np.float64), values.astype(np.float64)
    if np.any(np.diff(pins) <= 0):
        raise ValueError("PWC input knots must be strictly increasing")
    return pins, values


def _pwc_dense_lut(pins, values, maximum):
    points = pins.astype(np.int64)
    lut = np.zeros(max(int(maximum), int(points[-1])) + 1, dtype=np.float64)
    lut[:points[0]] = values[0]
    for i in range(len(points) - 1):
        start, end = points[i], points[i + 1]
        x = np.arange(start, end + 1)
        t = (x - start) / (end - start)
        lut[x] = values[i] + t * (values[i + 1] - values[i])
    lut[points[-1]:] = values[-1]
    return lut


def companding(image: RawTensor, sensor, params):
    torch = require_torch()
    data = _check_image(image, sensor)
    if not params["is_enable"]:
        return RawTensor(data, "uint32")
    pre = int(params.get("pre_linearization_black_level", params.get("pedestal", 0)))
    post = int(params.get("post_linearization_black_level", 0))
    maximum = _code_max(sensor) - pre
    if maximum < 0:
        raise ValueError("PWC pre-linearization black level exceeds the sensor range")
    pins, values = _pwc_knots(params)
    x = (data - pre).clamp(0, maximum)
    if max(maximum, pins[-1]) <= 65536:
        lut = torch.as_tensor(_pwc_dense_lut(pins, values, maximum), device=data.device)
        mapped = lut[x]
    else:
        knots = torch.as_tensor(pins, device=data.device)
        outputs = torch.as_tensor(values, device=data.device)
        slopes = torch.as_tensor(np.diff(values) / np.diff(pins), device=data.device)
        positions = x.to(torch.float64).contiguous()
        index = (torch.searchsorted(knots, positions, right=True) - 1).clamp(0, len(pins) - 2)
        mapped = slopes[index] * (positions - knots[index]) + outputs[index]
        mapped = torch.where(positions <= knots[0], outputs[0], mapped)
        mapped = torch.where(positions >= knots[-1], outputs[-1], mapped)
    if post:
        mapped = (mapped - post).clamp_min(0)
    if not bool(torch.isfinite(mapped).all()):
        raise ValueError("decompanding output must be finite")
    out_max = 2**int(sensor.get("hdr_bit_depth", sensor["bit_depth"])) - 1
    return RawTensor(mapped.clamp(0, out_max).to(torch.int64), "uint32")


def oecf(image: RawTensor, sensor, params):
    torch = require_torch()
    data = _check_image(image, sensor)
    if not params["is_enable"]:
        return image
    r_lut = params.get("r_lut")
    if r_lut is None:
        maximum = 2**int(sensor.get("hdr_bit_depth", sensor["bit_depth"])) - 1
        return RawTensor(data.clamp(0, maximum), "uint32" if maximum > 65535 else "uint16")
    bits = params.get("output_bit_depth", sensor["bit_depth"])
    if isinstance(bits, (bool, np.bool_)) or not isinstance(bits, numbers.Real) or not math.isfinite(bits) or not 1 <= bits <= 32 or int(bits) != bits:
        raise ValueError("oecf.output_bit_depth must be an integer in 1..32")
    maximum = 2**int(bits) - 1
    if bool((data < 0).any()):
        raise ValueError("oecf input codes must be nonnegative integers")
    result = torch.empty_like(data)
    names = ("r_lut", "gr_lut", "gb_lut", "bl_lut")
    for site, channel in enumerate(_layout(sensor)):
        row, col = divmod(site, 2)
        plane = data[:, row::2, col::2]
        if not plane.numel():
            continue
        name = names[channel]
        values = np.asarray(params.get(name, r_lut))
        if values.ndim != 1 or not values.size or values.dtype.kind not in "uif":
            raise ValueError(f"oecf {name} must be a nonempty 1-D LUT")
        if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > maximum) or np.any(values != np.floor(values)):
            raise ValueError(f"oecf {name} entries must be finite nonnegative integral <= {maximum}")
        if int(plane.max()) >= values.size:
            raise ValueError(f"oecf {name} has no entry for declared input code")
        lut = torch.as_tensor(values.astype(np.int64), device=data.device)
        result[:, row::2, col::2] = lut[plane]
    return RawTensor(result, "uint16" if bits <= 16 else "uint32")
