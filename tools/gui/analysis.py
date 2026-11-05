from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Integral
from typing import Any

import numpy as np

from util.color import encode_transfer, luminance, unpack_ycbcr422

ROI = tuple[int, int, int, int]


def normalize_roi(roi: Sequence[int] | None, width: int, height: int) -> ROI:
    """Intersect an ordered, half-open pixel rectangle with the image."""
    if width <= 0 or height <= 0:
        raise ValueError("image geometry must be positive")
    if roi is None:
        return 0, 0, width, height
    if len(roi) != 4 or any(isinstance(v, bool) or not isinstance(v, Integral) for v in roi):
        raise ValueError("ROI must contain four integral pixel coordinates")
    x0, y0, x1, y1 = (int(v) for v in roi)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("ROI must be ordered and nonempty")
    result = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("ROI does not intersect the image")
    return result


def _image(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values)
    if data.ndim not in (2, 3) or not data.size or data.dtype.kind not in "uif":
        raise ValueError("analysis requires a nonempty real 2-D or HWC array")
    if data.ndim == 3 and data.shape[2] != 3:
        raise ValueError("analysis HWC arrays must have three components")
    if not np.all(np.isfinite(data)):
        raise ValueError("analysis values must be finite")
    return data


def _region(values: np.ndarray, roi: Sequence[int] | None) -> tuple[np.ndarray, ROI]:
    data = _image(values)
    bounds = normalize_roi(roi, data.shape[1], data.shape[0])
    x0, y0, x1, y1 = bounds
    return data[y0:y1, x0:x1], bounds


def analyze_values(
    values: np.ndarray,
    *,
    nominal_ranges: Sequence[Sequence[float]],
    roi: Sequence[int] | None = None,
    bins: int = 256,
) -> dict[str, Any]:
    """Native-unit statistics; nominal-bound counts do not prove clipping."""
    data, bounds = _region(values, roi)
    planes = (data,) if data.ndim == 2 else tuple(data[..., c] for c in range(3))
    if len(nominal_ranges) != len(planes):
        raise ValueError("one declared nominal range is required per component")
    if isinstance(bins, bool) or not isinstance(bins, int) or not 2 <= bins <= 4096:
        raise ValueError("histogram bins must be an integer in 2..4096")
    components = []
    for plane, limits in zip(planes, nominal_ranges):
        if len(limits) != 2:
            raise ValueError("nominal ranges must be pairs")
        lo, hi = (float(v) for v in limits)
        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
            raise ValueError("nominal ranges must be finite and strictly ordered")
        mean = float(np.mean(plane, dtype=np.float64))
        std = float(np.std(plane, dtype=np.float64, ddof=0))
        if not math.isfinite(mean) or not math.isfinite(std):
            raise ValueError("statistics exceed the supported float64 magnitude")
        histogram, edges = np.histogram(plane, bins=bins, range=(lo, hi))
        below = int(np.count_nonzero(plane < lo))
        above = int(np.count_nonzero(plane > hi))
        if int(histogram.sum()) + below + above != plane.size:
            raise AssertionError("histogram accounting mismatch")
        components.append({
            "minimum": float(plane.min()), "maximum": float(plane.max()),
            "mean": mean, "population_std": std, "nominal_range": [lo, hi],
            "below_nominal": below, "at_nominal_low": int(np.count_nonzero(plane == lo)),
            "above_nominal": above, "at_nominal_high": int(np.count_nonzero(plane == hi)),
            "histogram": histogram.tolist(), "histogram_edges": edges.tolist(),
        })
    return {
        "roi": list(bounds), "pixel_count": int(data.shape[0] * data.shape[1]),
        "dtype": str(data.dtype), "components": components,
        "histogram_note": "Only nominal-range samples are binned; separate under/over counts preserve headroom.",
        "limit_note": "Nominal endpoints/out-of-range values are not a measurement of sensor saturation or lost detail.",
    }


def analyze_rgb8(rgb: np.ndarray, roi: Sequence[int] | None = None) -> dict[str, Any]:
    """Display sRGB8 statistics, never scene-linear luminance or sensor DR."""
    data, bounds = _region(rgb, roi)
    if data.dtype != np.uint8 or data.ndim != 3:
        raise ValueError("display analysis requires HWC3 uint8 RGB")
    result = analyze_values(data, nominal_ranges=((0, 255),) * 3)
    result["roi"] = list(bounds)
    result["domain"] = "display sRGB8 codes"
    result["component_names"] = ["R", "G", "B"]
    for c, component in enumerate(result["components"]):
        component["histogram"] = np.bincount(data[..., c].reshape(-1), minlength=256).tolist()
        component["histogram_codes"] = list(range(256))
    code_luma = luminance(data, matrix="bt709", dtype=np.float64)
    rounded_luma = np.clip(np.rint(code_luma), 0, 255).astype(np.uint8)
    percentiles = np.percentile(code_luma, [1, 99], method="linear")
    result["encoded_luma"] = {
        "label": "Rec.709-weighted encoded RGB code luma; not scene-linear luminance",
        "mean": float(np.mean(code_luma, dtype=np.float64)),
        "histogram": np.bincount(rounded_luma.reshape(-1), minlength=256).tolist(),
        "histogram_note": "Luma codes are rounded to nearest-even for histogram binning.",
        "percentile_1": float(percentiles[0]), "percentile_99": float(percentiles[1]),
        "percentile_code_span": float(percentiles[1] - percentiles[0]),
        "span_note": "A 1st–99th percentile code-value span, not sensor dynamic range or effective bit depth.",
    }
    result["any_display_zero_pixels"] = int(np.count_nonzero(np.any(data == 0, axis=2)))
    result["any_display_max_pixels"] = int(np.count_nonzero(np.any(data == 255, axis=2)))
    return result


def native_values(native: np.ndarray, metadata: Mapping[str, Any]) -> np.ndarray:
    """Unpack native 4:2:2 into pixel coordinates, repeating shared chroma."""
    data = _image(native)
    if metadata.get("color_model") == "ycbcr" and metadata.get("layout") == "422":
        return unpack_ycbcr422(data, packing=str(metadata.get("packing") or "yuyv"))
    return data


def native_pixel(
    native: np.ndarray, metadata: Mapping[str, Any], x: int, y: int,
) -> dict[str, Any]:
    """Read a native pixel without sampling the resized display image."""
    if any(isinstance(v, bool) or not isinstance(v, Integral) for v in (x, y)):
        raise ValueError("pixel coordinates must be integral")
    width, height = int(metadata["width"]), int(metadata["height"])
    if not 0 <= x < width or not 0 <= y < height:
        raise ValueError("pixel coordinates are outside the native frame")
    data = np.asarray(native)
    if metadata.get("color_model") == "ycbcr" and metadata.get("layout") == "422":
        if data.shape != (height, width * 2):
            raise ValueError("packed native geometry disagrees with metadata")
        group = (x // 2) * 4
        pair = unpack_ycbcr422(
            data[y:y + 1, group:group + 4],
            packing=str(metadata.get("packing") or "yuyv"),
        )
        sample = pair[0, x % 2]
    else:
        if data.shape[:2] != (height, width) or data.ndim not in (2, 3):
            raise ValueError("native geometry disagrees with metadata")
        sample = np.asarray(data[y, x]).reshape(-1)
    if sample.size not in (1, 3) or not np.all(np.isfinite(sample)):
        raise ValueError("native sample must have one or three finite components")
    names = ["Bayer/code"] if sample.size == 1 else (
        ["Y", "Cb", "Cr"] if metadata.get("color_model") == "ycbcr" else ["R", "G", "B"])
    return {
        "x": int(x), "y": int(y), "values": sample.tolist(), "component_names": names,
        "dtype": str(data.dtype), "units": metadata.get("value_units"),
        "domain": metadata.get("signal_domain"), "transfer": metadata.get("transfer"),
        "note": "Shared 4:2:2 chroma is repeated per pixel." if metadata.get("layout") == "422" else "",
    }


def sample_tone_curve(config: Mapping[str, Any], *, samples: int = 512) -> dict[str, Any]:
    """Shared native engine on a unit ramp; explicitly not a frame response."""
    from modules.tone_mapping.native import map_native
    from modules.tone_mapping.registry import resolve_tone_mapping
    from util.output_contract import OutputSpec
    from util.pipeline_migration import migrate_pipeline_config

    if isinstance(samples, bool) or not isinstance(samples, int) or not 2 <= samples <= 4096:
        raise ValueError("curve samples must be an integer in 2..4096")
    effective, _report = migrate_pipeline_config(config)
    resolved = resolve_tone_mapping(effective)
    spec = OutputSpec.from_mapping(effective.get("output"))
    x = np.linspace(0, 1, samples, dtype=np.float64).reshape(1, samples)
    mapped = map_native(x, config=resolved.spec)
    encoded = encode_transfer(np.clip(mapped, 0, 1), spec.transfer, gamma=spec.gamma, dtype=np.float64)
    return {
        "x": x.reshape(-1).tolist(),
        "display_linear": np.asarray(mapped).reshape(-1).tolist(),
        "after_transfer": np.asarray(encoded).reshape(-1).tolist(),
        "operator": resolved.spec.operator, "backend": resolved.spec.backend,
        "enabled": resolved.spec.enabled, "transfer": spec.transfer,
        "label": "Unit-ramp illustration through the shared native mapper and output transfer",
        "caveat": "Ramp-derived normalization, not the captured frame's response. Durand is local/frame-dependent; no single global curve describes its full-frame behavior.",
    }
