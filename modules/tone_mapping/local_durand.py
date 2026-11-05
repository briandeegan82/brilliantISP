"""Local Durand-style operator: Gaussian base + range-weighted detail.

Retains the existing Gaussian/range approximation honestly rather than
claiming a bilateral implementation; the default filter is SciPy
``gaussian_filter`` (reflect boundary, truncate 4) matching the legacy
CPU path. The whole-frame output is adaptively normalized to [0,1];
constant output falls back to the bounded normalized input. CPU-only; an
injectable Gaussian callback ``(float32 plane, sigma) -> plane`` allows a
future verified accelerator without duplicating tone math.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from modules.tone_mapping.normalization import (
    FrameStats,
    check_code_range,
    prepare_curve_input,
    require_input_reference,
    require_code_ranges,
    validate_tone_source,
)
from modules.tone_mapping.registry import ToneMappingSpec, ToneMappingError

GaussianFilter = Callable[[np.ndarray, float], np.ndarray]


def _gaussian(plane: np.ndarray, sigma: float) -> np.ndarray:
    return np.asarray(gaussian_filter(plane, sigma), dtype=np.float32)


def _durand_plane(
    curve_input: np.ndarray,
    *,
    sigma_space: float,
    sigma_color: float,
    contrast_factor: float,
    downsample_factor: float,
    gaussian: GaussianFilter,
) -> np.ndarray:
    x = np.asarray(curve_input, dtype=np.float32)
    log_y = np.log10(x + np.float32(1e-6))
    factor = int(downsample_factor)
    if factor > 1:
        h, w = x.shape[:2]
        dw = max(1, int(math.ceil(w / factor)))
        dh = max(1, int(math.ceil(h / factor)))
        small = cv2.resize(log_y, (dw, dh), interpolation=cv2.INTER_AREA)
        g_small = np.asarray(gaussian(small.astype(np.float32), sigma_space / factor), dtype=np.float32)
        if g_small.shape != small.shape:
            raise ValueError(
                f"injected gaussian must return the input shape; got "
                f"{g_small.shape} for {small.shape}"
            )
        if not np.all(np.isfinite(g_small)):
            raise ValueError("injected gaussian returned nonfinite values")
        diff_small = small - g_small
        base_small = g_small + np.exp(
            -0.5 * (diff_small / np.float32(sigma_color)) ** 2
        ) * diff_small
        base = cv2.resize(base_small, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        g = np.asarray(gaussian(log_y.astype(np.float32), sigma_space), dtype=np.float32)
        if g.shape != log_y.shape:
            raise ValueError(
                f"injected gaussian must return the input shape; got "
                f"{g.shape} for {log_y.shape}"
            )
        if not np.all(np.isfinite(g)):
            raise ValueError("injected gaussian returned nonfinite values")
        diff = log_y - g
        base = g + np.exp(-0.5 * (diff / np.float32(sigma_color)) ** 2) * diff
    out = np.power(10.0, base / np.float32(contrast_factor) + (log_y - base))
    lo = float(np.min(out))
    hi = float(np.max(out))
    if hi > lo and math.isfinite(hi) and math.isfinite(lo):
        return ((out - lo) / (hi - lo)).astype(np.float64)
    if not (math.isfinite(hi) and math.isfinite(lo)):
        raise ToneMappingError("durand produced nonfinite output")
    return np.clip(np.asarray(curve_input, dtype=np.float64), 0.0, 1.0)


def map_luminance(
    luminance: np.ndarray,
    *,
    input_max: float,
    output_max: int,
    config: ToneMappingSpec,
    frame_stats: FrameStats | None = None,
    gaussian: GaussianFilter | None = None,
    input_reference: int | float | None = None,
) -> np.ndarray:
    """Map luminance through the local Durand-style operator.

    Input is 2-D nonempty finite; the shared normalized -> gain ->
    domain-cap recipe supplies the curve input. ``input_max`` is the
    physical unsigned-code bound; ``input_reference`` is the optional
    white-reference denominator (defaults to ``input_max``).
    ``gaussian`` defaults to the built-in SciPy CPU filter.
    """
    input_max, output_max = require_code_ranges(input_max, output_max)
    reference = require_input_reference(input_reference, input_max)
    if not config.enabled:
        if input_max == output_max and reference == output_max:
            return luminance
        return np.clip(np.asarray(luminance, np.float64) * (output_max / reference), 0, output_max)
    if config.operator != "durand" or config.backend != "cpu":
        raise ToneMappingError("local Durand requires a durand/cpu spec")
    src = np.asarray(luminance)
    if src.ndim != 2 or src.size == 0:
        raise ToneMappingError(
            f"durand input must be a nonempty 2-D frame, got shape {src.shape}"
        )
    try:
        validate_tone_source(src, for_integer=False)
    except ValueError as exc:
        raise ToneMappingError(str(exc)) from exc
    check_code_range(src, input_max, ToneMappingError)
    params = config.parameters_map()
    gain = 2.0**config.exposure_ev * config.domain_scale
    curve_input = prepare_curve_input(
        src,
        input_max=reference,
        normalization=config.normalization,
        gain=gain,
        lut_domain=config.lut_domain,
        stats=frame_stats,
    )
    out = _durand_plane(
        curve_input,
        sigma_space=params["sigma_space"],
        sigma_color=params["sigma_color"],
        contrast_factor=params["contrast_factor"],
        downsample_factor=params["downsample_factor"],
        gaussian=gaussian or _gaussian,
    )
    return out * float(output_max)
