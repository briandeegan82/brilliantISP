"""Tone-mapping backends: float64 scalar, LUT, and fixed-point Q32.

Shared pipeline: validate input -> normalize once ->
``2**exposure_ev * domain_scale`` gain -> saturate curve input at
``lut_domain`` -> scalar curve -> optional theoretical output
normalization -> clip [0,1] -> output code scale. Integer paths evaluate
the exact recipes of the M3 contract without float frame arithmetic.
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from typing import Any, Callable

import numpy as np

from modules.tone_mapping import operators
from modules.tone_mapping.normalization import (
    FrameStats,
    check_code_range,
    frame_stats,
    normalize_integer,
    normalize_luminance,
    prepare_curve_input,
    require_input_reference,
    require_integral_reference,
    _integral_bound,
    require_code_ranges,
    validate_tone_source,
)
from modules.tone_mapping.registry import ToneMappingSpec, ToneMappingError

UINT32_MAX = np.uint64(2**32 - 1)
M = 2**32 - 1
LUT_BUDGET_BYTES = 8 * 1024 * 1024

_CURVES: dict[str, Callable[..., np.ndarray]] = {
    "aces_fitted": operators.aces_fitted,
    "hable": operators.hable,
    "reinhard": operators.reinhard,
}


def _curve(spec: ToneMappingSpec, x: np.ndarray) -> np.ndarray:
    if spec.operator not in _CURVES:
        raise ToneMappingError(f"{spec.operator} has no scalar backend curve")
    return _CURVES[spec.operator](x, **spec.parameters_map())


def _theoretical_reference(spec: ToneMappingSpec, gain: float) -> float:
    if not spec.normalize_output:
        return 1.0
    ref = float(_curve(spec, np.asarray([min(gain, spec.lut_domain)]))[0])
    if not np.isfinite(ref) or ref <= 0:
        raise ToneMappingError(
            "normalize_output has no representable positive curve reference "
            "at min(gain, lut_domain)"
        )
    return ref


def _require_range(input_max: Any, output_max: Any) -> tuple[float, int]:
    return require_code_ranges(input_max, output_max)


def _table_domain(spec: ToneMappingSpec, gain: float) -> float:
    """Normalized output is saturated above its theoretical reference."""
    return min(gain, spec.lut_domain) if spec.normalize_output else spec.lut_domain


def _require_integral_input_max(input_max: float) -> int:
    if not input_max.is_integer():
        raise ToneMappingError(
            "integer backends require an integral input_max (no silent truncation)"
        )
    return int(input_max)


def _float_backend(
    luminance: np.ndarray,
    spec: ToneMappingSpec,
    input_max: float,
    output_max: int,
    stats: FrameStats | None,
    *,
    reference: float | None = None,
) -> np.ndarray:
    gain = 2.0**spec.exposure_ev * spec.domain_scale
    x = prepare_curve_input(
        luminance,
        input_max=input_max if reference is None else reference,
        normalization=spec.normalization,
        gain=gain,
        lut_domain=spec.lut_domain,
        stats=stats,
    )
    mapped = _curve(spec, x) / _theoretical_reference(spec, gain)
    return np.clip(mapped, 0.0, 1.0) * float(output_max)


class _LutCache:
    """Bounded shared LUT cache; tables stored as immutable bytes."""

    def __init__(self, budget_bytes: int = LUT_BUDGET_BYTES) -> None:
        self._budget = budget_bytes
        self._tables: OrderedDict[tuple, bytes] = OrderedDict()
        self._retained = 0
        self._lock = threading.Lock()

    def key(self, spec: ToneMappingSpec, output_max: int) -> tuple:
        return (
            spec.operator,
            spec.parameters,
            spec.exposure_ev,
            spec.domain_scale,
            spec.normalization,
            spec.lut_domain,
            spec.lut_size,
            spec.normalize_output,
            output_max,
        )

    def get(self, spec: ToneMappingSpec, output_max: int) -> np.ndarray:
        key = self.key(spec, output_max)
        with self._lock:
            blob = self._tables.get(key)
            if blob is None:
                gain = 2.0**spec.exposure_ev * spec.domain_scale
                ref = _theoretical_reference(spec, gain)
                xs = np.linspace(0.0, _table_domain(spec, gain), spec.lut_size, dtype=np.float64)
                table = np.round(
                    np.clip(_curve(spec, xs) / ref, 0.0, 1.0) * float(output_max)
                ).astype("<u2")
                blob = table.tobytes()
                self._tables[key] = blob
                self._retained += len(blob)
                while self._retained > self._budget and len(self._tables) > 1:
                    _, evicted = self._tables.popitem(last=False)
                    self._retained -= len(evicted)
            else:
                self._tables.move_to_end(key)
        return np.frombuffer(blob, dtype="<u2")

    def clear(self) -> None:
        with self._lock:
            self._tables.clear()
            self._retained = 0

    def info(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._tables),
                "retained_bytes": self._retained,
                "budget_bytes": self._budget,
            }


_LUT_CACHE = _LutCache()


def _lut_cache_clear() -> None:
    _LUT_CACHE.clear()


def _lut_cache_info() -> dict[str, int]:
    return _LUT_CACHE.info()


def _q32_coordinate(
    n: np.ndarray,
    d: int,
    r: float,
) -> np.ndarray:
    """Saturating Q32 coordinate for ratio r per the integer recipe."""
    n = np.asarray(n, dtype=np.uint64)
    if d <= 0:
        raise ToneMappingError(f"normalization denominator must be positive, got {d}")
    if not math.isfinite(r) or r < 0:
        raise ToneMappingError(f"coordinate ratio must be finite nonnegative, got {r!r}")
    if r == 0:
        return np.zeros(n.shape, dtype=np.uint64)
    if r >= 2**64 / M:
        out = np.zeros(n.shape, dtype=np.uint64)
        out[n > 0] = np.uint64(M)
        return out
    c = round(r * M)
    if c <= 0:
        return np.zeros(n.shape, dtype=np.uint64)
    if c > 2**64 - 1:
        out = np.zeros(n.shape, dtype=np.uint64)
        out[n > 0] = np.uint64(M)
        return out
    threshold = (d * M) // c
    clipped = np.minimum(n, np.uint64(threshold))
    q = (clipped * np.uint64(c)) // np.uint64(d)
    q[n > np.uint64(threshold)] = np.uint64(M)
    return q


def _integer_lut(
    table: np.ndarray,
    n: np.ndarray,
    d: int,
    gain: float,
    lut_domain: float,
) -> np.ndarray:
    size = table.shape[0]
    q = _q32_coordinate(n, d, gain / lut_domain)
    position = (q * np.uint64((size - 1) * 65536)) // UINT32_MAX
    i = (position >> 16).astype(np.uint64)
    f = (position & np.uint64(0xFFFF)).astype(np.uint64)
    i = np.minimum(i, np.uint64(size - 1))
    j = np.minimum(i + np.uint64(1), np.uint64(size - 1))
    t = table.astype(np.uint64)
    out = (t[i] * (np.uint64(65536) - f) + t[j] * f + np.uint64(32768)) // np.uint64(65536)
    return out.astype(np.uint16)


def _fixed_reinhard(
    luminance: np.ndarray,
    spec: ToneMappingSpec,
    input_max: float,
    output_max: int,
    stats: FrameStats | None,
    *,
    reference: float | None = None,
    input_reference: int | float | None = None,
) -> np.ndarray:
    if luminance.dtype.kind != "u" or luminance.dtype.itemsize not in (1, 2, 4):
        raise ToneMappingError(
            f"fixed backend requires unsigned integer input (uint8/16/32), "
            f"got dtype {luminance.dtype}"
        )
    int_max = _require_integral_input_max(input_max)
    ref = float(input_max) if reference is None else float(reference)
    ref_int = require_integral_reference(input_reference, ref, input_max)
    params = spec.parameters_map()
    gain = 2.0**spec.exposure_ev * spec.domain_scale
    use = stats
    if spec.normalization == "per_frame" and use is None:
        use = frame_stats(luminance)
    nonconstant = False
    if spec.normalization == "per_frame":
        assert use is not None
        try:
            lo = _integral_bound("stats.minimum", use.minimum)
            hi = _integral_bound("stats.maximum", use.maximum)
        except ValueError as exc:
            raise ToneMappingError(str(exc)) from exc
        nonconstant = hi - lo > 0
    n, d = normalize_integer(
        luminance,
        normalization=spec.normalization,
        input_max=ref_int,
        stats=use,
    )
    H = 1.0 if nonconstant else int_max / ref
    if spec.normalize_output:
        u_ref = min(1.0, spec.lut_domain / gain)
    else:
        u_ref = min(H, spec.lut_domain / gain)
    effective_gain = gain * u_ref
    x = _q32_coordinate(n, d, 1.0 / u_ref)
    w = (params["knee"] / params["strength"]) / effective_gain
    if not math.isfinite(w) or w <= 0:
        raise ToneMappingError(
            "knee/(strength*effective_gain) must be representable positive"
        )
    if w <= 1.0:
        W = round(w * M)
        denom = x + np.uint64(W)
        denom[denom == 0] = np.uint64(1)
        if spec.normalize_output:
            half = (M + W + 1) // 2
            y = (x * np.uint64(half)) // denom
            yden = (M + 1) // 2
        else:
            y = (x * UINT32_MAX) // denom
            yden = M
    else:
        H = round(M / w)
        z = (x * np.uint64(H)) // UINT32_MAX
        if spec.normalize_output:
            half = (M + H + 1) // 2
            y = (x * np.uint64(half)) // (UINT32_MAX + z)
            yden = (M + 1) // 2
        else:
            y = (z * UINT32_MAX) // (UINT32_MAX + z)
            yden = M
    out = (y * np.uint64(output_max) + np.uint64(yden // 2)) // np.uint64(yden)
    return np.clip(out, 0, np.uint64(output_max)).astype(np.uint16)


def _float_lut(
    luminance: np.ndarray,
    spec: ToneMappingSpec,
    input_max: float,
    output_max: int,
    stats: FrameStats | None,
    *,
    reference: float | None = None,
) -> np.ndarray:
    table = _LUT_CACHE.get(spec, output_max)
    gain = 2.0**spec.exposure_ev * spec.domain_scale
    x = prepare_curve_input(
        luminance,
        input_max=input_max if reference is None else reference,
        normalization=spec.normalization,
        gain=gain,
        lut_domain=spec.lut_domain,
        stats=stats,
    )
    domain = _table_domain(spec, gain)
    position = np.minimum(x, domain) / domain * (spec.lut_size - 1)
    i = position.astype(np.uint64)
    f = position - i
    i = np.minimum(i, spec.lut_size - 1)
    j = np.minimum(i + np.uint64(1), np.uint64(spec.lut_size - 1))
    t = table.astype(np.float64)
    return t[i] * (1.0 - f) + t[j] * f


def _validate(luminance: np.ndarray) -> np.ndarray:
    try:
        return validate_tone_source(luminance, for_integer=False)
    except ValueError as exc:
        raise ToneMappingError(str(exc)) from exc


def map_luminance(
    luminance: np.ndarray,
    *,
    input_max: int | float,
    output_max: int,
    config: ToneMappingSpec,
    frame_stats: FrameStats | None = None,
    input_reference: int | float | None = None,
) -> np.ndarray:
    """Map code-space luminance through the declared tone-mapping spec.

    ``input_max`` is the explicit physical code maximum (positive,
    <=2**32-1; integral for integer backends). ``input_reference`` is the
    optional white-reference denominator (defaults to ``input_max``);
    active paths normalize against the reference while the unsigned-code
    bound stays physical. ``output_max`` the integer output code maximum
    (1..65535). Disabled specs return a domain-appropriate identity before
    any operator, LUT, or finite-input work — the same object only when
    physical maximum, reference, and output maximum all match.
    """
    input_max, output_max = _require_range(input_max, output_max)
    reference = require_input_reference(input_reference, input_max)
    if not config.enabled:
        if input_max == output_max and reference == output_max:
            return luminance
        scaled = np.asarray(luminance, dtype=np.float64) * (output_max / reference)
        return np.clip(scaled, 0.0, float(output_max))
    src = _validate(luminance)
    check_code_range(src, input_max, ToneMappingError)
    if config.operator == "durand":
        from modules.tone_mapping import local_durand

        return local_durand.map_luminance(
            src,
            input_max=input_max,
            output_max=output_max,
            config=config,
            frame_stats=frame_stats,
            input_reference=input_reference,
        )
    if config.backend == "fixed":
        return _fixed_reinhard(src, config, input_max, output_max, frame_stats,
                               reference=reference, input_reference=input_reference)
    if config.backend == "lut":
        if src.dtype.kind == "u":
            int_max = _require_integral_input_max(input_max)
            ref_int = require_integral_reference(input_reference, reference, input_max)
            n, d = normalize_integer(
                src,
                normalization=config.normalization,
                input_max=ref_int,
                stats=frame_stats,
            )
            table = _LUT_CACHE.get(config, output_max)
            gain = 2.0**config.exposure_ev * config.domain_scale
            return _integer_lut(table, n, d, gain, _table_domain(config, gain))
        return _float_lut(src, config, input_max, output_max, frame_stats,
                          reference=reference)
    return _float_backend(src, config, input_max, output_max, frame_stats,
                          reference=reference)
