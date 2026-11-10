"""Tone-mapping spec: declared operators, backend capabilities, validation.

A ``ToneMappingSpec`` is an immutable, hashable description of one tone
mapping request. The enabled flag is checked first: a globally disabled
spec bypasses all irrelevant operator/backend/curve validation (its
operator may be a legacy name resolved elsewhere) and discards supplied
parameters. Active specs validate parameter names/values, EV/domain
bounds, LUT settings, and scalar-curve endpoint representability using
the lead-owned operators.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from modules.tone_mapping import operators

OPERATOR_PARAMETERS: dict[str, dict[str, float]] = {
    "aces_fitted": {},
    "hable": {"exposure_bias": 2.0, "white_point": 11.2},
    "reinhard": {"knee": 0.25, "strength": 1.0},
    "durand": {
        "sigma_space": 2.0,
        "sigma_color": 0.4,
        "contrast_factor": 2.0,
        "downsample_factor": 1.0,
    },
}

OPERATOR_BACKENDS: dict[str, frozenset[str]] = {
    "aces_fitted": frozenset({"float", "lut"}),
    "hable": frozenset({"float", "lut"}),
    "reinhard": frozenset({"float", "lut", "fixed"}),
    "durand": frozenset({"cpu"}),
}

OPERATORS = frozenset(OPERATOR_PARAMETERS)
NORMALIZATIONS = frozenset({"declared_range", "per_frame"})
EXPOSURE_EV_RANGE = (-32.0, 32.0)
LUT_SIZE_RANGE = (256, 65536)


class ToneMappingError(ValueError):
    """Invalid tone-mapping spec; message names the offending field."""


def _finite_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToneMappingError(f"{name} must be a finite number, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ToneMappingError(
            f"{name} must be a finite number, got {value!r}"
        ) from exc
    if not math.isfinite(number):
        raise ToneMappingError(f"{name} must be finite, got {value!r}")
    return number


def _positive_finite(name: str, value: Any) -> float:
    number = _finite_number(name, value)
    if number <= 0:
        raise ToneMappingError(f"{name} must be strictly positive, got {value!r}")
    return number


def _check_curve_representable(
    operator: str, params: Mapping[str, float], gain: float, lut_domain: float,
    normalize_output: bool,
) -> None:
    scalar = {
        "aces_fitted": operators.aces_fitted,
        "hable": operators.hable,
        "reinhard": operators.reinhard,
    }
    fn = scalar.get(operator)
    if fn is None:
        return
    try:
        endpoint = float(fn(np.asarray([lut_domain]), **params)[0])
    except Exception as exc:
        raise ToneMappingError(
            f"curve evaluation at lut_domain failed: {exc}"
        ) from exc
    if not math.isfinite(endpoint):
        raise ToneMappingError(
            "curve output at lut_domain must be finite representable"
        )
    try:
        ref = float(fn(np.asarray([min(gain, lut_domain)]), **params)[0])
    except Exception as exc:
        raise ToneMappingError(
            f"theoretical reference evaluation failed: {exc}"
        ) from exc
    if not math.isfinite(ref):
        raise ToneMappingError(
            "theoretical normalization reference must be representable finite"
        )
    if normalize_output and ref <= 0:
        raise ToneMappingError(
            "theoretical normalization reference is not positive at "
            "min(gain, lut_domain)"
        )


@dataclass(frozen=True)
class ToneMappingSpec:
    """Declared tone-mapping request; validates itself on construction."""

    operator: str
    backend: str = "float"
    normalization: str = "declared_range"
    exposure_ev: float = 0.0
    domain_scale: float = 1.0
    normalize_output: bool = False
    parameters: tuple[tuple[str, float | int], ...] = ()
    lut_domain: float = 16.0
    lut_size: int = 65536
    enabled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ToneMappingError(f"enabled must be a boolean, got {self.enabled!r}")
        if not self.enabled:
            object.__setattr__(self, "operator", self.operator)
            object.__setattr__(self, "backend", self.backend)
            object.__setattr__(self, "parameters", ())
            return
        if not isinstance(self.normalize_output, bool):
            raise ToneMappingError(
                f"normalize_output must be a boolean, got {self.normalize_output!r}"
            )

        operator = str(self.operator).strip().lower()
        if operator not in OPERATORS:
            raise ToneMappingError(
                f"operator must be one of {sorted(OPERATORS)}, got {self.operator!r}"
            )
        object.__setattr__(self, "operator", operator)
        backend = str(self.backend).strip().lower()
        if backend not in OPERATOR_BACKENDS[operator]:
            raise ToneMappingError(
                f"backend must be one of {sorted(OPERATOR_BACKENDS[operator])} "
                f"for {operator}, got {self.backend!r}"
            )
        object.__setattr__(self, "backend", backend)
        normalization = str(self.normalization).strip().lower()
        if normalization not in NORMALIZATIONS:
            raise ToneMappingError(
                f"normalization must be one of {sorted(NORMALIZATIONS)}, "
                f"got {self.normalization!r}"
            )
        object.__setattr__(self, "normalization", normalization)
        ev = _finite_number("exposure_ev", self.exposure_ev)
        if not EXPOSURE_EV_RANGE[0] <= ev <= EXPOSURE_EV_RANGE[1]:
            raise ToneMappingError(
                f"exposure_ev must be in {EXPOSURE_EV_RANGE}, got {self.exposure_ev!r}"
            )
        object.__setattr__(self, "exposure_ev", ev)
        domain = _positive_finite("domain_scale", self.domain_scale)
        object.__setattr__(self, "domain_scale", domain)
        lut_domain = _positive_finite("lut_domain", self.lut_domain)
        object.__setattr__(self, "lut_domain", lut_domain)
        if (
            isinstance(self.lut_size, bool)
            or not isinstance(self.lut_size, int)
            or not LUT_SIZE_RANGE[0] <= self.lut_size <= LUT_SIZE_RANGE[1]
            or self.lut_size & (self.lut_size - 1)
        ):
            raise ToneMappingError(
                f"lut_size must be a power of two in "
                f"{LUT_SIZE_RANGE[0]}..{LUT_SIZE_RANGE[1]}, got {self.lut_size!r}"
            )
        gain = 2.0**ev * domain
        if not math.isfinite(gain) or gain <= 0:
            raise ToneMappingError(
                "combined gain 2**exposure_ev * domain_scale must be representable finite"
            )

        defaults = OPERATOR_PARAMETERS[operator]
        supplied: dict[str, float] = {}
        seen: set[str] = set()
        if not isinstance(self.parameters, (tuple, list)):
            raise ToneMappingError("parameters must be a sequence of (name, value) pairs")
        for item in self.parameters:
            if (
                not isinstance(item, (tuple, list))
                or len(item) != 2
                or not isinstance(item[0], str)
            ):
                raise ToneMappingError(
                    f"parameters entries must be (name, value) pairs, got {item!r}"
                )
            name, value = item
            if name in seen:
                raise ToneMappingError(f"duplicate parameter {name!r}")
            seen.add(name)
            if name not in defaults:
                raise ToneMappingError(
                    f"unknown parameter {name!r} for {operator}; "
                    f"allowed: {sorted(defaults)}"
                )
            supplied[name] = _positive_finite(f"parameters.{name}", value)
        if "downsample_factor" in supplied:
            ds = supplied["downsample_factor"]
            if not ds.is_integer() or ds < 1:
                raise ToneMappingError(
                    f"downsample_factor must be an integer >= 1, got {ds!r}"
                )
        merged = dict(defaults)
        merged.update(supplied)
        if operator == "durand" and self.normalize_output:
            raise ToneMappingError(
                "normalize_output is not applicable to local durand; "
                "frame adaptive normalization is inherent"
            )
        if operator == "aces_fitted" and supplied:
            raise ToneMappingError("aces_fitted has no curve parameters")
        if operator == "durand":
            if lut_domain > np.finfo(np.float32).max:
                raise ToneMappingError("durand lut_domain must be representable in float32")
            for name, value in merged.items():
                with np.errstate(over="ignore"):
                    f32 = np.float32(value)
                if not np.isfinite(f32) or (value != 0 and f32 == 0):
                    raise ToneMappingError(
                        f"parameters.{name} is not representable in float32"
                    )
        if not math.isfinite(gain / lut_domain) or gain / lut_domain <= 0:
            raise ToneMappingError(
                "gain/lut_domain ratio must be representable finite"
            )
        if backend == "fixed":
            w = (merged["knee"] / merged["strength"]) / min(gain, lut_domain)
            if not math.isfinite(w) or w <= 0:
                raise ToneMappingError(
                    "fixed knee/(strength*effective_gain) must be representable positive"
                )
        _check_curve_representable(
            operator, merged, gain, lut_domain, self.normalize_output
        )
        object.__setattr__(self, "parameters", tuple(sorted(merged.items())))

    def parameters_map(self) -> dict[str, float]:
        """Parameters as a plain dict."""
        return dict(self.parameters)


def spec_capabilities() -> dict[str, dict[str, Any]]:
    """Truthful capability table for reporting and UI."""
    return {
        name: {
            "backends": frozenset(OPERATOR_BACKENDS[name]),
            "parameters": dict(OPERATOR_PARAMETERS[name]),
            "integer_lut": "lut" in OPERATOR_BACKENDS[name],
            "local_operator": name == "durand",
        }
        for name in sorted(OPERATORS)
    }


LEGACY_SECTIONS = (
    "aces",
    "aces_integer",
    "hable",
    "hable_integer",
    "reinhard_integer",
    "integer_tmo",
    "hdr_durand",
    "durand",
)

LEGACY_MAPPER_ALIASES: dict[str, str] = {
    "aces": "aces",
    "aces_integer": "aces_integer",
    "hable": "hable",
    "hable_integer": "hable_integer",
    "reinhard_integer": "reinhard_integer",
    "integer_tmo": "reinhard_integer",
    "durand": "durand",
    "hdr_durand": "durand",
}


@dataclass(frozen=True)
class ResolvedToneMapping:
    """Effective tone-mapping request resolved from a full config."""

    spec: ToneMappingSpec
    placement: str
    is_save: bool
    is_plot_curve: bool
    notices: tuple[str, ...]
    legacy_section: str | None


def _num(section: Mapping[str, Any], key: str, default: float) -> float:
    return _finite_number(key, section.get(key, default))


def _alias_number(section: Mapping[str, Any], primary: str, fallback: str, default: float) -> float:
    return (
        _num(section, primary, default) if primary in section else _num(section, fallback, default)
    )


def _strict_bool(section: Mapping[str, Any], key: str, default: bool = False) -> bool:
    if key not in section:
        return bool(default)
    value = section[key]
    if not isinstance(value, bool):
        raise ToneMappingError(f"{key} must be a boolean, got {value!r}")
    return value


def _strict_placement(value: Any) -> str:
    placement = str(value)
    if placement not in ("pre_demosaic", "post_ccm"):
        raise ToneMappingError(
            f"placement must be pre_demosaic or post_ccm, got {value!r}"
        )
    return placement


def _check_aces_variant(section: Mapping[str, Any], name: str) -> None:
    variant = section.get("variant")
    if variant is not None and str(variant).lower() != "fitted":
        raise ToneMappingError(
            f"{name}.variant {variant!r} selects a canonical/full ACES "
            "variant that is not implemented; only aces_fitted exists"
        )


def _resolve_legacy(
    selector: str,
    section_name: str,
    section: Mapping[str, Any],
    notices: list[str],
) -> tuple[ToneMappingSpec, str]:
    enabled = _strict_bool(section, "is_enable", True)
    if selector in ("aces", "aces_integer"):
        for key in ("gamma", "apply_odt_gamma", "output_transform",
                    "surround", "white_point"):
            if key in section:
                notices.append(
                    f"{selector}.{key} is ignored: fitted operator has no "
                    "internal OETF/gamma stage (intentional output change)"
                )
    if not enabled:
        op, backend, norm, dscale = {
            "aces": ("aces_fitted", "float", "per_frame", 100.0),
            "aces_integer": ("aces_fitted", "lut", "per_frame", 1.0),
            "hable": ("hable", "float", "declared_range", 1.0),
            "hable_integer": ("hable", "lut", "per_frame", 1.0),
            "reinhard_integer": ("reinhard", "fixed", "declared_range", 1.0),
            "durand": ("durand", "cpu", "declared_range", 1.0),
        }[selector]
        return ToneMappingSpec(
            operator=op, backend=backend, normalization=norm,
            domain_scale=dscale, enabled=False,
        ), section_name
    if selector in ("aces",):
        _check_aces_variant(section, "aces")
        exposure = _alias_number(section, "exposure_adjust", "exposure_adjustment", 0.0)
        spec = ToneMappingSpec(
            operator="aces_fitted",
            backend="float",
            normalization="per_frame",
            exposure_ev=exposure,
            domain_scale=100.0,
            enabled=enabled,
        )
        return spec, section_name
    if selector == "aces_integer":
        _check_aces_variant(section, "aces_integer")
        exposure = _alias_number(section, "exposure_adjustment", "exposure_adjust", 0.0)
        normalization = (
            "per_frame" if _strict_bool(section, "use_normalization", True)
            else "declared_range"
        )
        spec = ToneMappingSpec(
            operator="aces_fitted",
            backend="lut",
            normalization=normalization,
            exposure_ev=exposure,
            domain_scale=_num(section, "hdr_scale", 1.0),
            normalize_output=_strict_bool(section, "normalize_output", False),
            enabled=enabled,
        )
        return spec, section_name
    if selector == "hable":
        bias = _alias_number(section, "exposure_bias", "exposure_adjust", 2.0)
        white = _num(section, "white_point", 11.2)
        spec = ToneMappingSpec(
            operator="hable",
            backend="float",
            normalization="declared_range",
            parameters=(
                ("exposure_bias", bias), ("white_point", white)),
            enabled=enabled,
        )
        for key in ("normalize_output", "use_normalization", "hdr_scale"):
            if key in section:
                notices.append(
                    f"hable.{key} is ignored by the float backend "
                    "(previously honored only by the integer backend)"
                )
        return spec, section_name
    if selector == "hable_integer":
        normalization = (
            "per_frame" if _strict_bool(section, "use_normalization", True)
            else "declared_range"
        )
        spec = ToneMappingSpec(
            operator="hable",
            backend="lut",
            normalization=normalization,
            domain_scale=_num(section, "hdr_scale", 1.0),
            normalize_output=_strict_bool(section, "normalize_output", False),
            parameters=(
                ("exposure_bias", _alias_number(section, "exposure_bias", "exposure_adjust", 2.0)),
                ("white_point", _num(section, "white_point", 11.2)),
            ),
            enabled=enabled,
        )
        return spec, section_name
    if selector == "reinhard_integer":
        spec = ToneMappingSpec(
            operator="reinhard",
            backend="fixed",
            normalization="declared_range",
            normalize_output=_strict_bool(section, "normalize_output", False),
            parameters=(
                ("knee", _num(section, "knee", 0.25)),
                ("strength", _num(section, "strength", 1.0)),
            ),
            enabled=enabled,
        )
        return spec, section_name
    if selector == "durand":
        downsample = section.get("downsample_factor", 1)
        if downsample != 1:
            notices.append(
                f"hdr_durand.downsample_factor {downsample!r} was unused "
                "in the legacy implementation; migrated to 1"
            )
        spec = ToneMappingSpec(
            operator="durand",
            backend="cpu",
            normalization="declared_range",
            parameters=(
                ("sigma_space", _num(section, "sigma_space", 2.0)),
                ("sigma_color", _num(section, "sigma_color", 0.4)),
                ("contrast_factor", _num(section, "contrast_factor", 2.0)),
                ("downsample_factor", 1.0),
            ),
            enabled=enabled,
        )
        return spec, section_name
    raise ToneMappingError(f"unknown legacy tone_mapper {selector!r}")


def _canonical_params(section: Mapping[str, Any]) -> tuple[tuple[str, float], ...]:
    raw = section.get("parameters", {})
    if not isinstance(raw, Mapping):
        raise ToneMappingError(
            f"tone_mapping.parameters must be a mapping, got {type(raw).__name__}"
        )
    return tuple(raw.items())


_LEGACY_SELECTOR_SECTION: dict[str, tuple[str, ...]] = {
    "aces": ("aces",),
    "aces_integer": ("aces_integer",),
    "hable": ("hable",),
    "hable_integer": ("hable_integer",),
    "reinhard_integer": ("reinhard_integer", "integer_tmo"),
    "integer_tmo": ("integer_tmo", "reinhard_integer"),
    "durand": ("hdr_durand",),
    "hdr_durand": ("hdr_durand",),
}


def select_legacy_section(
    selector_raw: str, full_config: Mapping[str, Any]
) -> tuple[str, Mapping[str, Any]]:
    """Pick the real legacy section for a selector, with alias fallback."""
    candidates = _LEGACY_SELECTOR_SECTION.get(selector_raw)
    if candidates is None:
        raise ToneMappingError(
            f"unknown tone_mapper {selector_raw!r}; expected one of "
            f"{sorted(LEGACY_MAPPER_ALIASES)}"
        )
    for name in candidates:
        if name not in full_config:
            continue
        section = full_config[name]
        if isinstance(section, Mapping):
            return name, section
        raise ToneMappingError(
            f"{name} section must be a mapping, got {type(section).__name__}"
        )
    return candidates[0], {}


def resolve_tone_mapping(full_config: Mapping[str, Any]) -> ResolvedToneMapping:
    """Resolve the effective tone-mapping spec from a whole pipeline config."""
    notices: list[str] = []
    if not isinstance(full_config, Mapping):
        raise ToneMappingError("full config must be a mapping")
    tm = full_config.get("tone_mapping", {})
    if not isinstance(tm, Mapping):
        raise ToneMappingError("tone_mapping section must be a mapping")
    global_enable = _strict_bool(tm, "is_enable", True)
    is_save = _strict_bool(tm, "is_save", False)
    is_plot_curve = _strict_bool(tm, "is_plot_curve", False)

    canonical_operator = tm.get("operator")
    if "operator" in tm:
        placement = _strict_placement(tm.get("placement", "post_ccm"))
        if "tone_mapper" in tm:
            notices.append(
                "canonical tone_mapping.operator overrides legacy "
                "tone_mapper selector"
            )
        if "tone_mapping_before_demosaic" in tm and "placement" not in tm:
            notices.append(
                "legacy tone_mapping_before_demosaic ignored; canonical "
                "placement defaults to post_ccm"
            )
        if not global_enable:
            spec = ToneMappingSpec(
                operator=str(canonical_operator), enabled=False)
            return ResolvedToneMapping(spec, placement, is_save,
                                       is_plot_curve, tuple(notices), None)
        allowed = {
            "operator", "backend", "placement", "normalization", "exposure_ev",
            "domain_scale", "normalize_output", "parameters", "lut_domain", "lut_size",
            "is_enable", "is_save", "is_plot_curve", "tone_mapper", "tone_mapping_before_demosaic",
        }
        unknown = set(tm) - allowed
        if unknown:
            raise ToneMappingError(
                f"unsupported canonical tone_mapping fields: {sorted(unknown, key=str)}"
            )
        if "backend" not in tm:
            raise ToneMappingError(
                "canonical tone_mapping.operator requires an explicit "
                "backend (float/lut/fixed/cpu)"
            )
        spec = ToneMappingSpec(
            operator=str(canonical_operator),
            backend=str(tm["backend"]),
            normalization=str(tm.get("normalization", "declared_range")),
            exposure_ev=tm.get("exposure_ev", 0.0),
            domain_scale=tm.get("domain_scale", 1.0),
            normalize_output=_strict_bool(tm, "normalize_output", False),
            parameters=_canonical_params(tm),
            lut_domain=tm.get("lut_domain", 16.0),
            lut_size=tm.get("lut_size", 65536),
            enabled=True,
        )
        return ResolvedToneMapping(
            spec, placement, is_save, is_plot_curve, tuple(notices), None
        )

    selector_raw = str(tm.get("tone_mapper", "reinhard_integer"))
    if "tone_mapping_before_demosaic" in tm:
        pre = _strict_bool(tm, "tone_mapping_before_demosaic", True)
    else:
        pre = True
    placement = "pre_demosaic" if pre else "post_ccm"

    if not global_enable:
        return ResolvedToneMapping(
            ToneMappingSpec(operator=selector_raw, enabled=False),
            placement, is_save, is_plot_curve, tuple(notices), selector_raw,
        )

    selector = LEGACY_MAPPER_ALIASES.get(selector_raw)
    if selector is None:
        raise ToneMappingError(
            f"unknown tone_mapper {selector_raw!r}; expected one of "
            f"{sorted(LEGACY_MAPPER_ALIASES)}"
        )
    section_name, section = select_legacy_section(selector_raw, full_config)
    if section_name != selector_raw:
        notices.append(
            f"{selector} section not present; using {section_name} settings "
            f"for selector {selector_raw!r}"
        )
    spec, legacy_section = _resolve_legacy(
        selector, section_name, section, notices
    )
    if "is_save" not in tm:
        is_save = _strict_bool(section, "is_save", False)
    if "is_plot_curve" not in tm:
        is_plot_curve = _strict_bool(section, "is_plot_curve", False)
    return ResolvedToneMapping(
        spec, placement, is_save, is_plot_curve, tuple(notices), legacy_section
    )


def placement_legacy(  # pylint: disable=unused-argument
    tm: Mapping[str, Any], section: Mapping[str, Any]
) -> str:
    """Legacy placement: the pre-demosaic boolean wins over section flags."""
    if _strict_bool(tm, "tone_mapping_before_demosaic", True):
        return "pre_demosaic"
    return "post_ccm"
