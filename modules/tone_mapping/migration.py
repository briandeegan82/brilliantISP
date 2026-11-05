"""Pure legacy->canonical tone-mapping migration (schema_version 1 only).

``migrate_tone_mapping`` returns a copied config whose selected TMO
becomes one canonical ``tone_mapping`` section, plus a structured report.
Source mappings are never mutated; schema_version 2 remains unsupported.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

from modules.tone_mapping.registry import (
    LEGACY_SECTIONS,
    ToneMappingError,
    resolve_tone_mapping,
    select_legacy_section,
)

LEGACY_TOP_SECTIONS = LEGACY_SECTIONS


def migrate_tone_mapping(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Migrate a config copy; returns (new_config, report).

    The selected section is resolved with a forced-enabled copy so its
    settings must be mathematically valid enough to translate even when
    disabled at runtime; the output preserves the original effective
    enable state. Unsupported/disabled-unknown selectors error rather
    than inventing an operator.
    """
    if not isinstance(config, Mapping):
        raise ToneMappingError("config must be a mapping")
    if "extends" in config:
        raise ToneMappingError("migration requires a resolved config; load extends with load_merged_yaml first")
    if "schema_version" in config:
        version = config["schema_version"]
        if type(version) is not int or version != 1:
            raise ToneMappingError(
                f"schema_version {version!r} is not supported by M3 "
                "migration (target schema_version 1 only)"
            )
        original_version: int | None = 1
    else:
        original_version = None
    source_tm = config.get("tone_mapping")
    if not isinstance(source_tm, Mapping):
        raise ToneMappingError("tone_mapping section must be a mapping")
    original = resolve_tone_mapping(config)
    canonical_input = "operator" in source_tm
    report: dict[str, Any] = {
        "migration": "tone_mapping_m3",
        "source_schema_version": original_version,
        "target_schema_version": 1,
        "source_tone_mapping": copy.deepcopy(dict(source_tm)),
        "notices": [],
        "potential_output_changes": [],
    }
    forced = copy.deepcopy(dict(config))
    forced["tone_mapping"]["is_enable"] = True
    if not canonical_input:
        selector = str(source_tm.get("tone_mapper", "reinhard_integer"))
        selected_name, selected = select_legacy_section(selector, config)
        if not isinstance(selected.get("is_enable", True), bool):
            raise ToneMappingError(f"{selected_name}.is_enable must be a boolean")
        forced[selected_name] = dict(forced.get(selected_name, {}))
        forced[selected_name]["is_enable"] = True
    resolved = resolve_tone_mapping(forced)

    spec = resolved.spec
    new_cfg = copy.deepcopy(dict(config))
    new_cfg["schema_version"] = 1
    canonical_tm: dict[str, Any] = {
        "is_enable": original.spec.enabled,
        "operator": spec.operator,
        "backend": spec.backend,
        "placement": resolved.placement,
        "normalization": spec.normalization,
        "exposure_ev": spec.exposure_ev,
        "domain_scale": spec.domain_scale,
        "normalize_output": spec.normalize_output,
        "parameters": dict(spec.parameters),
        "lut_domain": spec.lut_domain,
        "lut_size": spec.lut_size,
        "is_save": resolved.is_save,
        "is_plot_curve": resolved.is_plot_curve,
    }
    new_cfg["tone_mapping"] = canonical_tm

    legacy_sections: dict[str, Any] = {}
    inactive_presets: dict[str, Any] = {}
    for name in LEGACY_TOP_SECTIONS:
        if name in config:
            legacy_sections[name] = copy.deepcopy(config[name])
            if name != resolved.legacy_section:
                inactive_presets[name] = copy.deepcopy(config[name])
            new_cfg.pop(name, None)
    report["selected_legacy_section"] = resolved.legacy_section
    report["legacy_sections"] = legacy_sections
    report["inactive_presets"] = inactive_presets
    report["notices"] = list(resolved.notices)
    if not canonical_input:
        report["potential_output_changes"].append(
            "shared LUT interpolation / float64 math / clip-before-cast replaces duplicated legacy arithmetic"
        )
        if spec.backend in ("lut", "fixed"):
            report["potential_output_changes"].append(
                "declared bit depth replaces the legacy inferred/observed input range"
            )
    if not canonical_input and spec.operator == "aces_fitted":
        report["potential_output_changes"].append(
            "legacy ACES internal OETF/gamma stage removed; fitted curve "
            "only (exposure applied after normalization)"
        )
    if not canonical_input and spec.operator == "durand":
        report["potential_output_changes"].append(
            "local constant-output fallback preserves bounded normalized input rather than forcing white"
        )
    return new_cfg, report
