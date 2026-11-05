from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from modules.tone_mapping.registry import resolve_tone_mapping
from util.config_merge import (
    format_config_source,
    load_merged_yaml,
    normalize_config_paths,
    pipeline_config_paths,
)
from util.config_schema import (
    NATIVE_REQUIRED_CONFIG_KEYS,
    REQUIRED_CONFIG_KEYS,
    normalize_config,
)
from util.config_schema import validate_config
from util.output_contract import OutputSpec
from util.pipeline_migration import migrate_pipeline_config
from util.signal_range import LinearCodeRange, linear_code_range


@dataclass(frozen=True)
class PreparedConfig:
    source_config_original: dict[str, Any]
    effective_config: dict[str, Any]
    migration_report: dict[str, Any]
    warnings: tuple[str, ...]
    output_spec: OutputSpec
    code_range: LinearCodeRange
    config_source: str = field(default="<mapping>")


def _suppress_diagnostic_flags(config: Mapping[str, Any]) -> None:
    for value in config.values():
        if isinstance(value, dict):
            for key in ("is_save", "is_plot_curve", "is_debug"):
                if key in value:
                    value[key] = False
    platform = config.get("platform")
    if isinstance(platform, dict):
        platform["plot_histograms"] = False
        platform["generate_tv"] = False
        platform["debug_enabled"] = False
        platform["debug_log_file"] = None


def prepare_config(
    config: str | Path | list | tuple | Mapping[str, Any],
    *,
    options: Any | None = None,
    validate: bool = True,
) -> PreparedConfig:
    """Resolve a config exactly as ``BrilliantISP.load_config`` would, without
    constructing or running the facade.

    ``options`` is an optional object with ``diagnostics``/``output_dir``
    attributes; when present and diagnostics is false the same save/plot/debug
    flag suppression is applied to the internal copy. With ``validate`` the
    resolved effective config additionally passes the shared
    ``util.config_schema.validate_config`` error gate (the same check
    ``process_frame`` enforces) before being returned. The caller's mapping
    is never mutated.
    """
    if isinstance(config, Mapping):
        config_source = "<mapping>"
        normalized, notices = normalize_config(config)
    else:
        if isinstance(config, (str, Path)):
            paths = pipeline_config_paths(config)
        else:
            paths = normalize_config_paths(config)
        config_source = format_config_source(paths)
        normalized, notices = normalize_config(load_merged_yaml(paths))

    source_config_original = copy.deepcopy(normalized)
    migrated, migration_report = migrate_pipeline_config(
        normalized, canonicalize_tone_mapping=False
    )
    if options is not None and not getattr(options, "diagnostics", True):
        _suppress_diagnostic_flags(migrated)
    normalized, extra = normalize_config(migrated)
    for n in list(migration_report.get("notices", ())) + list(extra):
        if n not in notices:
            notices.append(n)

    required = (
        NATIVE_REQUIRED_CONFIG_KEYS
        if migration_report.get("target_version") == 2
        or int(normalized.get("schema_version") or 1) == 2
        else REQUIRED_CONFIG_KEYS
    )
    missing = [k for k in required if k not in normalized]
    if missing:
        raise KeyError(
            f"Config '{config_source}' missing required keys: {missing}. "
            "See config/base_hdr.yml plus a *_cam.yml overlay (e.g. svs_cam.yml)."
        )

    resolved_tm = resolve_tone_mapping(normalized)
    for notice in resolved_tm.notices:
        if notice not in notices:
            notices.append(notice)

    platform = normalized["platform"]
    platform["generate_tv"] = platform.get("generate_tv", False)
    if options is not None and getattr(options, "diagnostics", False):
        platform["output_dir"] = str(Path(options.output_dir) / "module_output")
        platform["debug_log_file"] = str(Path(options.output_dir) / "pipeline.log")
    elif "output_dir" not in platform:
        platform["output_dir"] = "module_output"
    if options is not None and getattr(options, "output_dir", None) is not None:
        platform["final_output_dir"] = str(options.output_dir)
    else:
        platform["final_output_dir"] = "out_frames/"
    sensor_info = normalized["sensor_info"]
    sensor_info.setdefault("horizontal_flip", False)
    sensor_info.setdefault("vertical_flip", False)
    normalized["digital_gain"]["exposure_correction_mode"] = normalized[
        "auto_exposure"
    ].get("exposure_correction_mode", "step")
    normalized["white_balance"].setdefault(
        "is_auto", normalized["auto_white_balance"]["is_enable"]
    )
    platform["rgb_output"] = normalized.get("rgb_conversion", {}).get("is_enable", True)
    initial = Path(platform["filename"]).stem
    platform["in_file"] = initial
    platform["out_file"] = (
        initial if platform.get("short_output_names", False) else "Out_" + initial
    )

    if validate:
        errors = [i for i in validate_config(normalized) if i.severity == "error"]
        if errors:
            raise ValueError(
                "config failed validation: "
                + "; ".join(f"{i.path}: {i.message}" for i in errors)
            )

    return PreparedConfig(
        source_config_original=source_config_original,
        effective_config=normalized,
        migration_report=migration_report,
        warnings=tuple(notices),
        output_spec=OutputSpec.from_mapping(normalized.get("output")),
        code_range=linear_code_range(normalized),
        config_source=config_source,
    )
