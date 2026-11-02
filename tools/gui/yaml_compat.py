"""YAML load/save using the BrilliantISP camera schema (source of truth)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from util.config_merge import load_merged_yaml, pipeline_config_paths

REQUIRED_CONFIG_KEYS = (
    "platform",
    "sensor_info",
    "dead_pixel_correction",
    "companding",
    "digital_gain",
    "lens_shading_correction",
    "bayer_noise_reduction",
    "black_level_correction",
    "white_balance",
    "auto_white_balance",
    "demosaic",
    "auto_exposure",
    "color_correction_matrix",
    "gamma_correction",
    "hdr_durand",
    "tone_mapping",
    "color_space_conversion",
    "color_saturation_enhancement",
    "ldci",
    "sharpen",
    "2d_noise_reduction",
    "rgb_conversion",
    "scale",
    "crop",
    "yuv_conversion_format",
)

# Document aliases that must never be written as YAML keys.
FORBIDDEN_ALIASES = frozenset({"filt_window", "isEnable", "gamma_lut_8", "gamma_lut_10", "gamma_lut_12", "gamma_lut_14"})


def yaml_safe_value(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: yaml_safe_value(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [yaml_safe_value(v) for v in obj]
    return obj


def load_preset_yaml(path: str | Path) -> dict[str, Any]:
    """Load a camera overlay or standalone YAML the same way the pipeline does."""
    paths = pipeline_config_paths(path)
    return copy.deepcopy(load_merged_yaml(paths))


def dump_config_yaml(config: dict[str, Any], path: str | Path) -> None:
    data = yaml_safe_value(copy.deepcopy(config))
    Path(path).write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )


def missing_required_keys(config: dict[str, Any]) -> list[str]:
    return [k for k in REQUIRED_CONFIG_KEYS if k not in config]


def contains_forbidden_aliases(config: dict[str, Any]) -> list[str]:
    found: list[str] = []

    def walk(obj: Any, prefix: str = "") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                path = f"{prefix}.{k}" if prefix else str(k)
                if k in FORBIDDEN_ALIASES:
                    found.append(path)
                walk(v, path)

    walk(config)
    return found


def merge_pipeline_feedback(dst: dict[str, Any], src: dict[str, Any]) -> None:
    si_s, si_d = src.get("sensor_info"), dst.get("sensor_info")
    if isinstance(si_s, dict) and isinstance(si_d, dict):
        # Crop mutates pipeline sensor_info width/height to the ROI. Copying that
        # back would make the next Process interpret the RAW at the cropped size.
        crop_on = bool((dst.get("crop") or {}).get("is_enable"))
        keys = ("bayer_pattern", "bit_depth", "hdr_bit_depth")
        if not crop_on:
            keys = ("width", "height", *keys)
        for k in keys:
            if k in si_s:
                si_d[k] = copy.deepcopy(si_s[k])
    wb_s, wb_d = src.get("white_balance"), dst.get("white_balance")
    if isinstance(wb_s, dict) and isinstance(wb_d, dict):
        for k in ("r_gain", "b_gain"):
            if k in wb_s:
                wb_d[k] = copy.deepcopy(wb_s[k])
    dg_s, dg_d = src.get("digital_gain"), dst.get("digital_gain")
    if isinstance(dg_s, dict) and isinstance(dg_d, dict):
        # Only mirror AE-driven gain when auto digital gain is actually active.
        # Otherwise Process must not overwrite a manual current_gain the user set.
        dg_auto = bool(dg_d.get("is_enable", True)) and bool(dg_d.get("is_auto", False))
        if dg_auto:
            for k in ("ae_feedback", "current_gain"):
                if k in dg_s:
                    dg_d[k] = copy.deepcopy(dg_s[k])
        elif "ae_feedback" in dg_s:
            dg_d["ae_feedback"] = copy.deepcopy(dg_s["ae_feedback"])
