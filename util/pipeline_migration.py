from __future__ import annotations

import copy
import math
from typing import Any, Mapping

from util.output_contract import OutputSpec

RETIRED_COLOR_SECTIONS = (
    "color_space_conversion", "gamma_correction", "rgb_conversion", "yuv_conversion_format",
)


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    result = config.get(name, {})
    if not isinstance(result, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return result


def _flag(section: Mapping[str, Any], key: str, name: str, default: bool = False) -> bool:
    result = section.get(key, default)
    if type(result) is not bool:
        raise ValueError(f"{name}.{key} must be a boolean")
    return result


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and numeric")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite and numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite and numeric")
    return result


def _legacy_output(config: Mapping[str, Any], notices: list[str]) -> OutputSpec:
    sensor = _section(config, "sensor_info")
    gamma = _section(config, "gamma_correction")
    rgb = _section(config, "rgb_conversion")
    yuv = _section(config, "yuv_conversion_format")
    csc = _section(config, "color_space_conversion")
    overrides = _section(config, "output")
    packed = _flag(yuv, "is_enable", "yuv_conversion_format")
    rgb_enabled = _flag(rgb, "is_enable", "rgb_conversion", True)
    model = overrides.get("model", "ycbcr" if packed or not rgb_enabled else "rgb")
    if not isinstance(model, str):
        raise ValueError("output.model must be rgb or ycbcr")
    model = model.lower()
    depth = overrides.get("bit_depth", sensor.get("output_bit_depth", 8))
    if "bit_depth" not in overrides:
        numeric = _number(depth, "sensor_info.output_bit_depth")
        if not numeric.is_integer():
            raise ValueError("sensor_info.output_bit_depth must be integral")
        depth = int(numeric)
    enabled = _flag(gamma, "is_enable", "gamma_correction")
    curve = gamma.get("curve", "gamma")
    transfer = "srgb" if isinstance(curve, str) and curve.lower() == "srgb" else "power"
    values: dict[str, Any] = {
        "model": model,
        "bit_depth": depth,
        "format": "yuv" if model == "ycbcr" else "png",
        "transfer": transfer if enabled else "linear",
        "matrix": "bt709" if csc.get("conv_standard", 0) == 1 else "bt601",
        "is_save": any((
            _flag(gamma, "is_save", "gamma_correction"),
            _flag(csc, "is_save", "color_space_conversion"),
            _flag(rgb, "is_save", "rgb_conversion"),
            _flag(yuv, "is_save", "yuv_conversion_format"),
        )),
    }
    if values["is_save"]:
        notices.append("legacy color-stage diagnostics consolidate as the typed native output stage; individual domains require explicit stage captures")
    if enabled and transfer == "power":
        values["gamma"] = gamma.get("gamma", 2.2)
        if not isinstance(curve, str) or curve.lower() != "gamma":
            notices.append("legacy gamma curve was not sRGB; its executed power-law behavior is retained")
    if model == "ycbcr":
        values.update(range="limited", layout=yuv.get("conv_type", "444") if packed else "444", packing="yuyv")
        if values["transfer"] == "linear" and "transfer" not in overrides:
            values["transfer"] = "rec709"
            notices.append("legacy linear/mixed-domain YUV is replaced by explicitly Rec.709-encoded standard Y'CbCr")
        notices.append("standard Y'CbCr coefficients, limited/full range and averaged 4:2:2 chroma replace legacy mixed ranges/packing")
        if packed and rgb_enabled:
            notices.append("legacy packing after RGB conversion was not standard Y'CbCr; packing now follows explicit RGB-to-Y'CbCr conversion")
    if depth == 32 and model == "rgb":
        values.update(transfer="linear", format="tiff")
        values.pop("gamma", None)
        notices.append("legacy encoded integer-valued float32 is replaced by normalized linear RGB float32 TIFF")
    values.update(dict(overrides))
    if values.get("transfer") != "power":
        values.pop("gamma", None)
    return OutputSpec.from_mapping(values)


def _migrate_luma(
    config: dict[str, Any], source: Mapping[str, Any],
    report: dict[str, Any], strict: bool,
) -> None:
    for name in ("ldci", "2d_noise_reduction"):
        if name not in source:
            continue
        old = _section(source, name)
        body = config[name]
        report["source_luma"][name] = copy.deepcopy(dict(old))
        try:
            if name == "ldci":
                wind = _number(old.get("wind", 64), "ldci.wind")
                count = _number(old.get("clip_limit", 2), "ldci.clip_limit")
                if not wind.is_integer() or wind < 1 or count < 0:
                    raise ValueError("legacy LDCI requires positive integral wind and nonnegative clip_limit")
                count = 0.08 * wind if count >= wind else count
                body["wind"] = int(wind)
                body["clip_limit"] = count * 256 / (wind * wind)
                post = _flag(old, "post_gamma", "ldci")
                placement = old.get("placement", "post_transfer" if post else "pre_transfer")
                if placement not in ("pre_transfer", "post_transfer"):
                    raise ValueError("ldci.placement must be pre_transfer or post_transfer")
                body["placement"] = placement
                body.pop("post_gamma", None)
                report["notices"].append("LDCI clip_limit is a ratio to mean histogram count; 4096 bins retain total prior mass 256")
                if placement == "pre_transfer":
                    report["notices"].append("legacy pre-gamma LDCI remains an advanced display-linear luma operation, not standard Y'CbCr")
            else:
                strength = _number(old.get("wts", 1), "2d_noise_reduction.wts")
                body["strength"] = (strength if strength > 0 else 1) / 255
                body.pop("wts", None)
                for field, default in (("window_size", 3), ("patch_size", 3)):
                    value = _number(old.get(field, default), f"2d_noise_reduction.{field}")
                    if not value.is_integer() or value < 1:
                        raise ValueError(f"2d_noise_reduction.{field} must be a positive integer")
                    integer = int(value)
                    body[field] = integer if integer % 2 else integer + 1
                    if body[field] != integer:
                        report["notices"].append(f"2d_noise_reduction.{field} {integer} rounded up to odd {integer + 1}")
        except ValueError:
            if strict or _flag(old, "is_enable", name):
                raise
            config[name] = copy.deepcopy(dict(old))
            report["notices"].append(f"inactive {name} controls must be fixed before native luma processing is enabled")


def migrate_pipeline_config(
    source: Mapping[str, Any],
    *,
    canonicalize_tone_mapping: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(source, Mapping):
        raise ValueError("config must be a mapping")
    if "extends" in source:
        raise ValueError("migration requires resolved inheritance")
    version = source.get("schema_version", 1)
    if type(version) is not int or version not in (1, 2):
        raise ValueError(f"unsupported schema_version {version!r}")
    config = copy.deepcopy(dict(source))
    report: dict[str, Any] = {
        "migration": "native_color_m4",
        "source_schema_version": source.get("schema_version"),
        "target_schema_version": 2,
        "removed_sections": {},
        "removed_fields": {},
        "source_luma": {},
        "notices": [],
        "potential_output_changes": [],
    }
    notices: list[str] = report["notices"]
    if version == 1:
        if canonicalize_tone_mapping:
            from modules.tone_mapping.migration import migrate_tone_mapping

            config, report["tone_mapping"] = migrate_tone_mapping(config)
        output = _legacy_output(source, notices)
        sensor = _section(source, "sensor_info")
        bits = _number(
            sensor.get("pipeline_rgb_bit_depth", sensor.get("bit_depth", 16)),
            "sensor_info.pipeline_rgb_bit_depth",
        )
        if not bits.is_integer() or not 1 <= bits <= 32:
            raise ValueError("legacy AE RGB bit depth must be an integer in 1..32")
        meter_max = ((2**int(bits)) - 1) >> max(int(bits) - 8, 0)
        ae = _section(source, "auto_exposure")
        if "auto_exposure" in config:
            for field in ("center_illuminance", "target_luminance"):
                if field not in ae or ae[field] is None:
                    continue
                try:
                    value = _number(ae[field], f"auto_exposure.{field}")
                    if value < 0:
                        raise ValueError(f"auto_exposure.{field} must be nonnegative")
                except ValueError:
                    if canonicalize_tone_mapping or _flag(ae, "is_enable", "auto_exposure"):
                        raise
                    notices.append(f"inactive auto_exposure.{field} is invalid; it must be fixed before enabling AE")
                    continue
                config["auto_exposure"][field] = value if value <= 1 else value / meter_max
        report["legacy_ae_meter_max"] = meter_max
        if "white_balance" in config:
            manual = dict(_section(config, "white_balance"))
            auto = _flag(_section(source, "auto_white_balance"), "is_enable", "auto_white_balance")
            if "is_auto" in manual and manual["is_auto"] is not auto:
                report["removed_fields"]["white_balance.is_auto"] = manual["is_auto"]
                notices.append("legacy white_balance.is_auto followed auto_white_balance.is_enable; the executed mode is retained explicitly")
            manual["is_auto"] = auto
            config["white_balance"] = manual
        _migrate_luma(config, source, report, canonicalize_tone_mapping)
        report["potential_output_changes"] = [
            "float Bayer gain/demosaic/CCM preserve precision and headroom",
            "AE uses a named pre-tone-map normalized scene-linear RGB tap",
            "saturation precedes gamut clipping and the single output transfer",
            "luma processing and scaling precede final quantization; no mandatory uint8 YUV round trip",
            "standard Y'CbCr output and explicit native/preview representations replace legacy mixed domains",
        ]
    else:
        output = OutputSpec.from_mapping(_section(config, "output"))
        for name in RETIRED_COLOR_SECTIONS:
            if name in config and _flag(_section(config, name), "is_enable", name):
                raise ValueError(f"schema_version 2 uses output, not active legacy section {name}; migrate a version 1 config first")
            if name in config and _flag(_section(config, name), "is_save", name):
                notices.append(f"schema_version 2 ignores retired {name}.is_save; use output.is_save or explicit stage captures")
    for name in RETIRED_COLOR_SECTIONS:
        if name in config:
            report["removed_sections"][name] = config.pop(name)
    for section, fields in (
        ("sensor_info", ("output_bit_depth", "pipeline_rgb_bit_depth")),
        ("platform", ("rgb_output", "save_format")),
        ("digital_gain", ("ae_feedback",)),
    ):
        if section in config:
            body = config[section]
            if not isinstance(body, dict):
                raise ValueError(f"{section} must be a mapping")
            for field in fields:
                if field in body:
                    report["removed_fields"][f"{section}.{field}"] = body.pop(field)
    config["schema_version"] = 2
    config["output"] = output.as_config()
    notices.append("native AE center_illuminance/target_luminance are normalized scene-linear luminance; nominal white is 1")
    return config, report
