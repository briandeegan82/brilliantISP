"""Shared pipeline-configuration metadata, normalization, and validation.

This module is the M1a foundation: it centralizes the required top-level
section list, provides frozen parameter/block descriptors seeded from the
audited historical defaults in ``config/unused/base_hdr.yml``, normalizes
loaded mappings without mutating them, and implements the numeric validation
shared by the calibration GUI and future CLI/migration tooling.

Scope notes:

- Descriptor coverage is intentionally partial. It includes the fields the
  current GUI validators consume plus the explicit enumerations exposed by
  ``tools/gui/ui_schema.ENUM_CHOICES``. It is not yet a complete per-block
  schema.
- ``normalize_config`` does not insert defaults or retune values. Legacy
  versionless mappings are accepted with a notice; ``schema_version: 2`` is
  rejected because the native v2 engine does not exist yet.
- Validation preserves the legacy geometry rules: odd crop origins are still
  allowed (warning only); the strict even-origin Bayer crop rule is deferred.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from util.raw_decode import RawFormat, RawFormatError

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

LEGACY_OPERATOR_SECTIONS = (
    "aces",
    "aces_integer",
    "hable",
    "hable_integer",
    "reinhard_integer",
    "integer_tmo",
    "hdr_durand",
    "durand",
)

KNOWN_TOP_LEVEL_KEYS = frozenset(
    REQUIRED_CONFIG_KEYS
    + LEGACY_OPERATOR_SECTIONS
    + (
        "oecf",
        "r_lut",
        "noise_reduction_2d",
        "yuv_conv_format",
        "schema_version",
        "extends",
        "output",
    )
)

BAYER_PATTERN_CHOICES = ("rggb", "grbg", "gbrg", "bggr")
VALID_ENDIAN_TYPES = ("ieee-le", "ieee-be")
VALID_DATA_FORMATS = ("uint8", "uint16", "uint32", "packed24", "word_pair32")

SUPPORTED_SCHEMA_VERSIONS = (1, 2)

# Native v2 required sections: legacy retired color sections are replaced by
# the typed output stage; sensor output_bit_depth/pipeline_rgb_bit_depth and
# platform rgb_output/save_format are retired by migration too.
NATIVE_REQUIRED_CONFIG_KEYS = tuple(
    key
    for key in REQUIRED_CONFIG_KEYS
    if key
    not in (
        "color_space_conversion",
        "gamma_correction",
        "rgb_conversion",
        "yuv_conversion_format",
    )
) + ("output",)


class _Missing:
    def __repr__(self) -> str:
        return "ABSENT"

    def __copy__(self) -> "_Missing":
        return self

    def __deepcopy__(self, memo: dict) -> "_Missing":
        return self


ABSENT: Any = _Missing()


@dataclass(frozen=True)
class ParameterSpec:
    path: str
    value_type: str
    default: Any = ABSENT
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] = ()
    units: str = ""
    help: str = ""
    doc_anchor: str = ""
    affected_stage: str = ""

    @property
    def has_default(self) -> bool:
        return self.default is not ABSENT


@dataclass(frozen=True)
class BlockSpec:
    block_id: str
    label: str
    sections: tuple[str, ...]
    parameters: tuple[ParameterSpec, ...] = ()
    doc_page: str = ""


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str
    severity: str = "error"


_BASE_GAIN_ARRAY = [
    0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 16, 32, 64,
]

PARAMETER_SPECS: tuple[ParameterSpec, ...] = (
    ParameterSpec("platform.filename", "str", default="frame_0000.raw", affected_stage="raw_input"),
    ParameterSpec("platform.save_format", "str", default="png", choices=("png", "jpg", "tiff"), affected_stage="output"),
    ParameterSpec("platform.short_output_names", "bool", default=True, affected_stage="output"),
    ParameterSpec("platform.debug_enabled", "bool", default=True, affected_stage="diagnostics"),
    ParameterSpec("platform.debug_log_level", "str", default="INFO",
                  choices=("DEBUG", "INFO", "WARNING", "ERROR"), affected_stage="diagnostics"),
    ParameterSpec("platform.plot_histograms", "bool", default=True, affected_stage="diagnostics"),
    ParameterSpec("sensor_info.bayer_pattern", "str", default="gbrg",
                  choices=BAYER_PATTERN_CHOICES, affected_stage="raw_input"),
    ParameterSpec("sensor_info.hdr_bit_depth", "int", default=24, units="bits", affected_stage="raw_input"),
    ParameterSpec("sensor_info.bit_depth", "int", default=12, minimum=1, maximum=32,
                  units="bits", affected_stage="raw_input"),
    ParameterSpec("sensor_info.pipeline_rgb_bit_depth", "int", default=16, units="bits"),
    ParameterSpec("sensor_info.width", "int", default=1920, minimum=1, units="px", affected_stage="raw_input"),
    ParameterSpec("sensor_info.height", "int", default=1563, minimum=1, units="px", affected_stage="raw_input"),
    ParameterSpec("sensor_info.output_bit_depth", "int", default=8, units="bits", affected_stage="output"),
    ParameterSpec("sensor_info.sensor", "str", default="base_camera"),
    ParameterSpec("sensor_info.endian_type", "str", default="ieee-be",
                  choices=VALID_ENDIAN_TYPES, affected_stage="raw_input"),
    ParameterSpec("sensor_info.data_format", "str", default="uint16",
                  choices=VALID_DATA_FORMATS, affected_stage="raw_input"),
    ParameterSpec("sensor_info.storage_format", "str", default=ABSENT,
                  choices=VALID_DATA_FORMATS, affected_stage="raw_input"),
    ParameterSpec("sensor_info.source_format", "str", default=ABSENT,
                  choices=("auto", "raw", "pgm", "tiff", "rawpy"),
                  affected_stage="raw_input"),
    ParameterSpec("sensor_info.header_bytes", "int", default=ABSENT, minimum=0,
                  units="bytes", affected_stage="raw_input"),
    ParameterSpec("sensor_info.row_stride_bytes", "int", default=ABSENT, minimum=1,
                  units="bytes", affected_stage="raw_input"),
    ParameterSpec("sensor_info.word_order", "str", default=ABSENT,
                  choices=("low_first", "high_first"), affected_stage="raw_input"),
    ParameterSpec("sensor_info.embedded_rows_top", "int", default=ABSENT, minimum=0,
                  units="rows", affected_stage="raw_input"),
    ParameterSpec("sensor_info.embedded_rows_bottom", "int", default=ABSENT, minimum=0,
                  units="rows", affected_stage="raw_input"),
    ParameterSpec("sensor_info.manual_bit_shift", "int", default=ABSENT, minimum=0,
                  units="bits", affected_stage="raw_input"),
    ParameterSpec("sensor_info.data_alignment", "str", default="LSB",
                  choices=("LSB", "MSB", "MANUAL"), affected_stage="raw_input"),
    ParameterSpec("sensor_info.horizontal_flip", "bool", default=False, affected_stage="raw_input"),
    ParameterSpec("sensor_info.vertical_flip", "bool", default=False, affected_stage="raw_input"),
    ParameterSpec("crop.is_enable", "bool", default=False, affected_stage="crop"),
    ParameterSpec("crop.crop_x_start", "int", default=1220, minimum=0, units="px", affected_stage="crop"),
    ParameterSpec("crop.crop_y_start", "int", default=440, minimum=0, units="px", affected_stage="crop"),
    ParameterSpec("crop.new_width", "int", default=300, minimum=1, units="px", affected_stage="crop"),
    ParameterSpec("crop.new_height", "int", default=300, minimum=1, units="px", affected_stage="crop"),
    ParameterSpec("dead_pixel_correction.is_enable", "bool", default=True, affected_stage="dead_pixel_correction"),
    ParameterSpec("dead_pixel_correction.dp_threshold", "number", default=64, affected_stage="dead_pixel_correction"),
    ParameterSpec("black_level_correction.is_enable", "bool", default=False, affected_stage="black_level_correction"),
    ParameterSpec("black_level_correction.r_offset", "number", default=16, affected_stage="black_level_correction"),
    ParameterSpec("black_level_correction.gr_offset", "number", default=16, affected_stage="black_level_correction"),
    ParameterSpec("black_level_correction.gb_offset", "number", default=16, affected_stage="black_level_correction"),
    ParameterSpec("black_level_correction.b_offset", "number", default=16, affected_stage="black_level_correction"),
    ParameterSpec("companding.is_enable", "bool", default=True, affected_stage="decompanding"),
    ParameterSpec("companding.pre_linearization_black_level", "number", default=240, affected_stage="decompanding"),
    ParameterSpec("companding.post_linearization_black_level", "number", default=0, affected_stage="decompanding"),
    ParameterSpec("digital_gain.is_enable", "bool", default=True, affected_stage="digital_gain"),
    ParameterSpec("digital_gain.is_auto", "bool", default=False, affected_stage="digital_gain"),
    ParameterSpec("digital_gain.gain_array", "list", default=tuple(_BASE_GAIN_ARRAY),
                  units="x", affected_stage="digital_gain"),
    ParameterSpec("digital_gain.current_gain", "int", default=10, minimum=0, affected_stage="digital_gain"),
    ParameterSpec("digital_gain.exposure_correction_mode", "str",
                  choices=("direct", "step"), affected_stage="auto_exposure"),
    ParameterSpec("lens_shading_correction.is_enable", "bool", default=False, affected_stage="lens_shading_correction"),
    ParameterSpec("lens_shading_correction.r_k1", "number", default=4.25, affected_stage="lens_shading_correction"),
    ParameterSpec("lens_shading_correction.r_k2", "number", default=4.08, affected_stage="lens_shading_correction"),
    ParameterSpec("bayer_noise_reduction.is_enable", "bool", default=True, affected_stage="bayer_noise_reduction"),
    ParameterSpec("bayer_noise_reduction.filter_window", "int", default=5, minimum=1,
                  units="px", affected_stage="bayer_noise_reduction"),
    ParameterSpec("auto_white_balance.is_enable", "bool", default=True, affected_stage="auto_white_balance"),
    ParameterSpec("auto_white_balance.algorithm", "str", default="pca",
                  choices=("grey_world", "norm_2", "pca"), affected_stage="auto_white_balance"),
    ParameterSpec("auto_white_balance.percentage", "number", default=3.5, affected_stage="auto_white_balance"),
    ParameterSpec("auto_white_balance.underexposed_percentage", "number", default=0.05,
                  affected_stage="auto_white_balance"),
    ParameterSpec("auto_white_balance.overexposed_percentage", "number", default=0.05,
                  affected_stage="auto_white_balance"),
    ParameterSpec("white_balance.is_enable", "bool", default=True, affected_stage="white_balance"),
    ParameterSpec("white_balance.is_auto", "bool", affected_stage="white_balance"),
    ParameterSpec("white_balance.r_gain", "number", default=1.0, minimum=0.0,
                  units="x", affected_stage="white_balance"),
    ParameterSpec("white_balance.b_gain", "number", default=1.0, minimum=0.0,
                  units="x", affected_stage="white_balance"),
    ParameterSpec("demosaic.algorithm", "str", default="bilinear",
                  choices=("bilinear", "malvar", "vng", "vng_opt", "hamilton_adams",
                           "hamilton_adams_opt", "ppg", "ppg_opt", "ahd", "ahd_opt",
                           "lmmse", "lmmse_opt", "lmmse_fast"),
                  affected_stage="demosaic"),
    ParameterSpec("color_correction_matrix.is_enable", "bool", default=True,
                  affected_stage="color_correction_matrix"),
    ParameterSpec("gamma_correction.is_enable", "bool", default=True, affected_stage="gamma_correction"),
    ParameterSpec("gamma_correction.curve", "str", default="srgb",
                  choices=("srgb", "gamma"), affected_stage="gamma_correction"),
    ParameterSpec("gamma_correction.gamma", "number", default=2.2, affected_stage="gamma_correction"),
    ParameterSpec("auto_exposure.is_enable", "bool", default=True, affected_stage="auto_exposure"),
    ParameterSpec("auto_exposure.center_illuminance", "number", default=0.4, affected_stage="auto_exposure"),
    ParameterSpec("auto_exposure.histogram_skewness", "number", default=0.9, affected_stage="auto_exposure"),
    ParameterSpec("auto_exposure.target_luminance", "number", default=0.5, affected_stage="auto_exposure"),
    ParameterSpec("auto_exposure.exposure_correction_mode", "str",
                  choices=("direct", "step"), affected_stage="auto_exposure"),
    ParameterSpec("color_space_conversion.conv_standard", "int", default=2,
                  affected_stage="color_space_conversion"),
    ParameterSpec("color_saturation_enhancement.is_enable", "bool", default=True, affected_stage="saturation"),
    ParameterSpec("color_saturation_enhancement.saturation_gain", "number", default=1.0,
                  units="x", affected_stage="saturation"),
    ParameterSpec("ldci.is_enable", "bool", default=False, affected_stage="ldci"),
    ParameterSpec("ldci.placement", "str", default="post_transfer",
                  choices=("pre_transfer", "post_transfer"), affected_stage="ldci"),
    ParameterSpec("2d_noise_reduction.strength", "number", default=10.0 / 255.0,
                  minimum=0.0, units="normalized", affected_stage="2d_noise_reduction"),
    ParameterSpec("oecf.output_bit_depth", "int", minimum=1.0, maximum=32.0,
                  affected_stage="oecf"),
    ParameterSpec("output.model", "str", default="rgb", choices=("rgb", "ycbcr"), affected_stage="output"),
    ParameterSpec("output.color_space", "str", default="srgb", choices=("srgb",), affected_stage="output"),
    ParameterSpec("output.transfer", "str", default="srgb",
                  choices=("srgb", "rec709", "power", "linear"), affected_stage="output"),
    ParameterSpec("output.gamma", "number", default=2.2, minimum=0.0, affected_stage="output"),
    ParameterSpec("output.bit_depth", "int", default=8, affected_stage="output"),
    ParameterSpec("output.format", "str", default="png",
                  choices=("png", "jpeg", "tiff", "yuv"), affected_stage="output"),
    ParameterSpec("output.matrix", "str", default="bt709", choices=("bt601", "bt709"), affected_stage="output"),
    ParameterSpec("output.range", "str", default="full", choices=("full", "limited"), affected_stage="output"),
    ParameterSpec("output.layout", "str", default="444", choices=("444", "422"), affected_stage="output"),
    ParameterSpec("output.packing", "str", default="yuyv", choices=("yuyv", "uyvy"), affected_stage="output"),
    ParameterSpec("output.is_save", "bool", default=False, affected_stage="output"),
    ParameterSpec("ldci.clip_limit", "number", default=2.0, affected_stage="ldci"),
    ParameterSpec("ldci.wind", "int", default=128, affected_stage="ldci"),
    ParameterSpec("sharpen.is_enable", "bool", default=True, affected_stage="sharpen"),
    ParameterSpec("sharpen.sharpen_sigma", "number", default=3.0, affected_stage="sharpen"),
    ParameterSpec("sharpen.sharpen_strength", "number", default=1.0, affected_stage="sharpen"),
    ParameterSpec("2d_noise_reduction.is_enable", "bool", default=False, affected_stage="noise_reduction_2d"),
    ParameterSpec("2d_noise_reduction.window_size", "int", default=9, affected_stage="noise_reduction_2d"),
    ParameterSpec("2d_noise_reduction.patch_size", "int", default=5, affected_stage="noise_reduction_2d"),
    ParameterSpec("rgb_conversion.is_enable", "bool", default=True, affected_stage="rgb_conversion"),
    ParameterSpec("scale.is_enable", "bool", default=False, affected_stage="scale"),
    ParameterSpec("scale.new_width", "int", default=1280, minimum=1, units="px", affected_stage="scale"),
    ParameterSpec("scale.new_height", "int", default=1024, minimum=1, units="px", affected_stage="scale"),
    ParameterSpec("scale.algorithm", "str", default="Bilinear",
                  choices=("Nearest_Neighbor", "Bilinear"), affected_stage="scale"),
    ParameterSpec("scale.upscale_method", "str", default="Bilinear",
                  choices=("Nearest_Neighbor", "Bilinear"), affected_stage="scale"),
    ParameterSpec("scale.downscale_method", "str", default="Bilinear",
                  choices=("Nearest_Neighbor", "Bilinear"), affected_stage="scale"),
    ParameterSpec("yuv_conversion_format.is_enable", "bool", default=True, affected_stage="yuv_output"),
    ParameterSpec("yuv_conversion_format.conv_type", "str", default="444",
                  choices=("444", "422"), affected_stage="yuv_output"),
    ParameterSpec("tone_mapping.is_enable", "bool", default=True, affected_stage="tone_mapping"),
    ParameterSpec("tone_mapping.tone_mapper", "str", default="reinhard_integer",
                  choices=("reinhard_integer", "aces", "aces_integer", "hable",
                           "hable_integer", "durand", "integer_tmo",
                           "hdr_durand"),
                  affected_stage="tone_mapping"),
    ParameterSpec("tone_mapping.tone_mapping_before_demosaic", "bool", default=True,
                  affected_stage="tone_mapping"),
    ParameterSpec("tone_mapping.operator", "str", default=ABSENT,
                  choices=("aces_fitted", "hable", "reinhard", "durand"),
                  affected_stage="tone_mapping"),
    ParameterSpec("tone_mapping.backend", "str", default=ABSENT,
                  choices=("float", "lut", "fixed", "cpu"),
                  affected_stage="tone_mapping"),
    ParameterSpec("tone_mapping.placement", "str", default=ABSENT,
                  choices=("pre_demosaic", "post_ccm"),
                  affected_stage="tone_mapping"),
    ParameterSpec("tone_mapping.normalization", "str", default=ABSENT,
                  choices=("declared_range", "per_frame"),
                  affected_stage="tone_mapping"),
    ParameterSpec("tone_mapping.exposure_ev", "number", default=ABSENT,
                  units="EV", affected_stage="tone_mapping"),
    ParameterSpec("tone_mapping.domain_scale", "number", default=ABSENT,
                  affected_stage="tone_mapping"),
    ParameterSpec("tone_mapping.normalize_output", "bool", default=ABSENT,
                  affected_stage="tone_mapping"),
    ParameterSpec("tone_mapping.lut_domain", "number", default=ABSENT,
                  affected_stage="tone_mapping"),
    ParameterSpec("tone_mapping.lut_size", "int", default=ABSENT,
                  affected_stage="tone_mapping"),
    ParameterSpec("aces.variant", "str", default="fitted",
                  choices=("fitted",), affected_stage="tone_mapping",
                  help="Only the fitted curve exists; canonical/full ACES is not implemented."),
    ParameterSpec("aces.exposure_adjustment", "number", default=0.0,
                  units="EV", affected_stage="tone_mapping"),
    ParameterSpec("hable.exposure_bias", "number", default=2.0, affected_stage="tone_mapping"),
    ParameterSpec("hable.white_point", "number", default=11.2, affected_stage="tone_mapping"),
)


def _specs_for(sections: Sequence[str]) -> tuple[ParameterSpec, ...]:
    prefixes = tuple(f"{s}." for s in sections)
    return tuple(p for p in PARAMETER_SPECS if p.path.startswith(prefixes))


_BLOCK_SECTIONS = (
    ("platform", "Platform / IO"),
    ("sensor_info", "Sensor information"),
    ("crop", "Crop"),
    ("dead_pixel_correction", "Dead pixel correction"),
    ("black_level_correction", "Black level correction"),
    ("companding", "Decompanding (PWC)"),
    ("digital_gain", "Digital gain"),
    ("lens_shading_correction", "Lens shading correction"),
    ("bayer_noise_reduction", "Bayer noise reduction"),
    ("auto_white_balance", "Auto white balance"),
    ("white_balance", "White balance"),
    ("demosaic", "Demosaic"),
    ("color_correction_matrix", "Color correction matrix"),
    ("gamma_correction", "Gamma correction"),
    ("auto_exposure", "Auto exposure"),
    ("color_space_conversion", "Color space conversion"),
    ("color_saturation_enhancement", "Color saturation enhancement"),
    ("ldci", "LDCI (CLAHE)"),
    ("sharpen", "Sharpen"),
    ("2d_noise_reduction", "2D noise reduction"),
    ("rgb_conversion", "RGB conversion"),
    ("scale", "Scale"),
    ("yuv_conversion_format", "YUV output format"),
    ("tone_mapping", "Tone mapping"),
)

BLOCK_SPECS: tuple[BlockSpec, ...] = tuple(
    BlockSpec(
        block_id=block_id,
        label=label,
        sections=(block_id,),
        parameters=_specs_for((block_id,)),
        doc_page=f"docs/blocks/{block_id}.md",
    )
    for block_id, label in _BLOCK_SECTIONS
)


def spec_map() -> dict[str, ParameterSpec]:
    return {p.path: p for p in PARAMETER_SPECS}


def normalize_config(mapping: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Deep-copy ``mapping`` and return ``(normalized, notices)``.

    Does not insert defaults or retune values. Versionless mappings and
    ``schema_version: 1`` are accepted as legacy input; ``schema_version: 2``
    uses the native color and output contracts.
    Unknown settings are retained and reported as warnings in ``notices``.
    """
    if not isinstance(mapping, Mapping):
        raise TypeError(
            f"normalize_config expects a mapping, got {type(mapping).__name__}"
        )
    cfg: dict[str, Any] = copy.deepcopy(dict(mapping))
    notices: list[str] = []

    if "schema_version" not in cfg:
        version: Any = None
        has_version = False
    else:
        version = cfg["schema_version"]
        has_version = True
    if not has_version:
        notices.append(
            "legacy config without schema_version: accepted as versionless legacy input"
        )
    elif isinstance(version, bool) or not isinstance(version, int):
        raise ValueError(
            f"schema_version must be an integer, got {version!r}"
        )
    elif version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"schema_version {version} is not supported by this configuration "
            f"loader: accepted versions are {SUPPORTED_SCHEMA_VERSIONS} plus "
            "versionless legacy input"
        )

    for key in cfg:
        if key not in KNOWN_TOP_LEVEL_KEYS and key != "output":
            notices.append(
                f"unknown top-level key '{key}' retained; it is not consumed by "
                "the current pipeline"
            )

    sensor = cfg.get("sensor_info")
    if isinstance(sensor, Mapping):
        data_format = str(sensor.get("data_format", "")).lower()
        endian = str(sensor.get("endian_type", "")).lower()
        if (
            data_format in ("uint32", "u32", "raw32")
            and endian in ("ieee-be", "big", "be")
            and "storage_format" not in sensor
        ):
            notices.append(
                "sensor_info.data_format uint32 with big-endian storage: the "
                "legacy heuristic could reinterpret low-word-first pairs; the "
                "new decoder reads standard 32-bit samples in the declared byte order — "
                "set storage_format: word_pair32 explicitly when the file is "
                "word-pair packed"
            )

    aces = cfg.get("aces")
    if isinstance(aces, Mapping):
        for key in ("variant", "output_transform", "surround", "white_point"):
            if key in aces:
                notices.append(
                    f"aces.{key} is present but not used by the current float "
                    "ACES implementation"
                )
        variant = aces.get("variant")
        if isinstance(variant, str) and variant.lower() in ("full", "canonical"):
            notices.append(
                f"aces.variant '{variant}' selects an ACES variant that is not "
                "implemented; only the fitted curve exists"
            )

    for section in LEGACY_OPERATOR_SECTIONS:
        body = cfg.get(section)
        if isinstance(body, Mapping) and "dark_enh" in body:
            notices.append(
                f"{section}.dark_enh is present but not used by the current "
                "integer tone-mapping implementation"
            )

    if "tone_mapping" in cfg:
        from modules.tone_mapping.registry import ToneMappingError, resolve_tone_mapping

        try:
            resolved = resolve_tone_mapping(cfg)
        except (ToneMappingError, TypeError):
            pass
        else:
            notices.extend(notice for notice in resolved.notices if notice not in notices)
    return cfg, notices


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool) or isinstance(value, (str, bytes)):
        return None
    if not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _section(config: Mapping[str, Any], key: str, issues: list[ValidationIssue]) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        return {}
    return value


def validate_black_level_ownership(config: Mapping[str, Any]) -> list[ValidationIssue]:
    """Reject black-level operations that overlap across BLC and PWC."""
    blc = config.get("black_level_correction")
    pwc = config.get("companding")
    if not isinstance(blc, Mapping) or not isinstance(pwc, Mapping):
        return []
    if blc.get("is_enable") is not True or pwc.get("is_enable") is not True:
        return []

    issues = []
    if blc.get("is_linear") is True:
        issues.append(ValidationIssue(
            "black_level_correction.is_linear",
            "disable BLC rescaling before PWC; its knots use input sensor codes",
        ))
    pre = pwc["pre_linearization_black_level"] if "pre_linearization_black_level" in pwc else pwc.get("pedestal", 0)
    if pre != 0:
        issues.append(ValidationIssue(
            "companding.pre_linearization_black_level",
            "BLC and PWC cannot both subtract companded-domain black; set the PWC pre level (or pedestal) to 0",
        ))
    if pwc.get("post_linearization_black_level", 0) != 0:
        issues.append(ValidationIssue(
            "companding.post_linearization_black_level",
            "BLC and PWC black subtraction cannot both be enabled; set the PWC post level to 0",
        ))
    return issues


def validate_config(
    config: Mapping[str, Any],
    *,
    require_raw: bool = False,
    has_raw: bool = False,
) -> list[ValidationIssue]:
    """Shared numeric/geometry validation for GUI and CLI use.

    Returns issues in deterministic order; each issue carries a ``severity``
    (``"error"`` blocks processing, ``"warning"`` is advisory only). Malformed
    sections produce issues rather than raising ``AttributeError``.
    """
    issues: list[ValidationIssue] = []
    if not isinstance(config, Mapping):
        return [ValidationIssue("config", "config must be a mapping")]
    issues.extend(validate_black_level_ownership(config))

    version = config.get("schema_version")
    if version == 2:
        from util.output_contract import OutputSpec
        try:
            OutputSpec.from_mapping(config.get("output"))
        except (ValueError, TypeError) as exc:
            issues.append(ValidationIssue("output", str(exc)))
        for name in ("gamma_correction", "color_space_conversion", "rgb_conversion", "yuv_conversion_format"):
            section = config.get(name)
            if isinstance(section, Mapping) and section.get("is_enable", False):
                issues.append(ValidationIssue(name, "active legacy color sections are not supported in schema_version 2"))
    required = (
        NATIVE_REQUIRED_CONFIG_KEYS
        if isinstance(version, int) and not isinstance(version, bool) and version == 2
        else REQUIRED_CONFIG_KEYS
    )
    for key in required:
        if key in config and not isinstance(config[key], Mapping):
            issues.append(
                ValidationIssue(key, f"section '{key}' must be a mapping")
            )

    sensor = _section(config, "sensor_info", issues)
    crop = _section(config, "crop", issues)
    wb = _section(config, "white_balance", issues)
    dg = _section(config, "digital_gain", issues)

    width = _as_number(sensor.get("width"))
    height = _as_number(sensor.get("height"))
    if width is None or not float(width).is_integer() or width <= 0:
        issues.append(ValidationIssue("sensor_info.width", "width must be a positive integer"))
    if height is None or not float(height).is_integer() or height <= 0:
        issues.append(ValidationIssue("sensor_info.height", "height must be a positive integer"))

    active_height = height
    if height is not None and active_height is not None:
        for embedded_key in ("embedded_rows_top", "embedded_rows_bottom"):
            embedded = _as_number(sensor.get(embedded_key))
            if embedded is not None and embedded > 0 and float(embedded).is_integer():
                active_height = active_height - embedded
    if (
        active_height is not None
        and active_height <= 0
        and height is not None
        and height > 0
    ):
        issues.append(
            ValidationIssue(
                "sensor_info.embedded_rows_top",
                "embedded row counts leave no active frame rows",
            )
        )

    if bool(crop.get("is_enable")) and width and active_height and width > 0 and active_height > 0:
        origins_ok = True
        x0 = 0.0
        y0 = 0.0
        for key, target in (("crop_x_start", "x0"), ("crop_y_start", "y0")):
            if key not in crop:
                continue
            value = _as_number(crop.get(key))
            if value is None or not float(value).is_integer():
                issues.append(
                    ValidationIssue(
                        f"crop.{key}",
                        "explicitly supplied crop origin must be an integer pixel offset",
                    )
                )
                origins_ok = False
            elif target == "x0":
                x0 = value
            else:
                y0 = value
        if origins_ok and (int(x0) % 2 or int(y0) % 2):
            issues.append(
                ValidationIssue(
                    "crop",
                    "odd crop origin is applied at the exact requested "
                    "coordinates with the effective CFA updated by the phase "
                    "shift; the legacy engine previously snapped odd origins "
                    "down to even",
                    severity="warning",
                )
            )
        nw = _as_number(crop.get("new_width"))
        nh = _as_number(crop.get("new_height"))
        if (
            nw is None
            or nh is None
            or not float(nw).is_integer()
            or not float(nh).is_integer()
        ):
            issues.append(
                ValidationIssue(
                    "crop.new_width",
                    "crop region size must be a positive integer",
                )
            )
        elif nw <= 0 or nh <= 0:
            issues.append(ValidationIssue("crop.new_width", "crop region must have positive size"))
        elif origins_ok and (x0 < 0 or y0 < 0 or (x0 + nw) > width or (y0 + nh) > active_height):
            issues.append(ValidationIssue("crop", "crop region must lie inside the sensor frame"))

    pattern = str(sensor.get("bayer_pattern", "")).lower()
    if pattern not in BAYER_PATTERN_CHOICES:
        issues.append(
            ValidationIssue(
                "sensor_info.bayer_pattern",
                f"bayer_pattern must be one of {', '.join(BAYER_PATTERN_CHOICES)}",
            )
        )

    bit_depth = _as_number(sensor.get("bit_depth"))
    if (
        bit_depth is None
        or not float(bit_depth).is_integer()
        or bit_depth < 1
        or bit_depth > 32
    ):
        issues.append(
            ValidationIssue(
                "sensor_info.bit_depth", "bit_depth must be a positive integer ≤ 32"
            )
        )

    endian = str(sensor.get("endian_type", "")).lower()
    if endian not in VALID_ENDIAN_TYPES:
        issues.append(
            ValidationIssue(
                "sensor_info.endian_type",
                "endian_type must be ieee-le or ieee-be",
            )
        )

    if isinstance(sensor, Mapping) and sensor:
        try:
            RawFormat.from_sensor_info(sensor)
        except RawFormatError as exc:
            path = f"sensor_info.{exc.field}"
            if exc.field == "sensor_info" or exc.field.startswith("sensor_info"):
                path = str(exc.field)
            reported = {
                i.path for i in issues if i.severity == "error"
            }
            if path not in reported:
                issues.append(ValidationIssue(path, str(exc)))
        except Exception as exc:
            issues.append(
                ValidationIssue("sensor_info", f"invalid RAW input contract: {exc}")
            )

    r_gain = _as_number(wb.get("r_gain"))
    b_gain = _as_number(wb.get("b_gain"))
    if r_gain is None or r_gain < 0 or (version != 2 and r_gain == 0):
        issues.append(ValidationIssue("white_balance.r_gain", "WB r_gain must be finite and nonnegative (positive in legacy configs)"))
    if b_gain is None or b_gain < 0 or (version != 2 and b_gain == 0):
        issues.append(ValidationIssue("white_balance.b_gain", "WB b_gain must be finite and nonnegative (positive in legacy configs)"))

    gain_raw = dg.get("gain_array")
    if gain_raw is None:
        gain_array: list[Any] = []
    elif isinstance(gain_raw, list):
        gain_array = gain_raw
        gains = [_as_number(g) for g in gain_array]
        if any(g is None or g <= 0 for g in gains):
            issues.append(
                ValidationIssue(
                    "digital_gain.gain_array",
                    "every digital gain must be a positive finite number",
                )
            )
        else:
            ordered = [float(g) for g in gains if g is not None]
            if any(b <= a for a, b in zip(ordered, ordered[1:])):
                issues.append(
                    ValidationIssue(
                        "digital_gain.gain_array",
                        "digital gain_array must be strictly increasing",
                    )
                )
    else:
        gain_array = []
        issues.append(
            ValidationIssue("digital_gain.gain_array", "gain_array must be a list")
        )
    idx = _as_number(dg.get("current_gain"))
    if idx is None or not float(idx).is_integer():
        issues.append(
            ValidationIssue(
                "digital_gain.current_gain",
                "current_gain is required and must be an integer index",
            )
        )
    else:
        i = int(idx)
        if not gain_array or i < 0 or i >= len(gain_array):
            issues.append(
                ValidationIssue(
                    "digital_gain.current_gain",
                    "current_gain index is out of range for gain_array",
                )
            )
        else:
            gval = _as_number(gain_array[i])
            if gval is None or gval <= 0:
                issues.append(
                    ValidationIssue(
                        "digital_gain.gain_array", "digital gain must be > 0"
                    )
                )

    for key, lo, hi in (
        ("hdr_bit_depth", 1, 32),
        ("pipeline_rgb_bit_depth", 1, 16),
    ):
        if key in sensor:
            value = _as_number(sensor.get(key))
            if value is None or not float(value).is_integer() or not lo <= value <= hi:
                issues.append(
                    ValidationIssue(
                        f"sensor_info.{key}",
                        f"{key} must be an integer in {lo}..{hi}",
                    )
                )

    if require_raw and not has_raw:
        issues.append(ValidationIssue("input.raw", "Load a test image before Process"))

    try:
        from modules.tone_mapping.registry import resolve_tone_mapping

        resolve_tone_mapping(config)
    except Exception as exc:
        issues.append(
            ValidationIssue("tone_mapping", f"tone_mapping resolution failed: {exc}")
        )

    return issues
