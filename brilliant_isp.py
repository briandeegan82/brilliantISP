"""
File: isp_pipeline.py
Description: Executes the complete pipeline
Code / Paper  Reference:
Author: Brian Deegan (based in part on 10xEngineers / Infinite-ISP)
------------------------------------------------------------
"""

import time
from pathlib import Path
from typing import Any, Callable, Mapping, cast
import numpy as np

try:
    import rawpy
except ImportError:  # optional; Bayer .raw / TIFF paths do not need it
    rawpy = None

try:
    import tifffile as tiff
except ImportError:
    tiff = None
import os

import util.utils as util
import copy
import dataclasses

from util.config_merge import (
    ConfigPathArg,
    format_config_source,
    load_merged_yaml,
    normalize_config_paths,
    pipeline_config_paths,
)
from util.config_schema import NATIVE_REQUIRED_CONFIG_KEYS, REQUIRED_CONFIG_KEYS, normalize_config, validate_black_level_ownership
from modules.tone_mapping.registry import ResolvedToneMapping

from util.pipeline_types import (
    FrameMetadata,
    ProcessingCancelled,
    ProcessingOptions,
    ProcessingState,
    StageEvent,
)
from util.bayer_repair import green_zero_fraction, repair_zero_green_cfa
from util import metering_roi
from util.raw_io import (
    alignment_from_sensor_info,
    apply_data_alignment,
    apply_raw_flips,
    flips_from_sensor_info,
    read_raw_payload,
)
from util.raw_decode import (
    RawFormat,
    decode_raw,
    decode_raw_bytes,
    resolve_source_format,
)
from util.debug_utils import DebugLogger
from util.debug_utils import get_debug_logger, set_global_debug_enabled
from util.histogram_utils import plot_histogram_comparison, estimate_dynamic_range
from util.isp_types import (
    AWBGains,
    AutoExposureConfig,
    BayerPattern,
    BayerNoiseReductionConfig,
    BlackLevelCorrectionConfig,
    ByteOrder,
    ColorCorrectionMatrixConfig,
    ColorSaturationEnhancementConfig,
    ColorSpaceConversionConfig,
    CropConfig,
    DeadPixelCorrectionConfig,
    DemosaicConfig,
    DigitalGainConfig,
    DigitalGainConfigOptional,
    ExtractedRawMetadata,
    GammaCorrectionConfig,
    GenericConfig,
    LDCIConfig,
    LensShadingCorrectionConfig,
    NoiseReduction2DConfig,
    ParsedFileNameInfo,
    PipelineConfig,
    PlatformConfig,
    RawBayerImage,
    RGBConversionConfig,
    ScaleConfig,
    SensorInfo,
    SharpenConfig,
    ToneMappingContext,
    ToneMappingConfig,
    ToneMappingParams,
    WhiteBalanceConfig,
    YUVConversionFormatConfig,
)

# Module imports
from modules.crop.crop import Crop
from modules.dead_pixel_correction.dead_pixel_correction import DeadPixelCorrection as DPC
from modules.black_level_correction.black_level_correction import BlackLevelCorrection as BLC
from modules.pwc_generation.pwc_generation import PiecewiseCurve as PWC
from modules.oecf.oecf import OECF
from modules.digital_gain.digital_gain import DigitalGain as DG
from modules.lens_shading_correction.lens_shading_correction import LensShadingCorrection as LSC
from modules.bayer_noise_reduction.bayer_noise_reduction import BayerNoiseReduction as BNR
from modules.auto_white_balance.auto_white_balance import AutoWhiteBalance as AWB
from modules.auto_white_balance import temporal as awb_temporal
from modules.auto_exposure import smooth_control as smooth_ae
from modules.white_balance.white_balance_optimized import WhiteBalanceOptimized as WBOPT
from modules.tone_mapping.tone_mapping import ToneMapping as tone_mapping
from modules.demosaic.demosaic import Demosaic
from modules.color_correction_matrix.color_correction_matrix_optimized import ColorCorrectionMatrixOptimized as CCMOPT
from modules.gamma_correction.gamma_correction import GammaCorrection as GC
from modules.auto_exposure.auto_exposure import AutoExposure as AE
from modules.color_space_conversion.color_space_conversion import ColorSpaceConversion as CSC
from modules.ldci.ldci import LDCI
from modules.ldci.ldci_rgb import apply_ldci_rgb8
from modules.sharpen.sharpen import Sharpening as SHARP
from modules.noise_reduction_2d.noise_reduction_2d import NoiseReduction2d as NR2D
from modules.rgb_conversion.rgb_conversion import RGBConversion as RGBC
from modules.scale.scale import Scale
from modules.yuv_conv_format.yuv_conv_format import YUVConvFormat as YUV_C

# HDR Image Reading Functions


def read_hdr_3byte(
    file_path: str,
    width: int,
    height: int,
    byte_order: ByteOrder = "little",
    data: bytes | None = None,
) -> np.ndarray | None:
    """
    Read HDR image using 3 consecutive bytes per pixel (24-bit packed).
    Little: LSB first (b0 | b1<<8 | b2<<16). Big: MSB first (b0<<16 | b1<<8 | b2).
    """
    if data is None:
        with open(file_path, "rb") as f:
            data = f.read()
    expected_size = width * height * 3
    actual_size = len(data)
    if actual_size < expected_size:
        return None
    fmt = RawFormat(
        width=width,
        height=height,
        bayer_pattern="rggb",
        bit_depth=24,
        storage_format="packed24",
        byte_order=byte_order,
    )
    return decode_raw_bytes(data[:expected_size], fmt).pixels


def read_hdr_uint16(
    file_path: str,
    width: int,
    height: int,
    byte_order: ByteOrder = "little",
    data: bytes | None = None,
) -> np.ndarray | None:
    """
    Read HDR image using uint16 pairs as uint32 pixels (low word | high word << 16).
    """
    if data is None:
        with open(file_path, "rb") as f:
            data = f.read()
    expected_size = width * height * 4
    actual_size = len(data)
    if actual_size < expected_size:
        return None
    fmt = RawFormat(
        width=width,
        height=height,
        bayer_pattern="rggb",
        bit_depth=32,
        storage_format="word_pair32",
        byte_order=byte_order,
        word_order="low_first",
    )
    return decode_raw_bytes(data[:expected_size], fmt).pixels


def analyze_file_size(file_path: str, logger=None) -> tuple[int, int]:
    """Analyze file size to suggest possible dimensions"""
    import logging

    log = logger or logging.getLogger("BrilliantISP.RawLoader")
    file_size = Path(file_path).stat().st_size
    log.debug(f"File size: {file_size:,} bytes")
    pixels_3byte = file_size // 3
    log.debug(f"Pixels (3-byte method): {pixels_3byte:,}")
    pixels_uint16 = file_size // 4
    log.debug(f"Pixels (uint16 method): {pixels_uint16:,}")
    return pixels_3byte, pixels_uint16


def infer_uint16_bayer_shape(file_size: int, config_width: int, config_height: int) -> tuple[int, int]:
    """
    Infer (width, height) for a uint16 Bayer buffer when file size != config W*H*2
    (e.g. bin2raw with --trim-top-rows / --image-offset-bytes).

    Picks a factorization W*H = file_size/2 that minimizes |W-cw| + |H-ch| with
    W,H in a sane range.
    """
    if file_size % 2 != 0:
        raise ValueError("file size must be even")
    pixels = file_size // 2
    if pixels <= 0:
        raise ValueError("empty file")
    if pixels == config_width * config_height:
        return config_width, config_height
    best_w, best_h = -1, -1
    best_score = float("inf")
    # Prefer factorizations near configured dimensions (bin2raw trims change W/H slightly)
    for h in range(max(64, config_height - 400), config_height + 400):
        if h <= 0 or pixels % h != 0:
            continue
        w = pixels // h
        if w < 64 or w > 16384:
            continue
        score = abs(w - config_width) + abs(h - config_height)
        if score < best_score:
            best_score = score
            best_w, best_h = w, h
    if best_w < 0:
        raise ValueError(f"no WxH with W*H={pixels} near {config_width}x{config_height}")
    return best_w, best_h


class BrilliantISP:
    """
    Brilliant-ISP Pipeline
    """

    def __init__(
        self,
        data_path: str,
        config_path: ConfigPathArg | Mapping[str, Any],
        outFileName: str,
        output_path: str | None = None,
        *,
        options: ProcessingOptions | None = None,
    ) -> None:
        """
        Constructor: Initialize with config and raw file path
        and Load configuration parameter from yaml file

        ``config_path`` may be YAML path(s) or an already-loaded mapping.
        ``options=None`` preserves the legacy save/diagnostic behavior;
        explicit ``ProcessingOptions`` controls saves, diagnostics and
        captures per request (compute-only by default).
        """
        self.options = options
        if (
            options is not None
            and (options.save_outputs or options.diagnostics)
            and options.output_dir is None
        ):
            raise ValueError(
                "ProcessingOptions with save_outputs or diagnostics requires output_dir"
            )
        self.data_path = data_path
        if options is not None and options.output_dir is not None:
            self.output_path = str(options.output_dir)
        else:
            self.output_path = output_path if output_path else "out_frames/"
        self.outFileName = outFileName
        self.logger: DebugLogger
        self.platform: PlatformConfig | None = None
        self.config_notices: list[str] = []
        self.sensor_info: SensorInfo | None = None
        self.c_yaml: PipelineConfig | None = None
        self.raw: np.ndarray | None = None
        self.decompanded_img: np.ndarray | None = None
        self.last_output_rgb: np.ndarray | None = None
        self.last_output_metadata: FrameMetadata | None = None
        self.last_feedback: dict[str, Any] = {}
        self.last_awb_status = "disabled"
        self.last_awb_count = 0
        self.last_awb_input_gain: float = 1.0
        self.processing_state: ProcessingState | None = None
        self.last_artifacts: tuple[Path, ...] = ()
        self.migration_report: dict[str, Any] = {}
        self.last_output_native: np.ndarray | None = None
        self.last_output_geometry: tuple[int, int] | None = None
        self.stage_events: list[StageEvent] = []
        self.captures: dict[str, np.ndarray] = {}
        self.pass_index: int = 0
        self.current_frame: np.ndarray | None = None
        self.awb_gains: AWBGains = (1.0, 1.0)
        self.ae_feedback: int | None = None
        self.dga_current_gain: int = 0
        self.param_durand: ToneMappingParams = {}
        self.param_aces: ToneMappingParams = {}
        self.param_integer_tmo: ToneMappingParams = {}
        self.param_aces_integer: ToneMappingParams = {}
        self.param_hable: ToneMappingParams = {}
        self.param_hable_integer: ToneMappingParams = {}
        # Ensure output directory exists
        if options is None or options.save_outputs or options.diagnostics:
            os.makedirs(self.output_path, exist_ok=True)
        self.load_config(config_path)
        assert self.platform is not None
        # Set global debug state from config
        set_global_debug_enabled(self.platform.get("debug_enabled", False))
        # Initialize debug logger after config is loaded
        self.logger = get_debug_logger("BrilliantISP", config=self.platform)

    _REQUIRED_CONFIG_KEYS = REQUIRED_CONFIG_KEYS

    def load_config(self, config_path: ConfigPathArg | Mapping[str, Any]) -> None:
        """
        Load config information to respective module parameters.
        Validates required keys and uses defaults for optional sections.

        ``config_path`` may be a single YAML file, a list/tuple of paths merged
        depth-first (later files override earlier ones for overlapping keys),
        or an already-loaded mapping (normalized and deep-copied; the caller's
        dict is never mutated or aliased).

        A single path whose name ends in ``_cam.yml`` is merged after
        ``config/base_hdr.yml`` automatically (see ``pipeline_config_paths``).
        """
        if isinstance(config_path, Mapping):
            self.config_path = "<mapping>"
            normalized, self.config_notices = normalize_config(config_path)
        else:
            if isinstance(config_path, (str, Path)):
                paths = pipeline_config_paths(config_path)
            else:
                paths = normalize_config_paths(config_path)
            self.config_path = format_config_source(paths)
            normalized, self.config_notices = normalize_config(load_merged_yaml(paths))
        # Snapshot the resolved source BEFORE diagnostics suppression so the
        # migration report keeps the requested legacy flags.
        self.source_config_original = copy.deepcopy(normalized)
        from util.pipeline_migration import migrate_pipeline_config
        migrated, self.migration_report = migrate_pipeline_config(
            normalized, canonicalize_tone_mapping=False
        )
        if self.options is not None and not self.options.diagnostics:
            self._suppress_diagnostic_flags(migrated)
        normalized, extra = normalize_config(migrated)
        for n in list(self.migration_report.get("notices", ())) + list(extra):
            if n not in self.config_notices:
                self.config_notices.append(n)
        black_level_issues = validate_black_level_ownership(normalized)
        if black_level_issues:
            raise ValueError(
                "config failed validation: "
                + "; ".join(f"{i.path}: {i.message}" for i in black_level_issues)
            )
        self._base_config_notices = list(self.config_notices)
        c_yaml = cast(PipelineConfig, normalized)

        required = (
            NATIVE_REQUIRED_CONFIG_KEYS
            if self.migration_report.get("target_version") == 2
            or int(cast(Any, c_yaml.get("schema_version") or 1)) == 2
            else self._REQUIRED_CONFIG_KEYS
        )
        missing = [k for k in required if k not in c_yaml]
        if missing:
            raise KeyError(
                f"Config '{self.config_path}' missing required keys: {missing}. "
                "See config/base_hdr.yml plus a *_cam.yml overlay (e.g. svs_cam.yml)."
            )

        # Extract workspace info
        self.platform = c_yaml["platform"]
        self.platform["generate_tv"] = self.platform.get("generate_tv", False)
        if self.options is not None and self.options.diagnostics:
            self.platform["output_dir"] = str(
                Path(cast(Path, self.options.output_dir)) / "module_output"
            )
            self.platform["debug_log_file"] = str(
                Path(cast(Path, self.options.output_dir)) / "pipeline.log"
            )
        elif "output_dir" not in self.platform:
            self.platform["output_dir"] = "module_output"  # Directory for module debug outputs (curves, etc)
        self.platform["final_output_dir"] = self.output_path
        self.raw_file = self.platform["filename"]
        self.render_3a = self.platform["render_3a"]
        self.sensor_info = c_yaml["sensor_info"]
        self.sensor_info.setdefault("horizontal_flip", False)
        self.sensor_info.setdefault("vertical_flip", False)

        # ISP module params
        self.parm_dpc: DeadPixelCorrectionConfig = c_yaml["dead_pixel_correction"]
        self.parm_cmpd: GenericConfig = c_yaml["companding"]
        self.parm_dga: DigitalGainConfigOptional = cast(
            DigitalGainConfigOptional, c_yaml["digital_gain"]
        )
        self.parm_lsc: LensShadingCorrectionConfig = c_yaml["lens_shading_correction"]
        self.parm_bnr: BayerNoiseReductionConfig = c_yaml["bayer_noise_reduction"]
        self.parm_blc: BlackLevelCorrectionConfig = c_yaml["black_level_correction"]
        self.parm_oec: GenericConfig = c_yaml.get("oecf", {"is_enable": False, "is_save": False})
        self.parm_wbc: WhiteBalanceConfig = c_yaml["white_balance"]
        self.parm_awb: GenericConfig = c_yaml["auto_white_balance"]
        # Single source of truth: manual vs auto follows auto_white_balance.is_enable
        # only when the (v2) white_balance.is_auto field is absent; an explicit
        # v2 field is authoritative and can mean suggestions-only AWB.
        self.parm_wbc.setdefault("is_auto", self.parm_awb["is_enable"])
        self.parm_dem: DemosaicConfig = c_yaml["demosaic"]
        self.parm_ae: AutoExposureConfig = c_yaml["auto_exposure"]
        self.parm_dga["exposure_correction_mode"] = self.parm_ae.get("exposure_correction_mode", "step")
        self.parm_ccm: ColorCorrectionMatrixConfig = c_yaml["color_correction_matrix"]
        self.parm_gmc: GammaCorrectionConfig = cast(GammaCorrectionConfig, c_yaml.get("gamma_correction", {}))
        self.param_durand = c_yaml.get("hdr_durand", {})
        self.param_aces = c_yaml.get("aces", {})
        self.parm_csc: ColorSpaceConversionConfig = cast(ColorSpaceConversionConfig, c_yaml.get("color_space_conversion", {}))
        self.parm_cse: ColorSaturationEnhancementConfig = c_yaml["color_saturation_enhancement"]
        self.parm_ldci: LDCIConfig = c_yaml["ldci"]
        self.parm_sha: SharpenConfig = c_yaml["sharpen"]
        self.parm_2dn: NoiseReduction2DConfig = c_yaml["2d_noise_reduction"]
        self.parm_rgb: RGBConversionConfig = cast(RGBConversionConfig, c_yaml.get("rgb_conversion", {}))
        self.parm_sca: ScaleConfig = c_yaml["scale"]
        self.parm_cro: CropConfig = c_yaml["crop"]
        self.parm_yuv: YUVConversionFormatConfig = cast(YUVConversionFormatConfig, c_yaml.get("yuv_conversion_format", {}))
        self.c_yaml = c_yaml
        self.platform["rgb_output"] = self.parm_rgb.get("is_enable", True)
        initial_in_file = Path(self.raw_file).stem
        self.platform["in_file"] = initial_in_file
        self.platform["out_file"] = (
            initial_in_file if self.platform.get("short_output_names", False) else "Out_" + initial_in_file
        )
        self.bit_depth = self.sensor_info["bit_depth"]
        self.tone_mapping: ToneMappingConfig = c_yaml["tone_mapping"]
        from modules.tone_mapping.registry import resolve_tone_mapping
        resolved_tm = resolve_tone_mapping(c_yaml)
        for notice in resolved_tm.notices:
            if notice not in self.config_notices:
                self.config_notices.append(notice)
        self.tone_mapping_before_demosaic = resolved_tm.placement == "pre_demosaic"
        legacy_mapper = str(self.tone_mapping.get("tone_mapper", "reinhard_integer"))
        # hdr_durand is the YAML block name; tone_mapper value must be durand.
        if legacy_mapper == "hdr_durand":
            legacy_mapper = "durand"
        self.tone_mapper = (
            str(self.tone_mapping["operator"])
            if "operator" in self.tone_mapping
            else legacy_mapper
        )
        self.param_aces = c_yaml.get("aces", {})
        # reinhard_integer section, or legacy integer_tmo section name
        integer_sections = ("integer_tmo", "reinhard_integer") if legacy_mapper == "integer_tmo" else ("reinhard_integer", "integer_tmo")
        self.param_integer_tmo = c_yaml.get(integer_sections[0], c_yaml.get(integer_sections[1], {}))
        self.param_aces_integer = c_yaml.get("aces_integer", {})
        self.param_hable = c_yaml.get("hable", {})
        self.param_hable_integer = c_yaml.get("hable_integer", {})

        # Snapshot the config values that a single frame mutates transiently, so
        # reusing one instance across a batch cannot leak state between frames.
        # Excludes intentional 3A carryover (WB gains / gain index), which is
        # threaded frame-to-frame by load_3a_statistics().
        self._frame_reset_baseline = {
            "sensor_wh": (self.sensor_info["width"], self.sensor_info["height"]),
            "bayer_pattern": self.sensor_info.get("bayer_pattern", "rggb"),
            "endian_type": self.sensor_info.get("endian_type"),
            "yuv_is_enable": self.parm_yuv.get("is_enable", False),
            "scale_is_debug": self.parm_sca.get("is_debug", False),
        }
        self._loaded_frame_geometry: tuple[int, int, str] | None = None
        self.decoded_raw = None

        # add rgb_output_conversion module

    def _reset_transient_frame_state(self) -> None:
        """
        Restore config that gets mutated in place while processing a frame back to
        its as-configured baseline, so each frame is processed independently when a
        BrilliantISP instance is reused across many frames.

        Covers: decoded sensor geometry and CFA phase (changed by crop),
        YUV is_enable (cleared by the YUV module when the output is RGB), and
        scale is_debug. Source byte order is restored separately at each file
        load. Intentional 3A carryover is deliberately not reset here.
        """
        baseline = getattr(self, "_frame_reset_baseline", None)
        if baseline is None or self.sensor_info is None:
            return
        loaded = getattr(self, "_loaded_frame_geometry", None)
        if loaded is not None:
            w, h, pattern = loaded
        else:
            w, h = baseline["sensor_wh"]
            pattern = baseline["bayer_pattern"]
        self.sensor_info["width"] = w
        self.sensor_info["height"] = h
        self.sensor_info["bayer_pattern"] = cast(BayerPattern, pattern)
        if self.c_yaml is not None:
            self.c_yaml["sensor_info"]["width"] = w
            self.c_yaml["sensor_info"]["height"] = h
            self.c_yaml["sensor_info"]["bayer_pattern"] = cast(BayerPattern, pattern)
        self.parm_yuv.setdefault("is_enable", baseline["yuv_is_enable"])
        self.parm_yuv["is_enable"] = baseline["yuv_is_enable"]
        self.parm_sca["is_debug"] = baseline["scale_is_debug"]
        self.captures.clear()
        self.stage_events.clear()
        self.last_artifacts = ()
        self.last_output_metadata = None
        self.last_feedback = {}
        self.last_output_native = self.last_output_rgb = None
        self.last_output_geometry = None

    @staticmethod
    def _suppress_diagnostic_flags(config: Mapping[str, Any]) -> None:
        """Disable save/plot/debug flags inside the internal config copy only.

        Applied for explicit ``ProcessingOptions`` with ``diagnostics=False`` so
        compute-only and save-only requests write no debug artifacts. Algorithm
        enable flags (``is_enable``) are never touched.
        """
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

    def _emit(self, event: StageEvent) -> None:
        callback = self.options.progress_callback if self.options else None
        if callback is not None:
            callback(event)

    def _check_cancelled(self) -> None:
        token = self.options.cancellation_token if self.options else None
        if token is not None and token.is_set():
            raise ProcessingCancelled("processing cancelled between stages")

    def _run_stage(
        self,
        stage_id: str,
        thunk: Callable[[], Any],
        ae_pass: int = 0,
    ) -> Any:
        """Run one pipeline stage with cancellation check and telemetry.

        Emits ``started``/``completed``/``failed`` events to the options
        progress callback, records completed/failed events in
        ``self.stage_events``, and stores an owned read-only copy in
        ``self.captures`` when the stage was requested via
        ``options.capture_stages``. ``elapsed_ns`` is measured with
        ``time.perf_counter_ns`` around the entire thunk, including module
        construction.
        """
        run_id = self.options.run_id if self.options else ""
        try:
            self._check_cancelled()
        except ProcessingCancelled:
            event = StageEvent(
                run_id=run_id,
                stage_id=stage_id,
                status="cancelled",
                elapsed_ns=0,
                pass_index=self.pass_index,
                ae_pass=ae_pass,
                error="cancelled",
            )
            self.stage_events.append(event)
            self._emit(event)
            raise
        self._emit(
            StageEvent(
                run_id=run_id,
                stage_id=stage_id,
                status="started",
                pass_index=self.pass_index,
                ae_pass=ae_pass,
            )
        )
        start = time.perf_counter_ns()
        try:
            result = thunk()
        except ProcessingCancelled:
            elapsed = time.perf_counter_ns() - start
            event = StageEvent(
                run_id=run_id,
                stage_id=stage_id,
                status="cancelled",
                elapsed_ns=elapsed,
                pass_index=self.pass_index,
                ae_pass=ae_pass,
                error="cancelled",
            )
            self.stage_events.append(event)
            self._emit(event)
            raise
        except Exception as exc:
            elapsed = time.perf_counter_ns() - start
            event = StageEvent(
                run_id=run_id,
                stage_id=stage_id,
                status="failed",
                elapsed_ns=elapsed,
                pass_index=self.pass_index,
                ae_pass=ae_pass,
                error=f"{type(exc).__name__}: {exc}",
            )
            self.stage_events.append(event)
            self._emit(event)
            raise
        elapsed = time.perf_counter_ns() - start
        image = result[0] if isinstance(result, tuple) and result and isinstance(result[0], np.ndarray) else result
        event = StageEvent(
            run_id=run_id,
            stage_id=stage_id,
            status="completed",
            elapsed_ns=elapsed,
            pass_index=self.pass_index,
            ae_pass=ae_pass,
            shape=tuple(image.shape) if isinstance(image, np.ndarray) else None,
            dtype=str(image.dtype) if isinstance(image, np.ndarray) else None,
        )
        self.stage_events.append(event)
        self._emit(event)
        if (
            self.options is not None
            and stage_id in self.options.capture_stages
            and isinstance(image, np.ndarray)
        ):
            captured = np.array(image, copy=True)
            captured.setflags(write=False)
            self.captures[stage_id] = captured
        return result

    def _run_pre_digital_gain(self) -> np.ndarray:
        """Crop -> DPC -> BLC -> green repair -> PWC -> OECF (shared front half)."""
        if self.raw is None or self.platform is None or self.sensor_info is None:
            raise RuntimeError("RAW image and configuration must be loaded before running the pipeline.")
        platform = self.platform
        sensor_info = self.sensor_info
        raw = self.raw
        # Set skip_disabled to avoid running disabled modules
        skip_disabled = platform.get("skip_disabled_modules", False)

        # =====================================================================
        # Cropping
        # Crop
        def _crop() -> np.ndarray:
            if skip_disabled and not self.parm_cro["is_enable"]:
                return raw
            crop = Crop(raw, platform, sensor_info, self.parm_cro)
            return crop.execute()

        cropped_img = self._run_stage("crop", _crop)

        # =====================================================================
        # Dead pixels correction
        # Dead Pixel Correction
        def _dpc() -> np.ndarray:
            if skip_disabled and not self.parm_dpc["is_enable"]:
                return cropped_img
            dpc = DPC(cropped_img, sensor_info, self.parm_dpc, platform)
            return dpc.execute()

        dpc_raw = self._run_stage("dead_pixel_correction", _dpc)

        # =====================================================================
        # Black level correction
        # Black Level Correction
        def _blc() -> np.ndarray:
            if skip_disabled and not self.parm_blc["is_enable"]:
                return dpc_raw
            blc = BLC(dpc_raw, platform, sensor_info, self.parm_blc)
            return blc.execute()

        blc_raw = self._run_stage("black_level_correction", _blc)

        # Reconstruct empty CFA greens after BLC so estimates use pedestal-corrected R/B
        blc_raw = self._run_stage(
            "green_repair", lambda: self._maybe_repair_zero_green_cfa(blc_raw)
        )

        # =====================================================================
        # decompanding
        # Decompanding
        def _pwc() -> np.ndarray:
            if skip_disabled and not self.parm_cmpd["is_enable"]:
                cmpd_raw = blc_raw.astype(np.uint32)
            else:
                cmpd = PWC(blc_raw, platform, sensor_info, self.parm_cmpd)
                cmpd_raw = cmpd.execute()
            # Store decompanded image for histogram comparison later
            if platform.get("plot_histograms", False) or (
                self.options is not None and "companding" in self.options.capture_stages
            ):
                self.decompanded_img = cmpd_raw.copy()
            return cmpd_raw

        cmpd_raw = self._run_stage("companding", _pwc)

        # =====================================================================
        # OECF
        def _oecf() -> Any:
            if skip_disabled and not self.parm_oec.get("is_enable", False):
                return cmpd_raw
            input_sensor = dict(sensor_info)
            input_sensor["hdr_bit_depth"] = (
                sensor_info.get("hdr_bit_depth", sensor_info["bit_depth"])
                if self.parm_cmpd.get("is_enable", False) else sensor_info["bit_depth"]
            )
            oecf = OECF(cmpd_raw, platform, input_sensor, self.parm_oec)
            return oecf.execute()

        return cast(np.ndarray, self._run_stage("oecf", _oecf))

    def _run_gain_through_white_balance(
        self,
        oecf_raw: np.ndarray,
        *,
        ae_pass: int = 0,
        mirror_gain: bool = True,
        gain_index: int | None = None,
    ) -> np.ndarray:
        """DG -> LSC -> BNR -> AWB -> WB (shared second half of the front half).

        ``mirror_gain`` is retained for call compatibility; automatic state
        never replaces the effective configuration's manual gain controls.
        """
        if self.platform is None or self.sensor_info is None or self.c_yaml is None:
            raise RuntimeError("Configuration must be loaded before running the pipeline.")
        platform = self.platform
        from util.signal_range import linear_code_range
        sensor_info = cast(SensorInfo, dict(self.sensor_info))
        sensor_info["hdr_bit_depth"] = linear_code_range(self.c_yaml).bit_depth

        # =====================================================================
        # Digital Gain (receives OECF output per pipeline order: PWC -> OECF -> DG)
        # Digital Gain
        from util.native_array import as_float32
        smooth_gain = getattr(self, "_smooth_gain", None)

        def _dg() -> tuple[np.ndarray, int]:
            if smooth_gain is not None:
                arr = np.asarray(self.parm_dga["gain_array"], dtype=np.float64)
                nearest = int(np.argmin(np.abs(np.log2(arr) - np.log2(smooth_gain))))
                with np.errstate(over="ignore", invalid="ignore"):
                    scaled = np.asarray(oecf_raw, dtype=np.float64) * float(smooth_gain)
                return as_float32(scaled, name="digital_gain"), nearest
            config = cast(DigitalGainConfig, dict(self.parm_dga))
            config["ae_feedback"] = None
            config["current_gain"] = int(self.parm_dga["current_gain"]) if gain_index is None else gain_index
            if gain_index is None and self.processing_state and self.parm_dga.get("is_enable", True) \
                    and self.parm_dga.get("is_auto", False) and self.processing_state.digital_gain_index is not None:
                config["current_gain"] = self.processing_state.digital_gain_index
            dga = DG(oecf_raw, platform, sensor_info, config)
            return dga.compute_linear()

        dga_raw, self.dga_current_gain = self._run_stage("digital_gain", _dg, ae_pass=ae_pass)
        # Keep automatic gain selection separate from the manual YAML controls.

        # =====================================================================
        # Lens shading correction
        # Lens Shading Correction
        def _lsc() -> np.ndarray:
            lsc = LSC(dga_raw, platform, sensor_info, self.parm_lsc)
            return lsc.compute_linear()

        lsc_raw = self._run_stage("lens_shading_correction", _lsc, ae_pass=ae_pass)

        # =====================================================================
        # Bayer noise reduction
        # Bayer Noise Reduction
        def _bnr() -> np.ndarray:
            bnr = BNR(lsc_raw, sensor_info, self.parm_bnr, platform, cpu_only=True)
            return bnr.compute_linear()

        bnr_raw = self._run_stage("bayer_noise_reduction", _bnr, ae_pass=ae_pass)

        # =====================================================================
        # Auto White Balance
        def _awb() -> Any:
            awb = AWB(bnr_raw, sensor_info, self.parm_awb, self.parm_wbc)
            hint = self.processing_state.previous_awb_gains if self.processing_state else None
            if smooth_gain is not None:
                self.last_awb_input_gain = float(smooth_gain)
            else:
                self.last_awb_input_gain = (
                    float(self.parm_dga["gain_array"][self.dga_current_gain])
                    if self.parm_dga.get("is_enable", True) else 1.0
                )
            gains = awb.compute_gains(fallback_gains=hint, input_gain=self.last_awb_input_gain)
            self.last_awb_status = awb.last_status
            self.last_awb_count = awb.sample_count
            self.last_awb_raw_estimate = (float(gains[0]), float(gains[1]))
            if awb.last_status == "estimated" and awb_temporal.is_active(self.parm_awb):
                gains = awb_temporal.filter_gains(gains, hint, self.parm_awb, input_gain=self.last_awb_input_gain)
            return gains

        self.awb_gains = cast(
            AWBGains,
            self._run_stage("auto_white_balance", _awb, ae_pass=ae_pass),
        )

        # =====================================================================
        # White balancing
        # White Balance
        # Use optimized version for better performance
        def _wb() -> np.ndarray:
            wbc = WBOPT(bnr_raw, platform, sensor_info, self.parm_wbc, self.awb_gains)
            return wbc.compute_linear()

        wb_raw = self._run_stage("white_balance", _wb, ae_pass=ae_pass)

        # Store current frame for external processing
        self.current_frame = wb_raw
        return cast(np.ndarray, wb_raw)

    def load_raw(self, byte_order: ByteOrder | None = None) -> None:
        """Load the configured RAW input through the declared-format decoder.

        Uses ``RawFormat.from_sensor_info`` + ``util.raw_decode.decode_raw``;
        there is no size-based heuristic guessing. ``byte_order`` overrides the
        configured endian only for explicit/raw-buffer input — container files
        (PGM/TIFF/rawpy) keep their authoritative byte order.
        """
        if self.platform is None or self.sensor_info is None:
            raise RuntimeError("Configuration must be loaded before loading RAW input.")
        sensor = self.sensor_info
        # Clear any per-frame state left by a previous frame before deriving this one.
        self._loaded_frame_geometry = None
        self.decoded_raw = None
        self.raw = None
        self.current_frame = None
        self.decompanded_img = None
        self.ae_feedback = None
        self.dga_current_gain = 0
        self.last_output_rgb = None
        self.last_output_native = None
        self.last_output_geometry = None
        self.last_output_metadata = None
        self.last_feedback = {}
        self.last_artifacts = ()
        self.captures.clear()
        self.stage_events.clear()
        self.pass_index = 0
        self.config_notices = list(getattr(self, "_base_config_notices", ()))
        self.awb_gains = (1.0, 1.0)
        self._check_cancelled()
        self._reset_transient_frame_state()
        baseline_endian = getattr(self, "_frame_reset_baseline", {}).get("endian_type")
        if baseline_endian is not None:
            sensor["endian_type"] = baseline_endian
            if self.c_yaml is not None:
                self.c_yaml["sensor_info"]["endian_type"] = baseline_endian
        elif "endian_type" in sensor:
            del sensor["endian_type"]
            if self.c_yaml is not None:
                self.c_yaml["sensor_info"].pop("endian_type", None)
        # Load raw image file information
        path_object = Path(self.data_path, self.raw_file)
        raw_path = str(path_object.resolve())
        self.in_file = path_object.stem
        short_names = self.platform.get("short_output_names", False)
        self.out_file = self.in_file if short_names else "Out_" + self.in_file

        self.platform["in_file"] = self.in_file
        self.platform["out_file"] = self.out_file

        fmt = RawFormat.from_sensor_info(sensor)
        resolved = resolve_source_format(path_object, fmt.source_format)
        if byte_order is not None and resolved == "raw":
            fmt = dataclasses.replace(fmt, byte_order=byte_order)
            sensor["endian_type"] = (
                "ieee-le" if fmt.byte_order == "little" else "ieee-be"
            )
        self.logger.info(f"Loading raw file: {raw_path}")
        self.logger.info(
            f"Expected dimensions: {fmt.width}x{fmt.height} "
            f"({fmt.storage_format}, {fmt.byte_order})"
        )
        decoded = decode_raw(raw_path, fmt)
        self.decoded_raw = decoded
        for warning in decoded.warnings:
            self.logger.warning(warning)
            if warning not in self.config_notices:
                self.config_notices.append(warning)
        self.raw = decoded.pixels
        self._loaded_frame_geometry = (
            decoded.width,
            decoded.height,
            decoded.bayer_pattern,
        )
        sensor["width"] = decoded.width
        sensor["height"] = decoded.height
        sensor["bayer_pattern"] = cast(BayerPattern, decoded.bayer_pattern)
        if self.c_yaml is not None:
            self.c_yaml["sensor_info"]["width"] = decoded.width
            self.c_yaml["sensor_info"]["height"] = decoded.height
            self.c_yaml["sensor_info"]["bayer_pattern"] = cast(BayerPattern, decoded.bayer_pattern)
        self.logger.info(
            f"Loaded {decoded.container} frame {decoded.width}x{decoded.height} "
            f"({decoded.bayer_pattern}, {decoded.storage_bits}-bit storage)"
        )

    def _load_raw_legacy(self, byte_order: ByteOrder = "little") -> None:
        """
        DEPRECATED file-size heuristic loader retained only for migration audit.

        NOT called by ``load_raw`` or any processing path. Kept to compare
        historical behavior during the staged RAW-decoder transition.
        """
        if self.platform is None or self.sensor_info is None:
            raise RuntimeError("Configuration must be loaded before loading RAW input.")
        platform = self.platform
        sensor = self.sensor_info
        # Load raw image file information
        path_object = Path(self.data_path, self.raw_file)
        raw_path = str(path_object.resolve())
        self.in_file = path_object.stem
        short_names = platform.get("short_output_names", False)
        self.out_file = self.in_file if short_names else "Out_" + self.in_file

        platform["in_file"] = self.in_file
        platform["out_file"] = self.out_file

        width = sensor["width"]
        height = sensor["height"]
        bit_depth = sensor["bit_depth"]
        suffix = path_object.suffix.lower()
        payload = read_raw_payload(path_object)
        raw_bytes = payload.data
        file_size = len(raw_bytes)
        if payload.is_pgm:
            self.logger.info(
                f"Stripped PGM header ({payload.header_bytes} bytes); "
                f"header size {payload.pgm_width}x{payload.pgm_height} maxval={payload.pgm_maxval}"
            )
            if payload.pgm_width and payload.pgm_height:
                if payload.pgm_width != width or payload.pgm_height != height:
                    self.logger.warning(
                        f"PGM header {payload.pgm_width}x{payload.pgm_height} differs from "
                        f"sensor_info {width}x{height}; using PGM dimensions."
                    )
                    width = payload.pgm_width
                    height = payload.pgm_height
                    sensor["width"] = width
                    sensor["height"] = height

        # Load Raw with enhanced HDR support (.raw and header-stripped .pgm)
        if suffix in {".raw", ".pgm"} or payload.is_pgm:
            expected_size_3byte = width * height * 3
            expected_size_uint16 = width * height * 4

            self.logger.info(f"Loading raw file: {raw_path}")
            self.logger.info(f"Expected dimensions: {width}x{height}")
            self.logger.info(f"File size: {file_size:,} bytes")
            self.logger.info(f"Expected size (3-byte method): {expected_size_3byte:,} bytes")
            self.logger.info(f"Expected size (uint16 method): {expected_size_uint16:,} bytes")

            # Try different loading methods based on file size and bit depth
            if bit_depth > 8:
                if abs(file_size - expected_size_3byte) < expected_size_3byte * 0.1:
                    self.logger.info(f"Trying 3-byte HDR method ({byte_order} endian)...")
                    self.raw = read_hdr_3byte(raw_path, width, height, byte_order, data=raw_bytes)
                    if self.raw is None:
                        raise RuntimeError(
                            f"Raw file too small for 3-byte HDR: expected {expected_size_3byte} bytes, " f"got {file_size}"
                        )
                    self.logger.info(f"Successfully loaded using 3-byte HDR method ({byte_order} endian)")
                    self._apply_sensor_flips()
                    return

                if abs(file_size - expected_size_uint16) < expected_size_uint16 * 0.1:
                    self.logger.info(f"Trying uint16 HDR method ({byte_order} endian)...")
                    self.raw = read_hdr_uint16(raw_path, width, height, byte_order, data=raw_bytes)
                    if self.raw is None:
                        raise RuntimeError(
                            f"Raw file too small for uint16 HDR: expected {expected_size_uint16} bytes, " f"got {file_size}"
                        )
                    self.logger.info(f"Successfully loaded using uint16 HDR method ({byte_order} endian)")
                    self._apply_sensor_flips()
                    return

                self.logger.info("Falling back to 2-byte uint16 method...")
                expected_2byte = width * height * 2
                load_w, load_h = width, height
                if file_size != expected_2byte:
                    try:
                        load_w, load_h = infer_uint16_bayer_shape(file_size, width, height)
                    except ValueError as e:
                        raise RuntimeError(
                            f"Raw size {file_size} bytes does not match config {width}x{height} "
                            f"({expected_2byte} B expected) and could not infer dimensions: {e}. "
                            f"Set sensor_info width/height to match bin2raw extraction_meta.txt "
                            f"(e.g. after --trim-top-rows / --image-offset-bytes)."
                        ) from e
                    if load_w != width or load_h != height:
                        self.logger.warning(
                            f"Inferred raw shape {load_w}x{load_h} from file size "
                            f"(config was {width}x{height}). Update sensor_info in your YAML."
                        )
                        sensor["width"] = load_w
                        sensor["height"] = load_h
                need_bytes = load_w * load_h * 2
                if file_size < need_bytes:
                    raise RuntimeError(f"Raw file too small: need {need_bytes} bytes for {load_w}x{load_h}, got {file_size}")
                if file_size > need_bytes:
                    self.logger.warning(
                        f"Raw file is {file_size - need_bytes} bytes longer than {load_w}x{load_h}x2; "
                        f"truncating to {need_bytes} bytes"
                    )
                et = str(sensor.get("endian_type", "")).lower()
                if "le" in et or "little" in et:
                    raw_dtype = "<u2"
                else:
                    # ieee-be / big / omitted (legacy default was always big-endian)
                    raw_dtype = ">u2"
                count = load_w * load_h
                self.raw = np.frombuffer(raw_bytes, dtype=raw_dtype, count=count).reshape((load_h, load_w)).copy()
            else:
                # For 8-bit or lower, use original method
                self.raw = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((height, width)).astype(np.uint16)
        elif suffix in {".tiff", ".tif"}:
            if tiff is None:
                raise ImportError("tifffile is required to load TIFF files. Install it with: pip install tifffile")
            # Load tiff file
            img = tiff.imread(raw_path)
            self.logger.info(f"Image shape: {img.shape}")
            if img.ndim == 3:
                self.raw = img[:, :, 0]
            else:
                self.raw = img
        else:
            if rawpy is None:
                raise ImportError(
                    "rawpy is required to load this image format "
                    f"({path_object.suffix or 'unknown'}). Install it with: pip install rawpy"
                )
            img = rawpy.imread(raw_path)
            self.raw = img.raw_image
        self._apply_sensor_flips()

    def _maybe_repair_zero_green_cfa(self, raw: np.ndarray) -> np.ndarray:
        """
        Optionally reconstruct empty CFA green sites from neighboring R/B.

        Controlled by sensor_info.repair_zero_green:
          - true: always repair
          - false: never repair
          - auto / omitted: repair when >50% of green sites are <= 0

        Prefer calling this after black-level correction so G is estimated from
        pedestal-corrected R/B (reduces magenta/green zipper on broken dumps).
        """
        if self.sensor_info is None:
            return raw
        mode = self.sensor_info.get("repair_zero_green", "auto")
        if mode is False or mode == "false":
            return raw
        pattern = str(self.sensor_info.get("bayer_pattern", "rggb")).strip().lower()
        if int(np.max(raw)) == 0:
            return raw
        frac = green_zero_fraction(raw, pattern)  # type: ignore[arg-type]
        should = mode is True or mode == "true" or (
            str(mode).lower() in ("auto", "") and frac > 0.5
        )
        if not should:
            if frac > 0.05:
                self.logger.info(
                    f"Green CFA zero fraction={frac:.1%} (below repair threshold)"
                )
            return raw
        self.logger.warning(
            f"Repairing empty CFA greens (zero fraction={frac:.1%}); "
            "true spectral green is missing in this RAW — colors will be approximate."
        )
        repaired, filled = repair_zero_green_cfa(
            raw, pattern  # type: ignore[arg-type]
        )
        self.logger.info(f"Filled {filled:.1%} of CFA green sites from R/B neighbors")
        return repaired

    def _apply_sensor_flips(self) -> None:
        if self.raw is None or self.sensor_info is None:
            return
        alignment = alignment_from_sensor_info(self.sensor_info)
        bit_depth = int(self.sensor_info.get("bit_depth", 12))
        container_bits = int(self.raw.dtype.itemsize) * 8
        manual_shift = self.sensor_info.get("manual_bit_shift")
        manual_shift = int(manual_shift) if manual_shift is not None else None
        aligned = apply_data_alignment(self.raw, bit_depth, alignment, manual_shift)
        if aligned is not self.raw:
            if alignment.upper() == "MANUAL" and manual_shift is not None:
                shift = manual_shift
            else:
                shift = max(0, container_bits - bit_depth)
            self.raw = aligned
            self.logger.info(
                f"Applied data_alignment={alignment} " f"(>>{shift} for {bit_depth}-bit in {container_bits}-bit container)"
            )
        hflip, vflip = flips_from_sensor_info(self.sensor_info)
        if hflip or vflip:
            self.raw = apply_raw_flips(self.raw, hflip, vflip)
            self.logger.info(f"Applied RAW flips horizontal={hflip} vertical={vflip}")

    def run_pipeline_up_to_wb(self) -> None:
        """
        Execute the pipeline up to and including white balance.
        Used for profiling demosaic and tone mapping modules.
        Stores result in self.current_frame.
        """
        self._reset_transient_frame_state()
        self.pass_index += 1

        # Run all stages up to white balance
        oecf_raw = self._run_pre_digital_gain()

        # Store result
        self._run_gain_through_white_balance(oecf_raw, mirror_gain=False)

    def run_pipeline(self, visualize_output: bool = True) -> None:
        """
        Simulation of ISP-Pipeline (single active native engine)
        """
        self._run_native(save_facade_output=visualize_output)

    def _run_legacy_pipeline(self, visualize_output: bool = True) -> None:
        """
        Retired legacy integer/YUV path kept only for audit history; not the
        active engine. Use run_pipeline.
        """
        if self.raw is None:
            raise RuntimeError("RAW image must be loaded before running the pipeline.")
        if self.platform is None or self.sensor_info is None or self.c_yaml is None:
            raise RuntimeError("Configuration must be loaded before running the pipeline.")
        self._reset_transient_frame_state()
        self.pass_index += 1
        platform = self.platform
        sensor_info = self.sensor_info
        skip_disabled = platform.get("skip_disabled_modules", False)

        # =====================================================================
        oecf_raw = self._run_pre_digital_gain()

        # =====================================================================
        # Digital Gain → … → Auto-Exposure (2nd pass for direct AE: pass 1 meters, pass 2 applies new index)
        # rerun_from_digital_gain defaults True when omitted so direct mode actually affects this frame.
        # Requires DG enabled + auto: otherwise current_gain must stay under manual control.
        dg_auto_ok = bool(self.parm_dga.get("is_enable", True)) and bool(self.parm_dga.get("is_auto", False))
        max_ae_passes = (
            2
            if (
                dg_auto_ok
                and self.parm_ae.get("exposure_correction_mode", "step") == "direct"
                and self.parm_ae.get("rerun_from_digital_gain", True)
                and self.parm_ae["is_enable"]
            )
            else 1
        )

        for ae_pass in range(max_ae_passes):
            prev_idx = self.parm_dga["current_gain"]

            # =====================================================================
            wb_raw = self._run_gain_through_white_balance(
                oecf_raw, ae_pass=ae_pass, mirror_gain=True
            )

            # =====================================================================
            # HDR tone mapping before Demosaicing
            def _tone_mapping_pre() -> np.ndarray:
                if self.tone_mapping_before_demosaic:
                    tone_mapper = tone_mapping(wb_raw, pipeline_self=cast(ToneMappingContext, self))
                    hdr = tone_mapper.execute()
                    self.logger.info(f"HDR Image mean: {np.mean(hdr)}")
                    return cast(np.ndarray, hdr)
                max_val = 2 ** sensor_info.get("hdr_bit_depth", 24) - 1
                return (wb_raw.astype(np.float32) * (65535.0 / max_val)).astype(np.uint16)

            hdr_raw = self._run_stage("tone_mapping_pre", _tone_mapping_pre, ae_pass=ae_pass)

            # =====================================================================
            # CFA demosaicing
            def _demosaic() -> np.ndarray:
                cfa_inter = Demosaic(cast(RawBayerImage, hdr_raw), platform, sensor_info, self.parm_dem)
                out = cfa_inter.execute()
                self.logger.info(f"Demosaiced Image mean: {np.mean(out)}")
                return cast(np.ndarray, out)

            demos_img = self._run_stage("demosaic", _demosaic, ae_pass=ae_pass)

            # =====================================================================
            # Color correction matrix
            # Use optimized version for better performance
            def _ccm() -> np.ndarray:
                ccm = CCMOPT(demos_img, platform, sensor_info, self.parm_ccm)
                out = ccm.execute()
                self.logger.info(f"CCM Image mean: {np.mean(out)}")
                return cast(np.ndarray, out)

            ccm_img = self._run_stage("color_correction_matrix", _ccm, ae_pass=ae_pass)

            # =====================================================================
            # HDR tone mapping after Demosaicing
            def _tone_mapping_post() -> np.ndarray:
                if not self.tone_mapping_before_demosaic:
                    tone_mapper = tone_mapping(ccm_img, pipeline_self=cast(ToneMappingContext, self))
                    mapped = tone_mapper.execute()
                    self.logger.info(f"HDR Image mean: {np.mean(mapped)}")
                    return cast(np.ndarray, mapped)
                return cast(np.ndarray, ccm_img)

            ccm_img = self._run_stage("tone_mapping_post", _tone_mapping_post, ae_pass=ae_pass)

            # =====================================================================
            # Auto-Exposure (operates on 16-bit linear RGB before bit conversion)
            # This provides maximum precision for exposure metering
            ae_holder: list[AE] = []

            def _ae() -> int | None:
                aef = AE(ccm_img, sensor_info, self.parm_ae)
                ae_holder.append(aef)
                return cast(int | None, aef.execute())

            self.ae_feedback = self._run_stage("auto_exposure", _ae, ae_pass=ae_pass)
            aef = ae_holder[0]
            self.logger.info(f"AE Feedback: {self.ae_feedback}")

            direct_ok = (
                dg_auto_ok
                and self.parm_ae.get("exposure_correction_mode", "step") == "direct"
                and self.parm_ae["is_enable"]
                and aef.last_meter_average is not None
            )
            new_idx = prev_idx
            if direct_ok:
                new_idx = aef.suggest_direct_gain_index(prev_idx, self.parm_dga["gain_array"])
                self.parm_dga["current_gain"] = new_idx
                if self.c_yaml is not None:
                    self.c_yaml["digital_gain"]["current_gain"] = new_idx
                self.dga_current_gain = new_idx
                self.parm_dga["ae_feedback"] = None
            if not direct_ok or ae_pass + 1 >= max_ae_passes or new_idx == prev_idx:
                break

        # =====================================================================
        # Convert 16-bit linear RGB to 8-bit linear RGB for YUV processing
        # CSC and downstream YUV modules expect 8-bit input
        # IMPORTANT: This is a LINEAR conversion (no gamma applied)
        # Formula: output = input × (255/65535) = input / 257
        # This differs from Infinite-ISP which uses gamma for bit conversion
        # See: GAMMA_CORRECTION_FINAL_SOLUTION.md for rationale
        def _linear_to_8bit() -> np.ndarray:
            self.logger.info(f"Converting 16-bit linear RGB to 8-bit linear RGB for YUV processing")
            self.logger.info(f"  Input range: [{np.min(ccm_img)}, {np.max(ccm_img)}]")
            linear = np.clip((ccm_img.astype(np.float32) / 65535.0 * 255.0), 0, 255).astype(np.uint8)
            self.logger.info(f"  Output range: [{np.min(linear)}, {np.max(linear)}]")
            return linear

        linear_8bit = self._run_stage("linear_16bit_to_8bit", _linear_to_8bit)

        # =====================================================================
        # Color space conversion (operates on 8-bit linear RGB)
        # ITU-R BT.601/709 standards require linear RGB input for correct color math
        def _csc() -> np.ndarray:
            csc = CSC(linear_8bit, platform, sensor_info, self.parm_csc, self.parm_cse)
            out = csc.execute()
            self.logger.info(f"CSC Image mean: {np.mean(out)}")
            return cast(np.ndarray, out)

        csc_img = self._run_stage("color_space_conversion", _csc)

        # =====================================================================
        # Local Dynamic Contrast Improvement
        # When post_gamma=true, LDCI is deferred to after gamma on 8-bit RGB
        # (matching boltISP's applyLdciRgb888 placement).
        ldci_post_gamma = bool(self.parm_ldci.get("post_gamma", False))

        def _ldci() -> np.ndarray:
            if ldci_post_gamma or (skip_disabled and not self.parm_ldci["is_enable"]):
                return csc_img
            ldci = LDCI(
                csc_img,
                platform,
                sensor_info,
                self.parm_ldci,
                self.parm_csc["conv_standard"],
            )
            return cast(np.ndarray, ldci.execute())

        ldci_img = self._run_stage("ldci", _ldci)

        # =====================================================================
        # Sharpening
        def _sharpen() -> np.ndarray:
            if skip_disabled and not self.parm_sha["is_enable"]:
                return ldci_img
            sharp = SHARP(
                ldci_img,
                platform,
                sensor_info,
                self.parm_sha,
                self.parm_csc["conv_standard"],
            )
            return cast(np.ndarray, sharp.execute())

        sharp_img = self._run_stage("sharpen", _sharpen)

        # =====================================================================
        # 2d noise reduction
        def _nr2d() -> np.ndarray:
            if skip_disabled and not self.parm_2dn["is_enable"]:
                return sharp_img
            nr2d = NR2D(
                sharp_img,
                sensor_info,
                self.parm_2dn,
                platform,
                self.parm_csc["conv_standard"],
            )
            return cast(np.ndarray, nr2d.execute())

        nr2d_img = self._run_stage("2d_noise_reduction", _nr2d)

        # =====================================================================
        # RGB conversion (YUV→RGB, outputs 8-bit linear RGB)
        rgbc_holder: list[RGBC] = []

        def _rgb_conversion() -> np.ndarray:
            rgbc = RGBC(nr2d_img, platform, sensor_info, self.parm_rgb, self.parm_csc)
            rgbc_holder.append(rgbc)
            out = rgbc.execute()
            self.logger.info(f"RGB Conversion output range: [{np.min(out)}, {np.max(out)}]")
            return cast(np.ndarray, out)

        rgbc_img = self._run_stage("rgb_conversion", _rgb_conversion)
        rgbc = rgbc_holder[0]

        # =====================================================================
        # Gamma correction (8-bit linear RGB → 8-bit gamma-corrected RGB)
        # IMPORTANT: Applied AFTER YUV processing, as final OETF encoding step
        # This is the correct position per IEC 61966-2-1 (sRGB) and industry standards
        # Differs from Infinite-ISP which applies gamma before CSC (pragmatic but incorrect)
        # Gamma acts as Opto-Electronic Transfer Function (OETF) for display encoding
        # See: GAMMA_CORRECTION_FINAL_SOLUTION.md for detailed rationale
        def _gamma() -> np.ndarray:
            gmc = GC(rgbc_img, platform, sensor_info, self.parm_gmc)
            out = gmc.execute()
            self.logger.info(f"Gamma output range: [{np.min(out)}, {np.max(out)}]")
            return cast(np.ndarray, out)

        gamma_img = self._run_stage("gamma_correction", _gamma)

        # =====================================================================
        # Post-gamma LDCI on 8-bit gamma-encoded RGB
        # Runs only when ldci.post_gamma: true (matches boltISP placement)
        def _ldci_post() -> np.ndarray:
            if ldci_post_gamma:
                self.logger.info("Applying post-gamma LDCI on 8-bit RGB (boltISP-compatible mode)")
                out = apply_ldci_rgb8(gamma_img, platform, sensor_info, self.parm_ldci)
                self.logger.info(f"  Post-gamma LDCI output range: [{np.min(out)}, {np.max(out)}]")
                return cast(np.ndarray, out)
            return cast(np.ndarray, gamma_img)

        gamma_img = self._run_stage("ldci_post_gamma", _ldci_post)

        # =====================================================================
        # Scaling
        def _scale() -> np.ndarray:
            if skip_disabled and not self.parm_sca["is_enable"]:
                return gamma_img
            scale = Scale(
                gamma_img,
                platform,
                sensor_info,
                self.parm_sca,
                self.parm_csc["conv_standard"],
            )
            return cast(np.ndarray, scale.execute())

        scaled_img = self._run_stage("scale", _scale)

        # =====================================================================
        # YUV saving format 444, 422 etc
        yuv_write = (
            True
            if self.options is None
            else (visualize_output and self.options.save_outputs)
        )

        def _yuv() -> np.ndarray:
            yuv = YUV_C(
                scaled_img,
                platform,
                sensor_info,
                self.parm_yuv,
                write_output=yuv_write,
            )
            return cast(np.ndarray, yuv.execute())

        yuv_conv = self._run_stage("yuv_conversion_format", _yuv)

        # only to view image if csc is off it does nothing
        out_img = yuv_conv
        out_dim = scaled_img.shape  # dimensions of Output Image

        # There can be two out_img formats depending upon which modules are
        # enabled 1. YUV    2. RGB
        if self.parm_yuv["is_enable"] is True:

            # YUV_C is enabled and RGB_C is disabled: Output is compressed YUV
            # To display : Need to decompress it and convert it to RGB.
            image_height, image_width, _ = out_dim
            yuv_custom_format = self.parm_yuv["conv_type"]

            yuv_conv = util.get_image_from_yuv_format_conversion(yuv_conv, image_height, image_width, yuv_custom_format)

            rgbc.yuv_img = yuv_conv
            out_rgb = rgbc.yuv_to_rgb()

        elif self.parm_rgb["is_enable"] is False:

            # RGB_C is disabled: Output is 3D - YUV
            # To display : Only convert it to RGB
            rgbc.yuv_img = yuv_conv
            out_rgb = rgbc.yuv_to_rgb()

        else:
            # RGB_C is enabled: Output is RGB
            # no further processing is needed for display
            out_rgb = out_img

        self.last_output_native = np.asarray(out_img)
        self.last_output_rgb = np.asarray(out_rgb).copy()
        self.last_output_geometry = (int(out_dim[0]), int(out_dim[1]))

        effective_save = (
            visualize_output
            if self.options is None
            else (visualize_output and self.options.save_outputs)
        )
        show_histograms = (
            visualize_output
            if self.options is None
            else self.options.diagnostics
        ) and self.platform.get("plot_histograms", False)
        # Is not part of ISP-pipeline only assists in visualizing output results

        # Plot histograms if enabled (debug feature)
        if show_histograms:
            try:
                # Estimate dynamic range of input (after decompanding)
                input_dr = estimate_dynamic_range(self.decompanded_img)
                self.logger.info(f"Input Dynamic Range (after decompanding): {input_dr['dynamic_range_ev']:.2f} EV")
                self.logger.info(f"Input Min: {input_dr['min_val']:.0f}, Max: {input_dr['max_val']:.0f}")
                self.logger.info(
                    f"Input Percentiles (0.1%, 99.9%): {input_dr['percentile_min']:.0f}, {input_dr['percentile_max']:.0f}"
                )
                self.logger.info(f"Input Bit Depth Utilized: {input_dr['bit_depth_utilized']:.1f} bits")

                # Estimate dynamic range of output
                output_dr = estimate_dynamic_range(out_rgb)
                self.logger.info(f"Output Dynamic Range: {output_dr['dynamic_range_ev']:.2f} EV")
                self.logger.info(f"Output Min: {output_dr['min_val']:.0f}, Max: {output_dr['max_val']:.0f}")
                self.logger.info(
                    f"Output Percentiles (0.1%, 99.9%): {output_dr['percentile_min']:.0f}, {output_dr['percentile_max']:.0f}"
                )
                self.logger.info(f"Output Bit Depth Utilized: {output_dr['bit_depth_utilized']:.1f} bits")

                # Generate histogram comparison plot
                show_log = self.platform.get("histogram_show_log", True)
                histogram_filename = f"{self.out_file}_histogram_comparison.png"

                plot_histogram_comparison(
                    self.decompanded_img,
                    out_rgb,
                    output_dir=self.platform.get("output_dir", "module_output"),
                    filename=histogram_filename,
                    input_label="Input (after decompanding)",
                    output_label="Output",
                    show_log=show_log,
                )

                output_dir = self.platform.get("output_dir", "module_output")
                self.logger.info(f"Histogram comparison saved to: {output_dir}/{histogram_filename}")
            except Exception as e:
                self.logger.warning(f"Failed to plot histograms: {e}")

        if effective_save:

            # If both RGB_C and YUV_C are enabled. Brilliant-ISP will generate
            # an output but it will be an invalid image.
            short_names = self.platform.get("short_output_names", False)
            # Derive the decorated name into a local; mutating self.outFileName here
            # accumulated the suffix on every frame when the instance is reused across
            # a batch, eventually overflowing the filesystem name limit.
            out_file_name = self.outFileName
            if not short_names:
                out_file_name = (
                    out_file_name
                    + "TM_"
                    + str(self.tone_mapper)
                    + "_s_"
                    + str(self.parm_cse["saturation_gain"])
                    + "_CCM_"
                    + str(self.parm_ccm["is_enable"])
                    + "_Before_Demosaic_"
                    + str(self.tone_mapping_before_demosaic)
                )

            util.save_pipeline_output(
                self.out_file, out_rgb, self.c_yaml, out_file_name, self.output_path, short_names=short_names
            )

    def execute(
        self,
        img_path: str | None = None,
        load_method: str = "auto",
        byte_order: ByteOrder | None = None,
    ) -> None:
        """
        Start execution of Brilliant-ISP

        Args:
            img_path (str): Optional path to image file
            load_method (str): only 'auto' is supported; explicit storage is
                declared via sensor_info.storage_format/data_format.
            byte_order (str): 'little' or 'big'; if None, the declared
                sensor_info.endian_type governs raw-buffer input.
        """
        if self.c_yaml is None:
            raise RuntimeError("Configuration must be loaded before execution.")
        if load_method != "auto":
            raise ValueError(
                f"load_method={load_method!r} is no longer honored: the decoder "
                "selects storage solely from sensor_info.storage_format/"
                "data_format; set those keys instead"
            )
        if img_path is not None:
            self.raw_file = img_path
            self.c_yaml["platform"]["filename"] = self.raw_file

        self.load_raw(byte_order=byte_order)

        # Print Logs to mark start of pipeline Execution
        self.logger.info(50 * "-" + "\nLoading RAW Image Done......\n")
        self.logger.info(f"Filename: {self.in_file}")

        # Note Initial Time for Pipeline Execution
        start = time.time()

        if not self.render_3a:
            # Run ISP-Pipeline once
            self.run_pipeline(visualize_output=True)
            # Display 3A Statistics
        else:
            # Run ISP-Pipeline till Correct Exposure with AWB gains
            self.execute_with_3a_statistics()

        util.display_ae_statistics(self.ae_feedback, self.awb_gains, self.logger)

        # Print Logs to mark end of pipeline Execution
        self.logger.info(50 * "-" + "\n")

        # Calculate pipeline execution time
        self.logger.info(f"\nPipeline Elapsed Time: {time.time() - start:.3f}s")

    def load_3a_statistics(self, awb_on: bool = True, ae_on: bool = True) -> None:
        """
        Update 3A Stats into WB and DG modules parameters
        """
        if self.c_yaml is None:
            raise RuntimeError("Configuration must be loaded before updating 3A statistics.")
        # Update 3A in c_yaml too because it is output config
        if awb_on is True and self.parm_dga["is_auto"] and self.parm_awb["is_enable"]:
            self.parm_wbc["r_gain"] = self.c_yaml["white_balance"]["r_gain"] = float(self.awb_gains[0])
            self.parm_wbc["b_gain"] = self.c_yaml["white_balance"]["b_gain"] = float(self.awb_gains[1])
        if ae_on is True and self.parm_dga.get("is_enable", True) and self.parm_dga["is_auto"] and self.parm_ae["is_enable"]:
            self.parm_dga["ae_feedback"] = self.c_yaml["digital_gain"]["ae_feedback"] = self.ae_feedback
            self.parm_dga["current_gain"] = self.c_yaml["digital_gain"]["current_gain"] = self.dga_current_gain

    def execute_with_3a_statistics(self) -> None:
        """Bounded native 3A controller; delegates to the shared native path."""
        self._run_native()

    def _execute_with_3a_statistics_legacy(self) -> None:
        """Retired unbounded legacy 3A loop; superseded by _run_native."""

        """
        Execute Brilliant-ISP with AWB gains and correct exposure
        """

        # Max valid gain index (len - 1).  Previous code used len() which caused
        # an off-by-one: the boundary condition was never satisfied when the gain
        # was already at its maximum, making the loop infinite.
        max_dg_idx = len(self.parm_dga["gain_array"]) - 1
        # Hard cap: at most (gain_array_size + 2) passes regardless of convergence
        max_iterations = max_dg_idx + 3

        # Run ISP-Pipeline
        self.run_pipeline(visualize_output=False)
        self.load_3a_statistics()
        iterations = 0
        while not (
            (self.ae_feedback == 0)
            or (self.ae_feedback == -1 and self.dga_current_gain >= max_dg_idx)
            or (self.ae_feedback == 1 and self.dga_current_gain <= 0)
            or self.ae_feedback is None
        ):
            iterations += 1
            if iterations >= max_iterations:
                self.logger.warning(
                    f"3A AE loop hit iteration cap ({max_iterations}) without converging "
                    f"(ae_feedback={self.ae_feedback}, gain_idx={self.dga_current_gain}). "
                    "Proceeding with current gain."
                )
                break
            self.run_pipeline(visualize_output=False)
            self.load_3a_statistics()

        self.run_pipeline(visualize_output=True)

    def update_sensor_info(
        self,
        sensor_info: ParsedFileNameInfo | ExtractedRawMetadata,
        update_blc_wb: bool = False,
    ) -> None:
        """
        Update sensor_info in config files
        """
        if self.sensor_info is None or self.c_yaml is None:
            raise RuntimeError("Configuration must be loaded before updating sensor info.")
        self.sensor_info["width"] = self.c_yaml["sensor_info"]["width"] = sensor_info["width"]

        self.sensor_info["height"] = self.c_yaml["sensor_info"]["height"] = sensor_info["height"]

        self.sensor_info["bit_depth"] = self.c_yaml["sensor_info"]["bit_depth"] = sensor_info["bit_depth"]

        self.sensor_info["bayer_pattern"] = self.c_yaml["sensor_info"]["bayer_pattern"] = sensor_info["bayer_pattern"]

        # Keep the per-frame reset baseline in sync so the new dimensions survive
        # _reset_transient_frame_state() on the next frame.
        if getattr(self, "_frame_reset_baseline", None) is not None:
            self._frame_reset_baseline["sensor_wh"] = (
                self.sensor_info["width"],
                self.sensor_info["height"],
            )
            self._frame_reset_baseline["bayer_pattern"] = sensor_info["bayer_pattern"]
        self._loaded_frame_geometry = None

        if update_blc_wb:
            black_level = sensor_info.get("black_level")
            white_level = sensor_info.get("white_level")
            wb_gains = sensor_info.get("wb_gains")
            if black_level is None or white_level is None or wb_gains is None:
                raise ValueError("update_blc_wb=True requires black_level, white_level, and wb_gains metadata.")
            self.parm_blc["r_offset"] = self.c_yaml["black_level_correction"]["r_offset"] = black_level[0]
            self.parm_blc["gr_offset"] = self.c_yaml["black_level_correction"]["gr_offset"] = black_level[1]
            self.parm_blc["gb_offset"] = self.c_yaml["black_level_correction"]["gb_offset"] = black_level[2]
            self.parm_blc["b_offset"] = self.c_yaml["black_level_correction"]["b_offset"] = black_level[3]

            self.parm_blc["r_sat"] = self.c_yaml["black_level_correction"]["r_sat"] = white_level
            self.parm_blc["gr_sat"] = self.c_yaml["black_level_correction"]["gr_sat"] = white_level
            self.parm_blc["gb_sat"] = self.c_yaml["black_level_correction"]["gb_sat"] = white_level
            self.parm_blc["b_sat"] = self.c_yaml["black_level_correction"]["b_sat"] = white_level

            self.parm_wbc["r_gain"] = self.c_yaml["white_balance"]["r_gain"] = wb_gains[0]
            self.parm_wbc["b_gain"] = self.c_yaml["white_balance"]["b_gain"] = wb_gains[2]

            # if sensor_info.get("ccm") is not None:
            #     self.parm_ccm["corrected_red"] = sensor_info["ccm"][0, 0:3]
            #     self.parm_ccm["corrected_green"] = sensor_info["ccm"][1, 0:3]
            #     self.parm_ccm["corrected_blue"] = sensor_info["ccm"][2, 0:3]

    def _save_native_tone_mapping_diagnostics(
        self, image: np.ndarray, *, stage: str, resolved: ResolvedToneMapping,
    ) -> None:
        """Native TMO diagnostics: IO/metadata only, no pixel math.

        Returns early before any array work when explicit options disable
        diagnostics or the resolved operator is disabled. Explicit
        ``diagnostics=True`` requires ``resolved.is_save`` for the NPY pair
        and ``resolved.is_plot_curve`` for the curve independently; the
        ``options=None`` facade honors the resolved flags directly.
        """
        if not resolved.spec.enabled:
            return
        if self.options is not None and not self.options.diagnostics:
            return
        if not resolved.is_save and not resolved.is_plot_curve:
            return
        if self.platform is None:
            return
        out_dir = self.platform.get("output_dir")
        if not out_dir:
            return
        platform = self.platform
        self._check_cancelled()
        import dataclasses
        from modules.tone_mapping.diagnostics import plot_curve, save_mapped_output
        image = np.asarray(image)
        module = (
            f"native_{stage}_{resolved.spec.operator}_{resolved.spec.backend}_normalized"
        )
        if resolved.is_save:
            sensor = self.sensor_info
            if sensor is None:
                raise RuntimeError("native tone-mapping diagnostics require sensor metadata")
            metadata = {
                "diagnostic_schema_version": 1,
                "stage": stage,
                "signal_domain": "tone_mapped_bayer" if image.ndim == 2 else "display_linear_rgb",
                "transfer": "linear",
                "value_units": "normalized",
                "reference_white": 1.0,
                "cfa_pattern": (
                    sensor["bayer_pattern"]
                    if image.ndim == 2 else None
                ),
                "shape": list(image.shape),
                "dtype": str(image.dtype),
                "tone_mapping": dataclasses.asdict(resolved.spec),
            }
            npy = save_mapped_output(
                image,
                platform=platform,
                input_name=getattr(self, "in_file", "native"),
                module_name=module,
                metadata=metadata,
            )
            self.last_artifacts += (npy, npy.with_suffix(".json"))
        if resolved.is_plot_curve:
            curve = plot_curve(resolved.spec, platform)
            if curve is not None:
                self.last_artifacts += (curve,)

    def _run_native(self, *, save_facade_output: bool = True) -> None:
        """One active native engine: linear compute_linear front half, named
        scene-linear tap feeding the bounded AE controller, unified tone map,
        luma pipeline, transfer/quantize, explicit writer. No module
        execute()/save in this path where a compute_linear exists.
        """
        from modules.tone_mapping.native import map_native
        from modules.tone_mapping.registry import resolve_tone_mapping
        from util.color import adjust_saturation, encode_transfer, luminance
        from util.luma import replace_luminance
        from util.native_array import as_float32
        from util.native_output import output_metadata, preview_from_native, quantize_output
        from util.output_contract import OutputSpec
        from util.signal_range import linear_code_range

        if self.raw is None or self.platform is None or self.sensor_info is None:
            raise RuntimeError("RAW image and configuration must be loaded before running the pipeline.")
        self._reset_transient_frame_state()
        self.pass_index += 1
        platform = self.platform
        c_yaml = self.c_yaml
        if c_yaml is None:
            raise RuntimeError("Configuration must be loaded before running the pipeline.")

        code_range = linear_code_range(c_yaml)
        declared_max = float(code_range.maximum)

        # ---------------------------------------------------------------
        # Front half (crop/DPC/BLC/green repair/PWC/OECF, once per frame)
        oecf_raw = np.array(self._run_pre_digital_gain(), copy=True)

        # Local processing sensor context: declared linear code range,
        # not the source HDR budget.
        p_sensor = dict(self.sensor_info)
        p_sensor["hdr_bit_depth"] = code_range.bit_depth

        # ---------------------------------------------------------------
        # Linear Bayer chain + bounded AE controller. Every pass restarts
        # from the immutable OECF output; the applied gain index is local.
        dg_auto = bool(self.parm_dga.get("is_enable", True)) and bool(self.parm_dga.get("is_auto", False))
        ae_enabled = bool(self.parm_ae.get("is_enable", False))
        ae_mode = str(self.parm_ae.get("exposure_correction_mode", "step"))
        ae_rerun = bool(self.parm_ae.get("rerun_from_digital_gain", True))
        if ae_enabled and ae_rerun:
            max_passes = 24 if self.render_3a else 2
        else:
            max_passes = 1

        hint_state = getattr(self, "processing_state", None)
        gain_idx = int(self.parm_dga.get("current_gain", 0))
        if dg_auto and hint_state is not None and hint_state.digital_gain_index is not None:
            gain_idx = int(hint_state.digital_gain_index)
        next_idx = gain_idx
        ae_stop = "disabled" if not ae_enabled else "no_rerun"
        ae_passes = 0
        ae = None
        suggested = gain_idx
        visited: set[tuple[int, int | None]] = set()
        rgb_linear: np.ndarray | None = None
        ae_rgb: np.ndarray | None = None

        resolved = resolve_tone_mapping(c_yaml)
        advanced_pre = self.tone_mapping_before_demosaic and resolved.spec.enabled
        if advanced_pre:
            self.logger.warning(
                "Advanced pre-demosaic tone mapping enabled: extra CPU branch; "
                "color fidelity limitations apply")

        smooth_mode = ae_enabled and dg_auto and bool(self.parm_ae.get("smooth_control", False))
        smooth_params = smooth_ae.SmoothAEParams.from_config(self.parm_ae) if smooth_mode else None
        smooth_desired: float | None = None
        self._smooth_gain = None
        if smooth_mode:
            max_passes = 2
            prev_gain = hint_state.ae_gain if hint_state is not None else None
            self._smooth_gain = float(prev_gain) if prev_gain is not None else float(self.parm_dga["gain_array"][gain_idx])
            smooth_linear = not bool(self.parm_bnr.get("is_enable", False))

            def _smooth_tone_map(y: np.ndarray) -> np.ndarray:
                if not resolved.spec.enabled:
                    return y
                return np.asarray(map_native(y, config=resolved.spec), dtype=np.float64)

        for ae_pass in range(max_passes):
            ae_passes = ae_pass + 1
            wb_code = self._run_gain_through_white_balance(oecf_raw, ae_pass=ae_pass, gain_index=gain_idx)
            awb_status = self.last_awb_status
            awb_count = self.last_awb_count
            self.current_frame = as_float32(wb_code, name="current_frame")

            # ---------------------------------------------------------------
            # Normalize the WB code float by the declared linear maximum, then
            # feed the demosaic its fixed 65535-code-equivalent domain; the
            # capture is normalized camera RGB.
            def _demosaic(c=wb_code) -> np.ndarray:
                normalized = c.astype(np.float64) / declared_max
                bayer65 = as_float32(normalized * 65535.0, name="demosaic_input")
                demos = Demosaic(bayer65, platform, cast(SensorInfo, p_sensor), self.parm_dem).compute_linear()
                out = np.asarray(demos, dtype=np.float64) / 65535.0
                return as_float32(out, name="demosaic")
            rgb_linear = np.asarray(self._run_stage("demosaic", _demosaic, ae_pass=ae_pass), dtype=np.float64)

            def _ccm(rgb=rgb_linear) -> np.ndarray:
                return CCMOPT(as_float32(rgb, name="ccm_input"), platform, cast(SensorInfo, p_sensor), self.parm_ccm).compute_linear()
            rgb_linear = self._run_stage("color_correction_matrix", _ccm, ae_pass=ae_pass)

            # ---------------------------------------------------------------
            # Named tap: scene-linear RGB feeds the AE meter (untonemapped).
            def _linear_tap(rgb=rgb_linear) -> np.ndarray:
                return np.asarray(rgb)
            ae_rgb = np.asarray(self._run_stage("linear_rgb_pre_tone_mapping", _linear_tap, ae_pass=ae_pass))
            ae = AE(ae_rgb, cast(SensorInfo, p_sensor), self.parm_ae, meter_domain="scene_linear_rgb")
            fb = self._run_stage("auto_exposure", lambda a=ae: a.get_exposure_feedback(), ae_pass=ae_pass)
            self.ae_feedback = fb

            if smooth_mode and smooth_params is not None:
                applied = float(self._smooth_gain)
                arr = np.log2(np.asarray(self.parm_dga["gain_array"], dtype=np.float64))
                if ae_pass > 0:
                    ae_stop = "smooth_rerun"
                    suggested = next_idx = int(np.argmin(np.abs(arr - np.log2(applied))))
                    break
                y_unity = smooth_ae.scene_luminance(rgb_linear, applied, smooth_params.meter_stride)
                peak_unity = (smooth_ae.scene_peak(rgb_linear, applied, smooth_params.meter_stride)
                              if smooth_params.max_clip_fraction < 1.0 else None)
                roi_w = metering_roi.weight_map(np.shape(rgb_linear)[:2], metering_roi.roi_config(self.parm_ae))
                stride = smooth_params.meter_stride
                smooth_desired = smooth_ae.solve_gain(
                    y_unity, _smooth_tone_map, smooth_params, peak_unity=peak_unity,
                    weights=None if roi_w is None else roi_w[::stride, ::stride])
                prev = hint_state.ae_gain if hint_state is not None else None
                new_gain = smooth_ae.next_gain(prev, smooth_desired, smooth_params)
                self._smooth_gain = new_gain
                suggested = next_idx = int(np.argmin(np.abs(arr - np.log2(new_gain))))
                ratio = new_gain / applied
                if ratio == 1.0 or smooth_linear:
                    if ratio != 1.0:
                        wb_code = as_float32(np.asarray(wb_code, dtype=np.float64) * ratio, name="white_balance")
                        rgb_linear = np.asarray(rgb_linear, dtype=np.float64) * ratio
                        self.current_frame = as_float32(wb_code, name="current_frame")
                    ae_stop = "smooth"
                    break
                continue

            suggested = gain_idx
            if ae_enabled and ae.last_meter_average is not None and dg_auto:
                if ae_mode == "direct":
                    suggested = ae.suggest_direct_gain_index(gain_idx, list(self.parm_dga["gain_array"]))
                elif fb == -1:
                    suggested = min(gain_idx + 1, len(self.parm_dga["gain_array"]) - 1)
                elif fb == 1:
                    suggested = max(gain_idx - 1, 0)
            if not ae_enabled:
                ae_stop = "disabled"
                break
            if ae.last_meter_average is None or ae.last_meter_average <= 0:
                ae_stop = "no_signal"
                break
            if not dg_auto:
                ae_stop = "manual_gain"
                next_idx = suggested if ae_mode == "direct" else gain_idx
                break
            if not ae_rerun:
                ae_stop = "no_rerun"
                next_idx = suggested
                break
            state_key = (suggested, fb)
            # Direct mode follows target_luminance independently of the
            # histogram feedback sign; fb==0 converges only step mode.
            converged = suggested == gain_idx or (ae_mode != "direct" and fb == 0)
            if converged:
                if ae_mode == "direct":
                    target = self.parm_ae.get("target_luminance")
                    target = 0.5 if target is None else float(target)
                    wants_more = target > ae.last_meter_average
                    wants_less = target < ae.last_meter_average
                else:
                    wants_more, wants_less = fb == -1, fb == 1
                gain_limited = (
                    (wants_more and gain_idx == len(self.parm_dga["gain_array"]) - 1)
                    or (wants_less and gain_idx == 0)
                )
                ae_stop = "gain_limit" if gain_limited else "converged"
                next_idx = suggested
                break
            if state_key in visited:
                ae_stop = "repeated_gain"
                next_idx = suggested
                break
            if ae_pass + 1 >= max_passes:
                ae_stop = "pass_limit"
                next_idx = suggested
                break
            visited.add(state_key)
            gain_idx = suggested
            next_idx = suggested
            self._check_cancelled()

        assert rgb_linear is not None
        if ae_stop == "disabled" and ae_enabled:
            ae_stop = "no_signal"

        # ---------------------------------------------------------------
        # Tone mapping (native map_native on normalized RGB). The advanced
        # pre-demosaic branch tone-maps the normalized WB plane then renders
        # without a second TMO; AE always sees the untonemapped linear tap.
        if advanced_pre:
            wb_norm = np.asarray(wb_code, dtype=np.float64) / declared_max
            mapped_plane = self._run_stage("tone_mapping_pre", lambda: map_native(wb_norm, config=resolved.spec))
            self._save_native_tone_mapping_diagnostics(mapped_plane, stage="tone_mapping_pre", resolved=resolved)

            def _render_demosaic() -> np.ndarray:
                pre_dem = Demosaic(
                    as_float32(mapped_plane * 65535.0, name="pre_demosaic_input"),
                    platform, cast(SensorInfo, p_sensor), self.parm_dem).compute_linear()
                pre_rgb = np.asarray(pre_dem, dtype=np.float64) / 65535.0
                return as_float32(pre_rgb, name="demosaic_render")

            pre_rgb = self._run_stage("demosaic_render", _render_demosaic)
            pre_ccm = self._run_stage("color_correction_matrix_render", lambda: CCMOPT(
                pre_rgb, platform, cast(SensorInfo, p_sensor), self.parm_ccm).compute_linear())
            mapped = self._run_stage("tone_mapping_post", lambda: pre_ccm)
        else:
            rgb_norm = str(c_yaml.get("tone_mapping", {}).get("rgb_norm", "luminance"))
            if rgb_norm not in ("luminance", "max_rgb"):
                raise ValueError("tone_mapping.rgb_norm must be 'luminance' or 'max_rgb'")

            def _tone_map_post() -> np.ndarray:
                if rgb_norm == "max_rgb":
                    from modules.tone_mapping.native import map_native_max_rgb
                    return map_native_max_rgb(rgb_linear, config=resolved.spec)
                return map_native(rgb_linear, config=resolved.spec)
            mapped = self._run_stage("tone_mapping_post", _tone_map_post)
            self._save_native_tone_mapping_diagnostics(mapped, stage="tone_mapping_post", resolved=resolved)
        mapped = np.asarray(mapped, dtype=np.float64)

        # Saturation adjustment on display-linear RGB (Rec.709 weights, enabled only)
        def _saturation(m=mapped) -> np.ndarray:
            if self.parm_cse.get("is_enable", False):
                return np.asarray(adjust_saturation(m, float(self.parm_cse["saturation_gain"]), matrix="bt709"))
            return np.asarray(m)
        mapped = np.asarray(self._run_stage("saturation", _saturation), dtype=np.float64)

        # Opt-in luminance-preserving highlight gamut compression: pull channels
        # that exceed display white toward luminance instead of hard-clipping.
        tm_section = c_yaml.get("tone_mapping", {})
        if bool(tm_section.get("gamut_compress", False)):
            white = float(tm_section.get("gamut_compress_peak", 1.0))
            if not 0.0 < white <= 1.0:
                raise ValueError("tone_mapping.gamut_compress_peak must be in (0, 1]")

            def _gamut_compress(m=mapped) -> np.ndarray:
                y = np.clip(luminance(m, matrix="bt709", dtype=np.float64), 0.0, white)[..., None]
                peak = np.max(m, axis=-1, keepdims=True)
                t = np.where(peak > white, (white - y) / np.maximum(peak - y, 1e-12), 1.0)
                return y + np.clip(t, 0.0, 1.0) * (m - y)
            mapped = np.asarray(self._run_stage("gamut_compress", _gamut_compress), dtype=np.float64)

        # ---------------------------------------------------------------
        # Output spec + luma/display blocks + scale
        spec = OutputSpec.from_mapping(cast(Mapping[str, Any] | None, c_yaml.get("output")))

        # Pre-transfer LDCI (advanced placement) on linear luma
        ldci_place = str(self.parm_ldci.get("placement", "post_transfer"))
        def _ldci_pre(m=mapped) -> np.ndarray:
            if self.parm_ldci.get("is_enable", False) and ldci_place == "pre_transfer":
                from modules.ldci.clahe import apply_luma as _clahe
                y = luminance(m, matrix=spec.matrix, dtype=np.float64)
                out = _clahe(y, scale=1.0,
                             wind=self.parm_ldci.get("wind", 64),
                             clip_limit=self.parm_ldci.get("clip_limit", 2.0))
                return np.asarray(replace_luminance(m, out, matrix=spec.matrix))
            return np.asarray(m)
        mapped = np.asarray(self._run_stage("ldci_pre_transfer", _ldci_pre), dtype=np.float64)

        # ONE transfer encode; gamut clip first for nonlinear integer output
        def _transfer(m=mapped) -> np.ndarray:
            if spec.floating_linear:
                return as_float32(m, name="linear_float_output").astype(np.float64)
            bounded = np.clip(m, 0.0, 1.0) if spec.transfer != "linear" else m
            return np.asarray(encode_transfer(bounded, spec.transfer, gamma=getattr(spec, "gamma", 2.2), dtype=np.float64))
        encoded = np.asarray(self._run_stage("transfer_encode", _transfer), dtype=np.float64)

        # Post-transfer luma blocks: LDCI (post), sharpen, NLM — each on
        # output.matrix luminance, recombined via replace_luminance.
        def _ldci_post(e=encoded) -> np.ndarray:
            if self.parm_ldci.get("is_enable", False) and ldci_place == "post_transfer":
                from modules.ldci.clahe import apply_luma as _clahe
                y = luminance(e, matrix=spec.matrix, dtype=np.float64)
                out = _clahe(y, scale=1.0,
                             wind=self.parm_ldci.get("wind", 64),
                             clip_limit=self.parm_ldci.get("clip_limit", 2.0))
                return np.asarray(replace_luminance(e, out, matrix=spec.matrix))
            return np.asarray(e)
        encoded = np.asarray(self._run_stage("ldci_post_transfer", _ldci_post), dtype=np.float64)

        def _sharpen(e=encoded) -> np.ndarray:
            if self.parm_sha.get("is_enable", False):
                from modules.sharpen.unsharp_masking import apply_luma as _ush
                y = luminance(e, matrix=spec.matrix, dtype=np.float64)
                out = _ush(y, scale=1.0,
                           sigma=self.parm_sha.get("sharpen_sigma", 1.0),
                           strength=self.parm_sha.get("sharpen_strength", 0.5))
                return np.asarray(replace_luminance(e, out, matrix=spec.matrix))
            return np.asarray(e)
        encoded = np.asarray(self._run_stage("sharpen", _sharpen), dtype=np.float64)

        def _nlm(e=encoded) -> np.ndarray:
            if self.parm_2dn.get("is_enable", False):
                from modules.noise_reduction_2d.non_local_means import apply_luma as _nl
                y = luminance(e, matrix=spec.matrix, dtype=np.float64)
                out = _nl(y, scale=1.0,
                          window_size=self.parm_2dn.get("window_size", 3),
                          patch_size=self.parm_2dn.get("patch_size", 3),
                          strength=self.parm_2dn.get("strength", 10.0 / 255.0))
                return np.asarray(replace_luminance(e, out, matrix=spec.matrix))
            return np.asarray(e)
        encoded = np.asarray(self._run_stage("noise_reduction_2d", _nlm), dtype=np.float64)

        # Display-domain scale on encoded RGB (same Scale2D workflow, no save)
        def _scl(e=encoded) -> np.ndarray:
            if not self.parm_sca.get("is_enable", False):
                return np.asarray(e, dtype=np.float64)
            from modules.scale.scale import Scale
            scl = Scale(as_float32(e, name="scale_input"), platform,
                        cast(SensorInfo, p_sensor), self.parm_sca,
                        int(self.parm_csc.get("conv_standard", 3)) if isinstance(self.parm_csc, dict) else 3,
                        cpu_only=True)
            return np.asarray(scl.compute_linear(), dtype=np.float64)
        encoded = np.asarray(self._run_stage("scale", _scl), dtype=np.float64)

        # ---------------------------------------------------------------
        # Cancellation boundary BEFORE quantize/writer
        self._check_cancelled()

        # Quantize + preview + metadata (lead-native_output contract)
        native = self._run_stage("output", lambda: quantize_output(np.asarray(encoded, dtype=np.float64), spec))
        self._check_cancelled()
        preview = preview_from_native(native, spec)
        metadata = output_metadata(
            native, spec,
            source_identity=getattr(self, "in_file", "native"),
            tone_mapped=bool(resolved.spec.enabled),
        )
        self.last_output_native = native
        self.last_output_rgb = preview
        self.last_output_metadata = metadata
        self.last_output_geometry = (int(metadata.height), int(metadata.width))

        # ---------------------------------------------------------------
        # Feedback (authoritative last_feedback for PipelineResult)
        self.last_feedback = {
            "awb_gains": (float(self.awb_gains[0]), float(self.awb_gains[1])),
            "digital_gain_index": int(self.dga_current_gain),
            "ae_feedback": self.ae_feedback,
            "awb_status": awb_status,
            "awb_sample_count": awb_count,
            "awb_input_gain": self.last_awb_input_gain,
            "awb_meter_domain": "gain_compensated_linear_bayer_codes",
            "ae_meter_domain": "scene_linear_rgb",
            "ae_meter_average": getattr(ae, "last_meter_average", None),
            "ae_sample_count": getattr(ae, "sample_count", 0),
            "ae_suggested_gain_index": int(suggested),
            "ae_stop_reason": ae_stop,
            "ae_passes": ae_passes,
            "linear_reference_code_max": declared_max,
            "next_digital_gain_index": int(next_idx),
        }
        if self._smooth_gain is not None:
            self.last_feedback.update({
                "ae_gain": float(self._smooth_gain),
                "ae_next_gain": float(self._smooth_gain),
                "ae_desired_gain": smooth_desired,
            })
        if getattr(self, "last_awb_raw_estimate", None) is not None and awb_temporal.is_active(self.parm_awb):
            self.last_feedback["awb_raw_estimate"] = self.last_awb_raw_estimate

        # ---------------------------------------------------------------
        # Explicit writer (only when options.save_outputs or facade default,
        # isolated). Compute-only creates nothing.
        write_dir: Path | None = None
        if self.options is not None and self.options.save_outputs and self.options.output_dir:
            write_dir = Path(self.options.output_dir)
        elif self.options is None and save_facade_output and self.output_path:
            # facade execute() default: write next to the declared output path
            write_dir = Path(self.output_path)
        if write_dir is not None:
            self._check_cancelled()
            from util.output_writer import write_output
            artifacts = write_output(
                native, spec, metadata,
                output_dir=write_dir,
                stem=getattr(self, "out_file", "native"),
                effective_config=c_yaml,
                migration_report=getattr(self, "migration_report", {}),
            )
            self.last_artifacts += tuple(artifacts)
