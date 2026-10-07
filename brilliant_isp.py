"""
File: isp_pipeline.py
Description: Executes the complete pipeline
Code / Paper  Reference:
Author: Brian Deegan (based in part on 10xEngineers / Infinite-ISP)
------------------------------------------------------------
"""

import time
from pathlib import Path
from typing import cast
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
from util.config_merge import (
    ConfigPathArg,
    format_config_source,
    load_merged_yaml,
    normalize_config_paths,
    pipeline_config_paths,
)
from util.raw_io import (
    alignment_from_sensor_info,
    apply_data_alignment,
    apply_raw_flips,
    flips_from_sensor_info,
    read_raw_payload,
)
from util.debug_utils import DebugLogger
from util.debug_utils import get_debug_logger, set_global_debug_enabled
from util.histogram_utils import plot_histogram_comparison, estimate_dynamic_range
from util.isp_types import (
    AWBGains,
    AutoExposureConfig,
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
    data = np.frombuffer(data[:expected_size], dtype=np.uint8)
    data = data.reshape(-1, 3)
    b0, b1, b2 = data[:, 0].astype(np.uint32), data[:, 1].astype(np.uint32), data[:, 2].astype(np.uint32)
    if byte_order == "little":
        pixels = b0 | (b1 << 8) | (b2 << 16)
    else:
        pixels = (b0 << 16) | (b1 << 8) | b2
    return pixels.reshape(height, width)


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
    dtype = "<u2" if byte_order == "little" else ">u2"
    uint16_data = np.frombuffer(data[:expected_size], dtype=dtype)
    pixels = uint16_data[0::2].astype(np.uint32) | (uint16_data[1::2].astype(np.uint32) << 16)
    return pixels.reshape(height, width)


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
        config_path: ConfigPathArg,
        outFileName: str,
        output_path: str | None = None,
    ) -> None:
        """
        Constructor: Initialize with config and raw file path
        and Load configuration parameter from yaml file
        """
        self.data_path = data_path
        self.output_path = output_path if output_path else "out_frames/"
        self.outFileName = outFileName
        self.logger: DebugLogger
        self.platform: PlatformConfig | None = None
        self.sensor_info: SensorInfo | None = None
        self.c_yaml: PipelineConfig | None = None
        self.raw: np.ndarray | None = None
        self.decompanded_img: np.ndarray | None = None
        self.last_output_rgb: np.ndarray | None = None
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
        os.makedirs(self.output_path, exist_ok=True)
        self.load_config(config_path)
        assert self.platform is not None
        # Set global debug state from config
        set_global_debug_enabled(self.platform.get("debug_enabled", False))
        # Initialize debug logger after config is loaded
        self.logger = get_debug_logger("BrilliantISP", config=self.platform)

    _REQUIRED_CONFIG_KEYS = (
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

    def load_config(self, config_path: ConfigPathArg) -> None:
        """
        Load config information to respective module parameters.
        Validates required keys and uses defaults for optional sections.

        ``config_path`` may be a single YAML file or a list/tuple of paths merged
        depth-first (later files override earlier ones for overlapping keys).

        A single path whose name ends in ``_cam.yml`` is merged after
        ``config/base_hdr.yml`` automatically (see ``pipeline_config_paths``).
        """
        if isinstance(config_path, (str, Path)):
            paths = pipeline_config_paths(config_path)
        else:
            paths = normalize_config_paths(config_path)
        self.config_path = format_config_source(paths)
        c_yaml = cast(PipelineConfig, load_merged_yaml(paths))

        missing = [k for k in self._REQUIRED_CONFIG_KEYS if k not in c_yaml]
        if missing:
            raise KeyError(
                f"Config '{self.config_path}' missing required keys: {missing}. "
                "See config/base_hdr.yml plus a *_cam.yml overlay (e.g. svs_cam.yml)."
            )

        # Extract workspace info
        self.platform = c_yaml["platform"]
        self.platform["generate_tv"] = self.platform.get("generate_tv", False)
        self.platform["output_dir"] = "module_output"  # Directory for module debug outputs (curves, etc)
        self.raw_file = self.platform["filename"]
        self.render_3a = self.platform["render_3a"]
        self.sensor_info = c_yaml["sensor_info"]
        self.sensor_info.setdefault("horizontal_flip", False)
        self.sensor_info.setdefault("vertical_flip", False)

        # ISP module params
        self.parm_dpc: DeadPixelCorrectionConfig = c_yaml["dead_pixel_correction"]
        self.parm_cmpd: GenericConfig = c_yaml["companding"]
        self.parm_dga: DigitalGainConfig = c_yaml["digital_gain"]
        self.parm_lsc: LensShadingCorrectionConfig = c_yaml["lens_shading_correction"]
        self.parm_bnr: BayerNoiseReductionConfig = c_yaml["bayer_noise_reduction"]
        self.parm_blc: BlackLevelCorrectionConfig = c_yaml["black_level_correction"]
        self.parm_oec: GenericConfig = c_yaml.get("oecf", {"is_enable": False, "is_save": False})
        self.parm_wbc: WhiteBalanceConfig = c_yaml["white_balance"]
        self.parm_awb: GenericConfig = c_yaml["auto_white_balance"]
        # Single source of truth: manual vs auto follows auto_white_balance.is_enable only.
        self.parm_wbc["is_auto"] = self.parm_awb["is_enable"]
        self.parm_dem: DemosaicConfig = c_yaml["demosaic"]
        self.parm_ae: AutoExposureConfig = c_yaml["auto_exposure"]
        self.parm_dga["exposure_correction_mode"] = self.parm_ae.get("exposure_correction_mode", "step")
        self.parm_ccm: ColorCorrectionMatrixConfig = c_yaml["color_correction_matrix"]
        self.parm_gmc: GammaCorrectionConfig = c_yaml["gamma_correction"]
        self.param_durand = c_yaml["hdr_durand"]
        self.param_aces = c_yaml.get("aces", {})
        self.parm_csc: ColorSpaceConversionConfig = c_yaml["color_space_conversion"]
        self.parm_cse: ColorSaturationEnhancementConfig = c_yaml["color_saturation_enhancement"]
        self.parm_ldci: LDCIConfig = c_yaml["ldci"]
        self.parm_sha: SharpenConfig = c_yaml["sharpen"]
        self.parm_2dn: NoiseReduction2DConfig = c_yaml["2d_noise_reduction"]
        self.parm_rgb: RGBConversionConfig = c_yaml["rgb_conversion"]
        self.parm_sca: ScaleConfig = c_yaml["scale"]
        self.parm_cro: CropConfig = c_yaml["crop"]
        self.parm_yuv: YUVConversionFormatConfig = c_yaml["yuv_conversion_format"]
        self.c_yaml = c_yaml
        self.platform["rgb_output"] = self.parm_rgb["is_enable"]
        initial_in_file = Path(self.raw_file).stem
        self.platform["in_file"] = initial_in_file
        self.platform["out_file"] = (
            initial_in_file if self.platform.get("short_output_names", False) else "Out_" + initial_in_file
        )
        self.bit_depth = self.sensor_info["bit_depth"]
        self.tone_mapping: ToneMappingConfig = c_yaml["tone_mapping"]
        self.tone_mapping_before_demosaic = self.tone_mapping["tone_mapping_before_demosaic"]
        # hdr_durand is the YAML block name; tone_mapper value must be durand.
        if self.tone_mapping.get("tone_mapper") == "hdr_durand":
            self.tone_mapping["tone_mapper"] = "durand"
        self.tone_mapper = self.tone_mapping["tone_mapper"]
        if self.tone_mapper == "aces":
            self.param_aces = c_yaml.get("aces", {})
        if self.tone_mapper == "reinhard_integer":
            # reinhard_integer section, or legacy integer_tmo section name
            self.param_integer_tmo = c_yaml.get("reinhard_integer", c_yaml.get("integer_tmo", {}))
        if self.tone_mapper == "aces_integer":
            self.param_aces_integer = c_yaml.get("aces_integer", {})
        if self.tone_mapper == "hable":
            self.param_hable = c_yaml.get("hable", {})
        if self.tone_mapper == "hable_integer":
            self.param_hable_integer = c_yaml.get("hable_integer", {})

        # Snapshot the config values that a single frame mutates transiently, so
        # reusing one instance across a batch cannot leak state between frames.
        # Excludes intentional 3A carryover (WB gains / gain index), which is
        # threaded frame-to-frame by load_3a_statistics().
        self._frame_reset_baseline = {
            "sensor_wh": (self.sensor_info["width"], self.sensor_info["height"]),
            "yuv_is_enable": self.parm_yuv["is_enable"],
            "scale_is_debug": self.parm_sca.get("is_debug", False),
        }

        # add rgb_output_conversion module

    def _reset_transient_frame_state(self) -> None:
        """
        Restore config that gets mutated in place while processing a frame back to
        its as-configured baseline, so each frame is processed independently when a
        BrilliantISP instance is reused across many frames.

        Covers: sensor width/height (overwritten by RAW-size shape inference in
        load_raw), YUV is_enable (cleared by the YUV module when the output is RGB),
        and scale is_debug. Intentional 3A carryover is deliberately not reset here.
        """
        baseline = getattr(self, "_frame_reset_baseline", None)
        if baseline is None or self.sensor_info is None:
            return
        w, h = baseline["sensor_wh"]
        self.sensor_info["width"] = w
        self.sensor_info["height"] = h
        if self.c_yaml is not None:
            self.c_yaml["sensor_info"]["width"] = w
            self.c_yaml["sensor_info"]["height"] = h
        self.parm_yuv["is_enable"] = baseline["yuv_is_enable"]
        self.parm_sca["is_debug"] = baseline["scale_is_debug"]

    def load_raw(self, byte_order: ByteOrder = "little") -> None:
        """
        Load raw image from provided path with enhanced HDR support

        Args:
            byte_order (str): 'little' or 'big' endian for HDR loading
            reverse_uint32 (bool): If True, reverse byte order within uint32 pixel values
        """
        if self.platform is None or self.sensor_info is None:
            raise RuntimeError("Configuration must be loaded before loading RAW input.")
        # Clear any per-frame state left by a previous frame before deriving this one.
        self._reset_transient_frame_state()
        # Load raw image file information
        path_object = Path(self.data_path, self.raw_file)
        raw_path = str(path_object.resolve())
        self.in_file = path_object.stem
        short_names = self.platform.get("short_output_names", False)
        self.out_file = self.in_file if short_names else "Out_" + self.in_file

        self.platform["in_file"] = self.in_file
        self.platform["out_file"] = self.out_file

        width = self.sensor_info["width"]
        height = self.sensor_info["height"]
        bit_depth = self.sensor_info["bit_depth"]
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
                    self.sensor_info["width"] = width
                    self.sensor_info["height"] = height

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
                        self.sensor_info["width"] = load_w
                        self.sensor_info["height"] = load_h
                need_bytes = load_w * load_h * 2
                if file_size < need_bytes:
                    raise RuntimeError(f"Raw file too small: need {need_bytes} bytes for {load_w}x{load_h}, got {file_size}")
                if file_size > need_bytes:
                    self.logger.warning(
                        f"Raw file is {file_size - need_bytes} bytes longer than {load_w}x{load_h}x2; "
                        f"truncating to {need_bytes} bytes"
                    )
                et = str(self.sensor_info.get("endian_type", "")).lower()
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
        # Set skip_disabled to avoid running disabled modules
        skip_disabled = self.platform.get("skip_disabled_modules", False)

        # Run all stages up to white balance
        # Crop
        if skip_disabled and not self.parm_cro["is_enable"]:
            cropped_img = self.raw
        else:
            crop = Crop(self.raw, self.platform, self.sensor_info, self.parm_cro)
            cropped_img = crop.execute()

        # Dead Pixel Correction
        if skip_disabled and not self.parm_dpc["is_enable"]:
            dpc_raw = cropped_img
        else:
            dpc = DPC(cropped_img, self.sensor_info, self.parm_dpc, self.platform)
            dpc_raw = dpc.execute()

        # Black Level Correction
        if skip_disabled and not self.parm_blc["is_enable"]:
            blc_raw = dpc_raw
        else:
            blc = BLC(dpc_raw, self.platform, self.sensor_info, self.parm_blc)
            blc_raw = blc.execute()

        # Decompanding
        if skip_disabled and not self.parm_cmpd["is_enable"]:
            cmpd_raw = blc_raw.astype(np.uint32)
        else:
            cmpd = PWC(blc_raw, self.platform, self.sensor_info, self.parm_cmpd)
            cmpd_raw = cmpd.execute()

        # OECF
        if skip_disabled and not self.parm_oec.get("is_enable", False):
            oecf_raw = cmpd_raw
        else:
            oecf = OECF(cmpd_raw, self.platform, self.sensor_info, self.parm_oec)
            oecf_raw = oecf.execute()

        # Digital Gain
        dga = DG(oecf_raw, self.platform, self.sensor_info, self.parm_dga)
        dga_raw, self.dga_current_gain = dga.execute()

        # Lens Shading Correction
        if skip_disabled and not self.parm_lsc.get("is_enable", True):
            lsc_raw = dga_raw
        else:
            lsc = LSC(dga_raw, self.platform, self.sensor_info, self.parm_lsc)
            lsc_raw = lsc.execute()

        # Bayer Noise Reduction
        if skip_disabled and not self.parm_bnr["is_enable"]:
            bnr_raw = lsc_raw
        else:
            bnr = BNR(lsc_raw, self.sensor_info, self.parm_bnr, self.platform)
            bnr_raw = bnr.execute()

        # Auto White Balance
        awb = AWB(bnr_raw, self.sensor_info, self.parm_awb, self.parm_wbc)
        self.awb_gains = cast(AWBGains, awb.execute())

        # White Balance
        wbc = WBOPT(bnr_raw, self.platform, self.sensor_info, self.parm_wbc, self.awb_gains)
        wb_raw = wbc.execute()

        # Store result
        self.current_frame = wb_raw

    def run_pipeline(self, visualize_output: bool = True) -> None:
        """
        Simulation of ISP-Pipeline
        """
        if self.raw is None:
            raise RuntimeError("RAW image must be loaded before running the pipeline.")
        if self.platform is None or self.sensor_info is None or self.c_yaml is None:
            raise RuntimeError("Configuration must be loaded before running the pipeline.")
        skip_disabled = self.platform.get("skip_disabled_modules", False)

        # =====================================================================
        # Cropping
        if skip_disabled and not self.parm_cro["is_enable"]:
            cropped_img = self.raw
        else:
            crop = Crop(self.raw, self.platform, self.sensor_info, self.parm_cro)
            cropped_img = crop.execute()

        # =====================================================================
        # Dead pixels correction
        if skip_disabled and not self.parm_dpc["is_enable"]:
            dpc_raw = cropped_img
        else:
            dpc = DPC(cropped_img, self.sensor_info, self.parm_dpc, self.platform)
            dpc_raw = dpc.execute()

        # =====================================================================
        # Black level correction
        if skip_disabled and not self.parm_blc["is_enable"]:
            blc_raw = dpc_raw
        else:
            blc = BLC(dpc_raw, self.platform, self.sensor_info, self.parm_blc)
            blc_raw = blc.execute()

        # =====================================================================
        # decompanding
        if skip_disabled and not self.parm_cmpd["is_enable"]:
            cmpd_raw = blc_raw.astype(np.uint32)
        else:
            cmpd = PWC(blc_raw, self.platform, self.sensor_info, self.parm_cmpd)
            cmpd_raw = cmpd.execute()

        # Store decompanded image for histogram comparison later
        self.decompanded_img = cmpd_raw.copy()

        # =====================================================================
        # OECF
        if skip_disabled and not self.parm_oec.get("is_enable", False):
            oecf_raw = cmpd_raw
        else:
            oecf = OECF(cmpd_raw, self.platform, self.sensor_info, self.parm_oec)
            oecf_raw = oecf.execute()
        oecf_raw = cast(np.ndarray, oecf_raw)

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
            # Digital Gain (receives OECF output per pipeline order: PWC -> OECF -> DG)
            dga = DG(oecf_raw, self.platform, self.sensor_info, self.parm_dga)
            dga_raw, self.dga_current_gain = dga.execute()
            # Keep YAML mirror in sync with any step-mode nudge inside DigitalGain.
            self.parm_dga["current_gain"] = self.dga_current_gain
            if self.c_yaml is not None:
                self.c_yaml["digital_gain"]["current_gain"] = self.dga_current_gain

            # =====================================================================
            # Lens shading correction
            if skip_disabled and not self.parm_lsc.get("is_enable", True):
                lsc_raw = dga_raw
            else:
                lsc = LSC(dga_raw, self.platform, self.sensor_info, self.parm_lsc)
                lsc_raw = lsc.execute()

            # =====================================================================
            # Bayer noise reduction
            if skip_disabled and not self.parm_bnr["is_enable"]:
                bnr_raw = lsc_raw
            else:
                bnr = BNR(lsc_raw, self.sensor_info, self.parm_bnr, self.platform)
                bnr_raw = bnr.execute()

            # =====================================================================
            # Auto White Balance
            awb = AWB(bnr_raw, self.sensor_info, self.parm_awb, self.parm_wbc)
            self.awb_gains = cast(AWBGains, awb.execute())

            # =====================================================================
            # White balancing
            # Use optimized version for better performance
            wbc = WBOPT(bnr_raw, self.platform, self.sensor_info, self.parm_wbc, self.awb_gains)
            wb_raw = wbc.execute()

            # Store current frame for external processing
            self.current_frame = wb_raw

            # =====================================================================
            # HDR tone mapping before Demosaicing
            if self.tone_mapping_before_demosaic:
                tone_mapper = tone_mapping(wb_raw, pipeline_self=cast(ToneMappingContext, self))
                hdr_raw = tone_mapper.execute()
                self.logger.info(f"HDR Image mean: {np.mean(hdr_raw)}")
            else:
                max_val = 2 ** self.sensor_info.get("hdr_bit_depth", 24) - 1
                hdr_raw = (wb_raw.astype(np.float32) * (65535.0 / max_val)).astype(np.uint16)

            # =====================================================================
            # CFA demosaicing
            cfa_inter = Demosaic(cast(RawBayerImage, hdr_raw), self.platform, self.sensor_info, self.parm_dem)
            demos_img = cfa_inter.execute()
            self.logger.info(f"Demosaiced Image mean: {np.mean(demos_img)}")

            # =====================================================================
            # Color correction matrix
            # Use optimized version for better performance
            ccm = CCMOPT(demos_img, self.platform, self.sensor_info, self.parm_ccm)
            ccm_img = ccm.execute()
            self.logger.info(f"CCM Image mean: {np.mean(ccm_img)}")

            # =====================================================================
            # HDR tone mapping after Demosaicing
            if not self.tone_mapping_before_demosaic:
                tone_mapper = tone_mapping(ccm_img, pipeline_self=cast(ToneMappingContext, self))
                CCM_tone_mapped = tone_mapper.execute()
                self.logger.info(f"HDR Image mean: {np.mean(CCM_tone_mapped)}")
                ccm_img = CCM_tone_mapped

            # =====================================================================
            # Auto-Exposure (operates on 16-bit linear RGB before bit conversion)
            # This provides maximum precision for exposure metering
            aef = AE(ccm_img, self.sensor_info, self.parm_ae)
            self.ae_feedback = aef.execute()
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
        self.logger.info(f"Converting 16-bit linear RGB to 8-bit linear RGB for YUV processing")
        self.logger.info(f"  Input range: [{np.min(ccm_img)}, {np.max(ccm_img)}]")
        linear_8bit = np.clip((ccm_img.astype(np.float32) / 65535.0 * 255.0), 0, 255).astype(np.uint8)
        self.logger.info(f"  Output range: [{np.min(linear_8bit)}, {np.max(linear_8bit)}]")

        # =====================================================================
        # Color space conversion (operates on 8-bit linear RGB)
        # ITU-R BT.601/709 standards require linear RGB input for correct color math
        csc = CSC(linear_8bit, self.platform, self.sensor_info, self.parm_csc, self.parm_cse)
        csc_img = csc.execute()
        self.logger.info(f"CSC Image mean: {np.mean(csc_img)}")

        # =====================================================================
        # Local Dynamic Contrast Improvement
        # When post_gamma=true, LDCI is deferred to after gamma on 8-bit RGB
        # (matching boltISP's applyLdciRgb888 placement).
        ldci_post_gamma = bool(self.parm_ldci.get("post_gamma", False))
        if ldci_post_gamma or (skip_disabled and not self.parm_ldci["is_enable"]):
            ldci_img = csc_img
        else:
            ldci = LDCI(
                csc_img,
                self.platform,
                self.sensor_info,
                self.parm_ldci,
                self.parm_csc["conv_standard"],
            )
            ldci_img = ldci.execute()

        # =====================================================================
        # Sharpening
        if skip_disabled and not self.parm_sha["is_enable"]:
            sharp_img = ldci_img
        else:
            sharp = SHARP(
                ldci_img,
                self.platform,
                self.sensor_info,
                self.parm_sha,
                self.parm_csc["conv_standard"],
            )
            sharp_img = sharp.execute()

        # =====================================================================
        # 2d noise reduction
        if skip_disabled and not self.parm_2dn["is_enable"]:
            nr2d_img = sharp_img
        else:
            nr2d = NR2D(
                sharp_img,
                self.sensor_info,
                self.parm_2dn,
                self.platform,
                self.parm_csc["conv_standard"],
            )
            nr2d_img = nr2d.execute()

        # =====================================================================
        # RGB conversion (YUV→RGB, outputs 8-bit linear RGB)
        rgbc = RGBC(nr2d_img, self.platform, self.sensor_info, self.parm_rgb, self.parm_csc)
        rgbc_img = rgbc.execute()
        self.logger.info(f"RGB Conversion output range: [{np.min(rgbc_img)}, {np.max(rgbc_img)}]")

        # =====================================================================
        # Gamma correction (8-bit linear RGB → 8-bit gamma-corrected RGB)
        # IMPORTANT: Applied AFTER YUV processing, as final OETF encoding step
        # This is the correct position per IEC 61966-2-1 (sRGB) and industry standards
        # Differs from Infinite-ISP which applies gamma before CSC (pragmatic but incorrect)
        # Gamma acts as Opto-Electronic Transfer Function (OETF) for display encoding
        # See: GAMMA_CORRECTION_FINAL_SOLUTION.md for detailed rationale
        gmc = GC(rgbc_img, self.platform, self.sensor_info, self.parm_gmc)
        gamma_img = gmc.execute()
        self.logger.info(f"Gamma output range: [{np.min(gamma_img)}, {np.max(gamma_img)}]")

        # =====================================================================
        # Post-gamma LDCI on 8-bit gamma-encoded RGB
        # Runs only when ldci.post_gamma: true (matches boltISP placement)
        if ldci_post_gamma:
            self.logger.info("Applying post-gamma LDCI on 8-bit RGB (boltISP-compatible mode)")
            gamma_img = apply_ldci_rgb8(gamma_img, self.platform, self.sensor_info, self.parm_ldci)
            self.logger.info(f"  Post-gamma LDCI output range: [{np.min(gamma_img)}, {np.max(gamma_img)}]")

        # =====================================================================
        # Scaling
        if skip_disabled and not self.parm_sca["is_enable"]:
            scaled_img = gamma_img
        else:
            scale = Scale(
                gamma_img,
                self.platform,
                self.sensor_info,
                self.parm_sca,
                self.parm_csc["conv_standard"],
            )
            scaled_img = scale.execute()

        # =====================================================================
        # YUV saving format 444, 422 etc
        yuv = YUV_C(scaled_img, self.platform, self.sensor_info, self.parm_yuv)
        yuv_conv = yuv.execute()

        # only to view image if csc is off it does nothing
        out_img = yuv_conv
        out_dim = scaled_img.shape  # dimensions of Output Image

        # Is not part of ISP-pipeline only assists in visualizing output results
        if visualize_output:

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

            self.last_output_rgb = np.asarray(out_rgb).copy()

            # Plot histograms if enabled (debug feature)
            if self.platform.get("plot_histograms", False):
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

            util.save_pipeline_output(
                self.out_file, out_rgb, self.c_yaml, out_file_name, self.output_path, short_names=short_names
            )

    def execute(
        self,
        img_path: str | None = None,
        load_method: str = "auto",
        byte_order: ByteOrder = "little",
    ) -> None:
        """
        Start execution of Brilliant-ISP

        Args:
            img_path (str): Optional path to image file
            load_method (str): 'auto', '3byte', 'uint16', or 'original'
            byte_order (str): 'little' or 'big'
            reverse_uint32 (bool): If True, reverse byte order within uint32 pixel values
        """
        if self.c_yaml is None:
            raise RuntimeError("Configuration must be loaded before execution.")
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
