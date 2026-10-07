from __future__ import annotations

from typing import Any, Literal, Protocol, TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray

BayerPattern: TypeAlias = Literal["rggb", "grbg", "gbrg", "bggr"]
ByteOrder: TypeAlias = Literal["little", "big"]
ToneMapperName: TypeAlias = Literal[
    "durand",
    "aces",
    "reinhard_integer",
    "aces_integer",
    "hable",
    "hable_integer",
]

UInt8Image: TypeAlias = NDArray[np.uint8]
UInt16Image: TypeAlias = NDArray[np.uint16]
UInt32Image: TypeAlias = NDArray[np.uint32]
Float32Image: TypeAlias = NDArray[np.float32]
Float64Image: TypeAlias = NDArray[np.float64]

RawBayerImage: TypeAlias = UInt16Image | UInt32Image
RGBImage: TypeAlias = UInt8Image | UInt16Image | Float32Image
YUVImage: TypeAlias = UInt8Image
AWBGains: TypeAlias = tuple[float, float] | tuple[float, float, float, float]
GenericConfig: TypeAlias = dict[str, Any]
DemosaicMasks: TypeAlias = tuple[np.ndarray, np.ndarray, np.ndarray]


class RequiredPlatformConfig(TypedDict):
    filename: str
    disable_progress_bar: bool
    leave_pbar_string: bool
    render_3a: bool
    generate_tv: bool
    save_format: str
    short_output_names: bool
    debug_enabled: bool
    output_dir: str
    in_file: str
    out_file: str
    rgb_output: bool


class PlatformConfig(RequiredPlatformConfig, total=False):
    debug_log_level: str
    debug_log_file: str | None
    plot_histograms: bool
    histogram_show_log: bool
    histogram_show_channels: bool
    skip_disabled_modules: bool


class RequiredSensorInfo(TypedDict):
    bayer_pattern: BayerPattern
    hdr_bit_depth: int
    bit_depth: int
    width: int
    height: int
    output_bit_depth: int
    pipeline_rgb_bit_depth: int


class SensorInfo(RequiredSensorInfo, total=False):
    sensor: str
    endian_type: str
    data_format: str
    data_alignment: str
    bit_alignment: str
    manual_bit_shift: int
    horizontal_flip: bool
    vertical_flip: bool
    embedded_rows_top: int
    embedded_rows_bottom: int
    orig_size: str


class ParsedFileNameInfo(TypedDict):
    width: int
    height: int
    bit_depth: int
    bayer_pattern: BayerPattern


class ExtractedRawMetadata(ParsedFileNameInfo, total=False):
    black_level: tuple[int, int, int, int]
    white_level: int
    wb_gains: tuple[float, float, float, float]
    ccm: NDArray[np.floating] | None


class DigitalGainConfig(TypedDict):
    is_debug: bool
    is_auto: bool
    is_enable: bool
    gain_array: list[float]
    current_gain: int
    ae_feedback: int | None
    is_save: bool


class WhiteBalanceConfig(TypedDict):
    is_enable: bool
    is_debug: bool
    is_auto: bool
    r_gain: float
    b_gain: float
    is_save: bool


class RequiredGammaCorrectionConfig(TypedDict):
    is_enable: bool
    is_save: bool


class GammaCorrectionConfig(RequiredGammaCorrectionConfig, total=False):
    curve: str
    gamma: float


class ColorSpaceConversionConfig(TypedDict):
    conv_standard: int
    is_save: bool


class ColorSaturationEnhancementConfig(TypedDict):
    is_enable: bool
    saturation_gain: float


class LDCIConfig(TypedDict):
    is_enable: bool
    clip_limit: float
    wind: int
    is_save: bool


class SharpenConfig(TypedDict):
    is_enable: bool
    sharpen_sigma: float
    sharpen_strength: float
    is_save: bool


class NoiseReduction2DConfig(TypedDict):
    is_enable: bool
    window_size: int
    patch_size: int
    wts: int
    is_save: bool


class RGBConversionConfig(TypedDict):
    is_enable: bool
    is_save: bool


class CropConfig(TypedDict):
    is_enable: bool
    is_debug: bool
    crop_x_start: int
    crop_y_start: int
    new_width: int
    new_height: int
    is_save: bool


class YUVConversionFormatConfig(TypedDict):
    is_enable: bool
    conv_type: str
    is_save: bool


class ScaleConfig(TypedDict):
    is_enable: bool
    is_debug: bool
    new_width: int
    new_height: int
    is_hardware: bool
    algorithm: str
    upscale_method: str
    downscale_method: str
    is_save: bool


class ColorCorrectionMatrixConfig(TypedDict):
    is_enable: bool
    is_save: bool
    corrected_red: list[float]
    corrected_green: list[float]
    corrected_blue: list[float]


class AutoExposureConfig(TypedDict):
    is_enable: bool
    is_debug: bool
    center_illuminance: float
    histogram_skewness: float


class DeadPixelCorrectionConfig(TypedDict):
    is_enable: bool
    dp_threshold: int
    is_debug: bool
    is_save: bool


class BlackLevelCorrectionConfig(TypedDict):
    is_enable: bool
    r_offset: int
    gr_offset: int
    gb_offset: int
    b_offset: int
    is_linear: bool
    r_sat: int
    gr_sat: int
    gb_sat: int
    b_sat: int
    is_save: bool


class LensShadingCorrectionConfig(TypedDict, total=False):
    is_enable: bool
    is_save: bool
    r_k1: float
    r_k2: float
    gr_k1: float
    gr_k2: float
    gb_k1: float
    gb_k2: float
    b_k1: float
    b_k2: float


class BayerNoiseReductionConfig(TypedDict):
    is_enable: bool
    filter_window: int
    r_std_dev_s: float
    r_std_dev_r: float
    g_std_dev_s: float
    g_std_dev_r: float
    b_std_dev_s: float
    b_std_dev_r: float
    is_save: bool


class RequiredDemosaicConfig(TypedDict):
    is_save: bool


class DemosaicConfig(RequiredDemosaicConfig, total=False):
    algorithm: str


class ToneMappingConfig(TypedDict):
    is_enable: bool
    tone_mapping_before_demosaic: bool
    tone_mapper: ToneMapperName
    is_save: bool


class ToneMappingParams(TypedDict, total=False):
    is_enable: bool
    is_save: bool
    is_debug: bool
    is_plot_curve: bool
    exposure_adjust: float
    exposure_adjustment: float
    gamma: float
    white_point: float
    surround: str
    sigma_space: float
    sigma_color: float
    contrast_factor: float
    downsample_factor: int
    knee: float
    strength: float
    dark_enh: float
    normalize_output: bool
    apply_odt_gamma: bool
    use_normalization: bool
    hdr_scale: float
    exposure_bias: float


class ToneMappingContext(Protocol):
    platform: PlatformConfig
    sensor_info: SensorInfo
    tone_mapping: ToneMappingConfig
    tone_mapping_before_demosaic: bool
    param_durand: ToneMappingParams
    logger: Any

    param_aces: ToneMappingParams
    param_integer_tmo: ToneMappingParams
    param_aces_integer: ToneMappingParams
    param_hable: ToneMappingParams
    param_hable_integer: ToneMappingParams


RequiredPipelineConfig = TypedDict(
    "RequiredPipelineConfig",
    {
        "platform": PlatformConfig,
        "sensor_info": SensorInfo,
        "dead_pixel_correction": DeadPixelCorrectionConfig,
        "companding": GenericConfig,
        "digital_gain": DigitalGainConfig,
        "lens_shading_correction": LensShadingCorrectionConfig,
        "bayer_noise_reduction": BayerNoiseReductionConfig,
        "black_level_correction": BlackLevelCorrectionConfig,
        "white_balance": WhiteBalanceConfig,
        "auto_white_balance": GenericConfig,
        "demosaic": DemosaicConfig,
        "auto_exposure": AutoExposureConfig,
        "color_correction_matrix": ColorCorrectionMatrixConfig,
        "gamma_correction": GammaCorrectionConfig,
        "hdr_durand": ToneMappingParams,
        "tone_mapping": ToneMappingConfig,
        "color_space_conversion": ColorSpaceConversionConfig,
        "color_saturation_enhancement": ColorSaturationEnhancementConfig,
        "ldci": LDCIConfig,
        "sharpen": SharpenConfig,
        "2d_noise_reduction": NoiseReduction2DConfig,
        "rgb_conversion": RGBConversionConfig,
        "scale": ScaleConfig,
        "crop": CropConfig,
        "yuv_conversion_format": YUVConversionFormatConfig,
    },
)


class PipelineConfig(RequiredPipelineConfig, total=False):
    oecf: GenericConfig
    aces: ToneMappingParams
    reinhard_integer: ToneMappingParams
    integer_tmo: ToneMappingParams
    aces_integer: ToneMappingParams
    hable: ToneMappingParams
    hable_integer: ToneMappingParams
