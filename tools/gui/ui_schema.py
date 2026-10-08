"""ISP Config window section order and known enum fields."""

ISP_SECTION_ORDER = [
    "platform",
    "sensor_info",
    "crop",
    "dead_pixel_correction",
    "black_level_correction",
    "companding",
    "oecf",
    "digital_gain",
    "lens_shading_correction",
    "bayer_noise_reduction",
    "auto_white_balance",
    "white_balance",
    "tone_mapping",
    "reinhard_integer",
    "aces_integer",
    "hdr_durand",
    "hable",
    "hable_integer",
    "aces",
    "demosaic",
    "color_correction_matrix",
    "auto_exposure",
    "color_space_conversion",
    "color_saturation_enhancement",
    "ldci",
    "sharpen",
    "2d_noise_reduction",
    "rgb_conversion",
    "gamma_correction",
    "scale",
    "yuv_conversion_format",
]

ENUM_CHOICES: dict[str, tuple[str, ...]] = {
    "sensor_info.bayer_pattern": ("rggb", "grbg", "gbrg", "bggr"),
    "sensor_info.endian_type": ("ieee-le", "ieee-be"),
    "sensor_info.data_format": ("uint8", "uint16", "uint32"),
    "demosaic.algorithm": (
        "bilinear",
        "malvar",
        "vng_opt",
        "hamilton_adams",
        "ppg",
        "ahd",
        "lmmse",
    ),
    "tone_mapping.tone_mapper": (
        "reinhard_integer",
        "aces",
        "aces_integer",
        "hable",
        "hable_integer",
        "durand",
    ),
    "gamma_correction.curve": ("srgb", "gamma"),
    "auto_white_balance.algorithm": ("grey_world", "norm_2", "pca"),
    "yuv_conversion_format.conv_type": ("444", "422"),
    "digital_gain.exposure_correction_mode": ("direct", "step"),
    "auto_exposure.exposure_correction_mode": ("direct", "step"),
    "aces.variant": ("fitted", "canonical"),
    "aces.output_transform": ("sRGB", "Rec709", "none"),
    "scale.algorithm": ("Nearest_Neighbor", "Bilinear"),
    "scale.upscale_method": ("Nearest_Neighbor", "Bilinear"),
    "scale.downscale_method": ("Nearest_Neighbor", "Bilinear"),
    "platform.debug_log_level": ("DEBUG", "INFO", "WARNING", "ERROR"),
    "platform.save_format": ("png", "jpg", "tiff"),
}

RGGB_START_LABELS = (
    "0 (R at 0,0) — RGGB",
    "1 (G at 0,0, R) — GRBG",
    "2 (G at 0,0, B) — GBRG",
    "3 (B at 0,0) — BGGR",
)

CFA_TILES: dict[str, tuple[tuple[str, str], tuple[str, str]]] = {
    "rggb": (("R", "G"), ("G", "B")),
    "grbg": (("G", "R"), ("B", "G")),
    "gbrg": (("G", "B"), ("R", "G")),
    "bggr": (("B", "G"), ("G", "R")),
}

CFA_COLORS = {"R": "#cc3333", "G": "#2e8b2e", "B": "#3355cc"}
