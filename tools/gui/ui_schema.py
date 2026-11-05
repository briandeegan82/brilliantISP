"""ISP Config window section order and known enum fields."""

from util.config_schema import spec_map

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

_SPEC_MAP = spec_map()

ENUM_CHOICES: dict[str, tuple[str, ...]] = {
    path: spec.choices for path, spec in _SPEC_MAP.items() if spec.choices
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
