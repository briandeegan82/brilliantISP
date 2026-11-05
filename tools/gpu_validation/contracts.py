from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NumericGate:
    name: str
    atol: float = 0.0
    rtol: float = 0.0
    max_code_error: int | None = None
    mean_code_error: float | None = None


EXACT = NumericGate("exact")
FLOAT32_NORMALIZED = NumericGate("float32_normalized", atol=2e-6, rtol=2e-5)
FLOAT64_HELPER = NumericGate("float64_helper", atol=1e-12, rtol=1e-10)
STATISTICS = NumericGate("statistics", atol=1e-10, rtol=1e-8)
FINAL_INTEGER = NumericGate("final_integer", max_code_error=1, mean_code_error=0.1)
FINAL_FLOAT = FLOAT32_NORMALIZED

DEMOSAIC_ALGORITHMS = (
    "bilinear", "malvar", "vng", "vng_opt", "hamilton_adams",
    "hamilton_adams_opt", "ppg", "ppg_opt", "lmmse", "lmmse_opt",
    "lmmse_fast", "ahd", "ahd_opt",
)
TONE_PAIRINGS = (
    ("aces_fitted", "float"), ("aces_fitted", "lut"),
    ("hable", "float"), ("hable", "lut"), ("reinhard", "float"),
    ("reinhard", "lut"), ("reinhard", "fixed"), ("durand", "cpu"),
)
INTEGER_STAGES = frozenset((
    "crop", "dead_pixel_correction", "black_level_correction", "green_repair",
    "companding", "oecf",
))
CODE_FLOAT_STAGES = frozenset((
    "digital_gain", "lens_shading_correction", "bayer_noise_reduction", "white_balance",
))
NORMALIZED_STAGES = frozenset((
    "demosaic", "color_correction_matrix", "linear_rgb_pre_tone_mapping",
    "tone_mapping_pre", "demosaic_render", "color_correction_matrix_render",
    "tone_mapping_post", "saturation", "ldci_pre_transfer", "transfer_encode",
    "ldci_post_transfer", "sharpen", "noise_reduction_2d", "scale",
))


def stage_gate(stage: str, *, logical_integer: bool = True) -> NumericGate:
    if stage in INTEGER_STAGES:
        return EXACT if logical_integer else FLOAT32_NORMALIZED
    if stage in CODE_FLOAT_STAGES or stage in NORMALIZED_STAGES:
        return FLOAT32_NORMALIZED
    if stage == "auto_white_balance":
        return STATISTICS
    if stage == "auto_exposure":
        return EXACT
    if stage == "output":
        return FINAL_INTEGER if logical_integer else FINAL_FLOAT
    raise ValueError(f"no numerical contract for stage {stage!r}")
