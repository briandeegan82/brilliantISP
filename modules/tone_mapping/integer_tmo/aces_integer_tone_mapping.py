"""
ACES integer/LUT tone mapping for production-style ISPs.

Uses precomputed LUTs for the ACES filmic curve and sRGB gamma.
No float ops in the hot path - suitable for hardware implementation.
Based on ACES 1.0 RRT (Knarkowicz 2016 fitted) + sRGB ODT.
"""
import numpy as np
from util.debug_utils import get_debug_logger


def _build_aces_rrt_lut(size=65536, hdr_scale=100.0):
    """
    Build ACES RRT (filmic) tone curve LUT.
    Formula: (x * (a*x + b)) / (x * (c*x + d) + e), Knarkowicz 2016.
    LUT index = normalized [0,1] maps to curve input [0, hdr_scale].
    Lower hdr_scale = more highlight compression (darker). Typical: 10-100.
    """
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    x_norm = np.linspace(0, 1, size, dtype=np.float64)
    x = x_norm * hdr_scale
    out = (x * (a * x + b)) / (np.maximum(x * (c * x + d) + e, 1e-8))
    return np.clip(np.round(out * 65535), 0, 65535).astype(np.uint16)


def _build_srgb_gamma_lut(size=65536, gamma=2.4):
    """
    Build sRGB gamma LUT (linear to display).
    Linear segment: x <= 0.0031308 -> 12.92 * x
    Power segment: x > 0.0031308 -> 1.055 * x^(1/gamma) - 0.055
    """
    x = np.linspace(0, 1, size, dtype=np.float64)
    linear = x <= 0.0031308
    out = np.zeros(size, dtype=np.float64)
    out[linear] = 12.92 * x[linear]
    out[~linear] = 1.055 * np.power(x[~linear], 1.0 / gamma) - 0.055
    return np.clip(np.round(out * 65535), 0, 65535).astype(np.uint16)


class ACESIntegerToneMapping:
    """
    ACES tone mapping via precomputed LUTs.
    Integer input → LUT lookup → integer output.
    """

    # Class-level LUT cache (shared across instances)
    _rrt_lut_cache = {}
    _srgb_lut_cache = {}

    def __init__(self, img, platform, sensor_info, params):
        self.img = np.asarray(img, dtype=np.uint32)
        self.platform = platform
        self.sensor_info = sensor_info
        self.params = params
        self.logger = get_debug_logger("ACESIntegerToneMapping", config=platform)

        self.is_enable = params.get("is_enable", True)
        self.is_save = params.get("is_save", False)
        self.exposure = params.get("exposure_adjustment", params.get("exposure_adjust", 0.0))
        self.gamma = params.get("gamma", 2.4)
        self.apply_gamma = params.get("apply_odt_gamma", True)
        self.use_normalization = params.get("use_normalization", True)  # Per-image min-max
        self.hdr_scale = params.get("hdr_scale", 1.0)  # Lower = more compression

        input_bits = sensor_info.get("hdr_bit_depth", 24)
        if self.img.size > 0 and self.img.max() <= 65535:
            input_bits = 16
        self.input_max = 2**input_bits - 1
        self.output_max = 65535
        self.lut_size = 65536

        self._build_luts()

    def _build_luts(self):
        """Build or retrieve cached LUTs."""
        gamma_key = (self.lut_size, self.gamma)
        if gamma_key not in self._srgb_lut_cache:
            self._srgb_lut_cache[gamma_key] = _build_srgb_gamma_lut(
                self.lut_size, self.gamma
            )
        rrt_key = (self.lut_size, self.hdr_scale)
        if rrt_key not in self._rrt_lut_cache:
            self._rrt_lut_cache[rrt_key] = _build_aces_rrt_lut(
                self.lut_size, self.hdr_scale
            )

        self.rrt_lut = self._rrt_lut_cache[(self.lut_size, self.hdr_scale)]
        self.srgb_lut = self._srgb_lut_cache[gamma_key]

    def _apply_curve(self, x: np.ndarray) -> np.ndarray:
        """
        Apply ACES RRT + optional sRGB gamma via LUT lookup.
        Per-image normalization (like float ACES) prevents washed-out output.
        """
        x = np.asarray(x, dtype=np.int64)

        # Exposure: scale by 2^exp (integer)
        if abs(self.exposure) > 1e-6:
            exp_scale = int(round((2.0 ** self.exposure) * 65536))
            x = (x * exp_scale) >> 16
            x = np.clip(x, 0, self.input_max)

        # Map input to LUT index: per-image norm (matches float ACES) or absolute
        if self.use_normalization:
            img_min = int(np.min(x))
            img_max = int(np.max(x))
            range_val = max(1, img_max - img_min)
            # Normalized [0,1]: (x - min) / range
            idx = ((x - img_min) * (self.lut_size - 1)) // range_val
        else:
            if self.input_max > 0:
                idx = (x * (self.lut_size - 1)) // self.input_max
            else:
                idx = np.zeros_like(x, dtype=np.int64)
        idx = np.clip(idx, 0, self.lut_size - 1)

        out = self.rrt_lut[idx]

        if self.apply_gamma:
            out = self.srgb_lut[out]

        return out.astype(np.uint16)

    def execute(self) -> np.ndarray:
        """Execute ACES integer tone mapping. Returns uint16."""
        if not self.is_enable:
            scale = self.output_max / max(1, self.input_max)
            return (self.img.astype(np.float64) * scale).astype(np.uint16)

        return self._apply_curve(self.img)
