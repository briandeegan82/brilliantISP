"""
Hable (Uncharted 2) integer/LUT tone mapping for production-style ISPs.

Uses precomputed LUT for the Hable filmic curve.
No float ops in the hot path - suitable for hardware implementation.
"""
import numpy as np
from util.debug_utils import get_debug_logger


def _hable_partial(x: np.ndarray, A=0.15, B=0.50, C=0.10, D=0.20, E=0.02, F=0.30) -> np.ndarray:
    """Uncharted 2 partial: ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F)) - E/F"""
    num = x * (A * x + C * B) + D * E
    den = np.maximum(x * (A * x + B) + D * F, 1e-10)
    return num / den - E / F


def _build_hable_lut(size=65536, exposure_bias=2.0, white_point=11.2, hdr_scale=1.0):
    """
    Build Hable filmic tone curve LUT.
    Input [0, hdr_scale] maps to [0, 1]; LUT index = normalized input.
    Lower hdr_scale = more highlight compression.
    """
    x_norm = np.linspace(0, 1, size, dtype=np.float64)
    x = x_norm * hdr_scale
    scaled = x * exposure_bias
    curr = _hable_partial(scaled)
    w_val = _hable_partial(np.array([white_point], dtype=np.float64))[0]
    white_scale = 1.0 / max(w_val, 1e-10)
    out = np.clip(curr * white_scale, 0.0, 1.0)
    return np.clip(np.round(out * 65535), 0, 65535).astype(np.uint16)


class HableIntegerToneMapping:
    """
    Hable tone mapping via precomputed LUT.
    Integer input → LUT lookup → integer output.
    """

    _lut_cache = {}

    def __init__(self, img, platform, sensor_info, params):
        self.img = np.asarray(img, dtype=np.uint32)
        self.platform = platform
        self.sensor_info = sensor_info
        self.params = params
        self.logger = get_debug_logger("HableIntegerToneMapping", config=platform)

        self.is_enable = params.get("is_enable", True)
        self.is_save = params.get("is_save", False)
        self.exposure_bias = params.get("exposure_bias", params.get("exposure_adjust", 2.0))
        self.white_point = params.get("white_point", 11.2)
        self.use_normalization = params.get("use_normalization", True)
        self.hdr_scale = params.get("hdr_scale", 1.0)

        input_bits = sensor_info.get("hdr_bit_depth", 24)
        if self.img.size > 0 and self.img.max() <= 65535:
            input_bits = 16
        self.input_max = 2**input_bits - 1
        self.output_max = 65535
        self.lut_size = 65536

        self._build_lut()

    def _build_lut(self):
        cache_key = (self.lut_size, self.exposure_bias, self.white_point, self.hdr_scale)
        if cache_key not in self._lut_cache:
            self._lut_cache[cache_key] = _build_hable_lut(
                self.lut_size, self.exposure_bias, self.white_point, self.hdr_scale
            )
        self.lut = self._lut_cache[cache_key]

    def _apply_curve(self, x: np.ndarray) -> np.ndarray:
        """Apply Hable curve via LUT lookup."""
        x = np.asarray(x, dtype=np.int64)

        if self.use_normalization:
            img_min = int(np.min(x))
            img_max = int(np.max(x))
            range_val = max(1, img_max - img_min)
            idx = ((x - img_min) * (self.lut_size - 1)) // range_val
        else:
            idx = (x * (self.lut_size - 1)) // max(1, self.input_max)

        idx = np.clip(idx, 0, self.lut_size - 1)
        return self.lut[idx].astype(np.uint16)

    def execute(self) -> np.ndarray:
        """Execute Hable integer tone mapping. Returns uint16."""
        if not self.is_enable:
            scale = self.output_max / max(1, self.input_max)
            return (self.img.astype(np.float64) * scale).astype(np.uint16)

        return self._apply_curve(self.img)
