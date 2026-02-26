"""
Integer-native tone mapping for production-style ISPs.

Operates directly on integer input (uint32/uint16) without float normalization.
Uses rational curve (Reinhard-style) with scaled integer arithmetic.
Similar in spirit to Arm Mali Iridix - parametric curve, hardware-friendly.
"""
import numpy as np
from util.debug_utils import get_debug_logger


class IntegerToneMapping:
    """
    Integer-domain tone mapping using rational curve.
    No float conversion - suitable for fixed-point / hardware implementation.
    """

    def __init__(self, img, platform, sensor_info, params):
        self.img = np.asarray(img, dtype=np.uint32)  # Luminance or raw (single channel)
        self.platform = platform
        self.sensor_info = sensor_info
        self.params = params
        self.logger = get_debug_logger("IntegerToneMapping", config=platform)

        self.is_enable = params.get("is_enable", True)
        self.is_save = params.get("is_save", False)

        # Rational curve: out = (in * out_max) / (in + knee)
        # knee controls white point: larger = more compression, darker
        self.knee = params.get("knee", 0.25)
        self.strength = params.get("strength", 1.0)
        self.dark_enh = params.get("dark_enh", 0.0)  # Shadow lift (0-1)

        input_bits = sensor_info.get("hdr_bit_depth", 24)
        if self.img.max() <= 65535:
            input_bits = 16
        self.input_max = 2**input_bits - 1
        self.output_max = 65535  # Pipeline convention

    def _apply_curve(self, x: np.ndarray) -> np.ndarray:
        """
        Apply rational tone curve in integer domain.
        out = (x * out_max) / (x + knee * input_max / strength)
        """
        x = np.asarray(x, dtype=np.int64)
        # Avoid div by zero
        knee_term = max(1, int(self.knee * self.input_max / self.strength))
        out = (x * self.output_max) // (x + knee_term)

        return np.clip(out, 0, self.output_max).astype(np.uint16)

    def execute(self) -> np.ndarray:
        """Execute integer tone mapping. Returns uint16."""
        if not self.is_enable:
            # Passthrough: scale to output range
            scale = self.output_max / max(1, self.input_max)
            return (self.img.astype(np.float64) * scale).astype(np.uint16)

        return self._apply_curve(self.img)
