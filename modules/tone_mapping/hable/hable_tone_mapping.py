# -*- coding: utf-8 -*-
"""
Hable (Uncharted 2) filmic tone mapping implementation.

Classic filmic curve from John Hable / Naughty Dog (Uncharted 2).
Based on Hejl-Burgess-Dawson rational approximation; provides
crisp blacks, soft shoulder for highlights.
Input: Luminance in [0,1] (float).
Output: [0,1] for pipeline.
"""
import numpy as np
import time
from util.debug_utils import get_debug_logger
from util.utils import save_output_array


def _hable_partial(x: np.ndarray, A=0.15, B=0.50, C=0.10, D=0.20, E=0.02, F=0.30) -> np.ndarray:
    """
    Uncharted 2 partial curve: ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F)) - E/F
    """
    num = x * (A * x + C * B) + D * E
    den = x * (A * x + B) + D * F
    den = np.maximum(den, 1e-10)
    return num / den - E / F


class HableToneMapping:
    """
    Hable / Uncharted 2 filmic tone mapping.
    Operates on luminance; preserves chromaticity.
    """

    def __init__(self, img, platform, sensor_info, params):
        self.img = img.astype(np.float32).copy()
        self.is_enable = params.get("is_enable", True)
        self.is_save = params.get("is_save", False)
        self.is_debug = params.get("is_debug", False)

        self.exposure_bias = params.get("exposure_bias", params.get("exposure_adjust", 2.0))
        self.white_point = params.get("white_point", 11.2)  # Hable default
        self.output_bit_depth = sensor_info.get("output_bit_depth", 8)
        self.sensor_info = sensor_info
        self.platform = platform

        self.logger = get_debug_logger("HableToneMapping", config=self.platform)

    def _apply_curve(self, x: np.ndarray) -> np.ndarray:
        """Apply Hable filmic curve."""
        scaled = np.maximum(x * self.exposure_bias, 0.0)
        curr = _hable_partial(scaled)
        # White scale so that white_point maps to 1.0
        w_val = _hable_partial(np.array([self.white_point], dtype=np.float32))[0]
        white_scale = 1.0 / max(w_val, 1e-10)
        return np.clip(curr * white_scale, 0.0, 1.0)

    def execute(self):
        """Execute Hable tone mapping. Returns [0,1] luminance."""
        if not self.is_enable:
            return np.clip(self.img, 0.0, 1.0)

        self.logger.info("Executing Hable (Uncharted 2) Tone Mapping...")
        start = time.time()

        self.img = self._apply_curve(self.img)

        execution_time = time.time() - start
        self.logger.info(f"  Execution time: {execution_time:.3f}s")

        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_hable_tonemapped_",
                self.platform,
                self.sensor_info.get("bit_depth", 8),
                self.sensor_info.get("bayer_pattern", "RGGB"),
            )
        return self.img
