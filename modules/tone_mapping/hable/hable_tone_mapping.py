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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


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
        self.is_plot_curve = params.get("is_plot_curve", False)

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

    def plot_tone_curve(self):
        """Plot and save the Hable tone mapping curve."""
        if not self.is_plot_curve:
            return
        
        try:
            # Generate input range (extended to show curve behavior)
            x = np.linspace(0, 2.0, 1000)
            y = self._apply_curve(x)
            
            # Create plot with actual values (not normalized)
            plt.figure(figsize=(10, 7))
            plt.plot(x, y, 'b-', linewidth=2, label='Hable (Uncharted 2) Filmic Curve')
            plt.plot([0, 1], [0, 1], 'r--', linewidth=1, alpha=0.5, label='Linear (no tone mapping)')
            plt.axvline(x=1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Input=1.0')
            plt.grid(True, alpha=0.3)
            plt.xlabel('Input (0 to 2.0)', fontsize=12)
            plt.ylabel('Output (0 to 1)', fontsize=12)
            plt.title(f'Hable Tone Mapping Curve\n(exposure_bias={self.exposure_bias}, white_point={self.white_point})', fontsize=14)
            plt.legend(fontsize=10)
            plt.xlim([0, 2.0])
            plt.ylim([0, 1.1])
            
            # Save plot
            output_dir = self.platform.get('output_dir', 'module_output')
            os.makedirs(output_dir, exist_ok=True)
            plot_filename = os.path.join(output_dir, 'tone_curve_hable.png')
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"  Tone mapping curve saved to: {plot_filename}")
            
        except Exception as e:
            self.logger.warning(f"  Failed to plot tone curve: {e}")

    def execute(self):
        """Execute Hable tone mapping. Returns [0,1] luminance."""
        if not self.is_enable:
            return np.clip(self.img, 0.0, 1.0)

        self.logger.info("Executing Hable (Uncharted 2) Tone Mapping...")
        start = time.time()

        # Plot the curve if debug option is enabled
        self.plot_tone_curve()
        
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
