from util.debug_utils import get_debug_logger
"""
File: gamma_correction.py
Description: Gamma correction with optional sRGB curve.
Supports: curve='gamma' (power) or curve='srgb' (IEC 61966-2-1).
"""
import time
import numpy as np

from util.utils import save_output_array


def _srgb_oetf(x):
    """
    sRGB OETF (linear to display). IEC 61966-2-1.
    x <= 0.0031308: 12.92 * x
    x > 0.0031308:  1.055 * x^(1/2.4) - 0.055
    """
    linear = x <= 0.0031308
    out = np.empty_like(x, dtype=np.float64)
    out[linear] = 12.92 * x[linear]
    out[~linear] = 1.055 * np.power(x[~linear], 1.0 / 2.4) - 0.055
    return np.clip(out, 0.0, 1.0)


class GammaCorrection:
    """
    Gamma correction. curve='gamma' (power) or curve='srgb'.
    """

    def __init__(self, img, platform, sensor_info, parm_gmm):
        self.img = img
        self.enable = parm_gmm["is_enable"]
        self.sensor_info = sensor_info
        self.output_bit_depth = sensor_info["output_bit_depth"]
        self.parm_gmm = parm_gmm
        self.is_save = parm_gmm["is_save"]
        self.platform = platform
        self.curve = parm_gmm.get("curve", "gamma").lower()
        self.gamma = parm_gmm.get("gamma", 2.2)
        self.logger = get_debug_logger("GammaCorrection", config=self.platform)

    def generate_gamma_lut(self, bit_depth):
        """Generate LUT: 'gamma' = power curve, 'srgb' = sRGB OETF."""
        max_val = 2**bit_depth - 1
        x = np.arange(0, max_val + 1, dtype=np.float64) / max_val

        if self.curve == "srgb":
            out = _srgb_oetf(x)
        else:
            out = np.power(np.maximum(x, 0), 1.0 / self.gamma)

        lut = np.clip(np.round(out * max_val), 0, max_val).astype(np.uint16)
        return lut

    def apply_gamma(self):
        """
        Apply Gamma LUT on n-bit Image.
        Input: 16-bit RGB (pipeline convention from CCM/tone mapping).
        """
        input_bit_depth = self.sensor_info.get("pipeline_rgb_bit_depth", 16)
        input_max = 2**input_bit_depth - 1
        lut = self.generate_gamma_lut(input_bit_depth).T

        # apply LUT
        gamma_img = lut[self.img]
        if self.output_bit_depth == 8:
            gamma_img = np.clip(
                (gamma_img.astype(np.float32) / input_max * 255), 0, 255
            ).astype(np.uint8)
            return gamma_img
        elif self.output_bit_depth == 16:
            return gamma_img
        elif self.output_bit_depth == 32:
            return gamma_img.astype(np.float32)
        else:
            raise ValueError("Unsupported output bit depth. Use 8, 16, or 32.")

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_gamma_correction_",
                self.platform,
                self.sensor_info["output_bit_depth"],
                self.sensor_info["bayer_pattern"],
            )

    def execute(self):
        """
        Exceute Gamma Correction
        """
        self.logger.info(f"Gamma Correction = {self.enable}")
        if self.enable is True:
            start = time.time()
            gc_out = self.apply_gamma()
            self.logger.info(f"  Execution time: {time.time() - start:.3f}s")
            self.img = gc_out

        self.save()
        return self.img
