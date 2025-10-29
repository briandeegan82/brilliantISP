from util.debug_utils import get_debug_logger
"""
File: gamma_correction.py
Description: Implements the gamma look up table provided in the config file
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import numpy as np

from util.utils import save_output_array


class GammaCorrection:
    """
    Gamma Correction
    """

    def __init__(self, img, platform, sensor_info, parm_gmm):
        self.img = img
        self.enable = parm_gmm["is_enable"]
        self.sensor_info = sensor_info
        self.output_bit_depth = sensor_info["output_bit_depth"]
        self.parm_gmm = parm_gmm
        self.is_save = parm_gmm["is_save"]
        self.platform = platform
        # Initialize debug logger
        self.logger = get_debug_logger("GammaCorrection", config=self.platform)

    def generate_gamma_lut(self, bit_depth):
        """
        Generates Gamma LUT for 8bit Image
        """
        max_val = 2**bit_depth - 1
        lut = np.arange(0, max_val + 1)
        lut = np.uint16(np.round(max_val * ((lut / max_val) ** (1 / 2.2))))
        return lut

    def apply_gamma(self):
        """
        Apply Gamma LUT on n-bit Image
        """
        # generate LUT
        # lut = self.generate_gamma_lut(self.output_bit_depth).T
        lut = self.generate_gamma_lut(16).T #bit_depth (input)=16

        # apply LUT
        gamma_img = lut[self.img]
        if self.output_bit_depth == 8:
            gamma_img = np.clip((gamma_img.astype(np.float32) / 65535 * 255), 0, 255).astype(np.uint8)
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
