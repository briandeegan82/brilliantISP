from util.debug_utils import get_debug_logger
"""
File: color_correction_matrix_optimized.py
Description: Optimized color correction matrix using vectorized operations
Code / Paper  Reference: https://www.imatest.com/docs/colormatrix/
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import numpy as np

from util.utils import save_output_array


class ColorCorrectionMatrixOptimized:
    "Apply the color correction 3x3 matrix with optimized vectorized operations"

    def __init__(self, img, platform, sensor_info, parm_ccm):
        self.img = img
        self.enable = parm_ccm["is_enable"]
        self.sensor_info = sensor_info
        self.parm_ccm = parm_ccm
        self.output_bit_depth = sensor_info["output_bit_depth"]
        self.ccm_mat = None
        self.is_save = parm_ccm["is_save"]
        self.platform = platform
        # Initialize debug logger
        self.logger = get_debug_logger("ColorCorrectionMatrixOptimized", config=self.platform)

    def apply_ccm_optimized(self):
        """
        Apply CCM Params with optimized vectorized operations
        """
        # OPTIMIZATION: Pre-compute the CCM matrix once
        r_1 = np.array(self.parm_ccm["corrected_red"], dtype=np.float32)
        r_2 = np.array(self.parm_ccm["corrected_green"], dtype=np.float32)
        r_3 = np.array(self.parm_ccm["corrected_blue"], dtype=np.float32)

        self.ccm_mat = np.float32([r_1, r_2, r_3])

        # OPTIMIZATION: Use vectorized normalization
        # normalize nbit to 0-1 img
        # self.img = np.float32(self.img) / (2**self.output_bit_depth - 1)
        self.img = np.float32(self.img) / 65535 #(2**16 - 1)
        # OPTIMIZATION: Use efficient reshape and matrix multiplication
        # convert to nx3 - use reshape with -1 for automatic dimension calculation
        img1 = self.img.reshape(-1, 3)

        # OPTIMIZATION: Use optimized matrix multiplication
        # keeping imatest convention of column sum to 1 mat. O*A => A = ccm
        # Use @ operator for optimized matrix multiplication
        out = img1 @ self.ccm_mat.transpose()

        # OPTIMIZATION: Use vectorized clipping
        # clipping after ccm is must to eliminate neg values
        out = np.clip(out, 0, 1, out=out)  # In-place clipping for efficiency

        # OPTIMIZATION: Use efficient reshape and conversion
        # convert back
        out = out.reshape(self.img.shape)
        # out = np.uint16(out * (2**self.output_bit_depth - 1))
        out = np.uint16(out * 65535) #(2**16 - 1)
        return out

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_color_correction_matrix_",
                self.platform,
                self.sensor_info["output_bit_depth"],
                self.sensor_info["bayer_pattern"],
            )

    def execute(self):
        """Execute ccm if enabled."""
        self.logger.info(f"Color Correction Matrix = {self.enable}")

        if self.enable:
            start = time.time()
            ccm_out = self.apply_ccm_optimized()
            execution_time = time.time() - start
            self.logger.info(f"  Execution time: {execution_time:.3f}s")
            self.img = ccm_out

        self.save()
        return self.img
