from util.debug_utils import get_debug_logger
"""
File: white_balance_optimized.py
Description: White balance on linear scene-referred raw (Bayer domain).
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import numpy as np

from util.utils import save_output_array


class WhiteBalanceOptimized:
    """
    Optimized White balance Module with vectorized operations
    """

    def __init__(self, img, platform, sensor_info, parm_wbc, awb_gains):
        """
        Class Constructor
        """
        self.img = img.copy()
        self.enable = parm_wbc["is_enable"]
        self.is_save = parm_wbc["is_save"]
        self.is_auto = parm_wbc["is_auto"]
        self.is_debug = parm_wbc["is_debug"]
        self.platform = platform
        self.sensor_info = sensor_info
        self.parm_wbc = parm_wbc
        self.bayer = self.sensor_info["bayer_pattern"]
        self.bpp = self.sensor_info.get("hdr_bit_depth", self.sensor_info["bit_depth"])
        self.raw = None
        self.awb_gains = awb_gains
        # Initialize debug logger
        self.logger = get_debug_logger("WhiteBalanceOptimized", config=self.platform)

    def apply_wb_parameters_optimized(self):
        """
        Applies white balance gains from config file to raw images with optimized vectorized operations
        """
        # get config params
        redgain = self.awb_gains[0]
        bluegain = self.awb_gains[1]
        self.raw = np.float32(self.img)

        if self.is_debug:
            self.logger.info(f"   - WB  - red gain : {redgain}")
            self.logger.info(f"   - WB  - blue gain: {bluegain}")

        # OPTIMIZATION: Use vectorized Bayer pattern application
        # Create gain masks for efficient multiplication
        if self.bayer == "rggb":
            # OPTIMIZATION: Use advanced indexing for vectorized operations
            self.raw[::2, ::2] *= redgain    # Red pixels
            self.raw[1::2, 1::2] *= bluegain # Blue pixels
        elif self.bayer == "bggr":
            self.raw[::2, ::2] *= bluegain   # Blue pixels
            self.raw[1::2, 1::2] *= redgain  # Red pixels
        elif self.bayer == "grbg":
            self.raw[1::2, ::2] *= bluegain  # Blue pixels
            self.raw[::2, 1::2] *= redgain   # Red pixels
        elif self.bayer == "gbrg":
            self.raw[1::2, ::2] *= redgain   # Red pixels
            self.raw[::2, 1::2] *= bluegain  # Blue pixels

        # OPTIMIZATION: Use vectorized clipping with bounds checking
        max_value = 2**self.bpp - 1
        if np.max(self.raw) >= max_value:
            self.logger.info(f"  Warning: Raw image values exceed {self.bpp} bits.")
            self.logger.info(f"  Max value: {np.max(self.raw)}")
            self.logger.info(f"  Clipping values to {self.bpp} bits.")
            # OPTIMIZATION: Use in-place clipping for efficiency
            np.clip(self.raw, 0, max_value, out=self.raw)
        
        raw_whitebal = np.uint32(self.raw)

        return raw_whitebal

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_white_balance_",
                self.platform,
                self.sensor_info["bit_depth"],
                self.sensor_info["bayer_pattern"],
            )

    def execute(self):
        """
        Execute White Balance Module
        """
        if self.enable is True:
            self.logger.info("White balancing = True")
            start = time.time()
            wb_out = self.apply_wb_parameters_optimized()
            execution_time = time.time() - start
            self.logger.info(f"  Execution time: {execution_time:.3f}s")
            self.img = wb_out

        self.save()
        return self.img
