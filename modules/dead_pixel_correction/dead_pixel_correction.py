"""
File: dead_pixel_correction.py
Description: Corrects the hot or dead pixels
Code / Paper  Reference: https://ieeexplore.ieee.org/document/9194921
Implementation inspired from: (OpenISP) https://github.com/cruxopen/openISP
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import numpy as np

from util.utils import save_output_array
from util.debug_utils import get_debug_logger
from modules.dead_pixel_correction.dynamic_dpc import DynamicDPC as DynDPC

# Try to import Numba version
try:
    from modules.dead_pixel_correction.dynamic_dpc_numba_optimized import DynamicDPCNumbaOptimized as DynDPCNumba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False



class DeadPixelCorrection:
    "Dead Pixel Correction"

    def __init__(self, img, sensor_info, parm_dpc, platform):
        self.img = img
        self.enable = parm_dpc["is_enable"]
        self.sensor_info = sensor_info
        self.parm_dpc = parm_dpc
        self.is_progress = platform["disable_progress_bar"]
        self.is_leave = platform["leave_pbar_string"]
        self.bpp = self.sensor_info["bit_depth"]
        self.threshold = self.parm_dpc["dp_threshold"]
        self.is_debug = self.parm_dpc["is_debug"]
        self.is_save = parm_dpc["is_save"]
        self.platform = platform
        # Initialize debug logger
        self.logger = get_debug_logger("DeadPixelCorrection", config=self.platform)

    def padding(self):
        """Return a mirror padded copy of image."""

        img_pad = np.pad(self.img, (2, 2), "reflect")
        return img_pad

    def apply_dynamic_dpc(self):
        """Apply DPC with Numba optimization if available"""
        # Use Numba version if available and beneficial
        if NUMBA_AVAILABLE:
            dpc = DynDPCNumba(self.img, self.sensor_info, self.parm_dpc)
        else:
            dpc = DynDPC(self.img, self.sensor_info, self.parm_dpc)
        return dpc.dynamic_dpc()

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_dead_pixel_correction_",
                self.platform,
                self.bpp,
                self.sensor_info["bayer_pattern"],
            )

    def execute(self):
        """Execute DPC Module"""

        self.logger.info(f"Dead Pixel Correction = {self.enable}")

        if self.enable:
            start = time.time()
            self.img = np.float32(self.img)
            dpc_out = self.apply_dynamic_dpc()
            execution_time = time.time() - start
            self.logger.info(f"Execution time: {execution_time:.3f}s")
            self.img = dpc_out

        self.save()
        return self.img
