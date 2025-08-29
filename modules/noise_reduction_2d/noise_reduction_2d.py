from util.debug_utils import get_debug_logger
"""
File: noise_reduction_2d.py
Description: Apply denoising algorithms on luminance channel with NumPy optimizations
Author: 10xEngineers
------------------------------------------------------------
"""
import time
from util.utils import save_output_array_yuv
from modules.noise_reduction_2d.non_local_means import NLM

# Try to import optimized version with NumPy broadcast
try:
    from modules.noise_reduction_2d.non_local_means_optimized import NLMOptimized as NLMOPT
    OPTIMIZED_VERSION_AVAILABLE = True
except ImportError:
    OPTIMIZED_VERSION_AVAILABLE = False


class NoiseReduction2d:
    """
    2D Noise Reduction with NumPy optimizations
    """

    def __init__(self, img, sensor_info, parm_2dnr, platform, conv_std):
        self.img = img
        self.enable = parm_2dnr["is_enable"]
        self.sensor_info = sensor_info
        self.parm_2dnr = parm_2dnr
        self.conv_std = conv_std
        self.is_progress = platform["disable_progress_bar"]
        self.is_leave = platform["leave_pbar_string"]
        self.is_save = parm_2dnr["is_save"]
        self.platform = platform
        # Initialize debug logger
        self.logger = get_debug_logger("NoiseReduction2d", config=self.platform)

    def apply_2dnr(self):
        """
        Applying noise reduction algorithms (EBF, NLM, Mean, BF)
        Uses optimized version with NumPy broadcast if available
        """
        # TEMPORARILY DISABLED: Use original version until numerical issues are fixed
        if False and OPTIMIZED_VERSION_AVAILABLE:
            # Use optimized version with NumPy broadcast operations
            self.logger.info("  Using optimized Non-local Means with NumPy broadcast")
            nlm = NLMOPT(self.img, self.sensor_info, self.parm_2dnr, self.platform)
        else:
            # Use original version
            self.logger.info("  Using original Non-local Means")
            nlm = NLM(self.img, self.sensor_info, self.parm_2dnr, self.platform)
        
        return nlm.apply_nlm()

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array_yuv(
                self.platform["in_file"],
                self.img,
                "Out_2d_noise_reduction_",
                self.platform,
                self.conv_std,
            )

    def execute(self):
        """
        Executing 2D noise reduction module
        """
        self.logger.info(f"Noise Reduction 2d = {self.enable}")

        if self.enable is True:
            start = time.time()
            s_out = self.apply_2dnr()
            execution_time = time.time() - start
            self.logger.info(f"  Execution time: {execution_time:.3f}s")
            self.img = s_out

        self.save()
        return self.img
