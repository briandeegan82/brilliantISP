from util.debug_utils import get_debug_logger
"""
File: bayer_noise_reduction.py
Description: Noise reduction in bayer domain with GPU acceleration and NumPy optimizations
Author: 10xEngineers
------------------------------------------------------------
"""


import time
from modules.bayer_noise_reduction.joint_bf import JointBF as JBF
from util.utils import save_output_array

# Try to import GPU-accelerated version
try:
    from modules.bayer_noise_reduction.joint_bf_gpu import JointBFGPU as JBFGPU
    GPU_VERSION_AVAILABLE = True
except ImportError:
    GPU_VERSION_AVAILABLE = False

# Try to import optimized version with NumPy broadcast
try:
    from modules.bayer_noise_reduction.joint_bf_optimized import JointBFOptimized as JBFOPT
    OPTIMIZED_VERSION_AVAILABLE = True
except ImportError:
    OPTIMIZED_VERSION_AVAILABLE = False


class BayerNoiseReduction:
    """
    Noise Reduction in Bayer domain with GPU acceleration and NumPy optimizations
    """

    def __init__(self, img, sensor_info, parm_bnr, platform):
        self.img = img
        self.enable = parm_bnr["is_enable"]
        self.sensor_info = sensor_info
        self.parm_bnr = parm_bnr
        self.is_progress = platform["disable_progress_bar"]
        self.is_leave = platform["leave_pbar_string"]
        self.is_save = parm_bnr["is_save"]
        self.platform = platform
        
        # Initialize debug logger
        self.logger = get_debug_logger("BayerNoiseReduction", config=self.platform)
        
        # Check if GPU acceleration should be used
        self.use_gpu = False
        if GPU_VERSION_AVAILABLE:
            try:
                from util.gpu_utils import is_gpu_available, should_use_gpu
                self.use_gpu = (is_gpu_available() and 
                               should_use_gpu((sensor_info["height"], sensor_info["width"]), 'filter2d'))
            except ImportError:
                self.use_gpu = False

    def apply_bnr(self):
        """
        Apply bnr to the input image and return the output image
        Uses optimized version with NumPy broadcast, GPU acceleration if available, falls back to CPU otherwise
        """
        # Priority: Optimized > GPU > Original
        # DISABLED: Optimization is causing image corruption
        if False and OPTIMIZED_VERSION_AVAILABLE:
            # Use optimized version with NumPy broadcast operations
            self.logger.info("  Using optimized Bayer Noise Reduction with NumPy broadcast")
            jbf = JBFOPT(self.img, self.sensor_info, self.parm_bnr, self.platform)
            bnr_out_img = jbf.apply_jbf()
        elif self.use_gpu and GPU_VERSION_AVAILABLE:
            # Use GPU-accelerated version
            self.logger.info("  Using GPU-accelerated Bayer Noise Reduction")
            jbf = JBFGPU(self.img, self.sensor_info, self.parm_bnr, self.platform)
            bnr_out_img = jbf.apply_jbf()
        else:
            # Use original CPU version
            self.logger.info("  Using original CPU Bayer Noise Reduction")
            jbf = JBF(self.img, self.sensor_info, self.parm_bnr, self.platform)
            bnr_out_img = jbf.apply_jbf()
        
        return bnr_out_img

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_bayer_noise_reduction_",
                self.platform,
                self.sensor_info["bit_depth"],
                self.sensor_info["bayer_pattern"],
            )

    def execute(self):
        """
        Applying BNR to input RAW image and returns the output image
        """
        self.logger.info(f"Bayer Noise Reduction = {self.enable}")

        if self.enable is True:
            start = time.time()
            bnr_out = self.apply_bnr()
            execution_time = time.time() - start
            self.logger.info(f"  Execution time: {execution_time:.3f}s")
            self.img = bnr_out

        self.save()
        return self.img
