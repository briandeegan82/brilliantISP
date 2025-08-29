from util.debug_utils import get_debug_logger
"""
File: sharpen.py
Description: Implements sharpening for Infinite-ISP with GPU acceleration.
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
"""

import time
from modules.sharpen.unsharp_masking import UnsharpMasking as USM

# Try to import GPU-accelerated version
try:
    from modules.sharpen.unsharp_masking_gpu import UnsharpMaskingGPU as USMGPU
    GPU_VERSION_AVAILABLE = True
except ImportError:
    GPU_VERSION_AVAILABLE = False

from util.utils import save_output_array_yuv


class Sharpening:
    """
    Sharpening with GPU acceleration
    """

    def __init__(self, img, platform, sensor_info, parm_sha, conv_std):
        self.img = img
        self.enable = parm_sha["is_enable"]
        self.sensor_info = sensor_info
        self.parm_sha = parm_sha
        self.is_save = parm_sha["is_save"]
        self.platform = platform
        self.conv_std = conv_std
        # Initialize debug logger
        self.logger = get_debug_logger("Sharpening", config=self.platform)
        
        # Check if GPU acceleration should be used
        self.use_gpu = False
        if GPU_VERSION_AVAILABLE:
            try:
                from util.gpu_utils import is_gpu_available, should_use_gpu
                self.use_gpu = (is_gpu_available() and 
                               should_use_gpu((img.shape[0], img.shape[1]), 'gaussian_blur'))
            except ImportError:
                self.use_gpu = False

    def apply_unsharp_masking(self):
        """
        Apply function for Sharpening Algorithm - Unsharp Masking
        Uses GPU acceleration if available, falls back to CPU otherwise
        """
        if self.use_gpu and GPU_VERSION_AVAILABLE:
            # Use GPU-accelerated version
            usm = USMGPU(
                self.img, self.parm_sha["sharpen_sigma"], self.parm_sha["sharpen_strength"]
            )
        else:
            # Use CPU version
            usm = USM(
                self.img, self.parm_sha["sharpen_sigma"], self.parm_sha["sharpen_strength"]
            )
        
        return usm.apply_sharpen()

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array_yuv(
                self.platform["in_file"],
                self.img,
                "Out_Shapening_",
                self.platform,
                self.conv_std,
            )

    def execute(self):
        """
        Applying sharpening to input image
        """
        self.logger.info(f"Sharpening = {self.enable}")

        if self.enable is True:
            start = time.time()
            s_out = self.apply_unsharp_masking()
            execution_time = time.time() - start
            self.logger.info(f"  Execution time: {execution_time:.3f}s")
            self.img = s_out

        self.save()
        return self.img
