from util.debug_utils import get_debug_logger

"""
File: bayer_noise_reduction.py
Description: Noise reduction in Bayer domain with optional GPU acceleration.
Author: Brian Deegan (based in part on 10xEngineers / Infinite-ISP)
------------------------------------------------------------
"""


import time
from typing import cast

import numpy as np
from modules.bayer_noise_reduction.joint_bf import JointBF as JBF
from util.isp_types import BayerImage, BayerNoiseReductionConfig, Float32Image, PlatformConfig, RawBayerImage, SensorInfo
from util.utils import save_output_array
from util.native_array import as_float32

# Try to import GPU-accelerated version
try:
    from modules.bayer_noise_reduction.joint_bf_gpu import JointBFGPU as JBFGPU

    GPU_VERSION_AVAILABLE = True
except ImportError:
    GPU_VERSION_AVAILABLE = False


class BayerNoiseReduction:
    """
    Noise Reduction in Bayer domain. Uses GPU if available, else CPU.
    """

    def __init__(
        self,
        img: BayerImage,
        sensor_info: SensorInfo,
        parm_bnr: BayerNoiseReductionConfig,
        platform: PlatformConfig,
        cpu_only: bool = False,
    ) -> None:
        self.img = img
        self.enable = parm_bnr["is_enable"]
        self.sensor_info = sensor_info
        self.parm_bnr = parm_bnr
        self.is_progress = platform["disable_progress_bar"]
        self.is_leave = platform["leave_pbar_string"]
        self.is_save = parm_bnr["is_save"]
        self.platform = platform
        self.cpu_only = cpu_only

        # Initialize debug logger
        self.logger = get_debug_logger("BayerNoiseReduction", config=self.platform)

        # Check if GPU acceleration should be used
        self.use_gpu = False
        if GPU_VERSION_AVAILABLE and not cpu_only:
            try:
                from util.gpu_utils import is_gpu_available, should_use_gpu

                self.use_gpu = is_gpu_available() and should_use_gpu((sensor_info["height"], sensor_info["width"]), "filter2d")
            except ImportError:
                self.use_gpu = False

    def apply_bnr(self) -> RawBayerImage:
        """
        Apply BNR to the input image. Uses GPU if available, else CPU.
        """
        if self.use_gpu and GPU_VERSION_AVAILABLE:
            # Use GPU-accelerated version
            self.logger.info("  Using GPU-accelerated Bayer Noise Reduction")
            jbf = JBFGPU(self.img, self.sensor_info, self.parm_bnr, self.platform)
            bnr_out_img = jbf.apply_jbf()
        else:
            # Use original CPU version
            self.logger.info("  Using original CPU Bayer Noise Reduction")
            jbf = JBF(self.img, self.sensor_info, self.parm_bnr, self.platform)
            bnr_out_img = jbf.apply_jbf()

        return cast(RawBayerImage, bnr_out_img)

    def compute_linear(self) -> Float32Image:
        """Native linear path: CPU JointBF, finite float32 code output.

        Odd shapes are reflect-padded only on the bottom/right to the next
        even size (full CFA cells), filtered, and cropped back. Frames
        with a dimension below 2 bypass filtering unchanged.
        """
        src = np.asarray(self.img)
        if src.ndim != 2 or src.size == 0:
            raise ValueError(f"BNR input must be a nonempty 2-D frame, got {src.shape}")
        if not np.all(np.isfinite(src)):
            raise ValueError("BNR input must be finite")
        if not self.enable:
            return as_float32(src, name="bayer_noise_reduction")
        h, w = src.shape
        if h < 2 or w < 2:
            return as_float32(src, name="bayer_noise_reduction")
        ph, pw = h + (h % 2), w + (w % 2)
        if (ph, pw) != (h, w):
            src = np.pad(src.astype(np.float64), ((0, ph - h), (0, pw - w)), "reflect")
        sensor = dict(self.sensor_info)
        sensor["height"], sensor["width"] = src.shape
        jbf = JBF(src, cast(SensorInfo, sensor), self.parm_bnr, self.platform)
        out = np.asarray(jbf.apply_jbf(preserve_float=True), dtype=np.float64)
        out = out[:h, :w]
        return as_float32(out, name="bayer_noise_reduction")

    def save(self) -> None:
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

    def execute(self) -> RawBayerImage:
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
        return cast(RawBayerImage, self.img)
