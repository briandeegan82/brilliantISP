"""
File: black_level_correction.py
Description: Implements black level correction and image linearization based on config file params
Code / Paper  Reference:
Author: Brian Deegan (based in part on 10xEngineers / Infinite-ISP)
------------------------------------------------------------
"""

import time
import numpy as np

from util.isp_types import BlackLevelCorrectionConfig, PlatformConfig, RawBayerImage, SensorInfo, UInt32Image
from util.utils import save_output_array
from util.debug_utils import get_debug_logger


class BlackLevelCorrection:
    """
    Black Level Correction
    """

    def __init__(
        self,
        img: RawBayerImage,
        platform: PlatformConfig,
        sensor_info: SensorInfo,
        parm_blc: BlackLevelCorrectionConfig,
    ) -> None:
        self.img = img
        self.enable = parm_blc["is_enable"]
        self.sensor_info = sensor_info
        self.param_blc = parm_blc
        self.is_linearize = self.param_blc["is_linear"]
        self.is_save = parm_blc["is_save"]
        self.platform = platform
        # Initialize debug logger
        self.logger = get_debug_logger("BlackLevelCorrection", config=self.platform)

    def apply_blc_parameters(self) -> UInt32Image:
        """
        Apply BLC parameters provided in config file
        """

        # get config parm
        bayer = str(self.sensor_info["bayer_pattern"]).strip().lower()
        bpp = self.sensor_info["bit_depth"]
        r_offset = self.param_blc["r_offset"]
        gb_offset = self.param_blc["gb_offset"]
        gr_offset = self.param_blc["gr_offset"]
        b_offset = self.param_blc["b_offset"]

        offsets = np.asarray((r_offset, gr_offset, gb_offset, b_offset))
        if offsets.dtype.kind not in "iuf" or not np.all(np.isfinite(offsets)) \
                or np.any(np.abs(offsets) > np.finfo(np.float64).max):
            raise ValueError("black-level offsets must be finite")
        offsets = offsets.astype(np.float64)
        denominators = np.ones(4, np.float64)
        if self.is_linearize:
            saturations = np.asarray([self.param_blc[key] for key in ("r_sat", "gr_sat", "gb_sat", "b_sat")])
            if saturations.dtype.kind not in "iuf" or not np.all(np.isfinite(saturations)) \
                    or np.any(np.abs(saturations) > np.finfo(np.float64).max):
                raise ValueError("black-level saturation values must be finite")
            with np.errstate(over="ignore", invalid="ignore"):
                denominators = saturations.astype(np.float64) - offsets
            if not np.all(np.isfinite(denominators)) or np.any(denominators <= 0):
                raise ValueError("black-level saturation minus offset must be finite and positive")
        raw = self.img.astype(np.float64)
        layouts = {"rggb": (0, 1, 2, 3), "bggr": (3, 2, 1, 0),
                   "grbg": (1, 0, 3, 2), "gbrg": (2, 3, 0, 1)}
        if bayer not in layouts:
            raise ValueError(f"Unsupported black-level Bayer pattern {bayer!r}")

        # implementing this formula with condition
        # ((img - blc) / (sat_level-blc)) * bitRange
        for site, channel in enumerate(layouts[bayer]):
            row, col = divmod(site, 2)
            view = raw[row::2, col::2]
            view -= offsets[channel]
            if self.is_linearize:
                span = denominators[channel]
                view[:] = np.clip(view, 0, span) / span * ((2**bpp) - 1)

        raw_blc = np.clip(raw, 0, (2**bpp) - 1).astype(np.uint32)
        return raw_blc

    def save(self) -> None:
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_black_level_correction_",
                self.platform,
                self.sensor_info["bit_depth"],
                self.sensor_info["bayer_pattern"],
            )

    def execute(self) -> RawBayerImage | UInt32Image:
        """
        Black Level Correction
        """
        self.logger.info(f"Black Level Correction = {self.enable}")

        if self.enable:
            start = time.time()
            blc_out = self.apply_blc_parameters()
            execution_time = time.time() - start
            self.logger.info(f"Execution time: {execution_time:.3f}s")
            self.img = blc_out
        self.save()
        return self.img
