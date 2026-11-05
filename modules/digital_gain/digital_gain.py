from util.debug_utils import get_debug_logger

"""
File: digital_gain.py
Description: Applies digital gain; operates on linear scene-referred data.
Code / Paper  Reference:
Author: Brian Deegan (based in part on 10xEngineers / Infinite-ISP)
------------------------------------------------------------
"""
import time
import numpy as np
from typing import cast

from util.isp_types import BayerImage, DigitalGainConfig, Float32Image, PlatformConfig, SensorInfo, UInt32Image
from util.native_array import as_float32
from util.utils import save_output_array


class DigitalGain:
    """
    Digital Gain
    """

    def __init__(
        self,
        img: BayerImage,
        platform: PlatformConfig,
        sensor_info: SensorInfo,
        parm_dga: DigitalGainConfig,
    ) -> None:
        self.img = img.copy()
        self.is_save = parm_dga["is_save"]
        self.is_debug = parm_dga["is_debug"]
        self.is_enable = bool(parm_dga.get("is_enable", True))
        self.is_auto = bool(parm_dga.get("is_auto", False))
        self.gains_array = parm_dga["gain_array"]
        self.current_gain = parm_dga["current_gain"]
        self.ae_feedback = parm_dga.get("ae_feedback")
        self.sensor_info = sensor_info
        self.platform = platform
        self.param_dga = parm_dga
        # Initialize debug logger
        self.logger = get_debug_logger("DigitalGain", config=self.platform)

    def apply_digital_gain(self) -> UInt32Image:
        """
        Apply Digital Gain - Provided in config file or
        according to AE Feedback
        """

        # Unified HDR path: use hdr_bit_depth (linear), fallback to bit_depth
        bpp = self.sensor_info.get("hdr_bit_depth", self.sensor_info["bit_depth"])
        max_code = (2**bpp) - 1

        if not self.is_enable:
            self.logger.info("  Digital gain disabled — passthrough (×1)")
            return cast(
                UInt32Image,
                np.clip(self.img, 0, max_code).astype(np.uint32),
            )

        # converting to float image
        gval = self._select_gain()
        self.img = self.img.astype(np.float32, copy=False)
        self.img = self._apply_gain_to(gval)

        # np.uint32 bit to contain the bpp bit raw
        self.img = np.clip(self.img, 0, max_code).astype(np.uint32)
        return cast(UInt32Image, self.img)

    def _select_gain(self) -> float:
        """Resolve the active gain index and return the multiplier."""
        # Gains are applied on the basis of AE-Feedback.
        # 'ae_correction == 0' - Default Gain is applied before AE feedback
        # 'ae_correction > 0' - Image is overexposed
        # 'ae_correction < 0' - Image is underexposed

        # "direct" AE mode sets current_gain in one shot (see brilliant_isp + AutoExposure);
        # do not also nudge by ae_feedback here.
        if self.is_auto and self.param_dga.get("exposure_correction_mode", "step") != "direct":
            if self.ae_feedback is not None and self.ae_feedback < 0:
                # max/min functions is applied to not allow digital gains exceed the defined limits
                self.current_gain = min(len(self.gains_array) - 1, self.current_gain + 1)

            elif self.ae_feedback is not None and self.ae_feedback > 0:
                self.current_gain = max(0, self.current_gain - 1)

        # Gain_Array is an array of pre-defined digital gains for ISP
        gval = float(self.gains_array[self.current_gain])
        self.logger.info(f"  Applied gain index {self.current_gain} × {gval:g} (linear multiplier on raw)")
        if self.is_debug:
            self.logger.info(f"   - DG  - Applied Gain = {gval}")
        return gval

    def _apply_gain_to(self, gval: float) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore"):
            out = gval * self.img.astype(np.float64)
        return as_float32(out, name="digital_gain")

    def compute_linear(self) -> tuple[Float32Image, int]:
        """Native linear path: float32 gain output, no clip/cast."""
        if not self.is_enable:
            return cast(Float32Image, as_float32(self.img, name="digital_gain")), self.current_gain
        gval = self._select_gain()
        out = self._apply_gain_to(gval)
        return cast(Float32Image, out), self.current_gain

    def save(self) -> None:
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_digital_gain_",
                self.platform,
                self.sensor_info.get("hdr_bit_depth", self.sensor_info["bit_depth"]),
                self.sensor_info["bayer_pattern"],
            )

    def execute(self) -> tuple[UInt32Image, int]:
        """
        Execute Digital Gain Module
        """
        self.logger.info(f"Digital Gain = {self.is_enable}")

        # ae_correction indicated if the gain is default digital gain or AE-correction gain.
        start = time.time()
        dg_out = self.apply_digital_gain()
        self.logger.info(f"  Execution time: {time.time() - start:.3f}s")
        self.img = dg_out
        self.save()
        return self.img, self.current_gain
