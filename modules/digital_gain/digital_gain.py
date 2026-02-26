from util.debug_utils import get_debug_logger
"""
File: digital_gain.py
Description: Applies digital gain; operates on linear scene-referred data.
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import numpy as np

from util.utils import save_output_array


class DigitalGain:
    """
    Digital Gain
    """

    def __init__(self, img, platform, sensor_info, parm_dga):
        self.img = img.copy()
        self.is_save = parm_dga["is_save"]
        self.is_debug = parm_dga["is_debug"]
        self.is_auto = parm_dga["is_auto"]
        self.gains_array = parm_dga["gain_array"]
        self.current_gain = parm_dga["current_gain"]
        self.ae_feedback = parm_dga["ae_feedback"]
        self.sensor_info = sensor_info
        self.platform = platform
        self.param_dga = parm_dga
        # Initialize debug logger
        self.logger = get_debug_logger("DigitalGain", config=self.platform)

    def apply_digital_gain(self):
        """
        Apply Digital Gain - Provided in config file or
        according to AE Feedback
        """

        # Unified HDR path: use hdr_bit_depth (linear), fallback to bit_depth
        bpp = self.sensor_info.get("hdr_bit_depth", self.sensor_info["bit_depth"])
        # dg = self.param_dga['dg_gain']

        # converting to float image
        self.img = np.float32(self.img)

        # Gains are applied on the basis of AE-Feedback.
        # 'ae_correction == 0' - Default Gain is applied before AE feedback
        # 'ae_correction > 0' - Image is overexposed
        # 'ae_correction < 0' - Image is underexposed

        if self.is_auto:

            if self.ae_feedback < 0:
                # max/min functions is applied to not allow digital gains exceed the defined limits
                self.current_gain = min(
                    len(self.gains_array) - 1, self.current_gain + 1
                )

            elif self.ae_feedback > 0:
                self.current_gain = max(0, self.current_gain - 1)

        # Gain_Array is an array of pre-defined digital gains for ISP
        self.img = self.gains_array[self.current_gain] * self.img

        if self.is_debug:
            self.logger.info(f"   - DG  - Applied Gain = {self.gains_array[self.current_gain]}")

        # np.uint32 bit to contain the bpp bit raw
        self.img = np.uint32(np.clip(self.img, 0, ((2**bpp) - 1)))
        return self.img

    def save(self):
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

    def execute(self):
        """
        Execute Digital Gain Module
        """
        self.logger.info("Digital Gain (default) = True ")

        # ae_correction indicated if the gain is default digital gain or AE-correction gain.
        start = time.time()
        dg_out = self.apply_digital_gain()
        self.logger.info(f"  Execution time: {time.time() - start:.3f}s")
        self.img = dg_out
        self.save()
        return self.img, self.current_gain
