from util.debug_utils import get_debug_logger
"""
File: ldci.py
Description: Implements the contrast adjustment in the yuv domain
Author: 10xEngineers
------------------------------------------------------------
"""
import time
from util.utils import save_output_array_yuv
from modules.ldci.clahe import CLAHE


class LDCI:
    """
    Local Dynamic Contrast Enhancement
    """

    def __init__(self, yuv, platform, sensor_info, parm_ldci, conv_std):
        self.yuv = yuv
        self.img = yuv
        self.enable = parm_ldci["is_enable"]
        self.sensor_info = sensor_info
        self.parm_ldci = parm_ldci
        self.is_save = parm_ldci["is_save"]
        self.platform = platform
        self.conv_std = conv_std
        # Initialize debug logger
        self.logger = get_debug_logger("LDCI", config=self.platform)

    def apply_ldci(self):
        """
        Applying LDCI module to the given image
        Uses optimized version with NumPy broadcast if available
        """
        # TEMPORARILY DISABLED: Use original version until numerical issues are fixed
        if False and OPTIMIZED_VERSION_AVAILABLE:
            # Use optimized version with NumPy broadcast operations
            self.logger.info("  Using optimized CLAHE with NumPy broadcast")
            clahe = CLAHEOPT(self.yuv, self.platform, self.sensor_info, self.parm_ldci)
        else:
            # Use original version
            self.logger.info("  Using original CLAHE")
            clahe = CLAHE(self.yuv, self.platform, self.sensor_info, self.parm_ldci)
        
        out_ceh = clahe.apply_clahe()
        return out_ceh

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array_yuv(
                self.platform["in_file"],
                self.img,
                "Out_ldci_",
                self.platform,
                self.conv_std,
            )

    def execute(self):
        """
        Executing LDCI module according to user choice
        """
        self.logger.info(f"LDCI = {self.enable}")

        if self.enable is True:
            start = time.time()
            s_out = self.apply_ldci()
            self.logger.info(f"  Execution time: {time.time() - start:.3f}s")
            self.img = s_out

        self.save()
        return self.img
