"""
File: demosaic.py
Description: Implements the cfa interpolation algorithms
Author: 10xEngineers
------------------------------------------------------------
"""
import time
import numpy as np
from util.utils import save_output_array
from util.debug_utils import get_debug_logger, is_debug_enabled
from modules.demosaic.malvar_he_cutler import Malvar as MAL

# Try to import CuPy version
try:
    from modules.demosaic.malvar_he_cutler_cupy import MalvarCuPy as MALCUPY
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# Import bilinear demosaic options
from modules.demosaic.bilinear_demosaic import (
    BilinearDemosaic as BILINEAR
)


class Demosaic:
    "CFA Interpolation"

    def __init__(self, img, platform, sensor_info, parm_dga):
        self.img = img
        self.bayer = sensor_info["bayer_pattern"]
        # self.bit_depth = sensor_info["output_bit_depth"]
        self.is_save = parm_dga["is_save"]
        self.sensor_info = sensor_info
        self.platform = platform
        # Get algorithm from config, default to "malvar" if not specified
        self.algorithm = parm_dga.get("algorithm", "malvar")
        # Initialize debug logger
        self.logger = get_debug_logger("Demosaic", config=self.platform)

    def masks_cfa_bayer(self):
        """
        Generating masks for the given bayer pattern
        """
        pattern = self.bayer
        # dict will be creating 3 channel boolean type array of given shape with the name
        # tag like 'r_channel': [False False ....] , 'g_channel': [False False ....] ,
        # 'b_channel': [False False ....]
        channels = dict(
            (channel, np.zeros(self.img.shape, dtype=bool)) for channel in "rgb"
        )

        # Following comment will create boolean masks for each channel r_channel,
        # g_channel and b_channel
        for channel, (y_channel, x_channel) in zip(
            pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]
        ):
            channels[channel][y_channel::2, x_channel::2] = True

        # tuple will return 3 channel boolean pattern for r_channel,
        # g_channel and b_channel with True at corresponding value
        # For example in rggb pattern, the r_channel mask would then be
        # [ [ True, False, True, False], [ False, False, False, False]]
        return tuple(channels[c] for c in "rgb")

    def apply_cfa(self, algorithm="malvar"):
        """
        Demosaicing the given raw image using given algorithm
        
        Args:
            algorithm (str): Demosaic algorithm to use
                - "malvar": Malvar-He-Cutler (default, high quality)
                - "bilinear": Simple bilinear interpolation
                - "bilinear_opt": Optimized bilinear using NumPy operations
                - "bilinear_fast": Fastest bilinear using simple averaging
        """
        # 3D masks according to the given bayer
        masks = self.masks_cfa_bayer()
        
        if algorithm == "malvar":
            # Use CuPy version if available and beneficial
            if CUPY_AVAILABLE:
                mal = MALCUPY(self.img, masks)
            else:
                mal = MAL(self.img, masks)
            demos_out = mal.apply_malvar()
            
        elif algorithm == "bilinear":
            bilinear = BILINEAR(self.img, masks)
            demos_out = bilinear.apply_bilinear()
            
            
        else:
            raise ValueError(f"Unknown demosaic algorithm: {algorithm}")

        # Pipeline convention: output 16-bit RGB for CCM
        output_bit_depth = self.sensor_info.get("pipeline_rgb_bit_depth", 16)
        output_max = 2**output_bit_depth - 1
        demos_out = np.clip(demos_out, 0, output_max)
        demos_out = np.uint16(demos_out)
        return demos_out

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_demosaic_",
                self.platform,
                self.sensor_info["bit_depth"],
                self.sensor_info["bayer_pattern"],
            )

    def execute(self, algorithm=None):
        """
        Applying demosaicing to bayer image
        
        Args:
            algorithm (str, optional): Demosaic algorithm to use. If None, uses algorithm from config.
                - "malvar": Malvar-He-Cutler (default, high quality)
                - "bilinear": Simple bilinear interpolation
                - "bilinear_opt": Optimized bilinear using NumPy operations
                - "bilinear_fast": Fastest bilinear using simple averaging
        """
        # Use algorithm from config if not specified
        if algorithm is None:
            algorithm = self.algorithm
            
        self.logger.info(f"CFA interpolation using {algorithm} algorithm")
        start = time.time()
        cfa_out = self.apply_cfa(algorithm)
        execution_time = time.time() - start
        self.logger.info(f"Execution time: {execution_time:.3f}s")
        self.img = cfa_out
        self.save()
        return self.img
