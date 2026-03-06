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

# Import VNG demosaic options
from modules.demosaic.vng_demosaic import (
    VNGDemosaic as VNG,
    VNGDemosaicOptimized as VNG_OPT
)

# Import Hamilton-Adams demosaic options
from modules.demosaic.hamilton_adams_demosaic import (
    HamiltonAdamsDemosaic as HA,
    HamiltonAdamsOptimized as HA_OPT
)

# Import PPG demosaic options
from modules.demosaic.ppg_demosaic import (
    PPGDemosaic as PPG,
    PPGDemosaicOptimized as PPG_OPT
)

# Import LMMSE demosaic options
from modules.demosaic.lmmse_demosaic import (
    LMMSEDemosaic as LMMSE,
    LMMSEDemosaicOptimized as LMMSE_OPT,
    LMMSEDemosaicFast as LMMSE_FAST
)

# Import AHD demosaic options
from modules.demosaic.ahd_demosaic import (
    AHDDemosaic as AHD,
    AHDDemosaicOptimized as AHD_OPT
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
                - "vng": Variable Number of Gradients (edge-directed, high quality)
                - "vng_opt": Optimized VNG using vectorized operations (faster)
                - "hamilton_adams": Hamilton-Adams (color ratio based, excellent quality)
                - "hamilton_adams_opt": Optimized Hamilton-Adams (faster)
                - "ppg": PPG - Patterned Pixel Grouping (iterative refinement, excellent quality)
                - "ppg_opt": Optimized PPG with enhanced pattern recognition (best quality)
                - "lmmse": LMMSE - Linear Minimum Mean Square Error (statistical, excellent quality)
                - "lmmse_opt": Optimized LMMSE with enhanced statistics (superior quality)
                - "lmmse_fast": Fast LMMSE with simplified estimation (good quality, faster)
                - "ahd": AHD - Adaptive Homogeneity-Directed (dual interpolation, excellent quality)
                - "ahd_opt": Optimized AHD with enhanced homogeneity analysis (superior quality)
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
            
        elif algorithm == "vng":
            vng = VNG(self.img, masks)
            demos_out = vng.apply_vng()
            
        elif algorithm == "vng_opt":
            vng_opt = VNG_OPT(self.img, masks)
            demos_out = vng_opt.apply_vng_optimized()
            
        elif algorithm == "hamilton_adams":
            ha = HA(self.img, masks)
            demos_out = ha.apply_hamilton_adams()
            
        elif algorithm == "hamilton_adams_opt":
            ha_opt = HA_OPT(self.img, masks)
            demos_out = ha_opt.apply_hamilton_adams_optimized()
            
        elif algorithm == "ppg":
            ppg = PPG(self.img, masks)
            demos_out = ppg.apply_ppg()
            
        elif algorithm == "ppg_opt":
            ppg_opt = PPG_OPT(self.img, masks)
            demos_out = ppg_opt.apply_ppg_optimized()
            
        elif algorithm == "lmmse":
            lmmse = LMMSE(self.img, masks)
            demos_out = lmmse.apply_lmmse()
            
        elif algorithm == "lmmse_opt":
            lmmse_opt = LMMSE_OPT(self.img, masks)
            demos_out = lmmse_opt.apply_lmmse_optimized()
            
        elif algorithm == "lmmse_fast":
            lmmse_fast = LMMSE_FAST(self.img, masks)
            demos_out = lmmse_fast.apply_lmmse_fast()
            
        elif algorithm == "ahd":
            ahd = AHD(self.img, masks)
            demos_out = ahd.apply_ahd()
            
        elif algorithm == "ahd_opt":
            ahd_opt = AHD_OPT(self.img, masks)
            demos_out = ahd_opt.apply_ahd_optimized()
            
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
                - "vng": Variable Number of Gradients (edge-directed, high quality)
                - "vng_opt": Optimized VNG using vectorized operations (faster)
                - "hamilton_adams": Hamilton-Adams (color ratio based, excellent quality)
                - "hamilton_adams_opt": Optimized Hamilton-Adams (faster)
                - "ppg": PPG - Patterned Pixel Grouping (iterative refinement, excellent quality)
                - "ppg_opt": Optimized PPG with enhanced pattern recognition (best quality)
                - "lmmse": LMMSE - Linear Minimum Mean Square Error (statistical, excellent quality)
                - "lmmse_opt": Optimized LMMSE with enhanced statistics (superior quality)
                - "lmmse_fast": Fast LMMSE with simplified estimation (good quality, faster)
                - "ahd": AHD - Adaptive Homogeneity-Directed (dual interpolation, excellent quality)
                - "ahd_opt": Optimized AHD with enhanced homogeneity analysis (superior quality)
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
