# -*- coding: utf-8 -*-
"""
ACES (Academy Color Encoding System) Tone Mapping Implementation
Based on ACES 1.0 Output Transform
@author: BrilliantISP
"""

import numpy as np
import time
from util.debug_utils import get_debug_logger
from util.utils import save_output_array


class ACESToneMapping:
    """
    ACES Tone Mapping Algorithm Implementation
    Implements the ACES 1.0 Output Transform (RRT + ODT)
    suitable for HDR to SDR conversion
    """
    
    def __init__(self, img, platform, sensor_info, params):
        """
        Initialize ACES Tone Mapping
        
        Parameters:
            img (numpy.ndarray): Input image (luminance or RGB)
            platform (dict): Platform configuration
            sensor_info (dict): Sensor information
            params (dict): ACES parameters
        """
        self.img = img.astype(np.float32).copy()
        self.is_enable = params.get("is_enable", True)
        self.is_save = params.get("is_save", False)
        self.is_debug = params.get("is_debug", False)
        
        # ACES parameters
        self.exposure_adjust = params.get("exposure_adjust",
            params.get("exposure_adjustment", 0.0))  # EV adjustment
        self.white_point = params.get("white_point", 7.2)  # nits
        self.surround = params.get("surround", "dark")  # 'dark', 'dim', 'avg'
        self.gamma = params.get("gamma", 2.4)  # Display gamma
        
        self.output_bit_depth = sensor_info.get("output_bit_depth", 8)
        self.sensor_info = sensor_info
        self.platform = platform
        
        # Initialize debug logger
        self.logger = get_debug_logger("ACESToneMapping", config=self.platform)
    
    def lin_to_log2(self, x, black_point=0.0):
        """Convert linear values to log2 domain"""
        return np.log2(np.maximum(x, black_point + 1e-5))
    
    def log2_to_lin(self, x):
        """Convert from log2 domain to linear"""
        return np.power(2.0, x)
    
    def apply_cctf_decoding(self, x):
        """
        Apply ACES CCTF (Color Component Transfer Function) decoding.
        Prepares linear values for RRT.
        """
        return np.maximum(x, 0.0)
    
    def rrt(self, x):
        """
        Reference Rendering Transform (RRT)
        Maps ACES RGB to RRT color space for tone mapping
        
        Parameters:
            x (numpy.ndarray): Input in ACES 2065-4 linear RGB or single-channel luminance
        
        Returns:
            numpy.ndarray: RRT RGB or luminance
        """
        # If single channel (luminance), apply tone curve directly
        if x.ndim == 2:
            return self._aces_tone_curve(x)
        
        # RRT uses a simple contrast-stretching sigmoid
        # This is a simplified version of the full RRT
        
        # Input scaling matrix (AP0 to AP1)
        M1 = np.array([
            [0.6954522, 0.1406786, 0.1638690],
            [0.0447945, 0.8596711, 0.0955343],
            [-0.0055258, 0.0040252, 1.0015006]
        ])
        
        # Apply input matrix
        x_ap1 = self._matmul(x, M1.T)
        
        # Tone mapping curve (simplified ACES tone mapper)
        # Based on ACES 1.0 RRT and ODT
        x_tm = self._aces_tone_curve(x_ap1)
        
        return x_tm
    
    def _aces_tone_curve(self, x):
        """
        ACES filmic tone mapping curve (fitted approximation).
        Maps linear HDR values to [0, 1] SDR.
        Formula: (x * (a*x + b)) / (x * (c*x + d) + e)
        Coefficients fitted to ACES RRT reference (Knarkowicz 2016).
        """
        # Exposure adjustment (support both param names from config)
        exp = self.exposure_adjust
        x = x * (2.0 ** exp)
        x = np.maximum(x, 0.0)
        
        # ACES filmic rational function - maps linear to [0, 1]
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        result = (x * (a * x + b)) / (x * (c * x + d) + e)
        return np.clip(result, 0.0, 1.0)
    
    def apply_odt(self, x):
        """
        Output Device Transform (ODT)
        Maps RRT output to SDR display
        
        Parameters:
            x (numpy.ndarray): RRT RGB [0, 1] or single-channel luminance
        
        Returns:
            numpy.ndarray: SDR display RGB [0, 1] or luminance
        """
        # If single channel, just apply gamma
        if x.ndim == 2:
            x_odt = self._apply_srgb_gamma(x, self.gamma)
            return np.clip(x_odt, 0.0, 1.0)
        
        # ODT color matrix (RRT to display primaries)
        M2 = np.array([
            [0.9999999, 0.0000000, 0.0000000],
            [0.0000000, 0.9999999, 0.0000000],
            [0.0000000, 0.0000000, 0.9999999]
        ])
        
        x_odt = self._matmul(x, M2.T)
        
        # Apply gamma correction (sRGB-like curve)
        x_odt = self._apply_srgb_gamma(x_odt, self.gamma)
        
        return np.clip(x_odt, 0.0, 1.0)
    
    def _apply_srgb_gamma(self, x, gamma=2.4):
        """Apply sRGB-like gamma correction"""
        # Linear segment: x <= 0.0031308
        linear = x <= 0.0031308
        result = np.zeros_like(x)
        
        # Linear part
        result[linear] = 12.92 * x[linear]
        
        # Power part
        power = x > 0.0031308
        result[power] = (1.0 + 0.055) * np.power(x[power], 1.0/gamma) - 0.055
        
        return result
    
    def _matmul(self, x, M):
        """
        Matrix multiplication for images
        Handles both single-channel (H, W) and multi-channel (H, W, C) images
        """
        orig_shape = x.shape
        
        # If single channel, skip matrix multiplication (just apply tone curve)
        if x.ndim == 2:
            return x
        
        # Flatten to (N, C) where N = H*W
        H, W, C = x.shape
        x_flat = x.reshape(-1, C)
        
        # Matrix multiplication
        result = x_flat @ M
        
        # Reshape back
        result = result.reshape(H, W, -1)
        
        return result
    
    def normalize(self, image):
        """Normalize image to [0, 1] range"""
        img_min = np.min(image)
        img_max = np.max(image)
        if img_max > img_min:
            return (image - img_min) / (img_max - img_min)
        return image
    
    def apply_tone_mapping(self):
        """
        Apply ACES tone mapping to the input image
        """
        # Prepare input - ensure we're working with normalized HDR values
        normalized_img = self.normalize(self.img)
        
        # Denormalize to typical HDR range (~0-10000 cd/m^2)
        hdr_img = normalized_img * 100.0
        
        # Apply ACES transforms
        aces_output = self.rrt(hdr_img)
        sdr_output = self.apply_odt(aces_output)
        
        return sdr_output
    
    def save(self):
        """Save tone mapping output"""
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_aces_tonemapped_",
                self.platform,
                self.sensor_info.get("bit_depth", 8),
                self.sensor_info.get("bayer_pattern", "RGGB")
            )
    
    def execute(self):
        """
        Execute ACES tone mapping
        
        Returns:
            numpy.ndarray: Tone-mapped image in [0, 1] range
        """
        if self.is_enable is True:
            self.logger.info("Executing ACES Tone Mapping...")
            start = time.time()
            
            try:
                self.img = self.apply_tone_mapping()
            except Exception as e:
                self.logger.error(f"ACES tone mapping failed: {e}")
                # Fallback to simple linear mapping
                self.img = self.normalize(self.img)
            
            execution_time = time.time() - start
            self.logger.info(f"  Execution time: {execution_time:.3f}s")
        
        self.save()
        return self.img
