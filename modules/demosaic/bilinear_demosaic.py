"""
File: bilinear_demosaic.py
Description: Simple bilinear filter demosaic algorithm
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import numpy as np
from scipy.ndimage import convolve


class BilinearDemosaic:
    """
    Simple bilinear filter demosaic algorithm
    Much simpler and potentially faster than Malvar-He-Cutler
    """
    
    def __init__(self, raw_in, masks):
        self.img = raw_in
        self.masks = masks
        
    def apply_bilinear(self):
        """
        Demosaicing using simple bilinear interpolation
        """
        mask_r, mask_g, mask_b = self.masks
        raw_in = np.float32(self.img)
        
        # Create output array
        demos_out = np.empty((raw_in.shape[0], raw_in.shape[1], 3))
        
        # Extract existing color channels
        r_channel = raw_in * mask_r
        g_channel = raw_in * mask_g
        b_channel = raw_in * mask_b
        
        # Simple bilinear interpolation kernels
        # For interpolating missing green pixels at red/blue locations
        green_kernel = np.array([
            [0, 0.25, 0],
            [0.25, 0, 0.25],
            [0, 0.25, 0]
        ], dtype=np.float32)
        
        # For interpolating missing red/blue pixels at green locations
        red_blue_kernel = np.array([
            [0, 0.5, 0],
            [0.5, 0, 0.5],
            [0, 0.5, 0]
        ], dtype=np.float32)
        
        # For interpolating missing red pixels at blue locations and vice versa
        cross_kernel = np.array([
            [0, 0, 0],
            [0, 0.25, 0],
            [0, 0, 0]
        ], dtype=np.float32)
        
        # Interpolate green channel at red and blue locations
        green_interpolated = convolve(raw_in, green_kernel, mode='mirror')
        g_channel = np.where(
            np.logical_or(mask_r == 1, mask_b == 1),
            green_interpolated,
            g_channel
        )
        
        # Interpolate red channel at green and blue locations
        red_interpolated = convolve(raw_in, red_blue_kernel, mode='mirror')
        r_channel = np.where(
            np.logical_or(mask_g == 1, mask_b == 1),
            red_interpolated,
            r_channel
        )
        
        # Interpolate blue channel at green and red locations
        blue_interpolated = convolve(raw_in, red_blue_kernel, mode='mirror')
        b_channel = np.where(
            np.logical_or(mask_g == 1, mask_r == 1),
            blue_interpolated,
            b_channel
        )
        
        # For red at blue locations and blue at red locations, use cross interpolation
        # This is a simplified approach - in practice, we'd need more sophisticated logic
        # based on the specific Bayer pattern
        
        # Assign channels to output
        demos_out[:, :, 0] = r_channel
        demos_out[:, :, 1] = g_channel
        demos_out[:, :, 2] = b_channel
        
        return demos_out


class BilinearDemosaicOptimized:
    """
    Optimized bilinear demosaic using NumPy operations
    """
    
    def __init__(self, raw_in, masks):
        self.img = raw_in
        self.masks = masks
        
    def apply_bilinear_optimized(self):
        """
        Optimized bilinear demosaic using NumPy operations
        """
        mask_r, mask_g, mask_b = self.masks
        raw_in = np.float32(self.img)
        
        # Create output array
        demos_out = np.empty((raw_in.shape[0], raw_in.shape[1], 3))
        
        # Extract existing color channels
        r_channel = raw_in * mask_r
        g_channel = raw_in * mask_g
        b_channel = raw_in * mask_b
        
        # Interpolate green channel at red and blue locations using NumPy operations
        # Average of 4 diagonal neighbors
        g_interp = 0.25 * (
            np.roll(np.roll(raw_in, -1, axis=0), -1, axis=1) +  # top-left
            np.roll(np.roll(raw_in, -1, axis=0), 1, axis=1) +   # top-right
            np.roll(np.roll(raw_in, 1, axis=0), -1, axis=1) +   # bottom-left
            np.roll(np.roll(raw_in, 1, axis=0), 1, axis=1)      # bottom-right
        )
        
        g_channel = np.where(
            np.logical_or(mask_r == 1, mask_b == 1),
            g_interp,
            g_channel
        )
        
        # Interpolate red channel at green and blue locations
        # Average of 2 horizontal neighbors
        r_interp_h = 0.5 * (
            np.roll(raw_in, -1, axis=1) +  # left
            np.roll(raw_in, 1, axis=1)     # right
        )
        
        # Average of 2 vertical neighbors
        r_interp_v = 0.5 * (
            np.roll(raw_in, -1, axis=0) +  # top
            np.roll(raw_in, 1, axis=0)     # bottom
        )
        
        # Use horizontal interpolation for green pixels in red rows
        # Use vertical interpolation for green pixels in blue rows
        r_channel = np.where(
            np.logical_and(mask_g == 1, mask_r.any(axis=1, keepdims=True)),
            r_interp_h,
            r_channel
        )
        
        r_channel = np.where(
            np.logical_and(mask_g == 1, mask_b.any(axis=1, keepdims=True)),
            r_interp_v,
            r_channel
        )
        
        # For red at blue locations, use diagonal interpolation
        r_interp_diag = 0.25 * (
            np.roll(np.roll(raw_in, -1, axis=0), -1, axis=1) +
            np.roll(np.roll(raw_in, -1, axis=0), 1, axis=1) +
            np.roll(np.roll(raw_in, 1, axis=0), -1, axis=1) +
            np.roll(np.roll(raw_in, 1, axis=0), 1, axis=1)
        )
        
        r_channel = np.where(mask_b == 1, r_interp_diag, r_channel)
        
        # Similar interpolation for blue channel
        b_interp_h = 0.5 * (
            np.roll(raw_in, -1, axis=1) +
            np.roll(raw_in, 1, axis=1)
        )
        
        b_interp_v = 0.5 * (
            np.roll(raw_in, -1, axis=0) +
            np.roll(raw_in, 1, axis=0)
        )
        
        b_channel = np.where(
            np.logical_and(mask_g == 1, mask_b.any(axis=1, keepdims=True)),
            b_interp_h,
            b_channel
        )
        
        b_channel = np.where(
            np.logical_and(mask_g == 1, mask_r.any(axis=1, keepdims=True)),
            b_interp_v,
            b_channel
        )
        
        b_interp_diag = 0.25 * (
            np.roll(np.roll(raw_in, -1, axis=0), -1, axis=1) +
            np.roll(np.roll(raw_in, -1, axis=0), 1, axis=1) +
            np.roll(np.roll(raw_in, 1, axis=0), -1, axis=1) +
            np.roll(np.roll(raw_in, 1, axis=0), 1, axis=1)
        )
        
        b_channel = np.where(mask_r == 1, b_interp_diag, b_channel)
        
        # Assign channels to output
        demos_out[:, :, 0] = r_channel
        demos_out[:, :, 1] = g_channel
        demos_out[:, :, 2] = b_channel
        
        return demos_out


class BilinearDemosaicFast:
    """
    Fast bilinear demosaic using simple averaging
    """
    
    def __init__(self, raw_in, masks):
        self.img = raw_in
        self.masks = masks
        
    def apply_bilinear_fast(self):
        """
        Fast bilinear demosaic using simple averaging
        """
        mask_r, mask_g, mask_b = self.masks
        raw_in = np.float32(self.img)
        
        # Create output array
        demos_out = np.empty((raw_in.shape[0], raw_in.shape[1], 3))
        
        # Extract existing color channels
        r_channel = raw_in * mask_r
        g_channel = raw_in * mask_g
        b_channel = raw_in * mask_b
        
        # Simple averaging for all missing pixels
        # This is the fastest but lowest quality approach
        
        # For green at red/blue locations: average of 4 neighbors
        g_avg = 0.25 * (
            np.roll(raw_in, -1, axis=0) + np.roll(raw_in, 1, axis=0) +
            np.roll(raw_in, -1, axis=1) + np.roll(raw_in, 1, axis=1)
        )
        
        g_channel = np.where(
            np.logical_or(mask_r == 1, mask_b == 1),
            g_avg,
            g_channel
        )
        
        # For red at green/blue locations: average of 4 neighbors
        r_avg = 0.25 * (
            np.roll(raw_in, -1, axis=0) + np.roll(raw_in, 1, axis=0) +
            np.roll(raw_in, -1, axis=1) + np.roll(raw_in, 1, axis=1)
        )
        
        r_channel = np.where(
            np.logical_or(mask_g == 1, mask_b == 1),
            r_avg,
            r_channel
        )
        
        # For blue at green/red locations: average of 4 neighbors
        b_avg = 0.25 * (
            np.roll(raw_in, -1, axis=0) + np.roll(raw_in, 1, axis=0) +
            np.roll(raw_in, -1, axis=1) + np.roll(raw_in, 1, axis=1)
        )
        
        b_channel = np.where(
            np.logical_or(mask_g == 1, mask_r == 1),
            b_avg,
            b_channel
        )
        
        # Assign channels to output
        demos_out[:, :, 0] = r_channel
        demos_out[:, :, 1] = g_channel
        demos_out[:, :, 2] = b_channel
        
        return demos_out
