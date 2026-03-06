"""
File: hamilton_adams_demosaic.py
Description: Implements the Hamilton-Adams demosaic algorithm
Reference: "Adaptive Color Plan Interpolation in Single Sensor Color Electronic Camera"
           by J.F. Hamilton Jr. and J.E. Adams Jr. (1997)
Author: Brian Deegan (via AI)
------------------------------------------------------------
"""
import numpy as np
from scipy.ndimage import convolve


class HamiltonAdamsDemosaic:
    """
    Hamilton-Adams demosaic algorithm.
    
    This algorithm uses:
    1. Second-order Laplacian for edge detection
    2. Color difference (constant hue assumption)
    3. Directional interpolation based on gradients
    
    The key insight is that color ratios (R/G, B/G) are more constant
    than absolute color values, especially across edges.
    """

    def __init__(self, raw_in, masks):
        self.img = np.asarray(raw_in, dtype=np.float32)
        self.masks = [m.astype(np.float32) for m in masks]
        self.height, self.width = self.img.shape
        
    def _safe_divide(self, numerator, denominator, epsilon=1e-6):
        """Safe division avoiding divide by zero."""
        return numerator / (denominator + epsilon)
    
    def _compute_gradients(self, channel):
        """
        Compute horizontal and vertical gradients using Laplacian.
        Returns gradient magnitude in each direction.
        """
        # Horizontal gradient (along rows)
        h_grad_kernel = np.array([[0, 0, 0],
                                   [1, -2, 1],
                                   [0, 0, 0]], dtype=np.float32)
        
        # Vertical gradient (along columns)
        v_grad_kernel = np.array([[0, 1, 0],
                                   [0, -2, 0],
                                   [0, 1, 0]], dtype=np.float32)
        
        h_grad = np.abs(convolve(channel, h_grad_kernel, mode='reflect'))
        v_grad = np.abs(convolve(channel, v_grad_kernel, mode='reflect'))
        
        return h_grad, v_grad
    
    def _interpolate_green(self):
        """
        Interpolate green channel at R and B locations using Hamilton-Adams method.
        Uses directional interpolation based on color differences.
        """
        raw = self.img
        mask_r, mask_g, mask_b = self.masks
        
        # Start with known green values
        G = raw * mask_g
        
        # Kernels for color difference estimation
        # Horizontal neighbors
        h_kernel = np.array([[0, 0, 0],
                            [1, 0, 1],
                            [0, 0, 0]], dtype=np.float32) / 2.0
        
        # Vertical neighbors
        v_kernel = np.array([[0, 1, 0],
                            [0, 0, 0],
                            [0, 1, 0]], dtype=np.float32) / 2.0
        
        # Second derivative (Laplacian) for edge detection
        h_laplacian = np.array([[0, 0, 0],
                               [1, -2, 1],
                               [0, 0, 0]], dtype=np.float32)
        
        v_laplacian = np.array([[0, 1, 0],
                               [0, -2, 0],
                               [0, 1, 0]], dtype=np.float32)
        
        # Interpolate green at R locations
        at_r = mask_r.astype(bool)
        if np.any(at_r):
            # Get green neighbors
            G_h = convolve(raw * mask_g, h_kernel, mode='reflect')
            G_v = convolve(raw * mask_g, v_kernel, mode='reflect')
            
            # Compute gradients using R channel (for edge detection)
            R_h_grad = np.abs(convolve(raw * mask_r, h_laplacian, mode='reflect'))
            R_v_grad = np.abs(convolve(raw * mask_r, v_laplacian, mode='reflect'))
            
            # Compute color difference correction
            # At R locations, estimate G using R and nearby G values
            R_at_r = raw * mask_r
            
            # Horizontal and vertical estimates with color difference
            G_est_h = G_h + 0.5 * convolve(raw - convolve(raw * mask_g, h_kernel, mode='reflect'), 
                                           h_kernel, mode='reflect')
            G_est_v = G_v + 0.5 * convolve(raw - convolve(raw * mask_g, v_kernel, mode='reflect'),
                                           v_kernel, mode='reflect')
            
            # Directional selection based on gradients
            # Use direction with smaller gradient
            G_at_r = np.where(R_h_grad < R_v_grad, G_est_h, G_est_v)
            
            # If gradients are similar, average both directions
            grad_threshold = 1.2
            similar_grads = (R_h_grad / (R_v_grad + 1e-6) > 1/grad_threshold) & \
                           (R_h_grad / (R_v_grad + 1e-6) < grad_threshold)
            G_at_r = np.where(similar_grads, (G_est_h + G_est_v) / 2.0, G_at_r)
            
            G[at_r] = G_at_r[at_r]
        
        # Interpolate green at B locations (symmetric to R)
        at_b = mask_b.astype(bool)
        if np.any(at_b):
            G_h = convolve(raw * mask_g, h_kernel, mode='reflect')
            G_v = convolve(raw * mask_g, v_kernel, mode='reflect')
            
            B_h_grad = np.abs(convolve(raw * mask_b, h_laplacian, mode='reflect'))
            B_v_grad = np.abs(convolve(raw * mask_b, v_laplacian, mode='reflect'))
            
            G_est_h = G_h + 0.5 * convolve(raw - convolve(raw * mask_g, h_kernel, mode='reflect'),
                                           h_kernel, mode='reflect')
            G_est_v = G_v + 0.5 * convolve(raw - convolve(raw * mask_g, v_kernel, mode='reflect'),
                                           v_kernel, mode='reflect')
            
            G_at_b = np.where(B_h_grad < B_v_grad, G_est_h, G_est_v)
            
            grad_threshold = 1.2
            similar_grads = (B_h_grad / (B_v_grad + 1e-6) > 1/grad_threshold) & \
                           (B_h_grad / (B_v_grad + 1e-6) < grad_threshold)
            G_at_b = np.where(similar_grads, (G_est_h + G_est_v) / 2.0, G_at_b)
            
            G[at_b] = G_at_b[at_b]
        
        return G
    
    def _interpolate_red_blue(self, G):
        """
        Interpolate R and B channels using color difference method.
        Uses the constant color ratio assumption: R/G and B/G are locally constant.
        """
        raw = self.img
        mask_r, mask_g, mask_b = self.masks
        
        # Start with known values
        R = raw * mask_r
        B = raw * mask_b
        
        # Kernels for different interpolation patterns
        h_kernel = np.array([[0, 0, 0],
                            [1, 0, 1],
                            [0, 0, 0]], dtype=np.float32) / 2.0
        
        v_kernel = np.array([[0, 1, 0],
                            [0, 0, 0],
                            [0, 1, 0]], dtype=np.float32) / 2.0
        
        diag_kernel = np.array([[1, 0, 1],
                               [0, 0, 0],
                               [1, 0, 1]], dtype=np.float32) / 4.0
        
        # === Interpolate Red Channel ===
        
        # R at green locations in red rows (horizontal neighbors)
        at_g = mask_g.astype(bool)
        has_R_h = convolve(mask_r, h_kernel * 2, mode='reflect') > 0.5
        has_R_v = convolve(mask_r, v_kernel * 2, mode='reflect') > 0.5
        
        # Compute R using color difference: R = G + (R - G)
        R_diff_h = convolve((raw - G) * mask_r, h_kernel, mode='reflect')
        R_diff_v = convolve((raw - G) * mask_r, v_kernel, mode='reflect')
        
        R_at_g_h = G + R_diff_h
        R_at_g_v = G + R_diff_v
        
        # Select based on which direction has red neighbors
        R_at_g = np.where(has_R_h, R_at_g_h, R_at_g_v)
        R[at_g] = R_at_g[at_g]
        
        # R at blue locations (diagonal neighbors)
        at_b = mask_b.astype(bool)
        R_diff_d = convolve((raw - G) * mask_r, diag_kernel, mode='reflect')
        R_at_b = G + R_diff_d
        R[at_b] = R_at_b[at_b]
        
        # === Interpolate Blue Channel ===
        
        # B at green locations
        has_B_h = convolve(mask_b, h_kernel * 2, mode='reflect') > 0.5
        has_B_v = convolve(mask_b, v_kernel * 2, mode='reflect') > 0.5
        
        B_diff_h = convolve((raw - G) * mask_b, h_kernel, mode='reflect')
        B_diff_v = convolve((raw - G) * mask_b, v_kernel, mode='reflect')
        
        B_at_g_h = G + B_diff_h
        B_at_g_v = G + B_diff_v
        
        B_at_g = np.where(has_B_h, B_at_g_h, B_at_g_v)
        B[at_g] = B_at_g[at_g]
        
        # B at red locations (diagonal neighbors)
        at_r = mask_r.astype(bool)
        B_diff_d = convolve((raw - G) * mask_b, diag_kernel, mode='reflect')
        B_at_r = G + B_diff_d
        B[at_r] = B_at_r[at_r]
        
        return R, B
    
    def apply_hamilton_adams(self):
        """
        Apply Hamilton-Adams demosaicing algorithm.
        
        Process:
        1. Interpolate green at R/B locations using directional gradients
        2. Interpolate R/B using color difference method with green
        """
        # Step 1: Interpolate green (most critical step)
        G = self._interpolate_green()
        
        # Step 2: Interpolate red and blue using color differences
        R, B = self._interpolate_red_blue(G)
        
        # Stack channels
        out = np.stack([R, G, B], axis=-1)
        return out.astype(np.float32)


class HamiltonAdamsOptimized:
    """
    Optimized Hamilton-Adams implementation with improved gradient computation.
    Uses more sophisticated edge detection and color ratio preservation.
    """
    
    def __init__(self, raw_in, masks):
        self.img = np.asarray(raw_in, dtype=np.float32)
        self.masks = [m.astype(np.float32) for m in masks]
        
    def apply_hamilton_adams_optimized(self):
        """
        Optimized Hamilton-Adams with enhanced gradient computation.
        """
        raw = self.img
        mask_r, mask_g, mask_b = self.masks
        
        # === Step 1: Interpolate Green ===
        G = raw * mask_g
        
        # Enhanced gradient kernels with second derivative
        h_interp = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 1, 0, 1, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]], dtype=np.float32) / 2.0
        
        v_interp = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0]], dtype=np.float32) / 2.0
        
        # Gradient estimation with second derivative
        h_grad = np.array([[0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [1, 0, -2, 0, 1],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0]], dtype=np.float32)
        
        v_grad = np.array([[0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, -2, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0]], dtype=np.float32)
        
        # Compute green interpolations and gradients
        G_h = convolve(raw * mask_g, h_interp, mode='reflect')
        G_v = convolve(raw * mask_g, v_interp, mode='reflect')
        
        grad_h = np.abs(convolve(raw, h_grad, mode='reflect'))
        grad_v = np.abs(convolve(raw, v_grad, mode='reflect'))
        
        # Directional interpolation with color difference correction
        # Add raw value contribution at center
        center_weight = 0.5
        G_h_corrected = G_h + center_weight * (raw - convolve(raw, h_interp, mode='reflect'))
        G_v_corrected = G_v + center_weight * (raw - convolve(raw, v_interp, mode='reflect'))
        
        # Select direction with minimum gradient
        G_interp = np.where(grad_h < grad_v, G_h_corrected, G_v_corrected)
        
        # If gradients similar, average both
        grad_ratio = grad_h / (grad_v + 1e-6)
        similar_grads = (grad_ratio > 0.8) & (grad_ratio < 1.25)
        G_interp = np.where(similar_grads, (G_h_corrected + G_v_corrected) / 2.0, G_interp)
        
        # Apply only at R and B locations
        G = G + (1.0 - mask_g) * G_interp
        
        # === Step 2: Interpolate R and B using color ratios ===
        
        # Kernels for R/B interpolation
        h_kernel = np.array([[0, 0, 0],
                            [1, 0, 1],
                            [0, 0, 0]], dtype=np.float32) / 2.0
        
        v_kernel = np.array([[0, 1, 0],
                            [0, 0, 0],
                            [0, 1, 0]], dtype=np.float32) / 2.0
        
        diag_kernel = np.array([[1, 0, 1],
                               [0, 0, 0],
                               [1, 0, 1]], dtype=np.float32) / 4.0
        
        # Red interpolation using color difference
        R = raw * mask_r
        R_diff_h = convolve((raw - G) * mask_r, h_kernel, mode='reflect')
        R_diff_v = convolve((raw - G) * mask_r, v_kernel, mode='reflect')
        R_diff_d = convolve((raw - G) * mask_r, diag_kernel, mode='reflect')
        
        has_R_h = convolve(mask_r, h_kernel * 2, mode='reflect') > 0.5
        has_R_v = convolve(mask_r, v_kernel * 2, mode='reflect') > 0.5
        
        at_g = mask_g.astype(bool)
        R_at_g = np.where(has_R_h, G + R_diff_h, G + R_diff_v)
        R = np.where(at_g, R_at_g, R)
        
        at_b = mask_b.astype(bool)
        R = np.where(at_b, G + R_diff_d, R)
        
        # Blue interpolation (symmetric)
        B = raw * mask_b
        B_diff_h = convolve((raw - G) * mask_b, h_kernel, mode='reflect')
        B_diff_v = convolve((raw - G) * mask_b, v_kernel, mode='reflect')
        B_diff_d = convolve((raw - G) * mask_b, diag_kernel, mode='reflect')
        
        has_B_h = convolve(mask_b, h_kernel * 2, mode='reflect') > 0.5
        has_B_v = convolve(mask_b, v_kernel * 2, mode='reflect') > 0.5
        
        B_at_g = np.where(has_B_h, G + B_diff_h, G + B_diff_v)
        B = np.where(at_g, B_at_g, B)
        
        at_r = mask_r.astype(bool)
        B = np.where(at_r, G + B_diff_d, B)
        
        # Stack channels
        out = np.stack([R, G, B], axis=-1)
        return out.astype(np.float32)
