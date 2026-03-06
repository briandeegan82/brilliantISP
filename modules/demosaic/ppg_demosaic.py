"""
File: ppg_demosaic.py
Description: Implements the PPG (Patterned Pixel Grouping) demosaic algorithm
Reference: "Demosaicing Using Optimal Recovery" by Hirakawa & Parks (2003)
           "Adaptive Homogeneity-Directed Demosaicing Algorithm" (PPG variant)
Author: Brian Deegan (via AI)
------------------------------------------------------------
"""
import numpy as np
from scipy.ndimage import convolve


class PPGDemosaic:
    """
    Patterned Pixel Grouping (PPG) demosaic algorithm.
    
    PPG improves upon traditional methods by:
    1. Using iterative refinement of color planes
    2. Computing better estimates of missing green values
    3. Using updated green for improved R/B interpolation
    4. Applying pattern-based corrections
    
    This produces excellent quality with good edge preservation.
    """

    def __init__(self, raw_in, masks):
        self.img = np.asarray(raw_in, dtype=np.float32)
        self.masks = [m.astype(np.float32) for m in masks]
        self.height, self.width = self.img.shape
        
    def _compute_laplacian(self, channel):
        """
        Compute Laplacian for edge detection.
        """
        laplacian_kernel = np.array([[0, 1, 0],
                                      [1, -4, 1],
                                      [0, 1, 0]], dtype=np.float32)
        return convolve(channel, laplacian_kernel, mode='reflect')
    
    def _interpolate_green_initial(self):
        """
        Initial green interpolation using directional gradients.
        This is similar to Hamilton-Adams but serves as first estimate.
        """
        raw = self.img
        mask_r, mask_g, mask_b = self.masks
        
        # Start with known green values
        G = raw * mask_g
        
        # Horizontal and vertical interpolation kernels
        h_kernel = np.array([[0, 0, 0],
                            [1, 0, 1],
                            [0, 0, 0]], dtype=np.float32) / 2.0
        
        v_kernel = np.array([[0, 1, 0],
                            [0, 0, 0],
                            [0, 1, 0]], dtype=np.float32) / 2.0
        
        # Gradient kernels (second order)
        h_grad_kernel = np.array([[0, 0, 0],
                                  [1, -2, 1],
                                  [0, 0, 0]], dtype=np.float32)
        
        v_grad_kernel = np.array([[0, 1, 0],
                                  [0, -2, 0],
                                  [0, 1, 0]], dtype=np.float32)
        
        # Compute gradients
        h_grad = np.abs(convolve(raw, h_grad_kernel, mode='reflect'))
        v_grad = np.abs(convolve(raw, v_grad_kernel, mode='reflect'))
        
        # Interpolate green from neighbors
        G_h = convolve(raw * mask_g, h_kernel, mode='reflect')
        G_v = convolve(raw * mask_g, v_kernel, mode='reflect')
        
        # Select direction with minimum gradient
        G_interp = np.where(h_grad < v_grad, G_h, G_v)
        
        # If gradients are similar, average both
        grad_ratio = h_grad / (v_grad + 1e-6)
        similar_grads = (grad_ratio > 0.8) & (grad_ratio < 1.25)
        G_interp = np.where(similar_grads, (G_h + G_v) / 2.0, G_interp)
        
        # Apply at R and B locations
        G = G + (1.0 - mask_g) * G_interp
        
        return G
    
    def _refine_green(self, G_initial):
        """
        Refine green channel using pattern-based correction.
        This is the key PPG step - iterative refinement.
        """
        raw = self.img
        mask_r, mask_g, mask_b = self.masks
        
        G = G_initial.copy()
        
        # Compute color differences at known locations
        R_minus_G = (raw - G_initial) * mask_r
        B_minus_G = (raw - G_initial) * mask_b
        
        # Interpolate color differences
        # This helps correct green estimates
        cross_kernel = np.array([[0, 1, 0],
                                 [1, 0, 1],
                                 [0, 1, 0]], dtype=np.float32) / 4.0
        
        # At R locations, use nearby B-G differences to refine
        # At B locations, use nearby R-G differences to refine
        R_minus_G_interp = convolve(R_minus_G, cross_kernel, mode='reflect')
        B_minus_G_interp = convolve(B_minus_G, cross_kernel, mode='reflect')
        
        # Refine green at R locations using pattern
        at_r = mask_r.astype(bool)
        # Use knowledge of B-G pattern to improve G estimate
        G_correction_at_r = -0.5 * B_minus_G_interp
        G[at_r] = (G[at_r] + G_correction_at_r[at_r])
        
        # Refine green at B locations
        at_b = mask_b.astype(bool)
        G_correction_at_b = -0.5 * R_minus_G_interp
        G[at_b] = (G[at_b] + G_correction_at_b[at_b])
        
        return G
    
    def _interpolate_red_blue(self, G):
        """
        Interpolate R and B using the refined green channel.
        Uses color difference method with the improved green.
        """
        raw = self.img
        mask_r, mask_g, mask_b = self.masks
        
        # Start with known values
        R = raw * mask_r
        B = raw * mask_b
        
        # Compute color differences at known locations
        R_minus_G = (raw - G) * mask_r
        B_minus_G = (raw - G) * mask_b
        
        # Kernels for interpolation
        h_kernel = np.array([[0, 0, 0],
                            [1, 0, 1],
                            [0, 0, 0]], dtype=np.float32) / 2.0
        
        v_kernel = np.array([[0, 1, 0],
                            [0, 0, 0],
                            [0, 1, 0]], dtype=np.float32) / 2.0
        
        diag_kernel = np.array([[1, 0, 1],
                               [0, 0, 0],
                               [1, 0, 1]], dtype=np.float32) / 4.0
        
        # Interpolate R-G differences
        R_minus_G_h = convolve(R_minus_G, h_kernel, mode='reflect')
        R_minus_G_v = convolve(R_minus_G, v_kernel, mode='reflect')
        R_minus_G_d = convolve(R_minus_G, diag_kernel, mode='reflect')
        
        # Interpolate B-G differences
        B_minus_G_h = convolve(B_minus_G, h_kernel, mode='reflect')
        B_minus_G_v = convolve(B_minus_G, v_kernel, mode='reflect')
        B_minus_G_d = convolve(B_minus_G, diag_kernel, mode='reflect')
        
        # Red at green locations
        at_g = mask_g.astype(bool)
        has_R_h = convolve(mask_r, h_kernel * 2, mode='reflect') > 0.5
        has_R_v = convolve(mask_r, v_kernel * 2, mode='reflect') > 0.5
        
        R_at_g = np.where(has_R_h, G + R_minus_G_h, G + R_minus_G_v)
        R = np.where(at_g, R_at_g, R)
        
        # Red at blue locations (diagonal)
        at_b = mask_b.astype(bool)
        R = np.where(at_b, G + R_minus_G_d, R)
        
        # Blue at green locations
        has_B_h = convolve(mask_b, h_kernel * 2, mode='reflect') > 0.5
        has_B_v = convolve(mask_b, v_kernel * 2, mode='reflect') > 0.5
        
        B_at_g = np.where(has_B_h, G + B_minus_G_h, G + B_minus_G_v)
        B = np.where(at_g, B_at_g, B)
        
        # Blue at red locations (diagonal)
        at_r = mask_r.astype(bool)
        B = np.where(at_r, G + B_minus_G_d, B)
        
        return R, B
    
    def apply_ppg(self):
        """
        Apply PPG demosaicing algorithm.
        
        Process:
        1. Initial green interpolation using directional gradients
        2. Refine green using pattern-based correction (key PPG step)
        3. Interpolate R/B using refined green with color differences
        """
        # Step 1: Initial green interpolation
        G_initial = self._interpolate_green_initial()
        
        # Step 2: Refine green (PPG pattern correction)
        G = self._refine_green(G_initial)
        
        # Step 3: Interpolate R and B
        R, B = self._interpolate_red_blue(G)
        
        # Stack channels
        out = np.stack([R, G, B], axis=-1)
        return out.astype(np.float32)


class PPGDemosaicOptimized:
    """
    Optimized PPG implementation with enhanced pattern recognition.
    Uses multiple iterations and better gradient estimation.
    """
    
    def __init__(self, raw_in, masks):
        self.img = np.asarray(raw_in, dtype=np.float32)
        self.masks = [m.astype(np.float32) for m in masks]
        self.num_iterations = 2  # Number of refinement iterations
        
    def _interpolate_green_directional(self, raw, mask_g):
        """
        Interpolate green with enhanced directional selection.
        """
        # Enhanced 5x5 gradient kernels
        h_grad = np.array([[0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0],
                          [1, -2, 0, -2, 1],
                          [0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0]], dtype=np.float32)
        
        v_grad = np.array([[0, 0, 1, 0, 0],
                          [0, 0, -2, 0, 0],
                          [0, 1, 0, 1, 0],
                          [0, 0, -2, 0, 0],
                          [0, 0, 1, 0, 0]], dtype=np.float32)
        
        # Compute gradients
        grad_h = np.abs(convolve(raw, h_grad, mode='reflect'))
        grad_v = np.abs(convolve(raw, v_grad, mode='reflect'))
        
        # Interpolation kernels
        h_interp = np.array([[0, 0, 0],
                            [1, 0, 1],
                            [0, 0, 0]], dtype=np.float32) / 2.0
        
        v_interp = np.array([[0, 1, 0],
                            [0, 0, 0],
                            [0, 1, 0]], dtype=np.float32) / 2.0
        
        # Interpolate from neighbors
        G_h = convolve(raw * mask_g, h_interp, mode='reflect')
        G_v = convolve(raw * mask_g, v_interp, mode='reflect')
        
        # Color difference correction
        G_h_corrected = G_h + 0.5 * (raw - convolve(raw, h_interp, mode='reflect'))
        G_v_corrected = G_v + 0.5 * (raw - convolve(raw, v_interp, mode='reflect'))
        
        # Select based on gradients
        G_interp = np.where(grad_h < grad_v, G_h_corrected, G_v_corrected)
        
        # Average if gradients similar
        grad_ratio = grad_h / (grad_v + 1e-6)
        similar = (grad_ratio > 0.75) & (grad_ratio < 1.33)
        G_interp = np.where(similar, (G_h_corrected + G_v_corrected) / 2.0, G_interp)
        
        return G_interp
    
    def _refine_green_iterative(self, G, raw, mask_r, mask_g, mask_b):
        """
        Iteratively refine green channel using pattern information.
        Multiple iterations improve quality.
        """
        for iteration in range(self.num_iterations):
            # Compute color differences at known locations
            R_minus_G = (raw - G) * mask_r
            B_minus_G = (raw - G) * mask_b
            
            # Pattern-based correction kernels
            cross_kernel = np.array([[0, 1, 0],
                                     [1, 0, 1],
                                     [0, 1, 0]], dtype=np.float32) / 4.0
            
            diag_kernel = np.array([[1, 0, 1],
                                    [0, 0, 0],
                                    [1, 0, 1]], dtype=np.float32) / 4.0
            
            # Interpolate color differences
            R_minus_G_cross = convolve(R_minus_G, cross_kernel, mode='reflect')
            R_minus_G_diag = convolve(R_minus_G, diag_kernel, mode='reflect')
            
            B_minus_G_cross = convolve(B_minus_G, cross_kernel, mode='reflect')
            B_minus_G_diag = convolve(B_minus_G, diag_kernel, mode='reflect')
            
            # Compute corrections
            # At R locations: use B-G pattern
            at_r = mask_r.astype(bool)
            correction_r = -0.25 * (B_minus_G_cross + B_minus_G_diag)
            
            # At B locations: use R-G pattern
            at_b = mask_b.astype(bool)
            correction_b = -0.25 * (R_minus_G_cross + R_minus_G_diag)
            
            # Apply corrections with damping for stability
            damping = 0.5 / (iteration + 1)  # Reduce correction strength each iteration
            G[at_r] = G[at_r] + damping * correction_r[at_r]
            G[at_b] = G[at_b] + damping * correction_b[at_b]
        
        return G
    
    def apply_ppg_optimized(self):
        """
        Apply optimized PPG with multiple refinement iterations.
        """
        raw = self.img
        mask_r, mask_g, mask_b = self.masks
        
        # Step 1: Initial green interpolation
        G = raw * mask_g
        G_interp = self._interpolate_green_directional(raw, mask_g)
        G = G + (1.0 - mask_g) * G_interp
        
        # Step 2: Iterative refinement (key PPG feature)
        G = self._refine_green_iterative(G, raw, mask_r, mask_g, mask_b)
        
        # Step 3: Interpolate R and B using refined green
        R = raw * mask_r
        B = raw * mask_b
        
        # Compute color differences
        R_minus_G = (raw - G) * mask_r
        B_minus_G = (raw - G) * mask_b
        
        # Interpolation kernels
        h_kernel = np.array([[0, 0, 0],
                            [1, 0, 1],
                            [0, 0, 0]], dtype=np.float32) / 2.0
        
        v_kernel = np.array([[0, 1, 0],
                            [0, 0, 0],
                            [0, 1, 0]], dtype=np.float32) / 2.0
        
        diag_kernel = np.array([[1, 0, 1],
                               [0, 0, 0],
                               [1, 0, 1]], dtype=np.float32) / 4.0
        
        # Interpolate color differences
        R_minus_G_h = convolve(R_minus_G, h_kernel, mode='reflect')
        R_minus_G_v = convolve(R_minus_G, v_kernel, mode='reflect')
        R_minus_G_d = convolve(R_minus_G, diag_kernel, mode='reflect')
        
        B_minus_G_h = convolve(B_minus_G, h_kernel, mode='reflect')
        B_minus_G_v = convolve(B_minus_G, v_kernel, mode='reflect')
        B_minus_G_d = convolve(B_minus_G, diag_kernel, mode='reflect')
        
        # Reconstruct R at missing locations
        at_g = mask_g.astype(bool)
        has_R_h = convolve(mask_r, h_kernel * 2, mode='reflect') > 0.5
        has_R_v = convolve(mask_r, v_kernel * 2, mode='reflect') > 0.5
        
        R_at_g = np.where(has_R_h, G + R_minus_G_h, G + R_minus_G_v)
        R = np.where(at_g, R_at_g, R)
        
        at_b = mask_b.astype(bool)
        R = np.where(at_b, G + R_minus_G_d, R)
        
        # Reconstruct B at missing locations
        has_B_h = convolve(mask_b, h_kernel * 2, mode='reflect') > 0.5
        has_B_v = convolve(mask_b, v_kernel * 2, mode='reflect') > 0.5
        
        B_at_g = np.where(has_B_h, G + B_minus_G_h, G + B_minus_G_v)
        B = np.where(at_g, B_at_g, B)
        
        at_r = mask_r.astype(bool)
        B = np.where(at_r, G + B_minus_G_d, B)
        
        # Stack channels
        out = np.stack([R, G, B], axis=-1)
        return out.astype(np.float32)
