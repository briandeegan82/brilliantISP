"""
File: ahd_demosaic.py
Description: Implements the AHD (Adaptive Homogeneity-Directed) demosaic algorithm
Reference: "Adaptive Homogeneity-Directed Demosaicing Algorithm" by Hirakawa & Parks (2005)
Author: Brian Deegan (via AI)
------------------------------------------------------------
"""
import numpy as np
from scipy.ndimage import convolve, median_filter


class AHDDemosaic:
    """
    AHD (Adaptive Homogeneity-Directed) demosaic algorithm.
    
    AHD is a sophisticated method that:
    1. Creates two initial interpolations (horizontal and vertical)
    2. Computes homogeneity maps for both directions
    3. Selects the best interpolation at each pixel based on homogeneity
    4. Applies median filtering to reduce artifacts
    
    This produces excellent quality with superior artifact reduction.
    """

    def __init__(self, raw_in, masks):
        self.img = np.asarray(raw_in, dtype=np.float32)
        self.masks = [m.astype(np.float32) for m in masks]
        self.height, self.width = self.img.shape
        self.epsilon = 1e-6
        
    def _interpolate_green_horizontal(self):
        """
        Interpolate green channel using horizontal neighbors.
        """
        raw = self.img
        mask_r, mask_g, mask_b = self.masks
        
        G = raw * mask_g
        
        # Horizontal kernel for green
        h_kernel = np.array([[0, 0, 0],
                            [1, 0, 1],
                            [0, 0, 0]], dtype=np.float32) / 2.0
        
        # Interpolate at R and B locations
        G_h = convolve(raw * mask_g, h_kernel, mode='reflect')
        
        # Simple interpolation without aggressive correction
        at_r = mask_r.astype(bool)
        at_b = mask_b.astype(bool)
        
        G[at_r] = G_h[at_r]
        G[at_b] = G_h[at_b]
        
        return G
    
    def _interpolate_green_vertical(self):
        """
        Interpolate green channel using vertical neighbors.
        """
        raw = self.img
        mask_r, mask_g, mask_b = self.masks
        
        G = raw * mask_g
        
        # Vertical kernel for green
        v_kernel = np.array([[0, 1, 0],
                            [0, 0, 0],
                            [0, 1, 0]], dtype=np.float32) / 2.0
        
        # Interpolate at R and B locations
        G_v = convolve(raw * mask_g, v_kernel, mode='reflect')
        
        # Simple interpolation without aggressive correction
        at_r = mask_r.astype(bool)
        at_b = mask_b.astype(bool)
        
        G[at_r] = G_v[at_r]
        G[at_b] = G_v[at_b]
        
        return G
    
    def _interpolate_rb_from_green(self, G):
        """
        Interpolate R and B channels using green as reference.
        Creates both horizontal and vertical versions.
        """
        raw = self.img
        mask_r, mask_g, mask_b = self.masks
        
        # Horizontal interpolation
        h_kernel = np.array([[0, 0, 0],
                            [1, 0, 1],
                            [0, 0, 0]], dtype=np.float32) / 2.0
        
        # Vertical interpolation
        v_kernel = np.array([[0, 1, 0],
                            [0, 0, 0],
                            [0, 1, 0]], dtype=np.float32) / 2.0
        
        # Diagonal interpolation
        d_kernel = np.array([[1, 0, 1],
                            [0, 0, 0],
                            [1, 0, 1]], dtype=np.float32) / 4.0
        
        # Red channel
        R = raw * mask_r
        R_minus_G = (raw - G) * mask_r
        
        R_h = convolve(R_minus_G, h_kernel, mode='reflect')
        R_v = convolve(R_minus_G, v_kernel, mode='reflect')
        R_d = convolve(R_minus_G, d_kernel, mode='reflect')
        
        # Determine orientation at each location
        at_g = mask_g.astype(bool)
        has_R_h = convolve(mask_r, h_kernel * 2, mode='reflect') > 0.5
        has_R_v = convolve(mask_r, v_kernel * 2, mode='reflect') > 0.5
        
        R_at_g = np.where(has_R_h, G + R_h, G + R_v)
        R = np.where(at_g, R_at_g, R)
        
        at_b = mask_b.astype(bool)
        R = np.where(at_b, G + R_d, R)
        
        # Blue channel (symmetric)
        B = raw * mask_b
        B_minus_G = (raw - G) * mask_b
        
        B_h = convolve(B_minus_G, h_kernel, mode='reflect')
        B_v = convolve(B_minus_G, v_kernel, mode='reflect')
        B_d = convolve(B_minus_G, d_kernel, mode='reflect')
        
        has_B_h = convolve(mask_b, h_kernel * 2, mode='reflect') > 0.5
        has_B_v = convolve(mask_b, v_kernel * 2, mode='reflect') > 0.5
        
        B_at_g = np.where(has_B_h, G + B_h, G + B_v)
        B = np.where(at_g, B_at_g, B)
        
        at_r = mask_r.astype(bool)
        B = np.where(at_r, G + B_d, B)
        
        return R, B
    
    def _compute_homogeneity_map(self, rgb_h, rgb_v):
        """
        Compute homogeneity map comparing horizontal and vertical interpolations.
        
        Homogeneity is measured by the variance in a local neighborhood.
        Lower variance = more homogeneous = better choice.
        """
        # Compute color differences
        diff_h = np.abs(rgb_h[:, :, 0] - rgb_h[:, :, 1]) + np.abs(rgb_h[:, :, 1] - rgb_h[:, :, 2])
        diff_v = np.abs(rgb_v[:, :, 0] - rgb_v[:, :, 1]) + np.abs(rgb_v[:, :, 1] - rgb_v[:, :, 2])
        
        # Compute local variance (homogeneity measure)
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)
        
        # Mean of differences
        mean_diff_h = convolve(diff_h, kernel, mode='reflect')
        mean_diff_v = convolve(diff_v, kernel, mode='reflect')
        
        # Variance of differences (measure of homogeneity)
        var_h = convolve((diff_h - mean_diff_h) ** 2, kernel, mode='reflect')
        var_v = convolve((diff_v - mean_diff_v) ** 2, kernel, mode='reflect')
        
        # Create decision map (True = use horizontal, False = use vertical)
        decision_map = var_h < var_v
        
        return decision_map
    
    def _blend_directions(self, rgb_h, rgb_v, decision_map):
        """
        Blend horizontal and vertical interpolations based on decision map.
        """
        # Stack decision map for all three channels
        decision_3d = np.stack([decision_map, decision_map, decision_map], axis=-1)
        
        # Select based on homogeneity
        rgb_blended = np.where(decision_3d, rgb_h, rgb_v)
        
        return rgb_blended
    
    def _median_filter_artifact_removal(self, rgb):
        """
        Apply very light median filtering only to extreme outliers.
        """
        result = rgb.copy()
        
        # Only apply minimal filtering to avoid destroying detail
        for c in range(3):
            # Compute local statistics
            mean_local = convolve(rgb[:, :, c], np.ones((3, 3)) / 9.0, mode='reflect')
            diff = np.abs(rgb[:, :, c] - mean_local)
            
            # Only filter extreme outliers (top 1%)
            threshold = np.percentile(diff, 99)
            outlier_mask = diff > threshold
            
            if np.any(outlier_mask):
                # Very light filtering only on outliers
                filtered = median_filter(rgb[:, :, c], size=3)
                result[:, :, c] = np.where(outlier_mask, 
                                          0.7 * rgb[:, :, c] + 0.3 * filtered,
                                          rgb[:, :, c])
        
        return result
    
    def apply_ahd(self):
        """
        Apply AHD demosaicing algorithm.
        
        Process:
        1. Create horizontal interpolation (G, R, B)
        2. Create vertical interpolation (G, R, B)
        3. Compute homogeneity maps
        4. Select best direction at each pixel
        5. Apply median filtering for artifact removal
        """
        # Step 1: Horizontal interpolation
        G_h = self._interpolate_green_horizontal()
        R_h, B_h = self._interpolate_rb_from_green(G_h)
        rgb_h = np.stack([R_h, G_h, B_h], axis=-1)
        
        # Step 2: Vertical interpolation
        G_v = self._interpolate_green_vertical()
        R_v, B_v = self._interpolate_rb_from_green(G_v)
        rgb_v = np.stack([R_v, G_v, B_v], axis=-1)
        
        # Step 3 & 4: Compute homogeneity and blend
        decision_map = self._compute_homogeneity_map(rgb_h, rgb_v)
        rgb_blended = self._blend_directions(rgb_h, rgb_v, decision_map)
        
        # Step 5: Artifact removal
        rgb_final = self._median_filter_artifact_removal(rgb_blended)
        
        return rgb_final.astype(np.float32)


class AHDDemosaicOptimized:
    """
    Optimized AHD implementation with enhanced homogeneity analysis.
    Uses more sophisticated metrics and smoother blending.
    """
    
    def __init__(self, raw_in, masks):
        self.img = np.asarray(raw_in, dtype=np.float32)
        self.masks = [m.astype(np.float32) for m in masks]
        self.epsilon = 1e-6
        
    def _directional_green_interpolation(self, direction='horizontal'):
        """
        Enhanced directional green interpolation with better color difference.
        """
        raw = self.img
        mask_r, mask_g, mask_b = self.masks
        
        G = raw * mask_g
        
        if direction == 'horizontal':
            kernel = np.array([[0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [1, 2, 0, 2, 1],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0]], dtype=np.float32) / 4.0
        else:  # vertical
            kernel = np.array([[0, 0, 1, 0, 0],
                              [0, 0, 2, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 2, 0, 0],
                              [0, 0, 1, 0, 0]], dtype=np.float32) / 4.0
        
        G_interp = convolve(raw * mask_g, kernel, mode='reflect')
        
        # Enhanced color difference correction
        at_rb = (1.0 - mask_g).astype(bool)
        correction = 0.5 * (raw - convolve(raw, kernel, mode='reflect'))
        
        G[at_rb] = G_interp[at_rb] + correction[at_rb]
        
        return G
    
    def _compute_enhanced_homogeneity(self, rgb_h, rgb_v):
        """
        Enhanced homogeneity computation using multiple metrics.
        """
        # Metric 1: Color difference variance
        diff_h = np.sum(np.abs(np.diff(rgb_h, axis=2)), axis=2)
        diff_v = np.sum(np.abs(np.diff(rgb_v, axis=2)), axis=2)
        
        # Metric 2: Gradient magnitude
        grad_kernel = np.array([[-1, 0, 1]], dtype=np.float32)
        
        grad_h_sum = 0
        grad_v_sum = 0
        for c in range(3):
            grad_h_sum += np.abs(convolve(rgb_h[:, :, c], grad_kernel, mode='reflect'))
            grad_v_sum += np.abs(convolve(rgb_v[:, :, c], grad_kernel, mode='reflect'))
        
        # Metric 3: Local variance
        kernel = np.ones((7, 7), dtype=np.float32) / 49.0
        
        var_h = convolve(diff_h ** 2, kernel, mode='reflect')
        var_v = convolve(diff_v ** 2, kernel, mode='reflect')
        
        # Combined metric (weighted)
        metric_h = 0.4 * var_h + 0.3 * diff_h + 0.3 * grad_h_sum
        metric_v = 0.4 * var_v + 0.3 * diff_v + 0.3 * grad_v_sum
        
        # Soft decision (with transition zone)
        ratio = metric_h / (metric_v + self.epsilon)
        
        # Smooth blending weight (0 = vertical, 1 = horizontal)
        # Use sigmoid-like function for smooth transition
        alpha = 1.0 / (1.0 + np.exp(5 * (ratio - 1.0)))
        
        return alpha
    
    def _smooth_blend(self, rgb_h, rgb_v, alpha):
        """
        Smooth blending with spatially varying weights.
        """
        # Expand alpha to 3 channels
        alpha_3d = np.stack([alpha, alpha, alpha], axis=-1)
        
        # Weighted blend
        rgb_blended = alpha_3d * rgb_h + (1.0 - alpha_3d) * rgb_v
        
        return rgb_blended
    
    def _adaptive_artifact_removal(self, rgb):
        """
        Very gentle artifact removal that preserves detail.
        """
        result = rgb.copy()
        
        # Only apply minimal filtering
        for c in range(3):
            # Detect only extreme outliers
            mean_local = convolve(rgb[:, :, c], np.ones((5, 5)) / 25.0, mode='reflect')
            diff = np.abs(rgb[:, :, c] - mean_local)
            
            # Only filter top 2% outliers
            threshold = np.percentile(diff, 98)
            outlier_mask = diff > threshold
            
            if np.any(outlier_mask):
                # Very gentle filtering
                filtered = median_filter(rgb[:, :, c], size=3)
                result[:, :, c] = np.where(outlier_mask,
                                          0.8 * rgb[:, :, c] + 0.2 * filtered,
                                          rgb[:, :, c])
        
        return result
    
    def apply_ahd_optimized(self):
        """
        Apply optimized AHD with enhanced homogeneity analysis.
        """
        # Horizontal interpolation
        G_h = self._directional_green_interpolation('horizontal')
        R_h, B_h = self._interpolate_rb_from_green_enhanced(G_h)
        rgb_h = np.stack([R_h, G_h, B_h], axis=-1)
        
        # Vertical interpolation
        G_v = self._directional_green_interpolation('vertical')
        R_v, B_v = self._interpolate_rb_from_green_enhanced(G_v)
        rgb_v = np.stack([R_v, G_v, B_v], axis=-1)
        
        # Enhanced homogeneity analysis with smooth blending
        alpha = self._compute_enhanced_homogeneity(rgb_h, rgb_v)
        rgb_blended = self._smooth_blend(rgb_h, rgb_v, alpha)
        
        # Adaptive artifact removal
        rgb_final = self._adaptive_artifact_removal(rgb_blended)
        
        return rgb_final.astype(np.float32)
    
    def _interpolate_rb_from_green_enhanced(self, G):
        """Enhanced R/B interpolation with better color difference handling."""
        raw = self.img
        mask_r, mask_g, mask_b = self.masks
        
        # Enhanced kernels
        h_kernel = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [1, 2, 0, 2, 1],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]], dtype=np.float32) / 4.0
        
        v_kernel = np.array([[0, 0, 1, 0, 0],
                            [0, 0, 2, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 2, 0, 0],
                            [0, 0, 1, 0, 0]], dtype=np.float32) / 4.0
        
        d_kernel = np.array([[1, 0, 0, 0, 1],
                            [0, 2, 0, 2, 0],
                            [0, 0, 0, 0, 0],
                            [0, 2, 0, 2, 0],
                            [1, 0, 0, 0, 1]], dtype=np.float32) / 8.0
        
        # Red interpolation
        R = raw * mask_r
        R_minus_G = (raw - G) * mask_r
        
        R_h = convolve(R_minus_G, h_kernel, mode='reflect')
        R_v = convolve(R_minus_G, v_kernel, mode='reflect')
        R_d = convolve(R_minus_G, d_kernel, mode='reflect')
        
        at_g = mask_g.astype(bool)
        has_R_h = convolve(mask_r, h_kernel * 2, mode='reflect') > 0.3
        R_at_g = np.where(has_R_h, G + R_h, G + R_v)
        R = np.where(at_g, R_at_g, R)
        
        at_b = mask_b.astype(bool)
        R = np.where(at_b, G + R_d, R)
        
        # Blue interpolation (symmetric)
        B = raw * mask_b
        B_minus_G = (raw - G) * mask_b
        
        B_h = convolve(B_minus_G, h_kernel, mode='reflect')
        B_v = convolve(B_minus_G, v_kernel, mode='reflect')
        B_d = convolve(B_minus_G, d_kernel, mode='reflect')
        
        has_B_h = convolve(mask_b, h_kernel * 2, mode='reflect') > 0.3
        B_at_g = np.where(has_B_h, G + B_h, G + B_v)
        B = np.where(at_g, B_at_g, B)
        
        at_r = mask_r.astype(bool)
        B = np.where(at_r, G + B_d, B)
        
        return R, B
