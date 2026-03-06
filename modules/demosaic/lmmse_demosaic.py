"""
File: lmmse_demosaic.py
Description: Implements the LMMSE (Linear Minimum Mean Square Error) demosaic algorithm
Reference: "Color Plane Interpolation Using Alternating Projections" 
           by Gunturk, Altunbasak, and Mersereau (2002)
Author: Brian Deegan (via AI)
------------------------------------------------------------
"""
import numpy as np
from scipy.ndimage import convolve, correlate


class LMMSEDemosaic:
    """
    LMMSE (Linear Minimum Mean Square Error) demosaic algorithm.
    
    LMMSE uses statistical estimation theory to minimize reconstruction error:
    1. Estimates local statistics (mean, variance, covariance)
    2. Computes optimal linear filter coefficients
    3. Applies adaptive filtering based on local image characteristics
    4. Uses color correlation for improved estimates
    
    This produces excellent quality with optimal MSE properties.
    """

    def __init__(self, raw_in, masks):
        self.img = np.asarray(raw_in, dtype=np.float32)
        self.masks = [m.astype(np.float32) for m in masks]
        self.height, self.width = self.img.shape
        
        # LMMSE parameters
        self.window_size = 5  # Local estimation window
        self.epsilon = 1e-6   # Small constant for numerical stability
        
    def _estimate_local_mean(self, channel, mask):
        """
        Estimate local mean of a channel using only known values.
        """
        # Use averaging kernel
        kernel = np.ones((self.window_size, self.window_size), dtype=np.float32)
        kernel = kernel / kernel.sum()
        
        # Compute sum and count of valid pixels
        sum_vals = convolve(channel * mask, kernel, mode='reflect')
        count_vals = convolve(mask, kernel, mode='reflect')
        
        # Compute mean (avoid division by zero)
        mean = sum_vals / (count_vals + self.epsilon)
        return mean
    
    def _estimate_local_variance(self, channel, mask, mean):
        """
        Estimate local variance using known values.
        """
        # Compute squared differences from mean
        diff_sq = ((channel - mean) ** 2) * mask
        
        # Average over local window
        kernel = np.ones((self.window_size, self.window_size), dtype=np.float32)
        kernel = kernel / kernel.sum()
        
        sum_diff_sq = convolve(diff_sq, kernel, mode='reflect')
        count_vals = convolve(mask, kernel, mode='reflect')
        
        variance = sum_diff_sq / (count_vals + self.epsilon)
        return variance
    
    def _estimate_covariance(self, channel1, mask1, channel2, mask2, mean1, mean2):
        """
        Estimate covariance between two channels.
        """
        # Only use locations where both channels have values
        both_mask = mask1 * mask2
        
        # Compute covariance
        diff1 = (channel1 - mean1) * both_mask
        diff2 = (channel2 - mean2) * both_mask
        cov_product = diff1 * diff2
        
        # Average over local window
        kernel = np.ones((self.window_size, self.window_size), dtype=np.float32)
        kernel = kernel / kernel.sum()
        
        sum_cov = convolve(cov_product, kernel, mode='reflect')
        count_vals = convolve(both_mask, kernel, mode='reflect')
        
        covariance = sum_cov / (count_vals + self.epsilon)
        return covariance
    
    def _lmmse_interpolate(self, target_channel, target_mask, ref_channel, ref_mask):
        """
        LMMSE interpolation of target channel using reference channel.
        
        Formula: X_est = mean_X + cov(X,Y)/var(Y) * (Y - mean_Y)
        
        This is the optimal linear estimator that minimizes MSE.
        """
        # Estimate local statistics for both channels
        mean_target = self._estimate_local_mean(target_channel, target_mask)
        mean_ref = self._estimate_local_mean(ref_channel, ref_mask)
        
        var_ref = self._estimate_local_variance(ref_channel, ref_mask, mean_ref)
        cov = self._estimate_covariance(target_channel, target_mask, 
                                       ref_channel, ref_mask, 
                                       mean_target, mean_ref)
        
        # Compute LMMSE gain (Wiener filter coefficient)
        gain = cov / (var_ref + self.epsilon)
        
        # Apply LMMSE estimation
        prediction = mean_target + gain * (ref_channel - mean_ref)
        
        return prediction
    
    def _interpolate_green_at_red_blue(self):
        """
        Interpolate green at red and blue locations using LMMSE.
        Uses directional interpolation combined with LMMSE refinement.
        """
        raw = self.img
        mask_r, mask_g, mask_b = self.masks
        
        # Start with known green values
        G = raw * mask_g
        
        # Initial directional interpolation for green
        h_kernel = np.array([[0, 0, 0],
                            [1, 0, 1],
                            [0, 0, 0]], dtype=np.float32) / 2.0
        
        v_kernel = np.array([[0, 1, 0],
                            [0, 0, 0],
                            [0, 1, 0]], dtype=np.float32) / 2.0
        
        # Gradient computation
        h_grad = np.abs(convolve(raw, np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=np.float32), mode='reflect'))
        v_grad = np.abs(convolve(raw, np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]], dtype=np.float32), mode='reflect'))
        
        # Initial estimate
        G_h = convolve(raw * mask_g, h_kernel, mode='reflect')
        G_v = convolve(raw * mask_g, v_kernel, mode='reflect')
        G_initial = np.where(h_grad < v_grad, G_h, G_v)
        
        # Combine with known values
        G_combined = G + (1.0 - mask_g) * G_initial
        
        # LMMSE refinement at R locations using color correlation
        R_channel = raw * mask_r
        G_at_r = self._lmmse_interpolate(G_combined, mask_g, R_channel, mask_r)
        
        # LMMSE refinement at B locations
        B_channel = raw * mask_b
        G_at_b = self._lmmse_interpolate(G_combined, mask_g, B_channel, mask_b)
        
        # Update G with refined estimates
        at_r = mask_r.astype(bool)
        at_b = mask_b.astype(bool)
        G[at_r] = G_at_r[at_r]
        G[at_b] = G_at_b[at_b]
        
        return G
    
    def _interpolate_red_blue(self, G):
        """
        Interpolate R and B using LMMSE with green as reference.
        """
        raw = self.img
        mask_r, mask_g, mask_b = self.masks
        
        # Extract known R and B values
        R = raw * mask_r
        B = raw * mask_b
        
        # Create full green mask (green is known everywhere after first step)
        G_mask = np.ones_like(G)
        
        # LMMSE interpolation for R using G as reference
        R_est = self._lmmse_interpolate(R, mask_r, G, G_mask)
        
        # LMMSE interpolation for B using G as reference
        B_est = self._lmmse_interpolate(B, mask_b, G, G_mask)
        
        # Combine known values with estimates
        R = R + (1.0 - mask_r) * R_est
        B = B + (1.0 - mask_b) * B_est
        
        return R, B
    
    def apply_lmmse(self):
        """
        Apply LMMSE demosaicing algorithm.
        
        Process:
        1. Initial directional interpolation for green
        2. LMMSE refinement of green using R/B correlation
        3. LMMSE interpolation of R/B using green as reference
        """
        # Step 1 & 2: Interpolate and refine green
        G = self._interpolate_green_at_red_blue()
        
        # Step 3: Interpolate R and B
        R, B = self._interpolate_red_blue(G)
        
        # Stack channels
        out = np.stack([R, G, B], axis=-1)
        return out.astype(np.float32)


class LMMSEDemosaicOptimized:
    """
    Optimized LMMSE implementation with enhanced statistical estimation.
    Uses larger windows and more sophisticated correlation modeling.
    """
    
    def __init__(self, raw_in, masks):
        self.img = np.asarray(raw_in, dtype=np.float32)
        self.masks = [m.astype(np.float32) for m in masks]
        
        # Enhanced parameters
        self.window_size = 7  # Larger window for better statistics
        self.epsilon = 1e-6
        
    def _compute_local_statistics(self, channel, mask):
        """
        Compute comprehensive local statistics efficiently.
        """
        kernel = np.ones((self.window_size, self.window_size), dtype=np.float32)
        kernel = kernel / kernel.sum()
        
        # Mean
        sum_vals = convolve(channel * mask, kernel, mode='reflect')
        count_vals = convolve(mask, kernel, mode='reflect')
        mean = sum_vals / (count_vals + self.epsilon)
        
        # Variance
        diff_sq = ((channel - mean) ** 2) * mask
        sum_diff_sq = convolve(diff_sq, kernel, mode='reflect')
        variance = sum_diff_sq / (count_vals + self.epsilon)
        
        return mean, variance
    
    def _compute_cross_correlation(self, ch1, mask1, ch2, mask2, mean1, mean2):
        """
        Compute cross-correlation between two channels.
        """
        both_mask = mask1 * mask2
        
        diff1 = (ch1 - mean1) * both_mask
        diff2 = (ch2 - mean2) * both_mask
        product = diff1 * diff2
        
        kernel = np.ones((self.window_size, self.window_size), dtype=np.float32)
        kernel = kernel / kernel.sum()
        
        sum_product = convolve(product, kernel, mode='reflect')
        count_vals = convolve(both_mask, kernel, mode='reflect')
        
        correlation = sum_product / (count_vals + self.epsilon)
        return correlation
    
    def _wiener_filter(self, target, target_mask, reference, ref_mask):
        """
        Apply Wiener filter for optimal LMMSE estimation.
        
        This is the optimal linear filter in MSE sense.
        """
        # Compute statistics
        mean_target, var_target = self._compute_local_statistics(target, target_mask)
        mean_ref, var_ref = self._compute_local_statistics(reference, ref_mask)
        
        # Compute cross-correlation
        cross_corr = self._compute_cross_correlation(target, target_mask, 
                                                     reference, ref_mask,
                                                     mean_target, mean_ref)
        
        # Wiener gain
        gain = cross_corr / (var_ref + self.epsilon)
        
        # Wiener filtering
        estimate = mean_target + gain * (reference - mean_ref)
        
        # Confidence weighting based on variance
        # Higher variance = less confident in estimate
        confidence = 1.0 / (1.0 + var_target / (var_ref + self.epsilon))
        
        return estimate, confidence
    
    def _interpolate_green_adaptive(self):
        """
        Adaptive green interpolation using LMMSE with confidence weighting.
        """
        raw = self.img
        mask_r, mask_g, mask_b = self.masks
        
        G = raw * mask_g
        
        # Enhanced directional gradients (5x5)
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
        
        grad_h = np.abs(convolve(raw, h_grad, mode='reflect'))
        grad_v = np.abs(convolve(raw, v_grad, mode='reflect'))
        
        # Initial interpolation
        h_kernel = np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]], dtype=np.float32) / 2.0
        v_kernel = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]], dtype=np.float32) / 2.0
        
        G_h = convolve(raw * mask_g, h_kernel, mode='reflect')
        G_v = convolve(raw * mask_g, v_kernel, mode='reflect')
        G_dir = np.where(grad_h < grad_v, G_h, G_v)
        
        # Average if gradients similar
        grad_ratio = grad_h / (grad_v + self.epsilon)
        similar = (grad_ratio > 0.7) & (grad_ratio < 1.43)
        G_dir = np.where(similar, (G_h + G_v) / 2.0, G_dir)
        
        G_temp = G + (1.0 - mask_g) * G_dir
        
        # LMMSE refinement with Wiener filtering
        R_channel = raw * mask_r
        G_at_r, conf_r = self._wiener_filter(G_temp, mask_g, R_channel, mask_r)
        
        B_channel = raw * mask_b
        G_at_b, conf_b = self._wiener_filter(G_temp, mask_g, B_channel, mask_b)
        
        # Apply with confidence weighting
        at_r = mask_r.astype(bool)
        at_b = mask_b.astype(bool)
        
        G[at_r] = conf_r[at_r] * G_at_r[at_r] + (1 - conf_r[at_r]) * G_dir[at_r]
        G[at_b] = conf_b[at_b] * G_at_b[at_b] + (1 - conf_b[at_b]) * G_dir[at_b]
        
        return G
    
    def apply_lmmse_optimized(self):
        """
        Apply optimized LMMSE with enhanced statistical modeling.
        """
        raw = self.img
        mask_r, mask_g, mask_b = self.masks
        
        # Step 1: Adaptive green interpolation
        G = self._interpolate_green_adaptive()
        
        # Step 2: LMMSE for R and B using G as reference
        R = raw * mask_r
        B = raw * mask_b
        
        G_mask = np.ones_like(G)
        
        # Wiener filtering for R
        R_est, conf_r = self._wiener_filter(R, mask_r, G, G_mask)
        R = R + (1.0 - mask_r) * R_est
        
        # Wiener filtering for B
        B_est, conf_b = self._wiener_filter(B, mask_b, G, G_mask)
        B = B + (1.0 - mask_b) * B_est
        
        # Stack channels
        out = np.stack([R, G, B], axis=-1)
        return out.astype(np.float32)


class LMMSEDemosaicFast:
    """
    Fast LMMSE implementation using simplified statistical estimation.
    Trades some quality for speed while maintaining LMMSE principles.
    """
    
    def __init__(self, raw_in, masks):
        self.img = np.asarray(raw_in, dtype=np.float32)
        self.masks = [m.astype(np.float32) for m in masks]
        self.window_size = 3  # Smaller window for speed
        self.epsilon = 1e-6
    
    def _fast_local_mean_var(self, channel, mask):
        """Fast local statistics using box filter."""
        kernel = np.ones((self.window_size, self.window_size), dtype=np.float32)
        
        sum_vals = convolve(channel * mask, kernel, mode='reflect')
        count_vals = convolve(mask, kernel, mode='reflect')
        mean = sum_vals / (count_vals + self.epsilon)
        
        sum_sq = convolve((channel ** 2) * mask, kernel, mode='reflect')
        mean_sq = sum_sq / (count_vals + self.epsilon)
        variance = mean_sq - mean ** 2
        variance = np.maximum(variance, 0)  # Ensure non-negative
        
        return mean, variance
    
    def _fast_lmmse(self, target, target_mask, ref, ref_mask):
        """Fast LMMSE using simplified covariance estimation."""
        mean_t, var_t = self._fast_local_mean_var(target, target_mask)
        mean_r, var_r = self._fast_local_mean_var(ref, ref_mask)
        
        # Simplified correlation estimate
        both_mask = target_mask * ref_mask
        product = target * ref * both_mask
        
        kernel = np.ones((self.window_size, self.window_size), dtype=np.float32)
        sum_prod = convolve(product, kernel, mode='reflect')
        count = convolve(both_mask, kernel, mode='reflect')
        
        mean_prod = sum_prod / (count + self.epsilon)
        cov = mean_prod - mean_t * mean_r
        
        # LMMSE estimation
        gain = cov / (var_r + self.epsilon)
        estimate = mean_t + gain * (ref - mean_r)
        
        return estimate
    
    def apply_lmmse_fast(self):
        """Fast LMMSE demosaicing."""
        raw = self.img
        mask_r, mask_g, mask_b = self.masks
        
        # Quick directional interpolation for green
        G = raw * mask_g
        
        h_k = np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]], dtype=np.float32) / 2.0
        v_k = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]], dtype=np.float32) / 2.0
        
        h_g = np.abs(convolve(raw, np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=np.float32), mode='reflect'))
        v_g = np.abs(convolve(raw, np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]], dtype=np.float32), mode='reflect'))
        
        G_h = convolve(raw * mask_g, h_k, mode='reflect')
        G_v = convolve(raw * mask_g, v_k, mode='reflect')
        G_init = np.where(h_g < v_g, G_h, G_v)
        
        G = G + (1.0 - mask_g) * G_init
        
        # Fast LMMSE for R and B
        R = raw * mask_r
        B = raw * mask_b
        
        G_mask = np.ones_like(G)
        
        R_est = self._fast_lmmse(R, mask_r, G, G_mask)
        B_est = self._fast_lmmse(B, mask_b, G, G_mask)
        
        R = R + (1.0 - mask_r) * R_est
        B = B + (1.0 - mask_b) * B_est
        
        out = np.stack([R, G, B], axis=-1)
        return out.astype(np.float32)
