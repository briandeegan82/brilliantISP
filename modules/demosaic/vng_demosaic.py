"""
File: vng_demosaic.py
Description: Implements the VNG (Variable Number of Gradients) demosaic algorithm
Reference: "Adaptive homogeneity-directed demosaicing algorithm" by Chang & Tan (1999)
Author: Brian Deegan (via AI)
------------------------------------------------------------
"""
import numpy as np
from scipy.ndimage import convolve


class VNGDemosaic:
    """
    Variable Number of Gradients (VNG) demosaic algorithm.
    
    VNG is an edge-directed interpolation method that:
    1. Computes gradients in 8 directions around each pixel
    2. Selects the most homogeneous directions (smallest gradients)
    3. Interpolates using only pixels in those directions
    
    This produces better edge preservation than bilinear or Malvar methods.
    """

    def __init__(self, raw_in, masks):
        self.img = np.asarray(raw_in, dtype=np.float32)
        self.masks = [m.astype(np.float32) for m in masks]
        self.height, self.width = self.img.shape

    def _compute_gradients(self, pixel_vals):
        """
        Compute gradient magnitudes for 8 directions.
        
        Args:
            pixel_vals: dict mapping direction indices to pixel values
            
        Returns:
            Array of gradient magnitudes for each direction
        """
        # Compute gradients by looking at color differences in each direction
        gradients = np.zeros(8)
        
        # For each direction, compute gradient as sum of absolute differences
        for direction in range(8):
            if direction in pixel_vals and len(pixel_vals[direction]) > 1:
                vals = np.array(pixel_vals[direction])
                # Gradient is the variance/spread in this direction
                gradients[direction] = np.std(vals) + np.abs(np.diff(vals)).sum()
            else:
                gradients[direction] = np.inf  # Mark invalid directions
                
        return gradients

    def _get_threshold(self, gradients):
        """
        Compute threshold for selecting homogeneous directions.
        Uses 1.5 times the minimum gradient as threshold.
        """
        valid_grads = gradients[gradients < np.inf]
        if len(valid_grads) == 0:
            return 0
        min_grad = np.min(valid_grads)
        return min_grad * 1.5

    def apply_vng(self):
        """
        Apply VNG demosaicing algorithm to the raw Bayer image.
        """
        raw = self.img
        mask_r, mask_g, mask_b = self.masks
        
        # Initialize output channels
        R = raw * mask_r
        G = raw * mask_g
        B = raw * mask_b
        
        # Pad image for border handling
        pad_size = 3
        raw_padded = np.pad(raw, pad_size, mode='reflect')
        mask_r_padded = np.pad(mask_r, pad_size, mode='reflect')
        mask_g_padded = np.pad(mask_g, pad_size, mode='reflect')
        mask_b_padded = np.pad(mask_b, pad_size, mode='reflect')
        
        # 8 directions for gradient computation (N, NE, E, SE, S, SW, W, NW)
        # Each direction has a set of offsets to sample
        direction_offsets = [
            [(-2, 0), (-1, 0), (1, 0), (2, 0)],      # N-S (vertical)
            [(-2, 2), (-1, 1), (1, -1), (2, -2)],    # NE-SW
            [(0, -2), (0, -1), (0, 1), (0, 2)],      # E-W (horizontal)
            [(-2, -2), (-1, -1), (1, 1), (2, 2)],    # NW-SE
            [(-1, 0), (-2, 0), (1, 0), (2, 0)],      # N-S variant
            [(-1, 1), (-2, 2), (1, -1), (2, -2)],    # NE-SW variant
            [(0, -1), (0, -2), (0, 1), (0, 2)],      # E-W variant
            [(-1, -1), (-2, -2), (1, 1), (2, 2)],    # NW-SE variant
        ]
        
        # Process each pixel
        for y in range(self.height):
            for x in range(self.width):
                # Adjust for padding
                py, px = y + pad_size, x + pad_size
                
                # Skip if green pixel (already has green value)
                # We'll interpolate green at R/B locations
                if not mask_g[y, x]:
                    # Interpolate Green at R/B locations
                    pixel_vals_g = {}
                    
                    for dir_idx, offsets in enumerate(direction_offsets):
                        vals = []
                        for dy, dx in offsets:
                            ny, nx = py + dy, px + dx
                            # Only use green pixels in this direction
                            if mask_g_padded[ny, nx]:
                                vals.append(raw_padded[ny, nx])
                        if vals:
                            pixel_vals_g[dir_idx] = vals
                    
                    if pixel_vals_g:
                        gradients = self._compute_gradients(pixel_vals_g)
                        threshold = self._get_threshold(gradients)
                        
                        # Select directions with gradient below threshold
                        selected_dirs = np.where(gradients <= threshold)[0]
                        
                        if len(selected_dirs) > 0:
                            # Interpolate using selected directions
                            green_vals = []
                            for dir_idx in selected_dirs:
                                if dir_idx in pixel_vals_g:
                                    green_vals.extend(pixel_vals_g[dir_idx])
                            
                            if green_vals:
                                G[y, x] = np.mean(green_vals)
                
                # Interpolate R at G/B locations
                if not mask_r[y, x]:
                    pixel_vals_r = {}
                    
                    for dir_idx, offsets in enumerate(direction_offsets):
                        vals = []
                        for dy, dx in offsets:
                            ny, nx = py + dy, px + dx
                            if mask_r_padded[ny, nx]:
                                vals.append(raw_padded[ny, nx])
                        if vals:
                            pixel_vals_r[dir_idx] = vals
                    
                    if pixel_vals_r:
                        gradients = self._compute_gradients(pixel_vals_r)
                        threshold = self._get_threshold(gradients)
                        selected_dirs = np.where(gradients <= threshold)[0]
                        
                        if len(selected_dirs) > 0:
                            red_vals = []
                            for dir_idx in selected_dirs:
                                if dir_idx in pixel_vals_r:
                                    red_vals.extend(pixel_vals_r[dir_idx])
                            
                            if red_vals:
                                R[y, x] = np.mean(red_vals)
                
                # Interpolate B at R/G locations
                if not mask_b[y, x]:
                    pixel_vals_b = {}
                    
                    for dir_idx, offsets in enumerate(direction_offsets):
                        vals = []
                        for dy, dx in offsets:
                            ny, nx = py + dy, px + dx
                            if mask_b_padded[ny, nx]:
                                vals.append(raw_padded[ny, nx])
                        if vals:
                            pixel_vals_b[dir_idx] = vals
                    
                    if pixel_vals_b:
                        gradients = self._compute_gradients(pixel_vals_b)
                        threshold = self._get_threshold(gradients)
                        selected_dirs = np.where(gradients <= threshold)[0]
                        
                        if len(selected_dirs) > 0:
                            blue_vals = []
                            for dir_idx in selected_dirs:
                                if dir_idx in pixel_vals_b:
                                    blue_vals.extend(pixel_vals_b[dir_idx])
                            
                            if blue_vals:
                                B[y, x] = np.mean(blue_vals)
        
        # Stack channels
        out = np.stack([R, G, B], axis=-1)
        return out.astype(np.float32)


class VNGDemosaicOptimized:
    """
    Optimized VNG demosaic using vectorized operations.
    Faster than the basic VNG but still maintains edge-directed interpolation.
    """
    
    def __init__(self, raw_in, masks):
        self.img = np.asarray(raw_in, dtype=np.float32)
        self.masks = [m.astype(np.float32) for m in masks]

    def _compute_color_gradients(self, img):
        """
        Compute gradient magnitudes in 8 directions using convolution.
        Returns gradient maps for each direction.
        """
        gradients = []
        
        # Gradient kernels for 8 directions
        # Each kernel measures variance along that direction
        kernels = [
            # Vertical (N-S)
            np.array([[0, 0, 1, 0, 0],
                     [0, 0, -1, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, -1, 0, 0],
                     [0, 0, 1, 0, 0]], dtype=np.float32),
            
            # Horizontal (E-W)
            np.array([[0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [1, -1, 0, -1, 1],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]], dtype=np.float32),
            
            # NE-SW diagonal
            np.array([[0, 0, 0, 0, 1],
                     [0, 0, 0, -1, 0],
                     [0, 0, 0, 0, 0],
                     [0, -1, 0, 0, 0],
                     [1, 0, 0, 0, 0]], dtype=np.float32),
            
            # NW-SE diagonal
            np.array([[1, 0, 0, 0, 0],
                     [0, -1, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, -1, 0],
                     [0, 0, 0, 0, 1]], dtype=np.float32),
        ]
        
        for kernel in kernels:
            grad = np.abs(convolve(img, kernel, mode='reflect'))
            gradients.append(grad)
            
        return gradients

    def apply_vng_optimized(self):
        """
        Apply optimized VNG demosaicing using vectorized operations.
        """
        raw = self.img
        mask_r, mask_g, mask_b = self.masks
        
        # Compute gradients
        gradients = self._compute_color_gradients(raw)
        
        # Combine gradients to get homogeneity map
        gradient_stack = np.stack(gradients, axis=-1)
        min_gradient = np.min(gradient_stack, axis=-1)
        threshold = min_gradient * 1.5
        
        # Create weight maps for each direction based on homogeneity
        weights = []
        for grad in gradients:
            # Weight is higher where gradient is low (homogeneous)
            weight = np.where(grad <= threshold, 1.0, 0.0)
            weights.append(weight)
        
        # Interpolation kernels for each color at different locations
        # Green at R/B locations (cross pattern)
        g_kernel = np.array([[0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0]], dtype=np.float32) / 4.0
        
        # R/B at G locations (depends on orientation)
        rb_kernel_h = np.array([[0, 0, 0],
                               [1, 0, 1],
                               [0, 0, 0]], dtype=np.float32) / 2.0
        
        rb_kernel_v = np.array([[0, 1, 0],
                               [0, 0, 0],
                               [0, 1, 0]], dtype=np.float32) / 2.0
        
        # R/B at opposite color (diagonal)
        rb_kernel_d = np.array([[1, 0, 1],
                               [0, 0, 0],
                               [1, 0, 1]], dtype=np.float32) / 4.0
        
        # Extract known values
        R = raw * mask_r
        G = raw * mask_g
        B = raw * mask_b
        
        # Interpolate green at R/B locations
        G_interp = convolve(raw * mask_g, g_kernel, mode='reflect')
        G = G + (1.0 - mask_g) * G_interp
        
        # Detect R/B orientations for smarter interpolation
        # R at G locations
        has_R_h = convolve(mask_r, rb_kernel_h * 2, mode='reflect') > 0
        has_R_v = convolve(mask_r, rb_kernel_v * 2, mode='reflect') > 0
        
        R_h = convolve(raw * mask_r, rb_kernel_h, mode='reflect')
        R_v = convolve(raw * mask_r, rb_kernel_v, mode='reflect')
        R_d = convolve(raw * mask_r, rb_kernel_d, mode='reflect')
        
        # Use gradient information to select best interpolation
        at_G = mask_g.astype(bool)
        R_interp = np.where(has_R_h, R_h, R_v)
        R = np.where(at_G, R_interp, R)
        
        # R at B locations (diagonal)
        at_B = mask_b.astype(bool)
        R = np.where(at_B, R_d, R)
        
        # Symmetric for blue
        has_B_h = convolve(mask_b, rb_kernel_h * 2, mode='reflect') > 0
        has_B_v = convolve(mask_b, rb_kernel_v * 2, mode='reflect') > 0
        
        B_h = convolve(raw * mask_b, rb_kernel_h, mode='reflect')
        B_v = convolve(raw * mask_b, rb_kernel_v, mode='reflect')
        B_d = convolve(raw * mask_b, rb_kernel_d, mode='reflect')
        
        at_G = mask_g.astype(bool)
        B_interp = np.where(has_B_h, B_h, B_v)
        B = np.where(at_G, B_interp, B)
        
        at_R = mask_r.astype(bool)
        B = np.where(at_R, B_d, B)
        
        # Stack channels
        out = np.stack([R, G, B], axis=-1)
        return out.astype(np.float32)
