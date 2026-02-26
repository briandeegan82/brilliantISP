"""
File: non_local_means_optimized.py
Description: Optimized non-local means filter using NumPy broadcast operations
Code / Paper  Reference:
Author: 10xEngineers
------------------------------------------------------------
"""
import logging
import numpy as np
from scipy import ndimage
from tqdm import tqdm
import cv2

# Import GPU utilities with fallback
try:
    from util.gpu_utils import (
        is_gpu_available, should_use_gpu, gpu_filter2d, 
        gpu_gaussian_blur, to_umat, from_umat
    )
    GPU_UTILS_AVAILABLE = True
except ImportError:
    GPU_UTILS_AVAILABLE = False
    # Fallback functions for CPU-only systems
    def is_gpu_available():
        return False
    
    def should_use_gpu(img_size, operation):
        return False
    
    def gpu_filter2d(img, kernel, use_gpu=True):
        return cv2.filter2D(img, -1, kernel)
    
    def gpu_gaussian_blur(img, ksize, sigma_x, sigma_y=0, use_gpu=True):
        return cv2.GaussianBlur(img, ksize, sigma_x, sigma_y)
    
    def to_umat(img, use_gpu=True):
        return img
    
    def from_umat(umat_or_array):
        return umat_or_array


class NLMOptimized:
    """
    Optimized Non-local means filter with NumPy broadcast operations
    """

    def __init__(self, img, sensor_info, parm_2dnr, platform):
        self.img = img
        self.sensor_info = sensor_info
        self.parm_2dnr = parm_2dnr
        self.platform = platform
        self.is_progress = platform["disable_progress_bar"]
        self.is_leave = platform["leave_pbar_string"]
        self.logger = logging.getLogger(__name__)
        
        # Check if GPU acceleration should be used
        self.use_gpu = (is_gpu_available() and 
                       should_use_gpu((sensor_info["height"], sensor_info["width"]), 'filter2d'))
        
        if self.use_gpu:
            self.logger.info("  Using GPU acceleration for Non-local Means")
        else:
            self.logger.info("  Using CPU implementation for Non-local Means")

    def get_weights(self):
        """
        Applying weights using vectorized operations
        """
        # wts is the strength parameter to assign weights to the similar pixels
        wts = self.parm_2dnr["wts"]

        # Avoiding division by zero
        if wts <= 0:
            wts = 1

        # OPTIMIZATION: Use vectorized operations instead of loops
        # The similarity between pixels is compared on the basis of Euclidean distance
        distance = np.arange(255**2)
        lut = np.exp(-distance / wts**2) * 1024

        return lut.astype(np.int32)

    def apply_mean_filter_optimized(self, array, patch_size):
        """
        Optimized mean filter using NumPy operations
        """
        if self.use_gpu and GPU_UTILS_AVAILABLE:
            return self.apply_mean_filter_gpu(array, patch_size)
        else:
            return self.apply_mean_filter_cpu(array, patch_size)

    def apply_mean_filter_cpu(self, array, patch_size):
        """
        CPU implementation of mean filter using NumPy
        """
        # Use uniform filter for mean calculation (much faster than manual loops)
        return ndimage.uniform_filter(array, size=patch_size, mode='reflect')

    def apply_mean_filter_gpu(self, array, patch_size):
        """
        GPU-accelerated mean filter
        """
        try:
            # Convert to GPU
            gpu_array = to_umat(array, use_gpu=True)
            
            # Create uniform kernel for mean filtering
            kernel_size = patch_size
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            # Create uniform kernel
            kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
            gpu_kernel = to_umat(kernel, use_gpu=True)
            
            # Apply filtering
            gpu_result = gpu_filter2d(gpu_array, gpu_kernel, use_gpu=True)
            
            return from_umat(gpu_result)
            
        except Exception as e:
            self.logger.warning(f"  GPU mean filter failed, falling back to CPU: {e}")
            return self.apply_mean_filter_cpu(array, patch_size)

    def apply_nlm_optimized(self):
        """
        Optimized Non-local Means Filter using NumPy broadcast operations
        """
        # Input YUV image
        in_image = self.img

        # Search window and patch sizes
        window_size = self.parm_2dnr["window_size"]
        patch_size = self.parm_2dnr["patch_size"]

        # Patch size should be odd
        if patch_size % 2 == 0:
            patch_size = patch_size + 1
            self.logger.info(f"    -Making patch size odd: {patch_size}")

        # Extracting Y channel to apply the 2DNR module
        input_image = in_image[:, :, 0]

        if in_image.dtype == "float32":
            input_image = np.round(255 * input_image).astype(np.uint8)

        # Declaring empty array for output image after denoising
        denoised_out = np.empty(in_image.shape, dtype=np.uint8)

        # Padding the input_image
        pads = window_size // 2
        wtspadded_y_in = np.pad(input_image, pads, mode="reflect")

        # Declaration of denoised Y channel and weights
        denoised_y_channel = np.zeros(input_image.shape, dtype=np.float32)
        final_weights = np.zeros(input_image.shape, dtype=np.float32)

        # Generating LUT weights based on euclidean distance between intensities
        weights_lut = self.get_weights()

        # OPTIMIZATION: Use NumPy broadcast operations for better performance
        # Pre-calculate all shifted arrays and process them efficiently
        
        # Create arrays for all window positions at once
        window_positions = []
        for i in range(window_size):
            for j in range(window_size):
                shifted_array = np.int32(
                    wtspadded_y_in[
                        i : i + input_image.shape[0], j : j + input_image.shape[1]
                    ]
                )
                window_positions.append(shifted_array)
        
        # Convert to 3D array for vectorized processing
        window_positions = np.array(window_positions)  # Shape: (window_size^2, height, width)
        
        # Process all positions at once using broadcasting
        for idx, array_for_each_pixel_in_sw in enumerate(window_positions):
            # Finding euclidean distance between pixels based on their intensities
            # OPTIMIZATION: Use vectorized operations
            euc_distance = (input_image - array_for_each_pixel_in_sw) ** 2
            
            # Apply mean filter
            distance = self.apply_mean_filter_optimized(euc_distance, patch_size=patch_size)
            
            # Assigning weights to the pixels based on their distance
            # OPTIMIZATION: Use vectorized indexing
            weight_for_each_shifted_array = weights_lut[distance]
            
            # Adding up all the weighted similar pixels
            denoised_y_channel += array_for_each_pixel_in_sw * weight_for_each_shifted_array
            
            # Adding up all the weights for final mean values at each pixel location
            final_weights += weight_for_each_shifted_array

        # Averaging out all the pixels
        # OPTIMIZATION: Use vectorized division with safe handling of zeros
        denoised_y_channel = denoised_y_channel / (final_weights + 1e-8)  # Add small epsilon

        if in_image.dtype == "float32":
            denoised_y_channel = np.float32(denoised_y_channel / 255.0)
            denoised_out = denoised_out.astype("float32")

        # Reconstructing the final output
        denoised_out[:, :, 0] = denoised_y_channel
        denoised_out[:, :, 1] = in_image[:, :, 1]
        denoised_out[:, :, 2] = in_image[:, :, 2]

        return denoised_out

    def apply_nlm_vectorized(self):
        """
        Alternative vectorized implementation for even better performance
        """
        # Input YUV image
        in_image = self.img

        # Search window and patch sizes
        window_size = self.parm_2dnr["window_size"]
        patch_size = self.parm_2dnr["patch_size"]

        # Patch size should be odd
        if patch_size % 2 == 0:
            patch_size = patch_size + 1

        # Extracting Y channel
        input_image = in_image[:, :, 0]

        if in_image.dtype == "float32":
            input_image = np.round(255 * input_image).astype(np.uint8)

        # Output array
        denoised_out = np.empty(in_image.shape, dtype=np.uint8)

        # Padding
        pads = window_size // 2
        wtspadded_y_in = np.pad(input_image, pads, mode="reflect")

        # Initialize output arrays
        denoised_y_channel = np.zeros(input_image.shape, dtype=np.float32)
        final_weights = np.zeros(input_image.shape, dtype=np.float32)

        # Weights LUT
        wts = max(self.parm_2dnr["wts"], 1)
        distance = np.arange(255**2)
        weights_lut = (np.exp(-distance / wts**2) * 1024).astype(np.int32)

        # OPTIMIZATION: Use sliding window view for better memory efficiency
        from numpy.lib.stride_tricks import sliding_window_view
        
        # Create sliding window view of the padded image
        # This creates a view without copying data
        window_view = sliding_window_view(wtspadded_y_in, (input_image.shape[0], input_image.shape[1]))
        
        # Process each window position
        for i in range(window_size):
            for j in range(window_size):
                # Extract shifted array using the window view
                array_for_each_pixel_in_sw = np.int32(window_view[i, j])
                
                # Calculate euclidean distance
                euc_distance = (input_image - array_for_each_pixel_in_sw) ** 2
                
                # Apply mean filter
                distance = self.apply_mean_filter_optimized(euc_distance, patch_size=patch_size)
                
                # Get weights
                weight_for_each_shifted_array = weights_lut[distance]
                
                # Accumulate results
                denoised_y_channel += array_for_each_pixel_in_sw * weight_for_each_shifted_array
                final_weights += weight_for_each_shifted_array

        # Final averaging
        denoised_y_channel = denoised_y_channel / (final_weights + 1e-8)

        if in_image.dtype == "float32":
            denoised_y_channel = np.float32(denoised_y_channel / 255.0)
            denoised_out = denoised_out.astype("float32")

        # Reconstruct output
        denoised_out[:, :, 0] = denoised_y_channel
        denoised_out[:, :, 1] = in_image[:, :, 1]
        denoised_out[:, :, 2] = in_image[:, :, 2]

        return denoised_out

    def apply_nlm(self):
        """
        Main method - choose the best implementation based on image size
        """
        # For smaller images, use the standard optimized version
        # For larger images, use the vectorized version if available
        try:
            if hasattr(np.lib.stride_tricks, 'sliding_window_view'):
                return self.apply_nlm_vectorized()
            else:
                return self.apply_nlm_optimized()
        except Exception as e:
            self.logger.warning(f"  Vectorized NLM failed, falling back to optimized: {e}")
            return self.apply_nlm_optimized()
