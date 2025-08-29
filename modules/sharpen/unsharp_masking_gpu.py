"""
File: unsharp_masking_gpu.py
Description: GPU-accelerated unsharp masking with frequency and strength control.
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
"""
import numpy as np
from scipy import ndimage
import cv2

# Import GPU utilities with fallback
try:
    from util.gpu_utils import (
        is_gpu_available, should_use_gpu, gpu_gaussian_blur, 
        to_umat, from_umat
    )
    GPU_UTILS_AVAILABLE = True
except ImportError:
    GPU_UTILS_AVAILABLE = False
    # Fallback functions for CPU-only systems
    def is_gpu_available():
        return False
    
    def should_use_gpu(img_size, operation):
        return False
    
    def gpu_gaussian_blur(img, ksize, sigma_x, sigma_y=0, use_gpu=True):
        return cv2.GaussianBlur(img, ksize, sigma_x, sigma_y)
    
    def to_umat(img, use_gpu=True):
        return img
    
    def from_umat(umat_or_array):
        return umat_or_array


class UnsharpMaskingGPU:
    """
    GPU-accelerated Unsharp Masking Algorithm with automatic CPU fallback
    """

    def __init__(self, img, sharpen_sigma, sharpen_strength):
        self.img = img
        self.sharpen_sigma = sharpen_sigma
        self.sharpen_strength = sharpen_strength
        
        # Check if GPU acceleration should be used
        self.use_gpu = (is_gpu_available() and 
                       should_use_gpu((img.shape[0], img.shape[1]), 'gaussian_blur'))
        
        if self.use_gpu:
            print("    Using GPU acceleration for Unsharp Masking")
        else:
            print("    Using CPU implementation for Unsharp Masking")

    def apply_sharpen(self):
        """
        GPU-accelerated sharpening to the input image
        """
        luma = np.float32(self.img[:, :, 0])

        # Apply GPU-accelerated Gaussian blur if available
        if self.use_gpu and GPU_UTILS_AVAILABLE:
            smoothened = self.apply_gaussian_blur_gpu(luma)
        else:
            smoothened = self.apply_gaussian_blur_cpu(luma)
        
        # Sharpen the image with unsharp mask
        # Strength is tuneable with the sharpen_strength parameter
        sharpened = luma + ((luma - smoothened) * self.sharpen_strength)

        if self.img.dtype == "float32":
            self.img[:, :, 0] = np.clip(sharpened, 0, 1)
        else:
            self.img[:, :, 0] = np.uint8(np.clip(sharpened, 0, 255))
        
        return self.img

    def apply_gaussian_blur_gpu(self, luma):
        """
        GPU-accelerated Gaussian blur
        """
        try:
            # Convert to GPU
            gpu_luma = to_umat(luma, use_gpu=True)
            
            # Calculate kernel size based on sigma
            kernel_size = int(self.sharpen_sigma * 6 + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            # Apply GPU Gaussian blur
            gpu_smoothened = gpu_gaussian_blur(gpu_luma, (kernel_size, kernel_size), 
                                             self.sharpen_sigma, use_gpu=True)
            
            return from_umat(gpu_smoothened)
            
        except Exception as e:
            print(f"    GPU Gaussian blur failed, falling back to CPU: {e}")
            return self.apply_gaussian_blur_cpu(luma)

    def apply_gaussian_blur_cpu(self, luma):
        """
        CPU implementation of Gaussian blur
        """
        return ndimage.gaussian_filter(luma, self.sharpen_sigma)
