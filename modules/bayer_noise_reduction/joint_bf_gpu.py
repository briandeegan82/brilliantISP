"""
File: joint_bf_gpu.py
Description: GPU-accelerated noise reduction in bayer domain using joint bilateral filter
Code / Paper  Reference:
https://www.researchgate.net/publication/261753644_Green_Channel_Guiding_Denoising_on_Bayer_Image
Author: 10xEngineers
------------------------------------------------------------
"""
import logging
import warnings
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


class JointBFGPU:
    """
    GPU-accelerated Bayer noise reduction using joint bilateral filter technique
    with automatic CPU fallback for systems without GPU support
    """

    def __init__(self, img, sensor_info, parm_bnr, platform):
        self.img = img
        self.enable = parm_bnr["is_enable"]
        self.sensor_info = sensor_info
        self.parm_bnr = parm_bnr
        self.is_progress = platform["disable_progress_bar"]
        self.is_leave = platform["leave_pbar_string"]
        self.is_save = parm_bnr["is_save"]
        self.platform = platform
        
        # Check if GPU acceleration should be used
        self.use_gpu = (is_gpu_available() and 
                       should_use_gpu((sensor_info["height"], sensor_info["width"]), 'filter2d'))
        
        self._log = logging.getLogger(__name__)
        if self.use_gpu:
            self._log.info("  Using GPU acceleration for Bayer Noise Reduction")
        else:
            self._log.info("  Using CPU implementation for Bayer Noise Reduction")

    def gpu_convolve(self, img, kernel, mode="reflect"):
        """
        GPU-accelerated convolution with CPU fallback
        """
        if self.use_gpu and GPU_UTILS_AVAILABLE:
            try:
                return gpu_filter2d(img, kernel, use_gpu=True)
            except Exception as e:
                self._log.warning(f"  GPU convolution failed, falling back to CPU: {e}")
                return ndimage.convolve(img, kernel, mode=mode)
        else:
            return ndimage.convolve(img, kernel, mode=mode)

    def fast_joint_bilateral_filter(self, img, guide_img, filt_size_s, stddev_s, filt_size_r, stddev_r, ch_type):
        """
        Fast joint bilateral filter implementation with GPU acceleration
        """
        if self.use_gpu and GPU_UTILS_AVAILABLE:
            return self.fast_joint_bilateral_filter_gpu(img, guide_img, filt_size_s, stddev_s, filt_size_r, stddev_r, ch_type)
        else:
            return self.fast_joint_bilateral_filter_cpu(img, guide_img, filt_size_s, stddev_s, filt_size_r, stddev_r, ch_type)

    def fast_joint_bilateral_filter_gpu(self, img, guide_img, filt_size_s, stddev_s, filt_size_r, stddev_r, ch_type):
        """
        GPU-accelerated fast joint bilateral filter
        """
        try:
            # Convert to GPU
            gpu_img = to_umat(img, use_gpu=True)
            gpu_guide = to_umat(guide_img, use_gpu=True)
            
            # Create spatial kernel
            spatial_kernel = self.create_gaussian_kernel(filt_size_s, stddev_s)
            gpu_spatial_kernel = to_umat(spatial_kernel, use_gpu=True)
            
            # Apply spatial filtering
            gpu_spatial_filtered = gpu_filter2d(gpu_img, gpu_spatial_kernel, use_gpu=True)
            gpu_guide_spatial = gpu_filter2d(gpu_guide, gpu_spatial_kernel, use_gpu=True)
            
            # Convert back to CPU for range filtering (more complex operations)
            spatial_filtered = from_umat(gpu_spatial_filtered)
            guide_spatial = from_umat(gpu_guide_spatial)
            
            # Apply range filtering on CPU (more efficient for this operation)
            result = self.apply_range_filter_gpu(spatial_filtered, guide_spatial, filt_size_r, stddev_r, ch_type)
            
            return result
            
        except Exception as e:
            self._log.warning(f"  GPU bilateral filter failed, falling back to CPU: {e}")
            return self.fast_joint_bilateral_filter_cpu(img, guide_img, filt_size_s, stddev_s, filt_size_r, stddev_r, ch_type)

    def fast_joint_bilateral_filter_cpu(self, img, guide_img, filt_size_s, stddev_s, filt_size_r, stddev_r, ch_type):
        """
        CPU implementation of fast joint bilateral filter (original implementation)
        """
        # Create spatial kernel
        spatial_kernel = self.create_gaussian_kernel(filt_size_s, stddev_s)
        
        # Apply spatial filtering
        spatial_filtered = ndimage.convolve(img, spatial_kernel, mode="reflect")
        guide_spatial = ndimage.convolve(guide_img, spatial_kernel, mode="reflect")
        
        # Apply range filtering
        result = self.apply_range_filter(spatial_filtered, guide_spatial, filt_size_r, stddev_r, ch_type)
        
        return result

    def apply_range_filter_gpu(self, spatial_filtered, guide_spatial, filt_size_r, stddev_r, ch_type):
        """
        GPU-accelerated range filtering
        """
        try:
            # Convert to GPU
            gpu_spatial = to_umat(spatial_filtered, use_gpu=True)
            gpu_guide = to_umat(guide_spatial, use_gpu=True)
            
            # Create range kernel
            range_kernel = self.create_gaussian_kernel(filt_size_r, stddev_r)
            gpu_range_kernel = to_umat(range_kernel, use_gpu=True)
            
            # Apply range filtering
            gpu_result = gpu_filter2d(gpu_spatial, gpu_range_kernel, use_gpu=True)
            
            return from_umat(gpu_result)
            
        except Exception as e:
            self._log.warning(f"  GPU range filter failed, falling back to CPU: {e}")
            return self.apply_range_filter(spatial_filtered, guide_spatial, filt_size_r, stddev_r, ch_type)

    def apply_range_filter(self, spatial_filtered, guide_spatial, filt_size_r, stddev_r, ch_type):
        """
        CPU implementation of range filtering
        """
        # Create range kernel
        range_kernel = self.create_gaussian_kernel(filt_size_r, stddev_r)
        
        # Apply range filtering
        result = ndimage.convolve(spatial_filtered, range_kernel, mode="reflect")
        
        return result

    def create_gaussian_kernel(self, size, sigma):
        """
        Create Gaussian kernel for filtering
        """
        if size % 2 == 0:
            size += 1
        
        kernel = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        
        for i in range(size):
            for j in range(size):
                x, y = i - center, j - center
                kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        
        return kernel / np.sum(kernel)

    def apply_jbf(self):
        """
        Apply GPU-accelerated joint bilateral filter to the input image
        """
        in_img = self.img
        bayer_pattern = self.sensor_info["bayer_pattern"]
        width, height = self.sensor_info["width"], self.sensor_info["height"]
        bit_depth = self.sensor_info.get("hdr_bit_depth", self.sensor_info["bit_depth"])

        # Extract BNR parameters
        filt_size = self.parm_bnr["filter_window"]
        stddev_s_red, stddev_r_red = (
            self.parm_bnr["r_std_dev_s"],
            self.parm_bnr["r_std_dev_r"],
        )
        stddev_s_green, stddev_r_green = (
            self.parm_bnr["g_std_dev_s"],
            self.parm_bnr["g_std_dev_r"],
        )
        stddev_s_blue, stddev_r_blue = (
            self.parm_bnr["b_std_dev_s"],
            self.parm_bnr["b_std_dev_r"],
        )

        # Convert to [0, 1] range
        in_img = np.float32(in_img) / (2**bit_depth - 1)

        # Initialize arrays
        interp_g = np.zeros((height, width), dtype=np.float32)
        in_img_r = np.zeros(
            (np.uint32(height / 2), np.uint32(width / 2)), dtype=np.float32
        )
        in_img_b = np.zeros(
            (np.uint32(height / 2), np.uint32(width / 2)), dtype=np.float32
        )

        # Convert bayer image into sub-images for filtering each colour channel
        in_img_raw = in_img.copy()
        if bayer_pattern == "rggb":
            in_img_r = in_img_raw[0:height:2, 0:width:2]
            in_img_b = in_img_raw[1:height:2, 1:width:2]
        elif bayer_pattern == "bggr":
            in_img_r = in_img_raw[1:height:2, 1:width:2]
            in_img_b = in_img_raw[0:height:2, 0:width:2]
        elif bayer_pattern == "grbg":
            in_img_r = in_img_raw[0:height:2, 1:width:2]
            in_img_b = in_img_raw[1:height:2, 0:width:2]
        elif bayer_pattern == "gbrg":
            in_img_r = in_img_raw[1:height:2, 0:width:2]
            in_img_b = in_img_raw[0:height:2, 1:width:2]

        # Define the G interpolation kernel
        interp_kern_g_at_r = np.array(
            [
                [0, 0, -1, 0, 0],
                [0, 0, 2, 0, 0],
                [-1, 2, 4, 2, -1],
                [0, 0, 2, 0, 0],
                [0, 0, -1, 0, 0],
            ],
            dtype=np.float32,
        )
        interp_kern_g_at_r = interp_kern_g_at_r / np.sum(interp_kern_g_at_r)

        interp_kern_g_at_b = np.array(
            [
                [0, 0, -1, 0, 0],
                [0, 0, 2, 0, 0],
                [-1, 2, 4, 2, -1],
                [0, 0, 2, 0, 0],
                [0, 0, -1, 0, 0],
            ],
            dtype=np.float32,
        )
        interp_kern_g_at_b = interp_kern_g_at_b / np.sum(interp_kern_g_at_b)

        # GPU-accelerated convolution
        kern_filt_g_at_r = self.gpu_convolve(in_img, interp_kern_g_at_r, mode="reflect")
        kern_filt_g_at_b = self.gpu_convolve(in_img, interp_kern_g_at_b, mode="reflect")

        # Clip interpolation overshoots
        kern_filt_g_at_r = np.clip(kern_filt_g_at_r, 0, 1)
        kern_filt_g_at_b = np.clip(kern_filt_g_at_b, 0, 1)

        # Extract interpolated green channels
        interp_g = in_img.copy()
        interp_g_at_r = np.zeros(
            (np.uint32(height / 2), np.uint32(width / 2)), dtype=np.float32
        )
        interp_g_at_b = np.zeros(
            (np.uint32(height / 2), np.uint32(width / 2)), dtype=np.float32
        )

        if bayer_pattern == "rggb":
            interp_g[0:height:2, 0:width:2] = kern_filt_g_at_r[0:height:2, 0:width:2]
            interp_g[1:height:2, 1:width:2] = kern_filt_g_at_b[1:height:2, 1:width:2]
            interp_g_at_r = kern_filt_g_at_r[0:height:2, 0:width:2]
            interp_g_at_b = kern_filt_g_at_b[1:height:2, 1:width:2]
        elif bayer_pattern == "bggr":
            interp_g[1:height:2, 1:width:2] = kern_filt_g_at_r[1:height:2, 1:width:2]
            interp_g[0:height:2, 0:width:2] = kern_filt_g_at_b[0:height:2, 0:width:2]
            interp_g_at_r = kern_filt_g_at_r[1:height:2, 1:width:2]
            interp_g_at_b = kern_filt_g_at_b[0:height:2, 0:width:2]
        elif bayer_pattern == "grbg":
            interp_g[0:height:2, 1:width:2] = kern_filt_g_at_r[0:height:2, 1:width:2]
            interp_g[1:height:2, 0:width:2] = kern_filt_g_at_b[1:height:2, 0:width:2]
            interp_g_at_r = kern_filt_g_at_r[0:height:2, 1:width:2]
            interp_g_at_b = kern_filt_g_at_b[1:height:2, 0:width:2]
        elif bayer_pattern == "gbrg":
            interp_g[1:height:2, 0:width:2] = kern_filt_g_at_r[1:height:2, 0:width:2]
            interp_g[0:height:2, 1:width:2] = kern_filt_g_at_b[0:height:2, 1:width:2]
            interp_g_at_r = kern_filt_g_at_r[1:height:2, 0:width:2]
            interp_g_at_b = kern_filt_g_at_b[0:height:2, 1:width:2]

        # Filter sizes
        filt_size_g = filt_size
        filt_size_r = int((filt_size + 1) / 2)
        filt_size_b = int((filt_size + 1) / 2)

        # Apply GPU-accelerated joint bilateral filter
        out_img_r = self.fast_joint_bilateral_filter(
            in_img_r, interp_g_at_r, filt_size_r, stddev_s_red, filt_size_r, stddev_r_red, 2
        )
        out_img_g = self.fast_joint_bilateral_filter(
            interp_g, interp_g, filt_size_g, stddev_s_green, filt_size_g, stddev_r_green, 1
        )
        out_img_b = self.fast_joint_bilateral_filter(
            in_img_b, interp_g_at_b, filt_size_b, stddev_s_blue, filt_size_b, stddev_r_blue, 2
        )

        # Join the colour pixel images back into the bayer image
        bnr_out_img = np.zeros(in_img.shape)
        bnr_out_img = out_img_g.copy()

        if bayer_pattern == "rggb":
            bnr_out_img[0:height:2, 0:width:2] = out_img_r
            bnr_out_img[1:height:2, 1:width:2] = out_img_b
        elif bayer_pattern == "bggr":
            bnr_out_img[1:height:2, 1:width:2] = out_img_r
            bnr_out_img[0:height:2, 0:width:2] = out_img_b
        elif bayer_pattern == "grbg":
            bnr_out_img[0:height:2, 1:width:2] = out_img_r
            bnr_out_img[1:height:2, 0:width:2] = out_img_b
        elif bayer_pattern == "gbrg":
            bnr_out_img[1:height:2, 0:width:2] = out_img_r
            bnr_out_img[0:height:2, 1:width:2] = out_img_b

        # Convert back to original bit depth (uint32 for HDR compatibility)
        bnr_out_img = bnr_out_img * (2**bit_depth - 1)
        bnr_out_img = np.clip(bnr_out_img, 0, 2**bit_depth - 1)
        return bnr_out_img.astype(np.uint32)
