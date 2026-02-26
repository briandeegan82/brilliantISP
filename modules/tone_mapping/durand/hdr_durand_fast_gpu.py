import logging
import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from util.utils import save_output_array
import time
import cv2

# Import GPU utilities with fallback
try:
    from util.gpu_utils import (
        is_gpu_available, should_use_gpu, gpu_bilateral_filter, 
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
    
    def gpu_bilateral_filter(img, d, sigma_color, sigma_space, use_gpu=True):
        return cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    
    def gpu_gaussian_blur(img, ksize, sigma_x, sigma_y=0, use_gpu=True):
        return cv2.GaussianBlur(img, ksize, sigma_x, sigma_y)
    
    def to_umat(img, use_gpu=True):
        return img
    
    def from_umat(umat_or_array):
        return umat_or_array

class HDRDurandToneMappingGPU:
    """
    GPU-accelerated HDR Durand Tone Mapping Algorithm Implementation
    with automatic CPU fallback for systems without GPU support
    """
    
    def __init__(self, img, platform, sensor_info, params):
        self.img = img.copy()
        self.is_enable = params.get("is_enable", True)
        self.is_save = params.get("is_save", False)
        self.is_debug = params.get("is_debug", False)
        self.sigma_space = params.get("sigma_space", 2.0)
        self.sigma_color = params.get("sigma_color", 0.4)
        self.contrast_factor = params.get("contrast_factor", 2.0)
        self.downsample_factor = params.get("downsample_factor", 4)
        self.output_bit_depth = sensor_info.get("output_bit_depth", 8)
        self.sensor_info = sensor_info
        self.platform = platform
        
        # Check if GPU acceleration should be used
        self.use_gpu = (is_gpu_available() and 
                       should_use_gpu((sensor_info.get("height", 1000), sensor_info.get("width", 1000)), 'bilateral_filter'))
        
        self._log = logging.getLogger(__name__)
        if self.use_gpu:
            self._log.info("  Using GPU acceleration for HDR Tone Mapping")
        else:
            self._log.info("  Using CPU implementation for HDR Tone Mapping")
    
    def normalize(self, image):
        """ Normalize image to [0,1] range."""
        return (image - np.min(image)) / (np.max(image) - np.min(image))
    
    def fast_bilateral_filter(self, image):
        """
        GPU-accelerated approximate bilateral filtering using a downsampled approach.
        """
        if self.use_gpu and GPU_UTILS_AVAILABLE:
            return self.fast_bilateral_filter_gpu(image)
        else:
            return self.fast_bilateral_filter_cpu(image)
    
    def fast_bilateral_filter_gpu(self, image):
        """
        GPU-accelerated bilateral filtering
        """
        try:
            small_img = zoom(image, 1 / self.downsample_factor, order=1)
            small_filtered = self.bilateral_filter_gpu(small_img, self.sigma_color, self.sigma_space)
            return zoom(small_filtered, self.downsample_factor, order=1)
        except Exception as e:
            self._log.warning(f"  GPU bilateral filter failed, falling back to CPU: {e}")
            return self.fast_bilateral_filter_cpu(image)
    
    def fast_bilateral_filter_cpu(self, image):
        """
        CPU implementation of bilateral filtering
        """
        small_img = zoom(image, 1 / self.downsample_factor, order=1)
        small_filtered = self.bilateral_filter_cpu(small_img, self.sigma_color, self.sigma_space)
        return zoom(small_filtered, self.downsample_factor, order=1)
    
    def bilateral_filter(self, image, sigma_color, sigma_space):
        """
        Bilateral filter with GPU acceleration if available
        """
        if self.use_gpu and GPU_UTILS_AVAILABLE:
            return self.bilateral_filter_gpu(image, sigma_color, sigma_space)
        else:
            return self.bilateral_filter_cpu(image, sigma_color, sigma_space)
    
    def bilateral_filter_gpu(self, image, sigma_color, sigma_space):
        """
        GPU-accelerated bilateral filter using direct CUDA
        """
        try:
            # Try direct CUDA bilateral filter first (faster than UMat)
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(image.astype(np.float32))
            gpu_result = cv2.cuda.bilateralFilter(gpu_img, 15, sigma_color * 100, sigma_space)
            return gpu_result.download()
        except Exception as e:
            # Fallback to UMat
            try:
                return gpu_bilateral_filter(image, 15, sigma_color * 100, sigma_space, use_gpu=True)
            except Exception as e2:
                self._log.warning(f"  GPU bilateral filter failed, falling back to CPU: {e2}")
                return self.bilateral_filter_cpu(image, sigma_color, sigma_space)
    
    def bilateral_filter_cpu(self, image, sigma_color, sigma_space):
        """
        CPU implementation of bilateral filter using Gaussian filtering approximation
        """
        spatial_filtered = gaussian_filter(image, sigma=sigma_space)
        intensity_diff = image - spatial_filtered
        range_kernel = np.exp(-0.5 * (intensity_diff / sigma_color) ** 2)
        return spatial_filtered + range_kernel * intensity_diff
    
    def apply_tone_mapping(self):
        """ GPU-accelerated Durand's tone mapping implementation. """
        # Convert to log domain
        epsilon = 1e-6  # Small value to avoid log(0)
        log_luminance = np.log10(self.img + epsilon)
    
        # Apply bilateral filter to get the base layer
        if self.use_gpu and GPU_UTILS_AVAILABLE:
            try:
                # Use GPU-accelerated bilateral filter
                log_base = self.bilateral_filter_gpu(log_luminance.astype(np.float32), 
                                               self.sigma_color, 
                                               self.sigma_space)
            except Exception as e:
                self._log.warning(f"  GPU tone mapping failed, falling back to CPU: {e}")
                log_base = self.bilateral_filter_cpu(log_luminance.astype(np.float32), 
                                               self.sigma_color, 
                                               self.sigma_space)
        else:
            # Use CPU bilateral filter
            log_base = self.bilateral_filter_cpu(log_luminance.astype(np.float32), 
                                           self.sigma_color, 
                                           self.sigma_space)
    
        # Extract the detail layer
        log_detail = log_luminance - log_base
    
        # Compress the base layer (reduce contrast)
        compressed_log_base = log_base / self.contrast_factor
    
        # Recombine base and detail layers
        log_output = compressed_log_base + log_detail
    
        # Convert back from log domain
        output_luminance = np.power(10, log_output)
    
        # Normalize to [0, 1] range
        output_luminance = (output_luminance - np.min(output_luminance)) / (np.max(output_luminance) - np.min(output_luminance))
    
        if self.output_bit_depth == 8:
            return (output_luminance * 255).astype(np.uint8)
        elif self.output_bit_depth == 16:
            return (output_luminance * 65535).astype(np.uint16)
        elif self.output_bit_depth == 32:
            return output_luminance.astype(np.float32)
        else:
            raise ValueError("Unsupported output bit depth. Use 8, 16, or 32.")
    
    def save(self):
        if self.is_save:
            save_output_array(self.platform["in_file"], self.img, "Out_hdr_durand_", 
                              self.platform, self.sensor_info["bit_depth"], self.sensor_info["bayer_pattern"])
    
    def execute(self):
        if self.is_enable is True:
            self._log.info("Executing HDR Durand Tone Mapping...")
            start = time.time()
            self.img = self.apply_tone_mapping()
            execution_time = time.time() - start
            self._log.info(f"Execution time: {execution_time:.3f}s")
            
        self.save()
        return self.img
