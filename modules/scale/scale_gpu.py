"""
File: scale_gpu.py
Description: GPU-accelerated scaling implementation with CPU fallback
Code / Paper  Reference:
https://patentimages.storage.googleapis.com/f9/11/65/a2b66f52c6dbd4/US8538199.pdf
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import re
import numpy as np
from util.utils import crop
from util.utils import save_output_array_yuv, save_output_array

# Import GPU utilities with fallback
try:
    from util.gpu_utils import (
        is_gpu_available, should_use_gpu, gpu_resize, 
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
    
    def gpu_resize(img, size, interpolation, use_gpu=True):
        import cv2
        return cv2.resize(img, size, interpolation=interpolation)
    
    def to_umat(img, use_gpu=True):
        return img
    
    def from_umat(umat_or_array):
        return umat_or_array


class ScaleGPU:
    """GPU-accelerated scale color image to given size with CPU fallback."""

    def __init__(self, img, platform, sensor_info, parm_sca, conv_std):
        self.img = img
        self.enable = parm_sca["is_enable"]
        self.sensor_info = sensor_info
        self.parm_sca = parm_sca
        self.is_save = parm_sca["is_save"]
        self.platform = platform
        self.conv_std = conv_std
        self.get_scaling_params()
        
        # Check if GPU acceleration should be used
        self.use_gpu = (is_gpu_available() and 
                       should_use_gpu((sensor_info["height"], sensor_info["width"]), 'resize'))
        
        if self.use_gpu:
            print("  Using GPU acceleration for Image Scaling")
        else:
            print("  Using CPU implementation for Image Scaling")

    def apply_scaling(self):
        """Execute GPU-accelerated scaling."""

        # check if no change in size
        if self.old_size == self.new_size:
            if self.is_debug:
                print("   - Output size is the same as input size.")
            return self.img

        if self.img.dtype == "float32":
            scaled_img = np.empty(
                (self.new_size[0], self.new_size[1], 3), dtype="float32"
            )
        else:
            scaled_img = np.empty(
                (self.new_size[0], self.new_size[1], 3), dtype="uint8"
            )

        # Determine interpolation method
        if self.parm_sca["algorithm"].lower() == "bilinear":
            interpolation = cv2.INTER_LINEAR
        elif self.parm_sca["algorithm"].lower() == "nearest_neighbor":
            interpolation = cv2.INTER_NEAREST
        else:
            interpolation = cv2.INTER_LINEAR  # Default to bilinear

        # Loop over each channel to resize the image
        for i in range(3):
            ch_arr = self.img[:, :, i]
            
            if self.use_gpu and GPU_UTILS_AVAILABLE:
                # Use GPU-accelerated scaling
                scaled_ch = self.scale_channel_gpu(ch_arr, interpolation)
            else:
                # Use CPU scaling
                scaled_ch = self.scale_channel_cpu(ch_arr, interpolation)

            # If input size is invalid, return the original image
            if scaled_ch.shape == self.old_size:
                return self.img
            else:
                scaled_img[:, :, i] = scaled_ch

            # Because each channel is scaled in the same way, the isDebug flag is turned
            # off after the first channel has been scaled.
            self.parm_sca["is_debug"] = False

        return scaled_img

    def scale_channel_gpu(self, ch_arr, interpolation):
        """
        GPU-accelerated channel scaling
        """
        try:
            # OpenCV expects (width, height) format
            new_size_cv2 = (self.new_size[1], self.new_size[0])
            
            # Use GPU-accelerated resize
            scaled_ch = gpu_resize(ch_arr, new_size_cv2, interpolation, use_gpu=True)
            
            return scaled_ch
            
        except Exception as e:
            print(f"    GPU scaling failed, falling back to CPU: {e}")
            return self.scale_channel_cpu(ch_arr, interpolation)

    def scale_channel_cpu(self, ch_arr, interpolation):
        """
        CPU implementation of channel scaling
        """
        import cv2
        # OpenCV expects (width, height) format
        new_size_cv2 = (self.new_size[1], self.new_size[0])
        
        # Use CPU resize
        scaled_ch = cv2.resize(ch_arr, new_size_cv2, interpolation=interpolation)
        
        return scaled_ch

    def get_scaling_params(self):
        """Save parameters as instance attributes."""
        self.is_debug = self.parm_sca["is_debug"]
        self.old_size = (self.sensor_info["height"], self.sensor_info["width"])
        self.new_size = (self.parm_sca["new_height"], self.parm_sca["new_width"])

    def save(self):
        """
        Function to save module output
        """
        # update size of array in filename
        self.platform["in_file"] = re.sub(
            r"\d+x\d+",
            f"{self.img.shape[1]}x{self.img.shape[0]}",
            self.platform["in_file"],
        )
        if self.is_save:
            if self.platform["rgb_output"]:
                save_output_array(
                    self.platform["in_file"],
                    self.img,
                    "Out_scale_",
                    self.platform,
                    self.sensor_info["bit_depth"],
                    self.sensor_info["bayer_pattern"],
                )
            else:
                save_output_array_yuv(
                    self.platform["in_file"],
                    self.img,
                    "Out_scale_",
                    self.platform,
                    self.conv_std,
                )

    def execute(self):
        """
        Applying scaling to input image
        """
        print("Scale = " + str(self.enable))

        if self.enable is True:
            start = time.time()
            s_out = self.apply_scaling()
            execution_time = time.time() - start
            print(f"  Execution time: {execution_time:.3f}s")
            self.img = s_out

        self.save()
        return self.img
