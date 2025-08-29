"""
File: scale.py
Description: Implements both hardware friendly and non hardware freindly scaling
Code / Paper  Reference:
https://patentimages.storage.googleapis.com/f9/11/65/a2b66f52c6dbd4/US8538199.pdf
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import numpy as np
import cv2
from util.utils import stride_convolve2d

################################################################################
class BilinearInterpolation:
    """Scale 2D image to given size using OpenCV for high performance."""

    def __init__(self, img, new_size, use_gpu=False):
        self.single_channel = np.float32(img)
        self.new_size = new_size
        self.use_gpu = use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0

    def bilinear_interpolation(self):
        """
        Upscale/Downscale 2D array by any scale factor using OpenCV's bilinear method.
        Much faster than the original nested loop implementation.
        GPU acceleration available when use_gpu=True and CUDA is available.
        """
        # OpenCV expects (width, height) format, so we reverse the size tuple
        new_size_cv2 = (self.new_size[1], self.new_size[0])
        
        if self.use_gpu:
            # GPU-accelerated version using UMat
            try:
                # Convert to UMat for GPU processing
                gpu_img = cv2.UMat(self.single_channel)
                
                # Use OpenCV's INTER_LINEAR for bilinear interpolation on GPU
                gpu_scaled = cv2.resize(gpu_img, new_size_cv2, 
                                      interpolation=cv2.INTER_LINEAR)
                
                # Get result back to CPU
                scaled_img = gpu_scaled.get()
                return scaled_img.astype("float32")
            except Exception as e:
                print(f"GPU acceleration failed, falling back to CPU: {e}")
                self.use_gpu = False
        
        # CPU version (fallback or when GPU not available)
        scaled_img = cv2.resize(self.single_channel, new_size_cv2, 
                               interpolation=cv2.INTER_LINEAR)
        
        return scaled_img.astype("float32")

    def downscale_by_int_factor(self):
        """
        Downscale a 2D array by an integer scale factor using Bilinear method with 2D convolution.
        Parameters
        ----------
        new_size: Required output size.
        Output: 16 bit scaled image in which each pixel is an average of box nxm
        determined by the scale factors.
        """

        scale_height = self.new_size[0] / self.single_channel.shape[0]
        scale_width = self.new_size[1] / self.single_channel.shape[1]

        box_height = int(np.ceil(1 / scale_height))
        box_width = int(np.ceil(1 / scale_width))

        scaled_img = np.zeros((self.new_size[0], self.new_size[1]), dtype="float32")
        kernel = np.ones((box_height, box_width)) / (box_height * box_width)

        scaled_img = stride_convolve2d(self.single_channel, kernel)
        return scaled_img.astype("float32")
