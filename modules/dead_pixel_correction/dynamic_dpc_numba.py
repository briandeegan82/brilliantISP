"""
File: dynamic_dpc_numba.py
Description: Numba-optimized dead pixel correction
Code / Paper  Reference: https://ieeexplore.ieee.org/document/9194921
Implementation inspired from: (OpenISP) https://github.com/cruxopen/openISP
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import logging
import numpy as np
from numba import jit, prange
import time

# Try to import Numba, fall back to CPU if not available
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logging.getLogger(__name__).info("Numba not available, using CPU implementation")


class DynamicDPCNumba:
    """
    Numba-optimized dynamic dead pixel correction
    Uses hybrid approach: NumPy for optimized operations, Numba for custom logic
    """

    def __init__(self, img, sensor_info, parm_dpc):
        self.img = img
        self.sensor_info = sensor_info
        self.bpp = self.sensor_info["bit_depth"]
        self.threshold = parm_dpc["dp_threshold"]
        self.is_debug = parm_dpc["is_debug"]
        self.use_numba = NUMBA_AVAILABLE and self._should_use_numba()
        
        self._log = logging.getLogger(__name__)
        if self.use_numba:
            self._log.info("  Using Numba-optimized dead pixel correction")
        else:
            self._log.info("  Using CPU dead pixel correction")

    def _should_use_numba(self):
        """Determine if Numba optimization should be used based on image size."""
        if not NUMBA_AVAILABLE:
            return False
        
        # Use Numba for images larger than 500K pixels
        image_size = self.img.shape[0] * self.img.shape[1]
        return image_size > 500000  # 500K threshold

    @staticmethod
    @jit(nopython=True, parallel=True)
    def fast_correction_application_numba(img, detection_mask, vertical_grad, horizontal_grad, 
                                         left_diagonal_grad, right_diagonal_grad, height, width):
        """
        Numba-optimized correction application (the most compute-intensive part)
        """
        corrected_img = np.copy(img)
        
        for i in prange(2, height - 2):
            for j in range(2, width - 2):
                if detection_mask[i, j] == 1:
                    # Find minimum gradient direction
                    gradients = np.array([
                        vertical_grad[i, j],
                        horizontal_grad[i, j],
                        left_diagonal_grad[i, j],
                        right_diagonal_grad[i, j]
                    ])
                    
                    min_grad_idx = np.argmin(gradients)
                    
                    # Compute correction based on minimum gradient direction
                    if min_grad_idx == 0:  # Vertical
                        corrected_img[i, j] = (img[i-2, j] + img[i+2, j]) / 2.0
                    elif min_grad_idx == 1:  # Horizontal
                        corrected_img[i, j] = (img[i, j-2] + img[i, j+2]) / 2.0
                    elif min_grad_idx == 2:  # Left diagonal
                        corrected_img[i, j] = (img[i-2, j+2] + img[i+2, j-2]) / 2.0
                    else:  # Right diagonal
                        corrected_img[i, j] = (img[i-2, j-2] + img[i+2, j+2]) / 2.0
        
        return corrected_img

    def dynamic_dpc_numba(self):
        """
        Hybrid Numba-optimized dynamic dead pixel correction
        Uses NumPy for optimized operations, Numba for custom logic
        """
        from scipy.ndimage import maximum_filter, minimum_filter, correlate
        
        height, width = self.sensor_info["height"], self.sensor_info["width"]

        # Use NumPy's optimized operations for the heavy lifting
        # Get 3x3 neighbourhood of each pixel using 5x5 window
        window = np.array(
            [
                [1, 0, 1, 0, 1],
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [1, 0, 1, 0, 1],
            ]
        )

        # Use NumPy's optimized filters
        max_value = maximum_filter(self.img, footprint=window, mode="mirror")
        min_value = minimum_filter(self.img, footprint=window, mode="mirror")

        # Condition 1: center_pixel needs to be corrected if it lies outside the interval
        mask_cond1 = (
            np.where((min_value > self.img) | (self.img > max_value), True, False)
        ).astype("int32")

        # Condition 2: Use NumPy's optimized correlate for gradient computation
        # Kernels to compute the difference between center pixel and each of the 8 neighbours
        ker_top_left = np.array(
            [
                [-1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        ker_top_mid = np.array(
            [
                [0, 0, -1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        ker_top_right = np.array(
            [
                [0, 0, 0, 0, -1],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        ker_mid_left = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [-1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        ker_mid_right = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, -1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        ker_bottom_left = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [-1, 0, 0, 0, 0],
            ]
        )
        ker_bottom_mid = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, -1, 0, 0],
            ]
        )
        ker_bottom_right = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, -1],
            ]
        )

        # Use NumPy's optimized correlate
        diff_top_left = np.abs(correlate(self.img, ker_top_left, mode="mirror"))
        diff_top_mid = np.abs(correlate(self.img, ker_top_mid, mode="mirror"))
        diff_top_right = np.abs(correlate(self.img, ker_top_right, mode="mirror"))
        diff_mid_left = np.abs(correlate(self.img, ker_mid_left, mode="mirror"))
        diff_mid_right = np.abs(correlate(self.img, ker_mid_right, mode="mirror"))
        diff_bottom_left = np.abs(correlate(self.img, ker_bottom_left, mode="mirror"))
        diff_bottom_mid = np.abs(correlate(self.img, ker_bottom_mid, mode="mirror"))
        diff_bottom_right = np.abs(correlate(self.img, ker_bottom_right, mode="mirror"))

        # Stack all arrays
        diff_array = np.stack(
            [
                diff_top_left,
                diff_top_mid,
                diff_top_right,
                diff_mid_left,
                diff_mid_right,
                diff_bottom_left,
                diff_bottom_mid,
                diff_bottom_right,
            ],
            axis=2,
        )

        # All gradients must be greater than the threshold for a pixel to be a DP
        mask_cond2 = np.all(np.where(diff_array > self.threshold, True, False), axis=2)

        # Detection mask with 1 for DPs and 0 for good pixels
        detection_mask = mask_cond1 * mask_cond2

        # Compute gradients using NumPy's optimized operations
        ker_v = np.array([[-1, 0, 2, 0, -1]]).T
        ker_h = np.array([[-1, 0, 2, 0, -1]])
        ker_left_dia = np.array(
            [
                [0, 0, 0, 0, -1],
                [0, 0, 0, 0, 0],
                [0, 0, 2, 0, 0],
                [0, 0, 0, 0, 0],
                [-1, 0, 0, 0, 0],
            ]
        )
        ker_right_dia = np.array(
            [
                [-1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 2, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, -1],
            ]
        )

        # Use NumPy's optimized correlate for gradients
        vertical_grad = np.abs(correlate(self.img, ker_v, mode="mirror"))
        horizontal_grad = np.abs(correlate(self.img, ker_h, mode="mirror"))
        left_diagonal_grad = np.abs(correlate(self.img, ker_left_dia, mode="mirror"))
        right_diagonal_grad = np.abs(correlate(self.img, ker_right_dia, mode="mirror"))

        # Compute the direction of the minimum gradient
        min_grad = np.min(
            np.stack(
                [
                    vertical_grad,
                    horizontal_grad,
                    left_diagonal_grad,
                    right_diagonal_grad,
                ],
                axis=2,
            ),
            axis=2,
        )

        # Compute mean values using NumPy's optimized operations
        ker_mean_v = np.array([[1, 0, 0, 0, 1]]).T / 2
        ker_mean_h = np.array([[1, 0, 0, 0, 1]]) / 2
        ker_mean_ldia = (
            np.array(
                [
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                ]
            )
            / 2
        )
        ker_mean_rdia = (
            np.array(
                [
                    [1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1],
                ]
            )
            / 2
        )

        # Use NumPy's optimized correlate for mean computation
        mean_v = correlate(self.img, ker_mean_v, mode="mirror")
        mean_h = correlate(self.img, ker_mean_h, mode="mirror")
        mean_ldia = correlate(self.img, ker_mean_ldia, mode="mirror")
        mean_rdia = correlate(self.img, ker_mean_rdia, mode="mirror")

        # Use Numba for the final correction application (most compute-intensive part)
        corrected_img = self.fast_correction_application_numba(
            self.img, detection_mask, vertical_grad, horizontal_grad,
            left_diagonal_grad, right_diagonal_grad, height, width
        )

        # Debug information
        if self.is_debug:
            num_corrected = np.count_nonzero(detection_mask)
            self._log.info(f"   - Number of corrected pixels = {num_corrected}")
            self._log.info(f"   - Threshold = {self.threshold}")

        # Clip to valid range and convert to uint16
        max_val = (2**self.bpp) - 1
        corrected_img = np.clip(corrected_img, 0, max_val)

        return corrected_img.astype(np.uint16)

    def dynamic_dpc_cpu(self):
        """
        CPU fallback implementation using original algorithm
        """
        # Import the original implementation
        from modules.dead_pixel_correction.dynamic_dpc import DynamicDPC
        dpc = DynamicDPC(self.img, self.sensor_info, {"dp_threshold": self.threshold, "is_debug": self.is_debug})
        return dpc.dynamic_dpc()

    def dynamic_dpc(self):
        """
        Apply dynamic dead pixel correction with Numba optimization
        """
        if self.use_numba:
            return self.dynamic_dpc_numba()
        else:
            return self.dynamic_dpc_cpu()
