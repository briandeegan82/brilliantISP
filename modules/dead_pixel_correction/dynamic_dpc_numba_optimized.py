"""
File: dynamic_dpc_numba_optimized.py
Description: Optimized dead pixel correction using Numba for compute-intensive loops
Code / Paper  Reference: https://ieeexplore.ieee.org/document/9194921
Implementation inspired from: (OpenISP) https://github.com/cruxopen/openISP
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import logging
import numpy as np
from util.debug_utils import get_debug_logger
from scipy.ndimage import maximum_filter, minimum_filter
from numba import njit, prange
import time

# Try to import Numba, fall back to CPU if not available
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logging.getLogger(__name__).info("Numba not available, using CPU implementation")


class DynamicDPCNumbaOptimized:
    """
    Dynamic Dead Pixel Correction (DPC):
    Detects and corrects defective pixels using neighborhood-based rules
    and gradient-guided interpolation. Optimized with Numba.
    """

    def __init__(self, img, sensor_info, parm_dpc, platform=None):
        self.img = img.astype(np.float32)
        self.sensor_info = sensor_info
        self.bpp = self.sensor_info.get("hdr_bit_depth", self.sensor_info["bit_depth"])
        self.threshold = parm_dpc["dp_threshold"]
        self.is_debug = parm_dpc.get("is_debug", False)
        self.use_numba = NUMBA_AVAILABLE and self._should_use_numba()
        self.logger = get_debug_logger("DeadPixelCorrection", config=platform or {})

        if self.use_numba:
            self.logger.info("  Using Numba-optimized dead pixel correction (hybrid approach)")
        else:
            self.logger.info("  Using CPU dead pixel correction")

    def _should_use_numba(self):
        """Determine if Numba optimization should be used based on image size."""
        if not NUMBA_AVAILABLE:
            return False
        
        # Use Numba for images larger than 500K pixels
        image_size = self.img.shape[0] * self.img.shape[1]
        return image_size > 500000  # 500K threshold

    def dynamic_dpc(self, return_mask=False):
        height, width = self.sensor_info["height"], self.sensor_info["width"]

        # Define 5x5 neighborhood footprint (cross-like)
        window = np.array(
            [
                [1, 0, 1, 0, 1],
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [1, 0, 1, 0, 1],
            ]
        )

        # Compute min/max of neighborhood using optimized NumPy operations
        max_value = maximum_filter(self.img, footprint=window, mode="mirror")
        min_value = minimum_filter(self.img, footprint=window, mode="mirror")

        # Condition 1: pixel outside valid range
        mask_cond1 = (self.img < min_value) | (self.img > max_value)

        # Condition 2: pixel differs from all neighbors by > threshold
        neighbor_offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

        diffs = [
            np.abs(self.img - np.roll(np.roll(self.img, dy, axis=0), dx, axis=1))
            for dy, dx in neighbor_offsets
        ]
        diff_array = np.stack(diffs, axis=2)
        mask_cond2 = np.all(diff_array > self.threshold, axis=2)

        # Final detection mask
        detection_mask = mask_cond1 & mask_cond2

        # Compute directional gradients
        vertical_grad = np.abs(self.img - np.roll(self.img, 2, axis=0))
        horizontal_grad = np.abs(self.img - np.roll(self.img, 2, axis=1))
        left_diag_grad = np.abs(self.img - np.roll(np.roll(self.img, 2, axis=0), 2, axis=1))
        right_diag_grad = np.abs(self.img - np.roll(np.roll(self.img, 2, axis=0), -2, axis=1))

        # Stack gradients and find min direction
        grads = np.stack([vertical_grad, horizontal_grad, left_diag_grad, right_diag_grad], axis=2)
        min_grad = np.min(grads, axis=2)

        # Directional means (2-neighbor average)
        mean_v = (np.roll(self.img, -2, axis=0) + np.roll(self.img, 2, axis=0)) / 2
        mean_h = (np.roll(self.img, -2, axis=1) + np.roll(self.img, 2, axis=1)) / 2
        mean_ld = (np.roll(np.roll(self.img, -2, axis=0), -2, axis=1) +
                   np.roll(np.roll(self.img, 2, axis=0), 2, axis=1)) / 2
        mean_rd = (np.roll(np.roll(self.img, -2, axis=0), 2, axis=1) +
                   np.roll(np.roll(self.img, 2, axis=0), -2, axis=1)) / 2

        # Apply correction with Numba kernel (only the compute-intensive part)
        if self.use_numba:
            corrected_img = _apply_correction_numba(
                self.img, detection_mask, min_grad,
                vertical_grad, horizontal_grad, left_diag_grad, right_diag_grad,
                mean_v, mean_h, mean_ld, mean_rd
            )
        else:
            corrected_img = _apply_correction_cpu(
                self.img, detection_mask, min_grad,
                vertical_grad, horizontal_grad, left_diag_grad, right_diag_grad,
                mean_v, mean_h, mean_ld, mean_rd
            )

        # Insert corrected pixels into image
        dpc_img = np.where(detection_mask, corrected_img, self.img)

        # Clip to sensor bit depth
        max_val = (1 << self.bpp) - 1
        dpc_img = np.clip(dpc_img, 0, max_val)

        # Cast back to suitable dtype
        dtype = np.uint16 if self.bpp <= 16 else np.uint32
        dpc_img = dpc_img.astype(dtype)

        if self.is_debug:
            self.logger.info(f"   - Number of corrected pixels = {np.count_nonzero(detection_mask)}")
            self.logger.info(f"   - Threshold = {self.threshold}")

        if return_mask:
            return dpc_img, detection_mask.astype(np.uint8)
        return dpc_img

    def dynamic_dpc_cpu(self):
        """
        CPU fallback implementation using original algorithm
        """
        # Import the original implementation
        from modules.dead_pixel_correction.dynamic_dpc import DynamicDPC
        dpc = DynamicDPC(self.img, self.sensor_info, {"dp_threshold": self.threshold, "is_debug": self.is_debug}, None)
        return dpc.dynamic_dpc()


@njit(parallel=True)
def _apply_correction_numba(img, detection_mask, min_grad,
                           vertical_grad, horizontal_grad, left_diag_grad, right_diag_grad,
                           mean_v, mean_h, mean_ld, mean_rd):
    """
    Numba-optimized correction application kernel
    Only the compute-intensive loop is optimized with Numba
    """
    h, w = img.shape
    corrected_img = np.zeros_like(img)

    for y in prange(h):
        for x in range(w):
            if detection_mask[y, x]:
                if min_grad[y, x] == vertical_grad[y, x]:
                    corrected_img[y, x] = mean_v[y, x]
                elif min_grad[y, x] == horizontal_grad[y, x]:
                    corrected_img[y, x] = mean_h[y, x]
                elif min_grad[y, x] == left_diag_grad[y, x]:
                    corrected_img[y, x] = mean_ld[y, x]
                else:
                    corrected_img[y, x] = mean_rd[y, x]
    return corrected_img


def _apply_correction_cpu(img, detection_mask, min_grad,
                         vertical_grad, horizontal_grad, left_diag_grad, right_diag_grad,
                         mean_v, mean_h, mean_ld, mean_rd):
    """
    CPU fallback for correction application
    """
    h, w = img.shape
    corrected_img = np.zeros_like(img)

    for y in range(h):
        for x in range(w):
            if detection_mask[y, x]:
                if min_grad[y, x] == vertical_grad[y, x]:
                    corrected_img[y, x] = mean_v[y, x]
                elif min_grad[y, x] == horizontal_grad[y, x]:
                    corrected_img[y, x] = mean_h[y, x]
                elif min_grad[y, x] == left_diag_grad[y, x]:
                    corrected_img[y, x] = mean_ld[y, x]
                else:
                    corrected_img[y, x] = mean_rd[y, x]
    return corrected_img

