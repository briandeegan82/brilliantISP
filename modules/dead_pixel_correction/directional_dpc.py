"""
File: directional_dpc.py
Description: Directional Dead Pixel Correction using median-based detection
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import numpy as np
from scipy.ndimage import median_filter
import time

# Try to import Numba, fall back to CPU if not available
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not available, using CPU implementation")


class DirectionalDPC:
    """
    Directional Dead Pixel Correction:
    Uses median-based detection and directional interpolation
    """
    
    def __init__(self, img, threshold, bpp=12, is_debug=False):
        self.img = img.astype(np.float32)
        self.threshold = threshold
        self.bpp = bpp
        self.is_debug = is_debug
        self.use_numba = NUMBA_AVAILABLE and self._should_use_numba()
        
        if self.use_numba:
            print("  Using Numba-optimized directional DPC")
        else:
            print("  Using CPU directional DPC")

    def _should_use_numba(self):
        """Determine if Numba optimization should be used based on image size."""
        if not NUMBA_AVAILABLE:
            return False
        
        # Use Numba for images larger than 500K pixels
        image_size = self.img.shape[0] * self.img.shape[1]
        return image_size > 500000  # 500K threshold

    def correct(self, return_mask=False):
        # 1. Detection via median deviation
        local_median = median_filter(self.img, size=3, mode="mirror")
        detection_mask = np.abs(self.img - local_median) > self.threshold

        # 2. Apply directional correction
        if self.use_numba:
            corrected_img = _directional_correct_numba(self.img, detection_mask)
        else:
            corrected_img = _directional_correct_cpu(self.img, detection_mask)

        # 3. Merge corrected pixels
        dpc_img = np.where(detection_mask, corrected_img, self.img)

        # Clip to sensor range
        max_val = (1 << self.bpp) - 1
        dpc_img = np.clip(dpc_img, 0, max_val).astype(
            np.uint16 if self.bpp <= 16 else np.uint32
        )

        if self.is_debug:
            print("   - Corrected pixels:", np.count_nonzero(detection_mask))
            print("   - Threshold:", self.threshold)

        if return_mask:
            return dpc_img, detection_mask.astype(np.uint8)
        return dpc_img


@njit(parallel=True)
def _directional_correct_numba(img, mask):
    """
    Numba-optimized directional correction
    """
    h, w = img.shape
    out = np.zeros_like(img)

    for y in prange(1, h-1):
        for x in range(1, w-1):
            if mask[y, x]:
                # Directional candidates
                v = 0.5 * (img[y-1, x] + img[y+1, x])
                hdir = 0.5 * (img[y, x-1] + img[y, x+1])
                ld = 0.5 * (img[y-1, x-1] + img[y+1, x+1])
                rd = 0.5 * (img[y-1, x+1] + img[y+1, x-1])

                # Gradients (lower = smoother direction)
                gv = abs(img[y-1, x] - img[y+1, x])
                gh = abs(img[y, x-1] - img[y, x+1])
                gld = abs(img[y-1, x-1] - img[y+1, x+1])
                grd = abs(img[y-1, x+1] - img[y+1, x-1])

                # Choose direction with min gradient (manual min finding for Numba)
                min_idx = 0
                min_val = gv
                if gh < min_val:
                    min_val = gh
                    min_idx = 1
                if gld < min_val:
                    min_val = gld
                    min_idx = 2
                if grd < min_val:
                    min_val = grd
                    min_idx = 3
                
                # Apply correction based on min gradient direction
                if min_idx == 0:
                    out[y, x] = v
                elif min_idx == 1:
                    out[y, x] = hdir
                elif min_idx == 2:
                    out[y, x] = ld
                else:
                    out[y, x] = rd
            else:
                out[y, x] = img[y, x]
    return out


def _directional_correct_cpu(img, mask):
    """
    CPU fallback for directional correction
    """
    h, w = img.shape
    out = np.zeros_like(img)

    for y in range(1, h-1):
        for x in range(1, w-1):
            if mask[y, x]:
                # Directional candidates
                v = 0.5 * (img[y-1, x] + img[y+1, x])
                hdir = 0.5 * (img[y, x-1] + img[y, x+1])
                ld = 0.5 * (img[y-1, x-1] + img[y+1, x+1])
                rd = 0.5 * (img[y-1, x+1] + img[y+1, x-1])

                # Gradients (lower = smoother direction)
                gv = abs(img[y-1, x] - img[y+1, x])
                gh = abs(img[y, x-1] - img[y, x+1])
                gld = abs(img[y-1, x-1] - img[y+1, x+1])
                grd = abs(img[y-1, x+1] - img[y+1, x-1])

                # Choose direction with min gradient
                gvals = [gv, gh, gld, grd]
                cvals = [v, hdir, ld, rd]
                out[y, x] = cvals[np.argmin(gvals)]
            else:
                out[y, x] = img[y, x]
    return out
