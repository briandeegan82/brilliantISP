"""
File: dynamic_dpc_numba_optimized.py
Description: Optimized dead pixel correction using Numba for compute-intensive loops
Code / Paper  Reference: https://ieeexplore.ieee.org/document/9194921
Implementation inspired from: (OpenISP) https://github.com/cruxopen/openISP
Author: Brian Deegan (based in part on 10xEngineers / Infinite-ISP)
------------------------------------------------------------
"""

import logging
import numpy as np
from util.debug_utils import get_debug_logger
from util.isp_types import DeadPixelCorrectionConfig, PlatformConfig, RawBayerImage, SensorInfo
from modules.dead_pixel_correction.dynamic_dpc import _same_color_prep
from numba import njit, prange
import time
from typing import cast

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

    def __init__(
        self,
        img: RawBayerImage,
        sensor_info: SensorInfo,
        parm_dpc: DeadPixelCorrectionConfig,
        platform: PlatformConfig | None = None,
    ) -> None:
        self.img = img
        self.sensor_info = sensor_info
        self.bpp = self.sensor_info.get("bit_depth", 16)
        self.threshold = parm_dpc["dp_threshold"]
        self.is_debug = parm_dpc.get("is_debug", False)
        self.use_numba = NUMBA_AVAILABLE and self._should_use_numba()
        self.logger = get_debug_logger("DeadPixelCorrection", config=platform or {})

        if self.use_numba:
            self.logger.info("  Using Numba-optimized dead pixel correction (hybrid approach)")
        else:
            self.logger.info("  Using CPU dead pixel correction")

    def _should_use_numba(self) -> bool:
        """Determine if Numba optimization should be used based on image size."""
        if not NUMBA_AVAILABLE:
            return False

        # Use Numba for images larger than 500K pixels
        image_size = self.img.shape[0] * self.img.shape[1]
        return image_size > 500000  # 500K threshold

    def dynamic_dpc(self, return_mask: bool = False) -> RawBayerImage | tuple[RawBayerImage, np.ndarray]:
        height, width = self.sensor_info["height"], self.sensor_info["width"]

        # Shared same-color prep: signed float64, +/-2 neighbors, reflected
        # boundaries (replaces the rolling 3x3 neighborhood prep).
        f = np.asarray(self.img, dtype=np.float64)
        if f.shape != (height, width):
            raise ValueError(f"image shape {f.shape} does not match declared {height}x{width}")
        (
            detection_mask,
            vertical_grad, horizontal_grad, left_diag_grad, right_diag_grad,
            mean_v, mean_h, mean_ld, mean_rd,
        ) = _same_color_prep(f, float(self.threshold))

        grads = np.stack([vertical_grad, horizontal_grad, left_diag_grad, right_diag_grad], axis=2)
        min_grad = np.min(grads, axis=2)

        # Apply correction with Numba kernel (only the compute-intensive part)
        if self.use_numba:
            corrected_img = _apply_correction_numba(
                f,
                detection_mask,
                min_grad,
                vertical_grad,
                horizontal_grad,
                left_diag_grad,
                right_diag_grad,
                mean_v,
                mean_h,
                mean_ld,
                mean_rd,
            )
        else:
            corrected_img = _apply_correction_cpu(
                f,
                detection_mask,
                min_grad,
                vertical_grad,
                horizontal_grad,
                left_diag_grad,
                right_diag_grad,
                mean_v,
                mean_h,
                mean_ld,
                mean_rd,
            )

        # Insert corrected pixels into image
        dpc_img = np.where(detection_mask, corrected_img, f)

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
            return cast(RawBayerImage, dpc_img), detection_mask.astype(np.uint8)
        return cast(RawBayerImage, dpc_img)

    def dynamic_dpc_cpu(self) -> RawBayerImage:
        """
        CPU fallback implementation using original algorithm
        """
        # Import the original implementation
        from modules.dead_pixel_correction.dynamic_dpc import DynamicDPC

        if self.bpp > 16:
            cpu_input = cast(RawBayerImage, self.img.astype(np.uint32))
        else:
            cpu_input = cast(RawBayerImage, self.img.astype(np.uint16))
        dpc = DynamicDPC(
            cpu_input,
            self.sensor_info,
            {
                "is_enable": True,
                "dp_threshold": self.threshold,
                "is_debug": self.is_debug,
                "is_save": False,
            },
            None,
        )
        return dpc.dynamic_dpc()


@njit(parallel=True)
def _apply_correction_numba(
    img: np.ndarray,
    detection_mask: np.ndarray,
    min_grad: np.ndarray,
    vertical_grad: np.ndarray,
    horizontal_grad: np.ndarray,
    left_diag_grad: np.ndarray,
    right_diag_grad: np.ndarray,
    mean_v: np.ndarray,
    mean_h: np.ndarray,
    mean_ld: np.ndarray,
    mean_rd: np.ndarray,
) -> np.ndarray:
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


def _apply_correction_cpu(
    img: np.ndarray,
    detection_mask: np.ndarray,
    min_grad: np.ndarray,
    vertical_grad: np.ndarray,
    horizontal_grad: np.ndarray,
    left_diag_grad: np.ndarray,
    right_diag_grad: np.ndarray,
    mean_v: np.ndarray,
    mean_h: np.ndarray,
    mean_ld: np.ndarray,
    mean_rd: np.ndarray,
) -> np.ndarray:
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
