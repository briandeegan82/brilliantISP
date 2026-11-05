"""
File: dynamic_dpc.py
Description: Dynamic dead pixel correction implementation
Code / Paper  Reference: https://ieeexplore.ieee.org/document/9194921
Implementation inspired from: (OpenISP) https://github.com/cruxopen/openISP
Author: Brian Deegan (based in part on 10xEngineers / Infinite-ISP)
------------------------------------------------------------
"""

from typing import cast

import numpy as np
from util.debug_utils import get_debug_logger
from util.isp_types import DeadPixelCorrectionConfig, PlatformConfig, RawBayerImage, SensorInfo
from scipy.ndimage import maximum_filter, minimum_filter, correlate


class DynamicDPC:
    """
    Dynamic dead pixel correction scans the entire image for potential
    dead pixels and apply correction using the neighboring pixels.
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
        self.is_debug = parm_dpc["is_debug"]
        self.logger = get_debug_logger("DeadPixelCorrection", config=platform or {})

    def dynamic_dpc(self) -> RawBayerImage:
        """This function detects and corrects Dead pixels using numpy
        array opertaions."""

        height, width = self.sensor_info["height"], self.sensor_info["width"]
        f = np.asarray(self.img, dtype=np.float64)
        if f.shape != (height, width):
            raise ValueError(
                f"image shape {f.shape} does not match declared {height}x{width}"
            )

        dpc_img = np.empty((height, width), np.float64)

        # Get 3x3 neighbourhood of each pixel.
        # 5x5 matrix is defined as this window is extarcted from raw image.
        window = np.array(
            [
                [1, 0, 1, 0, 1],
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [1, 0, 1, 0, 1],
            ]
        )

        # The maximum and minimum filters automatically pad the input image internally,
        # eliminating the need for manual padding.
        max_value = maximum_filter(f, footprint=window, mode="mirror")
        min_value = minimum_filter(f, footprint=window, mode="mirror")

        # Condition 1: center_pixel needs to be corrected if it lies outside the
        # interval(min_value,max) of the 3x3 neighbourhood.
        # min_value < center_pixel < max_value--> no correction needed
        mask_cond1 = (np.where((min_value > f) | (f > max_value), True, False)).astype("int32")

        # Condition 2:
        # center_pixel is corrected only if the difference of center_pixel and every
        # neighboring pixel is greater than the specified threshold.
        # The two if conditions are used in combination to reduce False positives.

        # Kernels to compute the difference between center pixel and
        # each of the 8 neighbours.
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

        # convolve each kernel over image to compute differences
        # The correlate function automatically pads the input image internally,
        # eliminating the need for manual padding.

        diff_top_left = np.abs(correlate(f, ker_top_left, mode="mirror"))
        diff_top_mid = np.abs(correlate(f, ker_top_mid, mode="mirror"))
        diff_top_right = np.abs(correlate(f, ker_top_right, mode="mirror"))
        diff_mid_left = np.abs(correlate(f, ker_mid_left, mode="mirror"))
        diff_mid_right = np.abs(correlate(f, ker_mid_right, mode="mirror"))
        diff_bottom_left = np.abs(correlate(f, ker_bottom_left, mode="mirror"))
        diff_bottom_mid = np.abs(correlate(f, ker_bottom_mid, mode="mirror"))
        diff_bottom_right = np.abs(correlate(f, ker_bottom_right, mode="mirror"))

        del (
            ker_top_left,
            ker_top_mid,
            ker_top_right,
            ker_mid_left,
            ker_mid_right,
            ker_bottom_left,
            ker_bottom_mid,
            ker_bottom_right,
        )

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

        del (
            diff_top_left,
            diff_top_mid,
            diff_top_right,
            diff_mid_left,
            diff_mid_right,
            diff_bottom_left,
            diff_bottom_mid,
            diff_bottom_right,
        )

        # all gradients must be greater than the threshold for a pixel to be a DP.
        mask_cond2 = np.all(np.where(diff_array > self.threshold, True, False), axis=2)

        # mask with 1 for DPs and 0 for good pixels (dead pixels are the ones for which
        # both conditions are true)
        detection_mask = mask_cond1 * mask_cond2

        # Compute gradients
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

        # Convole the kernels to compute respective gradients
        vertical_grad = np.abs(correlate(f, ker_v, mode="mirror"))
        horizontal_grad = np.abs(correlate(f, ker_h, mode="mirror"))
        left_diagonal_grad = np.abs(correlate(f, ker_left_dia, mode="mirror"))
        right_diagonal_grad = np.abs(correlate(f, ker_right_dia, mode="mirror"))

        # Delete temporary variables
        del ker_v, ker_h, ker_left_dia, ker_right_dia

        # compute the direction of the minimum gradient
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

        # corrected value is computed as the mean of the neighbours
        # in the direction of min_value gadient.
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

        # Convolve kernels to compute mean for each direction
        mean_v = correlate(f, ker_mean_v, mode="mirror")
        mean_h = correlate(f, ker_mean_h, mode="mirror")
        mean_ldia = correlate(f, ker_mean_ldia, mode="mirror")
        mean_rdia = correlate(f, ker_mean_rdia, mode="mirror")

        del ker_mean_v, ker_mean_h, ker_mean_ldia, ker_mean_rdia

        # Corrected image has the corrected pixel values in place of a detected dead pixel
        # and 0 elsewhere
        corrected_img = np.zeros(f.shape)
        applied = np.zeros(f.shape, bool)

        # In most cases only one directional mean applies per pixel. Where two or more
        # directions share the same gradient magnitude, priority is vertical, then
        # horizontal, left diagonal, and finally the right diagonal; the explicit
        # selection below replaces the earlier zero-as-unset sentinel.
        for grad, mean in (
            (vertical_grad, mean_v),
            (horizontal_grad, mean_h),
            (left_diagonal_grad, mean_ldia),
            (right_diagonal_grad, mean_rdia),
        ):
            selected = detection_mask.astype(bool) & ~applied & (min_grad == grad)
            corrected_img = np.where(selected, mean, corrected_img)
            applied |= selected

        del mean_h, mean_v, mean_ldia, mean_rdia

        # Insert correct value of the detected dead pixels
        dpc_img = np.where(applied, corrected_img, f)

        # Clip/cast in float64 to the declared sensor bit depth
        max_val = (2**self.bpp) - 1
        self.img = np.clip(dpc_img, 0, max_val)
        self.img = self.img.astype(np.uint32 if self.bpp > 16 else np.uint16)

        if self.is_debug:
            self.logger.info(f"   - Number of corrected pixels = {np.count_nonzero(detection_mask)}")
            self.logger.info(f"   - Threshold = {self.threshold}")
        return cast(RawBayerImage, self.img)


_PAIR_OFFSETS = (
    ((-2, 0), (2, 0)),
    ((0, -2), (0, 2)),
    ((-2, -2), (2, 2)),
    ((-2, 2), (2, -2)),
)


def _same_color_prep(
    img: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Detection mask, four directional gradients and four directional means.

    Signed float64 arithmetic (unsigned differences cannot wrap); neighbors
    are same-CFA-color samples at +/-2 offsets on a reflect-padded copy
    (scipy ``mirror`` boundary semantics). Accumulates over reflected
    neighbor views rather than materializing a stacked copy.
    """
    f = np.asarray(img, dtype=np.float64)
    ext = np.pad(f, 2, mode="reflect")
    h, w = f.shape
    lo = np.full((h, w), np.inf)
    hi = np.full((h, w), -np.inf)
    mask = np.ones((h, w), bool)
    grads: list[np.ndarray] = []
    means: list[np.ndarray] = []
    for (a_dy, a_dx), (b_dy, b_dx) in _PAIR_OFFSETS:
        for dy, dx in ((a_dy, a_dx), (b_dy, b_dx)):
            view = ext[2 + dy:2 + dy + h, 2 + dx:2 + dx + w]
            np.minimum(lo, view, out=lo)
            np.maximum(hi, view, out=hi)
            mask &= np.abs(f - view) > threshold
        a = ext[2 + a_dy:2 + a_dy + h, 2 + a_dx:2 + a_dx + w]
        b = ext[2 + b_dy:2 + b_dy + h, 2 + b_dx:2 + b_dx + w]
        grads.append(np.abs(2.0 * f - a - b))
        means.append((a + b) / 2.0)
    mask &= (f < lo) | (f > hi)
    return (
        mask,
        grads[0], grads[1], grads[2], grads[3],
        means[0], means[1], means[2], means[3],
    )
