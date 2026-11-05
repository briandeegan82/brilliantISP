"""
File: sharpen.py
Description: Simple unsharp masking with frequency and strength control.
Code / Paper  Reference:
Author: Brian Deegan (based in part on 10xEngineers / Infinite-ISP)
"""

import math
import numpy as np
from scipy import ndimage


class UnsharpMasking:
    """
    Implements Unsharp Masking Algorithm
    """

    def __init__(self, img: np.ndarray, sharpen_sigma: float, sharpen_strength: float) -> None:
        self.img = img
        self.sharpen_sigma = sharpen_sigma
        self.sharpen_strength = sharpen_strength

    def apply_sharpen(self) -> np.ndarray:
        """
        Applying sharpening to the input image
        """
        luma = np.float32(self.img[:, :, 0])

        # if self.img.dtype == "float32":
        #     luma = np.round(255 * luma).astype(np.uint8)

        # Filter the luma component of the image with a Gaussian LPF
        # Smoothing magnitude can be controlled with the sharpen_sigma parameter
        smoothened = ndimage.gaussian_filter(luma, self.sharpen_sigma)
        # Sharpen the image with upsharp mask
        # Strength is tuneable with the sharpen_strength parameter
        sharpened = luma + ((luma - smoothened) * self.sharpen_strength)

        if self.img.dtype == "float32":
            # sharpened = np.float32((sharpened) / 255.0)
            # out = sharpened.astype("float32")
            self.img[:, :, 0] = np.clip(sharpened, 0, 1)
        else:
            self.img[:, :, 0] = np.uint8(np.clip(sharpened, 0, 255))
        return self.img


def apply_luma(
    luma: np.ndarray,
    *,
    scale: float,
    sigma: float,
    strength: float,
) -> np.ndarray:
    """Native plane unsharp mask: clip(y + strength*(y - smooth), 0, 1).

    Same SciPy Gaussian (``gaussian_filter``, ``mode='reflect'``,
    ``truncate=4``); normalized math, checked float32 restore.
    """
    from scipy.ndimage import gaussian_filter
    from util.luma import normalize_plane, restore_plane

    for name, value in (("sigma", sigma), ("strength", strength)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{name} must be a finite nonnegative number, got {value!r}")
    y = normalize_plane(luma, scale=scale)
    smooth = gaussian_filter(y, float(sigma), mode="reflect", truncate=4)
    out = np.clip(y + float(strength) * (y - smooth), 0.0, 1.0)
    return restore_plane(out, scale=scale)
