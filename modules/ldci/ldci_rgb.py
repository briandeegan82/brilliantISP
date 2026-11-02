"""
Post-gamma LDCI on 8-bit RGB.

Mirrors boltISP's ``applyLdciRgb888``:
  1. Compute BT.709 luma from gamma-encoded 8-bit RGB.
  2. Apply CLAHE to the luma channel.
  3. Scale R/G/B by the per-pixel luma ratio (luma_out / luma_in).

This differs from the YUV-domain LDCI (which runs before gamma on linear data)
and produces results closer to boltISP's output when ``ldci.post_gamma: true``
is set in the config.
"""

from __future__ import annotations

import numpy as np

from modules.ldci.clahe import CLAHE
from util.isp_types import LDCIConfig, PlatformConfig, SensorInfo

# BT.709 luma coefficients (same as boltISP's applyLdciRgb888)
_BT709_R = 0.2126
_BT709_G = 0.7152
_BT709_B = 0.0722


def apply_ldci_rgb8(
    rgb: np.ndarray,
    platform: PlatformConfig,
    sensor_info: SensorInfo,
    parm_ldci: LDCIConfig,
) -> np.ndarray:
    """Apply CLAHE-based local contrast enhancement to 8-bit gamma-encoded RGB.

    Parameters
    ----------
    rgb:
        uint8 array of shape (H, W, 3) with R/G/B channels in the last axis.
    platform, sensor_info, parm_ldci:
        Same config dicts passed throughout the brilliantISP pipeline.

    Returns
    -------
    uint8 (H, W, 3) array with enhanced contrast.
    """
    if not parm_ldci.get("is_enable", False):
        return rgb

    rgb = rgb.astype(np.uint8)
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)

    # BT.709 luma (float, 0–255 range)
    luma_in = _BT709_R * r + _BT709_G * g + _BT709_B * b

    # Wrap the luma in a "fake" YUV array (U/V unused) so the existing CLAHE
    # class can process it without modification.
    h, w = rgb.shape[:2]
    fake_yuv = np.zeros((h, w, 3), dtype=np.uint8)
    fake_yuv[:, :, 0] = np.clip(np.round(luma_in), 0, 255).astype(np.uint8)

    clahe = CLAHE(fake_yuv, platform, sensor_info, parm_ldci)
    enhanced = clahe.apply_clahe()  # returns (H, W, 3) with modified channel 0

    luma_out = enhanced[:, :, 0].astype(np.float32)

    # Per-pixel luma-ratio scale; avoid division by zero (threshold matches boltISP)
    scale = np.where(luma_in > 1e-3, luma_out / np.maximum(luma_in, 1e-3), 0.0)

    # Clamp scale to avoid over-brightening
    scale = np.clip(scale, 0.0, 4.0)

    out = np.stack(
        [
            np.clip(np.round(r * scale), 0, 255),
            np.clip(np.round(g * scale), 0, 255),
            np.clip(np.round(b * scale), 0, 255),
        ],
        axis=-1,
    ).astype(np.uint8)

    return out
