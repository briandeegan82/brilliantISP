"""
Lens Shading Correction (LSC) - vignetting correction for raw Bayer data.

Corrects radial falloff (darker corners) using a per-channel polynomial gain model:
  gain(r) = 1 + k1*r² + k2*r⁴
where r is normalized distance from center (0 at center, 1 at corners).
Operates on raw Bayer before demosaic; preserves linearity.
"""
import time
import numpy as np
from util.debug_utils import get_debug_logger
from util.utils import save_output_array


def _build_radial_gain_map(height, width, k1, k2):
    """
    Build 2D gain map from radial polynomial: gain(r) = 1 + k1*r² + k2*r⁴.
    r = distance from center / max_distance, so r in [0, 1] at corners.
    """
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    max_dist = np.sqrt(cy**2 + cx**2)
    if max_dist < 1e-6:
        return np.ones((height, width), dtype=np.float32)

    y = np.arange(height, dtype=np.float32)[:, np.newaxis]
    x = np.arange(width, dtype=np.float32)[np.newaxis, :]
    r_sq = ((y - cy) ** 2 + (x - cx) ** 2) / (max_dist ** 2)
    gain = 1.0 + k1 * r_sq + k2 * (r_sq ** 2)
    return np.maximum(gain, 0.1).astype(np.float32)


class LensShadingCorrection:
    """
    Lens shading (vignetting) correction on raw Bayer.
    Uses radial polynomial gain per channel: gain(r) = 1 + k1*r² + k2*r⁴.
    """

    def __init__(self, img, platform, sensor_info, parm_lsc):
        self.img = np.asarray(img, dtype=np.float32)
        self.enable = parm_lsc.get("is_enable", True)
        self.is_save = parm_lsc.get("is_save", False)
        self.sensor_info = sensor_info
        self.parm_lsc = parm_lsc
        self.platform = platform
        self.logger = get_debug_logger("LensShadingCorrection", config=self.platform)

        # Radial model: gain(r) = 1 + k1*r² + k2*r⁴ per channel
        # Typical vignetting: k1, k2 > 0 (gain > 1 at edges to brighten corners)
        self.r_k1 = parm_lsc.get("r_k1", 0.0)
        self.r_k2 = parm_lsc.get("r_k2", 0.0)
        self.gr_k1 = parm_lsc.get("gr_k1", 0.0)
        self.gr_k2 = parm_lsc.get("gr_k2", 0.0)
        self.gb_k1 = parm_lsc.get("gb_k1", 0.0)
        self.gb_k2 = parm_lsc.get("gb_k2", 0.0)
        self.b_k1 = parm_lsc.get("b_k1", 0.0)
        self.b_k2 = parm_lsc.get("b_k2", 0.0)

        self.bayer = sensor_info.get("bayer_pattern", "rggb").lower()
        self.height, self.width = self.img.shape

        # Output max for clipping (HDR-aware)
        hdr_bits = sensor_info.get("hdr_bit_depth", 24)
        if self.img.max() <= 65535:
            hdr_bits = 16
        self.output_max = 2**hdr_bits - 1

    def _build_gain_maps(self):
        """Build per-channel gain maps."""
        h, w = self.height, self.width
        self._gain_r = _build_radial_gain_map(h, w, self.r_k1, self.r_k2)
        self._gain_gr = _build_radial_gain_map(h, w, self.gr_k1, self.gr_k2)
        self._gain_gb = _build_radial_gain_map(h, w, self.gb_k1, self.gb_k2)
        self._gain_b = _build_radial_gain_map(h, w, self.b_k1, self.b_k2)

    def _apply_lsc(self):
        """Apply per-channel gain to Bayer."""
        self._build_gain_maps()
        out = self.img.copy()

        if self.bayer == "rggb":
            out[0::2, 0::2] *= self._gain_r[0::2, 0::2]
            out[0::2, 1::2] *= self._gain_gr[0::2, 1::2]
            out[1::2, 0::2] *= self._gain_gb[1::2, 0::2]
            out[1::2, 1::2] *= self._gain_b[1::2, 1::2]
        elif self.bayer == "bggr":
            out[0::2, 0::2] *= self._gain_b[0::2, 0::2]
            out[0::2, 1::2] *= self._gain_gb[0::2, 1::2]
            out[1::2, 0::2] *= self._gain_gr[1::2, 0::2]
            out[1::2, 1::2] *= self._gain_r[1::2, 1::2]
        elif self.bayer == "grbg":
            out[0::2, 0::2] *= self._gain_gr[0::2, 0::2]
            out[0::2, 1::2] *= self._gain_r[0::2, 1::2]
            out[1::2, 0::2] *= self._gain_b[1::2, 0::2]
            out[1::2, 1::2] *= self._gain_gb[1::2, 1::2]
        elif self.bayer == "gbrg":
            out[0::2, 0::2] *= self._gain_gb[0::2, 0::2]
            out[0::2, 1::2] *= self._gain_b[0::2, 1::2]
            out[1::2, 0::2] *= self._gain_r[1::2, 0::2]
            out[1::2, 1::2] *= self._gain_gr[1::2, 1::2]
        else:
            raise ValueError(f"Unsupported bayer_pattern: {self.bayer}")

        out = np.clip(out, 0, self.output_max)
        return out

    def execute(self):
        """Execute lens shading correction."""
        self.logger.info(f"Lens Shading Correction = {self.enable}")

        if not self.enable:
            return self.img.astype(
                np.uint32 if self.output_max > 65535 else np.uint16,
                copy=False,
            )

        start = time.time()
        out = self._apply_lsc()

        if self.output_max > 65535:
            result = out.astype(np.uint32)
        else:
            result = out.astype(np.uint16)

        elapsed = time.time() - start
        self.logger.info(f"  LSC execution time: {elapsed:.3f}s")

        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                result,
                "Out_lsc_",
                self.platform,
                self.sensor_info.get("bit_depth", 12),
                self.sensor_info.get("bayer_pattern", "RGGB"),
            )

        return result
