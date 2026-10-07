from util.debug_utils import get_debug_logger

"""
File: color_space_conversion.py
Description: Converts RGB to YUV or YCbCr
Code / Paper  Reference: https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.709_conversion
                         https://www.itu.int/rec/R-REC-BT.601/
                         https://www.itu.int/rec/R-REC-BT.709-6-201506-I/en
                         https://learn.microsoft.com/en-us/windows/win32/medfound/recommended-
                         8-bit-yuv-formats-for-video-rendering
                         https://web.archive.org/web/20180423091842/http://www.equasys.de/
                         colorconversion.html
                         Author: Brian Deegan (based in part on 10xEngineers / Infinite-ISP)
------------------------------------------------------------------------------
"""

import time
import numpy as np

from util.isp_types import ColorSaturationEnhancementConfig, ColorSpaceConversionConfig, PlatformConfig, SensorInfo, UInt8Image
from util.utils import save_output_array_yuv


class ColorSpaceConversion:
    """
    Color Space Conversion
    """

    def __init__(
        self,
        img: UInt8Image,
        platform: PlatformConfig,
        sensor_info: SensorInfo,
        parm_csc: ColorSpaceConversionConfig,
        parm_cse: ColorSaturationEnhancementConfig,
    ) -> None:
        self.img = img.copy()
        self.is_save = parm_csc["is_save"]
        self.platform = platform
        self.sensor_info = sensor_info
        self.parm_csc = parm_csc
        self.bit_depth = sensor_info["output_bit_depth"]
        self.conv_std = self.parm_csc["conv_standard"]
        self.rgb2yuv_mat = None
        self.yuv_img = None
        self.parm_cse = parm_cse
        # Initialize debug logger
        self.logger = get_debug_logger("ColorSpaceConversion", config=self.platform)

    def _apply_luma_preserving_saturation(self, rgb_2d: np.ndarray) -> np.ndarray:
        """Luma-preserving saturation in linear RGB space.

        Implements c' = L + (c - L) * S, where L is BT.709 luma.
        This matches boltISP's CCM-stage saturation formula and is
        mathematically equivalent to chroma scaling without hue shift.

        Parameters
        ----------
        rgb_2d : (3, N) float64 array of R/G/B channel rows (0–255 range).

        Returns
        -------
        (3, N) float64 array with saturation applied.
        """
        gain = float(self.parm_cse["saturation_gain"])
        r, g, b = rgb_2d[0], rgb_2d[1], rgb_2d[2]
        luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
        out = np.empty_like(rgb_2d)
        out[0] = luma + (r - luma) * gain
        out[1] = luma + (g - luma) * gain
        out[2] = luma + (b - luma) * gain
        return np.clip(out, 0.0, 255.0)

    def rgb_to_yuv_8bit(self) -> UInt8Image:
        """
        RGB-to-YUV Colorspace conversion 8bit
        """

        if self.conv_std == 1:
            # for BT. 709
            self.rgb2yuv_mat = np.array([[47, 157, 16], [-26, -86, 112], [112, -102, -10]])
        else:

            # for BT.601/407
            # conversion metrix with 8bit integer co-efficients - m=8
            self.rgb2yuv_mat = np.array([[77, 150, 29], [131, -110, -21], [-44, -87, 138]])

        # make nx3 2d matrix of image
        mat_2d = self.img.reshape((self.img.shape[0] * self.img.shape[1], 3))

        # Color saturation: apply luma-preserving blend before RGB→YUV.
        # Formula: c' = L + (c - L) * S  (matches boltISP's CCM-stage saturation).
        # This replaces the old U/V scaling approach which is not equivalent.
        mat_2d_float = mat_2d.astype(np.float64)
        if self.parm_cse["is_enable"]:
            mat_2d_float = self._apply_luma_preserving_saturation(mat_2d_float.transpose()).transpose()

        # convert to 3xn for matrix multiplication
        mat2d_t = mat_2d_float.transpose()

        # convert to YUV
        yuv_2d = np.matmul(self.rgb2yuv_mat, mat2d_t)

        # convert image with its provided bit_depth
        yuv_2d = np.float64(yuv_2d) / (2**8)
        yuv_2d = np.where(yuv_2d >= 0, np.floor(yuv_2d + 0.5), np.ceil(yuv_2d - 0.5))

        # black-level/DC offset added to YUV values
        yuv_2d[0, :] = 2 ** (self.bit_depth / 2) + yuv_2d[0, :]
        yuv_2d[1, :] = 2 ** (self.bit_depth - 1) + yuv_2d[1, :]
        yuv_2d[2, :] = 2 ** (self.bit_depth - 1) + yuv_2d[2, :]

        # reshape the image back
        yuv2d_t = yuv_2d.transpose()

        yuv2d_t = np.clip(yuv2d_t, 0, (2**self.bit_depth) - 1)

        # Modules after CSC need 8-bit YUV so converting it into 8-bit after Normalizing.
        yuv2d_t = yuv2d_t / (2 ** (self.bit_depth - 8))
        yuv2d_t = np.where(yuv2d_t >= 0, np.floor(yuv2d_t + 0.5), np.ceil(yuv2d_t - 0.5))

        yuv2d_t = np.clip(yuv2d_t, 0, 255)

        self.img = yuv2d_t.reshape(self.img.shape).astype(np.uint8)
        return self.img

    def save(self) -> None:
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array_yuv(
                self.platform["in_file"],
                self.img,
                "Out_color_space_conversion_",
                self.platform,
                self.conv_std,
            )

    def execute(self) -> UInt8Image:
        """
        Execute Color Space Conversion
        """
        self.logger.info("Color Space Conversion (default) = True")

        start = time.time()
        csc_out = self.rgb_to_yuv_8bit()
        self.logger.info(f"  Execution time: {time.time() - start:.3f}s")
        self.img = csc_out
        self.save()
        return self.img
