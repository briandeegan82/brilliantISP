"""
File: crop.py
Description: Crops the bayer image keeping cfa pattern intact
Code / Paper  Reference:
Author: Brian Deegan (based in part on 10xEngineers / Infinite-ISP)
------------------------------------------------------------
"""

import time
import re
from typing import cast

import numpy as np
from util.isp_types import (
    BayerPattern,
    CropConfig,
    PlatformConfig,
    RawBayerImage,
    SensorInfo,
)
from util.raw_decode import shift_bayer_pattern
from util.utils import save_output_array
from util.debug_utils import get_debug_logger


class Crop:
    """
    Parameters:
    ----------
    img: 2D array
    new_size: 2-tuple with required height and width.
    Return:
    ------
    cropped_img: Generated coloured image of required size with same dtype
    as the input image.
    """

    def __init__(
        self,
        img: RawBayerImage,
        platform: PlatformConfig,
        sensor_info: SensorInfo,
        parm_cro: CropConfig,
    ) -> None:
        self.img = img
        self.sensor_info = sensor_info
        self.old_size = (img.shape[0], img.shape[1])
        self.new_size = (parm_cro["new_height"], parm_cro["new_width"])
        self.x_start = parm_cro.get("crop_x_start", 0)
        self.y_start = parm_cro.get("crop_y_start", 0)
        self.enable = parm_cro["is_enable"]
        self.is_debug = parm_cro["is_debug"]
        self.is_save = parm_cro["is_save"]

        self.platform = platform
        self.logger = get_debug_logger("Crop", config=self.platform)

    def update_sensor_info(self, dictionary: SensorInfo) -> None:
        """Update sensor geometry/CFA after a successful exact crop."""
        if dictionary["height"] != self.new_size[0] or dictionary["width"] != self.new_size[1]:
            dictionary["orig_size"] = str(self.old_size[::-1])
        dictionary["height"] = self.new_size[0]
        dictionary["width"] = self.new_size[1]
        dictionary["bayer_pattern"] = cast(
            BayerPattern,
            shift_bayer_pattern(
                str(dictionary.get("bayer_pattern", "rggb")),
                int(self.x_start),
                int(self.y_start),
            ),
        )

    def crop(self, img: RawBayerImage, rows_to_crop: int = 0, cols_to_crop: int = 0) -> RawBayerImage:
        """
        Crop 2D array.
        Parameter:
        ---------
        img: image (2D array) to be cropped.
        rows_to_crop: Number of rows to crop. If it is an even integer,
                      equal number of rows are cropped from either side of the image.
                      Otherwise the image is cropped from the extreme right/bottom.
        cols_to_crop: Number of columns to crop. Works exactly as rows_to_crop.
        Output: cropped image
        """

        if rows_to_crop or cols_to_crop:
            if rows_to_crop % 4 == 0 and cols_to_crop % 4 == 0:
                img = img[
                    rows_to_crop // 2 : -rows_to_crop // 2,
                    cols_to_crop // 2 : -cols_to_crop // 2,
                ]
            else:
                self.logger.warning(
                    "   - Input/Output heights are not compatible. " "Bayer pattern will be disturbed if cropped!"
                )
        return img

    def _validate(self) -> None:
        def _integer(name: str, value: object) -> int:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"crop {name} must be an integer, got {value!r}")
            return int(value)

        h, w = _integer("new_height", self.new_size[0]), _integer("new_width", self.new_size[1])
        x0, y0 = _integer("crop_x_start", self.x_start), _integer("crop_y_start", self.y_start)
        if h <= 0 or w <= 0:
            raise ValueError(f"crop output size must be positive, got {w}x{h}")
        if x0 < 0 or y0 < 0:
            raise ValueError(f"crop origins must be non-negative, got ({x0}, {y0})")
        self.new_size = (h, w)
        self.x_start, self.y_start = x0, y0

        # Validate crop region
        img_h, img_w = self.img.shape[0], self.img.shape[1]
        if y0 + h > img_h or x0 + w > img_w:
            raise ValueError(
                f"crop region ({x0}, {y0}) {w}x{h} exceeds input image "
                f"{img_w}x{img_h}; no implicit adjustment is applied"
            )

    def apply_cropping(self) -> RawBayerImage:
        """Crop Image using the exact requested origins and dimensions."""
        self._validate()

        if self.old_size == self.new_size and self.x_start == 0 and self.y_start == 0:
            self.logger.info("   - Output size is the same as input size.")
            return self.img

        # Extract the region from (y_start, x_start) with size (new_height, new_width)
        cropped_img = self.img[self.y_start : self.y_start + self.new_size[0], self.x_start : self.x_start + self.new_size[1]]

        if self.is_debug:
            self.logger.info(
                f"   - Crop region: x={self.x_start}, y={self.y_start}, width={self.new_size[1]}, height={self.new_size[0]}"
            )
            self.logger.info(f"   - Shape of cropped image = {cropped_img.shape}")
        return cropped_img

    def save(self, filename_tag: str) -> None:
        """
        Function to save module output
        """
        # update size of array in filename
        self.platform["in_file"] = re.sub(
            r"\d+x\d+",
            f"{self.img.shape[1]}x{self.img.shape[0]}",
            self.platform["in_file"],
        )
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                filename_tag,
                self.platform,
                self.sensor_info["bit_depth"],
                self.sensor_info["bayer_pattern"],
            )

    def execute(self) -> RawBayerImage:
        """Execute cropping if enabled."""
        self.logger.info(f"Crop = {self.enable}")

        # Save the input of crop module
        self.save("Inpipeline_crop_")

        # crop image if enabled
        if self.enable:
            start = time.time()
            cropped_img = self.apply_cropping()
            self.logger.info(f"  Execution time: {time.time() - start:.3f}s")
            self.img = cropped_img
            # Ensure Bayer pattern alignment
            self.update_sensor_info(self.sensor_info)

        # save the output of crop module
        self.save("Out_crop_")
        return self.img


##########################################################################
