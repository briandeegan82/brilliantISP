"""
File: auto_exposure.py
Description: 3A-AE Runs the Auto exposure algorithm in a loop
Code / Paper  Reference: https://www.atlantis-press.com/article/25875811.pdf
                         http://tomlr.free.fr/Math%E9matiques/Math%20Complete/Probability%20and%20statistics/CRC%20-%20standard%20probability%20and%20Statistics%20tables%20and%20formulae%20-%20DANIEL%20ZWILLINGER.pdf
Author: Brian Deegan (based in part on 10xEngineers / Infinite-ISP)
------------------------------------------------------------
"""

import time
import numpy as np
from util.debug_utils import get_debug_logger
from util.isp_types import AutoExposureConfig, SensorInfo


class AutoExposure:
    """
    Auto Exposure Module
    """

    def __init__(self, img: np.ndarray, sensor_info: SensorInfo, parm_ae: AutoExposureConfig) -> None:
        self.img = img
        self.enable = parm_ae["is_enable"]
        self.is_debug = parm_ae["is_debug"]
        self.center_illuminance = parm_ae["center_illuminance"]
        self.histogram_skewness_range = parm_ae["histogram_skewness"]
        self.sensor_info = sensor_info
        self.param_ae = parm_ae
        # AE runs on post-CCM RGB (see brilliant_isp). Use pipeline RGB bit depth for
        # downsampling — not hdr_bit_depth (scene/tone-map headroom), which broke 16-bit RGB.
        self._ae_rgb_bits = int(
            sensor_info.get(
                "pipeline_rgb_bit_depth",
                sensor_info.get("bit_depth", 16),
            )
        )
        # Initialize debug logger from parm_ae
        debug_config = {
            "debug_enabled": parm_ae.get("is_debug", False),
            "debug_log_level": "INFO",
            "debug_log_file": None,
        }
        self.logger = get_debug_logger("AutoExposure", config=debug_config)
        # Filled in determine_exposure() for direct gain mapping
        self.last_meter_average: float | None = None
        self.last_max_channel_meter: int = 0

    def _meter_shift(self) -> int:
        """Right-shift count to bring CCM RGB into an ~8-bit band for AE stats."""
        return max(0, self._ae_rgb_bits - 8)

    def _max_channel_after_shift(self) -> int:
        s = self._meter_shift()
        full = (2**self._ae_rgb_bits) - 1
        return full >> s if s else full

    def _skewness_center(self, max_ch: int) -> float:
        """
        center_illuminance: if in (0, 1], treat as fraction of max grey after meter
        scaling (legacy HDR YAML). Otherwise absolute in same units as metered grey (0..max_ch).
        """
        c = float(self.center_illuminance)
        if 0.0 <= c <= 1.0:
            return c * float(max_ch)
        return c

    def get_exposure_feedback(self) -> int:
        """
        Downscale post-CCM RGB to an ~8-bit range, then compute skewness feedback.
        """
        shift = self._meter_shift()
        self.img = self.img.astype(np.uint32, copy=False) >> shift

        # calculate the exposure metric
        return self.determine_exposure()

    def determine_exposure(self) -> int:
        """
        Image Exposure Estimation using Skewness Luminance of Histograms
        """

        # For Luminance Histograms, Image is first converted into greyscale image
        # Function also returns average luminance of image which is used as AE-Stat
        grey_img, avg_lum = self.get_greyscale_image(self.img)
        self.logger.info(f"Average luminance is = {avg_lum}")

        max_ch = self._max_channel_after_shift()
        self.last_meter_average = float(avg_lum)
        self.last_max_channel_meter = max_ch
        center = self._skewness_center(max_ch)

        # Histogram skewness Calculation for AE Stats
        skewness = self.get_luminance_histogram_skewness(grey_img, center)

        # get the ranges
        upper_limit = self.histogram_skewness_range
        lower_limit = -1 * upper_limit

        if self.is_debug:
            self.logger.info(
                f"AE - meter RGB bits={self._ae_rgb_bits}, "
                f"shift={self._meter_shift()}, max_ch={max_ch}, skewness_center={center:.3f}"
            )
            self.logger.info(f"AE - Histogram Skewness Range = {upper_limit}")

        # see if skewness is within range
        if skewness < lower_limit:
            return -1
        elif skewness > upper_limit:
            return 1
        else:
            return 0

    def get_greyscale_image(self, img: np.ndarray) -> tuple[np.ndarray, np.floating]:
        """
        Conversion of an Image into Greyscale Image
        """
        max_ch = float(self._max_channel_after_shift())
        # BT.709 luma weights (matches boltISP AGC luma computation)
        grey = np.dot(img[..., :3].astype(np.float64), [0.2126, 0.7152, 0.0722])
        grey_img = np.clip(grey, 0.0, max_ch)
        return grey_img, np.average(grey_img, axis=(0, 1))

    def get_luminance_histogram_skewness(self, img: np.ndarray, center: float) -> np.floating:
        """
        Skewness Calculation in reference to:
        Zwillinger, D. and Kokoska, S. (2000). CRC Standard Probability and Statistics
        Tables and Formulae. Chapman & Hall: New York. 2000. Section 2.2.24.1
        """

        # First subtract central luminance to calculate skewness around it
        img = img.astype(np.float64) - center

        # The sample skewness is computed as the Fisher-Pearson coefficient of
        # skewness, i.e. (m_3 / m_2**(3/2)) * g_1
        # where m_2 is 2nd moment (variance) and m_3 is third moment skewness

        img_size = img.size
        m_2 = np.sum(np.power(img, 2)) / img_size
        m_3 = np.sum(np.power(img, 3)) / img_size

        g_1 = np.sqrt(img_size * (img_size - 1)) / (img_size - 2)
        skewness = np.nan_to_num((m_3 / abs(m_2) ** (3 / 2)) * g_1)

        if self.is_debug:
            self.logger.info(f"AE - Histogram Skewness = {skewness}")

        return skewness

    def _absolute_target_luminance(self) -> float:
        """Target avg luma in the same units as last_meter_average (post meter downshift)."""
        max_ch = float(self.last_max_channel_meter)
        raw = self.param_ae.get("target_luminance")
        if raw is None:
            return 0.5 * max_ch
        c = float(raw)
        if 0.0 <= c <= 1.0:
            return c * max_ch
        return c

    def suggest_direct_gain_index(self, current_gain_index: int, gain_array: list[float]) -> int:
        """
        Pick gain_array index closest to the multiplier that would move
        last_meter_average toward target_luminance in one shot (first-order).
        """
        if self.last_meter_average is None or not gain_array:
            return current_gain_index
        target = self._absolute_target_luminance()
        ratio = target / max(self.last_meter_average, 1e-6)
        gi = int(np.clip(current_gain_index, 0, len(gain_array) - 1))
        cur = float(gain_array[gi])
        gmin = float(min(gain_array))
        gmax = float(max(gain_array))
        desired = float(np.clip(cur * ratio, gmin, gmax))
        arr = np.asarray(gain_array, dtype=np.float64)
        idx = int(np.argmin(np.abs(arr - desired)))
        if self.is_debug:
            self.logger.info(
                f"AE direct: target_luma={target:.3f}, avg={self.last_meter_average:.3f}, "
                f"ratio={ratio:.4f}, gain_idx {current_gain_index}→{idx} "
                f"({cur:.4f}→{float(gain_array[idx]):.4f})"
            )
        return idx

    def execute(self) -> int | None:
        """
        Execute Auto Exposure
        """
        self.logger.info(f"Auto Exposure = {self.enable}")

        if self.enable is False:
            self.last_meter_average = None
            return None
        else:
            start = time.time()
            ae_feedback = self.get_exposure_feedback()
            execution_time = time.time() - start
            self.logger.info(f"Execution time: {execution_time:.3f}s")
            return ae_feedback
