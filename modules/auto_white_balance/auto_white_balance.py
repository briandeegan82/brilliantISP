from util.debug_utils import get_debug_logger
"""
File: auto_white_balance.py
Description: 3A - AWB Runs the AWB algorithm based on selection from config file
Code / Paper  Reference: https://www.sciencedirect.com/science/article/abs/pii/0016003280900587
                         https://library.imaging.org/admin/apis/public/api/ist/website/downloadArticle/cic/12/1/art00008
                         https://opg.optica.org/josaa/viewmedia.cfm?uri=josaa-31-5-1049&seq=0
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import numpy as np
from modules.auto_white_balance.gray_world import GrayWorld as GW
from modules.auto_white_balance.norm_gray_world import NormGrayWorld as NGW
from modules.auto_white_balance.pca import PCAIlluminEstimation as PCA


class AutoWhiteBalance:
    """
    Auto White Balance Module
    """

    def __init__(self, raw, sensor_info, parm_awb, parm_wbc):

        self.raw = raw

        self.sensor_info = sensor_info
        self.parm_awb = parm_awb
        self.enable = parm_awb["is_enable"]
        self.hdr_bit_depth = sensor_info["hdr_bit_depth"]
        self.is_debug = parm_awb["is_debug"]
        self.underexposed_percentage = parm_awb["underexposed_percentage"]
        self.overexposed_percentage = parm_awb["overexposed_percentage"]
        self.flatten_img = None
        self.bayer = self.sensor_info["bayer_pattern"]
        # self.img = img
        self.algorithm = parm_awb["algorithm"]
        self.parm_wbc = parm_wbc
        
        # Initialize debug logger with config from parm_awb
        debug_config = {
            'debug_enabled': parm_awb.get('is_debug', False),
            'debug_log_level': 'INFO',
            'debug_log_file': None
        }
        self.logger = get_debug_logger("AutoWhiteBalance", config=debug_config)

    def determine_white_balance_gain(self):
        """
        Determine white balance gains calculated using AWB Algorithms to Raw Image
        """

        max_pixel_value = 2**self.hdr_bit_depth
        approx_percentage = max_pixel_value / 100
        # Removed overexposed and underexposed pixels for wb gain calculation
        overexposed_limit = (
            max_pixel_value - (self.overexposed_percentage) * approx_percentage
        )
        underexposed_limit = (self.underexposed_percentage) * approx_percentage

        if self.is_debug:
            self.logger.info(f"   - AWB - Underexposed Pixel Limit = {underexposed_limit}")
            self.logger.info(f"   - AWB - Overexposed Pixel Limit  = {overexposed_limit}")

        if self.bayer == "rggb":

            r_channel = self.raw[0::2, 0::2]
            gr_channel = self.raw[0::2, 1::2]
            gb_channel = self.raw[1::2, 0::2]
            b_channel = self.raw[1::2, 1::2]

        elif self.bayer == "bggr":
            b_channel = self.raw[0::2, 0::2]
            gb_channel = self.raw[0::2, 1::2]
            gr_channel = self.raw[1::2, 0::2]
            r_channel = self.raw[1::2, 1::2]

        elif self.bayer == "grbg":
            gr_channel = self.raw[0::2, 0::2]
            r_channel = self.raw[0::2, 1::2]
            b_channel = self.raw[1::2, 0::2]
            gb_channel = self.raw[1::2, 1::2]

        elif self.bayer == "gbrg":
            gb_channel = self.raw[0::2, 0::2]
            b_channel = self.raw[0::2, 1::2]
            r_channel = self.raw[1::2, 0::2]
            gr_channel = self.raw[1::2, 1::2]

        g_channel = (gr_channel + gb_channel) / 2
        bayer_channels = np.dstack((r_channel, g_channel, b_channel))
        # print(bayer_channels.shape)

        bad_pixels = np.sum(
            np.where(
                (bayer_channels < underexposed_limit)
                | (bayer_channels > overexposed_limit),
                1,
                0,
            ),
            axis=2,
        )
        self.flatten_img = bayer_channels[bad_pixels == 0]
        # print(self.flatten_raw.shape)

        if self.algorithm == "norm_2":
            rgain, bgain = self.apply_norm_gray_world()
        elif self.algorithm == "pca":
            rgain, bgain = self.apply_pca_illuminant_estimation()
        else:
            rgain, bgain = self.apply_gray_world()

        # Check if r_gain and b_gain go out of bound
        rgain = 1 if rgain <= 1 else rgain
        bgain = 1 if bgain <= 1 else bgain

        if self.is_debug:
            self.logger.info("   - AWB Actual Gains: ")
            self.logger.info(f"   - AWB - RGain = {rgain}")
            self.logger.info(f"   - AWB - Bgain = {bgain}")

        return rgain, bgain

    def apply_gray_world(self):
        """

        Gray World White Balance:
        Gray world algorithm calculates white balance (G/R and G/B)
        by average values of RGB channels
        """

        gwa = GW(self.flatten_img)
        return gwa.calculate_gains()

    def apply_norm_gray_world(self):
        """
        Norm 2 Gray World White Balance:

        Gray world algorithm calculates white balance (G/R and G/B)
        by average values of RGB channels. Average values for each channel
        are calculated by norm-2
        """

        ngw = NGW(self.flatten_img)
        return ngw.calculate_gains()

    def apply_pca_illuminant_estimation(self):
        """
        PCA Illuminant Estimation:

        This algorithm gets illuminant estimation directly from the color distribution
        The method that chooses bright and dark pixels using a projection distance in
        the color distribution and then applies principal component analysis to estimate
        the illumination direction
        """
        pixel_percentage = self.parm_awb["percentage"]
        pca = PCA(self.flatten_img, pixel_percentage)
        return pca.calculate_gains()

    def execute(self):
        """
        Execute Auto White Balance Module
        """
        self.logger.info(f"Auto White balancing = {self.enable}")

        # This module is enabled only when white balance 'enable' and 'auto' parameter both
        # are true.
        if self.enable is True:
            start = time.time()
            rgain, bgain = self.determine_white_balance_gain()
            self.logger.info(f"  Execution time: {time.time() - start:.3f}s")
            return np.array([rgain, bgain])
        else:
            rgain, bgain = self.parm_wbc["r_gain"], self.parm_wbc["b_gain"]
            self.logger.info(f"  Using default gains: {rgain}, {bgain}")
        return None
