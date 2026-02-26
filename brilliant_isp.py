"""
File: isp_pipeline.py
Description: Executes the complete pipeline
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
from pathlib import Path
import numpy as np
import yaml
import rawpy
from matplotlib import pyplot as plt
import tifffile as tiff
import os

import util.utils as util
from util.debug_utils import get_debug_logger

# HDR Image Reading Functions

def read_hdr_3byte(file_path, width, height, byte_order='little'):
    """
    Read HDR image using 3 consecutive bytes per pixel (24-bit packed).
    Little: LSB first (b0 | b1<<8 | b2<<16). Big: MSB first (b0<<16 | b1<<8 | b2).
    """
    with open(file_path, 'rb') as f:
        data = f.read()
    expected_size = width * height * 3
    actual_size = len(data)
    if actual_size < expected_size:
        return None
    data = np.frombuffer(data[:expected_size], dtype=np.uint8)
    data = data.reshape(-1, 3)
    b0, b1, b2 = data[:, 0].astype(np.uint32), data[:, 1].astype(np.uint32), data[:, 2].astype(np.uint32)
    if byte_order == 'little':
        pixels = b0 | (b1 << 8) | (b2 << 16)
    else:
        pixels = (b0 << 16) | (b1 << 8) | b2
    return pixels.reshape(height, width)

def read_hdr_uint16(file_path, width, height, byte_order='little'):
    """
    Read HDR image using uint16 pairs as uint32 pixels (low word | high word << 16).
    """
    with open(file_path, 'rb') as f:
        data = f.read()
    expected_size = width * height * 4
    actual_size = len(data)
    if actual_size < expected_size:
        return None
    dtype = '<u2' if byte_order == 'little' else '>u2'
    uint16_data = np.frombuffer(data[:expected_size], dtype=dtype)
    pixels = uint16_data[0::2].astype(np.uint32) | (uint16_data[1::2].astype(np.uint32) << 16)
    return pixels.reshape(height, width)

def analyze_file_size(file_path, logger=None):
    """Analyze file size to suggest possible dimensions"""
    import logging
    log = logger or logging.getLogger("BrilliantISP.RawLoader")
    file_size = Path(file_path).stat().st_size
    log.debug(f"File size: {file_size:,} bytes")
    pixels_3byte = file_size // 3
    log.debug(f"Pixels (3-byte method): {pixels_3byte:,}")
    pixels_uint16 = file_size // 4
    log.debug(f"Pixels (uint16 method): {pixels_uint16:,}")
    return pixels_3byte, pixels_uint16

from modules.crop.crop import Crop
from modules.dead_pixel_correction.dead_pixel_correction import (
    DeadPixelCorrection as DPC,
)
from modules.black_level_correction.black_level_correction import (
    BlackLevelCorrection as BLC,
)
from modules.pwc_generation.pwc_generation import (PiecewiseCurve as PWC)
from modules.oecf.oecf import OECF
from modules.digital_gain.digital_gain import DigitalGain as DG
from modules.lens_shading_correction.lens_shading_correction import (
    LensShadingCorrection as LSC,
)
from modules.bayer_noise_reduction.bayer_noise_reduction import (
    BayerNoiseReduction as BNR,
)
from modules.auto_white_balance.auto_white_balance import AutoWhiteBalance as AWB
from modules.white_balance.white_balance import WhiteBalance as WB
from modules.white_balance.white_balance_optimized import WhiteBalanceOptimized as WBOPT
from modules.tone_mapping.tone_mapping import ToneMapping as tone_mapping
from modules.demosaic.demosaic import Demosaic
from modules.color_correction_matrix.color_correction_matrix import (
    ColorCorrectionMatrix as CCM,
)
from modules.color_correction_matrix.color_correction_matrix_optimized import (
    ColorCorrectionMatrixOptimized as CCMOPT,
)
from modules.gamma_correction.gamma_correction import GammaCorrection as GC
from modules.auto_exposure.auto_exposure import AutoExposure as AE
from modules.color_space_conversion.color_space_conversion import (
    ColorSpaceConversion as CSC,
)
from modules.ldci.ldci import LDCI
from modules.sharpen.sharpen import Sharpening as SHARP
from modules.noise_reduction_2d.noise_reduction_2d import NoiseReduction2d as NR2D
from modules.rgb_conversion.rgb_conversion import RGBConversion as RGBC
from modules.scale.scale import Scale
from modules.yuv_conv_format.yuv_conv_format import YUVConvFormat as YUV_C


class BrilliantISP:
    """
    Brilliant-ISP Pipeline
    """

    def __init__(self, data_path, config_path, outFileName, output_path=None):
        """
        Constructor: Initialize with config and raw file path
        and Load configuration parameter from yaml file
        """
        self.data_path = data_path
        self.output_path = output_path if output_path else "out_frames/"
        self.outFileName=outFileName
        # Ensure output directory exists
        os.makedirs(self.output_path, exist_ok=True)
        self.load_config(config_path)
        # Set global debug state from config
        from util.debug_utils import set_global_debug_enabled
        set_global_debug_enabled(self.platform.get('debug_enabled', False))
        # Initialize debug logger after config is loaded
        self.logger = get_debug_logger("BrilliantISP", config=self.platform)

    _REQUIRED_CONFIG_KEYS = (
        "platform", "sensor_info", "dead_pixel_correction", "companding", "digital_gain",
        "lens_shading_correction", "bayer_noise_reduction", "black_level_correction",
        "white_balance", "auto_white_balance", "demosaic", "auto_exposure",
        "color_correction_matrix", "gamma_correction", "hdr_durand", "tone_mapping",
        "color_space_conversion", "color_saturation_enhancement", "ldci", "sharpen",
        "2d_noise_reduction", "rgb_conversion", "scale", "crop", "yuv_conversion_format",
    )

    def load_config(self, config_path):
        """
        Load config information to respective module parameters.
        Validates required keys and uses defaults for optional sections.
        """
        self.config_path = config_path
        with open(config_path, "r", encoding="utf-8") as file:
            c_yaml = yaml.safe_load(file)

        missing = [k for k in self._REQUIRED_CONFIG_KEYS if k not in c_yaml]
        if missing:
            raise KeyError(
                f"Config '{config_path}' missing required keys: {missing}. "
                "See config/svs_cam.yml for reference."
            )

        # Extract workspace info
        self.platform = c_yaml["platform"]
        self.raw_file = self.platform["filename"]
        self.render_3a = self.platform["render_3a"]
        self.sensor_info = c_yaml["sensor_info"]

        # ISP module params
        self.parm_dpc = c_yaml["dead_pixel_correction"]
        self.parm_cmpd = c_yaml["companding"]
        self.parm_dga = c_yaml["digital_gain"]
        self.parm_lsc = c_yaml["lens_shading_correction"]
        self.parm_bnr = c_yaml["bayer_noise_reduction"]
        self.parm_blc = c_yaml["black_level_correction"]
        self.parm_oec = c_yaml.get("oecf", {"is_enable": False, "is_save": False})
        self.parm_wbc = c_yaml["white_balance"]
        self.parm_awb = c_yaml["auto_white_balance"]
        self.parm_dem = c_yaml["demosaic"]
        self.parm_ae = c_yaml["auto_exposure"]
        self.parm_ccm = c_yaml["color_correction_matrix"]
        self.parm_gmc = c_yaml["gamma_correction"]
        self.param_durand = c_yaml["hdr_durand"]
        self.param_aces = c_yaml.get("aces", {})
        self.parm_csc = c_yaml["color_space_conversion"]
        self.parm_cse = c_yaml["color_saturation_enhancement"]
        self.parm_ldci = c_yaml["ldci"]
        self.parm_sha = c_yaml["sharpen"]
        self.parm_2dn = c_yaml["2d_noise_reduction"]
        self.parm_rgb = c_yaml["rgb_conversion"]
        self.parm_sca = c_yaml["scale"]
        self.parm_cro = c_yaml["crop"]
        self.parm_yuv = c_yaml["yuv_conversion_format"]
        self.c_yaml = c_yaml
        self.platform["rgb_output"] = self.parm_rgb["is_enable"]
        self.bit_depth = self.sensor_info["bit_depth"]
        self.tone_mapping = c_yaml["tone_mapping"]
        self.tone_mapping_before_demosaic = self.tone_mapping["tone_mapping_before_demosaic"]
        self.tone_mapper = self.tone_mapping["tone_mapper"]
        if self.tone_mapper == "aces":
            self.param_aces = c_yaml.get("aces", {})
        if self.tone_mapper == "integer":
            self.param_integer_tmo = c_yaml.get("integer_tmo", {})
        if self.tone_mapper == "aces_integer":
            self.param_aces_integer = c_yaml.get("aces_integer", {})
        if self.tone_mapper == "hable":
            self.param_hable = c_yaml.get("hable", {})
        if self.tone_mapper == "hable_integer":
            self.param_hable_integer = c_yaml.get("hable_integer", {})

        # add rgb_output_conversion module

    def load_raw(self, byte_order='little'):
        """
        Load raw image from provided path with enhanced HDR support
        
        Args:
            byte_order (str): 'little' or 'big' endian for HDR loading
            reverse_uint32 (bool): If True, reverse byte order within uint32 pixel values
        """
        # Load raw image file information
        path_object = Path(self.data_path, self.raw_file)
        raw_path = str(path_object.resolve())
        self.in_file = path_object.stem
        short_names = self.platform.get("short_output_names", False)
        self.out_file = self.in_file if short_names else "Out_" + self.in_file

        self.platform["in_file"] = self.in_file
        self.platform["out_file"] = self.out_file

        width = self.sensor_info["width"]
        height = self.sensor_info["height"]
        bit_depth = self.sensor_info["bit_depth"]

        # Load Raw with enhanced HDR support
        if path_object.suffix == ".raw":
            # Check if this might be an HDR file by analyzing file size
            file_size = path_object.stat().st_size
            expected_size_3byte = width * height * 3
            expected_size_uint16 = width * height * 4
            
            self.logger.info(f"Loading raw file: {raw_path}")
            self.logger.info(f"Expected dimensions: {width}x{height}")
            self.logger.info(f"File size: {file_size:,} bytes")
            self.logger.info(f"Expected size (3-byte method): {expected_size_3byte:,} bytes")
            self.logger.info(f"Expected size (uint16 method): {expected_size_uint16:,} bytes")
            
            # Try different loading methods based on file size and bit depth
            if bit_depth > 8:
                if abs(file_size - expected_size_3byte) < expected_size_3byte * 0.1:
                    self.logger.info(f"Trying 3-byte HDR method ({byte_order} endian)...")
                    self.raw = read_hdr_3byte(raw_path, width, height, byte_order)
                    if self.raw is None:
                        raise RuntimeError(
                            f"Raw file too small for 3-byte HDR: expected {expected_size_3byte} bytes, "
                            f"got {file_size}"
                        )
                    self.logger.info(f"Successfully loaded using 3-byte HDR method ({byte_order} endian)")
                    return

                if abs(file_size - expected_size_uint16) < expected_size_uint16 * 0.1:
                    self.logger.info(f"Trying uint16 HDR method ({byte_order} endian)...")
                    self.raw = read_hdr_uint16(raw_path, width, height, byte_order)
                    if self.raw is None:
                        raise RuntimeError(
                            f"Raw file too small for uint16 HDR: expected {expected_size_uint16} bytes, "
                            f"got {file_size}"
                        )
                    self.logger.info(f"Successfully loaded using uint16 HDR method ({byte_order} endian)")
                    return

                self.logger.info("Falling back to 2-byte uint16 method...")
                expected_2byte = width * height * 2
                if file_size < expected_2byte:
                    raise RuntimeError(
                        f"Raw file too small: expected at least {expected_2byte} bytes "
                        f"(for 2-byte {width}x{height}), got {file_size}"
                    )
                self.raw = np.fromfile(raw_path, dtype='>u2').reshape((height, width))
            else:
                # For 8-bit or lower, use original method
                self.raw = (
                    np.fromfile(raw_path, dtype=np.uint8)
                    .reshape((height, width))
                    .astype(np.uint16)
                )
        elif path_object.suffix == ".tiff":
            # Load tiff file
            img = tiff.imread(raw_path)
            self.logger.info(f"Image shape: {img.shape}")
            if img.ndim == 3:
                self.raw = img[:, :, 0]
            else:
                self.raw = img
        else:
            img = rawpy.imread(raw_path)
            self.raw = img.raw_image
            


    def run_pipeline(self, visualize_output=True):
        """
        Simulation of ISP-Pipeline
        """
        skip_disabled = self.platform.get("skip_disabled_modules", False)

        # =====================================================================
        # Cropping
        if skip_disabled and not self.parm_cro["is_enable"]:
            cropped_img = self.raw
        else:
            crop = Crop(self.raw, self.platform, self.sensor_info, self.parm_cro)
            cropped_img = crop.execute()

        # =====================================================================
        # Dead pixels correction
        if skip_disabled and not self.parm_dpc["is_enable"]:
            dpc_raw = cropped_img
        else:
            dpc = DPC(cropped_img, self.sensor_info, self.parm_dpc, self.platform)
            dpc_raw = dpc.execute()

        # =====================================================================
        # Black level correction
        if skip_disabled and not self.parm_blc["is_enable"]:
            blc_raw = dpc_raw
        else:
            blc = BLC(dpc_raw, self.platform, self.sensor_info, self.parm_blc)
            blc_raw = blc.execute()

        # =====================================================================
        # decompanding
        if skip_disabled and not self.parm_cmpd["is_enable"]:
            cmpd_raw = blc_raw.astype(np.uint32)
        else:
            cmpd = PWC(blc_raw, self.platform, self.sensor_info, self.parm_cmpd)
            cmpd_raw = cmpd.execute()

        # =====================================================================
        # OECF
        if skip_disabled and not self.parm_oec.get("is_enable", False):
            oecf_raw = cmpd_raw
        else:
            oecf = OECF(cmpd_raw, self.platform, self.sensor_info, self.parm_oec)
            oecf_raw = oecf.execute()

        # =====================================================================
        # Digital Gain (receives OECF output per pipeline order: PWC -> OECF -> DG)
        dga = DG(oecf_raw, self.platform, self.sensor_info, self.parm_dga)
        dga_raw, self.dga_current_gain = dga.execute()

        # =====================================================================
        # Lens shading correction
        if skip_disabled and not self.parm_lsc.get("is_enable", True):
            lsc_raw = dga_raw
        else:
            lsc = LSC(dga_raw, self.platform, self.sensor_info, self.parm_lsc)
            lsc_raw = lsc.execute()

        # =====================================================================
        # Bayer noise reduction
        if skip_disabled and not self.parm_bnr["is_enable"]:
            bnr_raw = lsc_raw
        else:
            bnr = BNR(lsc_raw, self.sensor_info, self.parm_bnr, self.platform)
            bnr_raw = bnr.execute()


        # =====================================================================
        # Auto White Balance
        awb = AWB(bnr_raw, self.sensor_info, self.parm_awb, self.parm_wbc)
        self.awb_gains = awb.execute()

        # =====================================================================
        # White balancing
        # Use optimized version for better performance
        wbc = WBOPT(bnr_raw, self.platform, self.sensor_info, self.parm_wbc, self.awb_gains)
        wb_raw = wbc.execute()


#%%
       # # =====================================================================
       # HDR tone mapping before Demosaicing
        if  self.tone_mapping_before_demosaic:
            tone_mapper = tone_mapping(wb_raw, pipeline_self=self)
            hdr_raw = tone_mapper.execute()
            self.logger.info(f"HDR Image mean: {np.mean(hdr_raw)}")
        else:
            max_val = 2**self.sensor_info.get("hdr_bit_depth", 24) - 1
            hdr_raw = (wb_raw.astype(np.float32) * (65535.0 / max_val)).astype(np.uint16)

#%%        # =====================================================================
        # CFA demosaicing
        cfa_inter = Demosaic(hdr_raw, self.platform, self.sensor_info, self.parm_dem)
        demos_img = cfa_inter.execute()
        self.logger.info(f"Demosaiced Image mean: {np.mean(demos_img)}")
        
        # =====================================================================
        # Color correction matrix
        # Use optimized version for better performance
        ccm = CCMOPT(demos_img, self.platform, self.sensor_info, self.parm_ccm)
        ccm_img = ccm.execute()
        self.logger.info(f"CCM Image mean: {np.mean(ccm_img)}")
        #%%

        # =====================================================================
        # HDR tone mapping after Demosaicing
        if not self.tone_mapping_before_demosaic:
            tone_mapper = tone_mapping(ccm_img, pipeline_self=self)
            CCM_tone_mapped = tone_mapper.execute()
            self.logger.info(f"HDR Image mean: {np.mean(CCM_tone_mapped)}")
            ccm_img = CCM_tone_mapped
            
        # =====================================================================
        # Gamma
        gmc = GC(ccm_img, self.platform, self.sensor_info, self.parm_gmc)
        gamma_raw = gmc.execute()
        self.logger.info(f"Gamma Image mean: {np.mean(gamma_raw)}")

        # =====================================================================
        # Auto-Exposure
        aef = AE(gamma_raw, self.sensor_info, self.parm_ae)
        self.ae_feedback = aef.execute()
        self.logger.info(f"AE Feedback: {self.ae_feedback}")

        # =====================================================================
        # Color space conversion
        csc = CSC(gamma_raw, self.platform, self.sensor_info, self.parm_csc, self.parm_cse )
        csc_img = csc.execute()
        self.logger.info(f"CSC Image mean: {np.mean(csc_img)}")

        # =====================================================================
        # Local Dynamic Contrast Improvement
        if skip_disabled and not self.parm_ldci["is_enable"]:
            ldci_img = csc_img
        else:
            ldci = LDCI(
                csc_img,
                self.platform,
                self.sensor_info,
                self.parm_ldci,
                self.parm_csc["conv_standard"],
            )
            ldci_img = ldci.execute()

        # =====================================================================
        # Sharpening
        if skip_disabled and not self.parm_sha["is_enable"]:
            sharp_img = ldci_img
        else:
            sharp = SHARP(
                ldci_img,
                self.platform,
                self.sensor_info,
                self.parm_sha,
                self.parm_csc["conv_standard"],
            )
            sharp_img = sharp.execute()

        # =====================================================================
        # 2d noise reduction
        if skip_disabled and not self.parm_2dn["is_enable"]:
            nr2d_img = sharp_img
        else:
            nr2d = NR2D(
                sharp_img,
                self.sensor_info,
                self.parm_2dn,
                self.platform,
                self.parm_csc["conv_standard"],
            )
            nr2d_img = nr2d.execute()

        # =====================================================================
        # RGB conversion
        rgbc = RGBC(
            nr2d_img, self.platform, self.sensor_info, self.parm_rgb, self.parm_csc
        )
        rgbc_img = rgbc.execute()

        # =====================================================================
        # Scaling
        if skip_disabled and not self.parm_sca["is_enable"]:
            scaled_img = rgbc_img
        else:
            scale = Scale(
                rgbc_img,
                self.platform,
                self.sensor_info,
                self.parm_sca,
                self.parm_csc["conv_standard"],
            )
            scaled_img = scale.execute()

        # =====================================================================
        # YUV saving format 444, 422 etc
        yuv = YUV_C(scaled_img, self.platform, self.sensor_info, self.parm_yuv)
        yuv_conv = yuv.execute()

        # only to view image if csc is off it does nothing
        out_img = yuv_conv
        out_dim = scaled_img.shape  # dimensions of Output Image

        # Is not part of ISP-pipeline only assists in visualizing output results
        if visualize_output:

            # There can be two out_img formats depending upon which modules are
            # enabled 1. YUV    2. RGB

            if self.parm_yuv["is_enable"] is True:

                # YUV_C is enabled and RGB_C is disabled: Output is compressed YUV
                # To display : Need to decompress it and convert it to RGB.
                image_height, image_width, _ = out_dim
                yuv_custom_format = self.parm_yuv["conv_type"]

                yuv_conv = util.get_image_from_yuv_format_conversion(
                    yuv_conv, image_height, image_width, yuv_custom_format
                )

                rgbc.yuv_img = yuv_conv
                out_rgb = rgbc.yuv_to_rgb()

            elif self.parm_rgb["is_enable"] is False:

                # RGB_C is disabled: Output is 3D - YUV
                # To display : Only convert it to RGB
                rgbc.yuv_img = yuv_conv
                out_rgb = rgbc.yuv_to_rgb()

            else:
                # RGB_C is enabled: Output is RGB
                # no further processing is needed for display
                out_rgb = out_img

            # If both RGB_C and YUV_C are enabled. Brilliant-ISP will generate
            # an output but it will be an invalid image.
            short_names = self.platform.get("short_output_names", False)
            if not short_names:
                self.outFileName = self.outFileName + "TM_" + str(self.tone_mapper) + "_s_" + str(self.parm_cse['saturation_gain']) + "_CCM_" + str(self.parm_ccm['is_enable']) + "_Before_Demosaic_" + str(self.tone_mapping_before_demosaic)

            util.save_pipeline_output(self.out_file, out_rgb, self.c_yaml, self.outFileName, self.output_path, short_names=short_names)

    def execute(self, img_path=None, load_method='auto', byte_order='little'):
        """
        Start execution of Brilliant-ISP
        
        Args:
            img_path (str): Optional path to image file
            load_method (str): 'auto', '3byte', 'uint16', or 'original'
            byte_order (str): 'little' or 'big'
            reverse_uint32 (bool): If True, reverse byte order within uint32 pixel values
        """
        if img_path is not None:
            self.raw_file = img_path
            self.c_yaml["platform"]["filename"] = self.raw_file
    
        self.load_raw(byte_order=byte_order)
    
        # Print Logs to mark start of pipeline Execution
        self.logger.info(50 * "-" + "\nLoading RAW Image Done......\n")
        self.logger.info(f"Filename: {self.in_file}")

        # Note Initial Time for Pipeline Execution
        start = time.time()

        if not self.render_3a:
            # Run ISP-Pipeline once
            self.run_pipeline(visualize_output=True)
            # Display 3A Statistics
        else:
            # Run ISP-Pipeline till Correct Exposure with AWB gains
            self.execute_with_3a_statistics()

        util.display_ae_statistics(self.ae_feedback, self.awb_gains, self.logger)

        # Print Logs to mark end of pipeline Execution
        self.logger.info(50 * "-" + "\n")

        # Calculate pipeline execution time
        self.logger.info(f"\nPipeline Elapsed Time: {time.time() - start:.3f}s")

    def load_3a_statistics(self, awb_on=True, ae_on=True):
        """
        Update 3A Stats into WB and DG modules parameters
        """
        # Update 3A in c_yaml too because it is output config
        if awb_on is True and self.parm_dga["is_auto"] and self.parm_awb["is_enable"]:
            self.parm_wbc["r_gain"] = self.c_yaml["white_balance"]["r_gain"] = float(
                self.awb_gains[0]
            )
            self.parm_wbc["b_gain"] = self.c_yaml["white_balance"]["b_gain"] = float(
                self.awb_gains[1]
            )
        if ae_on is True and self.parm_dga["is_auto"] and self.parm_ae["is_enable"]:
            self.parm_dga["ae_feedback"] = self.c_yaml["digital_gain"][
                "ae_feedback"
            ] = self.ae_feedback
            self.parm_dga["current_gain"] = self.c_yaml["digital_gain"][
                "current_gain"
            ] = self.dga_current_gain

    def execute_with_3a_statistics(self):
        """
        Execute Brilliant-ISP with AWB gains and correct exposure
        """

        # Maximum Iterations depend on total permissible gains
        max_dg = len(self.parm_dga["gain_array"])

        # Run ISP-Pipeline
        self.run_pipeline(visualize_output=False)
        self.load_3a_statistics()
        while not (
            (self.ae_feedback == 0)
            or (self.ae_feedback == -1 and self.dga_current_gain == max_dg)
            or (self.ae_feedback == 1 and self.dga_current_gain == 0)
            or self.ae_feedback is None
        ):
            self.run_pipeline(visualize_output=False)
            self.load_3a_statistics()

        self.run_pipeline(visualize_output=True)

    def update_sensor_info(self, sensor_info, update_blc_wb=False):
        """
        Update sensor_info in config files
        """
        self.sensor_info["width"] = self.c_yaml["sensor_info"]["width"] = sensor_info[0]

        self.sensor_info["height"] = self.c_yaml["sensor_info"]["height"] = sensor_info[
            1
        ]

        self.sensor_info["bit_depth"] = self.c_yaml["sensor_info"][
            "bit_depth"
        ] = sensor_info[2]

        self.sensor_info["bayer_pattern"] = self.c_yaml["sensor_info"][
            "bayer_pattern"
        ] = sensor_info[3]

        if update_blc_wb:
            self.parm_blc["r_offset"] = self.c_yaml["black_level_correction"][
                "r_offset"
            ] = sensor_info[4][0]
            self.parm_blc["gr_offset"] = self.c_yaml["black_level_correction"][
                "gr_offset"
            ] = sensor_info[4][1]
            self.parm_blc["gb_offset"] = self.c_yaml["black_level_correction"][
                "gb_offset"
            ] = sensor_info[4][2]
            self.parm_blc["b_offset"] = self.c_yaml["black_level_correction"][
                "b_offset"
            ] = sensor_info[4][3]

            self.parm_blc["r_sat"] = self.c_yaml["black_level_correction"][
                "r_sat"
            ] = sensor_info[5]
            self.parm_blc["gr_sat"] = self.c_yaml["black_level_correction"][
                "gr_sat"
            ] = sensor_info[5]
            self.parm_blc["gb_sat"] = self.c_yaml["black_level_correction"][
                "gb_sat"
            ] = sensor_info[5]
            self.parm_blc["b_sat"] = self.c_yaml["black_level_correction"][
                "b_sat"
            ] = sensor_info[5]

            self.parm_wbc["r_gain"] = self.c_yaml["white_balance"][
                "r_gain"
            ] = sensor_info[6][0]
            self.parm_wbc["b_gain"] = self.c_yaml["white_balance"][
                "b_gain"
            ] = sensor_info[6][2]

            # if sensor_info[7] is not None:
            #     self.parm_ccm["corrected_red"] = sensor_info[7][0,0:3]
            #     self.parm_ccm["corrected_green"] = sensor_info[7][1,0:3]
            #     self.parm_ccm["corrected_blue"] = sensor_info[7][2,0:3]
