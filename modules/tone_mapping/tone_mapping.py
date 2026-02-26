"""
Production-style tone mapping with bit-depth-aware normalization.

Input: Linear data from decompanding (before demosaic) or CCM (after demosaic).
- Before demosaic: uint32, range 2^hdr_bit_depth - 1 (e.g. 24-bit → 16,777,215)
- After demosaic:  uint16, range 65535
Output: uint16 for pipeline (gamma expects 16-bit input).

Supported tone_mapper values and their config sections:
  durand         -> hdr_durand
  aces           -> aces
  integer        -> integer_tmo
  aces_integer   -> aces_integer
  hable          -> hable
  hable_integer  -> hable_integer
"""
from util.utils import save_output_array
import numpy as np


class ToneMapping:
    def __init__(self, img, pipeline_self):
        self.img_orig = img  # Passthrough when disabled
        # Production-style: normalize using actual input bit depth
        if pipeline_self.tone_mapping_before_demosaic:
            input_max = 2 ** pipeline_self.sensor_info.get("hdr_bit_depth", 24) - 1
        else:
            input_max = 65535  # CCM outputs uint16
        self.img = img.astype(np.float32) / input_max
        self.input_max = input_max
        pipeline_rgb_bits = pipeline_self.sensor_info.get("pipeline_rgb_bit_depth", 16)
        self.output_max = 2**pipeline_rgb_bits - 1
        self.is_save = pipeline_self.tone_mapping["is_save"]
        self.method = pipeline_self.tone_mapping["tone_mapper"]
        self.enable=pipeline_self.tone_mapping["is_enable"]
        self.platform = pipeline_self.platform
        self.sensor_info = pipeline_self.sensor_info
        self.param_durand = pipeline_self.param_durand
        self.param_aces = getattr(pipeline_self, 'param_aces', {})
        self.logger = pipeline_self.logger
        self.bit_depth= self.sensor_info.get("output_bit_depth", 8)
        self.tone_mapping_before_demosaic = pipeline_self.tone_mapping_before_demosaic

        if self.tone_mapping_before_demosaic == True:
            self.Lw = self.img
        else:
            self.Lw = self.extract_luminance()

        if self.method == "durand":
            from modules.tone_mapping.durand.hdr_durand_fast import HDRDurandToneMapping
            self.hdr = HDRDurandToneMapping(self.Lw, self.platform, self.sensor_info, self.param_durand)

        elif self.method == "aces":
            from modules.tone_mapping.aces.aces_tone_mapping import ACESToneMapping
            self.hdr = ACESToneMapping(self.Lw, self.platform, self.sensor_info, self.param_aces)

        elif self.method == "integer":
            from modules.tone_mapping.integer_tmo.integer_tone_mapping import IntegerToneMapping
            param_int = getattr(pipeline_self, "param_integer_tmo", {})
            if self.tone_mapping_before_demosaic:
                self.hdr = IntegerToneMapping(self.img_orig, self.platform, self.sensor_info, param_int)
            else:
                L_int = self._extract_luminance_int()
                self.hdr = IntegerToneMapping(L_int, self.platform, self.sensor_info, param_int)
            self._use_integer_tmo = True
        elif self.method == "aces_integer":
            from modules.tone_mapping.integer_tmo.aces_integer_tone_mapping import ACESIntegerToneMapping
            param_aces_int = getattr(pipeline_self, "param_aces_integer", {})
            if self.tone_mapping_before_demosaic:
                self.hdr = ACESIntegerToneMapping(self.img_orig, self.platform, self.sensor_info, param_aces_int)
            else:
                L_int = self._extract_luminance_int()
                self.hdr = ACESIntegerToneMapping(L_int, self.platform, self.sensor_info, param_aces_int)
            self._use_integer_tmo = True
        elif self.method == "hable":
            from modules.tone_mapping.hable.hable_tone_mapping import HableToneMapping
            param_hable = getattr(pipeline_self, "param_hable", {})
            self.hdr = HableToneMapping(self.Lw, self.platform, self.sensor_info, param_hable)
        elif self.method == "hable_integer":
            from modules.tone_mapping.integer_tmo.hable_integer_tone_mapping import HableIntegerToneMapping
            param_hable_int = getattr(pipeline_self, "param_hable_integer", {})
            if self.tone_mapping_before_demosaic:
                self.hdr = HableIntegerToneMapping(self.img_orig, self.platform, self.sensor_info, param_hable_int)
            else:
                L_int = self._extract_luminance_int()
                self.hdr = HableIntegerToneMapping(L_int, self.platform, self.sensor_info, param_hable_int)
            self._use_integer_tmo = True
        else:
            raise ValueError(
                f"Unknown tone mapping method: {self.method}. "
                "Supported: 'durand', 'aces', 'integer', 'aces_integer', 'hable', 'hable_integer'."
            )
        self._use_integer_tmo = getattr(self, "_use_integer_tmo", False)
    
    def extract_luminance(self):
        epsilon = 1e-6
        Lw = 0.2126 * self.img[..., 0] + 0.7152 * self.img[..., 1] + 0.0722 * self.img[..., 2]
        return np.maximum(Lw, epsilon)

    def _extract_luminance_int(self):
        """Integer luminance for production-style path. L = (2126*R + 7152*G + 722*B) / 10000"""
        img = self.img_orig
        L = (2126 * img[..., 0] + 7152 * img[..., 1] + 722 * img[..., 2]) // 10000
        return np.maximum(L.astype(np.uint32), 1)

    def combine_luminance(self, output_luminance):
        chromaticity = self.img / self.Lw[..., np.newaxis]
        return chromaticity * output_luminance[..., np.newaxis]
    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_tonemapped_",
                self.platform,
                self.sensor_info["bit_depth"],
                self.sensor_info["bayer_pattern"],
            )



    def execute(self, visualize_output=True, save_output=True):
        if self.enable is False:
            # Passthrough: return input in pipeline format (uint16)
            if self.tone_mapping_before_demosaic:
                return (
                    self.img_orig.astype(np.float32) * (self.output_max / self.input_max)
                ).astype(np.uint16)
            return self.img_orig

        if self._use_integer_tmo:
            img_out = self.hdr.execute()  # Already uint16
            if self.tone_mapping_before_demosaic is False:
                # Combine luminance with chromaticity
                L_int = self._extract_luminance_int()
                chrom = self.img_orig.astype(np.float32) / np.maximum(L_int, 1)[..., np.newaxis]
                img_out = (chrom * img_out[..., np.newaxis]).astype(np.uint16)
            self.save()
            return np.clip(img_out, 0, self.output_max).astype(np.uint16)

        img_out = self.hdr.execute()
        if self.tone_mapping_before_demosaic is False:
            img_out = self.combine_luminance(img_out)
        self.save()
        img_out_int = (img_out * self.output_max).astype(np.uint16)
        return img_out_int
   

