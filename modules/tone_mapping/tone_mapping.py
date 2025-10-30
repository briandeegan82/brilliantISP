import matplotlib.pyplot as plt
from util.utils import save_output_array
import numpy as np
import os

class ToneMapping:
    def __init__(self, img, pipeline_self):
        self.img =img.astype(np.float32)
        self.img=self.img/655535
        self.is_save = pipeline_self.tone_mapping["is_save"]
        self.method = pipeline_self.tone_mapping["tone_mapper"]
        self.enable=pipeline_self.tone_mapping["is_enable"]
        self.platform = pipeline_self.platform
        self.sensor_info = pipeline_self.sensor_info
        self.param_durand = pipeline_self.param_durand
        # self.param_reinhard = pipeline_self.param_reinhard
        # self.param_tmoz = pipeline_self.param_tmoz
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

        elif self.method == "reinhardPTRGlobal":
            from modules.tone_mapping.reinhardPTR.reinhdard_PTRG import reinhdard_PTRG
            self.hdr = reinhdard_PTRG(self.Lw, self.platform, self.sensor_info, self.param_durand)
            
        elif self.method == "reinhdard_PTRLocal":
            from modules.tone_mapping.reinhardPTRLocal.reinhdard_PTRGLocal import reinhdard_PTRLocal
            self.hdr = reinhdard_PTRLocal(self.Lw, self.platform, self.sensor_info, self.param_durand)

        elif self.method == "TMOz" and self.tone_mapping_before_demosaic == False:
            self.cond=pipeline_self.TMOz_sorround_cond
            from modules.tone_mapping.TMOz.hdrRGB2TMOz import hdrRGB2TMOz 
            self.hdr = hdrRGB2TMOz(self.img, cond=self.cond)

        else:
            raise ValueError(f"Unknown tone mapping method: {self.method}. Make sure the TMOz must be applied after demossaicing.")
    
    def extract_luminance(self):
        epsilon = 1e-6
        Lw = 0.2126 * self.img[..., 0] + 0.7152 * self.img[..., 1] + 0.0722 * self.img[..., 2]
        return np.maximum(Lw, epsilon)

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
        if self.enable ==False:
            return self.img
        img_out = self.hdr.execute()
        if self.tone_mapping_before_demosaic == False and self.method!='TMOz':
            img_out = self.combine_luminance(img_out)
            
        self.save()
        img_out_int= (img_out * 65535).astype(np.uint16)
        return img_out_int
   

