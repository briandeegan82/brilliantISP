# -*- coding: utf-8 -*-
"""
TMOz Pipeline Class
Created on Mon Sep 22 17:08:03 2025
@author: Imra
"""

import numpy as np
from .TMOzUtils import utils
from .fast_bilateral_filter_imran import my_fast_bilateral_filter
import yaml

class hdrRGB2TMOz:
    def __init__(self, hdr, cond=None, sorroundPath=None, outFileName=None, output_dir=None):
        """
        Initialize TMOz pipeline.
        
        Parameters:
            hdr_path (str or Path): Path to HDR input image (.hdr).
            cond (dict): Tone mapping conditions (XYZw, Lw, etc.).
            outFileName (str): Output file base name (without extension).
            output_dir (str or Path): Folder where output will be saved.
        """
        
        self.hdr = np.maximum(hdr, 1e-4)  # Avoid zero/near-zero values
        
        # Set surround conditions
        if cond is not None:
            self.cond = cond
        elif sorroundPath is not None:
            with open(sorroundPath, "r") as file:
                data = yaml.safe_load(file)
            if "cond" in data:
                self.cond = data["cond"]
            else:
                raise ValueError(f"'cond' key not found in YAML file: {sorroundPath}")
        else:
            raise ValueError("Please provide the surround conditions or path to a YAML file.")
            
            
        if outFileName is not None:
            self.outFileName = outFileName
        else:
            self.outFileName ='TMOzImg'

        if output_dir is not None:
            self.output_dir = output_dir
        else:
            self.output_dir ='Out_hdr_TMOz'
            
        

            
        #Visualize HDR image
        

    def execute(self, visualize_output=True, save_output=True):
        """
        Run the tone mapping operator on the HDR input.
        
        Parameters:
            visualize_output (bool): Show result in a window.
            save_output (bool): Save SDR result to output_dir.
        """
        # Apply tone mapping
       
        
        sz = self.hdr.shape
        
        # Convert HDR image from sRGB to XYZ
        xyzi = utils.srgb2xyzLinear(self.hdr)  # Placeholder: implement srgb2xyzLinear
        del self.hdr
        
        

        # Extract luminance (Y) and calculate key value
        y = xyzi[:, 1]  # second column is Y
        key = utils.imgKey(y)  # Placeholder: implement imgKey

        # Get surround conditions
        XYZw1, La1, Yb, sr, XYZw2, La2 = utils.getcond(self.cond)

        # Normalize XYZ values
        normy = 100
        xyzi = xyzi / np.max(xyzi[:, 1]) * normy

        # Convert XYZ to CAM16Q
        Q, RGBa = utils.XYZ2CAM16Q_RGBa(xyzi, XYZw1, La1, Yb, sr)          # Placeholder
        
        Qimg = Q.reshape(sz[0], sz[1])
        del xyzi, Q

        # Normalize Q image
        maxq = np.max(Qimg)
        Qimg = Qimg / maxq

        # Apply Bilateral Filter to get base and detail layers
        # base_Q, detail_Q = utils.bilateral_filter(self,Qimg)  # Placeholder
        
        base_Q, detail_Q = my_fast_bilateral_filter(Qimg) 
        del Qimg

        # Enhance Details using Local Contrast
        detail_Qe = utils.Qimg_LocalContrast_Enhancement(detail_Q)  # Placeholder
        del detail_Q

        # Apply Tone Curve Compression to Base
        base_Qc = utils.tonecurveM(base_Q, key)  # Placeholder
        del base_Q

        # Combine Base and Detail Images
        Qimgo = base_Qc * detail_Qe * maxq
        # del base_Qc, detail_Qe

        # Color Correction based on QMh
        Mc, h = utils.newM(Qimgo.flatten(), XYZw2, La2, Yb, sr, RGBa)  # Placeholder
        QMh = np.stack([Qimgo.flatten(), Mc.flatten(), h.flatten()], axis=1)

        # Convert QMh back to XYZ
        xyzo = utils.CAM16UCS2XYZ_QMhs(QMh, XYZw2, La2, Yb, sr)  # Placeholder

        # Clipping: Simulate Incomplete Light Adaptation
        TonedXYZ = utils.TMOzclip(xyzo)  # Placeholder

        # Final XYZ to RGB image
        TonedXYZ = TonedXYZ / np.max(TonedXYZ[:, 1]) * XYZw2[1]
        TonedXYZ = TonedXYZ / np.max(TonedXYZ[:, 1])
        rgbimg = utils.xyz2srgb(TonedXYZ)  # Placeholder

        rgbimg = rgbimg.reshape(sz)
        # Display result
        if visualize_output:
            utils.imshow(np.clip(rgbimg.astype(np.float32)/65535*255, 0,255), "Tone Mapped Image")


        # Save result
        if save_output: 
            utils.saveFile(rgbimg, self.outFileName, self.output_dir)

        return rgbimg
    

