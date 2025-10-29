"""
File: isp_pipeline.py
Description: Executes the complete pipeline
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""

from brilliant_isp import BrilliantISP

CONFIG_PATH = "./config/triton_490.yml"
RAW_DATA = "./in_frames/hdr_mode/"
FILENAME = r'E:\IDM\Compressed\brilliantISP-Original\brilliantISP-main - color articiats corrected\in_frames\hdr_mode\Triton_Captures_new\Imran settings\BayerRG16.raw'
# FILENAME = r'C:\Users\Imra\Downloads\LUCID_TRI054S-C_221400748__20251027172736632_image0.raw'

# CONFIG_PATH = "./config/svs_cam.yml"
# RAW_DATA = "E:/UoG OneDrive\OneDrive - National University of Ireland, Galway/UoG\Dataset/IMX 490 Raw - Roshan/New folder/"
# FILENAME = 'frame_0402.raw'

# CONFIG_PATH = "./config/svs_cam.yml"
# RAW_DATA = "E:/UoG OneDrive\OneDrive - National University of Ireland, Galway/UoG\Dataset/IMX 490 Raw - Roshan/New folder/"
# FILENAME = 'frame_0402.raw'

if __name__ == "__main__":

    brilliant_isp = BrilliantISP(RAW_DATA, CONFIG_PATH)


    brilliant_isp.execute(img_path=FILENAME)

