"""
File: isp_pipeline.py
Description: Executes the complete pipeline
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""

from brilliant_isp import BrilliantISP

CONFIG_PATH = "./config/svs_cam.yml"
RAW_DATA = "./in_frames/hdr_mode/"
FILENAME = 'frame_0000.raw'



if __name__ == "__main__":

    brilliant_isp = BrilliantISP(RAW_DATA, CONFIG_PATH,outFileName="")


    brilliant_isp.execute(img_path=FILENAME)

