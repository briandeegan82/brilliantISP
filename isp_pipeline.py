"""
File: isp_pipeline.py
Description: Executes the complete pipeline
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""

from infinite_isp import InfiniteISP

CONFIG_PATH = "./config/svs_cam.yml"
RAW_DATA = "./in_frames/hdr_mode/"
FILENAME = 'frame_2880_fsin_38361195454327480.raw'

if __name__ == "__main__":

    infinite_isp = InfiniteISP(RAW_DATA, CONFIG_PATH)


    infinite_isp.execute(img_path=FILENAME)

