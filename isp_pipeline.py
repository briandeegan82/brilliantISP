"""
File: isp_pipeline.py
Description: Executes the complete pipeline
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""

from infinite_isp import InfiniteISP

CONFIG_PATH = "./config/svs_cam.yml"
RAW_DATA = "/media/brian/ssd-drive/drive/output/191-G-NUIG.RAW.DAI_3MPX_FV.BIN.20250723.134409_extracted/"
FILENAME = 'frame_6709.raw'

if __name__ == "__main__":

    infinite_isp = InfiniteISP(RAW_DATA, CONFIG_PATH)

    # set generate_tv flag to false
    infinite_isp.execute(img_path=FILENAME)
