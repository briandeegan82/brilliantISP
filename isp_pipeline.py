"""
File: isp_pipeline.py
Description: Executes the complete pipeline
Code / Paper  Reference:
Author: Brian Deegan (based in part on 10xEngineers / Infinite-ISP)
------------------------------------------------------------
"""

import os
from pathlib import Path

from brilliant_isp import BrilliantISP

# Working pair: hdr_mode with SVS_cam.yml
CONFIG_PATH = "./config/SVS_cam.yml"
_DEFAULT_RAW = "./in_frames/hdr_mode"
RAW_DATA = os.environ.get("ISP_RAW_DATA", _DEFAULT_RAW)
FILENAME = "FV.raw"

if __name__ == "__main__":
    raw_file = Path(RAW_DATA).expanduser().resolve() / FILENAME
    if not raw_file.is_file():
        raise SystemExit(
            f"Raw file not found: {raw_file}\n"
            "Place the file under in_frames/hdr_mode or set ISP_RAW_DATA to the directory "
            f"that contains {FILENAME!r} (example: ISP_RAW_DATA=/path/to/raw python isp_pipeline.py)."
        )
    brilliant_isp = BrilliantISP(RAW_DATA, CONFIG_PATH, outFileName="")
    brilliant_isp.execute(img_path=FILENAME)
