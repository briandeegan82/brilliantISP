"""
Batch-convert working RAW/config pairs from matches.py.
Each converted image is written to a "convert" subfolder next to the source RAW.
"""

import logging
import os
from pathlib import Path

from tqdm import tqdm
from brilliant_isp import BrilliantISP

from matches import matches as WORKING_MATCHES
from util.config_utils import parse_file_name, extract_raw_metadata
from util.raw_io import read_raw_payload

_log = logging.getLogger(__name__)

VIDEO_MODE = False
EXTRACT_SENSOR_INFO = True
UPDATE_BLC_WB = True
SKIP_DIR_NAMES = {"old", "unused", "convert"}


def process_single_raw_file(raw_file_path: str, config_path: str, output_dir: str) -> None:
    """
    Process a single .raw file and save the converted image to the specified output folder
    """
    file_dir = os.path.dirname(raw_file_path)
    file_name = os.path.basename(raw_file_path)

    os.makedirs(output_dir, exist_ok=True)

    file_stem = Path(file_name).stem
    config_file = os.path.join(file_dir, f"{file_stem}-configs.yml")

    brilliant_isp = BrilliantISP(file_dir, config_path, outFileName="", output_path=output_dir)
    assert brilliant_isp.c_yaml is not None

    brilliant_isp.c_yaml["platform"]["generate_tv"] = False

    if os.path.exists(config_file):
        _log.info(f"Found specific config: {config_file}")
        brilliant_isp.load_config(config_file)
        brilliant_isp.execute(file_name)
    else:
        _log.info(f"Using default config for: {file_name} ({config_path})")

        if EXTRACT_SENSOR_INFO:
            if file_name.lower().endswith(".raw"):
                _log.info(f"RAW file, extracting sensor info from filename: {file_name}")
                sensor_info = parse_file_name(file_name)
                if sensor_info:
                    brilliant_isp.update_sensor_info(sensor_info)
                    _log.info("Updated sensor_info in config")
                else:
                    _log.info("No information in filename - sensor_info not updated")
            elif file_name.lower().endswith(".pgm"):
                _log.info(f"PGM file, extracting dimensions from file: {file_name}")
                try:
                    payload = read_raw_payload(raw_file_path)
                    if payload.is_pgm and payload.pgm_width and payload.pgm_height:
                        sensor_info = {
                            "width": payload.pgm_width,
                            "height": payload.pgm_height,
                        }
                        brilliant_isp.update_sensor_info(sensor_info)
                        _log.info(f"Updated sensor dimensions from PGM: {payload.pgm_width}x{payload.pgm_height}")
                    else:
                        _log.info("Could not extract PGM dimensions")
                except Exception as e:
                    _log.warning(f"Error reading PGM file: {e}")
            else:
                sensor_info = extract_raw_metadata(raw_file_path)
                if sensor_info:
                    brilliant_isp.update_sensor_info(sensor_info, UPDATE_BLC_WB)
                    _log.info("Updated sensor_info in config")
                else:
                    _log.info("Not compatible file for metadata - sensor_info not updated")

        brilliant_isp.execute(file_name)


def batch_convert_working_matches() -> None:
    """
    Convert every working pair listed in matches.py
    """
    pairs = []
    for raw_rel, config_name in WORKING_MATCHES:
        raw_path = Path(raw_rel)
        if any(part in SKIP_DIR_NAMES for part in raw_path.parts):
            continue
        config_path = f"./config/{config_name}"
        if not raw_path.is_file():
            _log.warning(f"Working RAW not found, skipping: {raw_path}")
            continue
        if not Path(config_path).is_file():
            _log.warning(f"Working config not found, skipping: {config_path}")
            continue

        # Determine output path based on input location
        try:
            rel = raw_path.parent.relative_to("in_frames")
            output_dir = str(Path("out_frames") / rel)
        except ValueError:
            output_dir = str(Path("out_frames") / raw_path.parent.name)

        pairs.append((str(raw_path), config_path, output_dir))

    if not pairs:
        _log.warning("No working RAW/config pairs found in matches.py")
        return

    _log.info(f"Found {len(pairs)} working RAW files to convert")

    for raw_file, config_path, output_dir in tqdm(pairs, desc="Converting working RAWs", ncols=100):
        try:
            _log.info(f"Processing: {raw_file}")
            process_single_raw_file(raw_file, config_path, output_dir)
            _log.info(f"Successfully processed: {raw_file}")
        except Exception as e:
            _log.error(f"Error processing {raw_file}: {str(e)}")
            continue


def batch_convert_folder(input_folder: str, config_path: str, output_folder: str) -> None:
    """
    Convert all RAW files in a specific folder using a specific config.
    Each converted image is written to the specified output folder.
    """
    input_path = Path(input_folder)
    if not input_path.exists():
        _log.error(f"Input folder not found: {input_folder}")
        return

    if not Path(config_path).is_file():
        _log.error(f"Config file not found: {config_path}")
        return

    # Determine output directory structure
    try:
        rel = input_path.relative_to("in_frames")
        output_dir = str(Path(output_folder) / rel)
    except ValueError:
        output_dir = str(Path(output_folder) / input_path.name)

    os.makedirs(output_dir, exist_ok=True)

    # Find all raw/image files (including .pgm)
    raw_files = []
    for ext in [".raw", ".pgm", ".nef", ".dng"]:
        raw_files.extend(input_path.glob(f"*{ext}"))

    if not raw_files:
        _log.warning(f"No RAW files found in {input_folder}")
        return

    _log.info(f"Found {len(raw_files)} files to convert")
    _log.info(f"Config: {config_path}")
    _log.info(f"Output: {output_dir}")

    for raw_file in tqdm(raw_files, desc="Converting RAW files", ncols=100):
        try:
            _log.info(f"Processing: {raw_file}")
            process_single_raw_file(str(raw_file), config_path, output_dir)
            _log.info(f"Successfully processed: {raw_file.name}")
        except Exception as e:
            _log.error(f"Error processing {raw_file.name}: {str(e)}")
            continue


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Direct folder/config mode
    DIRECT_INPUT_FOLDER = "./in_frames/hdr_mode"
    DIRECT_CONFIG_PATH = "./config/SVS_cam.yml"
    DIRECT_OUTPUT_FOLDER = "./out_frames"

    if DIRECT_INPUT_FOLDER and Path(DIRECT_INPUT_FOLDER).exists():
        _log.info(f"BATCH CONVERTING FILES FROM: {DIRECT_INPUT_FOLDER}")
        batch_convert_folder(DIRECT_INPUT_FOLDER, DIRECT_CONFIG_PATH, DIRECT_OUTPUT_FOLDER)
    else:
        _log.info("BATCH CONVERTING WORKING RAW FILES FROM matches.py")
        batch_convert_working_matches()

    _log.info("Batch conversion completed!")
