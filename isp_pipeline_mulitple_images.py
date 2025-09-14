"""
This script is used to run isp_pipeline.py on a dataset placed in ./inframes/normal/data
It also fetches if a separate config of a raw image is present othewise uses the default config
"""

import os
from pathlib import Path
from tqdm import tqdm
from brilliant_isp import BrilliantISP

from util.config_utils import parse_file_name, extract_raw_metadata

# Define multiple input/output folder pairs
DATASET_CONFIGS = [
    {
        "input_path": "/media/brian/ssd-drive/colorchecker/raw_files/191-G-NUIG.RAW.DAI_3MPX_FV.BIN.20250903.151711_extracted",
        "output_path": "/media/brian/ssd-drive/UoG_drive_20250723/colorchecker/FV_ISP/"
    },
    # Add more folder pairs as needed:
    {
        "input_path": "/media/brian/ssd-drive/colorchecker/raw_files/191-G-NUIG.RAW.DAI_3MPX_MVL.BIN.20250903.151711_extracted",
        "output_path": "/media/brian/ssd-drive/UoG_drive_20250723/colorchecker/MVL_ISP/"
    },
    {
        "input_path": "/media/brian/ssd-drive/colorchecker/raw_files/191-G-NUIG.RAW.DAI_3MPX_MVR.BIN.20250903.151711_extracted",
        "output_path": "/media/brian/ssd-drive/UoG_drive_20250723/colorchecker/MVR_ISP/"
    },
    {
        "input_path": "/media/brian/ssd-drive/colorchecker/raw_files/191-G-NUIG.RAW.DAI_3MPX_RV.BIN.20250903.151711_extracted",
        "output_path": "/media/brian/ssd-drive/UoG_drive_20250723/colorchecker/RV_ISP/"
    },
]

# Examples of how to add more folder pairs:
# DATASET_CONFIGS = [
#     {
#         "input_path": "/media/brian/ssd-drive/drive/output/folder1/",
#         "output_path": "/media/brian/ssd-drive/drive/output/processed1/"
#     },
#     {
#         "input_path": "/media/brian/ssd-drive/drive/output/folder2/",
#         "output_path": "/media/brian/ssd-drive/drive/output/processed2/"
#     },
#     {
#         "input_path": "/media/brian/ssd-drive/drive/output/folder3/",
#         "output_path": "/media/brian/ssd-drive/drive/output/processed3/"
#     },
# ]

CONFIG_PATH = "./config/svs_cam.yml"
VIDEO_MODE = False
EXTRACT_SENSOR_INFO = True
UPDATE_BLC_WB = True


def video_processing(input_path, output_path):
    """
    Processed Images in a folder [input_path] like frames of an Image.
    - All images are processed with same config file located at CONFIG_PATH
    - 3A Stats calculated on a frame are applied on the next frame
    """

    raw_files = [f_name for f_name in os.listdir(input_path) if ".raw" in f_name]
    raw_files.sort()

    print(f"Processing {len(raw_files)} video frames...")
    print(f"Input directory: {input_path}")
    print(f"Output directory: {output_path}")

    brilliant_isp = BrilliantISP(DATASET_PATH, CONFIG_PATH, OUTPUT_PATH)

    # set generate_tv flag to false
    brilliant_isp.c_yaml["platform"]["generate_tv"] = False
    brilliant_isp.c_yaml["platform"]["render_3a"] = False

    for file in tqdm(raw_files, disable=False, leave=True):

        brilliant_isp.execute(file)
        brilliant_isp.load_3a_statistics()


def dataset_processing(input_path, output_path):
    """
    Processed each image as a single entity that may or may not have its config
    - If config file in the dataset folder has format filename-configs.yml it will
    be use to proocess the image otherwise default config is used.
    - For 3a-rendered output - set 3a_render flag in config file to true.
    """

    # The path for default config
    default_config = CONFIG_PATH

    # Get the list of all files in the input_path
    directory_content = os.listdir(input_path)

    # Get the list of all raw images in the input_path
    raw_images = [
        x
        for x in directory_content
        if (Path(input_path, x).suffix in [".raw", ".NEF", ".dng", ".nef"])
    ]

    brilliant_isp = BrilliantISP(DATASET_PATH, default_config, OUTPUT_PATH)

    # set generate_tv flag to false
    brilliant_isp.c_yaml["platform"]["generate_tv"] = False

    print(f"Processing {len(raw_images)} dataset images...")
    print(f"Input directory: {input_path}")
    print(f"Output directory: {output_path}")

    is_default_config = True

    for raw in tqdm(raw_images, ncols=100, leave=True):

        raw_path_object = Path(raw)
        config_file = raw_path_object.stem + "-configs.yml"

        # check if the config file exists in the input_path
        if find_files(config_file, input_path):

            print(f"Found {config_file}.")

            # use raw config file in dataset
            brilliant_isp.load_config(DATASET_PATH + config_file)
            is_default_config = False
            brilliant_isp.execute()

        else:
            print(f"Not Found {config_file}, Changing filename in default config file.")

            # copy default config file
            if not is_default_config:
                brilliant_isp.load_config(default_config)
                is_default_config = True

            if EXTRACT_SENSOR_INFO:
                if raw_path_object.suffix == ".raw":
                    print(
                        raw_path_object.suffix
                        + " file, sensor_info will be extracted from filename."
                    )
                    sensor_info = parse_file_name(raw)
                    if sensor_info:
                        brilliant_isp.update_sensor_info(sensor_info)
                        print("updated sensor_info into config")
                    else:
                        print("No information in filename - sensor_info not updated")
                else:
                    sensor_info = extract_raw_metadata(input_path + raw)
                    if sensor_info:
                        brilliant_isp.update_sensor_info(sensor_info, UPDATE_BLC_WB)
                        print("updated sensor_info into config")
                    else:
                        print(
                            "Not compatible file for metadata - sensor_info not updated"
                        )

            brilliant_isp.execute(raw)


def find_files(filename, search_path):
    """
    This function is used to find the files in the search_path
    """
    for _, _, files in os.walk(search_path):
        if filename in files:
            return True
    return False


def process_multiple_folders():
    """
    Process multiple input folders with corresponding output folders
    """
    print(f"Processing {len(DATASET_CONFIGS)} folder pairs...")
    
    for i, config in enumerate(DATASET_CONFIGS, 1):
        input_path = config["input_path"]
        output_path = config["output_path"]
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Processing folder pair {i}/{len(DATASET_CONFIGS)}")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print(f"{'='*60}")
        
        if VIDEO_MODE:
            video_processing(input_path, output_path)
        else:
            dataset_processing(input_path, output_path)
        
        print(f"Completed processing folder pair {i}/{len(DATASET_CONFIGS)}")


if __name__ == "__main__":

    if VIDEO_MODE:
        print("PROCESSING VIDEO FRAMES ONE BY ONE IN SEQUENCE")
        process_multiple_folders()

    else:
        print("PROCESSING DATSET IMAGES ONE BY ONE")
        process_multiple_folders()
