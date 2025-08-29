"""
This script is used to recursively process one .raw file per folder from a specified folder
and save ISP-converted images to a "convert" subfolder in each directory containing .raw files.
Based on isp_pipeline_mulitple_images.py
"""

import os
from pathlib import Path
from tqdm import tqdm
from infinite_isp import InfiniteISP

from util.config_utils import parse_file_name, extract_raw_metadata

# Configuration
INPUT_ROOT_PATH = "/media/brian/T7/Triton_Images"  # Root folder to search for .raw files
CONFIG_PATH = "./config/triton_490.yml"  # Default config file
VIDEO_MODE = False
EXTRACT_SENSOR_INFO = True
UPDATE_BLC_WB = True


def find_raw_files_recursively(root_path):
    """
    Recursively find one .raw file per folder in the given root path
    """
    raw_files = []
    processed_dirs = set()
    
    for root, dirs, files in os.walk(root_path):
        # Skip if we've already processed this directory
        if root in processed_dirs:
            continue
            
        # Find the first .raw file in this directory
        for file in files:
            if file.lower().endswith('.raw'):
                raw_files.append(os.path.join(root, file))
                processed_dirs.add(root)
                break  # Only take the first .raw file per directory
    
    return raw_files


def process_single_raw_file(raw_file_path, config_path):
    """
    Process a single .raw file and save the converted image to a 'convert' folder
    """
    # Get the directory containing the .raw file
    file_dir = os.path.dirname(raw_file_path)
    file_name = os.path.basename(raw_file_path)
    
    # Create 'convert' folder in the same directory as the .raw file
    convert_dir = os.path.join(file_dir, "convert")
    os.makedirs(convert_dir, exist_ok=True)
    
    # Check if there's a specific config file for this image
    file_stem = Path(file_name).stem
    config_file = os.path.join(file_dir, f"{file_stem}-configs.yml")
    
    # Initialize ISP with the file's directory and output to convert folder
    infinite_isp = InfiniteISP(file_dir, config_path, convert_dir)
    
    # Set generate_tv flag to false
    infinite_isp.c_yaml["platform"]["generate_tv"] = False
    
    # Check if specific config exists for this file
    if os.path.exists(config_file):
        print(f"Found specific config: {config_file}")
        infinite_isp.load_config(config_file)
        infinite_isp.execute(file_name, load_method='3byte', byte_order='big')
    else:
        print(f"Using default config for: {file_name}")
        
        # Extract sensor info if enabled
        if EXTRACT_SENSOR_INFO:
            if file_name.lower().endswith('.raw'):
                print(f"RAW file, extracting sensor info from filename: {file_name}")
                sensor_info = parse_file_name(file_name)
                if sensor_info:
                    infinite_isp.update_sensor_info(sensor_info)
                    print("Updated sensor_info in config")
                else:
                    print("No information in filename - sensor_info not updated")
            else:
                sensor_info = extract_raw_metadata(raw_file_path)
                if sensor_info:
                    infinite_isp.update_sensor_info(sensor_info, UPDATE_BLC_WB)
                    print("Updated sensor_info in config")
                else:
                    print("Not compatible file for metadata - sensor_info not updated")
        
        infinite_isp.execute(file_name, load_method='3byte', byte_order='big')


def batch_convert_raw_files():
    """
    Recursively find all .raw files in INPUT_ROOT_PATH and convert them
    """
    print(f"Searching for .raw files in: {INPUT_ROOT_PATH}")
    
    # Find all .raw files recursively
    raw_files = find_raw_files_recursively(INPUT_ROOT_PATH)
    
    if not raw_files:
        print(f"No .raw files found in {INPUT_ROOT_PATH}")
        return
    
    print(f"Found {len(raw_files)} folders with .raw files to process (one file per folder)")
    
    # Process each .raw file
    for raw_file in tqdm(raw_files, desc="Converting .raw files (one per folder)", ncols=100):
        try:
            print(f"\nProcessing: {raw_file}")
            process_single_raw_file(raw_file, CONFIG_PATH)
            print(f"Successfully processed: {raw_file}")
        except Exception as e:
            print(f"Error processing {raw_file}: {str(e)}")
            continue


def find_files(filename, search_path):
    """
    This function is used to find the files in the search_path
    """
    for _, _, files in os.walk(search_path):
        if filename in files:
            return True
    return False


if __name__ == "__main__":
    print("BATCH CONVERTING RAW FILES RECURSIVELY")
    print(f"Input root path: {INPUT_ROOT_PATH}")
    print(f"Default config: {CONFIG_PATH}")
    
    # Check if input path exists
    if not os.path.exists(INPUT_ROOT_PATH):
        print(f"Error: Input path {INPUT_ROOT_PATH} does not exist!")
        exit(1)
    
    batch_convert_raw_files()
    print("\nBatch conversion completed!") 