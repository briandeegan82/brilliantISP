"""
File: isp_pipeline_multiple_configs.py
Description: Executes the ISP pipeline with multiple configurations on the same image
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""

import os
import time
import yaml
from pathlib import Path
from infinite_isp import InfiniteISP

# Configuration
RAW_DATA = "./in_frames/hdr_mode/"
FILENAME = 'image_0.raw'
OUTPUT_BASE_PATH = "./out_frames/multiple_configs/"

# List of configuration files to test
CONFIG_FILES = [
    "triton_490.yml"
]

# Alternative: Define specific configurations to test
CUSTOM_CONFIGS = {
    "svs_high_contrast": {
        "base_config": "triton_490.yml",
        "modifications": {
            "ldci": {
                "clip_limit": 5.0,
                "wind": 128
            },
            "color_saturation_enhancement": {
                "saturation_gain": 2.8
            },
            "digital_gain":{
                "current_gain": 14
            },
            "hdr_durand":{
                "contrast_factor": 1.9,
                "downsample_factor": 4
            },
            "color_correction_matrix": {
                "is_enable": True,
                "corrected_red": [6.5, -0.2, -0.3],
                "corrected_green": [-0.2, 6.7, -0.2],
                "corrected_blue": [-0.1, -0.5, 6.0]
            }
        }
    },
    "svs_low_contrast": {
        "base_config": "triton_490.yml",
        "modifications": {
            "ldci": {
                "clip_limit": 0.5,
                "wind": 128
            },
            "color_saturation_enhancement": {
                "saturation_gain": 0.3
            },
            "digital_gain":{
                "current_gain": 4
            }
        }
    },
    "svs_low_noise": {
        "base_config": "triton_490.yml", 
        "modifications": {
            "bayer_noise_reduction": {
                "r_std_dev_s": 1.2,
                "g_std_dev_s": 1.2,
                "b_std_dev_s": 1.2
            },
            "2d_noise_reduction": {
                "is_enable": True,
                "window_size": 11,
                "patch_size": 7
            }
        }
    },
    "svs_sharp": {
        "base_config": "triton_490.yml",
        "modifications": {
            "sharpen": {
                "sharpen_strength": 3.0,
                "sharpen_sigma": 5
            }
        }
    },
    "svs_awb_shift": {
        "base_config": "triton_490.yml",
        "modifications": {
            "color_correction_matrix": {
                "is_enable": True,
                "corrected_red": [1.5, -0.2, -0.3],
                "corrected_green": [-0.2, 1.7, -0.2],
                "corrected_blue": [-0.1, -0.5, 6.0]
            }
        }
    },
}

def load_and_modify_config(base_config_path, modifications=None):
    """
    Load a base configuration and apply modifications
    
    Args:
        base_config_path (str): Path to base configuration file
        modifications (dict): Dictionary of modifications to apply
    
    Returns:
        dict: Modified configuration
    """
    config_path = os.path.join("./config/", base_config_path)
    
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    
    if modifications:
        for section, changes in modifications.items():
            if section in config:
                config[section].update(changes)
            else:
                print(f"Warning: Section '{section}' not found in config")
    
    return config

def save_temp_config(config, config_name):
    """
    Save a temporary configuration file
    
    Args:
        config (dict): Configuration dictionary
        config_name (str): Name for the configuration
    
    Returns:
        str: Path to temporary config file
    """
    temp_config_path = f"./config/temp_{config_name}.yml"
    
    with open(temp_config_path, "w", encoding="utf-8") as file:
        yaml.dump(config, file, default_flow_style=False)
    
    return temp_config_path

def process_with_config(config_path, output_suffix=""):
    """
    Process the image with a specific configuration
    
    Args:
        config_path (str): Path to configuration file
        output_suffix (str): Suffix to add to output filename
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create output directory for this config
        config_name = Path(config_path).stem
        output_path = os.path.join(OUTPUT_BASE_PATH, config_name)
        
        print(f"\n{'='*60}")
        print(f"Processing with config: {config_name}")
        print(f"{'='*60}")
        
        # Initialize ISP with the configuration
        infinite_isp = InfiniteISP(RAW_DATA, config_path, output_path)
        
        # Execute the pipeline
        start_time = time.time()
        infinite_isp.execute(img_path=FILENAME, load_method='3byte', byte_order='big')
        end_time = time.time()
        
        print(f"✓ Successfully processed with {config_name}")
        print(f"  Processing time: {end_time - start_time:.2f} seconds")
        print(f"  Output saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error processing with {config_name}: {str(e)}")
        return False

def process_multiple_configs():
    """
    Process the image with multiple configurations
    """
    print(f"Processing image: {FILENAME}")
    print(f"Available configurations: {len(CONFIG_FILES)}")
    print(f"Custom configurations: {len(CUSTOM_CONFIGS)}")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)
    
    successful_configs = []
    failed_configs = []
    
    # Process with standard configurations
    print(f"\n{'='*60}")
    print("PROCESSING WITH STANDARD CONFIGURATIONS")
    print(f"{'='*60}")
    
    for config_file in CONFIG_FILES:
        config_path = os.path.join("./config/", config_file)
        
        if os.path.exists(config_path):
            success = process_with_config(config_path)
            if success:
                successful_configs.append(config_file)
            else:
                failed_configs.append(config_file)
        else:
            print(f"✗ Configuration file not found: {config_path}")
            failed_configs.append(config_file)
    
    # Process with custom configurations
    print(f"\n{'='*60}")
    print("PROCESSING WITH CUSTOM CONFIGURATIONS")
    print(f"{'='*60}")
    
    temp_configs = []
    
    for config_name, config_spec in CUSTOM_CONFIGS.items():
        try:
            # Load and modify base configuration
            config = load_and_modify_config(
                config_spec["base_config"], 
                config_spec["modifications"]
            )
            
            # Save temporary configuration
            temp_config_path = save_temp_config(config, config_name)
            temp_configs.append(temp_config_path)
            
            # Process with modified configuration
            success = process_with_config(temp_config_path, f"_{config_name}")
            
            if success:
                successful_configs.append(config_name)
            else:
                failed_configs.append(config_name)
                
        except Exception as e:
            print(f"✗ Error creating custom config '{config_name}': {str(e)}")
            failed_configs.append(config_name)
    
    # Clean up temporary files
    for temp_config in temp_configs:
        try:
            os.remove(temp_config)
        except:
            pass
    
    # Print summary
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total configurations processed: {len(successful_configs) + len(failed_configs)}")
    print(f"Successful: {len(successful_configs)}")
    print(f"Failed: {len(failed_configs)}")
    
    if successful_configs:
        print(f"\nSuccessful configurations:")
        for config in successful_configs:
            print(f"  ✓ {config}")
    
    if failed_configs:
        print(f"\nFailed configurations:")
        for config in failed_configs:
            print(f"  ✗ {config}")
    
    print(f"\nOutput directory: {OUTPUT_BASE_PATH}")

def process_specific_configs(config_list):
    """
    Process with only specific configurations
    
    Args:
        config_list (list): List of configuration names to process
    """
    print(f"Processing image: {FILENAME}")
    print(f"Selected configurations: {config_list}")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)
    
    successful_configs = []
    failed_configs = []
    
    for config_name in config_list:
        # Check if it's a standard config
        if config_name in [Path(f).stem for f in CONFIG_FILES]:
            config_path = os.path.join("./config/", f"{config_name}.yml")
            success = process_with_config(config_path)
        # Check if it's a custom config
        elif config_name in CUSTOM_CONFIGS:
            try:
                config_spec = CUSTOM_CONFIGS[config_name]
                config = load_and_modify_config(
                    config_spec["base_config"], 
                    config_spec["modifications"]
                )
                temp_config_path = save_temp_config(config, config_name)
                success = process_with_config(temp_config_path, f"_{config_name}")
                os.remove(temp_config_path)
            except Exception as e:
                print(f"✗ Error with custom config '{config_name}': {str(e)}")
                success = False
        else:
            print(f"✗ Configuration '{config_name}' not found")
            success = False
        
        if success:
            successful_configs.append(config_name)
        else:
            failed_configs.append(config_name)
    
    # Print summary
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Successful: {len(successful_configs)}")
    print(f"Failed: {len(failed_configs)}")
    print(f"Output directory: {OUTPUT_BASE_PATH}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process image with multiple ISP configurations")
    parser.add_argument("--configs", nargs="+", help="Specific configurations to process")
    parser.add_argument("--image", default=FILENAME, help="Image filename to process")
    parser.add_argument("--list", action="store_true", help="List available configurations")
    
    args = parser.parse_args()
    
    # Update filename if provided
    if args.image != FILENAME:
        FILENAME = args.image
    
    if args.list:
        print("Available standard configurations:")
        for config in CONFIG_FILES:
            print(f"  - {Path(config).stem}")
        
        print("\nAvailable custom configurations:")
        for config_name in CUSTOM_CONFIGS.keys():
            print(f"  - {config_name}")
        
        print(f"\nCurrent image: {FILENAME}")
    
    elif args.configs:
        # Process specific configurations
        process_specific_configs(args.configs)
    
    else:
        # Process all configurations
        process_multiple_configs() 