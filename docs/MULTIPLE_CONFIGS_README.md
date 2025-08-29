# Multiple ISP Configuration Processing

This set of scripts allows you to process the same image with multiple ISP configurations and compare the results.

## Scripts Overview

### 1. `isp_pipeline_batch_convert.py` - Main Batch Processing Script
The primary script for processing an image with multiple configurations.

**Features:**
- Process with standard configurations (svs_cam.yml, triton_490.yml, etc.)
- Create custom parameter modifications
- Automatic output organization
- Progress tracking and error handling

**Usage:**
```bash
python isp_pipeline_batch_convert.py
```

**Configuration:**
Edit the script to modify:
- `FILENAME`: The image file to process
- `CONFIGURATIONS_TO_TEST`: List of standard configs to test
- `CUSTOM_MODIFICATIONS`: Custom parameter variations

### 2. `isp_pipeline_multiple_configs.py` - Advanced Script
More advanced script with command-line arguments and additional features.

**Usage:**
```bash
# List available configurations
python isp_pipeline_multiple_configs.py --list

# Process specific configurations
python isp_pipeline_multiple_configs.py --configs svs_cam triton_490

# Process with different image
python isp_pipeline_multiple_configs.py --image frame_0000.raw

# Process all configurations
python isp_pipeline_multiple_configs.py
```

### 3. `compare_isp_results.py` - Results Comparison
Analyze and visualize the differences between processed images.

**Features:**
- Grid comparison of all results
- Side-by-side comparisons
- Statistical analysis (brightness, contrast, saturation, etc.)
- Automatic image quality metrics

**Usage:**
```bash
python compare_isp_results.py
```

### 4. `example_multiple_configs.py` - Usage Examples
Demonstrates different ways to use the multiple configuration processing.

**Examples included:**
- Basic multiple configurations
- Custom parameter modifications
- Systematic parameter sweeps

**Usage:**
```bash
python example_multiple_configs.py
```

## Quick Start Guide

### Step 1: Basic Multiple Configuration Processing
```bash
# Edit the configuration in isp_pipeline_batch_convert.py
# Then run:
python isp_pipeline_batch_convert.py
```

### Step 2: Compare Results
```bash
python compare_isp_results.py
```

### Step 3: View Results
Check the output directories:
- `./out_frames/batch_processing/` - Individual processed images
- `./out_frames/comparison/` - Comparison visualizations

## Configuration Examples

### Standard Configurations
```python
CONFIGURATIONS_TO_TEST = [
    "svs_cam.yml",
    "triton_490.yml",
    "triton_490_optimized.yml",
    "samsung.yml",
    "blackfly.yml"
]
```

### Custom Parameter Modifications
```python
CUSTOM_MODIFICATIONS = {
    "high_contrast": {
        "base_config": "svs_cam.yml",
        "changes": {
            "hdr_durand": {
                "contrast_factor": 2.0,
                "sigma_space": 10.0
            },
            "color_saturation_enhancement": {
                "saturation_gain": 2.2
            }
        }
    },
    "low_noise": {
        "base_config": "svs_cam.yml",
        "changes": {
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
    }
}
```

## Output Structure

```
out_frames/
├── batch_processing/
│   ├── svs_cam/
│   │   └── Out_frame_2880_fsin_38361195454327480.png
│   ├── triton_490/
│   │   └── Out_frame_2880_fsin_38361195454327480.png
│   ├── svs_high_contrast/
│   │   └── Out_frame_2880_fsin_38361195454327480.png
│   └── ...
└── comparison/
    ├── comparison_grid.png
    ├── compare_svs_cam_vs_triton_490.png
    ├── image_statistics.png
    └── ...
```

## Common Use Cases

### 1. Camera Comparison
Compare how different camera configurations (SVS, Triton, Samsung) process the same image.

### 2. Parameter Optimization
Systematically test different parameter values to find optimal settings.

### 3. Quality Assessment
Compare image quality metrics across different configurations.

### 4. Algorithm Development
Test new algorithm variations against baseline configurations.

## Tips and Best Practices

1. **Start Small**: Begin with 2-3 configurations to test the workflow.

2. **Use Meaningful Names**: Give custom configurations descriptive names.

3. **Monitor Resources**: Processing multiple configurations can be resource-intensive.

4. **Check Compatibility**: Ensure your image format is compatible with all configurations.

5. **Backup Results**: Keep copies of important comparison results.

## Troubleshooting

### Common Issues

1. **Configuration Not Found**
   - Check that config files exist in `./config/`
   - Verify file names match exactly

2. **Image Processing Errors**
   - Ensure image format is compatible
   - Check sensor parameters in configuration

3. **Memory Issues**
   - Process fewer configurations at once
   - Use smaller images for testing

4. **Output Directory Issues**
   - Ensure write permissions
   - Check available disk space

### Getting Help

If you encounter issues:
1. Check the console output for error messages
2. Verify all dependencies are installed
3. Ensure file paths are correct
4. Test with a simple configuration first

## Dependencies

Make sure you have the following packages installed:
```bash
pip install opencv-python matplotlib numpy pyyaml
```

## Advanced Usage

### Custom Analysis
You can modify `compare_isp_results.py` to add custom analysis functions.

### Batch Processing Multiple Images
Modify the scripts to process multiple images with the same configurations.

### Integration with Other Tools
The output structure makes it easy to integrate with other image analysis tools. 