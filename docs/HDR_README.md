# HDR Image Reader Tools

This set of tools helps you read and analyze HDR (High Dynamic Range) image files where each pixel is stored as 24 bits (3 consecutive bytes).

## Files Included

1. **`hdr_image_reader.py`** - Command-line tool for reading HDR images
2. **`explore_hdr.py`** - Interactive tool to explore and determine image properties
3. **`example_usage.py`** - Example script showing how to use the tools

## Understanding Your HDR File

Your HDR image file has the following characteristics:
- **Pixel format**: 24 bits per pixel (3 bytes per pixel)
- **Data layout**: Each pixel is stored as 3 consecutive bytes
- **Possible byte orders**: Little-endian or big-endian
- **No header**: Raw pixel data only

## Quick Start

### Option 1: Interactive Explorer (Recommended for first-time use)

```bash
python explore_hdr.py
```

This will:
1. Ask for your file path
2. Analyze the file size and suggest possible dimensions
3. Try different byte orders
4. Show you the results and ask if they look correct
5. Save the image if desired

### Option 2: Command Line Tool

If you know your image dimensions:

```bash
python hdr_image_reader.py your_file.raw 1920 1080 --show
```

Or save to PNG:

```bash
python hdr_image_reader.py your_file.raw 1920 1080 --output output.png --normalize log
```

### Option 3: Example Script

```bash
python example_usage.py
```

This will try to read existing files in your project and show examples.

## Command Line Options

### `hdr_image_reader.py`

```bash
python hdr_image_reader.py <file_path> <width> <height> [options]
```

**Required arguments:**
- `file_path`: Path to your HDR image file
- `width`: Image width in pixels
- `height`: Image height in pixels

**Optional arguments:**
- `--byte-order {little,big}`: Byte order (default: little)
- `--output <file>`: Save as PNG file
- `--normalize {linear,log,sqrt}`: Normalization method (default: linear)
- `--show`: Display the image

**Examples:**

```bash
# Basic usage with display
python hdr_image_reader.py image.raw 1920 1080 --show

# Save with logarithmic normalization
python hdr_image_reader.py image.raw 1920 1080 --output result.png --normalize log

# Try big-endian byte order
python hdr_image_reader.py image.raw 1920 1080 --byte-order big --show
```

## Determining Image Dimensions

If you don't know your image dimensions:

1. **Calculate total pixels**: `file_size_in_bytes ÷ 3`
2. **Find reasonable dimensions**: Use common aspect ratios like 16:9, 4:3, 3:2, etc.

**Example:**
- File size: 6,220,800 bytes
- Total pixels: 6,220,800 ÷ 3 = 2,073,600 pixels
- Possible dimensions:
  - 1920×1080 (16:9) = 2,073,600 pixels ✓
  - 1440×1440 (1:1) = 2,073,600 pixels ✓
  - 1616×1293 (5:4) = 2,073,600 pixels ✓

## Byte Order

The tools support both byte orders:

- **Little-endian**: Least significant byte first (common on x86 systems)
- **Big-endian**: Most significant byte first (common on some embedded systems)

**Example for pixel value 0x123456:**
- Little-endian: `[0x56, 0x34, 0x12]`
- Big-endian: `[0x12, 0x34, 0x56]`

## Normalization Methods

Since HDR images have a wide dynamic range, normalization is needed for display:

- **Linear**: Simple linear scaling (good for most cases)
- **Log**: Logarithmic scaling (good for very high dynamic range)
- **Sqrt**: Square root scaling (good for moderate dynamic range)

## Troubleshooting

### File size doesn't match expected size
- Check if your dimensions are correct
- The file might have a header or footer
- Try different aspect ratios

### Image looks wrong
- Try the other byte order (little vs big endian)
- Check if dimensions are correct
- Try different normalization methods

### No image appears
- Check that your dimensions are correct
- Verify the file is not corrupted
- Try with a smaller test image first

## Integration with Your ISP Pipeline

These tools can be integrated with your existing ISP pipeline:

```python
from hdr_image_reader import read_hdr_image

# Read your HDR image
image = read_hdr_image("your_file.raw", width, height, byte_order='little')

# Now you can process it with your ISP pipeline
# The image will be a numpy array with shape (height, width)
```

## Example Output

When you run the tools, you'll see information like:

```
Reading HDR image: image.raw
Expected dimensions: 1920x1080
Byte order: little

Image Information:
File: image.raw
Shape: (1080, 1920)
Data type: uint32
Min value: 0
Max value: 16777215
Mean value: 12345.67
Standard deviation: 5678.90
```

## Requirements

- Python 3.6+
- numpy
- matplotlib
- pathlib (built-in)

Install requirements:
```bash
pip install numpy matplotlib
``` 