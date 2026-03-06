# Gamma Correction Pipeline Order Fix

## Issue
Gamma correction was being applied too early in the ISP pipeline, before Color Space Conversion (CSC) and other YUV-space processing operations. This is incorrect because:

1. **Color Space Conversion (RGB→YUV)** expects linear RGB data, not gamma-corrected data
2. **Auto-Exposure** should analyze linear RGB data for accurate metering
3. **YUV processing operations** (LDCI, sharpening, noise reduction) work best in linear or perceptually-uniform spaces

## Previous Pipeline Order (INCORRECT)
```
1. Demosaic (Bayer → RGB)
2. Color Correction Matrix (CCM)
3. Tone Mapping (HDR → linear 16-bit RGB)
4. ❌ Gamma Correction (applied too early)
5. ❌ Auto-Exposure (analyzing gamma-corrected data)
6. ❌ Color Space Conversion (RGB→YUV on gamma-corrected data)
7. LDCI
8. Sharpening
9. 2D Noise Reduction
10. RGB Conversion (YUV→RGB)
11. Scaling
12. YUV Conversion Format
```

## New Pipeline Order (CORRECT)
```
1. Demosaic (Bayer → RGB)
2. Color Correction Matrix (CCM)
3. Tone Mapping (HDR → linear 16-bit RGB)
4. ✅ Auto-Exposure (analyzing linear RGB data)
5. ✅ Color Space Conversion (RGB→YUV on linear data)
6. LDCI
7. Sharpening
8. 2D Noise Reduction
9. RGB Conversion (YUV→RGB) → outputs 8-bit uint8
10. ✅ Gamma Correction (applied after RGB conversion)
11. Scaling
12. YUV Conversion Format
```

## Changes Made

### 1. `brilliant_isp.py` - Pipeline Execution Order
**Lines 405-423:** Removed gamma correction from after tone mapping, moved Auto-Exposure and CSC to operate on linear RGB (`ccm_img` instead of `gamma_raw`)

**Lines 474-478:** Added gamma correction after RGB conversion (YUV→RGB), before scaling

**Lines 481-492:** Updated scaling module to use `gamma_raw` (gamma-corrected data) instead of `rgbc_img`

### 2. `modules/gamma_correction/gamma_correction.py` - Bit Depth Detection
**Lines 56-93:** Updated `apply_gamma()` to automatically detect input bit depth:
- RGB conversion outputs 8-bit uint8 data
- Gamma correction now detects this and uses an 8-bit LUT instead of 16-bit
- Added logging to debug bit depth and data range issues

### 3. `modules/rgb_conversion/rgb_conversion.py` - Enhanced Logging
**Lines 102-115:** Added logging to show input/output data types and ranges for debugging

### 4. Config Files - Documentation Updates
Updated comments in:
- `config/svs_cam.yml` line 24
- `config/triton_lab.yml` line 24

Changed comment from:
```yaml
pipeline_rgb_bit_depth: 16  # RGB stages (demosaic→gamma); see docs/PIPELINE_BIT_DEPTH_SPEC.md
```

To:
```yaml
pipeline_rgb_bit_depth: 16  # RGB stages (demosaic→CCM→CSC→YUV processing→RGB conversion→gamma)
```

## Technical Details

### Bit Depth Transition Issue
The original issue after moving gamma correction was that:
1. RGB Conversion (YUV→RGB) outputs **8-bit uint8** data (range 0-255)
2. Gamma Correction expected **16-bit uint16** input (range 0-65535)
3. When gamma tried to index into a 16-bit LUT with 8-bit values, it resulted in very dark output (values 0-255 in a 0-65535 LUT)

### Solution
Made gamma correction **adaptive**:
- Detects input bit depth from data type (`uint8` → 8-bit, `uint16` → 16-bit)
- Generates appropriate LUT size for the detected bit depth
- Properly converts to output bit depth (typically 8-bit for final display)

## Technical Justification

### Why Gamma Should Be Late in the Pipeline

1. **Linear Color Math**: Color space conversions (RGB↔YUV) use matrix multiplications that assume linear light values. Gamma-corrected values violate this assumption and produce incorrect colors.

2. **Perceptual Processing**: Operations like sharpening, noise reduction, and LDCI work better in perceptually-uniform spaces (YUV/Lab) computed from linear RGB, not gamma-corrected RGB.

3. **Auto-Exposure**: Exposure metering algorithms expect linear scene-referred values to calculate proper luminance statistics. Gamma-corrected values are display-referred and compress shadows/highlights non-linearly.

4. **HDR Tone Mapping**: Tone mappers output linear display values (0-1 range). Gamma should be the final OETF (Opto-Electronic Transfer Function) that encodes these linear values for display.

5. **Industry Standard**: Professional ISP pipelines apply gamma as one of the final stages, just before output encoding. This is the standard approach in cinema, broadcast, and high-end imaging.

### Why RGB Conversion Outputs 8-bit
The RGB Conversion module (YUV→RGB) is designed for final output preparation:
- It converts YUV back to RGB for display/saving
- The conversion uses 8-bit integer math with offsets [16, 128, 128]
- Output is clipped to [0, 255] range
- This is standard for video/display output pipelines

## Expected Impact

✅ **Improved color accuracy** in CSC output (proper RGB→YUV conversion)
✅ **Better auto-exposure** metering (analyzing linear data)
✅ **More effective YUV processing** (sharpening, LDCI work on proper perceptual space)
✅ **Correct gamma encoding** of final output
✅ **Proper bit depth handling** (8-bit to 8-bit conversion in gamma)

⚠️ **Note**: Output images may look slightly different due to the corrected processing order. This is expected and represents the proper behavior.

## Testing

To verify the fix:
1. Run the pipeline with tone mapping enabled
2. Check log messages for bit depth detection in gamma correction
3. Verify that output images are properly exposed (not dark or washed out)
4. Check that Auto-Exposure feedback values are appropriate for the scene
5. Verify that output colors look natural (no over-saturated or washed-out colors)

## References

- ITU-R BT.709: HDTV color space (assumes linear RGB input)
- ITU-R BT.601: SDTV color space (assumes linear RGB input)
- IEC 61966-2-1: sRGB standard (gamma as final encoding step)
- Industry ISP pipeline architectures (ARM, Qualcomm, etc.)

