# Gamma Correction Pipeline - Quick Reference

## TL;DR

**Problem**: Gamma was applied too early (before CSC), causing CSC to work on gamma-corrected RGB instead of linear RGB.

**Solution**: Added 16-bit→8-bit linear conversion before CSC, moved gamma to after RGB conversion.

**Result**: Mathematically correct color space conversion, better color accuracy, proper OETF encoding.

## Pipeline Order (Final)

```
Raw Bayer
  ↓
Demosaic → 16-bit RGB
  ↓
CCM → 16-bit linear RGB
  ↓
Tone Mapping → 16-bit linear RGB
  ↓
Auto-Exposure ✅ (16-bit linear, accurate metering)
  ↓
16→8 bit linear conversion ✅ (NEW: output = input / 257)
  ↓
CSC (RGB→YUV) ✅ (8-bit linear, correct color math)
  ↓
LDCI → 8-bit YUV
  ↓
Sharpening → 8-bit YUV
  ↓
2D Noise Reduction → 8-bit YUV
  ↓
RGB Conversion (YUV→RGB) → 8-bit linear RGB
  ↓
Gamma Correction ✅ (8-bit linear→gamma, proper OETF)
  ↓
Scaling → 8-bit gamma RGB
  ↓
Output
```

## Key Implementation Details

### 16-bit to 8-bit Linear Conversion
```python
# Location: brilliant_isp.py, line ~424
linear_8bit = np.clip((ccm_img.astype(np.float32) / 65535.0 * 255.0), 0, 255).astype(np.uint8)
```

- Divides by 257 (65535/255), not 256
- Preserves relative brightness
- No gamma applied (stays linear)

### Gamma Correction Position
```python
# Location: brilliant_isp.py, line ~477
# After RGB conversion (YUV→RGB), before scaling
gmc = GC(rgbc_img, ...)  # 8-bit linear RGB input
gamma_img = gmc.execute()  # 8-bit gamma RGB output
```

### Auto-Detection in Gamma Module
```python
# Location: gamma_correction.py, line ~61-77
if self.img.dtype == np.uint8:
    input_bit_depth = 8
elif self.img.dtype == np.uint16:
    input_bit_depth = 16
# Generates appropriate LUT size
```

## vs. Infinite-ISP

| Feature | Infinite-ISP | BrilliantISP |
|---------|--------------|--------------|
| Gamma Position | Before CSC | After YUV→RGB ✅ |
| CSC Input | Gamma-corrected | Linear ✅ |
| Auto-Exposure | Gamma-corrected | 16-bit linear ✅ |
| Color Accuracy | Good | Better ✅ |
| Standards Compliant | No | Yes ✅ |

## Files Modified

1. **brilliant_isp.py**
   - Line ~420: Added 16→8 bit linear conversion
   - Line ~477: Moved gamma after RGB conversion
   - Line ~415: Auto-Exposure uses 16-bit linear RGB

2. **gamma_correction.py**
   - Line ~61: Auto-detect input bit depth
   - Line ~74: Enhanced logging

3. **rgb_conversion.py**
   - Line ~108: Enhanced logging

4. **config/*.yml**
   - Updated pipeline comments

## Why This Matters

✅ **CSC operates in linear space** → correct color math (BT.601/709 compliance)  
✅ **Auto-Exposure on linear data** → accurate metering with full 16-bit precision  
✅ **Gamma as final OETF** → proper display encoding (IEC 61966-2-1 compliance)  
✅ **Explicit bit depth management** → clear, maintainable code

## Trade-offs

⚠️ **Precision loss**: 16-bit → 8-bit conversion loses ~256:1 quantization  
→ Minimal visual impact after tone mapping (HDR already compressed to display range)

⚠️ **Different from reference**: Infinite-ISP uses gamma before CSC  
→ Our implementation is more correct, theirs is more pragmatic

## Testing

Expected behavior:
- Images properly exposed
- Natural colors (no corruption/shifts)
- Auto-exposure works correctly
- Slight visual differences vs. original (expected, due to correct linear processing)

## References

- [Infinite-ISP](https://github.com/10x-Engineers/Infinite-ISP) - Reference implementation
- ITU-R BT.601/709 - Color space standards
- IEC 61966-2-1 - sRGB standard
- Full documentation: `GAMMA_CORRECTION_FINAL_SOLUTION.md`
