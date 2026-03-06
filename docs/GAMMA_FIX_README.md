# Gamma Correction Pipeline Fix - Documentation Index

This directory contains documentation for the gamma correction pipeline fix implemented in BrilliantISP.

## Quick Links

### 📋 Quick Reference
**[GAMMA_CORRECTION_QUICK_REFERENCE.md](GAMMA_CORRECTION_QUICK_REFERENCE.md)**
- TL;DR summary
- Pipeline diagram
- Key code snippets
- Comparison table with Infinite-ISP

### 📖 Complete Documentation
**[GAMMA_CORRECTION_FINAL_SOLUTION.md](GAMMA_CORRECTION_FINAL_SOLUTION.md)**
- Full problem description
- Detailed solution explanation
- Technical justification
- Implementation details
- Testing guidelines
- References

### 🔍 Technical Analysis
**[GAMMA_PIPELINE_ANALYSIS.md](GAMMA_PIPELINE_ANALYSIS.md)**
- Original problem analysis
- Options considered
- Decision rationale
- Comparison with reference implementation

## Summary

### The Problem
Gamma correction was applied too early (after tone mapping, before CSC), causing:
1. Color Space Conversion to work on gamma-corrected RGB (mathematically incorrect per BT.601/709)
2. Auto-Exposure to analyze non-linear data (less accurate metering)

### The Solution
1. Keep gamma late in pipeline (after YUV→RGB conversion)
2. Add explicit 16-bit→8-bit linear conversion before CSC
3. CSC operates on linear RGB (correct color math)
4. Gamma acts as proper OETF encoding

### Key Benefits
- ✅ Standards-compliant (ITU-R BT.601/709, IEC 61966-2-1)
- ✅ Better color accuracy
- ✅ More accurate auto-exposure
- ✅ Proper display encoding
- ✅ Clearer, more maintainable code

### Implementation
```python
# 16-bit→8-bit linear conversion (NEW)
linear_8bit = np.clip((ccm_img.astype(np.float32) / 65535.0 * 255.0), 0, 255).astype(np.uint8)

# CSC operates on linear RGB
csc = CSC(linear_8bit, ...)

# Gamma applied after YUV→RGB conversion
gmc = GC(rgbc_img, ...)  # 8-bit linear → 8-bit gamma
```

## Files Modified

1. `brilliant_isp.py` - Pipeline order changes
2. `modules/gamma_correction/gamma_correction.py` - Bit depth detection
3. `modules/rgb_conversion/rgb_conversion.py` - Enhanced logging
4. `config/svs_cam.yml`, `config/triton_lab.yml` - Documentation updates

## Comparison with Industry Standards

### Our Implementation (BrilliantISP)
```
Tone Mapping → AE (linear) → 16→8 linear → CSC (linear) → 
YUV processing → RGB Conv → Gamma → Output
```
✅ Follows professional cinema/broadcast standards

### Reference Implementation (Infinite-ISP)
```
CCM → Gamma → AE → CSC (gamma-corrected) → 
YUV processing → RGB Conv → Output
```
⚠️ Pragmatic but non-standard (common in mobile ISPs)

## Testing

Run the pipeline and verify:
- Images are properly exposed (not dark)
- Colors appear natural (no corruption)
- Auto-exposure provides appropriate feedback
- Output quality meets expectations

## References

- [Infinite-ISP GitHub](https://github.com/10x-Engineers/Infinite-ISP)
- ITU-R BT.601: Standard-definition television
- ITU-R BT.709: High-definition television  
- IEC 61966-2-1: sRGB color space standard
- Charles Poynton: "Digital Video and HD: Algorithms and Interfaces"

## Questions?

See the detailed documentation files for:
- **How the conversion is done**: `GAMMA_CORRECTION_QUICK_REFERENCE.md`
- **Why we made these choices**: `GAMMA_PIPELINE_ANALYSIS.md`
- **Complete technical details**: `GAMMA_CORRECTION_FINAL_SOLUTION.md`

---

**Last Updated**: 2026-03-06  
**Implementation Version**: BrilliantISP v1.0 (with gamma fix)
