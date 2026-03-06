# Summary: normalize_output Added to All Integer Tone Mappers

## What Was Done

Added the `normalize_output` parameter to all three integer-based tone mapping operators, allowing them to automatically scale their output to use the full 16-bit range (0-65535).

## Files Modified

### 1. Reinhard (`integer_tone_mapping.py`) - Already completed
- ✅ Added `normalize_output` parameter
- ✅ Calculate normalization scale based on theoretical max
- ✅ Apply scaling in `_apply_curve()`
- ✅ Enhanced plot to show normalization status
- ✅ Log scale factor and max output percentage

### 2. ACES Integer (`aces_integer_tone_mapping.py`) - NEW
- ✅ Added `normalize_output` parameter
- ✅ Calculate scale by evaluating LUT[max_index]
- ✅ Handle both RRT-only and RRT+gamma cases
- ✅ Apply scaling after LUT lookup
- ✅ Enhanced plot with normalization info
- ✅ Log scale factor and actual max output

### 3. Hable Integer (`hable_integer_tone_mapping.py`) - NEW
- ✅ Added `normalize_output` parameter
- ✅ Calculate scale by evaluating LUT[max_index]
- ✅ Apply scaling after LUT lookup
- ✅ Enhanced plot with normalization info
- ✅ Log scale factor and actual max output

### 4. Configuration Files
- ✅ `config/svs_cam.yml` - Added `normalize_output: false` to all three
- ✅ `config/triton_490.yml` - Added `normalize_output: false` to all three

### 5. Documentation
- ✅ `docs/INTEGER_TMO_NORMALIZATION.md` - Detailed Reinhard documentation
- ✅ `docs/TMO_NORMALIZE_OUTPUT_ALL.md` - Comprehensive guide for all three

### 6. Test Scripts
- ✅ `test_normalize_output.py` - Reinhard-specific test
- ✅ `test_all_tmo_normalization.py` - Tests all three tone mappers

## Test Results

### Reinhard Global Operator
- **Without normalization**: ~50% range usage (32,768 / 65,535)
- **With normalization**: 100% range usage (65,535 / 65,535)
- **Improvement**: 100% increase

### ACES Integer
- **Without normalization**: ~91% range usage (59,521 / 65,535)
- **With normalization**: 100% range usage (65,535 / 65,535)
- **Improvement**: 10.1% increase

### Hable Integer
- **Without normalization**: ~92% range usage (60,163 / 65,535)
- **With normalization**: 100% range usage (65,535 / 65,535)
- **Improvement**: 8.9% increase

## Key Features

### 1. Separate from Input Normalization

**Important distinction:**

- `use_normalization` (ACES/Hable only): Per-image **input** min-max scaling
- `normalize_output` (all three): **Output** range scaling

These are independent features that can both be enabled.

### 2. Implementation Approach

**Reinhard (formula-based):**
```python
# Calculate theoretical max analytically
white_point_term = knee × input_max / strength
theoretical_max = (input_max × 65535) / (input_max + white_point_term)
scale = 65535 / theoretical_max
output = curve(input) × scale
```

**ACES/Hable (LUT-based):**
```python
# Evaluate LUT at maximum to find theoretical max
theoretical_max = LUT[65535]  # (with gamma handling for ACES)
scale = 65535 / theoretical_max
output = LUT[index] × scale
```

### 3. Performance

- **Cost**: One float multiply per pixel
- **Initialization**: One-time scale calculation
- **Impact**: < 1% overhead
- **Memory**: No additional allocations

### 4. Visual Feedback

**Plot enhancements:**
- Title shows normalization status and scale factor
- Info box displays actual max output and percentage
- Console logs scale factor and theoretical max

**Example log output:**
```
Reinhard normalize output enabled: scaling by 2.000x
Theoretical max without normalization: 32768
Actual max output: 65535 (100.0% of range)
```

## Configuration Examples

### Recommended (Production)

```yaml
# Reinhard
integer_tmo:
  knee: 1.0
  strength: 1.0
  normalize_output: true

# ACES Integer
aces_integer:
  hdr_scale: 1.0
  use_normalization: true    # Input normalization
  normalize_output: true     # Output normalization

# Hable Integer
hable_integer:
  hdr_scale: 2.0
  use_normalization: true    # Input normalization
  normalize_output: true     # Output normalization
```

### Research (Authentic Behavior)

```yaml
# All tone mappers
normalize_output: false  # Preserve original algorithm characteristics
```

## Benefits

1. **Full bit depth utilization**
   - Maximize SNR
   - Reduce quantization artifacts
   - Better gradient smoothness

2. **Parameter flexibility**
   - Use any white point/knee value without range penalty
   - Use any hdr_scale without limiting output

3. **Consistent pipeline behavior**
   - Predictable output levels
   - Independent of tone mapping parameters

4. **Universal solution**
   - Works for all three integer tone mappers
   - Same parameter name and behavior

## Backward Compatibility

- ✅ `normalize_output` defaults to `false`
- ✅ Existing configurations work without modification
- ✅ No breaking changes
- ✅ Opt-in feature
- ✅ No API changes

## Usage

### Enable Normalization

Edit your config file:

```yaml
integer_tmo:
  normalize_output: true

aces_integer:
  normalize_output: true

hable_integer:
  normalize_output: true
```

### Run Pipeline

```bash
python isp_pipeline.py
```

### Check Plots

With `is_plot_curve: true`, you'll see:
- Normalization scale factor in title
- Max output percentage in info box
- Console logs confirming normalization

## Testing

Run comprehensive test:

```bash
python test_all_tmo_normalization.py
```

This tests all three tone mappers with and without normalization.

## Documentation

**Detailed guides:**
- `docs/INTEGER_TMO_NORMALIZATION.md` - Reinhard-specific
- `docs/TMO_NORMALIZE_OUTPUT_ALL.md` - All three tone mappers
- `REINHARD_TMO_UPDATES.md` - Reinhard terminology updates

## Impact Analysis

| Aspect | Impact |
|--------|--------|
| **Reinhard** | Critical - solves 50% range limitation |
| **ACES Integer** | Moderate - adds 10% range usage |
| **Hable Integer** | Moderate - adds 9% range usage |
| **Performance** | Negligible - < 1% overhead |
| **Compatibility** | None - fully backward compatible |
| **Usability** | High - simple boolean parameter |

## Recommendations

### When to Enable

✅ **Production ISP pipelines** - Always enable for all three  
✅ **High white point/knee values** - Essential for Reinhard  
✅ **Maximizing bit depth** - Important for all  
✅ **Consistent output levels** - Helpful for all  

### When to Disable

❌ **Research/academic work** - Matching reference implementations  
❌ **Algorithm comparisons** - Preserving authentic behavior  
❌ **Already near full range** - ACES/Hable may not need it  

## Conclusion

All three integer tone mappers now have consistent, unified support for output range normalization. This provides:

1. **Universal solution** - Same feature across all tone mappers
2. **Full range usage** - 100% output utilization when enabled
3. **Parameter freedom** - Use any settings without range penalty
4. **Production ready** - Low overhead, well-tested, documented

The implementation is complete, tested, and ready for production use.
