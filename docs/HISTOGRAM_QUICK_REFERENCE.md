# Histogram Debug Feature - Quick Reference

## Quick Setup

### 1. Enable in Config
```yaml
platform:
  plot_histograms: true       # Enable histogram plotting
  histogram_show_log: true    # Show log scale
```

### 2. Run Pipeline
```bash
source venv/bin/activate
python isp_pipeline.py
```

### 3. Check Output
Histograms saved to: `module_output/{filename}_histogram_comparison.png`

## Output Interpretation

### Dynamic Range (EV)
- **What it is**: Ratio between brightest and darkest values in stops
- **Formula**: DR (EV) = log₂(max / min)
- **Typical values**:
  - HDR sensor input: 16-20 EV
  - SDR output: 6-8 EV
  - Display capability: 8 EV (8-bit)

### Bit Depth Utilized
- **What it is**: Effective bits used by the data
- **Good**: Close to the nominal bit depth (e.g., 7.5-8.0 for 8-bit)
- **Bad**: Much lower (e.g., 5.0 for 8-bit indicates underutilization)

### Percentiles (0.1%, 99.9%)
- More robust than min/max
- Ignores extreme outliers
- Used for DR calculation

## Common Scenarios

### Scenario 1: Good Tone Mapping
```
Input DR: 19.26 EV
Output DR: 7.5 EV
Output bit depth: 7.9 bits
```
✓ Good utilization of output range

### Scenario 2: Clipping
```
Output Max: 255 (many pixels at 255)
Output DR: 4.5 EV
Output bit depth: 6.2 bits
```
⚠️ Highlights are clipped, adjust tone mapping

### Scenario 3: Underexposure
```
Output Max: 150
Output DR: 5.2 EV
Output bit depth: 7.2 bits
```
⚠️ Not using full output range, increase exposure/gain

### Scenario 4: Quantization
```
Histogram shows discrete "spikes" with gaps
```
⚠️ Severe quantization, check pipeline bit depth

## API Quick Reference

### Estimate Dynamic Range
```python
from util.histogram_utils import estimate_dynamic_range

dr = estimate_dynamic_range(image)
print(f"DR: {dr['dynamic_range_ev']:.2f} EV")
print(f"Min: {dr['min_val']}, Max: {dr['max_val']}")
print(f"Bit depth: {dr['bit_depth_utilized']:.1f}")
```

### Plot Comparison
```python
from util.histogram_utils import plot_histogram_comparison

input_dr, output_dr = plot_histogram_comparison(
    input_img, output_img,
    filename="my_comparison.png"
)
```

### Plot Single Image
```python
from util.histogram_utils import plot_single_histogram

dr = plot_single_histogram(
    image,
    filename="my_histogram.png",
    title="My Image",
    show_log=True
)
```

## Troubleshooting Quick Fixes

| Problem | Solution |
|---------|----------|
| No histogram generated | Set `plot_histograms: true` in config |
| Empty histogram | Check image is not all zeros |
| Weird DR values | Check image data type (uint8/uint16/uint32) |
| Plot not saved | Check `module_output/` directory exists and is writable |

## Config Options Summary

```yaml
platform:
  debug_enabled: true              # Required for all debug features
  plot_histograms: true            # Enable histogram plotting
  histogram_show_log: true         # Show log scale histograms
  histogram_show_channels: false   # Show RGB channels separately
```

## Console Output Example

```
Input Dynamic Range (after decompanding): 19.26 EV
Input Min: 1, Max: 1736000
Input Percentiles (0.1%, 99.9%): 1, 626059
Input Bit Depth Utilized: 20.7 bits

Output Dynamic Range: 7.52 EV
Output Min: 0, Max: 255
Output Percentiles (0.1%, 99.9%): 12, 248
Output Bit Depth Utilized: 8.0 bits

Histogram comparison saved to: module_output/frame_xxxx_histogram_comparison.png
```

## Test Script

Test the feature:
```bash
source venv/bin/activate
python test_histogram_feature.py
```

## Files

- **Implementation**: `util/histogram_utils.py`
- **Integration**: `brilliant_isp.py`
- **Documentation**: `docs/HISTOGRAM_DEBUG_FEATURE.md`
- **Test**: `test_histogram_feature.py`
- **Config**: `config/svs_cam.yml` (example)
