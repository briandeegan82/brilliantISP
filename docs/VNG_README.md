# VNG Demosaic - Quick Start Guide

## What is VNG?

VNG (Variable Number of Gradients) is an advanced demosaicing algorithm that excels at preserving edges and reducing color artifacts. It's now available in the brilliantISP pipeline!

## Quick Start

### 1. Update Your Config

Edit your pipeline config YAML:

```yaml
- name: demosaic
  is_save: true
  algorithm: vng_opt  # Recommended: optimized version
```

### 2. Run Your Pipeline

```bash
python isp_pipeline.py --config config/your_config.yml
```

### 3. Test It Out

Run the test suite to see VNG in action:

```bash
python test_vng_demosaic.py
```

This generates comparison images in `module_output/`:
- `vng_demosaic_comparison.png` - Compare all algorithms
- `vng_edge_preservation.png` - See edge preservation quality

## Algorithm Options

| Name | Description | Speed | Quality |
|------|-------------|-------|---------|
| `vng` | Reference implementation | Slow | Excellent |
| `vng_opt` | Optimized (recommended) | Fast | Excellent |
| `malvar` | Malvar-He-Cutler | Fast | Excellent |
| `bilinear` | Simple interpolation | Very Fast | Good |

## When to Use VNG

✓ **Best for:**
- Images with strong edges (buildings, text, graphics)
- When color accuracy is critical
- High-quality photography
- Scenes with fine details

✗ **Consider alternatives for:**
- Real-time video processing (use `bilinear`)
- When speed is critical (use `malvar`)
- Low-end hardware constraints

## Performance

On a typical 1920×1080 image:
- **VNG Optimized**: ~300ms
- **Malvar**: ~200ms  
- **Bilinear**: ~50ms

## Documentation

- **Full Documentation**: `docs/VNG_DEMOSAIC.md`
- **Config Examples**: `docs/VNG_DEMOSAIC_CONFIG_EXAMPLES.md`
- **Implementation Summary**: `VNG_IMPLEMENTATION_SUMMARY.md`

## Example Code

```python
from modules.demosaic.vng_demosaic import VNGDemosaicOptimized

# Create your Bayer masks (mask_r, mask_g, mask_b)
vng = VNGDemosaicOptimized(raw_bayer_image, masks)
rgb_output = vng.apply_vng_optimized()
```

## Questions?

Check the full documentation in `docs/VNG_DEMOSAIC.md` or run the test suite to see examples.

---

**Author**: Brian Deegan  
**Date**: March 2026  
**Status**: Production Ready
