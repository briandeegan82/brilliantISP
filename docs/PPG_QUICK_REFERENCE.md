# PPG Demosaic - Quick Reference

## What is PPG?

PPG (Patterned Pixel Grouping) is an advanced demosaic algorithm that uses **iterative refinement** with pattern recognition to achieve superior quality.

## Key Feature: Iterative Refinement

Unlike single-pass algorithms, PPG improves its estimates through multiple iterations:

```
Pass 1: Initial estimate → Good quality
Pass 2: Pattern refinement → Better quality
Pass 3: Fine tuning → Best quality
```

## Quick Start

### 1. Update Your Config

```yaml
- name: demosaic
  is_save: true
  algorithm: ppg_opt  # Recommended: 2 iterations
```

### 2. Run Your Pipeline

```bash
python isp_pipeline.py --config config/your_config.yml
```

### 3. Test It

```bash
python test_ppg_demosaic.py
```

## Algorithm Selection Guide

| Content Type | Best Algorithm | Reason |
|--------------|---------------|--------|
| Maximum Quality | **`ppg_opt`** | Iterative refinement |
| Natural Photography | `hamilton_adams_opt` | Color accuracy |
| Architecture | `vng_opt` | Edge preservation |
| General Purpose | `malvar` | Balanced |
| Real-time | `bilinear` | Fastest |

## Performance vs. Quality

| Algorithm | Speed | Quality | Note |
|-----------|-------|---------|------|
| Bilinear | ★★★★★ | ★★ | Fastest |
| Malvar | ★★★★ | ★★★★ | Balanced |
| Hamilton-Adams | ★★★ | ★★★★★ | Best color |
| VNG Opt | ★★★ | ★★★★★ | Best edges |
| **PPG** | **★★★** | **★★★★** | **1 iteration** |
| **PPG Optimized** | **★★** | **★★★★★** | **2 iterations** ⭐ |

## Typical Performance (1920×1080)

```
Bilinear:          ~50ms
Malvar:           ~200ms
Hamilton-Adams:   ~250ms
VNG Optimized:    ~300ms
PPG:              ~280ms   (1 iteration)
PPG Optimized:    ~320ms   (2 iterations) ← RECOMMENDED
```

## Available Algorithms (Now 8!)

```yaml
# Simple and fast
algorithm: bilinear

# Balanced (default)
algorithm: malvar

# Edge-directed
algorithm: vng_opt

# Best color accuracy
algorithm: hamilton_adams_opt

# Best overall quality
algorithm: ppg_opt          # ← RECOMMENDED FOR BEST QUALITY
```

## When to Use PPG

### ✓ Use PPG For:

- **Maximum quality** - Want the best available
- **Professional work** - Publication quality
- **Mixed content** - Edges + smooth regions
- **General high quality** - Best all-around
- **Final renders** - Not time-critical

### ✗ Consider Alternatives:

- **Real-time** → `bilinear` or `malvar`
- **Color-critical only** → `hamilton_adams_opt`
- **Edges-only** → `vng_opt`
- **Speed-critical** → `malvar`

## Key Advantages

✓ **Iterative refinement** - Gets better with each pass  
✓ **Pattern recognition** - Uses neighborhood information  
✓ **Superior quality** - Among the best available  
✓ **Versatile** - Works well on all content types  
✓ **Configurable** - Adjust iterations for quality/speed  

## How It Works (Simplified)

```
1. Initial green interpolation (like Hamilton-Adams)
   ↓
2. Analyze color patterns in neighborhood
   ↓
3. Compute pattern-based corrections
   ↓
4. Refine green estimates
   ↓
5. Repeat steps 2-4 (iteration)
   ↓
6. Interpolate R/B using refined green
```

## Example Code

```python
from modules.demosaic.ppg_demosaic import PPGDemosaicOptimized

ppg = PPGDemosaicOptimized(raw_bayer_image, masks)
ppg.num_iterations = 2  # Default, recommended
rgb_output = ppg.apply_ppg_optimized()
```

## Iteration Count Guide

```python
# Fast mode (1 iteration)
ppg.num_iterations = 1    # ~280ms, good quality

# Balanced mode (2 iterations) - RECOMMENDED
ppg.num_iterations = 2    # ~320ms, excellent quality

# Best quality (3 iterations)
ppg.num_iterations = 3    # ~360ms, superior quality
```

## Comparison with Other Methods

### PPG vs. All Others

| Feature | PPG | Hamilton-Adams | VNG | Malvar |
|---------|-----|----------------|-----|--------|
| Quality | ★★★★★ | ★★★★★ | ★★★★★ | ★★★★ |
| Speed | ★★★ | ★★★ | ★★★ | ★★★★ |
| Color | ★★★★ | ★★★★★ | ★★★★ | ★★★★ |
| Edges | ★★★★ | ★★★★ | ★★★★★ | ★★★★ |
| Overall | **★★★★★** | ★★★★ | ★★★★ | ★★★ |

### Winner by Category

- **Best Overall**: `ppg_opt`
- **Best Color**: `hamilton_adams_opt`
- **Best Edges**: `vng_opt`
- **Best Speed**: `bilinear`
- **Best Balance**: `malvar`

## Testing

Run comprehensive tests:

```bash
python test_ppg_demosaic.py
```

Outputs:
- `ppg_demosaic_comparison.png` - All algorithms
- `ppg_iterations.png` - Iteration comparison
- `ppg_edge_preservation.png` - Edge quality
- `ppg_pattern_sensitivity.png` - Pattern recognition

## Configuration Examples

### Maximum Quality
```yaml
- name: demosaic
  algorithm: ppg_opt
```

### Balanced Quality/Speed
```yaml
- name: demosaic
  algorithm: ppg
```

### Different Use Cases
```yaml
# For portraits
algorithm: hamilton_adams_opt

# For architecture
algorithm: vng_opt

# For best overall quality
algorithm: ppg_opt          # ← THIS ONE
```

## Tips for Best Results

1. **Use ppg_opt** for production work
2. **2 iterations** is the sweet spot
3. **Test on your content** to verify improvement
4. **Compare with malvar** to see quality gain
5. **Profile your pipeline** if speed matters

## What Makes PPG Special?

**Single-pass algorithms** (Bilinear, Malvar, H-A, VNG):
```
Raw → Process → Output (done in one pass)
```

**PPG (Iterative)**:
```
Raw → Initial estimate → Refine → Refine → Output
      (better each iteration)
```

This iterative refinement is why PPG achieves superior quality!

## Documentation

- **Full Documentation**: `docs/PPG_DEMOSAIC.md`
- **Test Script**: `test_ppg_demosaic.py`
- **Implementation**: `modules/demosaic/ppg_demosaic.py`

## Quick Decision

**"Which algorithm should I use?"**

- Need best quality? → **`ppg_opt`** ⭐
- Need best color? → `hamilton_adams_opt`
- Need best edges? → `vng_opt`
- Need fast? → `malvar`
- Need fastest? → `bilinear`

**For most production work: `ppg_opt`**

---

**Bottom Line**: PPG provides the best overall demosaic quality through iterative refinement. Use `ppg_opt` with 2 iterations for optimal results.
