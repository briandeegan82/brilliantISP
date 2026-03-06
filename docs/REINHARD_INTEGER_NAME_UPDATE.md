# Added 'reinhard_integer' as Tone Mapper Name

## Summary

The tone mapping dispatcher now recognizes `'reinhard_integer'` as a valid tone mapper name, in addition to the legacy `'integer'` name.

## Changes Made

### 1. Tone Mapping Dispatcher (`tone_mapping.py`)

**Added new elif branch:**
```python
elif self.method == "reinhard_integer":
    from modules.tone_mapping.integer_tmo.integer_tone_mapping import IntegerReinhardToneMapping
    param_int = getattr(pipeline_self, "param_integer_tmo", {})
    if self.tone_mapping_before_demosaic:
        self.hdr = IntegerReinhardToneMapping(self.img_orig, self.platform, self.sensor_info, param_int)
    else:
        L_int = self._extract_luminance_int()
        self.hdr = IntegerReinhardToneMapping(L_int, self.platform, self.sensor_info, param_int)
    self._use_integer_tmo = True
```

**Updated module docstring:**
```
Supported tone_mapper values and their config sections:
  durand            -> hdr_durand
  aces              -> aces
  integer           -> integer_tmo (Reinhard, legacy name)
  reinhard_integer  -> integer_tmo (Reinhard, recommended name)
  aces_integer      -> aces_integer
  hable             -> hable
  hable_integer     -> hable_integer
```

**Updated error message:**
```python
raise ValueError(
    f"Unknown tone mapping method: {self.method}. "
    "Supported: 'durand', 'aces', 'integer', 'reinhard_integer', 'aces_integer', 'hable', 'hable_integer'."
)
```

### 2. Configuration Files

**`config/svs_cam.yml`:**
```yaml
tone_mapping:
  tone_mapper: 'reinhard_integer'   # Reinhard integer (recommended name)
  
# Config section remains: integer_tmo
# Both 'integer' and 'reinhard_integer' use this section
```

**`config/triton_490.yml`:**
```yaml
tone_mapping:
  tone_mapper: 'reinhard_integer'   # Updated
```

## Usage

### Both Names Work

You can now use either:

**Option 1: New name (recommended)**
```yaml
tone_mapping:
  tone_mapper: 'reinhard_integer'
  
integer_tmo:
  knee: 0.25
  strength: 1.0
  normalize_output: true
```

**Option 2: Legacy name (for compatibility)**
```yaml
tone_mapping:
  tone_mapper: 'integer'
  
integer_tmo:
  knee: 0.25
  strength: 1.0
  normalize_output: true
```

Both point to the same implementation and use the same `integer_tmo:` config section.

## Behavior

### Class Selection

- `'integer'` → Uses `IntegerToneMapping` (the alias)
- `'reinhard_integer'` → Uses `IntegerReinhardToneMapping` (the actual class)

Both are functionally identical - the alias ensures backward compatibility.

### Config Section

Both tone mapper names read settings from:
```yaml
integer_tmo:
  # All Reinhard settings here
```

This maintains backward compatibility while allowing the more descriptive name.

## Verification

### Test Import
```bash
python3 -c "from modules.tone_mapping.tone_mapping import ToneMapping; print('OK')"
```

### Check Supported Methods
The module docstring lists all supported tone_mapper values:
```python
import modules.tone_mapping.tone_mapping as tm
print(tm.__doc__)
```

### Run Pipeline
Your config now uses:
```yaml
tone_mapper: 'reinhard_integer'
```

And the pipeline will:
- ✅ Recognize 'reinhard_integer' as valid
- ✅ Load settings from integer_tmo section
- ✅ Use IntegerReinhardToneMapping class
- ✅ Apply Reinhard tone mapping with your settings

## Advantages of 'reinhard_integer'

1. **Clearer naming**: Explicitly states the algorithm (Reinhard)
2. **Self-documenting**: Config is more readable
3. **Academic clarity**: Matches the paper reference
4. **Distinguishes from other integer TMOs**: Not confused with ACES/Hable integer

## Backward Compatibility

- ✅ Old configs using `'integer'` still work
- ✅ No breaking changes
- ✅ Both names supported indefinitely
- ✅ Same config section (`integer_tmo`)
- ✅ Same implementation code

## Recommendation

For new configurations, use:
```yaml
tone_mapper: 'reinhard_integer'
```

This makes it immediately clear that you're using the Reinhard operator.

## Your Current Config

Your `svs_cam.yml` is now set to:
```yaml
tone_mapping:
  tone_mapper: 'reinhard_integer'

integer_tmo:
  is_enable: true
  is_plot_curve: true
  knee: 0.25
  strength: 1.0
  normalize_output: true
```

This should work perfectly! The pipeline will recognize `'reinhard_integer'` and use the Reinhard tone mapper with your settings.
