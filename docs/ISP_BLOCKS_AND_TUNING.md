# ISP blocks: behavior and tuning

This document describes each processing block in **BrilliantISP** as wired in `brilliant_isp.py`, the YAML keys that control it, and practical tuning notes. Implementation details live under `modules/`; configuration merges from `config/base_hdr.yml` plus optional camera overlays (see below).

**BrilliantISP** is developed by **Brian Deegan**; portions of the codebase derive from **Infinite-ISP** (10xEngineers).

## How configuration is loaded

- **Merged base + camera**: If you pass a file whose name ends in `_cam.yml` (e.g. `config/AD_cam.yml`), `util/config_merge.py` loads `config/base_hdr.yml` first, then merges your camera file so later keys override earlier ones.
- **SDR-style base**: `config/base_sdr.yml` is an alternate full template (see its header comment). Use it by passing an explicit path list, e.g. `config_path=["config/base_sdr.yml", "config/your_overlay.yml"]`, unless you extend merge logic for a dedicated naming convention.
- **Standalone**: Any other YAML path is loaded as a single full config (must still contain all keys required by `BrilliantISP._REQUIRED_CONFIG_KEYS` in `brilliant_isp.py`).
- **Lists**: You can pass a list/tuple of paths to `BrilliantISP(..., config_path=[...])` for explicit ordering.

## End-to-end pipeline order

Order matches `BrilliantISP.run_pipeline()` (and `run_pipeline_up_to_wb()` for the prefix):

| Step | Block | YAML section(s) |
|------|--------|------------------|
| 1 | Crop | `crop` |
| 2 | Dead pixel correction | `dead_pixel_correction` |
| 3 | Black level correction | `black_level_correction` |
| 4 | Decompanding (PWC) | `companding` |
| 5 | OECF (optional LUT) | `oecf` |
| 6 | Digital gain | `digital_gain` |
| 7 | Lens shading correction | `lens_shading_correction` |
| 8 | Bayer noise reduction | `bayer_noise_reduction` |
| 9 | Auto white balance (estimates gains) | `auto_white_balance` |
| 10 | White balance (applies gains) | `white_balance` |
| 11a | **If** `tone_mapping.tone_mapping_before_demosaic: true` → tone mapping | `tone_mapping` + mapper-specific section |
| 12 | Demosaic | `demosaic` |
| 13 | Color correction matrix | `color_correction_matrix` |
| 11b | **If** `tone_mapping_before_demosaic: false` → tone mapping on RGB after CCM | same as 11a |
| 14 | Auto exposure (meters; can feed digital gain) | `auto_exposure` |
| 15 | Linear RGB 16-bit → 8-bit (for YUV path) | *(fixed scaling in code)* |
| 16 | RGB → YUV (incl. saturation) | `color_space_conversion`, `color_saturation_enhancement` |
| 17 | LDCI (CLAHE-style) | `ldci` |
| 18 | Sharpen | `sharpen` |
| 19 | 2D noise reduction (NLM) | `2d_noise_reduction` |
| 20 | RGB conversion (YUV → RGB) | `rgb_conversion` |
| 21 | Gamma | `gamma_correction` |
| 22 | Scale | `scale` |
| 23 | YUV format packing (optional) | `yuv_conversion_format` |

**Tone mapping placement:** `tone_mapping_before_demosaic: true` runs TMO on **Bayer** data (after WB, before demosaic). `false` runs TMO on **linear RGB** after CCM. Normalization uses `sensor_info.hdr_bit_depth` before demosaic and full scale after demosaic.

**Auto-exposure loop:** When `digital_gain.is_auto` and `auto_exposure.exposure_correction_mode` is `direct`, the pipeline may run **two** passes from digital gain through AE so the second pass applies the gain index chosen from the first meter reading (`rerun_from_digital_gain`, default `true`).

---

## Global settings

### `platform`

- **`filename`**: Input raw filename under `data_path`.
- **`debug_enabled`**, **`debug_log_level`**, **`debug_log_file`**: Logging.
- **`plot_histograms`**, **`histogram_show_log`**, **`histogram_show_channels`**: Optional histogram comparison (input vs output).
- **`save_format`**, **`short_output_names`**: Output naming.
- **`skip_disabled_modules`**: If `true`, disabled blocks are skipped (faster; avoids constructing some modules).

### `sensor_info`

- **`bayer_pattern`**: `rggb`, `grbg`, `gbrg`, or `bggr`.
- **`width`**, **`height`**: Frame size; must match raw buffer unless the loader infers size from file size (uint16 Bayer).
- **`bit_depth`**: Raw encoding (e.g. 12).
- **`hdr_bit_depth`**: Logical HDR range after decompanding (used for DPC, DG, AWB limits, tone-map normalization before demosaic).
- **`pipeline_rgb_bit_depth`**: Demosaic/CCM/AE linear RGB precision (typically 16).
- **`output_bit_depth`**: Final display bit depth (often 8).
- **`data_format`**, **`endian_type`**, **`data_alignment`**: Raw loading (e.g. `uint16`, `ieee-le`, `MSB`/`LSB`). `MSB` right-shifts left-justified samples into the low `bit_depth` bits.
- **`horizontal_flip`**, **`vertical_flip`**: Mirror the loaded RAW/PGM Bayer plane before the rest of the pipeline. `bayer_pattern` is the CFA after these flips.

**Tuning:** Wrong `width`/`height` or `bayer_pattern` causes color fringing, wrong WB, or load failures. Align `hdr_bit_depth` with your PWC/tone-map pipeline.

---

## Block-by-block reference

### Crop (`crop`)

**Role:** Crops the raw buffer before further processing.

**Keys:** `is_enable`, `crop_x_start`, `crop_y_start`, `new_width`, `new_height`.

**Tuning:** Use for ROI or to remove optical black / junk rows. Disable for full-frame processing.

---

### Dead pixel correction (`dead_pixel_correction`)

**Role:** Detects outlier Bayer pixels (hot/cold) and replaces them using neighbors (`modules/dead_pixel_correction/`).

**Keys:** `is_enable`, `dp_threshold` (larger = fewer aggressive corrections).

**Tuning:** Increase threshold if healthy edges are “over-corrected”; decrease if stuck pixels remain. Noise can trigger false positives on very dark frames.

---

### Black level correction (`black_level_correction`)

**Role:** Subtracts per-channel black levels and clips to saturation.

**Keys:** `r_offset`, `gr_offset`, `gb_offset`, `b_offset`, `r_sat`…`b_sat`, `is_linear`.

**Tuning:** Set offsets from sensor calibration. Wrong BLC shifts color balance and wastes dynamic range.

---

### Decompanding / PWC (`companding`)

**Role:** Reverses sensor **companding** with a piecewise linear LUT (`companded_pin` → `companded_pout`) to produce linear scene-referred values (`modules/pwc_generation/pwc_generation.py`).

**Keys:** `is_enable`, `companded_pin`, `companded_pout`, `pre_linearization_black_level`, `post_linearization_black_level`, `companded`. Legacy: `pedestal` aliases `pre_linearization_black_level` when the new key is absent.

**Order:** subtract `pre_linearization_black_level` → PWC LUT → subtract `post_linearization_black_level`. Use post when the sensor black (e.g. IMX728 240 LSB) lives in the linearized domain.

**Tuning:** Must match the sensor’s compression curve. Mismatched knees cause incorrect HDR scaling and bad tone mapping. If data are already linear, disable or replace with identity-like knees.

---

### OECF (`oecf`)

**Role:** Optional **opto-electronic conversion** LUT (sensor-specific), applied after decompanding when enabled.

**Keys:** `is_enable`, LUT content (see YAML comments in `base_hdr.yml`).

**Tuning:** Enable only when you have a calibrated LUT in the correct bit range. Often left off when PWC already linearizes.

---

### Digital gain (`digital_gain`)

**Role:** Multiplies linear raw by `gain_array[current_gain_index]`; optional **auto** mode steps index using AE feedback (`modules/digital_gain/digital_gain.py`).

**Keys:** `is_enable`, `is_auto`, `gain_array`, `current_gain`, `ae_feedback`.

**Tuning:** Choose a sensible `gain_array` span for your sensor. With `is_auto` and AE `step` mode, exposure feedback nudges the index. With AE `direct` mode, `suggest_direct_gain_index` picks an index toward `target_luminance`.

---

### Lens shading correction (`lens_shading_correction`)

**Role:** Per-channel radial polynomial gain: \(g(r) = 1 + k_1 r^2 + k_2 r^4\) from center to corners (`modules/lens_shading_correction/`).

**Keys:** `is_enable`, `r_k1`/`r_k2`, `gr_k1`/`gr_k2`, `gb_k1`/`gb_k2`, `b_k1`/`b_k2`.

**Tuning:** Typical **small** positive \(k_1, k_2\) correct vignetting. Very large values (see comments in `base_hdr.yml`) can cause seams or color splits—validate against a flat-field image.

---

### Bayer noise reduction (`bayer_noise_reduction`)

**Role:** Edge-preserving **joint bilateral filter** on Bayer data; optional GPU (`joint_bf_gpu`).

**Keys:** `filter_window` (odd size), per-channel `*_std_dev_s` (spatial sigma), `*_std_dev_r` (range sigma).

**Tuning:** Larger `filter_window` and sigmas = stronger smoothing and more risk of lost detail. Increase range sigmas slightly for noisy low light; decrease if edges look mushy. **Tune before demosaic**; heavy BNR can hide demosaic errors but also blur fine texture.

---

### Auto white balance (`auto_white_balance`) and white balance (`white_balance`)

**Role:** AWB estimates R/B (or channel) gains; WB applies them (`white_balance_optimized`).

**AWB keys:** `is_enable`, `algorithm` (`grey_world`, `norm_2`, `pca`), `underexposed_percentage`, `overexposed_percentage`, `percentage` (PCA).

**WB keys:** `is_enable`, `r_gain`, `b_gain`. **`white_balance.is_auto`** is synced from `auto_white_balance.is_enable` at load.

**Tuning:** Enable AWB for general scenes; use manual `r_gain`/`b_gain` for charts or when AWB is unstable. PCA `percentage` affects outlier rejection. For HDR, extremes are clipped using `hdr_bit_depth`—keep saturation settings consistent.

---

### Tone mapping (`tone_mapping` + algorithm sections)

**Role:** Maps high dynamic range linear data to displayable range. Router: `modules/tone_mapping/tone_mapping.py`.

**`tone_mapping` keys:** `is_enable`, `tone_mapping_before_demosaic`, `tone_mapper`.

**`tone_mapper` values and config sections:**

| `tone_mapper` | Section | Notes |
|----------------|---------|--------|
| `durand` | `hdr_durand` | Local TMO; bilateral-style base/detail; `sigma_space`, `sigma_color`, `contrast_factor`, `downsample_factor` |
| `aces` | `aces` | Float ACES (Knarkowicz filmic + sRGB ODT); `exposure_adjust` / `exposure_adjustment`, `gamma` |
| `reinhard_integer` | `reinhard_integer` (or legacy `integer_tmo`) | Global Reinhard-style; `knee`, `strength`, `normalize_output` |
| `aces_integer` | `aces_integer` | LUT ACES; `exposure_adjustment`, `hdr_scale`, normalization flags |
| `hable` | `hable` | Float Uncharted 2; `exposure_bias`, `white_point` |
| `hable_integer` | `hable_integer` | Integer/LUT Hable; `hdr_scale`, normalization |

**Tuning:**

- **Before demosaic:** Can reduce Bayer dynamic range early; interacts with demosaic and CCM. **After demosaic:** More traditional for “filmic” RGB grading; often easier to reason about color.
- **Global vs local:** Durand preserves local contrast; Reinhard/Hable/ACES are smoother globally but may flatten or clip highlights differently.
- **Washout:** Try `normalize_output` / `use_normalization` on integer paths, or adjust `hdr_scale` / `aces_integer` exposure.
- **Curves:** `is_plot_curve` in some sections saves debug curve plots under `module_output` when enabled.

#### Float vs integer ACES / Hable (equivalence and matching)

The **filmic math** is the same family (Knarkowicz rational for ACES; Hable `_hable_partial` for Uncharted 2), but **`aces` vs `aces_integer`** and **`hable` vs `hable_integer`** are **not equivalent by default**: different input scaling, defaults, and LUT quantization on the integer paths. Treat them as related curves, not bit-identical swaps.

**ACES (`aces` → approximating with `aces_integer`):**

- Float path: per-image min–max normalize, then multiply by **100** before the RRT curve, then sRGB-like gamma (`gamma`, default 2.4).
- Integer path: RRT is precomputed into a LUT over \([0, \texttt{hdr\_scale}]\). The shipped default **`hdr_scale` is often 1.0**, while float effectively drives the curve with inputs up to **100** after normalization—so align **`hdr_scale: 100`** (or the same numeric span you want as float’s post-normalize scale) if you want similar global contrast.
- Keep **`use_normalization: true`** (default) if you want frame-wise stretching similar in spirit to float’s normalize-then-scale behavior; set **`use_normalization: false`** only when you intend absolute mapping from full-range scene codes (`input_max` from `sensor_info`).
- Match exposure: use the same **`exposure_adjustment`** (or `exposure_adjust`) on both; integer applies it in fixed-point before LUT indexing.
- Keep **`apply_odt_gamma: true`** on `aces_integer` when you want the same RRT + ODT chain as float (gamma after filmic).
- **`normalize_output`** on integer rescales to full 16-bit range; float has no direct twin—disable on integer if you want closer qualitative match to float without a global output stretch.
- Expect **small differences** from LUT sampling, chained gamma LUT, and rounding even when parameters are aligned.

**Hable (`hable` → approximating with `hable_integer`):**

- Float path: operates on **luminance already normalized to about \([0,1]\)** by the tone-mapping stage.
- Integer path: LUT is built over \([0, \texttt{hdr\_scale}]\) with **`hdr_scale` default 1.0**, which matches a **normalized** input domain; indexing then maps either per-image min–max (**`use_normalization: true`**, default) or absolute HDR (**`use_normalization: false`**) into that LUT.
- Use the same **`exposure_bias`** and **`white_point`** on both; tune **`hdr_scale`** if you change how wide the LUT’s input domain should be relative to your scene encoding.
- **`normalize_output`** on integer is optional output range scaling; float has no identical step.
- Expect **LUT quantization** vs full-float Hable.

---

### Demosaic (`demosaic`)

**Role:** Bayer → RGB (`modules/demosaic/demosaic.py`). Output clipped to `pipeline_rgb_bit_depth`.

**Keys:** `algorithm` (see below), `is_save`.

**Algorithms:** `bilinear`, `malvar`, `vng`, `vng_opt`, `hamilton_adams`, `hamilton_adams_opt`, `ppg`, `ppg_opt`, `lmmse`, `lmmse_opt`, `lmmse_fast`, `ahd`, `ahd_opt`.

**Tuning:** `bilinear` is fast but soft. `malvar` is a good default. Edge-directed methods (`vng`, `hamilton_adams`, `ppg`, `ahd`, `lmmse`) reduce zippering at CPU cost. See `docs/PPG_DEMOSAIC.md`, `docs/VNG_DEMOSAIC.md`, `docs/HAMILTON_ADAMS_DEMOSAIC.md` for deep dives.

---

### Color correction matrix (`color_correction_matrix`)

**Role:** 3×3 linear RGB correction (`corrected_red/green/blue` rows). Optimized path: `color_correction_matrix_optimized`.

**Keys:** `is_enable`, nine coefficients (rows sum to 1 in convention).

**Tuning:** Calibrate with a color checker or factory matrix. Disable for quick raw pipeline debugging (identity matrix).

---

### Auto exposure (`auto_exposure`)

**Role:** After CCM, computes luminance histogram **skewness** vs a target band and returns feedback for digital gain (`modules/auto_exposure/auto_exposure.py`).

**Keys:** `is_enable`, `center_illuminance` (fraction in \([0,1]\) of meter max or absolute), `histogram_skewness` (acceptable skewness range), optional `target_luminance`, `exposure_correction_mode` (`step` vs `direct`), `rerun_from_digital_gain`.

**Tuning:** Narrow `histogram_skewness` = more aggressive correction. `center_illuminance` sets the desired “middle” of the histogram. Use **direct** mode with `digital_gain.is_auto` for one-shot gain selection; ensure `gain_array` is monotonic and covers the needed range.

---

### Color space conversion (`color_space_conversion`) and saturation (`color_saturation_enhancement`)

**Role:** Linear 8-bit RGB → YUV (BT.601 vs BT.709 via `conv_standard`), then optional **saturation** scaling on chroma (`modules/color_space_conversion/color_space_conversion.py`).

**Keys:** `conv_standard` (1 = BT.709, 2 = BT.601), `color_saturation_enhancement.saturation_gain`, `is_enable` on CSE.

**Tuning:** Match your display/export standard. `saturation_gain` > 1 boosts color; avoid large boosts before noise reduction if noise is visible in chroma.

---

### LDCI (`ldci`)

**Role:** Local dynamic contrast (CLAHE-style) on YUV path (`modules/ldci/`).

**Keys:** `is_enable`, `clip_limit`, `wind` (tile/window size).

**Tuning:** Higher `clip_limit` = stronger local contrast (can look “HDR-ish” or noisy). Increase `wind` for smoother tiles at cost of speed.

---

### Sharpen (`sharpen`)

**Role:** Unsharp masking on YUV pipeline (`modules/sharpen/`).

**Keys:** `sharpen_sigma`, `sharpen_strength`.

**Tuning:** Increase strength for crispness; combine carefully with NR2D to avoid amplifying noise.

---

### 2D noise reduction (`2d_noise_reduction`)

**Role:** Non-local means on YUV (`modules/noise_reduction_2d/`).

**Keys:** `window_size`, `patch_size`, `wts` (filtering strength).

**Tuning:** Larger window/patch = stronger denoising but slower and more blur. Apply after sharpening if you want sharpening to dominate edges; order in pipeline is fixed (sharpen → NR2D).

---

### RGB conversion (`rgb_conversion`)

**Role:** YUV → RGB after YUV-domain processing (`modules/rgb_conversion/`).

**Keys:** `is_enable`.

**Tuning:** Keep enabled for normal display path when CSC is on.

---

### Gamma correction (`gamma_correction`)

**Role:** Final **OETF**: power gamma or **sRGB** curve on 8-bit RGB (`modules/gamma_correction/`).

**Keys:** `is_enable`, `curve` (`gamma` or `srgb`), `gamma` (when `curve` is `gamma`).

**Tuning:** Use `srgb` for typical displays. See `docs/GAMMA_CORRECTION_FINAL_SOLUTION.md` for pipeline placement rationale.

---

### Scale (`scale`)

**Role:** Resize output (`bilinear` / nearest paths; optional GPU).

**Keys:** `new_width`, `new_height`, `algorithm`, `upscale_method`, `downscale_method`, `is_hardware`.

**Tuning:** Bilinear for general use; match output resolution to display or encoder.

---

### YUV conversion format (`yuv_conversion_format`)

**Role:** Packs YUV for saving (`444`, etc.).

**Keys:** `is_enable`, `conv_type`.

**Tuning:** If enabled without matching display path, you may need YUV→RGB for preview (handled in `brilliant_isp` when visualizing).

---

## Debugging tips

- Enable **`platform.debug_enabled`** and per-module **`is_debug`** where available.
- **`is_save`** on modules writes intermediate arrays when supported (see `util.utils.save_output_array`).
- **Histograms:** `plot_histograms` compares decompanded input vs final output.
- **Profiling:** See `reports/isp_block_profile_report.md` and `tools/profile_isp_blocks.py`.

## Related documents

- `README.md` — quick start and features
- `docs/PPG_DEMOSAIC.md`, `docs/VNG_DEMOSAIC.md`, `docs/HAMILTON_ADAMS_DEMOSAIC.md` — demosaic details
- `docs/GAMMA_CORRECTION_FINAL_SOLUTION.md` — gamma placement
- `config/base_hdr.yml` — full default key listing with inline comments
