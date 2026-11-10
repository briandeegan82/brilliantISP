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

**Auto-exposure loop:** When `digital_gain.is_auto` and `auto_exposure.exposure_correction_mode` is `direct`, the pipeline may run **two** passes from digital gain through AE so the second pass applies the gain index chosen from the first meter reading (`rerun_from_digital_gain`, default `true`). With `auto_exposure.smooth_control: true` the gain is continuous instead (see [Smooth AE controller](#smooth-ae-controller-auto_exposuresmooth_control)).

### Sequence processing and 3A state

The temporal controls below (smooth AE, AWB filtering, adaptive white) only act across frames when the caller threads one `util.pipeline_types.ProcessingState` through consecutive calls, e.g. `process_frame(raw, cfg, state=state)` with the same `state` object for every frame of a clip. The state carries `previous_awb_gains`, `digital_gain_index`, `ae_gain` and `tone_white`. Without it every frame starts from the configured defaults, so time constants and slew limits have no effect.

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
- **`embedded_rows_top`**, **`embedded_rows_bottom`**: Metadata rows trimmed from the raster before flips (see `docs/RAW_INPUT.md`).
- **`realign_columns_from`** (optional, default off): First column of a block that the sensor reads out one column early. Columns `[n, width)` are shifted right by one and column `n` is filled from the same CFA colour two columns to the left, restoring the CFA phase. The Limerick SVS cameras need `1888` (the last 32 columns otherwise show a magenta edge line).

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

Optional **`roi`** (see [Metering ROI](#metering-roi-auto_exposureroi-auto_white_balanceroi)): AWB samples only pixels with a positive ROI weight; positive zone weights are treated equally.

#### AWB plausibility limits and temporal filter (`modules/auto_white_balance/temporal.py`)

All optional; when none is present AWB behaves as before. Applied in this order to each new estimate:

| Key | Meaning |
|-----|---------|
| `low_light_prior: [r, b]`, `low_light_gain_range: [g0, g1]` | Blend the estimate toward fixed gains as the applied digital gain rises from `g0` to `g1` (log-linear weight 0→1). Stops night scenes lit by sodium/LED sources from dragging WB. |
| `gain_locus: [[r, b], ...]`, `locus_max_distance` | Illuminant locus polyline (≥ 2 points, gains). The estimate is pulled to within `locus_max_distance` (natural-log gain units) of the polyline. |
| `r_gain_range`, `b_gain_range: [lo, hi]` | Final plausibility box. |
| `time_constant_s`, `max_rate_per_s`, `frame_rate` | First-order IIR on log gains plus a slew limit (1/s); per-frame coefficients come from `frame_rate` (Hz). Needs sequence state. |

**Tuning:** derive the locus from AWB estimates on neutral scene content under known illuminants; a tight `locus_max_distance` (e.g. 0.10) suppresses green/magenta casts without blocking warm↔cool changes along the locus.

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

#### RGB recombination and white handling (native post-CCM path)

These `tone_mapping` keys apply when tone mapping runs on RGB after CCM:

- **`rgb_norm`**: the per-pixel value the global curve `f` is applied to; RGB is rescaled by `f(N)/N`, so hue is preserved.
  - `luminance` (default): `N = Y` (Rec.709). Saturated colours can push a channel past the output range.
  - `max_rgb`: `N = max(R, G, B)`; no channel can exceed the curve's output, at the cost of darker saturated colours.
  - `max_blend`: `N = max(Y, k * max(R, G, B))` with `rgb_norm_k = k` in [0, 1]. `k = 0` is `luminance`, `k = 1` is `max_rgb`. Monotone in exposure for every `k`, and most midtones follow the luminance path, so grey-preserving WB steps do not change their luma.
- **`near_white_desat: [d0, d1]`** (optional): pixels whose largest output channel is in `[d0, d1]` are pulled toward neutral at their own output luma, so near-white highlights roll off to white instead of a tinted plateau. Luma is unchanged, so monotonicity is kept. Caution: bright saturated reds/ambers (tail lights, signals) are also desaturated and can look grey.
- **`adaptive_white`** (optional mapping, `modules/tone_mapping/adaptive_white.py`): scene-adaptive global white point. Per frame, a high percentile (`percentile`, default 99.7) of max(R,G,B) sets the desired white W in [1, `max_white`] (default 8); W is low-pass filtered in the log domain (`time_constant_s`, default 1; `frame_rate`, default 10; `meter_stride`, default 8) and applied to the **next** frame as `f(g*n)/f(g*W)`. Needs sequence state.

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

Optional **`roi`**: the meter average and histogram moments use the ROI weights instead of the full frame.

#### Metering ROI (`auto_exposure.roi`, `auto_white_balance.roi`)

Opt-in and absent by default; when absent or `is_enable: false` the statistics use the full frame exactly as before (`util/metering_roi.py`). Coordinates are normalized to the statistics image (x: 0 left → 1 right, y: 0 top → 1 bottom) after embedded-row removal, sensor flips and crop, so the same values select the same scene region for AE (full-resolution RGB) and AWB (half-resolution CFA planes). Typical uses: limit statistics to a fisheye image circle, exclude the ego-vehicle body, or weight the lower (road) part of the frame.

```yaml
auto_exposure:
  roi:
    is_enable: true
    ellipse: {cx: 0.5, cy: 0.5, rx: 0.45, ry: 0.56}   # keep inside
    include_rect: [0.0, 0.1, 1.0, 1.0]                # keep inside, [x0, y0, x1, y1]
    exclude_rects: [[0.0, 0.0, 0.1, 0.2]]
    exclude_polygons: [[[0.0, 1.0], [0.0, 0.85], [0.5, 0.78], [1.0, 0.85], [1.0, 1.0]]]
    zone_weights: [[1, 1, 1], [1, 2, 1], [2, 3, 2]]   # rows top→bottom, nearest zone
```

All keys except `is_enable` are optional; unknown keys, out-of-range rectangles and an ROI that selects no pixels are errors. The CPU engine implements the ROI; `brilliantisp_gpu` does not read it.

#### Smooth AE controller (`auto_exposure.smooth_control`)

Opt-in (`modules/auto_exposure/smooth_control.py`); requires `auto_exposure.is_enable` and `digital_gain.is_auto`. Instead of stepping through `digital_gain.gain_array`, AE picks a **continuous** gain so that the mean **tone-mapped** luminance of the metered pixels (subsampled by `meter_stride`, weighted by `roi` when set) equals the output target. The demand is then filtered in the log2 domain: deadband → first-order IIR → slew limit. All time constants are in seconds; per-frame coefficients come from `frame_rate`, so behaviour does not depend on the frame rate. When `bayer_noise_reduction` is disabled the new gain is applied within the same pass by linear scaling; otherwise a second pass reruns from digital gain. Needs sequence state to smooth across frames.

| Key | Default | Meaning |
|-----|---------|---------|
| `output_target` | 0.18 | Target mean tone-mapped luminance (0–1, display-linear). |
| `time_constant_s` | 0.35 | IIR time constant. |
| `deadband_stops` | 0.05 | Errors smaller than this do not move the gain. |
| `max_rate_stops_per_s` | 5.0 | Slew limit. |
| `frame_rate` | 30.0 | Frame rate (Hz) used to derive per-frame coefficients. |
| `gain_min`, `gain_max` | 1/32, 16384 | Gain limits. |
| `meter_stride` | 8 | Meter subsampling. |
| `highlight_clip` | 1.0 | Tone-mapped meter values are clipped here before averaging. |
| `low_light_target_stops`, `low_light_gain_range` | 0, [1e4, 1e5] | Lower the target by up to this many stops as gain rises through the range (log-linear). |
| `target_schedule` | none | `[[gain, stops], ...]` (≥ 2 points): target = `output_target * 2**stops(gain)`, interpolated in log gain, flat beyond the ends. Overrides `low_light_target_stops`. |
| `low_light_deadband_stops` | none | Deadband widened toward this value across `low_light_gain_range`, so noisy night metering does not cause gain hunting. |
| `max_clip_fraction`, `clip_level`, `clip_max_backoff_stops` | 1.0, 0.98, 1.0 | Highlight protection: with `max_clip_fraction < 1` the gain is reduced so at most that fraction of metered pixels has a peak channel above `clip_level`, backing off at most `clip_max_backoff_stops`. |
| `model_rgb_norm` | false | Meter through the same `tone_mapping.rgb_norm` (`max_rgb`/`max_blend`) the output uses, so AE predicts output luma correctly. Use whenever `rgb_norm` is not `luminance`. |

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

## Example tuned configs: `config/r04q/`

Full (non-`extends`) configs from the 2026 3A tuning study for the Limerick vehicle, intended for sequence processing at 10 fps:

| File | Camera |
|------|--------|
| `AD.yml` | AD cameras (FL/FR/RL/RR); `rgb_norm: max_rgb`, lower-frame AE/AWB ROI |
| `SVS.yml` | SVS base (no ROI); `rgb_norm: max_blend`, `rgb_norm_k: 0.85`, `realign_columns_from: 1888` |
| `SVS_FV.yml`, `SVS_RV.yml` | SVS front/rear: base plus fisheye image-circle ROIs measured from raw |
| `SVS_MVL.yml`, `SVS_MVR.yml` | SVS mirror cameras: image circle from the vehicle intrinsics, minus car body and mirror housing |

All use smooth AE (`output_target` 0.18 AD / 0.16 SVS, SVS `target_schedule` and `model_rgb_norm`), AWB temporal filtering (`time_constant_s: 1.0`) with `r_gain_range`/`b_gain_range` limits (SVS also `gain_locus` with `locus_max_distance: 0.10`) and `near_white_desat: [0.8, 0.97]`. The first line of each file records how it was generated. Known open issues: SVS has no calibrated CCM, SVS road colour casts remain at dusk/night, and `near_white_desat` greys bright reds/ambers.

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
