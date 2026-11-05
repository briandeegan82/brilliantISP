# Code Audit — BrilliantISP (M0 baseline)

Status: **static source audit plus import-graph enumeration**. The findings
below are read from source and a mechanical caller scan; they have **not** been
reproduced at runtime unless explicitly stated. This document is the milestone
M0 audit ledger; per-block behavioral contracts land on the block pages in
later milestones.

Baseline commit: `47558570c5613aeb2a7b75ccbb5891d6a8737c0b`
Evidence directory (local, git-ignored): `tmp/modernization_m0/`

## 1. Baseline test run

- Command: `venv/bin/python -m pytest -p no:cacheprovider tests/test_raw_io.py tests/gui/`
  with `PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg`.
- Result: **40 passed** (8 `test_raw_io.py`, 30 `test_calibration_logic.py`,
  2 `test_gui_log.py`), ~3 s. Captured output:
  `tmp/modernization_m0/baseline_tests.txt`.
- Environment blocker: the project `venv` contains no `pytest`
  (`venv/pyvenv.cfg` has `include-system-site-packages = false`), and the
  system `python3` has `pytest` but lacks `tqdm`. Baseline was run with the
  venv interpreter plus a `PYTHONPATH` pointing at pure-Python pytest packages
  copied into `tmp/modernization_m0/pytest_pkgs/` (pytest, _pytest, pluggy,
  iniconfig, packaging, py). No packages were installed or modified. The venv
  needs a dev `pytest` install before this becomes the canonical command.
- Environment capture: `tmp/modernization_m0/environment.json` (Python 3.12.3,
  NumPy 1.26.4, SciPy 1.17.1, OpenCV 4.11.0, Matplotlib 3.10.8, PyYAML 6.0.3,
  Numba 0.64.0, rawpy 0.26.1; no CuPy; `cv2.cuda` reports 0 devices;
  12 logical CPUs (AMD Ryzen 5 7535HS), Linux x86_64; no thread-limit env
  vars set).

## 1a. Baseline processing captures

Two single executions of the existing pipeline were captured by
`tmp/modernization_m0/capture_baseline.py` (diagnostics/module saves off,
packed-YUV output off, OpenCL off, output redirected under
`tmp/modernization_m0/captures/`). Per-run details live in each
`manifest.json`; aggregate copy: `tmp/modernization_m0/baseline_captures.json`.
Captured stdout/stderr/exit: `capture_baseline_stdout.txt`,
`capture_baseline_stderr.txt`, `capture_baseline_exit.txt` (exit 0, no output).

| Capture | Source + config | Status | Result (shape, dtype, min..max) |
|---|---|---|---|
| `svs_fv` | `in_frames/hdr_mode/FV.raw` + `config/SVS_cam.yml` (1920x1536, gbrg) | success | `[1536, 1920, 3]` uint8, 0..255 |
| `ad_odd_height` | `in_frames/selected raw files/6ACE750_011_AD_FL_frame_000028.raw` + `config/AD_cam_2.yml` (3840x2185, gbrg) | success | `[2185, 3840, 3]` uint8, 0..217 |

Both manifests report `source_files_unchanged: true`; outputs
(`output_rgb.npy`, rendered PNG/YAML sidecar, `run_config.yml`,
`manifest.json`) are under `tmp/modernization_m0/captures/<name>/`. These are
single-run observations of the current pipeline, **not** verified-correct
baselines: known defects listed in this document still apply, so the arrays
are comparison inputs, not ground truth. sha256 of the four initially
modified `module_output` diagnostic PNGs was recorded before and after the
runs and is identical
(`dirty_png_sha256_before.txt` / `dirty_png_sha256_after.txt`).

## 2. Architecture and data flow (static findings)

- `brilliant_isp.py` is a ~1,200-line mutable orchestrator combining RAW
  loaders, configuration interpretation, execution, 3A loops, diagnostics, and
  export.
- Current stage order: load/alignment/flips -> crop -> DPC -> BLC -> optional
  green repair -> PWC -> OECF -> DG -> LSC -> BNR -> AWB/WB -> optional Bayer
  TMO -> demosaic -> CCM -> optional RGB TMO -> AE -> linear 16-to-8
  conversion -> CSC/saturation -> optional pre-gamma LDCI -> sharpen -> 2D NR
  -> RGB conversion -> gamma -> optional post-gamma LDCI -> scale -> optional
  YUV packing -> visualization/output.
- `run_pipeline_up_to_wb()` duplicates the front half of `run_pipeline()`.
  Direct AE can repeat DG through CCM/TMO; the outer 3A loop can repeat the
  complete path again.
- Demosaic kernels generally return float32, but the wrapper clips/casts to
  uint16. The pipeline hardcodes 65535 scaling at multiple boundaries.
- `load_config()`, crop, scale, YUV handling, and 3A mutate configuration.
  `_reset_transient_frame_state()` and GUI feedback merging compensate for
  some mutations.
- `platform.output_dir` is overwritten with `module_output` even when a
  session-specific output directory was requested. YUV packing writes to
  `./out_frames` directly.
- `last_output_rgb` is only produced when `visualize_output=True`, which also
  saves files; computation cannot be cleanly requested without
  presentation/export side effects.
- Logging uses mutable global debug state and named-handler replacement.
  Several modules select the Matplotlib backend at import time.

## 3. Tone mapping (static findings)

The bullets below record the **pre-M3 audit**, not the current execution
path. M3 replaced the duplicate operators with the shared registry,
normalization, curve, backend, and local-engine modules; retired classes
now delegate through `compat.py`. Input ranges, exposure semantics,
constant frames, finite-value checks, and clip-before-cast behavior have
regression coverage. Canonical configuration and migration are active.
Full ACES and verified GPU Durand remain unsupported; pipeline-wide
precision/transfer corrections remain M4. Current behavior and measured
legacy-output differences: `blocks/tone_mapping.md`.

- Dispatcher: `modules/tone_mapping/tone_mapping.py`.
- Separate float and integer/LUT files duplicate ACES rational coefficients,
  Hable partial-curve math, parameter extraction, normalization, plotting, and
  save logic.
- Float ACES min-max normalizes the image, multiplies by 100, evaluates an
  ACES-inspired fitted rational curve, then applies an sRGB-like transfer.
  LUT ACES uses separately configured domain scale and normalization.
- LUT ACES applies exposure before per-image min-max normalization; without
  clipping, normalization mathematically cancels uniform exposure.
- Both ACES paths may apply a transfer inside the operator while the main
  pipeline can apply gamma again (double OETF risk).
- `aces.variant` and `aces.output_transform` are exposed in configs/GUI enum
  metadata but unused by the float implementation; `canonical` is a
  misleading functional option. The implemented curve is a fitted
  (Narkowicz-style) curve, not a complete ACES RRT/ODT.
- `_extract_luminance_int()` multiplies the original integer array without
  widening first; the RGB recombination path casts to uint16 before clipping.
  Both are overflow hazards pending regression tests and fixes.
- Integer operators infer input bit depth from the image's observed maximum;
  LSC has a similar heuristic. A dark HDR frame can be reinterpreted as
  16-bit.
- `ToneMapping.__init__()` normalizes and constructs an operator before
  `execute()` checks the global enable flag; disabled processing still builds
  LUTs / validates irrelevant operators.
- CPU Durand returns normalized float luminance; its GPU implementation can
  return 8/16-bit integers — an inconsistent backend contract.
- Durand's active CPU path uses a Gaussian/range-weighted approximation, not
  an exact bilateral filter; `downsample_factor` is unused by the active
  `apply_tone_mapping()` path. It normalizes by max-minus-min with no
  constant-image guard.

## 4. GUI, IO, and configuration (static findings)

- Calibration GUI ~1,250 lines; older tuning GUI ~2,350. The older tuner has
  histograms, clipping, picker, ROI, and auto-processing features to extract.
- Calibration controls are generated from keys present in the loaded YAML, not
  a validated typed schema; defaulted parameters can be invisible and inactive
  operator sections remain visible.
- Numeric parsing sometimes silently retains/defaults/truncates invalid input;
  YAML text fields are not flushed by `_flush_isp_form_to_state()`.
- Calibration processing runs in a daemon thread capturing mutable
  app/session references; there is no request identity/revision check,
  cancellation, or progress protocol.
- `PreviewPane.show_rgb()` recreates axes content and refits every update,
  losing the comparison viewport.
- Session creation updates the last-session pointer before startup recovery
  checks it (recovery-ordering hazard).
- Session RAW import writes payload bytes to `input.raw` regardless of
  original container type (acceptable for stripped PGM; TIFF/rawpy containers
  lose identity).
- Preview uses an explicit dtype selector while processing guesses 2/3/4-byte
  storage from size with a 10% tolerance — two different interpretations.
- Header offset is session/UI state consumed by preview but not by the
  pipeline loader.
- Bayer flips change the sample layout without a shared effective-phase
  contract; crop silently adjusts odd offsets.
- `util/config_merge.py` expects `config/base_hdr.yml`, which lives under
  `config/unused/`; the implicit filename-based overlay merge does not occur
  in the present tree.
- Required-section lists and algorithm enums are duplicated across
  pipeline/types/GUI/export/profilers. `sensor_info.output_bit_depth`
  advertises paths the early 8-bit conversion cannot faithfully preserve.
- `isp_pipeline_batch_convert.py` imports `matches.py`, which is absent from
  the tree (confirmed: `from matches import matches` at line 13, no such file).

**M1 foundation status (completed):** the required-section tuple is shared,
`deep_merge` is fully independent-copying, explicit `extends` inheritance
works with cycle/chain errors, `normalize_config` now guards both loader
seams (`BrilliantISP.load_config` exposes notices via `config_notices`;
`load_preset_yaml` normalizes), GUI validation delegates to the shared
`validate_config`, and `tools/validate_config.py` provides a pipeline-free
CLI (exit 0/1/2). Verified by `tests/test_config_merge.py`,
`tests/test_config_schema.py` (all bundled presets error-free),
`tests/test_validate_config_cli.py`, and `tests/gui/test_calibration_logic.py`
regressions; frozen-scene RGB parity was bit-exact after the foundation
landed (see `tmp/modernization_m0/config_parity.json`). Deferred: canonical
v2 schema, TMO alias migration, per-block descriptor expansion.

**M2b status update:** `util/raw_decode.py` is now the sole decode path for
`load_raw`, `process_frame` file input, and the GUI preview; the old
heuristic survives only as the unused private `_load_raw_legacy` audit
copy. Crop applies exact origins (odd included) and updates the effective
CFA phase; `sensor_info` gained `storage_format`/`source_format`/
`header_bytes`/`row_stride_bytes`/`word_order`/`data_alignment=MANUAL`
fields wired through the shared validator, UI enum map, and session
staging (source bytes preserved under `input<suffix>`). See
`docs/RAW_INPUT.md` and `tests/test_raw_decode.py`,
`tests/test_raw_integration.py`, `tests/test_crop_coordinates.py`.

**M2a status update:** `util/pipeline_types.py` contracts and
`util/pipeline_runner.process_frame` now exist; `run_pipeline` and
`run_pipeline_up_to_wb` share extracted front-half helpers; every stage runs
through a `_run_stage` seam with cancellation and `perf_counter_ns`
telemetry; `last_output_rgb`/`last_output_native` are computed
unconditionally; final-output writes occur only on the final-pass effective
save flag (`visualize_output` for legacy, `visualize_output and
options.save_outputs` for explicit requests); diagnostics are independent
and never trigger final-output writes; and `YUVConvFormat` honors a
`write_output` flag plus `platform.final_output_dir`. Output metadata
reports the actual produced domain (`legacy_configured_rgb8` or
`legacy_ycbcr`), output dtype depths, and `cfa_pattern=None`. Legacy signal
domains/decoder and global loggers remain as documented pending M2b/M4.
See `docs/ARCHITECTURE.md` and `tests/test_pipeline_runner.py` /
`tests/test_pipeline_state.py`.

Earlier M1a items: the required-section tuple is now shared
(`util/config_schema.REQUIRED_CONFIG_KEYS`, aliased by the pipeline and GUI);
`deep_merge` returns a fully independent deep copy; `load_merged_yaml`
supports explicit relative `extends` inheritance with cycle detection; and
GUI numeric validation delegates to the shared `validate_config`. The
implicit `base_hdr.yml` filename merge remains intentionally inert until
audited v2 defaults are restored under `config/`. Enum/metadata deduplication
elsewhere and the `matches.py` gap are unchanged.

## 5. Documentation, tests, and performance evidence (static findings)

- `docs/ISP_BLOCKS_AND_TUNING.md` is an overview, not a uniform per-block
  reference; historical algorithm docs contain old keys and assumptions.
- `docs/GUI_DESIGN.md` still describes the calibration GUI as unimplemented.
- Gamma docs make overly broad claims that linear RGB is required by
  BT.601/709; Y'CbCr must be distinguished from linear luminance.
- Existing tests cover RAW/PGM helpers and GUI state/session/validation, with
  limited processing assertions. The "process writes artifacts" test exercises
  preparation, not an end-to-end image pipeline.
- No TMO/backend equivalence, color-transfer, demosaic, boundary, calibration,
  GUI lifecycle, or performance regression coverage exists.
- Historical profiling reports are leads, not current baselines; BNR/DPC and
  optional NLM were expensive in that scenario; the fast report uses a
  now-missing config.
- Profiling relies partly on rounded log-message parsing and misses inline
  conversion/constructor/orchestration cost.
- Confirmed by caller scan: `modules/ldci/ldci.py:54` and
  `modules/noise_reduction_2d/noise_reduction_2d.py:55` gate their "optimized"
  variants behind `if False`, and the BNR dispatcher selects `joint_bf.py`
  despite additional `_optimized`/`_numba` files. Algorithm/file names
  containing `_opt` do not prove equivalence or reachability.
- CI is pylint-centric, does not run the tests, includes a closed-PR lint
  workflow, and pre-commit uses a Windows-specific command.

## 6. Implementation inventory

Caller data was enumerated mechanically (import statements only; dynamic or
string-based dispatch is noted). "Active" means reachable from the current
pipeline entry point; "config-selectable" means reachable when the named
algorithm/backend is chosen in config. Unreferenced files are marked
**pending disposition** — not assumed redundant or equivalent until compared.

### Pipeline modules

| File | Role | Importers / callers | Classification |
|---|---|---|---|
| `modules/crop/crop.py` | Crop | `brilliant_isp.py` | Active |
| `modules/dead_pixel_correction/dead_pixel_correction.py` | DPC dispatcher | `brilliant_isp.py` | Active dispatcher |
| `modules/dead_pixel_correction/dynamic_dpc.py` | Dynamic DPC (reference) | DPC dispatcher (fallback), `dynamic_dpc_numba*.py` | Active |
| `modules/dead_pixel_correction/dynamic_dpc_numba_optimized.py` | Numba DPC | DPC dispatcher (preferred when importable) | Active |
| `modules/dead_pixel_correction/dynamic_dpc_numba.py` | Numba DPC variant | none | Unreferenced — pending disposition |
| `modules/dead_pixel_correction/directional_dpc.py` | Directional DPC | none | Unreferenced — pending disposition |
| `modules/dead_pixel_correction/directional_dpc_hybrid.py` | Hybrid DPC | none | Unreferenced — pending disposition |
| `modules/black_level_correction/black_level_correction.py` | BLC | `brilliant_isp.py` | Active |
| `modules/pwc_generation/pwc_generation.py` | Decompanding/PWC LUT | `brilliant_isp.py` | Active |
| `modules/oecf/oecf.py` | OECF LUT | `brilliant_isp.py` | Active |
| `modules/digital_gain/digital_gain.py` | Digital gain | `brilliant_isp.py`, `tests/gui/test_calibration_logic.py` | Active |
| `modules/lens_shading_correction/lens_shading_correction.py` | LSC | `brilliant_isp.py` | Active |
| `modules/bayer_noise_reduction/bayer_noise_reduction.py` | BNR dispatcher | `brilliant_isp.py` | Active dispatcher |
| `modules/bayer_noise_reduction/joint_bf.py` | Joint bilateral filter | BNR dispatcher (CPU default) | Active |
| `modules/bayer_noise_reduction/joint_bf_gpu.py` | GPU joint BF | BNR dispatcher (GPU branch) | Active — GPU-conditional |
| `modules/bayer_noise_reduction/joint_bf_numba.py` | Numba JBF | none | Unreferenced — pending disposition |
| `modules/bayer_noise_reduction/joint_bf_optimized.py` | Optimized JBF | none | Unreferenced — pending disposition |
| `modules/auto_white_balance/auto_white_balance.py` | AWB dispatcher | `brilliant_isp.py` | Active dispatcher |
| `modules/auto_white_balance/gray_world.py` | Gray-world AWB | AWB dispatcher | Active — config-selectable |
| `modules/auto_white_balance/norm_gray_world.py` | Norm-2 gray world | AWB dispatcher | Active — config-selectable |
| `modules/auto_white_balance/pca.py` | PCA AWB | AWB dispatcher | Active — config-selectable |
| `modules/white_balance/white_balance_optimized.py` | WB apply (optimized) | `brilliant_isp.py` | Active |
| `modules/white_balance/white_balance.py` | WB apply (reference) | none | Unreferenced — pending disposition |
| `modules/demosaic/demosaic.py` | Demosaic dispatcher | `brilliant_isp.py`, `tools/profile_demosaic_tmo_fast.py` | Active dispatcher |
| `modules/demosaic/malvar_he_cutler.py` | Malvar-He-Cutler | dispatcher, `util/utils.py` | Active — default algorithm |
| `modules/demosaic/malvar_he_cutler_cupy.py` | CuPy Malvar | dispatcher (optional import) | Active — GPU-conditional |
| `modules/demosaic/bilinear_demosaic.py` | Bilinear | dispatcher | Active — config-selectable |
| `modules/demosaic/vng_demosaic.py` | VNG + optimized class | dispatcher | Active — config-selectable |
| `modules/demosaic/hamilton_adams_demosaic.py` | Hamilton-Adams + opt | dispatcher | Active — config-selectable |
| `modules/demosaic/ppg_demosaic.py` | PPG + opt | dispatcher | Active — config-selectable |
| `modules/demosaic/lmmse_demosaic.py` | LMMSE + opt/fast | dispatcher | Active — config-selectable |
| `modules/demosaic/ahd_demosaic.py` | AHD + opt | dispatcher | Active — config-selectable |
| `modules/color_correction_matrix/color_correction_matrix_optimized.py` | CCM (optimized) | `brilliant_isp.py` | Active |
| `modules/color_correction_matrix/color_correction_matrix.py` | CCM (reference) | none | Unreferenced — pending disposition |
| `modules/tone_mapping/tone_mapping.py` | TMO dispatcher | `brilliant_isp.py`, `tools/profile_demosaic_tmo_fast.py` | Active dispatcher |
| `modules/tone_mapping/registry.py` | Immutable spec, capabilities, aliases | Adapter, compatibility base, schema, migration/export | Active — shared contract |
| `modules/tone_mapping/operators.py` | ACES-fitted/Hable/Reinhard curves | Backends, spec validation | Active — one curve definition each |
| `modules/tone_mapping/normalization.py` | Ranges, stats, luminance, recombination | Adapter, backends, local engine, compatibility base | Active — shared utilities |
| `modules/tone_mapping/backends.py` | Float/LUT/fixed dispatch and bounded cache | Adapter, compatibility base, diagnostics | Active — config-selectable |
| `modules/tone_mapping/local_durand.py` | Gaussian/range local approximation | Backend dispatcher | Active — CPU |
| `modules/tone_mapping/compat.py` | Shared legacy constructor/execute/save behavior | Retired wrapper classes | Compatibility — removal target M8 |
| `modules/tone_mapping/diagnostics.py` | Isolated native-array saves and curve plots | Adapter, compatibility base | Active — request-gated |
| `modules/tone_mapping/migration.py` | Pure canonical conversion and preservation report | Migration CLI | Explicit migration only |
| `modules/tone_mapping/aces/aces_tone_mapping.py` | Float ACES-fitted alias | Direct compatibility imports/tests | Wrapper — no separate operator |
| `modules/tone_mapping/hable/hable_tone_mapping.py` | Float Hable alias | Direct compatibility imports/tests | Wrapper — no separate operator |
| `modules/tone_mapping/integer_tmo/aces_integer_tone_mapping.py` | LUT ACES alias | Direct compatibility imports/tests | Wrapper — no separate operator |
| `modules/tone_mapping/integer_tmo/hable_integer_tone_mapping.py` | LUT Hable alias | Direct compatibility imports/tests | Wrapper — no separate operator |
| `modules/tone_mapping/integer_tmo/integer_tone_mapping.py` | Fixed Reinhard alias | Direct compatibility imports/tests | Wrapper — no separate operator |
| `modules/tone_mapping/durand/hdr_durand_fast.py` | Durand CPU alias | Direct compatibility imports/tests | Wrapper — no separate operator |
| `modules/tone_mapping/durand/hdr_durand_fast_gpu.py` | Explicit warned CPU fallback | Direct compatibility imports/tests | No verified GPU implementation |
| `modules/auto_exposure/auto_exposure.py` | Auto exposure | `brilliant_isp.py` | Active |
| `modules/color_space_conversion/color_space_conversion.py` | RGB<->YUV CSC | `brilliant_isp.py` | Active |
| `modules/ldci/ldci.py` | LDCI dispatcher | `brilliant_isp.py` | Active dispatcher |
| `modules/ldci/clahe.py` | CLAHE | `ldci.py`, `ldci_rgb.py` | Active |
| `modules/ldci/clahe_optimized.py` | Optimized CLAHE | `ldci.py` behind `if False` | Unreachable/disabled (static-confirmed) |
| `modules/ldci/ldci_rgb.py` | RGB-domain LDCI | `brilliant_isp.py` | Active |
| `modules/sharpen/sharpen.py` | Sharpen dispatcher | `brilliant_isp.py` | Active dispatcher |
| `modules/sharpen/unsharp_masking.py` | Unsharp mask | `sharpen.py` | Active |
| `modules/sharpen/unsharp_masking_gpu.py` | GPU unsharp | `sharpen.py` (optional) | Active — GPU-conditional |
| `modules/noise_reduction_2d/noise_reduction_2d.py` | 2D NR dispatcher | `brilliant_isp.py` | Active dispatcher |
| `modules/noise_reduction_2d/non_local_means.py` | NLM | dispatcher | Active |
| `modules/noise_reduction_2d/non_local_means_optimized.py` | Optimized NLM | dispatcher behind `if False` | Unreachable/disabled (static-confirmed) |
| `modules/rgb_conversion/rgb_conversion.py` | YUV->RGB | `brilliant_isp.py` | Active |
| `modules/gamma_correction/gamma_correction.py` | Gamma/transfer | `brilliant_isp.py` | Active |
| `modules/scale/scale.py` | Scale dispatcher | `brilliant_isp.py` | Active dispatcher |
| `modules/scale/bilinear_interpolation.py` | Bilinear scale | `scale.py` | Active — config-selectable |
| `modules/scale/nearest_neighbor.py` | Nearest scale | `scale.py` | Active — config-selectable |
| `modules/scale/scale_gpu.py` | GPU scale | `scale.py` (optional) | Active — GPU-conditional |
| `modules/yuv_conv_format/yuv_conv_format.py` | YUV packing | `brilliant_isp.py` | Active |

### Utilities and tools

| File | Role | Importers / callers | Classification |
|---|---|---|---|
| `util/raw_io.py` | RAW/PGM decode, flips, alignment | `brilliant_isp.py`, batch convert, GUI preview/session, tests | Active |
| `util/isp_types.py` | TypedDict config types | most modules | Active |
| `util/utils.py` | Save/plot helpers | most modules | Active |
| `util/debug_utils.py` | Logging/debug | most modules | Active |
| `util/histogram_utils.py` | Histograms | `brilliant_isp.py` | Active |
| `util/bayer_repair.py` | Zero-green repair | `brilliant_isp.py` | Active |
| `util/config_merge.py` | YAML overlay merge | `brilliant_isp.py`, profilers, tuning GUI | Active (HDR overlay inert — see §4) |
| `util/config_utils.py` | Filename metadata parse | `isp_pipeline_batch_convert.py` | Active |
| `util/gpu_utils.py` | GPU availability/policy | GPU-conditional modules | Active — GPU-conditional |
| `util/pipeline_validation.py` | Config validation | none | Unreferenced — pending disposition |
| `util/gen_requirements.py` | Requirements generator | none | Standalone utility script |
| `tools/isp_calibration_gui.py` | Calibration GUI entry | executed directly | Active entry point |
| `tools/isp_tuning_gui.py` | Legacy tuning GUI | executed directly | Active entry point (superseded pending parity) |
| `tools/gui/__init__.py` | Package init | importer convenience | Active (init) |
| `tools/gui/computed.py` | Derived GUI values | `isp_calibration_gui.py`, tests | Active |
| `tools/gui/format_map.py` | RAW format descriptors | preview/validation/GUI, tests | Active |
| `tools/gui/gui_log.py` | GUI logging | `isp_calibration_gui.py`, tests | Active |
| `tools/gui/process.py` | Processing preparation | `isp_calibration_gui.py`, tests | Active |
| `tools/gui/raw_preview.py` | RAW preview decode | `isp_calibration_gui.py` | Active |
| `tools/gui/session.py` | Session model | GUI modules, `isp_calibration_gui.py`, tests | Active |
| `tools/gui/state.py` | GUI state | `isp_calibration_gui.py`, tests | Active |
| `tools/gui/ui_schema.py` | Control metadata | `isp_calibration_gui.py` | Active |
| `tools/gui/validation.py` | Config validation UI-side | GUI modules, tests | Active |
| `tools/gui/yaml_compat.py` | Preset loading/required keys | both GUIs, session, tests | Active |
| `tools/profile_isp_blocks.py` | Block profiler | executed directly | Active tool (historical scenario) |
| `tools/profile_demosaic_tmo.py` | Demosaic/TMO profiler | executed directly | Active tool (missing-config scenario) |
| `tools/profile_demosaic_tmo_fast.py` | Fast profiler | executed directly | Active tool |
| `tools/extract_profile_results.py` | Report extractor | executed directly | Active tool |
| `tools/export_boltisp_yaml.py` | boltISP exporter | executed directly | Active tool |
| `isp_pipeline.py` / `isp_pipeline_multiple_configs.py` | Batch entry points | executed directly | Active entry points |
| `isp_pipeline_batch_convert.py` | Batch converter | executed directly | Broken import (`matches.py` absent) |

## 7. Test coverage baseline and fixture gaps

Current suite (40 tests) covers: PGM header/payload parsing, RAW flips and
data alignment, GUI session copy/import, format-map and validation logic,
calibration-logic helpers, and GUI log filtering. It does **not** execute the
image pipeline end-to-end.

Added this milestone (`tests/synthetic_fixtures.py`,
`tests/test_synthetic_fixtures.py`): deterministic constant/black/white,
uint16 and 24-bit uint32 ramps, primaries/neutral RGB, `rgb_to_bayer` for all
four CFA patterns, slanted-edge RGB and checkerboard texture frames,
odd-dimension and 2185-row constructors, hot/dead-pixel, and seeded-noise
fixtures.

Remaining fixture gaps for later milestones (from the required regression
matrix): packed24/word-pair storage samples, endian/header/stride decode
vectors, TIFF/rawpy adapters, malformed/truncated payloads, cropped/flipped
CFA-phase vectors, saturated-primary HDR frames, and per-stage known-answer
data. These are deliberately future work beyond M1a scope.

**M3 status update:** tone mapping consolidated into the shared engine
(`registry`, `normalization`, `backends`, `local_durand`, `diagnostics`,
`compat`); retired TMO files are thin wrappers. Canonical
`tone_mapping.operator/backend/placement` supported with legacy alias
resolution; `hdr_durand` is now an optional legacy section (required
count 24); migration CLI + report; boltISP export handles the supported
canonical subset. See `docs/blocks/tone_mapping.md`.

## 8. Native M4 follow-up candidates

The inventory above is historical; `ARCHITECTURE.md` describes the current
native execution graph. These candidates are not measured speedups or
validated algorithm replacements:

- **P1 — Headroom-aware tone normalization.** Global-maximum per-frame
  normalization changes midtones when previously clipped highlights are
  retained. Investigate explicit white anchors or robust percentile
  normalization without restoring early clipping. The local controlled
  attribution is in `tmp/modernization_m0/svs_headroom_probe/`.
- **P1 — AWB validity and confidence.** Virtual digital gain is now removed
  from AWB measurement units. Follow up with sensor-domain saturation/noise
  validity masks before companding and confidence handling for sparse
  low-light samples; bounded gains alone do not establish illuminant accuracy.
- **P1 — Native diagnostic-flag migration audit.** Keep request-scoped stage
  captures distinct from legacy per-module save flags, and document or
  migrate unsupported flags rather than presenting inactive controls.
- **P2 — CPU work reduction.** Measure the extra metering/render demosaic
  work in advanced pre-demosaic tone mapping, geometry-mask allocation, and
  reusable LUT construction after baseline acceptance. Any domain-changing
  shortcut needs a separate algorithm gate, not a mechanical speedup claim.

## 9. M5 CPU optimization evidence

- Four production files changed: `modules/demosaic/bilinear_demosaic.py`
  (per-call 2x2 periodic denominator tiles with binary/parity gating and
  full-mask fallback; OR-shift horizontal presence maps on the fast path;
  unused vertical presence convolutions removed),
  `util/native_array.py` (validated-f32 identity return),
  `util/pipeline_runner.py` (`copy=False` second endian cast),
  `util/native_output.py` (owned-copy sRGB8 preview fast path).
- Numerical evidence: `tests/test_cpu_optimizations.py` (128 cases — all
  CFA layouts, tiny/odd/even, signed HDR, const/ramp, weighted/irregular
  masks, read-only/non-contiguous inputs, endian/finite errors, all 256
  sRGB preview codes, structural fast-path asserts) plus the full suite:
  1252 passed, pyright 0 errors over 104 maintained files.
- Benchmark evidence is lead-derived and frozen in
  `tmp/modernization_m0/m5_cpu/final_summary.json` and `seal.json`;
  per-case raw records live alongside them. Recorded medians only; see
  `docs/ARCHITECTURE.md` for the claimed reductions and the descriptive
  (overlapping-range) AD array figure. No memory or export-speedup claims.
