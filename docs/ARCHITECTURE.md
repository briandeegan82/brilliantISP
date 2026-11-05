# Architecture (native M4 engine, current status)

This documents the processing API as it exists now. All active entry points
run the single native engine; the legacy module bodies are retired to the
private `_*_legacy` helpers for audit only — they are not a fallback.

## Entry points

- **Facade** — `BrilliantISP(data_path, config_path, outFileName,
  output_path=None, *, options=None)`. `config_path` accepts YAML path(s)
  (existing merge/`extends` rules) or an in-memory mapping (normalized and
  deep-copied; `self.config_path` becomes `"<mapping>"`). `options=None`
  keeps the facade default save behavior (writes to `output_path`);
  explicit `ProcessingOptions` controls saves, diagnostics, captures,
  cancellation, and telemetry per request.
- **New API** — `util.pipeline_runner.process_frame(source, config, *,
  state=None, options=None)`. `source` is a RAW file path (shared `util.raw_decode` loader) or
  a decoded 2-D Bayer `np.ndarray` matching declared `sensor_info` geometry
  exactly. Every call builds a fresh engine on a deep-copied config, so
  independent frames are repeatable and the caller's config is never mutated.

## Configuration loading / migration

Every source config — `schema_version: 1` or versionless — is migrated in
memory to the native schema_version 2 mapping at load via
`util.pipeline_migration.migrate_pipeline_config`. The pre-migration source
is kept on the engine as `source_config_original`; the migration report is
exposed on `PipelineResult.migration_report` and merged into the config
notices. The
native required-key set excludes the four retired color sections
(`gamma_correction`, `color_space_conversion`, `rgb_conversion`,
`yuv_conversion_format`) and requires `output`; v2 sources that still enable
a retired section are rejected. Output constraints are owned by
`OutputSpec.from_mapping`, not by scattered validation. The GUI preset
loader and the boltISP exporter both reject v2 and direct users to the
original v1 source; v1 workflows are unchanged. A fully migrated v2 config
runs end-to-end natively.

## Contracts (`util/pipeline_types.py`)

- `FrameMetadata` — frozen *output* descriptor: width/height from the final
  output stage (packed 4:2:2 reports the logical width, not the packed 2W),
  `cfa_pattern` (`None` on RGB/YUV output), `storage_bit_depth`/
  `code_bit_depth` from the actual output dtype/spec, `signal_domain`,
  `transfer`, `source_identity` (resolved file path or `array:{run_id}`).
  Domains are reported truthfully: `scene_linear_rgb` for linear output
  with `tone_mapped=false` (e.g. untone-mapped float linear TIFF),
  `display_linear_rgb`, `display_encoded_rgb`, or encoded YCbCr.
- `ProcessingState` — opt-in 3A carryover: `previous_awb_gains` is committed
  only when AWB actually *estimated* this frame (`last_status == "estimated"`,
  never on fallback); `digital_gain_index` seeds the next auto-DG pass and
  is ignored by manual DG; `ae_history` is bounded telemetry only. State is
  committed only after the result is fully built and the final cancellation
  check passes — never on failure or cancellation. `process_frame`
  snapshots `ProcessingOptions` per call so caller-side mutation cannot
  affect the active request.
- `process_frame` runs `validate_config` before engine creation; array
  sources must be unsigned 2-D Bayer matching declared geometry exactly —
  signed/float/object dtypes are rejected pending a designed float-raw path.
- `ProcessingOptions` — `run_id`, `capture_stages`, `diagnostics`,
  `save_outputs`, `output_dir`, `cancellation_token`, `progress_callback`.
  Defaults are compute-only: no directories, no writes, diagnostic flags
  suppressed in the internal config copy only.
- `StageEvent` — per-stage telemetry with `pass_index`/`ae_pass`.
- `PipelineResult` — `native_image` (the quantized/output-spec array, never
  a legacy packed-YUV heuristic), `native_metadata`, `preview_rgb8`,
  `effective_config`, `feedback` (authoritative last-stage feedback: AWB
  gains/status/sample count, applied DG index, `next_digital_gain_index`,
  AE feedback/meter average/stop reason/pass count), `timings`, `captures`
  (named-stage owned read-only arrays; demosaic/CCM captures are normalized
  F32 RGB), `warnings`, `artifacts`, `migration_report`.
- `ProcessingCancelled` — checked at every stage boundary and again before
  quantize/write; the writer itself is not interruptible, so cancellation is
  guaranteed at stage boundaries and before the writer starts.

## Execution graph (native)

Every stage runs through `_run_stage` (cancellation + telemetry + optional
capture). Disabled stages return identity and do not validate unused
controls.

1. Front half, once per frame: crop → DPC → BLC (float64 math) →
   zero-green repair (AUTO skips all-black frames, original-site fraction)
   → PWC → OECF (declared-code validated LUT, no integer wrap).
2. Per AE pass — every pass restarts from the immutable OECF output:
   DG → LSC → BNR (CPU) → AWB measurement → WB → normalize by the declared
   linear code maximum (ADC / PWC-HDR / OECF output bits, not dtype-inferred)
   → demosaic into normalized F32 RGB → CCM (normalized matrix, signed
   headroom preserved).
   AWB removes the applied virtual digital gain from measurement units
   before rejecting samples and estimating the illuminant; valid sensor
   highlights do not become ineligible merely because digital gain rises.
3. Named `linear_rgb_pre_tone_mapping` tap feeds the AE meter in
   scene-linear units — always the *untonemapped* signal.
4. Tone mapping via `map_native` (see `docs/blocks/tone_mapping.md`). The
   advanced pre-demosaic placement runs a separate render branch —
   `tone_mapping_pre` → `demosaic_render` → `color_correction_matrix_render`
   (extra CPU; a color-fidelity warning is logged) — while AE still meters
   the normal pre-tone-map tap. There is exactly one output transfer regardless
   of branch.
5. Saturation (enabled only), pre-transfer LDCI when
   `ldci.placement=pre_transfer`, the single `encode_transfer` for
   `srgb`/`rec709`/`power`/`linear` (F32-linear-TIFF bypasses clipping),
   post-transfer LDCI (`post_transfer`), sharpen and 2-D NLM — all three on
   `output.matrix` luminance via `apply_luma`/`replace_luminance` — then
   `Scale.compute_linear` (CPU-only, F32, no GPU/save path).
   The tail order is: pre-transfer LDCI → `encode_transfer` →
   post-transfer LDCI → sharpen → 2-D NLM → scale.
6. `quantize_output` → `preview_from_native` → `output_metadata` → explicit
   writer. Full finite float precision is carried through the chain with no
   accidental quantization — the tone/luma stages bound their own results
   algorithmically (and BLC/PWC clip by design); the output contract owns
   transfer encoding and final quantization. There is no 8-bit YUV roundtrip anywhere in
   the native chain.

## AE controller

Each pass rebuilds the frame from the immutable OECF output — gains never
compound across passes. The applied gain index is a *local* variable; the
effective config's manual `current_gain`/`ae_feedback` fields are never
written. `direct` mode follows `target_luminance` independently of the
histogram feedback sign (`fb==0` converges only `step` mode). Stop reasons
are truthful: `disabled`, `manual_gain`, `no_signal`, `converged`,
`repeated_gain`, `pass_limit`, `no_rerun`, `gain_limit` — a stable endpoint
that still requires exposure beyond the allowed gain range reports
`gain_limit`, not target convergence. `pass_limit` is reported, not
claimed as success. Pass count is bounded (24 max when `render_3a`, 2 for
the standard direct+auto rerun, 1 without rerun). `run_pipeline_up_to_wb`
profiles a single native pass through the same helpers — it does not run
the final AE convergence loop.

## Output / IO separation

- `last_output_native`, `last_output_rgb`, `last_output_metadata` are
  computed unconditionally after the final AE pass/render (not per pass).
- Writes happen only through `util.output_writer.write_output` when
  `options.save_outputs` (explicit) or the `options=None` facade default
  applies; `run_pipeline(visualize_output=False)` suppresses the facade
  default. Compute-only mode creates nothing.
- The writer reserves one shared collision-free stem for the image and all
  three sidecars (`.json` metadata, `_effective.yaml`, `_migration.json`),
  validates native dtype/shape against `OutputSpec` before encoding
  (unsigned `u8`/`u16` YUV exact bit depth, `<u2` little-endian bytes,
  HWC3 4:4:4 or Hx(2W) 4:2:2, F32 HWC3 TIFF, cv2 RGB→BGR PNG/JPEG),
  serializes sidecars before opening any file, and removes only files it
  created on failure — pre-existing files are never touched.
- `output.is_save` is the consolidated legacy color-diagnostic flag and
  is currently inactive; it does not enable the final writer and does not
  trigger native TIFF/diagnostic output. Native TMO diagnostics are
  separate: when `diagnostics=True` and the resolved operator's
  `is_save`/`is_plot_curve` flags allow, the final rendered TMO stage
  writes one normalized F32 `.npy` + paired `.json` metadata
  (`diagnostic_schema_version: 1`) and/or an exclusive-numbered curve PNG
  under `output_dir/module_output`; paths land in
  `PipelineResult.artifacts` alongside final-writer files. Cancellation is
  checked before the IO boundary but does not interrupt an in-flight
  write.

## Known limitations

- Per-block documentation pages, GUI v2 editing/integrated HMI, and the
  calibration workflow are future milestones (v2 GUI editing raises until
  M6).
- Module constructors still use the global debug logger and per-module
  Matplotlib backend selection at import time.
- Native output changes are intentional, not appearance-preserving: legacy
  AE targets now mean pre-tone linear meters (migration normalizes declared
  legacy code units only — no inverse-TMO parity). The earlier SVS color
  cast was traced to virtual-gain WB metering (fixed: AWB divides out the
  applied digital gain before sample thresholds). The separate brightness
  change comes from retained highlight headroom driving the existing
  per-frame global-maximum normalizer. Early clipping is not restored;
  no appearance-parity or IQ-superiority promise is made.
- RAW decode goes through `util.raw_decode` (see `docs/RAW_INPUT.md`); the
  legacy heuristic is retained as the unused private `_load_raw_legacy` for
  migration audit. RGB/multiframe container input is rejected.

## Evidence

- `tmp/modernization_m0/m4_accepted_baseline/manifest.json` — accepted local
  native regression baseline for the two reviewed scenes. It seals the
  capture data and both captured/accepted engine source snapshots; the
  later IO/reporting-only delta and AD `gain_limit` feedback amendment are
  explicit. This is not legacy appearance parity, universal IQ validation,
  or a CPU speedup result.
- `tmp/modernization_m0/m4_full_accepted_tests.txt`,
  `m4_full_accepted_types.txt` — 1124 tests passed; the 102-file maintained
  type gate has zero errors and two pre-existing warnings.
- `tmp/modernization_m0/svs_headroom_probe/` — controlled frozen-array
  brightness attribution; diagnostic clipping is not a production fix.
- `tmp/modernization_m0/m2*`/`m3*` artifacts — historical M2/M3 gates and
  the accepted M3 numerical checkpoint (3606-case finite-grid verification;
  a historical reference point, not current quality proof).
- `tmp/modernization_m0/m4_core_review_tests.txt`,
  `m4_core_review_types.txt` — focused M4 core gate.
- `tmp/modernization_m0/m4_side_tests.txt`, `m4_side_types.txt` —
  writer/CLI/export/GUI-side gate.
- `tmp/modernization_m0/m4_native_pipeline.json`,
  `m4_native_primitives.json` — lead-authored independent verification
  vectors.

## M5 CPU optimization pass (measured)

Four pure-CPU optimizations landed in the native path, verified by
`tests/test_cpu_optimizations.py` (128 lead-authored golden/structural
cases) and the full suite (1252 tests, pyright 0 errors over 104 files):

- **Bilinear demosaic denominators** — when every channel mask is a
  regular binary-periodic 2x2 pattern matching the raw shape and all four
  parity slices equal the tile value, the normalization denominators are
  convolved once on the 2x2 tile instead of the full image (mirror
  reflection `i -> 2*(N-1)-i` preserves `i%2` for N>=2, so the four tile
  denominator values equal the full-image parity values exactly).
  Tiny/weighted/irregular/signed or shape-mismatched masks fall back to
  the original full-mask path; the two unused vertical "has" convolutions
  were removed and the horizontal presence maps use binary OR-shifts only
  on the periodic fast path.
- **`as_float32` identity** — an input already validated as finite
  `float32` returns without a second cast/finite scan; all other dtype
  casts, overflow guards, and named errors are unchanged.
- **Endian conversion** — the second `astype(newbyteorder("="))` in the
  array-entry path uses `copy=False`; the first caller-isolating
  `np.array(source, copy=True)` is retained.
- **sRGB8 preview** — `preview_from_native` returns an owned copy of an
  already-uint8 RGB + 8-bit + sRGB native frame instead of a
  float64 normalize/requantize round-trip; all other previews unchanged.

Measured (lead-run benchmark, frozen in
`tmp/modernization_m0/m5_cpu/final_summary.json` + `seal.json`):

- RAW-to-result warm medians: SVS 4390.97 -> 3383.92 ms (-22.93%);
  AD 10394.93 -> 7408.92 ms (-28.73%). Both warm ranges are disjoint.
- Array-API warm medians: SVS 5207.93 -> 3855.63 ms (-25.97%);
  AD 16441.64 -> 10134.41 ms (-38.36%) — the AD array ranges overlap, so
  that figure is descriptive/tentative, not a strong separation claim.
- No memory improvement is claimed: peak RSS is whole-process (imports,
  input, validation included); AD array RSS actually increased
  (1535 -> 1598 MiB) and export SVS RSS increased (903 -> 951 MiB).
- Export timing is n=1 and descriptive only — no speedup claim.
- Methodology: 7 warm calls + 2 warmups + first call, new process per
  case variant, Linux, single thread, fixed CV/BLAS/Numba=1/OpenCL off,
  counterbalanced case-level batches (not paired samples). The array mode
  includes constructor/validation copy/preview; the raw mode includes
  decode; the writer is excluded.
- Malvar and the other alternate demosaic algorithms are unchanged and
  not selected by these two benchmark presets; the alternate BNR/NLM
  optimized kernels were not enabled or performance-validated in M5.
  No real-time, general-CPU, or GPU claims.

The recorded workstation is an AMD Ryzen 5 7535HS, Python 3.12.3,
NumPy 1.26.4, SciPy 1.17.1, and OpenCV 4.11.0.86. The scenes are
1920x1536 SVS and 3840x2185 AD. The first call is a fresh-interpreter
processing call, not an uncached-disk measurement: input checksums and
setup decoding have already read the RAW. Import time is recorded
separately. Seven samples do not establish a tail-latency SLA; the
recorded p95 is descriptive only.

`tools/benchmark_native_cpu.py` is the reusable worker. The local
`tmp/modernization_m0/m5_cpu/cases.json` supplies exact input/config paths,
SHA-256 identities, and frozen expected output/stage/feedback/metadata
values; the private RAW corpus is not shipped as a public fixture.
For example, from this checkout, using a **new** output filename:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMBA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 \
venv/bin/python tools/benchmark_native_cpu.py \
  --engine-root . --cases tmp/modernization_m0/m5_cpu/cases.json \
  --case svs_fv --mode raw \
  --output tmp/modernization_m0/m5_cpu/new_run.json
```

Modes `array` and `raw` measure the public API, `check` verifies the
captured F32 stages in a separate process, and `export` enables final
saving plus preset-requested diagnostics. All modes enforce frozen
output identities; the worker refuses existing result paths. The exact
counterbalanced commands are in `final_execution_sequence.json`.
`final_summary.json` is the frozen statistical result, and `seal.json`
records evidence/source hashes and verifies 102 protected files.
Documentation tables should be rendered from that summary rather than
rerunning measurements to regenerate presentation.

Remaining CPU candidates require new measurements: transfer encoding
and saturation, the extra metering/render path in pre-demosaic tone
mapping, and LUT/geometry reuse. Domain changes, approximate curves,
different demosaic kernels, and GPU substitutions require separate
numerical and image-quality gates.

## Integrated workspace boundary (M6)

The opt-in Tk workspace lives in `tools/gui/`; `GUI_DESIGN.md` describes its
workflow, launch policy, acceptance evidence, and measured limitations. The
legacy launcher remains the default during transition.

- `model.py` owns the editable draft, source identity, validation, grouped
  history, and saved/preset snapshots. Opening an editor must not add defaults.
- `controller.py` schedules interactive processing before derived rendering,
  auxiliary analysis, and serial batch work. Frozen snapshot descriptors are
  separate from the live draft. Late processing and render results are rejected
  by their respective context/request/descriptor checks.
- `jobs.py`, `protocol.py`, and `worker.py` isolate compute in a persistent spawn
  process. Worker initialization fixes CPU thread limits and the noninteractive
  plotting backend before importing the engine. Tk does not run image kernels.
  Staged source content and file identity are verified before and after
  processing, so a changed input cannot inherit its predecessor's fingerprint.
- `artifacts.py`, `analysis.py`, and `comparison.py` retain native domains,
  preview domains, geometry, and provenance. Wipe/difference files are derived
  display artifacts, not replacements for native output or perceptual metrics.
  Pixel picking uses native logical coordinates, including packed-422 chroma.
- `workspace_session.py` atomically saves versioned session bundles and checks
  portable source/snapshot paths. Export uses the selected result's frozen
  native specification, not a subsequently edited output setting.

The real-scene worker parity report is
`tmp/modernization_m0/m6_hmi/native_worker_parity.json`: native/preview pixels,
captured F32 stages, feedback, and metadata match M5, except for the deliberately
session-owned source path. Protected inputs and previous evidence are unchanged.
Synthetic GUI/worker tests are committed; private scene evidence is local.

Optional bounded array-cache primitives are tested but not enabled in the
workspace. Their storage budget counts array bytes, not total RSS; input hashing,
copying, and invalidation are not free. No M6 cache speedup or general memory
reduction is claimed. Guided calibration and the remaining per-block reference
work are later milestones, not hidden processing stages in this workspace.
