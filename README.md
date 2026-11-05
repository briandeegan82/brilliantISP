# BrilliantISP

BrilliantISP is a configurable software ISP pipeline for RAW and HDR imaging workflows.
It is developed by Brian Deegan and is based in part on [Infinite-ISP](https://github.com/10x-Engineers/Infinite-ISP) by 10xEngineers, with substantial modifications and extensions.

- License: Apache 2.0 (`LICENSE`, `NOTICE`)
- Language/runtime: Python
- Primary interfaces:
  - CLI pipeline scripts
  - Interactive tuning GUI (`tools/isp_tuning_gui.py`)
  - Calibration GUI (`tools/isp_calibration_gui.py`) — Image Format + ISP Configuration

## Highlights

- Full configurable ISP chain (crop -> denoise -> AWB/AE -> demosaic -> CCM -> gamma -> output format)
- HDR-aware RAW loading paths (including packed/high-bit-depth formats)
- Multiple tone mappers (`reinhard_integer`, `aces`, `aces_integer`, `hable`, `hable_integer`, `hdr_durand`)
- Config merge model (`*_cam.yml` overlays merged on top of `config/base_hdr.yml`)
- Optional debug histogram generation in pipeline
- Interactive tuning GUI with live reprocess and analysis tools

## Repository Layout

- `brilliant_isp.py` - core `BrilliantISP` pipeline class
- `isp_pipeline.py` - simplest single-image pipeline entry script
- `tools/isp_tuning_gui.py` - tuning GUI
- `tools/isp_calibration_gui.py` - two-window calibration GUI (format + ISP YAML)
- `config/` - base and camera-specific YAML configs
- `modules/` - individual ISP block implementations
- `util/` - config helpers, histogram/debug utilities, shared types
- `docs/` - deep dives and tuning references
- `brilliantisp_gpu/` + `isp_pipeline_gpu_batch.py` - optional additive PyTorch GPU engine (see Headless GPU Engine)

## Requirements

Use Python 3.10+ (recommended). Install dependencies from `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Main runtime dependencies include `numpy`, `opencv-python`, `matplotlib`, `rawpy`, `PyYAML`, `scipy`, `tifffile`, `tqdm`, and `numba`.

## Quick Start (CLI)

The fastest way to run one image is:

```bash
python isp_pipeline.py
```

By default `isp_pipeline.py` points to:
- `CONFIG_PATH = ./config/AD_cam.yml`
- RAW filename `frame_000000.raw`

Override raw folder without editing code:

```bash
ISP_RAW_DATA=/path/to/raw/folder python isp_pipeline.py
```

If the expected file is missing, the script exits with a clear error and the suggested `ISP_RAW_DATA` override.

## Interactive Tuning GUI

Run from repo root:

```bash
python tools/isp_tuning_gui.py
```

Optional initial config:

```bash
python tools/isp_tuning_gui.py --config config/AD_cam.yml
```

## Calibration GUI

Two windows (Image Format and ISP Configuration) that edit one YAML-shaped config and process a project-local session copy of the RAW file. Spec: `docs/GUI_DESIGN.md`.

```bash
python tools/isp_calibration_gui.py
```

Optional preset and skip session recover:

```bash
python tools/isp_calibration_gui.py --config config/SVS_cam.yml --no-recover
```

Working files land in `tmp/gui_session/<session_id>/` (`input.raw`, `config.yml`, `output.png`, `session.json`).

Edits and exceptions are logged to:
- `tmp/gui_session/<session_id>/gui.log` (this session)
- `tmp/gui_session/calibration_gui.log` (all sessions)

Use **Help → Open log folder…** to browse them.

### Core GUI workflow

1. `File -> Open config...`
2. `File -> Open raw...`
3. Adjust controls in `Blocks` and `Parameters`
4. Click `Process` (or enable `Auto-process`)
5. Save via `Save All` / `Save config` / `Save output image`

### Analysis menu features

The GUI includes an `Analysis` menu for fast image QA:

- Histogram viewer (RGB + luminance, linear/log)
- Global image statistics
- Clipped pixel highlighting (shadow/highlight overlay)
- Clipping threshold controls
- Color picker mode (pixel coordinates + RGB/luma in status bar)
- ROI statistics mode (drag-select region and inspect stats)

## Configuration Model

Pipeline behavior is config-driven via YAML files in `config/`.

- `config/base_hdr.yml` contains defaults
- `*_cam.yml` camera files are auto-merged over base config
- The GUI and scripts both use this merged-config behavior

Common blocks you will tune frequently:

- `sensor_info` - dimensions, bit depth, Bayer metadata
- `digital_gain` / `auto_exposure` - exposure behavior
- `auto_white_balance` / `white_balance` - WB controls
- `demosaic` - algorithm selection
- `tone_mapping` and related parameter sections
- `gamma_correction` - output transfer curve
- `color_correction_matrix`, `color_saturation_enhancement`

For a complete per-block explanation, see `docs/ISP_BLOCKS_AND_TUNING.md`.

## Additional Pipeline Scripts

The repository also includes dataset/batch helper scripts:

- `isp_pipeline_mulitple_images.py` - process multiple folder pairs (dataset or video-style flow)
- `isp_pipeline_batch_convert.py` - recursively process one RAW per folder into `convert/`
- `isp_pipeline_multiple_configs.py` - compare output across many config variants

These scripts are practical templates and may require local path edits before use.

## Documentation Index

- `docs/GUI_DESIGN.md` - calibration GUI specification
- `docs/ISP_BLOCKS_AND_TUNING.md` - pipeline order and tuning guide
- `docs/GAMMA_CORRECTION_FINAL_SOLUTION.md` - gamma placement and rationale
- `docs/PPG_DEMOSAIC.md` - PPG demosaic details
- `docs/VNG_DEMOSAIC.md` - VNG demosaic details
- `docs/HAMILTON_ADAMS_DEMOSAIC.md` - Hamilton-Adams details

## Headless GPU Engine (`brilliantisp_gpu`)

An additive, optional PyTorch GPU engine lives alongside the CPU pipeline.
The CPU engine, GUI, and config tooling are unchanged; GPU support is
additional code in `brilliantisp_gpu/` plus the launcher
`isp_pipeline_gpu_batch.py`. CPU-only users never need torch — the package
imports lazily and there is **no GPU->CPU fallback**: every image-processing
block runs on the selected GPU only. One GPU is selected per process;
multi-GPU scheduling is not included.

### Requirements (GPU only)

- A separate `.venv-gpu` environment with `requirements.txt` plus a
  backend-appropriate PyTorch build. The tested configuration is pinned in
  `tmp/gpu_validation/m0_environment/pip_freeze_v2.txt` (frozen artifact;
  torch 2.9.1+cu128, numpy 1.26.4, scipy 1.17.1, opencv-python 4.11.0.86).
- Official install matrix: https://pytorch.org/get-started/locally/

```bash
python -m venv .venv-gpu
source .venv-gpu/bin/activate
pip install -r requirements.txt
# NVIDIA CUDA example (backend/platform-specific; see the matrix):
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```
- NVIDIA CUDA block and listed frame checks have run on an RTX 4050 Laptop GPU
  (6 GB, SM 8.9); the tested coverage and explicit exclusions are listed
  below. ROCm/HIP code paths are implemented but **not**
  hardware-qualified — they require a suitable ROCm PyTorch build, HIPRTC
  tooling, and a supported device; do not treat any AMD GPU as qualified.
- Custom kernels compile through NVRTC/HIPRTC at runtime; if the backend
  compiler is missing the run fails with a named error (no fallback).

### Config

GPU processing uses the same YAML schema, migration, and merge semantics —
no config conversion. A single `*_cam.yml` path gets the filename-sensitive
helper merge (e.g. `SVS_cam.yml` -> `config/base_hdr.yml` + overlay when the base
  file exists; a missing base is an existing preset limitation, not
  something the GPU path repairs); an
explicit list of paths is an ordered merge (length-1 lists do **not** add
the base). Mapping configs are supported in the API. RAW decode is the
shared strict Bayer-format metadata path — no guessing; packed Bayer RG12p
is not supported. Unpacked TIFF input is covered by specific tests only.

### Python API

```python
from brilliantisp_gpu import process_frame_gpu, process_batch, BatchItem, GpuOptions
from util.pipeline_types import ProcessingOptions, ProcessingState
```

- `process_frame_gpu(source, config, *, state=None, options=None, gpu=None)`
  — one frame. `source` is a RAW path or a 2-D unsigned `np.ndarray`
  (uint8/16/32) already in processing coordinates (no flips/trim).
- `process_batch(items, *, gpu=None, options=None)` — lazy iterator of
  `BatchOutcome` (`status`, `result`, `error`, `device`, `metrics`,
  `attempts`). Items are `BatchItem(item_id, source, config, sequence_id=None,
  frame_index=None, output_dir=None, output_stem=None)`. Independent items
  may be grouped into real batched cohorts; sequence items
  (`sequence_id`, strictly increasing nonnegative `frame_index`) always run
  singly with private per-sequence 3A state.
- `GpuOptions(device="cuda:0", max_batch_size=4, prefetch=2,
  memory_fraction=0.70)` controls grouping and admission.
```python
items = [
    BatchItem("frame_a", "/data/ad_000.raw", "config/AD_cam_2.yml"),
    BatchItem("frame_b", "/data/ad_001.raw", "config/AD_cam_2.yml"),
]
options = ProcessingOptions(run_id="demo", save_outputs=True,
                            output_dir="out/ad")
for outcome in process_batch(iter(items), gpu=GpuOptions(max_batch_size=4),
                             options=options):
    if outcome.status == "success":
        print(outcome.item_id, outcome.result.native_image.shape)
    else:
        print(outcome.item_id, outcome.status, outcome.error)
```

- Outputs: `native_image` (authoritative dtype/bit-depth/metadata) and
  `preview_rgb8` derived from the quantized/packed output.
- Default is compute-only: no directories or files are created unless
  `ProcessingOptions(save_outputs=True, output_dir=...)` is given.
- Configs and state are copied; state commits only on success (never on
  failure or cancellation).

### Batch CLI

```bash
python isp_pipeline_gpu_batch.py --input /path/to/ad_raws \
    --config config/AD_cam_2.yml --output out/ad \
    --device cuda:0 --batch-size auto --max-batch-size 4
# or: python -m brilliantisp_gpu --manifest batch.jsonl --output out/batch
```

- `--input` scans a folder of RAW inputs (`--recursive` optional,
  `--sequence` forces lexical sequence semantics); `--manifest` takes a
  JSONL file. `--config` accepts a single path or an ordered list.
- Manifest rows: required `id`, `input`, `config`; optional
  `output_subdir`, `sequence_id`, `frame_index`. Relative paths resolve
  against the manifest's parent directory. Unknown fields, duplicate ids,
  decreasing sequence indices, absolute/`..`/symlink-escaping output
  subdirs are structural errors (exit 2).

```json
{"id": "ad_first", "input": "raws/ad_000.raw", "config": "presets/AD_cam_2.yml"}
{"id": "seq_a_0", "input": "raws/seq_000.raw", "config": "presets/AD_cam_2.yml", "sequence_id": "seq_a", "frame_index": 0}
{"id": "seq_a_1", "input": "raws/seq_001.raw", "config": "presets/AD_cam_2.yml", "sequence_id": "seq_a", "frame_index": 1}
```
- The CLI always writes images (save_outputs); use the API for
  compute-only runs.
- `--batch-size` larger than `--max-batch-size` is an invocation error
  (exit 2), not a silent cap; `auto` uses a bounded
  memory heuristic (an estimate, not a guarantee — actual cohort sizes may
  be lower). Front-half and render/post stages run on real leading-B
  tensors; the bounded 3A linear middle stays per-frame B=1 to preserve
  qualified controls. Input admission is bounded to
  `(prefetch+1) * max_batch_size` accepted frames.

### Outputs and reports

- Writers are collision-exclusive (numeric suffixes, shared-stem
  sidecars) and never overwrite existing outputs; relative subdirectories
  under the output root are preserved.
- JSONL reports contain one record per item in input order: actual
  artifact names, metadata, config SHA, warnings, feedback, error,
  actual batch size, device, and per-attempt metrics; a sibling
  `.summary.json` is written as a pair. Reports are exclusive-new —
  an existing report is an invocation error, never appended.
- Attempts are replicated per item in reports; dedupe by `attempt_id`
  before counting totals.
- Failures: unrelated items continue; a sequence failure blocks later
  frames of that sequence; `--fail-fast` stops new commits and marks
  remaining items blocked. Cancellation writes nothing past the boundary.
  Exit codes: 0 all-success (an empty input set reports 0 items), 1 any
  failed/blocked/cancelled, 2 invocation/manifest errors, 130 interrupt.
  A broken report sink exits 1 with a stderr diagnostic rather than a
  faked complete summary (an in-flight interrupt still exits 130).
- GPU OOM is classified by typed `torch.cuda.OutOfMemoryError` /
  `torch.OutOfMemoryError` only;
  cohorts recursively halve with fresh engines (never a CPU retry).

### Supported outputs and numerical qualification

RGB PNG 8/16, TIFF 16/float32-linear (signed headroom retained), JPEG 8
(lossy), YCbCr 4:4:4 and 4:2:2 (BT.601/709, full/limited, YUYV/UYVY).
Lossless formats are covered by lossless native-array gates; JPEG is
compared separately. Integer stages are qualified exact; final unsigned
8/16-bit outputs use max-1 / mean-0.1 gates at original bit depth;
RGB previews independently receive those same 8-bit gates.
floating outputs use `2e-6 + 2e-5 * |reference|` on finite values.
`profile_stages=False` timings are host-side launch/control wall time,
not GPU kernel time; `True` adds a CUDA synchronize per stage.

### Verification

- Local qualification artifacts (not distributed in a fresh checkout;
  reproducible via the validation harness and its frozen source/manifest
  SHA recipes; paths below are relative to `tmp/gpu_validation/`).
  Five representative full-frame real cases are currently qualified
  byte-exact on NVIDIA (native + preview, max 0 / mean 0 / mismatch 0;
  all stage, metadata and feedback groups passing):
  `m5/fv_ad_post_bnr_28b557d5/fv/report.json` — FV 1920x1536;
  `m5/fv_ad_post_bnr_28b557d5/ad/report.json` — AD 3840x2185 (very dark on
  both CPU and GPU due to the existing gain limit; no image-quality
  claims made);
  `m5/sdr_exp_fix_de876dc9/report.json` — SDR ColorChecker 2592x1536
  (initial preview failure was traced to float32 exponential rounding in
  BNR and fixed with a GPU-resident NumPy-reference arithmetic profile;
  the original failed evidence and unchanged thresholds are retained);
  `m5/triton_qualification_17739e72/triton_sdr_tiff/report.json` —
  Triton SDR TIFF 2880x1860, 24 stage comparisons;
  `m5/triton_qualification_17739e72/triton_hdr24_overlay/report.json` —
  HDR24 via an explicitly composed config (SDR base + partial HDR +
  packed-24 metadata overlay), 27 stage comparisons. The absent
  `base_hdr.yml` preset remains an existing config limitation, not a
  GPU-path fix; packed Bayer RG12p is not supported.
- Output-format coverage: `m5/real_output_roundtrip_e8996185/results.json`
  — 50 cases (5 real 512x512 Bayer ROI families x 10 output variants);
  all comparisons pass, all integer native and preview pixels byte-exact,
  45 lossless variants reopen identically, and float32 TIFF preserves
  signed/headroom ranges; the 5 JPEG variants are compared against the
  pre-compression native gate, not claimed lossless.
- All 13 demosaic variants also pass every integrated stage and final-image
  gate on a fixed real ColorChecker 128x128 ROI, with byte-exact native
  output and preview (`m5/real_demosaic_66b12e45/results.json`). This
  crop coverage is distinct from full-resolution algorithm-family checks.
- Broader corpus coverage: 20 full case/config pairs across 19 distinct
  source representations (15 additional declared/recorded-overlay cases
  + 5 previous full-frame cases) all pass with byte-identical native and
  preview
  (`m5/extended_corpus_299d1f55_summary.json`); the 17-path inventory has
  15 qualified under declared explicit settings — the other 2 are
  packed-RAW RG12p inputs blocked as unsupported. Four additional FV
  algorithm-family variants (HA-opt, PPG-opt, LMMSE-fast, AHD-opt after a
  float32-exponential fix in `_opt_alpha`) also pass byte-exact
  (first three under `m5/full_demosaic_families_49950d48/`; the corrected
  AHD-opt evidence is `m5/ahd_exp_fix_40350fe0/report.json`. The initial
  AHD failure is retained; the corrected run reused its frozen CPU capture
  and unchanged gates), bringing the full-frame set to 24 case-config
  comparisons. Nine real 128x128 ROI integration overlays
  covering tone-operator/backend pairs, LSC+sharpen, NLM, CLAHE,
  odd-origin CFA crop, scaling and output variants pass their fixed
  gates (max native diff 1 code, preview 0;
  `m5/real_interactions_a8575397_frozen_summary.json`) — these are
  integration overlays, not calibration or image-quality claims.
- Unit gates (fresh post-fix runs): the full GPU suite passed all 7111
  tests (0 failed/skipped, NVIDIA RTX 4050,
  `m5/final_gpu_full_e6cb26cd`); the CPU root suite passed 1437 with 42
  skips (41 no-Tk-display, 1 no-DISPLAY;
  `m5/final_cpu_full_fc3d7ea2`). An earlier leaked-fake-RTC-driver cache
  isolation bug in the GPU suite was fixed in test code.
- Run tests with `--require-gpu --gpu-vendor nvidia`; without qualifying
  hardware the required mode fails rather than silently qualifying
  anything (optional CPU-only runs may skip). ROCm/HIP paths are
  implemented but not hardware-qualified. Performance benchmarking is
  in progress — a local tool (`tools.gpu_validation.benchmark`) measures
  wall/GPU-event latency per workload mode against frozen data, but no
  speed or throughput comparisons are published yet.

## Troubleshooting

- RAW not found:
  - Check `platform.filename` in config or pass the right folder via `ISP_RAW_DATA`
- Wrong geometry / reshape errors:
  - Verify `sensor_info.width`, `sensor_info.height`, `sensor_info.bit_depth`
- Unexpected colors:
  - Confirm `sensor_info.bayer_pattern`, WB gains, and CCM entries
- GUI opens but no preview:
  - Load both config and RAW, then click `Process`
- Analysis menu items disabled:
  - Run at least one successful `Process` to produce preview image

## Attribution and License

- Developer: Brian Deegan
- Upstream basis (portions): [Infinite-ISP](https://github.com/10x-Engineers/Infinite-ISP), 10xEngineers
- Additional attribution details: `NOTICE`

Copyright 2026 Brian Deegan

Licensed under the Apache License, Version 2.0:
[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)
