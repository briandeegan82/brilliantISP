# GPU Validation — NVIDIA Correctness (Current Source)

Date: 2026-11-05
HEAD: `e1ffa600e7894223ffd12c4817f0d71b853c4a61` (working tree clean at run time)

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB |
| Python | 3.12.3 |
| NumPy | 1.26.4 |
| SciPy | 1.17.1 |
| PyTorch | 2.9.1+cu128 (CUDA 12.8, `cuda.is_available()=True`) |
| pytest | 7.4.4 (`.venv-gpu`) |
| DISPLAY | `:0` (present; no headless skips) |

No competing GPU processes were running during validation (only Xorg, 45 MiB).

## Unit test gates

### GPU suite (`tests/gpu`)

Command:

```
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLBACKEND=Agg .venv-gpu/bin/python -m pytest tests/gpu -q \
  --require-gpu --gpu-vendor nvidia --junitxml=tmp/gpu_validation/gpu_run_20261105_170800/gpu_tests.xml
```

Result: **7111 passed, 0 failed, 0 skipped** in 139.750 s (exit 0).

### CPU suite (`tests`, excluding `tests/gpu`)

Command:

```
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLBACKEND=Agg .venv-gpu/bin/python -m pytest tests --ignore=tests/gpu -q \
  --junitxml=tmp/gpu_validation/gpu_run_20261105_170800/cpu_tests.xml
```

Result: **1487 passed, 0 failed, 0 skipped** in 62.974 s (exit 0).

Note: the CPU suite was run with `.venv-gpu/bin/python` because `venv/` does not contain pytest. NumPy 1.26.4 / SciPy 1.17.1 are identical in both environments; PyTorch is lazily imported and the tests executed on CPU. No packages were installed; no environment modifications were made.

Evidence: `tmp/gpu_validation/gpu_run_20261105_170800/` — `gpu_tests.xml`, `pytest.log`, `cpu_tests.xml`, `cpu_pytest.log`, `environment.txt`.

## Current-source validation driver

Validation driver `tmp/gpu_validation/validate_current.py` was executed with phases `full`, `outputs`, and `demosaic` against evidence root `tmp/gpu_validation/current_e1ffa60_20261105` using `PYTHONPATH=<repo> MPLBACKEND=Agg .venv-gpu/bin/python`. `verified_evidence.json` records identities for the result summaries, JUnit reports, and environment record.

### `full` phase — 7 cases

| Case | Compared stages |
|---|---:|
| `fv` | 36 |
| `ad` | 33 |
| `sdr_colorchecker` | 36 |
| `triton_sdr_tiff` | 24 |
| `triton_hdr24_overlay` | 27 |
| `ad_night_hable_65536` | 24 |
| `localtone_sdr` | 33 |

- Every stage / control / feedback / config / metadata / native / preview gate passes.
- First 6 cases: byte-exact native and preview outputs.
- `localtone_sdr` (full 2592×1536 pipeline combining Durand tone mapping, NLM, CLAHE, LSC, sharpen):
  - native: max diff 1 code, mean 0.000020512501071673526
  - preview: max diff 1 code, mean 0.00000016744898834019203
  - physical `uint16` native / `uint8` preview; unchanged max-1 / mean-0.1 gates.
  - This is a pipeline-integration result, **not** an image-quality or calibration claim.

### `outputs` phase — 50 cases

5 real 512-ROI families × 10 output formats. All byte-exact CPU/GPU for native and preview. 45 lossless cases reopen byte-exact; the 5 JPEG cases compare pre-compression (JPEG is not lossless). Float TIFF cases retained negative values and headroom where applicable.

### `demosaic` phase — 13 algorithms

All 13 demosaic algorithms pass integrated-stage validation on a 128×128 real ColorChecker ROI with byte-exact native and preview output. This is **not** a full-size, all-13-algorithms claim.

## Source integrity

Verified unchanged between phases and at end of run:

- 174 CPU/config/test file hashes unchanged
- 130 GPU/harness/test file hashes unchanged

No processing code, CPU engine, GUI, camera presets, or numerical gates were modified.

## Limitations

- NVIDIA qualification only (RTX 4050 Laptop). AMD/ROCm has **not** been qualified on this host; AMD qualification requires a suitable ROCm PyTorch environment and `--gpu-vendor amd`.
- Bayer RG12p packed RAW is unsupported.
- `triton_hdr24` uses an explicit SDR base plus HDR and packed-24 overlay because no `base_hdr.yml` exists.
- Older, broader 19-source/corpus claims were not freshly re-run; the current tests do not recover historical missing evidence.
- The historical performance benchmark is a **separate, incomplete** experiment: it cannot be resumed as the same experiment because 6 frozen CPU/config/test hashes differ (the 130 GPU hashes are unchanged). **No speedup or throughput claim is made here.** Old benchmark matrix artifacts are untouched.
