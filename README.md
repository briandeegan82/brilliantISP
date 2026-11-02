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
