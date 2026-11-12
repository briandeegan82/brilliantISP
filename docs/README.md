# BrilliantISP documentation index

## Current references

- `ARCHITECTURE.md` — native processing API (single v2 engine), results,
  stage events, cancellation, explicit sequence state, and output isolation.
- `CONFIGURATION.md` — config file loading, explicit `extends` inheritance,
  shared schema metadata, v1→v2 in-memory migration, normalization and
  validation as implemented now.
- `CODE_AUDIT.md` — M0 static audit ledger: architecture findings, per-file
  implementation inventory, test baseline, fixture gaps.
- `ISP_BLOCKS_AND_TUNING.md` — overview of pipeline blocks and tuning notes,
  including smooth AE, AWB temporal filtering, metering ROI, `rgb_norm` and
  the tuned `config/r04q/` presets.
- `BOLTISP_YAML_COMPATIBILITY.md` — boltISP export compatibility notes.

## GUI

- `GUI_DESIGN.md`, `GUI_DESIGN_Additional_Requirements.md` — design
  requirements. Note: these predate the calibration GUI implementation;
  `tools/isp_calibration_gui.py` exists today, and an integrated-workspace
  redesign is planned (see the modernization plan).

## Demosaic algorithm notes

- `HAMILTON_ADAMS_DEMOSAIC.md`, `PPG_DEMOSAIC.md`, `VNG_DEMOSAIC.md` —
  per-variant notes; will be folded into canonical block pages in later
  milestones.

## Historical material

- `Algorithm Description Document v1.1.md` / `.pdf` — upstream design
  document kept for provenance; it describes the original design, not
  necessarily the current implementation.
- `GAMMA_CORRECTION_FINAL_SOLUTION.md` — historical rationale for the gamma
  stage ordering; the v2 native output path has landed, so this remains
  provenance only.

## Development

- Tests: `tests/` (pytest). The project venv currently lacks pytest; install
  `requirements-dev.txt` (`pytest==7.4.4`) into it, or see `CODE_AUDIT.md`
  for the shim used during M0/M1a evidence runs.
- Runtime dependencies: `requirements.txt` (includes pyright).
- `RAW_INPUT.md` — the declared RAW decode contract (`sensor_info` storage/
  layout fields, containers, effective-CFA rules) and migration notes.
- Config validator: `python tools/validate_config.py --config <yml>`
  (`--config` repeatable, merge in order; `--json` for machine-readable
  output). See `CONFIGURATION.md` for exit codes and details.
- Config migration: `python tools/migrate_pipeline_config.py --config <yml>
  --output <new.yml> --report <new.json>` — v1→v2 migration, exclusive
  destinations. See `CONFIGURATION.md` for the contract.
- `blocks/tone_mapping.md` — unified tone-mapping engine contract
  (operators, backends, normalization, migration).
- `R04Q_CALIBRATION_NOTES.md` — SVS fisheye intrinsics from the vehicle workbook and the
  derivation of the r04q MVL/MVR metering ROIs.
- `../tools/tuning_harness/README.md` — frozen KPI harness and protocol used to tune r04q.
