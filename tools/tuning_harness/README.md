# 3A tuning KPI harness (r04q)

The frozen KPI harness (v1.1) and experiment protocol used to tune the r04q AE/AWB/tone-mapping presets in
`config/r04q/`. It is kept for reproducibility. It is not part of the ISP and nothing imports it.

| File | Purpose |
|---|---|
| `PROTOCOL.md` | Experiment protocol: roles, user constraints C1–C3, scene sets, KPIs, scoring and stopping criteria. |
| `common.py` | Scene list and lighting classes, raw reference decode, active-area masks, metering ROIs, per-frame stats. |
| `scene_ref.py` | Builds the active-area masks (`masks/<cam>.npy`) and ISP-independent per-frame scene references. |
| `static_px.py` | Finds static pixels (ego body, overlays) for the masks. |
| `run_eval.py` | Runs every scene sequence through a brilliantISP checkout with 3A state carried frame to frame, then computes KPIs. |
| `kpi.py` | KPI definitions: frame-to-frame brightness stability, AWB cast, exposure error, saturation. |
| `overlay.py` | Debug overlays of masks and ROIs. |
| `FROZEN.sha256` | Checksums of the frozen harness files; any KPI run must match them. |
| `extract_scenes.sh`, `run_baseline.sh` | How the scene sequences were extracted and the baseline was run. |

## Running it

The harness expects the layout used on the tuning machine:
- `~/isp_tuning/harness/`: these files;
- `~/isp_tuning/scenes/<scene>/<cam>/raw/frame_*.raw`: the extracted sequences;
- `~/isp_tuning/runs/<name>/`: results.

`ROOT` in `common.py` sets this. The scenes come from the CAR recordings via `car_converter`; see
`extract_scenes.sh`.

```bash
cd ~/isp_tuning/harness && sha256sum -c FROZEN.sha256
python run_eval.py --isp-root /path/to/brilliantISP --svs-config config/r04q/SVS.yml \
  --ad-config config/r04q/AD.yml --name my_run --scenes tuning \
  --cam-config SVS_FV=config/r04q/SVS_FV.yml SVS_RV=config/r04q/SVS_RV.yml
python scene_ref.py   # first run only: builds masks/ and scene_ref.json
```

Not included: the masks (`masks/*.npy`, 22 MB) and static-pixel maps (44 MB). `scene_ref.py` and `static_px.py`
regenerate them. The scene data isn't included either.
