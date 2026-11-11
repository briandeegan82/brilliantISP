# brilliantISP 3A tuning experiment: protocol (v1.1)

User request (verbatim): "I want to get better performance from brilliantISP. Specifically, I want to tune the ISP
for both the SVS cameras and for the AD cameras. Right now, the frame to frame brightness is unstable, and the AWB is
prone to colour casts. Also, the SVS is too dark. The goal is stable frame to frame brightness and colour accuracy,
and better exposure accuracy. Spin one agent to tune the ISP, and a different agent to review and feedback.
Test and fine tune on multiple scenes. Use KPIs to verify performance. enter bypass mode to run the experiments.
Define experiment stopping criteria."

v1.1 changes vs v1.0: these follow the reviewer audit in `rounds/r00/REVIEW.md`, issues 1-9. See the KPI table and
scoring below. The stopping criteria are restated. `ORCHESTRATOR_NOTES.md` carries orchestrator messages, newest
first: re-read it at the end of every step and before starting any run.

All rounds were run with harness v1.1. The committed harness is v1.2, a lint-only reformat that
reproduces the v1.1 numbers exactly; see `README.md`.

## Roles
- Orchestrator (parent Devin session): owns the data, the frozen harness and this protocol, and relays between agents.
- Tuner: the only agent that edits ISP code/config, and only in `~/isp_tuning/brilliantISP-tuner` (branch `tuning/3a-stability`).
- Reviewer: independent. Never edits the tuner tree. Re-runs and inspects results, writes reviews, and can veto.

## User constraints (hard rules; any violation is an automatic REJECT)
- C1 (added 2026-11-09): every candidate config, SVS and AD, sets `sensor_info.embedded_rows_top: 4` and
  `sensor_info.embedded_rows_bottom: 4`.
- C2 (added 2026-11-09): tone mapping must be a global, monotone curve, applied after demosaic, for both SVS and AD
  (AD feeds perception).
  - Allowed: `hable` (a scene-adaptive white point is allowed), `reinhard`, `aces_fitted`.
  - Banned: `durand`, and any local, spatially varying or per-frame-renormalised operator.
  - Any adaptive curve parameter must come from global frame statistics, be temporally filtered with constants in
    seconds, and be applied identically to every pixel.
- C3 (added 2026-11-09): config granularity.
  - Use one base config for SVS (`rNN_SVS.yml`) and one for AD (`rNN_AD.yml`). AD_FL and AD_RR both use the AD base
    unchanged.
  - SVS_FV and SVS_RV may each have a per-position config (`rNN_SVS_FV.yml`, `rNN_SVS_RV.yml`). Each must equal the
    SVS base except for AE/AWB statistics-ROI keys: metering window or region, lens-circle centre/radius, ego-body
    exclusion boundary, and spatial metering weights. Nothing else may differ (no gains, caps, targets, CCM, tone
    or noise settings).
  - ROI geometry must describe the camera's mounting and optics: the lens circle and the ego body, measured by the
    tuner from the raw data.
    - It must not be copied from harness masks, `EGO_CUT`, `OCCLUDERS` or the road ROI.
    - It must not be fitted to KPI results.
  - The reviewer checks each per-position file with a YAML diff against the base.
  - Run per-position configs with
    `run_eval.py ... --cam-config SVS_FV=config/experiments/rNN_SVS_FV.yml SVS_RV=config/experiments/rNN_SVS_RV.yml`.
    That option was added to the runner on 2026-11-09; KPI definitions are unchanged and the harness is re-frozen.
- Saturation limit: K6 (any channel >= 250) at p95 must be <= 1% day / 2% low / 3% night. The user confirmed these
  limits. The 245-249 parking guard applies.

## Layout (GB10, home /home/ee5108)
- `~/isp_tuning/scenes/<scene>/<cam>/raw/frame_*.raw`: 200 consecutive frames at 10 fps per sequence (T6: 81, H3: 72).
  Cameras: SVS_FV and SVS_RV (1920x1536); AD_FL and AD_RR (3840x2185, stitched). The raster CFA is GBRG.
  - Rows 0-4 are embedded sensor data, and AD also has 20 at the bottom.
  - Both shipped configs currently set embedded_rows_* = 0.
- Tuning set: T1_urban_day, T2_canopy_static (fuel-station forecourt, not static), T3_bridge_sky, T4_motorway_dusk,
  T5_twilight_rural, T6_night_lights.
- Held-out set: H1_urban_day, H2_night_lights, H3_night_dark. Never tune on these, and never look at their per-frame
  images before the candidate is frozen for the round. Known caveat: T1, T3 and H1 come from the same recording.
- `~/isp_tuning/harness/`: the FROZEN KPI harness v1.1. Checksums are in `FROZEN.sha256`; verify with `sha256sum -c`.
  - Masks exclude the outside of the fisheye circle, the SVS ego-vehicle body and the AD_RR occluder.
  - Only the orchestrator edits the harness.
- `~/isp_tuning/runs/<name>/`: run outputs (`kpi.md`, `kpi.json`, per-frame `frame_stats.json`, `thumbs/`, configs, diff).
- `~/isp_tuning/rounds/rNN/`: `PROPOSAL.md` (tuner) and `REVIEW.md` (reviewer).
- Baselines, both on the shipped configs at brilliantISP 87bdbf4 (clean):
  - `runs/r00v11_baseline_shipped` is the v1.1 baseline. Compare every round against this run.
  - `runs/r00_baseline_shipped` is the v1.0 baseline (historical).

## Running
```
P=~/raw-dataset-converter/.venv/bin/python
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
$P ~/isp_tuning/harness/run_eval.py --isp-root ~/isp_tuning/brilliantISP-tuner \
   --svs-config config/SVS_cam.yml --ad-config config/AD_cam_2.yml --name rNN_<slug> --scenes tuning [--workers N]
$P ~/isp_tuning/harness/kpi.py compare ~/isp_tuning/runs/r00v11_baseline_shipped ~/isp_tuning/runs/rNN_<slug>
```
- `--scenes` accepts tuning | heldout | all | explicit scene names. Use `--cams` to run a subset of cameras.
- Each sequence carries a persistent `ProcessingState` frame to frame, as on the car.
- Throughput with ~12 parallel workers is roughly 6-10 s/frame for AD and 2-3 s/frame for SVS. A tuning run takes
  about 40-60 min.
- CPU sharing (20 cores): the tuner may use up to 14 workers, and the reviewer at most 4.
- Launch long runs with `setsid` and stop them by process group. Killing only the Pool parent leaves orphaned workers
  that keep writing into the run dir.
- Use `--max-frames` only for quick debugging, never for reported numbers.
- Run names are unique. Never delete another agent's runs.

## Bypass mode
"Bypass" means 3A bypass:
- AE off: `auto_exposure.is_enable: false`, `digital_gain.is_auto: false`, fixed `current_gain`.
- AWB off: `auto_white_balance.is_enable: false`, fixed `white_balance.r_gain/b_gain`.

Put bypass configs in separate files (e.g. `config/experiments/*.yml`) and use them for controlled one-variable
experiments:
1. Establish a 3A-off baseline.
2. Attribute each failure to its block: tone-mapper normalisation, AWB, embedded rows, CCM, AE target.
3. Re-enable 3A one block at a time.

Final candidates are always judged with 3A ON, because that is how the car runs.

## KPIs (harness v1.1, per sequence, first 10 frames excluded as warm-up)
eps = 0.005 linear Y. Scene-steady pair: |d log2 G_scene| < 0.03 stop.

| KPI | definition | day | low | night |
|---|---|---|---|---|
| K1 ISP step | p95 \|d log2(Y+eps) - d log2(G)\| on scene-steady pairs (stops) | <=0.03 | <=0.03 | <=0.03 |
| K2 gain hunting | sign reversals/s of log2(Y+eps) - log2(G), steps >=0.05 stop | <=1 | <=1 | <=1 |
| K3 AWB jitter | p95 frame-to-frame Delta u'v'; chroma-steady pairs (1% day, 2% low/night) with Ymean > 0.01 | <=0.003 | <=0.003 | <=0.003 |
| K4 road cast | median C*ab of the road ROI mean after scaling to L*=50; road L* < 10 means fail; n/a on AD_RR | <=5 | <=7 | <=10 |
| K5 exposure error | median log2(mean output linear Y / 0.18), stops | +-1/3 | -2/3..+1/3 | -1..+1/3 |
| K6 clipping | p95 fraction of active pixels with any channel >=250 | <=1% | <=2% | <=3% |
| K7 crushed shadows | median fraction of active pixels with luma <=4 | <=5% | <=15% | <=30% |
| K8 exposure spread | p90 \|e_t - median(e)\|, e_t = log2((Y_t+eps)/0.18) (responsiveness/consistency) | <=0.5 | <=0.75 | <=1.0 |
| K9 road lightness | median road ROI L* (lower bound); n/a on AD_RR | >=30 | >=20 | >=10 |

- n/a = not measurable from the scene data (too few steady pairs, or no road ROI). It is excluded from pass and score.
- nan = undefined because of the ISP output. It counts as the worst value.
- Failure score per sequence = mean over scored KPIs of max(0, |value/threshold| - 1), capped at 10. 0 means all
  passing; lower is better.
- The primary progress measure is the number of passing (sequence, KPI) cells. `kpi.md` also lists the 3A traces:
  gain-index changes, AE stop reasons and AWB gain ranges.
- Scalar KPIs are necessary but not sufficient. Always inspect thumbnails: sky, signs, night lamps, road colour.

## Round procedure (max 8 rounds)
1. Tuner:
   - Pursue one coherent hypothesis per round. It may touch several knobs, but say why.
   - Implement it on the branch and commit (one commit per round, message `rNN: ...`).
   - Run `--scenes tuning`, then, after freezing the commit, `--scenes heldout`.
   - Write `rounds/rNN/PROPOSAL.md`: hypothesis, change summary, commit SHA, run names, KPI deltas vs the v1.1 baseline
     and vs the previous best, 3A traces, image observations and risks. Then STOP and wait.
2. Reviewer:
   - Verify the harness checksums, and check that the run manifest matches the claimed commit.
   - Read the diff and check the KPIs and images independently (a spot re-run is allowed).
   - Look for metric gaming.
   - Write `rounds/rNN/REVIEW.md`: verdict ACCEPT / ACCEPT-WITH-CHANGES / REJECT, numbered actionable feedback, and any
     veto reasons. Then STOP and wait.
3. The orchestrator relays the review. The next round starts from the best ACCEPTED commit.

Rules:
- Algorithm changes go in the CPU engine only, as opt-in config keys (default off = old behaviour).
- Do not modify `brilliantisp_gpu/`, the GUI, or other repos.
- No per-scene or per-sequence special cases. The ISP must not use scene names, frame indices, harness ROIs/masks or
  reference data.
- Express temporal constants in seconds, with the frame rate as a config key: the data here is 10 fps, the car runs at 30 fps.
- No output pedestal above 2 codes. Darkening, desaturation or freezing 3A to move a KPI will be rejected.
- Keep existing tests passing: run `~/raw-dataset-converter/.venv/bin/python -m pytest -q` from the tuner tree,
  excluding tests/gpu.

## Stopping criteria (stop at the first that holds)
1. SUCCESS: every scored cell passes on every sequence (tuning AND held-out, SVS and AD), and the reviewer ACCEPTs.
2. PLATEAU: two consecutive accepted rounds each add no new passing tuning cells AND improve the tuning failure score
   by less than 2% relative.
3. BUDGET: 8 tuner/reviewer rounds completed.

Veto: the reviewer rejects any round where a sequence's KPI threshold-ratio worsens by more than 10% (and ends above
0.5) vs the previous best, or where a passing cell becomes failing. `kpi.py compare` flags these as REGRESS / NEW FAIL.
The exception is when the tuner justifies the change and the reviewer agrees it is a measurement artefact. Held-out
results are reported every round but never used to choose parameters.
