"""Frozen KPI definitions, thresholds, pass/fail and failure score (harness v1.2).

  python kpi.py <run_dir>                 # (re)compute kpi.json / kpi.md
  python kpi.py compare <runA> <runB>     # per-sequence deltas, flags >10% regressions

v1.1 changes (reviewer r00 audit): floored/ISP-induced K1/K2, K3 only on non-black frames,
K4 normalised to L*=50, new K8 (exposure responsiveness) and K9 (road lightness floor),
n/a-by-data excluded from pass/score, failure-only score, passing-cell count as primary
progress measure.

v1.2: pylint-clean reformat of v1.1; KPI definitions, thresholds and outputs are unchanged.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402  # pylint: disable=wrong-import-position
    CAMERAS, D65, FPS, HARNESS_VERSION, NO_ROAD_CAMS, ROOT, SCENE_CLASS, SCENES, STATIC_SCENES,
    xyz_to_lab)

WARMUP_FRAMES = 10            # first 1 s: AE/AWB start from the config seed and converge
EPS = 0.005                   # luminance floor (linear Y) for log-domain stability measures
LUM_STEADY_STOPS = 0.03       # |d log2 scene G| below this => scene brightness steady
# |d ln(R/G)|, |d ln(B/G)| gate on the scene reference
CHROMA_STEADY = {"day": 0.01, "low": 0.02, "night": 0.02}
K3_MIN_Y = 0.01               # K3 only on pairs where both frames have mean output Y above this
HUNT_STEP_STOPS = 0.05        # effective-gain steps smaller than this are ignored for hunting
EXPOSURE_TARGET_Y = 0.18      # mean linear output luminance target (L* ~ 50)
# road L* below this: road colour unmeasurable -> K4 fails (ISP's fault)
ROAD_MIN_L = 10.0
MIN_PAIRS = 10
FAIL_CAP = 10.0               # per-KPI failure excess cap
NA = "n/a"

# KPI -> per-class threshold. Tuples are (lo, hi) ranges; "min:" prefix in DESC marks lower bounds.
THRESH = {
    "K1_isp_step_p95": {"day": 0.03, "low": 0.03, "night": 0.03},
    "K2_gain_hunting_per_s": {"day": 1.0, "low": 1.0, "night": 1.0},
    "K3_awb_jitter_p95_duv": {"day": 0.003, "low": 0.003, "night": 0.003},
    "K4_road_cast_C50": {"day": 5.0, "low": 7.0, "night": 10.0},
    "K5_exposure_err_stops": {"day": (-1 / 3, 1 / 3), "low": (-2 / 3, 1 / 3),
                              "night": (-1.0, 1 / 3)},
    "K6_clip_frac_p95": {"day": 0.01, "low": 0.02, "night": 0.03},
    "K7_crush_frac_median": {"day": 0.05, "low": 0.15, "night": 0.30},
    "K8_exposure_spread_p90": {"day": 0.5, "low": 0.75, "night": 1.0},
    "K9_road_L_median": {"day": 30.0, "low": 20.0, "night": 10.0},
}
LOWER_BOUND = {"K9_road_L_median"}
DESC = {
    "K1_isp_step_p95": ("p95 |d log2(Y+eps) - d log2(G_scene)| on scene-steady pairs (stops): "
                        "ISP-induced brightness step"),
    "K2_gain_hunting_per_s": ("sign reversals/s of effective gain log2(Y+eps) - log2(G_scene), "
                              "steps >= 0.05 stop"),
    "K3_awb_jitter_p95_duv": ("p95 frame-to-frame Delta u'v' of output mean chromaticity, "
                              "scene-chroma-steady pairs with Ymean > 0.01"),
    "K4_road_cast_C50": ("median CIELAB C*ab of road ROI mean after scaling to L*=50; "
                         "road L* < 10 => fail"),
    "K5_exposure_err_stops": "median log2(mean output linear Y / 0.18)",
    "K6_clip_frac_p95": "p95 fraction of active pixels with any channel >= 250",
    "K7_crush_frac_median": "median fraction of active pixels with luma <= 4",
    "K8_exposure_spread_p90": ("p90 |e_t - median(e)|, e_t = log2((Y_t+eps)/0.18): "
                               "exposure responsiveness/consistency (stops)"),
    "K9_road_L_median": "min: median road ROI L* (road-referred exposure floor)",
}


def _ratio(k: str, v, cls: str) -> float:
    if v == NA:
        return float("nan")
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return 1.0 + FAIL_CAP  # undefined because of the ISP output => worst
    t = THRESH[k][cls]
    if k in LOWER_BOUND:
        return t / max(v, 1e-6)
    if isinstance(t, tuple):
        lo, hi = t
        return v / hi if v >= 0 else v / lo
    return v / t


def _lab_finv(f: float, d: float) -> float:
    return f ** 3 if f > d else 3 * d * d * (f - 4 / 29)


def _road_c50(road: dict) -> float:
    l_star, a, b = road.get("L"), road.get("a"), road.get("b")
    if l_star is None or not np.isfinite(l_star) or l_star < ROAD_MIN_L:
        return float("nan")
    fy = (l_star + 16) / 116
    fx, fz = fy + a / 500, fy - b / 200
    d = 6 / 29
    xyz = np.array([_lab_finv(fx, d), _lab_finv(fy, d), _lab_finv(fz, d)]) * D65
    xyz *= EXPOSURE_TARGET_Y / max(xyz[1], 1e-12)
    lab = xyz_to_lab(xyz)
    return float(np.hypot(lab[1], lab[2]))


def _info(st: list[dict], post: list[dict], y_out: np.ndarray, e: np.ndarray,
          road_l: np.ndarray) -> dict:
    fb = [s.get("feedback") or {} for s in st]
    gi = [f.get("digital_gain_index") for f in fb]
    awb = [f.get("awb_gains") for f in fb if f.get("awb_gains")]
    has_road_l = np.isfinite(road_l).any()
    return {
        "road_L_median": float(np.nanmedian(road_l)) if has_road_l else float("nan"),
        "road_hue_deg_median": float(np.degrees(np.nanmedian(
            [math.atan2(s["road"]["b"], s["road"]["a"]) for s in post]))),
        "Ymean_median": float(np.median(y_out[WARMUP_FRAMES:])),
        "K5_p10": float(np.percentile(e, 10)), "K5_p90": float(np.percentile(e, 90)),
        "gain_index_changes": int(sum(1 for a, b in zip(gi[1:], gi[:-1])
                                      if a != b and a is not None)),
        "gain_index_set": sorted({x for x in gi if x is not None}),
        "ae_stop_reasons": sorted({str(f.get("ae_stop_reason")) for f in fb
                                   if f.get("ae_stop_reason") is not None}),
        "awb_r_range": ([float(min(a[0] for a in awb)), float(max(a[0] for a in awb))]
                        if awb else None),
        "awb_b_range": ([float(min(a[1] for a in awb)), float(max(a[1] for a in awb))]
                        if awb else None),
        "luma_p1_median": (float(np.median([s["luma_p1"] for s in post]))
                           if "luma_p1" in st[0] else None),
        "luma_clip_p95": (float(np.percentile([s["luma_clip_frac"] for s in post], 95))
                          if "luma_clip_frac" in st[0] else None),
        "Lstar_rms_median": (float(np.median([s["Lstar_rms"] for s in post]))
                             if "Lstar_rms" in st[0] else None),
    }


def seq_kpis(scene: str, cam: str, stats: list[dict], ref: list[dict]) -> dict:
    """KPIs, threshold ratios, pass/fail and failure score of one scene/camera sequence."""
    cls = SCENE_CLASS[scene]
    n = min(len(stats), len(ref))
    st, rf = stats[:n], ref[:n]
    y_out = np.array([s["Ymean"] for s in st])
    up = np.array([s["up"] for s in st])
    vp = np.array([s["vp"] for s in st])
    g_ref = np.array([r["G"] for r in rf])
    lr = np.log(np.array([r["R"] for r in rf]) / g_ref)
    lb = np.log(np.array([r["B"] for r in rf]) / g_ref)
    idx = np.arange(WARMUP_FRAMES + 1, n)
    log_y = np.log2(y_out + EPS)
    log_g = np.log2(np.maximum(g_ref, 1e-9))
    d_log_g = np.abs(np.diff(log_g))[idx - 1]
    if scene not in STATIC_SCENES:
        lum_steady = d_log_g < LUM_STEADY_STOPS
    else:
        lum_steady = np.ones_like(d_log_g, bool)
    cs = CHROMA_STEADY[cls]
    chroma_steady = (lum_steady & (np.abs(np.diff(lr))[idx - 1] < cs)
                     & (np.abs(np.diff(lb))[idx - 1] < cs)
                     & (y_out[idx] > K3_MIN_Y) & (y_out[idx - 1] > K3_MIN_Y))
    step = np.abs(np.diff(log_y) - np.diff(log_g))[idx - 1]
    duv = np.hypot(np.diff(up), np.diff(vp))[idx - 1]
    k: dict = {}
    k["K1_isp_step_p95"] = (float(np.percentile(step[lum_steady], 95))
                            if lum_steady.sum() >= MIN_PAIRS else NA)
    g = log_y - log_g
    dg = np.diff(g[WARMUP_FRAMES:])
    sig = np.sign(dg[np.abs(dg) >= HUNT_STEP_STOPS])
    rev = int((sig[1:] * sig[:-1] < 0).sum()) if sig.size > 1 else 0
    k["K2_gain_hunting_per_s"] = rev / max((n - WARMUP_FRAMES) / FPS, 1e-9)
    k["K3_awb_jitter_p95_duv"] = (float(np.percentile(duv[chroma_steady], 95))
                                  if chroma_steady.sum() >= MIN_PAIRS else NA)
    post = st[WARMUP_FRAMES:]
    if cam in NO_ROAD_CAMS:
        k["K4_road_cast_C50"] = NA
    else:
        c50 = np.array([_road_c50(s["road"]) for s in post])
        k["K4_road_cast_C50"] = (float(np.median(c50)) if np.isfinite(c50).mean() >= 0.5
                                 else float("nan"))
    e = np.log2((y_out[WARMUP_FRAMES:] + EPS) / EXPOSURE_TARGET_Y)
    k["K5_exposure_err_stops"] = float(np.median(
        np.log2(np.maximum(y_out[WARMUP_FRAMES:], 1e-9) / EXPOSURE_TARGET_Y)))
    k["K6_clip_frac_p95"] = float(np.percentile([s["clip_frac"] for s in post], 95))
    k["K7_crush_frac_median"] = float(np.median([s["crush_frac"] for s in post]))
    k["K8_exposure_spread_p90"] = float(np.percentile(np.abs(e - np.median(e)), 90))
    road_l = np.array([s["road"]["L"] for s in post], float)
    if cam in NO_ROAD_CAMS:
        k["K9_road_L_median"] = NA
    else:
        k["K9_road_L_median"] = (float(np.nanmedian(road_l)) if np.isfinite(road_l).any()
                                 else 0.0)
    ratios = {kk: _ratio(kk, v, cls) for kk, v in k.items()}
    scored = {kk: r for kk, r in ratios.items() if not math.isnan(r)}
    excess = {kk: min(FAIL_CAP, max(0.0, abs(r) - 1.0)) for kk, r in scored.items()}
    info = {"frames": n, "lum_steady_pairs": int(lum_steady.sum()),
            "chroma_steady_pairs": int(chroma_steady.sum())}
    info.update(_info(st, post, y_out, e, road_l))
    return {"scene": scene, "cam": cam, "cls": cls, "kpi": k, "ratio": ratios,
            "na": [kk for kk in k if kk not in scored],
            "cells_pass": sum(1 for r in scored.values() if abs(r) <= 1.0),
            "cells_scored": len(scored),
            "pass": all(abs(r) <= 1.0 for r in scored.values()),
            "fail_score": float(np.mean(list(excess.values()))) if excess else 0.0, "info": info}


def _error_seq(scene: str, cam: str, scene_cls: str) -> dict:
    return {"scene": scene, "cam": cam, "cls": scene_cls, "error": True, "pass": False,
            "kpi": {k: float("nan") for k in THRESH}, "ratio": {k: 1 + FAIL_CAP for k in THRESH},
            "na": [], "cells_pass": 0, "cells_scored": len(THRESH), "fail_score": FAIL_CAP,
            "info": {}}


def compute_kpis(run: Path) -> dict:
    """Compute KPIs for every sequence of a run; write ``kpi.json`` and ``kpi.md``."""
    run = Path(run)
    seqs = []
    for scene, scene_cls in SCENE_CLASS.items():
        for cam in CAMERAS:
            f = run / scene / cam / "frame_stats.json"
            if not f.exists():
                continue
            stats = json.loads(f.read_text())
            ref = json.loads((SCENES / scene / cam / "scene_ref.json").read_text())
            if (run / scene / cam / "error.txt").exists() or len(stats) < len(ref):
                seqs.append(_error_seq(scene, cam, scene_cls))
                continue
            r = seq_kpis(scene, cam, stats, ref)
            r["error"] = False
            seqs.append(r)
    groups = {}
    for split in ("T", "H"):
        for cls in ("SVS", "AD"):
            ss = [s for s in seqs if s["scene"].startswith(split) and CAMERAS[s["cam"]] == cls]
            if ss:
                groups[f"{split}_{cls}"] = {
                    "fail_score": float(np.mean([s["fail_score"] for s in ss])),
                    "cells": (f"{sum(s['cells_pass'] for s in ss)}/"
                              f"{sum(s['cells_scored'] for s in ss)}"),
                    "seq_pass": f"{sum(s['pass'] for s in ss)}/{len(ss)}"}
    tun = [s for s in seqs if s["scene"].startswith("T")]
    out = {"harness_version": HARNESS_VERSION, "run": run.name,
           "fail_score_tuning": (float(np.mean([s["fail_score"] for s in tun])) if tun
                                 else float("nan")),
           "cells_tuning": (f"{sum(s['cells_pass'] for s in tun)}/"
                            f"{sum(s['cells_scored'] for s in tun)}"),
           "groups": groups, "sequences": seqs}
    (run / "kpi.json").write_text(json.dumps(out, indent=1))
    (run / "kpi.md").write_text(_markdown(out))
    return out


def _fmt(v) -> str:
    if v == NA:
        return NA
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "nan"
    return f"{v:.4f}" if abs(v) < 0.1 else f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"


def _markdown(o: dict) -> str:
    keys = list(THRESH)
    lines = [f"# KPI report: {o['run']} (harness v{o['harness_version']})", "",
             f"Tuning set: passing cells **{o['cells_tuning']}**, failure score "
             f"**{_fmt(o['fail_score_tuning'])}** "
             f"(mean over KPIs of max(0, |value/threshold| - 1), capped at {FAIL_CAP:g}; "
             "0 = all passing; lower is better)", ""]
    for g, v in o["groups"].items():
        lines.append(f"- {g}: cells {v['cells']}, sequences passing {v['seq_pass']}, "
                     f"failure score {_fmt(v['fail_score'])}")
    lines += ["", "| scene | cam | " + " | ".join(k.split('_', maxsplit=1)[0] for k in keys)
              + " | cells | fail | pass |",
              "|" + "---|" * (len(keys) + 5)]
    for s in o["sequences"]:
        cells = []
        for k in keys:
            v, r = s["kpi"][k], s["ratio"][k]
            bad = (not math.isnan(r)) and abs(r) > 1
            cells.append(("**" + _fmt(v) + "**") if bad else _fmt(v))
        lines.append(f"| {s['scene']} | {s['cam']} | " + " | ".join(cells) +
                     f" | {s['cells_pass']}/{s['cells_scored']} | {_fmt(s['fail_score'])} | "
                     f"{'Y' if s['pass'] else 'N'}{' ERR' if s.get('error') else ''} |")
    lines += ["", "Bold = fails threshold; n/a = not measurable from the scene data (excluded "
              "from pass/score); nan = undefined because of the ISP output (counted as worst). "
              "KPI definitions:"]
    for k in keys:
        lines.append(f"- {k}: {DESC[k]}; thresholds {THRESH[k]}")
    lines += ["", "3A traces (info): gain-index changes, AE stop reasons, AWB gain ranges", "",
              "| scene | cam | gain idx changes | gain idx set | AE stop | AWB r range | "
              "AWB b range | road L* | K5 p10..p90 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for s in o["sequences"]:
        i = s.get("info") or {}
        if not i:
            continue
        rr = i.get("awb_r_range") or [float("nan")] * 2
        br = i.get("awb_b_range") or [float("nan")] * 2
        lines.append(f"| {s['scene']} | {s['cam']} | {i['gain_index_changes']} | "
                     f"{i['gain_index_set'][:6]} | "
                     f"{','.join(i['ae_stop_reasons'])} | {rr[0]:.2f}-{rr[1]:.2f} | "
                     f"{br[0]:.2f}-{br[1]:.2f} | "
                     f"{_fmt(i['road_L_median'])} | {i['K5_p10']:.2f}..{i['K5_p90']:.2f} |")
    return "\n".join(lines) + "\n"


def compare(a: Path, b: Path) -> str:
    """Markdown table of per-cell changes from run ``a`` to run ``b`` (>10% regressions flagged)."""
    seqs_a = {(s["scene"], s["cam"]): s
              for s in json.loads((Path(a) / "kpi.json").read_text())["sequences"]}
    seqs_b = {(s["scene"], s["cam"]): s
              for s in json.loads((Path(b) / "kpi.json").read_text())["sequences"]}
    lines = [f"# compare {Path(a).name} -> {Path(b).name}", "",
             "| scene | cam | KPI | A | B | flag |", "|---|---|---|---|---|---|"]
    regress = newpass = newfail = 0
    for key in sorted(set(seqs_a) & set(seqs_b)):
        for k in THRESH:
            ra, rb = abs(seqs_a[key]["ratio"][k]), abs(seqs_b[key]["ratio"][k])
            if math.isnan(ra) or math.isnan(rb):
                continue
            flag = ""
            if ra > 1 >= rb:
                flag, newpass = "NEW PASS", newpass + 1
            elif rb > 1 >= ra:
                flag, newfail = "NEW FAIL", newfail + 1
            if rb > ra * 1.10 and rb > 0.5:
                flag, regress = (flag + " REGRESS>10%").strip(), regress + 1
            elif not flag and rb < ra * 0.9:
                flag = "better"
            if flag:
                lines.append(f"| {key[0]} | {key[1]} | {k} | {_fmt(seqs_a[key]['kpi'][k])} | "
                             f"{_fmt(seqs_b[key]['kpi'][k])} | {flag} |")
    lines.insert(1, f"New passing cells: {newpass}; new failing cells: {newfail}; "
                    f"regressions (>10% worse threshold-ratio and ratio > 0.5): {regress}")
    return "\n".join(lines) + "\n"


def main() -> None:
    """CLI: ``kpi.py <run>`` or ``kpi.py compare <runA> <runB>``."""
    if sys.argv[1] == "compare":
        print(compare(Path(sys.argv[2]), Path(sys.argv[3])))
    else:
        p = Path(sys.argv[1])
        p = p if p.is_absolute() else ROOT / "runs" / p
        compute_kpis(p)
        print((p / "kpi.md").read_text())


if __name__ == "__main__":
    main()
