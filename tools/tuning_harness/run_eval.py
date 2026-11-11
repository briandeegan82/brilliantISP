"""Frozen evaluation runner: process every scene sequence through a brilliantISP checkout
with 3A state carried frame to frame, record per-frame output stats, then compute KPIs.

Usage:
  ~/raw-dataset-converter/.venv/bin/python ~/isp_tuning/harness/run_eval.py \
      --isp-root ~/isp_tuning/brilliantISP-tuner --svs-config config/SVS_cam.yml \
      --ad-config config/AD_cam_2.yml --name r01_my_change \
      [--scenes tuning|heldout|all|T1_urban_day ...]

Config paths are relative to --isp-root unless absolute. Results: ~/isp_tuning/runs/<name>/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# pylint: disable=wrong-import-position
from common import (  # noqa: E402
    CAMERAS, HARNESS_VERSION, HELDOUT_SCENES, ROOT, SCENE_CLASS, SCENES, TUNING_SCENES,
    frame_stats, load_mask, raster_to_output, roi_slices)
from kpi import compute_kpis  # noqa: E402
# pylint: enable=wrong-import-position

WARMUP = 0  # KPIs apply their own warm-up exclusion; runner records every frame


def _jsonable(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, np.generic):
        return v.item()
    return str(v)


def run_sequence(job: dict) -> dict:
    """Process one scene/camera sequence with persistent 3A state; write its frame stats."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    isp_root = job["isp_root"]
    sys.path.insert(0, isp_root)
    os.chdir(isp_root)
    # OpenCV loads only after the thread env vars above are set, and brilliantISP is importable
    # only after isp_root is on sys.path (per job, in the worker).
    # pylint: disable=import-outside-toplevel
    import cv2
    from util.pipeline_runner import process_frame
    from util.pipeline_types import ProcessingOptions, ProcessingState
    # pylint: enable=import-outside-toplevel

    scene, cam, cls = job["scene"], job["cam"], CAMERAS[job["cam"]]
    cfg = yaml.safe_load(Path(job["config"]).read_text(encoding="utf-8"))
    out_dir = Path(job["out"]) / scene / cam
    (out_dir / "thumbs").mkdir(parents=True, exist_ok=True)
    mask = load_mask(cam)
    frames = sorted((SCENES / scene / cam / "raw").glob("frame_*.raw"))
    if job.get("max_frames"):
        frames = frames[: job["max_frames"]]
    state = ProcessingState()
    rows, t0, err = [], time.time(), None
    act = roi = None
    for f in frames:
        try:
            res = process_frame(str(f), cfg, state=state, options=ProcessingOptions())
        except Exception:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            err = traceback.format_exc()
            break
        rgb8 = np.asarray(res.preview_rgb8)
        if act is None:
            act = raster_to_output(mask, res.effective_config, cls, rgb8.shape[:2])
            rs = roi_slices(cam, mask.shape[0], mask.shape[1])
            roimask = np.zeros_like(mask)
            roimask[rs] = True
            roim = raster_to_output(roimask, res.effective_config, cls, rgb8.shape[:2])
            ys, xs = np.where(roim)
            roi = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
        s = frame_stats(rgb8, act, roi)
        s["frame"] = f.stem
        s["feedback"] = _jsonable(res.feedback)
        rows.append(s)
        th = cv2.resize(rgb8, (rgb8.shape[1] // 4, rgb8.shape[0] // 4),
                        interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_dir / "thumbs" / f"{f.stem}.jpg"), th[..., ::-1],
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    (out_dir / "frame_stats.json").write_text(json.dumps(rows))
    if err:
        (out_dir / "error.txt").write_text(err)
    return {"scene": scene, "cam": cam, "frames": len(rows), "sec": round(time.time() - t0, 1),
            "error": bool(err)}


def _git_info(isp_root: str) -> dict:
    def g(*a):
        return subprocess.run(["git", "-C", isp_root, *a], capture_output=True, text=True,
                              check=False).stdout.strip()
    diff = g("diff", "HEAD")
    return {"rev": g("rev-parse", "HEAD"), "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None}


def main() -> None:
    """CLI entry point: run the selected scenes and write ``runs/<name>/``."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--isp-root", required=True)
    ap.add_argument("--svs-config", required=True)
    ap.add_argument("--ad-config", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--scenes", nargs="*", default=["tuning"])
    ap.add_argument("--cams", nargs="*", default=list(CAMERAS))
    ap.add_argument("--cam-config", nargs="*", default=[], metavar="CAM=PATH",
                    help="per-position config override, e.g. SVS_FV=config/x_FV.yml")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="debug only; KPI runs must use all frames")
    a = ap.parse_args()

    isp_root = str(Path(a.isp_root).expanduser().resolve())
    scenes: list[str] = []
    for s in a.scenes:
        scenes += {"tuning": TUNING_SCENES, "heldout": HELDOUT_SCENES,
                   "all": list(SCENE_CLASS)}.get(s, [s])
    out = ROOT / "runs" / a.name
    if out.exists() and any(out.iterdir()):
        sys.exit(f"run dir {out} exists; choose a new --name")
    out.mkdir(parents=True)
    cfgs = {}
    for cls, p in (("SVS", a.svs_config), ("AD", a.ad_config)):
        pp = Path(p) if Path(p).is_absolute() else Path(isp_root) / p
        cfgs[cls] = str(pp)
        (out / f"config_{cls}.yml").write_text(pp.read_text(encoding="utf-8"), encoding="utf-8")
    cam_cfgs = {}
    for item in a.cam_config:
        cam, _, p = item.partition("=")
        if cam not in CAMERAS or not p:
            sys.exit(f"bad --cam-config {item!r}; expected CAM=PATH with CAM in "
                     f"{list(CAMERAS)}")
        pp = Path(p) if Path(p).is_absolute() else Path(isp_root) / p
        cam_cfgs[cam] = str(pp)
        (out / f"config_{cam}.yml").write_text(pp.read_text(encoding="utf-8"), encoding="utf-8")
    manifest = {"harness_version": HARNESS_VERSION, "isp_root": isp_root,
                "git": _git_info(isp_root), "configs": cfgs, "cam_configs": cam_cfgs,
                "scenes": scenes, "cams": a.cams, "max_frames": a.max_frames,
                "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    diff = subprocess.run(["git", "-C", isp_root, "diff", "HEAD"], capture_output=True, text=True,
                          check=False).stdout
    if diff:
        (out / "isp_uncommitted.diff").write_text(diff)
    jobs = [{"isp_root": isp_root, "scene": s, "cam": c,
             "config": cam_cfgs.get(c, cfgs[CAMERAS[c]]), "out": str(out),
             "max_frames": a.max_frames}
            for s in scenes for c in a.cams if (SCENES / s / ".done").exists()]
    # AD sequences are ~2x slower: schedule them first
    jobs.sort(key=lambda j: CAMERAS[j["cam"]] != "AD")
    t0 = time.time()
    with get_context("spawn").Pool(a.workers) as pool:
        results = list(pool.imap_unordered(run_sequence, jobs))
    manifest["results"] = sorted(results, key=lambda r: (r["scene"], r["cam"]))
    manifest["wall_sec"] = round(time.time() - t0, 1)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    compute_kpis(out)
    print((out / "kpi.md").read_text())


if __name__ == "__main__":
    main()
