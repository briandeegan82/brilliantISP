"""Build frozen active-area masks and ISP-independent per-frame scene references.

Run once by the orchestrator:  python scene_ref.py [--scenes ...]
Outputs: harness/masks/<cam>.npy and scenes/<scene>/<cam>/scene_ref.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402  # pylint: disable=wrong-import-position
    CAMERAS, EGO_CUT, GEOM, OCCLUDERS, PWL, SCENE_CLASS, SCENES, mask_path, read_raw,
    scene_ref_stats)

MASK_SCENES = ("T1_urban_day", "T2_canopy_static", "T3_bridge_sky", "T4_motorway_dusk")


def frames(scene: str, cam: str) -> list[Path]:
    """Sorted raw frame paths of one scene/camera sequence."""
    return sorted((SCENES / scene / cam / "raw").glob("frame_*.raw"))


def build_mask(cam: str) -> np.ndarray:
    """Active-area mask: lit lens image circle minus embedded rows, ego body and occluders."""
    cls = CAMERAS[cam]
    w, h, top, bot = GEOM[cls]
    # per-pixel max over day frames: anything ever lit is inside the lens image circle
    acc = np.zeros((h, w), np.float64)
    for sc in MASK_SCENES:
        for f in frames(sc, cam)[::10][:20]:
            acc = np.maximum(acc, read_raw(f, cls))
    black = PWL[cls]["black"]
    act = acc > black + 0.03 * (np.percentile(acc, 99) - black)
    act[:top] = False
    if bot:
        act[h - bot:] = False
    k = np.ones((15, 15), np.uint8)
    act = cv2.morphologyEx(act.astype(np.uint8), cv2.MORPH_OPEN, k)
    _, lab, st, _ = cv2.connectedComponentsWithStats(act, 8)
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    pts = cv2.findNonZero((lab == big).astype(np.uint8))
    hull = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(pts), 1)
    act = hull.astype(bool)
    if cam in EGO_CUT:
        a, b = EGO_CUT[cam]
        xs = (np.arange(w) / w - 0.5) / 0.5
        ycut = ((a + b * xs ** 2) * h).astype(int)
        act &= np.arange(h)[:, None] < ycut[None, :]
    for y0, y1, x0, x1 in OCCLUDERS.get(cam, []):
        act[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = False
    act[:top] = False
    if bot:
        act[h - bot:] = False
    # make the mask CFA-quad aligned (rows/cols pairs) so half-res planes index cleanly
    he = h - (h % 2)
    q = act[0:he:2, 0::2] & act[1:he:2, 1::2]
    out = np.zeros((h, w), bool)
    out[:he] = np.repeat(np.repeat(q, 2, 0), 2, 1)
    return out


def main() -> None:
    """CLI: build missing masks and per-frame scene references."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=list(SCENE_CLASS))
    ap.add_argument("--rebuild-masks", action="store_true")
    a = ap.parse_args()
    mask_path("x").parent.mkdir(parents=True, exist_ok=True)
    for cam in CAMERAS:
        if a.rebuild_masks or not mask_path(cam).exists():
            m = build_mask(cam)
            np.save(mask_path(cam), m)
            print(cam, "mask active fraction", round(float(m.mean()), 3))
    for scene in a.scenes:
        if not (SCENES / scene / ".done").exists():
            print("skip (not extracted)", scene)
            continue
        for cam, cls in CAMERAS.items():
            out = SCENES / scene / cam / "scene_ref.json"
            if out.exists():
                continue
            m = np.load(mask_path(cam))
            rows = []
            for f in frames(scene, cam):
                s = scene_ref_stats(read_raw(f, cls), cls, m)
                s["frame"] = f.stem
                rows.append(s)
            out.write_text(json.dumps(rows))
            print(scene, cam, len(rows), "frames")


if __name__ == "__main__":
    main()
