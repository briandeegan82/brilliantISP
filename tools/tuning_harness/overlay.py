"""Debug overlay: active mask (red = excluded) and road ROI (green) on raw G-plane previews."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import CAMERAS, SCENES, load_mask, read_raw, roi_slices  # noqa: E402

scenes = sys.argv[1:] or ["T1_urban_day", "T4_motorway_dusk", "T6_night_lights"]
rows = []
for sc in scenes:
    tiles = []
    for cam, cls in CAMERAS.items():
        fs = sorted((SCENES / sc / cam / "raw").glob("frame_*.raw"))
        raw = read_raw(fs[len(fs) // 2], cls).astype(np.float32)
        he = raw.shape[0] - raw.shape[0] % 2
        a = raw[:he]
        g = 0.5 * (a[0::2, 0::2] + a[1::2, 1::2])
        lo = np.percentile(g, 1)
        g = np.clip((g - lo) / max(np.percentile(g, 99.5) - lo, 1), 0, 1) ** 0.6
        img = cv2.cvtColor((g * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        m = load_mask(cam)[:he][0::2, 0::2]
        img[~m] = (img[~m] * 0.3 + np.array([0, 0, 120])).astype(np.uint8)
        ys, xs = roi_slices(cam, raw.shape[0], raw.shape[1])
        cv2.rectangle(img, (xs.start // 2, ys.start // 2), (xs.stop // 2, ys.stop // 2), (0, 255, 0), 3)
        cv2.putText(img, f"{sc} {cam}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
        tiles.append(cv2.resize(img, (480, 300)))
    rows.append(np.hstack(tiles))
cv2.imwrite(str(Path(__file__).parent / "mask_roi_check.jpg"), np.vstack(rows))
