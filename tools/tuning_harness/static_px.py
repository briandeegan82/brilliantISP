"""Orchestrator tool: find vehicle-fixed (ego body / occluder) pixels.

They show low temporal variation on moving scenes.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402  # pylint: disable=wrong-import-position
    CAMERAS, SCENES, linearize, load_mask, read_raw)

MOVING = ("T1_urban_day", "T3_bridge_sky", "T4_motorway_dusk", "H1_urban_day")


def tstd(cam: str) -> np.ndarray:
    """Median over moving scenes of the per-pixel temporal std of log2 G (half-res)."""
    cls = CAMERAS[cam]
    stds = []
    for sc in MOVING:
        fs = sorted((SCENES / sc / cam / "raw").glob("frame_*.raw"))[::8][:25]
        stack = []
        for f in fs:
            raw = read_raw(f, cls)
            he = raw.shape[0] - raw.shape[0] % 2
            lin = linearize(raw[:he], cls)
            g = 0.5 * (lin[0::2, 0::2] + lin[1::2, 1::2])
            g = cv2.blur(g, (9, 9))
            stack.append(np.log2(g + 64.0))
        s = np.std(np.stack(stack), axis=0)
        stds.append(s)
    return np.median(np.stack(stds), axis=0)


def main() -> None:
    """CLI: write static_px/<cam>_tstd.npy and a tstd.jpg contact sheet."""
    out = Path(__file__).parent / "static_px"
    out.mkdir(exist_ok=True)
    tiles = []
    for cam in sys.argv[1:] or list(CAMERAS):
        s = tstd(cam)
        np.save(out / f"{cam}_tstd.npy", s)
        m = load_mask(cam)
        he = m.shape[0] - m.shape[0] % 2
        mh = m[:he][0::2, 0::2]
        print(cam, "tstd percentiles in mask", np.percentile(s[mh], [1, 5, 10, 25, 50]).round(3))
        v = np.clip(s / 0.6, 0, 1)
        img = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_JET)
        img[~mh] = 0
        cv2.putText(img, cam, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
        tiles.append(cv2.resize(img, (480, 300)))
    cv2.imwrite(str(out / "tstd.jpg"), np.hstack(tiles))


if __name__ == "__main__":
    main()
