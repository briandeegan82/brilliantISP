"""Frozen KPI harness: shared geometry, raw reference decode, masks, ROIs, per-frame stats.

Do NOT edit during tuning. Changes require orchestrator + reviewer sign-off and a
new harness version (HARNESS_VERSION) so KPI numbers stay comparable across rounds.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

HARNESS_VERSION = "1.1"
ROOT = Path.home() / "isp_tuning"
SCENES = ROOT / "scenes"
FPS = 10.0  # recordings are decimated to 10 fps (sensor counter steps by 3 @ 30 fps)

# Scene list and lighting class. T* = tuning set, H* = held-out (never tuned on).
SCENE_CLASS = {
    "T1_urban_day": "day",
    "T2_canopy_static": "day",
    "T3_bridge_sky": "day",
    "T4_motorway_dusk": "day",
    "T5_twilight_rural": "low",
    "T6_night_lights": "night",
    "H1_urban_day": "day",
    "H2_night_lights": "night",
    "H3_night_dark": "night",
}
TUNING_SCENES = [s for s in SCENE_CLASS if s.startswith("T")]
HELDOUT_SCENES = [s for s in SCENE_CLASS if s.startswith("H")]
STATIC_SCENES: set[str] = set()  # v1.1: T2 is not static on AD_RR (pedestrians); use the normal scene-steady gate

CAMERAS = {
    "SVS_FV": "SVS",
    "SVS_RV": "SVS",
    "AD_FL": "AD",
    "AD_RR": "AD",
}

# Full-raster geometry (w, h, embedded rows top, embedded rows bottom). Verified from data:
# rows 0..4 are sensor embedded data; AD also has 20 embedded rows at the bottom.
GEOM = {"SVS": (1920, 1536, 5, 20 * 0), "AD": (3840, 2185, 5, 20)}

# Reference PWL decompanding (frozen copy of the shipped camera configs at harness v1.0).
# Used only for the ISP-independent scene reference (raw scene brightness / chroma).
PWL = {
    "SVS": dict(black=240,
                pin=[1, 256, 512, 831, 994, 1319, 1984, 2530, 2871, 2968, 3258, 3541, 3855],
                pout=[1, 256, 1024, 3900, 7800, 23400, 87315, 174630, 349260, 548241, 1736000, 5208000, 16777200]),
    "AD": dict(black=0,
               pin=[0, 100, 175, 308, 423, 624, 762, 1003, 1167, 1456, 1652, 1998, 2233, 2646, 2928, 3423, 3759, 4095],
               pout=[0, 100, 212, 450, 954, 2024, 4293, 9106, 19313, 40963, 86883, 184279, 390856, 829005,
                     1758320, 3729396, 7910050, 16777215]),
}
RAW_SAT_CODE = 4000  # companded code treated as (near) saturated for the reference stats

# v1.1: vehicle-fixed pixels excluded from the active mask.
# SVS ego body: exclude raster rows below y_cut(x) = a + b * ((x - 0.5) / 0.5) ** 2 (fractions of h, w).
EGO_CUT = {"SVS_FV": (0.80, -0.17), "SVS_RV": (0.78, -0.08)}
# Rectangular occluders (y0, y1, x0, x1) as raster fractions.
OCCLUDERS = {"AD_RR": [(0.10, 0.55, 0.0, 0.05)]}
# Cameras whose road ROI is not reliably road (K4/K9 reported n/a).
NO_ROAD_CAMS = {"AD_RR"}

# Road / near-neutral ROIs as fractions of the full raster (y0, y1, x0, x1).
ROAD_ROI = {
    "SVS_FV": (0.58, 0.76, 0.36, 0.64),
    "SVS_RV": (0.58, 0.76, 0.36, 0.64),
    "AD_FL": (0.80, 0.94, 0.35, 0.65),
    "AD_RR": (0.80, 0.94, 0.40, 0.75),
}

# ---------------------------------------------------------------------------
# sRGB / colour helpers
M_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                      [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]])
D65 = np.array([0.95047, 1.0, 1.08883])


def srgb8_to_linear(img8: np.ndarray) -> np.ndarray:
    lut = np.arange(256, dtype=np.float64) / 255.0
    lut = np.where(lut <= 0.04045, lut / 12.92, ((lut + 0.055) / 1.055) ** 2.4)
    return lut[img8]


def xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    t = xyz / D65
    d = 6 / 29
    f = np.where(t > d ** 3, np.cbrt(t), t / (3 * d * d) + 4 / 29)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def xyz_to_upvp(xyz: np.ndarray) -> tuple[float, float]:
    X, Y, Z = xyz
    den = X + 15 * Y + 3 * Z
    if den <= 0:
        return float("nan"), float("nan")
    return float(4 * X / den), float(9 * Y / den)


# ---------------------------------------------------------------------------
def read_raw(path: Path, cls: str) -> np.ndarray:
    w, h, _, _ = GEOM[cls]
    return np.fromfile(path, dtype=">u2").reshape(h, w)


def linearize(codes: np.ndarray, cls: str) -> np.ndarray:
    p = PWL[cls]
    c = np.clip(codes.astype(np.float64) - p["black"], 0, None)
    return np.interp(c, p["pin"], p["pout"])


def mask_path(cam: str) -> Path:
    return ROOT / "harness" / "masks" / f"{cam}.npy"


def load_mask(cam: str) -> np.ndarray:
    return np.load(mask_path(cam))


def roi_slices(cam: str, h: int, w: int) -> tuple[slice, slice]:
    y0, y1, x0, x1 = ROAD_ROI[cam]
    return slice(int(y0 * h), int(y1 * h)), slice(int(x0 * w), int(x1 * w))


def output_window(eff_cfg: dict, cls: str) -> tuple[int, int]:
    """Raster rows covered by the ISP output (handles embedded-row trimming)."""
    _, h, _, _ = GEOM[cls]
    si = eff_cfg.get("sensor_info", {})
    top = int(si.get("embedded_rows_top", 0) or 0)
    bot = int(si.get("embedded_rows_bottom", 0) or 0)
    return top, h - bot


def raster_to_output(arr: np.ndarray, eff_cfg: dict, cls: str, out_hw: tuple[int, int]) -> np.ndarray:
    """Map a full-raster boolean map onto the output image grid."""
    import cv2
    r0, r1 = output_window(eff_cfg, cls)
    sub = arr[r0:r1]
    if sub.shape != out_hw:
        sub = cv2.resize(sub.astype(np.uint8), (out_hw[1], out_hw[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    return sub


def scene_ref_stats(raw: np.ndarray, cls: str, mask: np.ndarray) -> dict:
    """ISP-independent scene reference: mean linear R, G, B over the active mask (half-res CFA planes).

    Raster CFA is GBRG (active area RGGB after the 5 embedded rows).
    """
    he = raw.shape[0] - (raw.shape[0] % 2)
    raw, mask = raw[:he], mask[:he]
    lin = linearize(raw, cls)
    sat = raw >= RAW_SAT_CODE
    gb, b, r, gr = lin[0::2, 0::2], lin[0::2, 1::2], lin[1::2, 0::2], lin[1::2, 1::2]
    m = mask[0::2, 0::2] & ~(sat[0::2, 0::2] | sat[0::2, 1::2] | sat[1::2, 0::2] | sat[1::2, 1::2])
    g = 0.5 * (gb + gr)
    R, G, B = float(r[m].mean()), float(g[m].mean()), float(b[m].mean())
    return {"R": R, "G": G, "B": B, "sat_frac": float(sat[mask].mean())}


def frame_stats(rgb8: np.ndarray, act: np.ndarray, roi: tuple[slice, slice]) -> dict:
    """Output-referred per-frame statistics on the active (image-circle / non-embedded) area."""
    lin = srgb8_to_linear(rgb8)
    px = lin[act]
    mean_rgb = px.mean(axis=0)
    xyz = M_RGB2XYZ @ mean_rgb
    up, vp = xyz_to_upvp(xyz)
    Y = px @ M_RGB2XYZ[1]
    clip = float((rgb8[act] >= 250).any(axis=1).mean())
    luma8 = rgb8[act].astype(np.float64) @ np.array([0.2126, 0.7152, 0.0722])
    crush = float((luma8 <= 4).mean())
    # road ROI: drop clipped + very bright (markings, lamps) + very dark pixels, then mean in linear RGB
    rr = lin[roi][act[roi]]
    r8 = rgb8[roi][act[roi]]
    ry = rr @ M_RGB2XYZ[1]
    ok = (r8.max(axis=1) < 250) & (ry > 1e-4)
    if ok.sum() > 100:
        med = np.median(ry[ok])
        ok &= ry < 2.0 * med
    if ok.sum() > 100:
        lab = xyz_to_lab(M_RGB2XYZ @ rr[ok].mean(axis=0))
        road = {"L": float(lab[0]), "a": float(lab[1]), "b": float(lab[2]),
                "C": float(np.hypot(lab[1], lab[2])), "n": int(ok.sum())}
    else:
        road = {"L": float("nan"), "a": float("nan"), "b": float("nan"), "C": float("nan"), "n": int(ok.sum())}
    Ls = 116 * np.cbrt(np.maximum(Y, (6 / 29) ** 3)) - 16
    return {"Ymean": float(Y.mean()), "Ymedian": float(np.median(Y)), "up": up, "vp": vp,
            "clip_frac": clip, "crush_frac": crush, "road": road,
            "mean_rgb_lin": [float(v) for v in mean_rgb],
            "luma_p1": float(np.percentile(luma8, 1)), "luma_clip_frac": float((luma8 >= 250).mean()),
            "Lstar_rms": float(Ls.std())}
