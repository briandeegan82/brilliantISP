"""Repair pathological Bayer CFA defects before demosaic."""

from __future__ import annotations

import numpy as np

from util.isp_types import BayerPattern


def green_zero_fraction(raw: np.ndarray, bayer_pattern: BayerPattern) -> float:
    """Fraction of CFA green sites that are <= 0."""
    pat = str(bayer_pattern).strip().lower()
    img = raw
    if pat == "rggb":
        g = np.concatenate([img[0::2, 1::2].ravel(), img[1::2, 0::2].ravel()])
    elif pat == "bggr":
        g = np.concatenate([img[0::2, 1::2].ravel(), img[1::2, 0::2].ravel()])
    elif pat == "grbg":
        g = np.concatenate([img[0::2, 0::2].ravel(), img[1::2, 1::2].ravel()])
    elif pat == "gbrg":
        g = np.concatenate([img[0::2, 0::2].ravel(), img[1::2, 1::2].ravel()])
    else:
        raise ValueError(f"Unsupported bayer_pattern: {bayer_pattern}")
    return float((g <= 0).mean())


def repair_zero_green_cfa(
    raw: np.ndarray,
    bayer_pattern: BayerPattern,
    *,
    zero_thresh: float = 0.0,
) -> tuple[np.ndarray, float]:
    """
    Replace zero/empty green CFA sites by averaging neighboring R and B.

    Some Lucid TRI054S dumps have ~97% of green sites literally zero while R/B
    are valid (magenta cast after demosaic). True spectral green cannot be
    recovered; this restores a usable luminance proxy so WB/TMO can run.

    Returns (repaired_raw, fraction_of_green_sites_filled).
    """
    pat = str(bayer_pattern).strip().lower()
    out = raw.astype(np.float64, copy=True)

    if pat == "rggb":
        filled = _fill_rggb_greens(out, zero_thresh)
    elif pat == "bggr":
        # BGGR greens sit on the same lattice as RGGB greens.
        filled = _fill_rggb_greens(out, zero_thresh)
    elif pat == "grbg":
        filled = _fill_grbg_greens(out, zero_thresh)
    elif pat == "gbrg":
        filled = _fill_grbg_greens(out, zero_thresh)
    else:
        raise ValueError(f"Unsupported bayer_pattern: {bayer_pattern}")

    return np.clip(out, 0, None).astype(raw.dtype, copy=False), filled


def _fill_rggb_greens(out: np.ndarray, zero_thresh: float) -> float:
    """Fill Gr (0,1) and Gb (1,0) from neighboring R/B on an RGGB/BGGR mosaic."""
    R = out[0::2, 0::2]
    Gr = out[0::2, 1::2]
    Gb = out[1::2, 0::2]
    B = out[1::2, 1::2]

    R_r = np.empty_like(R)
    R_r[:, :-1] = R[:, 1:]
    R_r[:, -1] = R[:, -1]
    B_u = np.empty_like(B)
    B_u[1:] = B[:-1]
    B_u[0] = B[0]
    Gr_est = 0.25 * (R + R_r + B_u + B)

    R_d = np.empty_like(R)
    R_d[:-1] = R[1:]
    R_d[-1] = R[-1]
    B_l = np.empty_like(B)
    B_l[:, 1:] = B[:, :-1]
    B_l[:, 0] = B[:, 0]
    Gb_est = 0.25 * (R + R_d + B_l + B)

    m_gr = Gr <= zero_thresh
    m_gb = Gb <= zero_thresh
    Gr[m_gr] = Gr_est[m_gr]
    Gb[m_gb] = Gb_est[m_gb]
    n = m_gr.size + m_gb.size
    return float((m_gr.sum() + m_gb.sum()) / max(n, 1))


def _fill_grbg_greens(out: np.ndarray, zero_thresh: float) -> float:
    """Fill G at (0,0) and (1,1) from neighboring R/B on a GRBG/GBRG mosaic."""
    G1 = out[0::2, 0::2]
    R = out[0::2, 1::2]
    B = out[1::2, 0::2]
    G2 = out[1::2, 1::2]

    R_l = np.empty_like(R)
    R_l[:, 1:] = R[:, :-1]
    R_l[:, 0] = R[:, 0]
    B_u = np.empty_like(B)
    B_u[1:] = B[:-1]
    B_u[0] = B[0]
    G1_est = 0.25 * (R_l + R + B_u + B)

    R_d = np.empty_like(R)
    R_d[:-1] = R[1:]
    R_d[-1] = R[-1]
    B_r = np.empty_like(B)
    B_r[:, :-1] = B[:, 1:]
    B_r[:, -1] = B[:, -1]
    G2_est = 0.25 * (R + R_d + B + B_r)

    m1 = G1 <= zero_thresh
    m2 = G2 <= zero_thresh
    G1[m1] = G1_est[m1]
    G2[m2] = G2_est[m2]
    n = m1.size + m2.size
    return float((m1.sum() + m2.sum()) / max(n, 1))
