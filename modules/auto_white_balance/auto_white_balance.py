from util.debug_utils import get_debug_logger

"""
File: auto_white_balance.py
Description: 3A - AWB Runs the AWB algorithm based on selection from config file
Code / Paper  Reference: https://www.sciencedirect.com/science/article/abs/pii/0016003280900587
                         https://library.imaging.org/admin/apis/public/api/ist/website/downloadArticle/cic/12/1/art00008
                         https://opg.optica.org/josaa/viewmedia.cfm?uri=josaa-31-5-1049&seq=0
Author: Brian Deegan (based in part on 10xEngineers / Infinite-ISP)
------------------------------------------------------------
"""
import time
import math
import numpy as np
from modules.auto_white_balance.gray_world import GrayWorld as GW
from modules.auto_white_balance.norm_gray_world import NormGrayWorld as NGW
from modules.auto_white_balance.pca import PCAIlluminEstimation as PCA
from util.metering_roi import roi_config, weight_map


class AutoWhiteBalance:
    """
    Auto White Balance Module
    """

    def __init__(self, raw, sensor_info, parm_awb, parm_wbc):

        self.raw = raw

        self.sensor_info = sensor_info
        self.parm_awb = parm_awb
        self.enable = parm_awb["is_enable"]
        self.hdr_bit_depth = sensor_info["hdr_bit_depth"]
        self.is_debug = parm_awb["is_debug"]
        self.underexposed_percentage = parm_awb["underexposed_percentage"]
        self.overexposed_percentage = parm_awb["overexposed_percentage"]
        self.flatten_img = None
        self.bayer = str(self.sensor_info["bayer_pattern"]).strip().lower()
        # self.img = img
        self.algorithm = parm_awb["algorithm"]
        self.parm_wbc = parm_wbc

        # Initialize debug logger with config from parm_awb
        debug_config = {"debug_enabled": parm_awb.get("is_debug", False), "debug_log_level": "INFO", "debug_log_file": None}
        self.logger = get_debug_logger("AutoWhiteBalance", config=debug_config)

    def determine_white_balance_gain(self):
        """
        Determine white balance gains calculated using AWB Algorithms to Raw Image
        """

        # Clamp to a sane range. Do NOT force gains >= 1: that prevents cutting
        # a strong blue/red channel (common under lab LED / fluorescent light)
        # and leaves a permanent color cast even with the correct Bayer phase.
        return self.compute_gains()

    def apply_gray_world(self):
        """

        Gray World White Balance:
        Gray world algorithm calculates white balance (G/R and G/B)
        by average values of RGB channels
        """

        gwa = GW(self.flatten_img)
        return gwa.calculate_gains()

    def apply_norm_gray_world(self):
        """
        Norm 2 Gray World White Balance:

        Gray world algorithm calculates white balance (G/R and G/B)
        by average values of RGB channels. Average values for each channel
        are calculated by norm-2
        """

        ngw = NGW(self.flatten_img)
        return ngw.calculate_gains()

    def apply_pca_illuminant_estimation(self):
        """
        PCA Illuminant Estimation:

        This algorithm gets illuminant estimation directly from the color distribution
        The method that chooses bright and dark pixels using a projection distance in
        the color distribution and then applies principal component analysis to estimate
        the illumination direction
        """
        pixel_percentage = self.parm_awb["percentage"]
        pca = PCA(self.flatten_img, pixel_percentage)
        return pca.calculate_gains()

    def execute(self):
        """
        Execute Auto White Balance Module
        """
        self.logger.info(f"Auto White balancing = {self.enable}")

        if self.enable:
            start = time.time()
            rgain, bgain = self.determine_white_balance_gain()
            self.logger.info(f"  Execution time: {time.time() - start:.3f}s")
            return np.array([rgain, bgain], dtype=np.float64)
        rgain, bgain = self.parm_wbc["r_gain"], self.parm_wbc["b_gain"]
        self.logger.info(f"  AWB off — using manual white_balance r_gain/b_gain: {rgain}, {bgain}")
        return np.array([float(rgain), float(bgain)], dtype=np.float64)


    def _awb_planes(self) -> np.ndarray | None:
        """Shared CFA->RGB float64 channel extraction with declared thresholds.

        Returns (rows, underexposed_limit, overexposed_limit) prepared as an
        Nx3 float64 sample matrix after the over/under rejection; None when
        the frame cannot produce samples. Channels are widened to float64
        BEFORE the green pair average so 32-bit unsigned sums cannot wrap.
        """
        raw = np.asarray(self.raw, dtype=np.float64)
        if raw.ndim != 2 or raw.size == 0:
            return None
        if self.bayer == "rggb":
            ch = {"r": raw[0::2, 0::2], "gr": raw[0::2, 1::2],
                  "gb": raw[1::2, 0::2], "b": raw[1::2, 1::2]}
        elif self.bayer == "bggr":
            ch = {"b": raw[0::2, 0::2], "gb": raw[0::2, 1::2],
                  "gr": raw[1::2, 0::2], "r": raw[1::2, 1::2]}
        elif self.bayer == "grbg":
            ch = {"gr": raw[0::2, 0::2], "r": raw[0::2, 1::2],
                  "b": raw[1::2, 0::2], "gb": raw[1::2, 1::2]}
        elif self.bayer == "gbrg":
            ch = {"gb": raw[0::2, 0::2], "b": raw[0::2, 1::2],
                  "r": raw[1::2, 0::2], "gr": raw[1::2, 1::2]}
        else:
            raise ValueError(f"unknown CFA pattern {self.bayer!r}")
        # Odd width or height: rows/cols from 0::2 vs 1::2 differ by one; crop to shared CFA size.
        mh = min(c.shape[0] for c in ch.values())
        mw = min(c.shape[1] for c in ch.values())
        if mh <= 0 or mw <= 0:
            return None
        r = ch["r"][:mh, :mw]
        g = (ch["gr"][:mh, :mw] + ch["gb"][:mh, :mw]) / 2.0
        b = ch["b"][:mh, :mw]
        bayer_channels = np.dstack((r, g, b))
        # print(bayer_channels.shape)
        return bayer_channels

    def _awb_controls(self) -> str:
        """Validate the active algorithm before any sample work; empty frames
        still surface unknown-algorithm errors."""
        algo = "grey_world" if self.algorithm == "gray_world" else self.algorithm
        if self.bayer not in ("rggb", "bggr", "grbg", "gbrg"):
            raise ValueError(f"unknown CFA pattern {self.bayer!r}")
        for name, value in (("underexposed_percentage", self.underexposed_percentage),
                            ("overexposed_percentage", self.overexposed_percentage)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100:
                raise ValueError(f"{name} must be finite in 0..100")
        if algo == "pca":
            try:
                pct = float(self.parm_awb["percentage"])
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"pca percentage must be in (0,100], got {self.parm_awb.get('percentage')!r}"
                ) from exc
            if not math.isfinite(pct) or pct <= 0 or pct > 100:
                raise ValueError(f"pca percentage must be in (0,100], got {pct!r}")
        elif algo not in ("grey_world", "norm_2"):
            raise ValueError(f"unknown AWB algorithm {self.algorithm!r}")
        return algo

    def _awb_rows(self, *, input_gain: float = 1.0) -> np.ndarray:
        """Rejection-filtered finite Nx3 sample rows; empty allowed."""
        rgb = self._awb_planes()
        if rgb is None:
            self.flatten_img = np.empty((0, 3))
            return self.flatten_img
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            rgb = rgb / input_gain
        max_pixel_value = 2.0 ** int(self.sensor_info["hdr_bit_depth"])
        approx_percentage = max_pixel_value / 100
        # Removed overexposed and underexposed pixels for wb gain calculation
        overexposed_limit = max_pixel_value - self.overexposed_percentage * approx_percentage
        underexposed_limit = self.underexposed_percentage * approx_percentage
        bad = np.sum((rgb < underexposed_limit) | (rgb > overexposed_limit), axis=2)
        keep = bad == 0
        weights = weight_map(rgb.shape[:2], roi_config(self.parm_awb))
        if weights is not None:
            keep &= weights > 0
        rows = rgb[keep]
        rows = rows[np.all(np.isfinite(rows), axis=1)]
        self.flatten_img = rows
        # print(self.flatten_raw.shape)
        return rows

    def _estimate_means(self, rows: np.ndarray, algo: str) -> np.ndarray:
        """Channel means for the requested algorithm; raises on ill-conditioned."""
        means = np.mean(rows, axis=0)
        if np.any(means <= 0) or not np.all(np.isfinite(means)):
            raise ValueError("AWB channel means are degenerate")
        if algo == "norm_2":
            return np.sqrt(np.mean(rows * rows, axis=0))
        if algo == "pca":
            percentage = float(self.parm_awb["percentage"])
            mean_rgb = means
            mean_vector = mean_rgb / np.linalg.norm(mean_rgb)
            data_p = rows @ mean_vector
            order = np.argsort(data_p)
            index = max(1, int(np.ceil(rows.shape[0] * percentage / 100)))
            filtered = np.concatenate((order[0:index], order[-index:None]))
            selected = rows[filtered, :].astype(np.float64)
            sigma = selected.T @ selected
            eig_value, eig_vector = np.linalg.eigh(sigma)
            return np.abs(eig_vector[:, int(np.argmax(eig_value))])
        return means

    @staticmethod
    def _gains_from_means(means: np.ndarray) -> tuple[float, float]:
        if not np.all(np.isfinite(means)) or np.any(means <= 0):
            raise ValueError("AWB channel means are degenerate")
        return float(means[1] / means[0]), float(means[1] / means[2])

    def _awb_estimate(self) -> tuple[float, float]:
        """Common guarded estimation shared by execute() and compute_gains()."""
        algo = self._awb_controls()
        rows = self._awb_rows()
        if rows.shape[0] == 0:
            raise ValueError("AWB produced no valid samples")
        means = self._estimate_means(rows, algo)
        return self._gains_from_means(means)

    def compute_gains(
        self, *, fallback_gains: tuple[float, float] | None = None, input_gain: float = 1.0
    ) -> tuple[float, float]:
        """Safe 3A variant: status-tracked gain estimation with honest fallbacks.

        Sets ``last_status`` ('disabled'|'estimated'|'fallback_no_samples'|
        'fallback_ill_conditioned') and ``sample_count``. Disabled returns
        the manual white-balance pair (ignores fallback and the inactive
        algorithm). Only estimated gains are clipped to .25..4; the
        fallback pair (provided finite nonnegative, then manual, else
        neutral) is returned exactly. ``input_gain`` removes virtual digital
        gain from measurement units before sample rejection and estimation.
        """
        self.last_status = "disabled"
        self.sample_count = 0
        manual = (self.parm_wbc.get("r_gain"), self.parm_wbc.get("b_gain"))
        if not self.enable:
            return self._awb_fallback(None, manual, "disabled")
        algo = self._awb_controls()
        if isinstance(input_gain, (bool, np.bool_)) or not isinstance(input_gain, (int, float, np.integer, np.floating)) \
                or not 0 < input_gain <= float(np.finfo(np.float64).max):
            raise ValueError("AWB input_gain must be finite and positive")
        rows = self._awb_rows(input_gain=float(input_gain))
        self.sample_count = int(rows.shape[0])
        if self.sample_count == 0:
            return self._awb_fallback(fallback_gains, manual, "fallback_no_samples")
        try:
            means = self._estimate_means(rows, algo)
            rgain_raw, bgain_raw = self._gains_from_means(means)
        except ValueError:
            return self._awb_fallback(fallback_gains, manual, "fallback_ill_conditioned")
        rgain = float(np.clip(rgain_raw, 0.25, 4.0))
        bgain = float(np.clip(bgain_raw, 0.25, 4.0))
        self.last_status = "estimated"
        return rgain, bgain

    def _awb_fallback(self, provided, manual, status):
        def _valid(pair):
            try:
                if not isinstance(pair, (tuple, list, np.ndarray)) or len(pair) != 2:
                    return False
                if any(isinstance(value, (bool, str)) for value in pair):
                    return False
                a, b = float(pair[0]), float(pair[1])
            except (TypeError, ValueError, IndexError, KeyError):
                return False
            return math.isfinite(a) and math.isfinite(b) and a >= 0 and b >= 0
        self.last_status = status
        if _valid(provided):
            return float(provided[0]), float(provided[1])
        if _valid(manual):
            return float(manual[0]), float(manual[1])
        return 1.0, 1.0
