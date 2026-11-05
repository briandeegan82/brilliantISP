"""
File: bilinear_demosaic.py
Description: Implements the bilinear demosaic algorithm
Author: Brian Deegan (chatGPT)
------------------------------------------------------------
"""

import numpy as np
from scipy.ndimage import convolve

from util.isp_types import BayerImage, DemosaicMasks


class BilinearDemosaic:
    """
    Correct bilinear demosaic using masks + proper kernels + normalization.
    Works for any Bayer layout as given by (mask_r, mask_g, mask_b).
    """

    def __init__(self, raw_in: BayerImage, masks: DemosaicMasks) -> None:
        self.img = np.asarray(raw_in, dtype=np.float32)
        self.masks = [m.astype(np.float32) for m in masks]  # accept bool/0-1

    def _periodic_mask_tiles(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Return the 2x2 periodic tile for each mask when every mask is a
        regular binary-periodic pattern matching the raw shape; else None."""
        tiles: list[np.ndarray] = []
        for mask in self.masks:
            if self.img.shape != mask.shape or mask.shape[0] < 2 or mask.shape[1] < 2:
                return None
            tile = mask[:2, :2]
            if not np.all((tile == 0) | (tile == 1)):
                return None
            if not (
                bool(np.all(mask[0::2, 0::2] == tile[0, 0]))
                and bool(np.all(mask[0::2, 1::2] == tile[0, 1]))
                and bool(np.all(mask[1::2, 0::2] == tile[1, 0]))
                and bool(np.all(mask[1::2, 1::2] == tile[1, 1]))
            ):
                return None
            tiles.append(tile)
        return (tiles[0], tiles[1], tiles[2])

    @staticmethod
    def _interp(
        channel_masked: np.ndarray,
        mask: np.ndarray,
        kernel: np.ndarray,
        mode: str = "mirror",
        *,
        mask_tile: np.ndarray | None = None,
    ) -> np.ndarray:
        """Convolve channel & mask with kernel and normalize safely."""
        num = convolve(channel_masked, kernel, mode=mode)
        if mask_tile is not None and mode == "mirror":
            den = convolve(mask_tile, kernel, mode="mirror")
            out = np.zeros(num.shape, dtype=np.float32)
            for dy in range(2):
                for dx in range(2):
                    if den[dy, dx] > 0:
                        np.divide(num[dy::2, dx::2], den[dy, dx], out=out[dy::2, dx::2])
            return out
        den = convolve(mask, kernel, mode=mode)
        out = np.zeros_like(num)
        np.divide(num, den, out=out, where=den > 0)
        return out

    def apply_bilinear(self) -> np.ndarray:
        raw = self.img
        mask_r, mask_g, mask_b = self.masks

        # Known samples
        R0 = raw * mask_r
        G0 = raw * mask_g
        B0 = raw * mask_b

        # Kernels (rows x cols). Symmetric so conv "flip" doesn’t matter.
        cross = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.float32)  # N,S,E,W avg

        horiz = np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]], np.float32) * 0.5  # left/right

        vert = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]], np.float32) * 0.5  # up/down

        diag = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], np.float32) * 0.25  # 4 diagonals

        # --- Green: fill at R/B using cross neighbors of *green* samples ---
        tiles = self._periodic_mask_tiles()
        tr = tiles[0] if tiles else None
        tg = tiles[1] if tiles else None
        tb = tiles[2] if tiles else None

        G_at_RB = self._interp(G0, mask_g, cross, mode="mirror", mask_tile=tg)
        G = G0 + (1.0 - mask_g) * G_at_RB  # only fill where green is missing

        # Helper: “does this pixel have same-color neighbors horizontally/vertically?”
        # Use constant padding (no wrap) for these boolean maps.
        if tiles is not None:
            has_R_h = np.zeros(mask_r.shape, dtype=bool)
            has_R_h[:, 1:] |= mask_r[:, :-1] > 0
            has_R_h[:, :-1] |= mask_r[:, 1:] > 0
            has_B_h = np.zeros(mask_b.shape, dtype=bool)
            has_B_h[:, 1:] |= mask_b[:, :-1] > 0
            has_B_h[:, :-1] |= mask_b[:, 1:] > 0
        else:
            has_R_h = convolve(mask_r, np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]], np.float32), mode="constant", cval=0.0) > 0
            has_B_h = convolve(mask_b, np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]], np.float32), mode="constant", cval=0.0) > 0

        # Precompute interpolants from same-color samples (proper normalization)
        R_h = self._interp(R0, mask_r, horiz, mask_tile=tr)  # red from left/right
        R_v = self._interp(R0, mask_r, vert, mask_tile=tr)  # red from up/down
        R_d = self._interp(R0, mask_r, diag, mask_tile=tr)  # red from diagonals

        B_h = self._interp(B0, mask_b, horiz, mask_tile=tb)
        B_v = self._interp(B0, mask_b, vert, mask_tile=tb)
        B_d = self._interp(B0, mask_b, diag, mask_tile=tb)

        # --- Red: keep known; fill missing according to classic bilinear rules ---
        R = R0.copy()
        # at green sites: if green is in a "red row" use horizontal, else vertical
        at_G = mask_g.astype(bool)
        R[at_G & has_R_h] = R_h[at_G & has_R_h]  # G pixels flanked by R horizontally
        R[at_G & ~has_R_h] = R_v[at_G & ~has_R_h]  # else use vertical (R above/below)

        # at blue sites: use diagonals
        at_B = mask_b.astype(bool)
        R[at_B] = R_d[at_B]

        # --- Blue: symmetric to red ---
        B = B0.copy()
        at_G = mask_g.astype(bool)
        B[at_G & has_B_h] = B_h[at_G & has_B_h]
        B[at_G & ~has_B_h] = B_v[at_G & ~has_B_h]

        at_R = mask_r.astype(bool)
        B[at_R] = B_d[at_R]

        # Stack
        out = np.stack([R, G, B], axis=-1)
        return out.astype(np.float32)
