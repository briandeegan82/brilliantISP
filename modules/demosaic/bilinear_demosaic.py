"""
File: bilinear_demosaic.py
Description: Implements the bilinear demosaic algorithm
Author: Brian Deegan (chatGPT)
------------------------------------------------------------
"""
import numpy as np
from scipy.ndimage import convolve

class BilinearDemosaic:
    """
    Correct bilinear demosaic using masks + proper kernels + normalization.
    Works for any Bayer layout as given by (mask_r, mask_g, mask_b).
    """

    def __init__(self, raw_in, masks):
        self.img = np.asarray(raw_in, dtype=np.float32)
        self.masks = [m.astype(np.float32) for m in masks]  # accept bool/0-1

    @staticmethod
    def _interp(channel_masked, mask, kernel, mode='mirror'):
        """Convolve channel & mask with kernel and normalize safely."""
        num = convolve(channel_masked, kernel, mode=mode)
        den = convolve(mask,           kernel, mode=mode)
        return np.where(den > 0, num / den, 0.0)

    def apply_bilinear(self):
        raw = self.img
        mask_r, mask_g, mask_b = self.masks

        # Known samples
        R0 = raw * mask_r
        G0 = raw * mask_g
        B0 = raw * mask_b

        # Kernels (rows x cols). Symmetric so conv "flip" doesn’t matter.
        cross = np.array([[0,1,0],
                          [1,0,1],
                          [0,1,0]], np.float32)              # N,S,E,W avg

        horiz = np.array([[0,0,0],
                          [1,0,1],
                          [0,0,0]], np.float32) * 0.5        # left/right

        vert  = np.array([[0,1,0],
                          [0,0,0],
                          [0,1,0]], np.float32) * 0.5        # up/down

        diag  = np.array([[1,0,1],
                          [0,0,0],
                          [1,0,1]], np.float32) * 0.25       # 4 diagonals

        # --- Green: fill at R/B using cross neighbors of *green* samples ---
        G_at_RB = self._interp(G0, mask_g, cross, mode='mirror')
        G = G0 + (1.0 - mask_g) * G_at_RB  # only fill where green is missing

        # Helper: “does this pixel have same-color neighbors horizontally/vertically?”
        # Use constant padding (no wrap) for these boolean maps.
        has_R_h = convolve(mask_r, np.array([[0,0,0],[1,0,1],[0,0,0]], np.float32),
                           mode='constant', cval=0.0) > 0
        has_R_v = convolve(mask_r, np.array([[0,1,0],[0,0,0],[0,1,0]], np.float32),
                           mode='constant', cval=0.0) > 0

        has_B_h = convolve(mask_b, np.array([[0,0,0],[1,0,1],[0,0,0]], np.float32),
                           mode='constant', cval=0.0) > 0
        has_B_v = convolve(mask_b, np.array([[0,1,0],[0,0,0],[0,1,0]], np.float32),
                           mode='constant', cval=0.0) > 0

        # Precompute interpolants from same-color samples (proper normalization)
        R_h = self._interp(R0, mask_r, horiz)   # red from left/right
        R_v = self._interp(R0, mask_r, vert)    # red from up/down
        R_d = self._interp(R0, mask_r, diag)    # red from diagonals

        B_h = self._interp(B0, mask_b, horiz)
        B_v = self._interp(B0, mask_b, vert)
        B_d = self._interp(B0, mask_b, diag)

        # --- Red: keep known; fill missing according to classic bilinear rules ---
        R = R0.copy()
        # at green sites: if green is in a "red row" use horizontal, else vertical
        at_G = mask_g.astype(bool)
        R[at_G & has_R_h] = R_h[at_G & has_R_h]     # G pixels flanked by R horizontally
        R[at_G & ~has_R_h] = R_v[at_G & ~has_R_h]   # else use vertical (R above/below)

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
