"""Durand wrapper: local Gaussian/range operator via the shared engine.

CPU-only verified path; legacy ``downsample_factor`` was unused and is
migrated to 1. Class name preserved for imports.
"""

from __future__ import annotations

import numpy as np

from modules.tone_mapping.compat import LegacyToneMapper
from modules.tone_mapping.normalization import normalize_luminance


class HDRDurandToneMapping(LegacyToneMapper):
    """Legacy-constructor wrapper over the shared local Durand operator."""

    legacy_selector = "durand"
    float_mode = True

    def normalize(self, image: np.ndarray) -> np.ndarray:
        return normalize_luminance(image, normalization="per_frame", input_max=1.0, stats=None)
