"""ACES wrapper: fitted scalar approximation via the shared engine.

``aces_fitted`` is the Narkowicz-style rational curve, not a complete ACES
color transform; the legacy internal OETF/gamma stage is intentionally
removed. Class name preserved for import compatibility.
"""

from __future__ import annotations

from modules.tone_mapping.compat import LegacyToneMapper


class ACESToneMapping(LegacyToneMapper):
    """Legacy-constructor wrapper over the shared aces_fitted float path."""

    legacy_selector = "aces"
    float_mode = True
