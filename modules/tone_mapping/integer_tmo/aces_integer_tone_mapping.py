"""Integer ACES wrapper: LUT backend via the shared engine.

Fitted curve only; the legacy integer OETF/LUT chain is intentionally
replaced by one domain-bounded LUT. Class name preserved.
"""

from __future__ import annotations

from modules.tone_mapping.compat import LegacyToneMapper


class ACESIntegerToneMapping(LegacyToneMapper):
    """Legacy-constructor wrapper over the shared aces_fitted LUT backend."""

    legacy_selector = "aces_integer"
    float_mode = False
