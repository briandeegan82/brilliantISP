"""Integer Reinhard wrapper: fixed-point Q32 path via the shared engine.

Class name and IntegerToneMapping alias preserved; the fixed path uses
declared code domain, never the observed image maximum.
"""

from __future__ import annotations

from modules.tone_mapping.compat import LegacyToneMapper


class IntegerReinhardToneMapping(LegacyToneMapper):
    """Legacy-constructor wrapper over the shared fixed Reinhard backend."""

    legacy_selector = "reinhard_integer"
    float_mode = False


IntegerToneMapping = IntegerReinhardToneMapping
