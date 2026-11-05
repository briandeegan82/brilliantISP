"""Integer Hable wrapper: LUT backend via the shared engine.

Class name preserved; declared code domain and engine-shared math replace
the duplicated legacy LUT implementation.
"""

from __future__ import annotations

from modules.tone_mapping.compat import LegacyToneMapper


class HableIntegerToneMapping(LegacyToneMapper):
    """Legacy-constructor wrapper over the shared hable LUT backend."""

    legacy_selector = "hable_integer"
    float_mode = False
