"""Hable wrapper: Uncharted-2 display-linear curve via the shared engine.

Class name and normalized-input contract preserved; math lives in
``modules.tone_mapping.operators``.
"""

from __future__ import annotations

from modules.tone_mapping import operators
from modules.tone_mapping.compat import LegacyToneMapper

_hable_partial = operators.hable_partial


class HableToneMapping(LegacyToneMapper):
    """Legacy-constructor wrapper over the shared hable float path."""

    legacy_selector = "hable"
    float_mode = True
