"""Headless logic for the BrilliantISP calibration GUI (no Tk)."""

from tools.gui.computed import compute_format_stats
from tools.gui.format_map import (
    BAYER_PATTERNS,
    RGGB_START_TO_PATTERN,
    bayer_from_rggb_start,
    crop_from_margins,
    margins_from_crop,
    update_crop_from_margins,
    rggb_start_from_bayer,
)
from tools.gui.session import GuiSession, SessionPaths
from tools.gui.state import SharedConfig
from tools.gui.validation import ValidationIssue, validate_config
from tools.gui.yaml_compat import (
    REQUIRED_CONFIG_KEYS,
    dump_config_yaml,
    load_preset_yaml,
    yaml_safe_value,
)

__all__ = [
    "BAYER_PATTERNS",
    "RGGB_START_TO_PATTERN",
    "GuiSession",
    "REQUIRED_CONFIG_KEYS",
    "SessionPaths",
    "SharedConfig",
    "ValidationIssue",
    "bayer_from_rggb_start",
    "compute_format_stats",
    "crop_from_margins",
    "update_crop_from_margins",
    "dump_config_yaml",
    "load_preset_yaml",
    "margins_from_crop",
    "rggb_start_from_bayer",
    "validate_config",
    "yaml_safe_value",
]
