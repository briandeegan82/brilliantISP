"""Prepare a validated Process run against a GUI session."""

from __future__ import annotations

from typing import Any

from tools.gui.session import GuiSession
from tools.gui.validation import ValidationIssue, can_process


def prepare_process(
    session: GuiSession,
    config: dict[str, Any],
) -> tuple[bool, list[ValidationIssue]]:
    """Validate and write ``config.yml``. Does not invoke BrilliantISP."""
    ok, issues = can_process(config, has_raw=session.has_raw)
    if not ok:
        return False, issues
    session.write_config(config)
    return True, []
