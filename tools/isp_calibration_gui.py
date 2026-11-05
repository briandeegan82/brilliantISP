#!/usr/bin/env python3
"""ISP calibration GUI launcher (stdlib-only bootstrap)."""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_LEGACY = "tools.gui.legacy_calibration"


def __getattr__(name: str):
    if name.startswith("__"):
        raise AttributeError(name)
    module = importlib.import_module(_LEGACY)
    return getattr(module, name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ISP calibration GUI")
    parser.add_argument("--config", default=None)
    parser.add_argument("--raw", default=None)
    parser.add_argument("--no-recover", action="store_true")
    parser.add_argument("--workspace", action="store_true",
                        help="Integrated workspace (opt-in)")
    args = parser.parse_args(argv)

    if args.workspace:
        from tools.gui.app import WorkspaceApp
    else:
        from tools.gui.legacy_calibration import CalibrationApp as WorkspaceApp
    import tkinter as tk
    root = tk.Tk()
    app = WorkspaceApp(
        root,
        preset=args.config,
        recover=not args.no_recover,
        initial_raw=args.raw)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
