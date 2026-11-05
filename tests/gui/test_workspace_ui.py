from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tools.gui.views.parameters import parse_numeric


REPO = Path(__file__).resolve().parents[2]
LAUNCHER = REPO / "tools" / "isp_calibration_gui.py"


def test_parse_numeric_types() -> None:
    assert parse_numeric("12", 0) == 12 and isinstance(parse_numeric("12", 0), int)
    assert parse_numeric("0x10", 0) == 16
    assert parse_numeric("1.5", 0.0) == 1.5
    with pytest.raises(ValueError):
        parse_numeric("abc", 0)
    with pytest.raises(ValueError):
        parse_numeric("1", True)


def test_launcher_help_no_heavy_imports() -> None:
    code = (
        "import sys; sys.path.insert(0, 'tools');"
        "import isp_calibration_gui as m;"
        "heavy = [x for x in ('tkinter','numpy','matplotlib','cv2','brilliant_isp')"
        " if x in sys.modules]; print(','.join(heavy))")
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, timeout=30,
                         cwd=str(REPO))
    assert out.stdout.strip() == "", out.stderr
    out = subprocess.run([sys.executable, str(LAUNCHER), "--help"],
                         capture_output=True, text=True, timeout=30)
    assert out.returncode == 0


def test_v2_yaml_loads() -> None:
    from tools.gui.yaml_compat import load_preset_yaml
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False) as fh:
        fh.write("schema_version: 2\n")
        path = Path(fh.name)
    assert load_preset_yaml(path)["schema_version"] == 2


def test_parse_numeric_rejects_bool_current() -> None:
    import pytest
    with pytest.raises(ValueError):
        parse_numeric("1", True)


def test_numeric_parse_strict() -> None:
    import pytest
    assert parse_numeric("0x1F", 0) == 31
    with pytest.raises(ValueError):
        parse_numeric("1.5", 0)


def _display_available() -> bool:
    import os
    try:
        import tkinter as tk
        root = tk.Tk()
        root.destroy()
        return True
    except Exception:
        return False


def test_app_constructs_and_rebinds(tmp_path) -> None:
    if not _display_available():
        import pytest
        pytest.skip("no DISPLAY")
    import tkinter as tk
    from tools.gui.app import WorkspaceApp
    from tools.gui.workspace_session import WorkspaceSession

    root = tk.Tk()
    try:
        app = WorkspaceApp(root, recover=False,
                           session_root=tmp_path / "sessions")
        old_model = app.model
        s2, m2 = WorkspaceSession.create(tmp_path / "other", app.model.config)
        app._rebind_workspace(m2, s2, {}, {})
        assert app.format_panel.model is m2
        assert app.params.model is m2
        assert app.controller.model is m2
        app._on_close()
        deadline = 50
        while not app.manager.shutdown_complete and deadline:
            import time
            time.sleep(0.05)
            deadline -= 1
        assert app.manager.shutdown_complete
    finally:
        try:
            root.destroy()
        except Exception:
            pass
