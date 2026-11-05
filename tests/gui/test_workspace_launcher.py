from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[2]
LAUNCHER = REPO / "tools" / "isp_calibration_gui.py"


def test_help_headless_no_heavy_imports() -> None:
    import os
    env = dict(os.environ)
    env.pop("DISPLAY", None)
    out = subprocess.run(
        [sys.executable, str(LAUNCHER), "--help"],
        capture_output=True, text=True, timeout=30, env=env)
    assert out.returncode == 0
    assert "--workspace" in out.stdout


def test_import_does_not_load_heavy_modules() -> None:
    code = (
        "import sys; sys.path.insert(0, 'tools');"
        "import isp_calibration_gui as m;"
        "heavy = [x for x in ('tkinter','numpy','matplotlib','cv2','brilliant_isp')"
        " if x in sys.modules];"
        "print(','.join(heavy))")
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, timeout=30,
                         cwd=str(REPO))
    assert out.stdout.strip() == "", out.stderr


def test_lazy_getattr_resolves_legacy_symbols() -> None:
    code = (
        "import sys; sys.path.insert(0, '.');"
        "import os; os.environ.setdefault('MPLBACKEND','Agg');"
        "from tools import isp_calibration_gui as m;"
        "print(type(m.CalibrationApp))")
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, timeout=60,
                         cwd=str(REPO))
    assert "class" in out.stdout, out.stderr


@pytest.mark.parametrize("workspace", [False, True])
def test_launcher_selects_requested_application(monkeypatch, workspace) -> None:
    from tools import isp_calibration_gui as launcher

    calls = []

    class Root:
        def mainloop(self):
            calls.append("mainloop")

    def application(name):
        def create(root, **kwargs):
            assert isinstance(root, Root)
            calls.append((name, kwargs))
        return create

    monkeypatch.setitem(sys.modules, "tkinter", SimpleNamespace(Tk=Root))
    monkeypatch.setitem(sys.modules, "tools.gui.app",
                        SimpleNamespace(WorkspaceApp=application("workspace")))
    monkeypatch.setitem(sys.modules, "tools.gui.legacy_calibration",
                        SimpleNamespace(CalibrationApp=application("legacy")))
    args = ["--config", "preset.yml", "--raw", "frame.raw", "--no-recover"]
    if workspace:
        args.append("--workspace")
    assert launcher.main(args) == 0
    assert calls == [
        ("workspace" if workspace else "legacy",
         {"preset": "preset.yml", "recover": False, "initial_raw": "frame.raw"}),
        "mainloop",
    ]
