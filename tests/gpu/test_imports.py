"""Import isolation: the package must work without torch and must never
pull GUI bindings. Run in clean subprocesses so module state cannot leak."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _run(code: str, extra_env: dict | None = None):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO)
    env.pop("DISPLAY", None)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env=env, cwd=REPO, timeout=120,
    )


def test_package_imports_without_torch():
    code = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "import brilliantisp_gpu\n"
        "from brilliantisp_gpu import types, config\n"
        "from brilliantisp_gpu.types import GpuOptions\n"
        "GpuOptions()\n"
        "try:\n"
        "    types.require_torch()\n"
        "except types.GpuUnavailableError:\n"
        "    pass\n"
        "else:\n"
        "    raise SystemExit('require_torch did not fail without torch')\n"
        "print('ok')\n"
    )
    proc = _run(code)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_ops_modules_import_torch_lazily():
    code = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "import brilliantisp_gpu.ops.uint64\n"
        "import brilliantisp_gpu.ops.borders\n"
        "import brilliantisp_gpu.ops.rtc\n"
        "import brilliantisp_gpu.ops.reference_sort\n"
        "import brilliantisp_gpu.blocks.statistics\n"
        "import brilliantisp_gpu.blocks.tone_mapping\n"
        "assert 'torch' not in {k for k, v in sys.modules.items() if v is not None}\n"
        "print('ok')\n"
    )
    proc = _run(code)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_no_gui_imports_headless():
    code = (
        "import sys\n"
        "import brilliantisp_gpu\n"
        "from brilliantisp_gpu import config\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in (\n"
        "    'PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'tkinter', 'dearpygui',\n"
        "    'matplotlib.backends.backend_tkagg')]\n"
        "assert not bad, bad\n"
        "print('ok')\n"
    )
    proc = _run(code)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout
