from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

tk = pytest.importorskip("tkinter")

PGM_BYTES = (b"P5\n64 48\n4095\n"
             + (np.arange(64 * 48, dtype=np.uint16)
                .reshape(48, 64) % 4096).astype(">u2").tobytes())


def _display() -> bool:
    try:
        r = tk.Tk(); r.destroy(); return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _display(), reason="no Tk display")


def _pump(app, predicate, deadline=90.0):
    t0 = time.monotonic()
    while not predicate() and time.monotonic() - t0 < deadline:
        app.root.update()
        time.sleep(0.05)
    return predicate()


class _App:
    def __init__(self, tmp_path, **kw):
        self.kw = kw
        self.tmp = tmp_path

    def __enter__(self):
        from tools.gui.app import WorkspaceApp
        self.root = tk.Tk()
        try:
            self.app = WorkspaceApp(
                self.root, recover=False,
                session_root=self.tmp / "sessions", **self.kw)
        except Exception:
            try:
                self.root.destroy()
            except Exception:
                pass
            raise
        return self.app

    def __exit__(self, *a):
        try:
            self.app.controller.close()
            self.app.manager.request_close()
            deadline = time.monotonic() + 15
            while not self.app.manager.shutdown_complete \
                    and time.monotonic() < deadline:
                self.app.manager.shutdown_step()
                time.sleep(0.05)
            assert self.app.manager.shutdown_complete
        finally:
            try:
                self.root.destroy()
            except Exception:
                pass


def _import_and_process(app, tmp_path, name="in.pgm") -> Path:
    src = tmp_path / name
    src.write_bytes(PGM_BYTES)
    with patch("tkinter.filedialog.askopenfilename",
               return_value=str(src)):
        app._on_import()
    assert _pump(app, lambda: app.model.source is not None, 60)
    app._on_process()
    def finished():
        res = app.controller.current_result
        return res is not None and bool(res.get("result_path"))
    assert _pump(app, finished, 90)
    res = app.controller.current_result
    assert res is not None
    return Path(res["result_path"]).parent


def test_real_flow_ab_side_by_side_wipe_difference(tmp_path) -> None:
    with _App(tmp_path) as app:
        d1 = _import_and_process(app, tmp_path)
        hash_d1 = sorted(p.read_bytes()
                         for p in d1.iterdir() if p.is_file())
        app.comparison.pin("A")
        cfg0 = app.model.config
        app.model.edit("digital_gain.current_gain", 1)
        app._on_edit()
        app._on_process()
        def moved():
            res = app.controller.current_result
            return (res is not None and res.get("result_path")
                    and Path(res["result_path"]).parent != d1)
        assert _pump(app, moved, 90)
        res = app.controller.current_result
        assert res is not None
        d2 = Path(res["result_path"]).parent
        hash_d2 = sorted(p.read_bytes()
                         for p in d2.iterdir() if p.is_file())
        cfg_after = app.model.config
        app.comparison.pin("B")
        assert app.comparison.pins["A"] is not None
        assert app.comparison.pins["A"]["manifest"] == str(d1)
        assert app.comparison.pins["B"] is not None
        app.comparison.mode.set("Side by side")
        app._comparison_changed()
        assert app._pair_dirs["A"] == d1
        assert app._display_dir == d2
        app.comparison.mode.set("Wipe")
        app._comparison_changed()
        wanted = app._render_wanted
        assert wanted is not None
        assert _pump(
            app,
            lambda: app._render_wanted is None
            and app.controller.render_result is not None, 90)
        render_jobs = [j for j in app.controller.jobs
                       if j.get("role") == "render"]
        assert render_jobs
        assert render_jobs[-1]["status"] == "done"
        assert app.source_label.cget("text").startswith("Derived")
        artist = app.preview._artist
        assert artist is not None
        arr = artist.get_array()
        assert arr is not None and arr.dtype == np.uint8
        assert arr.shape[0] == 48 and arr.shape[1] == 64
        app.controller.render_result = None
        app.controller.render_result = None
        app.comparison.mode.set("Difference")
        app._comparison_changed()
        assert _pump(
            app,
            lambda: app._render_wanted is None
            and app.controller.render_result is not None, 90)
        diff_jobs = [j for j in app.controller.jobs
                     if j.get("role") == "render"]
        assert diff_jobs[-1]["status"] == "done"
        assert "difference" in app.source_label.cget("text").lower()
        artist = app.preview._artist
        assert artist is not None
        arr = artist.get_array()
        assert arr is not None and arr.dtype == np.uint8 and arr.ndim == 3
        assert app.model.config == cfg_after
        after1 = sorted(p.read_bytes()
                        for p in d1.iterdir() if p.is_file())
        after2 = sorted(p.read_bytes()
                        for p in d2.iterdir() if p.is_file())
        assert after1 == hash_d1 and after2 == hash_d2


def test_foreign_fingerprint_registration(tmp_path) -> None:
    with _App(tmp_path) as app:
        d1 = _import_and_process(app, tmp_path)
        foreign = tmp_path / "foreign"
        shutil.copytree(d1, foreign)
        manifest = json.loads((foreign / "result.json").read_text())
        manifest["source_fingerprint"] = "different"
        (foreign / "result.json").write_text(json.dumps(manifest))
        app.comparison.pins["A"] = {"manifest": str(d1)}
        app.comparison.pins["B"] = {"manifest": str(foreign)}
        app.comparison.mode.set("Side by side")
        app._comparison_changed()
        assert app._pair_dirs["A"] == d1
        assert app.preview._link_allowed is False
        app.comparison.mode.set("Difference")
        app._comparison_changed()
        assert app._display_dir is None
        assert app._render_wanted is None
        jobs = [j for j in app.controller.jobs
                if j.get("role") == "render"]
        assert not jobs


def test_stale_feedback_guard(tmp_path) -> None:
    with _App(tmp_path) as app:
        _import_and_process(app, tmp_path)
        app.model.edit("digital_gain.current_gain", 1)
        with patch("tools.gui.feedback.feedback_edits",
                   side_effect=AssertionError("must not run")):
            app._apply_feedback("awb")
        assert "stale" in getattr(app, "_notice", "")


def test_feedback_apply_one_undo(tmp_path, monkeypatch) -> None:
    with _App(tmp_path) as app:
        _import_and_process(app, tmp_path)
        before = app.model.get("digital_gain.current_gain")
        def fake(config, feedback, kind):
            return {"digital_gain.current_gain": before + 2}
        monkeypatch.setattr("tools.gui.feedback.feedback_edits", fake)
        app._apply_feedback("gain")
        assert app.model.get("digital_gain.current_gain") == before + 2
        app._on_undo()
        assert app.model.get("digital_gain.current_gain") == before


def test_persistence_snapshot_fields(tmp_path) -> None:
    with _App(tmp_path) as app:
        d1 = _import_and_process(app, tmp_path)
        app.comparison.pin("A")
        with patch("tkinter.messagebox.showerror"), \
             patch("tkinter.messagebox.showwarning"):
            assert app._on_save_session() is True
        payload = json.loads(
            (app.session.directory / "workspace.json").read_text())
        snaps = payload["snapshots"]
        assert "current" in snaps
        assert snaps["A"] is not None
        assert isinstance(snaps["A"], dict)
        assert "current" in snaps
        assert payload["ui_state"]["comparison"]["mode"] == "Current"


def test_close_cancel_and_discard(tmp_path) -> None:
    with _App(tmp_path) as app:
        app.model.edit("digital_gain.current_gain", 1)
        with patch("tkinter.messagebox.askyesnocancel",
                   return_value=None):
            app._on_close()
        assert not app._closing
        assert app.root.winfo_exists()
