from __future__ import annotations

import json
from pathlib import Path
import time
from unittest.mock import patch

import pytest

from tests.gui.test_workspace_comparison_ui import _App, _display, _import_and_process, _pump

pytestmark = pytest.mark.skipif(not _display(), reason="no Tk display")


def _next_process(app, previous):
    app._on_process()

    def complete():
        result = app.controller.current_result
        return bool(result and result.get("result_path")
                    and Path(result["result_path"]).parent != previous and result.get("_rendered"))

    assert _pump(app, complete)
    return Path(app.controller.current_result["result_path"]).parent


def test_single_A_then_export_selected(tmp_path, monkeypatch) -> None:
    with _App(tmp_path) as app:
        first = _import_and_process(app, tmp_path)
        app.comparison.pin("A")
        app.comparison.mode.set("A")
        app._comparison_changed()
        assert app._display_dir == first
        app._on_process()
        assert _pump(app, lambda: app.controller.current_result is not None
                     and Path(app.controller.current_result["result_path"]).parent != first)
        assert app._display_dir == first
        app.comparison.mode.set("B")
        app._comparison_changed()
        assert app._display_dir is None
        assert app._render_wanted is None
        app.comparison.mode.set("A")
        app._comparison_changed()
        destination = tmp_path / "exported"
        destination.mkdir()
        requests = []
        original = app.controller.request_export

        def export(*args):
            request = original(*args)
            assert request is not None
            requests.append(request)
            return request

        monkeypatch.setattr(app.controller, "request_export", export)
        with patch("tkinter.filedialog.askdirectory", return_value=str(destination)), \
             patch("tkinter.simpledialog.askstring", return_value="selected"):
            app._on_export()
        assert len(requests) == 1
        assert Path(json.loads(requests[0].payload_json)["manifest_dir"]) == first
        assert _pump(app, lambda: any(job["request_id"] == requests[0].tag.request_id
                     and job["status"] == "done" for job in app.controller.jobs))
        frozen = json.loads((first / "result.json").read_text())
        assert (destination / f"selected.{frozen['native_metadata']['format']}").is_file()
        assert json.loads((destination / "selected.json").read_text()) == frozen["native_metadata"]


def test_restore_workspace_actual_new_app(tmp_path) -> None:
    with _App(tmp_path / "first") as app:
        first = _import_and_process(app, tmp_path)
        app.comparison.pin("A")
        second = _next_process(app, first)
        app.comparison.pin("B")
        app.comparison.mode.set("A")
        app._comparison_changed()
        app.root.update()
        app.view.paned.sashpos(0, 280)
        app.view.paned.sashpos(1, 950)
        app.preview.ax.set_xlim(5, 40)
        app.preview.ax.set_ylim(30, 3)
        viewport = app.preview.viewport()
        directory = app.session.directory
        assert app._on_save_session()
    with _App(tmp_path / "second") as app:
        with patch("tkinter.filedialog.askdirectory", return_value=str(directory)), \
             patch("tkinter.messagebox.askyesnocancel", return_value=False):
            app._on_open()
        app.root.update()
        pin_a, pin_b = app.comparison.pins["A"], app.comparison.pins["B"]
        assert pin_a is not None and pin_b is not None
        assert pin_a["manifest"] == str(first)
        assert pin_b["manifest"] == str(second)
        assert app.comparison.mode.get() == "A"
        assert app._display_dir == first
        assert app.preview.widget.winfo_width() > 600
        assert app.view.ui_state()["sashes"] == [280, 950]
        restored = app.preview.viewport()
        assert restored["xlim"] == pytest.approx(viewport["xlim"])
        assert restored["ylim"] == pytest.approx(viewport["ylim"])
        app.model.edit("digital_gain.current_gain", 2)
        app._on_edit()
        app.root.update()
        assert app.view.ui_state()["sashes"] == [280, 950]


def test_analysis_real_response_and_picker(tmp_path, monkeypatch) -> None:
    with _App(tmp_path) as app:
        directory = _import_and_process(app, tmp_path)
        app.view.inspector_notebook.select(app.view.analysis_frame)

        def presented():
            result = app.controller.analysis_result
            return bool(result and app.analysis._seen == result["request_id"])

        assert _pump(app, presented)
        assert app.analysis.table.get_children()
        assert "pixels" in app.analysis.heading.get()
        seen = []

        def pixel(path, x, y, cache):
            seen.append((path, x, y))
            return {"x": 10, "y": 12, "component_names": ["R", "G", "B"],
                    "values": [1, 2, 3], "units": "codes", "domain": "display_encoded_rgb", "note": ""}

        monkeypatch.setattr("tools.gui.app.artifact_pixel", pixel)
        app._set_notice("Background failure")
        app._on_picker(10., 12., "current")
        assert seen == [(directory, 10., 12.)]
        assert "codes" in app.analysis.picker.get()
        assert "display_encoded_rgb" in app.analysis.picker.get()
        assert app._notice == "Background failure"


def test_roi_scoped_and_grid_guard(tmp_path, monkeypatch) -> None:
    with _App(tmp_path) as app:
        app.capture.set("white_balance")
        first = _import_and_process(app, tmp_path)
        app.comparison.pin("A")
        second = _next_process(app, first)
        app.comparison.pin("B")
        app.comparison.mode.set("Side by side")
        app._comparison_changed()
        app._on_roi([0, 0, 8, 8], "A")
        assert app.analysis.directory == str(first)
        assert app.analysis.comparison_dir == str(second)
        assert app.analysis.roi == [0, 0, 8, 8]
        app._on_roi([1, 1, 9, 9], "B")
        assert app.analysis.directory == str(second)
        assert app.analysis.comparison_dir == str(first)
        assert app.analysis.roi == [1, 1, 9, 9]
        app._on_roi(None, "B")
        assert app.analysis.roi is None
        monkeypatch.setattr(app.controller, "request_analysis", lambda *args: None)
        app.analysis.stage.set("white_balance")
        app.analysis.preview_geometry = (3, 2)
        app.analysis.roi = [0, 0, 8, 8]
        app.analysis.request()
        assert app.analysis.roi is None
        assert "different grid" in app.analysis.roi_text.get()
        app._show_source()
        assert app.analysis.roi is None


def test_batch_select_wires_latest(tmp_path) -> None:
    with _App(tmp_path) as app:
        directory = _import_and_process(app, tmp_path)
        app.controller.batch.append({
            "batch_id": "batch-result", "label": "frozen", "status": "done",
            "source": dict(app.model.source), "result_path": str(directory / "result.json"),
        })
        app.batch.refresh()
        app.batch.tree.selection_set("batch-result")
        app.batch._select(None)
        assert app.comparison.mode.get() == "Current"
        assert app._latest_dir == directory
        assert app._display_dir == directory


def test_close_save_failure_keeps_root(tmp_path) -> None:
    with _App(tmp_path) as app:
        app.model.edit("digital_gain.current_gain", 1)
        with patch("tkinter.messagebox.askyesnocancel", return_value=True), \
             patch.object(app.session, "save", side_effect=OSError("write failed")), \
             patch("tkinter.messagebox.showerror"):
            app._on_close()
        assert not app._closing
        assert app.root.winfo_exists()
        assert app.model.is_dirty


def test_close_busy_worker_after_discard(tmp_path) -> None:
    with _App(tmp_path) as app:
        _import_and_process(app, tmp_path)
        app._on_process()
        assert app.controller.active is not None
        with patch("tkinter.messagebox.askyesnocancel", return_value=False):
            app._on_close()
        deadline = time.monotonic() + 15
        while not app.manager.shutdown_complete and time.monotonic() < deadline:
            app.root.update()
            time.sleep(.01)
        assert app._closing
        assert app.manager.shutdown_complete


def test_invaliddraft_prompts_preserve_and_reset_undo(tmp_path) -> None:
    with _App(tmp_path) as app:
        row = app.format_panel.rows["sensor_info.width"]
        row.var.set("garbage")
        with patch("tkinter.filedialog.askopenfilename", return_value="/not-read.yml"), \
             patch("tkinter.messagebox.askyesnocancel", return_value=None):
            app._load_preset()
        assert app.model.get("sensor_info.width") == "garbage"
        with patch("tkinter.messagebox.askyesno", return_value=False):
            app._reset_all()
            app._upgrade()
        assert app.model.get("sensor_info.width") == "garbage"
        with patch("tkinter.messagebox.askyesno", return_value=True):
            app._reset_all()
        assert app.model.get("sensor_info.width") != "garbage"
        app._on_undo()
        assert app.model.get("sensor_info.width") == "garbage"


def test_current_mode_clears_wanted_late_render(tmp_path) -> None:
    with _App(tmp_path) as app:
        directory = _import_and_process(app, tmp_path)
        app.comparison.pin("A")
        app.comparison.pin("B")
        app.comparison.mode.set("Wipe")
        app._comparison_changed()
        wanted = app._render_wanted
        assert wanted is not None
        app.comparison.mode.set("Current")
        app._comparison_changed()
        assert app._render_wanted is None
        app.controller.render_result = {
            "request_id": wanted[0], "render_path": "/not-loaded",
            "descriptor": {"mode": "wrong"},
        }
        app._show_render()
        assert app._display_dir == directory
        assert not app.preview._pair


def test_native_export_blocks_invalid_draft(tmp_path) -> None:
    with _App(tmp_path) as app:
        directory = _import_and_process(app, tmp_path)
        app.format_panel.rows["sensor_info.width"].var.set("garbage")
        with patch("tkinter.filedialog.askdirectory", side_effect=AssertionError("invalid draft must block export")):
            app._on_export()
        assert app._display_dir == directory
        assert "invalid" in app._notice
        assert app.view.buttons["export"].instate(["disabled"])
        app._on_process()
        assert "sensor_info.width" in app._notice
