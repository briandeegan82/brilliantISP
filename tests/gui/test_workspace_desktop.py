from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml

tk = pytest.importorskip("tkinter")

PGM_BYTES = (b"P5\n64 48\n4095\n"
             + (np.arange(64 * 48, dtype=np.uint16)
                .reshape(48, 64) % 4096).astype(">u2").tobytes())


def _display() -> bool:
    try:
        r = tk.Tk(); r.destroy(); return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _display(), reason="no Tk display")


def _fast_cfg() -> dict:
    from tests.test_pipeline_runner import _fast_config
    return _fast_config()


def _pump(app, predicate, deadline=30.0):
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


def test_app_default_construct_clean(tmp_path) -> None:
    with _App(tmp_path) as app:
        app.root.update()
        assert app.model.config_json_ready()
        assert not app.model.is_dirty
        assert app.controller.active is None


def test_on_new_rebinds_panels(tmp_path) -> None:
    with _App(tmp_path) as app:
        old = app.model
        old_cfg = old.config
        with patch("tkinter.messagebox.askyesno", return_value=True), \
             patch("tkinter.messagebox.showerror"):
            app._on_new()
        assert app.model is not old
        assert app.format_panel.model is app.model
        assert app.params.model is app.model
        app.format_panel.rows["sensor_info.width"].var.set("64")
        assert old.config["sensor_info"]["width"] == old_cfg["sensor_info"]["width"]
        assert app.model.get("sensor_info.width") == 64


def test_on_open_rebinds(tmp_path) -> None:
    from tools.gui.workspace_session import WorkspaceSession
    with _App(tmp_path) as app:
        old_cfg = app.model.config
        other, _m2 = WorkspaceSession.create(tmp_path / "other", _fast_cfg())
        old = app.model
        with patch("tkinter.filedialog.askdirectory",
                   return_value=str(other.directory)), \
             patch("tkinter.messagebox.askyesno", return_value=True), \
             patch("tkinter.messagebox.showerror"), \
             patch("tkinter.messagebox.showwarning"):
            app._on_open()
        assert app.model is not old
        assert app.format_panel.model is app.model
        assert app.params.model is app.model
        app.format_panel.rows["sensor_info.width"].var.set("64")
        assert app.model.get("sensor_info.width") == 64
        assert old.config["sensor_info"]["width"] == old_cfg["sensor_info"]["width"]


def test_import_process_export_flow(tmp_path) -> None:
    with _App(tmp_path) as app:
        src = tmp_path / "in.pgm"
        src.write_bytes(PGM_BYTES)
        with patch("tkinter.filedialog.askopenfilename",
                   return_value=str(src)):
            app._on_import()
        assert _pump(app, lambda: app.model.source is not None, 60), \
            "import did not adopt source"
        assert _pump(app, lambda: app.format_panel.rows[
            "sensor_info.width"].var.get() == "64", 10), \
            "format panel did not refresh to PGM dims"
        assert app.format_panel.rows["sensor_info.height"].var.get() == "48"
        record = app.model.source
        assert record is not None
        staged = Path(record["path"])
        assert staged.read_bytes() == PGM_BYTES
        view_manifest = record.get("view_manifest")
        assert view_manifest is not None
        assert (Path(view_manifest) / "result.json").is_file()
        cfg_before = app.model.config
        app._on_process()
        def processed():
            result = app.controller.current_result
            return result is not None and bool(result.get("result_path"))
        assert _pump(
            app, processed, 60), \
            "process did not finish"
        assert app.model.config == cfg_before
        result = app.controller.current_result
        assert result is not None
        result_path = Path(result["result_path"])
        assert result_path.exists()
        preview = np.load(result_path.parent / "preview.npy",
                          allow_pickle=False)
        assert preview.shape == (48, 64, 3)
        artist = app.preview._artist
        assert artist is not None
        arr = artist.get_array()
        assert arr is not None and arr.shape == (48, 64, 3)
        dest = tmp_path / "out"
        dest.mkdir()
        req = app.controller.request_export(result_path.parent, str(dest),
                                            "test")
        assert req is not None
        assert _pump(
            app,
            lambda: app.controller.active is None
            and app.controller.status in ("done", "error"), 30)
        assert app.controller.status == "done"
        exported = list(dest.iterdir())
        assert exported
        job = next(item for item in app.controller.jobs if item["request_id"] == req.tag.request_id)
        assert job["status"] == "done"
        assert job["payload"]["artifacts"]
        assert all(Path(artifact).is_file() for artifact in job["payload"]["artifacts"])
        assert any("test" in p.name for p in exported)


def test_invalid_entry_blocks_process_immediate(tmp_path) -> None:
    with _App(tmp_path) as app:
        row = app.format_panel.rows["sensor_info.width"]
        row.var.set("garbage")
        assert app.model.get("sensor_info.width") == "garbage"
        with patch("tkinter.messagebox.showerror"), \
             patch("tkinter.messagebox.showwarning"):
            app._on_process()
            assert app.controller.active is None
            assert app._on_save_session() is False
        row.var.set("64")
        with patch("tkinter.messagebox.showerror"), \
             patch("tkinter.messagebox.showwarning"):
            assert app._on_save_session() is True


def test_matrix_yaml_invalid_then_undo(tmp_path) -> None:
    with _App(tmp_path) as app:
        app.params.show_section("color_correction_matrix")
        path = next(p for p in app.params.rows
                    if p.startswith("color_correction_matrix.")
                    and isinstance(app.model.config.get(
                        "color_correction_matrix", {}).get(
                        p.split(".", 1)[1]), (list, dict)))
        row = app.params.rows[path]
        assert row._text is not None, "matrix row is not a YAML Text"
        row._text.delete("1.0", "end")
        row._text.insert("1.0", "[[1, 0], [0, 1")
        row._text.edit_modified(True)
        app._flush_all()
        cfg = app.model.config
        leaf = cfg.get("color_correction_matrix", {}).get(
            path.split(".", 1)[1])
        assert leaf == "[[1, 0], [0, 1"
        assert not app.model.config_json_ready()
        with patch("tkinter.messagebox.showerror"):
            assert app._on_save_session() is False
        app._on_undo()
        cfg2 = app.model.config
        restored = cfg2.get("color_correction_matrix", {}).get(
            path.split(".", 1)[1])
        assert isinstance(restored, (list, dict))


def test_unchanged_flush_not_dirty(tmp_path) -> None:
    with _App(tmp_path) as app:
        rev0 = app.model.revision
        app.params.flush_all()
        assert app.model.revision == rev0
        assert not app.model.is_dirty


def test_missing_default_tmo_no_insert(tmp_path) -> None:
    from util.pipeline_migration import migrate_pipeline_config
    cfg, _report = migrate_pipeline_config(_fast_cfg())
    cfg["tone_mapping"].update(operator="hable", backend="float", parameters={}, is_enable=True)
    from tools.gui.workspace_session import WorkspaceSession
    s, model = WorkspaceSession.create(tmp_path / "s", cfg)
    with _App(tmp_path) as app:
        app._rebind_workspace(model, s, {}, {})
        rev0 = app.model.revision
        app.params.show_section("tone_mapping")
        app._flush_all()
        assert app.model.revision == rev0
        assert app.model.config["tone_mapping"]["parameters"] == {}
        row = app.params.rows.get(
            "tone_mapping.parameters.exposure_bias")
        assert row is not None, "missing-default exposure_bias row absent"
        row.var.set("3")
        app._flush_all()
        assert app.model.config["tone_mapping"]["parameters"][
            "exposure_bias"] == 3.0
        app._on_undo()
        assert "exposure_bias" not in app.model.config[
            "tone_mapping"]["parameters"]


def test_close_cancel_keeps_root(tmp_path) -> None:
    with _App(tmp_path) as app:
        app.model.edit("digital_gain.current_gain", 1)
        with patch("tkinter.messagebox.askyesnocancel",
                   return_value=None):
            app._on_close()
        assert not app._closing
        assert app.root.winfo_exists()


def test_v2_ctor(tmp_path) -> None:
    from tools.migrate_pipeline_config import run_migrate
    src = tmp_path / "v1.yml"
    src.write_text(yaml.safe_dump(_fast_cfg()))
    run_migrate([str(src)], tmp_path / "v2.yml", tmp_path / "r.json")
    with _App(tmp_path, preset=str(tmp_path / "v2.yml")) as app:
        app.root.update()
        assert app.model.config.get("schema_version") == 2


def test_undo_grouped_typing(tmp_path) -> None:
    with _App(tmp_path) as app:
        r0 = app.model.revision
        row = app.format_panel.rows["sensor_info.width"]
        before = app.model.get("sensor_info.width")
        app.model.begin_edit_group("sensor_info.width")
        row.var.set("64")
        row.var.set("128")
        row.var.set("96")
        app.model.end_edit_group("sensor_info.width")
        app._on_undo()
        assert app.model.get("sensor_info.width") == before
        assert app.model.revision > r0
