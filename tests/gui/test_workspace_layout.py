from __future__ import annotations

import pytest

tk = pytest.importorskip("tkinter")

def _display() -> bool:
    try:
        r = tk.Tk(); r.destroy(); return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _display(), reason="no Tk display")


def _app(tmp_path):
    from tools.gui.app import WorkspaceApp
    from unittest.mock import patch
    with patch("tkinter.messagebox.askyesno", return_value=False):
        return WorkspaceApp(tk.Tk(), recover=False,
                            session_root=tmp_path / "s")


def _close(app):
    import time
    app._closing = True
    app.controller.close()
    deadline = time.monotonic() + 15
    while not app.manager.shutdown_complete and time.monotonic() < deadline:
        app.manager.shutdown_step()
        time.sleep(0.05)
    try:
        app.root.destroy()
    except Exception:
        pass
    assert app.manager.shutdown_complete


def test_toolbar_buttons_map(tmp_path) -> None:
    app = _app(tmp_path)
    try:
        root = app.root
        assert set(app.view.buttons) == {
            "process", "cancel", "stop", "import", "open", "new",
            "save_session", "export", "undo", "redo"}
        for key, btn in app.view.buttons.items():
            assert app.view.handlers[key]
    finally:
        _close(app)


def test_inspector_notebook_tabs(tmp_path) -> None:
    app = _app(tmp_path)
    try:
        nb = app.view.inspector_notebook
        tabs = [nb.tab(t, "text") for t in nb.tabs()]
        assert tabs == ["Parameters", "Analysis", "Feedback", "Jobs"]
        assert app.view.editor_frame.winfo_exists()
        assert app.view.analysis_frame.winfo_exists()
        assert app.view.feedback_frame.winfo_exists()
        assert app.view.jobs_frame.winfo_exists()
    finally:
        _close(app)


def test_set_busy_start_stop_once(tmp_path) -> None:
    app = _app(tmp_path)
    try:
        app.view.set_busy(True)
        app.view.set_busy(False)
        app.view.set_busy(False)
        app.root.update()
    finally:
        _close(app)


def test_ui_state_roundtrip(tmp_path) -> None:
    app = _app(tmp_path)
    try:
        app.root.geometry("1366x768")
        app.root.update()
        app.view.reset_layout()
        state = app.view.ui_state()
        assert isinstance(state["sashes"], list) and len(state["sashes"]) == 2
        app.view.restore_ui({"sashes": [280, 900]})
        app.root.update()
        positions = app.view.ui_state()["sashes"]
        assert positions[0] == 280
        assert positions[1] == 900
    finally:
        _close(app)


def test_pipeline_tree_in_place_and_alias(tmp_path) -> None:
    app = _app(tmp_path)
    try:
        tree = app.pipeline.tree
        assert tree.exists("noise_reduction_2d")
        assert not tree.exists("2d_noise_reduction")
        tree.selection_set("noise_reduction_2d")
        sel0 = tree.selection()
        app.pipeline.refresh()
        assert tree.selection() == sel0
        assert tree.exists("tone_mapping") or tree.exists("tone_mapping_post")             or tree.exists("tone_mapping_pre")
        app.model.edit("tone_mapping.is_enable", True)
        app.model.edit("tone_mapping.tone_mapping_before_demosaic", True)
        app.pipeline.refresh()
        assert tree.exists("tone_mapping_pre")
    finally:
        _close(app)


def test_set_timings_column(tmp_path) -> None:
    app = _app(tmp_path)
    try:
        app.pipeline.set_timings([
            {"stage_id": "demosaic", "elapsed_ns": 1_500_000},
            {"stage_id": "demosaic", "elapsed_ns": 500_000},
            {"stage_id": "x", "elapsed_ns": True},
            {"stage_id": "y", "elapsed_ns": -5}])
        vals = app.pipeline.tree.item("demosaic", "values")
        assert vals[2] == "2.00"
        app.pipeline.clear_timings()
        assert app.pipeline.tree.item("demosaic", "values")[2] == ""
    finally:
        _close(app)


def test_batch_panel_compact(tmp_path) -> None:
    app = _app(tmp_path)
    try:
        from tools.gui.views.batch import BatchPanel
        panel = BatchPanel(app.view.jobs_frame, app.controller,
                           lambda p: None, lambda s: None)
        panel.pack()
        app.root.geometry("1366x768")
        app.root.update()
        assert panel.winfo_reqwidth() < 400
    finally:
        _close(app)


def test_stage_selection_maps_section(tmp_path) -> None:
    app = _app(tmp_path)
    try:
        tree = app.pipeline.tree
        tree.selection_set("demosaic")
        tree.event_generate("<<TreeviewSelect>>")
        assert app.params.section == "demosaic"
        tree.selection_set("noise_reduction_2d")
        tree.event_generate("<<TreeviewSelect>>")
        assert app.params.section == "2d_noise_reduction"
    finally:
        _close(app)



def test_advanced_selection_stable_and_retired_labels(tmp_path) -> None:
    app = _app(tmp_path)
    try:
        tree = app.pipeline.tree
        adv_children = tree.get_children("advanced")
        assert adv_children
        target = adv_children[0]
        tree.selection_set(target)
        sel0 = tree.selection()
        app.pipeline.refresh()
        assert tree.exists("advanced")
        assert tree.selection() == sel0
        assert tree.exists(target)
        legacy = [c for c in tree.get_children("advanced")
                  if tree.item(c, "values")[1]
                  == "migration only—not native"]
        for iid in ("gamma", "color_space_conversion",
                    "rgb_conversion", "yuv_conversion_format"):
            if f"adv:{iid}" in tree.get_children("advanced"):
                assert f"adv:{iid}" in legacy
        ids = [c for c in tree.get_children("advanced")]
        assert ids == sorted(ids, key=lambda i:
            list(app.model.config.keys()).index(i.removeprefix("adv:")))
    finally:
        _close(app)


def test_editor_key_aliases(tmp_path) -> None:
    app = _app(tmp_path)
    try:
        tree = app.pipeline
        assert tree._editor_key("noise_reduction_2d") == "2d_noise_reduction"
        assert tree._editor_key("green_repair") == "sensor_info"
        for iid in ("ldci_pre_transfer", "ldci_post_transfer"):
            assert tree._editor_key(iid) == "ldci"
        assert tree._editor_key("transfer_encode") == "output"
        assert tree._editor_key("demosaic_render") == "demosaic"
        assert tree._editor_key(
            "color_correction_matrix_render") == "color_correction_matrix"
        assert tree._editor_key("adv:aces") == "aces"
        assert tree._state("sensor_info", app.model.config,
                           "green_repair") == "runtime"
    finally:
        _close(app)


def test_tmo_pre_domain_and_render_rows(tmp_path) -> None:
    app = _app(tmp_path)
    try:
        app.model.edit("tone_mapping.is_enable", True)
        app.model.edit("tone_mapping.tone_mapping_before_demosaic", True)
        app.pipeline.refresh()
        tree = app.pipeline.tree
        assert tree.exists("tone_mapping_pre")
        vals = tree.item("tone_mapping_pre", "values")
        assert vals[1] == "linear Bayer→display-linear Bayer"
        assert tree.exists("demosaic_render")
        assert tree.exists("color_correction_matrix_render")
        app.model.edit("tone_mapping.tone_mapping_before_demosaic", False)
        app.pipeline.refresh()
        assert tree.exists("tone_mapping_post")
        assert not tree.exists("demosaic_render")
    finally:
        _close(app)


def test_actual_mapped_layout_1366(tmp_path) -> None:
    app = _app(tmp_path)
    try:
        app.root.geometry("1366x768")
        for _ in range(30):
            app.root.update()
        assert app.preview.widget.winfo_width() >= 650, \
            f"centre width {app.preview.widget.winfo_width()}"
    finally:
        _close(app)


def test_actual_mapped_layout_1920(tmp_path) -> None:
    app = _app(tmp_path)
    try:
        app.root.geometry("1920x1080")
        for _ in range(30):
            app.root.update()
        assert app.preview.widget.winfo_width() >= 1000, \
            f"centre width {app.preview.widget.winfo_width()}"
    finally:
        _close(app)


def test_sash_not_forced_after_user_resize(tmp_path) -> None:
    app = _app(tmp_path)
    try:
        app.root.geometry("1366x768")
        for _ in range(30):
            app.root.update()
        app.view.paned.sashpos(0, 320)
        app.model.edit("digital_gain.current_gain", 1)
        app.root.update()
        assert app.view.paned.sashpos(0) == 320
    finally:
        _close(app)
