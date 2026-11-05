"""Integrated Tk workspace backed by isolated processing jobs."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

from tools.gui.artifacts import artifact_pixel, load_artifact, view_artifact
from tools.gui.comparison import compatibility_reason
from tools.gui.controller import WorkspaceController
from tools.gui.gui_log import attach_session_log, install_exception_hooks, setup_gui_logging
from tools.gui.jobs import WorkerManager
from tools.gui.session import DEFAULT_PRESET, DEFAULT_SESSION_ROOT
from tools.gui.views.batch import BatchPanel
from tools.gui.views.inputs import FormatPanel
from tools.gui.views.inspector import ParametersPanel
from tools.gui.views.preview import PreviewPane
from tools.gui.views.workspace import PipelineTree, WorkspaceView
from tools.gui.workspace_session import WorkspaceSession, read_last_workspace
from tools.gui.yaml_compat import dump_config_yaml, load_preset_yaml


class WorkspaceApp:
    def __init__(self, root: tk.Tk, preset: str | None = None, recover: bool = True,
                 initial_raw: str | None = None, session_root: Path | None = None) -> None:
        self.root = root
        style = ttk.Style(root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TFrame", background="#edf1f3")
        style.configure("TLabel", background="#edf1f3")
        style.configure("TButton", padding=(6, 3))
        style.configure("TNotebook.Tab", padding=(8, 4))
        style.map("Treeview", background=[("selected", "#147c76")], foreground=[("selected", "white")])
        self.session_root = Path(session_root) if session_root is not None else DEFAULT_SESSION_ROOT
        self._closing = False
        self._poll_after_id = None
        self._shutdown_after_id = None
        self._notice = ""
        self._seen_jobs: dict[str, str] = {}
        self._display_dir: Path | None = None
        self._pending_viewport: dict = {}
        self.snapshots: dict = {}
        self.log = setup_gui_logging()
        self.root.title("BrilliantISP — Calibration Workspace")
        self.root.minsize(1024, 640)
        self.root.geometry("1366x768")
        self.session, self.model, ui = self._initial_workspace(preset, recover)
        attach_session_log(self.session.directory)
        self.manager = WorkerManager()
        self.controller = WorkspaceController(self.model, self.manager, root_dir=self.session.directory.parent)
        self._generation = self.model.generation
        self._revision = -1
        self._operator = self.model.get("tone_mapping.operator")
        handlers = {name: getattr(self, "_on_" + name) for name in (
            "process", "cancel", "stop", "import", "open", "new", "export", "undo", "redo", "auto_toggle")}
        handlers["save_session"] = self._on_save_session
        self.view = WorkspaceView(root, handlers)
        self.view.pack(fill="both", expand=True)
        self.preview = PreviewPane(self.view.preview_frame)
        self.preview.widget.pack(fill="both", expand=True)
        self.format_panel = FormatPanel(self.view.format_tab, self.model, self._on_edit)
        self.format_panel.pack(fill="both", expand=True)
        self.params = ParametersPanel(self.view.editor_frame, self.model, self._on_edit)
        self.params.pack(fill="both", expand=True)
        self.pipeline = PipelineTree(self.view.pipeline_tab, self.model, self.params.show_section)
        self.pipeline.pack(fill="both", expand=True)
        batch_tab = ttk.Frame(self.view.left_notebook)
        self.view.left_notebook.add(batch_tab, text="Batch")
        self.batch = BatchPanel(batch_tab, self.controller, self._select_result, self._set_notice)
        self.batch.pack(fill="both", expand=True)
        self.source_label = ttk.Label(self.view.preview_frame, wraplength=650)
        self.source_label.pack(side="bottom", fill="x")
        from .views.comparison import ComparisonControls
        from .views.analysis_inspector import AnalysisInspector
        self.comparison = ComparisonControls(
            self.view.preview_frame,
            lambda: str(self._display_dir) if self._display_dir else None,
            self._comparison_changed, self._set_notice)
        self.comparison.pack(side="top", fill="x", before=self.preview.widget)
        self.analysis = AnalysisInspector(
            self.view.analysis_frame, self.controller, self._set_notice)
        self.analysis.pack(fill="both", expand=True)
        self.preview.on_picker = self._on_picker
        self.preview.on_roi = self._on_roi
        self.feedback_text = tk.Text(
            self.view.feedback_frame, state="disabled", wrap="word")
        self.feedback_text.pack(fill="both", expand=True)
        jobs_bar = ttk.Frame(self.view.feedback_frame)
        jobs_bar.pack(fill="x")
        ttk.Button(jobs_bar, text="Use measured AWB manually",
                   command=lambda: self._apply_feedback("awb")).pack(
                       side="left")
        ttk.Button(jobs_bar, text="Freeze applied gain",
                   command=lambda: self._apply_feedback("gain")).pack(
                       side="left")
        self.jobs_text = tk.Text(self.view.jobs_frame, state="disabled",
                                 wrap="word", height=8)
        self.jobs_text.pack(fill="both", expand=True)
        self._latest_dir: Path | None = None
        self._pair_dirs: dict = {}
        self._render_wanted: tuple | None = None
        self._render_failed = False
        self._picker_cache: dict = {}
        self._frozen_manifest: dict | None = None
        self._display_identity = None
        self._pair_registered = False
        self.view.inspector_notebook.bind("<<NotebookTabChanged>>", self._inspector_changed)
        self.capture = tk.StringVar(value="none")
        self._menus()
        for key, callback in (("<Control-s>", self._on_save_session), ("<Control-p>", self._on_process),
                              ("<Control-z>", self._on_undo), ("<Control-y>", self._on_redo)):
            self.root.bind(key, lambda e, cb=callback: cb())
        self.root.bind("<<WorkspaceReset>>", lambda e: self.root.after_idle(self._rebuild_forms))
        install_exception_hooks(tk_root=root, on_error=self._set_notice)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<Destroy>", self._root_destroyed, add="+")
        self._show_source()
        self._restore_ui(ui)
        if initial_raw:
            self.controller.request_import(initial_raw)
        self._refresh_status()
        self._poll_after_id = self.root.after(20, self._tick)

    def _initial_workspace(self, preset, recover):
        if recover and preset is None:
            try:
                last = read_last_workspace(self.session_root)
                if last is not None and messagebox.askyesno("Recover workspace", f"Recover {last.name}?", parent=self.root):
                    session, model, ui, snapshots, warnings = WorkspaceSession.open(last)
                    self.snapshots = snapshots
                    if warnings:
                        messagebox.showwarning("Workspace warnings", "\n".join(warnings), parent=self.root)
                    return session, model, ui
            except (ValueError, OSError) as exc:
                messagebox.showerror("Workspace recovery failed", str(exc), parent=self.root)
        path = Path(preset) if preset else DEFAULT_PRESET
        config = load_preset_yaml(path)
        session, model = WorkspaceSession.create(self.session_root, config, preset_path=str(path))
        return session, model, {}

    def _menus(self) -> None:
        menu = tk.Menu(self.root)
        file = tk.Menu(menu, tearoff=False)
        for label, command in (("New workspace", self._on_new), ("Open workspace…", self._on_open),
                               ("Save workspace", self._on_save_session), ("Import source…", self._on_import),
                               ("Load preset…", self._load_preset), ("Export preset YAML…", self._export_yaml),
                               ("Export original native result…", self._on_export), ("Close", self._on_close)):
            file.add_command(label=label, command=command)
        menu.add_cascade(label="File", menu=file)
        edit = tk.Menu(menu, tearoff=False)
        edit.add_command(label="Undo", command=self._on_undo)
        edit.add_command(label="Redo", command=self._on_redo)
        edit.add_command(label="Reset all to loaded preset", command=self._reset_all)
        edit.add_command(label="Upgrade configuration to native v2…", command=self._upgrade)
        menu.add_cascade(label="Edit", menu=edit)
        capture = tk.Menu(menu, tearoff=False)
        for stage in ("none", "white_balance", "demosaic", "color_correction_matrix", "linear_rgb_pre_tone_mapping"):
            capture.add_radiobutton(label=stage, variable=self.capture, value=stage)
        menu.add_cascade(label="Next Manual Capture", menu=capture)
        self.root.configure(menu=menu)

    def _tick(self) -> None:
        if self._closing:
            return
        self._poll_after_id = None
        try:
            self.controller.tick()
            if self.model.generation != self._generation:
                self._generation = self.model.generation
                self.controller.current_result = None
                self._rebuild_forms()
                self._show_source()
            result = self.controller.current_result
            if result and not result.get("_rendered") and not result.get("_render_failed"):
                if result.get("result_path"):
                    directory = Path(result["result_path"]).parent
                    self._latest_dir = directory
                    if self.comparison.mode.get() == "Current":
                        try:
                            self._show_artifact(str(directory))
                            self._render_failed = False
                        except Exception as exc:
                            self._render_failed = True
                            result["_render_failed"] = str(exc)
                            self._set_notice(f"preview failed: {exc}")
                        else:
                            result["_rendered"] = True
                    else:
                        result["_rendered"] = True
            if self.controller.render_result is not None:
                self._show_render()
            for job in self.controller.jobs[-200:]:
                if self._seen_jobs.get(job["request_id"]) != job["status"]:
                    self._seen_jobs[job["request_id"]] = job["status"]
                    if job["status"] == "error":
                        self._set_notice(job.get("payload", {}).get("error", "Worker failed"))
                    self.log.info("Job %s %s", job["request_id"], job["status"])
                    self.jobs_text.configure(state="normal")
                    self.jobs_text.insert("end", json.dumps(job, default=str) + "\n")
                    if int(self.jobs_text.index("end-1c").split(".")[0]) > 200:
                        self.jobs_text.delete("1.0", "2.0")
                    self.jobs_text.see("end")
                    self.jobs_text.configure(state="disabled")
            self.analysis.refresh()
            self._refresh_status()
            self.batch.refresh()
            live = {j["request_id"] for j in self.controller.jobs[-200:]}
            for stale in [k for k in self._seen_jobs if k not in live]:
                del self._seen_jobs[stale]
        except Exception as exc:
            self.log.exception("Workspace polling failed")
            self._set_notice(str(exc))
        finally:
            if not self._closing:
                self._poll_after_id = self.root.after(20, self._tick)

    def _show_artifact(self, directory: str) -> None:
        path = Path(directory)
        image, metadata, manifest, identity = view_artifact(path)
        if self._display_identity is not None and (
                identity.source_fingerprint != self._display_identity.source_fingerprint
                or identity.geometry != self._display_identity.geometry):
            self.analysis.roi = None
            self.analysis.roi_text.set("ROI: full frame")
        self.preview.show_rgb8(image, geometry_key=identity.source_fingerprint + identity.geometry)
        self._pair_dirs = {}
        self._display_identity = identity
        self._display_dir = path
        self._frozen_manifest = manifest
        if self._tag_is_current(manifest):
            self.pipeline.set_timings(manifest.get("timings") or [])
        else:
            self.pipeline.clear_timings()
        if self._pending_viewport:
            self.preview.restore_viewport(self._pending_viewport)
            self._pending_viewport = {}
        caption = "RAW visualization (stretched CFA, not processed RGB)" if manifest.get("kind") == "raw" else "sRGB display preview"
        name = Path(manifest["native_metadata"].get("source_identity", "")).name
        self.source_label.configure(text=f"{caption} | {name} [{identity.source_fingerprint[:10]}] | "
                                    f"r{manifest.get('tag', {}).get('revision', '?')} | "
                                    f"{metadata['width']} × {metadata['height']}")
        self._analysis_frame(path, None, metadata)
        self.feedback_text.configure(state="normal")
        self.feedback_text.delete("1.0", "end")
        self.feedback_text.insert("1.0", json.dumps({
            key: manifest.get(key) for key in ("tag", "source_fingerprint", "feedback", "warnings", "migration_report")
        }, indent=2))
        self.feedback_text.configure(state="disabled")

    def _tag_is_current(self, manifest: dict) -> bool:
        tag = manifest.get("tag", {})
        return (tag.get("session_id") == self.model.session_id
                and tag.get("generation") == self.model.generation
                and tag.get("revision") == self.model.revision
                and tag.get("source_fingerprint") == (self.model.source or {}).get("fingerprint"))

    def _analysis_frame(self, path, comparison, metadata) -> None:
        directory = str(Path(path))
        other = str(Path(comparison)) if comparison is not None else None
        self.analysis.preview_geometry = (metadata["width"], metadata["height"])
        if (self.analysis.directory, self.analysis.comparison_dir) != (directory, other):
            self.analysis.set_frame(directory, comparison_dir=other)
            self._inspector_changed()

    def _inspector_changed(self, _event=None) -> None:
        if self.view.inspector_notebook.select() == str(self.view.analysis_frame) and self.analysis.directory:
            self.analysis.request()

    def _select_result(self, directory) -> None:
        directory = Path(directory)
        self._render_wanted = None
        self._latest_dir = directory
        self.comparison.mode.set("Current")
        try:
            self._show_artifact(str(directory))
        except Exception as exc:
            self._set_notice(f"artifact load failed: {exc}")

    def _comparison_changed(self) -> None:
        self._notice = ""
        mode = self.comparison.mode.get()
        self._render_wanted = None
        try:
            if mode in ("Current", "A", "B"):
                pin = self.comparison.pins.get(mode)
                directory = self._latest_dir if mode == "Current" else pin and pin.get("manifest")
                if directory is None:
                    raise ValueError("No snapshot selected; pin a visible result first.")
                self._show_artifact(str(directory))
                return
            a, b = self.comparison.directories()
            pa, ma, _manifest_a, ia = view_artifact(a)
            pb, mb, _manifest_b, ib = view_artifact(b)
            reason = compatibility_reason(ia, ib)
            if mode == "Side by side":
                self.preview.show_pair(
                    pa, pb,
                    geometry_a=ia.source_fingerprint + ia.geometry,
                    geometry_b=ib.source_fingerprint + ib.geometry,
                    registered=reason is None)
                self._pair_dirs = {"A": Path(a), "B": Path(b)}
                self._pair_registered = reason is None
                self._display_dir = Path(b)
                self._display_identity = ib
                self._frozen_manifest = _manifest_b
                self._analysis_frame(b, a if reason is None else None, mb)
                self.source_label.configure(text=f"A (left) [{ia.source_fingerprint[:10]}] | "
                    f"B (right) [{ib.source_fingerprint[:10]}] | " + (reason or "Registered display previews"))
                return
            if reason is not None:
                raise ValueError(f"Registration required: {reason}")
            if mode not in ("Wipe", "Difference"):
                raise ValueError("Unknown comparison mode.")
            gain = self.comparison.difference_gain()
            if self._display_dir != Path(b) or self.preview._pair:
                self._show_artifact(b)
            self._pair_dirs = {"A": Path(a), "B": Path(b)}
            self._pair_registered = True
            self._analysis_frame(b, a, mb)
            req = self.controller.request_render(
                a, b, mode=mode.lower(),
                fraction=self.comparison.fraction.get(),
                difference_gain=gain)
            if req is None:
                raise ValueError("Workspace is closed.")
            self._render_wanted = (req.tag.request_id, req.payload_json)
            self.source_label.configure(text=f"{mode} rendering queued; previous display remains until ready.")
        except (OSError, ValueError, KeyError, tk.TclError) as exc:
            self._clear_display()
            self._set_notice(str(exc))

    def _show_render(self) -> None:
        result = self.controller.render_result
        if not result or not self._render_wanted:
            return
        want_id, want_payload = self._render_wanted
        if result["request_id"] != want_id or result["descriptor"] != json.loads(want_payload):
            return
        payload = json.loads(want_payload)
        if self.comparison.mode.get().lower() != payload["mode"]:
            self._render_wanted = None
            return
        try:
            import numpy as np
            path = Path(result["render_path"])
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest["preview"] != "preview.npy" or manifest["descriptor"] != payload:
                raise ValueError("Derived render descriptor does not match the requested view.")
            image = np.load(path.parent / manifest["preview"], allow_pickle=False, mmap_mode="r")
            if image.dtype != np.uint8 or list(image.shape) != manifest["shape"]:
                raise ValueError("Derived render array disagrees with its manifest.")
            if self._display_identity is None:
                raise ValueError("No registered reference geometry for derived view.")
            geometry = self._display_identity.source_fingerprint + self._display_identity.geometry
            self.preview.show_rgb8(image, geometry_key=geometry)
            self._display_dir = Path(payload["b_dir"])
            self.source_label.configure(text=f"Derived {manifest['domain']} {payload['mode']} "
                f"×{payload['difference_gain']} display-code visualization; native data unchanged.")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self._set_notice(f"render display failed: {exc}")
        finally:
            self._render_wanted = None

    def _on_picker(self, x: float, y: float, slot: str) -> None:
        dirs = self._pair_dirs if self.comparison.mode.get() in ("Wipe", "Difference") else {
            slot: self._pair_dirs.get(slot, self._display_dir)}
        labels = []
        for name, directory in dirs.items():
            if directory is None:
                continue
            info = artifact_pixel(directory, x, y, self._picker_cache)
            if info is not None:
                values = ", ".join(f"{key}={value:.7g}" for key, value in
                                   zip(info["component_names"], info["values"]))
                labels.append(f"{name}: ({info['x']}, {info['y']}) {values} "
                              f"{info['units']} | {info['domain']} {info['note']}")
        self.analysis.picker.set("\n".join(labels) or "Pointer is outside the native frame.")

    def _on_roi(self, bounds, slot: str) -> None:
        if bounds is None:
            self.analysis.clear_roi()
            return
        directory = self._pair_dirs.get(slot, self._display_dir)
        if directory is None:
            return
        _, metadata, _, _ = view_artifact(directory)
        other = next((value for key, value in self._pair_dirs.items() if key != slot), None)
        self.analysis.preview_geometry = (metadata["width"], metadata["height"])
        self.analysis.set_frame(str(directory), comparison_dir=str(other) if self._pair_registered and other else None)
        self.analysis.set_roi(bounds)

    def _apply_feedback(self, kind: str) -> None:
        manifest = self._frozen_manifest
        if not manifest:
            self._set_notice("no processed result")
            return
        tag = manifest.get("tag") or {}
        src = self.model.source or {}
        if (tag.get("session_id") != self.model.session_id
                or tag.get("generation") != self.model.generation
                or tag.get("revision") != self.model.revision
                or tag.get("source_fingerprint")
                != src.get("fingerprint")):
            self._set_notice("feedback is stale for current context")
            return
        from tools.gui.feedback import feedback_edits
        try:
            edits = feedback_edits(self.model.config, manifest.get("feedback") or {}, kind)
        except ValueError as exc:
            self._set_notice(str(exc))
            return
        self.model.edit_many(edits)
        self._on_edit()
        self._rebuild_forms()

    def _clear_display(self) -> None:
        self.preview.clear()
        self._display_dir = None
        self._display_identity = None
        self._frozen_manifest = None
        self._pair_dirs = {}
        self._pair_registered = False
        self._render_wanted = None
        self._picker_cache.clear()
        self.analysis.set_frame(None)
        self.analysis.roi = None
        self.analysis.roi_text.set("ROI: full frame")
        self.analysis.picker.set("Native picker: no current frame.")
        self.source_label.configure(text="No current preview — import a source or process this configuration.")

    def _show_source(self) -> None:
        source = self.model.source or {}
        self._clear_display()
        self._latest_dir = None
        self.comparison.mode.set("Current")
        manifest = source.get("view_manifest")
        if manifest:
            try:
                self._latest_dir = Path(manifest)
                self._show_artifact(manifest)
            except (ValueError, OSError, KeyError) as exc:
                self._set_notice(f"Source preview unavailable: {exc}")
        elif source.get("preview_error"):
            self._set_notice(source["preview_error"])

    def _set_notice(self, text: str) -> None:
        self._notice = str(text)
        self.log.warning("%s", text)

    def _on_edit(self) -> None:
        self.pipeline.clear_timings()
        if not self._source_ready():
            self.controller.auto_enabled = False
        self.controller.on_edit()
        operator = self.model.get("tone_mapping.operator")
        if operator != self._operator:
            self._operator = operator
            self.root.after_idle(self.params.refresh)
        self._refresh_status()

    def _flush_all(self) -> None:
        self.format_panel.flush_all()
        self.params.flush_all()
        self.model.end_edit_group()

    def _on_process(self) -> None:
        self._flush_all()
        self._notice = ""
        errors = [f"{issue.path}: {issue.message}" for issue in self.model.validation_issues()
                  if issue.severity == "error"]
        if errors:
            self._set_notice("Process blocked: " + "; ".join(errors[:3]))
            return
        if not self._source_ready():
            self._set_notice("Process blocked: import or reload a readable source file.")
            return
        captures = () if self.capture.get() == "none" else (self.capture.get(),)
        if not self.controller.request_process(captures):
            self._set_notice("Process blocked: import a source and repair validation errors.")

    def _on_cancel(self) -> None:
        self._render_wanted = None
        self.controller.cancel()
        self.view.auto_var.set(False)

    def _on_stop(self) -> None:
        self._render_wanted = None
        self.controller.stop_worker()
        self.view.auto_var.set(False)

    def _on_auto_toggle(self) -> None:
        self._flush_all()
        self.controller.auto_enabled = self.view.auto_var.get() and self._source_ready() and self.model.config_json_ready()
        if self.controller.auto_enabled:
            self.controller.on_edit()

    def _history(self, redo: bool) -> None:
        self.model.end_edit_group()
        generation = self.model.generation
        if (self.model.redo() if redo else self.model.undo()):
            if self.model.generation != generation:
                self.controller.context_changed()
            self._rebuild_forms()
            self._on_edit()

    def _on_undo(self) -> None:
        self._history(False)

    def _on_redo(self) -> None:
        self._history(True)

    def _rebuild_forms(self) -> None:
        self.format_panel.refresh()
        self.params.refresh()
        self.pipeline.refresh()
        self._revision = -1

    def _rebind_workspace(self, model, session, ui_state, snaps) -> None:
        self.controller.context_changed()
        self.controller.auto_enabled = False
        self.view.auto_var.set(False)
        self.model, self.session = model, session
        self.controller.model, self.controller.root_dir = model, session.directory.parent
        self.controller.current_result = None
        self._display_dir = None
        self._latest_dir = None
        self._pair_dirs = {}
        self._render_wanted = None
        self._picker_cache = {}
        self._frozen_manifest = None
        self._display_identity = None
        self.snapshots = snaps or {}
        self._generation = model.generation
        self._operator = model.get("tone_mapping.operator")
        for component in (self.format_panel, self.params, self.pipeline):
            component.model = model
        attach_session_log(session.directory)
        self._rebuild_forms()
        self._show_source()
        self._restore_ui(ui_state)

    def _restore_ui(self, ui) -> None:
        self.comparison.restore(self.snapshots)
        current = self.snapshots.get("current")
        if isinstance(current, dict) and current.get("manifest") and not current.get("missing"):
            try:
                self._select_result(current["manifest"])
            except (ValueError, OSError, KeyError) as exc:
                self._set_notice(f"Saved snapshot unavailable: {exc}")
        size = ui.get("window_size", "")
        parts = size.split("x") if isinstance(size, str) else []
        if len(parts) == 2 and all(part.isdigit() for part in parts):
            width, height = (int(part) for part in parts)
            self.root.geometry(f"{max(1024, min(width, self.root.winfo_screenwidth()))}x"
                               f"{max(640, min(height, self.root.winfo_screenheight()))}")
        comparison = ui.get("comparison", {})
        if isinstance(comparison, dict):
            fraction = comparison.get("fraction", .5)
            if isinstance(fraction, (int, float)) and not isinstance(fraction, bool) and math.isfinite(fraction) and 0 <= fraction <= 1:
                self.comparison.fraction.set(fraction)
            self.comparison.gain.set(str(comparison.get("gain", "1")))
            mode = comparison.get("mode", "Current")
            if mode in self.comparison.MODES and self._latest_dir is not None:
                self.comparison.mode.set(mode)
                self._comparison_changed()
        viewport = ui.get("viewport", {})
        if self._display_dir is not None and isinstance(viewport, dict):
            self.preview.restore_viewport(viewport)
        elif isinstance(viewport, dict):
            self._pending_viewport = viewport
        self.root.after_idle(lambda: self.view.restore_ui(ui.get("view", {})))

    def _confirm_unsaved(self) -> bool:
        self._flush_all()
        if not self.model.is_dirty:
            return True
        choice = messagebox.askyesnocancel("Unsaved changes", "Save this workspace before continuing?", parent=self.root)
        return False if choice is None else self._on_save_session() if choice else True

    def _on_new(self) -> None:
        if self._confirm_unsaved():
            session, model = WorkspaceSession.create(self.session_root, load_preset_yaml(DEFAULT_PRESET),
                                                      preset_path=str(DEFAULT_PRESET))
            self._rebind_workspace(model, session, {}, {})

    def _on_open(self) -> None:
        directory = filedialog.askdirectory(parent=self.root, initialdir=self.session_root)
        if not directory or not self._confirm_unsaved():
            return
        try:
            session, model, ui, snapshots, warnings = WorkspaceSession.open(Path(directory))
            self._rebind_workspace(model, session, ui, snapshots)
            if warnings:
                messagebox.showwarning("Workspace warnings", "\n".join(warnings), parent=self.root)
        except (ValueError, OSError) as exc:
            messagebox.showerror("Open workspace", str(exc), parent=self.root)

    def _on_import(self) -> None:
        path = filedialog.askopenfilename(parent=self.root)
        if path:
            self._flush_all()
            self.controller.context_changed()
            self.controller.request_import(path)

    def _on_save_session(self) -> bool:
        self._flush_all()
        try:
            self.snapshots = dict(self.comparison.pins)
            self.snapshots["current"] = {"manifest": str(self._display_dir), "stage": "preview"} if self._display_dir else None
            ui_state = {"viewport": self.preview.viewport(),
                        "view": self.view.ui_state(),
                        "window_size": self.root.geometry().split("+")[0],
                        "comparison": {
                            "mode": self.comparison.mode.get(),
                            "fraction": self.comparison.fraction.get(),
                            "gain": self.comparison.gain.get()}}
            self.session.save(self.model, ui_state, self.snapshots)
        except (ValueError, OSError) as exc:
            messagebox.showerror("Save workspace", str(exc), parent=self.root)
            return False
        try:
            self.session.save_last_pointer(self.session_root)
        except (ValueError, OSError) as exc:
            self._set_notice(f"Workspace saved; recovery pointer failed: {exc}")
        return True

    def _load_preset(self) -> None:
        path = filedialog.askopenfilename(parent=self.root, filetypes=[("YAML", "*.yml *.yaml")])
        if path and self._confirm_unsaved():
            try:
                config = load_preset_yaml(path)
                self.controller.context_changed()
                self.model.replace_config(config, preset_path=path, as_preset=True, label="Load preset")
                self._rebuild_forms()
                self._on_edit()
            except (ValueError, OSError) as exc:
                self._set_notice(str(exc))

    def _export_yaml(self) -> None:
        self._flush_all()
        if not self.model.config_json_ready():
            self._set_notice("Repair invalid editor values before exporting YAML.")
            return
        path = filedialog.asksaveasfilename(parent=self.root, defaultextension=".yml")
        if path:
            try:
                if Path(path).exists() and not messagebox.askyesno("Overwrite preset", f"Overwrite {path}?", parent=self.root):
                    return
                dump_config_yaml(self.model.config, path)
            except OSError as exc:
                self._set_notice(str(exc))

    def _upgrade(self) -> None:
        self._flush_all()
        if not self.model.config_json_ready():
            self._set_notice("Repair invalid editor values before upgrading the configuration.")
            return
        from util.pipeline_migration import migrate_pipeline_config
        try:
            config, report = migrate_pipeline_config(self.model.config)
            if messagebox.askyesno("Upgrade to native v2", "Apply explicit native migration as one undoable edit?\n"
                                  "The loaded preset file will not be overwritten.", parent=self.root):
                self.controller.context_changed()
                self.model.replace_config(config, label="Upgrade to native v2")
                self._rebuild_forms()
                self._on_edit()
                self.log.info("Migration report: %s", json.dumps(report))
        except ValueError as exc:
            self._set_notice(str(exc))

    def _reset_all(self) -> None:
        self._flush_all()
        if not messagebox.askyesno("Reset draft", "Reset all parameters to the loaded preset as one undoable edit?", parent=self.root):
            return
        self.model.reset_all()
        self._rebuild_forms()
        self._on_edit()

    def _on_export(self) -> None:
        self._flush_all()
        if not self.model.config_json_ready():
            self._set_notice("Repair invalid editor values before exporting.")
            return
        directory = self._display_dir
        if self.comparison.mode.get() in ("Side by side", "Wipe", "Difference"):
            slot = simpledialog.askstring("Native export", "Export original native result A or B?", initialvalue="B", parent=self.root)
            if slot is None:
                return
            directory = self._pair_dirs.get(slot.strip().upper())
        if directory is None:
            self._set_notice("No native processed result to export.")
            return
        manifest = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        if manifest.get("kind") == "raw":
            self._set_notice("Select a processed result, not the stretched RAW visualization, for native export.")
            return
        destination = filedialog.askdirectory(parent=self.root)
        if destination:
            stem = simpledialog.askstring("Native export", "Output stem (original result specification):",
                                          initialvalue="output", parent=self.root)
            if stem:
                self.controller.request_export(directory, destination, stem)

    def _source_ready(self) -> bool:
        source = self.model.source
        return bool(source and not source.get("missing") and Path(source.get("path", "")).is_file())

    def _refresh_status(self) -> None:
        if self._closing:
            return
        if self._revision != self.model.revision:
            self._revision = self.model.revision
            issues = self.model.validation_issues()
            self.params.show_issues(issues)
            fmt_rows = getattr(self.format_panel, "rows", {})
            for path, row in fmt_rows.items():
                if hasattr(row, "error_var"):
                    match = next((i for i in issues
                                  if i.path == path
                                  or i.path.startswith(path + ".")),
                                 None)
                    row.error_var.set(
                        f"{match.path}: {match.message}" if match else "")
            self.format_panel.update_readouts()
            self.pipeline.refresh()
            errors = [f"{i.path}: {i.message}" for i in issues if i.severity == "error"]
            self._validation_text = "; ".join(errors[:2])
        active = self.controller.active
        stage = ""
        if active is not None and self.controller.progress:
            for p in reversed(self.controller.progress):
                if p.get("request_id") == active.tag.request_id:
                    stage = p.get("stage_id", "")
                    break
        status = self._notice or getattr(self, "_validation_text", "") or self.controller.status
        self.view.status_var.set(f"{status} {stage}"[:320])
        parts = ["unsaved" if self.model.is_dirty else "saved", f"{len(self.model.modified_paths)} edited"]
        if self.controller.current_result and self.controller.current_result.get("stale"):
            parts.append("preview stale")
        self.view.dirty_var.set(" | ".join(parts))
        self.view.auto_var.set(self.controller.auto_enabled)
        self.view.set_busy(self.controller.active is not None)
        valid = not getattr(self, "_validation_text", "")
        for name, enabled in (
            ("process", valid and self._source_ready()), ("save_session", valid),
            ("export", valid and self._display_dir is not None and bool(self._frozen_manifest) and self._frozen_manifest.get("kind") != "raw"),
            ("undo", bool(self.model._undo)), ("redo", bool(self.model._redo))):
            self.view.buttons[name].configure(state="normal" if enabled else "disabled")

    def _on_close(self) -> None:
        if self._closing or not self._confirm_unsaved():
            return
        if self._poll_after_id is not None:
            try:
                self.root.after_cancel(self._poll_after_id)
            except Exception:
                pass
            self._poll_after_id = None
        self._closing = True
        self.controller.close()
        self._shutdown_step()

    def _shutdown_step(self) -> None:
        self.manager.shutdown_step()
        if self.manager.shutdown_complete:
            self.root.destroy()
        else:
            self._shutdown_after_id = self.root.after(
                50, self._shutdown_step)

    def _root_destroyed(self, event) -> None:
        if event.widget is not self.root:
            return
        self._closing = True
        for attr in ("_poll_after_id", "_shutdown_after_id"):
            aid = getattr(self, attr)
            if aid is not None:
                try:
                    self.root.after_cancel(aid)
                except tk.TclError:
                    pass
                setattr(self, attr, None)
