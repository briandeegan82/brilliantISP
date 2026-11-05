"""Integrated workspace layout: toolbar + paned notebook/preview/editor."""
from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk
from typing import Callable


class WorkspaceView(ttk.Frame):
    TOOLBAR = (
        ("process", "Process"), ("cancel", "Cancel"),
        ("stop", "Stop worker"), ("import", "Import…"),
        ("open", "Open…"), ("new", "New"),
        ("save_session", "Save session"), ("export", "Export…"),
        ("undo", "Undo"), ("redo", "Redo"))

    def __init__(self, parent, handlers: dict) -> None:
        super().__init__(parent)
        self.handlers = handlers
        self.buttons: dict[str, ttk.Button] = {}
        self._busy = False
        self._initial_layout_done = False
        self._pending_sashes: tuple[int, int] | None = None
        self._build_toolbar()
        self._build_body()
        self._build_status()
        self.paned.bind("<Map>", self._mapped_once)
        self.paned.bind("<Configure>", self._mapped_once)

    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self)
        bar.pack(fill="x")
        for key, label in self.TOOLBAR:
            btn = ttk.Button(bar, text=label, command=self.handlers[key])
            btn.pack(side="left", padx=1)
            self.buttons[key] = btn
        self.auto_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            bar, text="Auto", variable=self.auto_var,
            command=self.handlers["auto_toggle"]).pack(side="left", padx=6)

    def _build_body(self) -> None:
        self.paned = ttk.PanedWindow(self, orient="horizontal")
        self.paned.pack(fill="both", expand=True)
        left = ttk.Frame(self.paned, width=260)
        self.left_notebook = ttk.Notebook(left)
        self.left_notebook.pack(fill="both", expand=True)
        self.format_tab = ttk.Frame(self.left_notebook)
        self.left_notebook.add(self.format_tab, text="Format")
        self.pipeline_tab = ttk.Frame(self.left_notebook)
        self.left_notebook.add(self.pipeline_tab, text="Pipeline")
        self.paned.add(left, weight=0)
        self.preview_frame = ttk.Frame(self.paned)
        self.paned.add(self.preview_frame, weight=1)
        right = ttk.Frame(self.paned, width=300)
        self.inspector_notebook = ttk.Notebook(right)
        self.editor_frame = ttk.Frame(self.inspector_notebook)
        self.inspector_notebook.add(self.editor_frame, text="Parameters")
        self.analysis_frame = ttk.Frame(self.inspector_notebook)
        self.inspector_notebook.add(self.analysis_frame, text="Analysis")
        self.feedback_frame = ttk.Frame(self.inspector_notebook)
        self.inspector_notebook.add(self.feedback_frame, text="Feedback")
        self.jobs_frame = ttk.Frame(self.inspector_notebook)
        self.inspector_notebook.add(self.jobs_frame, text="Jobs")
        self.inspector_notebook.pack(fill="both", expand=True)
        self.paned.add(right, weight=0)

    def _build_status(self) -> None:
        bar = ttk.Frame(self)
        bar.pack(fill="x", side="bottom")
        self.status_var = tk.StringVar(value="idle")
        ttk.Label(bar, textvariable=self.status_var).pack(side="left", padx=4)
        self.dirty_var = tk.StringVar(value="")
        ttk.Label(bar, textvariable=self.dirty_var).pack(side="right", padx=4)
        self.progress = ttk.Progressbar(bar, length=140, mode="indeterminate")
        self.progress.pack(side="right", padx=4)

    def set_busy(self, busy: bool) -> None:
        if busy and not self._busy:
            self.progress.start(50)
        elif not busy and self._busy:
            self.progress.stop()
        self._busy = busy

    def ui_state(self) -> dict:
        return {"sashes": list(self._sash_positions())}

    def _sash_positions(self) -> list[int]:
        try:
            return [self.paned.sashpos(i) for i in (0, 1)]
        except Exception:
            return []

    def restore_ui(self, state: dict) -> None:
        sashes = state.get("sashes") if isinstance(state, dict) else None
        if not isinstance(sashes, (list, tuple)) or len(sashes) != 2:
            return
        p0, p1 = sashes
        for p in sashes:
            if not isinstance(p, (int, float)) or isinstance(p, bool):
                return
            if isinstance(p, float) and not math.isfinite(p):
                return
        self._pending_sashes = (int(p0), int(p1))
        self._mapped_once()

    def reset_layout(self) -> None:
        try:
            w = self.paned.winfo_width()
            self.paned.sashpos(1, w - 300)
            self.paned.sashpos(0, 260)
        except Exception:
            pass

    def _mapped_once(self, _event=None) -> None:
        if not self.paned.winfo_ismapped() \
                or self.paned.winfo_width() < 1024:
            return
        w = self.paned.winfo_width()
        if self._pending_sashes is not None:
            p0, p1 = self._pending_sashes
            left = min(max(p0, 180), w - 500)
            right = min(max(p1, left + 300), w - 220)
            try:
                self.paned.sashpos(1, right)
                self.paned.sashpos(0, left)
            except Exception:
                pass
            self._pending_sashes = None
            self._initial_layout_done = True
        elif not self._initial_layout_done:
            self.reset_layout()
            self._initial_layout_done = True


class PipelineTree(ttk.Frame):
    STAGES = (
        ("crop", "crop", "sensor Bayer"),
        ("dead_pixel_correction", "dead_pixel_correction", "sensor Bayer"),
        ("black_level_correction", "black_level_correction", "sensor Bayer"),
        ("green_repair", "sensor_info", "sensor Bayer"),
        ("companding", "companding", "sensor→linear Bayer"),
        ("oecf", "oecf", "linear Bayer"),
        ("digital_gain", "digital_gain", "linear Bayer"),
        ("lens_shading_correction", "lens_shading_correction", "linear Bayer"),
        ("bayer_noise_reduction", "bayer_noise_reduction", "linear Bayer"),
        ("auto_white_balance", "auto_white_balance", "linear Bayer"),
        ("white_balance", "white_balance", "linear Bayer"),
        ("demosaic", "demosaic", "linear Bayer→RGB"),
        ("color_correction_matrix", "color_correction_matrix",
         "camera-linear RGB"),
        ("linear_rgb_pre_tone_mapping", "linear_rgb_pre_tone_mapping",
         "scene-linear RGB (tap, read-only)"),
        ("auto_exposure", "auto_exposure", "metering"),
    )
    TAIL = (
        ("color_saturation_enhancement", "color_saturation_enhancement",
         "display-linear RGB"),
        ("ldci_pre_transfer", "ldci", "luma (pre-transfer placement)"),
        ("transfer_encode", "output", "output-transfer RGB"),
        ("ldci_post_transfer", "ldci", "luma (post-transfer placement)"),
        ("sharpen", "sharpen", "output-transfer RGB"),
        ("noise_reduction_2d", "2d_noise_reduction",
         "output-transfer RGB"),
        ("scale", "scale", "output-transfer RGB"),
        ("output", "output", "output"))
    TMO_NODE = "tone_mapping"
    RENDER_EXTRAS = (
        ("demosaic_render", "demosaic", "display-linear Bayer"),
        ("color_correction_matrix_render", "color_correction_matrix",
         "display-linear RGB"))
    ADVANCED = "advanced"

    def __init__(self, parent, model, on_select: Callable[[str], None]) -> None:
        super().__init__(parent)
        self.model = model
        self.on_select = on_select
        self._elapsed: dict[str, int] = {}
        cols = ("state", "domain", "ms")
        self.tree = ttk.Treeview(self, columns=cols, height=18)
        self.tree.heading("#0", text="Stage")
        self.tree.column("#0", width=170)
        for c, w in (("state", 66), ("domain", 132), ("ms", 52)):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, stretch=False)
        vscroll = ttk.Scrollbar(self, orient="vertical",
                                command=self.tree.yview)
        hscroll = ttk.Scrollbar(self, orient="horizontal",
                                command=self.tree.xview)
        self.tree.configure(yscrollcommand=vscroll.set,
                            xscrollcommand=hscroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vscroll.pack(side="right", fill="y")
        hscroll.pack(side="bottom", fill="x")
        self.caption_var = tk.StringVar(
            value="Total recorded completed stage time across passes")
        ttk.Label(self, textvariable=self.caption_var).pack(
            side="bottom", anchor="w")
        self.tree.bind("<<TreeviewSelect>>", self._selected)
        self.refresh()

    def _ldci_pre(self, cfg: dict) -> bool:
        ldci = cfg.get("ldci", {})
        return isinstance(ldci, dict) and ldci.get(
            "placement") == "pre_transfer"

    def _state(self, cfg_key: str, cfg: dict,
               iid: str | None = None) -> str:
        if iid == "green_repair":
            return "runtime"
        if cfg_key is None or cfg_key not in cfg:
            return "tap"
        block = cfg.get(cfg_key)
        if not isinstance(block, dict):
            return "INVALID"
        enabled = block.get("is_enable", True)
        if isinstance(enabled, bool):
            return "on" if enabled else "off"
        return "INVALID"

    def _ldci_state(self, which: str, cfg: dict) -> str:
        if "ldci" not in cfg:
            return "tap"
        enabled = self._state("ldci", cfg)
        if enabled in ("INVALID",):
            return enabled
        active = (self._ldci_pre(cfg) if which == "pre"
                  else not self._ldci_pre(cfg))
        if enabled == "off":
            return "off"
        return "on" if active else "inactive"

    def _order(self, cfg: dict) -> list[tuple[str, str | None, str]]:
        tmo = cfg.get("tone_mapping", {})
        tmo_enabled = isinstance(tmo, dict) and tmo.get("is_enable") is True
        pre = tmo_enabled and tmo.get("tone_mapping_before_demosaic") is True
        rows: list[tuple[str, str | None, str]] = list(self.STAGES)
        tmo_iid = "tone_mapping_pre" if pre else (
            "tone_mapping_post" if tmo_enabled else "tone_mapping")
        domain = ("linear Bayer→display-linear Bayer" if pre
                  else "scene-linear RGB→display")
        rows.append((tmo_iid, "tone_mapping", domain))
        if pre:
            rows.extend(self.RENDER_EXTRAS)
        rows.extend(self.TAIL)
        return rows

    def _state_of(self, iid: str, cfg_key: str | None, cfg: dict) -> str:
        if iid in ("ldci_pre_transfer", "ldci_post_transfer"):
            return self._ldci_state(
                "pre" if iid == "ldci_pre_transfer" else "post", cfg)
        if cfg_key is None:
            return "tap"
        return self._state(cfg_key, cfg, iid)

    def refresh(self) -> None:
        cfg = self.model.config
        mod = {p.split(".")[0] for p in self.model.modified_paths}
        known = set(cfg)
        ordered = self._order(cfg)
        wanted = [i for i, _k, _d in ordered]
        existing = list(self.tree.get_children())
        for iid, cfg_key, domain in ordered:
            mark = " *" if cfg_key and cfg_key in mod else ""
            label = iid + mark
            state = self._state_of(iid, cfg_key, cfg)
            ms = (f"{self._elapsed[iid] / 1e6:.2f}"
                  if iid in self._elapsed else "")
            if self.tree.exists(iid):
                self.tree.item(iid, text=label,
                               values=(state, domain, ms))
            else:
                self.tree.insert("", "end", iid=iid, text=label,
                                 values=(state, domain, ms))
        ordered_keys = {k for _i, k, _d in ordered if k}
        advanced = [s for s in cfg
                    if s not in ordered_keys
                    and isinstance(cfg[s], dict)]
        legacy_only = {"gamma", "gamma_correction", "color_space_conversion",
                       "rgb_conversion", "yuv_conversion_format"}
        adv_wanted = {f"adv:{s}" for s in advanced}
        for iid in existing:
            if iid == self.ADVANCED:
                continue
            if iid not in wanted and iid not in adv_wanted                     and not iid.startswith("adv:"):
                self.tree.delete(iid)
        for i, iid in enumerate(wanted):
            self.tree.move(iid, "", i)
        if not self.tree.exists(self.ADVANCED):
            self.tree.insert("", "end", iid=self.ADVANCED,
                             text="Advanced")
        self.tree.move(self.ADVANCED, "", len(wanted))
        for i, section in enumerate(advanced):
            aid = f"adv:{section}"
            label = section + (" *" if section in mod else "")
            state = self._state(section, cfg)
            domain = ("migration only—not native"
                      if section in legacy_only else "config")
            if self.tree.exists(aid):
                self.tree.item(aid, text=label,
                               values=(state, domain, ""))
            else:
                self.tree.insert(self.ADVANCED, "end", iid=aid,
                                 text=label,
                                 values=(state, domain, ""))
            self.tree.move(aid, self.ADVANCED, i)
        for child in self.tree.get_children(self.ADVANCED):
            if child not in adv_wanted:
                self.tree.delete(child)

    def set_timings(self, timings) -> None:
        self._elapsed = {}
        for t in timings:
            ns = t.get("elapsed_ns")
            if isinstance(ns, int) and not isinstance(ns, bool) and ns >= 0:
                sid = t["stage_id"]
                self._elapsed[sid] = self._elapsed.get(sid, 0) + ns
        self.refresh()

    def clear_timings(self) -> None:
        self._elapsed = {}
        self.refresh()

    def _editor_key(self, iid: str) -> str:
        if iid.startswith("adv:"):
            return iid[4:]
        if iid in ("tone_mapping_pre", "tone_mapping_post",
                   "tone_mapping"):
            return "tone_mapping"
        if iid == "demosaic_render":
            return "demosaic"
        if iid == "color_correction_matrix_render":
            return "color_correction_matrix"
        if iid == "green_repair":
            return "sensor_info"
        if iid in ("ldci_pre_transfer", "ldci_post_transfer"):
            return "ldci"
        if iid == "transfer_encode":
            return "output"
        if iid == "linear_rgb_pre_tone_mapping":
            return "linear_rgb_pre_tone_mapping"
        if iid == "noise_reduction_2d":
            return "2d_noise_reduction"
        return iid

    def _selected(self, _e) -> None:
        sel = self.tree.selection()
        if sel and sel[0] != self.ADVANCED:
            self.on_select(self._editor_key(sel[0]))
