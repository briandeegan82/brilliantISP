#!/usr/bin/env python3
"""
BrilliantISP calibration GUI — Image Format + ISP Configuration windows.

Run from repo root:

    python tools/isp_calibration_gui.py
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "TkAgg")

import cv2
import numpy as np
import yaml
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from tools.gui.computed import compute_format_stats
from tools.gui.format_map import (
    bayer_from_rggb_start,
    margins_from_crop,
    rggb_start_from_bayer,
    update_crop_from_margins,
)
from tools.gui.gui_log import (
    attach_session_log,
    diff_dicts,
    get_gui_logger,
    install_exception_hooks,
    setup_gui_logging,
    truncate,
)
from tools.gui.process import prepare_process
from tools.gui.raw_preview import preview_from_config
from tools.gui.session import (
    DEFAULT_PRESET,
    DEFAULT_SESSION_ROOT,
    GuiSession,
    recover_last_session,
)
from tools.gui.state import SharedConfig
from tools.gui.ui_schema import (
    CFA_COLORS,
    CFA_TILES,
    ENUM_CHOICES,
    ISP_SECTION_ORDER,
    RGGB_START_LABELS,
)
from tools.gui.validation import ValidationIssue
from tools.gui.yaml_compat import dump_config_yaml, load_preset_yaml, merge_pipeline_feedback, yaml_safe_value


def _to_int(text: str, default: int = 0) -> int:
    try:
        return int(float(text.strip()))
    except (TypeError, ValueError):
        return default


def _to_float(text: str, default: float = 0.0) -> float:
    try:
        return float(text.strip())
    except (TypeError, ValueError):
        return default


class PreviewPane:
    def __init__(self, parent: Any, tk: Any) -> None:
        self.tk = tk
        self._image_artist = None
        self._image_width = 0
        self._image_height = 0
        self._zoom_factor = 1.2
        self._pan_start: tuple[float, float, tuple[float, float], tuple[float, float]] | None = None
        self.last_rgb: np.ndarray | None = None

        fig = Figure(figsize=(6, 4), dpi=100)
        fig.patch.set_facecolor("#f0f0f0")
        self._ax = fig.add_axes((0.01, 0.01, 0.98, 0.98))
        self._ax.axis("off")
        self._ax.set_facecolor("#e8e8e8")
        self._canvas = FigureCanvasTkAgg(fig, master=parent)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._canvas.mpl_connect("scroll_event", self._on_scroll)
        self._canvas.mpl_connect("button_press_event", self._on_press)
        self._canvas.mpl_connect("button_release_event", self._on_release)
        self._canvas.mpl_connect("motion_notify_event", self._on_motion)

    def show_rgb(self, rgb: np.ndarray) -> None:
        display = np.clip(np.asarray(rgb), 0, 255).astype(np.uint8)
        self.last_rgb = display.copy()
        self._ax.clear()
        self._ax.axis("off")
        self._image_height, self._image_width = display.shape[:2]
        if display.ndim == 2:
            self._image_artist = self._ax.imshow(display, cmap="gray", vmin=0, vmax=255)
        else:
            self._image_artist = self._ax.imshow(display)
        self.fit()

    def refresh_from_path(self, path: Path) -> bool:
        if not path.is_file():
            return False
        bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if bgr is None:
            return False
        if bgr.ndim == 3 and bgr.shape[2] == 3:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            rgb = bgr
        self.show_rgb(rgb)
        return True

    def fit(self) -> None:
        if self._image_width <= 0:
            return
        self._ax.set_xlim(-0.5, self._image_width - 0.5)
        self._ax.set_ylim(self._image_height - 0.5, -0.5)
        self._canvas.draw_idle()

    def one_to_one(self) -> None:
        if self._image_width <= 0:
            return
        widget = self._canvas.get_tk_widget()
        vw = max(widget.winfo_width(), 1)
        vh = max(widget.winfo_height(), 1)
        cx, cy = self._image_width / 2, self._image_height / 2
        self._ax.set_xlim(cx - vw / 2, cx + vw / 2)
        self._ax.set_ylim(cy + vh / 2, cy - vh / 2)
        self._canvas.draw_idle()

    def zoom(self, scale: float) -> None:
        if self._image_artist is None or self._image_width <= 0:
            return
        x0, x1 = self._ax.get_xlim()
        y0, y1 = self._ax.get_ylim()
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        nx0 = cx - (cx - x0) * scale
        nx1 = cx + (x1 - cx) * scale
        ny0 = cy - (cy - y0) * scale
        ny1 = cy + (y1 - cy) * scale
        self._ax.set_xlim(*self._clamp(nx0, nx1, self._image_width))
        self._ax.set_ylim(*self._clamp(ny0, ny1, self._image_height))
        self._canvas.draw_idle()

    def _clamp(self, a: float, b: float, size: int) -> tuple[float, float]:
        lo, hi = (a, b) if a < b else (b, a)
        span = hi - lo
        if span >= size:
            return (-0.5, size - 0.5) if a < b else (size - 0.5, -0.5)
        if lo < -0.5:
            lo, hi = -0.5, -0.5 + span
        if hi > size - 0.5:
            hi, lo = size - 0.5, size - 0.5 - span
        return (lo, hi) if a < b else (hi, lo)

    def _on_scroll(self, event: Any) -> None:
        if event.inaxes != self._ax or event.xdata is None:
            return
        scale = 1.0 / self._zoom_factor if event.button == "up" else self._zoom_factor
        x0, x1 = self._ax.get_xlim()
        y0, y1 = self._ax.get_ylim()
        nx0 = event.xdata - (event.xdata - x0) * scale
        nx1 = event.xdata + (x1 - event.xdata) * scale
        ny0 = event.ydata - (event.ydata - y0) * scale
        ny1 = event.ydata + (y1 - event.ydata) * scale
        self._ax.set_xlim(*self._clamp(nx0, nx1, self._image_width))
        self._ax.set_ylim(*self._clamp(ny0, ny1, self._image_height))
        self._canvas.draw_idle()

    def _on_press(self, event: Any) -> None:
        if event.inaxes != self._ax or event.button != 1:
            return
        self._pan_start = (event.x, event.y, self._ax.get_xlim(), self._ax.get_ylim())

    def _on_release(self, _event: Any) -> None:
        self._pan_start = None

    def _on_motion(self, event: Any) -> None:
        if self._pan_start is None or event.inaxes != self._ax:
            return
        sx, sy, xlim, ylim = self._pan_start
        bbox = self._ax.get_window_extent()
        if bbox.width == 0 or bbox.height == 0:
            return
        dx = (event.x - sx) / bbox.width * (xlim[1] - xlim[0])
        dy = (event.y - sy) / bbox.height * (ylim[1] - ylim[0])
        self._ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
        self._ax.set_ylim(ylim[0] - dy, ylim[1] - dy)
        self._canvas.draw_idle()


class CalibrationApp:
    def __init__(
        self,
        root: Any,
        *,
        preset: Path | None,
        recover: bool,
        initial_raw: Path | None = None,
    ) -> None:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk

        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.root = root
        self._processing = False
        self._highlight: dict[str, Any] = {}
        self._isp_vars: dict[str, Any] = {}
        self._show_modified_only = tk.BooleanVar(value=False)
        self._syncing = False
        self._format_log_after: str | None = None
        self._last_format_snapshot: dict[str, Any] = {}
        self.log = setup_gui_logging()
        self.log.info("Calibration GUI starting (preset=%s recover=%s raw=%s)", preset, recover, initial_raw)

        self.session, cfg = GuiSession.create_new(preset_path=preset or DEFAULT_PRESET)
        self.state = SharedConfig(cfg)
        self.state.preset_path = str(preset or DEFAULT_PRESET)

        if recover:
            recovered = recover_last_session()
            if recovered is not None:
                rec_session, rec_cfg = recovered
                if messagebox.askyesno(
                    "Recover session",
                    f"Restore previous session {rec_session.session_id}?",
                ):
                    self.session = rec_session
                    self.state.load_from(rec_cfg, preset_path=rec_session.ui_state.get("last_preset_path"))
                    self.log.info("Recovered session %s", rec_session.session_id)

        attach_session_log(self.session.paths.directory)
        install_exception_hooks(
            tk_root=root,
            on_error=lambda text: self._on_logged_exception(text),
        )
        self.log.info("Session %s at %s", self.session.session_id, self.session.paths.directory)

        root.title("ISP calibration tool | Image format")
        root.minsize(1100, 720)
        root.geometry("1280x800")
        self._build_menu(root)
        self._build_format_window(root)

        self.isp_win = tk.Toplevel(root)
        self.isp_win.title("ISP calibration tool | ISP configuration")
        self.isp_win.minsize(520, 640)
        self.isp_win.geometry("640x800")
        self._build_menu(self.isp_win)
        self._build_isp_window(self.isp_win)
        self.isp_win.protocol("WM_DELETE_WINDOW", self._hide_isp)

        self._load_format_from_state()
        self._rebuild_isp_form()
        self._update_computed()
        self._set_status(f"Session {self.session.session_id}")
        if self.session.paths.output_png.is_file():
            self.preview.refresh_from_path(self.session.paths.output_png)
        elif self.session.has_raw:
            self._show_loaded_raw_preview()
        if initial_raw is not None:
            self._load_raw_path(Path(initial_raw))

    def _hide_isp(self) -> None:
        self.isp_win.withdraw()

    def _show_isp(self) -> None:
        self.isp_win.deiconify()
        self.isp_win.lift()

    def _build_menu(self, win: Any) -> None:
        menubar = self.tk.Menu(win)
        win.config(menu=menubar)
        file_m = self.tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_m)
        file_m.add_command(label="New session", command=self._new_session)
        file_m.add_command(label="Open session…", command=self._open_session)
        file_m.add_command(label="Save session", command=self._save_session)
        file_m.add_separator()
        file_m.add_command(label="Open preset YAML…", command=self._open_preset)
        file_m.add_command(label="Export YAML…", command=self._export_yaml)
        file_m.add_separator()
        file_m.add_command(label="Process", command=self.process)
        file_m.add_separator()
        file_m.add_command(label="Exit", command=self.root.quit)

        view_m = self.tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_m)
        view_m.add_command(label="ISP configuration window", command=self._show_isp)
        view_m.add_checkbutton(
            label="Show modified only",
            variable=self._show_modified_only,
            command=self._rebuild_isp_form,
        )

        edit_m = self.tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_m)
        edit_m.add_command(label="Reset all", command=self._reset_all)

        help_m = self.tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_m)
        help_m.add_command(
            label="About…",
            command=lambda: self.messagebox.showinfo(
                "About",
                "BrilliantISP calibration GUI\nSee docs/GUI_DESIGN.md",
            ),
        )
        help_m.add_command(label="Open log folder…", command=self._open_log_folder)

    def _build_format_window(self, root: Any) -> None:
        tk, ttk = self.tk, self.ttk
        top = ttk.Frame(root, padding=8)
        top.pack(fill=tk.X)
        ttk.Label(top, text="IMAGE FORMAT", font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT)
        ttk.Button(top, text="Save session", command=self._save_session).pack(side=tk.RIGHT, padx=4)
        ttk.Button(top, text="ISP config…", command=self._show_isp).pack(side=tk.RIGHT)

        body = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        left = ttk.Frame(body)
        body.add(left, weight=0)
        right = ttk.Frame(body)
        body.add(right, weight=1)

        geo = ttk.LabelFrame(left, text="Sensor / geometry", padding=8)
        geo.pack(fill=tk.X, pady=4)
        self.var_sensor = tk.StringVar()
        self.var_workplace = tk.StringVar(value=str(self.session.paths.directory))
        self.var_width = tk.StringVar()
        self.var_height = tk.StringVar()
        self.var_ml = tk.StringVar(value="0")
        self.var_mr = tk.StringVar(value="0")
        self.var_mt = tk.StringVar(value="0")
        self.var_mb = tk.StringVar(value="0")
        self.var_flip_h = tk.BooleanVar(value=False)
        self.var_flip_v = tk.BooleanVar(value=False)
        self.var_eff_w = tk.StringVar()
        self.var_eff_h = tk.StringVar()
        self.var_rggb = tk.StringVar(value=RGGB_START_LABELS[0])
        self._row(geo, 0, "Sensor name", ttk.Entry(geo, textvariable=self.var_sensor, width=28))
        self.var_raw_path = tk.StringVar(value="")
        raw_fr = ttk.Frame(geo)
        ttk.Entry(raw_fr, textvariable=self.var_raw_path, width=22, state="readonly").pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(raw_fr, text="browse", command=self._load_image).pack(side=tk.LEFT, padx=4)
        self._row(geo, 1, "RAW image", raw_fr)
        path_fr = ttk.Frame(geo)
        ttk.Entry(path_fr, textvariable=self.var_workplace, width=22, state="readonly").pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        self._row(geo, 2, "Session folder", path_fr)
        wh = ttk.Frame(geo)
        ttk.Entry(wh, textvariable=self.var_width, width=8).pack(side=tk.LEFT)
        ttk.Entry(wh, textvariable=self.var_height, width=8).pack(side=tk.LEFT, padx=4)
        self._row(geo, 3, "Sensor width/height", wh)
        margins = ttk.Frame(geo)
        for v in (self.var_ml, self.var_mr, self.var_mt, self.var_mb):
            ttk.Entry(margins, textvariable=v, width=5).pack(side=tk.LEFT, padx=1)
        self._row(geo, 4, "Crop margins L-R-T-B", margins)
        flip = ttk.Frame(geo)
        ttk.Checkbutton(flip, text="H", variable=self.var_flip_h, command=self._on_flip_changed).pack(side=tk.LEFT)
        ttk.Checkbutton(flip, text="V", variable=self.var_flip_v, command=self._on_flip_changed).pack(side=tk.LEFT)
        self._row(geo, 5, "Flip", flip)
        eff = ttk.Frame(geo)
        ttk.Entry(eff, textvariable=self.var_eff_w, width=8, state="readonly").pack(side=tk.LEFT)
        ttk.Entry(eff, textvariable=self.var_eff_h, width=8, state="readonly").pack(side=tk.LEFT, padx=4)
        self._row(geo, 6, "Effective width/height", eff)
        ttk.Combobox(
            geo,
            textvariable=self.var_rggb,
            values=RGGB_START_LABELS,
            state="readonly",
            width=28,
        ).grid(row=7, column=1, sticky="w", pady=2)
        ttk.Label(geo, text="rggb_start").grid(row=7, column=0, sticky="e", padx=(0, 8))
        self.cfa_canvas = tk.Canvas(geo, width=120, height=120, highlightthickness=1, highlightbackground="#888")
        self.cfa_canvas.grid(row=8, column=1, sticky="w", pady=6)
        ttk.Label(geo, text="CFA").grid(row=8, column=0, sticky="ne", padx=(0, 8), pady=6)

        rawf = ttk.LabelFrame(left, text="RAW file format", padding=8)
        rawf.pack(fill=tk.X, pady=4)
        self.var_header = tk.StringVar(value="0")
        self.var_bit = tk.StringVar()
        self.var_fmt = tk.StringVar()
        self.var_align = tk.StringVar(value="MSB")
        self.var_manual_shift = tk.StringVar(value="4")
        self.var_endian = tk.StringVar()
        self.var_hdr_bits = tk.StringVar()
        self.var_pipe_bits = tk.StringVar()
        self.var_out_bits = tk.StringVar()
        self._row(rawf, 0, "Header size (bytes)", ttk.Entry(rawf, textvariable=self.var_header, width=12))
        self._row(rawf, 1, "Raw bit-depth", ttk.Entry(rawf, textvariable=self.var_bit, width=12))
        self._row(
            rawf,
            2,
            "Data format",
            ttk.Combobox(
                rawf,
                textvariable=self.var_fmt,
                values=("uint8", "uint16", "uint32"),
                width=14,
            ),
        )
        self._row(
            rawf,
            3,
            "Data alignment",
            ttk.Combobox(rawf, textvariable=self.var_align, values=("MSB", "LSB", "MANUAL"), width=14, state="readonly"),
        )
        # Manual bit-shift control (only shown when MANUAL mode is selected)
        manual_shift_frame = ttk.Frame(rawf)
        ttk.Label(manual_shift_frame, text="    Shift right by:").pack(side=tk.LEFT)
        self.manual_shift_spinbox = ttk.Spinbox(
            manual_shift_frame, textvariable=self.var_manual_shift, from_=0, to=16, width=5, command=self._push_format_to_state
        )
        self.manual_shift_spinbox.pack(side=tk.LEFT, padx=4)
        ttk.Label(manual_shift_frame, text="bits").pack(side=tk.LEFT)
        self._row(rawf, 4, "", manual_shift_frame)
        self.manual_shift_row_widgets = (manual_shift_frame,)  # Store for show/hide
        self._row(
            rawf,
            5,
            "Machine format",
            ttk.Combobox(
                rawf,
                textvariable=self.var_endian,
                values=("ieee-le", "ieee-be"),
                width=14,
                state="readonly",
            ),
        )
        adv = ttk.Frame(rawf)
        ttk.Entry(adv, textvariable=self.var_hdr_bits, width=5).pack(side=tk.LEFT)
        ttk.Entry(adv, textvariable=self.var_pipe_bits, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Entry(adv, textvariable=self.var_out_bits, width=5).pack(side=tk.LEFT)
        self._row(rawf, 6, "HDR / pipe / out bits", adv)

        stats = ttk.LabelFrame(left, text="Computed", padding=8)
        stats.pack(fill=tk.X, pady=4)
        self.lbl_computed = ttk.Label(stats, text="", justify=tk.LEFT)
        self.lbl_computed.pack(anchor="w")

        actions = ttk.LabelFrame(left, text="Test image", padding=8)
        actions.pack(fill=tk.X, pady=4)
        self.var_ae = tk.BooleanVar(value=True)
        self.var_awb = tk.BooleanVar(value=True)
        self.var_dg = tk.StringVar(value="1")
        ttk.Button(actions, text="Load test image", command=self._load_image).pack(fill=tk.X)
        row = ttk.Frame(actions)
        row.pack(fill=tk.X, pady=4)
        ttk.Checkbutton(row, text="Apply AE", variable=self.var_ae, command=self._push_format_to_state).pack(side=tk.LEFT)
        ttk.Checkbutton(row, text="AWB", variable=self.var_awb, command=self._push_format_to_state).pack(side=tk.LEFT)
        ttk.Label(row, text="Digital gain idx").pack(side=tk.LEFT, padx=(8, 2))
        # Use Spinbox with bounds instead of plain Entry to prevent out-of-range values
        self.dg_spinbox = ttk.Spinbox(
            row, textvariable=self.var_dg, width=6, from_=0, to=15, command=self._push_format_to_state
        )
        self.dg_spinbox.pack(side=tk.LEFT)
        self.reload_btn = ttk.Button(actions, text="Reload test image", command=self._reload_image, state="disabled")
        self.reload_btn.pack(fill=tk.X, pady=2)
        self.process_btn = ttk.Button(actions, text="Process", command=self.process)
        self.process_btn.pack(fill=tk.X, pady=2)

        prev_bar = ttk.Frame(right)
        prev_bar.pack(fill=tk.X)
        ttk.Button(prev_bar, text="Fit", command=lambda: self.preview.fit()).pack(side=tk.LEFT, padx=2)
        ttk.Button(prev_bar, text="1:1", command=lambda: self.preview.one_to_one()).pack(side=tk.LEFT, padx=2)
        ttk.Button(prev_bar, text="Zoom +", command=lambda: self.preview.zoom(1 / 1.2)).pack(side=tk.LEFT, padx=2)
        ttk.Button(prev_bar, text="Zoom −", command=lambda: self.preview.zoom(1.2)).pack(side=tk.LEFT, padx=2)
        ttk.Button(prev_bar, text="Refresh preview", command=self._refresh_preview).pack(side=tk.LEFT, padx=2)
        preview_holder = ttk.LabelFrame(right, text="Preview", padding=2)
        preview_holder.pack(fill=tk.BOTH, expand=True)
        self.preview = PreviewPane(preview_holder, tk)

        self.status = ttk.Label(root, text="", relief=tk.SUNKEN, anchor="w")
        self.status.pack(fill=tk.X, side=tk.BOTTOM)

        for var in (
            self.var_sensor,
            self.var_width,
            self.var_height,
            self.var_ml,
            self.var_mr,
            self.var_mt,
            self.var_mb,
            self.var_rggb,
            self.var_header,
            self.var_bit,
            self.var_fmt,
            self.var_align,
            self.var_manual_shift,
            self.var_endian,
            self.var_hdr_bits,
            self.var_pipe_bits,
            self.var_out_bits,
            self.var_flip_h,
            self.var_flip_v,
            self.var_ae,
            self.var_awb,
            self.var_dg,
        ):
            var.trace_add("write", lambda *_: self._on_format_changed())

        # Add special handler for alignment mode to show/hide manual shift control
        self.var_align.trace_add("write", lambda *_: self._update_manual_shift_visibility())
        self._update_manual_shift_visibility()  # Initialize visibility

        self._widget_by_path = {
            "sensor_info.width": None,
            "sensor_info.height": None,
        }

    def _row(self, parent: Any, row: int, label: str, widget: Any) -> None:
        self.ttk.Label(parent, text=label).grid(row=row, column=0, sticky="e", padx=(0, 8), pady=2)
        widget.grid(row=row, column=1, sticky="w", pady=2)
        parent.columnconfigure(1, weight=1)

    def _build_isp_window(self, win: Any) -> None:
        tk, ttk = self.tk, self.ttk
        bar = ttk.Frame(win, padding=6)
        bar.pack(fill=tk.X)
        ttk.Button(bar, text="Process", command=self.process).pack(side=tk.LEFT)
        ttk.Button(bar, text="Reset module", command=self._reset_selected_module).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Reset all", command=self._reset_all).pack(side=tk.LEFT)
        self.isp_status = ttk.Label(bar, text="")
        self.isp_status.pack(side=tk.RIGHT)

        wrap = ttk.Frame(win)
        wrap.pack(fill=tk.BOTH, expand=True)
        self._isp_canvas = tk.Canvas(wrap, highlightthickness=0)
        vsb = ttk.Scrollbar(wrap, orient=tk.VERTICAL, command=self._isp_canvas.yview)
        self._isp_inner = ttk.Frame(self._isp_canvas)
        self._isp_inner.bind(
            "<Configure>",
            lambda e: self._isp_canvas.configure(scrollregion=self._isp_canvas.bbox("all")),
        )
        self._isp_canvas.create_window((0, 0), window=self._isp_inner, anchor="nw")
        self._isp_canvas.configure(yscrollcommand=vsb.set)
        self._isp_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        win.bind("<MouseWheel>", self._on_isp_wheel, add="+")

        self._selected_module = tk.StringVar(value="crop")

    def _on_isp_wheel(self, event: Any) -> None:
        self._isp_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _rebuild_isp_form(self) -> None:
        for child in self._isp_inner.winfo_children():
            child.destroy()
        self._isp_vars.clear()
        self._highlight.clear()
        cfg = self.state.config
        keys = [k for k in ISP_SECTION_ORDER if k in cfg] + [
            k for k in cfg if k not in ISP_SECTION_ORDER and isinstance(cfg.get(k), dict)
        ]
        modified_only = bool(self._show_modified_only.get())
        for section in keys:
            if not isinstance(cfg.get(section), dict):
                continue
            if modified_only and not self.state.is_dirty(section):
                continue
            self._add_section(section, cfg[section])
        self._isp_inner.update_idletasks()

    def _add_section(self, section: str, data: dict[str, Any]) -> None:
        tk, ttk = self.tk, self.ttk
        title = section.replace("_", " ")
        if self.state.is_dirty(section):
            title += " *"
        box = ttk.LabelFrame(self._isp_inner, text=title, padding=6)
        box.pack(fill=tk.X, padx=8, pady=4)
        box.bind("<Button-1>", lambda _e, s=section: self._selected_module.set(s))
        row = 0
        for key, val in data.items():
            path = f"{section}.{key}"
            # Skip platform.filename — it's the session copy name "input.raw", not the user's file.
            # The original filename is shown in the Image Format window's "RAW image" field.
            if path == "platform.filename":
                continue
            dirty = self.state.is_dirty(path)
            label = key + (" *" if dirty else "")
            ttk.Label(box, text=label).grid(row=row, column=0, sticky="ne", padx=(0, 6), pady=2)
            widget = self._make_value_widget(box, path, val)
            widget.grid(row=row, column=1, sticky="ew", pady=2)
            box.columnconfigure(1, weight=1)
            self._highlight[path] = widget
            rst = ttk.Button(box, text="reset", width=6, command=lambda p=path: self._reset_parameter(p))
            rst.grid(row=row, column=2, padx=4)
            row += 1

    def _make_value_widget(self, parent: Any, path: str, value: Any) -> Any:
        ttk = self.ttk
        tk = self.tk
        if isinstance(value, bool):
            var = tk.BooleanVar(value=value)
            self._isp_vars[path] = var
            cb = ttk.Checkbutton(parent, variable=var, command=lambda: self._isp_changed(path, var.get()))
            return cb
        if path in ENUM_CHOICES:
            var = tk.StringVar(value=str(value))
            self._isp_vars[path] = var
            cb = ttk.Combobox(parent, textvariable=var, values=ENUM_CHOICES[path], state="readonly", width=28)
            cb.bind("<<ComboboxSelected>>", lambda _e: self._isp_changed(path, var.get()))
            return cb
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            var = tk.StringVar(value=str(value))
            self._isp_vars[path] = var
            ent = ttk.Entry(parent, textvariable=var, width=28)
            ent.bind("<FocusOut>", lambda _e, p=path, v=var: self._isp_number(p, v.get(), value))
            ent.bind("<Return>", lambda _e, p=path, v=var: self._isp_number(p, v.get(), value))
            return ent
        if isinstance(value, str) or value is None:
            var = tk.StringVar(value="" if value is None else value)
            self._isp_vars[path] = var
            ent = ttk.Entry(parent, textvariable=var, width=28)
            ent.bind("<FocusOut>", lambda _e, p=path, v=var: self._isp_changed(p, v.get() or None))
            return ent
        text = tk.Text(parent, height=min(6, 1 + str(value).count("\n")), width=36, wrap="none")
        dumped = yaml.safe_dump(yaml_safe_value(value), default_flow_style=True).strip()
        text.insert("1.0", dumped)
        text.bind("<FocusOut>", lambda _e, p=path, w=text: self._isp_yaml(p, w))
        self._isp_vars[path] = text
        return text

    def _isp_number(self, path: str, raw: str, original: Any) -> None:
        if isinstance(original, int) and not isinstance(original, bool):
            try:
                self._isp_changed(path, int(float(raw)))
                return
            except ValueError:
                pass
        try:
            self._isp_changed(path, float(raw))
        except ValueError:
            pass

    def _isp_yaml(self, path: str, widget: Any) -> None:
        raw = widget.get("1.0", "end").strip()
        try:
            parsed = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            self.log.warning("Invalid YAML for %s: %s", path, exc)
            return
        self._isp_changed(path, parsed)

    def _isp_changed(self, path: str, value: Any) -> None:
        if self._syncing:
            return
        previous = self.state.get(path)
        self.state.set(path, value)
        if previous != value:
            self.log.info("ISP change %s: %s -> %s", path, truncate(previous), truncate(value))
        self._set_status(f"Updated {path}")
        # Keep Image Format digital-gain index in sync with ISP form current_gain.
        if path == "digital_gain.current_gain":
            try:
                self.var_dg.set(str(int(value)))
            except (TypeError, ValueError, self.tk.TclError):
                pass
        if path.startswith("sensor_info.") or path.startswith("crop.") or path.startswith("scale."):
            self._load_format_from_state()
            self._update_computed()
            if path in (
                "sensor_info.horizontal_flip",
                "sensor_info.vertical_flip",
                "sensor_info.width",
                "sensor_info.height",
                "sensor_info.bayer_pattern",
                "sensor_info.endian_type",
                "sensor_info.bit_depth",
                "sensor_info.data_format",
            ):
                self._show_loaded_raw_preview()

    def _flush_isp_form_to_state(self) -> None:
        """Commit ISP form widgets (including Entry values that only save on FocusOut)."""
        if self._syncing:
            return
        for path, var in list(self._isp_vars.items()):
            if isinstance(var, self.tk.BooleanVar):
                self.state.set(path, bool(var.get()))
            elif isinstance(var, self.tk.Variable):
                raw = var.get()
                existing = self.state.get(path)
                if isinstance(existing, bool):
                    self.state.set(path, bool(raw))
                elif isinstance(existing, int) and not isinstance(existing, bool):
                    try:
                        self.state.set(path, int(float(str(raw))))
                    except (TypeError, ValueError):
                        pass
                elif isinstance(existing, float):
                    try:
                        self.state.set(path, float(str(raw)))
                    except (TypeError, ValueError):
                        pass
                elif isinstance(existing, str) or existing is None:
                    self.state.set(path, (str(raw) if raw is not None else None) or None)
            else:
                # YAML Text widgets — already handled via FocusOut; skip here.
                continue
        # Keep format-bar gain index aligned after flush.
        try:
            self.var_dg.set(str(int(self.state.get("digital_gain.current_gain") or 0)))
        except (TypeError, ValueError, self.tk.TclError):
            pass

    def _on_flip_changed(self) -> None:
        self._push_format_to_state()
        self._show_loaded_raw_preview()

    def _update_manual_shift_visibility(self) -> None:
        """Show/hide the manual bit-shift control based on alignment mode."""
        alignment = self.var_align.get()
        for widget in self.manual_shift_row_widgets:
            if alignment == "MANUAL":
                widget.grid()  # Show the control
            else:
                widget.grid_remove()  # Hide but keep in layout

    def _on_format_changed(self) -> None:
        if self._syncing:
            return
        self._push_format_to_state()
        self._draw_cfa()
        self._update_computed()
        self._schedule_format_log()

    def _format_snapshot(self) -> dict[str, Any]:
        return {
            "sensor": self.var_sensor.get(),
            "raw_path": self.var_raw_path.get(),
            "width": self.var_width.get(),
            "height": self.var_height.get(),
            "crop_lrtb": (self.var_ml.get(), self.var_mr.get(), self.var_mt.get(), self.var_mb.get()),
            "alignment": self.var_align.get(),
            "manual_shift": self.var_manual_shift.get() if self.var_align.get() == "MANUAL" else None,
            "flip_h": self.var_flip_h.get(),
            "flip_v": self.var_flip_v.get(),
            "rggb": self.var_rggb.get(),
            "header": self.var_header.get(),
            "bit_depth": self.var_bit.get(),
            "data_format": self.var_fmt.get(),
            "alignment": self.var_align.get(),
            "endian": self.var_endian.get(),
            "hdr_pipe_out": (self.var_hdr_bits.get(), self.var_pipe_bits.get(), self.var_out_bits.get()),
            "ae": self.var_ae.get(),
            "awb": self.var_awb.get(),
            "digital_gain_idx": self.var_dg.get(),
        }

    def _schedule_format_log(self) -> None:
        if self._format_log_after is not None:
            try:
                self.root.after_cancel(self._format_log_after)
            except Exception:
                pass
        self._format_log_after = self.root.after(400, self._flush_format_log)

    def _flush_format_log(self) -> None:
        self._format_log_after = None
        snap = self._format_snapshot()
        diffs = diff_dicts(self._last_format_snapshot, snap)
        if diffs:
            self.log.info("Image format changes:\n  %s", "\n  ".join(diffs))
        self._last_format_snapshot = snap

    def _push_format_to_state(self) -> None:
        if self._syncing:
            return
        w = _to_int(self.var_width.get())
        h = _to_int(self.var_height.get())
        self.state.set("sensor_info.sensor", self.var_sensor.get())
        self.state.set("sensor_info.width", w)
        self.state.set("sensor_info.height", h)
        self.state.set("sensor_info.bit_depth", _to_int(self.var_bit.get(), 12))
        self.state.set("sensor_info.data_format", self.var_fmt.get() or "uint16")
        alignment = self.var_align.get() or "LSB"
        self.state.set("sensor_info.data_alignment", alignment)
        # Only set manual_bit_shift if MANUAL mode is selected
        if alignment == "MANUAL":
            self.state.set("sensor_info.manual_bit_shift", _to_int(self.var_manual_shift.get(), 0))
        else:
            # Remove manual_bit_shift if not in MANUAL mode
            if "manual_bit_shift" in self.state.config.get("sensor_info", {}):
                del self.state.config["sensor_info"]["manual_bit_shift"]
        self.state.set("sensor_info.endian_type", self.var_endian.get() or "ieee-le")
        self.state.set("sensor_info.hdr_bit_depth", _to_int(self.var_hdr_bits.get(), 24))
        self.state.set("sensor_info.pipeline_rgb_bit_depth", _to_int(self.var_pipe_bits.get(), 16))
        self.state.set("sensor_info.output_bit_depth", _to_int(self.var_out_bits.get(), 8))
        self.state.set("sensor_info.horizontal_flip", bool(self.var_flip_h.get()))
        self.state.set("sensor_info.vertical_flip", bool(self.var_flip_v.get()))
        try:
            start = RGGB_START_LABELS.index(self.var_rggb.get())
        except ValueError:
            start = 0
        self.state.set("sensor_info.bayer_pattern", bayer_from_rggb_start(start))
        crop = self.state.config.setdefault("crop", {})
        update_crop_from_margins(
            crop,
            w or 0,
            h or 0,
            _to_int(self.var_ml.get()),
            _to_int(self.var_mr.get()),
            _to_int(self.var_mt.get()),
            _to_int(self.var_mb.get()),
        )
        self.state.apply_ae(bool(self.var_ae.get()))
        self.state.apply_awb(bool(self.var_awb.get()))
        # Clamp digital gain index to valid range before setting
        dg_config = self.state.config.get("digital_gain") or {}
        gain_array = dg_config.get("gain_array", [1.0])
        max_idx = len(gain_array) - 1 if isinstance(gain_array, list) and len(gain_array) > 0 else 0
        dg_idx = max(0, min(_to_int(self.var_dg.get(), 0), max_idx))
        self.state.set_digital_gain_index(dg_idx)
        self.session.ui_state["header_size"] = _to_int(self.var_header.get())
        self.session.ui_state["data_alignment"] = self.var_align.get()
        self.session.ui_state["flip_h"] = bool(self.var_flip_h.get())
        self.session.ui_state["flip_v"] = bool(self.var_flip_v.get())

    def _load_format_from_state(self) -> None:
        self._syncing = True
        try:
            si = self.state.config.get("sensor_info") or {}
            crop = self.state.config.get("crop") or {}
            self.var_sensor.set(str(si.get("sensor") or ""))
            self.var_width.set(str(si.get("width", "")))
            self.var_height.set(str(si.get("height", "")))
            self.var_bit.set(str(si.get("bit_depth", "")))
            self.var_fmt.set(str(si.get("data_format") or "uint16"))
            self.var_endian.set(str(si.get("endian_type") or "ieee-le"))
            self.var_hdr_bits.set(str(si.get("hdr_bit_depth", "")))
            self.var_pipe_bits.set(str(si.get("pipeline_rgb_bit_depth", "")))
            self.var_out_bits.set(str(si.get("output_bit_depth", "")))
            try:
                idx = rggb_start_from_bayer(str(si.get("bayer_pattern", "rggb")))
            except ValueError:
                idx = 0
            self.var_rggb.set(RGGB_START_LABELS[idx])
            l, r, t, b = margins_from_crop(int(si.get("width") or 0), int(si.get("height") or 0), crop)
            self.var_ml.set(str(l))
            self.var_mr.set(str(r))
            self.var_mt.set(str(t))
            self.var_mb.set(str(b))
            self.var_ae.set(bool((self.state.config.get("auto_exposure") or {}).get("is_enable", False)))
            self.var_awb.set(bool((self.state.config.get("auto_white_balance") or {}).get("is_enable", False)))
            dg_config = self.state.config.get("digital_gain") or {}
            current_gain = int(dg_config.get("current_gain", 0))
            self.var_dg.set(str(current_gain))
            # Update Spinbox bounds based on gain_array length
            self._update_digital_gain_bounds()
            ui = self.session.ui_state
            self.var_header.set(str(ui.get("header_size", 0)))
            align = si.get("data_alignment") or si.get("bit_alignment") or ui.get("data_alignment", "LSB")
            self.var_align.set(str(align))
            # Load manual bit shift value
            manual_shift = si.get("manual_bit_shift", 4)
            self.var_manual_shift.set(str(manual_shift))
            if "horizontal_flip" in si:
                self.var_flip_h.set(bool(si.get("horizontal_flip")))
            else:
                self.var_flip_h.set(bool(ui.get("flip_h", False)))
            if "vertical_flip" in si:
                self.var_flip_v.set(bool(si.get("vertical_flip")))
            else:
                self.var_flip_v.set(bool(ui.get("flip_v", False)))
            self.var_workplace.set(str(self.session.paths.directory))
            src = ui.get("original_source_path")
            self.var_raw_path.set(str(src) if src else "")
            self.reload_btn.config(state="normal" if src else "disabled")
            self._draw_cfa()
            self._last_format_snapshot = self._format_snapshot()
        finally:
            self._syncing = False

    def _draw_cfa(self) -> None:
        pattern = str(self.state.get("sensor_info.bayer_pattern") or "rggb")
        tiles = CFA_TILES.get(pattern, CFA_TILES["rggb"])
        c = self.cfa_canvas
        c.delete("all")
        size = 54
        for y, row in enumerate(tiles):
            for x, ch in enumerate(row):
                x0, y0 = 6 + x * size, 6 + y * size
                c.create_rectangle(x0, y0, x0 + size, y0 + size, fill=CFA_COLORS[ch], outline="#222")
                c.create_text(x0 + size / 2, y0 + size / 2, text=ch, fill="white", font=("Segoe UI", 16, "bold"))

    def _update_computed(self) -> None:
        stats = compute_format_stats(
            self.state.config,
            header_bytes=_to_int(self.var_header.get()),
        )
        self.var_eff_w.set(str(stats["effective_width"]))
        self.var_eff_h.set(str(stats["effective_height"]))
        raw_mib = stats["estimated_raw_size"] / (1024 * 1024)
        self.lbl_computed.config(
            text=(
                f"Active pixels: {stats['active_pixel_count']:,}\n"
                f"Crop: {stats['crop_percentage']:.2f} %\n"
                f"Estimated RAW: {stats['estimated_raw_size']:,} bytes ({raw_mib:.3f} MiB)\n"
                f"Output: {stats['output_width']} × {stats['output_height']}"
            )
        )

    def _update_digital_gain_bounds(self) -> None:
        """Update the digital gain Spinbox bounds based on the current gain_array length."""
        dg_config = self.state.config.get("digital_gain") or {}
        gain_array = dg_config.get("gain_array", [1.0])
        if isinstance(gain_array, list) and len(gain_array) > 0:
            max_idx = len(gain_array) - 1
            self.dg_spinbox.config(from_=0, to=max_idx)
            # Ensure current value is within bounds
            try:
                current = int(self.var_dg.get())
                if current > max_idx:
                    self.var_dg.set(str(max_idx))
                    self.log.warning("Digital gain index %d exceeds max %d; clamped to %d", current, max_idx, max_idx)
                elif current < 0:
                    self.var_dg.set("0")
                    self.log.warning("Digital gain index %d is negative; clamped to 0", current)
            except (ValueError, TypeError):
                self.var_dg.set("0")
                self.log.warning("Invalid digital gain index; reset to 0")

    def _load_image(self) -> None:
        path = self.filedialog.askopenfilename(
            title="Load test image",
            filetypes=[
                ("RAW / PGM / image", "*.raw *.RAW *.pgm *.PGM *.tiff *.tif *.dng *.png *.jpg"),
                ("PGM", "*.pgm *.PGM"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._load_raw_path(Path(path))

    def _load_raw_path(self, source: Path) -> None:
        self.log.info("Load RAW requested: %s", source)
        try:
            self.session.copy_raw_from(source)
        except OSError as exc:
            self.log.exception("Failed to copy RAW from %s", source)
            self.messagebox.showerror("Load image", str(exc))
            return
        except ValueError as exc:
            self.log.exception("Failed to parse RAW/PGM from %s", source)
            self.messagebox.showerror("Load image", str(exc))
            return
        meta = getattr(self.session, "last_payload_meta", None)
        if meta is not None and getattr(meta, "is_pgm", False):
            if meta.pgm_width:
                self.state.set("sensor_info.width", int(meta.pgm_width))
            if meta.pgm_height:
                self.state.set("sensor_info.height", int(meta.pgm_height))
            self._load_format_from_state()
            self._update_computed()
            self._rebuild_isp_form()
        self.var_raw_path.set(str(Path(source).resolve()))
        self.reload_btn.config(state="normal")
        self._show_loaded_raw_preview()
        self.log.info("RAW copied to %s (%s bytes)", self.session.paths.input_raw, self.session.paths.input_raw.stat().st_size)
        self._set_status(f"Loaded {source} → {self.session.paths.input_raw}")

    def _reload_image(self) -> None:
        try:
            self.session.reload_raw()
        except OSError as exc:
            self.messagebox.showerror("Reload", str(exc))
            self.log.exception("Reload RAW failed")
            return
        src = self.session.ui_state.get("original_source_path")
        if src:
            self.var_raw_path.set(str(src))
        self._show_loaded_raw_preview()
        self._set_status("Reloaded RAW copy")

    def _show_loaded_raw_preview(self) -> None:
        if not self.session.has_raw:
            return
        self._push_format_to_state()
        try:
            rgb = preview_from_config(
                self.session.paths.input_raw,
                self.state.config,
                header_bytes=int(self.session.ui_state.get("header_size") or 0),
            )
        except Exception as exc:
            self.log.exception("RAW preview failed")
            self._set_status(f"RAW preview failed: {exc}")
            return
        self.log.info("RAW preview displayed shape=%s", tuple(rgb.shape))
        self.preview.show_rgb(rgb)

    def _refresh_preview(self) -> None:
        if self.preview.refresh_from_path(self.session.paths.output_png):
            return
        if self.session.has_raw:
            self._show_loaded_raw_preview()
            return
        if self.preview.last_rgb is not None:
            self.preview.show_rgb(self.preview.last_rgb)
        else:
            self._set_status("No preview yet — load a RAW or run Process")

    def _highlight_issues(self, issues: list[ValidationIssue]) -> None:
        msg = "\n".join(f"{i.path}: {i.message}" for i in issues)
        self.log.warning("Validation blocked Process:\n%s", msg)
        self.messagebox.showerror("Validation", msg)
        self._set_status("Process blocked — fix validation errors")

    def process(self) -> None:
        if self._processing:
            return
        # Format bar first, then ISP form wins for overlapping keys (esp. current_gain).
        self._push_format_to_state()
        self._flush_isp_form_to_state()
        ok, issues = prepare_process(self.session, self.state.config)
        if not ok:
            self._highlight_issues(issues)
            return
        self._processing = True
        self.process_btn.config(state="disabled")
        self._set_status("Processing…")
        endian = str(self.state.get("sensor_info.endian_type") or "ieee-le")
        byte_order = "big" if "be" in endian else "little"
        session_dir = str(self.session.paths.directory)
        cfg_path = str(self.session.paths.config_yml)

        self.log.info(
            "Process start session=%s endian=%s raw=%s",
            session_dir,
            byte_order,
            self.session.ui_state.get("original_source_path"),
        )

        def work() -> None:
            rgb = None
            isp_done = None
            err = None
            try:
                from brilliant_isp import BrilliantISP

                isp_done = BrilliantISP(session_dir, cfg_path, outFileName="", output_path=session_dir)
                isp_done.execute(img_path="input.raw", byte_order=byte_order)  # type: ignore[arg-type]
                rgb = isp_done.last_output_rgb
            except Exception:
                err = traceback.format_exc()
                get_gui_logger().error("Pipeline exception:\n%s", err)

            def finish() -> None:
                self._processing = False
                self.process_btn.config(state="normal")
                if err:
                    self.log.error("Process failed")
                    self.messagebox.showerror("Pipeline error", err)
                    self._set_status("Error during processing")
                    return
                if isp_done is not None and isp_done.c_yaml is not None:
                    merge_pipeline_feedback(self.state.config, isp_done.c_yaml)
                    self._load_format_from_state()
                    self._rebuild_isp_form()
                if rgb is not None:
                    self.preview.show_rgb(rgb)
                    arr = np.clip(np.asarray(rgb), 0, 255).astype(np.uint8)
                    bgr = arr if arr.ndim == 2 else cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(self.session.paths.output_png), bgr)
                    self.session.write_ui_state()
                    self.log.info("Process done preview_shape=%s", tuple(arr.shape))
                else:
                    self.log.warning("Process finished with no preview RGB")
                self._set_status("Done")

            self.root.after(0, finish)

        threading.Thread(target=work, daemon=True).start()

    def _new_session(self) -> None:
        path = self.filedialog.askopenfilename(
            title="Preset YAML for new session",
            initialdir=str(REPO_ROOT / "config"),
            filetypes=[("YAML", "*.yml *.yaml")],
        )
        preset = Path(path) if path else DEFAULT_PRESET
        self.session, cfg = GuiSession.create_new(preset_path=preset)
        attach_session_log(self.session.paths.directory)
        self.log.info("New session %s preset=%s", self.session.session_id, preset)
        self.state.load_from(cfg, preset_path=str(preset))
        self._load_format_from_state()
        self._rebuild_isp_form()
        self._update_computed()
        self._set_status(f"New session {self.session.session_id}")

    def _open_session(self) -> None:
        path = self.filedialog.askdirectory(title="Open session directory", initialdir=str(DEFAULT_SESSION_ROOT))
        if not path:
            return
        try:
            session, cfg = GuiSession.open_existing(path)
        except OSError as exc:
            self.log.exception("Open session failed: %s", path)
            self.messagebox.showerror("Open session", str(exc))
            return
        self.session = session
        attach_session_log(self.session.paths.directory)
        self.log.info("Opened session %s", session.session_id)
        self.state.load_from(cfg, preset_path=session.ui_state.get("last_preset_path"))
        self._load_format_from_state()
        self._rebuild_isp_form()
        self._update_computed()
        self.preview.refresh_from_path(session.paths.output_png)
        self._set_status(f"Opened {session.session_id}")

    def _save_session(self) -> None:
        self._push_format_to_state()
        self.session.save(self.state.config)
        self.log.info("Session saved %s", self.session.paths.directory)
        self._set_status("Session saved")

    def _open_preset(self) -> None:
        path = self.filedialog.askopenfilename(
            title="Open preset YAML",
            initialdir=str(REPO_ROOT / "config"),
            filetypes=[("YAML", "*.yml *.yaml")],
        )
        if not path:
            return
        try:
            cfg = load_preset_yaml(path)
        except Exception as exc:
            self.log.exception("Failed to load preset %s", path)
            self.messagebox.showerror("Open preset", str(exc))
            return
        self.log.info("Loaded preset %s", path)
        self.state.load_from(cfg, preset_path=path)
        self.session.ui_state["last_preset_path"] = path
        self.session.ensure_platform_filename(self.state.config)
        self._load_format_from_state()
        self._rebuild_isp_form()
        self._update_computed()
        self._set_status(f"Loaded preset {Path(path).name}")

    def _export_yaml(self) -> None:
        self._push_format_to_state()
        path = self.filedialog.asksaveasfilename(
            title="Export YAML",
            defaultextension=".yml",
            filetypes=[("YAML", "*.yml *.yaml")],
        )
        if not path:
            return
        dump_config_yaml(self.state.config, path)
        self.log.info("Exported YAML %s", path)
        self._set_status(f"Exported {path}")

    def _reset_parameter(self, path: str) -> None:
        self.log.info("Reset parameter %s", path)
        self.state.reset_parameter(path)
        self._load_format_from_state()
        self._rebuild_isp_form()
        self._update_computed()

    def _reset_selected_module(self) -> None:
        section = self._selected_module.get()
        self.log.info("Reset module %s", section)
        self.state.reset_module(section)
        self._load_format_from_state()
        self._rebuild_isp_form()
        self._update_computed()

    def _reset_all(self) -> None:
        self.log.info("Reset all parameters to preset snapshot")
        self.state.reset_all()
        self._load_format_from_state()
        self._rebuild_isp_form()
        self._update_computed()

    def _on_logged_exception(self, text: str) -> None:
        try:
            self.messagebox.showerror("Unhandled error", text[-4000:])
        except Exception:
            pass

    def _open_log_folder(self) -> None:
        folder = self.session.paths.directory
        self.log.info("Open log folder %s", folder)
        try:
            if sys.platform == "win32":
                os.startfile(folder)  # type: ignore[attr-defined]
            else:
                import subprocess

                subprocess.Popen(["xdg-open", str(folder)])
        except Exception:
            self.log.exception("Could not open log folder")
            self.messagebox.showinfo(
                "Logs", f"Session log:\n{folder / 'gui.log'}\n\nGlobal log:\n{DEFAULT_SESSION_ROOT / 'calibration_gui.log'}"
            )

    def _set_status(self, text: str) -> None:
        self.status.config(text=text)
        if hasattr(self, "isp_status"):
            self.isp_status.config(text=text)


def main() -> None:
    parser = argparse.ArgumentParser(description="BrilliantISP calibration GUI")
    parser.add_argument("--config", type=Path, default=None, help="Initial preset YAML")
    parser.add_argument("--raw", type=Path, default=None, help="RAW/image to load into the session")
    parser.add_argument("--no-recover", action="store_true", help="Skip auto-recover prompt")
    args = parser.parse_args()
    import tkinter as tk

    root = tk.Tk()
    CalibrationApp(
        root,
        preset=args.config,
        recover=not args.no_recover,
        initial_raw=args.raw,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
