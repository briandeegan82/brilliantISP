"""Inputs/format panel: CFA tile, format fields, gain shortcuts, AE/AWB."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable


class _Scroll(ttk.Frame):
    def __init__(self, parent) -> None:
        super().__init__(parent)
        canvas = tk.Canvas(self, highlightthickness=0)
        scroll = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.inner = ttk.Frame(canvas)
        self.inner.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.inner, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(-e.delta // 120, "units"))


class FormatPanel(ttk.Frame):
    FIELDS = (
        "width", "height", "bayer_pattern", "bit_depth", "hdr_bit_depth",
        "output_bit_depth", "pipeline_rgb_bit_depth", "storage_format",
        "data_format", "data_alignment", "endian_type", "source_format",
        "header_bytes", "row_stride_bytes", "horizontal_flip",
        "vertical_flip", "embedded_rows_top", "embedded_rows_bottom")

    def __init__(self, parent, model, on_change: Callable[[], None]) -> None:
        super().__init__(parent)
        self.model = model
        self.on_change = on_change
        self._sync = False
        sc = _Scroll(self)
        sc.pack(fill="both", expand=True)
        self._host = sc.inner
        self._build()

    def _build(self) -> None:
        self._sync = True
        for w in self._host.winfo_children():
            w.destroy()
        sensor = self.model.config.get("sensor_info", {})
        if not isinstance(sensor, dict):
            sensor = {}
        ttk.Label(self._host, text="Sensor format",
                  font="TkHeadingFont").pack(anchor="w")
        self._cfa_row(sensor)
        for f in self.FIELDS:
            self._field_row(sensor, f)
        ttk.Separator(self._host).pack(fill="x", pady=6)
        self._toggles()
        self._gain_row()
        self._stats()
        self._sync = False

    def _cfa_row(self, sensor: dict) -> None:
        row = ttk.Frame(self._host)
        row.pack(fill="x")
        ttk.Label(row, text="bayer_pattern", width=22).pack(side="left")
        colors = {"r": "#d55", "g": "#5c5", "b": "#55d"}
        tile = ttk.Frame(row)
        tile.pack(side="left", padx=4)
        self._tile = tile
        current = str(sensor.get("bayer_pattern", "rggb"))
        cb = ttk.Combobox(row, values=["rggb", "grbg", "gbrg", "bggr"],
                          width=6, state="readonly")
        cb.set(current)
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda e:
                self._emit("sensor_info.bayer_pattern", cb.get(), current))
        self._paint_tile(current)

    def _paint_tile(self, pattern: str) -> None:
        colors = {"r": "#d55", "g": "#5c5", "b": "#55d"}
        for w in self._tile.winfo_children():
            w.destroy()
        for i, ch in enumerate(str(pattern).lower()[:4]):
            tk.Label(self._tile, text=ch.upper(), width=2,
                     bg=colors.get(ch, "#999"), fg="white").grid(
                         row=i // 2, column=i % 2)

    def _field_row(self, sensor: dict, f: str) -> None:
        row = ttk.Frame(self._host)
        row.pack(fill="x")
        ttk.Label(row, text=f, width=22).pack(side="left")
        val = sensor.get(f)
        if isinstance(val, bool):
            var = tk.BooleanVar(value=val)
            ttk.Checkbutton(
                row, variable=var,
                command=lambda v=var, k=f, c=val:
                self._emit(f"sensor_info.{k}", v.get(), c)).pack(side="left")
        else:
            ent = ttk.Entry(row, width=14)
            ent.insert(0, "" if val is None else str(val))
            ent.pack(side="left")
            ent.bind("<KeyRelease>", lambda e, en=ent, k=f, c=val:
                     self._emit_typed(en, f"sensor_info.{k}", c))

    def _toggles(self) -> None:
        row = ttk.Frame(self._host)
        row.pack(fill="x")
        ae = tk.BooleanVar(
            value=bool(self.model.get("auto_exposure.is_enable", False)))
        ttk.Checkbutton(row, text="Auto exposure", variable=ae,
                        command=lambda: self._emit(
                            "auto_exposure.is_enable", ae.get(), None)
                        ).pack(side="left")
        awb = tk.BooleanVar(
            value=bool(self.model.get("auto_white_balance.is_enable", True)))
        ttk.Checkbutton(row, text="Auto WB", variable=awb,
                        command=lambda: self._emit(
                            "auto_white_balance.is_enable", awb.get(), None)
                        ).pack(side="left")

    def _gain_row(self) -> None:
        row = ttk.Frame(self._host)
        row.pack(fill="x", pady=4)
        gains = self.model.get("digital_gain.gain_array", [])
        count = len(gains) if isinstance(gains, list) else 0
        cur = self.model.get("digital_gain.current_gain", 0)
        ttk.Label(row, text=f"DG index {cur} / {max(count - 1, 0)}").pack(
            side="left")
        ttk.Button(row, text="-", width=3,
                   command=lambda: self._bump(-1)).pack(side="left")
        ttk.Button(row, text="+", width=3,
                   command=lambda: self._bump(1)).pack(side="left")

    def _stats(self) -> None:
        try:
            from tools.gui.computed import compute_format_stats
            stats = compute_format_stats(self.model.config)
            ttk.Label(self._host, text=str(stats)).pack(anchor="w")
        except Exception:
            pass

    def _emit(self, path: str, value: Any, current: Any) -> None:
        if self._sync:
            return
        self.model.edit(path, value)
        self.on_change()

    def _emit_typed(self, entry: ttk.Entry, path: str, current: Any) -> None:
        if self._sync:
            return
        raw = entry.get().strip()
        if raw == "":
            self.model.edit(path, None)
        elif isinstance(current, bool):
            self.model.edit(path, raw)
        elif isinstance(current, int):
            try:
                self.model.edit(path, int(raw, 0))
            except ValueError:
                self.model.edit(path, raw)
        elif isinstance(current, float):
            try:
                self.model.edit(path, float(raw))
            except ValueError:
                self.model.edit(path, raw)
        else:
            self.model.edit(path, raw)
        self.on_change()

    def _bump(self, delta: int) -> None:
        gains = self.model.get("digital_gain.gain_array", [])
        limit = len(gains) - 1 if isinstance(gains, list) and gains else 0
        try:
            cur = int(self.model.get("digital_gain.current_gain", 0))
        except (TypeError, ValueError):
            cur = 0
        self.model.edit("digital_gain.current_gain",
                        max(0, min(limit, cur + delta)))
        self.on_change()

    def refresh(self) -> None:
        self._build()
