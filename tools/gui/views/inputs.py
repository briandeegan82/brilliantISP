from __future__ import annotations

import copy
import tkinter as tk
from tkinter import ttk
from typing import Callable

from tools.gui.computed import compute_format_stats
from tools.gui.editing import SPECS
from tools.gui.format_map import margins_from_crop, update_crop_from_margins
from tools.gui.views.fields import FieldRow, ScrollFrame


class FormatPanel(ttk.Frame):
    FIELDS = (
        "width", "height", "bayer_pattern", "bit_depth", "hdr_bit_depth",
        "data_format", "storage_format", "data_alignment", "endian_type", "source_format",
        "header_bytes", "row_stride_bytes", "horizontal_flip", "vertical_flip",
        "embedded_rows_top", "embedded_rows_bottom",
    )

    def __init__(self, parent, model, on_change: Callable[[], None]) -> None:
        super().__init__(parent)
        self.model, self.on_change = model, on_change
        self.rows: dict[str, FieldRow] = {}
        self.scroll = ScrollFrame(self)
        self.scroll.pack(fill="both", expand=True)
        self.refresh()

    def refresh(self) -> None:
        for w in self.scroll.inner.winfo_children():
            w.destroy()
        self.rows.clear()
        sensor = self.model.get("sensor_info", {})
        self.tile = ttk.Frame(self.scroll.inner)
        self.tile.pack(anchor="w", padx=6, pady=4)
        self._tile_pattern = None
        self._update_tile()
        fields = self.FIELDS if isinstance(sensor, dict) else ()
        for name in fields:
            path = "sensor_info." + name
            spec = SPECS.get(path)
            present = name in sensor
            default = spec.default if spec and spec.has_default else None
            row = FieldRow(self.scroll.inner, self.model, path, sensor.get(name, default),
                           present, self.on_change)
            row.pack(fill="x", padx=6, pady=2)
            self.rows[path] = row
        if not fields:
            row = FieldRow(self.scroll.inner, self.model, "sensor_info", sensor, True, self.on_change)
            row.pack(fill="x", padx=6)
            self.rows["sensor_info"] = row
        controls = ttk.Frame(self.scroll.inner)
        controls.pack(fill="x", padx=6, pady=6)
        ttk.Button(controls, text="DG −", command=lambda: self._bump(-1)).pack(side="left")
        ttk.Button(controls, text="DG +", command=lambda: self._bump(1)).pack(side="left")
        self.gain_label = ttk.Label(self.scroll.inner, wraplength=255)
        self.gain_label.pack(anchor="w", padx=6)
        for section, title in (("auto_exposure", "Auto exposure"), ("auto_white_balance", "Auto white balance")):
            value = self.model.get(section + ".is_enable", False)
            row = FieldRow(self.scroll.inner, self.model, section + ".is_enable", value, True, self.on_change)
            row.pack(fill="x", padx=6)
            self.rows[section + ".is_enable"] = row
        self._crop_controls()
        self.stats = ttk.Label(self.scroll.inner, wraplength=255)
        self.stats.pack(anchor="w", padx=6, pady=6)
        self.update_readouts()

    def _crop_controls(self) -> None:
        frame = ttk.LabelFrame(self.scroll.inner, text="Crop margins (decoded Bayer coordinates)")
        frame.pack(fill="x", padx=6, pady=6)
        self.margins: dict[str, tk.StringVar] = {}
        sensor, crop = self.model.get("sensor_info", {}), self.model.get("crop", {})
        self._crop_baseline = copy.deepcopy(crop) if isinstance(crop, dict) else {}
        self._margin_dirty = False
        try:
            height = sensor["height"] - sensor.get("embedded_rows_top", 0) - sensor.get("embedded_rows_bottom", 0)
            values = margins_from_crop(sensor["width"], height, crop)
        except (TypeError, KeyError, ValueError):
            values = (0, 0, 0, 0)
        for name, value in zip(("left", "right", "top", "bottom"), values):
            var = tk.StringVar(value=str(value))
            self.margins[name] = var
            row = ttk.Frame(frame)
            row.pack(fill="x")
            ttk.Label(row, text=name, width=8).pack(side="left")
            ttk.Entry(row, textvariable=var, width=9).pack(side="left")
            var.trace_add("write", lambda *a: self._margin_changed())
        ttk.Button(frame, text="Apply margins as one edit", command=self._apply_margins).pack(anchor="w")
        self.margin_error = ttk.Label(frame, wraplength=240, foreground="#a34713")
        self.margin_error.pack(anchor="w")

    def _apply_margins(self) -> None:
        try:
            values = [int(self.margins[n].get(), 10) for n in ("left", "right", "top", "bottom")]
            if any(v < 0 for v in values):
                raise ValueError("Margins must be nonnegative integers")
            sensor, crop = self.model.get("sensor_info"), self._crop_baseline
            height = sensor["height"] - sensor.get("embedded_rows_top", 0) - sensor.get("embedded_rows_bottom", 0)
            updated = update_crop_from_margins(copy.deepcopy(crop), sensor["width"], height, *values)
            self.model.begin_edit_group("crop_margins")
            self.model.edit("crop", updated, "Set crop margins")
            self.margin_error.configure(text="")
            self.on_change()
        except (TypeError, ValueError, KeyError) as exc:
            self.margin_error.configure(text=str(exc))
            self.model.begin_edit_group("crop_margins")
            self.model.edit("crop", "Invalid crop margins: " + repr(
                {name: var.get() for name, var in self.margins.items()}), "Edit crop margins")
            self.on_change()

    def _margin_changed(self) -> None:
        self._margin_dirty = True
        self._apply_margins()

    def _bump(self, delta: int) -> None:
        current, gains = self.model.get("digital_gain.current_gain"), self.model.get("digital_gain.gain_array")
        if isinstance(current, bool) or not isinstance(current, int) or not isinstance(gains, list) or not gains:
            self.margin_error.configure(text="Repair digital_gain.current_gain / gain_array first")
            return
        self.model.end_edit_group()
        self.model.edit("digital_gain.current_gain", max(0, min(len(gains) - 1, current + delta)), "Digital gain shortcut")
        self.on_change()
        self.update_readouts()

    def flush_all(self) -> None:
        for row in self.rows.values():
            row.commit(flush=True)
        if self._margin_dirty:
            self._apply_margins()
        self.model.end_edit_group()

    def update_readouts(self) -> None:
        self._update_tile()
        gain = self.model.get("digital_gain.current_gain", "?")
        reason = "" if self.model.get("digital_gain.is_enable", False) else " (digital gain disabled)"
        self.gain_label.configure(text=f"Manual gain index: {gain}{reason}")
        try:
            errors = [i for i in self.model.validation_issues() if i.severity == "error"
                      and i.path.split(".")[0] in ("sensor_info", "crop", "scale")]
            if errors:
                self.stats.configure(text="Layout estimates unavailable until format/crop/scale errors are fixed.")
                return
            stats = compute_format_stats(self.model.config)
            text = stats["format_error"] or (
                f"Decoded active: {stats['effective_width']} × {stats['effective_height']}\n"
                f"Declared RAW size: {stats['estimated_raw_size']:,} bytes\n"
                f"Output: {stats['output_width']} × {stats['output_height']}")
            self.stats.configure(text=text)
        except (TypeError, ValueError, AttributeError) as exc:
            self.stats.configure(text=f"Invalid format: {exc}")

    def _update_tile(self) -> None:
        pattern = str(self.model.get("sensor_info.bayer_pattern", "")).lower()
        if pattern == self._tile_pattern:
            return
        self._tile_pattern = pattern
        for child in self.tile.winfo_children():
            child.destroy()
        colors = {"r": "#b94f49", "g": "#438d6c", "b": "#547caa"}
        ttk.Label(self.tile, text="Declared CFA").grid(row=0, column=0, columnspan=2)
        for i, channel in enumerate(pattern[:4]):
            tk.Label(self.tile, text=channel.upper(), width=3, bg=colors.get(channel, "#777777"),
                     fg="white").grid(row=i // 2 + 1, column=i % 2)
