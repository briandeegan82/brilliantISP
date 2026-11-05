from __future__ import annotations

import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from typing import Callable


class AnalysisInspector(ttk.Frame):
    def __init__(self, parent, controller, on_error: Callable[[str], None]) -> None:
        super().__init__(parent)
        self.controller, self.on_error = controller, on_error
        self.directory: str | None = None
        self.comparison_dir: str | None = None
        self.roi: list[int] | None = None
        self._requested_id: str | None = None
        self._seen: str | None = None
        self._descriptor: dict | None = None
        self.preview_geometry: tuple[int, int] | None = None
        self._is_raw = False
        self.stage = tk.StringVar(value="preview")
        row = ttk.Frame(self)
        row.pack(fill="x", padx=5, pady=4)
        self.selector = ttk.Combobox(row, textvariable=self.stage, values=("preview", "native"),
                                     state="readonly", width=23)
        self.selector.pack(side="left", fill="x", expand=True)
        self.selector.bind("<<ComboboxSelected>>", lambda e: self.request())
        ttk.Button(row, text="Analyze", command=self.request).pack(side="left")
        self.compare_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(self, text="Compare with reference when registered", variable=self.compare_enabled,
                        command=self.request).pack(anchor="w", padx=5)
        self.heading = tk.StringVar(value="Select a processed result or RAW source.")
        ttk.Label(self, textvariable=self.heading, wraplength=290).pack(anchor="w", padx=5)
        self.roi_text = tk.StringVar(value="ROI: full frame")
        ttk.Label(self, textvariable=self.roi_text).pack(anchor="w", padx=5)
        ttk.Button(self, text="Analyze full frame", command=self.clear_roi).pack(anchor="w", padx=5)
        self.picker = tk.StringVar(value="Native picker: move the pointer over the image.")
        ttk.Label(self, textvariable=self.picker, wraplength=290).pack(anchor="w", padx=5, pady=4)
        table = ttk.Frame(self)
        table.pack(fill="x", padx=5)
        self.table = ttk.Treeview(table, columns=("min", "mean", "std", "max", "under", "over"),
                                  show="tree headings", height=3)
        self.table.heading("#0", text="Component")
        self.table.column("#0", width=72, stretch=False)
        for name, label in (("min", "Min"), ("mean", "Mean"), ("std", "Population σ"),
                            ("max", "Max"), ("under", "< nominal"), ("over", "> nominal")):
            self.table.heading(name, text=label)
            self.table.column(name, width=79, stretch=False)
        self.table.pack(fill="x")
        scrollbar = ttk.Scrollbar(table, orient="horizontal", command=self.table.xview)
        scrollbar.pack(fill="x")
        self.table.configure(xscrollcommand=scrollbar.set)
        self.notes = tk.StringVar()
        ttk.Label(self, textvariable=self.notes, wraplength=290).pack(anchor="w", padx=5, pady=4)
        self.difference = tk.StringVar()
        ttk.Label(self, textvariable=self.difference, wraplength=290).pack(anchor="w", padx=5)
        self.plot_frame = ttk.Frame(self)
        self.plot_frame.pack(fill="both", expand=True)
        self._canvas = None
        self._hist = None
        self._curve = None
        self._hist_lines = []
        self._curve_lines = []

    def set_frame(self, directory: str | None, *, comparison_dir: str | None = None) -> None:
        self.directory, self.comparison_dir = directory, comparison_dir
        self._requested_id, self._seen, self._descriptor = None, None, None
        self.heading.set("Analysis not yet requested for this snapshot.")
        self.difference.set("")
        self.notes.set("")
        for key in self.table.get_children():
            self.table.delete(key)
        for line in self._hist_lines + self._curve_lines:
            line.set_data([], [])
        if self._canvas is not None:
            self._canvas.draw_idle()
        if directory is None:
            return
        manifest = json.loads((Path(directory) / "result.json").read_text(encoding="utf-8"))
        self._is_raw = manifest.get("kind") == "raw"
        stages = ["preview", "native"] + [item["stage"] for item in manifest["files"].get("captures", [])]
        self.selector.configure(values=stages)
        if self.stage.get() not in stages:
            self.stage.set("preview")

    def set_roi(self, roi) -> None:
        self.roi = list(roi) if roi is not None else None
        self.roi_text.set(f"ROI [x0,y0,x1,y1), native pixels: {self.roi}" if roi else "ROI: full frame")
        self.request()

    def clear_roi(self) -> None:
        self.set_roi(None)

    def request(self) -> None:
        if self.directory is None:
            return
        if self.roi is not None and self.preview_geometry is not None:
            from tools.gui.artifacts import load_artifact
            stage = "native" if self._is_raw and self.stage.get() == "preview" else self.stage.get()
            _, metadata = load_artifact(self.directory, stage)
            if (metadata["width"], metadata["height"]) != self.preview_geometry:
                self.roi = None
                self.roi_text.set("ROI cleared: selected capture uses a different grid; analyzing full frame.")
        comparison = self.comparison_dir if self.compare_enabled.get() else None
        desired = {"manifest_dir": self.directory, "stage": self.stage.get(),
                   "roi": self.roi, "comparison_dir": comparison}
        result = self.controller.analysis_result
        if self._descriptor == desired and result and result.get("request_id") == self._requested_id:
            return
        req = self.controller.request_analysis(self.directory, self.stage.get(), self.roi, comparison)
        if req is not None:
            self._requested_id = req.tag.request_id
            self._descriptor = json.loads(req.payload_json)
            self.heading.set("Analysis queued; previous values are not current.")

    def refresh(self) -> None:
        result = self.controller.analysis_result
        if (result is None or result.get("request_id") != self._requested_id
                or result.get("request_id") == self._seen or result.get("descriptor") != self._descriptor):
            return
        self._seen = result["request_id"]
        try:
            response = json.loads(Path(result["analysis_path"]).read_text(encoding="utf-8"))
            self._present(response)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            self.on_error(f"Analysis display failed: {exc}")

    def _present(self, response: dict) -> None:
        stats, metadata = response["stats"], response["metadata"]
        units = metadata.get("value_units", "")
        self.heading.set(f"{stats['domain']} | {units} | {stats['pixel_count']:,} pixels")
        names = stats["component_names"]
        for index, component in enumerate(stats["components"]):
            key = str(index)
            values = tuple(f"{component[name]:.7g}" for name in (
                "minimum", "mean", "population_std", "maximum", "below_nominal", "above_nominal"))
            if self.table.exists(key):
                self.table.item(key, text=names[index], values=values)
            else:
                self.table.insert("", "end", iid=key, text=names[index], values=values)
        for key in self.table.get_children():
            if int(key) >= len(names):
                self.table.delete(key)
        notes = [response.get("note", ""), stats["histogram_note"], stats["limit_note"]]
        if response.get("curve"):
            notes.extend([response["curve"]["label"], response["curve"]["caveat"]])
        if "encoded_luma" in stats:
            luma = stats["encoded_luma"]
            notes.extend([luma["label"],
                          f"Encoded luma mean {luma['mean']:.7g}; P1–P99 code span {luma['percentile_code_span']:.7g}.",
                          luma["span_note"]])
        self.notes.set("\n".join(note for note in notes if note))
        difference = response.get("difference")
        self.difference.set("" if not difference else (
            f"Registered difference ({units}): {difference['changed_pixels']:,}/{difference['pixel_count']:,} pixels changed\n"
            f"Mean absolute {difference['mean_absolute']:.7g}; RMSE {difference['rmse']:.7g}; "
            f"max absolute {difference['maximum_absolute']:.7g}\n{difference['note']}"))
        self._plot(response)

    def _init_plot(self) -> None:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        figure = Figure(figsize=(3.3, 4), dpi=100)
        self._hist, self._curve = figure.subplots(2, 1)
        self._hist_lines = [self._hist.plot([], [], color=color, drawstyle="steps-post", linewidth=1)[0]
                            for color in ("#b24740", "#30826a", "#4b75ab")]
        self._curve_lines = [self._curve.plot([], [], color=color, label=label)[0]
                            for color, label in (("#30826a", "Display linear"), ("#4b75ab", "After transfer"))]
        self._curve.legend(fontsize=7)
        figure.subplots_adjust(left=.2, right=.96, top=.93, bottom=.13, hspace=.65)
        self._canvas = FigureCanvasTkAgg(figure, master=self.plot_frame)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

    def _plot(self, response: dict) -> None:
        if self._canvas is None:
            self._init_plot()
        assert self._hist is not None and self._curve is not None and self._canvas is not None
        stats = response["stats"]
        for index, line in enumerate(self._hist_lines):
            if index >= len(stats["components"]):
                line.set_data([], [])
                continue
            component = stats["components"][index]
            line.set_color("#777777" if len(stats["components"]) == 1 else ("#b24740", "#30826a", "#4b75ab")[index])
            if "histogram_codes" in component:
                line.set_data(component["histogram_codes"] + [component["histogram_codes"][-1] + 1],
                              component["histogram"] + [component["histogram"][-1]])
            else:
                line.set_data(component["histogram_edges"], component["histogram"] + [component["histogram"][-1]])
        self._hist.set_title("Nominal-range histogram", fontsize=9)
        self._hist.set_xlabel(response["metadata"]["value_units"], fontsize=8)
        self._hist.set_ylabel("Samples / bin", fontsize=8)
        self._hist.relim()
        self._hist.autoscale_view()
        curve = response.get("curve")
        for name, line in zip(("display_linear", "after_transfer"), self._curve_lines):
            line.set_data(curve["x"], curve[name]) if curve else line.set_data([], [])
        self._curve.set_title("Unit-ramp illustration, not frame response" if curve else "No RGB curve for sensor codes", fontsize=8)
        self._curve.set_xlabel("Normalized input", fontsize=8)
        self._curve.set_ylabel("Normalized output", fontsize=8)
        self._curve.relim()
        self._curve.autoscale_view()
        self._canvas.draw_idle()
