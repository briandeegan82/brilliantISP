"""Stable single-canvas RGB preview (artists updated, never refit)."""
from __future__ import annotations

import math
from typing import Any, Callable
import numpy as np
from tools.gui.analysis import normalize_roi


class PreviewPane:
    def __init__(self, parent) -> None:
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.figure.set_facecolor("#172126")
        self.figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_axis_off()
        self.ax.set_facecolor("#172126")
        self._artist = None
        self._shape: tuple[int, int] | None = None
        self._geometry_key: str | None = None
        self._secondary = None
        self._artist_secondary = None
        self._secondary_shape = None
        self._secondary_key = None
        self._pair = False
        self._link_allowed = False
        self.on_roi: Callable[[Any, str], None] | None = None
        self.on_picker: Callable[[float, float, str], None] | None = None
        self._roi_start = None
        self._rectangles = {}
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.widget = self.canvas.get_tk_widget()
        bar = ttk.Frame(parent)
        bar.pack(side="top", fill="x")
        self.widget.pack(side="top", fill="both", expand=True)
        ttk.Button(bar, text="Fit", command=self.fit).pack(side="left")
        ttk.Button(bar, text="1:1", command=self.actual_size).pack(side="left")
        self.roi_mode = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="Select ROI", variable=self.roi_mode).pack(side="left")
        ttk.Button(bar, text="Clear ROI", command=lambda: self.clear_roi(notify=True)).pack(side="left")
        self.linked = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Link registered views", variable=self.linked).pack(side="left")
        bar.lift()
        self._pan_start: tuple | None = None
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)

    def show_rgb8(self, image: np.ndarray, *, geometry_key: str | None = None) -> None:
        if self._secondary is not None:
            self._secondary.set_visible(False)
        self._pair = False
        self.ax.set_position((0, 0, 1, 1))
        self._artist, self._shape, self._geometry_key = self._show(
            self.ax, self._artist, image, self._shape, self._geometry_key, geometry_key)
        self.canvas.draw_idle()

    def _show(self, ax, artist, image, shape, previous_key, geometry_key):
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("preview requires HWC3 uint8 RGB")
        h, w = image.shape[:2]
        same = shape == (h, w) and artist is not None and previous_key == geometry_key
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        if artist is None:
            artist = ax.imshow(
                image, extent=(0, w, h, 0), interpolation="nearest")
        else:
            artist.set_data(image)
            artist.set_extent((0, w, h, 0))
            artist.set_visible(True)
        if same:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        else:
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
            self.clear_roi()
        return artist, (h, w), geometry_key

    def show_pair(self, a, b, *, geometry_a: str, geometry_b: str, registered: bool) -> None:
        self._pair, self._link_allowed = True, registered
        self.ax.set_position((0, 0, .497, 1))
        if self._secondary is None:
            self._secondary = self.figure.add_axes((.503, 0, .497, 1))
            self._secondary.set_axis_off()
            self._secondary.set_facecolor("#172126")
        self._secondary.set_visible(True)
        self._artist, self._shape, self._geometry_key = self._show(
            self.ax, self._artist, a, self._shape, self._geometry_key, geometry_a)
        self._artist_secondary, self._secondary_shape, self._secondary_key = self._show(
            self._secondary, self._artist_secondary, b, self._secondary_shape, self._secondary_key, geometry_b)
        self.canvas.draw_idle()

    def clear(self) -> None:
        for artist in (self._artist, self._artist_secondary):
            if artist is not None:
                artist.set_data(np.zeros((1, 1, 3), dtype=np.uint8))
                artist.set_visible(False)
        self._shape = self._secondary_shape = None
        self._geometry_key = self._secondary_key = None
        self.clear_roi()
        self.canvas.draw_idle()

    def _axes(self):
        axes = [(self.ax, self._shape)]
        if self._pair and self._secondary is not None:
            axes.append((self._secondary, self._secondary_shape))
        return axes

    def _fit_axes(self) -> None:
        for ax, shape in self._axes():
            if shape is not None:
                h, w = shape
                ax.set_xlim(0, w)
                ax.set_ylim(h, 0)

    def fit(self) -> None:
        self._fit_axes()
        self.canvas.draw_idle()

    def actual_size(self) -> None:
        for ax, shape in self._axes():
            if shape is None:
                continue
            bbox = ax.get_window_extent()
            cx, cy = sum(ax.get_xlim()) / 2, sum(ax.get_ylim()) / 2
            ax.set_xlim(cx - bbox.width / 2, cx + bbox.width / 2)
            ax.set_ylim(cy + bbox.height / 2, cy - bbox.height / 2)
        self.canvas.draw_idle()

    def viewport(self) -> dict:
        return {"xlim": list(self.ax.get_xlim()),
                "ylim": list(self.ax.get_ylim())}

    def restore_viewport(self, state: dict) -> None:
        try:
            limits = state["xlim"] + state["ylim"]
            if len(limits) != 4 or not all(math.isfinite(float(v)) for v in limits):
                return
            self.ax.set_xlim(state["xlim"])
            self.ax.set_ylim(state["ylim"])
            self.canvas.draw_idle()
        except Exception:
            pass

    def _on_scroll(self, event) -> None:
        ax = event.inaxes
        if self._shape is None or ax is None or ax not in [a for a, _ in self._axes()] or event.xdata is None or event.ydata is None:
            return
        factor = 1.25 if event.button == "up" else 0.8
        x, y = ax.get_xlim(), ax.get_ylim()
        cx, cy = event.xdata, event.ydata
        ax.set_xlim(cx - (cx - x[0]) / factor, cx + (x[1] - cx) / factor)
        ax.set_ylim(cy - (cy - y[0]) / factor, cy + (y[1] - cy) / factor)
        self._sync_views(ax)
        self.canvas.draw_idle()

    def _on_press(self, event) -> None:
        if event.xdata is None or event.ydata is None:
            return
        if event.button == 1 and self.roi_mode.get():
            self._roi_start = (event.inaxes, event.xdata, event.ydata)
        elif event.button in (2, 3):
            self._pan_start = (
                event.inaxes, event.x, event.y, event.inaxes.get_xlim(), event.inaxes.get_ylim(),
                event.inaxes.transData.frozen().inverted())

    def _on_release(self, event) -> None:
        self._pan_start = None
        if self._roi_start is None:
            return
        ax, sx, sy = self._roi_start
        self._roi_start = None
        ex, ey = ax.transData.inverted().transform((event.x, event.y))
        bounds = (math.floor(min(sx, ex)), math.floor(min(sy, ey)),
                  math.ceil(max(sx, ex)), math.ceil(max(sy, ey)))
        shape = self._shape if ax is self.ax else self._secondary_shape
        if shape is None:
            return
        try:
            bounds = normalize_roi(bounds, shape[1], shape[0])
        except ValueError:
            return
        self._draw_roi(ax, bounds)
        if self.on_roi is not None:
            self.on_roi(bounds, self._slot(ax))

    def _on_motion(self, event) -> None:
        if self._pan_start is not None:
            ax, sx, sy, xlim, ylim, inverse = self._pan_start
            start, end = inverse.transform((sx, sy)), inverse.transform((event.x, event.y))
            dx, dy = start - end
            ax.set_xlim(xlim[0] + dx, xlim[1] + dx)
            ax.set_ylim(ylim[0] + dy, ylim[1] + dy)
            self._sync_views(ax)
            self.canvas.draw_idle()
        if self._roi_start is not None and event.inaxes is self._roi_start[0] and event.xdata is not None and event.ydata is not None:
            ax, sx, sy = self._roi_start
            self._draw_roi(ax, (min(sx, event.xdata), min(sy, event.ydata),
                                max(sx, event.xdata), max(sy, event.ydata)))
        if self.on_picker is not None and event.inaxes in [a for a, _ in self._axes()] and event.xdata is not None and event.ydata is not None:
            self.on_picker(event.xdata, event.ydata, self._slot(event.inaxes))

    def _slot(self, ax) -> str:
        return "B" if self._pair and ax is self._secondary else "A" if self._pair else "current"

    def _sync_views(self, ax) -> None:
        if self._pair and self._link_allowed and self.linked.get():
            other = self._secondary if ax is self.ax else self.ax
            if other is not None:
                other.set_xlim(ax.get_xlim())
                other.set_ylim(ax.get_ylim())

    def _draw_roi(self, ax, bounds) -> None:
        from matplotlib.patches import Rectangle
        x0, y0, x1, y1 = bounds
        rect = self._rectangles.get(ax)
        if rect is None:
            rect = Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#42d4c6", linewidth=1.2)
            self._rectangles[ax] = rect
            ax.add_patch(rect)
        else:
            rect.set_bounds(x0, y0, x1 - x0, y1 - y0)
            rect.set_visible(True)
        self.canvas.draw_idle()

    def clear_roi(self, *, notify: bool = False) -> None:
        self._roi_start = None
        for rectangle in self._rectangles.values():
            rectangle.set_visible(False)
        if notify and self.on_roi is not None:
            self.on_roi(None, "current")
