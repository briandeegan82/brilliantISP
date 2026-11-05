"""Batch queue panel: frozen snapshots, serial run, retry."""
from __future__ import annotations

from pathlib import Path
from tkinter import filedialog, ttk
from typing import Callable


class BatchPanel(ttk.Frame):
    COLS = ("label", "source", "status")

    def __init__(self, parent, controller,
                 on_result: Callable[[str], None],
                 on_status: Callable[[str], None]) -> None:
        super().__init__(parent)
        self.controller = controller
        self.on_result = on_result
        self.on_status = on_status
        self.tree = ttk.Treeview(self, columns=self.COLS,
                                 show="headings", height=12)
        widths = {"label": 72, "source": 85, "status": 70}
        for c in self.COLS:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=widths[c], stretch=False)
        hscroll = ttk.Scrollbar(self, orient="horizontal",
                                command=self.tree.xview)
        self.tree.configure(xscrollcommand=hscroll.set)
        self.tree.pack(fill="both", expand=True)
        hscroll.pack(fill="x")
        self.tree.bind("<<TreeviewSelect>>", self._select)
        bar = ttk.Frame(self)
        bar.pack(fill="x")
        buttons = (("Add current", self._add_current),
                   ("Add preset…", self._add_preset),
                   ("Start", self._start),
                   ("Pause", self._pause),
                   ("Cancel", self._cancel),
                   ("Retry", self._retry))
        for i, (label, cmd) in enumerate(buttons):
            ttk.Button(bar, text=label, command=cmd).grid(
                row=i // 3, column=i % 3, sticky="ew")
        for c in range(3):
            bar.columnconfigure(c, weight=1)

    def _add_current(self) -> None:
        if self.controller.enqueue_batch(label="current") is None:
            self.on_status("batch enqueue rejected (invalid config or full)")
        self.refresh()

    def _add_preset(self) -> None:
        from tools.gui.yaml_compat import load_preset_yaml
        path = filedialog.askopenfilename(parent=self)
        if not path:
            return
        try:
            cfg = load_preset_yaml(Path(path))
        except Exception as exc:
            self.on_status(f"preset load failed: {exc}")
            return
        if self.controller.enqueue_batch(config_snapshot=cfg,
                                         label=Path(path).stem) is None:
            self.on_status("batch enqueue rejected (invalid config or full)")
        self.refresh()

    def _start(self) -> None:
        self.controller.start_batch()

    def _pause(self) -> None:
        self.controller.pause_batch()

    def _cancel(self) -> None:
        self.controller.cancel()

    def _retry(self) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        item = next((i for i in self.controller.batch
                     if i["batch_id"] == sel[0]), None)
        if item is not None:
            self.controller.retry_batch(item["batch_id"])
        self.refresh()

    def _select(self, _e) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        item = next((i for i in self.controller.batch
                     if i["batch_id"] == sel[0]), None)
        if item is not None and item.get("result_path"):
            self.on_result(str(Path(item["result_path"]).parent))

    def refresh(self) -> None:
        existing = set(self.tree.get_children())
        for item in self.controller.batch:
            values = (item["label"],
                      Path(item["source"].get("path", "")).name,
                      item["status"])
            if item["batch_id"] in existing:
                self.tree.item(item["batch_id"], values=values)
            else:
                self.tree.insert("", "end", iid=item["batch_id"],
                                 values=values)
        alive = {i["batch_id"] for i in self.controller.batch}
        for stale in existing - alive:
            self.tree.delete(stale)
