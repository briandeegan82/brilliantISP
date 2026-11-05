from __future__ import annotations

from pathlib import Path
from tkinter import ttk
from typing import Callable
import webbrowser

from tools.gui.editing import SPECS, visible_fields
from tools.gui.views.fields import FieldRow, ScrollFrame


class ParametersPanel(ttk.Frame):
    def __init__(self, parent, model, on_change: Callable[[], None]) -> None:
        super().__init__(parent)
        self.model, self.on_change = model, on_change
        self.section: str | None = None
        self.rows: dict[str, FieldRow] = {}
        self._editor = self
        self.title = ttk.Label(self, text="Select a pipeline block", font="TkHeadingFont", wraplength=270)
        self.title.pack(anchor="w", padx=6, pady=4)
        self.reason = ttk.Label(self, wraplength=270, foreground="#936500")
        self.reason.pack(anchor="w", padx=6)
        self.scroll = ScrollFrame(self)
        self.scroll.pack(fill="both", expand=True)
        bar = ttk.Frame(self)
        bar.pack(fill="x")
        ttk.Button(bar, text="Reset block", command=self._reset).pack(side="left")
        ttk.Button(bar, text="Documentation", command=self._docs).pack(side="left")

    def show_section(self, section: str) -> None:
        self.flush_all()
        self.section = section
        self.refresh()

    def refresh(self) -> None:
        for child in self.scroll.inner.winfo_children():
            child.destroy()
        self.rows.clear()
        if self.section is None:
            return
        cfg = self.model.config
        self.title.configure(text=self.section.replace("_", " ").title())
        enabled = self.model.get(f"{self.section}.is_enable", True)
        reason = "Block disabled: values are editable but apply only when enabled." if enabled is False else ""
        if self.section == "tone_mapping" and cfg.get("schema_version") == 2:
            reason += " Switching operator applies its documented defaults as one undoable edit."
        self.reason.configure(text=reason)
        for path, value, present in visible_fields(self.section, cfg):
            choices = None
            if path == "tone_mapping.backend" and cfg.get("schema_version") == 2:
                from modules.tone_mapping.registry import OPERATOR_BACKENDS
                choices = sorted(OPERATOR_BACKENDS.get(self.model.get("tone_mapping.operator"), ()))
            row = FieldRow(self.scroll.inner, self.model, path, value, present, self.on_change, choices=choices)
            row.pack(fill="x", padx=6, pady=3)
            self.rows[path] = row
        if not self.rows:
            ttk.Label(self.scroll.inner, text="Read-only execution tap. Capture it on the next run for native analysis.",
                      wraplength=265).pack(anchor="w", padx=6)

    def flush_all(self) -> None:
        for row in self.rows.values():
            row.commit(flush=True)
        self.model.end_edit_group()

    def show_issues(self, issues) -> None:
        for path, row in self.rows.items():
            row.error_var.set("\n".join(i.message for i in issues if i.path == path or i.path.startswith(path + ".")))

    def _reset(self) -> None:
        if self.section:
            self.model.end_edit_group()
            self.model.reset_module(self.section)
            self.refresh()
            self.on_change()

    def _docs(self) -> None:
        if self.section is None:
            return
        page = Path(__file__).resolve().parents[3] / "docs" / "blocks" / f"{self.section}.md"
        if not page.is_file():
            page = page.parent.parent / "GUI_DESIGN.md"
        webbrowser.open(page.as_uri())
