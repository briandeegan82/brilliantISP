"""Schema-driven parameter editor (right panel)."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable

import yaml

from util.config_schema import spec_map


def parse_numeric(text: str, current: Any) -> Any:
    if isinstance(current, bool):
        raise ValueError("bool not numeric")
    if isinstance(current, int):
        return int(text, 0)
    if isinstance(current, float):
        return float(text)
    raise ValueError("not a numeric parameter")


class BlockEditor(ttk.Frame):
    """Edits one config section; every change is a model.edit()."""

    def __init__(self, parent, model, section: str,
                 on_change: Callable[[], None]) -> None:
        super().__init__(parent)
        self.model = model
        self.section = section
        self.on_change = on_change
        self._sync = False
        self._pending: dict[str, tuple[Any, Any]] = {}
        canvas = tk.Canvas(self, highlightthickness=0)
        scroll = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self._inner = ttk.Frame(canvas)
        self._inner.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self._inner, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self._build()

    def _spec(self, path: str):
        try:
            specs = spec_map()
        except Exception:
            return None
        return specs.get(path)

    def _build(self) -> None:
        self._sync = True
        for w in self._inner.winfo_children():
            w.destroy()
        self._pending.clear()
        block = self.model.config.get(self.section)
        if not isinstance(block, dict):
            ttk.Label(self._inner,
                      text=f"{self.section}: not editable").pack(anchor="w")
            self._sync = False
            return
        self._rows(block, self.section)
        ttk.Separator(self._inner).pack(fill="x", pady=6)
        bar = ttk.Frame(self._inner)
        bar.pack(fill="x")
        ttk.Button(bar, text="Reset module",
                   command=lambda: self._reset_module()).pack(side="left")
        self._sync = False

    def _rows(self, block: dict, prefix: str) -> None:
        for key, value in block.items():
            path = f"{prefix}.{key}"
            spec = self._spec(path)
            choices = getattr(spec, "choices", None)
            if isinstance(value, dict):
                ttk.Label(self._inner, text=key,
                          font="TkHeadingFont").pack(anchor="w", padx=6)
                self._rows(value, path)
            elif choices:
                self._choice_row(key, value, path, choices)
            elif isinstance(value, bool):
                var = tk.BooleanVar(value=value)
                ttk.Checkbutton(self._inner, text=key, variable=var,
                                command=lambda v=var, p=path, c=value:
                                self._emit(p, v.get(), c)).pack(anchor="w",
                                                                padx=12)
            elif isinstance(value, (int, float)):
                self._numeric_row(key, value, path)
            else:
                self._yaml_row(key, value, path)

    def _choice_row(self, key: str, value: Any, path: str,
                    choices: list) -> None:
        row = ttk.Frame(self._inner)
        row.pack(fill="x")
        ttk.Label(row, text=key, width=22).pack(side="left")
        cb = ttk.Combobox(row, values=list(choices), width=18,
                          state="readonly")
        cb.set(str(value))
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>",
                lambda e: self._emit(path, cb.get(), value))

    def _numeric_row(self, key: str, value: Any, path: str) -> None:
        row = ttk.Frame(self._inner)
        row.pack(fill="x")
        ttk.Label(row, text=key, width=22).pack(side="left")
        ent = ttk.Entry(row, width=16)
        ent.insert(0, str(value))
        ent.pack(side="left")
        ent.bind("<KeyRelease>", lambda e, en=ent, p=path, c=value:
                 self._emit_numeric(en, p, c))

    def _yaml_row(self, key: str, value: Any, path: str) -> None:
        row = ttk.Frame(self._inner)
        row.pack(fill="x")
        ttk.Label(row, text=key, width=22).pack(side="left")
        text = tk.Text(row, width=26, height=3)
        try:
            text.insert("1.0", yaml.safe_dump(value, default_flow_style=True))
        except Exception:
            text.insert("1.0", str(value))
        text.pack(side="left")
        text.bind("<<Modified>>", lambda e, t=text, p=path:
                  self._emit_yaml(t, p))

    def _emit(self, path: str, value: Any, current: Any) -> None:
        if self._sync:
            return
        self.model.edit(path, value)
        self.on_change()

    def _emit_numeric(self, entry: ttk.Entry, path: str, current: Any) -> None:
        if self._sync:
            return
        raw = entry.get().strip()
        try:
            self.model.edit(path, parse_numeric(raw, current))
        except ValueError:
            self.model.edit(path, raw)
        self.on_change()

    def _emit_yaml(self, text: tk.Text, path: str) -> None:
        if self._sync or not text.edit_modified():
            return
        text.edit_modified(False)
        raw = text.get("1.0", "end").strip()
        try:
            self.model.edit(path, yaml.safe_load(raw))
        except yaml.YAMLError:
            self.model.edit(path, raw)
        self.on_change()

    def _reset_module(self) -> None:
        self.model.reset_module(self.section)
        self.on_change()
        self.refresh()

    def refresh(self) -> None:
        self._build()

    def flush_all(self) -> None:
        self._sync = True
        for w in self._inner.winfo_children():
            w.event_generate("<FocusOut>")
        self._sync = False


class ParametersPanel(ttk.Frame):
    def __init__(self, parent, model,
                 on_change: Callable[[], None]) -> None:
        super().__init__(parent)
        self.model = model
        self.on_change = on_change
        self._editor: BlockEditor | None = None

    def show_section(self, section: str) -> None:
        if self._editor is not None:
            self._editor.destroy()
        self._editor = BlockEditor(self, self.model, section, self.on_change)
        self._editor.pack(fill="both", expand=True)

    def refresh(self) -> None:
        if self._editor is not None:
            self._editor.refresh()

    def flush_all(self) -> None:
        if self._editor is not None:
            self._editor.flush_all()
