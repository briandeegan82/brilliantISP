from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable

import yaml

from tools.gui.editing import SPECS, field_kind, parse_draft


class ScrollFrame(ttk.Frame):
    def __init__(self, parent) -> None:
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0, width=240)
        self.inner = ttk.Frame(self.canvas)
        window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(window, width=e.width))
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        for event in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.bind_all(event, self._wheel, add="+")

    def _wheel(self, event):
        if not str(event.widget).startswith(str(self)):
            return None
        steps = -1 if getattr(event, "num", None) == 4 else 1 if getattr(event, "num", None) == 5 else (
            -1 if event.delta > 0 else 1)
        self.canvas.yview_scroll(steps, "units")
        return "break"


class FieldRow(ttk.Frame):
    def __init__(self, parent, model, path: str, value: Any, present: bool,
                 on_change: Callable[[], None], *, choices=None) -> None:
        super().__init__(parent)
        self.model, self.path, self.original = model, path, value
        self.on_change, self.present = on_change, present
        self._initial = True
        self._dirty = False
        self._text: tk.Text | None = None
        spec = SPECS.get(path)
        kind = field_kind(path, value)
        label = path.split(".", 1)[-1] + (" (default)" if not present else "")
        ttk.Label(self, text=label, wraplength=250).pack(anchor="w")
        editor = ttk.Frame(self)
        editor.pack(fill="x", pady=(1, 1))
        self.var: tk.Variable
        if kind == "bool" and isinstance(value, bool):
            self.var = tk.BooleanVar(value=value)
            widget = ttk.Checkbutton(editor, text="Enabled / true", variable=self.var, command=self.commit)
        elif choices or (spec and spec.choices):
            self.var = tk.StringVar(value=str(value))
            options = choices or (spec.choices if spec else ())
            widget = ttk.Combobox(editor, textvariable=self.var, values=list(options), state="readonly", width=15)
            widget.bind("<<ComboboxSelected>>", lambda e: self.commit())
        elif kind in ("matrix", "list", "array", "yaml") or isinstance(value, (list, tuple, dict)):
            self.var = tk.StringVar()
            widget = tk.Text(editor, height=3, width=20, undo=True, wrap="none")
            widget.insert("1.0", yaml.safe_dump(value, default_flow_style=False, sort_keys=False))
            widget.edit_modified(False)
            self._text = widget
            widget.bind("<<Modified>>", self._text_changed)
        else:
            self.var = tk.StringVar(value="" if value is None else str(value))
            widget = ttk.Entry(editor, textvariable=self.var, width=12)
            self.var.trace_add("write", lambda *a: self.commit())
        self.input = widget
        widget.pack(side="left", fill="x", expand=True)
        widget.bind("<FocusIn>", lambda e: self.model.begin_edit_group(path), add="+")
        widget.bind("<FocusOut>", lambda e: self.model.end_edit_group(path), add="+")
        ttk.Button(editor, text="Reset", width=5, command=self.reset).pack(side="right", padx=(3, 0))
        help_text = ((spec.help if spec else "") + (" [" + spec.units + "]" if spec and spec.units else "")).strip()
        if path.endswith((".is_save", ".is_debug", ".is_plot_curve")):
            help_text = "Preview jobs override diagnostics flags. This setting is retained for preset export."
        self.help_var = tk.StringVar(value=help_text)
        if help_text:
            ttk.Label(self, textvariable=self.help_var, wraplength=265, foreground="#4b6067").pack(anchor="w")
        self.error_var = tk.StringVar()
        self.error_label = ttk.Label(self, textvariable=self.error_var, wraplength=265, foreground="#a34713")
        self.error_var.trace_add("write", lambda *a: self._show_error())
        self._initial = False

    def _show_error(self) -> None:
        if self.error_var.get():
            self.error_label.pack(anchor="w")
        else:
            self.error_label.pack_forget()

    def _text_changed(self, event) -> None:
        if self._text is not None and self._text.edit_modified():
            self._text.edit_modified(False)
            self.commit()

    def commit(self, *, flush: bool = False) -> None:
        if self._initial:
            return
        if flush and not self._dirty and not (self._text is not None and self._text.edit_modified()):
            return
        value = self.var.get()
        if self._text is not None:
            raw = self._text.get("1.0", "end-1c")
            try:
                value = yaml.safe_load(raw)
            except yaml.YAMLError:
                value = raw
        elif not isinstance(self.var, tk.BooleanVar):
            value = parse_draft(self.path, str(value), self.original)
        before = self.model.revision
        if self.present and value == self.model.get(self.path):
            return
        self._dirty = True
        try:
            if not flush:
                if isinstance(self.var, tk.BooleanVar) or isinstance(self.input, ttk.Combobox):
                    self.model.end_edit_group()
                else:
                    self.model.begin_edit_group(self.path)
            if self.path == "auto_white_balance.is_enable":
                self.model.edit_many({self.path: value, "white_balance.is_auto": value}, "Toggle automatic white balance")
            elif self.path == "tone_mapping.operator":
                from modules.tone_mapping.registry import OPERATOR_BACKENDS, OPERATOR_PARAMETERS
                if value in OPERATOR_PARAMETERS and value != self.model.get(self.path):
                    backend = self.model.get("tone_mapping.backend")
                    if backend not in OPERATOR_BACKENDS[value]:
                        backend = "cpu" if value == "durand" else "float"
                    self.model.edit_many({
                        self.path: value, "tone_mapping.backend": backend,
                        "tone_mapping.parameters": dict(OPERATOR_PARAMETERS[value]),
                    }, "Switch tone operator and documented defaults")
                else:
                    self.model.edit(self.path, value)
            else:
                self.model.edit(self.path, value)
        except (TypeError, ValueError, AttributeError) as exc:
            self.error_var.set(str(exc))
            return
        self.present = True
        if self.model.revision != before:
            self.on_change()

    def reset(self) -> None:
        self.model.end_edit_group()
        self.model.reset_parameter(self.path)
        self.on_change()
        self.winfo_toplevel().event_generate("<<WorkspaceReset>>", when="tail")
