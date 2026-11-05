from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from typing import Callable


class ComparisonControls(ttk.Frame):
    MODES = ("Current", "A", "B", "Side by side", "Wipe", "Difference")

    def __init__(self, parent, visible_snapshot: Callable[[], str | None],
                 on_change: Callable[[], None], on_error: Callable[[str], None]) -> None:
        super().__init__(parent)
        self.visible_snapshot, self.on_change, self.on_error = visible_snapshot, on_change, on_error
        self.pins: dict[str, dict | None] = {"A": None, "B": None}
        self._pending = None
        self.mode = tk.StringVar(value="Current")
        self.fraction = tk.DoubleVar(value=.5)
        self.gain = tk.StringVar(value="1")
        row = ttk.Frame(self)
        row.pack(fill="x")
        for slot in ("A", "B"):
            ttk.Button(row, text=f"Pin visible → {slot}", command=lambda s=slot: self.pin(s)).pack(side="left")
        self.mode_selector = ttk.Combobox(row, textvariable=self.mode, values=self.MODES, state="readonly", width=14)
        self.mode_selector.pack(side="left", padx=4)
        self.mode_selector.bind("<<ComboboxSelected>>", lambda e: self.on_change())
        ttk.Label(row, text="Wipe").pack(side="left")
        ttk.Scale(row, variable=self.fraction, from_=0, to=1, command=lambda v: self._defer()).pack(
            side="left", fill="x", expand=True)
        ttk.Label(row, text="Difference ×").pack(side="left")
        gain = ttk.Entry(row, textvariable=self.gain, width=5)
        gain.pack(side="left")
        gain.bind("<Return>", lambda e: self.on_change())
        gain.bind("<FocusOut>", lambda e: self.on_change())
        self.labels = tk.StringVar(value="A: not pinned | B: not pinned")
        ttk.Label(self, textvariable=self.labels, wraplength=800).pack(anchor="w")

    def _defer(self) -> None:
        if self._pending is not None:
            self.after_cancel(self._pending)
        self._pending = self.after(70, self._changed)

    def _changed(self) -> None:
        self._pending = None
        if self.mode.get() == "Wipe":
            self.on_change()

    def pin(self, slot: str) -> None:
        directory = self.visible_snapshot()
        if directory is None:
            self.on_error("There is no visible source/result snapshot to pin.")
            return
        path = Path(directory)
        try:
            manifest = json.loads((path / "result.json").read_text(encoding="utf-8"))
            tag = manifest["tag"]
            stamp = datetime.fromtimestamp((path / "result.json").stat().st_mtime, timezone.utc).isoformat(
                timespec="seconds")
            self.pins[slot] = {
                "manifest": str(path), "stage": "preview", "timestamp": stamp,
                "source_fingerprint": manifest["source_fingerprint"],
                "request_id": tag.get("request_id", ""), "revision": tag.get("revision"),
                "source_name": Path(manifest["native_metadata"]["source_identity"]).name,
            }
        except (ValueError, OSError, KeyError) as exc:
            self.on_error(f"Cannot pin snapshot: {exc}")
            return
        self._labels()
        self.on_change()

    def _labels(self) -> None:
        labels = []
        for slot, pin in self.pins.items():
            labels.append(f"{slot}: not pinned" if pin is None else (
                f"{slot}: {pin.get('source_name', '')} [{str(pin.get('source_fingerprint', ''))[:10]}] "
                f"r{pin.get('revision')} {pin.get('timestamp', '')}"))
        self.labels.set(" | ".join(labels))

    def directories(self) -> tuple[str, str]:
        a, b = self.pins["A"], self.pins["B"]
        if a is None or b is None:
            raise ValueError("Pin both A and B for this comparison mode.")
        return a["manifest"], b["manifest"]

    def difference_gain(self) -> float:
        try:
            gain = float(self.gain.get())
        except ValueError as exc:
            raise ValueError("Difference gain must be a number in (0, 64].") from exc
        if not math.isfinite(gain) or not 0 < gain <= 64:
            raise ValueError("Difference gain must be in (0, 64].")
        return gain

    def restore(self, snapshots: dict) -> None:
        self.pins = {slot: snapshots[slot] if isinstance(snapshots.get(slot), dict) else None for slot in ("A", "B")}
        self.mode.set("Current")
        self._labels()
