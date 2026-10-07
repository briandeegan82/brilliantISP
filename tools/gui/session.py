"""GUI session directories: temp YAML + RAW working copies."""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.gui.yaml_compat import dump_config_yaml, load_preset_yaml
from util.raw_io import read_raw_payload

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SESSION_ROOT = REPO_ROOT / "tmp" / "gui_session"
DEFAULT_PRESET = REPO_ROOT / "config" / "SVS_cam.yml"
LAST_SESSION_NAME = "last_session.json"
INPUT_RAW = "input.raw"
CONFIG_YML = "config.yml"
OUTPUT_PNG = "output.png"
SESSION_JSON = "session.json"

DEFAULT_UI_STATE: dict[str, Any] = {
    "original_source_path": None,
    "header_size": 0,
    "data_alignment": "LSB",  # Default to LSB (no shift) - most RAW files are pre-aligned
    "flip_h": False,
    "flip_v": False,
    "flip_90": False,
    "last_preset_path": None,
    "zoom": 1.0,
    "pan": [0.0, 0.0],
    "fit_mode": "fit",
}


@dataclass
class SessionPaths:
    directory: Path

    @property
    def input_raw(self) -> Path:
        return self.directory / INPUT_RAW

    @property
    def config_yml(self) -> Path:
        return self.directory / CONFIG_YML

    @property
    def output_png(self) -> Path:
        return self.directory / OUTPUT_PNG

    @property
    def session_json(self) -> Path:
        return self.directory / SESSION_JSON


class GuiSession:
    def __init__(
        self,
        session_id: str,
        paths: SessionPaths,
        ui_state: dict[str, Any] | None = None,
    ) -> None:
        self.session_id = session_id
        self.paths = paths
        self.ui_state: dict[str, Any] = {**DEFAULT_UI_STATE, **(ui_state or {})}
        self.last_payload_meta = None

    @property
    def has_raw(self) -> bool:
        return self.paths.input_raw.is_file()

    @classmethod
    def create_new(
        cls,
        *,
        root: Path | None = None,
        preset_path: Path | None = None,
        config: dict[str, Any] | None = None,
    ) -> tuple["GuiSession", dict[str, Any]]:
        root = Path(root) if root is not None else DEFAULT_SESSION_ROOT
        session_id = uuid.uuid4().hex[:12]
        directory = root / session_id
        directory.mkdir(parents=True, exist_ok=True)
        session = cls(session_id, SessionPaths(directory))
        if config is None:
            preset = Path(preset_path) if preset_path is not None else DEFAULT_PRESET
            config = load_preset_yaml(preset)
            session.ui_state["last_preset_path"] = str(preset)
        session.ensure_platform_filename(config)
        session.write_config(config)
        session.write_ui_state()
        write_last_session_id(session_id, root=root)
        return session, config

    @classmethod
    def open_existing(cls, directory: str | Path, *, root: Path | None = None) -> tuple["GuiSession", dict[str, Any]]:
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"Session directory not found: {directory}")
        session_id = directory.name
        ui_state = _read_json(directory / SESSION_JSON, DEFAULT_UI_STATE)
        session = cls(session_id, SessionPaths(directory), ui_state=ui_state)
        if not session.paths.config_yml.is_file():
            raise FileNotFoundError(f"Missing {CONFIG_YML} in {directory}")
        config = load_preset_yaml(session.paths.config_yml)
        write_last_session_id(session_id, root=root if root is not None else directory.parent)
        return session, config

    def save(self, config: dict[str, Any]) -> None:
        self.ensure_platform_filename(config)
        self.write_config(config)
        self.write_ui_state()
        write_last_session_id(self.session_id, root=self.paths.directory.parent)

    def ensure_platform_filename(self, config: dict[str, Any]) -> None:
        platform = config.setdefault("platform", {})
        if isinstance(platform, dict):
            platform["filename"] = INPUT_RAW

    def write_config(self, config: dict[str, Any]) -> None:
        self.ensure_platform_filename(config)
        dump_config_yaml(config, self.paths.config_yml)

    def write_ui_state(self) -> None:
        self.paths.session_json.write_text(
            json.dumps(self.ui_state, indent=2),
            encoding="utf-8",
        )

    def copy_raw_from(self, source: str | Path) -> Path:
        source = Path(source)
        if not source.is_file():
            raise FileNotFoundError(f"RAW not found: {source}")
        payload = read_raw_payload(source)
        self.paths.input_raw.write_bytes(payload.data)
        self.ui_state["original_source_path"] = str(source)
        self.ui_state["header_size"] = 0 if payload.is_pgm else self.ui_state.get("header_size", 0)
        self.last_payload_meta = payload
        self.write_ui_state()
        return self.paths.input_raw

    def reload_raw(self) -> Path:
        src = self.ui_state.get("original_source_path")
        if not src:
            raise FileNotFoundError("No original source path to reload")
        return self.copy_raw_from(src)

    def write_output_png(self, rgb_bytes: bytes | None = None, source_png: str | Path | None = None) -> Path:
        if source_png is not None:
            shutil.copy2(source_png, self.paths.output_png)
        elif rgb_bytes is not None:
            self.paths.output_png.write_bytes(rgb_bytes)
        else:
            raise ValueError("Provide rgb_bytes or source_png")
        return self.paths.output_png


def write_last_session_id(session_id: str, *, root: Path | None = None) -> None:
    root = Path(root) if root is not None else DEFAULT_SESSION_ROOT
    root.mkdir(parents=True, exist_ok=True)
    (root / LAST_SESSION_NAME).write_text(
        json.dumps({"session_id": session_id}, indent=2),
        encoding="utf-8",
    )


def read_last_session_id(*, root: Path | None = None) -> str | None:
    root = Path(root) if root is not None else DEFAULT_SESSION_ROOT
    path = root / LAST_SESSION_NAME
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    sid = data.get("session_id")
    return str(sid) if sid else None


def recover_last_session(*, root: Path | None = None) -> tuple[GuiSession, dict[str, Any]] | None:
    root = Path(root) if root is not None else DEFAULT_SESSION_ROOT
    sid = read_last_session_id(root=root)
    if not sid:
        return None
    directory = root / sid
    if not directory.is_dir() or not (directory / CONFIG_YML).is_file():
        return None
    return GuiSession.open_existing(directory, root=root)


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        return dict(default)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return dict(default)
    return {**default, **data}
