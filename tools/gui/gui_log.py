"""File + console logging for the calibration GUI."""

from __future__ import annotations

import logging
import sys
import threading
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable

from tools.gui.session import DEFAULT_SESSION_ROOT

LOGGER_NAME = "brilliantisp.calibration_gui"
GLOBAL_LOG_NAME = "calibration_gui.log"
SESSION_LOG_NAME = "gui.log"
_MAX_BYTES = 2 * 1024 * 1024
_BACKUP_COUNT = 3


def get_gui_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def setup_gui_logging(*, session_dir: Path | None = None) -> logging.Logger:
    """Attach stderr + rotating global log, and optional session ``gui.log``."""
    log = get_gui_logger()
    log.setLevel(logging.DEBUG)
    log.propagate = False
    _ensure_stream_handler(log)
    global_path = DEFAULT_SESSION_ROOT / GLOBAL_LOG_NAME
    _ensure_file_handler(log, global_path, key="global")
    if session_dir is not None:
        attach_session_log(session_dir)
    return log


def attach_session_log(session_dir: Path) -> Path:
    log = get_gui_logger()
    path = Path(session_dir) / SESSION_LOG_NAME
    _ensure_file_handler(log, path, key="session")
    log.info("Session log: %s", path)
    return path


def _ensure_stream_handler(log: logging.Logger) -> None:
    if any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in log.handlers):
        return
    h = logging.StreamHandler(sys.stderr)
    h.setLevel(logging.INFO)
    h.setFormatter(_formatter())
    setattr(h, "_gui_log_key", "stderr")
    log.addHandler(h)


def _ensure_file_handler(log: logging.Logger, path: Path, *, key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for h in list(log.handlers):
        if getattr(h, "_gui_log_key", None) == key:
            if key == "session":
                log.removeHandler(h)
                h.close()
            else:
                return
    h = RotatingFileHandler(path, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8")
    h.setLevel(logging.DEBUG)
    h.setFormatter(_formatter())
    setattr(h, "_gui_log_key", key)
    log.addHandler(h)


def _formatter() -> logging.Formatter:
    return logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def install_exception_hooks(
    *,
    tk_root: Any | None = None,
    on_error: Callable[[str], None] | None = None,
) -> None:
    log = get_gui_logger()

    def handle(exc_type, exc, tb) -> None:
        if exc_type is None:
            return
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        log.error("Unhandled exception:\n%s", text)
        if on_error is not None:
            on_error(text)

    def sys_hook(exc_type, exc, tb) -> None:
        handle(exc_type, exc, tb)

    sys.excepthook = sys_hook

    def thread_hook(args: threading.ExceptHookArgs) -> None:
        handle(args.exc_type, args.exc_value, args.exc_traceback)

    threading.excepthook = thread_hook

    if tk_root is not None:

        def report(exc, val, tb) -> None:
            handle(exc, val, tb)

        tk_root.report_callback_exception = report


def truncate(value: Any, limit: int = 240) -> str:
    text = repr(value)
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def diff_dicts(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    keys = sorted(set(old) | set(new))
    for key in keys:
        a, b = old.get(key), new.get(key)
        if a != b:
            lines.append(f"{key}: {truncate(a)} -> {truncate(b)}")
    return lines
