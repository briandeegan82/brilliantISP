"""Stdlib-only IPC protocol shared by the workspace core and worker.

This module must never import Tk or NumPy at top level: the spawn child
imports it before its own environment is configured.
"""
from __future__ import annotations

from dataclasses import dataclass, field

ALLOWED_OPERATIONS = ("import", "process", "export", "analysis", "render")


@dataclass(frozen=True)
class JobTag:
    request_id: str
    session_id: str
    generation: int
    revision: int
    source_fingerprint: str


@dataclass(frozen=True)
class JobRequest:
    tag: JobTag
    operation: str
    config_json: str
    source_path: str
    output_dir: str
    capture_stages: tuple[str, ...] = ()
    payload_json: str = "{}"


@dataclass(frozen=True)
class WorkerEvent:
    tag: JobTag
    kind: str
    payload_json: str = "{}"
