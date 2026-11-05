"""Headless workspace controller: serial job scheduling, staleness guards."""
from __future__ import annotations

import copy
import json
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from .jobs import WorkerManager
from .model import WorkspaceModel
from .protocol import JobRequest, JobTag, WorkerEvent

_TERMINAL = ("done", "error", "cancelled")


class WorkspaceController:
    """Owns job publication. No Tk: the UI layer drives ``tick()``."""

    def __init__(self, model: WorkspaceModel, manager: WorkerManager,
                 *, clock: Callable[[], float] = time.monotonic,
                 debounce_seconds: float = 0.4, max_batch: int = 16,
                 auto_default: bool = False,
                 root_dir: Path | str = "sessions") -> None:
        self.model = model
        self.manager = manager
        self.clock = clock
        self.debounce_seconds = debounce_seconds
        self.max_batch = max_batch
        self.root_dir = Path(root_dir)
        self.auto_enabled = auto_default
        self.current_result: dict | None = None
        self.active: JobRequest | None = None
        self.active_role: str | None = None
        self._pending: dict | None = None
        self._pending_at = 0.0
        self._pending_manual = False
        self._discard: set[str] = set()
        self._retiring = False
        self.status = "idle"
        self.progress: list[dict] = []
        self.jobs: list[dict] = []
        self.batch: list[dict] = []
        self.batch_paused = True
        self._closed_ctl = False
        self._pending_aux: JobRequest | None = None
        self.latest_analysis_id: str | None = None
        self.analysis_result: dict | None = None
        self._pending_render: JobRequest | None = None
        self.latest_render_id: str | None = None
        self.render_result: dict | None = None

    def _tag(self, source: dict | None = None) -> JobTag:
        src = source if source is not None else (self.model.source or {})
        return JobTag(
            request_id=uuid.uuid4().hex,
            session_id=self.model.session_id,
            generation=self.model.generation,
            revision=self.model.revision,
            source_fingerprint=src.get("fingerprint", ""))

    def _job_dir(self, tag: JobTag) -> Path:
        return (self.root_dir / self.model.session_id / "jobs"
                / tag.request_id)

    def _config_json_for(self, cfg: dict, source: dict | None) -> str:
        cfg = copy.deepcopy(cfg)
        if source:
            cfg.setdefault("platform", {})
            cfg["platform"]["filename"] = Path(source["path"]).name
        return json.dumps(cfg, allow_nan=False)

    def _build_request(self, role: str, cfg_snapshot: dict | None = None,
                       source_snapshot: dict | None = None,
                       capture_stages: tuple[str, ...] = (),
                       payload: dict | None = None,
                       source_path: str | None = None) -> JobRequest:
        src = source_snapshot if source_snapshot is not None else self.model.source
        cfg = cfg_snapshot if cfg_snapshot is not None else self.model.config
        tag = self._tag(src)
        return JobRequest(
            tag=tag, operation=role,
            config_json=self._config_json_for(cfg, src),
            source_path=source_path if source_path is not None
            else (src["path"] if src else ""),
            output_dir=str(self._job_dir(tag)),
            capture_stages=capture_stages,
            payload_json=json.dumps(payload or {}))

    # ---- user entry points ----

    def on_edit(self) -> None:
        if self.current_result is not None:
            self.current_result["stale"] = True
        if self.active_role == "interactive" and self.active is not None:
            self._discard.add(self.active.tag.request_id)
            self.manager.cancel()
            self.status = "cancelling"
        if self._pending is not None and self._pending["role"] == "export":
            return
        if self.auto_enabled:
            self._pending = {"role": "process", "manual": False,
                             "capture_stages": ()}
            self._pending_at = self.clock()
        else:
            self._pending = None

    def context_changed(self) -> None:
        if self.active is not None:
            self._discard.add(self.active.tag.request_id)
            self.manager.cancel()
            self._retiring = True
        self._pending_aux = None
        self.latest_analysis_id = None
        self.analysis_result = None
        self._pending_render = None
        self.latest_render_id = None
        self.render_result = None
        self._pending = None
        self.batch.clear()
        self.batch_paused = True
        if self.current_result is not None:
            self.current_result["stale"] = True
        self.status = "idle" if self.active is None else "cancelling"

    def request_process(self, capture_stages: tuple[str, ...] = ()) -> bool:
        if self._closed_ctl or not self.model.config_json_ready() or self.model.source is None:
            return False
        req = self._build_request("process", capture_stages=capture_stages)
        if self.active is not None or not self.manager.ready:
            self._pending = {"role": "process", "manual": True,
                             "request": req, "capture_stages": capture_stages}
            self._pending_at = self.clock()
            return True
        self._submit(req, "interactive")
        return True

    def request_import(self, path: str | Path) -> JobRequest | None:
        if self._closed_ctl:
            return None
        tag = self._tag()
        try:
            cfg_json = json.dumps(self.model.config, allow_nan=False)
        except (TypeError, ValueError):
            cfg_json = "{}"
        req = JobRequest(
            tag=tag, operation="import", config_json=cfg_json,
            source_path=str(path), output_dir=str(self._job_dir(tag)))
        if self.active is not None or not self.manager.ready:
            self._pending = {"role": "import", "manual": True, "request": req}
            self._pending_at = self.clock()
            if self.active is not None:
                self._discard.add(self.active.tag.request_id)
                self.manager.cancel()
            self.status = "cancelling"
            return req
        self._submit(req, "import")
        return req

    def request_export(self, manifest_dir: str | Path,
                       dest_dir: str | Path, stem: str) -> JobRequest | None:
        if self._closed_ctl:
            return None
        tag = self._tag()
        req = JobRequest(
            tag=tag, operation="export", config_json="{}",
            source_path="", output_dir=str(self._job_dir(tag)),
            payload_json=json.dumps({
                "manifest_dir": str(manifest_dir),
                "dest_dir": str(dest_dir), "stem": stem}))
        if self.active is not None or not self.manager.ready:
            self._pending = {"role": "export", "manual": True, "request": req}
            self._pending_at = self.clock()
            return req
        self._submit(req, "export")
        return req

    def _submit(self, req: JobRequest, role: str) -> None:
        self.manager.submit(req)
        self.progress.clear()
        self.active = req
        self.active_role = role
        self.status = "running"
        self.jobs.append({"request_id": req.tag.request_id, "role": role,
                          "status": "running"})

    def _context_ok(self, tag: JobTag) -> bool:
        return (tag.session_id == self.model.session_id
                and tag.generation == self.model.generation
                and tag.revision == self.model.revision
                and tag.source_fingerprint
                == (self.model.source or {}).get("fingerprint", ""))

    def request_analysis(self, manifest_dir, stage: str = "preview",
                         roi=None, comparison_dir=None) -> JobRequest | None:
        if self._closed_ctl:
            return None
        tag = self._tag()
        payload = {
            "manifest_dir": str(manifest_dir), "stage": stage,
            "roi": list(roi) if roi is not None else None,
            "comparison_dir": (str(comparison_dir)
                               if comparison_dir is not None else None)}
        req = JobRequest(
            tag=tag, operation="analysis", config_json="{}",
            source_path="", output_dir=str(self._job_dir(tag)),
            payload_json=json.dumps(payload, allow_nan=False))
        self.latest_analysis_id = tag.request_id
        if self.active_role == "aux" and self.active is not None:
            self._discard.add(self.active.tag.request_id)
            self.manager.cancel()
        self._pending_aux = req
        self._schedule()
        return req

    def request_render(self, a_dir, b_dir, *, mode: str,
                       fraction: float = 0.5,
                       difference_gain: float = 1.0) -> JobRequest | None:
        if self._closed_ctl:
            return None
        tag = self._tag()
        payload = {
            "a_dir": str(a_dir), "b_dir": str(b_dir), "mode": mode,
            "fraction": fraction, "difference_gain": difference_gain}
        req = JobRequest(
            tag=tag, operation="render", config_json="{}",
            source_path="", output_dir=str(self._job_dir(tag)),
            payload_json=json.dumps(payload, allow_nan=False))
        self.latest_render_id = tag.request_id
        if self.active_role == "render" and self.active is not None:
            self._discard.add(self.active.tag.request_id)
            self.manager.cancel()
        self._pending_render = req
        self._schedule()
        return req

    # ---- scheduler ----

    def tick(self) -> list[WorkerEvent]:
        events = self.manager.poll()
        for evt in events:
            self._handle(evt)
        self._schedule()
        return events

    def _schedule(self) -> None:
        if self._closed_ctl:
            return
        if self.active is not None or not self.manager.ready:
            return
        if self._pending is not None:
            manual = self._pending.get("manual", False)
            due = manual or (
                self.auto_enabled
                and self.clock() - self._pending_at >= self.debounce_seconds)
            if due:
                pend = self._pending
                role = pend["role"]
                if role == "process":
                    if self.model.source is None:
                        self._pending = None
                        self.status = "source_required"
                        return
                    if not self.model.config_json_ready():
                        self._pending = None
                        self.status = "validation_blocked"
                        return
                self._pending = None
                req = pend.get("request")
                if req is None:
                    req = self._build_request(
                        "process",
                        capture_stages=pend.get("capture_stages", ()))
                self._submit(req, "interactive" if role == "process" else role)
            return
        if self._pending_render is not None and self.manager.ready:
            req = self._pending_render
            self._pending_render = None
            self._submit(req, "render")
            return
        if self._pending_aux is not None and self.manager.ready:
            req = self._pending_aux
            self._pending_aux = None
            self._submit(req, "aux")
            return
        if not self.batch_paused:
            self._start_next_batch()

    def _handle(self, evt: WorkerEvent) -> None:
        if self.active is None:
            return  # no active job: ignore all events
        if evt.tag != self.active.tag:
            return  # stale event for a different request
        kind = evt.kind
        payload = json.loads(evt.payload_json or "{}")
        job = next((j for j in self.jobs
                    if j["request_id"] == evt.tag.request_id), None)
        if kind not in _TERMINAL and kind != "progress":
            return  # unknown kinds cannot affect the active job
        if kind == "progress":
            if (not self._closed_ctl and evt.tag.request_id not in self._discard
                    and (self.active_role == "batch" or self._context_ok(evt.tag))):
                payload["request_id"] = evt.tag.request_id
                self.progress.append(payload)
            return
        role = self.active_role
        discarded = evt.tag.request_id in self._discard
        obsolete = discarded or (
            role in ("interactive", "import") and not self._context_ok(evt.tag))
        if job is not None:
            job["status"] = ("discarded" if kind == "done" and obsolete
                             else kind)
            job["payload"] = payload
        self._discard.discard(evt.tag.request_id)
        self.active = None
        self.active_role = None
        if kind == "done":
            if role == "interactive" and self._context_ok(evt.tag) and not discarded:
                self.current_result = {
                    "result_path": payload.get("result_path"),
                    "tag": evt.tag, "stale": False,
                    "provenance": "interactive"}
                self.status = "done"
            elif role == "import" and self._context_ok(evt.tag) and not discarded:
                record = dict(payload["source"])
                overrides = record.pop("format_overrides", {})
                self.model.adopt_source(record, overrides or None)
                self.status = "done"
            elif role == "batch":
                self._batch_result(
                    evt.tag.request_id,
                    "done" if not discarded
                    else "discarded", payload)
            elif role == "render":
                if (not discarded
                        and evt.tag.request_id == self.latest_render_id
                        and not self._closed_ctl):
                    self.render_result = {
                        "render_path": payload.get("render_path"),
                        "request_id": evt.tag.request_id,
                        "descriptor": payload.get("descriptor")}
                    self.status = "done"
                else:
                    self.status = "discarded"
            elif role == "aux":
                if (not discarded
                        and evt.tag.request_id == self.latest_analysis_id
                        and not self._closed_ctl):
                    self.analysis_result = {
                        "analysis_path": payload.get("analysis_path"),
                        "request_id": evt.tag.request_id,
                        "descriptor": payload.get("descriptor")}
                    self.status = "done"
                else:
                    self.status = "discarded"
            elif role == "export":
                self.status = "done" if not discarded else "discarded"
            else:
                self.status = "discarded"
        elif kind == "cancelled":
            if role == "batch":
                self._batch_result(evt.tag.request_id, "cancelled", payload)
            self.status = "cancelled"
        else:
            if role == "batch":
                self._batch_result(evt.tag.request_id, "error", payload)
            self.status = "error"
        self._retiring = False

    # ---- batch ----

    def start_batch(self) -> None:
        self.batch_paused = False

    def pause_batch(self) -> None:
        self.batch_paused = True

    def enqueue_batch(self, config_snapshot: dict | None = None,
                      label: str = "Batch",
                      source_snapshot: dict | None = None) -> str | None:
        unfinished = sum(1 for i in self.batch
                         if i["status"] in ("queued", "running"))
        if unfinished >= self.max_batch:
            return None
        cfg = config_snapshot if config_snapshot is not None else self.model.config
        src = (source_snapshot if source_snapshot is not None
               else self.model.source)
        if src is None:
            return None
        try:
            json.dumps(cfg, allow_nan=False)
        except (TypeError, ValueError):
            return None
        from util.config_schema import validate_config
        if any(i.severity == "error" for i in validate_config(cfg)):
            return None
        batch_id = uuid.uuid4().hex
        self.batch.append({
            "batch_id": batch_id, "label": label,
            "config": copy.deepcopy(cfg), "source": copy.deepcopy(src),
            "status": "queued", "result_path": None, "error": None,
            "request_id": None,
            "generation": self.model.generation,
            "revision": self.model.revision})
        return batch_id

    def retry_batch(self, batch_id: str) -> bool:
        unfinished = sum(1 for i in self.batch
                         if i["status"] in ("queued", "running"))
        if unfinished >= self.max_batch:
            return False
        for item in self.batch:
            if item["batch_id"] == batch_id:
                if item["status"] not in ("done", "error", "cancelled", "discarded",
                                          "interrupted"):
                    return False
                item["status"] = "queued"
                item["error"] = None
                item["result_path"] = None
                item["request_id"] = None
                return True
        return False

    def _start_next_batch(self) -> None:
        item = next((i for i in self.batch if i["status"] == "queued"), None)
        if item is None:
            return
        req = self._build_request(
            "process", cfg_snapshot=item["config"],
            source_snapshot=item["source"])
        req = JobRequest(
            tag=req.tag, operation="process", config_json=req.config_json,
            source_path=req.source_path, output_dir=req.output_dir,
            capture_stages=req.capture_stages)
        item["request_id"] = req.tag.request_id
        item["status"] = "running"
        self._submit(req, "batch")

    def _batch_result(self, request_id: str, status: str, payload: dict) -> None:
        for item in self.batch:
            if item["request_id"] == request_id:
                item["status"] = status
                item["error"] = payload.get("error")
                item["result_path"] = payload.get("result_path")

    # ---- lifecycle ----

    def cancel(self) -> None:
        self.auto_enabled = False
        self._pending = None
        self._pending_aux = None
        self.latest_analysis_id = None
        self.analysis_result = None
        self._pending_render = None
        self.latest_render_id = None
        self.render_result = None
        self.batch_paused = True
        if self.active is not None:
            self._discard.add(self.active.tag.request_id)
            self.manager.cancel()
            self.status = "cancelling"

    def stop_worker(self) -> None:
        if self.active is not None:
            job = next((j for j in self.jobs
                        if j["request_id"] == self.active.tag.request_id), None)
            if job is not None:
                job["status"] = "interrupted"
            self._discard.add(self.active.tag.request_id)
        if self.active_role == "batch":
            for item in self.batch:
                if item["status"] == "running":
                    item["status"] = "interrupted"
        self._pending = None
        self._pending_aux = None
        self.latest_analysis_id = None
        self.analysis_result = None
        self._pending_render = None
        self.latest_render_id = None
        self.render_result = None
        self.auto_enabled = False
        self.batch_paused = True
        self.manager.terminate()
        self.active = None
        self.active_role = None
        self._pending = None
        self.status = "interrupted"

    def close(self) -> None:
        self._closed_ctl = True
        self._pending = None
        self._pending_aux = None
        self.latest_analysis_id = None
        self.analysis_result = None
        self._pending_render = None
        self.latest_render_id = None
        self.render_result = None
        self.batch_paused = True
        if self.active is not None:
            self._discard.add(self.active.tag.request_id)
        self.manager.request_close()
