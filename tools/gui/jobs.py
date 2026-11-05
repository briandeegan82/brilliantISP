"""Spawn-context persistent worker manager. No Tk imports anywhere."""
from __future__ import annotations

import multiprocessing as mp
import queue as _queue
import time
from typing import Any, Optional

from .protocol import JobRequest, WorkerEvent

_POLL_BUDGET = 64
_TERMINAL = ("done", "error", "cancelled")


class WorkerManager:
    """One persistent spawn worker; strictly one active job at a time."""

    def __init__(self) -> None:
        self._ctx = mp.get_context("spawn")
        self._proc: Any = None
        self._req_q: Any = None
        self._evt_q: Any = None
        self._cancel: Any = None
        self._stop: Any = None
        self._active: JobRequest | None = None
        self._closed = False
        self._stopping = False
        self._shutdown_deadline: float | None = None
        self._kill_deadline: float | None = None

    @property
    def active(self) -> JobRequest | None:
        return self._active

    @property
    def ready(self) -> bool:
        return not self._closed and not self._stopping

    def _close_queues(self) -> None:
        for q in (self._req_q, self._evt_q):
            if q is not None:
                try:
                    q.cancel_join_thread()
                    q.close()
                except Exception:
                    pass
        self._req_q = self._evt_q = None
        self._cancel = self._stop = None

    def _ensure(self) -> None:
        if self._stopping:
            raise RuntimeError("previous worker is still stopping")
        if self._proc is not None and self._proc.is_alive():
            return
        if self._proc is not None:
            self._proc.join(timeout=0)
            if self._proc.is_alive():
                raise RuntimeError("old worker not yet reaped")
            self._proc = None
        self._close_queues()
        self._req_q = self._ctx.Queue(maxsize=4)
        self._evt_q = self._ctx.Queue(maxsize=256)
        self._cancel = self._ctx.Event()
        self._stop = self._ctx.Event()
        from .worker import worker_main
        self._proc = self._ctx.Process(
            target=worker_main,
            args=(self._req_q, self._evt_q, self._cancel, self._stop),
            daemon=True)
        self._proc.start()

    def submit(self, req: JobRequest) -> None:
        if self._closed:
            raise RuntimeError("manager closed")
        if not self.ready:
            raise RuntimeError("worker is stopping")
        if self._active is not None:
            raise RuntimeError("a job is already active")
        self._ensure()
        self._cancel.clear()
        self._req_q.put_nowait(req)
        self._active = req

    def cancel(self) -> None:
        if self._cancel is not None:
            self._cancel.set()

    def poll(self) -> list[WorkerEvent]:
        out: list[WorkerEvent] = []
        if self._evt_q is not None:
            while len(out) < _POLL_BUDGET:
                try:
                    out.append(self._evt_q.get_nowait())
                except _queue.Empty:
                    break
        if self._active is not None:
            if any(e.kind in _TERMINAL
                   and e.tag == self._active.tag
                   for e in out):
                if self._proc is not None and not self._proc.is_alive():
                    self._proc.join(timeout=0)
                    self._proc = None
                self._active = None
            elif self._proc is not None and not self._proc.is_alive():
                out.append(WorkerEvent(
                    self._active.tag, "error",
                    '{"error": "worker exited unexpectedly"}'))
                self._proc.join(timeout=0)
                self._proc = None
                self._active = None
        self.shutdown_step()
        return out

    def terminate(self) -> None:
        self._stopping = True
        if self._cancel is not None:
            self._cancel.set()
        if self._stop is not None:
            self._stop.set()
        if self._proc is not None:
            self._proc.terminate()
            self._proc.join(timeout=0)
            if self._proc.is_alive():
                self._kill_deadline = time.monotonic() + 0.5
            else:
                self._proc = None
        self._active = None

    def request_close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stopping = True
        if self._cancel is not None:
            self._cancel.set()
        if self._stop is not None:
            self._stop.set()
        self._shutdown_deadline = time.monotonic() + 2.0
        self._kill_deadline = None

    def shutdown_step(self) -> None:
        if self._proc is None:
            self._close_queues()
            self._stopping = False
            self._active = None
            self._shutdown_deadline = self._kill_deadline = None
            return
        now = time.monotonic()
        if not self._proc.is_alive():
            self._proc.join(timeout=0)
            self._proc = None
            self._close_queues()
            self._stopping = False
            self._shutdown_deadline = None
            self._kill_deadline = None
            return
        if self._shutdown_deadline is not None and now >= self._shutdown_deadline:
            if self._kill_deadline is None:
                self._proc.terminate()
                self._kill_deadline = now + 0.5
            elif now >= self._kill_deadline:
                self._proc.kill()
        elif self._stopping and self._kill_deadline is not None and now >= self._kill_deadline:
            self._proc.kill()

    @property
    def shutdown_complete(self) -> bool:
        return self._proc is None

    def close(self) -> None:
        if self._shutdown_deadline is None:
            self.request_close()
