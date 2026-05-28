"""
In-memory job queue for long-running Monte Carlo executions (Phase 12).

The HTTP analysis/optimization endpoints used to be fully synchronous:
the browser blocked for the entire Monte Carlo run, with no way to show
a progress bar. This module moves those workloads to a background
ThreadPoolExecutor and exposes a small ``JobStore`` so the API can:

1. Accept a job submission and return immediately with a job id.
2. Let the client poll a status endpoint that reports
   ``status ∈ {pending, running, done, failed}`` plus a ``progress``
   counter wired to the Monte Carlo ``progress_callback``.
3. When the job finishes, expose the persisted ``run_id`` so the
   wizard can redirect to the Dashboard and auto-select the new run.

Scope: single-process, single-uvicorn-worker deployments (matches our
``Dockerfile.backend``). For multi-worker setups this should be backed
by Redis or a real task queue (Celery / RQ / Dramatiq). Not in scope
for Phase 12.

Memory: jobs are kept indefinitely so the polling client always finds
them; in practice the store is a few KB per job. A simple LRU prune is
applied to cap the store at a few hundred entries.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Literal, Optional

logger = logging.getLogger(__name__)


JobKind = Literal["analysis", "optimization"]
JobStatus = Literal["pending", "running", "done", "failed"]


@dataclass
class JobRecord:
    """Lightweight snapshot of a background simulation job.

    The ``progress_done`` / ``progress_total`` pair drives the UI bar.
    For an analysis it counts Monte Carlo paths; for a design (optimization
    sweep) it counts completed scenario configurations. The frontend
    reads the values raw and formats the % itself.
    """

    id: str
    kind: JobKind
    status: JobStatus = "pending"
    progress_done: int = 0
    progress_total: int = 0
    message: str = ""
    run_id: Optional[int] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "progress_done": self.progress_done,
            "progress_total": self.progress_total,
            "progress_fraction": (
                self.progress_done / self.progress_total
                if self.progress_total > 0
                else 0.0
            ),
            "message": self.message,
            "run_id": self.run_id,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
        }


class JobStore:
    """
    Thread-safe job registry. Singleton at the module level (``_STORE``).

    Use ``submit(...)`` to register and execute a callable in the
    background, ``get(...)`` to fetch a status snapshot, and ``update(...)``
    from inside the worker to mutate progress fields atomically.
    """

    def __init__(self, max_workers: int = 2, max_entries: int = 256) -> None:
        self._lock = threading.Lock()
        self._jobs: "OrderedDict[str, JobRecord]" = OrderedDict()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._max_entries = max_entries

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_recent(self, limit: int = 20) -> list[JobRecord]:
        with self._lock:
            return list(self._jobs.values())[-limit:][::-1]

    def update(self, job_id: str, **fields: Any) -> None:
        """Mutate a job record in-place. Unknown keys are silently ignored."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for k, v in fields.items():
                if hasattr(job, k):
                    setattr(job, k, v)

    def submit(
        self,
        kind: JobKind,
        worker: Callable[["JobHandle"], None],
    ) -> JobRecord:
        """
        Register a new job and dispatch it to the background pool.

        Args:
            kind: ``'analysis'`` or ``'optimization'``. Stored for routing.
            worker: Callable receiving a :class:`JobHandle` it can use to
                report progress and the final ``run_id``. The callable is
                executed inside the thread pool; any exception is captured
                and surfaces in the job's ``error`` field.

        Returns:
            The freshly created ``JobRecord`` (with ``status='pending'``).
        """
        job_id = uuid.uuid4().hex
        record = JobRecord(id=job_id, kind=kind)
        with self._lock:
            self._jobs[job_id] = record
            self._prune_unlocked()

        handle = JobHandle(store=self, job_id=job_id)

        def run() -> None:
            self.update(job_id, status="running", started_at=datetime.now(timezone.utc))
            try:
                worker(handle)
                self.update(
                    job_id,
                    status="done",
                    completed_at=datetime.now(timezone.utc),
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Job %s failed", job_id)
                self.update(
                    job_id,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                    completed_at=datetime.now(timezone.utc),
                )

        self._executor.submit(run)
        return record

    def _prune_unlocked(self) -> None:
        """Drop the oldest entries to keep the store under ``max_entries``."""
        while len(self._jobs) > self._max_entries:
            self._jobs.popitem(last=False)


@dataclass
class JobHandle:
    """Small façade passed to worker functions so they can publish progress."""

    store: JobStore
    job_id: str

    def set_progress(self, done: int, total: int, message: str = "") -> None:
        self.store.update(
            self.job_id,
            progress_done=int(done),
            progress_total=int(total),
            message=message,
        )

    def set_run_id(self, run_id: int) -> None:
        self.store.update(self.job_id, run_id=int(run_id))


# Module-level singleton — small in-memory store, fine for the single-worker
# uvicorn deployment used in production. Tests can build their own JobStore.
_STORE = JobStore()


def get_default_store() -> JobStore:
    """Return the module-level :class:`JobStore` singleton."""
    return _STORE
