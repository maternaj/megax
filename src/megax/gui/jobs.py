"""In-process background jobs with progress for long MC runs."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from megax.storage import load_round_record

JobKind = Literal["optimize", "simulate"]
JobStatus = Literal["running", "done", "error"]

# Keep finished jobs in memory long enough for the GUI poll to observe them.
FINISHED_JOB_TTL_SECONDS = 300.0
# Fallback when in-memory state was lost but results were persisted to disk.
RECOVERY_MAX_AGE_SECONDS = 7200.0


@dataclass
class JobState:
    round_key: str
    kind: JobKind
    status: JobStatus = "running"
    phase: str = ""
    message: str = ""
    done: float = 0.0
    total: float = 1.0
    started_at: float = field(default_factory=time.monotonic)
    updated_at: float = field(default_factory=time.monotonic)
    eta_seconds: float | None = None
    error: str | None = None
    redirect_extra: str = ""

    def elapsed_seconds(self) -> float:
        return max(0.0, time.monotonic() - self.started_at)

    def finished_age_seconds(self) -> float:
        return max(0.0, time.monotonic() - self.updated_at)

    def percent(self) -> float:
        if self.total <= 0:
            return 0.0
        return max(0.0, min(100.0, 100.0 * self.done / self.total))

    def _refresh_eta(self) -> None:
        if self.done <= 0:
            self.eta_seconds = None
            return
        elapsed = self.elapsed_seconds()
        remaining_units = max(0.0, self.total - self.done)
        self.eta_seconds = (elapsed / self.done) * remaining_units

    def to_dict(self) -> dict[str, Any]:
        self._refresh_eta()
        return {
            "round_key": self.round_key,
            "kind": self.kind,
            "running": self.status == "running",
            "status": self.status,
            "phase": self.phase,
            "message": self.message,
            "done": self.done,
            "total": self.total,
            "percent": round(self.percent(), 1),
            "elapsed_seconds": round(self.elapsed_seconds(), 1),
            "eta_seconds": round(self.eta_seconds, 1) if self.eta_seconds is not None else None,
            "error": self.error,
            "redirect_extra": self.redirect_extra,
        }


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, JobState] = {}

    def get(self, round_key: str) -> JobState | None:
        with self._lock:
            job = self._jobs.get(round_key)
            if job is None:
                return None
            if (
                job.status != "running"
                and job.finished_age_seconds() > FINISHED_JOB_TTL_SECONDS
            ):
                del self._jobs[round_key]
                return None
            return job

    def is_running(self, round_key: str) -> bool:
        job = self.get(round_key)
        return job is not None and job.status == "running"

    def start(
        self,
        round_key: str,
        *,
        kind: JobKind,
        total: float,
        message: str,
        redirect_extra: str,
    ) -> JobState:
        with self._lock:
            existing = self._jobs.get(round_key)
            if existing is not None and existing.status == "running":
                raise RuntimeError("Job already running for this round")
            job = JobState(
                round_key=round_key,
                kind=kind,
                total=max(1.0, total),
                message=message,
                redirect_extra=redirect_extra,
            )
            self._jobs[round_key] = job
            return job

    def update(
        self,
        round_key: str,
        *,
        phase: str | None = None,
        message: str | None = None,
        done: float | None = None,
        total: float | None = None,
    ) -> None:
        with self._lock:
            job = self._jobs.get(round_key)
            if job is None or job.status != "running":
                return
            if phase is not None:
                job.phase = phase
            if message is not None:
                job.message = message
            if done is not None:
                job.done = done
            if total is not None:
                job.total = max(1.0, total)
            job.updated_at = time.monotonic()

    def complete(self, round_key: str, *, message: str | None = None) -> None:
        with self._lock:
            job = self._jobs.get(round_key)
            if job is None:
                return
            job.status = "done"
            job.done = job.total
            if message is not None:
                job.message = message
            job.updated_at = time.monotonic()

    def fail(self, round_key: str, error: str) -> None:
        with self._lock:
            job = self._jobs.get(round_key)
            if job is None:
                return
            job.status = "error"
            job.error = error
            job.message = error
            job.updated_at = time.monotonic()


def _parse_iso_timestamp(raw: str | None) -> datetime | None:
    if not raw:
        return None
    text = str(raw)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _age_seconds(ts: datetime) -> float:
    return max(0.0, (datetime.now(timezone.utc) - ts).total_seconds())


def recover_job_payload(round_key: str, expect: JobKind) -> dict[str, Any] | None:
    """Rebuild a terminal job payload from persisted round state after store loss."""
    record = load_round_record(round_key)
    if record is None:
        return None

    state = record.state
    started_at = _parse_iso_timestamp(state.pending_mc_started_at)
    pending = state.pending_mc_job

    if expect == "optimize":
        opt = state.last_optimization
        if opt is None or opt.error:
            if pending == "optimize":
                return {
                    "running": False,
                    "status": "interrupted",
                    "kind": "optimize",
                    "interrupted": True,
                    "message": "Běh přerušen (restart serveru?) — spusťte optimalizaci znovu.",
                }
            return None
        finished_at = _parse_iso_timestamp(opt.optimized_at)
        if finished_at is None:
            return None
        if started_at is not None and finished_at < started_at:
            if pending == "optimize":
                return {
                    "running": False,
                    "status": "interrupted",
                    "kind": "optimize",
                    "interrupted": True,
                    "message": "Běh přerušen — výsledek zatím není k dispozici.",
                }
            return None
        if started_at is None and _age_seconds(finished_at) > RECOVERY_MAX_AGE_SECONDS:
            return None
        return {
            "round_key": round_key,
            "kind": "optimize",
            "running": False,
            "status": "done",
            "recovered": True,
            "message": opt.note or "Optimalizace dokončena",
            "percent": 100.0,
            "redirect_extra": "optimized=1",
        }

    sim = state.last_simulation
    if sim is None or sim.error:
        if pending == "simulate":
            return {
                "running": False,
                "status": "interrupted",
                "kind": "simulate",
                "interrupted": True,
                "message": "Běh přerušen (restart serveru?) — spusťte simulaci znovu.",
            }
        return None
    finished_at = _parse_iso_timestamp(sim.simulated_at)
    if finished_at is None:
        return None
    if started_at is not None and finished_at < started_at:
        if pending == "simulate":
            return {
                "running": False,
                "status": "interrupted",
                "kind": "simulate",
                "interrupted": True,
                "message": "Běh přerušen — výsledek zatím není k dispozici.",
            }
        return None
    if started_at is None and _age_seconds(finished_at) > RECOVERY_MAX_AGE_SECONDS:
        return None
    return {
        "round_key": round_key,
        "kind": "simulate",
        "running": False,
        "status": "done",
        "recovered": True,
        "message": sim.note or "Simulace dokončena",
        "percent": 100.0,
        "redirect_extra": "sim=1",
    }


job_store = JobStore()
