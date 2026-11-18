"""Background job manager with streaming logs and cancellation."""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Callable

from .utils import kill_process, now_iso


class Job:
    def __init__(self, kind: str, title: str):
        self.id = f"job_{uuid.uuid4().hex[:8]}"
        self.kind = kind
        self.title = title
        self.status = "running"  # running | done | error | cancelled
        self.progress = 0.0
        self.message = "准备中 ..."
        self.logs: list[str] = []
        self.log_total = 0
        self.result = None
        self.error = ""
        self.created = now_iso()
        self.started = time.time()
        self.cancel_event = threading.Event()
        self.process = None
        self.processes: set = set()
        self.log_path = None
        self._lock = threading.Lock()

    def log(self, line: str) -> None:
        text = str(line).rstrip()
        if not text:
            return
        with self._lock:
            self.logs.append(text)
            self.log_total += 1
            if len(self.logs) > 3000:
                del self.logs[:1000]
            if self.log_path is not None:
                try:
                    with open(self.log_path, "a", encoding="utf-8") as handle:
                        handle.write(text + "\n")
                except OSError:
                    pass
        if len(text) < 160:
            self.message = text

    def set_progress(self, value: float, message: str | None = None) -> None:
        self.progress = max(0.0, min(1.0, float(value)))
        if message:
            self.message = message

    def set_process(self, proc) -> None:
        self.process = proc
        with self._lock:
            self.processes.add(proc)

    def cancel(self) -> None:
        self.cancel_event.set()
        with self._lock:
            processes = list(self.processes)
        for proc in processes:
            kill_process(proc)
        if self.process is not None:
            kill_process(self.process)

    def snapshot(self, since: int = 0) -> dict:
        with self._lock:
            # ``since`` is an absolute log index; trimmed entries have been
            # dropped from the list, so shift it into the retained window.
            offset = self.log_total - len(self.logs)
            start = max(0, int(since) - offset)
            logs = self.logs[start:]
            total = self.log_total
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "status": self.status,
            "progress": round(self.progress, 3),
            "message": self.message,
            "error": self.error,
            "result": self.result,
            "logs": logs,
            "log_index": total,
            "elapsed": round(time.time() - self.started, 1),
        }


class JobManager:
    log_dir_getter: Callable[[], object] | None = None

    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, kind: str, title: str, fn: Callable[[Job], None]) -> Job:
        job = Job(kind, title)
        if self.log_dir_getter is not None:
            try:
                directory = self.log_dir_getter()
                if directory:
                    directory = Path(directory)
                    directory.mkdir(parents=True, exist_ok=True)
                    job.log_path = directory / f"{job.id}.log"
            except Exception:
                job.log_path = None
        with self._lock:
            self.jobs[job.id] = job
            self._cleanup()
        thread = threading.Thread(target=self._run, args=(job, fn), daemon=True)
        thread.start()
        return job

    def _run(self, job: Job, fn: Callable[[Job], None]) -> None:
        try:
            fn(job)
            if job.status == "running":
                if job.cancel_event.is_set():
                    job.status = "cancelled"
                    job.message = "已取消"
                else:
                    job.status = "done"
                    job.progress = 1.0
                    job.message = job.message or "完成"
        except Exception as exc:
            job.status = "error"
            job.error = str(exc)
            job.message = "失败"
            job.log(f"[错误] {exc}")
            job.log(traceback.format_exc())
        finally:
            job.process = None
            with job._lock:
                job.processes.clear()

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def active(self) -> list[Job]:
        return [job for job in self.jobs.values() if job.status == "running"]

    def _cleanup(self) -> None:
        cutoff = time.time() - 3600 * 6
        # Never evict a job that is still running: its result and logs would
        # become unreachable (HTTP 404) while the worker is still writing.
        stale = [
            jid for jid, j in self.jobs.items()
            if j.status != "running" and j.started < cutoff
        ]
        for jid in stale:
            self.jobs.pop(jid, None)
