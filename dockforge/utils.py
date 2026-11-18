"""Small shared helpers."""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_filename(value: str, fallback: str = "item") -> str:
    """Filesystem-safe file name that keeps the extension intact.

    ``safe_name`` truncates at 80 characters and would cut off a long name's
    ``.smi`` / ``.sdf`` suffix, which made uploads of files with long names fail
    with "unsupported ligand format".  Keep the suffix and trim only the stem.
    """
    path = Path(str(value or ""))
    suffix = re.sub(r"[^0-9A-Za-z.]", "", path.suffix.lower())[:10]
    stem = safe_name(path.stem, fallback)
    limit = max(1, 80 - len(suffix))
    return f"{stem[:limit]}{suffix}"


def safe_name(value: str, fallback: str = "item") -> str:
    """Filesystem-safe name that still keeps CJK characters readable."""
    value = unicodedata.normalize("NFKC", str(value or "")).strip()
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value)
    value = re.sub(r"\s+", "_", value)
    value = value.strip("._ ")
    return value[:80] or fallback


def unique_dir(parent: Path, name: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    base = safe_name(name)
    candidate = parent / base
    i = 2
    while candidate.exists():
        candidate = parent / f"{base}_{i}"
        i += 1
    candidate.mkdir(parents=True)
    return candidate


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def load_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def run_stream(
    cmd: list[str],
    cwd: Path | None = None,
    on_line: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
    on_process: Callable[[subprocess.Popen], None] | None = None,
    timeout: float | None = None,
) -> tuple[int, list[str]]:
    """Run a command, stream stdout+stderr line by line."""
    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=creationflags,
    )
    if on_process:
        on_process(proc)

    lines: list[str] = []
    started = time.time()

    def reader():
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = strip_ansi(raw.rstrip("\r\n"))
            lines.append(line)
            if on_line:
                try:
                    on_line(line)
                except Exception:
                    pass

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()

    while proc.poll() is None:
        if cancel_event is not None and cancel_event.is_set():
            kill_process(proc)
            break
        if timeout is not None and time.time() - started > timeout:
            kill_process(proc)
            break
        time.sleep(0.08)

    thread.join(timeout=5)
    try:
        proc.wait(timeout=5)
    except Exception:
        pass
    return proc.returncode if proc.returncode is not None else -1, lines


def kill_process(proc: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        else:
            proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def tail_lines(lines: Iterable[str], n: int = 40) -> str:
    seq = list(lines)
    return "\n".join(seq[-n:])
