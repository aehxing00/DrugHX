"""DrugHX launcher: starts the local web UI and opens the browser."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

from dockforge.config import APP_VERSION, VINA_DOWNLOAD_URL, VINA_EXE  # noqa: E402

LOCK_FILE = APP_DIR / "data" / "launcher.lock"


def _health_ok(port: int, timeout: float = 0.4) -> bool:
    if not port:
        return False
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/health", timeout=timeout
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return payload.get("app") == "DrugHX"
    except Exception:
        return False


def _read_lock() -> dict:
    try:
        payload = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _create_lock(port: int) -> bool:
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "port": port, "time": time.time()}, stream)
    return True


def _clear_lock() -> None:
    try:
        LOCK_FILE.unlink()
    except OSError:
        pass


def _pid_alive(pid: int) -> bool:
    """True when the process still exists (Windows only)."""
    if not pid or not hasattr(ctypes, "WinDLL"):
        return False
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_bool, ctypes.c_ulong)
    kernel32.GetExitCodeProcess.restype = ctypes.c_bool
    kernel32.GetExitCodeProcess.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ulong),
    )
    kernel32.CloseHandle.restype = ctypes.c_bool
    kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
    handle = kernel32.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == 259  # STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def _wait_for_instance(port: int, seconds: float = 20.0) -> str | None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if _health_ok(port, 0.6):
            return f"http://127.0.0.1:{port}"
        time.sleep(0.25)
    return None


def ensure_vina(allow_download: bool = True) -> bool:
    if VINA_EXE.is_file() and VINA_EXE.stat().st_size > 500_000:
        return True
    if not allow_download:
        return False
    print("[DrugHX] 未找到 Vina，正在从官方 GitHub 下载 vina 1.2.5 ...")
    try:
        VINA_EXE.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(VINA_DOWNLOAD_URL, timeout=30) as source:
            with VINA_EXE.open("wb") as target:
                shutil.copyfileobj(source, target)
        print(f"[DrugHX] 已下载：{VINA_EXE}")
        return True
    except Exception as exc:
        print(f"[DrugHX] Vina 下载失败：{exc}")
        print(f"[DrugHX] 请手动下载 {VINA_DOWNLOAD_URL}")
        print(f"[DrugHX] 并保存为 {VINA_EXE}")
        return False


def free_port(start: int) -> int:
    port = start
    while port < start + 50:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                port += 1
    raise RuntimeError(
        f"端口 {start}–{start + 49} 均被占用，请关闭占用程序或使用 --port 指定其他端口。"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="DrugHX molecular docking workbench")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--no-download", action="store_true", help="do not download Vina automatically")
    parser.add_argument("--version", action="version", version=APP_VERSION)
    args = parser.parse_args()

    # Re-use an instance already started from this installation instead of
    # launching a second server that would write the same project files.
    # Only the lock file is probed: scanning 50 ports can block for 20 s when
    # the local firewall silently drops connections to closed ports.
    try:
        port = free_port(args.port)
    except RuntimeError as exc:
        print(f"[DrugHX] {exc}")
        sys.exit(1)
    if not _create_lock(port):
        lock = _read_lock()
        lock_port = int(lock.get("port") or 0)
        pid = int(lock.get("pid") or 0)
        age = time.time() - float(lock.get("time") or 0)
        if pid and not _pid_alive(pid):
            # The previous instance was killed; its lock is stale.
            _clear_lock()
            if not _create_lock(port):
                print("[DrugHX] 无法获取启动锁，请关闭其他 DrugHX 窗口后重试。")
                sys.exit(1)
        else:
            existing = _wait_for_instance(lock_port) if lock_port else None
            if existing:
                print(f"[DrugHX] 软件已在运行：{existing}")
                if not args.no_browser:
                    webbrowser.open(existing)
                return
            if age < 30:
                print("[DrugHX] 另一个实例正在启动，请稍候再双击启动器。")
                sys.exit(1)
            _clear_lock()
            if not _create_lock(port):
                print("[DrugHX] 无法获取启动锁，请关闭其他 DrugHX 窗口后重试。")
                sys.exit(1)

    vina_ok = ensure_vina(allow_download=not args.no_download)
    url = f"http://{args.host}:{port}"

    print("")
    print("  DrugHX  ·  AutoDock Vina 分子对接工作台")
    print(f"  版本 {APP_VERSION}    Vina {'就绪' if vina_ok else '未找到'}")
    print(f"  界面 {url}")
    print("  按 Ctrl+C 退出")
    print("")

    if not args.no_browser:
        def open_browser():
            # Cold starts normally finish in 2–5 s; keep trying for 60 s on
            # slow disks instead of opening a dead page too early.
            for _ in range(240):
                try:
                    urllib.request.urlopen(url, timeout=0.6)
                    break
                except Exception:
                    time.sleep(0.25)
            webbrowser.open(url)

        threading.Thread(target=open_browser, daemon=True).start()

    import uvicorn

    try:
        uvicorn.run("dockforge.api:app", host=args.host, port=port, log_level="warning")
    finally:
        _clear_lock()


if __name__ == "__main__":
    main()
