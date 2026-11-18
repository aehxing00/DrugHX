"""DrugHX launcher: starts the local web UI and opens the browser."""
from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from dockforge.config import APP_VERSION, VINA_DOWNLOAD_URL, VINA_EXE  # noqa: E402


def ensure_vina(allow_download: bool = True) -> bool:
    if VINA_EXE.is_file() and VINA_EXE.stat().st_size > 500_000:
        return True
    if not allow_download:
        return False
    print("[DrugHX] 未找到 Vina，正在从官方 GitHub 下载 vina 1.2.5 ...")
    try:
        VINA_EXE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(VINA_DOWNLOAD_URL, VINA_EXE)
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
    return start


def main() -> None:
    parser = argparse.ArgumentParser(description="DrugHX molecular docking workbench")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--no-download", action="store_true", help="do not download Vina automatically")
    parser.add_argument("--version", action="version", version=APP_VERSION)
    args = parser.parse_args()

    vina_ok = ensure_vina(allow_download=not args.no_download)
    port = free_port(args.port)
    url = f"http://{args.host}:{port}"

    print("")
    print("  DrugHX  ·  AutoDock Vina 分子对接工作台")
    print(f"  版本 {APP_VERSION}    Vina {'就绪' if vina_ok else '未找到'}")
    print(f"  界面 {url}")
    print("  按 Ctrl+C 退出")
    print("")

    if not args.no_browser:
        def open_browser():
            for _ in range(60):
                try:
                    urllib.request.urlopen(url, timeout=0.6)
                    break
                except Exception:
                    time.sleep(0.25)
            webbrowser.open(url)

        threading.Thread(target=open_browser, daemon=True).start()

    import uvicorn

    uvicorn.run("dockforge.api:app", host=args.host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
