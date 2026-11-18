"""Optional docking-engine installation helpers.

Windows builds are installed from conda-forge and run with their own runtime
DLLs so they do not clash with QuickVina's newer boost libraries.
"""
from __future__ import annotations

import io
import json
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

from .config import BIN_DIR, resolve_engine_path, save_engine_path

CONDA_API = "https://api.anaconda.org/package/conda-forge"
SMINA_RUNTIME = BIN_DIR / "smina_runtime"

# smina 2020.12.10 needs boost 1.74, OpenBabel 3.1 and the MSVC runtime.  All
# packages below are the exact win-64 builds that provide the required DLLs.
_SMINA_PACKAGES = [
    ("smina", "smina-2020.12.10-h94ab393_0", {"smina.exe"}),
    (
        "boost-cpp",
        "boost-cpp-1.74.0-h54f0996_3",
        {
            "boost_iostreams.dll",
            "boost_timer.dll",
            "boost_chrono.dll",
            "boost_regex.dll",
            "boost_thread.dll",
            "boost_serialization.dll",
            "boost_filesystem.dll",
            "boost_program_options.dll",
        },
    ),
    ("openbabel", "openbabel-3.1.1-py38h62647bb_5", {"openbabel-3.dll", "inchi.dll"}),
    (
        "vs2015_runtime",
        "vs2015_runtime-14.29.30139-h890b9b1_9",
        {
            "msvcp140.dll",
            "msvcp140_1.dll",
            "msvcp140_2.dll",
            "msvcp140_atomic_wait.dll",
            "msvcp140_codecvt_ids.dll",
            "vcruntime140.dll",
            "vcruntime140_1.dll",
            "concrt140.dll",
        },
    ),
    ("bzip2", "bzip2-1.0.8-h8ffe710_4", {"libbz2.dll", "bzip2.dll"}),
    ("zstd", "zstd-1.5.2-h7755175_4", {"libzstd.dll", "zstd.dll"}),
    ("libzlib", "libzlib-1.2.13-hcfcfb64_4", {"zlib.dll"}),
    ("xz", "xz-5.2.5-h2fa13f4_0", {"liblzma.dll"}),
]

def _json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "DrugHX/2.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def _is_batch(binary: Path) -> bool:
    return binary.suffix.lower() in (".bat", ".cmd")


def engine_command(binary: Path, args: list[str]) -> list[str]:
    """Build a subprocess command, routing .bat launchers through cmd.exe."""
    if _is_batch(binary):
        return ["cmd.exe", "/c", str(binary), *args]
    return [str(binary), *args]


def _probe(binary: Path) -> str:
    try:
        result = subprocess.run(
            engine_command(binary, ["--version"]),
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"无法启动 {binary.name}：{exc}") from exc
    text = (result.stdout or result.stderr or "").strip()
    if result.returncode != 0:
        raise ValueError(
            f"{binary.name} 启动失败（退出码 {result.returncode}）：{text[:300]}"
        )
    return text.splitlines()[0] if text else ""


def _conda_tar(package: str, suffix: str) -> bytes:
    data = _json(f"{CONDA_API}/{package}")
    files = [
        item
        for item in data.get("files") or []
        if item.get("attrs", {}).get("subdir") == "win-64"
        and str(item.get("basename", "")).endswith(".tar.bz2")
        and (not suffix or suffix in item.get("basename", ""))
    ]
    if not files:
        raise ValueError(f"conda-forge 上没有 {package} 的 Windows 构建")
    url = "https:" + files[-1]["download_url"]
    with urllib.request.urlopen(url, timeout=600) as response:
        return response.read()


def _install_smina() -> Path:
    bin_dir = SMINA_RUNTIME / "Library" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    for package, build, wanted in _SMINA_PACKAGES:
        data = _conda_tar(package, build)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:bz2") as archive:
            found = set()
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                name = Path(member.name).name.lower()
                if name in wanted:
                    (bin_dir / Path(member.name).name).write_bytes(
                        archive.extractfile(member).read()
                    )
                    found.add(name)
                elif package == "openbabel" and name.endswith(".obf"):
                    # OpenBabel format/operation plugins are .obf shared objects.
                    (bin_dir / Path(member.name).name).write_bytes(
                        archive.extractfile(member).read()
                    )
                elif package == "openbabel" and member.name.startswith("share/openbabel/"):
                    target = SMINA_RUNTIME / member.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.extractfile(member).read())
            missing = wanted - found
            if missing:
                raise ValueError(
                    f"{package} 缺少运行文件：{'、'.join(sorted(missing))}"
                )
    wrapper = SMINA_RUNTIME / "smina.bat"
    wrapper.write_text(
        "@echo off\r\n"
        'set "BABEL_DATADIR=%~dp0share\\openbabel"\r\n'
        'set "BABEL_LIBDIR=%~dp0Library\\bin"\r\n'
        '"%~dp0Library\\bin\\smina.exe" %*\r\n',
        encoding="utf-8",
    )
    _probe(wrapper)
    save_engine_path("smina", str(wrapper))
    return wrapper


def install(engine_id: str) -> Path:
    """Install an engine with a verified runtime and return its path."""
    engine = str(engine_id or "").strip().lower()
    if engine == "qvina2":
        data = _conda_tar("qvina", "qvina-2.1.0-h720a4ee_1")
        target = BIN_DIR / "qvina2.exe"
        BIN_DIR.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:bz2") as archive:
            member = next(
                (item for item in archive.getmembers() if item.name.endswith("qvina2.exe")),
                None,
            )
            if member is None:
                raise ValueError("下载包中未找到 qvina2.exe")
            target.write_bytes(archive.extractfile(member).read())
        _probe(target)
        save_engine_path("qvina2", str(target))
        return target
    if engine == "smina":
        return _install_smina()
    raise ValueError(f"未知对接引擎：{engine_id}")


def install_defaults() -> dict[str, str]:
    """Install the default optional engines, reporting instead of raising.

    A failed download must not abort the whole installer, so every failure is
    returned as a readable message.
    """
    result: dict[str, str] = {}
    for engine_id in ("smina", "qvina2"):
        try:
            existing = resolve_engine_path(engine_id)
            if existing and Path(existing).is_file():
                result[engine_id] = f"已安装：{existing}"
                continue
            result[engine_id] = f"已安装：{install(engine_id)}"
        except Exception as exc:  # noqa: BLE001
            result[engine_id] = f"未安装：{exc}"
    return result


if __name__ == "__main__":
    for engine_id, message in install_defaults().items():
        print(f"[{engine_id}] {message}")


def current_path(engine_id: str) -> Path | None:
    return resolve_engine_path(engine_id)
