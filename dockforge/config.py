"""Global paths and defaults for DrugHX."""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

APP_NAME = "DrugHX"
APP_AUTHOR = "HX"
APP_VERSION = "2.15.0"

# 商业合规模式：默认关闭，全部在线功能保持可用。设置环境变量
# DRUGHX_COMMERCIAL=1 后，软件会自动跳过仅授权学术使用的数据源
# （SwissTargetPrediction、DGIdb），并把 KEGG 富集默认库换成 Reactome，
# 其余功能不受影响。
COMMERCIAL_MODE = os.environ.get("DRUGHX_COMMERCIAL", "0").strip().lower() in {
    "1", "true", "yes", "on",
}

APP_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = APP_DIR / "web"
BIN_DIR = APP_DIR / "bin"
VINA_EXE = Path(os.environ.get("VINA_PATH", BIN_DIR / "vina.exe"))
VINA_GPU_EXE = Path(os.environ.get("VINA_GPU_PATH", BIN_DIR / "vina-gpu.exe"))
VINA_GPU_KERNELS = BIN_DIR / "vina-gpu-kernels"
SMINA_EXE = Path(os.environ.get("SMINA_PATH", BIN_DIR / "smina_runtime" / "smina.bat"))
QVINA2_EXE = Path(os.environ.get("QVINA2_PATH", BIN_DIR / "qvina2.exe"))

VINA_DOWNLOAD_URL = (
    "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/"
    "v1.2.5/vina_1.2.5_win.exe"
)

# Selectable docking engines.  "vina" is the default and keeps the original
# CPU/GPU behaviour; the other entries are Vina-CLI compatible binaries.
DOCKING_ENGINES = {
    "vina": {
        "label": "AutoDock Vina 1.2.5",
        "path": VINA_EXE,
        "description": "官方 AutoDock Vina 1.2.5，CPU 多线程，作为打分与 Vinardo 复算基准。",
        "supports_rescore": True,
        "kind": "vina-like",
    },
    "vina-gpu": {
        "label": "Vina-GPU 2.0",
        "path": VINA_GPU_EXE,
        "description": "Vina-GPU 2.0，OpenCL 显卡加速，多次随机复算后取最优并用 CPU 复核。",
        "supports_rescore": True,
        "kind": "gpu",
    },
    "smina": {
        "label": "smina",
        "path": SMINA_EXE,
        "description": "基于 Vina 的增强版，支持自定义打分函数与更精细的局部优化，命令行与 Vina 兼容。",
        "supports_rescore": False,
        "kind": "vina-like",
    },
    "qvina2": {
        "label": "QuickVina 2",
        "path": QVINA2_EXE,
        "description": "QuickVina 2，加速版 Vina，搜索速度更快，命令行参数为 Vina 的子集。",
        "supports_rescore": False,
        "kind": "vina-like",
    },
}

# All user data lives under <DrugHX>/data so the install root stays clean.
DATA_DIR = APP_DIR / "data"
PROJECTS_DIR = Path(os.environ.get("DOCKFORGE_PROJECTS", DATA_DIR / "projects"))
LOGS_DIR = Path(os.environ.get("DOCKFORGE_LOGS", DATA_DIR / "logs"))
LEGACY_PROJECTS_DIR = Path.home() / "DockForgeProjects"


def _load_plugin_engines() -> None:
    """Register user-supplied Vina-CLI engines from data/engines.json.

    The file is optional; each entry needs ``id``, ``label`` and ``path``.
    Custom engines follow the Vina command line and are treated as
    ``vina-like`` so they can join multi-engine consensus runs.
    """
    try:
        payload = json.loads((DATA_DIR / "engines.json").read_text(encoding="utf-8"))
    except Exception:
        return
    items = payload.get("engines") if isinstance(payload, dict) else payload
    for item in items or []:
        if not isinstance(item, dict):
            continue
        engine_id = str(item.get("id") or "").strip().lower()
        engine_path = str(item.get("path") or "").strip()
        if not engine_id or not engine_path or engine_id in DOCKING_ENGINES:
            continue
        DOCKING_ENGINES[engine_id] = {
            "label": str(item.get("label") or engine_id),
            "path": Path(engine_path),
            "description": str(
                item.get("description") or "自定义 Vina 兼容引擎（data/engines.json）"
            ),
            "supports_rescore": False,
            "kind": "vina-like",
        }


_load_plugin_engines()

RECENT_FILE = Path.home() / ".dockforge_recent.json"

DEFAULT_DOCK_PARAMS = {
    "exhaustiveness": 16,
    "num_modes": 9,
    "energy_range": 4.0,
    "cpu": 0,
    "seed": 42,
    "scoring": "vina",
    "verbosity": 1,
    "rescore": True,
    "engine": "cpu",
    "docking_engine": "vina",
    "parallel_jobs": 1,
    "gpu_threads": 8000,
    "gpu_search_depth": 20,
    "gpu_replicas": 1,
}

# Default drug-likeness bounds.  Users can override every value in the manual
# and automatic preparation panels; filters.normalize_limits clamps them.
FILTER_LIMIT_DEFAULTS = {
    "mw": 500.0,
    "logp": 4.5,
    "tpsa": 140.0,
    "hbd": 5,
    "hba": 10,
    "rotb": 10,
    "heavy": 45,
    "rings": 6,
    "max_ring": 8,
    "qed": 0.30,
}

DEFAULT_LIGAND_OPTIONS = {
    "protonation": "ph7.4",   # keep | neutral | ph7.4
    "minimize": True,
    "seed": 42,
    "threads": 8,
    # Reject PAINS / non-drug-like molecules before 3D embedding and PDBQT
    # generation.  Can be switched off in the manual preparation panel.
    "filter": True,
    "filter_limits": dict(FILTER_LIMIT_DEFAULTS),
}

# One-click docking presets.  Defaults target a 16-core / 32-thread desktop
# (Ryzen 9 7945HX) with an RTX 4070; ``cpu: 0`` means "use all logical cores".
# The frontend only pre-fills these values — every field stays editable.
DOCK_PRESETS = [
    {
        "id": "single-fine",
        "label": "单配体精细（Vina CPU）",
        "description": "少量关键配体的最终结合模式：高 exhaustiveness、保留 20 个构象并做 Vinardo 交叉打分。",
        "params": {
            "docking_engine": "vina", "exhaustiveness": 32, "num_modes": 20,
            "energy_range": 5.0, "cpu": 0, "seed": 42, "scoring": "vina",
            "parallel_jobs": 1, "rescore": True,
        },
    },
    {
        "id": "batch-balanced",
        "label": "批量标准（Vina CPU）",
        "description": "几十到几百个配体的常规筛选：并行 6 任务、exhaustiveness 8，兼顾速度与可信度。",
        "params": {
            "docking_engine": "vina", "exhaustiveness": 8, "num_modes": 9,
            "energy_range": 4.0, "cpu": 0, "seed": 42, "scoring": "vina",
            "parallel_jobs": 6, "rescore": True,
        },
    },
    {
        "id": "ultra-fast",
        "label": "超多配体极速粗筛（Vina CPU）",
        "description": "上万配体的第一轮粗排：exhaustiveness 1、每个配体只留 1 个构象、并行 16 任务、关闭交叉打分。",
        "params": {
            "docking_engine": "vina", "exhaustiveness": 1, "num_modes": 1,
            "energy_range": 3.0, "cpu": 0, "seed": 42, "scoring": "vina",
            "parallel_jobs": 16, "rescore": False,
        },
    },
    {
        "id": "gpu-fast",
        "label": "GPU 极速粗筛（Vina-GPU 批量）",
        "description": "RTX 4070 上的一次性整批粗筛：单个 GPU 进程连续处理整个配体库，search_depth 1、每配体 1 次复算，优先保证吞吐。",
        "params": {
            "docking_engine": "vina-gpu", "exhaustiveness": 8, "num_modes": 1,
            "energy_range": 3.0, "cpu": 0, "seed": 42, "scoring": "vina",
            "parallel_jobs": 1, "rescore": False,
            "gpu_threads": 2000, "gpu_search_depth": 1, "gpu_replicas": 1,
        },
    },
    {
        "id": "gpu-fine",
        "label": "GPU 精细（Vina-GPU 复算）",
        "description": "GPU 多次随机复算并用 CPU Vina 复核前两名，适合几十个候选的精细排序。",
        "params": {
            "docking_engine": "vina-gpu", "exhaustiveness": 8, "num_modes": 9,
            "energy_range": 4.0, "cpu": 0, "seed": 42, "scoring": "vina",
            "parallel_jobs": 1, "rescore": True,
            "gpu_threads": 4000, "gpu_search_depth": 20, "gpu_replicas": 4,
        },
    },
    {
        "id": "smina-fine",
        "label": "smina 精细（自定义打分）",
        "description": "smina 的局部优化更充分：exhaustiveness 32、20 个构象，适合先导化合物精细评估。",
        "params": {
            "docking_engine": "smina", "exhaustiveness": 32, "num_modes": 20,
            "energy_range": 5.0, "cpu": 0, "seed": 42,
            "parallel_jobs": 1, "rescore": False,
        },
    },
    {
        "id": "qvina-fine",
        "label": "QuickVina 2 精细",
        "description": "QuickVina 2 的精细模式：exhaustiveness 8、保留 9 个构象，速度约为主流 Vina 的数倍。",
        "params": {
            "docking_engine": "qvina2", "exhaustiveness": 8, "num_modes": 9,
            "energy_range": 4.0, "cpu": 0, "seed": 42,
            "parallel_jobs": 1, "rescore": False,
        },
    },
    {
        "id": "smina-screen",
        "label": "smina 批量筛选",
        "description": "smina 并行 8 任务、exhaustiveness 8，适合几百个类似物的相对排序。",
        "params": {
            "docking_engine": "smina", "exhaustiveness": 8, "num_modes": 9,
            "energy_range": 4.0, "cpu": 0, "seed": 42,
            "parallel_jobs": 8, "rescore": False,
        },
    },
    {
        "id": "qvina-fast",
        "label": "QuickVina 2 极速粗筛",
        "description": "上万配体的首选：QuickVina 2 搜索速度约为 Vina 的数十倍，exhaustiveness 2、只留 1 个构象、并行 16 任务，适合先粗排再回精细对接。",
        "params": {
            "docking_engine": "qvina2", "exhaustiveness": 2, "num_modes": 1,
            "energy_range": 3.0, "cpu": 0, "seed": 42,
            "parallel_jobs": 16, "rescore": False,
        },
    },
]

DEFAULT_RECEPTOR_OPTIONS = {
    "remove_water": True,
    "remove_hetero": True,
    "keep_ions": False,
    "chains": [],
    "altloc": "A",
    "charge_model": "template",  # template | gasteiger
    "delete_unknown": True,
}

STANDARD_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "HID", "HIE", "HIP", "CYX", "CYM", "ASH", "GLH", "LYN", "ACE", "NME",
    "UNK",
}

WATER_NAMES = {"HOH", "WAT", "DOD", "H2O", "TIP", "TIP3", "SOL"}

ION_NAMES = {
    "ZN", "MG", "CA", "MN", "FE", "FE2", "CU", "CU1", "NA", "K", "CL", "BR",
    "IOD", "NI", "CO", "CD", "HG", "SR", "CS", "BA", "MO", "W",
}

ION_CHARGES = {
    "ZN": 2.0, "MG": 2.0, "CA": 2.0, "MN": 2.0, "FE": 2.0, "FE2": 2.0,
    "CU": 2.0, "CU1": 1.0, "NA": 1.0, "K": 1.0, "CL": -1.0, "BR": -1.0,
    "IOD": -1.0, "NI": 2.0, "CO": 2.0, "CD": 2.0, "HG": 2.0, "SR": 2.0,
    "CS": 1.0, "BA": 2.0, "MO": 4.0, "W": 4.0,
}

# AutoDock Vina 1.2.x supports large boxes; this is only a soft warning.
# 30 Å per side (~27 000 Å³) is already generous for a single binding site.
BOX_VOLUME_HINT = 27000


def python_exe() -> str:
    return sys.executable


ENGINE_PATH_FILE = DATA_DIR / "engine_paths.json"


def load_engine_paths() -> dict:
    try:
        data = json.loads(ENGINE_PATH_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_engine_path(engine_id: str, path: str) -> None:
    data = load_engine_paths()
    data[str(engine_id)] = str(path)
    ENGINE_PATH_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def resolve_engine_path(engine_id: str) -> Path | None:
    """Resolve an engine executable: saved path, env var, bin/, then PATH."""
    key = str(engine_id or "").strip().lower()
    spec = DOCKING_ENGINES.get(key)
    if not spec:
        return None
    saved = str(load_engine_paths().get(key) or "").strip()
    if saved:
        return Path(saved)
    env_name = {
        "vina": "VINA_PATH",
        "vina-gpu": "VINA_GPU_PATH",
        "smina": "SMINA_PATH",
        "qvina2": "QVINA2_PATH",
    }.get(key)
    if env_name:
        env_value = os.environ.get(env_name, "").strip()
        if env_value:
            return Path(env_value)
    default = spec.get("path")
    if default and Path(default).is_file():
        return Path(default)
    found = shutil.which(str(default.name if default else key))
    if found:
        return Path(found)
    found = shutil.which(key)
    return Path(found) if found else (Path(default) if default else None)


def engine_status() -> list[dict]:
    """Return availability of every docking engine for the API/UI."""
    installable = {"smina", "qvina2"}
    status = []
    for engine_id, spec in DOCKING_ENGINES.items():
        path = resolve_engine_path(engine_id)
        available = bool(path) and Path(path).is_file()
        status.append(
            {
                "id": engine_id,
                "label": spec.get("label", engine_id),
                "available": available,
                "path": str(path) if path else "",
                "default_path": str(spec.get("path") or ""),
                "installable": engine_id in installable,
                "description": spec.get("description", ""),
            }
        )
    return status
