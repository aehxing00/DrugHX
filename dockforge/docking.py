"""AutoDock Vina execution, optional GPU engine and result parsing."""
from __future__ import annotations

import os
import random
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import ligands as ligmod
from . import pdbio
from .config import (
    BIN_DIR,
    DOCKING_ENGINES,
    VINA_EXE,
    VINA_GPU_EXE,
    VINA_GPU_KERNELS,
    resolve_engine_path,
)
from .engines import engine_command
from .project import Project
from .utils import run_stream, safe_name

_MODE_RE = re.compile(
    r"^\s*(\d+)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*$"
)
_RESULT_RE = re.compile(
    r"REMARK VINA RESULT:\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)"
)
_SCORE_ONLY_RE = re.compile(
    r"Estimated Free Energy of Binding\s*:\s*(-?\d+(?:\.\d+)?)"
)

# Environment variable that overrides each engine's executable path.
_ENGINE_ENV = {
    "vina": "VINA_PATH",
    "vina-gpu": "VINA_GPU_PATH",
    "smina": "SMINA_PATH",
    "qvina2": "QVINA2_PATH",
}


def parse_vina_log(lines: list[str]) -> list[dict]:
    modes = []
    for line in lines:
        if "affinity" in line or "rmsd" in line.lower():
            continue
        match = _MODE_RE.match(line)
        if not match:
            continue
        modes.append(
            {
                "mode": int(match.group(1)),
                "affinity": float(match.group(2)),
                "rmsd_lb": float(match.group(3)),
                "rmsd_ub": float(match.group(4)),
            }
        )
    return modes


def parse_pose_modes(pose_path: Path) -> list[dict]:
    modes = []
    if not pose_path.exists():
        return modes
    text = pose_path.read_text(encoding="utf-8", errors="replace")
    for match in _RESULT_RE.finditer(text):
        modes.append(
            {
                "mode": len(modes) + 1,
                "affinity": float(match.group(1)),
                "rmsd_lb": float(match.group(2)),
                "rmsd_ub": float(match.group(3)),
            }
        )
    return modes


def _vina_command(
    receptor_path: Path,
    ligand_path: Path,
    pose_path: Path,
    box: dict,
    params: dict,
    seed: int,
) -> list[str]:
    center = box["center"]
    size = box["size"]
    return [
        str(VINA_EXE),
        "--receptor", str(receptor_path),
        "--ligand", str(ligand_path),
        "--center_x", f"{center[0]:.3f}",
        "--center_y", f"{center[1]:.3f}",
        "--center_z", f"{center[2]:.3f}",
        "--size_x", f"{size[0]:.3f}",
        "--size_y", f"{size[1]:.3f}",
        "--size_z", f"{size[2]:.3f}",
        "--exhaustiveness", str(int(params.get("exhaustiveness", 8))),
        "--num_modes", str(int(params.get("num_modes", 9))),
        "--energy_range", f"{float(params.get('energy_range', 3.0)):.2f}",
        "--cpu", str(int(params.get("cpu", 0))),
        "--seed", str(int(seed)),
        "--scoring", str(params.get("scoring", "vina")),
        "--verbosity", str(int(params.get("verbosity", 1))),
        "--out", str(pose_path),
    ]


def _engine_binary(name: str) -> Path | None:
    """Return the executable configured for a docking engine id."""
    spec = DOCKING_ENGINES.get(str(name or "").strip().lower())
    if not spec:
        return None
    return resolve_engine_path(name)


def _require_engine(name: str) -> Path:
    binary = _engine_binary(name)
    if binary is None:
        raise RuntimeError(
            f"未知对接引擎：{name}。可选：{'、'.join(DOCKING_ENGINES)}。"
        )
    if not binary.is_file():
        label = DOCKING_ENGINES.get(name, {}).get("label", name)
        env_name = _ENGINE_ENV.get(name, f"{name.upper()}_PATH")
        raise RuntimeError(
            f"未找到 {label} 可执行文件。请将 {binary.name} 放到 {BIN_DIR} 目录，"
            f"或设置环境变量 {env_name} 指向该文件。"
        )
    return binary


def _vina_like_command(
    binary: Path,
    receptor_path: Path,
    ligand_path: Path,
    pose_path: Path,
    box: dict,
    params: dict,
    seed: int,
    engine: str,
) -> list[str]:
    """Build a smina/QuickVina 2 command from the Vina argument set.

    All three tools accept the Vina CLI, but differ slightly: we never pass
    ``--scoring`` (smina uses its own defaults) and QuickVina 2 does not
    understand ``--energy_range``, ``--cpu`` or ``--verbosity``.
    """
    engine = str(engine).lower()
    base = _vina_command(receptor_path, ligand_path, pose_path, box, params, seed)
    drop = {"--scoring"}
    if engine == "qvina2":
        drop.update({"--energy_range", "--cpu", "--verbosity"})
    command: list[str] = []
    index = 1
    while index < len(base):
        flag = base[index]
        if flag in drop:
            index += 2
            continue
        command.append(flag)
        if index + 1 < len(base) and not base[index + 1].startswith("--"):
            command.append(base[index + 1])
            index += 2
        else:
            index += 1
    return engine_command(binary, command)


def _score_only(
    receptor_path: Path,
    pose_path: Path,
    box: dict,
    output_dir: Path,
    stem: str,
    scoring: str,
) -> float | None:
    """Rescore the top pose with a chosen Vina scoring function."""
    pose1 = output_dir / f"{stem}_pose1.pdbqt"
    pose1.write_text(
        pdbio.pose_pdbqt(
            pose_path.read_text(encoding="utf-8", errors="replace"), 1
        ),
        encoding="utf-8",
    )
    cmd = [
        str(VINA_EXE),
        "--receptor", str(receptor_path),
        "--ligand", str(pose1),
        "--center_x", f"{box['center'][0]:.3f}",
        "--center_y", f"{box['center'][1]:.3f}",
        "--center_z", f"{box['center'][2]:.3f}",
        "--size_x", f"{box['size'][0]:.3f}",
        "--size_y", f"{box['size'][1]:.3f}",
        "--size_z", f"{box['size'][2]:.3f}",
        "--scoring", scoring,
        "--score_only",
    ]
    try:
        _, lines = run_stream(cmd, cwd=output_dir, timeout=180)
        for line in lines:
            match = _SCORE_ONLY_RE.search(line)
            if match:
                return float(match.group(1))
    except Exception:
        return None
    finally:
        pose1.unlink(missing_ok=True)
    return None


def _rescore_vinardo(
    receptor_path: Path,
    pose_path: Path,
    box: dict,
    output_dir: Path,
    stem: str,
) -> float | None:
    return _score_only(receptor_path, pose_path, box, output_dir, stem, "vinardo")


def _run_vina_gpu(
    receptor_path: Path,
    ligand_path: Path,
    output_dir: Path,
    stem: str,
    box: dict,
    params: dict,
    seed: int,
    job,
) -> tuple[int, list[str], Path | None, str, float | None]:
    """Run Vina-GPU 2.0 with multiple independent starts.

    Vina-GPU is a stochastic Monte-Carlo search and a single run can settle in a
    shallow local minimum.  The GPU is fast enough to repeat the search a few
    times; the two best GPU poses are then rescored with the standard CPU Vina
    scoring function and the best pose is kept.
    """
    replicas = max(1, min(int(params.get("gpu_replicas", 1) or 1), 8))
    cwd = VINA_GPU_KERNELS if VINA_GPU_KERNELS.is_dir() else output_dir
    candidates: list[dict] = []
    combined: list[str] = []

    for replica in range(1, replicas + 1):
        if job.cancel_event.is_set():
            break
        replica_seed = int(seed) + (replica - 1) * 7919
        replica_stem = f"{stem}_gpu_r{replica}"
        replica_pose = output_dir / f"{replica_stem}.pdbqt"
        replica_log = output_dir / f"{replica_stem}.log"
        cmd = [
            str(VINA_GPU_EXE),
            "--receptor", str(receptor_path),
            "--ligand", str(ligand_path),
            "--center_x", f"{box['center'][0]:.3f}",
            "--center_y", f"{box['center'][1]:.3f}",
            "--center_z", f"{box['center'][2]:.3f}",
            "--size_x", f"{box['size'][0]:.3f}",
            "--size_y", f"{box['size'][1]:.3f}",
            "--size_z", f"{box['size'][2]:.3f}",
            "--thread", str(int(params.get("gpu_threads", 8000))),
            "--search_depth", str(int(params.get("gpu_search_depth", 20))),
            "--num_modes", str(int(params.get("num_modes", 9))),
            "--energy_range", f"{float(params.get('energy_range', 3.0)):.2f}",
            "--seed", str(replica_seed),
            "--out", str(replica_pose),
            "--log", str(replica_log),
        ]
        code, lines = run_stream(
            cmd,
            cwd=cwd,
            on_line=job.log,
            cancel_event=job.cancel_event,
            on_process=job.set_process,
        )
        if replicas > 1:
            job.log(
                f"[GPU] {stem} 复算 {replica}/{replicas}（seed {replica_seed}）"
            )
        combined.append(
            f"===== GPU replica {replica}/{replicas} | seed {replica_seed} ====="
        )
        combined.extend(lines)
        if code != 0 or not replica_pose.exists():
            continue
        modes = parse_vina_log(lines) or parse_pose_modes(replica_pose)
        gpu_score = modes[0]["affinity"] if modes else None
        candidates.append(
            {
                "replica": replica,
                "seed": replica_seed,
                "pose": replica_pose,
                "log": replica_log,
                "gpu": gpu_score,
                "cpu": None,
                "score": gpu_score,
            }
        )

    if not candidates:
        return 0, combined, None, "Vina-GPU 未生成输出文件", None

    # CPU rescoring is the slow part of a GPU run.  Rescore only the two best
    # GPU candidates and pick the better CPU score.
    ranked = sorted(
        candidates,
        key=lambda item: item["gpu"] if item["gpu"] is not None else 999,
    )
    for item in ranked[:2]:
        item["cpu"] = _score_only(
            receptor_path,
            item["pose"],
            box,
            output_dir,
            f"{stem}_gpu_r{item['replica']}",
            "vina",
        )
        if item["cpu"] is not None:
            item["score"] = item["cpu"]
    best = min(
        ranked[:2],
        key=lambda item: item["score"] if item["score"] is not None else 999,
    )
    pose_path = output_dir / f"{stem}.pdbqt"
    pose_path.write_text(
        best["pose"].read_text(encoding="utf-8", errors="replace"),
        encoding="utf-8",
    )
    if best["cpu"] is not None:
        gpu_text = (
            f"GPU {best['gpu']:.2f}" if best["gpu"] is not None else "GPU 未解析"
        )
        job.log(
            f"[GPU] 采用第 {best['replica']} 次复算（{gpu_text}，"
            f"CPU 复核 {best['cpu']:.2f} kcal/mol）"
        )
    # Keep the combined log for reproducibility and remove per-replica files.
    for path in output_dir.glob(f"{stem}_gpu_r*"):
        path.unlink(missing_ok=True)
    return 0, combined, pose_path, "", best["gpu"]


def _run_single(
    project: Project,
    index: int,
    total: int,
    receptor: dict,
    ligand: dict,
    params: dict,
    output_dir: Path,
    job,
    seed_override: int | None = None,
    replica: int = 1,
) -> dict:
    base = {
        "task_index": index,
        "receptor_id": receptor["id"],
        "receptor_name": receptor["name"],
        "ligand_id": ligand["id"],
        "ligand_name": ligand["name"],
        "modes": [],
    }
    if job.cancel_event.is_set():
        return {**base, "status": "cancelled", "error": "已取消"}
    box = receptor.get("box")
    if not box or not box.get("center") or not box.get("size"):
        return {**base, "status": "error", "error": "未设置对接盒"}

    seed_base = int(params.get("seed") or 0)
    if seed_override is not None:
        seed = int(seed_override)
    else:
        seed = random.randint(1, 2_147_483_647) if seed_base == 0 else seed_base + index
    receptor_path = project.root / receptor["pdbqt"]
    ligand_path = project.root / ligand["pdbqt"]
    # Keep a receptor snapshot with the run so old results stay analysable
    # even if the receptor is re-prepared or deleted later.
    receptor_pdb = project.root / receptor.get("pdb", "")
    if receptor.get("pdb") and receptor_pdb.exists():
        snapshot = output_dir / f"receptor_{receptor['id']}.pdb"
        if not snapshot.exists():
            shutil.copyfile(receptor_pdb, snapshot)
        base["receptor_pdb"] = str(snapshot.relative_to(project.root))
    stem_base = safe_name(f"{receptor['name']}__{ligand['name']}", "docking")
    stem = f"{index:03d}_{stem_base}" + (f"_r{replica}" if replica > 1 else "")
    pose_path = output_dir / f"{stem}.pdbqt"
    log_path = output_dir / f"{stem}.log"
    config_path = output_dir / f"{stem}.txt"

    config_path.write_text(
        "\n".join(
            [
                f"receptor = {receptor_path}",
                f"ligand = {ligand_path}",
                f"center_x = {box['center'][0]:.3f}",
                f"center_y = {box['center'][1]:.3f}",
                f"center_z = {box['center'][2]:.3f}",
                f"size_x = {box['size'][0]:.3f}",
                f"size_y = {box['size'][1]:.3f}",
                f"size_z = {box['size'][2]:.3f}",
                f"exhaustiveness = {params.get('exhaustiveness', 8)}",
                f"num_modes = {params.get('num_modes', 9)}",
                f"energy_range = {params.get('energy_range', 3.0)}",
                f"seed = {seed}",
                f"cpu = {params.get('cpu', 0)}",
                f"scoring = {params.get('scoring', 'vina')}",
                f"engine = {params.get('engine', 'cpu')}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    job.log(f"[{index}/{total}] {ligand['name']} → {receptor['name']}")
    engine = str(params.get("engine", "cpu")).lower()
    docking_engine = str(params.get("docking_engine") or "vina").strip().lower()
    if docking_engine == "vina-gpu":
        # "vina-gpu" is the GPU variant of Vina; keep the existing CPU/GPU flow.
        engine = "gpu"
        docking_engine = "vina"
    elif docking_engine not in DOCKING_ENGINES:
        docking_engine = "vina"
    used_engine = "cpu" if docking_engine == "vina" else docking_engine
    score_label = (
        "Vina"
        if docking_engine == "vina"
        else DOCKING_ENGINES[docking_engine]["label"]
    )
    gpu_reported = None
    if docking_engine != "vina":
        binary = _require_engine(docking_engine)
        job.log(f"[引擎] {score_label}：{binary}")
        code, lines = run_stream(
            _vina_like_command(
                binary, receptor_path, ligand_path, pose_path,
                box, params, seed, docking_engine,
            ),
            cwd=output_dir,
            on_line=job.log,
            cancel_event=job.cancel_event,
            on_process=job.set_process,
        )
    elif engine == "gpu" and VINA_GPU_EXE.is_file():
        code, lines, gpu_pose, gpu_error, gpu_reported = _run_vina_gpu(
            receptor_path, ligand_path, output_dir, stem, box, params, seed, job
        )
        if gpu_pose is not None and code == 0:
            pose_path = gpu_pose
            used_engine = "gpu"
        else:
            job.log(f"[GPU] {ligand['name']} 未完成：{gpu_error or '未知错误'}，回退 CPU")
            code, lines = run_stream(
                _vina_command(receptor_path, ligand_path, pose_path, box, params, seed),
                cwd=output_dir,
                on_line=job.log,
                cancel_event=job.cancel_event,
                on_process=job.set_process,
            )
    else:
        if engine == "gpu":
            job.log("[GPU] 未找到 Vina-GPU，回退 CPU")
        code, lines = run_stream(
            _vina_command(receptor_path, ligand_path, pose_path, box, params, seed),
            cwd=output_dir,
            on_line=job.log,
            cancel_event=job.cancel_event,
            on_process=job.set_process,
        )
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if job.cancel_event.is_set():
        return {**base, "status": "cancelled", "error": "已取消"}

    if code == 0 and pose_path.exists():
        pose_path.write_text(
            pdbio.normalize_pdbqt(
                pose_path.read_text(encoding="utf-8", errors="replace")
            ),
            encoding="utf-8",
        )
    if docking_engine != "vina":
        modes = parse_pose_modes(pose_path) or parse_vina_log(lines)
    elif used_engine == "gpu":
        modes = parse_pose_modes(pose_path)
    else:
        modes = parse_vina_log(lines) or parse_pose_modes(pose_path)
    if code != 0 or not pose_path.exists():
        tail = (
            " | ".join(lines[-6:])
            if lines
            else f"{score_label} 退出码 {code}"
        )
        job.log(f"[失败] {ligand['name']} × {receptor['name']}: {tail}")
        return {
            **base,
            "status": "error",
            "error": tail,
            "box": box,
            "seed": seed,
            "engine": used_engine,
        }

    best = modes[0]["affinity"] if modes else None
    gpu_affinity = gpu_reported if used_engine == "gpu" else None
    if used_engine == "gpu" and modes:
        # Vina-GPU reports its own search score; recompute the pose with the
        # standard Vina scoring function so CPU and GPU results are comparable.
        rescored = _score_only(
            receptor_path, pose_path, box, output_dir, stem, "vina"
        )
        if rescored is not None:
            best = rescored
            modes[0]["gpu_affinity"] = gpu_affinity
            modes[0]["affinity"] = rescored
    vinardo = None
    if params.get("rescore", True) and modes and docking_engine == "vina":
        vinardo = _rescore_vinardo(
            receptor_path, pose_path, box, output_dir, stem
        )
    result = {
        **base,
        "status": "ok" if modes else "error",
        "error": "" if modes else "未能解析打分结果",
        "affinity": best,
        "gpu_affinity": gpu_affinity,
        "vinardo": vinardo,
        "modes": modes,
        "pose": str(pose_path.relative_to(project.root)),
        "log": str(log_path.relative_to(project.root)),
        "box": box,
        "seed": seed,
        "engine": used_engine,
        "engine_scores": {
            ("vina" if docking_engine == "vina" else docking_engine): best
        },
    }
    job.log(
        f"[完成] {ligand['name']} × {receptor['name']}：{score_label} {best:.2f}"
        + (f"（GPU 原始 {gpu_affinity:.2f}）" if gpu_affinity is not None else "")
        + (f"，Vinardo {vinardo:.2f}" if vinardo is not None else "")
        + " kcal/mol"
        if best is not None
        else f"[完成] {ligand['name']} × {receptor['name']}"
    )
    return result


def _spearman(pairs: list[tuple[float, float]]) -> float | None:
    """Spearman rank correlation without an extra scipy dependency."""
    if len(pairs) < 3:
        return None
    import numpy as np

    xs = np.array([item[0] for item in pairs], dtype=float)
    ys = np.array([item[1] for item in pairs], dtype=float)
    xs = np.argsort(np.argsort(xs)).astype(float)
    ys = np.argsort(np.argsort(ys)).astype(float)
    if xs.std() == 0 or ys.std() == 0:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


_ENGINE_WEIGHTS = {
    "vina": 1.0,
    "vinardo": 0.6,
    "smina": 0.8,
    "qvina2": 0.6,
}


def _assign_consensus(results: list[dict], primary: str = "vina") -> dict:
    """Rank hits per engine and build one mean-rank consensus.

    ``vina_rank`` is the rank of the score that actually drove the docking
    (Vina, or the selected primary engine).  It must not be counted twice when
    that engine also appears in ``engine_scores``, otherwise the consensus
    silently double-weights the primary engine.
    """
    ok = [r for r in results if r.get("status") == "ok"]
    vina_sorted = sorted(ok, key=lambda r: r.get("affinity", 999))
    for rank, item in enumerate(vina_sorted, start=1):
        item["vina_rank"] = rank
    vinardo_items = [r for r in ok if r.get("vinardo") is not None]
    vinardo_sorted = sorted(vinardo_items, key=lambda r: r["vinardo"])
    for rank, item in enumerate(vinardo_sorted, start=1):
        item["vinardo_rank"] = rank
    engines: list[str] = []
    for item in ok:
        for name, value in (item.get("engine_scores") or {}).items():
            if value is not None and name not in engines:
                engines.append(name)
    for name in engines:
        scored = [
            r for r in ok
            if (r.get("engine_scores") or {}).get(name) is not None
        ]
        scored.sort(key=lambda r: r["engine_scores"][name])
        for rank, item in enumerate(scored, start=1):
            item.setdefault("engine_ranks", {})[name] = rank
    for item in ok:
        # Keyed by engine name so a repeated primary engine overwrites rather
        # than adds an extra vote.
        parts: dict[str, float] = {primary: float(item.get("vina_rank", len(ok)))}
        if item.get("vinardo_rank"):
            parts["vinardo"] = float(item["vinardo_rank"])
        for name, rank in (item.get("engine_ranks") or {}).items():
            if isinstance(rank, (int, float)):
                parts[name] = float(rank)
        item["consensus_parts"] = sorted(parts)
        weights = {name: _ENGINE_WEIGHTS.get(name, 0.5) for name in parts}
        total_weight = sum(weights.values()) or 1.0
        item["consensus_rank"] = round(
            sum(parts[name] * weights[name] for name in parts) / total_weight, 2
        )
        values = list(parts.values())
        spread = (max(values) - min(values)) if len(values) > 1 else 0.0
        item["consensus_spread"] = round(spread, 2)
        # Selective prediction: when engines disagree strongly, say so instead
        # of pretending the mean rank is a confident answer.
        if len(parts) < 2:
            item["consensus_confidence"] = "single"
        elif spread >= max(3.0, 0.35 * len(ok)):
            item["consensus_confidence"] = "low"
        elif spread <= 1.0:
            item["consensus_confidence"] = "high"
        else:
            item["consensus_confidence"] = "medium"
    low_confidence = sum(
        1 for item in ok if item.get("consensus_confidence") == "low"
    )
    correlations: dict[str, float] = {}
    for index, first in enumerate(engines):
        for second in engines[index + 1:]:
            pairs = [
                (r["engine_scores"][first], r["engine_scores"][second])
                for r in ok
                if (r.get("engine_scores") or {}).get(first) is not None
                and (r.get("engine_scores") or {}).get(second) is not None
            ]
            value = _spearman(pairs)
            if value is not None:
                correlations[f"{first}|{second}"] = round(value, 3)
    return {
        "engines": engines,
        "correlations": correlations,
        "count": len(ok),
        "weights": {name: _ENGINE_WEIGHTS.get(name, 0.5) for name in engines},
        "low_confidence": low_confidence,
    }


_CONSENSUS_ENGINES = ("vina", "smina", "qvina2")


def _consensus_engines(primary: str) -> list[str]:
    """Engines that can be added to the primary Vina run on this machine."""
    found: list[str] = []
    for name in _CONSENSUS_ENGINES:
        if name == primary:
            continue
        binary = _engine_binary(name)
        if binary is not None and binary.is_file():
            found.append(name)
    return found


def _run_secondary_engine(
    project: Project,
    receptor: dict,
    ligand: dict,
    engine: str,
    params: dict,
    output_dir: Path,
    stem: str,
    seed: int,
    job,
) -> tuple[str, float | None, str]:
    """Dock one ligand with a secondary engine and return its best score."""
    binary = _require_engine(engine)
    box = receptor.get("box") or {}
    if not box.get("center") or not box.get("size"):
        return engine, None, "未设置对接盒"
    receptor_path = project.root / receptor["pdbqt"]
    ligand_path = project.root / ligand["pdbqt"]
    pose_path = output_dir / f"{stem}_{engine}.pdbqt"
    log_path = output_dir / f"{stem}_{engine}.log"
    local = {
        **params,
        "rescore": False,
        "cpu": max(1, min(int(params.get("cpu") or 1), 4)),
    }
    code, lines = run_stream(
        _vina_like_command(
            binary, receptor_path, ligand_path, pose_path, box, local, seed, engine
        ),
        cwd=output_dir,
        on_line=job.log,
        cancel_event=job.cancel_event,
        on_process=job.set_process,
    )
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if code != 0 or not pose_path.exists():
        label = DOCKING_ENGINES.get(engine, {}).get("label", engine)
        return engine, None, f"{label} 退出码 {code}"
    modes = parse_pose_modes(pose_path) or parse_vina_log(lines)
    return engine, (modes[0]["affinity"] if modes else None), ""



def _repair_macrocycle_pdbqt(project: Project, ligand: dict, job) -> None:
    """Re-prepare old ligands that contain Meeko macrocycle glue atoms.

    Older DrugHX versions let Meeko open macrocycles, producing "G0"/"CG0"
    atom types.  Vina-GPU 2.0 (and some Vina builds) cannot parse them, which
    can lead to a native memory fault.  Re-prepare the ligand with rigid
    macrocycles so the PDBQT only contains standard AutoDock types.
    """
    rel = ligand.get("pdbqt")
    if not rel:
        return
    path = project.root / rel
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    if " CG0" not in text and " G0" not in text:
        return
    smiles = ligand.get("smiles") or ligand.get("meta", {}).get("smiles") or ""
    if not smiles:
        return
    job.log(f"[配体] {ligand.get('name', '')} 含宏环 glue 原子，自动重新准备 ...")
    mol = ligmod.mol_from_smiles(smiles)
    _, pdbqt, meta = ligmod.prepare_mol_for_docking(
        mol,
        protonation=str(ligand.get("protonation", "ph7.4")),
        minimize=True,
        seed=42,
    )
    path.write_text(pdbqt, encoding="utf-8")
    ligand["meta"] = {**ligand.get("meta", {}), **meta}
    project.save()


_VALID_AD_TYPES = {
    "H", "HD", "C", "A", "N", "NA", "NS", "OA", "OS", "S", "SA",
    "F", "CL", "BR", "I", "P", "MG", "MN", "ZN", "CA", "FE",
    "CU", "NI", "CO", "CD", "HG", "W", "G0", "CG0", "CG1", "CG2", "CG3",
}

_PARSE_ERROR_RE = re.compile(
    r'Parse error on line \d+ in file "([^"]+)":\s*(.+)'
)


def _pdbqt_issue(path: Path) -> str:
    """Return why Vina cannot parse a PDBQT file, or '' when it looks valid."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return f"无法读取文件：{exc}"
    if "ROOT" not in text or "ENDROOT" not in text:
        return "缺少 ROOT/ENDROOT 段"
    atoms = 0
    for line in text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if len(line) < 54:
            return "原子行长度不足"
        try:
            float(line[30:38])
            float(line[38:46])
            float(line[46:54])
        except ValueError:
            return "原子坐标格式错误"
        tokens = line.split()
        atom_type = tokens[-1] if tokens else ""
        if atom_type.upper() not in _VALID_AD_TYPES:
            return f"AutoDock 不支持的原子类型 {atom_type or '?'}"
        atoms += 1
    return "" if atoms > 0 else "没有原子记录"


def _pdbqt_readable(path: Path) -> bool:
    return not _pdbqt_issue(path)


_GPU_STATS_CMD = [
    "nvidia-smi",
    "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
    "--format=csv,noheader,nounits",
]


def _gpu_stats_text() -> str:
    """Return a one-line NVIDIA GPU utilisation summary (empty when unavailable)."""
    try:
        proc = subprocess.run(
            _GPU_STATS_CMD, capture_output=True, text=True, timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    lines = (proc.stdout or "").strip().splitlines()
    if not lines:
        return ""
    parts = [part.strip() for part in lines[0].split(",")]
    if len(parts) < 4:
        return ""
    try:
        util = int(float(parts[0]))
        used = float(parts[1])
        total = float(parts[2])
    except ValueError:
        return ""
    power = parts[3]
    return (
        f"利用率 {util}%，显存 {used / 1024:.1f}/{total / 1024:.1f} GB，"
        f"功耗 {power} W"
    )



class GpuBatchError(RuntimeError):
    """Vina-GPU produced no result at all for a whole batch."""


_GPU_NOISE = ("Perform docking", "######", "Output will be in")


def _gpu_output_tail(lines: list[str], limit: int = 10) -> list[str]:
    """Return the last meaningful Vina-GPU output lines for error reporting."""
    tail: list[str] = []
    for raw in lines:
        line = (raw or "").strip()
        if not line or any(token in line for token in _GPU_NOISE):
            continue
        tail.append(line[:300])
    return tail[-limit:]


def _gpu_batch_once(
    project: Project,
    receptor: dict,
    group: list[dict],
    work_dir: Path,
    params: dict,
    job,
    produced: dict[str, Path],
    progress: dict,
    total: int,
) -> tuple[int, dict[str, str]]:
    lig_dir = work_dir / "ligands"
    out_dir = work_dir / "out"
    lig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ligand in group:
        source = project.root / ligand["pdbqt"]
        target = lig_dir / f"{ligand['id']}.pdbqt"
        if target.exists():
            continue
        try:
            os.link(source, target)
        except FileExistsError:
            continue
        except OSError:
            try:
                if not target.exists():
                    shutil.copyfile(source, target)
            except (FileExistsError, shutil.SameFileError, OSError):
                # Never let one duplicate/unreadable ligand abort a batch;
                # it will simply have no output and be recorded as failed.
                continue

    box = receptor.get("box") or {}
    center = box.get("center") or [0, 0, 0]
    size = box.get("size") or [20, 20, 20]
    # Vina-GPU rejects --thread below 1000 with a usage error.
    threads = max(1000, min(int(params.get("gpu_threads", 2000) or 2000), 20000))
    depth = max(1, min(int(params.get("gpu_search_depth", 2) or 2), 100))
    modes_wanted = max(1, min(int(params.get("num_modes", 1) or 1), 9))
    energy_range = float(params.get("energy_range", 3.0) or 3.0)
    seed = int(params.get("seed") or 42)
    command = [
        str(VINA_GPU_EXE),
        "--receptor", str(project.root / receptor["pdbqt"]),
        "--ligand_directory", str(lig_dir),
        "--output_directory", str(out_dir),
        "--center_x", f"{center[0]:.3f}",
        "--center_y", f"{center[1]:.3f}",
        "--center_z", f"{center[2]:.3f}",
        "--size_x", f"{size[0]:.3f}",
        "--size_y", f"{size[1]:.3f}",
        "--size_z", f"{size[2]:.3f}",
        "--thread", str(threads),
        "--search_depth", str(depth),
        "--num_modes", str(modes_wanted),
        "--energy_range", f"{energy_range:.2f}",
        "--seed", str(seed),
    ]

    def on_line(line: str) -> None:
        if "output...done" in line:
            progress["done"] += 1
            job.set_progress(
                min(1.0, progress["done"] / max(total, 1)),
                f"GPU 粗筛 {progress['done']}/{total}",
            )
            if progress["done"] % 500 == 0:
                job.log(f"[GPU] 已完成 {progress['done']}/{total}")
        elif any(
            key in line.lower()
            for key in ("error", "parse error", "not enough memory", "cuda")
        ):
            job.log(line)

    cwd = VINA_GPU_KERNELS if VINA_GPU_KERNELS.is_dir() else work_dir
    code, lines = run_stream(
        command,
        cwd=cwd,
        on_line=on_line,
        cancel_event=job.cancel_event,
        on_process=job.set_process,
    )
    parse_errors: dict[str, str] = {}
    for line in lines:
        match = _PARSE_ERROR_RE.search(line)
        if match:
            parse_errors[Path(match.group(1)).stem] = match.group(2).strip()
    for ligand in group:
        output = out_dir / f"{ligand['id']}_out.pdbqt"
        if output.exists() and output.stat().st_size > 0:
            produced[ligand["id"]] = output
    return code, parse_errors, lines


def _gpu_batch_group(
    project: Project,
    receptor: dict,
    group: list[dict],
    work_dir: Path,
    params: dict,
    job,
    produced: dict[str, Path],
    errors: dict[str, str],
    progress: dict,
    total: int,
    tag: str = "b0",
    depth_left: int = 10,
    top: bool = True,
) -> None:
    """Run a GPU batch, isolating individual ligands only when necessary."""
    if not group or job.cancel_event.is_set():
        return
    before = len(produced)
    code, parse_errors, lines = _gpu_batch_once(
        project, receptor, group, work_dir / tag, params, job,
        produced, progress, total,
    )
    missing = [ligand for ligand in group if ligand["id"] not in produced]
    if not missing:
        return

    # Vina-GPU aborts the whole batch on a PDBQT parse error.  Identify the
    # offending file from its message, record it as failed, and retry the rest.
    bad = [ligand for ligand in group if ligand["id"] in parse_errors]
    if bad:
        skip = set()
        for ligand in bad:
            message = parse_errors[ligand["id"]]
            errors[ligand["id"]] = message
            skip.add(ligand["id"])
            job.log(f"[跳过] {ligand['name']}：{message}")
        remaining = [ligand for ligand in missing if ligand["id"] not in skip]
        if remaining and not job.cancel_event.is_set() and depth_left > 0:
            _gpu_batch_group(
                project, receptor, remaining, work_dir, params, job,
                produced, errors, progress, total, tag + "r", depth_left - 1, False,
            )
        elif remaining:
            for ligand in remaining:
                errors[ligand["id"]] = "Vina-GPU 批量重试次数用尽"
        return

    all_missing = len(missing) == len(group)
    if all_missing and top:
        # No output and no parse error means the engine itself did not start
        # (driver, VRAM pressure, or antivirus locking the fresh PDBQT files).
        # Retry the untouched batch once before escalating to the CPU.
        tail = _gpu_output_tail(lines)
        job.log(
            f"[GPU] 本批 {len(group)} 个配体无任何输出（退出码 {code}），"
            "2 秒后整批重试一次"
        )
        for line in tail:
            job.log(f"[GPU]   {line}")
        if not job.cancel_event.is_set():
            time.sleep(2)
            _gpu_batch_group(
                project, receptor, group, work_dir, params, job,
                produced, errors, progress, total,
                tag + "r", max(0, depth_left - 1), False,
            )
        if not any(ligand["id"] in produced for ligand in group):
            detail = tail[-1] if tail else "无详细错误输出"
            raise GpuBatchError(
                f"Vina-GPU 整批无输出（退出码 {code}，{len(group)} 个配体）：{detail}"
            )
        return

    if all_missing and len(group) > 64:
        # A whole batch failing without a single output points at the GPU
        # runtime rather than one ligand; retrying thousands of times would
        # freeze the queue, so surface the real error instead.
        detail = (_gpu_output_tail(lines) or ["无详细错误输出"])[-1]
        raise GpuBatchError(
            f"Vina-GPU 批量粗筛未能启动（退出码 {code}，{len(group)} 个配体）：{detail}"
        )
    if len(group) == 1:
        errors[group[0]["id"]] = "Vina-GPU 未能解析该配体"
        return
    if depth_left <= 0:
        for ligand in missing:
            errors[ligand["id"]] = "Vina-GPU 批量重试后仍无结果"
        return
    mid = max(1, len(missing) // 2)
    if before == len(produced) and len(group) <= 4:
        for ligand in missing:
            errors[ligand["id"]] = "Vina-GPU 未能处理该配体"
        return
    _gpu_batch_group(
        project, receptor, missing[:mid], work_dir, params, job,
        produced, errors, progress, total, tag + "a", depth_left - 1, False,
    )
    _gpu_batch_group(
        project, receptor, missing[mid:], work_dir, params, job,
        produced, errors, progress, total, tag + "b", depth_left - 1, False,
    )


def _run_vina_gpu_library(
    project: Project,
    run: dict,
    receptors: list[dict],
    ligands: list[dict],
    params: dict,
    job,
) -> dict:
    """Screen the whole ligand library in one or a few Vina-GPU batch calls."""
    out_root = project.run_dir(run) / "gpu_batch"
    if out_root.exists():
        shutil.rmtree(out_root, ignore_errors=True)
    out_root.mkdir(parents=True, exist_ok=True)
    total = len(receptors) * len(ligands)
    results: list[dict] = []
    progress = {"done": 0}
    # Vina-GPU rejects --thread below 1000 with a usage error.
    threads = max(1000, min(int(params.get("gpu_threads", 2000) or 2000), 20000))
    depth = max(1, min(int(params.get("gpu_search_depth", 2) or 2), 100))
    chunk_size = max(100, min(int(params.get("gpu_chunk", 1500) or 1500), 5000))
    job.log(
        f"[GPU] 批量粗筛：{len(ligands)} 个配体 × {len(receptors)} 个受体，"
        f"thread={threads}，search_depth={depth}，每批 {chunk_size} 个"
    )
    job.log(
        "[GPU] 提示：Vina-GPU 是单卡单进程引擎，解析配体和精修结果时 GPU 会短暂空闲；"
        "对接阶段 GPU 利用率可达 90–100%，下面的实时监控日志会给出真实数值。"
    )

    monitor_stop = threading.Event()

    def _monitor() -> None:
        while not monitor_stop.wait(30):
            stats = _gpu_stats_text()
            if stats:
                job.log(f"[GPU] {stats}　已完成 {progress['done']}/{total}")

    watcher = threading.Thread(target=_monitor, daemon=True)
    watcher.start()
    started_at = time.monotonic()


    for rec_index, receptor in enumerate(receptors):
        rec_dir = out_root / receptor["id"]
        rec_dir.mkdir(parents=True, exist_ok=True)
        receptor_pdb_rel = ""
        if receptor.get("pdb"):
            source_pdb = project.root / receptor["pdb"]
            if source_pdb.exists():
                snapshot = out_root / f"receptor_{receptor['id']}.pdb"
                if not snapshot.exists():
                    shutil.copyfile(source_pdb, snapshot)
                receptor_pdb_rel = str(snapshot.relative_to(project.root))

        valid: list[dict] = []
        invalid: list[tuple[dict, str]] = []
        for ligand in ligands:
            source = project.root / ligand["pdbqt"]
            issue = _pdbqt_issue(source)
            if issue:
                invalid.append((ligand, issue))
            else:
                valid.append(ligand)
        if invalid:
            preview = "、".join(
                f"{ligand['name']}（{reason}）"
                for ligand, reason in invalid[:8]
            )
            job.log(
                f"[配体] 跳过 {len(invalid)} 个 AutoDock 无法解析的配体："
                + preview
                + ("…" if len(invalid) > 8 else "")
            )
        produced: dict[str, Path] = {}
        errors: dict[str, str] = {}
        try:
            for chunk_index, offset in enumerate(range(0, len(valid), chunk_size)):
                chunk = valid[offset:offset + chunk_size]
                if not chunk:
                    continue
                _gpu_batch_group(
                    project, receptor, chunk, rec_dir / f"batch{chunk_index}",
                    params, job, produced, errors, progress, total,
                )
                done = rec_index * len(ligands) + len(produced) + len(invalid)
                elapsed = max(time.monotonic() - started_at, 1e-6)
                rate = done / elapsed
                left = (total - done) / rate if rate > 0 else 0.0
                job.log(
                    f"[GPU] 进度 {done}/{total}，平均 {1 / rate:.2f} s/配体，"
                    f"预计剩余 {left / 60:.1f} 分钟"
                )
                job.set_progress(
                    min(1.0, done / max(total, 1)),
                    f"GPU 粗筛 {done}/{total}",
                )
        finally:
            monitor_stop.set()
        for ligand, reason in invalid:
            errors[ligand["id"]] = reason

        for lig_index, ligand in enumerate(ligands):
            index = rec_index * len(ligands) + lig_index + 1
            base = {
                "task_index": index,
                "receptor_id": receptor["id"],
                "receptor_name": receptor["name"],
                "ligand_id": ligand["id"],
                "ligand_name": ligand["name"],
                "modes": [],
                "engine": "gpu",
            }
            if receptor_pdb_rel:
                base["receptor_pdb"] = receptor_pdb_rel
            output = produced.get(ligand["id"])
            if output is None:
                results.append(
                    {
                        **base,
                        "status": "error",
                        "error": errors.get(ligand["id"], "未生成对接结果"),
                    }
                )
                continue
            modes = parse_pose_modes(output)
            if not modes:
                results.append(
                    {
                        **base,
                        "status": "error",
                        "error": "未能解析打分结果",
                        "pose": str(output.relative_to(project.root)),
                    }
                )
                continue
            results.append(
                {
                    **base,
                    "status": "ok",
                    "affinity": modes[0]["affinity"],
                    "seed": int(params.get("seed") or 42),
                    "pose": str(output.relative_to(project.root)),
                }
            )
        results.sort(key=lambda item: item.get("task_index", 0))
        run["results"] = list(results)
        project.save()
        progress["done"] = rec_index * len(ligands) + len(produced) + len(invalid)
        job.set_progress(
            min(1.0, progress["done"] / max(total, 1)),
            f"GPU 粗筛 {progress['done']}/{total}",
        )

    results.sort(key=lambda item: item.get("task_index", 0))
    run["results"] = list(results)
    run["status"] = "cancelled" if job.cancel_event.is_set() else "done"
    run["finished"] = True
    project.save()
    ok = [item for item in results if item.get("status") == "ok"]
    best = min(ok, key=lambda item: item["affinity"]) if ok else None
    job.result = {
        "run_id": run["id"],
        "total": total,
        "ok": len(ok),
        "best": (
            {
                "ligand": best["ligand_name"],
                "receptor": best["receptor_name"],
                "affinity": best["affinity"],
                "vinardo": None,
            }
            if best
            else None
        ),
    }
    if job.cancel_event.is_set():
        job.message = "已取消"
        job.status = "cancelled"
    return job.result


def run_docking(
    project: Project,
    run: dict,
    receptor_ids: list[str],
    ligand_ids: list[str],
    params: dict,
    job,
) -> dict:
    docking_engine = str(params.get("docking_engine") or "vina").strip().lower()
    if docking_engine == "vina-gpu":
        # Selecting the GPU entry is equivalent to engine="gpu" plus Vina.
        params = {**params, "engine": "gpu"}
        docking_engine = "vina"
    elif docking_engine not in DOCKING_ENGINES:
        raise RuntimeError(
            f"未知对接引擎：{docking_engine}。可选：{'、'.join(DOCKING_ENGINES)}。"
        )
    if docking_engine != "vina":
        # smina / QuickVina 2 are CPU tools; the Vina CPU/GPU switch
        # does not apply to them.
        params = {**params, "engine": "cpu"}
    if docking_engine == "vina":
        if not VINA_EXE.is_file():
            raise RuntimeError(
                "未找到 Vina 可执行文件。请把 vina.exe 放到 bin/ 目录，"
                "或设置环境变量 VINA_PATH。"
            )
    else:
        _require_engine(docking_engine)
    receptors = [r for r in (project.get_receptor(rid) for rid in receptor_ids) if r]
    ligand_index = {item["id"]: item for item in project.data.get("ligands", [])}
    # Old projects may contain duplicate ligand IDs (6-hex collisions); each
    # ligand must appear only once in a run.
    ligand_ids = list(dict.fromkeys(ligand_ids))
    ligands = [ligand_index[lid] for lid in ligand_ids if lid in ligand_index]
    if not receptors or not ligands:
        raise RuntimeError("请至少选择一个受体和一个配体")

    invalid_rec = [
        r["name"]
        for r in receptors
        if not r.get("pdbqt") or not (project.root / r["pdbqt"]).exists()
    ]
    invalid_lig = [
        l["name"]
        for l in ligands
        if not l.get("pdbqt") or not (project.root / l["pdbqt"]).exists()
    ]
    if invalid_rec:
        raise RuntimeError("以下结构尚未准备完成：" + "、".join(invalid_rec))
    if invalid_lig:
        if not params.get("skip_invalid"):
            raise RuntimeError("以下结构尚未准备完成：" + "、".join(invalid_lig))
        job.log(
            f"[对接] 跳过 {len(invalid_lig)} 个未准备完成的配体："
            + "、".join(invalid_lig[:10])
            + ("…" if len(invalid_lig) > 10 else "")
        )
        valid = {
            l["id"] for l in ligands
            if l.get("pdbqt") and (project.root / l["pdbqt"]).exists()
        }
        ligands = [l for l in ligands if l["id"] in valid]
        if not ligands:
            raise RuntimeError("所有配体都尚未准备完成")
        ligand_ids = [l["id"] for l in ligands]

    if not params.get("skip_repair"):
        for ligand in ligands:
            try:
                _repair_macrocycle_pdbqt(project, ligand, job)
            except Exception as exc:
                job.log(f"[配体] {ligand.get('name', '')} 宏环重准备失败：{exc}")

    pairs = [(rec, lig) for rec in receptors for lig in ligands]
    total = len(pairs)
    output_dir = project.run_dir(run)
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = str(params.get("engine", "cpu")).lower()
    if engine == "gpu" and not VINA_GPU_EXE.is_file():
        raise RuntimeError(
            "未找到 Vina-GPU。请将 Vina-GPU 2.0 可执行文件放到 bin/vina-gpu.exe，"
            "或设置环境变量 VINA_GPU_PATH。"
        )

    if params.get("bulk") and engine == "gpu" and docking_engine == "vina":
        # Thousands of ligands: one GPU process handles the whole directory,
        # avoiding per-ligand process start-up and repeated CPU rescoring.
        try:
            return _run_vina_gpu_library(project, run, receptors, ligands, params, job)
        except GpuBatchError as exc:
            # A driver / VRAM / antivirus hiccup must not abort a long run:
            # continue the same screening on CPU so the pipeline still finishes.
            job.log(f"[GPU] {exc}")
            job.log(
                "[GPU] 已自动回退到 CPU（Vina）继续本次粗筛，结果同样有效；"
                "速度会明显变慢，如需 GPU 可稍后重新运行。"
            )
            params = {**params, "engine": "cpu", "bulk": False}
            engine = "cpu"

    parallel = max(1, min(int(params.get("parallel_jobs", 1) or 1), 16))
    if engine == "gpu" and parallel > 1:
        # Vina-GPU 2.0 is not safe to launch from several threads at once on
        # Windows; concurrent OpenCL contexts can crash the native process.
        job.log("[GPU] 检测到并行任务，已改为串行执行以避免 OpenCL 冲突")
        parallel = 1
    cpu_total = int(params.get("cpu", 0) or 0) or (os.cpu_count() or 1)
    params = dict(params)
    if engine == "cpu":
        params["cpu"] = max(1, cpu_total // parallel) if parallel > 1 else cpu_total
    job.log(f"对接任务：{len(receptors)} 个受体 × {len(ligands)} 个配体 = {total} 次计算")
    if docking_engine == "vina":
        engine_desc = f"{engine.upper()}（{VINA_EXE if engine == 'cpu' else VINA_GPU_EXE}）"
    else:
        engine_desc = f"{DOCKING_ENGINES[docking_engine]['label']}（{_engine_binary(docking_engine)}）"
    job.log(
        f"引擎：{engine_desc}，并行任务：{parallel}，每任务 CPU：{params.get('cpu')}，"
        f"双打分：{'开' if params.get('rescore', True) and docking_engine == 'vina' else '关'}"
    )

    results: list[dict] = []
    results_lock = threading.Lock()
    completed = 0
    # Bulk screens can contain thousands of ligands; saving the whole project
    # after every result made the job quadratic in project size.
    save_every = max(1, min(500, max(1, total) // 20))

    def record(result: dict) -> None:
        nonlocal completed
        with results_lock:
            results.append(result)
            completed += 1
            if completed % save_every == 0 or completed == total:
                # Sorting thousands of results after every single ligand was
                # quadratic; sort only when flushing to disk.
                results.sort(key=lambda item: item.get("task_index", 0))
                run["results"] = list(results)
                project.save()
            job.set_progress(completed / total, f"已完成 {completed}/{total}")

    replicates = max(1, min(int(params.get("seed_replicates", 1) or 1), 10))

    def run_pair(index: int, receptor: dict, ligand: dict) -> dict:
        """Dock one pair, optionally repeating with independent random seeds."""
        if replicates <= 1:
            return _run_single(
                project, index, total, receptor, ligand, params, output_dir, job
            )
        base_seed = int(params.get("seed") or 0) or random.randint(
            1, 2_000_000_000
        )
        attempts = []
        for replica in range(1, replicates + 1):
            attempts.append(
                _run_single(
                    project, index, total, receptor, ligand, params, output_dir, job,
                    seed_override=base_seed + replica * 7919,
                    replica=replica,
                )
            )
            if job.cancel_event.is_set():
                break
        ok = [
            item for item in attempts
            if item.get("status") == "ok" and item.get("affinity") is not None
        ]
        if not ok:
            return attempts[-1]
        best = dict(min(ok, key=lambda item: float(item["affinity"])))
        values = [float(item["affinity"]) for item in ok]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        best["replicates"] = [
            {
                "seed": item.get("seed"),
                "affinity": item.get("affinity"),
                "status": item.get("status"),
            }
            for item in attempts
        ]
        best["replicate_mean"] = round(mean, 3)
        best["replicate_sd"] = round(variance ** 0.5, 3)
        best["replicate_range"] = round(max(values) - min(values), 3)
        best["seed_replicates"] = replicates
        return best

    if parallel > 1 and total > 1:
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(
                    run_pair, index, receptor, ligand,
                ): index
                for index, (receptor, ligand) in enumerate(pairs, start=1)
            }
            for future in as_completed(futures):
                if job.cancel_event.is_set():
                    continue
                try:
                    record(future.result())
                except Exception as exc:
                    index = futures[future]
                    receptor, ligand = pairs[index - 1]
                    job.log(f"[失败] {ligand['name']} × {receptor['name']}: {exc}")
                    record(
                        {
                            "task_index": index,
                            "receptor_id": receptor["id"],
                            "receptor_name": receptor["name"],
                            "ligand_id": ligand["id"],
                            "ligand_name": ligand["name"],
                            "status": "error",
                            "error": str(exc),
                            "modes": [],
                        }
                    )
    else:
        for index, (receptor, ligand) in enumerate(pairs, start=1):
            if job.cancel_event.is_set():
                break
            try:
                result = run_pair(index, receptor, ligand)
                record(result)
            except Exception as exc:
                job.log(f"[失败] {ligand['name']} × {receptor['name']}: {exc}")
                record(
                    {
                        "task_index": index,
                        "receptor_id": receptor["id"],
                        "receptor_name": receptor["name"],
                        "ligand_id": ligand["id"],
                        "ligand_name": ligand["name"],
                        "status": "error",
                        "error": str(exc),
                        "modes": [],
                    }
                )

    if params.get("consensus") and not job.cancel_event.is_set():
        # One job runs the primary Vina docking and every other installed
        # Vina-compatible engine over the same receptor/ligand pairs, so the
        # results can be compared without launching several jobs by hand.
        secondary = _consensus_engines(docking_engine)
        targets = [item for item in results if item.get("status") == "ok"]
        if secondary and targets:
            labels = "、".join(
                DOCKING_ENGINES.get(name, {}).get("label", name) for name in secondary
            )
            job.log(
                f"[共识] 追加引擎：{labels}，共 {len(targets)} × {len(secondary)} 次计算"
            )
            pair_map = {
                index: (rec, lig)
                for index, (rec, lig) in enumerate(pairs, start=1)
            }
            total_secondary = len(targets) * len(secondary)
            done_secondary = 0
            counter_lock = threading.Lock()

            def run_extra(item: dict) -> None:
                nonlocal done_secondary
                if job.cancel_event.is_set():
                    return
                rec, lig = pair_map[item["task_index"]]
                stem = f"{item['task_index']:03d}_" + safe_name(
                    f"{rec['name']}__{lig['name']}", "docking"
                )
                for engine in secondary:
                    if job.cancel_event.is_set():
                        return
                    _, score, error = _run_secondary_engine(
                        project, rec, lig, engine, params, output_dir, stem,
                        int(item.get("seed") or params.get("seed") or 42), job,
                    )
                    if score is not None:
                        item.setdefault("engine_scores", {})[engine] = score
                    elif error:
                        job.log(f"[共识] {lig['name']} · {engine}：{error}")
                    with counter_lock:
                        done_secondary += 1
                        job.set_progress(
                            0.95 + 0.05 * done_secondary / max(total_secondary, 1),
                            f"多引擎共识 {done_secondary}/{total_secondary}",
                        )

            with ThreadPoolExecutor(max_workers=max(1, min(4, len(targets)))) as executor:
                list(executor.map(run_extra, targets))
        elif params.get("consensus"):
            job.log("[共识] 未检测到 smina / QuickVina 2，已跳过多引擎复算")

    results.sort(key=lambda item: item.get("task_index", 0))
    consensus_stats = _assign_consensus(results, primary=docking_engine)
    if len(consensus_stats.get("engines") or []) >= 2:
        run["consensus"] = consensus_stats
    run["results"] = list(results)
    run["status"] = "cancelled" if job.cancel_event.is_set() else "done"
    run["finished"] = True
    project.save()

    ok = [r for r in results if r.get("status") == "ok"]
    best = min(ok, key=lambda r: r["affinity"]) if ok else None
    job.result = {
        "run_id": run["id"],
        "total": total,
        "ok": len(ok),
        "best": (
            {
                "ligand": best["ligand_name"],
                "receptor": best["receptor_name"],
                "affinity": best["affinity"],
                "vinardo": best.get("vinardo"),
            }
            if best
            else None
        ),
    }
    if job.cancel_event.is_set():
        job.message = "已取消"
        job.status = "cancelled"
    return job.result
