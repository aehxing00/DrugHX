"""Reproducibility manifest for a docking run and its report.

A report is only trustworthy if the exact inputs, engine versions, seeds and
parameters can be recovered later.  This module writes that record next to the
PDF and can verify later that the underlying files have not changed.

The manifest is deliberately plain JSON: it is easy to archive, diff in a
repository, attach to a paper's supplementary material or hand to a reviewer.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

from .config import (
    APP_VERSION,
    QVINA2_EXE,
    SMINA_EXE,
    VINA_EXE,
    VINA_GPU_EXE,
)

_PACKAGES = ("rdkit", "meeko", "gemmi", "numpy", "scipy", "matplotlib", "pypdf", "posebusters")


def _digest(path: Path) -> dict | None:
    try:
        if not path.is_file():
            return None
        sha = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                sha.update(chunk)
        return {"sha256": sha.hexdigest(), "bytes": path.stat().st_size}
    except OSError:
        return None


def _version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "?"


def _rel(project_root: Path, value: str | Path) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    try:
        return str(path.relative_to(project_root)).replace("\\", "/")
    except ValueError:
        return str(path)


def build(project, run: dict, params: dict | None = None, extra: dict | None = None) -> dict:
    """Collect everything needed to reproduce ``run``."""
    root = project.root
    results = [item for item in run.get("results", []) if item.get("status") == "ok"]
    receptor_files: dict[str, dict] = {}
    ligand_files: dict[str, dict] = {}
    seen_rec: set[str] = set()
    seen_lig: set[str] = set()
    for item in results[:200]:
        receptor_id = str(item.get("receptor_id") or "")
        if receptor_id and receptor_id not in seen_rec:
            seen_rec.add(receptor_id)
            receptor = project.get_receptor(receptor_id) or {}
            for key in ("pdb", "pdbqt"):
                value = receptor.get(key)
                if value:
                    receptor_files[f"{receptor_id}:{key}"] = {
                        "path": _rel(root, value),
                        "digest": _digest(root / value),
                    }
        ligand_id = str(item.get("ligand_id") or "")
        if ligand_id and ligand_id not in seen_lig:
            seen_lig.add(ligand_id)
            ligand = project.get_ligand(ligand_id) or {}
            for key in ("sdf", "pdbqt"):
                value = ligand.get(key)
                if value:
                    ligand_files[f"{ligand_id}:{key}"] = {
                        "path": _rel(root, value),
                        "digest": _digest(root / value),
                    }

    engines = {}
    for name, path in (
        ("vina", VINA_EXE),
        ("vina-gpu", VINA_GPU_EXE),
        ("smina", SMINA_EXE),
        ("qvina2", QVINA2_EXE),
    ):
        digest = _digest(Path(path))
        if digest:
            engines[name] = {"path": str(path), **digest}

    manifest = {
        "schema": "drughx-manifest/1",
        "app": {"name": "DrugHX", "version": APP_VERSION},
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "project": {
            "name": project.data.get("name"),
            "path": str(root),
            "created": project.data.get("created"),
        },
        "run": {
            "id": run.get("id"),
            "engine": run.get("engine") or run.get("docking_engine"),
            "params": params or run.get("params") or {},
            "seed": (params or run.get("params") or {}).get("seed"),
            "results_ok": len(results),
            "results_total": len(run.get("results", [])),
        },
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "packages": {name: _version(name) for name in _PACKAGES},
        },
        "engines": engines,
        "inputs": {
            "receptors": receptor_files,
            "ligands": ligand_files,
        },
        "notes": extra or {},
    }
    return manifest


def write(path: Path, manifest: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def write_for_run(project, run: dict, outdir: Path, params: dict | None = None, extra: dict | None = None) -> tuple[Path, dict]:
    manifest = build(project, run, params=params, extra=extra)
    path = Path(outdir) / f"{run.get('id', 'run')}_manifest.json"
    return write(path, manifest), manifest


def verify(project_root: Path, manifest: dict) -> dict:
    """Re-hash every recorded input and report mismatches."""
    root = Path(project_root)
    checked = 0
    mismatches: list[dict] = []
    for group in ("receptors", "ligands"):
        for key, entry in ((manifest.get("inputs") or {}).get(group) or {}).items():
            expected = (entry or {}).get("digest")
            if not expected:
                continue
            checked += 1
            current = _digest(root / (entry or {}).get("path", ""))
            if not current or current.get("sha256") != expected.get("sha256"):
                mismatches.append(
                    {
                        "input": key,
                        "path": (entry or {}).get("path"),
                        "expected": expected.get("sha256"),
                        "current": (current or {}).get("sha256"),
                    }
                )
    return {"checked": checked, "mismatches": mismatches, "ok": not mismatches}


def summary(manifest: dict) -> list[tuple[str, str]]:
    """Compact key/value rows for the report appendix."""
    env = manifest.get("environment") or {}
    run = manifest.get("run") or {}
    inputs = manifest.get("inputs") or {}
    return [
        ("软件版本", f"DrugHX {(manifest.get('app') or {}).get('version', '—')}"),
        ("运行编号", str(run.get("id") or "—")),
        ("对接引擎", str(run.get("engine") or "—")),
        ("随机种子", str(run.get("seed") or "—")),
        ("环境", f"{env.get('platform', '—')} · Python {env.get('python', '—')}"),
        (
            "核心库",
            " · ".join(
                f"{name} {version}"
                for name, version in (env.get("packages") or {}).items()
                if name in ("rdkit", "meeko", "posebusters")
            )
            or "—",
        ),
        ("受体输入", str(len(inputs.get("receptors") or {}))),
        ("配体输入", str(len(inputs.get("ligands") or {}))),
    ]
