"""Project model and persistence."""
from __future__ import annotations

import os
import shutil
import time
import uuid
from pathlib import Path

from .config import (
    APP_DIR,
    DEFAULT_DOCK_PARAMS,
    DEFAULT_LIGAND_OPTIONS,
    DEFAULT_RECEPTOR_OPTIONS,
    ENGINE_PATH_FILE,
    LEGACY_PROJECTS_DIR,
    LOGS_DIR,
    PROJECTS_DIR,
    RECENT_FILE,
)
from .utils import load_json, now_iso, safe_name, stamp, unique_dir, write_json

_LEGACY_MIGRATED = False

# Old job logs are only useful while debugging; keep a month by default.
LOG_RETENTION_DAYS = int(os.environ.get("DOCKFORGE_LOG_RETENTION_DAYS", "30"))


def _prune_logs(folder: Path, days: int = LOG_RETENTION_DAYS) -> None:
    """Delete rotated job logs older than ``days`` (0 disables pruning)."""
    if days <= 0 or not folder.is_dir():
        return
    cutoff = time.time() - days * 86400
    for file in folder.glob("*.log*"):
        try:
            if file.is_file() and file.stat().st_mtime < cutoff:
                file.unlink()
        except OSError:
            continue


def _new_id(prefix: str) -> str:
    # 12 hex characters keep collisions negligible even for 20k+ ligands
    # (6 characters produced a handful of duplicate IDs in large libraries).
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _migrate_legacy_projects() -> None:
    """Move projects and settings into the new ``data`` layout once."""
    global _LEGACY_MIGRATED
    if _LEGACY_MIGRATED:
        return
    _LEGACY_MIGRATED = True
    try:
        PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        _prune_logs(LOGS_DIR)
        legacy = Path(LEGACY_PROJECTS_DIR)
        if legacy.exists() and legacy.resolve() != PROJECTS_DIR.resolve():
            for item in legacy.iterdir():
                if not item.is_dir() or not (item / "project.json").is_file():
                    continue
                target = PROJECTS_DIR / item.name
                if not target.exists():
                    shutil.copytree(item, target)
        old_projects = APP_DIR / "prj"
        if old_projects.exists() and old_projects.resolve() != PROJECTS_DIR.resolve():
            for item in list(old_projects.iterdir()):
                if not item.is_dir() or not (item / "project.json").is_file():
                    continue
                target = PROJECTS_DIR / item.name
                if not target.exists():
                    shutil.move(str(item), str(target))
            try:
                old_projects.rmdir()
            except OSError:
                pass
        # Every project keeps its own logs folder for debugging.
        for item in PROJECTS_DIR.iterdir():
            if item.is_dir() and (item / "project.json").is_file():
                (item / "logs").mkdir(exist_ok=True)
        old_engine_paths = APP_DIR / "engine_paths.json"
        if old_engine_paths.is_file() and not ENGINE_PATH_FILE.exists():
            shutil.copy2(old_engine_paths, ENGINE_PATH_FILE)
        # Recent-project entries still point at <root>/prj; rewrite them.
        if RECENT_FILE.is_file():
            recent = load_json(RECENT_FILE, [])
            changed = False
            for item in recent if isinstance(recent, list) else []:
                path = Path(str(item.get("path") or ""))
                if old_projects in path.parents and (PROJECTS_DIR / path.name).exists():
                    item["path"] = str(PROJECTS_DIR / path.name)
                    changed = True
            if changed:
                write_json(RECENT_FILE, recent)
    except Exception:
        # Never block startup because a legacy folder could not be moved.
        pass


def _normalise_consensus(data: dict) -> None:
    """Rebuild ``consensus_rank`` as the mean of the available engine ranks.

    Projects written by earlier builds stored the *sum* of ranks, which made
    the value incomparable with new runs (2 engines × 20 ligands produced
    ranks up to 40). Recomputing on load keeps every project consistent.
    """
    for run in data.get("runs", []):
        for item in run.get("results", []):
            if item.get("status") != "ok":
                continue
            # Key by engine name: the rank that drove docking is stored in
            # ``vina_rank`` even when the primary engine was smina, and
            # it must not be counted twice via ``engine_ranks``.
            scores = item.get("engine_scores") or {}
            primary = next(
                (
                    name for name, value in scores.items()
                    if isinstance(value, (int, float))
                    and isinstance(item.get("affinity"), (int, float))
                    and abs(value - item["affinity"]) < 1e-6
                ),
                next(iter(scores), "vina"),
            )
            parts: dict[str, float] = {}
            if item.get("vina_rank"):
                parts[primary] = float(item["vina_rank"])
            if item.get("vinardo_rank"):
                parts["vinardo"] = float(item["vinardo_rank"])
            for name, value in (item.get("engine_ranks") or {}).items():
                if isinstance(value, (int, float)):
                    parts[name] = float(value)
            if parts:
                item["consensus_rank"] = round(
                    sum(parts.values()) / len(parts), 2
                )


class Project:
    def __init__(self, root: Path, data: dict):
        self.root = Path(root)
        log_dir = self.root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        _prune_logs(log_dir)
        self.data = data
        self.data.setdefault("params", dict(DEFAULT_DOCK_PARAMS))
        params = self.data["params"]
        # Upgrade projects that still carry the old accuracy defaults, without
        # touching values the user has customised.
        if (
            params.get("exhaustiveness") == 8
            and float(params.get("energy_range", 3.0)) == 3.0
            and int(params.get("seed", 0)) == 0
        ):
            params["exhaustiveness"] = 16
            params["energy_range"] = 4.0
            params["seed"] = 42
        if params.get("gpu_search_depth") == 20:
            params["gpu_search_depth"] = 40
        self.data.setdefault("ligand_options", dict(DEFAULT_LIGAND_OPTIONS))
        self.data.setdefault("receptor_options", dict(DEFAULT_RECEPTOR_OPTIONS))
        self.data.setdefault("receptors", [])
        self.data.setdefault("ligands", [])
        self.data.setdefault("runs", [])
        self.data.setdefault("synthesis", [])
        self.data.setdefault("ddi", [])
        self.data.setdefault("drugref", [])
        _normalise_consensus(self.data)

    # ------------------------------------------------------------------
    # creation / loading
    # ------------------------------------------------------------------
    @classmethod
    def create(cls, name: str, parent: Path | None = None) -> "Project":
        _migrate_legacy_projects()
        parent = Path(parent) if parent else PROJECTS_DIR
        parent.mkdir(parents=True, exist_ok=True)
        root = unique_dir(parent, safe_name(name, "Docking_Project"))
        data = {
            "version": 1,
            "name": root.name,
            "created": now_iso(),
            "receptors": [],
            "ligands": [],
            "runs": [],
            "params": dict(DEFAULT_DOCK_PARAMS),
            "ligand_options": dict(DEFAULT_LIGAND_OPTIONS),
            "receptor_options": dict(DEFAULT_RECEPTOR_OPTIONS),
        }
        project = cls(root, data)
        for sub in ("receptors", "ligands", "runs", "exports"):
            (root / sub).mkdir(exist_ok=True)
        project.save()
        project.touch_recent()
        return project

    @classmethod
    def load(cls, path: Path) -> "Project":
        path = Path(path)
        if path.is_file():
            path = path.parent
        project_file = path / "project.json"
        data = load_json(project_file)
        if not data:
            raise ValueError(f"不是有效的 DrugHX 项目：{path}")
        project = cls(path, data)
        if project._repair_duplicate_ligand_ids():
            project.save()
        project.touch_recent()
        return project

    def _repair_duplicate_ligand_ids(self) -> bool:
        """Give duplicate ligand IDs (old 6-hex collisions) their own IDs.

        Duplicate IDs make batch linking fail and make runs ambiguous.  The
        first record keeps the original ID; later duplicates get a fresh ID
        and a copy of the shared ligand directory.
        """
        seen: set[str] = set()
        changed = False
        for ligand in self.data.get("ligands", []):
            lid = str(ligand.get("id") or "")
            if not lid:
                ligand["id"] = _new_id("lig")
                seen.add(ligand["id"])
                changed = True
                continue
            if lid not in seen:
                seen.add(lid)
                continue
            new_id = _new_id("lig")
            while new_id in seen:
                new_id = _new_id("lig")
            old_dir = self.root / "ligands" / lid
            new_dir = self.root / "ligands" / new_id
            copied = not old_dir.is_dir()
            if old_dir.is_dir():
                try:
                    shutil.copytree(old_dir, new_dir)
                    copied = True
                except Exception:
                    copied = False
            if not copied:
                # Leave the record untouched rather than pointing it at a
                # directory that was never copied; runtime dedup still applies.
                seen.add(lid)
                continue
            for key in ("pdbqt", "sdf", "source"):
                value = ligand.get(key)
                if not value:
                    continue
                text = str(value).replace("\\", "/")
                prefix = f"ligands/{lid}/"
                if text.startswith(prefix):
                    ligand[key] = "ligands/" + new_id + "/" + text[len(prefix):]
            ligand["id"] = new_id
            seen.add(new_id)
            changed = True
        return changed

    def save(self) -> None:
        write_json(self.root / "project.json", self.data)

    def touch_recent(self) -> None:
        data = load_json(RECENT_FILE, {}) or {}
        recents = data.get("items", []) if isinstance(data, dict) else data
        recents = [r for r in recents if r.get("path") != str(self.root)]
        recents.insert(0, {"path": str(self.root), "name": self.data.get("name", ""), "time": now_iso()})
        write_json(RECENT_FILE, {"items": recents[:8]})

    @staticmethod
    def recent() -> list[dict]:
        _migrate_legacy_projects()
        data = load_json(RECENT_FILE, {}) or {}
        items = data.get("items", []) if isinstance(data, dict) else data
        valid = []
        changed = False
        for item in items:
            path = Path(item.get("path", ""))
            try:
                if path.parent.resolve() != PROJECTS_DIR.resolve():
                    candidate = PROJECTS_DIR / path.name
                    if (candidate / "project.json").is_file():
                        item["path"] = str(candidate)
                        path = candidate
                        changed = True
            except Exception:
                pass
            if not path.exists():
                candidate = PROJECTS_DIR / path.name
                if (candidate / "project.json").is_file():
                    item["path"] = str(candidate)
                    path = candidate
                    changed = True
            if path.exists():
                valid.append(item)
            else:
                changed = True
        if changed:
            write_json(RECENT_FILE, {"items": valid[:8]})
        return valid

    # ------------------------------------------------------------------
    # receptors / ligands
    # ------------------------------------------------------------------
    def receptor_dir(self, receptor: dict) -> Path:
        return self.root / "receptors" / receptor["id"]

    def ligand_dir(self, ligand: dict) -> Path:
        return self.root / "ligands" / ligand["id"]

    def new_receptor(self, name: str) -> dict:
        rid = _new_id("rec")
        rec_dir = self.root / "receptors" / rid
        rec_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "id": rid,
            "name": name or "Receptor",
            "status": "raw",
            "message": "",
            "created": now_iso(),
            "meta": {},
            "pockets": [],
            "box": None,
            "options": dict(DEFAULT_RECEPTOR_OPTIONS),
        }
        self.data["receptors"].append(record)
        return record

    def new_ligand(self, name: str) -> dict:
        lid = _new_id("lig")
        lig_dir = self.root / "ligands" / lid
        lig_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "id": lid,
            "name": name or "Ligand",
            "status": "raw",
            "message": "",
            "created": now_iso(),
            "meta": {},
        }
        self.data["ligands"].append(record)
        return record

    def get_receptor(self, receptor_id: str) -> dict | None:
        return next((r for r in self.data["receptors"] if r["id"] == receptor_id), None)

    def get_ligand(self, ligand_id: str) -> dict | None:
        return next((l for l in self.data["ligands"] if l["id"] == ligand_id), None)

    def remove_receptor(self, receptor_id: str) -> bool:
        rec = self.get_receptor(receptor_id)
        if not rec:
            return False
        shutil.rmtree(self.receptor_dir(rec), ignore_errors=True)
        self.data["receptors"] = [r for r in self.data["receptors"] if r["id"] != receptor_id]
        self.save()
        return True

    def remove_ligand(self, ligand_id: str) -> bool:
        lig = self.get_ligand(ligand_id)
        if not lig:
            return False
        shutil.rmtree(self.ligand_dir(lig), ignore_errors=True)
        self.data["ligands"] = [l for l in self.data["ligands"] if l["id"] != ligand_id]
        self.save()
        return True

    def new_run(self, params: dict) -> dict:
        run_id = f"run_{stamp()}_{uuid.uuid4().hex[:4]}"
        run_dir = self.root / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "id": run_id,
            "time": now_iso(),
            "status": "running",
            "params": params,
            "results": [],
            "output": f"runs/{run_id}",
        }
        self.data["runs"].insert(0, record)
        self.save()
        return record

    def get_run(self, run_id: str) -> dict | None:
        return next((r for r in self.data["runs"] if r["id"] == run_id), None)

    def run_dir(self, run: dict) -> Path:
        path = Path(run["output"])
        return path if path.is_absolute() else self.root / path

    # ------------------------------------------------------------------
    # quality control
    # ------------------------------------------------------------------
    def qc(self) -> dict:
        """Compact health report for the project (used by the QC panel)."""
        receptors = self.data.get("receptors", [])
        ligands = self.data.get("ligands", [])
        runs = self.data.get("runs", [])
        issues: list[dict] = []

        def exists(rel_path: str | None) -> bool:
            if not rel_path:
                return False
            path = Path(rel_path)
            return (path if path.is_absolute() else self.root / path).is_file()

        rec_ready = 0
        rec_missing = []
        for rec in receptors:
            if rec.get("status") != "ready":
                continue
            if exists(rec.get("pdbqt")):
                rec_ready += 1
            else:
                rec_missing.append(rec.get("name") or rec.get("id"))
        if rec_missing:
            issues.append(
                {
                    "level": "error",
                    "text": f"{len(rec_missing)} 个已标记就绪的受体缺少 PDBQT："
                    + "、".join(rec_missing[:5]),
                }
            )

        status_counts: dict[str, int] = {}
        seen: set[str] = set()
        duplicates: list[str] = []
        lig_missing = []
        for lig in ligands:
            status = str(lig.get("status") or "raw")
            status_counts[status] = status_counts.get(status, 0) + 1
            lid = str(lig.get("id") or "")
            if lid:
                if lid in seen:
                    duplicates.append(lig.get("name") or lid)
                seen.add(lid)
            if status == "ready" and not exists(lig.get("pdbqt")):
                lig_missing.append(lig.get("name") or lid)
        if duplicates:
            issues.append(
                {
                    "level": "error",
                    "text": f"{len(duplicates)} 个配体 ID 重复，批量对接可能漏算："
                    + "、".join(duplicates[:5]),
                }
            )
        if lig_missing:
            issues.append(
                {
                    "level": "error",
                    "text": f"{len(lig_missing)} 个已就绪配体缺少 PDBQT："
                    + "、".join(lig_missing[:5]),
                }
            )
        if status_counts.get("error"):
            issues.append(
                {
                    "level": "warn",
                    "text": f"{status_counts['error']} 个配体准备失败，请检查配体准备的日志与 SMILES。",
                }
            )
        if not receptors:
            issues.append({"level": "info", "text": "项目还没有受体。"})
        if not ligands:
            issues.append({"level": "info", "text": "项目还没有配体。"})

        failed_runs = [run for run in runs if run.get("status") == "error"]
        interrupted = [run for run in runs if run.get("status") == "interrupted"]
        if failed_runs:
            issues.append(
                {"level": "warn", "text": f"{len(failed_runs)} 次对接任务失败，可在任务日志中查看原因。"}
            )
        if interrupted:
            issues.append(
                {
                    "level": "warn",
                    "text": f"{len(interrupted)} 次对接任务中断；可在结果页对失败的组合重新对接。",
                }
            )

        latest = runs[0] if runs else None
        latest_failed = 0
        latest_total = 0
        if latest:
            latest_results = latest.get("results") or []
            latest_total = len(latest_results)
            latest_failed = sum(
                1 for item in latest_results if item.get("status") != "ok"
            )

        return {
            "ok": not any(item["level"] == "error" for item in issues),
            "issues": issues,
            "receptors": {
                "total": len(receptors),
                "ready": rec_ready,
                "missing_files": len(rec_missing),
            },
            "ligands": {
                "total": len(ligands),
                "by_status": status_counts,
                "duplicate_ids": len(duplicates),
                "missing_files": len(lig_missing),
            },
            "runs": {
                "total": len(runs),
                "failed": len(failed_runs),
                "interrupted": len(interrupted),
                "latest_id": latest.get("id") if latest else "",
                "latest_failed": latest_failed,
                "latest_total": latest_total,
            },
            "logs_dir": str(self.root / "logs"),
        }

    # ------------------------------------------------------------------
    # API state
    # ------------------------------------------------------------------
    def state(
        self,
        system: dict | None = None,
        ligand_limit: int = 300,
        ligand_offset: int = 0,
        ligand_query: str = "",
        ligand_status: str = "",
    ) -> dict:
        def exists(rel_path: str | None) -> bool:
            if not rel_path:
                return False
            path = Path(rel_path)
            return (path if path.is_absolute() else self.root / path).exists()

        receptors = []
        for rec in self.data["receptors"]:
            receptors.append(
                {
                    **{k: v for k, v in rec.items() if k not in ("pdb", "pdb_clean", "pdbqt")},
                    "has_pdb": exists(rec.get("pdb")),
                    "has_pdbqt": exists(rec.get("pdbqt")),
                }
            )
        all_ligands = self.data["ligands"]
        ligand_smiles = {
            lig["id"]: (
                lig.get("smiles")
                or (lig.get("meta") or {}).get("smiles")
                or ""
            )
            for lig in all_ligands
        }
        ligand_stats: dict[str, int] = {}
        for lig in all_ligands:
            status = str(lig.get("status") or "raw")
            ligand_stats[status] = ligand_stats.get(status, 0) + 1
        query = str(ligand_query or "").strip().lower()
        status_filter = str(ligand_status or "").strip().lower()
        filtered = [
            lig for lig in all_ligands
            if (not status_filter or str(lig.get("status") or "").lower() == status_filter)
            and (
                not query
                or query in str(lig.get("name") or "").lower()
                or query in str(lig.get("smiles") or (lig.get("meta") or {}).get("smiles") or "").lower()
            )
        ]
        start = max(0, int(ligand_offset or 0))
        limit = max(1, min(int(ligand_limit or 300), 500))
        page = filtered[start:start + limit]
        ligands = []
        for lig in page:
            ligands.append(
                {
                    **{k: v for k, v in lig.items() if k not in ("pdbqt", "sdf", "source")},
                    "has_pdbqt": exists(lig.get("pdbqt")),
                    "has_sdf": exists(lig.get("sdf")),
                }
            )
        runs = []
        for run in self.data["runs"]:
            results = run.get("results", [])
            result_rows = []
            for item in results[:200]:
                row = dict(item)
                # Make every result self-contained: the frontend can follow the
                # selected ligand even when it is outside the current 300-row
                # project page.
                row.setdefault(
                    "ligand_smiles", ligand_smiles.get(item.get("ligand_id"), "")
                )
                result_rows.append(row)
            entry = {
                "id": run["id"],
                "time": run["time"],
                "status": run["status"],
                "params": run.get("params", {}),
                "results": result_rows,
                "result_count": len(results),
            }
            if (run.get("params") or {}).get("bulk"):
                ok = [
                    item for item in results
                    if item.get("status") == "ok" and item.get("affinity") is not None
                ]
                values = sorted(float(item["affinity"]) for item in ok)
                if values:
                    median = (
                        values[len(values) // 2]
                        if len(values) % 2
                        else (
                            values[len(values) // 2 - 1] + values[len(values) // 2]
                        ) / 2
                    )
                    entry["bulk_summary"] = {
                        "ok": len(values),
                        "failed": len(results) - len(values),
                        "best": values[0],
                        "worst": values[-1],
                        "median": round(median, 2),
                        "mean": round(sum(values) / len(values), 2),
                        "below_8": sum(1 for value in values if value <= -8.0),
                        "below_9": sum(1 for value in values if value <= -9.0),
                    }
                else:
                    entry["bulk_summary"] = {
                        "ok": 0, "failed": len(results),
                        "best": None, "worst": None,
                        "median": None, "mean": None,
                        "below_8": 0, "below_9": 0,
                    }
            runs.append(entry)
        # Only ligands that actually took part in fine docking (bulk screens
        # are excluded); downstream modules use this as their candidate list.
        fine_ligands = []
        fine_seen: set[str] = set()
        ligand_index = {lig["id"]: lig for lig in all_ligands}
        for run in self.data["runs"]:
            if (run.get("params") or {}).get("bulk"):
                continue
            for result_row in run.get("results", []):
                lid = result_row.get("ligand_id")
                if not lid or lid in fine_seen or lid not in ligand_index:
                    continue
                fine_seen.add(lid)
                lig = ligand_index[lid]
                fine_ligands.append(
                    {
                        "id": lid,
                        "name": lig.get("name") or lid,
                        "smiles": ligand_smiles.get(lid, ""),
                        "status": lig.get("status"),
                        "formula": (lig.get("meta") or {}).get("formula", ""),
                    }
                )
                if len(fine_ligands) >= 500:
                    break
            if len(fine_ligands) >= 500:
                break
        return {
            "name": self.data.get("name", "DrugHX"),
            "path": str(self.root),
            "created": self.data.get("created"),
            "receptors": receptors,
            "ligands": ligands,
            "fine_ligands": fine_ligands,
            "ligand_total": len(all_ligands),
            "ligand_filtered": len(filtered),
            "ligand_offset": start,
            "ligand_stats": ligand_stats,
            "runs": runs,
            "params": self.data["params"],
            "ligand_options": self.data["ligand_options"],
            "receptor_options": self.data["receptor_options"],
            "optimizations": self.data.get("optimizations", []),
            "designs": self.data.get("designs", []),
            "fishing": self.data.get("fishing", []),
            "netpharm": self.data.get("netpharm", []),
            # Paused automatic pipeline (fine docking done, waiting for the
            # user to pick a candidate).  Exposed so a page refresh can
            # restore the confirmation panel instead of losing the run.
            "auto_pending": self.data.get("auto_pending") or {},
            "synthesis": self.data.get("synthesis", []),
            "ddi": self.data.get("ddi", []),
            "drugref": self.data.get("drugref", []),
            "assays": self.data.get("assays") or {},
            "pbpk": self.data.get("pbpk") or {},
            "benchmarks": self.data.get("benchmarks") or {},
            "recent": self.recent(),
            "system": system or {},
        }
