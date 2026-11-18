"""FastAPI application: project, receptor/ligand preparation, docking, results."""
from __future__ import annotations

import copy
import csv
import io
import json
import os
import re
import shutil
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

import numpy as np

from . import (
    activelearn,
    admet,
    analysis,
    assay,
    benchmark,
    ddi,
    design,
    decision,
    drugref,
    energy,
    figures,
    filters as ligfilters,
    fishing,
    i18n,
    ligands as ligmod,
    manifest as runmanifest,
    mmp,
    mmgbsa,
    mpo,
    netpharm,
    online,
    optimization,
    pbpk,
    pdbio,
    pharmacophore,
    pharm,
    posecheck,
    profiling,
    receptors as recmod,
    references,
    synth,
    trust,
)
from .boxes import box_from_coords, box_volume, detect_pockets, suggest_box
from .config import (
    APP_VERSION,
    COMMERCIAL_MODE,
    DEFAULT_DOCK_PARAMS,
    DOCK_PRESETS,
    DEFAULT_LIGAND_OPTIONS,
    DEFAULT_RECEPTOR_OPTIONS,
    LOGS_DIR,
    PROJECTS_DIR,
    VINA_EXE,
    VINA_GPU_EXE,
    WEB_DIR,
    engine_status,
    resolve_engine_path,
    save_engine_path,
)
from .docking import run_docking
from .jobs import JobManager
from .project import Project

import logging
from logging.handlers import RotatingFileHandler

LOGS_DIR.mkdir(parents=True, exist_ok=True)


class _SafeRotatingFileHandler(RotatingFileHandler):
    """Rotate the log without crashing when a second instance holds the file.

    Windows forbids renaming an open file.  Two DrugHX processes (for example
    the launcher plus a manual ``uvicorn`` start) would otherwise raise a
    PermissionError on every record once the log exceeds ``maxBytes`` and flood
    the console with tracebacks.  Keeping the stream open is harmless: the
    file simply grows until the other process releases it.
    """

    def doRollover(self) -> None:  # noqa: D102
        try:
            super().doRollover()
        except PermissionError:
            if self.stream is None:
                self.stream = self._open()


_handler = _SafeRotatingFileHandler(
    LOGS_DIR / "dockforge.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
)
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[_handler])
logger = logging.getLogger("dockforge")
from .utils import safe_filename, safe_name, write_json


def _version(module_name: str) -> str:
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", "?")
    except Exception:
        return "未安装"


def _system_info() -> dict:
    return {
        "app_version": APP_VERSION,
        "commercial_mode": COMMERCIAL_MODE,
        "vina_path": str(VINA_EXE),
        "vina_version": recmod.vina_version(),
        "gpu_path": str(VINA_GPU_EXE),
        "gpu_available": VINA_GPU_EXE.is_file(),
        "engines": engine_status(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "rdkit": _version("rdkit"),
        "meeko": _version("meeko"),
        "projects_dir": str(PROJECTS_DIR),
    }


class AppState:
    def __init__(self):
        self.jobs = JobManager()
        self.jobs.log_dir_getter = self._job_log_dir
        self.lock = threading.RLock()
        self.system = _system_info()
        self.project = self._initial_project()

    def _job_log_dir(self) -> Path:
        project = getattr(self, "project", None)
        return (project.root / "logs") if project is not None else LOGS_DIR

    def _initial_project(self) -> Project:
        for item in Project.recent():
            try:
                project = Project.load(Path(item["path"]))
            except Exception:
                continue
            changed = False
            for run in project.data.get("runs", []):
                if run.get("status") == "running":
                    run["status"] = "interrupted"
                    changed = True
            if changed:
                project.save()
            return project
        # No usable recent entry: open the most recently touched project
        # instead of silently creating an empty one.
        try:
            candidates = [
                item for item in PROJECTS_DIR.iterdir()
                if item.is_dir() and (item / "project.json").is_file()
            ]
        except OSError:
            candidates = []
        for folder in sorted(
            candidates,
            key=lambda item: (item / "project.json").stat().st_mtime,
            reverse=True,
        ):
            try:
                return Project.load(folder)
            except Exception:
                continue
        return Project.create(f"Docking Project {datetime.now():%Y-%m-%d}")

    def snapshot(
        self,
        ligand_limit: int = 300,
        ligand_offset: int = 0,
        ligand_query: str = "",
        ligand_status: str = "",
    ) -> dict:
        with self.lock:
            data = self.project.state(
                self.system,
                ligand_limit=ligand_limit,
                ligand_offset=ligand_offset,
                ligand_query=ligand_query,
                ligand_status=ligand_status,
            )
            active = []
            for job in self.jobs.active():
                snap = job.snapshot(since=max(0, len(job.logs) - 200))
                snap["log_index"] = len(job.logs)
                active.append(snap)
            data["active_jobs"] = active
            return data


STATE = AppState()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    for job in STATE.jobs.active():
        job.cancel()


app = FastAPI(title="DrugHX", version=APP_VERSION, lifespan=lifespan)


@app.exception_handler(Exception)
async def _unhandled_error(request, exc: Exception):
    """Log unexpected failures with a traceback and a user-readable message."""
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "内部错误，已写入 data/logs/dockforge.log 和当前项目的 logs 目录，请把日志发给开发者。"
        },
    )
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.middleware("http")
async def no_static_cache(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store"
    return response


def _abs(rel_path: str | None) -> Path | None:
    if not rel_path:
        return None
    path = Path(rel_path)
    return path if path.is_absolute() else STATE.project.root / path


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(STATE.project.root))
    except ValueError:
        return str(path)


def _find_result_receptor(result: dict) -> dict | None:
    """Resolve a result's receptor, tolerating a re-prepared receptor."""
    receptor = STATE.project.get_receptor(result.get("receptor_id"))
    if receptor:
        return receptor
    name = result.get("receptor_name")
    if not name:
        return None
    candidates = [
        item for item in STATE.project.data.get("receptors", [])
        if item.get("name") == name and item.get("pdb")
    ]
    return candidates[0] if candidates else None


def _result_receptor_pdb(result: dict) -> Path | None:
    """Prefer the receptor copy stored with the run, then the project record."""
    stored = _abs(result.get("receptor_pdb"))
    if stored and stored.exists():
        return stored
    receptor = _find_result_receptor(result)
    if not receptor:
        return None
    path = _abs(receptor.get("pdb"))
    return path if path and path.exists() else None


def _result_receptor_pdbqt(result: dict) -> str | None:
    """Receptor PDBQT text, used to recover donor hydrogens for H-bond angles."""
    candidates: list[Path] = []
    receptor = _find_result_receptor(result)
    if receptor and receptor.get("pdbqt"):
        candidates.append(_abs(receptor["pdbqt"]))
    pdb = _result_receptor_pdb(result)
    if pdb is not None:
        candidates.append(pdb.with_suffix(".pdbqt"))
    for path in candidates:
        if path and path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
    return None


def _ensure_no_active_job() -> None:
    if STATE.jobs.active():
        raise HTTPException(409, "已有任务正在运行，请等待完成或先取消当前任务")


@app.get("/", response_class=HTMLResponse)
def index():
    return (WEB_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/favicon.ico")
def favicon():
    icon = WEB_DIR / "favicon.svg"
    if icon.exists():
        return FileResponse(icon, media_type="image/svg+xml")
    return Response(status_code=204)


@app.get("/api/health")
def health():
    """Identity probe used by the launcher to avoid duplicate instances."""
    return {"app": "DrugHX", "version": APP_VERSION}


# ---------------------------------------------------------------------------
# state / project
# ---------------------------------------------------------------------------
@app.get("/api/state")
def get_state(
    ligand_limit: int = 300,
    ligand_offset: int = 0,
    ligand_query: str = "",
    ligand_status: str = "",
):
    data = STATE.snapshot(
        ligand_limit=ligand_limit,
        ligand_offset=ligand_offset,
        ligand_query=ligand_query,
        ligand_status=ligand_status,
    )
    data["dock_presets"] = DOCK_PRESETS
    return data


@app.get("/api/ligands")
def list_ligands(
    offset: int = 0,
    limit: int = 100,
    query: str = "",
    status: str = "",
):
    """Paginated compact ligand list for the fine-preparation module."""
    with STATE.lock:
        items = STATE.project.state(
            None,
            ligand_limit=limit,
            ligand_offset=offset,
            ligand_query=query,
            ligand_status=status,
        )
    return {
        "total": items["ligand_filtered"],
        "offset": items["ligand_offset"],
        "items": items["ligands"],
        "stats": items["ligand_stats"],
    }


@app.get("/api/dock/presets")
def dock_presets():
    return {"presets": DOCK_PRESETS}


@app.get("/api/engines")
def engines():
    return {"engines": engine_status()}


@app.post("/api/engines/{engine_id}/path")
def set_engine_path(engine_id: str, payload: dict):
    path = Path(str(payload.get("path") or "").strip())
    if not path.is_file():
        raise HTTPException(400, f"找不到可执行文件：{path}")
    save_engine_path(engine_id, str(path))
    with STATE.lock:
        STATE.system = _system_info()
    return {"ok": True, "engines": engine_status()}


@app.post("/api/engines/{engine_id}/install")
def install_engine(engine_id: str):
    from . import engines as enginemod

    try:
        binary = enginemod.install(engine_id)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    with STATE.lock:
        STATE.system = _system_info()
    return {"ok": True, "path": str(binary), "engines": engine_status()}


@app.post("/api/project/new")
def new_project(payload: dict):
    _ensure_no_active_job()
    name = str(payload.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "项目名称不能为空")
    with STATE.lock:
        STATE.project = Project.create(name)
    return STATE.snapshot()


@app.post("/api/project/open")
def open_project(payload: dict):
    _ensure_no_active_job()
    path = Path(str(payload.get("path") or "").strip())
    try:
        project = Project.load(path)
    except Exception as exc:
        raise HTTPException(400, f"打开项目失败：{exc}") from exc
    with STATE.lock:
        STATE.project = project
    return STATE.snapshot()


@app.post("/api/project/params")
def save_params(payload: dict):
    with STATE.lock:
        params = dict(DEFAULT_DOCK_PARAMS)
        params.update(payload.get("params") or {})
        STATE.project.data["params"] = params
        if "ligand_options" in payload:
            opts = dict(DEFAULT_LIGAND_OPTIONS)
            opts.update(payload["ligand_options"] or {})
            STATE.project.data["ligand_options"] = opts
        if "receptor_options" in payload:
            opts = dict(DEFAULT_RECEPTOR_OPTIONS)
            opts.update(payload["receptor_options"] or {})
            STATE.project.data["receptor_options"] = opts
        STATE.project.save()
    return {"ok": True, "params": STATE.project.data["params"]}


@app.post("/api/project/reveal")
def reveal_project(payload: dict | None = None):
    kind = str((payload or {}).get("kind") or "project").lower()
    with STATE.lock:
        root = STATE.project.root
    target = root / "exports" / "full_report" if kind == "report" else root
    if kind == "report":
        target.mkdir(parents=True, exist_ok=True)
    try:
        os.startfile(str(target))  # type: ignore[attr-defined]
    except Exception as exc:
        raise HTTPException(400, f"无法打开目录：{exc}") from exc
    return {"ok": True, "path": str(target)}


BACKUPS_DIR = PROJECTS_DIR.parent / "backups"


@app.get("/api/project/qc")
def project_qc():
    """Project health check: missing files, duplicate IDs, failed jobs."""
    with STATE.lock:
        return STATE.project.qc()


@app.post("/api/project/backup")
def backup_project():
    """Zip the current project into data/backups and keep the last ten."""
    _ensure_no_active_job()
    with STATE.lock:
        project = STATE.project
        root = project.root
        name = safe_name(str(project.data.get("name") or root.name), "project")
    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
    target = BACKUPS_DIR / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.zip"
    try:
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in root.rglob("*"):
                if not path.is_file() or "__pycache__" in path.parts:
                    continue
                archive.write(path, path.relative_to(root.parent))
    except Exception as exc:
        try:
            target.unlink(missing_ok=True)
        except OSError:
            pass
        raise HTTPException(500, f"项目备份失败：{exc}") from exc
    backups = sorted(
        BACKUPS_DIR.glob(f"{name}_*.zip"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for old in backups[10:]:
        try:
            old.unlink()
        except OSError:
            pass
    return {
        "ok": True,
        "path": str(target),
        "size_mb": round(target.stat().st_size / 1048576, 2),
    }


@app.get("/api/references.bib")
def references_bib():
    return Response(
        references.bibtex(),
        media_type="application/x-bibtex; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="drughx_references.bib"'},
    )


@app.get("/api/references.ris")
def references_ris():
    return Response(
        references.ris(),
        media_type="application/x-research-info-systems; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="drughx_references.ris"'},
    )


# ---------------------------------------------------------------------------
# receptors
# ---------------------------------------------------------------------------
def _store_receptor_pdb(name: str, text: str, pdb_id: str | None = None) -> dict:
    with STATE.lock:
        project = STATE.project
        record = project.new_receptor(name)
        rec_dir = project.receptor_dir(record)
        source = rec_dir / "source.pdb"
        source.write_text(text, encoding="utf-8")
        record["source"] = _rel(source)
        record["pdb_id"] = pdb_id or ""
        record["meta"] = pdbio.pdb_info(text)
        record["hetero"] = pdbio.hetero_ligands(text)
        record["pdb"] = _rel(source)
        project.save()
    return record


@app.post("/api/receptors/fetch")
def fetch_receptor(payload: dict):
    pdb_id = str(payload.get("pdb_id") or "").strip()
    text = recmod.fetch_pdb(pdb_id)
    record = _store_receptor_pdb(pdb_id.upper(), text, pdb_id.upper())
    return {"ok": True, "receptor_id": record["id"], "state": STATE.snapshot()}


@app.post("/api/receptors/upload")
async def upload_receptors(files: list[UploadFile] = File(...)):
    added = []
    for upload in files:
        filename = Path(upload.filename or "receptor.pdb").name
        suffix = Path(filename).suffix.lower()
        data = await upload.read()
        if len(data) > 120 * 1024 * 1024:
            raise HTTPException(400, f"{filename} 超过 120 MB")
        text = data.decode("utf-8", errors="replace")
        name = Path(filename).stem
        if suffix in (".pdb", ".ent", ".txt"):
            if "ATOM" not in text:
                raise HTTPException(400, f"{filename} 不是有效的 PDB 文件")
            added.append(_store_receptor_pdb(name, text))
        elif suffix == ".pdbqt":
            with STATE.lock:
                project = STATE.project
                record = project.new_receptor(name)
                rec_dir = project.receptor_dir(record)
                source = rec_dir / "source.pdbqt"
                source.write_text(text, encoding="utf-8")
                pdb_text = pdbio.pdbqt_to_pdb(text)
                view = rec_dir / "receptor.pdb"
                view.write_text(pdb_text, encoding="utf-8")
                target = rec_dir / "receptor.pdbqt"
                target.write_text(text, encoding="utf-8")
                record.update(
                    {
                        "source": _rel(source),
                        "pdb": _rel(view),
                        "pdb_clean": _rel(view),
                        "pdbqt": _rel(target),
                        "status": "ready",
                        "message": "使用上传的 PDBQT",
                        "meta": {
                            **pdbio.pdb_info(pdb_text),
                            "pdbqt_atoms": pdbio.pdbqt_stats(text)["atoms"],
                            "from_pdbqt": True,
                        },
                        "pockets": detect_pockets(pdb_text, top=5),
                    }
                )
                if record["pockets"]:
                    top = record["pockets"][0]
                    record["box"] = {
                        "center": top["center"],
                        "size": top["size"],
                        "source": "pocket",
                        "label": top["label"],
                    }
                project.save()
                added.append(record)
        else:
            raise HTTPException(400, f"不支持的受体格式：{suffix}")
    return {"ok": True, "added": len(added), "state": STATE.snapshot()}


@app.post("/api/receptors/{receptor_id}/prepare")
def prepare_receptor_endpoint(receptor_id: str, payload: dict):
    _ensure_no_active_job()
    with STATE.lock:
        receptor = STATE.project.get_receptor(receptor_id)
        if not receptor:
            raise HTTPException(404, "受体不存在")
        source = _abs(receptor.get("source"))
        if not source or not source.exists():
            raise HTTPException(400, "找不到受体源文件")
        if source.suffix.lower() == ".pdbqt":
            raise HTTPException(400, "该受体来自已准备的 PDBQT，无需再次运行准备流程")
        options = dict(DEFAULT_RECEPTOR_OPTIONS)
        options.update(payload.get("options") or {})
        receptor["options"] = options
        receptor["status"] = "preparing"
        receptor["message"] = "准备中 ..."
        STATE.project.save()

    def task(job):
        with STATE.lock:
            project = STATE.project
            receptor = project.get_receptor(receptor_id)
            rec_dir = project.receptor_dir(receptor)
            source_text = source.read_text(encoding="utf-8", errors="replace")
            job.log(f"受体 {receptor['name']}：开始准备")
            job.set_progress(0.05, "清洁结构 ...")
        try:
            hetero = pdbio.hetero_ligands(source_text)
            selected_chains = options.get("chains") or []
            if selected_chains:
                hetero = [h for h in hetero if h.get("chain") in selected_chains]
            result = recmod.prepare_receptor(
                rec_dir, source_text, options, job.log,
                hetero=hetero,
            )
            with STATE.lock:
                receptor = project.get_receptor(receptor_id)
                receptor.update(
                    {
                        "status": "ready",
                        "message": "",
                        "pdb": _rel(Path(result["pdb"])),
                        "pdb_clean": _rel(Path(result["pdb_clean"])),
                        "pdbqt": _rel(Path(result["pdbqt"])),
                        "meta": result["meta"],
                        "pockets": result["pockets"],
                        "hetero": hetero,
                    }
                )
                if not receptor.get("box") or (receptor["box"].get("source") != "manual"):
                    receptor["box"] = result["box"]
                project.save()
                job.set_progress(1.0, "受体准备完成")
                job.result = {
                    "receptor_id": receptor_id,
                    "pockets": result["pockets"],
                    "box": receptor.get("box"),
                    "meta": result["meta"],
                }
        except Exception as exc:
            with STATE.lock:
                receptor = project.get_receptor(receptor_id)
                if receptor:
                    if job.cancel_event.is_set():
                        receptor["status"] = "raw"
                        receptor["message"] = "已取消"
                    else:
                        receptor["status"] = "error"
                        receptor["message"] = str(exc)
                project.save()
            if job.cancel_event.is_set():
                return
            raise

    job = STATE.jobs.create("receptor", f"准备受体 · {receptor['name']}", task)
    return {"job_id": job.id}


@app.post("/api/receptors/{receptor_id}/box")
def set_receptor_box(receptor_id: str, payload: dict):
    center = payload.get("center")
    size = payload.get("size")
    if not (isinstance(center, list) and isinstance(size, list) and len(center) == 3 and len(size) == 3):
        raise HTTPException(400, "对接盒参数不完整")
    try:
        center = [round(float(v), 3) for v in center]
        size = [round(float(v), 3) for v in size]
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, "对接盒参数必须是数字") from exc
    if any(v <= 0 or v > 100 for v in size):
        raise HTTPException(400, "对接盒边长必须在 0–100 Å 之间")
    with STATE.lock:
        receptor = STATE.project.get_receptor(receptor_id)
        if not receptor:
            raise HTTPException(404, "受体不存在")
        receptor["box"] = {
            "center": center,
            "size": size,
            "source": payload.get("source", "manual"),
            "label": payload.get("label", "手动设置"),
        }
        STATE.project.save()
    return {"ok": True, "box": receptor["box"], "volume": box_volume(size)}


@app.get("/api/receptors/{receptor_id}/pdb", response_class=PlainTextResponse)
def receptor_pdb(receptor_id: str):
    with STATE.lock:
        receptor = STATE.project.get_receptor(receptor_id)
        if not receptor:
            raise HTTPException(404, "受体不存在")
        path = _abs(receptor.get("pdb")) or _abs(receptor.get("source"))
    if not path or not path.exists():
        raise HTTPException(404, "受体结构尚未准备")
    return PlainTextResponse(path.read_text(encoding="utf-8", errors="replace"))


@app.get("/api/receptors/{receptor_id}/original")
def receptor_original(receptor_id: str):
    with STATE.lock:
        receptor = STATE.project.get_receptor(receptor_id)
        if not receptor:
            raise HTTPException(404, "受体不存在")
        path = _abs(receptor.get("source"))
    if not path or not path.exists():
        raise HTTPException(404, "源文件不存在")
    return FileResponse(path, filename=f"{receptor['name']}{path.suffix}")


@app.delete("/api/receptors/{receptor_id}")
def delete_receptor(receptor_id: str):
    with STATE.lock:
        ok = STATE.project.remove_receptor(receptor_id)
    if not ok:
        raise HTTPException(404, "受体不存在")
    return {"ok": True, "state": STATE.snapshot()}


# ---------------------------------------------------------------------------
# ligands
# ---------------------------------------------------------------------------
def _unique_ligand_name(base: str, names: set[str] | None = None) -> str:
    names = names if names is not None else {l["name"] for l in STATE.project.data["ligands"]}
    if base not in names:
        return base
    i = 2
    while f"{base}_{i}" in names:
        i += 1
    return f"{base}_{i}"


@app.post("/api/ligands/upload")
async def upload_ligands(files: list[UploadFile] = File(...)):
    added = []
    for upload in files:
        filename = Path(upload.filename or "ligand.sdf").name
        suffix = Path(filename).suffix.lower()
        data = await upload.read()
        if len(data) > 120 * 1024 * 1024:
            raise HTTPException(400, f"{filename} 超过 120 MB")
        base = Path(filename).stem
        if suffix == ".pdbqt":
            with STATE.lock:
                record = STATE.project.new_ligand(_unique_ligand_name(base))
                lig_dir = STATE.project.ligand_dir(record)
                source = lig_dir / "source.pdbqt"
                source.write_bytes(data)
                target = lig_dir / "ligand.pdbqt"
                text = data.decode("utf-8", errors="replace")
                target.write_text(pdbio.normalize_pdbqt(text), encoding="utf-8")
                record.update(
                    {
                        "status": "ready",
                        "source": _rel(source),
                        "pdbqt": _rel(target),
                        "format": "pdbqt",
                        "message": "使用上传的 PDBQT",
                        "meta": {"name": record["name"]},
                    }
                )
                STATE.project.save()
                added.append(record["id"])
            continue

        source = STATE.project.root / "tmp_upload" / safe_filename(filename)
        source.parent.mkdir(exist_ok=True)
        source.write_bytes(data)
        try:
            mols = ligmod.read_ligand_file(source)
        except Exception as exc:
            source.unlink(missing_ok=True)
            raise HTTPException(400, f"{filename}: {exc}") from exc

        with STATE.lock:
            names = {item["name"] for item in STATE.project.data["ligands"]}
        added_here = 0
        for index, (mol_name, mol) in enumerate(mols, start=1):
            suffix_name = base if len(mols) == 1 else f"{base}_{index}"
            name = mol_name if len(mols) > 1 and mol_name else suffix_name
            with STATE.lock:
                record = STATE.project.new_ligand(_unique_ligand_name(name, names))
                names.add(record["name"])
                lig_dir = STATE.project.ligand_dir(record)
                src_copy = lig_dir / "source.sdf"
                try:
                    ligmod.write_sdf(mol, src_copy, record["name"])
                except Exception:
                    src_copy.write_bytes(data)
                record.update(
                    {
                        "source": _rel(src_copy),
                        "sdf": _rel(src_copy),
                        "format": "sdf",
                        "meta": ligmod.ligand_metadata(mol),
                    }
                )
                added.append(record["id"])
                added_here += 1
                # Saving a 20k-ligand project for every molecule made large
                # imports quadratic; flush in batches instead.
                if added_here % 200 == 0:
                    STATE.project.save()
        with STATE.lock:
            STATE.project.save()
        source.unlink(missing_ok=True)

    return {"ok": True, "added": len(added), "ids": added, "state": STATE.snapshot()}


@app.post("/api/ligands/smiles")
def add_smiles(payload: dict):
    text = str(payload.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "SMILES 不能为空")
    items = ligmod.read_smiles_lines(text)
    if not items:
        raise HTTPException(400, "没有解析到有效的 SMILES")
    added = []
    with STATE.lock:
        names = {item["name"] for item in STATE.project.data["ligands"]}
    for name, smiles in items:
        try:
            mol = ligmod.mol_from_smiles(smiles)
        except Exception as exc:
            raise HTTPException(400, f"{name}: {exc}") from exc
        with STATE.lock:
            record = STATE.project.new_ligand(_unique_ligand_name(name, names))
            names.add(record["name"])
            lig_dir = STATE.project.ligand_dir(record)
            src = lig_dir / "source.smi"
            src.write_text(f"{smiles} {record['name']}\n", encoding="utf-8")
            sdf_path = lig_dir / "source.sdf"
            try:
                ligmod.write_sdf(mol, sdf_path, record["name"])
            except Exception:
                pass
            record.update(
                {
                    "source": _rel(src),
                    "sdf": _rel(sdf_path) if sdf_path.exists() else "",
                    "format": "smi",
                    "smiles": smiles,
                    "meta": ligmod.ligand_metadata(mol),
                }
            )
            added.append(record["id"])
    with STATE.lock:
        STATE.project.save()
    return {"ok": True, "added": len(added), "ids": added, "state": STATE.snapshot()}


@app.get("/api/ligands/{ligand_id}/svg")
def ligand_structure_svg(ligand_id: str, width: int = 300, height: int = 220):
    """2D depiction used by the filter report and the candidate picker."""
    with STATE.lock:
        ligand = STATE.project.get_ligand(ligand_id)
        smiles = (ligand or {}).get("smiles") or ""
    mol = ligmod.mol_from_smiles(smiles) if smiles else None
    if mol is None:
        raise HTTPException(404, "该配体没有可用的 SMILES")
    from rdkit.Chem.Draw import rdMolDraw2D

    drawer = rdMolDraw2D.MolDraw2DSVG(
        max(140, min(int(width), 800)),
        max(100, min(int(height), 600)),
    )
    drawer.drawOptions().clearBackground = False
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return Response(drawer.GetDrawingText(), media_type="image/svg+xml")


@app.post("/api/ligands/prepare")
def prepare_ligands(payload: dict):
    _ensure_no_active_job()
    options = dict(DEFAULT_LIGAND_OPTIONS)
    options.update(payload.get("options") or {})
    scope = str(payload.get("scope") or "").strip().lower()
    with STATE.lock:
        all_ligands = STATE.project.data["ligands"]
        if scope in ("all", "all_raw"):
            candidates = [item for item in all_ligands if item.get("status") != "ready"]
            if not candidates:
                candidates = list(all_ligands)
        else:
            index = {item["id"]: item for item in all_ligands}
            candidates = [
                index[lid]
                for lid in list(dict.fromkeys(payload.get("ids") or []))
                if lid in index
            ]
        targets = [item["id"] for item in candidates]
        for ligand in candidates:
            ligand["status"] = "preparing"
            ligand["message"] = "排队中 ..."
        STATE.project.data["ligand_options"] = options
        STATE.project.save()
    if not targets:
        raise HTTPException(400, "请选择要准备的配体")

    project = STATE.project
    job = STATE.jobs.create(
        "ligands",
        f"准备配体 · {len(targets)} 个",
        lambda job: _prepare_ligands_task(project, targets, options, job),
    )
    return {"job_id": job.id}


def _prepare_ligands_task(
    project: Project,
    targets: list[str],
    options: dict,
    job,
    progress_from: float = 0.0,
    progress_to: float = 1.0,
) -> None:
    """Prepare ligand records in parallel (shared by the API and auto mode)."""

    def task(job):
        total = len(targets)
        ok = 0
        filtered = 0
        completed = 0
        with STATE.lock:
            index = {item["id"]: item for item in project.data["ligands"]}
        threads = max(1, min(int(options.get("threads", 4) or 4), 32))
        job.log(f"[准备] 并行线程 {threads}，共 {total} 个配体")
        flush_every = max(1, min(500, max(1, total) // 20))
        log_every = max(1, min(200, max(1, total) // 20))
        dirty = 0

        def flush(force: bool = False) -> None:
            nonlocal dirty
            if dirty or force:
                with STATE.lock:
                    project.save()
                dirty = 0

        def prepare_one(lid: str):
            ligand = index.get(lid)
            if not ligand:
                return lid, None, "配体不存在"
            lig_dir = project.ligand_dir(ligand)
            if (
                ligand.get("pdbqt")
                and _abs(ligand["pdbqt"]).exists()
                and ligand.get("format") == "pdbqt"
            ):
                # Uploaded PDBQT files used to skip the drug-likeness filter
                # entirely.  Rebuild the molecule from the stored SMILES when
                # possible; otherwise record the gap instead of hiding it.
                verdict = None
                smiles = ligand.get("smiles") or ""
                if smiles:
                    try:
                        mol = ligmod.mol_from_smiles(smiles)
                    except Exception:
                        mol = None
                    if mol is not None:
                        verdict = ligfilters.evaluate(mol, ligand["name"])
                if options.get("filter", True):
                    if verdict and not verdict["passed"]:
                        return lid, {"filtered": True, "filter": verdict}, None
                    if verdict is None:
                        verdict = {
                            "passed": True,
                            "skipped": True,
                            "reasons": [],
                            "warnings": [
                                "PDBQT 上传：无法重建结构，未执行成药性过滤"
                            ],
                            "pains": [],
                            "alerts": [],
                            "props": {},
                        }
                return lid, {"skip": True, "filter": verdict}, None
            mol = None
            if ligand.get("smiles"):
                mol = ligmod.mol_from_smiles(ligand["smiles"])
            else:
                source = _abs(ligand.get("source"))
                if not source or not source.exists():
                    raise ValueError("找不到配体源文件")
                mols = ligmod.read_ligand_file(source)
                if not mols:
                    raise ValueError("无法读取分子")
                mol = mols[0][1]
            verdict = (
                ligfilters.evaluate(mol, ligand["name"]) if mol is not None else None
            )
            if (
                options.get("filter", True)
                and verdict is not None
                and not verdict["passed"]
            ):
                return lid, {"filtered": True, "filter": verdict}, None
            mol3d, pdbqt, meta = ligmod.prepare_mol_for_docking(
                mol,
                protonation=ligand.get("protonation_hint") or options.get("protonation", "ph7.4"),
                minimize=bool(options.get("minimize", True)),
                seed=int(options.get("seed", 42)),
            )
            pdbqt_path = lig_dir / "ligand.pdbqt"
            pdbqt_path.write_text(pdbqt, encoding="utf-8")
            sdf_path = lig_dir / "ligand.sdf"
            ligmod.write_sdf(mol3d, sdf_path, ligand["name"])
            return lid, {
                "pdbqt": _rel(pdbqt_path),
                "sdf": _rel(sdf_path),
                "meta": meta,
                "filter": verdict,
            }, None

        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = {
                executor.submit(prepare_one, lid): lid
                for lid in targets
                if not job.cancel_event.is_set()
            }
            for future in as_completed(futures):
                lid = futures[future]
                ligand = index.get(lid)
                if not ligand:
                    continue
                try:
                    _, payload, error = future.result()
                except Exception as exc:  # noqa: BLE001
                    payload, error = None, str(exc)
                with STATE.lock:
                    if error:
                        ligand["status"] = "error"
                        ligand["message"] = error
                    elif isinstance(payload, dict) and payload.get("skip"):
                        ligand["status"] = "ready"
                        ligand["message"] = "使用上传的 PDBQT"
                        if (payload or {}).get("filter"):
                            ligand["filter"] = payload["filter"]
                        ok += 1
                    elif isinstance(payload, dict) and payload.get("filtered"):
                        ligand["status"] = "filtered"
                        ligand["message"] = "成药性过滤：" + "；".join(
                            (payload.get("filter") or {}).get("reasons") or []
                        )
                        ligand["filter"] = payload.get("filter") or {}
                        filtered += 1
                    else:
                        ligand.update(
                            {
                                "status": "ready",
                                "message": "",
                                "pdbqt": (payload or {}).get("pdbqt", ""),
                                "sdf": (payload or {}).get("sdf", ""),
                                "meta": (payload or {}).get("meta", {}),
                                "filter": (payload or {}).get("filter")
                                or ligand.get("filter")
                                or {},
                                "smiles": ((payload or {}).get("meta") or {}).get(
                                    "smiles", ligand.get("smiles", "")
                                ),
                                "protonation": options.get("protonation"),
                            }
                        )
                        ok += 1
                    dirty += 1
                completed += 1
                if completed % log_every == 0 or completed == total:
                    job.set_progress(
                        progress_from
                        + (progress_to - progress_from) * (completed / total),
                        f"配体准备 {completed}/{total}",
                    )
                    job.log(
                        f"[准备] {completed}/{total} 完成，成功 {ok}"
                        + (f"，成药性过滤 {filtered}" if filtered else "")
                        + (f"，最近失败：{error}" if error else "")
                    )
                if dirty >= flush_every:
                    flush()
        flush(force=True)
        job.result = {"total": total, "ok": ok, "filtered": filtered}
        if job.cancel_event.is_set():
            with STATE.lock:
                for lid in targets:
                    ligand = index.get(lid)
                    if ligand and ligand.get("status") == "preparing":
                        ligand["status"] = "raw"
                        ligand["message"] = "已取消"
                project.save()
            job.status = "cancelled"

    task(job)


@app.post("/api/ligands/clear")
def clear_ligands(payload: dict):
    """Remove all prepared or all unprepared ligand records in one action."""
    _ensure_no_active_job()
    scope = str(payload.get("scope") or "").strip().lower()
    if scope not in ("prepared", "unprepared"):
        raise HTTPException(400, "scope 必须是 prepared 或 unprepared")
    with STATE.lock:
        project = STATE.project
        keep = []
        removed = []
        for item in project.data["ligands"]:
            ready = item.get("status") == "ready"
            drop = ready if scope == "prepared" else not ready
            (removed if drop else keep).append(item)
        project.data["ligands"] = keep
        project.save()

    def task(job):
        total = len(removed)
        for index, item in enumerate(removed, start=1):
            if job.cancel_event.is_set():
                break
            shutil.rmtree(project.ligand_dir(item), ignore_errors=True)
            if index % 100 == 0 or index == total:
                job.set_progress(index / max(total, 1), f"清理文件 {index}/{total}")
        job.result = {"removed": total}

    job = STATE.jobs.create(
        "ligands", f"清除{'已准备' if scope == 'prepared' else '未准备'}配体 · {len(removed)} 个", task
    )
    return {"job_id": job.id, "removed": len(removed)}


@app.get("/api/ligands/bulk/export")
def export_bulk_ligands():
    """Pack all prepared ligands into a portable ZIP for reuse across projects."""
    with STATE.lock:
        project = STATE.project
        root = project.root
        items = [
            item
            for item in project.data.get("ligands", [])
            if item.get("status") == "ready" and item.get("pdbqt")
        ]
    prepared = []
    for item in items:
        pdbqt = _abs(item.get("pdbqt"))
        if not pdbqt or not pdbqt.is_file():
            continue
        sdf = _abs(item.get("sdf")) if item.get("sdf") else None
        prepared.append((item, pdbqt, sdf if sdf and sdf.is_file() else None))
    if not prepared:
        raise HTTPException(400, "当前项目没有已准备的配体可储存")

    tmp_dir = root / "tmp_export"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    archive = tmp_dir / f"ligands_ready_{datetime.now():%Y%m%d_%H%M%S}.zip"
    manifest = {
        "app": "DrugHX",
        "version": APP_VERSION,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "count": len(prepared),
        "ligands": [],
    }
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for index, (item, pdbqt, sdf) in enumerate(prepared, start=1):
            stem = f"{index:05d}_{safe_name(item.get('name') or 'ligand')}"
            entry = {
                "name": item.get("name") or stem,
                "smiles": item.get("smiles") or (item.get("meta") or {}).get("smiles", ""),
                "meta": item.get("meta") or {},
                "pdbqt": f"files/{stem}.pdbqt",
            }
            zf.write(pdbqt, entry["pdbqt"])
            if sdf:
                entry["sdf"] = f"files/{stem}.sdf"
                zf.write(sdf, entry["sdf"])
            manifest["ligands"].append(entry)
        zf.writestr(
            "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=1)
        )
    return FileResponse(
        archive,
        filename=archive.name,
        media_type="application/zip",
        background=BackgroundTask(lambda: archive.unlink(missing_ok=True)),
    )


@app.post("/api/ligands/bulk/import")
async def import_bulk_ligands(file: UploadFile = File(...)):
    """Load a ZIP produced by /api/ligands/bulk/export into the current project."""
    _ensure_no_active_job()
    filename = Path(file.filename or "ligands_ready.zip").name
    if Path(filename).suffix.lower() != ".zip":
        raise HTTPException(400, "请选择“储存准备结果”导出的 ZIP 文件")
    with STATE.lock:
        project = STATE.project
        tmp_dir = project.root / "tmp_upload"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        archive = tmp_dir / safe_filename(filename, "ligands_ready.zip")
    size = 0
    with archive.open("wb") as handle:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > 2048 * 1024 * 1024:
                handle.close()
                archive.unlink(missing_ok=True)
                raise HTTPException(400, "ZIP 文件超过 2 GB")
            handle.write(chunk)

    imported = 0
    skipped = 0
    try:
        with zipfile.ZipFile(archive) as zf:
            try:
                manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
            except Exception as exc:
                raise HTTPException(400, f"ZIP 内缺少有效的 manifest.json：{exc}") from exc
            entries = manifest.get("ligands") or []
            with STATE.lock:
                names = {item["name"] for item in project.data["ligands"]}
                ready_names = {
                    item["name"]
                    for item in project.data["ligands"]
                    if item.get("status") == "ready"
                }
            dirty = 0
            for index, entry in enumerate(entries, start=1):
                member = str(entry.get("pdbqt") or "")
                try:
                    text = zf.read(member).decode("utf-8", errors="replace")
                except KeyError:
                    skipped += 1
                    continue
                name = str(entry.get("name") or f"ligand_{index}")
                if name in ready_names:
                    skipped += 1
                    continue
                with STATE.lock:
                    record = project.new_ligand(_unique_ligand_name(name, names))
                    names.add(record["name"])
                    ready_names.add(record["name"])
                    lig_dir = project.ligand_dir(record)
                    target = lig_dir / "ligand.pdbqt"
                    target.write_text(pdbio.normalize_pdbqt(text), encoding="utf-8")
                    sdf_target = lig_dir / "ligand.sdf"
                    has_sdf = False
                    sdf_member = str(entry.get("sdf") or "")
                    if sdf_member:
                        try:
                            sdf_target.write_bytes(zf.read(sdf_member))
                            has_sdf = True
                        except KeyError:
                            pass
                    meta = entry.get("meta") or {}
                    record.update(
                        {
                            "status": "ready",
                            "message": "从准备结果导入",
                            "pdbqt": _rel(target),
                            "sdf": _rel(sdf_target) if has_sdf else "",
                            "format": "pdbqt",
                            "smiles": entry.get("smiles") or meta.get("smiles", ""),
                            "meta": meta,
                        }
                    )
                    dirty += 1
                    if dirty % 200 == 0:
                        project.save()
                imported += 1
        with STATE.lock:
            project.save()
    finally:
        archive.unlink(missing_ok=True)
    return {"ok": True, "imported": imported, "skipped": skipped, "state": STATE.snapshot()}


@app.get("/api/ligands/{ligand_id}/sdf", response_class=PlainTextResponse)
def ligand_sdf(ligand_id: str):
    with STATE.lock:
        ligand = STATE.project.get_ligand(ligand_id)
        if not ligand:
            raise HTTPException(404, "配体不存在")
        path = _abs(ligand.get("sdf"))
    if not path or not path.exists():
        raise HTTPException(404, "配体还没有 3D 结构")
    return PlainTextResponse(path.read_text(encoding="utf-8", errors="replace"))


@app.get("/api/ligands/{ligand_id}/download")
def ligand_download(ligand_id: str):
    with STATE.lock:
        ligand = STATE.project.get_ligand(ligand_id)
        if not ligand:
            raise HTTPException(404, "配体不存在")
        path = _abs(ligand.get("pdbqt")) or _abs(ligand.get("sdf")) or _abs(ligand.get("source"))
    if not path or not path.exists():
        raise HTTPException(404, "文件不存在")
    return FileResponse(path, filename=f"{ligand['name']}{path.suffix}")


@app.delete("/api/ligands/{ligand_id}")
def delete_ligand(ligand_id: str):
    with STATE.lock:
        ok = STATE.project.remove_ligand(ligand_id)
    if not ok:
        raise HTTPException(404, "配体不存在")
    return {"ok": True, "state": STATE.snapshot()}


# ---------------------------------------------------------------------------
# toolbox: descriptor export, diversity picking, batch standardisation
# ---------------------------------------------------------------------------
@app.get("/api/tools/ligands.csv")
def export_ligand_descriptors():
    rows = [[
        "name", "smiles", "status", "MW", "cLogP", "TPSA", "HBD", "HBA",
        "RotB", "QED", "alerts", "Lipinski", "Veber",
    ]]
    with STATE.lock:
        ligands = [dict(item) for item in STATE.project.data["ligands"]]
    for ligand in ligands:
        smiles = ligand.get("smiles") or (ligand.get("meta") or {}).get("smiles")
        if not smiles:
            continue
        try:
            block = admet.compact(admet.profile_smiles(smiles, ligand.get("name", "")))
        except Exception:
            continue
        rows.append([
            ligand.get("name", ""),
            smiles,
            ligand.get("status", ""),
            round(float(block.get("mw", 0)), 2),
            round(float(block.get("logp", 0)), 2),
            round(float(block.get("tpsa", 0)), 1),
            block.get("hbd", ""),
            block.get("hba", ""),
            block.get("rotb", ""),
            round(float(block.get("qed", 0)), 3),
            block.get("alerts", ""),
            "yes" if block.get("lipinski") else "no",
            "yes" if block.get("veber") else "no",
        ])
    buffer = io.StringIO()
    csv.writer(buffer).writerows(rows)
    return Response(
        content="\ufeff" + buffer.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="drughx_ligands.csv"'},
    )


@app.post("/api/tools/diversity")
def pick_diverse_ligands(payload: dict):
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    count = max(1, min(int(payload.get("count") or 12), 200))
    with STATE.lock:
        ligands = [dict(item) for item in STATE.project.data["ligands"]]
    items = []
    for ligand in ligands:
        smiles = ligand.get("smiles") or (ligand.get("meta") or {}).get("smiles")
        if not smiles:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            items.append(ligand)
    if not items:
        raise HTTPException(400, "项目里还没有带 SMILES 的配体")
    if count >= len(items):
        selected = list(range(len(items)))
    else:
        generator = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
        fingerprints = []
        for ligand in items:
            smiles = ligand.get("smiles") or (ligand.get("meta") or {}).get("smiles")
            fingerprints.append(generator.GetFingerprint(Chem.MolFromSmiles(smiles)))
        selected = [0]
        while len(selected) < count:
            best_index = None
            best_similarity = 2.0
            for index in range(len(items)):
                if index in selected:
                    continue
                similarity = max(
                    DataStructs.TanimotoSimilarity(
                        fingerprints[index], fingerprints[chosen]
                    )
                    for chosen in selected
                )
                if similarity < best_similarity:
                    best_similarity = similarity
                    best_index = index
            if best_index is None:
                break
            selected.append(best_index)
    chosen = [items[index] for index in sorted(selected)]
    return {
        "ok": True,
        "ids": [item["id"] for item in chosen],
        "names": [item.get("name", "") for item in chosen],
        "count": len(chosen),
    }


@app.post("/api/tools/structure-to-smiles")
async def structure_to_smiles(file: UploadFile = File(...)):
    """Convert the first molecule in an SDF/SMILES/CSV upload to SMILES."""
    from rdkit import Chem

    filename = Path(file.filename or "input.sdf").name
    suffix = Path(file.filename or "input.sdf").suffix.lower() or ".sdf"
    data = await file.read()
    if len(data) > 60 * 1024 * 1024:
        raise HTTPException(400, "文件超过 60 MB")
    outdir = STATE.project.root / "exports" / "tools"
    outdir.mkdir(parents=True, exist_ok=True)
    source = outdir / f"extract_{datetime.now():%Y%m%d_%H%M%S}{suffix}"
    source.write_bytes(data)
    try:
        molecules = ligmod.read_ligand_file(source)
    except Exception as exc:
        raise HTTPException(400, f"{filename}: {exc}") from exc
    finally:
        source.unlink(missing_ok=True)
    rows = []
    for name, mol in molecules:
        try:
            heavy = Chem.RemoveHs(mol)
            smiles = Chem.MolToSmiles(heavy, isomericSmiles=True)
        except Exception:
            continue
        if smiles:
            rows.append({"name": name or f"ligand_{len(rows) + 1}", "smiles": smiles})
    if not rows:
        raise HTTPException(400, "没有从文件中读取到有效分子")
    return {"ok": True, "molecules": rows, "count": len(rows)}


@app.post("/api/tools/smiles-to-sdf")
def smiles_to_sdf(payload: dict):
    """Convert one SMILES to a 2D SDF file for download."""
    from rdkit import Chem
    from rdkit.Chem import rdDepictor

    smiles = str(payload.get("smiles") or "").strip()
    name = str(payload.get("name") or "ligand").strip() or "ligand"
    if not smiles:
        raise HTTPException(400, "请输入 SMILES")
    try:
        mol = ligmod.mol_from_smiles(smiles)
    except Exception as exc:
        raise HTTPException(400, f"SMILES 无法解析：{exc}") from exc
    rdDepictor.Compute2DCoords(mol)
    mol.SetProp("_Name", name)
    block = Chem.MolToMolBlock(mol) + "$$$$\n"
    filename = f"{safe_name(name)}.sdf"
    return PlainTextResponse(
        block,
        media_type="chemical/x-mdl-sdfile",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/tools/standardize")
async def standardize_uploaded_file(file: UploadFile = File(...)):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    filename = Path(file.filename or "input.sdf").name
    suffix = Path(filename).suffix.lower() or ".sdf"
    data = await file.read()
    if len(data) > 120 * 1024 * 1024:
        raise HTTPException(400, "文件超过 120 MB")
    outdir = STATE.project.root / "exports" / "tools"
    outdir.mkdir(parents=True, exist_ok=True)
    source = outdir / f"upload_{datetime.now():%Y%m%d_%H%M%S}{suffix}"
    source.write_bytes(data)
    try:
        molecules = ligmod.read_ligand_file(source)
    except Exception as exc:
        raise HTTPException(400, f"{filename}: {exc}") from exc
    finally:
        source.unlink(missing_ok=True)
    blocks = []
    for index, (name, mol) in enumerate(molecules, start=1):
        try:
            clean = ligmod.standardize_mol(mol, protonation="ph7.4")
        except Exception:
            continue
        if clean is None or clean.GetNumAtoms() == 0:
            continue
        label = name or f"ligand_{index}"
        clean.SetProp("_Name", label)
        if clean.GetNumConformers() == 0:
            AllChem.Compute2DCoords(clean)
        blocks.append(
            Chem.MolToMolBlock(clean)
            + f">  <NAME>\n{label}\n\n$$$$\n"
        )
    if not blocks:
        raise HTTPException(400, "没有可标准化的分子")
    target = outdir / f"standardized_{datetime.now():%Y%m%d_%H%M%S}.sdf"
    target.write_text("".join(blocks), encoding="utf-8")
    return FileResponse(
        target, filename="standardized.sdf", media_type="chemical/x-mdl-sdfile"
    )


# ---------------------------------------------------------------------------
# boxes / pockets
# ---------------------------------------------------------------------------
@app.get("/api/receptors/{receptor_id}/pockets")
def receptor_pockets(receptor_id: str):
    with STATE.lock:
        receptor = STATE.project.get_receptor(receptor_id)
        if not receptor:
            raise HTTPException(404, "受体不存在")
        pockets = receptor.get("pockets") or []
        hetero = receptor.get("hetero") or []
    return {"pockets": pockets, "hetero": hetero}


@app.post("/api/receptors/{receptor_id}/pockets/detect")
def re_detect_pockets(receptor_id: str):
    with STATE.lock:
        receptor = STATE.project.get_receptor(receptor_id)
        if not receptor:
            raise HTTPException(404, "受体不存在")
        path = _abs(receptor.get("pdb")) or _abs(receptor.get("source"))
        if not path or not path.exists():
            raise HTTPException(400, "请先准备受体")
        text = path.read_text(encoding="utf-8", errors="replace")
    pockets = detect_pockets(text, top=5)
    with STATE.lock:
        receptor["pockets"] = pockets
        STATE.project.save()
    return {"ok": True, "pockets": pockets}


@app.post("/api/receptors/{receptor_id}/box/from-ligand")
def box_from_ligand(receptor_id: str, payload: dict):
    ref = payload.get("ligand") or {}
    with STATE.lock:
        receptor = STATE.project.get_receptor(receptor_id)
        if not receptor:
            raise HTTPException(404, "受体不存在")
        source = _abs(receptor.get("source"))
        if not source or not source.exists():
            raise HTTPException(400, "找不到受体源文件")
        text = source.read_text(encoding="utf-8", errors="replace")
    block = pdbio.extract_residue_pdb(
        text, ref.get("chain", ""), int(ref.get("resseq", 0)),
        ref.get("icode", ""), ref.get("name", "")
    )
    if not block:
        raise HTTPException(400, "未能提取该共晶配体")
    coords = [
        [a.x, a.y, a.z]
        for a in pdbio.parse_pdb(block).atoms
        if a.element != "H"
    ]
    box = box_from_coords(coords, padding=6.0, min_size=20.0)
    box["source"] = "ligand"
    box["label"] = f"共晶配体 {ref.get('name', '')}"
    with STATE.lock:
        receptor = STATE.project.get_receptor(receptor_id)
        receptor["box"] = box
        STATE.project.save()
    return {"ok": True, "box": box, "volume": box_volume(box["size"])}


@app.post("/api/receptors/{receptor_id}/extract-ligand")
def extract_reference_ligand(receptor_id: str, payload: dict):
    ref = payload.get("ligand") or {}
    with STATE.lock:
        receptor = STATE.project.get_receptor(receptor_id)
        if not receptor:
            raise HTTPException(404, "受体不存在")
        source = _abs(receptor.get("source"))
        if not source or not source.exists():
            raise HTTPException(400, "找不到受体源文件")
        text = source.read_text(encoding="utf-8", errors="replace")
    block = pdbio.extract_residue_pdb(
        text, ref.get("chain", ""), int(ref.get("resseq", 0)),
        ref.get("icode", ""), ref.get("name", "")
    )
    if not block:
        raise HTTPException(400, "未能提取该共晶配体")

    from rdkit import Chem

    # Prefer the RCSB ideal SDF: it carries correct bond orders and
    # aromaticity, unlike a bare PDB HETATM block. The chemical component
    # ID is global, so this also works for uploaded receptor PDB files.
    ideal_text = None
    try:
        ideal_text = recmod.fetch_ligand_ideal_sdf(ref.get("name", ""))
    except Exception:
        ideal_text = None

    with STATE.lock:
        project = STATE.project
        base_name = f"{ref.get('name', 'ligand')}_{ref.get('chain', '')}{ref.get('resseq', '')}"
        record = project.new_ligand(_unique_ligand_name(base_name))
        lig_dir = project.ligand_dir(record)
        pdb_path = lig_dir / "source.pdb"
        pdb_path.write_text(block, encoding="utf-8")
        sdf_path = lig_dir / "source.sdf"
        meta = {"name": record["name"]}
        source_rel = _rel(pdb_path)
        format_name = "pdb"

        if ideal_text:
            sdf_path.write_text(ideal_text, encoding="utf-8")
            mol = None
            try:
                mol = Chem.MolFromMolBlock(
                    ideal_text.split("$$$$")[0], removeHs=False, sanitize=True
                )
            except Exception:
                mol = None
            if mol is not None:
                meta = ligmod.ligand_metadata(mol)
                source_rel = _rel(sdf_path)
                format_name = "sdf"
        else:
            # Unknown bond orders: avoid heuristic protonation changes.
            try:
                mol = Chem.MolFromPDBBlock(
                    block, removeHs=False, sanitize=False, proximityBonding=True
                )
                if mol is not None:
                    try:
                        Chem.SanitizeMol(mol)
                    except Exception:
                        pass
                    ligmod.write_sdf(mol, sdf_path, record["name"])
                    meta = ligmod.ligand_metadata(mol)
            except Exception:
                pass

        record.update(
            {
                "source": source_rel,
                "sdf": _rel(sdf_path) if sdf_path.exists() else "",
                "format": format_name,
                "meta": meta,
                "extracted_from": ref.get("key", ""),
                "protonation_hint": "" if ideal_text else "keep",
            }
        )
        project.save()
    return {
        "ok": True,
        "ligand_id": record["id"],
        "ideal_sdf": bool(ideal_text),
        "state": STATE.snapshot(),
    }


# ---------------------------------------------------------------------------
# docking
# ---------------------------------------------------------------------------
@app.post("/api/dock")
def dock(payload: dict):
    _ensure_no_active_job()
    receptor_ids = payload.get("receptor_ids") or []
    ligand_ids = payload.get("ligand_ids") or []
    scope = str(payload.get("scope") or "").strip().lower()
    bulk = scope in ("all_ready", "ready") or bool(payload.get("bulk"))
    params = dict(DEFAULT_DOCK_PARAMS)
    params.update(payload.get("params") or {})
    if bulk:
        # Bulk screens must tolerate a handful of broken ligands: skip them
        # instead of aborting the whole 10k+ ligand run.
        params["bulk"] = True
        params["skip_invalid"] = True
        params["skip_repair"] = True
    with STATE.lock:
        project = STATE.project
        lig_index = {item["id"]: item for item in project.data["ligands"]}
        if bulk:
            ligand_ids = []
            seen_ids = set()
            for item in project.data["ligands"]:
                lid = item["id"]
                if lid in seen_ids:
                    continue
                if (
                    item.get("status") == "ready"
                    and item.get("pdbqt")
                    and _abs(item["pdbqt"]).exists()
                ):
                    seen_ids.add(lid)
                    ligand_ids.append(lid)
        if not receptor_ids or not ligand_ids:
            raise HTTPException(400, "请选择受体和配体")
        for rid in receptor_ids:
            receptor = project.get_receptor(rid)
            if not receptor:
                raise HTTPException(404, "受体不存在")
            if not receptor.get("pdbqt") or not _abs(receptor["pdbqt"]).exists():
                raise HTTPException(400, f"受体 {receptor['name']} 尚未准备")
            if not receptor.get("box"):
                raise HTTPException(400, f"受体 {receptor['name']} 尚未设置对接盒")
        valid_ids = []
        for lid in ligand_ids:
            ligand = lig_index.get(lid)
            if not ligand:
                if bulk:
                    continue
                raise HTTPException(404, "配体不存在")
            if not ligand.get("pdbqt") or not _abs(ligand["pdbqt"]).exists():
                if bulk:
                    continue
                raise HTTPException(400, f"配体 {ligand['name']} 尚未准备")
            valid_ids.append(lid)
        if bulk:
            if not valid_ids:
                raise HTTPException(400, "没有可用于大规模对接的已准备配体")
            ligand_ids = valid_ids
        if not bulk:
            # Keep project defaults for the fine-docking page; a bulk screen
            # must not overwrite them with ultra-fast settings.
            project.data["params"] = params
        run = project.new_run(params)
        project.save()

    def task(job):
        try:
            run_docking(project, run, receptor_ids, ligand_ids, params, job)
        except Exception:
            with STATE.lock:
                run["status"] = "error"
                run["finished"] = True
                project.save()
            raise

    job = STATE.jobs.create("dock", f"分子对接 · {len(receptor_ids)}×{len(ligand_ids)}", task)
    return {"job_id": job.id, "run_id": run["id"]}


# ---------------------------------------------------------------------------
# one-click automatic pipeline
# ---------------------------------------------------------------------------
class _JobScope:
    """Map a sub-step's 0..1 progress onto a slice of the parent job."""

    def __init__(self, job, start: float = 0.0, end: float = 1.0):
        object.__setattr__(self, "_job", job)
        object.__setattr__(self, "_start", float(start))
        object.__setattr__(self, "_end", float(end))
        object.__setattr__(self, "log", job.log)
        object.__setattr__(self, "cancel_event", job.cancel_event)

    def set_progress(self, value: float, message: str | None = None) -> None:
        value = max(0.0, min(1.0, float(value or 0.0)))
        self._job.set_progress(
            self._start + (self._end - self._start) * value, message
        )

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_job"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_job"), name, value)


def _auto_prepare_receptor(
    project: Project, filename: str, suffix: str, text: str, job
) -> str:
    """Create, prepare and box one receptor; return its record id."""
    stem = Path(filename).stem or "Receptor"
    with STATE.lock:
        record = project.new_receptor(stem)
        receptor_id = record["id"]
        rec_dir = project.receptor_dir(record)
        project.save()

    if suffix == ".pdbqt":
        source = rec_dir / "source.pdbqt"
        source.write_text(text, encoding="utf-8")
        pdb_text = pdbio.pdbqt_to_pdb(text)
        view = rec_dir / "receptor.pdb"
        view.write_text(pdb_text, encoding="utf-8")
        target = rec_dir / "receptor.pdbqt"
        target.write_text(text, encoding="utf-8")
        pockets = detect_pockets(pdb_text, top=5)
        with STATE.lock:
            record = project.get_receptor(receptor_id)
            record.update(
                {
                    "source": _rel(source),
                    "pdb": _rel(view),
                    "pdb_clean": _rel(view),
                    "pdbqt": _rel(target),
                    "status": "ready",
                    "message": "使用上传的 PDBQT",
                    "meta": {
                        **pdbio.pdb_info(pdb_text),
                        "pdbqt_atoms": pdbio.pdbqt_stats(text)["atoms"],
                        "from_pdbqt": True,
                    },
                    "pockets": pockets,
                }
            )
            if pockets:
                top = pockets[0]
                record["box"] = {
                    "center": top["center"],
                    "size": top["size"],
                    "source": "pocket",
                    "label": top["label"],
                }
            project.save()
        job.log(f"受体 {stem}：直接使用上传的 PDBQT")
        return receptor_id

    source = rec_dir / "source.pdb"
    source.write_text(text, encoding="utf-8")
    hetero = pdbio.hetero_ligands(text)
    with STATE.lock:
        record = project.get_receptor(receptor_id)
        record.update(
            {
                "source": _rel(source),
                "pdb": _rel(source),
                "meta": pdbio.pdb_info(text),
                "hetero": hetero,
            }
        )
        project.save()

    job.log(f"受体 {stem}：清洁结构并转换为 PDBQT ...")
    result = recmod.prepare_receptor(
        rec_dir, text, dict(DEFAULT_RECEPTOR_OPTIONS), job.log, hetero=hetero
    )
    box = result.get("box")
    if not box:
        # No co-crystal ligand and no detected cavity: fall back to a box around
        # the whole cleaned structure so the pipeline can still run.
        try:
            atoms = pdbio.parse_pdb(
                Path(result["pdb"]).read_text(encoding="utf-8", errors="replace")
            ).atoms
            coords = [[a.x, a.y, a.z] for a in atoms if a.element != "H"]
            if coords:
                box = box_from_coords(coords, padding=0.0, min_size=20.0)
                box["source"] = "structure"
                box["label"] = "全结构包围盒"
        except Exception:
            box = None
    if not box:
        raise RuntimeError("无法确定对接盒，请改用带共晶配体或明显口袋的受体结构")

    with STATE.lock:
        record = project.get_receptor(receptor_id)
        record.update(
            {
                "status": "ready",
                "message": "",
                "pdb": _rel(Path(result["pdb"])),
                "pdb_clean": _rel(Path(result["pdb_clean"])),
                "pdbqt": _rel(Path(result["pdbqt"])),
                "meta": result["meta"],
                "pockets": result["pockets"],
                "hetero": hetero,
                "box": box,
            }
        )
        project.save()
    job.log(f"受体准备完成：{len(result['pockets'])} 个候选口袋，对接盒来源 {box.get('source')}")
    return receptor_id


def _auto_resolve_receptor(project: Project, payload: dict, job) -> tuple[str, dict]:
    """Resolve a receptor from an uploaded file, a gene/UniProt id or a sequence."""
    mode = str(payload.get("mode") or "file").lower()
    if mode == "file":
        return _auto_prepare_receptor(
            project,
            payload.get("filename") or "receptor.pdb",
            payload.get("suffix") or ".pdb",
            payload.get("text") or "",
            job,
        ), {}

    annotation: dict = {}
    if mode == "pdb":
        pdb_id = str(payload.get("pdb_id") or "").strip().upper()
        if len(pdb_id) < 4:
            raise RuntimeError("请输入 4 位 RCSB PDB ID，例如 6LU7")
        job.set_progress(0.02, f"从 RCSB PDB 下载 {pdb_id} ...")
        try:
            text = recmod.fetch_pdb(pdb_id)
        except Exception as exc:
            raise RuntimeError(f"PDB {pdb_id} 下载失败：{exc}") from exc
        job.log(f"[自动] 已下载 RCSB PDB {pdb_id}（{len(text.splitlines())} 行）")
        receptor_id = _auto_prepare_receptor(
            project, f"{pdb_id}.pdb", ".pdb", text, job
        )
        return receptor_id, {
            "source": f"RCSB PDB {pdb_id}",
            "gene": pdb_id,
            "pdb_id": pdb_id,
        }

    if mode == "gene":
        query = str(payload.get("query") or "").strip()
        species = str(payload.get("species") or "human")
        job.set_progress(0.02, f"在线查询靶点 {query} 并获取 AlphaFold 模型 ...")
        annotation = online.uniprot_lookup(query, species)
        if not annotation.get("available"):
            raise RuntimeError(f"在线靶点查询失败：{annotation.get('error')}")
        job.log(
            f"[自动] UniProt {annotation.get('accession')} "
            f"({annotation.get('gene')}, {annotation.get('length')} aa)"
        )
        structure = online.alphafold_model(annotation.get("accession") or "")
        if not structure.get("available"):
            job.log(f"[自动] AlphaFold DB 不可用（{structure.get('error')}），改用 ESMFold")
            structure = online.esmfold_model(annotation.get("sequence") or "")
    else:
        job.set_progress(0.02, "使用 ESMFold 预测蛋白结构 ...")
        structure = online.esmfold_model(payload.get("sequence") or "")

    if not structure.get("available"):
        raise RuntimeError(
            f"在线结构获取失败：{structure.get('error')}"
            "。ESM Atlas 公共接口偶发限流，请稍后重试，或改用基因 / UniProt 登录号获取 AlphaFold 模型。"
        )
    source = str(structure.get("source") or "Online")
    name = (
        annotation.get("gene")
        or payload.get("query")
        or "Predicted"
    )
    name = f"{name}_{source.split(' ')[0]}"
    job.log(f"[自动] 结构来源 {source}，长度 {structure.get('length')} aa")
    receptor_id = _auto_prepare_receptor(
        project, f"{name}.pdb", ".pdb", structure.get("pdb") or "", job
    )
    info = {
        "source": source,
        "accession": structure.get("accession") or annotation.get("accession") or "",
        "gene": annotation.get("gene") or str(payload.get("query") or ""),
        "length": structure.get("length"),
        "plddt": (structure.get("plddt") or {}).get("mean"),
        "model_version": structure.get("model_version"),
        "uniprot": annotation if annotation.get("available") else {},
    }
    return receptor_id, info


def _auto_import_ligands(project: Project, payloads: list[tuple], job) -> list[str]:
    """Create ligand records from uploaded files; return their ids."""
    targets: list[str] = []
    names = {item["name"] for item in project.data["ligands"]}
    tmp_dir = project.root / "tmp_auto"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        for filename, suffix, data in payloads:
            base = Path(filename).stem or "Ligand"
            if suffix == ".pdbqt":
                text = data.decode("utf-8", errors="replace")
                with STATE.lock:
                    record = project.new_ligand(_unique_ligand_name(base, names))
                    names.add(record["name"])
                    lig_dir = project.ligand_dir(record)
                    source = lig_dir / "source.pdbqt"
                    source.write_bytes(data)
                    target = lig_dir / "ligand.pdbqt"
                    target.write_text(pdbio.normalize_pdbqt(text), encoding="utf-8")
                    record.update(
                        {
                            "status": "ready",
                            "source": _rel(source),
                            "pdbqt": _rel(target),
                            "format": "pdbqt",
                            "message": "使用上传的 PDBQT",
                            "meta": {"name": record["name"]},
                        }
                    )
                    targets.append(record["id"])
                    project.save()
                continue

            tmp = tmp_dir / safe_filename(filename)
            tmp.write_bytes(data)
            try:
                mols = ligmod.read_ligand_file(tmp)
            except Exception as exc:
                job.log(f"[跳过] {filename}: {exc}")
                tmp.unlink(missing_ok=True)
                continue
            tmp.unlink(missing_ok=True)
            if not mols:
                job.log(f"[跳过] {filename}: 未解析到分子")
                continue
            with STATE.lock:
                for index, (mol_name, mol) in enumerate(mols, start=1):
                    default = base if len(mols) == 1 else f"{base}_{index}"
                    name = mol_name if len(mols) > 1 and mol_name else default
                    record = project.new_ligand(_unique_ligand_name(name, names))
                    names.add(record["name"])
                    lig_dir = project.ligand_dir(record)
                    src_copy = lig_dir / "source.sdf"
                    try:
                        ligmod.write_sdf(mol, src_copy, record["name"])
                    except Exception:
                        src_copy.write_bytes(data)
                    record.update(
                        {
                            "source": _rel(src_copy),
                            "sdf": _rel(src_copy),
                            "format": "sdf",
                            "meta": ligmod.ligand_metadata(mol),
                        }
                    )
                    targets.append(record["id"])
                project.save()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return targets


def _auto_refine_box(
    project: Project, receptor_id: str, ligand_ids: list[str], job
) -> None:
    """Grow the docking box when the prepared ligands do not fit comfortably."""
    from rdkit import Chem

    with STATE.lock:
        receptor = project.get_receptor(receptor_id)
        box = dict((receptor or {}).get("box") or {})
        ligands = [
            project.get_ligand(lid) for lid in ligand_ids[:30]
        ]
    if not receptor or not box:
        return
    spans = []
    for ligand in ligands:
        path = _abs((ligand or {}).get("sdf"))
        if not path or not path.exists():
            continue
        try:
            supplier = Chem.SDMolSupplier(str(path), removeHs=True, sanitize=False)
            mol = next((item for item in supplier if item is not None), None)
            if mol is None or mol.GetNumConformers() == 0:
                continue
            coords = mol.GetConformer().GetPositions()
            spans.append((coords.max(axis=0) - coords.min(axis=0)).tolist())
        except Exception:
            continue
    if not spans:
        return
    suggestion = suggest_box(box, spans, padding=4.0, max_size=40.0)
    if not suggestion["expanded"]:
        return
    with STATE.lock:
        receptor["box"] = {
            "center": suggestion["center"],
            "size": suggestion["size"],
            "label": (box.get("label") or "自动") + " · 配体自适应",
        }
        project.save()
    job.log(
        f"[自动] 对接盒按配体尺寸扩展：{suggestion['original_size']} → "
        f"{suggestion['size']} Å"
    )


def _auto_coarse_params() -> dict:
    """Fast pre-screen: Vina-GPU when installed, otherwise Vina CPU.

    Coarse screening only needs a reliable *relative* ranking, so the GPU
    search uses the cheapest legal setting (thread=1000, depth=2) instead of
    the full ``DEFAULT_DOCK_PARAMS`` grid (thread=8000, depth=20).
    Measured on a 20-ligand library with an RTX 4070: 0.21 s/ligand versus
    1.79 s/ligand, i.e. about 8x faster on an 800-ligand library.
    """
    params = dict(DEFAULT_DOCK_PARAMS)
    if VINA_GPU_EXE.is_file():
        params.update(
            {
                "docking_engine": "vina-gpu",
                "engine": "gpu",
                "exhaustiveness": 8,
                "num_modes": 1,
                "energy_range": 3.0,
                "rescore": False,
                "gpu_threads": 1000,
                "gpu_search_depth": 2,
            }
        )
    else:
        params.update(
            {
                "docking_engine": "vina",
                "engine": "cpu",
                "exhaustiveness": 1,
                "num_modes": 1,
                "energy_range": 3.0,
                "rescore": False,
                "parallel_jobs": 4,
            }
        )
    params.update({"bulk": True, "skip_invalid": True, "skip_repair": True})
    return params


def _auto_fine_params() -> dict:
    """Detailed multi-engine consensus docking for the shortlisted ligands."""
    params = dict(DEFAULT_DOCK_PARAMS)
    params.update(
        {
            "docking_engine": "vina",
            "engine": "cpu",
            "exhaustiveness": 16,
            "num_modes": 9,
            "consensus": True,
            "bulk": False,
            # Four independent Vina processes; each keeps 8 of the 32 logical
            # cores.  Single-process Vina scales poorly past ~8 threads, so
            # this is ~3x faster on a 10-ligand shortlist with identical
            # per-ligand search depth.
            "parallel_jobs": 4,
        }
    )
    return params


def _auto_online_summary(best: dict, receptor_info: dict, job) -> dict:
    """Optional, failure-tolerant online intel for the automatic pipeline."""
    summary: dict = {}
    if receptor_info:
        summary["receptor"] = receptor_info
        gene = receptor_info.get("gene") or ""
        if gene:
            try:
                job.log(f"[自动] 在线获取靶点情报：{gene}")
                intel = online.target_intel(gene, "human")
                if intel.get("available"):
                    ot = intel.get("opentargets") or {}
                    pdb = intel.get("pdb") or {}
                    summary["target"] = {
                        "gene": (intel.get("uniprot") or {}).get("gene") or gene,
                        "accession": (intel.get("uniprot") or {}).get("accession") or "",
                        "disease_count": ot.get("disease_count"),
                        "top_diseases": (ot.get("diseases") or [])[:5],
                        "pdb_count": pdb.get("total") or len(pdb.get("entries") or []),
                        "tractability": (ot.get("tractability") or [])[:6],
                    }
            except Exception as exc:
                job.log(f"[自动] 靶点情报获取失败：{exc}")
    name = str(best.get("ligand_name") or "")
    smiles = _result_smiles(best) or ""
    try:
        if name or smiles:
            job.log(f"[自动] 在线获取最优分子记录：{name or 'SMILES'}")
            record = online.pubchem_compound(name, "name") if name else online.pubchem_compound(smiles, "smiles")
            if record.get("available"):
                compound = {
                    "query": name or smiles,
                    "cid": record.get("cid"),
                    "title": record.get("title"),
                    "formula": record.get("formula"),
                    "mw": record.get("mw"),
                    "xlogp": record.get("xlogp"),
                    "tpsa": record.get("tpsa"),
                    "smiles": record.get("smiles"),
                    "similar": [],
                }
                if record.get("smiles"):
                    similar = online.pubchem_similar(record["smiles"], 0.9, 5)
                    compound["similar"] = [
                        {
                            "cid": row.get("cid"),
                            "title": row.get("title"),
                            "smiles": row.get("smiles"),
                        }
                        for row in (similar.get("rows") or [])
                    ]
                summary["compound"] = compound
    except Exception as exc:
        job.log(f"[自动] 化合物情报获取失败：{exc}")
    return summary


def _run_auto_pipeline(
    project: Project,
    receptor_payload: dict,
    ligand_payloads: list[tuple],
    langs: list[str],
    job,
    online_intel: bool = True,
) -> None:
    """Receptor → ligands → screen → consensus docking → optimisation → report."""
    job.log(f"[自动] 项目 {project.data['name']}，输出目录 {project.root}")
    job.set_progress(0.01, "准备受体 ...")
    receptor_id, receptor_online = _auto_resolve_receptor(project, receptor_payload, job)

    job.set_progress(0.08, "导入配体 ...")
    targets = _auto_import_ligands(project, ligand_payloads, job)
    if not targets:
        raise RuntimeError("没有导入任何有效配体")
    job.log(f"[自动] 已导入 {len(targets)} 个配体")

    _prepare_ligands_task(
        project,
        targets,
        dict(DEFAULT_LIGAND_OPTIONS),
        job,
        progress_from=0.10,
        progress_to=0.24,
    )
    ready = [
        lid for lid in targets
        if (project.get_ligand(lid) or {}).get("status") == "ready"
    ]
    if not ready:
        raise RuntimeError("所有配体准备失败，请检查配体文件")
    job.log(f"[自动] 配体准备完成：{len(ready)}/{len(targets)}")
    _auto_refine_box(project, receptor_id, ready, job)

    coarse_run = None
    selected = list(ready)
    if len(ready) > 10:
        coarse_params = _auto_coarse_params()
        label = "Vina-GPU" if VINA_GPU_EXE.is_file() else "Vina CPU"
        job.set_progress(
            0.24,
            f"配体 {len(ready)} 个超过 10 个，先用 {label} 粗筛 ...",
        )
        coarse_run = project.new_run(coarse_params)
        project.save()
        if coarse_params.get("engine") == "gpu":
            coarse_settings = (
                f"thread={coarse_params.get('gpu_threads')}，"
                f"search_depth={coarse_params.get('gpu_search_depth')}"
            )
        else:
            coarse_settings = (
                f"exhaustiveness={coarse_params.get('exhaustiveness')}，"
                f"并行 {coarse_params.get('parallel_jobs')}"
            )
        job.log(f"[自动] 粗筛 {len(ready)} 个配体（{label}，{coarse_settings}）")
        coarse_started = time.monotonic()
        run_docking(
            project,
            coarse_run,
            [receptor_id],
            ready,
            coarse_params,
            _JobScope(job, 0.24, 0.44),
        )
        coarse_seconds = time.monotonic() - coarse_started
        hits = [
            item for item in coarse_run.get("results", [])
            if item.get("status") == "ok" and item.get("affinity") is not None
        ]
        hits.sort(key=lambda item: float(item["affinity"]))
        selected = list(
            dict.fromkeys(
                str(item.get("ligand_id")) for item in hits[:24]
            )
        )
        if not selected:
            raise RuntimeError("粗筛没有产生成功结果")
        job.log(
            f"[自动] 粗筛完成，用时 {coarse_seconds:.1f} s，"
            f"取前 {len(selected)} 个配体进入精细对接"
        )
    else:
        job.log(f"[自动] 配体 {len(ready)} 个，不超过 10 个，跳过粗筛")

    job.set_progress(0.46, "多工具共识精细对接 ...")
    fine_params = _auto_fine_params()
    fine_run = project.new_run(fine_params)
    project.save()
    job.log(f"[自动] 精细对接 {len(selected)} 个配体（多引擎共识）")
    run_docking(
        project,
        fine_run,
        [receptor_id],
        selected,
        fine_params,
        _JobScope(job, 0.46, 0.72),
    )
    pairs = [
        (index, item)
        for index, item in enumerate(fine_run.get("results", []))
        if item.get("status") == "ok" and item.get("affinity") is not None
    ]
    if not pairs:
        raise RuntimeError("精细对接没有产生成功结果")
    best_index, best = min(pairs, key=lambda pair: float(pair[1]["affinity"]))
    job.log(
        f"[自动] 精细对接完成，当前最优 {best.get('ligand_name')}"
        f"（{best.get('affinity')} kcal/mol）；等待手动确认候选分子"
    )
    online_summary: dict = {}
    if online_intel:
        job.set_progress(0.73, "在线获取靶点与候选分子情报 ...")
        online_summary = _auto_online_summary(best, receptor_online, job)
        if online_summary:
            with STATE.lock:
                project.data["auto_online"] = online_summary
                project.save()

    candidates = _auto_candidates(
        project, fine_run, receptor_pdb=_result_receptor_pdb(best)
    )
    pending = {
        "run_id": fine_run["id"],
        "coarse_run_id": coarse_run["id"] if coarse_run else "",
        "receptor_id": receptor_id,
        "langs": langs,
        "screened": len(ready),
        "fine_count": len(selected),
        "candidates": candidates,
        "online": online_summary,
    }
    with STATE.lock:
        project.data["auto_pending"] = pending
        project.save()
    job.set_progress(1.0, "精细对接完成，请选择候选分子")
    job.result = {
        "run_id": fine_run["id"],
        "coarse_run_id": pending["coarse_run_id"],
        "screened": len(ready),
        "fine_count": len(selected),
        "best_ligand": best.get("ligand_name"),
        "best_affinity": best.get("affinity"),
        "project": project.data["name"],
        "project_path": str(project.root),
        "report_dir": str(project.root / "exports" / "full_report"),
        "awaiting": True,
        "candidates": candidates,
        "online": online_summary,
    }


def _auto_candidates(
    project: Project,
    run: dict,
    limit: int = 20,
    receptor_pdb: Path | None = None,
) -> list[dict]:
    """Per-ligand docking + property table for the manual confirmation step.

    Candidates are ranked by the multi-criteria decision score (affinity,
    pose validity, ligand efficiency, QED, synthetic accessibility) instead of
    affinity alone.  A medicinal-chemistry tier is applied first so that up to
    20 molecules with genuine drug-like properties reach the confirmation list;
    only when too few pass are the remaining best-scoring molecules appended.
    PoseBusters is run once for the top 15 poses so the score can use a real
    physical-validity term.
    """
    from rdkit import Chem

    rows = [
        item for item in run.get("results", [])
        if item.get("status") == "ok" and item.get("affinity") is not None
    ]
    rows.sort(key=lambda item: float(item["affinity"]))
    with STATE.lock:
        index = {item["id"]: item for item in project.data.get("ligands", [])}

    pose_rates: dict[str, float] = {}
    if receptor_pdb:
        entries: list[tuple[str, object]] = []
        for item in rows[:15]:
            ligand = index.get(item.get("ligand_id"))
            pose = _abs(item.get("pose"))
            if not ligand or not pose or not pose.exists():
                continue
            try:
                mol, _ = analysis._load_pose_mol(pose, 1)
            except Exception:
                continue
            entries.append((ligand["id"], mol))
        if entries:
            try:
                checked = posecheck.run_batch(Path(receptor_pdb), entries)
                for row in checked.get("rows") or []:
                    total = max(1, int(row.get("total") or 0))
                    pose_rates[row.get("name")] = int(row.get("passed") or 0) / total
            except Exception as exc:
                logger.warning("候选构象物理校验失败：%s", exc)

    payload = []
    for item in rows[: max(60, limit * 3)]:
        ligand = index.get(item.get("ligand_id"))
        if not ligand:
            continue
        meta = ligand.get("meta") or {}
        verdict = ligand.get("filter") or {}
        props = verdict.get("props") or {}
        smiles = ligand.get("smiles") or meta.get("smiles") or ""
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        qed = props.get("qed")
        efficiency = None
        pains_count = len(verdict.get("pains") or [])
        alerts_count = len(verdict.get("alerts") or [])
        alert_count = pains_count + alerts_count
        warning_count = len(verdict.get("warnings") or [])
        if mol is not None:
            try:
                profile = admet.profile_mol(
                    mol, affinity=float(item["affinity"])
                )
                qed = (profile.get("druglike") or {}).get("qed", qed)
                efficiency = (profile.get("efficiency") or {}).get("le")
            except Exception:
                pass
        sa = decision.sa_score(mol)
        heavy = meta.get("heavy") or props.get("heavy")
        scored = decision.score_candidate(
            float(item["affinity"]),
            heavy=heavy,
            qed=qed,
            sa=sa,
            pose_pass_rate=pose_rates.get(ligand["id"]),
            alerts=alert_count,
            warnings=warning_count,
        )
        payload.append(
            {
                "ligand_id": ligand["id"],
                "name": ligand.get("name") or item.get("ligand_name") or "",
                "affinity": item.get("affinity"),
                "smiles": smiles,
                "formula": meta.get("formula") or props.get("formula") or "",
                "mw": meta.get("mw") if meta.get("mw") is not None else props.get("mw"),
                "logp": meta.get("logp") if meta.get("logp") is not None else props.get("logp"),
                "tpsa": meta.get("tpsa") if meta.get("tpsa") is not None else props.get("tpsa"),
                "hbd": meta.get("hbd") if meta.get("hbd") is not None else props.get("hbd"),
                "hba": meta.get("hba") if meta.get("hba") is not None else props.get("hba"),
                "rotb": meta.get("rotb") if meta.get("rotb") is not None else props.get("rotb"),
                "heavy": heavy,
                "rings": props.get("rings"),
                "max_ring": props.get("max_ring"),
                "qed": qed,
                "le": efficiency,
                "sa": sa,
                "pose_pass": pose_rates.get(ligand["id"]),
                "decision_score": scored["score"],
                "decision_components": scored["components"],
                "decision_confidence": scored["confidence"],
                "decision_notes": scored["notes"],
                "warnings": verdict.get("warnings") or [],
                "pains": pains_count,
                "alerts": alerts_count,
                "svg": f"/api/ligands/{ligand['id']}/svg",
            }
        )
    payload.sort(
        key=lambda item: (
            -float(item.get("decision_score") or 0.0),
            float(item.get("affinity") or 0.0),
        )
    )

    def preferred(item: dict) -> bool:
        """Strict medchem gate: keep only molecules that look developable."""
        qed_value = item.get("qed")
        sa_value = item.get("sa")
        le_value = item.get("le")
        heavy = item.get("heavy")
        pose = item.get("pose_pass")
        return (
            qed_value is not None and float(qed_value) >= 0.45
            and (sa_value is None or float(sa_value) <= 4.5)
            and le_value is not None and float(le_value) >= 0.25
            and heavy is not None and 12 <= int(heavy) <= 45
            and len(item.get("warnings") or []) <= 1
            and int(item.get("pains") or 0) == 0
            and int(item.get("alerts") or 0) == 0
            and (pose is None or float(pose) >= 0.5)
        )

    for item in payload:
        item["tier"] = "preferred" if preferred(item) else "backup"
    strict = [item for item in payload if item["tier"] == "preferred"]
    relaxed = [item for item in payload if item["tier"] != "preferred"]
    chosen = strict[:limit]
    if len(chosen) < limit:
        chosen += relaxed[: limit - len(chosen)]
    return chosen


def _auto_finish(project: Project, job, ligand_id: str = "") -> None:
    """Optimisation → design → reports for the manually confirmed candidate."""
    with STATE.lock:
        pending = dict(project.data.get("auto_pending") or {})
    if not pending:
        raise RuntimeError("没有等待确认的自动分析任务，请重新运行自动模式")
    run = project.get_run(pending.get("run_id") or "")
    if not run:
        raise RuntimeError("精细对接记录不存在")
    pairs = [
        (index, item)
        for index, item in enumerate(run.get("results", []))
        if item.get("status") == "ok" and item.get("affinity") is not None
    ]
    if not pairs:
        raise RuntimeError("精细对接没有成功结果")
    if ligand_id:
        chosen = [pair for pair in pairs if pair[1].get("ligand_id") == ligand_id]
        if not chosen:
            raise RuntimeError("所选分子不在精细对接结果中")
        best_index, best = chosen[0]
    else:
        best_index, best = min(pairs, key=lambda pair: float(pair[1]["affinity"]))
    job.log(
        f"[自动] 手动确认 {best.get('ligand_name')}"
        f"（{best.get('affinity')} kcal/mol），继续优化、设计与报告"
    )
    receptor_id = str(pending.get("receptor_id") or "")
    langs = list(pending.get("langs") or ["zh"])
    coarse_run_id = str(pending.get("coarse_run_id") or "")
    online_summary = pending.get("online") or {}

    optimization_id = ""
    job.set_progress(0.04, "基于所选分子进行药物优化 ...")
    try:
        optimization.run_optimization(
            project,
            copy.deepcopy(run),
            best_index,
            {
                "max_candidates": 8,
                "exhaustiveness": 4,
                "num_modes": 3,
                "engine": "gpu" if VINA_GPU_EXE.is_file() else "cpu",
                "parallel_jobs": 2,
                "seed": 42,
            },
            _JobScope(job, 0.04, 0.26),
        )
        optimization_id = (project.data.get("optimizations") or [{}])[0].get("run_id", "")
        job.log(f"[自动] 药物优化完成：{optimization_id}")
    except Exception as exc:
        job.log(f"[自动] 药物优化失败，继续生成报告：{exc}")

    design_id = ""
    job.set_progress(0.28, "基于所选分子进行分子设计 ...")
    try:
        design.run_design(
            project,
            receptor_id,
            {
                "max_designs": 8,
                "mode": "auto",
                "reference_smiles": _result_smiles(best),
                "exhaustiveness": 2,
                "active": True,
                "engine": "gpu" if VINA_GPU_EXE.is_file() else "cpu",
                "parallel_jobs": 2,
                "seed": 42,
            },
            _JobScope(job, 0.28, 0.44),
        )
        design_id = (project.data.get("designs") or [{}])[0].get("run_id", "")
        job.log(f"[自动] 分子设计完成：{design_id}")
    except Exception as exc:
        job.log(f"[自动] 分子设计失败，继续生成报告：{exc}")

    fishing_id = ""
    netpharm_id = ""
    fishing_data: dict = {}
    job.set_progress(0.46, "靶点垂钓：检索相似分子与实验靶点 ...")
    try:
        fishing_data = fishing.run_fishing(
            project,
            _result_smiles(best),
            best.get("ligand_name") or "候选分子",
            {
                "similarity": 40,
                "molecules": 120,
                "targets": 220,
                "depth": "deep",
                "swiss_organisms": [
                    "Homo_sapiens",
                    "Mus_musculus",
                    "Rattus_norvegicus",
                ],
            },
            _JobScope(job, 0.46, 0.58),
        )
        fishing_id = (fishing_data or {}).get("id", "")
        job.log(
            f"[自动] 靶点垂钓完成：{len((fishing_data or {}).get('targets') or [])} 个靶点"
            f"（{fishing_id}）"
        )
    except Exception as exc:
        job.log(f"[自动] 靶点垂钓失败，报告将不含该节：{exc}")

    if fishing_id:
        job.set_progress(0.60, "网络药理学：构建 PPI 与多库富集 ...")
        try:
            genes = _fishing_human_genes(fishing_data)
            if len(genes) < 2:
                job.log("[自动] 垂钓到的基因不足 2 个，跳过网络药理学")
            else:
                net = netpharm.run_netpharm(
                    project,
                    genes,
                    (fishing_data or {}).get("compound")
                    or {"name": best.get("ligand_name") or "", "smiles": _result_smiles(best)},
                    {"genes": 80, "string_score": 400, "expand": 20},
                    _JobScope(job, 0.60, 0.70),
                )
                netpharm_id = (net or {}).get("id", "")
                job.log(
                    f"[自动] 网络药理学完成：{len(genes)} 个基因，"
                    f"{((net or {}).get('stats') or {}).get('edges', 0)} 条互作边（{netpharm_id}）"
                )
        except Exception as exc:
            job.log(f"[自动] 网络药理学失败，报告将不含该节：{exc}")

    reports: dict[str, str] = {}
    total = max(1, len(langs))
    for index, one in enumerate(langs):
        job.set_progress(0.72 + 0.26 * index / total, f"生成完整报告（{one}）...")
        response = full_project_report(
            run_id=run["id"],
            optimization_id=optimization_id,
            design_id=design_id,
            fishing_id=fishing_id,
            netpharm_id=netpharm_id,
            lang=one,
        )
        path = getattr(response, "path", None)
        if not path:
            path = project.root / "exports" / "full_report" / (
                f"{run['id']}_complete_report_en.pdf"
                if one == "en"
                else f"{run['id']}_complete_report.pdf"
            )
        reports[one] = str(path)
        job.log(f"[自动] 报告已写入：{path}")
    with STATE.lock:
        project.data.pop("auto_pending", None)
        project.save()
    job.set_progress(1.0, "全自动分析完成")
    job.result = {
        "run_id": run["id"],
        "coarse_run_id": coarse_run_id,
        "optimization_id": optimization_id,
        "design_id": design_id,
        "fishing_id": fishing_id,
        "netpharm_id": netpharm_id,
        "screened": pending.get("screened"),
        "fine_count": pending.get("fine_count"),
        "best_ligand": best.get("ligand_name"),
        "best_affinity": best.get("affinity"),
        "project": project.data["name"],
        "project_path": str(project.root),
        "report_dir": str(project.root / "exports" / "full_report"),
        "reports": reports,
        "report": reports.get(langs[0]),
        "online": online_summary,
        "confirmed": True,
    }


@app.post("/api/auto/continue")
def auto_continue(payload: dict):
    """Continue the paused automatic pipeline with the user-selected ligand."""
    _ensure_no_active_job()
    ligand_id = str(payload.get("ligand_id") or "").strip()
    with STATE.lock:
        project = STATE.project
        pending = project.data.get("auto_pending") or {}
        name = project.data.get("name", "")
    if not pending:
        raise HTTPException(400, "没有等待确认的自动分析任务")
    job = STATE.jobs.create(
        "auto",
        f"自动分析（继续）· {name}",
        lambda job: _auto_finish(project, job, ligand_id=ligand_id),
    )
    return {"job_id": job.id}


@app.post("/api/auto")
async def auto_pipeline(
    receptor: UploadFile | None = File(None),
    ligands: list[UploadFile] = File(...),
    project_name: str = Form(""),
    report_lang: str = Form("both"),
    receptor_query: str = Form(""),
    receptor_pdb_id: str = Form(""),
    receptor_sequence: str = Form(""),
    species: str = Form("human"),
    online_intel: bool = Form(True),
):
    """One receptor (file / gene / sequence) + ligands → automatic full report."""
    _ensure_no_active_job()
    receptor_payload: dict
    if receptor is not None and receptor.filename:
        receptor_name = Path(receptor.filename or "receptor.pdb").name
        receptor_suffix = Path(receptor_name).suffix.lower()
        if receptor_suffix not in (".pdb", ".ent", ".txt", ".pdbqt"):
            raise HTTPException(400, "受体只支持 .pdb / .ent / .pdbqt")
        receptor_bytes = await receptor.read()
        if len(receptor_bytes) > 120 * 1024 * 1024:
            raise HTTPException(400, f"{receptor_name} 超过 120 MB")
        receptor_text = receptor_bytes.decode("utf-8", errors="replace")
        if receptor_suffix != ".pdbqt" and "ATOM" not in receptor_text:
            raise HTTPException(400, f"{receptor_name} 不是有效的 PDB 文件")
        receptor_payload = {
            "mode": "file",
            "filename": receptor_name,
            "suffix": receptor_suffix,
            "text": receptor_text,
        }
    elif str(receptor_pdb_id or "").strip():
        pdb_id = str(receptor_pdb_id).strip().upper()
        if not re.fullmatch(r"[0-9][A-Z0-9]{3}", pdb_id):
            raise HTTPException(
                400, "PDB ID 必须是 4 位字母数字（如 6LU7），且以数字开头"
            )
        receptor_payload = {
            "mode": "pdb",
            "pdb_id": pdb_id,
        }
    elif str(receptor_sequence or "").strip():
        sequence = str(receptor_sequence).strip()
        if len(online._clean_sequence(sequence)) < 20:
            raise HTTPException(400, "氨基酸序列至少需要 20 个残基")
        receptor_payload = {"mode": "sequence", "sequence": sequence}
    elif str(receptor_query or "").strip():
        receptor_payload = {
            "mode": "gene",
            "query": str(receptor_query).strip(),
            "species": str(species or "human"),
        }
    else:
        raise HTTPException(400, "请上传受体文件，或输入 PDB ID / 基因或 UniProt 登录号 / 氨基酸序列")

    ligand_payloads = []
    for upload in ligands:
        filename = Path(upload.filename or "ligand.sdf").name
        suffix = Path(filename).suffix.lower()
        if suffix not in (".sdf", ".sd", ".mol", ".mol2", ".smi", ".smiles", ".pdb", ".pdbqt"):
            raise HTTPException(400, f"不支持的配体格式：{suffix or filename}")
        data = await upload.read()
        if len(data) > 120 * 1024 * 1024:
            raise HTTPException(400, f"{filename} 超过 120 MB")
        ligand_payloads.append((filename, suffix, data))
    if not ligand_payloads:
        raise HTTPException(400, "请至少上传一个配体文件")

    choice = str(report_lang).lower()
    langs = ["zh", "en"] if choice == "both" else (["en"] if choice.startswith("en") else ["zh"])
    name = project_name.strip() or f"Auto_{datetime.now():%Y%m%d_%H%M}"
    with STATE.lock:
        project = Project.create(name)
        STATE.project = project
        project.data["params"] = _auto_fine_params()
        project.save()

    def task(job):
        _run_auto_pipeline(
            project,
            receptor_payload,
            ligand_payloads,
            langs,
            job,
            online_intel=bool(online_intel),
        )

    job = STATE.jobs.create("auto", f"全自动分析 · {project.data['name']}", task)
    return {
        "job_id": job.id,
        "project": project.data["name"],
        "project_path": str(project.root),
    }


@app.get("/api/auto/report")
def auto_report(run_id: str, lang: str = "zh"):
    """Serve the report that the automatic pipeline already wrote."""
    lang = "en" if str(lang).lower().startswith("en") else "zh"
    with STATE.lock:
        project = STATE.project
        if not project.get_run(run_id):
            raise HTTPException(404, "对接记录不存在")
        path = project.root / "exports" / "full_report" / (
            f"{run_id}_complete_report_en.pdf"
            if lang == "en"
            else f"{run_id}_complete_report.pdf"
        )
    if not path.is_file():
        raise HTTPException(404, "报告尚未生成")
    return FileResponse(path, filename=path.name, media_type="application/pdf")


@app.post("/api/runs/{run_id}/resume")
def resume_run(run_id: str):
    """Re-run failed/interrupted pairs and merge them with earlier successes."""
    _ensure_no_active_job()
    with STATE.lock:
        project = STATE.project
        source = project.get_run(run_id)
        if not source:
            raise HTTPException(404, "对接记录不存在")
        old_results = list(source.get("results") or [])
        failed = [
            item for item in old_results
            if item.get("status") != "ok"
            and item.get("receptor_id")
            and item.get("ligand_id")
        ]
        if not failed:
            raise HTTPException(400, "该记录没有需要续跑的失败或中断组合")
        receptor_ids = list(dict.fromkeys(item["receptor_id"] for item in failed))
        ligand_ids = list(dict.fromkeys(item["ligand_id"] for item in failed))
        params = dict(source.get("params") or {})
        params.pop("resume_from", None)
        run = project.new_run(params)
        run["resumed_from"] = run_id
        project.save()
        carried = [dict(item) for item in old_results if item.get("status") == "ok"]

    def task(job):
        try:
            run_docking(project, run, receptor_ids, ligand_ids, params, job)
            with STATE.lock:
                merged: dict[tuple[str, str], dict] = {}
                for item in carried:
                    merged[
                        (str(item.get("receptor_id")), str(item.get("ligand_id")))
                    ] = item
                for item in run.get("results") or []:
                    key = (str(item.get("receptor_id")), str(item.get("ligand_id")))
                    if item.get("status") == "ok" or key not in merged:
                        merged[key] = item
                final = list(merged.values())
                for index, item in enumerate(final, start=1):
                    item["task_index"] = index
                run["results"] = final
                run["status"] = "cancelled" if job.cancel_event.is_set() else "done"
                run["finished"] = True
                project.save()
        except Exception:
            with STATE.lock:
                run["status"] = "error"
                run["finished"] = True
                project.save()
            raise

    job = STATE.jobs.create(
        "dock", f"续跑 · {len(receptor_ids)}×{len(ligand_ids)}", task
    )
    return {"job_id": job.id, "run_id": run["id"]}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str, since: int = 0):
    job = STATE.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    return job.snapshot(since=since)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    job = STATE.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    job.cancel()
    return {"ok": True}


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------
def _get_result(run_id: str, index: int):
    with STATE.lock:
        run = STATE.project.get_run(run_id)
        if not run:
            raise HTTPException(404, "对接记录不存在")
        results = run.get("results", [])
        if index < 0 or index >= len(results):
            raise HTTPException(404, "结果不存在")
        return run, results[index]


@app.get("/api/runs/{run_id}/result/{index}/pose", response_class=PlainTextResponse)
def result_pose(run_id: str, index: int, mode: int = 1):
    _, result = _get_result(run_id, index)
    pose = _abs(result.get("pose"))
    if not pose or not pose.exists():
        raise HTTPException(404, "构象文件不存在")
    text = pose.read_text(encoding="utf-8", errors="replace")
    return PlainTextResponse(pdbio.pose_pdbqt(text, mode))


@app.get("/api/runs/{run_id}/result/{index}/log", response_class=PlainTextResponse)
def result_log(run_id: str, index: int):
    _, result = _get_result(run_id, index)
    path = _abs(result.get("log"))
    if not path or not path.exists():
        return PlainTextResponse("（没有日志）")
    return PlainTextResponse(path.read_text(encoding="utf-8", errors="replace"))


@app.get("/api/runs/{run_id}/result/{index}/receptor.pdb")
def result_receptor_structure(run_id: str, index: int):
    _, result = _get_result(run_id, index)
    with STATE.lock:
        path = _result_receptor_pdb(result)
    if not path or not path.exists():
        raise HTTPException(404, "缺少受体结构")
    return FileResponse(path, filename=f"{result.get('receptor_name', 'receptor')}.pdb", media_type="chemical/x-pdb")


@app.post("/api/runs/{run_id}/result/{index}/analyze")
def result_analyze(run_id: str, index: int, payload: dict):
    _, result = _get_result(run_id, index)
    mode = int(payload.get("mode", 1))
    pose = _abs(result.get("pose"))
    with STATE.lock:
        receptor_pdb = _result_receptor_pdb(result)
    if not pose or not pose.exists() or not receptor_pdb:
        raise HTTPException(404, "缺少受体或构象文件")
    try:
        data = analysis.analyse_interactions(
            receptor_pdb.read_text(encoding="utf-8", errors="replace"),
            pose,
            mode=mode,
            receptor_pdbqt=_result_receptor_pdbqt(result),
        )
    except Exception as exc:
        raise HTTPException(400, f"作用力分析失败：{exc}") from exc
    data["ligand"] = result.get("ligand_name")
    data["receptor"] = result.get("receptor_name")
    return data


@app.post("/api/runs/{run_id}/result/{index}/posebusters")
def result_posebusters(run_id: str, index: int, payload: dict | None = None):
    """PoseBusters physical-validity checks for one docked binding mode."""
    body = payload or {}
    mode = int(body.get("mode", 1))
    _, result = _get_result(run_id, index)
    pose = _abs(result.get("pose"))
    with STATE.lock:
        receptor_pdb = _result_receptor_pdb(result)
    if result.get("status") != "ok" or not pose or not pose.exists() or not receptor_pdb:
        raise HTTPException(404, "缺少受体或构象文件")
    try:
        mol, _ = analysis._load_pose_mol(pose, mode)
    except Exception as exc:
        raise HTTPException(400, f"无法重建该构象的分子：{exc}") from exc
    data = posecheck.run(receptor_pdb, mol)
    data["ligand"] = result.get("ligand_name")
    data["mode"] = mode
    if data.get("available") and not data.get("error"):
        with STATE.lock:
            result["posebusters"] = data
            STATE.project.save()
    return data


@app.post("/api/runs/{run_id}/posebusters")
def run_posebusters_batch(run_id: str, payload: dict | None = None):
    """Check every successful pose of a run (protein parsed once)."""
    body = payload or {}
    limit = max(1, min(int(body.get("limit", 50)), 200))
    with STATE.lock:
        run = STATE.project.get_run(run_id)
        if not run:
            raise HTTPException(404, "对接记录不存在")
        entries: list[tuple[int, str, object, dict]] = []
        for position, item in enumerate(run.get("results", [])):
            if len(entries) >= limit:
                break
            if item.get("status") != "ok":
                continue
            pose = _abs(item.get("pose"))
            if not pose or not pose.exists():
                continue
            try:
                mol, _ = analysis._load_pose_mol(pose, 1)
            except Exception:
                continue
            entries.append(
                (position, str(item.get("ligand_name") or f"#{position + 1}"), mol, item)
            )
        receptor_pdb = _result_receptor_pdb(entries[0][3]) if entries else None
    if not entries or receptor_pdb is None:
        raise HTTPException(404, "没有可校验的成功构象")
    data = posecheck.run_batch(receptor_pdb, [(name, mol) for _, name, mol, _ in entries])
    if not data.get("available") and data.get("error"):
        raise HTTPException(400, str(data.get("error")))
    for (position, name, _mol, item), row in zip(entries, data.get("rows") or []):
        item["posebusters"] = {
            "available": bool(row.get("available", True)),
            "passed": row.get("passed", 0),
            "total": row.get("total", len(posecheck.CHECKS)),
            "all_passed": row.get("all_passed", False),
            "failed": row.get("failed", []),
            "failed_labels": row.get("failed_labels", []),
            "config": "dock",
            "module_version": data.get("module_version", ""),
        }
        row["index"] = position
        row["ligand"] = name
        row["affinity"] = item.get("affinity")
    with STATE.lock:
        STATE.project.save()
    data["run_id"] = run_id
    return data


@app.get("/api/runs/{run_id}/manifest")
def run_manifest(run_id: str, verify: bool = False):
    """Return (and optionally verify) the reproducibility manifest of a run."""
    with STATE.lock:
        project = STATE.project
        run = project.get_run(run_id)
        params = dict(project.data.get("params") or {})
    if not run:
        raise HTTPException(404, "对接记录不存在")
    path = project.root / "exports" / "full_report" / f"{run_id}_manifest.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = runmanifest.build(project, run, params=params)
    else:
        data = runmanifest.build(project, run, params=params)
        runmanifest.write(path, data)
    if verify:
        data["verification"] = runmanifest.verify(project.root, data)
    return data


@app.post("/api/runs/{run_id}/result/{index}/rmsd")
def result_rmsd(run_id: str, index: int, payload: dict):
    _, result = _get_result(run_id, index)
    mode = int(payload.get("mode", 1))
    ref = payload.get("reference") or {}
    pose = _abs(result.get("pose"))
    with STATE.lock:
        receptor = _find_result_receptor(result)
        if not receptor:
            raise HTTPException(404, "受体已被删除")
        source = _abs(receptor.get("source"))
        hetero = receptor.get("hetero") or []
    if not source or not source.exists():
        # A re-prepared receptor may have dropped the original source; fall back
        # to the cleaned structure so the user gets a clear message instead of 404.
        source = _result_receptor_pdb(result)
    if not source or not source.exists():
        raise HTTPException(404, "缺少受体源文件")
    if not ref:
        if not hetero:
            raise HTTPException(400, "受体中没有共晶配体，无法计算 RMSD")
        # Receptors often carry several hetero records (ions, cofactors, more
        # than one ligand).  Try them in turn instead of trusting the first
        # entry, which may be an unrelated 48-atom molecule.
        last: dict | None = None
        receptor_text = source.read_text(encoding="utf-8", errors="replace")
        for candidate in hetero:
            try:
                data = analysis.rmsd_to_reference(
                    receptor_text, candidate, pose, mode=mode
                )
            except Exception as exc:  # noqa: BLE001
                last = {"available": False, "error": str(exc)}
                continue
            if data.get("available"):
                return data
            last = data
        return last or {
            "available": False,
            "error": "所有共晶配体均无法与该对接构象匹配",
        }
    try:
        data = analysis.rmsd_to_reference(
            source.read_text(encoding="utf-8", errors="replace"), ref, pose, mode=mode
        )
    except Exception as exc:
        raise HTTPException(400, f"RMSD 计算失败：{exc}") from exc
    return data


@app.get("/api/runs/{run_id}/result/{index}/complex")
def result_complex(run_id: str, index: int, mode: int = 1):
    run, result = _get_result(run_id, index)
    pose = _abs(result.get("pose"))
    with STATE.lock:
        receptor_pdb = _result_receptor_pdb(result)
    if not pose or not pose.exists() or not receptor_pdb:
        raise HTTPException(404, "缺少受体或构象文件")
    pose_text = pdbio.pose_pdbqt(pose.read_text(encoding="utf-8", errors="replace"), mode)
    complex_text = pdbio.complex_pdb(
        receptor_pdb.read_text(encoding="utf-8", errors="replace"),
        pdbio.pdbqt_to_pdb(pose_text),
        name=f"{result['ligand_name']} / {result['receptor_name']}",
    )
    exports = STATE.project.root / "exports"
    exports.mkdir(exist_ok=True)
    filename = safe_name(f"{run['id']}_{index + 1}_{result['ligand_name']}_mode{mode}", "complex")
    path = exports / f"{filename}.pdb"
    path.write_text(complex_text, encoding="utf-8")
    return FileResponse(path, filename=f"{filename}.pdb", media_type="chemical/x-pdb")


@app.get("/api/runs/{run_id}/export.csv")
def export_csv(run_id: str):
    with STATE.lock:
        run = STATE.project.get_run(run_id)
        if not run:
            raise HTTPException(404, "对接记录不存在")
        results = list(run.get("results", []))
    buffer = io.StringIO()
    buffer.write("\ufeff")
    writer = csv.writer(buffer)
    writer.writerow(
        ["排名", "配体", "受体", "Vina 打分 (kcal/mol)", "GPU 原始打分", "Vinardo 打分 (kcal/mol)", "共识排名", "构象数", "RMSD l.b.", "RMSD u.b.", "状态", "引擎", "随机种子", "盒子中心", "盒子尺寸"]
    )
    ok = sorted(
        [r for r in results if r.get("status") == "ok" and r.get("affinity") is not None],
        key=lambda r: r["affinity"],
    )
    for rank, item in enumerate(ok, start=1):
        best = item["modes"][0] if item.get("modes") else {}
        box = item.get("box") or {}
        writer.writerow(
            [
                rank,
                item.get("ligand_name"),
                item.get("receptor_name"),
                item.get("affinity"),
                item.get("gpu_affinity", ""),
                item.get("vinardo", ""),
                item.get("consensus_rank", ""),
                len(item.get("modes", [])),
                best.get("rmsd_lb", ""),
                best.get("rmsd_ub", ""),
                item.get("status"),
                item.get("engine", "cpu"),
                item.get("seed", ""),
                ";".join(f"{v:.2f}" for v in box.get("center", [])),
                ";".join(f"{v:.2f}" for v in box.get("size", [])),
            ]
        )
    for item in results:
        if item.get("status") != "ok":
            writer.writerow(["", item.get("ligand_name"), item.get("receptor_name"), "", "", "", "", 0, "", "", item.get("status"), item.get("engine", ""), item.get("seed", ""), "", ""])
    return Response(
        content=buffer.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{run_id}_scores.csv"'},
    )


@app.get("/api/runs/{run_id}/bulk/summary")
def bulk_run_summary(run_id: str):
    with STATE.lock:
        run = STATE.project.get_run(run_id)
        if not run:
            raise HTTPException(404, "对接记录不存在")
        results = [dict(item) for item in run.get("results", [])]
        ligand_meta = {
            item["id"]: dict(item.get("meta") or {})
            for item in STATE.project.data["ligands"]
        }
    ok = [r for r in results if r.get("status") == "ok" and r.get("affinity") is not None]
    scores = sorted(float(item["affinity"]) for item in ok)
    if not scores:
        return {
            "run_id": run_id, "total": len(results), "ok": 0, "failed": len(results),
            "scores": [], "top": [],
        }
    top = sorted(ok, key=lambda item: float(item["affinity"]))[:10]
    median = (
        scores[len(scores) // 2]
        if len(scores) % 2
        else (scores[len(scores) // 2 - 1] + scores[len(scores) // 2]) / 2
    )
    return {
        "run_id": run_id,
        "total": len(results),
        "ok": len(ok),
        "failed": len(results) - len(ok),
        "best": scores[0],
        "worst": scores[-1],
        "median": round(median, 2),
        "mean": round(float(np.mean(scores)), 2),
        "below_8": sum(1 for value in scores if value <= -8.0),
        "below_9": sum(1 for value in scores if value <= -9.0),
        "top": [
            {
                "ligand": item.get("ligand_name") or item.get("ligand_id"),
                "affinity": float(item["affinity"]),
                "mw": (ligand_meta.get(str(item.get("ligand_id") or "")) or {}).get("mw"),
            }
            for item in top
        ],
    }


@app.get("/api/runs/{run_id}/top")
def bulk_run_top(run_id: str, limit: int = 50):
    """Top-scored ligands of a bulk run, ready for fine preparation/docking."""
    limit = max(1, min(int(limit or 50), 500))
    with STATE.lock:
        project = STATE.project
        run = project.get_run(run_id)
        if not run:
            raise HTTPException(404, "对接记录不存在")
        index = {item["id"]: item for item in project.data["ligands"]}
        ok = [
            item for item in run.get("results", [])
            if item.get("status") == "ok" and item.get("affinity") is not None
        ]
        ok.sort(key=lambda item: float(item["affinity"]))
        items = []
        for rank, result in enumerate(ok[:limit], start=1):
            ligand = index.get(result.get("ligand_id")) or {}
            meta = ligand.get("meta") or {}
            heavy = float(meta.get("heavy") or 0)
            pdbqt = ligand.get("pdbqt")
            items.append(
                {
                    "rank": rank,
                    "ligand_id": result.get("ligand_id"),
                    "ligand_name": result.get("ligand_name") or ligand.get("name") or "",
                    "affinity": float(result["affinity"]),
                    "mw": meta.get("mw"),
                    "logp": meta.get("logp"),
                    "heavy": meta.get("heavy"),
                    "le": round(-float(result["affinity"]) / heavy, 3) if heavy > 0 else None,
                    "status": ligand.get("status", ""),
                    "has_pdbqt": bool(pdbqt and (project.root / pdbqt).exists()),
                }
            )
    return {"run_id": run_id, "limit": limit, "total": len(ok), "items": items}


@app.get("/api/runs/{run_id}/bulk.png")
def bulk_run_png(run_id: str):
    with STATE.lock:
        run = STATE.project.get_run(run_id)
        if not run:
            raise HTTPException(404, "对接记录不存在")
        root = STATE.project.root
        results = [dict(item) for item in run.get("results", [])]
        ligand_meta = {
            item["id"]: dict(item.get("meta") or {})
            for item in STATE.project.data["ligands"]
        }
    outdir = root / "exports" / "bulk"
    path = outdir / f"{run_id}_screening.png"
    figures.bulk_screen_png(
        {"results": results, "params": dict(run.get("params") or {})},
        ligand_meta,
        path,
    )
    return FileResponse(path, media_type="image/png", filename=f"{run_id}_screening.png")


@app.get("/api/runs/{run_id}/bulk/figures.pdf")
def bulk_run_figures(run_id: str, lang: str = "zh"):
    lang = "en" if str(lang).lower().startswith("en") else "zh"
    i18n.set_lang(lang)
    with STATE.lock:
        run = STATE.project.get_run(run_id)
        if not run:
            raise HTTPException(404, "对接记录不存在")
        root = STATE.project.root
        snapshot = {
            "results": [dict(item) for item in run.get("results", [])],
            "params": dict(run.get("params") or {}),
        }
        ligand_meta = {
            item["id"]: dict(item.get("meta") or {})
            for item in STATE.project.data["ligands"]
        }
    outdir = root / "exports" / "bulk"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{run_id}_statistics_{lang}.pdf"
    figures.bulk_screening_report_pdf(snapshot, ligand_meta, path)
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=f"{run_id}_screening_statistics.pdf",
    )


@app.get("/api/runs/{run_id}/consensus.pdf")
def run_consensus_pdf(run_id: str, lang: str = "zh"):
    lang = "en" if str(lang).lower().startswith("en") else "zh"
    i18n.set_lang(lang)
    with STATE.lock:
        run = STATE.project.get_run(run_id)
        if not run:
            raise HTTPException(404, "对接记录不存在")
        root = STATE.project.root
        snapshot = copy.deepcopy(run)
    if len((snapshot.get("consensus") or {}).get("engines") or []) < 2:
        raise HTTPException(400, "该记录没有多引擎共识结果")
    outdir = root / "exports" / "consensus"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{run_id}_consensus_{lang}.pdf"
    figures.multi_engine_report_pdf(snapshot, path)
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=f"{run_id}_multi_engine_consensus.pdf",
    )


@app.get("/api/runs/{run_id}/figures")
def export_publication_figures(run_id: str):
    with STATE.lock:
        run = STATE.project.get_run(run_id)
        if not run:
            raise HTTPException(404, "对接记录不存在")
        results = [dict(item) for item in run.get("results", [])]
        root = STATE.project.root
    ok = [r for r in results if r.get("status") == "ok" and r.get("pose")]
    if not ok:
        raise HTTPException(400, "没有可生成图表的结果")
    ok.sort(key=lambda r: r.get("affinity", 999))
    best = ok[0]

    analyses = {}
    for item in ok[:40]:
        receptor_pdb = _result_receptor_pdb(item)
        pose = _abs(item.get("pose"))
        if not receptor_pdb or not pose or not pose.exists():
            continue
        try:
            analyses[id(item)] = analysis.analyse_interactions(
                receptor_pdb.read_text(encoding="utf-8", errors="replace"),
                pose,
                mode=1,
                receptor_pdbqt=_result_receptor_pdbqt(item),
            )
        except Exception:
            continue

    receptor_pdb = _result_receptor_pdb(best)
    pose_pdbqt = _abs(best.get("pose"))
    if not receptor_pdb or not pose_pdbqt:
        raise HTTPException(400, "缺少受体结构或构象文件")

    outdir = root / "exports" / f"{run_id}_figures"
    if outdir.exists():
        shutil.rmtree(outdir, ignore_errors=True)
    try:
        zip_path = figures.build_publication_bundle(
            run, ok, analyses, receptor_pdb, pose_pdbqt, best, outdir
        )
    except Exception as exc:
        raise HTTPException(400, f"论文图生成失败：{exc}") from exc
    return FileResponse(
        zip_path,
        filename=f"{run_id}_publication_figures.zip",
        media_type="application/zip",
    )


@app.get("/api/runs/{run_id}/result/{index}/figures.pdf")
def result_vector_figures(run_id: str, index: int, mode: int = 1):
    _, result = _get_result(run_id, index)
    pose = _abs(result.get("pose"))
    with STATE.lock:
        receptor_pdb = _result_receptor_pdb(result)
    if not pose or not pose.exists() or not receptor_pdb:
        raise HTTPException(404, "缺少受体或构象文件")
    try:
        interaction = analysis.analyse_interactions(
            receptor_pdb.read_text(encoding="utf-8", errors="replace"),
            pose,
            mode=mode,
            receptor_pdbqt=_result_receptor_pdbqt(result),
        )
    except Exception as exc:
        raise HTTPException(400, f"作用力分析失败：{exc}") from exc
    outdir = STATE.project.root / "exports" / "vector"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{run_id}_{index + 1}_mode{mode}.pdf"
    try:
        figures.result_figure_pdf(
            pose,
            result.get("ligand_name", "ligand"),
            result.get("receptor_name", "receptor"),
            interaction,
            path,
            mode=mode,
        )
    except Exception as exc:
        raise HTTPException(400, f"矢量图生成失败：{exc}") from exc
    return FileResponse(
        path,
        filename=f"{run_id}_{index + 1}_mode{mode}_vector.pdf",
        media_type="application/pdf",
    )


# ---------------------------------------------------------------------------
# professional add-ons: ADMET, pharmacophore, MM-GBSA decomposition
# ---------------------------------------------------------------------------
def _result_smiles(result: dict) -> str:
    ligand = STATE.project.get_ligand(result.get("ligand_id") or "")
    if ligand:
        smiles = ligand.get("smiles") or (ligand.get("meta") or {}).get("smiles")
        if smiles:
            return str(smiles)
    return ""


def _fishing_human_genes(data: dict | None) -> list[str]:
    """Unique human gene symbols extracted from a target-fishing payload."""
    genes: list[str] = []
    for item in (data or {}).get("targets", []):
        if "homo sapiens" not in str(item.get("organism") or "").lower():
            continue
        for gene in (item.get("genes") or [item.get("gene")]):
            if gene and gene not in genes:
                genes.append(gene)
    return genes


def _molecule_keys(smiles: str) -> set[str]:
    """Canonical InChIKeys for a SMILES field that may hold several molecules.

    Fishing and network pharmacology records can pool multiple query ligands;
    the field then contains one SMILES per line.  Comparing only the first line
    would silently mark every pooled record as a different molecule.
    """
    keys: set[str] = set()
    try:
        from rdkit import Chem

        for line in str(smiles).splitlines():
            line = line.strip()
            if not line:
                continue
            mol = Chem.MolFromSmiles(line)
            if mol is None:
                continue
            key = Chem.MolToInchiKey(mol)
            if key:
                keys.add(key)
    except Exception:
        return keys
    return keys


def _profiling_records(run: dict) -> list[dict]:
    """Successful docking results with SMILES, used by series-level panels."""
    records = []
    for result in run.get("results") or []:
        if result.get("status") != "ok":
            continue
        smiles = _result_smiles(result)
        if not smiles:
            continue
        records.append(
            {
                "name": result.get("ligand_name") or result.get("ligand_id"),
                "smiles": smiles,
                "affinity": result.get("affinity"),
            }
        )
    return records


def _ligand_props(result: dict) -> dict | None:
    """Compact physicochemical block used by the report figures."""
    smiles = _result_smiles(result)
    if not smiles:
        return None
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        profile = admet.profile_mol(
            mol,
            name=str(result.get("ligand_name") or "ligand"),
            affinity=result.get("affinity"),
        )
        data = admet.compact(profile)
        data["heavy"] = profile["physchem"]["heavy"]
        return data
    except Exception:
        return None


def _redocking_validation(project: Project, run: dict | None) -> dict:
    """Symmetry-corrected RMSD between co-crystal ligands and their docked poses."""
    from rdkit import Chem
    from rdkit.Chem import rdMolAlign

    rows: list[dict] = []
    overlay: dict | None = None
    ordered_runs = []
    if run:
        ordered_runs.append(run)
    ordered_runs.extend(
        item for item in project.data.get("runs", [])
        if not run or item.get("id") != run.get("id")
    )
    for candidate_run in ordered_runs:
        for result in candidate_run.get("results", []):
            if result.get("status") != "ok":
                continue
            ligand = project.get_ligand(result.get("ligand_id") or "")
            if not ligand or not ligand.get("extracted_from"):
                continue
            crystal_path = _abs(ligand.get("source"))
            pose_path = _abs(result.get("pose"))
            if not crystal_path or not pose_path:
                continue
            if not crystal_path.exists() or not pose_path.exists():
                continue
            try:
                from rdkit import rdBase

                # Crystal SDFs exported from PDB often carry a 2D flag with
                # non-zero Z coordinates; RDKit logs a harmless warning while
                # parsing them.  Keep the report console clean.
                with rdBase.BlockLogs():
                    supplier = Chem.SDMolSupplier(
                        str(crystal_path), removeHs=False, sanitize=True
                    )
                    crystal = next(
                        (mol for mol in supplier if mol is not None), None
                    )
                if crystal is None:
                    continue
                reference = analysis._strip_hydrogens(crystal)
                probe = analysis._strip_hydrogens(analysis._load_pose_mol(pose_path, 1)[0])
                rmsd, transform, atom_map = rdMolAlign.GetBestAlignmentTransform(
                    probe, reference
                )
                rmsd = float(rmsd)
            except Exception:
                continue
            name = str(result.get("ligand_name") or ligand.get("name") or "ligand")
            if all(existing["name"] != name for existing in rows):
                rows.append(
                    {
                        "name": name,
                        "rmsd": round(rmsd, 2),
                        "passed": rmsd <= 2.0,
                        "atoms": reference.GetNumAtoms(),
                        "score": result.get("affinity"),
                    }
                )
            if overlay is None or rmsd < overlay["rmsd"]:
                reference_conf = reference.GetConformer()
                probe_conf = probe.GetConformer()
                crystal_points = np.array([
                    list(reference_conf.GetAtomPosition(ref_index))
                    for _, ref_index in atom_map
                ])
                pose_points = np.array([
                    list(probe_conf.GetAtomPosition(probe_index))
                    for probe_index, _ in atom_map
                ])
                homogeneous = np.hstack(
                    [pose_points, np.ones((len(pose_points), 1))]
                )
                pose_points = (np.asarray(transform) @ homogeneous.T).T[:, :3]
                center = crystal_points.mean(axis=0)
                _, _, basis = np.linalg.svd(crystal_points - center)
                overlay = {
                    "name": name,
                    "rmsd": round(rmsd, 2),
                    "atoms": len(atom_map),
                    "crystal": ((crystal_points - center) @ basis[:2].T)
                    .round(3).tolist(),
                    "pose": ((pose_points - center) @ basis[:2].T)
                    .round(3).tolist(),
                }
        if rows:
            break
    rows.sort(key=lambda item: item["rmsd"])
    return {"rows": rows, "overlay": overlay}


def _optimization_parent(project: Project, optimization: dict | None) -> dict | None:
    """The molecule that every ΔVina in this optimization run refers to."""
    if not optimization:
        return None
    parent_run = project.get_run(optimization.get("parent_run", ""))
    if not parent_run:
        return None
    results = [
        item for item in parent_run.get("results", [])
        if item.get("status") == "ok"
    ]
    try:
        index = int(optimization.get("parent_index") or 0)
    except (TypeError, ValueError):
        index = 0
    if not 0 <= index < len(results):
        return None
    parent = results[index]
    return {
        "name": parent.get("ligand_name", ""),
        "affinity": parent.get("affinity"),
        "props": _ligand_props(parent) or {},
    }


def _result_context(run_id: str, index: int) -> tuple[dict, Path, Path, str]:
    _, result = _get_result(run_id, index)
    pose = _abs(result.get("pose"))
    with STATE.lock:
        receptor_pdb = _result_receptor_pdb(result)
        smiles = _result_smiles(result)
    if not pose or not pose.exists() or not receptor_pdb:
        raise HTTPException(404, "缺少受体或构象文件")
    if not smiles:
        raise HTTPException(400, "配体缺少 SMILES，无法计算分子性质")
    return result, pose, receptor_pdb, smiles


def _professional_dir() -> Path:
    path = STATE.project.root / "exports" / "professional"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _admet_for_result(result: dict, smiles: str) -> dict:
    return admet.profile_smiles(
        smiles,
        name=str(result.get("ligand_name") or "ligand"),
        affinity=result.get("affinity"),
    )


def _interaction_for_result(result: dict, pose: Path, receptor_pdb: Path, mode: int) -> dict:
    try:
        return analysis.analyse_interactions(
            receptor_pdb.read_text(encoding="utf-8", errors="replace"),
            pose,
            mode=mode,
            receptor_pdbqt=_result_receptor_pdbqt(result),
        )
    except Exception as exc:
        raise HTTPException(400, f"作用力分析失败：{exc}") from exc


def _expert_pages(
    result: dict,
    pose: Path,
    receptor_pdb: Path,
    smiles: str,
    outdir: Path,
    prefix: str,
    mode: int = 1,
) -> list[Path]:
    """Generate the ADMET, pharmacophore and MM-GBSA pages for one result."""
    if not smiles:
        raise HTTPException(400, "配体缺少 SMILES，无法计算分子性质")
    interaction = _interaction_for_result(result, pose, receptor_pdb, mode)
    name = str(result.get("ligand_name") or "ligand")
    receptor_name = str(result.get("receptor_name") or "receptor")
    try:
        profile = _admet_for_result(result, smiles)
        pages = [admet.admet_figure_pdf(profile, outdir / f"{prefix}_admet.pdf")]
        pharm = pharmacophore.extract(pose, receptor_pdb, mode, interaction)
        pages.append(
            pharmacophore.pharmacophore_pdf(
                pharm, name, receptor_name, smiles, outdir / f"{prefix}_pharm.pdf"
            )
        )
        mm = mmgbsa.decompose(pose, receptor_pdb, mode, interaction, result.get("affinity"))
        pages.append(
            mmgbsa.mmgbsa_pdf(mm, name, receptor_name, outdir / f"{prefix}_mmgbsa.pdf")
        )
        try:
            consensus = energy.consensus(
                result, interaction, _ligand_props(result), mm
            )
            pages.append(
                energy.energy_pdf(
                    consensus, name, receptor_name, outdir / f"{prefix}_energy.pdf"
                )
            )
        except Exception:
            pass
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"专业分析报告生成失败：{exc}") from exc
    return pages


@app.get("/api/admet")
def admet_lookup(smiles: str, name: str = "", affinity: float | None = None):
    try:
        return admet.profile_smiles(smiles, name=name, affinity=affinity)
    except Exception as exc:
        raise HTTPException(400, f"ADMET 计算失败：{exc}") from exc


@app.get("/api/admet.pdf")
def admet_pdf(smiles: str, name: str = "", affinity: float | None = None):
    try:
        profile = admet.profile_smiles(smiles, name=name, affinity=affinity)
    except Exception as exc:
        raise HTTPException(400, f"ADMET 计算失败：{exc}") from exc
    path = _professional_dir() / admet.admet_filename(name or "ligand")
    try:
        admet.admet_figure_pdf(profile, path)
    except Exception as exc:
        raise HTTPException(400, f"ADMET 图生成失败：{exc}") from exc
    return FileResponse(path, filename=path.name, media_type="application/pdf")


@app.post("/api/admet/batch")
def admet_batch(payload: dict):
    molecules = payload.get("molecules") or []
    if not isinstance(molecules, list) or not molecules:
        raise HTTPException(400, "请提供至少一个 SMILES")
    profiles = []
    errors = []
    for index, item in enumerate(molecules[:60], start=1):
        if isinstance(item, str):
            smiles = item.strip()
            name = f"分子 {index}"
        else:
            smiles = str(item.get("smiles") or "").strip()
            name = str(item.get("name") or f"分子 {index}").strip()
        if not smiles:
            continue
        try:
            profiles.append(admet.profile_smiles(smiles, name=name))
        except Exception as exc:
            errors.append({"name": name, "smiles": smiles, "error": str(exc)})
    if not profiles:
        detail = errors[0]["error"] if errors else "没有可解析的 SMILES"
        raise HTTPException(400, f"批量 ADMET 计算失败：{detail}")
    return {"profiles": profiles, "errors": errors}


@app.get("/api/runs/{run_id}/result/{index}/admet")
def result_admet(run_id: str, index: int):
    result, _pose, _receptor, smiles = _result_context(run_id, index)
    return _admet_for_result(result, smiles)


@app.get("/api/runs/{run_id}/result/{index}/admet.pdf")
def result_admet_pdf(run_id: str, index: int):
    result, _pose, _receptor, smiles = _result_context(run_id, index)
    profile = _admet_for_result(result, smiles)
    path = _professional_dir() / f"{run_id}_{index + 1}_{admet.admet_filename(result.get('ligand_name', 'ligand'))}"
    try:
        admet.admet_figure_pdf(profile, path)
    except Exception as exc:
        raise HTTPException(400, f"ADMET 图生成失败：{exc}") from exc
    return FileResponse(path, filename=path.name, media_type="application/pdf")


def _gene_list(value: str | list | None) -> list[str]:
    if isinstance(value, list):
        items = value
    else:
        items = str(value or "").replace(";", ",").replace("，", ",").split(",")
    genes = []
    for item in items:
        gene = str(item).strip()
        if gene and gene not in genes:
            genes.append(gene)
    return genes


def _pharm_payload(
    smiles: str,
    name: str,
    species: str,
    genes: str | list | None,
    online_lookup: bool,
    affinity: float | None = None,
) -> dict:
    if not smiles:
        raise HTTPException(400, "请输入化合物的 SMILES")
    try:
        return pharm.predict(
            smiles,
            name=name or "候选分子",
            species=species,
            genes=_gene_list(genes),
            affinity=affinity,
            include_online=online_lookup,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"综合药理计算失败：{exc}") from exc


@app.get("/api/pharm")
def pharm_lookup(
    smiles: str,
    name: str = "",
    species: str = "human",
    genes: str = "",
    online: bool = True,
    affinity: float | None = None,
):
    return _pharm_payload(smiles, name, species, genes, online, affinity)


@app.get("/api/pharm.pdf")
def pharm_report(
    smiles: str,
    name: str = "",
    species: str = "human",
    genes: str = "",
    online: bool = True,
    affinity: float | None = None,
    lang: str = "zh",
):
    lang = "en" if str(lang).lower().startswith("en") else "zh"
    i18n.set_lang(lang)
    try:
        data = _pharm_payload(smiles, name, species, genes, online, affinity)
        path = _professional_dir() / (
            f"pharm_{safe_name(name or 'compound')}"
            f"{'_en' if lang == 'en' else ''}.pdf"
        )
        try:
            pharm.pharm_pdf(data, path)
        except Exception as exc:
            raise HTTPException(400, f"综合药理报告生成失败：{exc}") from exc
    finally:
        i18n.set_lang("zh")
    return FileResponse(path, filename=path.name, media_type="application/pdf")


@app.get("/api/ligands/{ligand_id}/pharm")
def ligand_pharm(ligand_id: str, species: str = "human", genes: str = "", online: bool = True):
    with STATE.lock:
        ligand = STATE.project.get_ligand(ligand_id)
    if not ligand:
        raise HTTPException(404, "配体不存在")
    smiles = ligand.get("smiles") or (ligand.get("meta") or {}).get("smiles") or ""
    return _pharm_payload(
        smiles,
        ligand.get("name") or "候选分子",
        species,
        genes,
        online,
    )


@app.get("/api/ligands/{ligand_id}/pharm.pdf")
def ligand_pharm_pdf(
    ligand_id: str,
    species: str = "human",
    genes: str = "",
    online: bool = True,
    lang: str = "zh",
):
    with STATE.lock:
        ligand = STATE.project.get_ligand(ligand_id)
    if not ligand:
        raise HTTPException(404, "配体不存在")
    smiles = ligand.get("smiles") or (ligand.get("meta") or {}).get("smiles") or ""
    return pharm_report(
        smiles=smiles,
        name=ligand.get("name") or "候选分子",
        species=species,
        genes=genes,
        online=online,
        lang=lang,
    )


@app.get("/api/targets/tissue")
def target_tissue(genes: str = "", species: str = "human"):
    gene_list = _gene_list(genes)
    if not gene_list:
        raise HTTPException(400, "请提供至少一个基因符号")
    return online.target_profiles(gene_list, species)


# ---------------------------------------------------------------------------
# online intelligence (structure / target / compound / enrichment)
# ---------------------------------------------------------------------------
@app.post("/api/online/structure")
def online_structure(payload: dict):
    """Fetch an AlphaFold DB model or predict a structure with ESMFold."""
    mode = str(payload.get("mode") or "alphafold").lower()
    if mode == "esmfold":
        result = online.esmfold_model(payload.get("sequence") or "")
        return result
    accession = str(payload.get("accession") or "").strip()
    query = str(payload.get("gene") or payload.get("query") or "").strip()
    if not accession:
        if not query:
            raise HTTPException(400, "请输入基因名、UniProt 登录号或氨基酸序列")
        lookup = online.uniprot_lookup(query, str(payload.get("species") or "human"))
        if not lookup.get("available"):
            return {"available": False, "error": lookup.get("error"), "uniprot": lookup}
        accession = lookup.get("accession") or ""
        result = online.alphafold_model(accession)
        result["uniprot"] = lookup
    else:
        result = online.alphafold_model(accession)
    return result


@app.post("/api/online/structure/save")
def online_structure_save(payload: dict):
    """Register a predicted/downloaded PDB as a project receptor."""
    text = str(payload.get("pdb") or "")
    if "ATOM" not in text:
        raise HTTPException(400, "PDB 内容无效")
    name = str(payload.get("name") or "predicted_structure").strip()[:80]
    record = _store_receptor_pdb(name, text)
    result = {
        "ok": True,
        "receptor_id": record["id"],
        "name": record.get("name"),
        "state": STATE.snapshot(),
    }
    if payload.get("prepare", True):
        try:
            result["job_id"] = prepare_receptor_endpoint(record["id"], {})["job_id"]
        except Exception as exc:
            result["prepare_error"] = str(exc)
    return result


@app.post("/api/online/target")
def online_target(payload: dict):
    query = str(payload.get("query") or payload.get("gene") or "").strip()
    if not query:
        raise HTTPException(400, "请输入基因名或 UniProt 登录号")
    return online.target_intel(query, str(payload.get("species") or "human"))


@app.post("/api/online/compound")
def online_compound(payload: dict):
    query = str(payload.get("query") or "").strip()
    if not query:
        raise HTTPException(400, "请输入化合物名称或 SMILES")
    kind = "smiles" if str(payload.get("kind") or "name").lower() == "smiles" else "name"
    record = online.pubchem_compound(query, kind)
    similar = {"available": False, "rows": [], "error": "未查询相似化合物"}
    if record.get("available") and payload.get("similar", True):
        reference = record.get("smiles") or (query if kind == "smiles" else "")
        if reference:
            try:
                threshold = float(payload.get("threshold") or 0.9)
            except (TypeError, ValueError):
                threshold = 0.9
            try:
                limit = int(payload.get("limit") or 10)
            except (TypeError, ValueError):
                limit = 10
            similar = online.pubchem_similar(reference, threshold, limit)
    return {"record": record, "similar": similar}


@app.post("/api/online/enrichment")
def online_enrichment(payload: dict):
    genes = payload.get("genes") or []
    if not isinstance(genes, list):
        genes = _gene_list(str(genes).replace("\n", ",").replace(" ", ","))
    else:
        genes = [str(item).strip() for item in genes if str(item).strip()]
    library = str(
        payload.get("library")
        or ("Reactome_2022" if COMMERCIAL_MODE else "KEGG_2021_Human")
    )
    return online.enrichr_enrich(genes, library)


@app.get("/api/reports/compound.pdf")
def compound_report(
    ligand_id: str = "",
    smiles: str = "",
    name: str = "",
    species: str = "human",
    genes: str = "",
    online: bool = True,
    lang: str = "zh",
):
    """Compound-only report that works without any docking run."""
    lang = "en" if str(lang).lower().startswith("en") else "zh"
    with STATE.lock:
        project = STATE.project
        ligand = project.get_ligand(ligand_id) if ligand_id else None
        root = project.root
    if ligand:
        smiles = smiles or ligand.get("smiles") or (ligand.get("meta") or {}).get("smiles") or ""
        name = name or ligand.get("name") or "候选分子"
    if not smiles:
        raise HTTPException(400, "请输入 SMILES 或选择项目中的配体")
    i18n.set_lang(lang)
    try:
        data = _pharm_payload(smiles, name, species, genes, online)
        outdir = root / "exports" / "reports"
        outdir.mkdir(parents=True, exist_ok=True)
        stem = safe_name(name or "compound")
        admet_path = _professional_dir() / f"compound_{stem}{'_en' if lang == 'en' else ''}_admet.pdf"
        try:
            admet.admet_figure_pdf(data["profile"], admet_path)
        except Exception as exc:
            raise HTTPException(400, f"ADMET 页生成失败：{exc}") from exc
        outpath = outdir / f"{stem}_compound_report{'_en' if lang == 'en' else ''}.pdf"
        try:
            pharm.compound_report_pdf(data, admet_path, outpath)
        except Exception as exc:
            raise HTTPException(400, f"化合物综合报告生成失败：{exc}") from exc
    finally:
        i18n.set_lang("zh")
    return FileResponse(outpath, filename=outpath.name, media_type="application/pdf")


@app.get("/api/runs/{run_id}/result/{index}/pharmacophore")
def result_pharmacophore(run_id: str, index: int, mode: int = 1):
    result, pose, receptor_pdb, _smiles = _result_context(run_id, index)
    interaction = _interaction_for_result(result, pose, receptor_pdb, mode)
    try:
        return pharmacophore.extract(pose, receptor_pdb, mode, interaction)
    except Exception as exc:
        raise HTTPException(400, f"药效团分析失败：{exc}") from exc


@app.get("/api/runs/{run_id}/result/{index}/pharmacophore.pdf")
def result_pharmacophore_pdf(run_id: str, index: int, mode: int = 1):
    result, pose, receptor_pdb, smiles = _result_context(run_id, index)
    interaction = _interaction_for_result(result, pose, receptor_pdb, mode)
    try:
        data = pharmacophore.extract(pose, receptor_pdb, mode, interaction)
    except Exception as exc:
        raise HTTPException(400, f"药效团分析失败：{exc}") from exc
    name = str(result.get("ligand_name") or "ligand")
    path = _professional_dir() / f"{run_id}_{index + 1}_{pharmacophore.filename(name)}"
    try:
        pharmacophore.pharmacophore_pdf(
            data, name, str(result.get("receptor_name") or "receptor"), smiles, path
        )
    except Exception as exc:
        raise HTTPException(400, f"药效团图生成失败：{exc}") from exc
    return FileResponse(path, filename=path.name, media_type="application/pdf")


@app.get("/api/runs/{run_id}/result/{index}/mmgbsa")
def result_mmgbsa(run_id: str, index: int, mode: int = 1):
    result, pose, receptor_pdb, _smiles = _result_context(run_id, index)
    interaction = _interaction_for_result(result, pose, receptor_pdb, mode)
    try:
        return mmgbsa.decompose(
            pose, receptor_pdb, mode, interaction, result.get("affinity")
        )
    except Exception as exc:
        raise HTTPException(400, f"自由能分解失败：{exc}") from exc


@app.get("/api/runs/{run_id}/result/{index}/mmgbsa.pdf")
def result_mmgbsa_pdf(run_id: str, index: int, mode: int = 1):
    result, pose, receptor_pdb, _smiles = _result_context(run_id, index)
    interaction = _interaction_for_result(result, pose, receptor_pdb, mode)
    try:
        data = mmgbsa.decompose(
            pose, receptor_pdb, mode, interaction, result.get("affinity")
        )
    except Exception as exc:
        raise HTTPException(400, f"自由能分解失败：{exc}") from exc
    name = str(result.get("ligand_name") or "ligand")
    path = _professional_dir() / f"{run_id}_{index + 1}_{mmgbsa.filename(name)}"
    try:
        mmgbsa.mmgbsa_pdf(
            data, name, str(result.get("receptor_name") or "receptor"), path
        )
    except Exception as exc:
        raise HTTPException(400, f"自由能分解图生成失败：{exc}") from exc
    return FileResponse(path, filename=path.name, media_type="application/pdf")


@app.get("/api/runs/{run_id}/result/{index}/energy")
def result_energy(run_id: str, index: int, mode: int = 1):
    result, pose, receptor_pdb, _smiles = _result_context(run_id, index)
    interaction = _interaction_for_result(result, pose, receptor_pdb, mode)
    try:
        mm = mmgbsa.decompose(pose, receptor_pdb, mode, interaction, result.get("affinity"))
        return energy.consensus(result, interaction, _ligand_props(result), mm)
    except Exception as exc:
        raise HTTPException(400, f"结合能比较失败：{exc}") from exc


@app.get("/api/runs/{run_id}/result/{index}/energy.pdf")
def result_energy_pdf(run_id: str, index: int, mode: int = 1):
    result, pose, receptor_pdb, _smiles = _result_context(run_id, index)
    interaction = _interaction_for_result(result, pose, receptor_pdb, mode)
    try:
        mm = mmgbsa.decompose(pose, receptor_pdb, mode, interaction, result.get("affinity"))
        data = energy.consensus(result, interaction, _ligand_props(result), mm)
    except Exception as exc:
        raise HTTPException(400, f"结合能比较失败：{exc}") from exc
    name = str(result.get("ligand_name") or "ligand")
    receptor = str(result.get("receptor_name") or "receptor")
    path = _professional_dir() / f"{run_id}_{index + 1}_energy.pdf"
    try:
        energy.energy_pdf(data, name, receptor, path)
    except Exception as exc:
        raise HTTPException(400, f"结合能比较图生成失败：{exc}") from exc
    return FileResponse(path, filename=path.name, media_type="application/pdf")


@app.get("/api/runs/{run_id}/result/{index}/expert.pdf")
def result_expert_pdf(run_id: str, index: int, mode: int = 1):
    """Merge ADMET, pharmacophore and MM-GBSA pages into one expert report."""
    from pypdf import PdfWriter

    result, pose, receptor_pdb, smiles = _result_context(run_id, index)
    name = str(result.get("ligand_name") or "ligand")
    outdir = _professional_dir()
    pages = _expert_pages(
        result, pose, receptor_pdb, smiles, outdir, f"{run_id}_{index + 1}", mode
    )
    writer = PdfWriter()
    for page in pages:
        writer.append(str(page))
    target = outdir / f"{run_id}_{index + 1}_expert.pdf"
    with target.open("wb") as handle:
        writer.write(handle)
    writer.close()
    return FileResponse(
        target, filename=f"{name}_expert_analysis.pdf", media_type="application/pdf"
    )


@app.get("/api/runs/{run_id}/result/{index}/fingerprint.pdf")
def result_fingerprint_pdf(run_id: str, index: int, mode: int = 1):
    result, pose, receptor_pdb, _smiles = _result_context(run_id, index)
    interaction = _interaction_for_result(result, pose, receptor_pdb, mode)
    name = str(result.get("ligand_name") or "ligand")
    receptor = str(result.get("receptor_name") or "receptor")
    path = _professional_dir() / f"{run_id}_{index + 1}_fingerprint.pdf"
    generated = figures.interaction_fingerprint_pdf(interaction, name, receptor, path)
    if not generated:
        raise HTTPException(400, "没有可用的接触指纹数据")
    return FileResponse(path, filename=path.name, media_type="application/pdf")


@app.get("/api/runs/{run_id}/result/{index}/pocket.pdf")
def result_pocket_pdf(run_id: str, index: int, mode: int = 1):
    result, pose, receptor_pdb, _smiles = _result_context(run_id, index)
    interaction = _interaction_for_result(result, pose, receptor_pdb, mode)
    name = str(result.get("ligand_name") or "ligand")
    receptor = str(result.get("receptor_name") or "receptor")
    path = _professional_dir() / f"{run_id}_{index + 1}_pocket.pdf"
    generated = figures.pocket_profile_pdf(interaction, name, receptor, path)
    if not generated:
        raise HTTPException(400, "没有可用的口袋画像数据")
    return FileResponse(path, filename=path.name, media_type="application/pdf")


@app.get("/api/runs/{run_id}/profiling.pdf")
def result_profiling_pdf(run_id: str):
    """Developability, chemical-space and metabolic soft-spot pages."""
    from pypdf import PdfWriter

    run = STATE.project.get_run(run_id)
    if not run:
        raise HTTPException(404, "对接记录不存在")
    records = _profiling_records(run)
    if len(records) < 2:
        raise HTTPException(400, "至少需要两个带 SMILES 的成功结果")
    outdir = STATE.project.root / "exports" / "profiling"
    outdir.mkdir(parents=True, exist_ok=True)
    parts = []
    for name, builder in (
        ("developability", profiling.developability_pdf),
        ("chemspace", profiling.chemspace_pdf),
        ("metabolism", profiling.metabolism_pdf),
    ):
        path = outdir / f"{run_id}_{name}.pdf"
        builder(records, path)
        parts.append(path)
    writer = PdfWriter()
    for part in parts:
        writer.append(str(part))
    target = outdir / f"{run_id}_profiling.pdf"
    with target.open("wb") as handle:
        writer.write(handle)
    writer.close()
    return FileResponse(
        target, filename="DrugHX_profiling.pdf", media_type="application/pdf"
    )


@app.get("/api/runs/{run_id}/profiling.json")
def result_profiling_json(run_id: str):
    """Metabolic soft spots and developability data for the expert panel."""
    with STATE.lock:
        run = STATE.project.get_run(run_id)
        if not run:
            raise HTTPException(404, "对接记录不存在")
        records = _profiling_records(run)
    if not records:
        raise HTTPException(400, "该记录没有带 SMILES 的成功结果")
    rows = profiling.metabolism_rows(records)
    return {"rows": rows, "count": len(rows)}


@app.get("/api/runs/{run_id}/validation.json")
def result_validation_json(run_id: str):
    with STATE.lock:
        project = STATE.project
        run = project.get_run(run_id)
        if not run:
            raise HTTPException(404, "对接记录不存在")
        data = _redocking_validation(project, run)
    return data


@app.post("/api/optimize")
def start_optimization(payload: dict):
    _ensure_no_active_job()
    run_id = str(payload.get("run_id") or "")
    index = int(payload.get("index") or 0)
    options = dict(payload.get("options") or {})
    with STATE.lock:
        run = STATE.project.get_run(run_id)
        if not run:
            raise HTTPException(404, "对接记录不存在")
        results = run.get("results") or []
        if index < 0 or index >= len(results):
            raise HTTPException(404, "结果不存在")
        if results[index].get("status") != "ok":
            raise HTTPException(400, "该结果未成功，无法进行药物优化")
        project = STATE.project
        parent_run = copy.deepcopy(run)
    def task(job):
        try:
            optimization.run_optimization(project, parent_run, index, options, job)
        except Exception as exc:
            job.log(f"[优化失败] {exc}")
            raise

    job = STATE.jobs.create(
        "optimize",
        f"药物优化 · {results[index].get('ligand_name', '')}",
        task,
    )
    return {"job_id": job.id, "run_id": run_id, "parent_index": index}


@app.post("/api/design")
def start_design(payload: dict):
    _ensure_no_active_job()
    receptor_id = str(payload.get("receptor_id") or "")
    options = dict(payload.get("options") or {})
    with STATE.lock:
        receptor = STATE.project.get_receptor(receptor_id)
        if not receptor:
            raise HTTPException(404, "受体不存在")
        if not receptor.get("box"):
            raise HTTPException(400, "请先设置结合位点盒子")
        project = STATE.project
        target_name = receptor.get("name", "target")
    def task(job):
        try:
            design.run_design(project, receptor_id, options, job)
        except Exception as exc:
            job.log(f"[设计失败] {exc}")
            raise

    job = STATE.jobs.create("design", f"分子设计 · {target_name}", task)
    return {"job_id": job.id, "receptor_id": receptor_id}


@app.post("/api/design/database")
def save_design_database(payload: dict):
    text = str(payload.get("text") or "")
    with STATE.lock:
        count = design.save_database(STATE.project, text)
        STATE.project.save()
    return {"ok": True, "count": count}


@app.get("/api/designs/{run_id}/report.pdf")
def design_report(run_id: str):
    with STATE.lock:
        project = STATE.project
        path = project.root / "exports" / f"design_{run_id}" / "design_report.pdf"
        record = next(
            (
                item for item in project.data.get("designs", [])
                if item.get("run_id") == run_id or item.get("id") == run_id
            ),
            None,
        )
    if record:
        # Rebuild from stored results so the downloaded PDF matches the current
        # layout (full candidate table, ADMET table, scaffold labels).
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            design.rebuild_report(project, record, path)
        except Exception:
            pass
    if not path.exists():
        raise HTTPException(404, "设计报告尚未生成")
    return FileResponse(path, filename=f"{run_id}_design_report.pdf", media_type="application/pdf")


@app.get("/api/fishing")
def get_fishing(record_id: str = ""):
    with STATE.lock:
        data = fishing.load_fishing(STATE.project, record_id)
    return {"record": data}


@app.get("/api/fishing/resolve")
def fishing_resolve_name(name: str = ""):
    """Resolve a molecule name to SMILES from project ligands or the FDA set."""
    text = str(name or "").strip().lower()
    if not text:
        return {"smiles": "", "source": ""}
    with STATE.lock:
        for ligand in STATE.project.data.get("ligands", []):
            if str(ligand.get("name") or "").strip().lower() == text:
                smiles = ligand.get("smiles") or (ligand.get("meta") or {}).get("smiles") or ""
                if smiles:
                    return {"smiles": smiles, "source": "project", "name": ligand.get("name")}
    for item in drugref.FDA_DRUGS:
        if str(item.get("name") or "").strip().lower() == text:
            return {"smiles": item["smiles"], "source": "FDA", "name": item["name"]}
    return {"smiles": "", "source": ""}


@app.get("/api/fishing/{record_id}/report.pdf")
def fishing_standalone_report(record_id: str, lang: str = "zh"):
    lang = "en" if str(lang).lower().startswith("en") else "zh"
    i18n.set_lang(lang)
    with STATE.lock:
        data = fishing.load_fishing(STATE.project, record_id)
    if not data:
        raise HTTPException(404, "靶点垂钓记录不存在")
    outdir = STATE.project.root / "exports" / "fishing"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{record_id}_report_{lang}.pdf"
    figures.fishing_report_pdf(data, path)
    return FileResponse(
        path, media_type="application/pdf",
        filename=f"fishing_{record_id}_{lang}.pdf",
    )


@app.get("/api/netpharm/{record_id}/report.pdf")
def netpharm_standalone_report(record_id: str, lang: str = "zh"):
    lang = "en" if str(lang).lower().startswith("en") else "zh"
    i18n.set_lang(lang)
    with STATE.lock:
        data = netpharm.load_netpharm(STATE.project, record_id)
    if not data:
        raise HTTPException(404, "网络药理学记录不存在")
    outdir = STATE.project.root / "exports" / "netpharm"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{record_id}_report_{lang}.pdf"
    figures.netpharm_report_pdf(data, path)
    return FileResponse(
        path, media_type="application/pdf",
        filename=f"netpharm_{record_id}_{lang}.pdf",
    )


@app.post("/api/fishing/run")
def start_fishing(payload: dict):
    _ensure_no_active_job()
    smiles = str(payload.get("smiles") or "").strip()
    if not smiles:
        raise HTTPException(400, "请输入待垂钓分子的 SMILES")
    from rdkit import Chem

    if Chem.MolFromSmiles(smiles) is None:
        raise HTTPException(400, "SMILES 无法解析，请检查输入")
    name = str(payload.get("name") or "查询分子").strip()
    options = dict(payload.get("options") or {})
    project = STATE.project

    def task(job):
        try:
            fishing.run_fishing(project, smiles, name, options, job)
        except Exception as exc:
            job.log(f"[垂钓失败] {exc}")
            raise

    job = STATE.jobs.create("fishing", f"靶点垂钓 · {name}", task)
    return {"job_id": job.id}


@app.get("/api/netpharm")
def get_netpharm(record_id: str = ""):
    with STATE.lock:
        data = netpharm.load_netpharm(STATE.project, record_id)
    return {"record": data}


@app.post("/api/netpharm/run")
def start_netpharm(payload: dict):
    _ensure_no_active_job()

    options = dict(payload.get("options") or {})
    extra_genes = [
        str(gene).strip()
        for gene in (payload.get("genes") or [])
        if str(gene).strip()
    ]
    with STATE.lock:
        project = STATE.project
        source = fishing.load_fishing(
            project, str(payload.get("fishing_id") or "")
        )
    if not source:
        raise HTTPException(
            400, "请先在“靶点垂钓”中完成一次分析；网络药理学将直接使用垂钓得到的靶点与基因集。"
        )
    genes = _fishing_human_genes(source)
    for gene in extra_genes:
        if gene not in genes:
            genes.append(gene)
    compound = source.get("compound") or {"name": "查询分子", "smiles": ""}
    if len(genes) < 2:
        raise HTTPException(
            400, "垂钓结果中带基因符号的靶点不足 2 个，无法构建网络；请补充人源基因"
        )

    options["supplied_genes"] = list(extra_genes)

    def task(job):
        try:
            netpharm.run_netpharm(
                project, genes, compound, options, job
            )
        except Exception as exc:
            job.log(f"[网络药理学失败] {exc}")
            raise

    job = STATE.jobs.create(
        "netpharm", f"网络药理学 · {compound.get('name', '') or '查询分子'}", task
    )
    return {"job_id": job.id, "genes": len(genes)}


@app.get("/api/synthesis")
def get_synthesis(record_id: str = ""):
    with STATE.lock:
        data = synth.load_synthesis(STATE.project, record_id)
    return {"record": data}


@app.post("/api/synthesis/run")
def start_synthesis(payload: dict):
    smiles = str(payload.get("smiles") or "").strip()
    name = str(payload.get("name") or "候选分子").strip()
    if not smiles:
        raise HTTPException(400, "请输入待评估分子的 SMILES")
    with STATE.lock:
        project = STATE.project
        try:
            data = synth.run_synthesis(project, smiles, name)
        except Exception as exc:
            raise HTTPException(400, f"可合成性评估失败：{exc}") from exc
    return {"record": data}


@app.get("/api/synthesis.pdf")
def synthesis_pdf(record_id: str = "", smiles: str = "", name: str = ""):
    with STATE.lock:
        project = STATE.project
        if record_id:
            data = synth.load_synthesis(project, record_id)
        elif smiles:
            try:
                data = synth.assess(smiles, name or "候选分子")
            except Exception as exc:
                raise HTTPException(400, f"SMILES 无法解析：{exc}") from exc
        else:
            data = synth.load_synthesis(project, "")
        if not data:
            raise HTTPException(404, "还没有可合成性记录")
        target = synth.synthesis_dir(project, data["id"])
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"{synth.filename(name or data.get('compound', {}).get('name', 'candidate'))}_synthesis.pdf"
        if not path.is_file():
            synth.synthesis_report_pdf(data, path)
    return FileResponse(
        path, filename=path.name, media_type="application/pdf"
    )


@app.get("/api/runs/{run_id}/result/{index}/synthesis")
def result_synthesis(run_id: str, index: int):
    with STATE.lock:
        run = STATE.project.get_run(run_id)
        results = run.get("results", []) if run else []
        if not results or index < 0 or index >= len(results):
            raise HTTPException(404, "结果不存在")
        result = results[index]
        if result.get("status") != "ok":
            raise HTTPException(400, "该结果未成功，无法评估可合成性")
        smiles = _result_smiles(result)
        if not smiles:
            raise HTTPException(400, "该配体没有可用的 SMILES，无法评估可合成性")
        data = synth.run_synthesis(
            STATE.project, smiles, result.get("ligand_name", "候选分子")
        )
    return {"record": data}


@app.get("/api/runs/{run_id}/result/{index}/synthesis.pdf")
def result_synthesis_pdf(run_id: str, index: int):
    with STATE.lock:
        run = STATE.project.get_run(run_id)
        results = run.get("results", []) if run else []
        if not results or index < 0 or index >= len(results):
            raise HTTPException(404, "结果不存在")
        result = results[index]
        smiles = _result_smiles(result)
        if not smiles:
            raise HTTPException(400, "该配体没有可用的 SMILES")
        data = synth.run_synthesis(
            STATE.project, smiles, result.get("ligand_name", "候选分子")
        )
        path = Path(data.get("report_pdf") or "")
    if not path.is_file():
        raise HTTPException(404, "合成路线 PDF 尚未生成")
    return FileResponse(
        path, filename=path.name, media_type="application/pdf"
    )


def _targets_for_smiles(project: Project, smiles: str) -> list[str]:
    """Best-effort gene symbols from a fishing record for the same molecule."""
    if not smiles:
        return []
    keys = _molecule_keys(smiles)
    if not keys:
        return []
    for record in project.data.get("fishing", []):
        record_smiles = (record.get("compound") or {}).get("smiles") or ""
        if not record_smiles:
            continue
        if not (keys & _molecule_keys(record_smiles)):
            continue
        genes = []
        for item in record.get("targets", []):
            for gene in (item.get("genes") or [item.get("gene")]):
                if gene and gene not in genes:
                    genes.append(gene)
        return genes
    return []


@app.get("/api/ddi")
def get_ddi(record_id: str = ""):
    with STATE.lock:
        records = STATE.project.data.get("ddi", [])
        record = next(
            (item for item in records if item.get("id") == record_id),
            records[0] if records else None,
        )
    return {"record": record or {}}


@app.post("/api/ddi/run")
def run_ddi(payload: dict):
    smiles = str(payload.get("smiles") or "").strip()
    name = str(payload.get("name") or "候选分子").strip()
    co_drugs = payload.get("co_drugs") or []
    if payload.get("use_fda_all"):
        # Default behaviour: compare the candidate with every FDA reference
        # drug in the local set, optionally plus user-supplied entries.
        seen = {
            str(item.get("name") if isinstance(item, dict) else item or "").strip().lower()
            for item in co_drugs
        }
        for item in drugref.FDA_DRUGS:
            key = str(item.get("name") or "").strip().lower()
            if not key or key in seen:
                continue
            co_drugs.append(
                {
                    "name": item["name"],
                    "smiles": item["smiles"],
                    "targets": item.get("primary_targets"),
                    "drug_class": item.get("drug_class"),
                    "indication": item.get("indication"),
                    "metabolized_by": item.get("metabolized_by"),
                }
            )
    if not smiles:
        raise HTTPException(400, "请输入候选分子 SMILES")
    if not co_drugs:
        raise HTTPException(400, "请至少选择一个同时使用的药物")
    with STATE.lock:
        project = STATE.project
        targets = payload.get("targets") or _targets_for_smiles(project, smiles)
        try:
            data = ddi.analyse(
                smiles,
                name,
                co_drugs=co_drugs,
                candidate_targets=targets,
                candidate_class=str(payload.get("drug_class") or ""),
            )
        except Exception as exc:
            raise HTTPException(400, f"药物互作分析失败：{exc}") from exc
        record_id = f"ddi_{datetime.now():%Y%m%d_%H%M%S}"
        outdir = project.root / "exports" / "ddi" / record_id
        outdir.mkdir(parents=True, exist_ok=True)
        data["id"] = record_id
        data["created"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pdf_path = outdir / "ddi_report.pdf"
        try:
            ddi.ddi_pdf(data, pdf_path)
            data["report_pdf"] = _rel(pdf_path)
        except Exception:
            data["report_pdf"] = ""
        write_json(outdir / "ddi.json", data)
        project.data.setdefault("ddi", [])
        project.data["ddi"] = [
            item for item in project.data["ddi"] if item.get("id") != record_id
        ]
        project.data["ddi"].insert(
            0,
            {
                "id": record_id,
                "created": data["created"],
                "candidate": data.get("candidate"),
                "summary": data.get("summary"),
                "report_pdf": data.get("report_pdf", ""),
                "co_drugs": [
                    {
                        "name": item.get("name"),
                        "score": item.get("score"),
                        "category": item.get("category"),
                    }
                    for item in (data.get("co_drugs") or [])[:12]
                ],
            },
        )
        project.data["ddi"] = project.data["ddi"][:20]
        project.save()
    return {"record": data}


@app.get("/api/ddi.pdf")
def ddi_pdf(record_id: str = ""):
    with STATE.lock:
        records = STATE.project.data.get("ddi", [])
        record = next(
            (item for item in records if item.get("id") == record_id),
            records[0] if records else None,
        )
        if not record:
            raise HTTPException(404, "还没有药物互作记录")
        path = _abs(record.get("report_pdf"))
    if not path or not path.is_file():
        raise HTTPException(404, "药物互作 PDF 尚未生成")
    return FileResponse(path, filename=path.name, media_type="application/pdf")


@app.get("/api/drugref/list")
def drugref_list():
    return {"drugs": drugref.list_drugs()}


@app.get("/api/drugref")
def get_drugref(record_id: str = ""):
    with STATE.lock:
        records = STATE.project.data.get("drugref", [])
        record = next(
            (item for item in records if item.get("id") == record_id),
            records[0] if records else None,
        )
    return {"record": record or {}}


@app.post("/api/drugref/run")
def run_drugref(payload: dict):
    smiles = str(payload.get("smiles") or "").strip()
    name = str(payload.get("name") or "候选分子").strip()
    if not smiles:
        raise HTTPException(400, "请输入候选分子 SMILES")
    with STATE.lock:
        project = STATE.project
        targets = payload.get("targets") or _targets_for_smiles(project, smiles)
        try:
            data = drugref.compare(
                smiles,
                name,
                candidate_targets=targets,
                top_n=int(payload.get("top_n") or 12),
            )
        except Exception as exc:
            raise HTTPException(400, f"FDA 参考分析失败：{exc}") from exc
        record_id = f"ref_{datetime.now():%Y%m%d_%H%M%S}"
        outdir = project.root / "exports" / "drugref" / record_id
        outdir.mkdir(parents=True, exist_ok=True)
        data["id"] = record_id
        data["created"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pdf_path = outdir / "drugref_report.pdf"
        try:
            drugref.drugref_pdf(data, pdf_path)
            data["report_pdf"] = _rel(pdf_path)
        except Exception:
            data["report_pdf"] = ""
        write_json(outdir / "drugref.json", data)
        project.data.setdefault("drugref", [])
        project.data["drugref"] = [
            item for item in project.data["drugref"] if item.get("id") != record_id
        ]
        project.data["drugref"].insert(
            0,
            {
                "id": record_id,
                "created": data["created"],
                "candidate": data.get("candidate"),
                "top": data.get("top"),
                "report_pdf": data.get("report_pdf", ""),
            },
        )
        project.data["drugref"] = project.data["drugref"][:20]
        project.save()
    return {"record": data}


@app.get("/api/drugref.pdf")
def drugref_pdf(record_id: str = ""):
    with STATE.lock:
        records = STATE.project.data.get("drugref", [])
        record = next(
            (item for item in records if item.get("id") == record_id),
            records[0] if records else None,
        )
        if not record:
            raise HTTPException(404, "还没有 FDA 参考分析记录")
        path = _abs(record.get("report_pdf"))
    if not path or not path.is_file():
        raise HTTPException(404, "FDA 参考 PDF 尚未生成")
    return FileResponse(path, filename=path.name, media_type="application/pdf")


@app.post("/api/reports/prepare")
def prepare_report_modules(payload: dict):
    """Fill in target fishing + network pharmacology before a full report.

    Both steps query public web services and can take minutes; running them
    inside the PDF download (a plain GET) made the browser look frozen.  The
    front end now calls this job first and opens the PDF afterwards.
    """
    _ensure_no_active_job()
    run_id = str(payload.get("run_id") or "")
    force = bool(payload.get("force"))
    project = STATE.project
    run = project.get_run(run_id) if run_id else None
    if not run:
        candidates = [item for item in project.data.get("runs", []) if item.get("results")]
        run = candidates[0] if candidates else None
    if not run:
        raise HTTPException(400, "还没有可用的对接结果")
    results = [
        item for item in run.get("results", []) if item.get("status") == "ok"
    ]
    if not results:
        raise HTTPException(400, "该记录没有成功结果")
    best = min(results, key=lambda item: float(item.get("affinity") or 999))
    smiles = _result_smiles(best)
    if not smiles:
        raise HTTPException(400, "无法获取候选分子的 SMILES")
    name = best.get("ligand_name") or "候选分子"

    def task(job):
        fishing_id = ""
        netpharm_id = ""
        notice = ""
        with STATE.lock:
            has_fishing = bool(project.data.get("fishing"))
            empty_attempt = dict(project.data.get("fishing_empty") or {})
        if not force and not has_fishing and empty_attempt.get("smiles") == smiles:
            job.log("[报告] 该分子此前未检索到靶点，跳过重复检索")
            job.set_progress(1.0, "跳过重复检索")
            job.result = {
                "fishing_id": "",
                "netpharm_id": "",
                "skipped": True,
                "reason": "此前未检索到该分子的蛋白靶点，报告暂不含垂钓与网络药理学",
            }
            return
        job.set_progress(0.02, "靶点垂钓 ...")
        try:
            existing = None if force else fishing.load_fishing(project, "")
            if existing and existing.get("targets"):
                fishing_id = existing.get("id", "")
                job.log(f"[报告] 复用已有垂钓记录 {fishing_id}")
            else:
                data = fishing.run_fishing(
                    project,
                    smiles,
                    name,
                    {
                        "similarity": 40,
                        "molecules": 120,
                        "targets": 220,
                        "depth": "deep",
                        "swiss_organisms": [
                            "Homo_sapiens",
                            "Mus_musculus",
                            "Rattus_norvegicus",
                        ],
                    },
                    _JobScope(job, 0.02, 0.60),
                )
                fishing_id = (data or {}).get("id", "")
                job.log(
                    f"[报告] 靶点垂钓完成："
                    f"{len((data or {}).get('targets') or [])} 个靶点（{fishing_id}）"
                )
                with STATE.lock:
                    project.data.pop("fishing_empty", None)
                    project.save()
        except Exception as exc:
            notice = f"未检索到靶点，报告暂不含垂钓与网络药理学：{exc}"
            job.log(f"[报告] 靶点垂钓失败：{exc}")
            with STATE.lock:
                project.data["fishing_empty"] = {"smiles": smiles}
                project.save()
        job.set_progress(0.62, "网络药理学 ...")
        try:
            source = fishing.load_fishing(project, fishing_id)
            genes = _fishing_human_genes(source)
            existing_net = None if force else netpharm.load_netpharm(project, "")
            if existing_net and (existing_net.get("ppi") or {}).get("edges"):
                netpharm_id = existing_net.get("id", "")
                job.log(f"[报告] 复用已有网络药理学记录 {netpharm_id}")
            elif len(genes) >= 2:
                record = netpharm.run_netpharm(
                    project,
                    genes,
                    (source or {}).get("compound")
                    or {"name": name, "smiles": smiles},
                    {"genes": 80, "string_score": 400, "expand": 20},
                    _JobScope(job, 0.62, 0.98),
                )
                netpharm_id = (record or {}).get("id", "")
                job.log(f"[报告] 网络药理学完成：{len(genes)} 个基因（{netpharm_id}）")
            else:
                job.log("[报告] 人源基因不足 2 个，跳过网络药理学")
        except Exception as exc:
            job.log(f"[报告] 网络药理学失败：{exc}")
        job.set_progress(1.0, "报告所需分析已补齐")
        job.result = {
            "fishing_id": fishing_id,
            "netpharm_id": netpharm_id,
            "notice": notice,
        }

    job = STATE.jobs.create("report", f"补齐报告分析 · {name}", task)
    return {"job_id": job.id}


@app.get("/api/reports/full.pdf")
def full_project_report(
    run_id: str = "",
    optimization_id: str = "",
    design_id: str = "",
    fishing_id: str = "",
    netpharm_id: str = "",
    result_index: int | None = None,
    lang: str = "zh",
    species: str = "human",
):
    lang = "en" if str(lang).lower().startswith("en") else "zh"
    i18n.set_lang(lang)
    from . import design as design_module
    from . import optimization as optimization_module
    with STATE.lock:
        project = STATE.project
        run = project.get_run(run_id) if run_id else None
        if not run:
            # Prefer the newest run with at least two successful hits: the
            # optimisation pipeline also stores a one-ligand run, and silently
            # using it would turn the "complete report" into a single-molecule
            # summary.
            candidates = [
                item for item in project.data.get("runs", []) if item.get("results")
            ]
            run = next(
                (
                    item for item in candidates
                    if sum(
                        1 for hit in item["results"]
                        if hit.get("status") == "ok"
                    ) >= 2
                ),
                candidates[0] if candidates else None,
            )
        if not run:
            raise HTTPException(400, "还没有可生成报告的对接结果")
        run = copy.deepcopy(run)
        optimizations = list(project.data.get("optimizations", []))
        designs = list(project.data.get("designs", []))
        optimization = next(
            (item for item in optimizations if item.get("run_id") == optimization_id),
            optimizations[0] if optimizations else None,
        )
        design = next(
            (item for item in designs if item.get("run_id") == design_id),
            designs[0] if designs else None,
        )
        if optimization and not optimization.get("count"):
            opt_run = project.get_run(optimization.get("run_id", ""))
            optimization = dict(optimization)
            optimization["count"] = len(opt_run.get("results", [])) if opt_run else 0
        if design and not design.get("count"):
            des_run = project.get_run(design.get("run_id", ""))
            design = dict(design)
            design["count"] = len(des_run.get("results", [])) if des_run else 0
        opt_run = project.get_run(optimization.get("run_id", "")) if optimization else None
        design_run = project.get_run(design.get("run_id", "")) if design else None
        opt_candidates = list(opt_run.get("results", [])) if opt_run else []
        design_candidates = list(design_run.get("results", [])) if design_run else []
        fishing_data = fishing.load_fishing(project, fishing_id)
        netpharm_data = netpharm.load_netpharm(project, netpharm_id)
        project_info = {
            "name": project.data.get("name", "DrugHX"),
            "path": str(project.root),
            "now": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        ligand_meta = {
            item["id"]: dict(item.get("meta") or {})
            for item in project.data.get("ligands", [])
        }
        root = project.root
    results = [item for item in run.get("results", []) if item.get("status") == "ok"]
    if not results:
        raise HTTPException(400, "该记录没有成功结果")
    def _affinity_key(item):
        value = item.get("affinity")
        return float(value) if isinstance(value, (int, float)) else 999.0

    results.sort(key=_affinity_key)
    best = results[0]
    if result_index is not None:
        # The results page lets the user pick the compound that all downstream
        # modules (and this report) should default to.  Keep the full ranking,
        # but lead with the explicit choice.
        try:
            picked = run.get("results", [])[int(result_index)]
        except (IndexError, ValueError, TypeError):
            picked = None
        if picked and picked.get("status") == "ok":
            best = picked
            results = [item for item in results if item is not picked]
            results.insert(0, picked)
    module_notes: dict[str, str] = {}
    main_smiles = _result_smiles(best)
    fish_smiles = (
        (fishing_data or {}).get("compound") or {}
    ).get("smiles") or ""
    main_keys = _molecule_keys(main_smiles)
    fish_keys = _molecule_keys(fish_smiles)
    if fishing_data and main_keys and fish_keys:
        mismatch_note = ""
        if not (main_keys & fish_keys):
            mismatch_note = (
                (
                    f"The query molecule in the latest target-fishing record "
                    f"({fish_smiles.splitlines()[0][:36]}…) differs from the report's "
                    f"main ligand {best.get('ligand_name', '—')} "
                    f"({main_smiles[:36]}…). The fishing and network pharmacology "
                    "results are still included as a cross-molecule reference and "
                    "must not be read as target conclusions for the main ligand."
                )
                if lang == "en"
                else (
                    f"本项目最新垂钓记录的查询分子（{fish_smiles.splitlines()[0][:36]}…）"
                    f"与报告主配体 {best.get('ligand_name', '—')}（{main_smiles[:36]}…）"
                    "结构不一致；本节仍保留该垂钓与网络药理学结果，仅供跨分子参考，"
                    "不能当作主配体的靶点结论。"
                )
            )
        elif len(fish_keys) > 1:
            mismatch_note = (
                (
                    f"This target-fishing record pools {len(fish_keys)} query "
                    "molecules, and the report's main ligand is one of them; the "
                    "targets and the derived network pharmacology describe the pooled "
                    "gene set and cannot be attributed to the main ligand alone."
                )
                if lang == "en"
                else (
                    f"该垂钓记录合并了 {len(fish_keys)} 个查询分子，报告主配体是其中之一；"
                    "下方靶点与网络药理学结果为这批分子的合并基因集，"
                    "不能全部归因于主配体。"
                )
            )
        if mismatch_note:
            module_notes["fishing"] = mismatch_note
            module_notes["netpharm"] = mismatch_note
            fishing_data = dict(fishing_data)
            fishing_data["report_note"] = mismatch_note
            if netpharm_data:
                netpharm_data = dict(netpharm_data)
                netpharm_data["report_note"] = mismatch_note
    if netpharm_data:
        net_keys = _molecule_keys(
            (netpharm_data.get("compound") or {}).get("smiles") or ""
        )
        note = ""
        if main_keys and net_keys:
            if not (main_keys & net_keys):
                note = (
                    "The compound in the network pharmacology record differs from the "
                    "report's main ligand; the record is shown as a cross-molecule "
                    "reference and must not be read as a conclusion for the main ligand."
                    if lang == "en"
                    else (
                        "网络药理学记录的化合物与报告主配体结构不一致；"
                        "该记录作为跨分子参考保留，不能当作主配体的靶点结论。"
                    )
                )
            elif len(net_keys) > 1:
                note = (
                    f"This network pharmacology record pools {len(net_keys)} query "
                    "molecules (the report's main ligand is one of them); the "
                    "enrichment reflects the pooled gene set."
                    if lang == "en"
                    else (
                        f"该网络药理学记录合并了 {len(net_keys)} 个查询分子"
                        "（报告主配体是其中之一）；富集结果反映合并基因集，"
                        "不等同于主配体单分子的靶点结论。"
                    )
                )
        if note:
            module_notes["netpharm"] = note
            netpharm_data = dict(netpharm_data)
            netpharm_data["report_note"] = note
        ppi_edges = (
            ((netpharm_data or {}).get("ppi") or {}).get("edges") or []
        )
        if netpharm_data and len(ppi_edges) < 1:
            module_notes["netpharm"] = (
                "The STRING query returned no interaction edge, so the PPI page was "
                "omitted; the enrichment, C-T-D and Sankey pages are still included."
                if lang == "en"
                else (
                    "STRING 未返回互作边，已略去蛋白互作网络页；"
                    "富集、C-T-D 与桑基图仍保留在本报告中。"
                )
            )
    synthesis_data = None
    if main_smiles:
        try:
            synthesis_data = synth.assess(
                main_smiles, best.get("ligand_name", "候选分子")
            )
        except Exception as exc:
            module_notes["synthesis"] = f"可合成性评估失败：{exc}"
    else:
        module_notes["synthesis"] = (
            "主配体缺少 SMILES，无法进行可合成性与合成路线评估。"
        )
    fishing_genes: list[str] = []
    for item in (fishing_data or {}).get("targets", []):
        if "homo sapiens" not in str(item.get("organism") or "").lower():
            continue
        for gene in (item.get("genes") or [item.get("gene")]):
            if gene and gene not in fishing_genes:
                fishing_genes.append(gene)
    for gene in (netpharm_data or {}).get("genes", []):
        if gene and gene not in fishing_genes:
            fishing_genes.append(gene)
    pharm_data = None
    if main_smiles:
        try:
            pharm_data = pharm.predict(
                main_smiles,
                name=best.get("ligand_name", "候选分子"),
                species=species,
                genes=fishing_genes,
                affinity=best.get("affinity"),
                include_online=True,
            )
        except Exception as exc:
            module_notes["pharm"] = f"综合药理预测失败：{exc}"
    drugref_data = None
    ddi_data = None
    if main_smiles:
        try:
            drugref_data = drugref.compare(
                main_smiles,
                best.get("ligand_name", "候选分子"),
                candidate_targets=fishing_genes,
            )
        except Exception as exc:
            module_notes["drugref"] = f"FDA 参考分析失败：{exc}"
        if drugref_data:
            try:
                co_drugs = [
                    {
                        "name": item.get("name"),
                        "smiles": item.get("smiles"),
                        "targets": item.get("targets"),
                        "drug_class": item.get("drug_class"),
                        "indication": item.get("indication"),
                        "metabolized_by": item.get("metabolized_by"),
                    }
                    for item in (drugref_data.get("references") or [])
                ]
                ddi_data = ddi.analyse(
                    main_smiles,
                    best.get("ligand_name", "候选分子"),
                    co_drugs=co_drugs,
                    candidate_targets=fishing_genes,
                )
            except Exception as exc:
                module_notes["ddi"] = f"药物互作分析失败：{exc}"
    receptor_pdb = _result_receptor_pdb(best)
    pose = _abs(best.get("pose"))
    if not receptor_pdb or not pose:
        raise HTTPException(400, "缺少受体或构象文件")
    outdir = root / "exports" / "full_report"
    outdir.mkdir(parents=True, exist_ok=True)
    source_pdfs = []
    is_bulk = bool((run.get("params") or {}).get("bulk"))
    if is_bulk:
        # Large-library runs get a dedicated statistics section (7 figures)
        # instead of the 16-pose fine-docking layout.
        bulk_pdf = outdir / f"{run['id']}_bulk_statistics.pdf"
        try:
            figures.bulk_screening_report_pdf(run, ligand_meta, bulk_pdf)
            source_pdfs.append(bulk_pdf)
        except Exception:
            logger.exception("批量粗筛统计图生成失败（run=%s）", run["id"])
    else:
        # Even when the report is generated from a refined re-dock, the latest
        # large-library screen is the first figure chapter: those panels are
        # the ones cited as the screening result in a manuscript.
        bulk_runs = [
            item for item in (project.data.get("runs") or [])
            if (item.get("params") or {}).get("bulk") and item.get("results")
        ]
        if bulk_runs:
            latest_bulk = max(bulk_runs, key=lambda item: str(item.get("time") or ""))
            bulk_pdf = outdir / f"{latest_bulk['id']}_bulk_statistics.pdf"
            try:
                figures.bulk_screening_report_pdf(latest_bulk, ligand_meta, bulk_pdf)
                source_pdfs.append(bulk_pdf)
            except Exception:
                logger.exception(
                    "历史粗筛统计图生成失败（run=%s）", latest_bulk["id"]
                )
    try:
        interaction = analysis.analyse_interactions(
            receptor_pdb.read_text(encoding="utf-8", errors="replace"),
            pose,
            1,
            receptor_pdbqt=_result_receptor_pdbqt(best),
        )
        result_pdf = outdir / f"{run['id']}_best_result.pdf"
        figures.result_figure_pdf(
            pose,
            best.get("ligand_name", "ligand"),
            best.get("receptor_name", "receptor"),
            interaction,
            result_pdf,
            1,
        )

        analyses = {}
        for item in results[:30]:
            rec_pdb = _result_receptor_pdb(item)
            item_pose = _abs(item.get("pose"))
            if not rec_pdb or not item_pose:
                continue
            try:
                analyses[id(item)] = analysis.analyse_interactions(
                    rec_pdb.read_text(encoding="utf-8", errors="replace"),
                    item_pose,
                    1,
                    receptor_pdbqt=_result_receptor_pdbqt(item),
                )
            except Exception:
                continue
        fig_dir = outdir / "figures"
        if fig_dir.exists():
            shutil.rmtree(fig_dir, ignore_errors=True)
        if not is_bulk:
            # A 12k-ligand screen would turn the fine-docking panel into an
            # unreadable wall of bars; the dedicated statistics section above
            # already covers the library.
            figures.build_publication_bundle(
                run, results, analyses, receptor_pdb, pose, best, fig_dir
            )

        # 论文级正文图 1–6：亲和力全景、配体效率、作用力指纹、构象谱、优化与设计。
        props = {}
        for item in results[:30]:
            data = _ligand_props(item)
            if data:
                props[id(item)] = data
        parent = {
            "affinity": best.get("affinity"),
            "name": best.get("ligand_name", ""),
            "props": props.get(id(best), {}),
        }
        optimization_rows = []
        for item in opt_candidates:
            if item.get("status") != "ok" or item.get("affinity") is None:
                continue
            optimization_rows.append(
                {
                    "label": item.get("ligand_name", ""),
                    "affinity": item["affinity"],
                    "props": _ligand_props(item) or {},
                }
            )
        design_rows = []
        for item in design_candidates:
            if item.get("status") != "ok" or item.get("affinity") is None:
                continue
            design_rows.append(
                {
                    "label": item.get("ligand_name", ""),
                    "affinity": item["affinity"],
                    "props": _ligand_props(item) or {},
                }
            )
        optimization_parent = _optimization_parent(project, optimization)
        validation = _redocking_validation(project, run)
        analysis_pdf = outdir / f"{run['id']}_analysis_figures.pdf"
        figures.report_analysis_pdf(
            results,
            analyses,
            props,
            optimization_parent or parent,
            optimization_rows,
            design_rows,
            analysis_pdf,
        )
        source_pdfs.append(analysis_pdf)
        source_pdfs.append(result_pdf)
        if len((run.get("consensus") or {}).get("engines") or []) >= 2:
            try:
                consensus_pdf = outdir / f"{run['id']}_multi_engine.pdf"
                figures.multi_engine_report_pdf(run, consensus_pdf)
                source_pdfs.append(consensus_pdf)
            except Exception as exc:
                module_notes["consensus"] = f"多引擎共识图生成失败：{exc}"
        if fishing_data and fishing_data.get("targets"):
            fish_pdf = outdir / f"{run['id']}_fishing.pdf"
            try:
                figures.fishing_report_pdf(fishing_data, fish_pdf)
                source_pdfs.append(fish_pdf)
            except Exception as exc:
                module_notes["fishing"] = f"靶点垂钓图生成失败：{exc}"
        if netpharm_data and ((netpharm_data.get("ppi") or {}).get("edges")):
            net_pdf = outdir / f"{run['id']}_netpharm.pdf"
            try:
                figures.netpharm_report_pdf(netpharm_data, net_pdf)
                source_pdfs.append(net_pdf)
            except Exception as exc:
                module_notes["netpharm"] = f"网络药理学图生成失败：{exc}"
        profiling_records = _profiling_records(run)
        if profiling_records:
            for suffix, builder in (
                ("developability", profiling.developability_pdf),
                ("chemspace", profiling.chemspace_pdf),
                ("metabolism", profiling.metabolism_pdf),
            ):
                try:
                    panel_pdf = outdir / f"{run['id']}_{suffix}.pdf"
                    builder(profiling_records, panel_pdf)
                    source_pdfs.append(panel_pdf)
                except Exception as exc:
                    module_notes[f"profiling_{suffix}"] = f"{suffix} 图生成失败：{exc}"
        # 图 14–15：原子–残基接触指纹、空间作用图与口袋理化画像。
        fingerprint_pdf = outdir / f"{run['id']}_contact_fingerprint.pdf"
        try:
            generated = figures.interaction_fingerprint_pdf(
                interaction,
                best.get("ligand_name", "ligand"),
                best.get("receptor_name", "receptor"),
                fingerprint_pdf,
            )
            if generated:
                source_pdfs.append(generated)
        except Exception:
            logger.warning("接触指纹图生成失败，已跳过", exc_info=True)
        pocket_pdf = outdir / f"{run['id']}_pocket_profile.pdf"
        try:
            generated = figures.pocket_profile_pdf(
                interaction,
                best.get("ligand_name", "ligand"),
                best.get("receptor_name", "receptor"),
                pocket_pdf,
            )
            if generated:
                source_pdfs.append(generated)
        except Exception:
            logger.warning("口袋画像图生成失败，已跳过", exc_info=True)
        if synthesis_data:
            try:
                synthesis_pdf = outdir / f"{run['id']}_synthesis.pdf"
                synth.synthesis_report_pdf(synthesis_data, synthesis_pdf)
                source_pdfs.append(synthesis_pdf)
            except Exception as exc:
                module_notes["synthesis"] = f"合成路线图生成失败：{exc}"
        # 图 18–22：FDA 参考、药物互作与网络药理学扩展图。
        if drugref_data:
            try:
                ref_pdf = outdir / f"{run['id']}_drugref.pdf"
                drugref.drugref_pdf(drugref_data, ref_pdf)
                source_pdfs.append(ref_pdf)
            except Exception as exc:
                module_notes["drugref"] = f"FDA 参考图生成失败：{exc}"
        if ddi_data:
            try:
                ddi_pdf = outdir / f"{run['id']}_ddi.pdf"
                ddi.ddi_pdf(ddi_data, ddi_pdf)
                source_pdfs.append(ddi_pdf)
            except Exception as exc:
                module_notes["ddi"] = f"药物互作图生成失败：{exc}"
        if netpharm_data:
            for key, func in (
                ("enrichment", netpharm.enrichment_pdf),
                ("ctd", netpharm.ctd_pdf),
                ("sankey", netpharm.sankey_pdf),
            ):
                if key == "ctd" and not netpharm_data.get("ctd"):
                    continue
                if key == "sankey" and not netpharm_data.get("sankey"):
                    continue
                path = outdir / f"{run['id']}_netpharm_{key}.pdf"
                try:
                    func(netpharm_data, path)
                    source_pdfs.append(path)
                except Exception as exc:
                    module_notes[f"netpharm_{key}"] = f"{key} 图生成失败：{exc}"
        if (
            optimization or design or fishing_data or netpharm_data
            or synthesis_data or drugref_data or ddi_data
        ):
            legend_pdf = outdir / f"{run['id']}_module_legends.pdf"
            try:
                figures.module_legends_pdf(
                    optimization, design, fishing_data, netpharm_data, legend_pdf,
                    synthesis_data=synthesis_data,
                    drugref_data=drugref_data,
                    ddi_data=ddi_data,
                )
                source_pdfs.append(legend_pdf)
            except Exception as exc:
                module_notes["legends"] = f"模块图例页生成失败：{exc}"
        # ADMET / 药效团 / 自由能分解三页随完整报告一并收录；缺少 SMILES 时跳过。
        expert_smiles = _result_smiles(best)
        if expert_smiles:
            try:
                source_pdfs.extend(
                    _expert_pages(
                        best,
                        pose,
                        receptor_pdb,
                        expert_smiles,
                        outdir,
                        f"{run['id']}_best",
                        1,
                    )
                )
            except Exception as exc:
                module_notes["expert"] = f"ADMET/药效团/能量专家页生成失败：{exc}"
        if pharm_data:
            try:
                pharm_path = outdir / f"{run['id']}_pharm_profile.pdf"
                pharm.pharm_pdf(pharm_data, pharm_path)
                source_pdfs.append(pharm_path)
            except Exception as exc:
                module_notes["pharm"] = f"综合药理页生成失败：{exc}"
        if optimization and optimization.get("report"):
            rebuild_path = outdir / (
                f"{optimization.get('run_id', 'optimization')}_optimization_report_en.pdf"
                if lang == "en"
                else f"{optimization.get('run_id', 'optimization')}_optimization_report.pdf"
            )
            rebuilt = False
            try:
                optimization_module.rebuild_report(project, optimization, rebuild_path)
                source_pdfs.append(rebuild_path)
                rebuilt = True
            except Exception as exc:
                module_notes["optimization"] = f"优化报告重建失败，使用原始报告：{exc}"
            if not rebuilt:
                path = root / (
                    optimization.get("report_en")
                    if lang == "en" and optimization.get("report_en")
                    else optimization["report"]
                )
                if path.exists():
                    source_pdfs.append(path)
        if design and design.get("report"):
            rebuild_path = outdir / (
                f"{design.get('run_id', 'design')}_design_report_en.pdf"
                if lang == "en"
                else f"{design.get('run_id', 'design')}_design_report.pdf"
            )
            rebuilt = False
            try:
                design_module.rebuild_report(project, design, rebuild_path)
                source_pdfs.append(rebuild_path)
                rebuilt = True
            except Exception as exc:
                module_notes["design"] = f"设计报告重建失败，使用原始报告：{exc}"
            if not rebuilt:
                path = root / (
                    design.get("report_en")
                    if lang == "en" and design.get("report_en")
                    else design["report"]
                )
                if path.exists():
                    source_pdfs.append(path)
        posebusters_payload: dict = {}
        try:
            checked = [
                item for item in run.get("results", [])
                if item.get("status") == "ok"
            ][:12]
            receptor_for_check = _result_receptor_pdb(best) if best else None
            molecules = []
            if receptor_for_check:
                for item in checked:
                    pose = _abs(item.get("pose"))
                    if not pose or not pose.exists():
                        continue
                    try:
                        mol, _ = analysis._load_pose_mol(pose, 1)
                    except Exception:
                        continue
                    molecules.append((str(item.get("ligand_name") or ""), mol))
            if molecules:
                batch = posecheck.run_batch(receptor_for_check, molecules)
                if batch.get("available") and batch.get("rows"):
                    posebusters_payload = batch
        except Exception as exc:  # never block report generation
            logger.warning("PoseBusters report page skipped: %s", exc)
        mpo_payload: dict = {}
        records: list[dict] = []
        try:
            ligand_index = {
                item["id"]: item for item in project.data.get("ligands", [])
            }
            records = []
            for item in run.get("results", []):
                if item.get("status") != "ok" or item.get("affinity") is None:
                    continue
                ligand = ligand_index.get(item.get("ligand_id"))
                if not ligand:
                    continue
                smiles = (
                    ligand.get("smiles")
                    or (ligand.get("meta") or {}).get("smiles")
                    or item.get("ligand_smiles")
                    or ""
                )
                if not smiles:
                    continue
                records.append(
                    {
                        "ligand_id": ligand.get("id"),
                        "name": ligand.get("name") or item.get("ligand_name") or "",
                        "smiles": smiles,
                        "affinity": float(item["affinity"]),
                    }
                )
            records.sort(key=lambda row: row["affinity"])
            if records:
                mpo_payload = mpo.analyse(records[:100])
        except Exception as exc:  # never block report generation
            logger.warning("MPO report page skipped: %s", exc)
        assay_payload = project.data.get("assays") or {}
        trust_payload: dict = {}
        try:
            best_ligand = project.get_ligand((best or {}).get("ligand_id") or "")
            best_smiles = (
                (best_ligand or {}).get("smiles")
                or ((best_ligand or {}).get("meta") or {}).get("smiles")
                or (best or {}).get("ligand_smiles")
                or ""
            )
            best_receptor = project.get_receptor((best or {}).get("receptor_id") or "")
            best_pdb = (
                project.root / (best_receptor.get("source") or best_receptor.get("pdb"))
                if best_receptor else (_result_receptor_pdb(best) if best else None)
            )
            trust_payload = trust.assess(
                best_pdb,
                best_smiles,
                run.get("results", []),
                assay_payload=assay_payload,
                receptor_meta=(best_receptor or {}).get("meta") or {},
            )
        except Exception as exc:
            logger.warning("Trust report page skipped: %s", exc)
        mmp_payload: dict = {}
        try:
            mmp_payload = mmp.mine(
                project.data.get("ligands", []),
                run.get("results", []),
                assay_payload,
            )
        except Exception as exc:
            logger.warning("MMP report page skipped: %s", exc)
        active_payload: dict = {}
        try:
            if records:
                active_payload = activelearn.recommend(
                    records, assay_payload=assay_payload, batch=12,
                )
        except Exception as exc:
            logger.warning("Active-learning report page skipped: %s", exc)
        pbpk_payload = project.data.get("pbpk") or {}
        benchmark_payload = (project.data.get("benchmarks") or {}).get(run["id"]) or {}
        report_path = outdir / (
            f"{run['id']}_complete_report_en.pdf"
            if lang == "en"
            else f"{run['id']}_complete_report.pdf"
        )
        manifest_data: dict = {}
        try:
            _, manifest_data = runmanifest.write_for_run(
                project,
                run,
                outdir,
                params=project.data.get("params"),
                extra={"report_lang": lang, "result_index": result_index},
            )
        except Exception as exc:  # never block the report
            logger.warning("manifest generation skipped: %s", exc)
        figures.full_report_pdf(
            project_info,
            run,
            optimization,
            design,
            source_pdfs,
            report_path,
            analyses,
            props,
            validation,
            optimization_parent,
            fishing_data,
            netpharm_data,
            module_notes,
            synthesis_data=synthesis_data,
            drugref_data=drugref_data,
            ddi_data=ddi_data,
            pharm_data=pharm_data,
            manifest=manifest_data,
            species=species,
            primary_ligand=best,
            posebusters=posebusters_payload,
            mpo_data=mpo_payload,
            assay_data=assay_payload,
            trust_data=trust_payload,
            mmp_data=mmp_payload,
            pbpk_data=pbpk_payload,
            active_data=active_payload,
            benchmark_data=benchmark_payload,
        )
    except Exception as exc:
        raise HTTPException(400, f"完整报告生成失败：{exc}") from exc
    finally:
        i18n.set_lang("zh")
    return FileResponse(
        report_path,
        filename=report_path.name,
        media_type="application/pdf",
    )


@app.get("/api/optimizations/{run_id}/report.pdf")
def optimization_report(run_id: str):
    with STATE.lock:
        project = STATE.project
        record = next(
            (
                item for item in project.data.get("optimizations", [])
                if item.get("run_id") == run_id or item.get("id") == run_id
            ),
            None,
        )
    path = project.root / "exports" / f"optimization_{run_id}" / "optimization_report.pdf"
    if record:
        # Rebuild from stored results so the downloaded PDF reflects the current
        # layout (ligand-efficiency column, ADMET table, size/stability guards).
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            optimization.rebuild_report(project, record, path)
        except Exception:
            pass
    if not path.exists():
        raise HTTPException(404, "优化报告尚未生成")
    return FileResponse(
        path,
        filename=f"{run_id}_optimization_report.pdf",
        media_type="application/pdf",
    )


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str):
    with STATE.lock:
        run = STATE.project.get_run(run_id)
        if not run:
            raise HTTPException(404, "对接记录不存在")
        shutil.rmtree(STATE.project.run_dir(run), ignore_errors=True)
        STATE.project.data["runs"] = [r for r in STATE.project.data["runs"] if r["id"] != run_id]
        STATE.project.save()
    return {"ok": True, "state": STATE.snapshot()}


@app.get("/api/runs/{run_id}/mpo")
def run_mpo(run_id: str, limit: int = 100):
    """Multi-parameter optimisation panel for one docking run."""
    with STATE.lock:
        project = STATE.project
        run = project.get_run(run_id)
        if not run:
            raise HTTPException(404, "对接记录不存在")
        index = {item["id"]: item for item in project.data.get("ligands", [])}
    records = []
    for item in run.get("results", []):
        if item.get("status") != "ok" or item.get("affinity") is None:
            continue
        ligand = index.get(item.get("ligand_id"))
        if not ligand:
            continue
        smiles = (
            ligand.get("smiles")
            or (ligand.get("meta") or {}).get("smiles")
            or item.get("ligand_smiles")
            or ""
        )
        if not smiles:
            continue
        records.append(
            {
                "ligand_id": ligand.get("id"),
                "name": ligand.get("name") or item.get("ligand_name") or "",
                "smiles": smiles,
                "affinity": float(item["affinity"]),
            }
        )
    records.sort(key=lambda row: row["affinity"])
    records = records[: max(1, min(int(limit or 100), 500))]
    payload = mpo.analyse(records)
    payload["run_id"] = run_id
    return payload


@app.get("/api/projects/current/assays")
def get_assay_calibration():
    with STATE.lock:
        return STATE.project.data.get("assays") or {}


@app.post("/api/projects/current/assays")
async def upload_assay_calibration(file: UploadFile = File(...), run_id: str = Form("")):
    """Import measured IC50/Kd values and calibrate the docking ranking."""
    raw = await file.read()
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(400, "实验数据文件超过 20 MB")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("gbk", errors="replace")
    rows = assay.parse_table(text)
    if not rows:
        raise HTTPException(
            400,
            "未解析到有效数据：需要包含名称或 SMILES 列，以及 IC50/Kd/EC50/Ki 数值列"
            "（单位 nM、μM、mM 或 M）。",
        )
    with STATE.lock:
        project = STATE.project
        run = project.get_run(run_id) if run_id else None
        if run is None:
            run = next(
                (
                    item for item in project.data.get("runs", [])
                    if any(row.get("status") == "ok" for row in item.get("results", []))
                ),
                None,
            )
        payload = assay.calibrate(
            rows,
            project.data.get("ligands", []),
            (run or {}).get("results", []),
        )
        payload["file"] = file.filename or ""
        payload["run_id"] = (run or {}).get("id", "")
        project.data["assays"] = payload
        project.save()
    return payload


@app.delete("/api/projects/current/assays")
def clear_assay_calibration():
    with STATE.lock:
        STATE.project.data.pop("assays", None)
        STATE.project.save()
    return {"ok": True}


@app.post("/api/dose")
def dose_endpoint(payload: dict):
    """First-dose projection from target potency and predicted PK."""
    try:
        potency = float(payload.get("potency_nm") or 0)
        mw = float(payload.get("mw") or 0)
    except (TypeError, ValueError):
        raise HTTPException(400, "potency_nm 与 mw 必须是数字") from None
    if potency <= 0 or mw <= 0:
        raise HTTPException(400, "请提供有效的效价（nM）与分子量")
    return mpo.dose_projection(
        potency_nm=potency,
        mw=mw,
        species=str(payload.get("species") or "human"),
        f_percent=float(payload.get("f_percent") or 50),
        cl_ml_min_kg=float(payload.get("cl_ml_min_kg") or 5),
        fu=float(payload.get("fu") or 0.2),
        vd_l_kg=float(payload.get("vd_l_kg") or 1.5),
        target_multiplier=float(payload.get("target_multiplier") or 2.0),
    )


@app.get("/api/runs/{run_id}/trust")
def run_trust(run_id: str, result_index: int = 0):
    """Decision confidence: structure, domain, consensus and calibration."""
    with STATE.lock:
        project = STATE.project
        run = project.get_run(run_id)
        if not run:
            raise HTTPException(404, "对接记录不存在")
        results = [
            item for item in run.get("results", [])
            if item.get("status") == "ok" and item.get("affinity") is not None
        ]
        if not results:
            raise HTTPException(400, "该记录没有成功结果")
        results.sort(key=lambda item: float(item["affinity"]))
        index = max(0, min(int(result_index), len(results) - 1))
        best = results[index]
        ligand = project.get_ligand(best.get("ligand_id") or "")
        smiles = (
            (ligand or {}).get("smiles")
            or ((ligand or {}).get("meta") or {}).get("smiles")
            or best.get("ligand_smiles")
            or ""
        )
        receptor = project.get_receptor(best.get("receptor_id") or "")
        pdb_path = (
            project.root / (receptor.get("source") or receptor.get("pdb"))
            if receptor else _result_receptor_pdb(best)
        )
        receptor_meta = (receptor or {}).get("meta") or {}
        assays = project.data.get("assays") or {}
    payload = trust.assess(
        pdb_path,
        smiles,
        run.get("results", []),
        assay_payload=assays,
        receptor_meta=receptor_meta,
    )
    payload["run_id"] = run_id
    payload["ligand"] = best.get("ligand_name")
    return payload


@app.get("/api/runs/{run_id}/mmp")
def run_mmp(run_id: str):
    """Matched molecular pair SAR rules mined from the current project."""
    with STATE.lock:
        project = STATE.project
        run = project.get_run(run_id)
        if not run:
            raise HTTPException(404, "对接记录不存在")
        payload = mmp.mine(
            project.data.get("ligands", []),
            run.get("results", []),
            project.data.get("assays") or {},
        )
        payload["run_id"] = run_id
        return payload


@app.get("/api/runs/{run_id}/active")
def run_active_learning(run_id: str, batch: int = 12):
    """Recommend the next experimental batch by Expected Improvement."""
    with STATE.lock:
        project = STATE.project
        run = project.get_run(run_id)
        if not run:
            raise HTTPException(404, "对接记录不存在")
        index = {item["id"]: item for item in project.data.get("ligands", [])}
        records = []
        for item in run.get("results", []):
            if item.get("status") != "ok" or item.get("affinity") is None:
                continue
            ligand = index.get(item.get("ligand_id"))
            if not ligand:
                continue
            smiles = (
                ligand.get("smiles")
                or (ligand.get("meta") or {}).get("smiles")
                or item.get("ligand_smiles")
                or ""
            )
            if not smiles:
                continue
            records.append(
                {
                    "ligand_id": ligand.get("id"),
                    "name": ligand.get("name") or item.get("ligand_name") or "",
                    "smiles": smiles,
                    "affinity": float(item["affinity"]),
                }
            )
        assays = project.data.get("assays") or {}
    payload = activelearn.recommend(
        records,
        assay_payload=assays,
        batch=max(1, min(int(batch or 12), 40)),
    )
    payload["run_id"] = run_id
    return payload


@app.post("/api/runs/{run_id}/benchmark")
async def run_benchmark(run_id: str, file: UploadFile = File(...)):
    """Retrospective calibration: actives vs decoys in one docking run."""
    raw = await file.read()
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(400, "活性数据文件超过 20 MB")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("gbk", errors="replace")
    rows = assay.parse_table(text)
    if not rows:
        raise HTTPException(
            400,
            "未解析到活性数据：需要名称或 SMILES 列，以及 IC50/Kd/EC50/Ki 数值列与单位。",
        )
    with STATE.lock:
        project = STATE.project
        run = project.get_run(run_id)
        if not run:
            raise HTTPException(404, "对接记录不存在")
        ligands = project.data.get("ligands", [])
        try:
            payload = benchmark.analyse(run, ligands, rows)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        payload["file"] = file.filename or ""
        benchmarks = project.data.setdefault("benchmarks", {})
        benchmarks[run_id] = payload
        project.save()
    return payload


@app.get("/api/runs/{run_id}/benchmark")
def get_benchmark(run_id: str):
    with STATE.lock:
        payload = (STATE.project.data.get("benchmarks") or {}).get(run_id)
    if not payload:
        raise HTTPException(404, "该记录尚无回溯校准结果")
    return payload


@app.post("/api/pbpk")
def pbpk_endpoint(payload: dict):
    """Seven-species PBPK simulation with optional Monte-Carlo withdrawal."""
    smiles = str(payload.get("smiles") or "").strip()
    mw = payload.get("mw")
    logp = payload.get("logp")
    if smiles and (mw is None or logp is None):
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mw = mw if mw is not None else float(Descriptors.MolWt(mol))
            logp = logp if logp is not None else float(Crippen.MolLogP(mol))
    try:
        mw = float(mw or 0)
        logp = float(logp or 0)
    except (TypeError, ValueError):
        raise HTTPException(400, "需要有效的分子量与 logP") from None
    if mw <= 0:
        raise HTTPException(400, "需要有效的分子量")
    species = str(payload.get("species") or "human")
    result = pbpk.simulate(
        dose_mg_kg=float(payload.get("dose_mg_kg") or 5),
        mw=mw,
        logp=logp,
        fu=float(payload.get("fu") or 0.2),
        cl_ml_min_kg=float(payload.get("cl_ml_min_kg") or 5),
        species=species,
        route=str(payload.get("route") or "oral"),
        f_percent=float(payload.get("f_percent") or 50),
        ka=float(payload.get("ka") or 1.2),
        duration_days=float(payload.get("duration_days") or 14),
        mrl_ug_kg=payload.get("mrl_ug_kg") or None,
    )
    if payload.get("monte_carlo", True):
        result["monte_carlo"] = pbpk.withdrawal_monte_carlo(
            dose_mg_kg=float(payload.get("dose_mg_kg") or 5),
            mw=mw,
            logp=logp,
            species=species,
            route=str(payload.get("route") or "oral"),
            f_percent=float(payload.get("f_percent") or 50),
            fu=float(payload.get("fu") or 0.2),
            cl_ml_min_kg=float(payload.get("cl_ml_min_kg") or 5),
            ka=float(payload.get("ka") or 1.2),
            duration_days=float(payload.get("duration_days") or 14),
            mrl_ug_kg=payload.get("mrl_ug_kg") or None,
            samples=int(payload.get("samples") or 60),
        )
    result["smiles"] = smiles
    with STATE.lock:
        STATE.project.data["pbpk"] = result
        STATE.project.save()
    return result


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------
@app.post("/api/tools/enumerate")
def enumerate_library(payload: dict):
    smiles = str(payload.get("smiles") or "").strip()
    substituents = payload.get("substituents") or ["C", "O", "N", "F", "Cl", "C(F)(F)F", "C#N", "OC"]
    limit = int(payload.get("limit", 300))
    if "[R]" not in smiles and "R" not in smiles:
        raise HTTPException(400, "SMILES 中需要包含 R 或 [R] 占位符")
    from rdkit import Chem

    results = []
    clean_subs = [s.strip() for s in substituents if str(s).strip()]
    for i, sub in enumerate(clean_subs, start=1):
        smi = smiles.replace("[R]", sub).replace("R", sub)
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        results.append({"name": f"R{i}_{safe_name(sub, 'sub')}", "smiles": Chem.MolToSmiles(mol)})
        if len(results) >= limit:
            break
    return {"ok": True, "count": len(results), "molecules": results}
