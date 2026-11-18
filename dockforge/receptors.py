"""Receptor retrieval, cleaning and PDBQT preparation."""
from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

from . import pdbio
from .boxes import box_from_coords, detect_pockets
from .config import VINA_EXE, python_exe
from .utils import run_stream, tail_lines

RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
LIGAND_SDF_URL = "https://files.rcsb.org/ligands/download/{het_id}_ideal.sdf"

_COMMON_ADDITIVES = {
    "NAG", "GOL", "EDO", "SO4", "PO4", "ACT", "DMS", "MPD", "PEG", "FMT",
    "CIT", "TLA", "IMD", "BME", "MSE", "CL", "BR", "IOD", "NO3", "NH4",
}


def fetch_pdb(pdb_id: str) -> str:
    pdb_id = pdb_id.strip().upper()
    if len(pdb_id) != 4 or not pdb_id.isalnum():
        raise ValueError("PDB ID 必须是 4 位字母或数字，例如 3PTB")
    url = RCSB_URL.format(pdb_id=pdb_id)
    request = urllib.request.Request(url, headers={"User-Agent": "DrugHX/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            data = response.read()
    except urllib.error.HTTPError as exc:
        raise ValueError(f"RCSB 上找不到 {pdb_id}（HTTP {exc.code}）") from exc
    except Exception as exc:
        raise ValueError(f"下载 {pdb_id} 失败：{exc}") from exc
    text = data.decode("utf-8", errors="replace")
    if "ATOM" not in text:
        raise ValueError(f"{pdb_id} 下载内容不是有效的 PDB 文件")
    return text


def fetch_ligand_ideal_sdf(het_id: str) -> str:
    """Fetch the chemically correct ideal SDF for a PDB chemical component."""
    het_id = het_id.strip().upper()
    if not het_id or len(het_id) > 3:
        raise ValueError("无效的配体编号")
    url = LIGAND_SDF_URL.format(het_id=het_id)
    request = urllib.request.Request(url, headers={"User-Agent": "DrugHX/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        text = response.read().decode("utf-8", errors="replace")
    if "V2000" not in text and "V3000" not in text:
        raise ValueError("RCSB 返回的不是 SDF")
    return text


def prepare_receptor(
    rec_dir: Path,
    original_pdb: str,
    options: dict,
    log,
    hetero: list[dict] | None = None,
) -> dict:
    """Clean a receptor and convert it to PDBQT with Meeko.

    Returns metadata to merge into the project receptor record.
    """
    rec_dir.mkdir(parents=True, exist_ok=True)
    remove_hetero = bool(options.get("remove_hetero", True))
    keep_ions = bool(options.get("keep_ions", False))
    clean = pdbio.clean_pdb(
        original_pdb,
        chains=options.get("chains") or [],
        remove_water=bool(options.get("remove_water", True)),
        remove_hetero=remove_hetero,
        keep_ions=keep_ions and not remove_hetero,
        altloc=options.get("altloc", "A") or "A",
    )
    clean_path = rec_dir / "receptor_clean.pdb"
    clean_path.write_text(clean["pdb"], encoding="utf-8")

    # Ions are appended separately only when hetero records were stripped;
    # otherwise clean_pdb already kept them (when requested).
    ions = []
    if keep_ions and remove_hetero:
        ions = pdbio.extract_ions(original_pdb)
    view_pdb = clean["pdb"]
    if ions:
        lines = view_pdb.rstrip().splitlines()
        if lines and lines[-1] == "END":
            lines.pop()
        if lines and lines[-1] == "TER":
            pass
        lines.extend(a.line[:66] for a in ions)
        lines.append("END")
        view_pdb = "\n".join(lines) + "\n"
    view_path = rec_dir / "receptor.pdb"
    view_path.write_text(view_pdb, encoding="utf-8")

    log(f"清洁受体：{clean['atoms']} 个原子，{clean['residues']} 个残基，链 {','.join(clean['chains']) or '-'}")
    if ions:
        log(f"保留金属/离子：{len(ions)} 个原子")

    base = rec_dir / "receptor"
    pdbqt_path = rec_dir / "receptor.pdbqt"
    cmd = [
        python_exe(), "-m", "meeko.cli.mk_prepare_receptor",
        "--read_pdb", str(clean_path),
        "-o", str(base),
        "-p",
        "-x",
    ]
    if options.get("charge_model") == "gasteiger":
        cmd += ["--compute_charges", "--charge_model", "gasteiger"]

    log("运行 Meeko 受体准备 ...")
    code, lines = run_stream(cmd, cwd=rec_dir, on_line=log, timeout=900)
    if code != 0 or not pdbqt_path.exists():
        raise RuntimeError("Meeko 受体准备失败：\n" + tail_lines(lines, 25))

    pdbqt = pdbqt_path.read_text(encoding="utf-8", errors="replace")
    if ions:
        serial = 1
        for line in pdbqt.splitlines():
            if line.startswith(("ATOM", "HETATM")):
                serial = max(serial, int(line[6:11].strip() or 0) + 1)
        extra = []
        for ion in ions:
            ion.serial = serial
            extra.append(pdbio.format_ion_pdbqt(ion))
            serial += 1
        pdbqt = pdbqt.rstrip() + "\n" + "\n".join(extra) + "\n"
        pdbqt_path.write_text(pdbqt, encoding="utf-8")

    stats = pdbio.pdbqt_stats(pdbqt)
    if stats["atoms"] == 0:
        raise RuntimeError("受体 PDBQT 为空，请检查输入结构")

    receptor_info = pdbio.pdb_info(original_pdb)
    log("识别结合口袋 ...")
    pockets = detect_pockets(view_pdb, top=5)
    if pockets:
        log(f"口袋预测完成：{len(pockets)} 个候选口袋")

    box = None
    hetero = hetero or pdbio.hetero_ligands(original_pdb)
    if hetero:
        # A co-crystallised ligand is the most reliable default site.  Ignore
        # cryoprotectants/additives (DMS, glycerol, sulfate, ...): when only
        # those are present, use the predicted pocket instead.
        candidates = [
            h for h in hetero if str(h.get("name", "")).upper() not in _COMMON_ADDITIVES
        ]
        if candidates:
            ref = max(candidates, key=lambda item: item.get("heavy", 0))
            block = pdbio.extract_residue_pdb(
                original_pdb,
                ref.get("chain", ""),
                int(ref.get("resseq", 0)),
                ref.get("icode", ""),
                ref.get("name", ""),
            )
            coords = [
                [a.x, a.y, a.z]
                for a in pdbio.parse_pdb(block).atoms
                if a.element != "H"
            ]
            if coords:
                box = box_from_coords(coords, padding=6.0, min_size=20.0)
                box["source"] = "ligand"
                box["label"] = f"共晶配体 {ref.get('name', '')}"
                log(f"默认对接盒来自共晶配体 {ref.get('name', '')}（{ref.get('heavy', 0)} 个重原子）")
    if box is None and pockets:
        box = {
            "center": pockets[0]["center"],
            "size": pockets[0]["size"],
            "source": "pocket",
            "label": pockets[0]["label"],
        }

    return {
        "pdb": str(view_path),
        "pdb_clean": str(clean_path),
        "pdbqt": str(pdbqt_path),
        "meta": {
            **receptor_info,
            "atoms": clean["atoms"],
            "residues": clean["residues"],
            "chains": clean["chains"],
            "pdbqt_atoms": stats["atoms"],
            "ions": len(ions),
        },
        "pockets": pockets,
        "box": box,
    }


def vina_version() -> str | None:
    if not VINA_EXE.is_file():
        return None
    try:
        code, lines = run_stream([str(VINA_EXE), "--version"], timeout=20)
        for line in lines:
            if "AutoDock Vina" in line:
                return line.strip()
    except Exception:
        return None
    return None
