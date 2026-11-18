"""Geometric pharmacophore perception from a docked pose.

Features are perceived with the RDKit chemical feature definitions, projected
onto the docked 3D coordinates and annotated with the receptor residues that
line each feature.  The result describes why a pose is complementary to the
pocket and which pharmacophore points are already satisfied.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures
from scipy.spatial import cKDTree

from . import analysis, i18n, pdbio
from .utils import safe_name

FAMILIES = {
    "Donor": ("氢键供体", "#1d4ed8"),
    "Acceptor": ("氢键受体", "#b91c1c"),
    "Aromatic": ("芳香中心", "#7e22ce"),
    "Hydrophobe": ("疏水中心", "#0f766e"),
    "LumpedHydrophobe": ("疏水中心", "#0f766e"),
    "PosIonizable": ("正电中心", "#c2410c"),
    "NegIonizable": ("负电中心", "#0369a1"),
    "ZnBinder": ("金属结合", "#475569"),
}

_ACCEPTOR_ELEMENTS = {"O", "N", "S"}
_DONOR_RESIDUES = {"SER", "THR", "TYR", "ASN", "GLN", "HIS", "TRP", "CYS", "LYS", "ARG"}
_ACIDIC = {"ASP", "GLU"}
_BASIC = {"LYS", "ARG", "HIS"}


def _factory():
    src = Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"
    if not src.exists():
        return None
    target = src
    # RDKit 的 C++ 层用窄字符 fopen 打开文件，含中文的安装路径会失败；
    # 此时把定义文件复制到纯 ASCII 的临时目录再加载。
    try:
        str(target).encode("ascii")
    except UnicodeEncodeError:
        cached = Path(tempfile.gettempdir()) / "dockforge_BaseFeatures.fdef"
        if not cached.exists() or cached.stat().st_mtime < src.stat().st_mtime:
            shutil.copyfile(src, cached)
        target = cached
    return ChemicalFeatures.BuildFeatureFactory(str(target))


def extract(
    pose_path: Path,
    receptor_pdb: Path,
    mode: int = 1,
    interaction: dict | None = None,
) -> dict:
    mol, conf_id = analysis._load_pose_mol(pose_path, mode)
    factory = _factory()
    raw = factory.GetFeaturesForMol(mol) if factory is not None else []
    conf = mol.GetConformer(conf_id)

    st = pdbio.parse_pdb(receptor_pdb.read_text(encoding="utf-8", errors="replace"))
    protein = [a for a in st.atoms if a.element != "H"]
    if not protein:
        raise ValueError("受体结构为空")
    coords = np.array([[a.x, a.y, a.z] for a in protein])
    tree = cKDTree(coords)
    residues: dict[tuple, list] = {}
    for atom in protein:
        residues.setdefault((atom.chain, atom.resseq, atom.resname), []).append(atom)

    features = []
    for feat in raw:
        family = feat.GetFamily()
        if family not in FAMILIES:
            continue
        atom_ids = list(feat.GetAtomIds())
        if not atom_ids:
            continue
        points = [
            conf.GetAtomPosition(idx) for idx in atom_ids
            if idx < mol.GetNumAtoms()
        ]
        if not points:
            continue
        centroid = np.array(
            [sum(p.x for p in points) / len(points),
             sum(p.y for p in points) / len(points),
             sum(p.z for p in points) / len(points)]
        )
        contacts = []
        best = 99.0
        for j in tree.query_ball_point(centroid, 5.0):
            atom = protein[j]
            dist = float(np.linalg.norm(coords[j] - centroid))
            best = min(best, dist)
            partner = _classify_complement(family, atom)
            contacts.append(
                {
                    "residue": atom.resname,
                    "chain": atom.chain,
                    "resseq": atom.resseq,
                    "atom": atom.name,
                    "distance": round(dist, 2),
                    "complement": partner,
                }
            )
        contacts.sort(key=lambda item: item["distance"])
        seen = set()
        unique = []
        for item in contacts:
            key = (item["chain"], item["resseq"], item["residue"])
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        label, color = FAMILIES[family]
        features.append(
            {
                "family": family,
                "type": feat.GetType(),
                "label": label,
                "color": color,
                "atoms": atom_ids,
                "centroid": [round(float(v), 3) for v in centroid],
                "contacts": unique[:4],
                "min_distance": round(best, 2) if best < 99 else None,
                "matched": bool(unique) and best <= 4.6,
            }
        )

    counts = {}
    for item in features:
        counts[item["label"]] = counts.get(item["label"], 0) + 1
    matched = sum(1 for item in features if item["matched"])
    coverage = round(100.0 * matched / len(features), 1) if features else 0.0

    unsatisfied = [
        {
            "label": item["label"],
            "color": item["color"],
            "residue": (item["contacts"][0]["residue"] + str(item["contacts"][0]["resseq"]))
            if item["contacts"] else "无",
            "distance": item["min_distance"],
        }
        for item in features
        if not item["matched"]
    ]

    return {
        "available": bool(features),
        "method": "RDKit BaseFeatures + 几何互补判定",
        "features": features,
        "counts": counts,
        "total": len(features),
        "matched": matched,
        "coverage": coverage,
        "unsatisfied": unsatisfied,
        "interaction_counts": (interaction or {}).get("counts", {}),
    }


def _classify_complement(family: str, atom) -> str:
    if family in ("Donor",):
        return "互补" if atom.element in _ACCEPTOR_ELEMENTS else "邻近"
    if family in ("Acceptor",):
        return "互补" if atom.element in ("N", "O") else "邻近"
    if family == "Aromatic":
        return "π-堆积" if atom.element in ("C", "N") and atom.name in ("CZ", "CG", "CD1", "CE1", "CH2") else "邻近"
    if family in ("Hydrophobe", "LumpedHydrophobe"):
        return "疏水" if atom.element in ("C", "S") else "邻近"
    if family == "PosIonizable":
        return "盐桥" if atom.resname in _ACIDIC else "邻近"
    if family == "NegIonizable":
        return "盐桥" if atom.resname in _BASIC else "邻近"
    return "邻近"


def pharmacophore_pdf(data: dict, name: str, receptor: str, smiles: str, outpath: Path) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    from . import figures

    figures._style_cjk()
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    mol = Chem.MolFromSmiles(smiles) if smiles else None

    fig = plt.figure(figsize=(8.27, 11.69))
    grid = fig.add_gridspec(
        2, 2, height_ratios=[1.0, 0.72],
        width_ratios=[1.18, 1.0],
        hspace=0.26, wspace=0.24,
        left=0.07, right=0.95, top=0.885, bottom=0.07,
    )
    fig.text(0.07, 0.955, "药效团分析", fontsize=20, fontweight="bold", color=figures.INK)
    fig.text(
        0.07, 0.928,
        f"{name} × {receptor}   ·   {data['total']} 个特征   ·   互补覆盖 {data['coverage']:.0f}%",
        fontsize=11.5, color=figures.MUTED,
    )

    # ---- 2D structure with feature markers ------------------------------
    ax = fig.add_subplot(grid[0, 0])
    if mol is not None:
        depict = figures._mol_with_2d(mol)
        figures._draw_molecule(ax, depict)
        # Keep the depiction inside its grid cell even for large macrocycles.
        ax.set_aspect("equal", adjustable="datalim")
        conf = depict.GetConformer()
        for index, item in enumerate(data["features"], start=1):
            ids = [idx for idx in item["atoms"] if idx < depict.GetNumAtoms()]
            if not ids:
                continue
            px = sum(conf.GetAtomPosition(idx).x for idx in ids) / len(ids)
            py = sum(conf.GetAtomPosition(idx).y for idx in ids) / len(ids)
            ax.add_patch(
                Circle(
                    (px, py), 0.42, fill=False,
                    edgecolor=item["color"], linewidth=2.0, alpha=0.85, zorder=5,
                )
            )
            offset = ((0.34, 0.34), (-0.34, 0.34), (0.34, -0.34), (-0.34, -0.34))[
                (index - 1) % 4
            ]
            ax.text(
                px + offset[0], py + offset[1], str(index), fontsize=7.8,
                fontweight="bold", color=item["color"], ha="center",
                va="center", zorder=7,
                bbox={
                    "boxstyle": "circle,pad=0.16", "facecolor": "white",
                    "edgecolor": item["color"], "linewidth": 1.0, "alpha": 0.95,
                },
            )
        ax.set_title("二维特征映射", fontsize=12, loc="left", pad=8)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "缺少结构信息", ha="center", color=figures.MUTED)

    # ---- summary + family counts -----------------------------------------
    ax = fig.add_subplot(grid[0, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.0, 0.94, "互补性摘要", fontsize=13, fontweight="bold", va="top", color=figures.INK)
    metrics = [
        ("药效团特征", f"{data['total']}"),
        ("已满足", f"{data['matched']}"),
        ("覆盖率", f"{data['coverage']:.0f}%"),
        ("未满足", f"{len(data['unsatisfied'])}"),
    ]
    for i, (label, value) in enumerate(metrics):
        y = 0.83 - i * 0.11
        if i18n.get_lang() == "en":
            label = figures._trunc(label, 20)
            ax.text(0.0, y, label, fontsize=10.2, color=figures.MUTED, va="top")
            ax.text(0.98, y, value, fontsize=14, fontweight="bold",
                    color=figures.INK, va="top", ha="right")
        else:
            ax.text(0.0, y, label, fontsize=11, color=figures.MUTED, va="top")
            ax.text(0.52, y, value, fontsize=15, fontweight="bold", color=figures.INK, va="top")
    y = 0.42
    ax.text(0.0, y, "特征构成", fontsize=12, fontweight="bold", color=figures.INK, va="top")
    for i, (label, count) in enumerate(sorted(data["counts"].items(), key=lambda kv: -kv[1])[:6]):
        y -= 0.045
        ax.text(0.02, y, figures._trunc(label, 16) if i18n.get_lang() == "en" else label,
                fontsize=10.5, color=figures.MUTED, va="top")
        ax.plot([0.52, 0.52 + 0.32 * count / max(1, data["total"])], [y - 0.016] * 2,
                color=figures.TEAL, linewidth=6, solid_capstyle="round")
        ax.text(0.90, y, str(count), fontsize=10, fontweight="bold", color=figures.INK, va="top")

    # ---- feature table ----------------------------------------------------
    ax = fig.add_subplot(grid[1, :])
    ax.axis("off")
    ax.set_title("特征–残基对应关系（前 8 个）", fontsize=13, loc="left", pad=12)
    header = ["#", "特征", "配体原子", "最近残基", "距离 (Å)", "互补性"]
    rows = []
    for index, item in enumerate(data["features"][:8], start=1):
        contact = item["contacts"][0] if item["contacts"] else {}
        residue = (
            f"{contact.get('residue', '—')} {contact.get('chain', '')}{contact.get('resseq', '')}"
            if contact else "无接触"
        )
        rows.append(
            [
                str(index),
                item["label"],
                ", ".join(str(idx + 1) for idx in item["atoms"][:4]),
                residue,
                f"{item['min_distance']:.2f}" if item["min_distance"] is not None else "—",
                contact.get("complement", "—") if contact else "—",
            ]
        )
    if rows:
        table = ax.table(
            cellText=rows, colLabels=header,
            colWidths=[0.05, 0.20, 0.20, 0.27, 0.14, 0.14],
            loc="upper center", cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9.5)
        table.scale(1, 1.5)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor(figures.GRID)
            cell.set_linewidth(0.6)
            if row == 0:
                cell.set_facecolor("#f1f5f9")
                cell.set_text_props(fontweight="bold", color=figures.INK)
            elif col == 1:
                cell.set_text_props(color=figures.TEAL, fontweight="bold")
    if data["unsatisfied"]:
        ax.text(
            0.0, 0.02,
            "未满足特征：" + "、".join(
                f"{item['label']}({item['residue']})" for item in data["unsatisfied"][:6]
            ),
            fontsize=9.5, color=figures.AMBER, va="bottom", transform=ax.transAxes,
        )

    fig.text(
        0.07, 0.020,
        figures._wrap_cjk(
            "方法：RDKit BaseFeatures 感知供体/受体/芳香/疏水/可电离中心，"
            "按 5 Å 邻域判定残基互补性。",
            116,
        ),
        fontsize=8.4, color=figures.MUTED, va="bottom",
    )
    fig.savefig(outpath, dpi=600, facecolor="white")
    plt.close(fig)
    return outpath


def filename(name: str) -> str:
    return f"{safe_name(name, 'ligand')}_pharmacophore.pdf"
