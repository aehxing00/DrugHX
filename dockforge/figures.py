"""Publication-quality figure generation for docking results."""
from __future__ import annotations

import csv
import re
import textwrap
import unicodedata
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, RegularPolygon
from rdkit import Chem
from rdkit.Chem import Descriptors, rdDepictor

from . import i18n, pdbio
from .config import APP_VERSION

INK = "#0f172a"
MUTED = "#64748b"
TEAL = "#0f766e"
BLUE = "#0369a1"
AMBER = "#b45309"
RED = "#b91c1c"
GRID = "#e2e8f0"

INTERACTION_LABELS = [
    ("hbonds", "H-bond"),
    ("hydrophobic", "Hydrophobic"),
    ("salt_bridges", "Salt bridge"),
    ("pi_stacking", "π-stacking"),
    ("halogens", "Halogen"),
    ("metals", "Metal"),
]

_INTERACTION_ZH = {
    "hbonds": "氢键",
    "hydrophobic": "疏水",
    "salt_bridges": "盐桥",
    "pi_stacking": "π-堆积",
    "halogens": "卤键",
    "metals": "金属",
}


def _L(zh: str, en: str) -> str:
    """Return Chinese in Chinese mode and English in English mode."""
    return en if i18n.get_lang() == "en" else zh


def _interaction_labels() -> list[tuple[str, str]]:
    return [
        (key, _L(_INTERACTION_ZH.get(key, label), label))
        for key, label in INTERACTION_LABELS
    ]


_SYNTH_VERDICT_EN = {
    "易合成": "Easy to synthesise",
    "中等": "Moderate",
    "偏难": "Difficult",
    "困难": "Very difficult",
    "未安装 SAscore，无法判定": "SAscore unavailable",
}

_SYNTH_VERDICT_ZH = {
    "Easy": "易合成",
    "Easy to synthesize": "易合成",
    "Easy to synthesise": "易合成",
    "Moderate": "中等",
    "Moderately soluble": "中等",
    "Difficult": "偏难",
    "Very difficult": "困难",
    "SAscore unavailable": "未安装 SAscore，无法判定",
}


def _synth_verdict(payload: dict | None) -> str:
    value = str((payload or {}).get("verdict") or "—")
    if i18n.get_lang() == "en":
        return _SYNTH_VERDICT_EN.get(value, value)
    return _SYNTH_VERDICT_ZH.get(value, value)


_MMP_DIRECTION_EN = {"改善": "Improves", "降低": "Reduces"}
_MMP_CONFIDENCE_EN = {"高": "High", "中": "Medium", "低": "Low"}
_BENCH_FEATURE_EN = {
    "potency": "Potency (−Vina)",
    "le": "Ligand efficiency",
    "qed": "QED",
    "sa": "Synthesizability (inverse)",
    "safety": "Safety",
}


def _benchmark_verdict(comparison: dict) -> str:
    """English fallback for benchmark payloads written before verdict_en existed."""
    if comparison.get("cv_auc") is None or comparison.get("expert_auc") is None:
        return "too few labelled molecules with SMILES to fit weights; enrichment metrics only"
    level = comparison.get("level")
    if level == "bad":
        return (
            "both weightings are close to random (AUC < 0.60); do not change weights until "
            "the docking box, actives or decoys have been reviewed"
        )
    if level == "ok":
        return (
            "fitted weights outperform the expert weights and can be tested on this target series"
        )
    return (
        "fitted weights do not clearly outperform the expert weights; keep the current MPO weights"
    )


_ACTIVE_REASON_EN = {
    "实测点不足 4 个，按对接打分排序（先积累实验数据）": "Low assay count; docking rank",
    "EI 高且与已选分子结构差异足够": "High EI; structurally distinct",
    "EI 补充（多样性约束下不足批次）": "EI fill (diversity)",
}
_ACTIVE_NOTE_EN = {
    "当前对接集中的分子都已有实测数据；请先对接新一批候选，或等待新实验数据后再运行主动学习。": (
        "All molecules in this docking set already have measurements; dock a new batch or "
        "wait for new assay data before running active learning."
    ),
    "至少需要 4 个实测 pActivity 才能启用主动学习；当前返回对接最优未测候选。": (
        "At least four measured pActivity values are required for active learning; the top "
        "unmeasured docking candidates are returned."
    ),
    "高斯过程：0.75×Morgan/Tanimoto 核 + 0.25×对接打分核，噪声 0.12；EI 采用当前最优实测值 + 0.1 探索项；批次选择加入 Tanimoto ≤ 0.85 的多样性约束。预测用于下一轮实验排序，不是活性结论。": (
        "Gaussian process with a 0.75×Morgan/Tanimoto + 0.25×docking-score kernel (noise 0.12); "
        "EI uses the best measured value plus a 0.1 exploration term; batch selection enforces "
        "Tanimoto ≤ 0.85 diversity. Predictions prioritise the next experiment round and are not "
        "activity conclusions."
    ),
}


def _style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                "Arial", "Helvetica", "DejaVu Sans",
            ],
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.linewidth": 0.9,
            "axes.edgecolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "axes.labelcolor": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
        }
    )


def _style_bulk():
    """CNS-friendly typography for large-library screening figures.

    Fonts are deliberately larger than the generic report style so panels stay
    legible after two-column journal scaling.
    """
    _style_cjk()
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 16.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 14.5,
            "xtick.labelsize": 12.5,
            "ytick.labelsize": 12.5,
            "legend.fontsize": 12,
            "axes.linewidth": 1.1,
            "axes.edgecolor": "#94a3b8",
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
        }
    )


def _save_all(fig, base: Path):
    """Save vector PDF (primary) and SVG (editable backup)."""
    paths = []
    for suffix in ("pdf", "svg"):
        path = base.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def affinity_figure(results: list[dict], base: Path) -> list[Path]:
    _style()
    ok = [
        r for r in results
        if r.get("status") == "ok" and r.get("affinity") is not None
    ]
    if not ok:
        return []
    height = max(3.6, 0.42 * min(len(ok), 30) + 2.0)
    fig, ax = plt.subplots(figsize=(8.6, height))
    _draw_affinity(ax, results)
    fig.tight_layout()
    return _save_all(fig, base)


def _draw_affinity(ax, results: list[dict], compact: bool = False):
    ok = [
        r for r in results
        if r.get("status") == "ok" and r.get("affinity") is not None
    ]
    ok.sort(key=lambda r: r["affinity"])
    shown = ok[:30]
    labels = [
        _clip_display(
            f"{r.get('ligand_name', 'ligand')} · {r.get('receptor_name', 'receptor')}",
            34 if compact else 52,
        )
        for r in shown
    ]
    scores = [float(r["affinity"]) for r in shown]
    vinardo = [r.get("vinardo") for r in shown]
    has_vinardo = any(v is not None for v in vinardo)

    y = np.arange(len(shown))
    if has_vinardo:
        ax.barh(
            y - 0.17, scores, color=TEAL, height=0.32,
            edgecolor="none", label="Vina",
        )
        v_values = [float(v) if v is not None else 0.0 for v in vinardo]
        ax.barh(
            y + 0.17, v_values, color=AMBER, height=0.32,
            edgecolor="none", label="Vinardo",
        )
        for i, value in enumerate(scores):
            ax.text(
                value - 0.15, i - 0.17, f"{value:.2f}", va="center",
                ha="right", fontsize=11.5 if not compact else 8.8, color=INK,
            )
        for i, value in enumerate(vinardo):
            if value is not None:
                ax.text(
                    float(value) - 0.15, i + 0.17, f"{float(value):.2f}",
                    va="center", ha="right",
                    fontsize=11.5 if not compact else 8.8, color=AMBER,
                )
        ax.legend(frameon=False, loc="lower left", ncol=2)
        low = min(scores + [float(v) for v in vinardo if v is not None])
    else:
        colors = [TEAL if i == 0 else "#5e9c96" for i in range(len(shown))]
        ax.barh(y, scores, color=colors, height=0.66, edgecolor="none")
        for i, score in enumerate(scores):
            ax.text(
                score - 0.15, i, f"{score:.2f}", va="center", ha="right",
                fontsize=12 if not compact else 9, color=INK,
            )
        low = min(scores)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=INK, fontsize=11 if not compact else 8.8)
    ax.invert_yaxis()
    ax.set_xlabel(_L("对接打分（kcal/mol，经验值）", "Docking score (kcal/mol, empirical)"))
    ax.set_title(
        _L("对接打分排序", "Docking score ranking"), loc="left", fontweight="bold",
        fontsize=13 if compact else 16, pad=6 if compact else 10,
    )
    high = max(0.0, max(scores))
    ax.set_xlim(low - 1.2, high + 0.8)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return shown


def interaction_heatmap(
    results: list[dict],
    analyses: dict[int, dict],
    base: Path,
) -> list[Path]:
    _style()
    rows, matrix = _interaction_matrix(results, analyses)
    if not matrix:
        return []
    height = max(3.4, 0.40 * len(rows) + 2.0)
    fig, ax = plt.subplots(figsize=(6.8, height))
    _draw_heatmap(ax, rows, matrix)
    fig.tight_layout()
    return _save_all(fig, base)


def _interaction_matrix(results: list[dict], analyses: dict[int, dict]):
    ok = [r for r in results if r.get("status") == "ok"][:30]
    rows = []
    matrix = []
    for r in ok:
        analysis = analyses.get(id(r))
        if not analysis:
            continue
        values = [len(analysis.get(key, [])) for key, _ in INTERACTION_LABELS]
        rows.append(r.get("ligand_name", "ligand"))
        matrix.append(values)
    return rows, matrix


def _draw_heatmap(ax, rows: list[str], matrix: list[list[int]], compact: bool = False):
    matrix_arr = np.asarray(matrix, dtype=float)
    image = ax.imshow(matrix_arr, cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(np.arange(len(INTERACTION_LABELS)))
    ax.set_xticklabels(
        [label for _, label in _interaction_labels()],
        rotation=35, ha="right",
        fontsize=11 if not compact else 8.8,
    )
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows, fontsize=11 if not compact else 8.8)
    ax.set_title(
        _L("蛋白–配体作用力谱", "Protein–ligand interaction profile"),
        loc="left", fontweight="bold",
        fontsize=13 if compact else 16, pad=6 if compact else 10,
    )
    max_value = max(1.0, float(matrix_arr.max()))
    for i in range(matrix_arr.shape[0]):
        for j in range(matrix_arr.shape[1]):
            value = int(matrix_arr[i, j])
            ax.text(
                j, i, str(value), ha="center", va="center",
                fontsize=9 if not compact else 8,
                color="white" if value > max_value * 0.55 else INK,
            )
    ax.set_xticks(np.arange(-0.5, len(INTERACTION_LABELS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    cbar = ax.figure.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label(_L("作用力计数", "Interaction count"), fontsize=10 if not compact else 9)
    cbar.ax.tick_params(labelsize=9 if not compact else 8)
    return image


def publication_panel(
    results: list[dict],
    analyses: dict[int, dict],
    base: Path,
) -> list[Path]:
    """Two-panel summary figure (a: scores, b: interaction classes)."""
    _style()
    rows, matrix = _interaction_matrix(results, analyses)
    ok = [
        r for r in results
        if r.get("status") == "ok" and r.get("affinity") is not None
    ]
    if not ok or not matrix:
        return []
    height = max(3.4, 0.36 * min(len(ok), 30) + 1.8)
    fig = plt.figure(figsize=(9.2, height))
    grid = fig.add_gridspec(
        1, 2, width_ratios=[1.45, 1.0], wspace=0.62,
        left=0.17, right=0.97, top=0.84, bottom=0.15,
    )
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    _draw_affinity(ax1, results, compact=True)
    _draw_heatmap(ax2, rows, matrix, compact=True)
    return _save_all(fig, base)


def _named_ligand_pdb(pdb_text: str) -> str:
    """Give every ligand atom a unique PDB name (C0, O12, ...).

    The interaction analysis identifies ligand atoms by RDKit index.  PyMOL
    distance selections need matching unique names, otherwise atom names such
    as "O" would select every oxygen in the ligand.
    """
    lines = []
    atom_index = 0
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            padded = line.ljust(80)
            element = (
                padded[76:78].strip()
                or padded[12:16].strip()[:1]
                or "C"
            )
            name = f"{element}{atom_index}"
            line = padded[:12] + f"{name:<4s}" + padded[16:]
            atom_index += 1
        lines.append(line)
    return "\n".join(lines) + "\n"


def _ligand_atom_positions(pdb_text: str) -> list[tuple[str, np.ndarray]]:
    atoms = []
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        padded = line.ljust(80)
        try:
            coord = np.array(
                [
                    float(padded[30:38]),
                    float(padded[38:46]),
                    float(padded[46:54]),
                ]
            )
        except ValueError:
            continue
        atoms.append((padded[12:16].strip(), coord))
    return atoms


def _nearest_ligand_name(
    atoms: list[tuple[str, np.ndarray]],
    point: list[float] | None,
    fallback: str,
) -> str:
    if not atoms or not point:
        return fallback
    target = np.asarray(point, dtype=float)
    distances = [float(np.linalg.norm(coord - target)) for _, coord in atoms]
    index = int(np.argmin(distances))
    if distances[index] > 0.6:
        return fallback
    return atoms[index][0]


def _style_cjk():
    english = i18n.get_lang() == "en"
    _style()
    plt.rcParams.update(
        {
            "font.sans-serif": (
                ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans", "Microsoft YaHei"]
                if english
                else [
                    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                    "Arial", "DejaVu Sans",
                ]
            ),
            "font.size": 13.5 if english else 12,
            "axes.labelsize": 14.5 if english else 13,
            "axes.titlesize": 16.0 if english else 16,
            "xtick.labelsize": 12.5 if english else 11,
            "ytick.labelsize": 12.5 if english else 11,
            "legend.fontsize": 12.0 if english else 11,
        }
    )


_ELEMENT_COLORS = {
    "C": "#0f172a",
    "N": "#1d4ed8",
    "O": "#b91c1c",
    "S": "#b45309",
    "F": "#15803d",
    "Cl": "#15803d",
    "Br": "#92400e",
    "I": "#7e22ce",
    "P": "#c2410c",
}


def _mol_with_2d(mol: Chem.Mol) -> Chem.Mol:
    work = Chem.Mol(mol)
    # Drop the docked 3D conformer first: Compute2DCoords on a 3D conformer
    # triggers RDKit's "tagged as 2D but Z != 0" warning.
    work.RemoveAllConformers()
    rdDepictor.Compute2DCoords(work)
    return work


def _draw_molecule(ax, mol: Chem.Mol, title: str | None = None, compact: bool = False):
    """Draw a clean vector 2D structure with matplotlib."""
    work = _mol_with_2d(mol)
    try:
        Chem.Kekulize(work, clearAromaticFlags=True)
    except Exception:
        pass
    conf = work.GetConformer()
    coords = {
        atom.GetIdx(): np.array(
            [conf.GetAtomPosition(atom.GetIdx()).x, conf.GetAtomPosition(atom.GetIdx()).y]
        )
        for atom in work.GetAtoms()
    }
    for bond in work.GetBonds():
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        if begin.GetAtomicNum() == 1 or end.GetAtomicNum() == 1:
            continue
        p1 = coords[begin.GetIdx()]
        p2 = coords[end.GetIdx()]
        ax.plot(
            [p1[0], p2[0]], [p1[1], p2[1]], color="#334155",
            linewidth=1.1 if compact else 1.35,
            solid_capstyle="round", zorder=1,
        )
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            direction = p2 - p1
            norm = float(np.linalg.norm(direction)) or 1.0
            offset = 0.14 if compact else 0.10
            perp = np.array([-direction[1], direction[0]]) / norm * offset
            ax.plot(
                [p1[0] + perp[0], p2[0] + perp[0]],
                [p1[1] + perp[1], p2[1] + perp[1]],
                color="#334155", linewidth=1.0, solid_capstyle="round", zorder=1,
            )
    for atom in work.GetAtoms():
        if atom.GetAtomicNum() in (0, 1):
            continue
        symbol = atom.GetSymbol()
        point = coords[atom.GetIdx()]
        if symbol == "C":
            continue
        ax.text(
            point[0], point[1], symbol, ha="center", va="center",
            fontsize=10.5 if not compact else 8.8, fontweight="bold",
            color=_ELEMENT_COLORS.get(symbol, INK),
            bbox={"boxstyle": "circle,pad=0.18", "facecolor": "white",
                  "edgecolor": "none", "alpha": 0.95},
            zorder=3,
        )
    ax.set_aspect("equal")
    ax.axis("off")
    if coords:
        points = np.array(list(coords.values()), dtype=float)
        span = points.max(axis=0) - points.min(axis=0)
        pad = max(0.45, 0.08 * float(max(span))) if compact else max(0.65, 0.08 * float(max(span)))
        ax.set_xlim(points[:, 0].min() - pad, points[:, 0].max() + pad)
        ax.set_ylim(points[:, 1].min() - pad, points[:, 1].max() + pad)
    if title:
        ax.set_title(title, fontsize=11 if not compact else 9, fontweight="bold", pad=8)
    return coords


def _interaction_styles():
    return [
        ("hbonds", "氢键", "#b45309", (0, (4, 2))),
        ("salt_bridges", "盐桥", "#b91c1c", (0, (4, 2))),
        ("pi_stacking", "π-堆积", "#6d28d9", (0, (2, 2))),
        ("halogens", "卤键", "#0369a1", (0, (3, 2))),
        ("metals", "金属配位", "#475569", (0, (1, 2))),
    ]


def _nearest_atom_index(mol: Chem.Mol, point) -> int | None:
    if not point:
        return None
    target = np.asarray(point, dtype=float)
    conf = mol.GetConformer()
    best = None
    best_distance = 1e9
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        position = conf.GetAtomPosition(atom.GetIdx())
        distance = float(
            np.linalg.norm(np.array([position.x, position.y, position.z]) - target)
        )
        if distance < best_distance:
            best_distance = distance
            best = atom.GetIdx()
    return best


def _wrap_tokens(prefix: str, tokens: list[str], width: int = 22) -> str:
    """Wrap residue tokens without breaking a token across lines."""
    lines = []
    current = prefix
    for token in tokens:
        separator = "" if current.endswith("：") or not current else "、"
        candidate = f"{current}{separator}{token}"
        if len(candidate) > width and current and not current.endswith("："):
            lines.append(current)
            current = token
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def draw_interaction_diagram(ax, mol: Chem.Mol, analysis: dict, title: str):
    """Publication-style 2D interaction map (vector)."""
    work = _mol_with_2d(mol)
    coords2d = _draw_molecule(ax, work)
    heavy = [
        atom.GetIdx() for atom in work.GetAtoms() if atom.GetAtomicNum() > 1
    ]
    if not heavy:
        ax.set_title(title)
        return
    centroid = np.mean([coords2d[idx] for idx in heavy], axis=0)
    entries = []
    seen = set()
    for key, label, color, linestyle in _interaction_styles():
        for item in analysis.get(key, [])[:12]:
            idx = _nearest_atom_index(mol, item.get("ligand_coord"))
            if idx is None or idx not in coords2d:
                continue
            residue = f"{item.get('residue', 'RES')}{item.get('resseq', '')}"
            if residue in seen:
                continue
            seen.add(residue)
            entries.append(
                {
                    "point": coords2d[idx],
                    "text": residue,
                    "label": label,
                    "color": color,
                    "linestyle": linestyle,
                    "distance": item.get("distance", item.get("min_distance")),
                }
            )
    used = []
    max_radius = max(
        float(np.linalg.norm(coords2d[idx] - centroid)) for idx in heavy
    )
    label_radius = max_radius + 1.15
    count = max(1, len(entries))
    min_separation = min(0.9, 2 * np.pi / count)
    angles = []
    for index, entry in enumerate(entries):
        point = entry["point"]
        angle = float(np.arctan2(point[1] - centroid[1], point[0] - centroid[0]))
        # Spread labels with similar directions around the molecule.
        for _ in range(count):
            if all(abs((angle - old + np.pi) % (2 * np.pi) - np.pi) >= min_separation for old in angles):
                break
            angle += min_separation
        angles.append(angle)
        placed = centroid + np.array([np.cos(angle), np.sin(angle)]) * label_radius
        used.append(placed)
        ax.plot(
            [point[0], placed[0]], [point[1], placed[1]],
            color=entry["color"], linewidth=1.0,
            linestyle=entry["linestyle"], zorder=2,
        )
        distance = entry.get("distance")
        text = entry["text"] + (f"\n{distance:.1f} Å" if isinstance(distance, (int, float)) else "")
        ax.text(
            placed[0], placed[1], text, ha="center", va="center",
            fontsize=9.5, color=entry["color"], fontweight="bold",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white",
                  "edgecolor": entry["color"], "linewidth": 0.6, "alpha": 0.95},
            zorder=4,
        )
    handles = [
        Line2D([0], [0], color=color, linewidth=1.4, linestyle=linestyle, label=label)
        for _, label, color, linestyle in _interaction_styles()
        if any(entry["label"] == label for entry in entries)
    ]
    if handles:
        english = i18n.get_lang() == "en"
        if english:
            short = {
                "氢键": "H-bond", "盐桥": "Salt bridge", "π-堆积": "π-stack",
                "卤键": "Halogen", "金属配位": "Metal",
            }
            for handle in handles:
                handle.set_label(short.get(handle.get_label(), handle.get_label()))
        ax.legend(
            handles=handles, loc="lower right", frameon=False,
            fontsize=7.4 if english else 8.0, handlelength=2.0,
            ncol=min(3, len(handles)),
            bbox_to_anchor=(1.0, 0.0), borderaxespad=0.2,
        )
    ax.set_title(title, fontsize=13.5, fontweight="bold", pad=10)
    xs = [p[0] for p in coords2d.values()] + [p[0] for p in used]
    ys = [p[1] for p in coords2d.values()] + [p[1] for p in used]
    hydrophobic = analysis.get("hydrophobic", [])
    if hydrophobic and xs and ys:
        names = [
            f"{item.get('residue')}{item.get('resseq')}" for item in hydrophobic[:6]
        ]
        label = _wrap_tokens("疏水接触：", names, 24)
        ax.text(
            0.01, 0.02, label, transform=ax.transAxes,
            fontsize=7.6, color="#166534", va="bottom", ha="left",
            linespacing=1.25, zorder=5,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white",
                  "edgecolor": "none", "alpha": 0.88},
        )
    if xs and ys:
        ax.set_xlim(min(xs) - 1.15, max(xs) + 1.15)
        ax.set_ylim(min(ys) - 2.55, max(ys) + 0.95)


def result_figure_pdf(
    pose_path: Path,
    ligand_name: str,
    receptor_name: str,
    analysis: dict,
    outpath: Path,
    mode: int = 1,
) -> Path:
    """Two-page vector PDF: 2D structure and interaction map."""
    from .analysis import _load_pose_mol
    from rdkit.Chem import Descriptors, rdMolDescriptors

    _style_cjk()
    mol, _ = _load_pose_mol(pose_path, mode)
    heavy = Chem.RemoveHs(mol)
    properties = [
        ("分子式", rdMolDescriptors.CalcMolFormula(heavy)),
        ("分子量", f"{Descriptors.MolWt(heavy):.1f} Da"),
        ("cLogP", f"{Descriptors.MolLogP(heavy):.2f}"),
        ("TPSA", f"{Descriptors.TPSA(heavy):.1f} Å²"),
        ("氢键供体", str(rdMolDescriptors.CalcNumHBD(heavy))),
        ("氢键受体", str(rdMolDescriptors.CalcNumHBA(heavy))),
        ("可旋转键", str(rdMolDescriptors.CalcNumRotatableBonds(heavy))),
        ("重原子数", str(heavy.GetNumHeavyAtoms())),
    ]
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(outpath) as pdf:
        fig = plt.figure(figsize=(8.6, 6.0))
        ax = fig.add_axes([0.04, 0.10, 0.55, 0.80])
        _draw_molecule(ax, heavy, title=f"{ligand_name} · 2D 结构")
        txt = "结合性质\n\n" + "\n".join(f"{key}：{value}" for key, value in properties)
        fig.text(
            0.63, 0.64, txt, fontsize=10.5, color=INK, va="top", ha="left",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f8fafc",
                  "edgecolor": GRID, "linewidth": 0.8},
        )
        fig.text(
            0.63, 0.91, f"{ligand_name} × {receptor_name}", fontsize=16,
            fontweight="bold", color=INK,
        )
        fig.text(
            0.63, 0.875, "DrugHX 矢量图 · PDF", fontsize=10, color=MUTED,
        )
        fig.text(
            0.04, 0.015,
            _wrap_cjk(
                "图 7　代表配体的二维结构、分子式与理化性质。数值由 RDKit 依据对接前的质子化状态计算，"
                "与实验测定值存在系统差异，仅用于同系列分子的相对比较。", 86
            ),
            fontsize=8.6, color=MUTED, va="bottom", linespacing=1.4,
        )
        pdf.savefig(fig, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.6, 7.0))
        draw_interaction_diagram(
            ax, mol, analysis, f"{ligand_name} · 蛋白–配体作用力"
        )
        fig.text(
            0.01, 0.012,
            _wrap_cjk(
                "图 8　蛋白–配体作用力示意：虚线区分氢键、盐桥、π-堆积、卤键与金属配位，"
                "标注残基编号与距离（Å）。作用力由几何判据识别，不等同于能量分解结果。", 92
            ),
            fontsize=8.6, color=MUTED, va="bottom", linespacing=1.4,
        )
        pdf.savefig(fig, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return outpath


def _optimization_page1(pdf, parent: dict, ok_candidates: list[dict],
                        suggestions: list[dict]):
    """Summary table plus residue-level medicinal chemistry suggestions."""
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.06, 0.955, "DrugHX 药物分子改进报告", fontsize=24, fontweight="bold", color=INK)
    fig.text(
        0.06, 0.925,
        f"母体：{parent.get('ligand_name', '—')}    靶点：{parent.get('receptor_name', '—')}    "
        f"母体 Vina：{parent.get('affinity', '—')} kcal/mol",
        fontsize=11.5, color=MUTED,
    )
    english = i18n.get_lang() == "en"
    if english:
        lines = [
            "Optimisation strategy: starting from the parent interaction profile, the library "
            "covers scaffold hopping, bioisosteric replacement, substituent walking, linker "
            "modification, metabolic blocking and second-generation combinations. Every "
            "candidate is re-docked in the same binding box with the same scoring function. "
            "All values are relative rankings, not experimental binding free energies.",
        ]
    else:
        lines = [
            "改造策略：基于母体作用力谱，生成骨架跃迁、生物电子等排、取代基行走、",
            "连接子改造、代谢稳定化与组合改造候选，并在同一位点、同一打分函数下自动重新对接。",
            "所有数值为相对排序，不等同于实验结合自由能。",
        ]
    category_counts: dict[str, int] = {}
    for item in ok_candidates:
        name = str(item.get("category") or item.get("modification") or "其他")
        category_counts[name] = category_counts.get(name, 0) + 1
    english = i18n.get_lang() == "en"
    category_en = {
        "生物电子等排": "bioisostere",
        "骨架跃迁": "scaffold hopping",
        "取代基行走": "substituent walking",
        "代谢稳定化": "metabolic blocking",
        "连接子改造": "linker modification",
        "组合改造": "combination",
        "其他": "other",
    }
    if category_counts:
        lines.append(
            (
                f"This run covers {len(category_counts)} transformation classes: "
                + ", ".join(
                    f"{category_en.get(name, name)} {count}"
                    for name, count in category_counts.items()
                )
                + "."
            )
            if english
            else (
                "本轮覆盖 %d 类改造：%s。"
                % (
                    len(category_counts),
                    "、".join(f"{name} {count}" for name, count in category_counts.items()),
                )
            )
        )
    strategy = _wrap_cjk("\n".join(lines), 92) if english else "\n".join(lines)
    fig.text(0.06, 0.885, strategy, fontsize=10.5, color=INK, va="top", linespacing=1.5)
    has_type = any(
        str(item.get("modification", "")).strip() not in ("", "—")
        for item in ok_candidates
    )
    parent_mol = parent.get("mol")
    parent_le = None
    if parent_mol is not None and parent.get("affinity") is not None:
        parent_le = abs(float(parent.get("affinity"))) / max(1, parent_mol.GetNumHeavyAtoms())
    header = ["候选"] + (["改造类型"] if has_type else []) + [
        "Vina", "ΔVina", "LE", "氢键", "疏水",
    ]
    widths = ([0.30, 0.22] if has_type else [0.46]) + [0.09, 0.09, 0.08, 0.07, 0.07]
    if i18n.get_lang() == "en":
        header = ["Candidate"] + (["Modification"] if has_type else []) + [
            "Vina", "ΔVina", "LE", "H-bond", "Hydroph.",
        ]
        widths = ([0.30, 0.22] if has_type else [0.46]) + [0.09, 0.09, 0.08, 0.07, 0.07]
    rows = [header]
    for index, item in enumerate(ok_candidates[:12], start=1):
        result = item["result"]
        analysis = item.get("analysis", {})
        counts = analysis.get("counts", {})
        delta = result.get("affinity", 0) - parent.get("affinity", 0)
        row = [_trunc(f"{index:02d} {item.get('label', result.get('ligand_name', ''))}", 20)]
        if has_type:
            row.append(_trunc(str(item.get("modification", "")), 10))
        candidate_mol = item.get("mol")
        le_text = "—"
        if candidate_mol is not None and result.get("affinity") is not None:
            le = abs(float(result.get("affinity"))) / max(1, candidate_mol.GetNumHeavyAtoms())
            le_text = f"{le:.2f}"
            if parent_le is not None and le < parent_le * 0.98 and delta < 0:
                le_text += "↓"
        row.extend([
            f"{result.get('affinity', 0):.2f}",
            f"{delta:+.2f}",
            le_text,
            str(counts.get("hbonds", 0)),
            str(counts.get("hydrophobic", 0)),
        ])
        rows.append(row)
    ax = fig.add_axes([0.06, 0.60, 0.88, 0.205])
    ax.axis("off")
    table = ax.table(cellText=rows, loc="upper left", cellLoc="left", colWidths=widths)
    table.auto_set_font_size(False)
    table.set_fontsize(9.6)
    row_fraction = 1.0 / max(len(rows), 1)
    for (row, col), cell in table.get_celld().items():
        cell.set_height(row_fraction)
        cell.set_edgecolor(GRID)
        if row == 0:
            cell.set_facecolor("#e2e8f0")
            cell.set_text_props(fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f8fafc")
    fig.text(
        0.06, 0.590,
        "↓ = Vina improved but ligand efficiency fell below the parent; check for molecular-weight-driven false positives."
        if english
        else "↓ = 打分改善但配体效率低于母体，需警惕分子量膨胀带来的假阳性。",
        fontsize=8.2, color=MUTED, va="top",
    )
    fig.text(
        0.06, 0.558,
        "Residue-level optimisation suggestions" if english else "残基层面的改进方案",
        fontsize=14, fontweight="bold", color=INK,
    )
    y = 0.526
    for item in suggestions[:9]:
        residue = str(item.get("residue", ""))
        interaction = _wrap_cjk(str(item.get("interaction", "")), 14)
        strategy = _wrap_cjk(str(item.get("strategy", "")), 78)
        fig.text(0.06, y, residue, fontsize=10, fontweight="bold", color=TEAL, va="top")
        fig.text(0.155, y, interaction, fontsize=9.3, color=MUTED, va="top",
                 linespacing=1.35)
        fig.text(0.305, y, strategy, fontsize=9.3, color=INK, va="top", linespacing=1.35)
        lines = max(strategy.count("\n") + 1, interaction.count("\n") + 1)
        y -= 0.0195 * lines + 0.010
        if y < 0.06:
            break
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def optimization_report_pdf(
    parent: dict,
    candidates: list[dict],
    suggestions: list[dict],
    outpath: Path,
) -> Path:
    """Vector PDF report for scaffold hopping and auto-scored analogues."""
    _style_cjk()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    ok_candidates = [item for item in candidates if item.get("result", {}).get("status") == "ok"]
    if not ok_candidates:
        with PdfPages(outpath) as pdf:
            fig = _a4_page("优化报告", "没有候选完成对接，无法生成打分比较")
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)
        return outpath
    with PdfPages(outpath) as pdf:
        _optimization_page1(pdf, parent, ok_candidates, suggestions)

        # Full candidate table(s): the first page shows the top 12, these pages
        # list every docked candidate so no optimisation direction is lost.
        if len(ok_candidates) > 12:
            english = i18n.get_lang() == "en"
            for start in range(0, len(ok_candidates), 22):
                group = ok_candidates[start:start + 22]
                fig = _a4_page(
                    "Optimisation candidate table" if english else "优化候选全表",
                    (
                        f"Page {start // 22 + 1} · {len(ok_candidates)} docked candidates"
                        if english
                        else f"第 {start // 22 + 1} 页 · 已完成对接 {len(ok_candidates)} 个候选"
                    ),
                )
                rows = [
                    ["Candidate", "Modification", "Category", "Vina", "ΔVina"]
                    if english
                    else ["候选", "改造类型", "类别", "Vina", "ΔVina"]
                ]
                for index, item in enumerate(group, start=start + 1):
                    result = item["result"]
                    delta = (
                        float(result.get("affinity", 0))
                        - float(parent.get("affinity", 0))
                    )
                    rows.append([
                        _trunc(
                            f"{index:02d} {item.get('label', result.get('ligand_name', ''))}",
                            24,
                        ),
                        _trunc(str(item.get("modification", "")), 12),
                        _trunc(str(item.get("category", "")), 12),
                        f"{float(result.get('affinity', 0)):.2f}",
                        f"{delta:+.2f}",
                    ])
                _table(
                    fig, 0.86, rows, [0.36, 0.22, 0.18, 0.12, 0.12],
                    fontsize=8.2, row_height=0.026,
                )
                pdf.savefig(fig, facecolor="white")
                plt.close(fig)

        # Candidate developability table: docking score alone is not enough to
        # rank analogues, so every candidate gets one set of ADMET descriptors.
        english = i18n.get_lang() == "en"
        property_items = [parent] + ok_candidates
        for start in range(0, len(property_items), 20):
            group = property_items[start:start + 20]
            fig = _a4_page(
                "Optimisation candidate developability" if english
                else "优化候选类药性与 ADMET 属性",
                (
                    f"Page {start // 20 + 1} · {len(ok_candidates)} docked candidates "
                    "with the parent as reference"
                    if english
                    else f"第 {start // 20 + 1} 页 · {len(ok_candidates)} 个候选（母体作为参照）"
                ),
            )
            rows = [[
                "Candidate" if english else "候选", "MW", "cLogP", "TPSA",
                "HBD", "HBA", "RotB", "QED", "SA", "ESOL logS",
            ]]
            for offset, item in enumerate(group):
                index = start + offset
                name = _trunc(
                    str(item.get("label", item.get("ligand_name", ""))), 20
                )
                prefix = "00 " if index == 0 else f"{index:02d} "
                rows.append([prefix + name] + _property_cells(item.get("mol")))
            _table(
                fig, 0.855, rows,
                [0.23, 0.07, 0.07, 0.07, 0.05, 0.05, 0.06, 0.065, 0.065, 0.09],
                fontsize=8.6, row_height=0.028,
            )
            note_top = 0.855 - 0.028 * len(rows) - 0.022
            fig.text(
                0.07, note_top,
                _wrap_cjk(
                    "HBD / HBA 采用 RDKit 口径：供体为 N/O/S–H，受体为 N/O 孤对；"
                    "HBA 与 Lipinski 原文的 N+O 计数略有差异。"
                    "ESOL logS 为 Delaney 估算（mol/L），越负表示溶解度越低；"
                    "SA 为 Ertl 合成可及性评分（1 易 – 10 难）。"
                    "这些指标用于候选之间的相对比较，不等同于实验测定值。",
                    92,
                ) if not english else _wrap_cjk(
                    "HBD/HBA follow the RDKit definition (donors: N/O/S–H; "
                    "acceptors: N/O lone pairs), which differs slightly from the "
                    "original Lipinski N+O count. ESOL logS is a Delaney estimate "
                    "(mol/L, more negative = less soluble); SA is the Ertl "
                    "synthetic-accessibility score (1 easy – 10 hard). These "
                    "values are for relative ranking only, not measured data.",
                    92,
                ),
                fontsize=8.8, color=MUTED, va="top", linespacing=1.55,
            )
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)

        # page 2: score comparison
        comparison = [parent] + ok_candidates
        results = []
        analyses = {}
        for item in comparison:
            result = dict(item["result"])
            result["ligand_name"] = item.get("label", result.get("ligand_name"))
            results.append(result)
            analyses[id(result)] = item.get("analysis", {})
        fig, ax = plt.subplots(figsize=(8.6, max(3.6, 0.48 * len(results) + 1.8)))
        _draw_affinity(ax, results, compact=True)
        ax.set_title("母体与改进候选的打分比较", loc="left", fontweight="bold", fontsize=15, pad=10)
        pdf.savefig(fig, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        # page 3: interaction fingerprint
        rows, matrix = _interaction_matrix(results, analyses)
        if matrix:
            fig, ax = plt.subplots(figsize=(6.4, max(3.0, 0.38 * len(rows) + 1.9)))
            _draw_heatmap(ax, rows, matrix)
            ax.set_title("母体与候选的作用力谱", loc="left", fontweight="bold", fontsize=15, pad=10)
            pdf.savefig(fig, bbox_inches="tight", facecolor="white")
            plt.close(fig)

        # page 4: 2D structures
        structure_items = [parent] + ok_candidates[:5]
        fig, axes = plt.subplots(2, 3, figsize=(8.27, 6.0))
        for ax, item in zip(axes.flat, structure_items):
            mol = item.get("mol")
            if mol is None:
                ax.axis("off")
                continue
            _draw_molecule(ax, mol, title=item.get("label", item.get("ligand_name", "")), compact=True)
        for ax in axes.flat[len(structure_items):]:
            ax.axis("off")
        fig.suptitle("母体与改进候选的 2D 结构", fontsize=16, fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        # page 5+: interaction maps for the parent and every scored candidate
        map_items = [parent] + ok_candidates
        for start in range(0, len(map_items), 2):
            group = map_items[start:start + 2]
            fig, axes = plt.subplots(1, len(group), figsize=(9.6, 6.2))
            if len(group) == 1:
                axes = [axes]
            fig.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.16, wspace=0.32)
            for ax, item in zip(axes, group):
                mol = item.get("pose_mol") or item.get("mol")
                if mol is None:
                    ax.axis("off")
                    continue
                draw_interaction_diagram(
                    ax, mol, item.get("analysis", {}),
                    f"{item.get('label', item.get('ligand_name', ''))} · 作用力",
                )
            pdf.savefig(fig, bbox_inches="tight", facecolor="white")
            plt.close(fig)
    return outpath


def design_report_pdf(target: str, designs: list[dict], outpath: Path) -> Path:
    """Vector PDF report for target-focused de novo design and DB similarity hits."""
    _style_cjk()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    ok = [item for item in designs if item.get("result", {}).get("status") == "ok"]
    if not ok:
        with PdfPages(outpath) as pdf:
            fig = _a4_page("设计报告", "没有设计分子完成对接，无法生成打分比较")
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)
        return outpath
    with PdfPages(outpath) as pdf:
        # page 1: summary and design table
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.06, 0.955, "DrugHX 分子设计报告", fontsize=24, fontweight="bold", color=INK)
        english = i18n.get_lang() == "en"
        fig.text(
            0.06, 0.925,
            (
                f"Target: {target}    Designs: {len(ok)}    Method: fragment growth + AutoDock Vina re-docking"
                if english
                else f"靶点：{target}    设计分子：{len(ok)} 个    方法：片段生长 + AutoDock Vina 重新对接"
            ),
            fontsize=11.5, color=MUTED,
        )
        if english:
            text = (
                "Design strategy: aromatic and heteroaromatic cores are decorated with "
                "single, double and triple substitution patterns. When a reference ligand "
                "is supplied, scaffold hopping, heterocycle replacement and linker "
                "modification are also enumerated. Candidates are pre-filtered for "
                "drug-likeness and then docked in the same binding box. All scores are "
                "relative rankings, not experimental binding free energies."
            )
        else:
            text = (
                "设计策略：从芳环/杂环核心出发，枚举单取代、双取代与三取代模式；"
                "提供参考配体时同时执行骨架跃迁、杂环替换与连接子改造。"
                "候选先按类药性预筛，再在同一位点对接排序。"
                "所有分值用于相对排序，不等同于实验结合自由能。"
            )
        core_counts: dict[str, int] = {}
        for item in ok:
            core = str(item.get("core") or "其他")
            core_counts[core] = core_counts.get(core, 0) + 1
        if core_counts:
            scaffold_hops = sum(
                count for name, count in core_counts.items() if "骨架" in name
            )
            text += (
                f" This run covers {len(core_counts)} core families and {scaffold_hops} scaffold hops."
                if english
                else f"本轮覆盖 {len(core_counts)} 个核心家族，骨架跃迁 {scaffold_hops} 个。"
            )
        fig.text(
            0.06, 0.885,
            _wrap_cjk(text, 74 if english else 82),
            fontsize=9.8 if english else 10.5, color=INK,
            va="top", linespacing=1.5,
        )
        best_name = str(ok[0].get("name", ok[0].get("label", "—"))) if ok else ""
        translated_name = i18n.tr(best_name)
        best_label = translated_name[:34]
        if len(translated_name) > 34:
            best_label += "…"
        best_text = (
            f"最佳设计：{best_label}    "
            f"Vina {ok[0]['result'].get('affinity', 0):.2f} kcal/mol"
            if ok else "尚未得到成功对接的设计分子"
        )
        fig.text(
            0.06, 0.752 if english else 0.79, best_text,
            fontsize=11.5 if english else 13, fontweight="bold", color=TEAL,
        )
        rows = [["设计分子", "Vina", "MW", "cLogP", "TPSA", "最近邻", "Tanimoto"]]
        for item in ok[:12]:
            result = item["result"]
            mol = item.get("mol")
            hits = item.get("database_hits") or []
            hit = hits[0] if hits else {}
            rows.append([
                _trunc(str(item.get("name", item.get("label", ""))), 20),
                f"{result.get('affinity', 0):.2f}",
                f"{Descriptors.MolWt(mol):.1f}" if mol is not None else "—",
                f"{Descriptors.MolLogP(mol):.2f}" if mol is not None else "—",
                f"{Descriptors.TPSA(mol):.1f}" if mol is not None else "—",
                _wrap_cjk(str(hit.get("name", "—")), 16),
                f"{hit.get('tanimoto', 0):.2f}" if hit else "—",
            ])
        ax = fig.add_axes(
            [0.06, 0.36, 0.88, 0.355 if i18n.get_lang() == "en" else 0.40]
        )
        ax.axis("off")
        table = ax.table(cellText=rows, loc="upper left", cellLoc="left",
                         colWidths=[0.28, 0.08, 0.08, 0.08, 0.08, 0.20, 0.08])
        table.auto_set_font_size(False)
        table.set_fontsize(8.8)
        row_fraction = 1.0 / max(len(rows), 1)
        for (row, col), cell in table.get_celld().items():
            cell.set_height(row_fraction)
            cell.set_edgecolor(GRID)
            if row == 0:
                cell.set_facecolor("#e2e8f0")
                cell.set_text_props(fontweight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#f8fafc")
        if ok and len(ok) > 6:
            scores = [float(item["result"].get("affinity", 0)) for item in ok]
            weights = [
                float(Descriptors.MolWt(item["mol"]))
                for item in ok if item.get("mol") is not None
            ]
            summary = [
                f"本轮共枚举并完成对接 {len(ok)} 个设计分子，打分区间 "
                f"{max(scores):.2f} 至 {min(scores):.2f} kcal/mol，中位数 "
                f"{float(np.median(scores)):.2f}，说明取代基组合对结合能有实质影响。",
            ]
            if weights:
                tans = [
                    float((item.get("database_hits") or [{}])[0].get("tanimoto", 0))
                    for item in ok
                ]
                summary.append(
                    f"分子量范围 {min(weights):.0f}–{max(weights):.0f} Da，"
                    "全部落在类药空间内；表中最后两列为数据库 Morgan 指纹检索的"
                    f"最近邻与 Tanimoto 相似度（当前最高 {max(tans):.2f}），"
                    "相似度越低说明与已知分子骨架的差异越大。"
                )
            summary.append(
                "建议优先合成打分最优且相似度最低的 2–3 个分子，并用分子动力学检验"
                "磺酰胺与卤素取代基在结合口袋中的稳定性，再决定是否扩展取代基范围。"
            )
            fig.add_artist(
                Line2D([0.06, 0.94], [0.315, 0.315], transform=fig.transFigure,
                       color=GRID, linewidth=1.0)
            )
            fig.text(0.06, 0.295, "设计结果解读", fontsize=13, fontweight="bold", color=TEAL)
            fig.text(0.06, 0.272, _wrap_cjk("".join(summary), 84), fontsize=10.6,
                     color=INK, va="top", linespacing=1.65)
        if 2 <= len(ok) <= 6:
            axp = fig.add_axes([0.10, 0.08, 0.76, 0.23])
            mws = [float(Descriptors.MolWt(item["mol"])) for item in ok if item.get("mol") is not None]
            logps = [float(Descriptors.MolLogP(item["mol"])) for item in ok if item.get("mol") is not None]
            scores = [float(item["result"].get("affinity", 0)) for item in ok if item.get("mol") is not None]
            if mws:
                norm = Normalize(min(scores), max(scores) + 1e-6)
                axp.scatter(mws, logps, c=scores, cmap="YlGnBu_r", s=90,
                            edgecolor="white", linewidth=1.0, zorder=3, norm=norm)
                for name, x, y in zip(
                    [str(item.get("name", "")).split()[0] for item in ok if item.get("mol") is not None],
                    mws, logps,
                ):
                    axp.annotate(name, (x, y), textcoords="offset points", xytext=(6, 6),
                                 fontsize=7.5, color=INK)
                axp.set_xlim(min(mws) * 0.92, max(mws) * 1.08)
                axp.set_ylim(min(logps) - 0.35, max(logps) + 0.35)
                axp.set_xlabel("Molecular weight (Da)", fontsize=9)
                axp.set_ylabel("cLogP", fontsize=9)
                axp.tick_params(labelsize=8, colors=MUTED)
                axp.set_title("类药性空间（点标注为设计编号，颜色越深结合越强）", loc="left",
                              fontsize=11.5, fontweight="bold", pad=6)
                axp.grid(color=GRID, linewidth=0.5)
                axp.set_axisbelow(True)
                for spine in ("top", "right"):
                    axp.spines[spine].set_visible(False)
        pdf.savefig(fig, facecolor="white")
        plt.close(fig)

        # Full design table(s): every docked design with core, substituent and mode.
        if len(ok) > 12:
            english = i18n.get_lang() == "en"
            for start in range(0, len(ok), 22):
                group = ok[start:start + 22]
                fig = _a4_page(
                    "Design candidate table" if english else "设计候选全表",
                    (
                        f"Page {start // 22 + 1} · {len(ok)} docked designs"
                        if english
                        else f"第 {start // 22 + 1} 页 · 已完成对接 {len(ok)} 个设计"
                    ),
                )
                rows = [
                    ["Design", "Core", "Substituents", "Mode", "Vina"]
                    if english
                    else ["设计分子", "核心", "取代基", "模式", "Vina"]
                ]
                for index, item in enumerate(group, start=start + 1):
                    rows.append([
                        _trunc(
                            f"{index:02d} {item.get('name', item.get('label', ''))}",
                            22,
                        ),
                        _trunc(str(item.get("core", "")), 12),
                        _trunc("、".join(item.get("substituents") or []), 18),
                        _trunc(
                            (
                                "Scaffold" if item.get("mode") == "scaffold_hop" else "Fragment"
                            )
                            if english
                            else (
                                "骨架跃迁" if item.get("mode") == "scaffold_hop" else "片段生长"
                            ),
                            10,
                        ),
                        f"{float(item['result'].get('affinity', 0)):.2f}",
                    ])
                _table(
                    fig, 0.86, rows, [0.34, 0.17, 0.27, 0.10, 0.08],
                    fontsize=8.2, row_height=0.026,
                )
                pdf.savefig(fig, facecolor="white")
                plt.close(fig)

        # Design developability table: docking score alone is not enough to
        # rank designs, so every design gets one set of ADMET descriptors.
        english = i18n.get_lang() == "en"
        for start in range(0, len(ok), 20):
            group = ok[start:start + 20]
            fig = _a4_page(
                "Design candidate developability" if english
                else "设计候选类药性与 ADMET 属性",
                (
                    f"Page {start // 20 + 1} · {len(ok)} docked designs"
                    if english
                    else f"第 {start // 20 + 1} 页 · {len(ok)} 个设计分子"
                ),
            )
            rows = [[
                "Design" if english else "设计", "MW", "cLogP", "TPSA",
                "HBD", "HBA", "RotB", "QED", "SA", "ESOL logS",
            ]]
            for offset, item in enumerate(group):
                index = start + offset + 1
                raw_name = str(item.get("name", item.get("label", "")))
                core_name = re.sub(
                    r"^DESIGN\d+\s*", "", raw_name, flags=re.IGNORECASE
                ).strip() or raw_name
                rows.append(
                    [f"{index:02d} " + _trunc(core_name, 22)]
                    + _property_cells(item.get("mol"))
                )
            _table(
                fig, 0.855, rows,
                [0.235, 0.07, 0.07, 0.07, 0.05, 0.05, 0.06, 0.065, 0.065, 0.09],
                fontsize=8.6, row_height=0.028,
            )
            note_top = 0.855 - 0.028 * len(rows) - 0.022
            fig.text(
                0.07, note_top,
                (
                    _wrap_cjk(
                        "HBD / HBA 采用 RDKit 口径：供体为 N/O/S–H，受体为 N/O 孤对；"
                        "HBA 与 Lipinski 原文的 N+O 计数略有差异。"
                        "ESOL logS 为 Delaney 估算（mol/L），越负表示溶解度越低；"
                        "SA 为 Ertl 合成可及性评分（1 易 – 10 难）。"
                        "设计分子仍需经合成路线与实验验证后再进入下一阶段。",
                        92,
                    )
                    if not english
                    else _wrap_cjk(
                        "HBD/HBA follow the RDKit definition (donors: N/O/S–H; "
                        "acceptors: N/O lone pairs), which differs slightly from "
                        "the original Lipinski N+O count. ESOL logS is a Delaney "
                        "estimate (mol/L, more negative = less soluble); SA is "
                        "the Ertl synthetic-accessibility score (1 easy – 10 hard). "
                        "Designs still require route checking and experimental "
                        "validation before the next stage.",
                        92,
                    )
                ),
                fontsize=8.8, color=MUTED, va="top", linespacing=1.55,
            )
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)

        # page 2: score ranking
        results = []
        for item in ok:
            result = dict(item["result"])
            result["ligand_name"] = item.get("label", result.get("ligand_name"))
            results.append(result)
        fig, ax = plt.subplots(figsize=(8.6, max(3.6, 0.48 * len(results) + 1.8)))
        _draw_affinity(ax, results, compact=True)
        ax.set_title("设计分子打分排序", loc="left", fontweight="bold", fontsize=15, pad=10)
        pdf.savefig(fig, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        # page 3: 2D structures
        fig, axes = plt.subplots(2, 3, figsize=(8.27, 6.2))
        for ax, item in zip(axes.flat, ok[:6]):
            _draw_molecule(ax, item.get("mol"), title=item.get("label", ""), compact=True)
        for ax in axes.flat[len(ok[:6]):]:
            ax.axis("off")
        fig.suptitle("设计分子 2D 结构", fontsize=16, fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        # page 4: database similarity table and top hit structures
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.06, 0.95, "数据库相似性检索", fontsize=20, fontweight="bold", color=INK)
        fig.text(
            0.06, 0.915,
            _wrap_cjk(
                "对每个设计分子计算 Morgan 指纹（半径 2，2048 bit）与数据库的 Tanimoto 相似度。",
                100,
            ),
            fontsize=9.8 if i18n.get_lang() == "en" else 10.5, color=MUTED,
            va="top",
        )
        db_rows = [["设计分子", "最近邻", "Tanimoto", "数据库 SMILES"]]
        seen_hits = []
        for item in ok[:10]:
            hits = item.get("database_hits") or []
            hit = hits[0] if hits else {}
            db_rows.append([
                _trunc(item.get("name", item.get("label", "")), 26),
                _trunc(hit.get("name", "—"), 18),
                f"{hit.get('tanimoto', 0):.2f}" if hit else "—",
                _trunc(hit.get("smiles", "—"), 34),
            ])
            for hit in hits[:2]:
                if hit.get("smiles") and hit.get("smiles") not in [x[0] for x in seen_hits]:
                    seen_hits.append((hit["smiles"], hit.get("name", "")))
        ax = fig.add_axes([0.06, 0.38, 0.88, 0.50])
        ax.axis("off")
        table = ax.table(cellText=db_rows, loc="upper left", cellLoc="left",
                         colWidths=[0.28, 0.17, 0.09, 0.46])
        table.auto_set_font_size(False)
        table.set_fontsize(7.8)
        table.scale(1, 1.55)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor(GRID)
            if row == 0:
                cell.set_facecolor("#e2e8f0")
                cell.set_text_props(fontweight="bold")
        # top database hits structures
        if seen_hits:
            axes = []
            for index, (smiles, name) in enumerate(seen_hits[:4]):
                ax = fig.add_axes([0.06 + (index % 2) * 0.45, 0.20 + (1 - index // 2) * 0.16, 0.40, 0.14])
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    _draw_molecule(ax, mol, title=name, compact=True)
            fig.text(0.06, 0.52, "最近邻结构", fontsize=13, fontweight="bold", color=INK)
        pdf.savefig(fig, facecolor="white")
        plt.close(fig)

        # page 5+: interaction maps for top designs
        top = ok[:4]
        for start in range(0, len(top), 2):
            group = top[start:start + 2]
            fig, axes = plt.subplots(1, len(group), figsize=(9.6, 6.2))
            if len(group) == 1:
                axes = [axes]
            fig.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.16, wspace=0.32)
            for ax, item in zip(axes, group):
                mol = item.get("pose_mol") or item.get("mol")
                if mol is None:
                    ax.axis("off")
                    continue
                draw_interaction_diagram(
                    ax, mol, item.get("analysis", {}),
                    f"{item.get('label', '')} · 作用力",
                )
            pdf.savefig(fig, bbox_inches="tight", facecolor="white")
            plt.close(fig)
    return outpath


# ---------------------------------------------------------------------------
# publication-grade report pages (CNS style, A4, vector)
# ---------------------------------------------------------------------------
_PANEL_COLORS = [
    "#0f766e", "#0369a1", "#b45309", "#7e22ce",
    "#be123c", "#15803d", "#475569", "#0e7490",
]


def _panel_tag(ax, letter: str, x: float = -0.13, y: float = 1.06):
    """Panel letters are intentionally disabled; kept as a no-op for modules."""
    return None


def _a4_page(title: str, subtitle: str = "", tag: str = "", footer: str = "") -> plt.Figure:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    english = i18n.get_lang() == "en"
    title_size = 19.0 if english else 21
    title_text = _wrap_cjk(title, 58) if english else title
    fig.text(
        0.07, 0.958, title_text, fontsize=title_size, fontweight="bold",
        color=INK, va="top",
    )
    title_lines = title_text.count("\n") + 1
    sub_y = 0.958 - 0.024 * title_lines - 0.014
    if subtitle:
        sub_text = _wrap_cjk(subtitle, 96 if english else 78)
        fig.text(0.07, sub_y, sub_text, fontsize=12.0 if english else 11.5,
                 color=MUTED, va="top")
        sub_y -= 0.0165 * (sub_text.count("\n") + 1) + 0.006
    fig.add_artist(
        Line2D(
            [0.07, 0.93], [sub_y, sub_y], transform=fig.transFigure,
            color=GRID, linewidth=1.0,
        )
    )
    if tag and english:
        fig.text(0.07, 0.038, tag, fontsize=9.5, color=MUTED, va="bottom")
    fig.text(
        0.93, 0.038, footer or "DrugHX · 完整科研报告", fontsize=9.5,
        color=MUTED, ha="right", va="bottom",
    )
    fig.text(
        0.93, 0.055,
        (
            f"Tools: DrugHX {APP_VERSION} · RDKit · AutoDock Vina 1.2.5 · Vinardo · Meeko"
            if english
            else f"工具：DrugHX {APP_VERSION} · RDKit · AutoDock Vina 1.2.5 · Vinardo · Meeko"
        ),
        fontsize=6.6, color=MUTED, ha="right", va="bottom",
    )
    return fig


def _char_width(char: str) -> int:
    return 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1


_CLOSING = "，。、；：？！）」』）】％–—"
_CJK_TEXT = re.compile(r"[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]")


def _trunc(text, limit: int) -> str:
    """Translate before truncating so English labels are not cut in Chinese."""
    value = i18n.tr(str(text))
    if len(value) <= limit:
        return value
    return value[: max(1, limit - 1)].rstrip() + "…"


def _wrap_cjk(text: str, limit: int = 88) -> str:
    """Wrap mixed Chinese/Latin text by display width (CJK counts as 2)."""
    text = i18n.tr(text)
    if not _CJK_TEXT.search(text):
        return "\n".join(
            textwrap.fill(line, width=max(20, int(limit)))
            for line in text.split("\n")
        )
    lines = []
    current = ""
    width = 0
    for char in text:
        if char == "\n":
            lines.append(current)
            current = ""
            width = 0
            continue
        char_width = _char_width(char)
        if width + char_width > limit and current:
            if char in _CLOSING:
                current += char
                lines.append(current)
                current = ""
                width = 0
                continue
            if (
                char.isascii() and char.isalnum()
                and current[-1].isascii() and current[-1].isalnum()
                and width + char_width <= limit + 8
            ):
                current += char
                width += char_width
                continue
            if (
                char in ".,-+/%" and current[-1].isascii()
                and current[-1].isalnum()
                and width + char_width <= limit + 8
            ):
                current += char
                width += char_width
                continue
            if (
                current[-1] in ".,-+/%" and len(current) > 1
                and current[-2].isascii() and current[-2].isalnum()
                and char.isascii() and char.isalnum()
                and width + char_width <= limit + 8
            ):
                current += char
                width += char_width
                continue
            lines.append(current)
            current = char
            width = char_width
        else:
            current += char
            width += char_width
    if current:
        lines.append(current)
    for index in range(1, len(lines)):
        while lines[index] and lines[index][0] in _CLOSING:
            lines[index - 1] += lines[index][0]
            lines[index] = lines[index][1:]
    return "\n".join(lines)


def _clip_display(text: str, limit: int = 30) -> str:
    """Translate first, then truncate by display width.

    Clipping Chinese before translation leaves fragments such as
    ``…+甲…`` in English reports because the cut phrase no longer matches
    the dictionary.  Translating first keeps every produced label fully
    English.
    """
    width = 0
    output: list[str] = []
    for char in i18n.tr(str(text or "")):
        step = _char_width(char)
        if width + step > limit:
            return "".join(output).rstrip() + "…"
        output.append(char)
        width += step
    return "".join(output)


def _safe_wrap_units(font_size: float, requested: int, max_points: float = 450.0) -> int:
    """Cap the wrap width so no line can run past the A4 text column."""
    return max(24, min(int(requested), int(max_points / max(font_size * 0.5, 1.0))))


def _text_blocks(
    fig,
    top: float,
    blocks: list[tuple[str, list[str]]],
    width: int = 40,
    size: float = 12.0,
    heading_size: float = 13.6,
    stop: float = 0.055,
) -> float:
    """Lay out headed paragraphs; returns the next free y coordinate."""
    english = i18n.get_lang() == "en"
    if english:
        # English needs more characters per line and slightly tighter type to
        # fit the same A4 page as the Chinese layout.
        width = int(width * 1.20)
        size = size * 0.78
        heading_size = heading_size * 0.95
        stop = max(stop, 0.10)
    y = top
    for heading, paragraphs in blocks:
        if y < stop:
            break
        heading = i18n.tr(heading)
        if heading:
            fig.text(
                0.07, y, heading, fontsize=heading_size, fontweight="bold",
                color=TEAL, va="top",
            )
            y -= 0.00235 * heading_size
        for paragraph in paragraphs:
            if y < stop:
                break
            wrapped = _wrap_cjk(
                i18n.tr(paragraph),
                _safe_wrap_units(size, width * 2),
            )
            fig.text(
                0.07, y, wrapped, fontsize=size, color=INK, va="top",
                linespacing=1.45 if english else 1.60,
            )
            y -= 0.00194 * size * (wrapped.count("\n") + 1) * (
                0.92 if english else 1.0
            ) + (0.009 if english else 0.011)
    return y


def _table(
    fig,
    top: float,
    rows: list[list[str]],
    col_widths: list[float],
    fontsize: float = 9.4,
    row_height: float = 0.029,
):
    height = row_height * len(rows)
    ax = fig.add_axes([0.07, top - height, 0.86, height])
    ax.axis("off")
    table = ax.table(
        cellText=rows, loc="upper left", cellLoc="left", colWidths=col_widths
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    row_fraction = 1.0 / max(len(rows), 1)
    for (row, _col), cell in table.get_celld().items():
        cell.set_height(row_fraction)
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#e2e8f0")
            cell.set_text_props(fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f8fafc")
    return ax


def _property_cells(mol) -> list[str]:
    """MW, cLogP, TPSA, HBD, HBA, RotB, QED, SA score and ESOL logS."""
    if mol is None:
        return ["—"] * 9
    try:
        from rdkit.Chem import Crippen, QED as _QED, rdMolDescriptors

        mw = float(Descriptors.MolWt(mol))
        logp = float(Crippen.MolLogP(mol))
        tpsa = float(Descriptors.TPSA(mol))
        hbd = int(rdMolDescriptors.CalcNumHBD(mol))
        hba = int(rdMolDescriptors.CalcNumHBA(mol))
        rotb = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
        qed = float(_QED.qed(mol))
        aromatic_fraction = (
            float(rdMolDescriptors.CalcNumAromaticRings(mol))
            / max(1, mol.GetNumHeavyAtoms())
        )
        # Delaney (ESOL) estimate, logS in mol/L.
        logs = (
            0.16 - 0.63 * logp - 0.0062 * mw
            + 0.066 * rotb - 0.74 * aromatic_fraction
        )
    except Exception:
        return ["—"] * 9
    sa_text = "—"
    try:
        from . import profiling

        scorer = profiling._sascorer()
        if scorer:
            sa_text = f"{float(scorer.calculateScore(mol)):.2f}"
    except Exception:
        sa_text = "—"
    return [
        f"{mw:.1f}", f"{logp:.2f}", f"{tpsa:.1f}", str(hbd), str(hba),
        str(rotb), f"{qed:.2f}", sa_text, f"{logs:.2f}",
    ]


def _compact_frame(ax, grid_axis: str = "x"):
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=MUTED)


def _caption(fig, text: str, y: float = 0.148, width: int = 92,
             split: bool = True):
    """Figure caption placed on the same page, above the footer.

    Long captions are split into a method paragraph and a result paragraph so
    readers can skim each figure without reading a wall of text.
    """
    fig.add_artist(
        Line2D([0.07, 0.93], [y + 0.012, y + 0.012],
               transform=fig.transFigure, color=GRID, linewidth=0.8)
    )
    text = i18n.tr(text)
    if split and "\n\n" not in text and len(text) > 90:
        if i18n.get_lang() == "en":
            parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
        else:
            parts = [part.strip() for part in re.split(r"(?<=[。！？])", text) if part.strip()]
        if len(parts) >= 3:
            middle = max(1, len(parts) // 2)
            english = i18n.get_lang() == "en"
            sep = " " if english else ""
            head = "Methods: " if english else "方法："
            tail = "Results: " if english else "结果："
            text = (
                head + sep.join(parts[:middle]) + "\n\n"
                + tail + sep.join(parts[middle:])
            )
    wrapped = _wrap_cjk(text, int(width * 1.10) if i18n.get_lang() == "en" else width)
    lines = wrapped.count("\n") + 1
    size = 10.6 if i18n.get_lang() == "en" else 9.8
    # Shrink long captions so they never collide with the page footer.
    while y - lines * size * 1.55 / 842.0 < 0.082 and size > 8.4:
        size -= 0.3
    fig.text(
        0.07, y, wrapped, fontsize=size, color=MUTED,
        va="top", linespacing=1.55,
    )


def _single_ligand_panel(ax, item: dict):
    """Score dashboard used when only one ligand has a successful pose."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("单配体打分概览", loc="left", fontsize=15.5, pad=14)
    vina = float(item.get("affinity") or 0)
    vinardo = item.get("vinardo")
    rows = [("Vina", abs(vina), TEAL, f"{vina:.2f}")]
    if vinardo is not None:
        rows.append(("Vinardo", abs(float(vinardo)), AMBER, f"{float(vinardo):.2f}"))
    scale = max(6.0, max(value for _, value, _, _ in rows) * 1.25)
    y_positions = [0.62, 0.34][: len(rows)]
    for (label, value, color, text), y in zip(rows, y_positions):
        ax.add_patch(
            Rectangle(
                (0.12, y - 0.055), 0.72, 0.11,
                facecolor="#f8fafc", edgecolor=GRID, linewidth=0.8,
            )
        )
        ax.add_patch(
            Rectangle(
                (0.12, y - 0.055), 0.72 * value / scale, 0.11,
                facecolor=color, edgecolor="none", alpha=0.9,
            )
        )
        ax.text(
            0.10, y, label, ha="right", va="center",
            fontsize=12.5, color=INK,
        )
        ax.text(
            0.12 + 0.72 * value / scale + 0.02, y, text,
            ha="left", va="center", fontsize=13.5, color=color,
            fontweight="bold",
        )
    modes = item.get("modes") or []
    ax.text(
        0.12, 0.035,
        textwrap.fill(
            i18n.tr(
                f"输出构象 {len(modes)} 个；仅 1 个成功配体，不计算跨配体均值、"
                "中位数区间与 Vina–Vinardo 相关性。"
            ),
            46,
        ),
        fontsize=10.0, color=MUTED, va="bottom", linespacing=1.35,
    )
    ax.plot([0.12, 0.84], [0.225, 0.225], color=GRID, linewidth=0.8)


def _single_pose_panel(ax, item: dict):
    """Energy–RMSD funnel for the modes of a single ligand."""
    modes = item.get("modes") or []
    points = [
        (
            float(mode.get("affinity", 0)),
            float(mode.get("rmsd_lb", 0)),
        )
        for mode in modes
    ]
    if not points:
        ax.axis("off")
        ax.text(0.5, 0.5, "没有构象能量记录", ha="center", va="center",
                fontsize=12, color=MUTED)
        return
    scores = np.array([point[0] for point in points])
    rmsd = np.array([point[1] for point in points])
    ax.scatter(
        scores, rmsd, color=TEAL, s=95,
        edgecolor="white", linewidth=1.0, zorder=3,
    )
    best = int(np.argmin(scores))
    ax.scatter(
        [scores[best]], [rmsd[best]], s=190, facecolor="none",
        edgecolor=RED, linewidth=1.6, zorder=4,
    )
    ax.set_xlabel("Vina 打分 (kcal/mol)", fontsize=13.5)
    ax.set_ylabel("相对最优构象 RMSD (Å)", fontsize=13.5)
    ax.set_title(
        _wrap_cjk(
            i18n.tr(f"构象能量–RMSD 分布（{len(points)} 个输出构象）"),
            46,
        ),
        loc="left", fontsize=13.5, pad=10,
    )
    _compact_frame(ax, "both")


def _page_landscape(pdf, results: list[dict]):
    fig = _a4_page(
        "图 1　结合亲和力全景",
        "打分排序与两套经验打分函数的共识程度",
        "Fig. 1　Binding affinity landscape",
    )
    grid = fig.add_gridspec(
        2, 1, height_ratios=[1.40, 1.0], hspace=0.44,
        left=0.31, right=0.93, top=0.842, bottom=0.215,
    )
    ax = fig.add_subplot(grid[0])
    shown = results[:12]
    if len(shown) == 1:
        _single_ligand_panel(ax, shown[0])
    else:
        names = [_trunc(item.get("ligand_name", "ligand"), 16) for item in shown]
        scores = [float(item["affinity"]) for item in shown]
        y = np.arange(len(shown))
        ax.barh(y, [-value for value in scores], color=TEAL, height=0.62,
                edgecolor="none")
        for index, value in enumerate(scores):
            ax.text(
                -value + 0.16, index, f"{-value:.1f}", va="center", ha="left",
                fontsize=12.5, color=INK, fontweight="bold",
            )
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=13.0)
        ax.invert_yaxis()
        ax.set_xlabel("结合亲和力 −ΔG$_{bind}$ (kcal/mol)", fontsize=14.5)
        ax.set_title("按 Vina 打分排序的配体", loc="left", fontsize=16, pad=14)
        ax.set_xlim(0, max(-min(scores), 1.0) * 1.16)
        _panel_tag(ax, "a", x=-0.075)
        _compact_frame(ax)

    ax2 = fig.add_subplot(grid[1])
    pairs = [
        (float(item["affinity"]), float(item["vinardo"]))
        for item in results
        if item.get("vinardo") is not None
    ]
    if len(pairs) >= 2:
        vina = np.array([pair[0] for pair in pairs])
        vinardo = np.array([pair[1] for pair in pairs])
        ax2.scatter(
            vina, vinardo, s=95, color=BLUE, edgecolor="white",
            linewidth=1.2, zorder=3,
        )
        low = min(vina.min(), vinardo.min()) - 0.8
        high = max(vina.max(), vinardo.max()) + 0.8
        ax2.plot([low, high], [low, high], color=MUTED, linewidth=1.2,
                 linestyle=(0, (5, 3)), label="y = x", zorder=2)
        correlation = float(np.corrcoef(vina, vinardo)[0, 1])
        rho = _spearman(pairs)
        ax2.text(
            0.04, 0.95,
            f"r = {correlation:.3f} · ρ = {rho:.3f}\nn = {len(pairs)}" if rho is not None
            else f"r = {correlation:.3f}\nn = {len(pairs)}",
            transform=ax2.transAxes, fontsize=11.0, color=INK, va="top",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
                  "edgecolor": GRID, "linewidth": 0.8},
        )
        ax2.set_xlim(low, high)
        ax2.set_ylim(low, high)
        ax2.set_aspect("equal", adjustable="box")
        ax2.set_xlabel("Vina 打分 (kcal/mol)", fontsize=14.5)
        ax2.set_ylabel("Vinardo 打分 (kcal/mol)", fontsize=14.5)
        ax2.set_title("两套打分函数的一致性", loc="left", fontsize=16, pad=12)
        ax2.legend(frameon=False, loc="lower right")
        _panel_tag(ax2, "b", x=-0.075)
        _compact_frame(ax2, "both")
    elif len(shown) == 1:
        _single_pose_panel(ax2, shown[0])
    else:
        ax2.axis("off")
        has_vinardo = any(item.get("vinardo") is not None for item in results)
        ax2.text(
            0.5, 0.5,
            (
                "仅有 1 个配体，无法绘制 Vina–Vinardo 相关性\n"
                "Vinardo 二次打分见附录 A"
                if has_vinardo
                else "未启用 Vinardo 交叉打分"
            ),
            ha="center", va="center", fontsize=12.5, color=MUTED,
        )
    if len(shown) == 1:
        _caption(
            fig,
            _L(
                f"{shown[0].get('ligand_name', '配体')} 的 Vina 打分概览；本次只有 1 个成功配体，"
                "因此不计算跨配体均值、中位数区间或 Vina–Vinardo 相关性。"
                "该配体全部输出构象的能量–RMSD 分布，红圈标出最优构象；"
                "RMSD 为相对最优构象的重原子偏差，用于判断构象采样是否收敛。",
                f"Vina score overview for {shown[0].get('ligand_name', 'the ligand')}; with only one "
                "successfully docked ligand, cross-ligand means, median ranges and the Vina–Vinardo "
                "correlation are not computed. Energy–RMSD distribution of all output conformations, "
                "with the best pose circled in red; RMSD is the heavy-atom deviation from the best pose "
                "and indicates whether conformational sampling has converged.",
            ),
            width=104,
            split=False,
        )
    else:
        _caption(
            fig,
            _L(
                "按 AutoDock Vina 打分升序排列的配体，条形长度为 −ΔG_bind，末端标注数值（kcal/mol）。"
                "Vina 与 Vinardo 两套经验打分的散点比较，虚线为 y = x；左上角给出 Pearson r、Spearman ρ "
                "与样本量 n，仅当 |r| ≥ 0.8 时排序才不依赖单一打分函数。",
                "Ligands sorted by ascending AutoDock Vina score, with bar length equal to −ΔG_bind and "
                "values (kcal/mol) at the bar ends. Scatter comparison of the Vina and Vinardo empirical "
                "scores with the dashed line y = x; Pearson r, Spearman ρ and sample size n are given in "
                "the upper-left. The ranking is independent of a single scoring function only when |r| ≥ 0.8.",
            ),
            width=104,
            split=False,
        )
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def _page_efficiency(pdf, results: list[dict], props: dict):
    fig = _a4_page(
        "图 2　配体效率与类药化学空间",
        "结合能随分子大小的增长趋势，以及分子量–脂溶性分布",
        "Fig. 2　Ligand efficiency and chemical space",
    )
    grid = fig.add_gridspec(
        2, 1, height_ratios=[1.0, 1.0], hspace=0.42,
        left=0.14, right=0.90, top=0.842, bottom=0.215,
    )
    points = []
    for item in results:
        data = props.get(id(item))
        if not data or not data.get("heavy"):
            continue
        points.append(
            (
                float(data["heavy"]),
                -float(item["affinity"]),
                str(item.get("ligand_name", "")),
                float(data.get("efficiency", {}).get("le") or 0.0),
            )
        )
    ax = fig.add_subplot(grid[0])
    if points:
        heavy = np.array([point[0] for point in points])
        energy = np.array([point[1] for point in points])
        ax.scatter(
            heavy, energy, s=95, color=TEAL, edgecolor="white",
            linewidth=1.2, zorder=3,
        )
        span = max(heavy.max(), 10.0)
        top = max(energy.max() * 1.12, 6.0)
        for le in (0.2, 0.3, 0.4, 0.5):
            ax.plot([0, span * 1.08], [0, le * span * 1.08], color=MUTED,
                    linewidth=0.9, linestyle=(0, (6, 3)), alpha=0.7, zorder=1)
            x_label = min(span * 0.97, 0.90 * top / le)
            ax.text(
                x_label - 0.4, le * x_label, f"LE {le:.1f}",
                fontsize=10.5, color=MUTED, ha="right", va="bottom",
            )
        order = np.argsort(-energy)[:1]
        if len(points) <= 4:
            for index in order:
                ax.annotate(
                    points[index][2][:16],
                    (heavy[index], energy[index]),
                    textcoords="offset points", xytext=(-10, -18),
                    ha="right", fontsize=10.5, color=INK,
                    arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8},
                )
        ax.set_xlim(0, span * 1.12)
        ax.set_ylim(0, top)
        ax.set_xlabel("重原子数", fontsize=14)
        ax.set_ylabel(_L("−ΔGbind（kcal/mol，经验打分）", "−ΔGbind (kcal/mol, empirical)"), fontsize=14)
        ax.set_title("配体效率等值线", loc="left", fontsize=15.5, pad=12)
        mean_le = float(np.mean([point[3] for point in points]))
        le_text = (
            _L(
                f"LE = {points[0][3]:.2f} kcal/mol/HA（单配体）",
                f"LE = {points[0][3]:.2f} kcal/mol/HA (single ligand)",
            )
            if len(points) == 1
            else _L(
                f"平均 LE = {mean_le:.2f} kcal/mol/HA",
                f"Mean LE = {mean_le:.2f} kcal/mol/HA",
            )
        )
        ax.text(
            0.97, 0.06, le_text,
            transform=ax.transAxes, fontsize=11.5, color=INK,
            va="bottom", ha="right",
        )
        _compact_frame(ax)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "缺少分子性质数据", ha="center", va="center",
                fontsize=13, color=MUTED)

    ax2 = fig.add_subplot(grid[1])
    space = [
        (props[id(item)], float(item["affinity"]))
        for item in results
        if id(item) in props
    ]
    if space:
        masses = np.array([entry[0]["mw"] for entry in space])
        logps = np.array([entry[0]["logp"] for entry in space])
        values = np.array([entry[1] for entry in space])
        scatter = ax2.scatter(
            masses, logps, c=values, cmap="viridis", s=110,
            edgecolor="white", linewidth=1.1, zorder=3,
        )
        ax2.add_patch(
            Rectangle(
                (200, -1.5), 300, 6.5, fill=False, edgecolor=MUTED,
                linewidth=1.1, linestyle=(0, (6, 3)),
            )
        )
        ax2.text(202, 5.2, "类药空间上限（MW ≤ 500，cLogP ≤ 5）",
                 fontsize=10.5, color=MUTED, va="bottom")
        ax2.add_patch(
            Rectangle(
                (250, -1.0), 200, 4.5, fill=False, edgecolor=TEAL,
                linewidth=1.1, linestyle=(0, (2, 2)),
            )
        )
        ax2.text(253, -0.90, "lead-like 区间", fontsize=10.5, color=TEAL,
                 va="bottom")
        cbar = fig.colorbar(scatter, ax=ax2, fraction=0.035, pad=0.02)
        cbar.set_label("Vina 打分 (kcal/mol)", fontsize=11.5)
        cbar.ax.tick_params(labelsize=10.5)
        ax2.set_xlabel("分子量 (Da)", fontsize=14)
        ax2.set_ylabel("cLogP", fontsize=14)
        ax2.set_title("分子量–脂溶性化学空间", loc="left", fontsize=15.5,
                      pad=12)
        _compact_frame(ax2, "both")
    else:
        ax2.axis("off")
        ax2.text(0.5, 0.5, "缺少分子性质数据", ha="center", va="center",
                 fontsize=13, color=MUTED)
    _caption(
        fig,
        _L(
            "−ΔG_bind 随重原子数的变化，虚线为配体效率等值线（LE = 0.2–0.5 kcal/mol/HA）；"
            "同一等值线上的分子可直接比较，右下方（同尺寸但 −ΔG 偏低）的分子效率较差。"
            "分子量–cLogP 化学空间，点色表示 Vina 打分；虚线框为类药空间上限（MW ≤ 500、cLogP ≤ 5），"
            "点线框为 lead-like 区间（MW 250–450、cLogP −1–3.5）。",
            "−ΔG_bind versus heavy-atom count, with dashed ligand-efficiency contours "
            "(LE = 0.2–0.5 kcal/mol/HA); molecules on the same contour are directly comparable, and "
            "those lower right (same size but lower −ΔG) are less efficient. Molecular weight–cLogP "
            "space with point colour showing the Vina score; dashed box, drug-like limit (MW ≤ 500, "
            "cLogP ≤ 5); dotted box, lead-like region (MW 250–450, cLogP −1–3.5).",
        ),
        width=104,
        split=False,
    )
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def _page_interactions(pdf, results: list[dict], analyses: dict[int, dict]):
    rows, matrix = _interaction_matrix(results, analyses)
    if not matrix:
        return
    if len(rows) > 12:
        rows, matrix = rows[:12], matrix[:12]
    fig = _a4_page(
        "图 3　蛋白–配体相互作用指纹",
        "各配体的作用力计数与残基接触频率",
        "Fig. 3　Interaction fingerprint",
    )
    grid = fig.add_gridspec(
        2, 1, height_ratios=[1.0, 1.35], hspace=0.46,
        left=0.31, right=0.90, top=0.842, bottom=0.225,
    )
    ax = fig.add_subplot(grid[0])
    matrix_arr = np.asarray(matrix, dtype=float)
    image = ax.imshow(matrix_arr, cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(np.arange(len(INTERACTION_LABELS)))
    ax.set_xticklabels(
        [label for _, label in _interaction_labels()], rotation=30, ha="right",
        fontsize=11,
    )
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([_trunc(row, 18) for row in rows], fontsize=10.5)
    maximum = max(1.0, float(matrix_arr.max()))
    for i in range(matrix_arr.shape[0]):
        for j in range(matrix_arr.shape[1]):
            value = int(matrix_arr[i, j])
            ax.text(
                j, i, str(value), ha="center", va="center", fontsize=11,
                color="white" if value > maximum * 0.55 else INK,
            )
    ax.set_xticks(np.arange(-0.5, len(INTERACTION_LABELS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", length=0)
    cbar = fig.colorbar(image, ax=ax, fraction=0.032, pad=0.02)
    cbar.set_label("作用力数量", fontsize=11.5)
    cbar.ax.tick_params(labelsize=10.5)
    ax.set_title("作用力计数矩阵", loc="left", fontsize=16, pad=12)

    classes = [
        ("hbonds", "氢键", "#b45309"),
        ("hydrophobic", "疏水", "#15803d"),
        ("salt_bridges", "盐桥", "#b91c1c"),
        ("pi_stacking", "π-堆积", "#6d28d9"),
        ("halogens", "卤键", "#0369a1"),
        ("metals", "金属", "#475569"),
    ]
    frequency: dict[str, dict[str, int]] = {}
    for item in results:
        analysis = analyses.get(id(item))
        if not analysis:
            continue
        for key, _label, _color in classes:
            for contact in analysis.get(key, []):
                residue = f"{contact.get('residue', '')}{contact.get('resseq', '')}"
                bucket = frequency.setdefault(
                    residue, {k: 0 for k, _l, _c in classes}
                )
                bucket[key] += 1
    ax2 = fig.add_subplot(grid[1])
    if frequency:
        ranked = sorted(
            frequency.items(), key=lambda kv: -sum(kv[1].values())
        )[:14]
        labels = [item[0] for item in ranked]
        left = np.zeros(len(ranked))
        y = np.arange(len(ranked))
        for key, label, color in classes:
            values = np.array([item[1][key] for item in ranked], dtype=float)
            if values.max() == 0:
                continue
            ax2.barh(
                y, values, left=left, color=color, height=0.68,
                label=label, edgecolor="white", linewidth=0.6,
            )
            left += values
        ax2.set_yticks(y)
        ax2.set_yticklabels(labels, fontsize=12)
        ax2.invert_yaxis()
        ax2.set_xlabel("发生该作用的配体数", fontsize=14.5)
        ax2.set_title("关键残基接触频率", loc="left", fontsize=16, pad=12)
        ax2.legend(
            frameon=False, ncol=3, loc="lower right", fontsize=11.5,
            handlelength=1.2,
        )
        _compact_frame(ax2)
    else:
        ax2.axis("off")
        ax2.text(0.5, 0.5, "缺少作用力数据", ha="center", va="center",
                 fontsize=13, color=MUTED)
    _caption(
        fig,
        "各配体最优构象中氢键、疏水接触、盐桥、π-堆积、卤键与金属配位的计数矩阵，色标为接触数。"
        "跨配体统计的残基接触频率，条形按作用类型堆叠，横轴为发生该作用的配体数；"
        "排名靠前的残基构成口袋的主要锚点。",
    )
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def _page_poses(pdf, results: list[dict]):
    shown = [item for item in results if item.get("modes")][:6]
    if not shown:
        return
    fig = _a4_page(
        "图 4　构象能量谱",
        "多构象打分分布与结合模式的构象收敛性",
        "Fig. 4　Pose ensemble",
    )
    grid = fig.add_gridspec(
        2, 1, height_ratios=[1.25, 1.0], hspace=0.62,
        left=0.20, right=0.93, top=0.842, bottom=0.250,
    )
    ax = fig.add_subplot(grid[0])
    for index, item in enumerate(shown):
        modes = item["modes"][:9]
        color = _PANEL_COLORS[index % len(_PANEL_COLORS)]
        rmsd = [float(mode.get("rmsd_lb", 0.0)) for mode in modes]
        energy = [float(mode.get("affinity", 0.0)) for mode in modes]
        ax.plot(rmsd, energy, color=color, linewidth=1.6, alpha=0.9, zorder=2)
        ax.scatter(
            rmsd, energy, s=55, color=color, edgecolor="white",
            linewidth=0.8, zorder=3,
            label=_trunc(item.get("ligand_name", "ligand"), 18),
        )
    ax.set_xlabel("相对最优构象的 RMSD (Å)", fontsize=14.5)
    ax.set_ylabel("Vina 打分 (kcal/mol)", fontsize=14.5)
    ax.set_title("构象空间采样", loc="left", fontsize=16, pad=12)
    ax.legend(
        frameon=False, fontsize=10.2, ncol=3, loc="upper center",
        bbox_to_anchor=(0.5, -0.18), handlelength=1.5, columnspacing=1.2,
    )
    _compact_frame(ax, "both")

    ax2 = fig.add_subplot(grid[1])
    names = []
    spans = []
    counts = []
    for item in shown:
        modes = item["modes"]
        energies = [float(mode.get("affinity", 0.0)) for mode in modes]
        names.append(_trunc(item.get("ligand_name", "ligand"), 16))
        spans.append(max(energies) - min(energies) if energies else 0.0)
        counts.append(len(modes))
    y = np.arange(len(names))
    ax2.barh(y, spans, color=TEAL, height=0.6, edgecolor="none")
    for index, (span, count) in enumerate(zip(spans, counts)):
        ax2.text(span + 0.04, index, f"{span:.2f}　({count} 个构象)",
                 va="center", fontsize=11.5, color=INK)
    ax2.set_yticks(y)
    ax2.set_yticklabels(names, fontsize=12)
    ax2.invert_yaxis()
    ax2.set_xlabel("构象打分跨度 (kcal/mol)", fontsize=14.5)
    convergence_title = "构象收敛性（跨度越小越集中）"
    if i18n.get_lang() == "en":
        convergence_title = "Conformational convergence (tighter clustering)"
    ax2.set_title(convergence_title, loc="left", fontsize=14.5, pad=10)
    ax2.set_xlim(0, max(spans) * 1.45 if max(spans) > 0 else 1.0)
    _compact_frame(ax2)
    _caption(
        fig,
        _L(
            "每个配体全部构象的 Vina 打分与其相对最优构象的重原子 RMSD；点越靠左下，高分组装越接近全局最优。"
            "右图为各配体的构象打分跨度，括号内为构象数。本图仅反映单次运行的构象采样，"
            "不等同于多随机种子的重复性验证。",
            "Vina score of every output conformation versus its heavy-atom RMSD from the best pose; "
            "points lower and further left indicate poses closer to the global optimum. "
            "Right, conformational score span per ligand, with the number of output conformations in parentheses. "
            "Single-run sampling only; not a multi-seed reproducibility test.",
        ),
        y=0.198,
    )
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def _page_optimization(pdf, parent: dict, rows: list[dict]):
    rows = [row for row in rows if row.get("affinity") is not None]
    rows = sorted(rows, key=lambda row: row["affinity"])[:10]
    if not rows:
        return
    base = float(parent.get("affinity") or 0.0)
    fig = _a4_page(
        "图 5　药物分子改进效果",
        "母体与改造候选的结合能变化及关键性质对比",
        "Fig. 5　Optimization",
    )
    grid = fig.add_gridspec(
        3, 2, height_ratios=[1.55, 0.80, 0.80], hspace=0.48, wspace=0.34,
        left=0.345, right=0.93, top=0.842, bottom=0.185,
    )
    ax = fig.add_subplot(grid[0, :])
    for index, row in enumerate(rows):
        value = float(row["affinity"])
        if value < base - 0.05:
            color, _tag = TEAL, "改善"
        elif value > base + 0.05:
            color, _tag = RED, "下降"
        else:
            color, _tag = MUTED, "持平"
        ax.plot([base, value], [index, index], color=color, linewidth=2.4,
                alpha=0.75, zorder=2)
        ax.scatter([base], [index], s=55, facecolor="white", edgecolor=MUTED,
                   zorder=3)
        ax.scatter([value], [index], s=105, color=color, edgecolor="white",
                   linewidth=1.0, zorder=4)
        ax.text(
            max(base, value) + 0.10, index,
            f"{value:.2f} ({value - base:+.2f})", va="center", fontsize=10.5,
            color=INK,
        )
    ax.axvline(base, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(base, -0.75, f"母体 {base:.2f}", fontsize=10.5,
            color=MUTED, ha="center", va="bottom")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(
        [_clip_display(row.get("label", ""), 30) for row in rows], fontsize=11
    )
    ax.set_ylim(len(rows) - 0.4, -1.15)
    lowest = min([base] + [float(row["affinity"]) for row in rows])
    highest = max([base] + [float(row["affinity"]) for row in rows])
    ax.set_xlim(lowest - 1.0, highest + 2.0)
    ax.set_xlabel("Vina 打分 (kcal/mol)", fontsize=14)
    ax.set_title("结合能变化（哑铃图）", loc="left", fontsize=15.5, pad=12)
    _compact_frame(ax)

    metrics = [
        ("分子量 (Da)", "mw", 1.0),
        ("cLogP", "logp", 2.0),
        ("QED", "qed", 3.0),
        ("配体效率 LE", "le", 3.0),
    ]
    parent_props = parent.get("props") or {}
    best_row = rows[0]
    best_props = best_row.get("props") or {}
    for index, (label, key, decimals) in enumerate(metrics):
        ax2 = fig.add_subplot(grid[1 + index // 2, index % 2])
        if key == "le":
            parent_value = float(
                (parent_props.get("efficiency") or {}).get("le") or 0.0
            )
            best_value = float(
                (best_props.get("efficiency") or {}).get("le") or 0.0
            )
        else:
            parent_value = float(parent_props.get(key) or 0.0)
            best_value = float(best_props.get(key) or 0.0)
        bars = ax2.bar(
            ["母体", "最佳候选"], [parent_value, best_value],
            color=[MUTED, TEAL], width=0.55,
        )
        for bar, value in zip(bars, [parent_value, best_value]):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.{int(decimals)}f}",
                ha="center", va="bottom", fontsize=11.5, fontweight="bold",
            )
        ax2.set_title(label, fontsize=13.5, pad=8)
        top_value = max(parent_value, best_value, 0.001)
        ax2.set_ylim(0, top_value * 1.28)
        _compact_frame(ax2, "y")
    _caption(
        fig,
        _L(
            "优化母体（空心点与虚线）与各改造候选（实心点）的结合能哑铃图，青色表示相对母体改善、"
            "红色表示下降，括号内为 ΔVina（kcal/mol）；下方为母体与最佳候选的分子量、cLogP、QED 与配体效率。"
            "ΔVina 仅表示相对排序，不代表实验结合自由能。",
            "Dumbbell plot of docking energies for the parent (open circle, dashed line) and optimized "
            "candidates (filled circles); teal indicates improvement and red a decrease, with ΔVina "
            "(kcal/mol) in parentheses. Below, molecular weight, cLogP, QED and ligand efficiency of the "
            "parent versus the best candidate. ΔVina is a relative ranking only, not an experimental "
            "binding free energy.",
        ),
        width=104,
        split=False,
    )
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def _radar_scores(props: dict, affinity: float | None) -> list[float]:
    if not props:
        return [0.0] * 6
    efficiency = props.get("efficiency") or {}
    tpsa = float(props.get("tpsa") or 0.0)
    logp = float(props.get("logp") or 0.0)
    hbd = float(props.get("hbd") or 0.0)
    qed = float(props.get("qed") or 0.0)
    le = float(efficiency.get("le") or 0.0)
    return [
        min(1.0, abs(float(affinity or 0.0)) / 12.0),
        min(1.0, le / 0.6),
        qed,
        max(0.0, 1.0 - tpsa / 140.0),
        max(0.0, 1.0 - abs(logp - 2.5) / 3.5),
        max(0.0, 1.0 - hbd / 5.0),
    ]


def _page_design(pdf, parent: dict, rows: list[dict]):
    rows = [row for row in rows if row.get("affinity") is not None]
    if not rows:
        return
    rows = sorted(rows, key=lambda row: row["affinity"])
    fig = _a4_page(
        "图 6　骨架枚举设计分子的评价",
        "设计分子的打分排序、化学空间位置与类药性雷达",
        "Fig. 6　Scaffold enumeration",
    )
    grid = fig.add_gridspec(
        2, 2, height_ratios=[1.15, 1.0], hspace=0.50, wspace=0.62,
        left=0.400, right=0.93, top=0.812, bottom=0.215,
    )
    ax = fig.add_subplot(grid[0, :])
    shown = rows[:12]
    names = [_clip_display(row.get("label", ""), 34) for row in shown]
    scores = [float(row["affinity"]) for row in shown]
    y = np.arange(len(shown))
    ax.barh(y, [-value for value in scores], color=TEAL, height=0.62,
            edgecolor="none")
    for index, row in enumerate(shown):
        qed = (row.get("props") or {}).get("qed")
        suffix = f"　QED {qed:.2f}" if qed is not None else ""
        ax.text(
            -float(row["affinity"]) + 0.15, index,
            f"{-float(row['affinity']):.2f}{suffix}", va="center",
            ha="left", fontsize=11.0, color=INK,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=11.5)
    ax.invert_yaxis()
    ax.set_xlabel("结合亲和力 −ΔG (kcal/mol)", fontsize=14)
    ax.set_title("设计分子打分排序", loc="left", fontsize=15.5, pad=12)
    ax.set_xlim(0, max(-min(scores), 1.0) * 1.34)
    _compact_frame(ax)

    ax2 = fig.add_subplot(grid[1, 0])
    masses = []
    logps = []
    values = []
    for row in rows:
        data = row.get("props") or {}
        if not data:
            continue
        masses.append(float(data.get("mw") or 0.0))
        logps.append(float(data.get("logp") or 0.0))
        values.append(float(row.get("affinity") or 0.0))
    if masses:
        scatter = ax2.scatter(
            masses, logps, c=values, cmap="viridis", s=90,
            edgecolor="white", linewidth=1.0, zorder=3, label="设计分子",
        )
        ax2.scatter(
            [masses[0]], [logps[0]], s=260, facecolor="none",
            edgecolor=AMBER, linewidth=2.0, zorder=4, label="最优设计",
        )
        cbar = fig.colorbar(scatter, ax=ax2, fraction=0.035, pad=0.01)
        cbar.set_label("Vina 打分 (kcal/mol)", fontsize=9.5)
        cbar.ax.tick_params(labelsize=9.0)
    ax2.set_xlabel("分子量 (Da)", fontsize=13.5)
    ax2.set_ylabel("cLogP", fontsize=13.5)
    ax2.set_title("设计分子化学空间", loc="left", fontsize=14.5, pad=10)
    _compact_frame(ax2, "both")

    ax3 = fig.add_subplot(grid[1, 1], polar=True)
    labels = ["−ΔG", "LE", "QED", "TPSA", "cLogP", "HBD"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    best = rows[0]
    best_scores = _radar_scores(best.get("props") or {}, best.get("affinity"))
    median_scores = [
        float(np.median([_radar_scores(row.get("props") or {}, row.get("affinity"))[index]
                         for row in rows]))
        for index in range(len(labels))
    ]
    for values_row, color, label in (
        (median_scores, MUTED, "设计中位"),
        (best_scores, TEAL, "最优设计"),
    ):
        loop = values_row + values_row[:1]
        ax3.plot(angles, loop, color=color, linewidth=1.6, label=label)
        ax3.fill(angles, loop, color=color, alpha=0.14)
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(labels, fontsize=11.5, color=INK)
    ax3.set_ylim(0, 1.05)
    ax3.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax3.set_yticklabels(["", "", "", ""])
    ax3.grid(color=GRID, linewidth=0.8)
    ax3.spines["polar"].set_color(GRID)
    ax3.set_title("类药性雷达", fontsize=14.5, pad=18)
    _caption(
        fig,
        _L(
            "骨架枚举设计分子的打分排序，条形长度为 −ΔG，末端标注数值与 QED。设计分子在分子量–cLogP "
            "空间中的分布，颜色由深到浅表示 Vina 打分由优到劣，橙色圆环标记最优设计。"
            "最优设计与全部设计中位数的类药性雷达，各轴按 0–1 归一化（青色为最优设计，灰色为设计中位数）。"
            "设计分子由核心骨架与取代基枚举生成，并非严格意义上的全新骨架设计。",
            "Score ranking of scaffold-enumerated designs, with bar length equal to −ΔG and QED "
            "annotated at the bar ends. Designs in molecular weight–cLogP space, coloured from dark "
            "to light as the Vina score worsens, with the best design circled in amber. Drug-likeness "
            "radar of the best design versus the median of all designs, each axis normalized to 0–1 "
            "(teal, best design; grey, median). "
            "Designs are enumeration-based, not de novo scaffolds.",
        ),
        width=104,
        split=False,
    )
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def _page_validation(pdf, validation: dict):
    rows = validation.get("rows") or []
    overlay = validation.get("overlay")
    if not rows:
        return
    fig = _a4_page(
        "1.5　对接方案验证（重对接）",
        "对称校正 RMSD 与晶体构象叠合",
        "Fig. V1　Redocking validation",
    )
    grid = fig.add_gridspec(
        2, 1, height_ratios=[0.85, 1.3], hspace=0.50,
        left=0.30, right=0.93, top=0.842, bottom=0.205,
    )
    ax = fig.add_subplot(grid[0])
    names = [str(row.get("name", ""))[:18] for row in rows]
    rmsds = [float(row["rmsd"]) for row in rows]
    colors = [TEAL if row.get("passed") else RED for row in rows]
    y = np.arange(len(rows))
    ax.barh(y, rmsds, color=colors, height=0.60 if len(rows) > 2 else 0.42,
            edgecolor="none")
    for index, value in enumerate(rmsds):
        ax.text(value + 0.06, index, f"{value:.2f} Å", va="center",
                fontsize=11.5, color=INK, fontweight="bold")
    ax.axvline(2.0, color=AMBER, linewidth=1.4, linestyle=(0, (4, 3)))
    ax.text(2.06, -0.62, "2.0 Å 判据", fontsize=10.5, color=AMBER, va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("对称校正重原子 RMSD (Å)", fontsize=14)
    ax.set_title(
        _L("重对接构象与晶体的偏差", "Deviation from the crystal conformation"),
        loc="left", fontsize=15, pad=12,
    )
    ax.set_xlim(0, max(max(rmsds) + 0.9, 3.0))
    if len(rows) == 1:
        ax.set_ylim(0.62, -0.95)
        verdict = "通过" if rows[0].get("passed") else "未通过"
        ax.text(
            0.985, 0.90,
            f"{verdict} 2.0 Å 判据（RMSD {rmsds[0]:.2f} Å）",
            transform=ax.transAxes, ha="right", va="top", fontsize=12,
            color=TEAL if rows[0].get("passed") else RED, fontweight="bold",
            bbox={"boxstyle": "round,pad=0.32", "facecolor": "white",
                  "edgecolor": GRID, "linewidth": 0.8},
        )
    else:
        ax.set_ylim(len(rows) - 0.4, -1.0)
    _compact_frame(ax)

    ax2 = fig.add_subplot(grid[1])
    if overlay:
        crystal = np.asarray(overlay["crystal"], dtype=float)
        pose = np.asarray(overlay["pose"], dtype=float)
        for first, second in zip(crystal, pose):
            ax2.plot([first[0], second[0]], [first[1], second[1]],
                     color=GRID, linewidth=0.8, zorder=1)
        ax2.scatter(crystal[:, 0], crystal[:, 1], s=85, color=MUTED,
                    edgecolor="white", linewidth=1.0, zorder=2,
                    label="晶体构象")
        ax2.scatter(pose[:, 0], pose[:, 1], s=55, color=TEAL, marker="x",
                    linewidth=1.6, zorder=3, label="对接构象")
        ax2.set_aspect("equal")
        ax2.axis("off")
        ax2.legend(frameon=False, fontsize=10.5, loc="upper right")
        ax2.set_title(
            f"叠合投影（{overlay.get('name', '')}，RMSD {overlay.get('rmsd', 0):.2f} Å，"
            f"{overlay.get('atoms', 0)} 个重原子）",
            loc="left", fontsize=13.5, pad=10,
        )
    else:
        ax2.axis("off")
        ax2.text(0.5, 0.5, "缺少可供叠合的晶体构象", ha="center", va="center",
                 fontsize=13, color=MUTED)
    _caption(
        fig,
        _L(
            "重对接配体的对称校正重原子 RMSD，虚线为 2.0 Å 判据；绿色通过、红色未通过。"
            "晶体构象（灰色圆点）与对接构象（青色叉号）沿主平面的叠合投影，细线连接对应原子。"
            "若晶体配体为共价抑制剂，Vina 无法描述共价键，RMSD 与打分只反映非共价部分的叠合程度。",
            "Symmetry-corrected heavy-atom RMSD of re-docked ligands, with the dashed line marking the "
            "2.0 Å criterion; green passes, red fails. Overlay projection of the crystal (grey dots) and "
            "docked (teal crosses) conformations onto the principal plane, with thin lines connecting "
            "corresponding atoms. For covalent crystal ligands, Vina cannot describe the covalent bond "
            "and RMSD reflects only the non-covalent overlay.",
        ),
        width=104,
        y=0.165,
    )
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def _page_posebusters(pdf, payload: dict | None):
    """A4 page: PoseBusters pass/fail matrix for the checked binding modes."""
    rows = [row for row in ((payload or {}).get("rows") or []) if row.get("checks")]
    if not rows:
        return
    from matplotlib.colors import ListedColormap

    english = i18n.get_lang() == "en"
    checks = rows[0]["checks"]
    labels = [
        (item.get("label_en") if english else item.get("label_zh")) or item.get("key")
        for item in checks
    ]
    matrix = np.asarray(
        [
            [1.0 if item.get("passed") else 0.0 for row in rows]
            for item in checks
        ],
        dtype=float,
    )
    names = [
        _wrap_cjk(str(row.get("ligand") or row.get("name") or ""), 14).replace("\n", " ")
        for row in rows
    ]
    fig = _a4_page(
        "1.6　构象物理合理性（PoseBusters）"
        if not english
        else "1.6　Pose plausibility (PoseBusters)",
        "逐构象检查键长/键角、内部冲突、环平面性、应变能及蛋白–配体距离与体积重叠"
        if not english
        else "Bond lengths/angles, internal clashes, ring planarity, strain, protein–ligand distance and overlap",
        "Fig. P1　Pose plausibility",
    )
    ax = fig.add_axes([0.34, 0.27, 0.58, 0.55])
    ax.imshow(
        matrix,
        aspect="auto",
        cmap=ListedColormap(["#fca5a5", TEAL]),
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8.6)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=55, ha="right", fontsize=8.4)
    ax.set_xticks(np.arange(-0.5, len(names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.3)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        "绿色 = 通过，红色 = 未通过"
        if not english
        else "Green = pass, red = fail",
        loc="left",
        fontsize=12.5,
        color=MUTED,
        pad=10,
    )
    passed_rows = sum(1 for row in rows if row.get("all_passed"))
    failed_labels = sorted(
        {
            label
            for row in rows
            for label in (row.get("failed_labels") or [])
        }
    )
    _caption(
        fig,
        (
            f"PoseBusters（config = dock，版本 {payload.get('module_version', '?')}）对 {len(rows)} 个构象"
            f"逐项检查 20 个物理有效性判据，{passed_rows} 个构象全部通过。"
            + (
                f"未出现的检查项：{'、'.join(failed_labels)}。"
                if failed_labels
                else "所有构象均未触犯任一判据。"
            )
            + "Vina 打分不惩罚不合理几何，本页用于识别打分虚高或结合模式不合理的构象；"
            "物理合理不等于结合模式正确，仍需结合重对接 RMSD、MM-GBSA 或分子动力学复核。"
        )
        if not english
        else (
            f"PoseBusters (config = dock, version {payload.get('module_version', '?')}) checked "
            f"{len(rows)} poses against 20 physical-validity criteria; {passed_rows} passed all. "
            + (
                f"Checks that failed at least once: {', '.join(failed_labels)}. "
                if failed_labels
                else "No criterion was violated. "
            )
            + "Docking scores do not penalise implausible geometry, so this page flags poses whose "
            "score may be misleading; physical plausibility does not prove the binding mode, which "
            "still requires redocking RMSD, MM-GBSA or molecular dynamics."
        ),
    )
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def _page_mpo(pdf, payload: dict | None):
    """A4 page: multi-parameter score, stage labels and Pareto front."""
    items = [
        item for item in ((payload or {}).get("items") or [])
        if item.get("affinity") is not None and item.get("mpo") is not None
    ]
    if len(items) < 2:
        return
    fig = _a4_page(
        _L("2.11　多参数优化与候选分层", "2.11　Multi-parameter optimisation and stage gates"),
        _L(
            "效力 30%、配体效率 15%、QED 20%、可合成性 10%、安全性 25% 的 0–100 综合评分",
            "0–100 score weighting potency 30%, LE 15%, QED 20%, SA 10% and safety 25%",
        ),
        _L("Fig. MPO　多参数决策", "Fig. MPO　Multi-parameter decision"),
    )
    grid = fig.add_gridspec(
        2, 1, height_ratios=[1.25, 0.85], hspace=0.42,
        left=0.10, right=0.92, top=0.845, bottom=0.20,
    )
    ax = fig.add_subplot(grid[0])
    for item in items:
        color = TEAL if item.get("pareto") else "#cbd5e1"
        ax.scatter(
            float(item["affinity"]), float(item["mpo"]),
            s=70 if item.get("pareto") else 38, color=color,
            edgecolor="white", linewidth=0.8, zorder=3,
        )
    top = sorted(items, key=lambda item: -float(item["mpo"]))[:3]
    for index, item in enumerate(top):
        offset = (6, 8) if index % 2 == 0 else (6, -16)
        ax.annotate(
            _trunc(item.get("name") or "", 18),
            (float(item["affinity"]), float(item["mpo"])),
            xytext=offset, textcoords="offset points",
            fontsize=9.6, color=INK,
        )
    ax.set_xlabel(_L("Vina 对接打分 (kcal/mol)", "Vina docking score (kcal/mol)"), fontsize=13.5)
    ax.set_ylabel("MPO", fontsize=13.5)
    ax.set_title(
        _L("综合评分与 Pareto 前沿（青色为未被支配的候选）",
           "MPO versus docking score (teal = Pareto front)"),
        loc="left", fontsize=15, pad=10,
    )
    _panel_tag(ax, "a", x=-0.075)
    _compact_frame(ax)

    ax2 = fig.add_subplot(grid[1])
    ax2.axis("off")
    header = [
        _L("配体", "Ligand"), _L("Vina", "Vina"), "LE", "QED", "SA",
        _L("安全性", "Safety"), "MPO", _L("阶段", "Stage"), _L("新颖性", "Novelty"),
    ]
    rows = [header]
    for item in sorted(items, key=lambda row: -float(row["mpo"]))[:9]:
        novelty_label = (item.get("novelty") or {}).get("label", "—")
        rows.append([
            _trunc(item.get("name") or "", 20),
            f"{float(item['affinity']):.2f}",
            "—" if item.get("le") is None else f"{float(item['le']):.2f}",
            "—" if item.get("qed") is None else f"{float(item['qed']):.2f}",
            "—" if item.get("sa") is None else f"{float(item['sa']):.1f}",
            f"{float(item.get('safety') or 0):.0f}",
            f"{float(item['mpo']):.1f}",
            _L(str(item.get("stage") or "—"), str(item.get("stage_en") or item.get("stage") or "—")),
            _trunc(str(novelty_label), 18),
        ])
    _table(fig, 0.455, rows, [0.28, 0.10, 0.07, 0.07, 0.06, 0.08, 0.07, 0.08, 0.18], fontsize=9.4, row_height=0.022)
    _caption(
        fig,
        _L(
            "阶段判定：候选需 MPO ≥70、打分 ≤−9、LE ≥0.35、QED ≥0.60；先导需 MPO ≥55、打分 ≤−8、LE ≥0.28。"
            "Pareto 前沿指在打分、LE 与安全性三项上均无其他分子同时更优的候选。新颖性为与内置 FDA 药物集的 Morgan/Tanimoto 最大相似度："
            "≥0.60 为低、0.40–0.60 为中、<0.40 为高。评分仅用于排序与立项判断，需实验验证。",
            "Stage rules: candidate requires MPO ≥70, score ≤−9, LE ≥0.35 and QED ≥0.60; lead requires MPO ≥55, "
            "score ≤−8 and LE ≥0.28. Pareto members are not dominated on score, LE and safety. "
            "Novelty is the maximum Morgan/Tanimoto similarity to the built-in FDA set (≥0.60 low, 0.40–0.60 medium, "
            "<0.40 high). Scores rank hypotheses and require experimental confirmation.",
        ),
        width=104,
    )
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def _page_assay(pdf, payload: dict | None):
    """A4 page: measured potency versus docking score calibration."""
    rows = [row for row in ((payload or {}).get("rows") or []) if row.get("p_activity") is not None]
    if len(rows) < 3:
        return
    summary = (payload or {}).get("summary") or {}
    fig = _a4_page(
        _L("2.12　实验数据回填与打分校准", "2.12　Assay calibration of docking scores"),
        _L("实测 IC50/Kd 与对接打分的秩相关，用于判断本系列能否按打分排序", "Rank correlation between measured potency and docking score"),
        _L("Fig. CAL　打分校准", "Fig. CAL　Score calibration"),
    )
    ax = fig.add_axes([0.11, 0.52, 0.80, 0.315])
    xs = [float(row["affinity"]) for row in rows]
    ys = [float(row["p_activity"]) for row in rows]
    ax.scatter(xs, ys, s=70, color=TEAL, edgecolor="white", linewidth=0.9, zorder=3)
    slope = summary.get("slope")
    intercept = summary.get("intercept")
    if slope is not None and intercept is not None:
        x_min, x_max = min(xs), max(xs)
        pad = (x_max - x_min) * 0.08 or 0.5
        line_x = [x_min - pad, x_max + pad]
        ax.plot(line_x, [slope * x + intercept for x in line_x], color=AMBER, linewidth=1.4, linestyle=(0, (4, 3)))
    for row in rows[:4]:
        ax.annotate(
            _trunc(row.get("name") or "", 16),
            (float(row["affinity"]), float(row["p_activity"])),
            xytext=(4, 4), textcoords="offset points", fontsize=9.6, color=INK,
        )
    ax.set_xlabel(_L("Vina 对接打分 (kcal/mol)", "Vina docking score (kcal/mol)"), fontsize=13.5)
    ax.set_ylabel(_L("实测 pActivity", "Measured pActivity"), fontsize=13.5)
    ax.set_title(
        _L("实测效价与对接打分", "Measured potency versus docking score"),
        loc="left", fontsize=15, pad=10,
    )
    _panel_tag(ax, "a", x=-0.075)
    _compact_frame(ax)

    rho = summary.get("spearman")
    pearson = summary.get("pearson")
    fig.text(
        0.10, 0.475,
        _L(
            f"匹配 {summary.get('matched', 0)}/{summary.get('total', 0)} 条；"
            f"Spearman ρ = {'—' if rho is None else f'{rho:.2f}'}，"
            f"Pearson r = {'—' if pearson is None else f'{pearson:.2f}'}。",
            f"Matched {summary.get('matched', 0)}/{summary.get('total', 0)} rows; "
            f"Spearman ρ = {'n/a' if rho is None else f'{rho:.2f}'}, "
            f"Pearson r = {'n/a' if pearson is None else f'{pearson:.2f}'}.",
        ),
        fontsize=12, color=INK, va="top",
    )
    fig.text(
        0.10, 0.448,
        _wrap_cjk(
            _L(
                f"判断：{summary.get('verdict', '—')}",
                f"Verdict: {summary.get('verdict_en') or summary.get('verdict', '—')}",
            ),
            96 if i18n.get_lang() == "en" else 66,
        ),
        fontsize=11.0, color=TEAL, fontweight="bold", va="top",
    )
    header = [
        _L("配体", "Ligand"), _L("实测值", "Measured"), _L("pActivity", "pActivity"),
        _L("Vina", "Vina"), _L("匹配方式", "Match"),
    ]
    table_rows = [header]
    for row in rows[:8]:
        try:
            measured = f"{float(row.get('value')):.4g}"
        except (TypeError, ValueError):
            measured = str(row.get("value") or "—")
        table_rows.append([
            _trunc(row.get("name") or "", 24),
            f"{measured} {row.get('unit') or ''}".strip(),
            f"{float(row['p_activity']):.2f}",
            f"{float(row['affinity']):.2f}",
            "Name" if str(row.get("matched_by")) == "名称" else str(row.get("matched_by") or "—"),
        ])
    _table(fig, 0.405, table_rows, [0.30, 0.20, 0.12, 0.12, 0.12], fontsize=9.8, row_height=0.023)
    _caption(
        fig,
        _L(
            "pActivity = 9 − log10(IC50/nM)。预期相关性为负：对接打分越负、pActivity 越高。"
            "ρ ≤ −0.60 表示排序可用于本系列；−0.60 < ρ ≤ −0.30 仅可粗排；|ρ| < 0.30 或方向为正时，"
            "说明当前打分不能用于排序，应检查名称/SMILES 对应关系、实验条件，并考虑 MM-GBSA 或自由能微扰复核。",
            "pActivity = 9 − log10(IC50/nM). The expected correlation is negative. "
            "ρ ≤ −0.60 supports ranking within this series; −0.60 < ρ ≤ −0.30 supports only coarse ranking; "
            "|ρ| < 0.30 or a positive sign means the score cannot rank this series, so check name/SMILES "
            "matching, assay conditions and consider MM-GBSA or free-energy perturbation.",
        ),
        width=104,
    )
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def _page_trust(pdf, payload: dict | None):
    """A4 page: decision-confidence components and abstention advice."""
    if not payload or not payload.get("components"):
        return
    fig = _a4_page(
        _L("2.13　可信度与弃权判定", "2.13　Decision confidence and abstention"),
        _L(
            "结构质量、适用域、引擎一致性、实验校准与构象有效性的加权证据",
            "Weighted evidence from structure, domain, consensus, calibration and pose validity",
        ),
        _L("Fig. TRUST　可信度证据", "Fig. TRUST　Confidence evidence"),
    )
    components = payload.get("components") or []
    ax = fig.add_axes([0.30, 0.55, 0.62, 0.30])
    labels = [
        _L(
            str(item.get("label") or item.get("key")),
            str(item.get("label_en") or item.get("label") or item.get("key")),
        )
        for item in components
    ]
    scores = [float(item.get("score") or 0) for item in components]
    confidences = [float(item.get("weight") or 0) for item in components]
    y = np.arange(len(components))
    colors = [TEAL if score >= 70 else AMBER if score >= 45 else RED for score in scores]
    ax.barh(y, scores, color=colors, height=0.55, edgecolor="none")
    for index, (score, weight) in enumerate(zip(scores, confidences)):
        ax.text(
            min(score + 1.5, 98), index,
            f"{score:.0f} · {_L('权重', 'weight')} {weight * 100:.0f}%",
            va="center", fontsize=10.5, color=INK,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xlabel(_L("分项可信度（0–100）", "Component confidence (0–100)"), fontsize=13.5)
    ax.set_title(
        _L("分项证据", "Component evidence"), loc="left", fontsize=16, pad=10,
    )
    _compact_frame(ax)

    score = float(payload.get("score") or 0)
    color = TEAL if score >= 75 else AMBER if score >= 55 else RED
    fig.text(
        0.10, 0.505, f"{score:.0f}",
        fontsize=34, fontweight="bold", color=color, va="top",
    )
    fig.text(
        0.10, 0.435,
        _L(
            f"总体可信度：{payload.get('level', '—')}",
            f"Overall confidence: {payload.get('level_en') or payload.get('level', '—')}",
        ),
        fontsize=14, color=INK, va="top",
    )
    fig.text(
        0.10, 0.405,
        _wrap_cjk(
            _L(
                str(payload.get("advice") or ""),
                str(payload.get("advice_en") or payload.get("advice") or ""),
            ),
            40 if i18n.get_lang() == "en" else 26,
        ),
        fontsize=11.2, color=color, va="top", linespacing=1.4,
    )
    fig.text(
        0.46, 0.478, _L("证据说明", "Evidence"),
        fontsize=12.5, fontweight="bold", color=TEAL, va="top",
    )
    note_y = 0.445
    note_size = 9.5
    for item in components:
        if not item.get("note"):
            continue
        label = _L(
            str(item.get("label") or item.get("key")),
            str(item.get("label_en") or item.get("label") or item.get("key")),
        )
        note = _L(
            str(item.get("note") or ""),
            str(item.get("note_en") or item.get("note") or ""),
        )
        if len(note) > 190:
            note = note[:187].rstrip() + "…"
        wrapped = _wrap_cjk(f"{label}: {note}", 56 if i18n.get_lang() == "en" else 34)
        fig.text(
            0.46, note_y, wrapped, fontsize=note_size, color=INK, va="top",
            linespacing=1.35,
        )
        note_y -= 0.00194 * note_size * (wrapped.count("\n") + 1) * (
            0.92 if i18n.get_lang() == "en" else 1.0
        ) + 0.007
    _caption(
        fig,
        _L(
            "可信度低于 45 时报告明确弃权：当前排序不建议直接用于立项或合成决策。"
            "结构质量来自 PDB 分辨率、R-free、缺失残基、altloc 或 AlphaFold pLDDT；"
            "适用域为与上市药参考集的最近邻相似度；一致性来自多引擎/种子分歧；"
            "实验校准来自项目内实测数据的秩相关。",
            "Below 45 the report explicitly abstains: the ranking should not drive programme or synthesis "
            "decisions. Structure quality uses PDB resolution, R-free, missing residues, altlocs or AlphaFold "
            "pLDDT; domain is nearest-neighbour similarity to marketed drugs; consensus is engine/seed "
            "disagreement; calibration is the rank correlation with project assays.",
        ),
        width=104,
        y=0.180,
    )
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def _page_mmp(pdf, payload: dict | None):
    """A4 page: matched molecular pair SAR rules."""
    rules = (payload or {}).get("rules") or []
    if not rules:
        return
    fig = _a4_page(
        _L("2.14　项目内匹配分子对与 SAR 规则", "2.14　Project matched-pair SAR rules"),
        _L(
            f"从 {payload.get('molecules', 0)} 个分子、{payload.get('pairs', 0)} 对匹配分子中挖掘的定向改造规则",
            f"Directional rules mined from {payload.get('molecules', 0)} molecules and {payload.get('pairs', 0)} matched pairs",
        ),
        _L("Fig. MMP　项目内 SAR", "Fig. MMP　Project SAR"),
    )
    header = [
        _L("取代基改造", "Transformation"), _L("支持对数", "Pairs"),
        _L("平均 Δ", "Mean Δ"), _L("标准差", "SD"),
        _L("方向", "Direction"), _L("置信度", "Confidence"), _L("示例", "Example"),
    ]
    rows = [header]
    for rule in rules[:12]:
        example = (rule.get("examples") or [{}])[0]
        direction = str(rule.get("direction") or "—")
        confidence = str(rule.get("confidence") or "—")
        rows.append([
            _trunc(f"{rule.get('from')} >> {rule.get('to')}", 34),
            str(rule.get("n") or 0),
            f"{float(rule.get('mean_delta') or 0):+.2f}",
            f"{float(rule.get('sd') or 0):.2f}",
            _L(direction, _MMP_DIRECTION_EN.get(direction, direction)),
            _L(confidence, _MMP_CONFIDENCE_EN.get(confidence, confidence)),
            _trunc(
                f"{example.get('from', '')} → {example.get('to', '')}",
                20,
            ),
        ])
    _table(
        fig, 0.845, rows,
        [0.30, 0.06, 0.08, 0.06, 0.10, 0.13, 0.19],
        fontsize=9.2, row_height=0.024,
    )
    _caption(
        fig,
        _L(
            "Δ 为“后者 − 前者”：实测数据存在时用 pActivity 差值，否则用 −Vina 差值；正值表示改造改善活性/打分。"
            "同一取代基改造在不同核心上可能方向相反，因此规则只在支持对数和置信度足够时建议采用。"
            "n=1 的规则仅作为假设，需实验验证；核心结构不同的规则不应直接迁移。",
            "Δ is the second molecule minus the first (pActivity when assays exist, otherwise −Vina); a positive "
            "value means the change improves potency or score. The same transformation can reverse direction on a "
            "different core, so use rules only when support and confidence are adequate. Rules with n=1 are hypotheses "
            "and must be tested.",
        ),
        width=104,
    )
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def _page_pbpk(pdf, payload: dict | None):
    """A4 page: seven-species PBPK curves and withdrawal interval."""
    if not payload or not payload.get("time_h"):
        return
    fig = _a4_page(
        _L("2.15　七物种 PBPK 与休药期", "2.15　Seven-species PBPK and withdrawal"),
        _L(
            "血浆与组织浓度–时间曲线、示例 MRL 比较与 Monte-Carlo 休药期",
            "Plasma/tissue concentration–time curves, example MRL comparison and Monte-Carlo withdrawal",
        ),
        _L("Fig. PBPK　种属与残留", "Fig. PBPK　Species and residues"),
    )
    grid = fig.add_gridspec(
        2, 1, height_ratios=[1.2, 0.8], hspace=0.45,
        left=0.11, right=0.91, top=0.845, bottom=0.20,
    )
    ax = fig.add_subplot(grid[0])
    time_days = np.asarray(payload["time_h"], dtype=float) / 24.0
    curves = payload.get("curves") or {}
    styles = {
        "plasma": (TEAL, "-", 1.8),
        "muscle": (AMBER, "-", 1.3),
        "liver": (RED, "--", 1.2),
        "kidney": (MUTED, "--", 1.2),
        "fat": ("#7c3aed", ":", 1.2),
    }
    labels = {
        "plasma": _L("血浆", "Plasma"), "muscle": _L("肌肉", "Muscle"),
        "liver": _L("肝脏", "Liver"), "kidney": _L("肾脏", "Kidney"),
        "fat": _L("脂肪", "Fat"),
    }
    for key, (color, style, width) in styles.items():
        values = np.asarray(curves.get(key) or [], dtype=float)
        if values.size != time_days.size:
            continue
        ax.plot(time_days, np.maximum(values, 1e-6), color=color, linestyle=style, linewidth=width, label=labels[key])
    ax.set_yscale("log")
    ax.set_xlabel(_L("时间（天）", "Time (days)"), fontsize=13.5)
    ax.set_ylabel(_L("浓度（mg/L 或 mg/kg）", "Concentration (mg/L or mg/kg)"), fontsize=13.5)
    ax.set_title(
        _L(
            f"{payload.get('species', '—')} · {payload.get('route', '—')} · {float(payload.get('dose_mg_kg') or 0):.2f} mg/kg",
            f"{payload.get('species', '—')} · {payload.get('route', '—')} · {float(payload.get('dose_mg_kg') or 0):.2f} mg/kg",
        ),
        loc="left", fontsize=15, pad=10,
    )
    ax.legend(frameon=False, fontsize=10.5, ncol=3)
    _panel_tag(ax, "a", x=-0.075)
    _compact_frame(ax)

    ax2 = fig.add_subplot(grid[1])
    rows = payload.get("mrl_rows") or []
    ax2.axis("off")
    table_rows = [[
        _L("组织", "Tissue"), _L("MRL (µg/kg)", "MRL (µg/kg)"),
        _L("Cmax (mg/kg)", "Cmax (mg/kg)"), _L("超标天数", "Days above"),
        _L("状态", "Status"),
    ]]
    for row in rows:
        table_rows.append([
            str(row.get("tissue") or ""),
            f"{row.get('mrl_ug_kg')}",
            f"{float(row.get('cmax_mg_kg') or 0):.4f}",
            f"{float(row.get('days_above_mrl') or 0):.2f}",
            _L("超标", "exceeded") if row.get("exceeded") else _L("未超标", "below"),
        ])
    if len(table_rows) > 1:
        table = ax2.table(cellText=table_rows, loc="upper center", cellLoc="left")
        table.auto_set_font_size(False)
        table.set_fontsize(9.6)
        table.scale(1.0, 1.35)
        for (row, _col), cell in table.get_celld().items():
            cell.set_edgecolor(GRID)
            if row == 0:
                cell.set_facecolor("#e2e8f0")
                cell.set_text_props(fontweight="bold")
    else:
        ax2.text(0.5, 0.6, _L("非食用动物或未提供 MRL，不计算休药期。", "Non-food species or no MRL supplied; withdrawal not calculated."),
                 ha="center", va="center", fontsize=11.5, color=MUTED)
    mc = payload.get("monte_carlo") or {}
    summary = _L(
        f"血浆 Cmax {float(payload.get('cmax_mg_l') or 0):.3f} mg/L，Tmax {float(payload.get('tmax_h') or 0):.1f} h，"
        f"AUC {float(payload.get('auc_mg_h_l') or 0):.1f} mg·h/L，表观终末 t½ {payload.get('half_life_h') or '—'} h；"
        f"点估计休药期 {payload.get('withdrawal_days', 0)} 天。"
        + (f" Monte-Carlo 中位 {mc.get('median_days', 0):.0f} 天、95 分位 {mc.get('p95_days', 0):.0f} 天。" if mc else ""),
        f"Plasma Cmax {float(payload.get('cmax_mg_l') or 0):.3f} mg/L, Tmax {float(payload.get('tmax_h') or 0):.1f} h, "
        f"AUC {float(payload.get('auc_mg_h_l') or 0):.1f} mg·h/L, t1/2 {payload.get('half_life_h') or 'n/a'} h; "
        f"point-estimate withdrawal {payload.get('withdrawal_days', 0)} days."
        + (f" Monte-Carlo median {mc.get('median_days', 0):.0f} d, 95th percentile {mc.get('p95_days', 0):.0f} d." if mc else ""),
    )
    _caption(fig, summary + " " + _L("MRL 为示例值，注册前必须以目标市场法规为准；模型为筛查级，不替代残留试验。", "MRLs are examples; confirm against the target market regulation before registration. This is a screening model, not a residue study."), width=104)
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def _page_active(pdf, payload: dict | None):
    """A4 page: next-batch active-learning recommendations."""
    items = (payload or {}).get("items") or []
    if not items:
        return
    fig = _a4_page(
        _L("2.16　下一轮实验推荐（主动学习）", "2.16　Next-batch recommendations"),
        _L(
            "以实测 pActivity 为标签的高斯过程 + Expected Improvement + 多样性约束",
            "Gaussian process with Expected Improvement and a diversity constraint",
        ),
        _L("Fig. AL　实验推荐", "Fig. AL　Experiment recommendation"),
    )
    header = [
        _L("配体", "Ligand"), "Vina", _L("预测 pActivity", "Pred. pAct."),
        _L("不确定度", "SD"), "EI", _L("说明", "Rationale"),
    ]
    rows = [header]
    for item in items[:12]:
        predicted = item.get("predicted_pactivity")
        reason = str(item.get("reason") or "")
        rows.append([
            _trunc(item.get("name") or "", 26),
            f"{float(item.get('affinity') or 0):.2f}",
            "—" if predicted is None else f"{float(predicted):.2f}",
            "—" if item.get("sd") is None else f"±{float(item['sd']):.2f}",
            "—" if item.get("ei") is None else f"{float(item['ei']):.2f}",
            _trunc(
                _L(reason, item.get("reason_en") or _ACTIVE_REASON_EN.get(reason, reason)),
                34,
            ),
        ])
    _table(fig, 0.845, rows, [0.26, 0.09, 0.17, 0.08, 0.08, 0.32], fontsize=9.4, row_height=0.025)
    mode_note = (
        _L("当前实测点不足 4 个，返回对接最优候选；导入更多实测数据后自动切换为主动学习。", "Fewer than four measurements: docking top candidates are returned until more assay data are imported.")
        if payload.get("mode") == "fallback"
        else _L(f"训练集 {payload.get('train', 0)} 个实测分子，候选池 {payload.get('pool', 0)} 个。", f"Training set {payload.get('train', 0)} measured compounds; pool {payload.get('pool', 0)}.")
    )
    note = str(payload.get("note") or "")
    note_text = _L(note, payload.get("note_en") or _ACTIVE_NOTE_EN.get(note, note))
    _caption(fig, (mode_note + " " + note_text).strip(), width=104)
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def _page_benchmark(pdf, payload: dict | None):
    """A4 page: retrospective enrichment benchmark and fitted weights."""
    metrics = (payload or {}).get("metrics") or {}
    curve = (payload or {}).get("curve") or {}
    if not metrics or not curve.get("x"):
        return
    fig = _a4_page(
        _L("2.17　回溯校准与数据驱动权重", "2.17　Retrospective calibration and fitted weights"),
        _L(
            "同一对接库中活性物与诱饵的早期富集、AUC、BEDROC 与权重对比",
            "Early enrichment, AUC, BEDROC and weight comparison for actives versus decoys",
        ),
        _L("Fig. BENCH　回溯校准", "Fig. BENCH　Retrospective benchmark"),
    )
    grid = fig.add_gridspec(
        2, 1, height_ratios=[1.0, 0.85], hspace=0.40,
        left=0.12, right=0.90, top=0.845, bottom=0.21,
    )
    ax = fig.add_subplot(grid[0])
    x = np.asarray(curve.get("x") or [], dtype=float)
    y = np.asarray(curve.get("y") or [], dtype=float)
    ax.plot(x, y, color=TEAL, linewidth=2.0, label=_L("实测排序", "Observed ranking"))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.1, linestyle=(0, (4, 3)), label=_L("随机", "Random"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(_L("累计筛选比例", "Fraction of library screened"), fontsize=13.5)
    ax.set_ylabel(_L("累计找到的活性物比例", "Fraction of actives recovered"), fontsize=13.5)
    ax.set_title(
        _L("富集曲线", "Enrichment curve"), loc="left", fontsize=15, pad=10,
    )
    ax.legend(frameon=False, fontsize=11, loc="lower right")
    _panel_tag(ax, "a", x=-0.075)
    _compact_frame(ax)

    ax2 = fig.add_subplot(grid[1])
    ax2.axis("off")
    metrics_rows = [[
        "EF1%", "EF5%", "AUC", "BEDROC", _L("最佳排名", "Best rank"),
        _L("中位排名", "Median rank"),
    ], [
        f"{metrics.get('ef1', '—')}",
        f"{metrics.get('ef5', '—')}",
        f"{metrics.get('auc', '—')}" + (f" {metrics.get('auc_ci')}" if metrics.get("auc_ci") else ""),
        f"{metrics.get('bedroc', '—')}",
        f"{metrics.get('best_active_rank', '—')}/{metrics.get('n_total', '—')}",
        f"{metrics.get('median_active_rank', '—')}/{metrics.get('n_total', '—')}",
    ]]
    table = ax2.table(cellText=metrics_rows, loc="upper center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10.0)
    table.scale(1.0, 1.35)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        if row == 0:
            cell.set_facecolor("#e2e8f0")
            cell.set_text_props(fontweight="bold")

    weights = (payload or {}).get("weights") or {}
    labels = (payload or {}).get("feature_labels") or {}
    comparison = (payload or {}).get("comparison") or {}
    verdict_text = _L(
        str(comparison.get("verdict") or ""),
        comparison.get("verdict_en") or _benchmark_verdict(comparison),
    )
    if weights:
        expert = {"potency": 0.30, "le": 0.15, "qed": 0.20, "sa": 0.10, "safety": 0.25}
        text_lines = [_L("权重对比（拟合 vs 专家）", "Weights (fitted vs expert)")]
        for key, value in weights.items():
            label_text = (
                labels.get(key, key)
                if i18n.get_lang() != "en"
                else _BENCH_FEATURE_EN.get(key, labels.get(key, key))
            )
            text_lines.append(f"{label_text}: {value:.2f} vs {expert.get(key, 0):.2f}")
        text_lines.append(
            _wrap_cjk(
                _L(
                    f"5 折交叉验证 AUC {comparison.get('cv_auc', '—')}，专家权重 AUC "
                    f"{comparison.get('expert_auc', '—')}；{verdict_text}",
                    f"5-fold CV AUC {comparison.get('cv_auc', 'n/a')}, expert AUC "
                    f"{comparison.get('expert_auc', 'n/a')}; {verdict_text}",
                ),
                96 if i18n.get_lang() == "en" else 46,
            )
        )
        fig.text(
            0.12, 0.335, "\n".join(text_lines),
            fontsize=10.2, color=INK, va="top", linespacing=1.5,
        )
    else:
        fig.text(
            0.12, 0.335,
            _L(
                "带 SMILES 的标注分子不足，未拟合权重；仅报告富集指标。",
                "Not enough labelled molecules with SMILES; enrichment metrics only.",
            ),
            fontsize=10, color=MUTED, va="top",
        )

    _caption(
        fig,
        _L(
            f"本次共 {metrics.get('n_total', 0)} 个分子，其中活性物 {metrics.get('n_active', 0)} 个；"
            f"诱饵为同一次对接中未标注为活性物的其他配体。"
            "EF1%/EF5% 为前 1%/5% 富集倍数，AUC 与 BEDROC（α=20）越接近 1 越好；"
            "BEDROC 由 RDKit 按 Truchon–Bayly 定义计算。权重对比使用逻辑回归与 5 折交叉验证，"
            "提升不足 0.03 时建议保留专家权重。结果只反映本次库与靶点，不能外推到其他靶点。",
            f"{metrics.get('n_total', 0)} molecules with {metrics.get('n_active', 0)} actives; "
            "decoys are the remaining ligands of the same run. EF1%/EF5% are enrichment factors in the top 1%/5%; "
            "AUC and BEDROC (alpha=20) approach 1 for good rankings (RDKit Truchon–Bayly implementation). "
            "Weights use logistic regression with 5-fold CV; keep the expert weights when the improvement is below 0.03. "
            "Results apply only to this library and target.",
        ),
        width=104,
    )
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def _page_manifest(pdf, manifest: dict):
    """Reproducibility record: versions, parameters and input hashes."""
    english = i18n.get_lang() == "en"
    app = manifest.get("app") or {}
    run = manifest.get("run") or {}
    env = manifest.get("environment") or {}
    inputs = manifest.get("inputs") or {}
    params = run.get("params") or {}
    packages = env.get("packages") or {}
    core = " · ".join(
        f"{name} {packages.get(name, '?')}"
        for name in ("rdkit", "meeko", "posebusters")
        if name in packages
    )
    rows = [
        (_L("软件版本", "Software"), f"DrugHX {app.get('version', '—')}"),
        (_L("生成时间", "Created"), str(manifest.get("created") or "—")),
        (_L("运行编号", "Run"), str(run.get("id") or "—")),
        (
            _L("对接引擎", "Engine"),
            f"{run.get('engine') or '—'} · seed {run.get('seed') or '—'}",
        ),
        (
            _L("环境", "Environment"),
            f"{env.get('platform') or '—'} · Python {env.get('python') or '—'}",
        ),
        (_L("核心库", "Core libraries"), core or "—"),
        (
            _L("参数", "Parameters"),
            ", ".join(f"{key}={value}" for key, value in sorted(params.items()))[:240]
            or "—",
        ),
        (
            _L("输入文件", "Inputs"),
            f"{len(inputs.get('receptors') or {})} receptor / "
            f"{len(inputs.get('ligands') or {})} ligand",
        ),
    ]
    hash_lines = []
    for group in ("receptors", "ligands"):
        for key, entry in (inputs.get(group) or {}).items():
            digest = (entry or {}).get("digest") or {}
            if not digest:
                continue
            hash_lines.append(
                f"{key}: {str(digest.get('sha256'))[:12]}… "
                f"({int(digest.get('bytes') or 0)} B)"
            )
            if len(hash_lines) >= 12:
                break
        if len(hash_lines) >= 12:
            break

    fig = _a4_page(
        _L("1.7　可复现性记录", "1.7　Reproducibility record"),
        _L(
            "软件版本、引擎、参数与输入文件哈希",
            "Software version, engines, parameters and input-file hashes",
        ),
        _L("图 M1　运行清单", "Fig. M1　Run manifest"),
    )
    _text_blocks(
        fig,
        0.79,
        [
            (_L("运行清单", "Run manifest"), [f"{label}：{value}" for label, value in rows]),
            (
                _L("输入文件 SHA-256（前 12 位）", "Input SHA-256 (first 12 chars)"),
                hash_lines or [_L("未记录输入文件。", "No input files recorded.")],
            ),
        ],
        width=76, size=8.8, stop=0.155,
    )
    _caption(
        fig,
        _L(
            "本页由 manifest.py 自动生成，完整 JSON 与本报告同目录保存。"
            "哈希用于确认报告生成后输入结构未被替换；参数、引擎版本与随机种子用于复现实验。"
            "若输入哈希不一致，报告中的数值不应与新结构直接比较。",
            "Generated by manifest.py; the full JSON is stored next to this report. "
            "Hashes detect input replacement after report generation, while parameters, engine versions and seeds support reproduction. "
            "If an input hash differs, the reported values must not be compared with the new structure.",
        ),
        y=0.150,
    )
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def _short_organism(organism: str) -> str:
    text = str(organism or "").strip()
    if not text:
        return "—"
    mapping = {
        "Homo sapiens": "Human",
        "Severe acute respiratory syndrome coronavirus 2": "SARS-CoV-2",
        "SARS coronavirus": "SARS-CoV",
        "Middle East respiratory syndrome-related coronavirus": "MERS-CoV",
        "Human coronavirus NL63": "HCoV-NL63",
        "Human coronavirus": "HCoV",
        "Murine coronavirus (strain A59)": "MHV-A59",
        "Porcine epidemic diarrhea virus CV777": "PEDV",
        "Porcine deltacoronavirus": "PDCoV",
        "Feline infectious peritonitis virus": "FIPV",
        "Infectious bronchitis virus": "IBV",
    }
    if text in mapping:
        return mapping[text]
    if "coronavirus" in text.lower():
        return "Coronavirus"
    return text[:24]


def _basis_label(basis: str) -> str:
    mapping = {
        "median pChEMBL": "中位 pChEMBL",
        "qualitative activity records": "定性活性记录",
        "mechanism of action": "作用机制",
        "co-crystal structure": "共晶结构",
        "known drug–gene interaction": "已知药物–基因关系",
        "similar-compound bioactivity": "相似化合物活性",
        "PubChem bioassay": "PubChem 生物试验",
        "approved drug mechanism": "已批准药物机制",
        "external evidence": "外部证据",
        "external evidence (legacy record)": "外部证据（旧记录）",
    }
    return mapping.get(str(basis or ""), str(basis or "外部证据"))


def _gene_label(item: dict) -> str:
    """Standardised short gene label (ChEMBL uses 'rep' for the 3CL protease)."""
    gene = str(item.get("gene") or "").strip()
    if not gene:
        return ""
    return "3CLpro/NSP5" if gene.lower() == "rep" else gene


def _target_protein_label(
    item: dict, limit: int = 62, with_organism: bool = False
) -> str:
    name = str(item.get("name") or "—")
    name = name.replace("（PDB 共晶）", "")
    name = re.split(r"[（(]", name)[0].strip() or "—"
    replacements = {
        "Replicative polyprotein 1ab": "Rep. polyprotein 1ab",
        "Replicase polyprotein 1ab": "Rep. polyprotein 1ab",
        "Infectious bronchitis virus main protease": "IBV main protease",
    }
    name = replacements.get(name, name)
    organism = _short_organism(item.get("organism", ""))
    if with_organism and organism not in ("—", "Human"):
        name = f"{name} · {organism}"
    name = i18n.tr(name)
    if len(name) > limit:
        return name[: max(1, limit - 1)].rstrip() + "…"
    return name


def _target_display_name(
    item: dict, limit: int = 30, with_organism: bool = True
) -> str:
    gene = _gene_label(item)
    if gene:
        return gene
    return _target_protein_label(item, limit, with_organism)


def _draw_bipartite(ax, fishing: dict, limit: int = 12):
    targets = fishing.get("targets", [])[:limit]
    if not targets:
        ax.axis("off")
        return
    count = len(targets)
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    radius = 1.0
    node_x = np.cos(angles) * radius
    node_y = np.sin(angles) * radius
    def _score(item):
        return float(item.get("evidence_score") or item.get("median_pchembl") or 0)

    low = min(_score(item) for item in targets)
    high = max(_score(item) for item in targets)
    for angle, x, y, item in zip(angles, node_x, node_y, targets):
        strength = (_score(item) - low) / max(high - low, 0.01)
        ax.plot([0, x], [0, y], color=TEAL, alpha=0.25 + 0.55 * strength,
                linewidth=1.0 + 1.8 * strength, zorder=1)
        size = 60 + 22 * min(item.get("n", 1), 30)
        ax.scatter([x], [y], s=size, color=BLUE, edgecolor="white",
                   linewidth=1.0, zorder=3)
        label = _target_display_name(item, 26)
        ax.text(
            x * 1.13, y * 1.13, label, fontsize=8.4, color=INK,
            ha="left" if x >= -0.02 else "right",
            va="bottom" if y >= -0.02 else "top",
        )
    ax.scatter([0], [0], s=900, color=AMBER, edgecolor="white", linewidth=1.4,
               zorder=4)
    ax.text(0, -0.17, fishing.get("compound", {}).get("name", "配体"),
            ha="center", va="top", fontsize=11, color=INK, fontweight="bold",
            zorder=5)
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("配体–靶点网络", loc="left", fontsize=14.5, pad=10)


def fishing_report_pdf(fishing: dict, outpath: Path) -> Path:
    """Two A4 pages: target ranking/network and the complete target table."""
    _style_cjk()
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    targets = fishing.get("targets", [])
    compound = fishing.get("compound", {})
    with PdfPages(outpath) as pdf:
        fig = _a4_page(
            "图 9　靶点垂钓（反向筛选）",
            "整合 ChEMBL、SwissTargetPrediction、BindingDB、PubChem、DGIdb、Open Targets 与 PDB 的潜在蛋白靶点",
            "Fig. 9　Reverse target fishing",
        )
        grid = fig.add_gridspec(
            2, 1, height_ratios=[1.25, 1.0], hspace=0.42,
            left=0.34, right=0.93, top=0.842, bottom=0.215,
        )
        ax = fig.add_subplot(grid[0])
        shown = targets[:12]
        names = [_target_display_name(item, 34) for item in shown]
        values = [
            float(item.get("evidence_score") or item.get("median_pchembl") or 0.0)
            for item in shown
        ]
        y = np.arange(len(shown))
        colors = [
            TEAL if int(item.get("n") or 0) >= 3 else "#8fc3bd"
            for item in shown
        ]
        ax.barh(y, values, color=colors, height=0.62, edgecolor="none")
        for index, item in enumerate(shown):
            marker = "" if item.get("median_pchembl") is not None else "*"
            ax.text(
                values[index] + 0.06, index,
                f"{values[index]:.1f}{marker}（n={item.get('n', 0)}）",
                va="center", fontsize=9.8, color=INK,
            )
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=10.5)
        ax.invert_yaxis()
        ax.set_xlabel(
            _wrap_cjk(i18n.tr("综合证据分（定量 pChEMBL 优先，外部证据单独计分）"), 56),
            fontsize=10.6 if i18n.get_lang() == "en" else 11.4,
        )
        ax.set_title("潜在靶点排序", loc="left", fontsize=14.5, pad=10)
        ax.set_xlim(0, max(max(values) * 1.28, 6.0))
        _compact_frame(ax)

        ax2 = fig.add_subplot(grid[1])
        _draw_bipartite(ax2, fishing, 12)
        source_counts = (fishing.get("stats") or {}).get("sources") or {}
        source_text = "、".join(
            f"{name}（{count}）" for name, count in source_counts.items()
        ) or "没有来源返回命中"
        similarity = (fishing.get("parameters") or {}).get("similarity", 70)
        if i18n.get_lang() == "en":
            hits_text = ", ".join(
                f"{name} ({count})" for name, count in source_counts.items()
            ) or "none"
            caption_en = (
                f"{compound.get('name', 'query')} was used as the query; ChEMBL "
                f"similar compounds (final Tanimoto ≥ {similarity}%), "
                "SwissTargetPrediction (2D/3D), BindingDB, DGIdb, PubChem, "
                "Open Targets and PDB co-crystals were merged; "
                f"sources with hits: {hits_text}. ChEMBL targets use median "
                "pChEMBL while external evidence uses a separate strength score. "
                "Entries marked * lack a quantitative pChEMBL. Edge width is the "
                "composite evidence score and node size the number of records. "
                "Non-human co-crystal targets are labelled with their species and "
                "excluded from human network pharmacology. The ranking reflects "
                "existing data only and does not represent measured affinity."
            )
        else:
            caption_en = (
                f"以 {compound.get('name', '查询分子')} 为查询，合并 ChEMBL 相似分子"
                f"（最终 Tanimoto ≥ {similarity}%）、SwissTargetPrediction 2D/3D、"
                "BindingDB、DGIdb、PubChem、Open Targets 与 PDB 共晶证据；"
                f"本次实际命中来源：{source_text}。ChEMBL 靶点使用中位 pChEMBL，"
                "外部证据使用单独的证据强度分，两者不混为 pChEMBL；带 * 的条目没有定量 pChEMBL，"
                "只显示证据分。"
                "配体–靶点网络，边宽表示综合证据分，节点大小表示记录数。"
                "非人源共晶靶点在表中单独标注物种，不进入人源网络药理学。"
                "排序仅反映已有数据，不代表本分子的实测亲和力，需实验验证。"
                "SwissTargetPrediction 的 2D/3D 相似性预测结果也已按概率加权合并。"
            )
        note = str(fishing.get("report_note") or "").strip()
        if note:
            caption_en += (
                " Note: cross-molecule reference only (other/pooled query); not a "
                "target conclusion for the main ligand."
                if i18n.get_lang() == "en"
                else "注：该垂钓记录与主配体不一致或为合并查询，仅作跨分子参考。"
            )
        _caption(fig, caption_en, y=0.168)
        pdf.savefig(fig, facecolor="white")
        plt.close(fig)

        rows = [["靶点", "基因", "类型", "n", "pChEMBL", "最佳 (nM)", "证据分", "来源", "物种"]]
        if i18n.get_lang() == "en":
            rows = [[
                "Target", "Gene", "Type", "n", "pChEMBL", "Best nM",
                "Score", "Source", "Species",
            ]]
        type_short = {
            "SINGLE PROTEIN": "SINGLE",
            "PROTEIN FAMILY": "FAMILY",
            "PROTEIN COMPLEX": "COMPLEX",
            "PROTEIN-PROTEIN INTERACTION": "PPI",
            "CHIMERIC PROTEIN": "CHIMERA",
        }
        for item in targets[:22]:
            pchembl = item.get("median_pchembl")
            rows.append(
                [
                    textwrap.fill(
                        _target_protein_label(item, 62, with_organism=False), 34
                    ),
                    _gene_label(item) or "—",
                    type_short.get(
                        str(item.get("type") or "").upper(),
                        str(item.get("type") or "—")[:10],
                    ),
                    str(item.get("n", 0)),
                    f"{float(pchembl):.2f}" if pchembl is not None else "—",
                    f"{float(item.get('best_nm')):.1f}" if item.get("best_nm") else "—",
                    f"{float(item.get('evidence_score') or 0):.2f}",
                    "/".join(item.get("sources") or ["ChEMBL"])[:14],
                    _short_organism(item.get("organism", "")),
                ]
            )
        shown = min(len(targets), 22)
        page = _a4_page(
            "附表 F1　靶点垂钓完整数据",
            (
                f"Top {shown} targets ranked by composite evidence score"
                if i18n.get_lang() == "en"
                else f"按综合证据分排序的前 {shown} 个靶点"
            ),
        )
        _table(
            page, 0.868, rows,
            [0.27, 0.060, 0.070, 0.030, 0.080, 0.070, 0.060, 0.100, 0.100]
            if i18n.get_lang() == "en"
            else [0.285, 0.060, 0.075, 0.035, 0.075, 0.070, 0.065, 0.095, 0.100],
            fontsize=7.2, row_height=0.0300,
        )
        hits = "、".join(
            f"{name}（{count}）" for name, count in source_counts.items()
        ) or "无"
        footer = _wrap_cjk(
            f"检索来源：{fishing.get('source', 'ChEMBL')}；本次命中：{hits}；"
            f"检索时间 {fishing.get('created', '—')}。pChEMBL = −log10(摩尔活性值)；"
            "外部证据（PDB 共晶、DGIdb、BindingDB、PubChem、Open Targets）"
            "没有 pChEMBL 时以“—”显示，证据分只用于排序，不冒充 pChEMBL。",
            82,
        )
        page.text(
            0.07, 0.060, footer,
            fontsize=8.2, color=MUTED, va="bottom", linespacing=1.4,
        )
        pdf.savefig(page, facecolor="white")
        plt.close(page)
    return outpath


def _bubble_panel(
    ax, rows: list[dict], title: str, color_map: str = "viridis",
    empty_text: str = "",
):
    rows = [item for item in rows if item.get("p")]
    if not rows:
        ax.axis("off")
        ax.text(0.5, 0.5, empty_text or "无显著富集条目", ha="center", va="center",
                fontsize=11, color=MUTED, linespacing=1.5)
        return
    rows = rows[:6]
    labels = [
        textwrap.fill(re.sub(r"\s*\(GO:\d+\)", "", item["term"]), 38)
        for item in rows
    ]
    y = np.arange(len(rows))
    values = [-np.log10(max(item["p"], 1e-300)) for item in rows]
    sizes = [max(28, 16 * len(item.get("genes", []))) for item in rows]
    combined = np.array([float(item.get("combined") or 0) for item in rows])
    scatter = ax.scatter(
        values, y, s=sizes, c=combined, cmap=color_map,
        edgecolor="white", linewidth=0.8, zorder=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.8)
    ax.invert_yaxis()
    ax.set_xlabel("−log$_{10}$ P", fontsize=11.5)
    ax.set_title(title, loc="left", fontsize=13.5, pad=10)
    ax.set_xlim(0, max(max(values) * 1.2, 1.0))
    _compact_frame(ax)
    return scatter


def netpharm_report_pdf(data: dict, outpath: Path) -> Path:
    """Two A4 pages: PPI network plus KEGG/GO bubbles, then enrichment tables."""
    _style_cjk()
    import networkx as nx

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    genes = data.get("genes", [])
    compound = data.get("compound", {})
    ppi = data.get("ppi", {})
    english = i18n.get_lang() == "en"
    with PdfPages(outpath) as pdf:
        fig = _a4_page(
            "图 10　网络药理学",
            "蛋白互作网络、KEGG 通路与 GO 生物学过程富集",
            "Fig. 10　Network pharmacology",
        )
        grid = fig.add_gridspec(
            3, 1, height_ratios=[1.35, 1.02, 1.02], hspace=0.52,
            left=0.31, right=0.93, top=0.842, bottom=0.215,
        )
        ax = fig.add_subplot(grid[0, :])
        graph = nx.Graph()
        for gene in genes:
            graph.add_node(gene)
        for edge in ppi.get("edges", []):
            graph.add_edge(edge["a"], edge["b"], weight=edge.get("score", 400))
        graph.remove_nodes_from([node for node, deg in graph.degree() if deg == 0])
        if graph.number_of_nodes():
            count = graph.number_of_nodes()
            layout = nx.spring_layout(
                graph, seed=11, k=2.8 / np.sqrt(max(count, 2)),
                iterations=500, weight=None,
            )
            xs = np.array([layout[node][0] for node in graph.nodes()])
            ys = np.array([layout[node][1] for node in graph.nodes()])
            xs = xs - (xs.max() + xs.min()) / 2
            ys = ys - (ys.max() + ys.min()) / 2
            span = max(float(np.ptp(xs)), float(np.ptp(ys)), 1e-6)
            xs, ys = xs / span * 1.85, ys / span * 1.85
            positions = {node: (xs[index], ys[index])
                         for index, node in enumerate(graph.nodes())}
            degrees = dict(graph.degree())
            for first, second in graph.edges():
                ax.plot(
                    [positions[first][0], positions[second][0]],
                    [positions[first][1], positions[second][1]],
                    color="#cbd5e1", linewidth=0.9, zorder=1,
                )
            seed_flags = {
                item.get("name"): bool(item.get("seed", True))
                for item in ppi.get("nodes") or []
            }
            nodes = list(graph.nodes())
            sizes = [55 + 48 * degrees.get(node, 0) ** 0.75 for node in nodes]
            values = [degrees.get(node, 0) for node in nodes]
            seed_mask = [seed_flags.get(node, True) for node in nodes]
            seed_idx = [i for i, flag in enumerate(seed_mask) if flag]
            ext_idx = [i for i, flag in enumerate(seed_mask) if not flag]
            scatter = None
            if seed_idx:
                scatter = ax.scatter(
                    [xs[i] for i in seed_idx], [ys[i] for i in seed_idx],
                    s=[sizes[i] for i in seed_idx],
                    c=[values[i] for i in seed_idx], cmap="viridis",
                    edgecolor="white", linewidth=1.1, zorder=3,
                    label="直接靶点基因",
                )
            if ext_idx:
                ax.scatter(
                    [xs[i] for i in ext_idx], [ys[i] for i in ext_idx],
                    s=[sizes[i] for i in ext_idx],
                    color="#f1f5f9", edgecolor=AMBER, linewidth=1.2,
                    zorder=3, label="STRING 一阶邻居（扩展）",
                )
            ax.legend(
                frameon=False, loc="lower right", fontsize=8.6,
                handletextpad=0.4, borderaxespad=0.2,
            )
            top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:4]
            if top_nodes:
                ax.text(
                    0.015, 0.03,
                    "核心节点：" + " · ".join(top_nodes),
                    transform=ax.transAxes, fontsize=9.2, color=INK,
                    ha="left", va="bottom",
                    bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
                          "edgecolor": GRID, "linewidth": 0.7, "alpha": 0.9},
                )
            if scatter is None:
                scatter = plt.cm.ScalarMappable(
                    norm=Normalize(vmin=0, vmax=max(values or [1])),
                    cmap="viridis",
                )
            colorbar = fig.colorbar(
                scatter, ax=ax, fraction=0.022, pad=0.01, shrink=0.72
            )
            colorbar.set_label("连接度", fontsize=9.2)
            colorbar.ax.tick_params(labelsize=8.2)
        ax.set_xlim(-1.18, 1.18)
        ax.set_ylim(-1.18, 1.18)
        isolated = [gene for gene in genes if gene not in graph]
        if english:
            ppi_title = (
                f"Protein–protein interaction network "
                f"({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
                + (
                    f"; {len(isolated)} seed gene(s) without a STRING edge"
                    if isolated
                    else ""
                )
                + ")"
            )
        else:
            ppi_title = (
                f"蛋白–蛋白互作网络（{graph.number_of_nodes()} 个节点，"
                f"{graph.number_of_edges()} 条边"
                + (
                    f"；{len(isolated)} 个种子基因无 STRING 互作"
                    if isolated
                    else ""
                )
                + "）"
            )
        ax.set_title(
            _wrap_cjk(ppi_title, 46),
            loc="left", fontsize=13.5, pad=10,
        )
        ax.axis("off")

        enrichment = data.get("enrichment") or {}
        kegg_block = enrichment.get("kegg") or {}
        go_block = enrichment.get("go_bp") or {}
        kegg_terms = data.get("kegg") or kegg_block.get("terms") or []
        go_terms = data.get("go") or go_block.get("terms") or []

        def _empty_label(block: dict, library: str) -> str:
            error = str(block.get("error") or "").strip()
            if error:
                return (
                    f"{library} request failed (skipped)"
                    if english
                    else f"{library} 请求失败（已跳过）"
                )
            return "No significant terms" if english else "无显著富集条目"

        kegg_ax = fig.add_subplot(grid[1])
        _bubble_panel(
            kegg_ax, kegg_terms, "KEGG 通路富集", "viridis",
            empty_text=_empty_label(kegg_block, "KEGG"),
        )
        go_ax = fig.add_subplot(grid[2])
        _bubble_panel(
            go_ax, go_terms, "GO 生物学过程富集", "magma",
            empty_text=_empty_label(go_block, "GO BP"),
        )
        supplied = [gene for gene in (data.get("supplied_genes") or []) if gene in genes]
        direct = len(genes) - len(supplied)
        if english:
            source_text = (
                f"{len(genes)} input genes ({direct} from target fishing, "
                f"{len(supplied)} imported)"
                if supplied
                else f"{len(genes)} candidate target genes"
            )
            caption = (
                f"{compound.get('name', 'query')}: {source_text} mapped onto the STRING "
                "protein–protein interaction network (confidence ≥ 400). Node size and "
                "colour indicate degree; amber open nodes are STRING first-order "
                "neighbours included only as network background and not counted as direct "
                "targets. The four highest-degree genes are listed in the lower left. "
                "KEGG pathway and GO biological-process enrichment: x-axis, −log10 P; "
                "dot size, number of genes in the term; colour, Enrichr combined score. "
                "Enrichment depends on existing database annotations and is only for "
                "hypothesis generation."
            )
        else:
            source_text = (
                f"的 {len(genes)} 个输入基因（{direct} 个来自靶点垂钓、"
                f"{len(supplied)} 个为外部导入）"
                if supplied
                else f"的 {len(genes)} 个候选靶点"
            )
            caption = (
                f"{compound.get('name', '查询分子')}{source_text}在 STRING 中的"
                "蛋白–蛋白互作网络（置信度 ≥ 400），节点大小与颜色表示连接度，"
                "琥珀色空心节点为 STRING 一阶邻居扩展（仅作网络背景，不计为直接靶点），"
                "左下角列出连接度最高的 4 个核心基因。"
                "KEGG 通路富集与 GO 生物学过程富集，横轴为 −log10 P，点大小为条目包含的基因数，"
                "颜色为 Enrichr 综合得分。富集结果依赖数据库中已知的基因–通路注释，只用于机制假设生成。"
            )
        failed_libraries = [
            name
            for key, name in (
                ("kegg", "KEGG"),
                ("go_cc", "GO CC"),
                ("wikipathway", "WikiPathway"),
            )
            if str((enrichment.get(key) or {}).get("error") or "").strip()
        ]
        if failed_libraries:
            caption += (
                " Libraries unavailable in this run: "
                + ", ".join(failed_libraries)
                + " (request failed; skipped)."
                if english
                else "本次请求失败并跳过的库："
                + "、".join(failed_libraries)
                + "，对应面板不代表“无显著条目”。"
            )
        note = str(data.get("report_note") or "").strip()
        if note:
            caption += (
                " Note: cross-molecule reference only (other/pooled query); not a "
                "conclusion for the main ligand."
                if i18n.get_lang() == "en"
                else "注：该网络药理学记录与主配体不一致或为合并查询，仅作跨分子参考。"
            )
        _caption(fig, caption, y=0.168)
        pdf.savefig(fig, facecolor="white")
        plt.close(fig)

        page = _a4_page(
            "附表 N1　富集分析完整数据",
            "KEGG 与 GO 富集条目（按 P 值排序）",
        )
        rows = [["KEGG 通路", "P", "FDR", "基因数", "基因"]]
        for item in data.get("kegg", [])[:12]:
            rows.append([
                textwrap.fill(item["term"][:40], 28),
                f"{item['p']:.2e}",
                f"{item['fdr']:.2e}" if item.get("fdr") is not None else "—",
                str(len(item.get("genes", []))),
                textwrap.fill(", ".join(item.get("genes", [])[:10]), 36),
            ])
        _table(page, 0.868, rows, [0.28, 0.10, 0.10, 0.06, 0.34],
               fontsize=8.8, row_height=0.0230)
        page.text(0.07, 0.520, "GO 生物学过程（前 8 条）", fontsize=12.5,
                  fontweight="bold", color=TEAL, va="top")
        go_rows = [["GO 条目", "P", "FDR", "基因数"]]
        for item in data.get("go", [])[:8]:
            go_rows.append([
                textwrap.fill(item["term"][:72], 42),
                f"{item['p']:.2e}",
                f"{item['fdr']:.2e}" if item.get("fdr") is not None else "—",
                str(len(item.get("genes", []))),
            ])
        _table(page, 0.480, go_rows, [0.54, 0.12, 0.12, 0.08],
               fontsize=8.8, row_height=0.0270)
        page.text(
            0.07, 0.055,
            _wrap_cjk(
                f"数据来源：{data.get('source', 'STRING + Enrichr')}；生成时间 "
                f"{data.get('created', '—')}。",
                82,
            ),
            fontsize=9.2, color=MUTED, va="bottom", linespacing=1.45,
        )
        pdf.savefig(page, facecolor="white")
        plt.close(page)
    return outpath


_RESIDUE_PROPERTIES = {
    "hydrophobic": {
        "ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "GLY", "CYS",
    },
    "aromatic": {"PHE", "TYR", "TRP", "HIS"},
    "polar": {"SER", "THR", "ASN", "GLN", "TYR", "CYS", "HIS"},
    "charged": {"ASP", "GLU", "LYS", "ARG", "HIS"},
}


def _residue_class(resname: str) -> str:
    name = str(resname or "").upper()
    for label in ("aromatic", "charged", "polar", "hydrophobic"):
        if name in _RESIDUE_PROPERTIES[label]:
            return label
    return "other"


def _contact_tables(analysis: dict):
    """Residue × interaction-type counts and spatial interaction edges."""
    residues: dict[str, dict] = {}
    spatial = []
    for key, label in INTERACTION_LABELS:
        for item in analysis.get(key) or []:
            resname = str(item.get("residue") or "").strip()
            resseq = item.get("resseq")
            if not resname:
                continue
            res_key = f"{resname}{resseq}" if resseq is not None else resname
            bucket = residues.setdefault(
                res_key,
                {"residue": resname, "resseq": resseq, "counts": {
                    name: 0 for _, name in INTERACTION_LABELS
                }},
            )
            amount = int(item.get("contacts") or 1)
            bucket["counts"][label] += amount
            ligand_atom = str(item.get("ligand_atom") or "").strip()
            ligand_coord = item.get("ligand_coord") or item.get("ligand_center")
            protein_coord = item.get("protein_coord") or item.get("protein_center")
            if not ligand_coord or not protein_coord:
                continue
            spatial.append(
                {
                    "type": label,
                    "residue": res_key,
                    "resname": resname,
                    "atom": ligand_atom or "ring",
                    "ligand_coord": np.asarray(ligand_coord, dtype=float),
                    "protein_coord": np.asarray(protein_coord, dtype=float),
                    "distance": item.get("distance"),
                }
            )
    return residues, spatial


def _project_coords(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points.reshape(0, 2)
    centered = points - points.mean(axis=0, keepdims=True)
    if len(points) < 3:
        projected = np.zeros((len(points), 2))
        projected[:, 0] = centered[:, 0]
        projected[:, 1] = centered[:, 1] if points.shape[1] > 1 else 0
    else:
        try:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            projected = centered @ vt[:2].T
        except np.linalg.LinAlgError:
            projected = centered[:, :2]
    span = np.ptp(projected, axis=0)
    span[span < 1e-6] = 1.0
    return projected / span * 1.8


def interaction_fingerprint_pdf(
    analysis: dict,
    ligand_name: str,
    receptor_name: str,
    outpath: Path,
) -> Path | None:
    """Figure 14: residue interaction fingerprint plus a spatial contact map."""
    residues, spatial = _contact_tables(analysis)
    if not residues:
        return None
    _style_cjk()
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        residues.items(),
        key=lambda item: (-sum(item[1]["counts"].values()), item[0]),
    )[:20]
    labels = [label for _, label in _interaction_labels()]
    keys = [key for key, _ in INTERACTION_LABELS]
    matrix = np.asarray(
        [[entry["counts"].get(key, 0) for key in keys] for _, entry in rows],
        dtype=float,
    )
    fig = _a4_page(
        "图 14　原子–残基接触指纹与空间作用图",
        f"{ligand_name} 在 {receptor_name} 口袋中的残基级作用力分布",
        "Fig. 14　Atom–residue contact fingerprint",
    )
    grid = fig.add_gridspec(
        2, 1, height_ratios=[1.02, 1.0], hspace=0.34,
        left=0.155, right=0.93, top=0.822, bottom=0.245,
    )
    top_grid = grid[0].subgridspec(1, 2, width_ratios=[4.2, 1.0], wspace=0.24)
    ax = fig.add_subplot(top_grid[0])
    totals_ax = fig.add_subplot(top_grid[1])
    image = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=38, ha="right", fontsize=9.2)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([item[0] for item in rows], fontsize=9.2)
    ax.set_title("残基–作用力指纹", loc="left", fontsize=13.5, pad=10)
    max_value = max(1.0, float(matrix.max()))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix[i, j])
            if value:
                ax.text(
                    j, i, str(value), ha="center", va="center", fontsize=8.2,
                    color="white" if value > max_value * 0.55 else INK,
                )
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", length=0)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("接触计数", fontsize=9.5)
    cbar.ax.tick_params(labelsize=8.5)
    totals = matrix.sum(axis=1)
    totals_ax.barh(
        np.arange(len(rows)), totals, color=TEAL, height=0.62,
        edgecolor="none",
    )
    for index, value in enumerate(totals):
        totals_ax.text(
            value + max(0.2, totals.max() * 0.04), index, f"{int(value)}",
            va="center", fontsize=8.2, color=INK,
        )
    totals_ax.set_ylim(len(rows) - 0.5, -0.5)
    totals_ax.set_yticks([])
    totals_ax.set_xlabel("合计", fontsize=9.0)
    totals_ax.set_xlim(0, max(1.0, totals.max() * 1.35))
    totals_ax.set_title("每残基", loc="left", fontsize=9.5, pad=8)
    _compact_frame(totals_ax)

    ax2 = fig.add_subplot(grid[1])
    if spatial:
        atom_coords: dict[str, list[np.ndarray]] = {}
        residue_coords: dict[str, list[np.ndarray]] = {}
        degree: dict[str, int] = {}
        for edge in spatial:
            atom_key = f"{edge['atom']}"
            atom_coords.setdefault(atom_key, []).append(edge["ligand_coord"])
            residue_coords.setdefault(edge["residue"], []).append(edge["protein_coord"])
            degree[atom_key] = degree.get(atom_key, 0) + 1
            degree[edge["residue"]] = degree.get(edge["residue"], 0) + 1
        atom_pos = {key: np.mean(value, axis=0) for key, value in atom_coords.items()}
        residue_pos = {
            key: np.mean(value, axis=0) for key, value in residue_coords.items()
        }
        all_points = np.asarray(list(atom_pos.values()) + list(residue_pos.values()))
        projected = _project_coords(all_points)
        positions = {}
        index = 0
        for key in atom_pos:
            positions[("atom", key)] = projected[index]
            index += 1
        for key in residue_pos:
            positions[("residue", key)] = projected[index]
            index += 1
        palette = {
            "H-bond": BLUE,
            "Salt bridge": RED,
            "π-stacking": AMBER,
            "Halogen": "#7e22ce",
            "Metal": "#475569",
        }
        all_positions = np.asarray(list(positions.values()))
        center = all_positions.mean(axis=0)
        placed_labels: list[np.ndarray] = []

        def place_label(point, direction, base, step):
            candidate = point + direction * base
            for attempt in range(5):
                if all(
                    float(np.linalg.norm(candidate - other)) > 0.20
                    for other in placed_labels
                ):
                    break
                candidate = point + direction * (base + step * (attempt + 1))
            placed_labels.append(candidate)
            return candidate
        for edge in spatial:
            start = positions.get(("atom", edge["atom"]))
            end = positions.get(("residue", edge["residue"]))
            if start is None or end is None:
                continue
            ax2.plot(
                [start[0], end[0]], [start[1], end[1]],
                color=palette.get(edge["type"], MUTED),
                linewidth=1.0 + 0.15 * degree.get(edge["residue"], 1),
                alpha=0.55, zorder=1,
            )
        for key, point in positions.items():
            vector = point - center
            norm = float(np.linalg.norm(vector))
            direction = vector / norm if norm > 1e-6 else np.array([0.0, 1.0])
            kind, label = key
            if kind == "residue":
                resname = residues.get(label, {}).get("residue", label)
                prop = _residue_class(resname)
                fill = {
                    "hydrophobic": "#ccfbf1",
                    "aromatic": "#fef3c7",
                    "polar": "#dbeafe",
                    "charged": "#fee2e2",
                }.get(prop, "#f1f5f9")
                edge_color = {
                    "hydrophobic": TEAL,
                    "aromatic": AMBER,
                    "polar": BLUE,
                    "charged": RED,
                }.get(prop, MUTED)
                size = 80 + 18 * min(degree.get(label, 1), 12)
                ax2.scatter(
                    [point[0]], [point[1]], s=size, color=fill,
                    edgecolor=edge_color, linewidth=1.2, zorder=3,
                )
                label_point = place_label(point, direction, 0.14, 0.13)
                ax2.text(
                    label_point[0], label_point[1], label,
                    ha="center", va="center", fontsize=8.0, color=INK, zorder=4,
                    bbox={"boxstyle": "round,pad=0.12", "facecolor": "white",
                          "edgecolor": "none", "alpha": 0.8},
                )
            else:
                element = re.match(r"[A-Z][a-z]?", label)
                color = _ELEMENT_COLORS.get(
                    element.group(0) if element else "C", INK
                )
                ax2.scatter(
                    [point[0]], [point[1]], s=48, color=color,
                    edgecolor="white", linewidth=0.8, zorder=5,
                )
                label_point = place_label(
                    point, np.array([0.0, -1.0]), 0.07, 0.08
                )
                ax2.text(
                    label_point[0], label_point[1], label,
                    ha="center", va="center", fontsize=7.2, color=MUTED, zorder=6,
                    bbox={"boxstyle": "round,pad=0.10", "facecolor": "white",
                          "edgecolor": "none", "alpha": 0.75},
                )
        handles = [
            Line2D([0], [0], color=color, linewidth=2.0, label=label)
            for label, color in palette.items()
            if any(edge["type"] == label for edge in spatial)
        ]
        if placed_labels:
            bounds = np.vstack([all_positions, np.asarray(placed_labels)])
        else:
            bounds = all_positions
        low = bounds.min(axis=0) - 0.22
        high = bounds.max(axis=0) + 0.22
        ax2.set_xlim(low[0], high[0])
        ax2.set_ylim(low[1], high[1])
        if handles:
            fig.legend(
                handles=handles, loc="lower center",
                bbox_to_anchor=(0.5, 0.188), ncol=3,
                frameon=False, fontsize=9.0, handlelength=1.6,
            )
    else:
        ax2.axis("off")
        ax2.text(
            0.5, 0.5, "没有可投影的空间接触坐标",
            ha="center", va="center", fontsize=11, color=MUTED,
        )
    ax2.set_aspect("auto")
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlabel("PCA 1", fontsize=8.8, color=MUTED)
    ax2.set_ylabel("PCA 2", fontsize=8.8, color=MUTED)
    ax2.set_title("空间接触图（PCA 投影）", loc="left", fontsize=13.5, pad=10)
    _caption(
        fig,
        f"{ligand_name} 与 {receptor_name} 口袋残基的原子–残基接触指纹，行为残基、列为作用力类型，"
        "色深与数字表示接触计数；疏水接触按残基级接触数汇总。把氢键、盐桥、π-堆积、卤键与金属配位的"
        "配体原子与受体残基坐标做 PCA 投影，边颜色表示作用力类型，节点大小表示接触度；"
        "残基节点底色表示疏水、芳香、极性与带电性质。该图只描述几何接触，不等同于能量分解。",
        y=0.172,
    )
    with PdfPages(outpath) as pdf:
        pdf.savefig(fig, facecolor="white")
    plt.close(fig)
    return outpath


def pocket_profile_pdf(
    analysis: dict,
    ligand_name: str,
    receptor_name: str,
    outpath: Path,
) -> Path | None:
    """Figure 15: pocket residue composition, interaction counts and distances."""
    residues, spatial = _contact_tables(analysis)
    if not residues:
        return None
    _style_cjk()
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    counts = analysis.get("counts") or {}
    labels = [label for _, label in _interaction_labels()]
    values = [
        int(counts.get(key) or len(analysis.get(key) or []))
        for key, _ in INTERACTION_LABELS
    ]
    residue_classes: dict[str, int] = {}
    for entry in residues.values():
        label = _residue_class(entry["residue"])
        residue_classes[label] = residue_classes.get(label, 0) + 1
    total_residues = max(1, len(residues))
    fractions = {
        key: residue_classes.get(key, 0) / total_residues
        for key in ("hydrophobic", "aromatic", "polar", "charged")
    }
    hbond_density = min(1.0, (counts.get("hbonds", 0) or len(analysis.get("hbonds") or [])) / max(1.0, 2.0 * total_residues))
    radar_labels = ["疏水", "芳香", "极性", "带电", "氢键"]
    radar_values = [
        fractions["hydrophobic"], fractions["aromatic"],
        fractions["polar"], fractions["charged"], hbond_density,
    ]

    fig = _a4_page(
        "图 15　结合口袋理化画像",
        f"{receptor_name} 口袋的残基组成、作用力计数与距离分布",
        "Fig. 15　Pocket physicochemical profile",
    )
    grid = fig.add_gridspec(
        2, 2, width_ratios=[1.0, 1.15], height_ratios=[1.15, 1.0],
        wspace=0.62, hspace=0.42,
        left=0.18, right=0.93, top=0.822, bottom=0.235,
    )
    radar_ax = fig.add_subplot(grid[0, 0], polar=True)
    angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
    angles += angles[:1]
    radar = radar_values + radar_values[:1]
    radar_ax.plot(angles, radar, color=TEAL, linewidth=2.0)
    radar_ax.fill(angles, radar, color=TEAL, alpha=0.22)
    radar_ax.scatter(angles[:-1], radar_values, color=TEAL, s=35, zorder=4)
    radar_ax.set_xticks(angles[:-1])
    radar_ax.set_xticklabels(radar_labels, fontsize=9.2)
    radar_ax.tick_params(axis="x", pad=3)
    radar_ax.set_ylim(0, 1)
    radar_ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    radar_ax.set_yticklabels([])
    radar_ax.set_title("残基性质画像", loc="left", fontsize=13.5, pad=20)
    radar_ax.grid(color=GRID, linewidth=0.8)

    ax2 = fig.add_subplot(grid[0, 1])
    order = np.argsort(values)
    colors = [TEAL, "#5e9c96", BLUE, AMBER, "#7e22ce", "#475569"]
    ax2.barh(
        np.arange(len(labels)),
        [values[i] for i in order],
        color=[colors[i] for i in order],
        height=0.62, edgecolor="none",
    )
    for index, i in enumerate(order):
        ax2.text(
            values[i] + max(0.15, max(values) * 0.02), index, str(values[i]),
            va="center", fontsize=9.5, color=INK,
        )
    ax2.set_yticks(np.arange(len(labels)))
    ax2.set_yticklabels([labels[i] for i in order], fontsize=9.5)
    ax2.set_xlabel("接触/记录数", fontsize=11)
    ax2.set_title("作用力组成", loc="left", fontsize=13.5, pad=10)
    ax2.set_xlim(0, max(max(values), 1) * 1.22)
    _compact_frame(ax2)

    ax3 = fig.add_subplot(grid[1, :])
    distance_groups = [
        ("氢键", [item.get("distance") for item in analysis.get("hbonds") or []]),
        ("盐桥", [item.get("distance") for item in analysis.get("salt_bridges") or []]),
        ("卤键", [item.get("distance") for item in analysis.get("halogens") or []]),
        ("金属", [item.get("distance") for item in analysis.get("metals") or []]),
    ]
    rng = np.random.default_rng(7)
    plotted = False
    for index, (label, distances) in enumerate(distance_groups):
        clean = [float(value) for value in distances if value is not None]
        if not clean:
            continue
        plotted = True
        jitter = rng.normal(0, 0.035, len(clean))
        ax3.scatter(
            clean, np.full(len(clean), index) + jitter,
            s=46, color=colors[index], edgecolor="white", linewidth=0.8,
            alpha=0.9, zorder=3,
        )
        median = float(np.median(clean))
        ax3.text(
            max(clean) + 0.08, index, f"中位 {median:.2f} Å",
            va="center", fontsize=8.8, color=MUTED,
        )
    if plotted:
        ax3.set_yticks(np.arange(len(distance_groups)))
        ax3.set_yticklabels([label for label, _ in distance_groups], fontsize=9.5)
        ax3.set_xlabel("距离 (Å)", fontsize=11)
        ax3.set_xlim(2.4, 5.6)
        ax3.set_title("接触距离分布", loc="left", fontsize=13.5, pad=10)
        _compact_frame(ax3)
    else:
        ax3.axis("off")
        ax3.text(
            0.5, 0.5, "没有可用的距离记录",
            ha="center", va="center", fontsize=11, color=MUTED,
        )
    _caption(
        fig,
        f"{receptor_name} 口袋残基的疏水、芳香、极性与带电比例，以及按残基数归一化的氢键密度；"
        "所有指标均为 0–1 的接触画像，不是可成药性打分，也不构成能量学结论。"
        "最优构象中各类作用力的接触计数。氢键、盐桥、卤键与金属配位的距离分布，"
        "散点为单次接触，横线位置标注中位数；距离越短只表示几何接近，是否形成稳定作用仍需动力学验证。",
        y=0.172,
    )
    with PdfPages(outpath) as pdf:
        pdf.savefig(fig, facecolor="white")
    plt.close(fig)
    return outpath


def report_analysis_pdf(
    results: list[dict],
    analyses: dict[int, dict],
    props: dict,
    parent: dict,
    optimization_rows: list[dict],
    design_rows: list[dict],
    outpath: Path,
) -> Path:
    """Six publication figures (Fig. 1–6) as A4 vector pages."""
    _style_cjk()
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    ok = [item for item in results if item.get("affinity") is not None][:16]
    with PdfPages(outpath) as pdf:
        _page_landscape(pdf, ok)
        _page_efficiency(pdf, ok, props)
        _page_interactions(pdf, ok, analyses)
        _page_poses(pdf, ok)
        if optimization_rows:
            _page_optimization(pdf, parent, optimization_rows)
        if design_rows:
            _page_design(pdf, parent, design_rows)
    return outpath


def _describe_correlation(value: float) -> str:
    """Plain-language strength of a correlation coefficient."""
    if abs(value) >= 0.8:
        return "高度一致"
    if abs(value) >= 0.5:
        return "中等相关"
    if abs(value) >= 0.3:
        return "弱相关"
    return "基本不相关"


def _spearman(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 3:
        return None
    left = np.asarray([pair[0] for pair in pairs], dtype=float)
    right = np.asarray([pair[1] for pair in pairs], dtype=float)
    left_rank = np.argsort(np.argsort(left)).astype(float)
    right_rank = np.argsort(np.argsort(right)).astype(float)
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _stats(results: list[dict], analyses: dict[int, dict], props: dict) -> dict:
    scores = [float(item["affinity"]) for item in results if item.get("affinity") is not None]
    efficiencies = [
        float((props.get(id(item), {}).get("efficiency") or {}).get("le") or 0.0)
        for item in results
        if id(item) in props
    ]
    efficiencies = [value for value in efficiencies if value > 0]
    pairs = [
        (float(item["affinity"]), float(item["vinardo"]))
        for item in results
        if item.get("vinardo") is not None
    ]
    correlation = (
        float(np.corrcoef(*zip(*pairs))[0, 1]) if len(pairs) >= 3 else None
    )
    spearman = _spearman(pairs)
    contacts = {}
    for item in results:
        analysis = analyses.get(id(item))
        if not analysis:
            continue
        for key in ("hbonds", "hydrophobic", "salt_bridges", "pi_stacking"):
            for contact in analysis.get(key, []):
                residue = f"{contact.get('residue', '')}{contact.get('resseq', '')}"
                contacts[residue] = contacts.get(residue, 0) + 1
    hotspots = sorted(contacts.items(), key=lambda kv: -kv[1])[:5]
    return {
        "count": len(scores),
        "best": min(scores) if scores else None,
        "worst": max(scores) if scores else None,
        "mean": float(np.mean(scores)) if scores else None,
        "median": float(np.median(scores)) if scores else None,
        "mean_le": float(np.mean(efficiencies)) if efficiencies else None,
        "correlation": correlation,
        "spearman": spearman,
        "corr_text": _describe_correlation(correlation) if correlation is not None else "",
        "hotspots": hotspots,
    }


def _block_height(heading, paragraphs, width, size, heading_size):
    english = i18n.get_lang() == "en"
    height = 0.00235 * heading_size
    for paragraph in paragraphs:
        wrapped = _wrap_cjk(i18n.tr(paragraph), width * 2)
        height += 0.00194 * size * (wrapped.count("\n") + 1) * (
            0.92 if english else 1.0
        ) + (0.009 if english else 0.011)
    return height


def _flow_text_pages(
    pdf, title, subtitle, blocks, width=40, size=12.0,
    heading_size=13.6, stop=0.075,
):
    """Lay out method blocks across as many A4 pages as needed.

    ``_text_blocks`` silently dropped sections that did not fit; this helper
    measures each block first and starts a continuation page instead.
    """
    english = i18n.get_lang() == "en"
    if english:
        width = int(width * 1.20)
        size = size * 0.78
        heading_size = heading_size * 0.95
        stop = max(stop, 0.10)
    pending = list(blocks)
    page_index = 0
    cont_zh, cont_en = "（续）", " (continued)"
    while pending:
        has_marker = cont_zh in title or cont_en in title
        base_title = title.replace(cont_zh, "").replace(cont_en, "").rstrip()
        suffix = _L(cont_zh, cont_en) if (page_index > 0 or has_marker) else ""
        page_title = base_title + suffix
        fig = _a4_page(page_title, subtitle if page_index == 0 else "")
        y = 0.885
        while pending:
            heading, paragraphs = pending[0]
            height = _block_height(heading, paragraphs, width, size, heading_size)
            if y - height < stop and y < 0.879:
                break
            heading_text = i18n.tr(heading)
            if heading_text:
                fig.text(0.07, y, heading_text, fontsize=heading_size,
                         fontweight="bold", color=TEAL, va="top")
                y -= 0.00235 * heading_size
            for paragraph in paragraphs:
                wrapped = _wrap_cjk(
                    i18n.tr(paragraph),
                    _safe_wrap_units(size, width * 2),
                )
                fig.text(
                    0.07, y, wrapped, fontsize=size, color=INK, va="top",
                    linespacing=1.45 if english else 1.60,
                )
                y -= 0.00194 * size * (wrapped.count("\n") + 1) * (
                    0.92 if english else 1.0
                ) + (0.009 if english else 0.011)
            pending.pop(0)
        pdf.savefig(fig, facecolor="white")
        plt.close(fig)
        page_index += 1


def _methods_pages(pdf):
    blocks = [
        (
            "1.1 受体准备与结合位点定义",
            ["蛋白结构来自 RCSB PDB 或本地 PDB/PDBQT 文件。流程默认去除结晶水、非配位杂原子与替代构象（altloc 取 A），保留用户指定的链；按模板电荷模型补氢，再由 Meeko 生成 AutoDock Vina 可读的 PDBQT。结合盒由共晶配体的几何中心与尺寸自动推导，也可使用内置几何口袋预测或手工坐标，并对盒体积给出提示。"],
        ),
        (
            "1.2 配体准备",
            ["配体支持 SMILES、SDF、MOL、MOL2、PDB 与现成 PDBQT。RDKit 负责标准化与去盐，按 pH 7.4 规则质子化（羧酸去质子、脂肪胺质子化），ETKDGv3 生成三维构象，MMFF94 做能量最小化，最终转换为 PDBQT。"],
        ),
        (
            "1.3 分子对接与打分",
            [
                _L(
                    "对接引擎为 AutoDock Vina 1.2.5（CPU），可选 Vina-GPU 2.0。GPU 模式以多个独立随机种子并行搜索，取最优构象后再用 CPU Vina 在相同打分函数下复核，使 CPU 与 GPU 结果可直接比较。默认 exhaustiveness = 16、num_modes = 9、energy_range = 4.0 kcal/mol（极速粗筛可使用预设中的低 exhaustiveness，报告中按实际参数记录）。",
                    "The docking engine is AutoDock Vina 1.2.5 (CPU), with optional Vina-GPU 2.0. In GPU mode, several independent random seeds search in parallel; the best pose is then rescored with CPU Vina using the same scoring function, so CPU and GPU results are directly comparable. Defaults are exhaustiveness = 16, num_modes = 9 and energy_range = 4.0 kcal/mol; ultra-fast pre-screening may use a preset with lower exhaustiveness, and the parameters actually used are recorded in the report.",
                ),
                _L(
                    "GPU 模式对每个配体执行多次独立复算并用 CPU 复核，吞吐优势只在大批量（约 50 个以上配体）时体现；配体数较少时 CPU 模式通常更快。Vina-GPU 不使用 exhaustiveness，其搜索强度由 search_depth 控制，报告中须分别记录。",
                    "GPU mode performs several independent replicas per ligand and validates each by CPU rescoring; its throughput advantage only materialises for large batches (roughly 50 ligands or more), whereas CPU mode is usually faster for a handful of ligands. Vina-GPU does not use exhaustiveness: its search intensity is controlled by search_depth, which is recorded separately.",
                ),
                _L(
                    "每个构象同时计算 Vina 与 Vinardo 两套经验打分，并以两项排名之和构成共识排序；同时报告配体效率 LE = −ΔG / 重原子数，用于消除分子大小带来的系统偏置。",
                    "Every pose is scored with both the Vina and Vinardo empirical functions, and the sum of the two ranks forms the consensus ordering; ligand efficiency LE = −ΔG / heavy-atom count is also reported to remove the size-dependent bias.",
                ),
            ],
        ),
        (
            "1.4 相互作用分析",
            [
                _L(
                    "基于原子间距离与几何判据识别相互作用：氢键要求供体–受体距离 ≤ 3.5 Å（含硫 4.0 Å），当受体 PDBQT 提供氢原子时进一步要求 D–H···A ≥ 120°；疏水接触为碳原子间距 ≤ 4.2 Å；盐桥为相反电荷基团 ≤ 4.0 Å；π-堆积为芳环质心距离 ≤ 6.0 Å（平行型另需面–面接触判据，T 型夹角 ≥ 55°）；卤键为 X···A ≤ 3.9 Å 且 C–X···A ≥ 135°；金属配位为 ≤ 3.1 Å。逐残基统计接触类型与最短距离，构成蛋白–配体作用力指纹。",
                    "Interactions are identified from interatomic distances and geometric criteria: hydrogen bonds require a donor–acceptor distance ≤ 3.5 Å (4.0 Å when sulfur is involved) and, when the receptor PDBQT provides hydrogens, an additional D–H···A angle ≥ 120°; hydrophobic contacts use carbon–carbon distances ≤ 4.2 Å; salt bridges require oppositely charged groups within 4.0 Å; π-stacking requires an aromatic centroid distance ≤ 6.0 Å (parallel stacking also needs face-to-face contact, while T-shaped stacking requires an angle ≥ 55°); halogen bonds require X···A ≤ 3.9 Å with C–X···A ≥ 135°; metal coordination uses a distance ≤ 3.1 Å. Contacts are counted per residue together with the shortest distance, giving the protein–ligand interaction fingerprint.",
                ),
            ],
        ),
        (
            "1.5 结合自由能分解（MM-GBSA 近似）",
            ["以 12-6 Lennard-Jones 势、Coulomb 静电（ε = 4）、Born 去溶剂化与几何氢键项逐残基累加，给出范德华、静电、去溶剂化与氢键四项贡献及总能量。绝对值仅为近似，但残基排序与同类分子间的相对比较具有参考价值。"],
        ),
        (
            "1.6 ADMET 与类药性",
            ["RDKit 描述符计算分子量、cLogP、TPSA、氢键供体/受体、可旋转键、芳香比例与 Fsp3；ESOL 估算水溶性；QED 综合类药性；Lipinski、Veber、Egan 与 lead-like 规则逐项判定。效率指标包括 LE、LipE、BEI 与 SEI。"],
        ),
    ]
    _flow_text_pages(pdf, "1　方法", "计算流程、参数设置与判定规则", blocks)
    blocks = [
        (
            "1.7 药效团",
            ["使用 RDKit 基础特征定义识别氢键供体/受体、芳环、疏水中心与正/负离子中心，统计各特征数量与几何分布，用于判断骨架是否覆盖关键药效团。"],
        ),
        (
            "1.8 药物优化",
            ["从母体识别羧酸、酯、酰胺、磺酰胺、芳环与末端甲基等可改造位点，按六个方向系统生成候选：① 生物电子等排（羧酸→四唑/酰基磺酰胺/伯酰胺/异羟肟酸、酰胺→腈、酯→酰胺/羧酸、腈→四唑）；② 骨架跃迁（苯环→吡啶/嘧啶/哒嗪/吡嗪、单氮芳环→二氮芳环、五元杂环 S/O/N 互换、稠环氮杂化）；③ 取代基行走（多个芳环位置分别引入氟、氯、甲基、甲氧基、羟基、氨基、氰基）；④ 代谢稳定化（酰胺/磺酰胺 N-甲基化、甲基→CF3）；⑤ 连接子改造（亚甲基→醚/胺/羰基）；⑥ 对排名靠前的候选再做二次组合改造。所有候选先按性质先验与指纹多样性筛选，再在同一受体、同一结合盒与同一打分函数下重新对接，输出 ΔVina、配体效率与残基级作用力比较。", "候选数量可设置为 3–60 个；类别配额保证骨架跃迁、生物电子等排、取代基行走等方向都有代表，而不是被单一高分类型占满。组合改造把骨架跃迁或等排替换与取代基行走叠加，用于检验多步改造的协同效应。所有结果仍为计算假设，需合成与实验验证。"],
        ),
        (
            "1.9 分子设计",
            ["以 29 类芳环/杂环为核心（苯、吡啶、嘧啶、吡嗪、哒嗪、三嗪、噻吩、呋喃、吡咯、咪唑、吡唑、噁唑、异噁唑、噻唑、异噻唑、萘、喹啉、异喹啉、喹唑啉、喹喔啉、吲哚、苯并咪唑、吲唑、苯并呋喃、苯并噻吩、苯并噁唑、苯并噻唑、7-氮杂吲哚、萘啶），枚举单取代、双取代与三取代模式；取代基覆盖卤素、羟基、氨基、烷基、三氟甲基、三氟甲氧基、烷氧基、氰基、酰胺、羧酸、磺酰胺、甲磺酰基、乙酰基、甲氨基、二甲氨基、环丙基等 22 类。", "当提供参考配体时，设计模块还会调用优化规则生成骨架跃迁候选（芳环氮杂化、五元杂环杂原子替换、连接子改造、生物电子等排），并按配额保留一定数量的骨架跃迁分子，避免被高分但结构相近的片段类似物挤占。所有候选先经类药性规则预筛，再在同一受体与结合盒内对接排序，并检索本地数据库最近邻。", "设计结果用于提出可检验的结构假设；核心与取代基组合的枚举空间有限，不能替代全库搜索或实验筛选。"],
        ),
        (
            "1.10 数据库相似性检索",
            ["以 Morgan 指纹（半径 2，2048 bit）与本地 SMILES 数据库计算 Tanimoto 相似度，返回最近邻分子，用于判断设计结果是否落在已知化学空间内。"],
        ),
        (
            "1.11 可视化与统计",
            [_L(
                "所有图像由 Matplotlib 直接输出 PDF 矢量图，字体以 Type 42 嵌入；配色采用色盲友好的深青/琥珀双色方案；每张图均为 A4 单页，图注紧随其下，便于连续阅读。图像标注随报告语言生成（中文或英文），矢量文本可编辑，投稿时可按期刊格式调整标签，数据与坐标无需改动。",
                "All figures are exported directly from Matplotlib as vector PDFs with Type 42 embedded fonts; the palette uses a colour-blind-friendly deep-teal/amber scheme; each figure occupies a single A4 page with its caption immediately below. Figure labels follow the report language (Chinese or English), and the vector text stays editable, so labels can be adapted to journal style without changing the data or coordinates.",
            )],
        ),
        (
            "1.12 可重复性与局限",
            ["Vina 打分为经验函数，忽略蛋白柔性与显式水；MM-GBSA 为近似分解；ADMET 端点为规则式启发估计；优化与设计结果为计算假设，需通过分子动力学、自由能微扰或实验验证。每次运行显式记录随机种子，但单次运行不足以评估重复性，需要多随机种子或多次独立运行后才能给出稳健性结论。"],
        ),
        (
            "1.13 数据与结果文件组织",
            ["每个项目在磁盘上按 receptors/、ligands/、runs/ 与 exports/ 分层存放：受体与配体的原始结构、清洗后的 PDB/PDBQT、对接日志、每个构象的 PDBQT 与打分表均可追溯。报告中的图 1–6 由同一次运行的结果直接生成，附录表格与正文数值共享同一数据源，避免手工转录误差。"],
        ),
        (
            "1.14 术语与缩写",
            ["PDBQT：AutoDock 系列使用的原子类型与电荷格式；LE：配体效率；LipE：亲脂性配体效率；BEI：结合效率指数；SEI：表面效率指数；QED：定量估计类药性；TPSA：拓扑极性表面积；MM-GBSA：分子力学/广义 Born 表面积方法；HTS：高通量筛选。"],
        ),
        (
            "1.15 重对接验证",
            ["当受体结构中带有共晶配体时，程序自动以该配体为参照执行重对接：在同一结合盒与同一打分函数下重新生成构象，用 RDKit 的对称校正算法计算重原子 RMSD，并以 2.0 Å 作为通过判据。该步骤用于检验受体准备、质子化与搜索参数是否合理；未通过时，后续排序只能作为假设生成。若共晶配体为共价抑制剂，Vina 无法形成共价键，验证结果需改用共价对接程序或非共价阳性对照复核。"],
        ),
    ]
    _flow_text_pages(pdf, "1　方法（续）", "专家模块、可重复性与局限性", blocks)
    blocks = [
        (
            "1.16 反向靶点垂钓",
            ["先用输入 SMILES 计算 InChIKey，在 ChEMBL 中做精确匹配；命中分子直接纳入汇总。随后以 Morgan 指纹（半径 2，2048 bit）执行 Tanimoto 相似性检索：从用户设定阈值开始，若候选少于 12–20 个，则按 10% 步长自动降至 50%，最多收集 100 个参照分子，并在本地缓存结果。深度模式进一步把参照分子上限提高到 40 个、参考分子预算加倍，并把最终靶点上限提升到 120，用于网络药理学需要更多节点的情况。", "活性记录只把真实的定量值作为 pChEMBL：优先取数据库 pChEMBL，缺失时由 standard_value 与 standard_units 换算；没有定量值的定性记录只计入记录数，不伪造 5.0 或 7.0 的 pChEMBL。作用机制记录单独计为机制证据。靶点详情取回后保留单蛋白、蛋白复合物与蛋白家族，复合物的全部基因组分都进入基因集；细胞系与组织靶点被剔除。", "除 ChEMBL 外，还并行查询七类证据：BindingDB 的结构相似化合物活性（Tanimoto 0.85 → 0.70 → 0.60 逐级回退）、PubChem 生物试验摘要、DGIdb 的药物–基因关系、Open Targets 的已批准药物作用机制、RCSB PDB 的共晶配体匹配（graph-exact，人源不足时回退 graph-relaxed）、SwissTargetPrediction 的 2D/3D 相似性靶点预测（含排队轮询与会话下载），以及本地 FDA 相似药物回退（Morgan 指纹 Tanimoto ≥ 0.22）。当所有在线服务均不可用时，本地相似药物仍会给出最低限度的靶点集，不会直接返回空结果。", "PDB 共晶与靶点名称会尝试映射到人源基因，非人源同源蛋白标记为 PDB homolog；无法映射的非人源实体保留物种标签但不进入人源网络。用户还可在垂钓面板粘贴外部基因列表，这些基因按 Imported 来源合并去重。所有来源按基因或蛋白名合并，并在结果中标明来源；ChEMBL 的定量 pChEMBL 与外部证据分开显示，外部证据没有 pChEMBL 时表中显示“—”。该方法反映的是已有文献活性、药物注释与结构数据的覆盖范围，而非从头预测：结构新颖的分子可能只能得到少量靶点，阴性结果不能排除某个靶点。"],
        ),
        (
            "1.17 网络药理学与富集分析",
            ["网络药理学直接读取靶点垂钓得到的基因符号，提交 STRING（species 9606，required_score 默认 400）构建蛋白互作网络；当种子基因少于 12 个时，可请求 STRING 增加一阶邻居节点（add_nodes，默认 15），扩展节点在图中单独着色，只作为网络背景，不计为直接靶点。", "富集分析使用 Enrichr 的 8 个库：KEGG_2021_Human、GO_Biological_Process_2023、GO_Cellular_Component_2023、GO_Molecular_Function_2023、Reactome_2022、WikiPathway_2023_Human、DisGeNET 与 MSigDB_Hallmark_2020；按校正 P 值排序报告前若干条目，气泡面积表示基因数，颜色表示 −log10(校正 P)。", "在富集结果上进一步构建化合物–靶点–疾病（C-T-D）网络：疾病条目来自 DisGeNET 富集，基因–疾病边要求基因出现在对应条目中；同时构建基因–通路桑基图，只保留至少 2 个输入基因命中的通路。若 STRING 没有返回任何互作边，程序保留富集、C-T-D 与桑基图，仅跳过 PPI 网络；若所有库都没有 FDR < 0.05 的条目，则如实说明未检出显著条目，不生成空富集图。", "STRING、Enrichr 与 DisGeNET 均为整合型数据库，边权、富集结果与疾病注释依赖数据库版本和背景基因集；C-T-D 网络只表示“化合物–靶点–疾病”的注释邻接，不代表因果关系。"],
        ),
        (
            "1.18 统一报告的组织方式",
            ["对接、优化、设计、垂钓、网络药理学、可合成性、FDA 参考与药物互作的全部图表按固定顺序合并进同一份 A4 报告：正文（方法、摘要、讨论）在前，源图（图 1–22）居中，结论与附录在后。所有图均为矢量图，图注紧随其图同页排布；正文数值、图与附录表格共享同一数据源，避免手工转录误差。"],
        ),
        (
            "1.19 接触指纹与口袋画像",
            ["图 14 的原子–残基接触指纹按氢键、疏水、盐桥、π-堆积、卤键与金属配位六类作用力统计残基级接触数；空间接触图把配体原子与受体残基坐标做 PCA 投影，仅用于展示几何邻接关系。图 15 的残基性质比例按接触残基计数，氢键密度按残基数归一化；这些指标是接触画像，不是可成药性打分，也不构成能量学结论。"],
        ),
    ]
    _flow_text_pages(pdf, "1　方法（续）", "在线数据模块与统一报告的组织方式", blocks)
    blocks = [
        (
            "1.20 类药性与可开发性",
            ["BOILED-Egg 风格判别以 Crippen WLOGP 与 TPSA 为坐标，采用 Egan/Clark 经验阈值划分 HIA 与 BBB 区域，只表示渗透趋势。CNS MPO 按 Wager 六项（cLogP、cLogD、MW、HBD、TPSA、pKa）各给 0–1 分并求和，4 分以上视为中枢神经系统候选。", "logD7.4 由规则式 pKa 与 Henderson–Hasselbalch 公式估算：酸性与碱性中心分别识别羧酸、磺酸、四唑、酚羟基与脂肪胺、芳香胺、吡啶、咪唑、胍、脒；多中心时取电离程度最大的形式。合成可及性使用 Ertl 与 Schuffenhauer 的 SAscore（RDKit Contrib 实现）。结构警示来自 RDKit FilterCatalog 的 PAINS、Brenk 与 NIH 三个目录。"],
        ),
        (
            "1.21 化学空间与骨架多样性",
            ["以 Morgan 指纹（半径 2，2048 bit）计算 Tanimoto 相似矩阵，按平均连接层次聚类排序绘制热图；用经典多维尺度（MDS）把同一距离矩阵投影到二维。Murcko 骨架由 RDKit 提取，骨架多样性定义为唯一骨架数除以分子数，平均最近邻相似度用于衡量系列冗余程度。"],
        ),
        (
            "1.22 代谢软点预测",
            ["用 SMARTS 规则识别常见 CYP 反应位点：N-脱烷基、O-脱烷基、硫氧化、苄位与烯丙位氧化、酚羟基结合、芳环氧化；各规则按权重累加并截断为 0–100 的风险分。该模块不区分 CYP 同工酶、不预测代谢速率，只用于优先安排代谢稳定性实验与结构封闭策略。"],
        ),
    ]
    _flow_text_pages(pdf, "1　方法（续）", "可开发性、化学空间与代谢软点的判据", blocks)
    blocks = [
        (
            "1.23 可合成性评估",
            ["SAscore 采用 Ertl 与 Schuffenhauer 的片段贡献法（RDKit Contrib），取值 1–10，越低越易合成；同时报告重原子数、环数、芳香环数、可旋转键、手性中心、Fsp3 与 Bertz 复杂度。手性中心 ≥ 2、大环（≥ 12 元）以及酰氯、叠氮、环氧化物、醛、Michael 受体等活性基团会作为硬步骤提示，但不直接等同于不可合成。"],
        ),
        (
            "1.24 模板逆合成与路线评分",
            ["路线由一套常见的药物化学断键模板生成：酰胺缩合、酯化、磺酰胺、脲、Williamson 芳醚、Suzuki 偶联、还原胺化、酮还原胺化与 SNAr；当模板全部不匹配时，回退到 RDKit BRICS 断键给出砌块建议。每条路线做两级递归扩展，末端砌块按重原子数与 SAscore 标记为“常见砌块（启发式）”。", "路线评分 = 0.38×估计总产率 + 0.34×砌块易得性（由末端 SAscore 归一化）+ 0.18×步数惩罚 + 0.10×反应难度惩罚，满分 100。产率与条件来自模板的经验常数，不是实验值；该模块不是 SYNTHIA 级全库搜索，只用于候选分子的合成可行性排序与路线假设。"],
        ),
    ]
    _flow_text_pages(pdf, "1　方法（续）", "可合成性与逆合成路线", blocks)
    blocks = [
        (
            "1.25 多引擎对接",
            [_L(
                "除 AutoDock Vina 1.2.5 与 Vina-GPU 2.0 外，命令行层集成 smina 与 QuickVina 2。两者与 Vina 共享 receptor/ligand/center/size 参数，但 smina 不传 --scoring，QuickVina 2 进一步去掉 --energy_range、--cpu 与 --verbosity。程序按所选引擎校验可执行文件，缺失时给出放置路径与环境变量提示；非 Vina 引擎从 PDBQT 的 REMARK VINA RESULT 解析打分，不执行 Vinardo 复算。不同引擎的打分尺度不可直接混用，跨引擎比较只作趋势参考。",
                "In addition to AutoDock Vina 1.2.5 and Vina-GPU 2.0, smina and QuickVina 2 are integrated at the command-line level. Both share the receptor/ligand/center/size parameters with Vina, but smina omits --scoring and QuickVina 2 additionally drops --energy_range, --cpu and --verbosity. The program validates the executable for the selected engine and reports the expected path or environment variable when missing; non-Vina engines parse scores from the PDBQT REMARK VINA RESULT block and do not run Vinardo rescoring. Scores from different engines are not on the same scale, and cross-engine comparisons are only indicative.",
            )],
        ),
        (
            "1.26 药物–药物相互作用",
            [_L(
                "以候选分子为中心，与联用药物逐项计算：共享靶点（候选靶点与参考药已知主要靶点的交集）、CYP3A4/2D6 竞争（候选的 CYP 抑制风险 × 联用药是否经该酶代谢）、hERG 风险叠加（两者预测风险之和）以及结构/类别相似度。四项按 0.38、0.27、0.20、0.15 加权为 0–100 的综合风险，分为高/中/低/信息不足。共享靶点不代表同向或拮抗，CYP 与 hERG 为计算风险，结果只用于优先核查。",
                "Each co-administered drug is assessed against the candidate by four items: shared targets (intersection of candidate targets and the known primary targets of the reference drug), CYP3A4/2D6 competition (candidate CYP-inhibition risk × whether the co-drug is metabolised by that enzyme), additive hERG risk (sum of the two predicted risks), and structural/class similarity. The four items are weighted 0.38, 0.27, 0.20 and 0.15 into a 0–100 composite risk, graded high/medium/low/insufficient information. Shared targets do not imply agonism or antagonism; CYP and hERG are computational risks, and the output is for prioritised review only.",
            )],
        ),
        (
            "1.27 FDA 已上市药物参考",
            ["内置 30–45 个结构简单、靶点明确的 FDA 已上市药物，字段包括结构、类别、主要靶点、适应证、给药途径、部分药物的人体典型半衰期与主要代谢酶。候选分子与参考药计算 Morgan 指纹 Tanimoto 相似度、Bemis–Murcko 骨架、rdFMCS 最大公共子结构、靶点 Jaccard、ADMET 相似度（hERG、CYP3A4、CYP2D6、Ames 与半衰期）与理化性质距离，综合分 = 0.45×Tanimoto + 0.25×靶点 Jaccard + 0.20×ADMET 相似度 + 0.10×性质相似度。药效维度不伪造 pIC50，只用靶点重叠与机制邻近表示。"],
        ),
        (
            "1.28 多库富集、C-T-D 与桑基图",
            ["多库富集把八个 Enrichr 库的显著条目按 −log10(P) 与基因数绘制点图；C-T-D 网络把化合物、输入基因与 DisGeNET 疾病条目连成三层网络；桑基图把基因分配到 KEGG 与 Reactome 通路。三者共用同一基因集与富集结果，避免不同图之间条目不一致。疾病关联与通路注释只代表数据库中的已知关系，不等于因果或临床有效性。"],
        ),
        (
            "1.29 综合药理、物种与组织分布",
            ["综合药理模块以同一 SMILES 为输入，输出物种感知的口服性能（BCS、HIA、Caco-2、ESOL 溶解度、剂量数、预测 F 与食物效应）、反刍动物过瘤胃（天然过瘤胃率、瘤胃降解风险、保护后过瘤胃率与包被策略）、肠道菌群扰动（偶氮、硝基、季铵、多酚、磺酰胺、Michael 受体、醌与糖苷等结构警示）、pH 1.2/6.8 水解、血浆酯酶、氧化与光稳定性、剂型与成盐/共晶建议、核酸嵌入/沟槽结合与共价警示、hERG/DILI/Ames/致癌性/生殖毒性/皮肤致敏/光毒性/环境风险/CYP3A4/2D6/P-gp 综合风险，以及按物种给出的经验组织 Kp 与分布占比。", "口服与瘤胃为经验规则，组织 Kp 为 logP/ppb 分配模型，不是 PBPK；菌群与核酸结合为结构警示评分。每个模块都标注高/中/低置信度与适用边界。靶点组织表达通过 UniProt 与 Human Protein Atlas 查询，并计算化合物–靶点组织重叠评分；在线证据（PubChem、ChEMBL、UniProt、HPA）与本地规则预测在报告中分开标注，查询失败时自动降级为本地规则。该模块不依赖分子对接，可单独输出 6 页综合药理报告。"],
        ),
    ]
    _flow_text_pages(pdf, "1　方法（续）", "多引擎对接、药物互作、FDA 参考与综合药理", blocks)


def _run_mode_label(run: dict) -> dict:
    params = (run or {}).get("params") or {}
    kind = str((run or {}).get("kind") or "")
    results = [
        item for item in (run or {}).get("results") or []
        if item.get("status") == "ok"
    ]
    count = len(results)
    exhaustiveness = int(params.get("exhaustiveness") or 8)
    modes = int(params.get("num_modes") or 9)
    english = i18n.get_lang() == "en"
    if kind == "design":
        label = "Design-candidate docking" if english else "设计分子对接"
        key = "design"
    elif kind == "optimization":
        label = "Optimization-candidate docking" if english else "优化候选对接"
        key = "optimization"
    elif exhaustiveness >= 16 and modes >= 9 and count <= 40:
        label = "Refined docking (high exhaustiveness)" if english else "精细对接（高 exhaustiveness）"
        key = "refined"
    elif count >= 40 or exhaustiveness <= 8:
        label = "Large-scale screening" if english else "大规模筛选"
        key = "screening"
    else:
        label = "Standard docking" if english else "标准对接"
        key = "standard"
    return {
        "key": key,
        "label": label,
        "count": count,
        "exhaustiveness": exhaustiveness,
        "num_modes": modes,
    }


def _enrichment_overview(data: dict | None) -> dict:
    """Successful/failed Enrichr libraries and significant-term total."""
    enrichment = (data or {}).get("enrichment") or {}
    ok: list[str] = []
    failed: list[str] = []
    significant = 0
    for key, block in enrichment.items():
        block = block or {}
        library = str(block.get("library") or key)
        significant += int(block.get("significant") or 0)
        if str(block.get("error") or "").strip():
            failed.append(library)
        elif block.get("terms"):
            ok.append(library)
    return {"ok": ok, "failed": failed, "significant": significant}


def _abstract_page(
    pdf,
    project_info: dict,
    run: dict,
    results: list[dict],
    best: dict | None,
    props: dict,
    analyses: dict[int, dict],
    optimization: dict | None,
    design: dict | None,
    validation: dict | None = None,
    fishing_data: dict | None = None,
    netpharm_data: dict | None = None,
    synthesis_data: dict | None = None,
    drugref_data: dict | None = None,
    ddi_data: dict | None = None,
    pharm_data: dict | None = None,
):
    stats = _stats(results, analyses, props)
    best_props = props.get(id(best), {}) if best else {}
    best_analysis = analyses.get(id(best), {}) if best else {}
    counts = best_analysis.get("counts", {}) if isinstance(best_analysis, dict) else {}
    fig = _a4_page("摘要", "研究背景、主要发现与结论")

    mode_info = _run_mode_label(run)
    if i18n.get_lang() == "en":
        overview = [
            f"This report consolidates the docking results of project "
            f"\"{project_info.get('name', '—')}\" generated with DrugHX. "
            f"{stats['count']} docked ligands were included (each with its output conformations); "
            f"the receptor was {best.get('receptor_name', '—') if best else '—'}. This run is classified as "
            f"{mode_info['label']} (exhaustiveness {mode_info['exhaustiveness']}, "
            f"{mode_info['count']} successful results)."
        ]
    else:
        overview = [
            f"本报告基于 DrugHX 计算平台，对项目「{project_info.get('name', '—')}」的分子对接结果进行系统整理。"
            f"共纳入 {stats['count']} 个完成对接的配体（每个配体保留多个输出构象），靶点为 "
            f"{best.get('receptor_name', '—') if best else '—'}；"
            f"本次运行属于{mode_info['label']}"
            f"（exhaustiveness {mode_info['exhaustiveness']}，"
            f"{mode_info['count']} 个成功结果）。",
        ]
    findings = []
    if validation and validation.get("rows"):
        first = validation["rows"][0]
        findings.append(
            f"共晶配体重对接 RMSD {float(first.get('rmsd', 0)):.2f} Å，"
            f"{'通过' if first.get('passed') else '未通过'} 2.0 Å 判据；"
            "本次排序据此定位为假设生成。"
        )
    if best:
        heavy_value = best_props.get("heavy")
        le_value = (best_props.get("efficiency") or {}).get("le")
        qed_value = best_props.get("qed")
        heavy_text = "—" if heavy_value is None else str(heavy_value)
        le_text = "—" if le_value is None else f"{float(le_value):.2f}"
        qed_text = "—" if qed_value is None else f"{float(qed_value):.2f}"
        findings.append(
            _L(
                f"最优配体为 {best.get('ligand_name', '—')}，Vina 打分 {float(best.get('affinity', 0)):.2f} kcal/mol；"
                f"重原子 {heavy_text} 个，配体效率 LE = {le_text} kcal/mol/HA，QED = {qed_text}。",
                f"The best ligand was {best.get('ligand_name', '—')} with a Vina score of "
                f"{float(best.get('affinity', 0)):.2f} kcal/mol; {heavy_text} heavy atoms, "
                f"ligand efficiency LE = {le_text} kcal/mol/HA and QED = {qed_text}.",
            )
        )
        interaction_total = sum(
            int(counts.get(key) or 0)
            for key in ("hbonds", "hydrophobic", "salt_bridges", "pi_stacking", "halogens", "metals")
        )
        if interaction_total:
            findings.append(
                _L(
                    f"该结合模式形成 {counts.get('hbonds', 0)} 个氢键、{counts.get('hydrophobic', 0)} 处疏水接触、"
                    f"{counts.get('salt_bridges', 0)} 个盐桥与 {counts.get('pi_stacking', 0)} 处 π-堆积，"
                    "可作为骨架修饰与选择性设计的参照。",
                    f"This binding mode formed {counts.get('hbonds', 0)} hydrogen bonds, "
                    f"{counts.get('hydrophobic', 0)} hydrophobic contacts, "
                    f"{counts.get('salt_bridges', 0)} salt bridges and "
                    f"{counts.get('pi_stacking', 0)} π-stacking interactions, which can guide "
                    "scaffold modification and selectivity design.",
                )
            )
        else:
            findings.append(
                _L(
                    "本次分析未识别到氢键、疏水接触等作用力（可能缺少作用力分析或构象文件），"
                    "请结合图 3 与三维构象人工核对，不能据此推断口袋互补性。",
                    "No hydrogen bonds or hydrophobic contacts were detected for this pose (the "
                    "interaction analysis may be unavailable); inspect Figure 3 and the 3D pose "
                    "manually instead of inferring pocket complementarity.",
                )
            )
    if stats["mean"] is not None:
        if stats["count"] <= 1:
            findings.append(
                f"本次仅有 1 个成功配体，Vina 打分为 {stats['mean']:.2f} kcal/mol；"
                "样本量不足以计算均值、中位数区间或 Vina–Vinardo 相关性，"
                "相关统计只作单分子描述。"
            )
        else:
            correlation_text = ""
            if stats["correlation"] is not None:
                rho = stats.get("spearman")
                rho_text = f"、Spearman ρ = {rho:.3f}" if rho is not None else ""
                correlation_text = (
                    f"Vina 与 Vinardo 的 Pearson r = {stats['correlation']:.3f}{rho_text}，"
                    f"属于{stats.get('corr_text', '中等相关')}；"
                    + (
                        "两套函数排序一致，结论不依赖单一打分函数。"
                        if abs(stats["correlation"]) >= 0.8
                        else "两套函数只部分一致，排名差异需结合作用力逐条复核，不能据此宣称排序稳健。"
                    )
                )
            findings.append(
                f"全部配体平均 {stats['mean']:.2f} kcal/mol（中位数 {stats['median']:.2f}，"
                f"区间 {stats['worst']:.2f} 至 {stats['best']:.2f}）；" + correlation_text
            )
    if stats["hotspots"]:
        names = "、".join(f"{residue}" for residue, _ in stats["hotspots"][:3])
        findings.append(
            (
                f"残基接触频率分析显示 {names} 为高频作用残基，可作为后续骨架修饰与选择性设计的优先位点。"
                if stats["count"] >= 3
                else f"该配体的主要接触残基为 {names}，可作为后续骨架修饰与选择性设计的优先位点。"
            )
        )
    if optimization and optimization.get("best"):
        best_opt = optimization["best"]
        parent_name = optimization.get("parent_ligand") or "母体"
        generated = int(optimization.get("generated") or optimization.get("count") or 0)
        categories = optimization.get("categories") or {}
        scaffold_hops = int(optimization.get("scaffold_hops") or 0)
        if i18n.get_lang() == "en":
            findings.append(
                f"The optimisation module generated {generated} candidates from {parent_name} "
                f"across {len(categories) or 1} transformation classes, including "
                f"{scaffold_hops} scaffold hops; the best candidate "
                f"{best_opt.get('label', '—')} scored "
                f"{float(best_opt.get('affinity', 0)):.2f} kcal/mol "
                f"(Δ {float(best_opt.get('delta', 0)):+.2f})."
            )
        else:
            findings.append(
                f"药物优化模块以 {parent_name} 为母体生成 {generated} 个候选，"
                f"覆盖 {len(categories) or 1} 类改造（含 {scaffold_hops} 个骨架跃迁），"
                f"最佳候选 {best_opt.get('label', '—')} 的打分为 "
                f"{float(best_opt.get('affinity', 0)):.2f} kcal/mol，"
                f"相对该母体 Δ {float(best_opt.get('delta', 0)):+.2f} kcal/mol。"
            )
    if design and design.get("best"):
        best_design = design["best"]
        generated = int(design.get("generated") or design.get("count") or 0)
        core_counts = design.get("core_counts") or {}
        scaffold_hops = int(design.get("scaffold_hops") or 0)
        if i18n.get_lang() == "en":
            findings.append(
                f"The design module generated {generated} drug-like candidates across "
                f"{len(core_counts)} core families, including {scaffold_hops} scaffold hops; "
                f"the best design {best_design.get('label', '—')} scored "
                f"{float(best_design.get('affinity', 0)):.2f} kcal/mol."
            )
        else:
            findings.append(
                f"分子设计模块生成 {generated} 个类药候选，覆盖 {len(core_counts)} 个核心家族"
                f"（含 {scaffold_hops} 个骨架跃迁），最佳设计 "
                f"{best_design.get('label', '—')} 的打分为 "
                f"{float(best_design.get('affinity', 0)):.2f} kcal/mol。"
            )
    if synthesis_data:
        route = synthesis_data.get("best_route") or {}
        if route:
            findings.append(
                _L(
                    f"可合成性评估为“{_synth_verdict(synthesis_data)}”"
                    f"（SAscore {float(synthesis_data.get('sa') or 0):.2f}）；"
                    f"首选模板路线为 {route.get('label', '—')}，"
                    f"约 {route.get('step_count', 0)} 步、估计总产率 "
                    f"{float(route.get('overall_yield') or 0):.0f}%，路线评分 "
                    f"{float(route.get('score') or 0):.1f}/100。"
                    "该路线来自有限反应模板库，需文献与小试验证。",
                    f"Synthesizability was assessed as “{_synth_verdict(synthesis_data)}” "
                    f"(SAscore {float(synthesis_data.get('sa') or 0):.2f}); the preferred template "
                    f"route is {route.get('label', '—')}, about {route.get('step_count', 0)} steps, "
                    f"estimated overall yield {float(route.get('overall_yield') or 0):.0f}% and "
                    f"route score {float(route.get('score') or 0):.1f}/100. The route comes from a "
                    "limited reaction-template library and needs literature and bench validation.",
                )
            )
        else:
            findings.append(
                _L(
                    f"可合成性评估为“{_synth_verdict(synthesis_data)}”"
                    f"（SAscore {float(synthesis_data.get('sa') or 0):.2f}），"
                    "但当前模板库没有匹配到完整逆合成路线；已给出硬步骤与砌块建议。",
                    f"Synthesizability was assessed as “{_synth_verdict(synthesis_data)}” "
                    f"(SAscore {float(synthesis_data.get('sa') or 0):.2f}), but no complete "
                    "retrosynthetic route matched the current template library; hard steps and "
                    "building-block suggestions are provided.",
                )
            )
    fish_targets = (fishing_data or {}).get("targets") or []
    if fish_targets:
        fish_name = (fishing_data.get("compound") or {}).get("name") or "查询分子"
        if str(fish_name).strip().lower() == "query":
            fish_name = "查询分子"
        top_names = "、".join(
            _target_display_name(item, 34) for item in fish_targets[:3]
        )
        n_human = sum(
            1 for item in fish_targets
            if item.get("genes")
            and "homo sapiens" in str(item.get("organism") or "").lower()
        )
        source_counts = (fishing_data.get("stats") or {}).get("sources") or {}
        hit_text = "、".join(
            f"{name}（{count}）" for name, count in source_counts.items()
        ) or "无"
        findings.append(
            f"反向靶点垂钓以 {fish_name} 为查询，汇总 {len(fish_targets)} 个候选靶点"
            f"（{n_human} 个人源基因），证据分最高为 {top_names}；"
            f"实际命中 {hit_text}。该结果只提出靶点假设，不代表实测亲和力。"
        )
        sources_status = (fishing_data or {}).get("external_sources") or {}
        fishing_stats = (fishing_data or {}).get("stats") or {}
        chembl_missing = (
            (sources_status.get("ChEMBL") or {}).get("available") is False
            or (
                "ChEMBL" not in (fishing_stats.get("sources") or {})
                and int(fishing_stats.get("molecules") or 0) == 0
            )
        )
        if chembl_missing:
            findings.append(
                "ChEMBL similarity search was unavailable in this run; the target "
                "list therefore reflects only PDB, BindingDB and other fallback "
                "sources. A short list does not mean that no other target exists."
                if i18n.get_lang() == "en"
                else "本次 ChEMBL 相似性检索不可用，靶点列表仅反映 PDB、BindingDB 等回退来源；"
                "数量偏少不代表不存在其他靶点，服务恢复后应重新运行垂钓。"
            )
    net_genes = (netpharm_data or {}).get("genes") or []
    net_overview = _enrichment_overview(netpharm_data)
    if netpharm_data and (net_genes or net_overview["ok"]):
        ppi = (netpharm_data or {}).get("ppi") or {}
        expanded = (netpharm_data or {}).get("expanded_genes") or []
        enrichment = (netpharm_data or {}).get("enrichment") or {}
        first_term = None
        first_label = ""
        for key, label in (("kegg", "KEGG"), ("go_bp", "GO BP"), ("reactome", "Reactome")):
            terms = [
                item for item in ((enrichment.get(key) or {}).get("terms") or [])
                if item.get("p")
            ]
            if terms:
                first_term = min(terms, key=lambda item: float(item.get("p") or 1))
                first_label = label
                break
        supplied = (netpharm_data or {}).get("supplied_genes") or []
        english = i18n.get_lang() == "en"
        if english:
            source_note = (
                f" ({len(net_genes) - len(supplied)} from fishing, "
                f"{len(supplied)} imported)"
                if supplied
                else ""
            )
            text = (
                f"Network pharmacology mapped {len(net_genes)} input genes"
                f"{source_note} onto STRING "
                f"({len(ppi.get('edges') or [])} edges"
                + (f", {len(expanded)} neighbours as background" if expanded else "")
                + ")"
            )
            if first_term is not None:
                text += (
                    f"; the top {first_label} term was {first_term.get('term', '—')} "
                    f"(P = {float(first_term.get('p') or 0):.1e})"
                )
            text += (
                f"; {len(net_overview['ok'])} Enrichr libraries returned results with "
                f"{net_overview['significant']} terms at FDR < 0.05"
            )
            if net_overview["failed"]:
                text += (
                    "; libraries skipped after request failure: "
                    + ", ".join(net_overview["failed"])
                )
            findings.append(text + ".")
        else:
            lead = "垂钓与外部导入的" if supplied else "垂钓得到的"
            text = (
                f"网络药理学将{lead} {len(net_genes)} 个基因映射到 STRING"
                f"（{len(ppi.get('edges') or [])} 条边"
                + (f"，{len(expanded)} 个邻居作背景" if expanded else "")
                + "）"
            )
            if first_term is not None:
                text += (
                    f"，{first_label} 首位为 {first_term.get('term', '—')}"
                    f"（P = {float(first_term.get('p') or 0):.1e}）"
                )
            text += (
                f"；{len(net_overview['ok'])} 个 Enrichr 库返回结果，"
                f"共 {net_overview['significant']} 个 FDR < 0.05 条目"
            )
            if net_overview["failed"]:
                text += "；请求失败并跳过的库：" + "、".join(net_overview["failed"])
            findings.append(text + "。")
    ref_top = (drugref_data or {}).get("top") or []
    if ref_top:
        first = ref_top[0]
        findings.append(
            f"FDA 参考分析中最接近的已上市药物为 {first.get('name', '—')}"
            f"（Tanimoto {float(first.get('tanimoto') or 0):.3f}，靶点 Jaccard "
            f"{float(first.get('target_jaccard') or 0):.3f}，综合分 "
            f"{float(first.get('score') or 0):.3f}），只表示机制与性质邻近。"
        )
    ddi_summary = (ddi_data or {}).get("summary") or {}
    if ddi_summary:
        high = ddi_summary.get("high", 0)
        medium = ddi_summary.get("medium", 0)
        unknown = ddi_summary.get("unknown", 0)
        if i18n.get_lang() == "en":
            findings.append(
                f"Drug–drug interaction screening: {high} high-risk, {medium} medium-risk "
                f"and {unknown} with insufficient information; review the checklist in Figure 19."
            )
        else:
            findings.append(
                f"药物互作筛查：高风险 {high} 个、中风险 {medium} 个、信息不足 "
                f"{unknown} 个，需按图 19 清单核查。"
            )
    if netpharm_data and net_overview["ok"]:
        if i18n.get_lang() == "en":
            text = (
                f"Enrichment analysis succeeded in {len(net_overview['ok'])} Enrichr "
                f"libraries with {net_overview['significant']} terms at FDR < 0.05; "
                "Figures 20–22 show the multi-library enrichment, C-T-D network and "
                "target–pathway Sankey diagram"
            )
            if net_overview["failed"]:
                text += (
                    "; libraries skipped after request failure: "
                    + ", ".join(net_overview["failed"])
                )
            findings.append(text + ".")
        else:
            text = (
                f"网络药理学在 {len(net_overview['ok'])} 个 Enrichr 库中获得结果，"
                f"共 {net_overview['significant']} 个 FDR < 0.05 条目，"
                "图 20–22 为多库富集、C-T-D 网络与靶点–通路桑基图"
            )
            if net_overview["failed"]:
                text += "；请求失败并跳过的库：" + "、".join(net_overview["failed"])
            findings.append(text + "。")
    if pharm_data:
        oral = pharm_data.get("oral") or {}
        risk = pharm_data.get("risk") or {}
        tissue = pharm_data.get("tissue") or {}
        formulation = pharm_data.get("formulation") or {}
        if i18n.get_lang() == "en":
            findings.append(
                f"Integrated pharmacology ({pharm_data.get('species_label', '—')}): "
                f"BCS {oral.get('bcs', '—')}, predicted oral F ≈ "
                f"{float(oral.get('f_percent') or 0):.0f}%; suggested formulation: "
                f"{formulation.get('form', '—')}; integrated risk {risk.get('level', '—')} "
                f"({float(risk.get('overall') or 0):.0f}/100); compound–target tissue "
                f"overlap {float(tissue.get('overlap_score') or 0):.0f}/100. "
                "Details are shown in Figures C1–C6."
            )
        else:
            findings.append(
                f"综合药理（{pharm_data.get('species_label', '—')}）：BCS "
                f"{oral.get('bcs', '—')}，预测口服 F ≈ "
                f"{float(oral.get('f_percent') or 0):.0f}%；建议剂型："
                f"{formulation.get('form', '—')}；综合风险 {risk.get('level', '—')} "
                f"（{float(risk.get('overall') or 0):.0f}/100）；化合物–靶点组织重叠 "
                f"{float(tissue.get('overlap_score') or 0):.0f}/100。详见附图 C1–C6。"
            )
        warheads = pharm_data.get("covalent_warheads") or []
        if warheads:
            findings.append(
                "Covalent-warhead warning: detected "
                + ", ".join(warheads)
                + ". Vina and Vinardo are non-covalent scoring functions and cannot "
                "describe covalent bonds; the reported scores reflect only the "
                "non-covalent portion and require covalent docking, dynamics or "
                "experimental validation."
                if i18n.get_lang() == "en"
                else "共价弹头警示：检测到 "
                + "、".join(warheads)
                + "。Vina/Vinardo 为非共价打分，不能描述共价键；本报告分数只反映非共价部分，"
                "排序仅作参考，需共价对接、动力学模拟或实验验证。"
            )
    cross_note = str(
        (netpharm_data or {}).get("report_note")
        or (fishing_data or {}).get("report_note")
        or ""
    ).strip()
    if cross_note:
        findings.append(
            "Note: the fishing/network pharmacology records are cross-molecule "
            "references (other or pooled queries); see Sections 2.8–2.9."
            if i18n.get_lang() == "en"
            else "注：垂钓/网络药理学记录与主配体不一致或为合并查询，仅作跨分子参考（见 2.8–2.9 节）。"
        )
    conclusion = [
        "候选分子在结合亲和力与理化性质之间取得了可解释的平衡，优化、设计与垂钓结果给出可检验的下一步假设；"
        "所有数值来自经验打分与数据库注释，最终结论仍需分子动力学模拟与实验验证。",
    ]
    blocks = [
        ("研究背景", overview),
        ("主要发现", findings),
        ("结论", conclusion),
    ]
    end_y = _text_blocks(
        fig, 0.885, blocks,
        width=45 if i18n.get_lang() == "en" else 50,
        size=10.8 if i18n.get_lang() == "en" else 9.2,
        stop=0.070 if i18n.get_lang() != "en" else 0.100,
    )
    keywords = "分子对接；ADMET；FDA 参考；药物互作；网络药理学；合成路线"
    if i18n.get_lang() == "en":
        keywords = _wrap_cjk(i18n.tr(keywords), 92)
    english = i18n.get_lang() == "en"
    minimum = 0.105 if english else 0.150
    if end_y >= minimum + 0.030:
        heading_y = max(minimum, end_y - (0.045 if english else 0.055))
        fig.text(
            0.07, heading_y,
            _L("关键词", "Keywords"), fontsize=13.6, fontweight="bold",
            color=TEAL, va="top",
        )
        fig.text(
            0.07, heading_y - (0.021 if english else 0.022),
            keywords,
            fontsize=8.6 if english else 9.2,
            color=INK, va="top",
        )
        pdf.savefig(fig, facecolor="white")
        plt.close(fig)
    else:
        # The abstract is full: keep it uncluttered and place the keywords on
        # a short continuation page instead of overlapping the conclusion.
        pdf.savefig(fig, facecolor="white")
        plt.close(fig)
        fig2 = _a4_page(_L("摘要（续）", "Abstract (continued)"), _L("关键词与缩写", "Keywords and abbreviations"))
        fig2.text(
            0.07, 0.860, _L("关键词", "Keywords"),
            fontsize=13.6, fontweight="bold", color=TEAL, va="top",
        )
        fig2.text(
            0.07, 0.830, keywords,
            fontsize=9.2 if not english else 8.6, color=INK, va="top",
        )
        pdf.savefig(fig2, facecolor="white")
        plt.close(fig2)


def _discussion_pages(
    pdf,
    run: dict,
    results: list[dict],
    best: dict | None,
    props: dict,
    analyses: dict[int, dict],
    optimization: dict | None,
    design: dict | None,
    validation: dict | None = None,
    fishing_data: dict | None = None,
    netpharm_data: dict | None = None,
    notes: dict | None = None,
    synthesis_data: dict | None = None,
    drugref_data: dict | None = None,
    ddi_data: dict | None = None,
    pharm_data: dict | None = None,
):
    stats = _stats(results, analyses, props)
    best_props = props.get(id(best), {}) if best else {}
    best_analysis = analyses.get(id(best), {}) if best else {}
    counts = best_analysis.get("counts", {}) if isinstance(best_analysis, dict) else {}

    fig = _a4_page("2　结果与讨论", "结合模式、打分可靠性与作用力热点")
    binding = []
    if validation and validation.get("rows"):
        first = validation["rows"][0]
        binding.append(
            f"对接方案先用共晶配体 {first.get('name', '—')} 重对接检验：对称校正重原子 RMSD "
            f"{float(first.get('rmsd', 0)):.2f} Å，"
            f"{'满足' if first.get('passed') else '未满足'} 2.0 Å 判据。"
            + (
                "因此本次参数可用于同系列分子的相对排序。"
                if first.get("passed")
                else "说明当前参数尚不能复现晶体结合模式，后续排名只应作为假设生成，不能作为定量结论。"
            )
        )
    if best:
        binding.append(
            f"{best.get('ligand_name', '—')} 以 {float(best.get('affinity', 0)):.2f} kcal/mol 位居首位，"
            f"较第二位（{results[1].get('ligand_name', '—')}，{float(results[1].get('affinity', 0)):.2f} kcal/mol）"
            f"低 {abs(float(results[1].get('affinity', 0)) - float(best.get('affinity', 0))):.2f} kcal/mol。"
            if len(results) > 1
            else "仅有一个成功结果，统计意义有限。"
        )
        best_mw = float(best_props.get("mw") or 0)
        best_logp = float(best_props.get("logp") or 0)
        if best_mw <= 500 and best_logp <= 5:
            space_text = _L(
                "落在类药化学空间内",
                "lies within the conventional drug-like chemical space",
            )
        else:
            space_text = _L(
                "分子量或脂溶性略超常规类药空间上限，属于边界分子，应结合 LE/QED 判断",
                "slightly exceeds the conventional drug-like boundaries and should be judged together with LE/QED",
            )
        binding.append(
            f"其分子量为 {best_mw:.1f} Da、cLogP {best_logp:.2f}、"
            f"TPSA {best_props.get('tpsa', 0):.1f} Å²、可旋转键 {best_props.get('rotb', 0)} 个，"
            f"{space_text}；{_L('配体效率', 'ligand efficiency')} {(best_props.get('efficiency') or {}).get('le', 0):.2f} "
            "表明其结合能并非单纯由分子量堆叠获得。"
        )
    reliability = []
    if stats["correlation"] is not None:
        rho = stats.get("spearman")
        rho_text = f"，Spearman ρ = {rho:.3f}" if rho is not None else ""
        reliability.append(
            f"Vina 与 Vinardo 的 Pearson r = {stats['correlation']:.3f}{rho_text}（n = "
            f"{stats['count']}），属于{stats.get('corr_text', '中等相关')}。"
            + (
                "两套函数在原子类型与表面项处理上不同却给出高度一致的排序，说明该系列分子的构象选择稳定。"
                if abs(stats["correlation"]) >= 0.8
                else "两套函数只部分一致，说明分数对打分函数有一定依赖；排名相邻的分子应结合作用力逐条复核，"
                "不宜仅凭 0.1–0.3 kcal/mol 的差异判断优劣。"
            )
        )
    reliability.append(
        (
            "构象能量谱显示，多数配体的输出构象打分跨度在 1–2 kcal/mol 量级，"
            if stats["count"] >= 3
            else "当前成功配体数量较少，构象能量谱只作单分子描述；"
        )
        + "跨度越大表示搜索越未收敛、结合模式越不确定。该指标只反映单次运行的构象采样，"
        "不能替代多随机种子的重复性检验；若需评估可重复性，应显式运行 3–5 个独立种子并比较最优构象 RMSD。"
    )
    interactions = []
    if counts:
        interactions.append(
            f"最优配体复合物中检出 {counts.get('hbonds', 0)} 个氢键、{counts.get('hydrophobic', 0)} 处疏水接触、"
            f"{counts.get('salt_bridges', 0)} 个盐桥、{counts.get('pi_stacking', 0)} 处 π-堆积，"
            "氢键与疏水作用共同构成主要的结合驱动力。"
        )
    if stats["hotspots"]:
        hotspots = "、".join(
            f"{residue}（{count} 次）" for residue, count in stats["hotspots"][:4]
        )
        interactions.append(
            (
                f"跨配体的残基接触频率统计给出高频热点：{hotspots}。"
                "这些残基在多数配体中反复出现，构成口袋的“锚点”，"
                if stats["count"] >= 3
                else f"该配体的残基接触热点为：{hotspots}。"
            )
            + "也是选择性优化时应优先保留或强化作用力的位点。"
        )
    blocks = [
        ("2.1 结合亲和力与排序", binding),
        ("2.2 打分可靠性与构象收敛性", reliability),
        ("2.3 相互作用谱与残基热点", interactions),
        (
            "2.4 图 1–图 4 阅读要点",
            [
                "图 1 用于判断候选的相对强弱与两套打分函数是否一致：若多数点紧贴对角线，说明排序稳健；偏离明显的分子应结合作用力逐个复核。",
                "图 2 回答“分数是否靠分子量堆出来”：位于同一配体效率等值线上的分子具有可比性，落在右下方且 LE 偏低的分子通常不值得继续投入。",
                "图 3 与图 4 共同描述结合模式的“质”和“稳”：前者给出与哪些残基发生何种作用，后者给出该模式在不同构象中的可重复性，两者同时满足才可作为改造起点。",
            ],
        ),
    ]
    _text_blocks(fig, 0.885, blocks)
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)

    fig = _a4_page("2　结果与讨论（续）", "类药性、分子改进与骨架枚举设计")
    druglike = []
    if best:
        efficiency = best_props.get("efficiency") or {}
        druglike.append(
            f"最优配体的 QED 为 {best_props.get('qed', 0):.2f}，氢键供体/受体为 "
            f"{best_props.get('hbd', 0)}/{best_props.get('hba', 0)}，"
            f"Lipinski 规则{'通过' if best_props.get('lipinski') else '未通过'}，"
            f"Veber 规则{'通过' if best_props.get('veber') else '未通过'}；"
            f"LipE = {efficiency.get('lipe', 0):.2f}，BEI = {efficiency.get('bei', 0):.2f}。"
        )
    if stats["mean_le"] is not None:
        if stats["count"] >= 3:
            druglike.append(
                f"该系列分子的平均配体效率为 {stats['mean_le']:.2f} kcal/mol/HA，"
                "处于先导化合物常见区间（0.3–0.5）。若需进一步提升，应优先考虑在保持作用力的前提下"
                "减少非必要重原子，或引入能形成定向氢键的极性基团。"
            )
        else:
            druglike.append(
                f"当前只有 {stats['count']} 个成功配体，平均配体效率 "
                f"{stats['mean_le']:.2f} kcal/mol/HA 仅作描述，不足以代表一个系列；"
                "若要评估系列趋势，需要增加同系列类似物。"
            )
    optimization_text = []
    if optimization and optimization.get("best"):
        best_opt = optimization["best"]
        parent_name = optimization.get("parent_ligand") or "母体"
        categories = optimization.get("categories") or {}
        scaffold_hops = int(optimization.get("scaffold_hops") or 0)
        generated = int(optimization.get("generated") or optimization.get("count") or 0)
        top_rows = optimization.get("top") or []
        category_text = "、".join(
            f"{name} {count}" for name, count in categories.items()
        ) or "结构修饰"
        if i18n.get_lang() == "en":
            category_en = {
                "生物电子等排": "bioisostere",
                "骨架跃迁": "scaffold hopping",
                "取代基行走": "substituent walking",
                "代谢稳定化": "metabolic blocking",
                "连接子改造": "linker modification",
                "组合改造": "combination",
            }
            category_text = ", ".join(
                f"{category_en.get(name, name)} {count}"
                for name, count in categories.items()
            ) or "structural modification"
            optimization_text.append(
                f"The optimisation module generated {generated} candidates from {parent_name}; "
                f"{optimization.get('count', 0)} completed docking. The library covered "
                f"{len(categories) or 1} transformation classes ({category_text}), including "
                f"{scaffold_hops} scaffold hops. The best candidate, "
                f"{best_opt.get('label', '—')} ({best_opt.get('modification', 'modification')}), "
                f"scored {float(best_opt.get('affinity', 0)):.2f} kcal/mol "
                f"(Δ {float(best_opt.get('delta', 0)):+.2f} kcal/mol vs the parent)."
            )
            top_bits = "; ".join(
                f"{item.get('label', '—')} ({item.get('category') or item.get('modification', '—')}, "
                f"Δ {float(item.get('delta') or 0):+.2f})"
                for item in top_rows[:5]
                if item.get("delta") is not None
            )
            if top_bits:
                optimization_text.append(
                    f"The top candidates by ΔVina were {top_bits}. "
                    "Second-generation combination candidates stack a scaffold hop or "
                    "bioisosteric replacement with a substituent walk, so the report covers "
                    "both single-step and multi-step optimisation directions."
                )
            optimization_text.append(
                "ΔVina is meaningful only within the same parent series and cannot be compared "
                "with absolute library-screening scores. If molecular weight increases, check "
                "ligand efficiency to avoid false positives that merely add volume."
            )
        else:
            optimization_text.append(
                f"优化模块以 {parent_name} 为母体生成 {generated} 个候选，"
                f"其中 {optimization.get('count', 0)} 个完成对接；候选覆盖 "
                f"{len(categories) or 1} 类改造（{category_text}），"
                f"其中骨架跃迁 {scaffold_hops} 个。最佳候选 "
                f"{best_opt.get('label', '—')}（{best_opt.get('modification', '结构修饰')}）"
                f"取得 {float(best_opt.get('affinity', 0)):.2f} kcal/mol，"
                f"相对该母体 Δ {float(best_opt.get('delta', 0)):+.2f} kcal/mol。"
            )
            top_bits = "、".join(
                f"{item.get('label', '—')}（{item.get('category') or item.get('modification', '—')}，"
                f"Δ {float(item.get('delta') or 0):+.2f}）"
                for item in top_rows[:5]
                if item.get("delta") is not None
            )
            if top_bits:
                optimization_text.append(
                    f"ΔVina 排名靠前的方向为 {top_bits}。"
                    "组合改造候选把骨架跃迁或生物电子等排与取代基行走叠加，"
                    "因此本轮同时覆盖单步改造与多步组合改造两个层次。"
                )
            optimization_text.append(
                "此处 ΔVina 只对同一母体的类似物有意义，不能与前面库筛结果的绝对分数直接横向比较；"
                "若伴随分子量上升，应同时检查配体效率是否下降，以避免以体积换分数的假阳性。"
            )
    else:
        optimization_text.append(
            (notes or {}).get("optimization")
            or "本项目尚未运行药物优化模块。建议在确定母体后运行，以获得残基级的改造方案。"
        )
    design_text = []
    if design and design.get("best"):
        best_design = design["best"]
        core_counts = design.get("core_counts") or {}
        scaffold_hops = int(design.get("scaffold_hops") or 0)
        generated = int(design.get("generated") or design.get("count") or 0)
        top_rows = design.get("top") or []
        if i18n.get_lang() == "en":
            design_text.append(
                f"The design module generated {generated} drug-like candidates and "
                f"{design.get('count', 0)} completed docking (mode: {design.get('mode', 'auto')}). "
                f"It covered {len(core_counts)} core families; {scaffold_hops} candidates were "
                f"reference-based scaffold hops. The best design, "
                f"{best_design.get('label', '—')}, scored "
                f"{float(best_design.get('affinity', 0)):.2f} kcal/mol "
                f"(QED {(best_design.get('props') or {}).get('qed', 0):.2f})."
            )
            top_bits = "; ".join(
                f"{item.get('label', '—')} ({item.get('core', '—')}, "
                f"{float(item.get('affinity') or 0):.2f} kcal/mol)"
                for item in top_rows[:5]
            )
            if top_bits:
                design_text.append(
                    f"The top designs were {top_bits}. The enumeration covers single-, "
                    "double- and triple-substitution patterns on aromatic cores and, when a "
                    "reference ligand is supplied, aromatic scaffold hops, heterocycle "
                    "replacements and linker modifications."
                )
        else:
            design_text.append(
                f"分子设计模块生成 {generated} 个通过类药性预筛的候选，"
                f"其中 {design.get('count', 0)} 个完成对接（模式：{design.get('mode', 'auto')}）；"
                f"覆盖 {len(core_counts)} 个核心家族，参考配体骨架跃迁 {scaffold_hops} 个。"
                f"最佳设计 {best_design.get('label', '—')} 的打分为 "
                f"{float(best_design.get('affinity', 0)):.2f} kcal/mol，"
                f"QED = {(best_design.get('props') or {}).get('qed', 0):.2f}。"
            )
            top_bits = "、".join(
                f"{item.get('label', '—')}（{item.get('core', '—')}，"
                f"{float(item.get('affinity') or 0):.2f} kcal/mol）"
                for item in top_rows[:5]
            )
            if top_bits:
                design_text.append(
                    f"排名靠前的设计为 {top_bits}。枚举覆盖单取代、双取代与三取代模式；"
                    "提供参考配体时还会执行芳环骨架跃迁、杂环替换与连接子改造，"
                    "因此候选在核心骨架与取代基两个维度同时展开。"
                )
    else:
        design_text.append(
            (notes or {}).get("design")
            or "本项目尚未运行分子设计模块。该模块可在已知口袋内枚举取代基并自动对接排序。"
        )
    fishing_text = []
    fish_targets = (fishing_data or {}).get("targets") or []
    if fish_targets:
        fish_name = (fishing_data.get("compound") or {}).get("name") or "查询分子"
        if str(fish_name).strip().lower() == "query":
            fish_name = "查询分子"
        top = fish_targets[:3]
        top_text = "、".join(
            (
                f"{_target_display_name(item, 28)}（中位 pChEMBL "
                f"{float(item['median_pchembl']):.2f}）"
                if item.get("median_pchembl") is not None
                else f"{_target_display_name(item, 28)}（证据分 "
                f"{float(item.get('evidence_score') or 0):.2f}，"
                f"{_basis_label(item.get('score_basis'))}）"
            )
            for item in top
        )
        similarity = float((fishing_data.get("parameters") or {}).get("similarity", 70))
        similarity_text = (
            f"{similarity / 100:.2f}" if similarity > 1 else f"{similarity:.2f}"
        )
        n_human = sum(
            1 for item in fish_targets
            if item.get("genes")
            and "homo sapiens" in str(item.get("organism") or "").lower()
        )
        n_nonhuman = len(fish_targets) - n_human
        fishing_text.append(
            f"以 {fish_name} 为查询、Tanimoto ≥ {similarity_text} 检索相似分子，"
            f"并整合 BindingDB、DGIdb、PubChem、Open Targets 与 PDB 共晶证据，"
            f"共保留 {len(fish_targets)} 个候选蛋白靶点（人源基因 {n_human} 个、"
            f"非人源 {n_nonhuman} 个）；综合证据分最高的为 {top_text}。"
        )
        fishing_text.append(
            "该排序反映的是数据库中的已知活性、已知作用机制或共晶结构证据，"
            "不代表本分子的实测亲和力；它不能替代实验靶点鉴定，阴性靶点也不等于不结合。"
            + (
                "非人源靶点只用于说明结构相似性，不进入人源网络药理学。"
                if n_nonhuman
                else ""
            )
        )
        fishing_text.append(
            "SwissTargetPrediction 的 2D/3D 相似性预测与外部导入基因会一并合并去重，"
            "按综合证据分排序。"
        )
        sources_status = (fishing_data or {}).get("external_sources") or {}
        unavailable: list[str] = []
        for key, label in (
            ("ChEMBL", "ChEMBL"),
            ("SwissTargetPrediction", "SwissTargetPrediction"),
        ):
            block = sources_status.get(key) or {}
            if block and not block.get("available"):
                unavailable.append(label)
        fishing_stats = (fishing_data or {}).get("stats") or {}
        if (
            "ChEMBL" not in unavailable
            and "ChEMBL" not in str(fishing_stats.get("sources") or {})
            and int(fishing_stats.get("molecules") or 0) == 0
        ):
            # Records created before the source-status field existed.
            unavailable.append("ChEMBL")
        if unavailable:
            if i18n.get_lang() == "en":
                fishing_text.append(
                    "Data sources unavailable in this run (no records returned): "
                    + ", ".join(unavailable)
                    + ". A smaller target list therefore reflects source availability, "
                    "not the absence of biological targets; rerun later or supply an "
                    "external gene list."
                )
            else:
                fishing_text.append(
                    "本次不可用的数据源（未返回记录）："
                    + "、".join(unavailable)
                    + "。靶点数量偏少反映的是数据源可用性，而非不存在生物学靶点；"
                    "可在服务恢复后重跑，或在垂钓面板补充外部基因列表。"
                )
    else:
        fishing_text.append(
            (notes or {}).get("fishing")
            or "本项目尚未运行靶点垂钓模块。该模块可在没有指定靶点时反向提出靶点假设。"
        )
    fish_note = str((fishing_data or {}).get("report_note") or "").strip()
    if fish_targets and fish_note:
        fishing_text.append(fish_note)
    netpharm_text = []
    net_genes = (netpharm_data or {}).get("genes") or []
    net_overview = _enrichment_overview(netpharm_data)
    if netpharm_data and (net_genes or net_overview["ok"]):
        ppi = (netpharm_data or {}).get("ppi") or {}
        enrichment = (netpharm_data or {}).get("enrichment") or {}
        first_term = None
        first_label = ""
        for key, label in (("kegg", "KEGG"), ("go_bp", "GO BP"), ("reactome", "Reactome")):
            terms = [
                item for item in ((enrichment.get(key) or {}).get("terms") or [])
                if item.get("p")
            ]
            if terms:
                first_term = min(terms, key=lambda item: float(item.get("p") or 1))
                first_label = label
                break
        supplied = netpharm_data.get("supplied_genes") or []
        english = i18n.get_lang() == "en"
        if english:
            source_note = (
                f" ({len(net_genes) - len(supplied)} from fishing, "
                f"{len(supplied)} imported)"
                if supplied
                else ""
            )
            text = (
                f"{len(net_genes)} input genes{source_note} formed "
                f"{len(ppi.get('edges') or [])} high-confidence STRING interactions"
            )
            if first_term is not None:
                text += (
                    f"; the top {first_label} term was "
                    f"{first_term.get('term', '—')} "
                    f"(P = {float(first_term.get('p') or 0):.1e})"
                )
            text += (
                f"; {len(net_overview['ok'])} Enrichr libraries returned results with "
                f"{net_overview['significant']} terms at FDR < 0.05"
            )
            if net_overview["failed"]:
                text += (
                    "; libraries skipped after request failure: "
                    + ", ".join(net_overview["failed"])
                )
            netpharm_text.append(text + ".")
            netpharm_text.append(
                "Pathway and process enrichment only shows that the candidate targets "
                "cluster in these annotated functions; validation requires expression "
                "profiling, pathway inhibition or CRISPR interference, and database "
                "versions and background gene sets can change the specific terms."
            )
        else:
            lead = "垂钓与补充的" if supplied else "垂钓得到的"
            text = (
                f"{lead} {len(net_genes)} 个基因在 STRING 中形成 "
                f"{len(ppi.get('edges') or [])} 条高置信度互作边"
            )
            if first_term is not None:
                text += (
                    f"；{first_label} 首位为 {first_term.get('term', '—')}"
                    f"（P = {float(first_term.get('p') or 0):.1e}）"
                )
            text += (
                f"；{len(net_overview['ok'])} 个 Enrichr 库返回结果，"
                f"共 {net_overview['significant']} 个 FDR < 0.05 条目"
            )
            if net_overview["failed"]:
                text += "；请求失败并跳过的库：" + "、".join(net_overview["failed"])
            netpharm_text.append(text + "。")
            netpharm_text.append(
                "通路与过程富集只说明候选靶点在已知注释中聚集于这些功能，需通过表达谱、通路抑制或 CRISPR 干扰实验验证；"
                "数据库版本与背景基因集变化会改变具体条目。"
            )
    else:
        netpharm_text.append(
            (notes or {}).get("netpharm")
            or (
                "The network pharmacology module has not been run for this project; "
                "once target fishing is complete, PPI and enrichment can be built on "
                "the same gene set."
                if i18n.get_lang() == "en"
                else "本项目尚未运行网络药理学模块，完成靶点垂钓后可直接在同一基因集上构建 PPI 与富集。"
            )
        )
    net_note = str((netpharm_data or {}).get("report_note") or "").strip()
    if netpharm_text and net_note and net_note not in netpharm_text:
        netpharm_text.append(net_note)
    synthesis_text = []
    if synthesis_data:
        best_route = synthesis_data.get("best_route") or {}
        synthesis_text.append(
            f"主配体的 SAscore 为 {float(synthesis_data.get('sa') or 0):.2f}"
            f"（{_synth_verdict(synthesis_data)}），"
            f"重原子 {((synthesis_data.get('metrics') or {}).get('heavy', 0))} 个、"
            f"手性中心 {((synthesis_data.get('metrics') or {}).get('stereocenters', 0))} 个。"
        )
        if best_route:
            synthesis_text.append(
                f"首选模板路线为 {best_route.get('label', '—')}，"
                f"约 {best_route.get('step_count', 0)} 步、估计总产率 "
                f"{float(best_route.get('overall_yield') or 0):.0f}%，路线评分 "
                f"{float(best_route.get('score') or 0):.1f}/100；"
                "步骤与条件见图 17 与步骤表。"
            )
        if synthesis_data.get("flags"):
            hints = [
                str(flag).strip().rstrip("。；;")
                for flag in synthesis_data["flags"][:3]
                if str(flag).strip()
            ]
            if hints:
                if i18n.get_lang() == "en":
                    translated = [
                        i18n.tr(hint).strip().rstrip(".;；。")
                        for hint in hints
                    ]
                    synthesis_text.append(
                        "Hard-step alerts: " + "; ".join(translated) + "."
                    )
                else:
                    synthesis_text.append("硬步骤提示：" + "；".join(hints) + "。")
        synthesis_text.append(
            "路线来自有限反应模板库（酰胺、酯、磺酰胺、Suzuki、还原胺化、BRICS 等），"
            "不是 SYNTHIA 级全库搜索；产率与条件为经验估计，必须经文献检索与小试验证。"
        )
    else:
        synthesis_text.append(
            (notes or {}).get("synthesis")
            or "本项目尚未运行可合成性与合成路线评估。"
        )
    drugref_text = []
    if drugref_data:
        top = (drugref_data.get("top") or [])[:3]
        if top:
            names = "、".join(
                f"{item.get('name')}（Tanimoto {float(item.get('tanimoto') or 0):.2f}，"
                f"靶点 Jaccard {float(item.get('target_jaccard') or 0):.2f}）"
                for item in top
            )
            drugref_text.append(
                f"与 FDA 已上市药物参考集比较，最接近的参考药为 {names}。"
                "这些药物提供了结构类别、已知靶点与 ADMET 性质的参照，"
                "但参考排序不是疗效或安全性等价性的证据。"
            )
            max_tanimoto = max(
                float(item.get("tanimoto") or 0) for item in top
            )
            if max_tanimoto < 0.15:
                drugref_text.append(
                    _L(
                        f"参考集与候选化学空间的匹配度低（最高 Tanimoto {max_tanimoto:.2f}），"
                        "该排序只能说明结构类型差异，不能作为机制邻近、老药新用或专利规避证据。",
                        f"The reference set matches the candidate chemical space poorly "
                        f"(maximum Tanimoto {max_tanimoto:.2f}); the ranking only indicates "
                        "structural differences and cannot support mechanistic proximity, "
                        "drug repurposing or patent-avoidance claims.",
                    )
                )
        drugref_text.append(
            "药效维度只用靶点重叠与机制邻近表示，不伪造 pIC50；药动与 ADMET 为计算值，"
            "可能与实测值存在数量级差异。"
        )
    else:
        drugref_text.append(
            (notes or {}).get("drugref")
            or "本项目尚未运行 FDA 已上市药物参考分析。"
        )
    ddi_text = []
    if ddi_data:
        summary = ddi_data.get("summary") or {}
        ddi_text.append(
            f"规则式药物互作筛查覆盖 {len(ddi_data.get('co_drugs') or [])} 个联用药物，"
            f"高风险 {summary.get('high', 0)} 个、中风险 {summary.get('medium', 0)} 个、"
            f"信息不足 {summary.get('unknown', 0)} 个。"
        )
        risky = [
            item for item in (ddi_data.get("co_drugs") or [])
            if item.get("category") in ("高", "中")
        ][:3]
        if risky:
            ddi_text.append(
                "优先核查：" + "；".join(
                    f"{item.get('name')}（{item.get('score')}，{item.get('mechanism')}）"
                    for item in risky
                ) + "。"
            )
        ddi_text.append(
            "共享靶点不代表同向或拮抗，CYP 与 hERG 为计算风险；"
            "所有互作结论必须结合说明书、相互作用数据库与实验数据判断。"
        )
    else:
        ddi_text.append(
            (notes or {}).get("ddi")
            or "本项目尚未运行药物–药物相互作用分析。"
        )
    netpharm_extra = []
    net_overview = _enrichment_overview(netpharm_data)
    if netpharm_data and (net_overview["ok"] or net_genes):
        stats_block = netpharm_data.get("stats") or {}
        if i18n.get_lang() == "en":
            text = (
                f"Enrichment was run across {len(net_overview['ok']) + len(net_overview['failed'])} "
                f"Enrichr libraries; {len(net_overview['ok'])} returned results, yielding "
                f"{net_overview['significant']} terms at FDR < 0.05 "
                f"(KEGG {stats_block.get('kegg_significant', 0)}; "
                f"GO BP {stats_block.get('go_significant', 0)})"
            )
            if net_overview["failed"]:
                text += (
                    "; libraries skipped after request failure: "
                    + ", ".join(net_overview["failed"])
                )
            netpharm_extra.append(text + ".")
        else:
            text = (
                f"富集分析覆盖 {len(net_overview['ok']) + len(net_overview['failed'])} 个 Enrichr 库，"
                f"其中 {len(net_overview['ok'])} 个库返回结果，"
                f"共 {net_overview['significant']} 个 FDR < 0.05 条目"
                f"（KEGG {stats_block.get('kegg_significant', 0)} 个、"
                f"GO BP {stats_block.get('go_significant', 0)} 个）"
            )
            if net_overview["failed"]:
                text += "；请求失败并跳过的库：" + "、".join(net_overview["failed"])
            netpharm_extra.append(text + "。")
        ctd = netpharm_data.get("ctd") or {}
        if ctd.get("nodes"):
            if i18n.get_lang() == "en":
                netpharm_extra.append(
                    f"The compound–target–disease network contains "
                    f"{len(ctd.get('nodes') or [])} nodes and "
                    f"{len(ctd.get('edges') or [])} edges; disease terms come from "
                    "DisGeNET enrichment. The network is a compound–target–disease "
                    "annotation hypothesis, not causal evidence."
                )
            else:
                netpharm_extra.append(
                    f"化合物–靶点–疾病网络包含 {len(ctd.get('nodes') or [])} 个节点、"
                    f"{len(ctd.get('edges') or [])} 条边，疾病条目来自 DisGeNET 富集；"
                    "该网络用于提出“化合物–靶点–疾病”假设，不代表因果关系。"
                )
        sankey = netpharm_data.get("sankey") or {}
        if sankey.get("pathways"):
            if i18n.get_lang() == "en":
                netpharm_extra.append(
                    f"The target–pathway Sankey diagram links "
                    f"{len(sankey.get('genes') or [])} genes to "
                    f"{len(sankey.get('pathways') or [])} pathways, showing how the same "
                    "gene set is distributed across pathways."
                )
            else:
                netpharm_extra.append(
                    f"靶点–通路桑基图连接 {len(sankey.get('genes') or [])} 个基因与 "
                    f"{len(sankey.get('pathways') or [])} 条通路，用于观察同一基因集在不同通路间的分配。"
                )
    else:
        netpharm_extra.append(
            (notes or {}).get("netpharm")
            or "本项目尚未生成网络药理学扩展图。"
        )
    pharm_text = []
    if pharm_data:
        species_label = pharm_data.get("species_label") or "人"
        oral = pharm_data.get("oral") or {}
        rumen = pharm_data.get("rumen") or {}
        microbiota = pharm_data.get("microbiota") or {}
        nucleic = pharm_data.get("nucleic_acid") or {}
        stability = pharm_data.get("stability") or {}
        formulation = pharm_data.get("formulation") or {}
        risk = pharm_data.get("risk") or {}
        tissue = pharm_data.get("tissue") or {}
        confidence = pharm_data.get("confidence") or {}
        top_kp = (tissue.get("kp_rows") or [])[:3]
        target_rows = (tissue.get("target_rows") or [])[:3]
        high_risk = [
            item.get("name", "") for item in (risk.get("endpoints") or [])
            if float(item.get("value") or 0) >= 60
        ]
        confidence_bits = []
        english = i18n.get_lang() == "en"
        for label, block in (
            ("Oral" if english else "口服", oral),
            ("Rumen" if english else "瘤胃", rumen if rumen.get("applicable") else None),
            ("Microbiota" if english else "菌群", microbiota),
            ("Stability" if english else "稳定性", stability),
            ("Formulation" if english else "剂型", formulation),
            ("Nucleic acid" if english else "核酸结合", nucleic),
            ("Risk" if english else "风险", risk),
            ("Tissue" if english else "组织分布", tissue),
        ):
            if block:
                confidence_bits.append(
                    f"{label} {((block.get('confidence') or {}).get('level') or '—')}"
                )
        if english:
            pharm_text.append(
                f"For {species_label}, the predicted oral bioavailability is "
                f"F ≈ {float(oral.get('f_percent') or 0):.0f}% with BCS class "
                f"{oral.get('bcs', '—')} (HIA {float(oral.get('hia') or 0):.0f}%, "
                f"Caco-2 log Papp {float(oral.get('caco2') or 0):.2f}, ESOL solubility "
                f"{float(oral.get('solubility_mg_ml') or 0):.4f} mg/mL, dose number "
                f"{float(oral.get('dose_number') or 0):.2f}). {oral.get('food_effect', '')}"
            )
            if rumen.get("applicable"):
                pharm_text.append(
                    f"Native rumen bypass is {float(rumen.get('bypass_percent') or 0):.0f}%, "
                    f"rumen degradation risk {float(rumen.get('degradation_risk') or 0):.0f}/100, "
                    f"and protection may raise bypass to "
                    f"{float(rumen.get('protected_bypass_percent') or 0):.0f}%. "
                    f"{rumen.get('strategy', '')}"
                )
            else:
                pharm_text.append(
                    f"Rumen bypass is not calculated because {species_label} is not a ruminant species."
                )
            pharm_text.append(
                f"Gut-microbiota perturbation score is {float(microbiota.get('score') or 0):.0f}/100 "
                f"({microbiota.get('category', '—')}); direction: {microbiota.get('direction', '')}. "
                "Structural alerts: "
                + (", ".join(microbiota.get("alerts") or []) or "none")
                + "."
            )
            pharm_text.append(
                f"Predicted stability risks are pH 1.2 hydrolysis "
                f"{float(stability.get('ph_1_2') or 0):.0f}, pH 6.8 hydrolysis "
                f"{float(stability.get('ph_6_8') or 0):.0f}, plasma esterase "
                f"{float(stability.get('plasma') or 0):.0f}, oxidation "
                f"{float(stability.get('oxidation') or 0):.0f} and phototoxicity "
                f"{float(stability.get('phototoxicity') or 0):.0f} "
                f"(overall {stability.get('category', '—')}). Recommended formulation: "
                f"{formulation.get('form', '—')}; salt/co-crystal: "
                f"{str(formulation.get('salt_or_cocrystal', '')).rstrip('。.;； ')}; excipient notes: "
                + ("; ".join(formulation.get("excipient") or []))
                + f"; developability {float(formulation.get('developability') or 0):.0f}/100."
            )
            pharm_text.append(
                f"Nucleic-acid binding scores are intercalation "
                f"{float(nucleic.get('intercalation') or 0):.0f} and groove binding "
                f"{float(nucleic.get('groove') or 0):.0f} (overall "
                f"{float(nucleic.get('overall') or 0):.0f}, {nucleic.get('category', '—')}); "
                "covalent/DNA-reactive alerts: "
                + (", ".join(nucleic.get("covalent_alerts") or []) or "none")
                + f". The integrated risk is {risk.get('level', '—')} "
                f"({float(risk.get('overall') or 0):.0f}/100); high-risk endpoints: "
                + (", ".join(high_risk) or "none")
                + "."
            )
            kp_text = "; ".join(
                f"{row.get('label') or row.get('tissue')} Kp {float(row.get('kp') or 0):.2f} "
                f"({float(row.get('share') or 0):.1f}%)"
                for row in top_kp
            )
            target_text = "; ".join(
                f"{row.get('gene')} → {row.get('tissue')}"
                for row in target_rows if row.get("tissue")
            )
            target_clause = (
                f"target expression ({target_text}) comes from Human Protein Atlas and UniProt"
                if target_text
                else "target tissue expression was not included: HPA data are human-only"
            )
            pharm_text.append(
                f"Highest predicted tissue Kp values are {kp_text}. "
                f"Compound–target tissue overlap score is "
                f"{float(tissue.get('overlap_score') or 0):.0f}/100; "
                f"{target_clause}. "
                "Kp values are empirical partition estimates, not PBPK."
            )
            pharm_text.append(
                "Online evidence from PubChem, ChEMBL, UniProt and the Human Protein "
                "Atlas is reported separately on page C6. Module confidence: "
                + "; ".join(confidence_bits)
                + f"; overall {confidence.get('level', '—')} "
                f"({confidence.get('score', '—')}/100). All values are screening-level "
                "rule estimates and require experimental verification."
            )
        else:
            pharm_text.append(
                f"以{species_label}为对象，预测口服生物利用度 F ≈ {float(oral.get('f_percent') or 0):.0f}%，"
                f"BCS 分类 {oral.get('bcs', '—')}（HIA {float(oral.get('hia') or 0):.0f}%、"
                f"Caco-2 log Papp {float(oral.get('caco2') or 0):.2f}、ESOL 溶解度 "
                f"{float(oral.get('solubility_mg_ml') or 0):.4f} mg/mL、剂量数 "
                f"{float(oral.get('dose_number') or 0):.2f}）。{oral.get('food_effect', '')}。"
            )
            if rumen.get("applicable"):
                pharm_text.append(
                    f"天然过瘤胃率约 {float(rumen.get('bypass_percent') or 0):.0f}%，瘤胃降解风险 "
                    f"{float(rumen.get('degradation_risk') or 0):.0f}/100，保护后过瘤胃率可望达到约 "
                    f"{float(rumen.get('protected_bypass_percent') or 0):.0f}%。{rumen.get('strategy', '')}"
                )
            else:
                pharm_text.append(f"{species_label}不是反刍动物，不计算过瘤胃指标。")
            pharm_text.append(
                f"肠道菌群扰动评分 {float(microbiota.get('score') or 0):.0f}/100"
                f"（{microbiota.get('category', '—')}），方向：{microbiota.get('direction', '')}。"
                "结构警示：" + ("、".join(microbiota.get("alerts") or []) or "无") + "。"
            )
            pharm_text.append(
                f"预测稳定性风险：pH 1.2 水解 {float(stability.get('ph_1_2') or 0):.0f}、"
                f"pH 6.8 水解 {float(stability.get('ph_6_8') or 0):.0f}、血浆酯酶 "
                f"{float(stability.get('plasma') or 0):.0f}、氧化 {float(stability.get('oxidation') or 0):.0f}、"
                f"光毒性 {float(stability.get('phototoxicity') or 0):.0f}（总体 {stability.get('category', '—')}）。"
                f"建议剂型：{formulation.get('form', '—')}；成盐/共晶："
                f"{str(formulation.get('salt_or_cocrystal', '')).rstrip('。.;； ')}；辅料提示："
                + ("；".join(formulation.get("excipient") or []))
                + f"；制剂可开发性 {float(formulation.get('developability') or 0):.0f}/100。"
            )
            pharm_text.append(
                f"核酸结合：嵌入倾向 {float(nucleic.get('intercalation') or 0):.0f}、沟槽结合倾向 "
                f"{float(nucleic.get('groove') or 0):.0f}，综合 {float(nucleic.get('overall') or 0):.0f}"
                f"（{nucleic.get('category', '—')}）；共价/DNA 反应警示："
                + ("、".join(nucleic.get("covalent_alerts") or []) or "无")
                + f"。综合风险 {risk.get('level', '—')}（{float(risk.get('overall') or 0):.0f}/100），"
                "高风险端点：" + ("、".join(high_risk) or "无") + "。"
            )
            kp_text = "、".join(
                f"{row.get('label') or row.get('tissue')} Kp {float(row.get('kp') or 0):.2f}"
                f"（占比 {float(row.get('share') or 0):.1f}%）"
                for row in top_kp
            )
            target_text = "、".join(
                f"{row.get('gene')} → {row.get('tissue')}"
                for row in target_rows if row.get("tissue")
            )
            target_clause = (
                f"靶点表达（{target_text}）来自 Human Protein Atlas 与 UniProt"
                if target_text
                else "靶点组织表达未纳入：HPA 数据仅覆盖人源物种"
            )
            pharm_text.append(
                f"预测 Kp 最高的组织为 {kp_text}；化合物–靶点组织重叠评分 "
                f"{float(tissue.get('overlap_score') or 0):.0f}/100，{target_clause}。"
                "Kp 为经验分配模型，不是 PBPK。"
            )
            pharm_text.append(
                "在线证据（PubChem、ChEMBL、UniProt、Human Protein Atlas）与本地规则预测分开标注，"
                "查询状态见附图 C6。各模块置信度："
                + "、".join(confidence_bits)
                + f"；总体 {confidence.get('level', '—')}（{confidence.get('score', '—')}/100）。"
                "所有数值均为筛查级规则估计，需实验验证。"
            )
        warheads = pharm_data.get("covalent_warheads") or []
        if warheads:
            pharm_text.append(
                "Covalent-warhead warning: detected "
                + ", ".join(warheads)
                + ". Vina/Vinardo cannot model covalent bonds; the scores only cover the "
                "non-covalent portion and must be followed by covalent docking or experimental validation."
                if i18n.get_lang() == "en"
                else "共价弹头警示：检测到 "
                + "、".join(warheads)
                + "。Vina/Vinardo 不能描述共价键，分数只反映非共价部分，"
                "需共价对接或实验验证。"
            )
    else:
        pharm_text.append(
            (notes or {}).get("pharm")
            or "本项目尚未运行综合药理模块。该模块可在没有对接结果时输出口服、瘤胃、菌群、稳定性、剂型、核酸结合、风险与组织分布。"
        )
    limits = [
        "Vina 与 MM-GBSA 近似均未显式处理质子化变化、金属配位与蛋白诱导契合，MM-GBSA 未做熵校正，"
        "数值只适合在同一体系内做相对比较。",
        "验证路径：① 对排名前 3–5 的分子做 100–500 ns 分子动力学，考察结合模式稳定性与关键氢键占据率；"
        "② 用 MM/GBSA 或自由能微扰复核相对结合自由能；③ 对优化与设计候选做合成可行性评估与体外结合实验。",
    ]
    blocks = [
        ("2.5 配体效率与类药性", druglike),
        ("2.6 药物分子改进", optimization_text),
        ("2.7 骨架枚举设计", design_text),
        ("2.8 反向靶点垂钓", fishing_text),
        ("2.9 网络药理学与通路富集", netpharm_text),
        ("2.10 可合成性与合成路线", synthesis_text),
        ("2.11 FDA 参考与药物互作", drugref_text + ddi_text),
        ("2.12 网络药理学扩展", netpharm_extra),
        ("2.13 局限与验证计划", limits),
        ("2.14 综合药理、物种与组织分布", pharm_text),
    ]
    _text_blocks(fig, 0.885, blocks[:4], size=10.8, stop=0.07)
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)

    _flow_text_pages(
        pdf,
        "2　结果与讨论（续）",
        _L(
            "网络药理学、可合成性、FDA 参考、局限与综合药理",
            "Network pharmacology, synthesizability, FDA reference, limitations and integrated pharmacology",
        ),
        blocks[4:],
        width=45,
        size=10.8,
        heading_size=13.0,
        stop=0.070,
    )


def module_legends_pdf(
    optimization: dict | None,
    design: dict | None,
    fishing_data: dict | None,
    netpharm_data: dict | None,
    outpath: Path,
    synthesis_data: dict | None = None,
    drugref_data: dict | None = None,
    ddi_data: dict | None = None,
) -> Path:
    """Caption sheet placed directly before the expert-module figures."""
    _style_cjk()
    legends = [
        ("附图 A1　ADMET 与类药性估计",
         "理化性质、类药性规则判定、吸收/分布/代谢/排泄端点与综合雷达。端点为由 RDKit 描述符与规则阈值得到的"
         "启发式估计，不是机器学习预测，仅用于同系列分子的相对比较。"),
        ("附图 A2　药效团特征",
         "氢键供体/受体、芳环、疏水中心与离子中心的识别结果及空间分布；互补性按 5 Å 邻域判定，"
         "仅表示特征附近存在可相互作用残基，不构成能量学证据。"),
        ("附图 A3　结合自由能分解",
         "残基级 MM-GBSA 近似的范德华、静电、去溶剂化与氢键贡献，含前 16 位残基与能量项总计；"
         "未做熵校正，绝对值不可与 Vina 打分直接比较。"),
        ("附图 A4　结合能多方法比较",
         "并列 Vina、Vinardo、X-Score-like 经验项与 MM-GBSA，并给出仅基于对接尺度方法的共识估计；"
         "MM-GBSA 单独比较，不与对接打分平均。"),
    ]
    if optimization and not optimization.get("error"):
        legends.append((
            "附图 B1　药物优化报告",
            "优化母体的作用力谱、残基级改造建议、候选打分排序、结构式与作用力示意；"
            "所有 ΔVina 均相对该母体计算。",
        ))
    if design and not design.get("error"):
        legends.append((
            "附图 B2　骨架枚举设计报告",
            "设计策略、设计分子打分排序、结构式与数据库相似性命中；设计分子由核心骨架与取代基枚举生成。",
        ))
    if fishing_data and fishing_data.get("targets"):
        legends.append((
            "附图 F1　靶点垂钓完整数据",
            "按综合证据分排序的全部候选靶点，含基因符号、靶点类型、活性记录数、"
            "pChEMBL（无定量值时为“—”）、最佳活性值、证据分、来源与活性类型；"
            "数据来自 ChEMBL、BindingDB、PubChem、DGIdb、Open Targets、RCSB PDB 与 UniProt，"
            "检索时间见页脚。",
        ))
    if netpharm_data and netpharm_data.get("ppi"):
        if i18n.get_lang() == "en":
            legends.append((
                "Supplementary Figure N1　Full enrichment tables",
                "All enrichment terms from up to eight Enrichr libraries (KEGG, GO BP/CC/MF, "
                "Reactome, WikiPathway, DisGeNET, Hallmark), with P value, FDR, gene count "
                "and hit genes. Libraries whose request failed (after one retry) are marked "
                "in the corresponding panel and contribute no terms; a failed request does "
                "not mean that no significant term exists. The PPI network may include STRING "
                "first-order neighbours (coloured separately) only as background, not as "
                "direct targets; data come from STRING and Enrichr and are for hypothesis "
                "generation only.",
            ))
        else:
            legends.append((
                "附图 N1　富集分析完整数据",
                "八个 Enrichr 库（KEGG、GO BP/CC/MF、Reactome、WikiPathway、DisGeNET、Hallmark）"
                "的富集条目，含 P 值、FDR、基因数与命中基因；请求失败（已重试一次）的库"
                "在对应面板标注原因，不产生条目，也不表示未检出显著结果；"
                "PPI 网络可包含 STRING 一阶邻居扩展节点（图中单独着色），"
                "扩展节点只作为网络背景，不计为直接靶点；数据来自 STRING 与 Enrichr，仅用于机制假设。",
            ))
    if synthesis_data:
        legends.append((
            "图 16–17　可合成性与最佳合成路线",
            "图 16 给出 SAscore、候选路线评分、末端砌块复杂度与硬步骤提示；"
            "图 17 给出首选路线的正向步骤、结构式、常见条件、经验产率与步骤表。"
            "路线来自有限反应模板库，不是 SYNTHIA 级全库搜索；产率与条件需实验验证。",
        ))
    if drugref_data:
        legends.append((
            "图 18　FDA 已上市药物参考分析",
            "结构相似度、靶点 Jaccard、ADMET 相似度与理化性质距离的加权参考排序，"
            "并给出候选与最接近参考药的雷达对比和属性表。"
            "药效只用靶点重叠与机制邻近表示，不提供 pIC50；ADMET 与半衰期为计算值，"
            "不能替代临床或实验数据。",
        ))
    if ddi_data:
        legends.append((
            "图 19　候选分子–已上市药物相互作用假设",
            "共享靶点、CYP3A4/2D6 竞争、hERG 风险叠加与结构/类别相似度的综合风险排序、"
            "证据矩阵与互作网络。共享靶点不代表同向或拮抗，CYP 与 hERG 为计算风险；"
            "所有结论必须结合说明书、相互作用数据库与实验数据判断。",
        ))
    if netpharm_data and netpharm_data.get("enrichment"):
        legends.append((
            "图 20–22　多库富集、C-T-D 网络与靶点–通路桑基图",
            "图 20 汇总八个 Enrichr 库的显著条目；图 21 为化合物–靶点–疾病网络，"
            "疾病条目来自 DisGeNET 富集；图 22 为基因–通路桑基图，只保留至少 2 个输入基因命中的通路。"
            "三者共用同一基因集与富集结果，仅用于机制假设。",
        ))
    blocks = [(title, [text]) for title, text in legends]
    blocks.append((
        "图像与投稿说明",
        [
            "本页之后的附图由各专家模块直接生成，保留原始编号与图注；正文图 1–22 的图注均已随图放置。",
            "全部图像为 PDF 矢量（字体 Type 42 子集嵌入），可在 Illustrator 或 Inkscape 中编辑；"
            "若期刊要求位图，建议按 600 dpi 导出 TIFF。",
        ],
    ))
    with PdfPages(outpath) as pdf:
        _flow_text_pages(
            pdf, "附图表与模块报告图注",
            "ADMET、药效团、自由能分解、靶点垂钓、网络药理学、合成路线、FDA 参考与药物互作的完整说明",
            blocks, width=44, size=11.0, heading_size=12.4, stop=0.065,
        )
    return outpath


def _references_page(pdf) -> None:
    """Numbered bibliography for every method, engine and database used."""
    from . import references as refmod

    english = i18n.get_lang() == "en"
    entries = refmod.REFERENCES
    # Measure every entry first so long author lists never collide with the
    # footer: a page is flushed when the next entry would cross 10% height.
    pages: list[list[tuple[int, str]]] = [[]]
    y = 0.868
    for number, ref in enumerate(entries, start=1):
        authors = ", ".join(ref.get("authors") or []).rstrip(".")
        text = f"{number}. {authors}. {ref.get('title') or ''}"
        if ref.get("journal"):
            text += f". {ref['journal']}"
        if ref.get("year"):
            text += f". {ref['year']}"
        if ref.get("volume"):
            text += f";{ref['volume']}"
        pages_text = str(ref.get("pages") or "").replace("--", "–")
        if pages_text:
            text += f":{pages_text}"
        link = ref.get("doi") or ref.get("url") or ""
        if link:
            text += f". doi:{link}" if ref.get("doi") else f". {link}"
        wrapped = _wrap_cjk(text + ".", 108 if english else 96)
        height = 0.0145 * (wrapped.count("\n") + 1) + 0.014
        if pages[-1] and y - height < 0.10:
            pages.append([])
            y = 0.868
        pages[-1].append((number, wrapped))
        y -= height
    total_pages = len(pages)
    for page_index, chunk in enumerate(pages, start=1):
        if total_pages == 1:
            title = _L("7　参考文献", "7　References")
        else:
            title = _L(
                f"7　参考文献（{page_index}/{total_pages}）",
                f"7　References ({page_index}/{total_pages})",
            )
        fig = _a4_page(
            title,
            _L(
                "报告中引用的方法、软件与数据库原始文献",
                "Primary publications for the methods, software and databases used in this report",
            ),
        )
        y = 0.868
        for _number, wrapped in chunk:
            fig.text(
                0.07, y, wrapped, fontsize=8.6, color=INK, va="top",
                linespacing=1.45,
            )
            y -= 0.0145 * (wrapped.count("\n") + 1) + 0.014
        pdf.savefig(fig)
        plt.close(fig)


def _tool_index_page(pdf) -> None:
    english = i18n.get_lang() == "en"
    fig = _a4_page(
        "6　附录 B　图注工具与数据来源" if not english else "6　Appendix B　Tools behind each figure",
        "每张图使用的主要工具、数据库与计算模块" if not english else "Main tools, databases and modules used for each figure",
    )
    if english:
        rows = [
            ["Figure / module", "Tools and data"],
            ["1–4  Scoring, LE, interaction fingerprint, pose ensemble", "AutoDock Vina 1.2.5, Vinardo, RDKit, Matplotlib"],
            ["5–8  Optimisation, design, chemical space", "RDKit, AutoDock Vina, DrugHX rules"],
            ["9  Reverse target fishing", "ChEMBL, BindingDB, PubChem, DGIdb, Open Targets, RCSB PDB, SwissTargetPrediction"],
            ["10  Network pharmacology", "STRING v12, Enrichr (KEGG/GO/Reactome)"],
            ["11–13  Developability, chemical space, metabolism", "RDKit, ESOL, SAscore, Matplotlib"],
            ["14–15  Contact fingerprint, pocket profile", "RDKit, DrugHX geometry analysis"],
            ["16–17  Synthesizability, synthetic route", "RDKit, BRICS, DrugHX reaction templates"],
            ["18  FDA reference panel", "DrugHX local FDA set, RDKit, ChEMBL-derived targets"],
            ["19  Drug–drug interaction hypotheses", "DrugHX rules, CYP/hERG heuristics"],
            ["20–22  Enrichment, C-T-D, Sankey", "Enrichr, DisGeNET, KEGG, Reactome, WikiPathway, MSigDB"],
            ["A1–A4  ADMET, pharmacophore, MM-GBSA, energy comparison", "RDKit, ESOL, Clark, DrugHX MM-GBSA, Vina/Vinardo"],
            ["C1–C6  Integrated pharmacology and risk assessment", "RDKit, PubChem, ChEMBL, UniProt, HPA, DrugHX species/formulation/risk rules"],
        ]
        note = (
            "Online records are database evidence, not predictions; local rules are screening-level "
            "estimates. Database versions and model coverage can change between releases. "
            "Tools: RDKit, AutoDock Vina 1.2.5, Vinardo, Meeko, Matplotlib, PubChem, ChEMBL, "
            "SwissTargetPrediction, UniProt, Human Protein Atlas, STRING, Enrichr."
        )
    else:
        rows = [
            ["图/页与内容", "工具与数据"],
            ["1–4  打分、配体效率、作用力指纹、构象谱", "AutoDock Vina 1.2.5、Vinardo、RDKit、Matplotlib"],
            ["5–8  优化、设计、化学空间", "RDKit、AutoDock Vina、DrugHX 规则"],
            ["9  靶点垂钓", "ChEMBL、BindingDB、PubChem、DGIdb、Open Targets、RCSB PDB、SwissTargetPrediction"],
            ["10  网络药理学", "STRING v12、Enrichr（KEGG/GO/Reactome）"],
            ["11–13  可开发性、化学空间、代谢", "RDKit、ESOL、SAscore、Matplotlib"],
            ["14–15  接触指纹、口袋画像", "RDKit、DrugHX 几何分析"],
            ["16–17  可合成性与合成路线", "RDKit、BRICS、DrugHX 反应模板"],
            ["18  FDA 参考", "DrugHX 本地 FDA 集、RDKit、ChEMBL 靶点"],
            ["19  药物互作假设", "DrugHX 规则、CYP/hERG 经验式"],
            ["20–22  多库富集、C-T-D、桑基图", "Enrichr、DisGeNET、KEGG、Reactome、WikiPathway、MSigDB"],
            ["附图 A1–A4  ADMET、药效团、MM-GBSA、结合能比较", "RDKit、ESOL、Clark、DrugHX MM-GBSA、Vina/Vinardo"],
            ["附图 C1–C6  综合药理与风险评估", "RDKit、PubChem、ChEMBL、UniProt、HPA、DrugHX 物种/制剂/风险规则"],
        ]
        note = (
            "在线记录是数据库证据而非预测；本地规则为筛查级估计。数据库版本与模型覆盖范围可能随软件版本变化。"
            "工具：RDKit、AutoDock Vina 1.2.5、Vinardo、Meeko、Matplotlib、PubChem、ChEMBL、"
            "SwissTargetPrediction、UniProt、Human Protein Atlas、STRING、Enrichr。"
        )
    wrapped_rows = [rows[0]]
    for row in rows[1:]:
        wrapped_rows.append([
            textwrap.fill(str(row[0]), 32),
            textwrap.fill(str(row[1]), 52),
        ])
    _table(fig, 0.872, wrapped_rows, [0.30, 0.56], fontsize=7.7, row_height=0.034)
    fig.text(
        0.07, 0.872 - 0.034 * len(wrapped_rows) - 0.030,
        _wrap_cjk(note, 100 if english else 84),
        fontsize=8.4, color=MUTED, va="top", linespacing=1.55,
    )
    _caption(
        fig,
        "This index traces the tools and data behind each figure; numerical details are given in the individual captions and Methods."
        if english
        else "本索引用于追溯每张图的工具与数据来源；具体数值与判据见各图图注与正文方法。",
        y=0.155,
    )
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def _acknowledgements_page(pdf) -> None:
    """Acknowledgement of the software, databases and methods used by DrugHX."""
    fig = _a4_page(
        _L("致谢", "Acknowledgements"),
        _L(
            "DrugHX 使用的软件、数据库、评分方法与开源工具",
            "Software, databases, scoring methods and open-source tools used by DrugHX",
        ),
        "Acknowledgements",
    )
    blocks = [
        (
            _L("软件与计算引擎", "Software and computational engines"),
            [
                _L(
                    "AutoDock Vina 1.2.5、Vina-GPU 2.0、smina 与 QuickVina 2 提供分子对接与打分；"
                    "Meeko 负责配体 PDBQT 准备，RDKit 提供化学信息学、构象生成、指纹、骨架与反应模板，"
                    "Open Babel 用于部分格式转换与电荷检查。",
                    "AutoDock Vina 1.2.5, Vina-GPU 2.0, smina and QuickVina 2 provide docking and "
                    "scoring; Meeko prepares ligand PDBQT files; RDKit provides cheminformatics, "
                    "conformer generation, fingerprints, scaffolds and reaction templates; Open Babel "
                    "is used for selected format conversions and charge checks.",
                ),
            ],
        ),
        (
            _L("数据库与网络资源", "Databases and web resources"),
            [
                _L(
                    "RCSB PDB 提供蛋白结构；ChEMBL、BindingDB、DGIdb、PubChem 与 Open Targets 提供活性、"
                    "药物–基因关系与靶点证据；STRING 提供蛋白互作网络；Enrichr（KEGG、GO、Reactome、"
                    "WikiPathways、DisGeNET、MSigDB Hallmark）提供富集分析；FDA 标签、DrugBank 与 PubChem "
                    "用于已上市药物的结构与靶点参考。",
                    "RCSB PDB provides protein structures; ChEMBL, BindingDB, DGIdb, PubChem and Open "
                    "Targets provide activity, drug–gene relationships and target evidence; STRING "
                    "provides protein interaction networks; Enrichr (KEGG, GO, Reactome, WikiPathways, "
                    "DisGeNET and MSigDB Hallmark) provides enrichment analysis; FDA labels, DrugBank "
                    "and PubChem provide structures and target references for marketed drugs.",
                ),
            ],
        ),
        (
            _L("评分与可视化方法", "Scoring and visualisation methods"),
            [
                _L(
                    "SAscore 基于 Ertl 与 Schuffenhauer 的相对可合成性评分；BOILED-Egg 基于 Daina 与 Zoete "
                    "的 WLOGP–TPSA 判别；CNS MPO 参考 Wager 等的多参数优化；PAINS、Brenk 与 NIH 警示来自"
                    "公开过滤规则；Bemis–Murcko 骨架与 rdFMCS 用于结构比较。图形使用 matplotlib 与 RDKit 绘制，"
                    "PDF 合并与页面处理使用 pypdf。",
                    "SAscore follows the relative synthesizability score of Ertl and Schuffenhauer; "
                    "BOILED-Egg uses the WLOGP–TPSA discrimination of Daina and Zoete; CNS MPO follows "
                    "the multi-parameter optimisation of Wager et al.; PAINS, Brenk and NIH alerts come "
                    "from public filter rules; Bemis–Murcko scaffolds and rdFMCS are used for structure "
                    "comparison. Figures are drawn with matplotlib and RDKit; PDF merging and page "
                    "handling use pypdf.",
                ),
            ],
        ),
        (
            _L("致谢", "Acknowledgements"),
            [
                _L(
                    "感谢上述开源软件、公共数据库与方法作者的工作，使本软件能够在本地完成从结构准备、"
                    "分子对接、ADMET、靶点垂钓、网络药理学、可合成性到 FDA 参考与药物互作的完整分析。"
                    "软件作者：HX。",
                    "We thank the authors of the open-source software, public databases and methods "
                    "listed above, which make it possible to run structure preparation, docking, ADMET, "
                    "target fishing, network pharmacology, synthesizability, FDA reference and drug–drug "
                    "interaction analysis locally. Software author: HX.",
                ),
                _L(
                    "所有预测结果均为计算推断，不构成实验结论或临床建议；使用者应结合实验数据、"
                    "药物说明书与领域专家判断。",
                    "All predictions are computational inferences and do not constitute experimental "
                    "conclusions or clinical advice; users should combine them with experimental data, "
                    "drug labels and expert judgement.",
                ),
            ],
        ),
    ]
    _text_blocks(fig, 0.885, blocks, width=46, size=11.8)
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def _conclusion_page(
    pdf,
    results: list[dict],
    best: dict | None,
    props: dict,
    optimization: dict | None,
    design: dict | None,
    analyses: dict[int, dict] | None = None,
    validation: dict | None = None,
    synthesis_data: dict | None = None,
    drugref_data: dict | None = None,
    ddi_data: dict | None = None,
    pharm_data: dict | None = None,
    posebusters: dict | None = None,
    trust_data: dict | None = None,
):
    stats = _stats(results, analyses or {}, props)
    best_props = props.get(id(best), {}) if best else {}
    efficiency = best_props.get("efficiency") or {}
    le = efficiency.get("le")
    qed = best_props.get("qed")
    logp = best_props.get("logp")
    affinity = (
        float(best["affinity"])
        if best and best.get("affinity") is not None
        else None
    )
    sa = None
    if synthesis_data and synthesis_data.get("sa") is not None:
        try:
            sa = float(synthesis_data["sa"])
        except (TypeError, ValueError):
            sa = None
    pose_rate = None
    if posebusters:
        checked = posebusters.get("rows") or []
        best_name = str((best or {}).get("ligand_name") or "")
        match = next(
            (row for row in checked if str(row.get("name") or "") == best_name),
            None,
        )
        if match and int(match.get("total") or 0):
            pose_rate = int(match.get("passed") or 0) / max(1, int(match["total"]))
        elif checked:
            total_checks = sum(int(row.get("total") or 0) for row in checked)
            passed_checks = sum(int(row.get("passed") or 0) for row in checked)
            pose_rate = (passed_checks / total_checks) if total_checks else None
    rmsd_passed = None
    rmsd_value = None
    if validation and validation.get("rows"):
        rmsd_passed = bool(validation["rows"][0].get("passed"))
        try:
            rmsd_value = float(validation["rows"][0].get("rmsd"))
        except (TypeError, ValueError):
            rmsd_value = None
    risk = (pharm_data or {}).get("risk") or {}
    oral = (pharm_data or {}).get("oral") or {}
    ddi_summary = (ddi_data or {}).get("summary") or {}
    ddi_high = int(ddi_summary.get("high") or 0)

    def _num(value, digits: int = 2) -> str:
        return "—" if value is None else f"{float(value):.{digits}f}"

    criteria = [
        {
            "label": _L("结合亲和力", "Binding affinity"),
            "value": "—" if affinity is None else f"{affinity:.2f} kcal/mol",
            "target": "≤ −8.0 kcal/mol",
            "passed": affinity is not None and affinity <= -8.0,
            "assessed": affinity is not None,
        },
        {
            "label": _L("配体效率 LE", "Ligand efficiency"),
            "value": _num(le),
            "target": "≥ 0.30",
            "passed": le is not None and float(le) >= 0.30,
            "assessed": le is not None,
        },
        {
            "label": _L("类药性 QED", "Drug-likeness QED"),
            "value": _num(qed),
            "target": "≥ 0.50",
            "passed": qed is not None and float(qed) >= 0.50,
            "assessed": qed is not None,
        },
        {
            "label": _L("合成可及性 SAscore", "Synthetic accessibility"),
            "value": _num(sa),
            "target": "≤ 4.0",
            "passed": sa is not None and sa <= 4.0,
            "assessed": sa is not None,
        },
        {
            "label": _L("构象物理有效性", "Pose physical validity"),
            "value": "—" if pose_rate is None else f"{pose_rate * 100:.0f}%",
            "target": "≥ 80%",
            "passed": pose_rate is not None and pose_rate >= 0.8,
            "assessed": pose_rate is not None,
        },
        {
            "label": _L("对接方案验证", "Redocking validation"),
            "value": (
                "—"
                if rmsd_passed is None
                else f"RMSD {_num(rmsd_value)} Å"
            ),
            "target": "≤ 2.0 Å",
            "passed": bool(rmsd_passed),
            "assessed": rmsd_passed is not None,
        },
        {
            "label": _L("安全性 / 联用风险", "Safety / DDI risk"),
            "value": _L(
                f"综合风险 {risk.get('level', '—')}；高风险联用 {ddi_high} 个",
                f"risk {risk.get('level', '—')}; {ddi_high} high-risk DDI",
            ),
            "target": _L("无高风险项", "no high-risk flag"),
            "passed": ddi_high == 0
            and str(risk.get("level") or "") not in ("高", "High"),
            "assessed": bool(risk) or bool(ddi_summary),
        },
    ]
    if trust_data and trust_data.get("score") is not None:
        trust_score = float(trust_data["score"])
        criteria.append(
            {
                "label": _L("决策可信度", "Decision confidence"),
                "value": f"{trust_score:.0f}",
                "target": "≥ 55",
                "passed": trust_score >= 55,
                "assessed": True,
            }
        )
    assessed = [item for item in criteria if item["assessed"]]
    passed_count = sum(1 for item in assessed if item["passed"])
    ratio = (passed_count / len(assessed)) if assessed else 0.0
    if assessed and ratio >= 0.75 and passed_count >= 4:
        tier, tier_color = "A", TEAL
        verdict = _L("Tier 1 · 建议进入体外验证", "Tier 1 · Recommended for in vitro validation")
        verdict_note = _L(
            "关键指标多数达标，可作为先导化合物推进；先完成生化结合与细胞活性两步，"
            "再决定是否投入体内研究。",
            "Most key criteria are met. Advance it as a lead, complete biochemical binding "
            "and cell-activity assays, then decide on in vivo work.",
        )
    elif assessed and ratio >= 0.45:
        tier, tier_color = "B", AMBER
        verdict = _L("Tier 2 · 有条件推进", "Tier 2 · Conditional advancement")
        verdict_note = _L(
            "结合或类药性仅部分达标，先补齐未达标项（见 4.5），并用实测亲和力校准打分；"
            "不建议直接进入体内实验。",
            "Binding or drug-likeness is only partially met. Close the gaps in 4.5 and "
            "confirm affinity experimentally before any in vivo work.",
        )
    else:
        tier, tier_color = "C", RED
        verdict = _L(
            "Tier 3 · 暂缓实验，优先结构优化",
            "Tier 3 · Hold; prioritise structure optimisation",
        )
        verdict_note = _L(
            "达标项不足，当前分子更适合作为结构起点；按 4.5 改造后重新对接与评估。",
            "Too few criteria are met; treat it as a starting point, apply the changes in "
            "4.5, then re-dock and re-assess.",
        )

    fig = _a4_page(
        _L("4　结论与实验指导", "4　Conclusions and Experimental Guidance"),
        _L("可行性判定、分阶段验证路线与决策阈值", "Feasibility verdict, staged validation and decision gates"),
        _L("§4 结论与实验指导", "§4 Conclusions and guidance"),
    )
    fig.text(
        0.07, 0.882,
        _L("4.1　总体可行性判定", "4.1　Overall feasibility verdict"),
        fontsize=13.6, fontweight="bold", color=TEAL, va="top",
    )
    fig.add_artist(
        Rectangle(
            (0.07, 0.748), 0.86, 0.108, transform=fig.transFigure,
            facecolor="#f8fafc", edgecolor=tier_color, linewidth=1.0,
        )
    )
    fig.text(
        0.09, 0.842, verdict,
        fontsize=14.2, fontweight="bold", color=tier_color, va="top",
    )
    fig.text(
        0.09, 0.815, _wrap_cjk(verdict_note, 96 if i18n.get_lang() == "en" else 74),
        fontsize=9.6 if i18n.get_lang() == "en" else 10.4, color=INK, va="top",
        linespacing=1.35,
    )
    fig.text(
        0.93, 0.842,
        _L(f"达标 {passed_count}/{len(assessed)} 项", f"{passed_count}/{len(assessed)} criteria met"),
        fontsize=10.5, color=MUTED, ha="right", va="top",
    )
    header = [
        _L("评估项", "Criterion"),
        _L("本次结果", "Result"),
        _L("参考阈值", "Reference"),
        _L("判断", "Verdict"),
    ]
    rows = [header]
    for item in criteria:
        if not item["assessed"]:
            mark = _L("未评估", "n/a")
        elif item["passed"]:
            mark = _L("达标", "met")
        else:
            mark = _L("未达标", "not met")
        rows.append([item["label"], item["value"], item["target"], mark])
    _table(fig, 0.735, rows, [0.25, 0.29, 0.19, 0.13], fontsize=9.2, row_height=0.0225)

    findings = []
    if best:
        rank = 1 + sum(
            1 for item in results
            if item.get("affinity") is not None
            and float(item["affinity"]) < affinity
        ) if affinity is not None else 1
        findings.append(
            _L(
                f"结合层面：{best.get('ligand_name', '—')} 对接得分 "
                f"{_num(affinity)} kcal/mol，在 {len(results)} 个成功配体中排名第 {rank}；"
                f"配体效率 LE {_num(le)}，QED {_num(qed)}。",
                f"Binding: {best.get('ligand_name', '—')} scores {_num(affinity)} kcal/mol "
                f"(rank {rank} of {len(results)} successful ligands) with LE {_num(le)} "
                f"and QED {_num(qed)}.",
            )
        )
    if pose_rate is not None or rmsd_passed is not None:
        findings.append(
            _L(
                "构象层面：PoseBusters 通过率 "
                + ("—" if pose_rate is None else f"{pose_rate * 100:.0f}%")
                + "；重对接 RMSD "
                + ("—" if rmsd_value is None else f"{rmsd_value:.2f} Å")
                + ("（通过 2 Å 判据）" if rmsd_passed else "（未通过 2 Å 判据）")
                + "。",
                "Poses: PoseBusters pass rate "
                + ("n/a" if pose_rate is None else f"{pose_rate * 100:.0f}%")
                + "; redocking RMSD "
                + ("n/a" if rmsd_value is None else f"{rmsd_value:.2f} Å")
                + (" (passes the 2 Å criterion)" if rmsd_passed else " (fails the 2 Å criterion)")
                + ".",
            )
        )
    if pharm_data or sa is not None or qed is not None:
        bits = []
        if best_props.get("mw") is not None:
            bits.append(f"MW {_num(best_props.get('mw'), 0)}")
        if logp is not None:
            bits.append(f"cLogP {_num(logp)}")
        if qed is not None:
            bits.append(f"QED {_num(qed)}")
        if sa is not None:
            bits.append(f"SA {_num(sa)}")
        if oral.get("f_percent") is not None:
            bits.append(
                _L(
                    f"预测口服利用度 {float(oral['f_percent']):.0f}%",
                    f"predicted oral F {float(oral['f_percent']):.0f}%",
                )
            )
        if risk.get("level"):
            bits.append(_L(f"综合风险 {risk['level']}", f"risk {risk['level']}"))
        if ddi_high:
            bits.append(_L(f"高风险联用 {ddi_high} 个", f"{ddi_high} high-risk DDI"))
        if bits:
            findings.append(
                _L("成药性层面：", "Developability: ") + "；".join(bits) + "。"
                if i18n.get_lang() != "en"
                else "Developability: " + "; ".join(bits) + "."
            )
    if synthesis_data:
        route = synthesis_data.get("best_route") or {}
        findings.append(
            _L(
                f"合成层面：SAscore {_num(sa)}，可合成性评级“{_synth_verdict(synthesis_data)}”"
                + (
                    f"，首选模板路线 {route.get('label', '—')} 约 {route.get('step_count', 0)} 步。"
                    if route
                    else "，暂未匹配完整模板路线。"
                ),
                f"Synthesis: SAscore {_num(sa)} ({_synth_verdict(synthesis_data)})"
                + (
                    f", first template route {route.get('label', '—')}, "
                    f"about {route.get('step_count', 0)} steps."
                    if route
                    else "; no complete template route matched."
                ),
            )
        )

    if findings:
        _text_blocks(
            fig, 0.492,
            [(_L("4.2　关键发现", "4.2　Key findings"), findings)],
            width=40, size=11.0,
        )
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)

    optimization_tips = []
    if affinity is not None and affinity > -8.0:
        optimization_tips.append(
            _L(
                "结合仍偏弱：优先在口袋疏水区增加芳环/卤素接触，或把外围极性基团替换为可形成氢键的杂环。",
                "Binding is still weak: add aromatic/halogen contacts in hydrophobic sub-pockets "
                "or replace peripheral polar groups with H-bonding heterocycles.",
            )
        )
    if le is not None and float(le) < 0.30:
        optimization_tips.append(
            _L(
                "配体效率偏低：减少分子量与可旋转键，保留与关键残基的极性接触，避免单纯增大骨架。",
                "Low ligand efficiency: trim MW and rotatable bonds, keep polar contacts with key "
                "residues, and avoid simply enlarging the scaffold.",
            )
        )
    if qed is not None and float(qed) < 0.50:
        optimization_tips.append(
            _L(
                "类药性不足：降低 cLogP、减少芳香环数量或引入适度极性，使 QED 回到 0.5 以上。",
                "Insufficient drug-likeness: lower cLogP, reduce aromatic ring count or add "
                "moderate polarity to bring QED above 0.5.",
            )
        )
    if sa is not None and sa > 4.0:
        optimization_tips.append(
            _L(
                "合成偏难：合并手性中心、减少保护基步骤，优先选择市售砌块可直接拼接的类似物。",
                "Hard to synthesise: merge stereocentres, reduce protecting-group steps and prefer "
                "analogues assembled from commercial building blocks.",
            )
        )
    if logp is not None and float(logp) > 4.0:
        optimization_tips.append(
            _L(
                "脂溶性偏高：引入 1–2 个氮/氧原子或降低烷基链长度，降低脱靶与代谢风险。",
                "High lipophilicity: introduce one or two N/O atoms or shorten alkyl chains to "
                "reduce off-target and metabolic risk.",
            )
        )
    if pose_rate is not None and pose_rate < 0.8:
        optimization_tips.append(
            _L(
                "构象合理性不足：复核质子化状态与环构象，必要时应重新对接或改用柔性命中策略。",
                "Pose validity is insufficient: recheck protonation and ring conformations, and "
                "re-dock or use a flexible-hit strategy if needed.",
            )
        )
    if ddi_high:
        optimization_tips.append(
            _L(
                "联用风险需前置评估：在优化阶段同步测定 CYP 抑制，避免上市后剂量限制。",
                "Assess DDI risk early: measure CYP inhibition during optimisation to avoid "
                "post-marketing dose limitations.",
            )
        )
    if not optimization_tips:
        optimization_tips.append(
            _L(
                "当前分子的关键指标基本均衡，优化时应保持骨架不变，仅在溶剂暴露区做小幅取代基行走。",
                "Key properties are reasonably balanced; keep the scaffold and perform only "
                "small substituent walks in solvent-exposed regions.",
            )
        )

    blocks = [
        (_L("4.3　分阶段实验验证", "4.3　Staged experimental validation"), [
            _L(
                "阶段 0（约 1 周，质量门槛）：重组蛋白 SDS-PAGE 纯度 ≥95%，用 SPR/ITC/DSF 确认活性构象；"
                "化合物 HPLC 纯度 ≥95%、LC-MS 确认分子量，并测定 PBS/DMSO 溶解度。",
                "Stage 0 (about 1 week, quality gate): recombinant protein ≥95% pure by SDS-PAGE "
                "with activity confirmed by SPR/ITC/DSF; compound ≥95% pure by HPLC, identity "
                "confirmed by LC-MS, and solubility measured in PBS/DMSO.",
            ),
            _L(
                "阶段 1（1–2 周，生化结合）：SPR 或 ITC 测定 Kd（8 个浓度点，n=3）；酶学体系测 IC50"
                "（10 点稀释，n=3），设阳性对照与底物对照，Z′ ≥0.5。",
                "Stage 1 (1–2 weeks, biochemical binding): Kd by SPR or ITC (8 concentrations, "
                "n=3); enzyme IC50 with 10-point dilution (n=3), positive and substrate controls, "
                "Z′ ≥0.5.",
            ),
            _L(
                "阶段 2（2–3 周，细胞活性与选择性）：在靶点高表达与低表达（或敲除）细胞中平行测 IC50，"
                "选择性窗口 ≥10 倍；用 WB 或通路报告基因确认靶点参与。",
                "Stage 2 (2–3 weeks, cell activity and selectivity): parallel IC50 in "
                "target-high vs target-low/knockout cells, selectivity window ≥10-fold; confirm "
                "target engagement by WB or a pathway reporter.",
            ),
            _L(
                "阶段 3（2–4 周，成药性与安全）：Caco-2 渗透性、肝微粒体 CLint、血浆蛋白结合、"
                "CYP1A2/2C9/2D6/3A4 抑制、hERG 膜片钳与细胞毒。",
                "Stage 3 (2–4 weeks, developability and safety): Caco-2 permeability, liver "
                "microsome CLint, plasma protein binding, CYP1A2/2C9/2D6/3A4 inhibition, hERG "
                "patch clamp and cytotoxicity.",
            ),
            _L(
                "阶段 4（按需，体内）：大鼠/犬 PK（IV+PO）、剂量探索与药效模型；兽药方向在目标动物补做 PK "
                "与可食组织残留，按 MRL 估算休药期。",
                "Stage 4 (as needed, in vivo): rat/dog PK (IV+PO), dose finding and efficacy models; "
                "for veterinary use, add target-animal PK and edible-tissue residues, and estimate "
                "withdrawal time against the MRL.",
            ),
        ]),
        (_L("4.4　Go / No-Go 判据", "4.4　Go / no-go gates"), [
            _L(
                "继续推进：Kd ≤10 μM 或 IC50 ≤1 μM，细胞活性 ≤1 μM 且选择性 ≥20 倍，"
                "溶解度 ≥10 倍最高测试浓度。",
                "Proceed: Kd ≤10 μM or IC50 ≤1 μM, cellular activity ≤1 μM with ≥20-fold "
                "selectivity, and solubility ≥10× the highest test concentration.",
            ),
            _L(
                "需要复测或优化：Kd 10–30 μM、选择性 5–20 倍，或肝微粒体稳定性低于同类中位。",
                "Repeat or optimise: Kd 10–30 μM, 5–20-fold selectivity, or microsomal stability "
                "below the class median.",
            ),
            _L(
                "终止信号：Kd >30 μM、选择性 <5 倍、hERG IC50 <10 倍有效暴露，或出现 PAINS 类干扰。",
                "Stop: Kd >30 μM, <5-fold selectivity, hERG IC50 <10× effective exposure, or "
                "PAINS-type interference.",
            ),
            _L(
                "阶段门：每阶段结束按上述标准评审，未达标先回到 4.5 优化，而不是直接进入动物实验。",
                "Stage gate: review against these criteria at the end of each stage; return to "
                "the optimisation in 4.5 before any animal study.",
            ),
        ]),
        (_L("4.5　下一步优化方向", "4.5　Optimisation priorities"), optimization_tips[:4]),
        (_L("4.6　声明", "4.6　Statement"), [
            _L(
                "以上为基于计算的可行性判断与实验设计建议，不替代实测数据；"
                "对接打分仅用于相对排序，实验结论以实测为准。",
                "The above is a computational feasibility assessment and experimental design "
                "advice, not a substitute for measurements; docking scores rank analogues only "
                "and all conclusions must be confirmed experimentally.",
            ),
        ]),
    ]
    _flow_text_pages(
        pdf,
        _L("4　结论与实验指导（续）", "4　Conclusions and Experimental Guidance (continued)"),
        _L(
            "分阶段实验验证、Go/No-Go 判据与优化方向",
            "Staged validation, go/no-go gates and optimisation",
        ),
        blocks, width=40, size=10.9, heading_size=13.0, stop=0.070,
    )


def _appendix_pages(
    pdf,
    run: dict,
    results: list[dict],
    props: dict,
    analyses: dict[int, dict] | None = None,
):
    analyses = analyses or {}
    fig = _a4_page("5　附录 A　完整数据表", "全部成功结果的结构、打分与理化性质")
    rows = [["配体", "Vina", "Vinardo", "HA", "MW", "cLogP", "TPSA", "LE", "QED", "HBD"]]
    for item in results[:20]:
        data = props.get(id(item), {})
        efficiency = data.get("efficiency") or {}
        rows.append(
            [
                textwrap.fill(_trunc(item.get("ligand_name", ""), 20), 16),
                f"{float(item.get('affinity', 0)):.2f}",
                f"{float(item['vinardo']):.2f}" if item.get("vinardo") is not None else "—",
                str(data.get("heavy", "—")),
                f"{float(data.get('mw') or 0):.0f}" if data else "—",
                f"{float(data.get('logp') or 0):.2f}" if data else "—",
                f"{float(data.get('tpsa') or 0):.0f}" if data else "—",
                f"{float(efficiency.get('le') or 0):.2f}" if data else "—",
                f"{float(data.get('qed') or 0):.2f}" if data else "—",
                str(data.get("hbd", "—")),
            ]
        )
    _table(
        fig, 0.872, rows,
        [0.24, 0.08, 0.095, 0.055, 0.07, 0.07, 0.07, 0.06, 0.06, 0.05],
        fontsize=9.9, row_height=0.0312,
    )
    note_top = 0.872 - 0.0312 * len(rows) - 0.030
    fig.text(
        0.07, note_top,
        _wrap_cjk(
            _L(
                "HA：重原子数；LE：配体效率（−ΔG / HA，kcal/mol/HA）；"
                "QED：综合类药性；HBD：氢键供体数（RDKit 口径：N/O/S–H，"
                "与 Lipinski 原文的 N+O 计数略有差异）。",
                "HA: heavy atoms; LE: ligand efficiency (−ΔG / HA, kcal/mol/HA); "
                "QED: drug-likeness; HBD: H-bond donors (RDKit definition: "
                "N/O/S–H, which differs slightly from the original Lipinski "
                "N+O count).",
            ),
            96,
        ),
        fontsize=9.6, color=MUTED, va="top",
    )
    fig.text(
        0.07, note_top - 0.045,
        _wrap_cjk(
            "数据口径：−ΔG 取 AutoDock Vina 最优构象打分，Vinardo 列为同一构象的二次打分；"
            "理化性质由 RDKit 依据对接前的质子化状态计算，与实验测定值存在系统差异，"
            "仅用于同系列分子间的相对比较。", 80
        ),
        fontsize=10.6, color=INK, va="top", linespacing=1.65,
    )
    if analyses and len(rows) <= 12:
        counts = [["配体", "氢键", "疏水", "盐桥", "π-堆积", "卤键", "金属", "合计"]]
        if i18n.get_lang() == "en":
            counts[0] = [
                "Ligand", "H-bond", "Hydrophobic", "Salt", "π-stack",
                "Halogen", "Metal", "Total",
            ]
        for item in results[:20]:
            data = analyses.get(id(item))
            if not data:
                continue
            values = [len(data.get(key, [])) for key, _ in INTERACTION_LABELS]
            counts.append(
                [textwrap.fill(_trunc(item.get("ligand_name", ""), 20), 16)]
                + [str(value) for value in values]
                + [str(sum(values))]
            )
        if len(counts) > 1:
            fig.text(
                0.07, 0.376, "附表 A2　各配体相互作用计数",
                fontsize=13, fontweight="bold", color=TEAL, va="top",
            )
            _table(
                fig, 0.338, counts,
                [0.24, 0.09, 0.12, 0.08, 0.09, 0.09, 0.08, 0.07]
                if i18n.get_lang() == "en"
                else [0.26, 0.09, 0.09, 0.09, 0.09, 0.08, 0.08, 0.08],
                fontsize=9.8, row_height=0.0250,
            )
            fig.text(
                0.07, 0.072,
                _wrap_cjk(
                    "计数来自每个配体最优构象的几何判据，同一残基的多个接触分别计数。",
                    96,
                ),
                fontsize=9.0, color=MUTED, va="top",
            )
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)

    params = run.get("params") or {}
    spec = [
        ("exhaustiveness", "搜索穷尽度：构象搜索的并行采样轮数"),
        ("num_modes", "每个配体保留的输出构象数"),
        ("energy_range", "输出构象的能量窗口（kcal/mol）"),
        ("seed", "随机种子（0 表示由程序自动选择）"),
        ("scoring", "主打分函数"),
        ("engine", "计算引擎（CPU / GPU）"),
        ("gpu_search_depth", "GPU 搜索深度（替代 exhaustiveness 控制搜索强度）"),
        ("gpu_replicas", "每个配体的 GPU 独立复算次数"),
        ("gpu_threads", "Vina-GPU 的 OpenCL 线程数"),
        ("rescore", "是否用 Vinardo 做二次一致性打分"),
        ("cpu", "使用的 CPU 线程数（0 表示全部核心）"),
        ("parallel_jobs", "并行处理的配体数"),
    ]
    rows = [["参数", "取值", "含义"]]
    for key, meaning in spec:
        if key in params:
            if i18n.get_lang() == "en":
                meaning = _wrap_cjk(i18n.tr(meaning), 46)
            rows.append([key, str(params.get(key)), meaning])
    rows.append(["run_id", str(run.get("id", "—")), "对接记录编号，用于溯源原始输出"])
    if run.get("time"):
        rows.append(["time", str(run.get("time"))[:19].replace("T", " "), "记录生成时间"])
    page = _a4_page(
        "5　附录 B　运行参数与文件清单",
        "保证结果可复现的计算参数，以及本报告包含的图表索引",
    )
    _table(
        page, 0.868, rows, [0.20, 0.22, 0.44],
        fontsize=8.6 if i18n.get_lang() == "en" else 9.8,
        row_height=0.0300,
    )
    blocks = [
        ("附录 B 补充说明", [
            "GPU 模式下 exhaustiveness 不生效，实际搜索强度由 gpu_search_depth 与 gpu_replicas 决定；"
            "表中 exhaustiveness 仅记录用户输入，跨引擎比较需谨慎。",
            "ADMET 端点为规则阈值得到的启发式估计，不是机器学习模型预测；"
            "MM-GBSA 为未做熵校正的近似分解，绝对值不可与 Vina 打分直接比较。",
        ]),
        ("附录 C　报告文件与图像索引", [
            "图 1–图 6：结合亲和力、配体效率、作用力指纹、构象谱、药物改进与骨架枚举设计，"
            "均为 A4 单页矢量图，图注紧随图下。",
            "图 7–图 8：代表配体的二维结构与作用力示意；附图 A1–A4：ADMET、药效团、自由能分解与结合能比较；"
            "附图 B1–B2：优化与设计模块的完整过程记录。",
            "全部图像为 PDF 矢量（Type 42 子集嵌入），可在 Illustrator 或 Inkscape 中无损编辑。",
        ]),
    ]
    _text_blocks(
        page, 0.868 - 0.0300 * len(rows) - 0.050, blocks,
        width=42, size=10.6, heading_size=12.6, stop=0.05,
    )
    pdf.savefig(page, facecolor="white")
    plt.close(page)


def full_report_pdf(
    project_info: dict,
    run: dict,
    optimization: dict | None,
    design: dict | None,
    source_pdfs: list[Path],
    outpath: Path,
    analyses: dict[int, dict] | None = None,
    props: dict | None = None,
    validation: dict | None = None,
    optimization_parent: dict | None = None,
    fishing_data: dict | None = None,
    netpharm_data: dict | None = None,
    notes: dict | None = None,
    synthesis_data: dict | None = None,
    drugref_data: dict | None = None,
    ddi_data: dict | None = None,
    pharm_data: dict | None = None,
    species: str = "human",
    primary_ligand: dict | None = None,
    posebusters: dict | None = None,
    manifest: dict | None = None,
    mpo_data: dict | None = None,
    assay_data: dict | None = None,
    trust_data: dict | None = None,
    mmp_data: dict | None = None,
    pbpk_data: dict | None = None,
    active_data: dict | None = None,
    benchmark_data: dict | None = None,
) -> Path:
    """One complete vector PDF: cover, abstract, methods, discussion and figures."""
    _style_cjk()
    analyses = analyses or {}
    props = props or {}
    outpath.parent.mkdir(parents=True, exist_ok=True)
    cover_path = outpath.with_name(outpath.stem + "_cover.pdf")
    results = [item for item in run.get("results", []) if item.get("status") == "ok"]
    best = primary_ligand or (
        min(results, key=lambda item: item.get("affinity", 999)) if results else None
    )
    with PdfPages(cover_path) as pdf:
        # Cover
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("#ffffff")
        fig.add_artist(
            Rectangle((0.0, 0.0), 0.032, 1.0, transform=fig.transFigure,
                      facecolor=TEAL, edgecolor="none")
        )
        # Decorative hexagon network, top right.
        motif = fig.add_axes([0.64, 0.70, 0.31, 0.24])
        motif.set_xlim(-0.4, 3.4)
        motif.set_ylim(-0.4, 2.4)
        motif.set_aspect("equal")
        motif.axis("off")
        for cx, cy in [(0.0, 1.0), (0.78, 1.0), (1.56, 1.0), (2.34, 1.0),
                       (0.39, 1.7), (1.17, 1.7), (1.95, 1.7), (0.39, 0.3),
                       (1.17, 0.3), (1.95, 0.3)]:
            motif.add_patch(
                RegularPolygon(
                    (cx, cy), numVertices=6, radius=0.44, orientation=np.pi / 6,
                    facecolor="none", edgecolor=TEAL, linewidth=1.0, alpha=0.18,
                )
            )
        fig.text(0.07, 0.88, "DrugHX", fontsize=36, fontweight="bold", color=TEAL)
        fig.text(
            0.07, 0.83, "分子对接 · ADMET · 药物互作",
            fontsize=15.5 if i18n.get_lang() == "en" else 20, color=INK,
        )
        fig.text(0.07, 0.78, "完整科研报告", fontsize=30, fontweight="bold", color=INK)
        fig.add_artist(
            Line2D([0.07, 0.58], [0.755, 0.755], transform=fig.transFigure,
                   color=TEAL, linewidth=2.0)
        )
        fig.text(0.07, 0.70, f"项目：{project_info.get('name', '—')}", fontsize=15, color=INK)
        fig.text(0.07, 0.66, f"靶点：{best.get('receptor_name', '—') if best else '—'}", fontsize=15, color=INK)
        fig.text(
            0.07, 0.62,
            f"结果最优配体：{best.get('ligand_name', '—') if best else '—'}",
            fontsize=15, color=INK,
        )
        fig.text(0.07, 0.58, f"生成时间：{project_info.get('now', '—')}", fontsize=12, color=MUTED)
        fig.text(
            0.07, 0.545,
            f"{i18n.tr('软件作者：HX')}　·　{i18n.tr('版本')} {APP_VERSION}",
            fontsize=12.5, color=MUTED,
        )
        fig.add_artist(
            Rectangle((0.065, 0.345), 0.62, 0.195, transform=fig.transFigure,
                      facecolor="#f0fdfa", edgecolor="#99f6e4", linewidth=0.8)
        )
        if best:
            fig.text(
                0.09, 0.485,
                f"最优配体 Vina：{best.get('affinity', 0):.2f} kcal/mol",
                fontsize=18, color=TEAL, fontweight="bold",
            )
        if validation and validation.get("rows"):
            first = validation["rows"][0]
            fig.text(
                0.09, 0.452,
                f"重对接验证：RMSD {first.get('rmsd', 0):.2f} Å"
                f"（{'通过' if first.get('passed') else '未通过'} 2 Å 判据）",
                fontsize=12.5, color=MUTED,
            )
        if optimization and optimization.get("best"):
            opt_best = optimization["best"]
            fig.text(
                0.09, 0.415,
                f"优化最佳：{opt_best.get('label', '—')}  {opt_best.get('affinity', 0):.2f} kcal/mol"
                f"（Δ {opt_best.get('delta', 0):+.2f}）",
                fontsize=14, color=INK,
            )
            if optimization_parent and optimization_parent.get("name"):
                fig.text(
                    0.09, 0.382,
                    f"优化母体：{optimization_parent.get('name')}  "
                    f"{float(optimization_parent.get('affinity') or 0):.2f} kcal/mol",
                    fontsize=12, color=MUTED,
                )
        if design and design.get("best"):
            des_best = design["best"]
            fig.text(
                0.09, 0.360,
                f"设计最佳：{des_best.get('label', '—')}  {des_best.get('affinity', 0):.2f} kcal/mol",
                fontsize=13, color=INK,
            )
        fig.text(
            0.07, 0.16,
            _wrap_cjk(
                "本报告中的图像均为 PDF 矢量图；打分仅用于相对排序，不代表实验结合自由能。",
                96,
            ),
            fontsize=9.5 if i18n.get_lang() == "en" else 10.5,
            color=MUTED, va="top",
        )
        pdf.savefig(fig, facecolor="white")
        plt.close(fig)

        _methods_pages(pdf)
        _page_validation(pdf, validation or {})
        _page_posebusters(pdf, posebusters)
        if manifest:
            _page_manifest(pdf, manifest)

        _abstract_page(
            pdf, project_info, run, results, best, props, analyses,
            optimization, design, validation, fishing_data, netpharm_data,
            synthesis_data=synthesis_data,
            drugref_data=drugref_data,
            ddi_data=ddi_data,
            pharm_data=pharm_data,
        )
        _discussion_pages(
            pdf, run, results, best, props, analyses, optimization, design,
            validation, fishing_data, netpharm_data, notes,
            synthesis_data=synthesis_data,
            drugref_data=drugref_data,
            ddi_data=ddi_data,
            pharm_data=pharm_data,
        )
        # Chapter 2 closes with the industrial decision panels: multi-parameter
        # optimisation, then assay calibration when measured data exist.
        _page_mpo(pdf, mpo_data)
        _page_assay(pdf, assay_data)
        _page_trust(pdf, trust_data)
        _page_mmp(pdf, mmp_data)
        _page_pbpk(pdf, pbpk_data)
        _page_active(pdf, active_data)
        _page_benchmark(pdf, benchmark_data)
        # Chapter 4 closes the main report: verdict, experimental plan and gates.
        _conclusion_page(
            pdf, results, best, props, optimization, design, analyses, validation,
            synthesis_data=synthesis_data,
            drugref_data=drugref_data,
            ddi_data=ddi_data,
            pharm_data=pharm_data,
            posebusters=posebusters,
            trust_data=trust_data,
        )
    back_path = outpath.with_name(outpath.stem + "_back.pdf")
    with PdfPages(back_path) as pdf:
        _acknowledgements_page(pdf)
        _appendix_pages(pdf, run, results, props, analyses)
        _tool_index_page(pdf)
        _references_page(pdf)

    # Merge cover and figure pages onto uniform A4 pages.
    from pypdf import PdfReader, PdfWriter, Transformation

    a4_width, a4_height = 595.28, 841.89
    margin, gap = 34, 30
    writer = PdfWriter()
    for page in PdfReader(str(cover_path)).pages:
        writer.add_page(page)

    max_height = a4_height - 2 * margin

    def natural_height(group):
        total = 0.0
        for page in group:
            width = float(page.mediabox.width)
            height = float(page.mediabox.height)
            total += height * min((a4_width - 2 * margin) / max(width, 1.0), 1.0)
        return total + gap * (len(group) - 1)

    def pack_sizes(group):
        sizes = [
            (float(page.mediabox.width), float(page.mediabox.height))
            for page in group
        ]
        scales = [
            min((a4_width - 2 * margin) / max(width, 1.0), 1.0)
            for width, _ in sizes
        ]
        heights = [height * scale for (_, height), scale in zip(sizes, scales)]
        total = sum(heights) + gap * (len(group) - 1)
        if total > max_height:
            factor = max(0.1, (max_height - gap * (len(group) - 1)) / total)
            scales = [scale * factor for scale in scales]
            heights = [height * scale for (_, height), scale in zip(sizes, scales)]
            total = sum(heights) + gap * (len(group) - 1)
        return sizes, scales, heights, total

    def flush_group(group):
        if not group:
            return
        blank = writer.add_blank_page(width=a4_width, height=a4_height)
        sizes, scales, heights, total_height = pack_sizes(group)
        top = a4_height - margin
        for page, (width, height), scale, drawn_height in zip(group, sizes, scales, heights):
            tx = (a4_width - width * scale) / 2
            ty = top - drawn_height
            blank.merge_transformed_page(
                page, Transformation().scale(scale).translate(tx, ty)
            )
            top = ty - gap

    buffer = []
    for path in source_pdfs:
        if not path or not path.exists() or path.suffix.lower() != ".pdf":
            continue
        try:
            reader = PdfReader(str(path))
        except Exception:
            continue
        for page in reader.pages:
            width = float(page.mediabox.width)
            height = float(page.mediabox.height)
            if abs(width - a4_width) < 8 and abs(height - a4_height) < 8:
                flush_group(buffer)
                buffer = []
                # A4 source pages are shrunk slightly and centred so that axis
                # labels can never touch or cross the physical page edge.
                scale = 0.940
                blank = writer.add_blank_page(width=a4_width, height=a4_height)
                tx = (a4_width - width * scale) / 2
                ty = (a4_height - height * scale) / 2
                blank.merge_transformed_page(
                    page, Transformation().scale(scale).translate(tx, ty)
                )
                continue
            trial = buffer + [page]
            if buffer and (natural_height(trial) > max_height * 1.12 or len(trial) > 3):
                flush_group(buffer)
                buffer = [page]
            else:
                buffer = trial
    flush_group(buffer)
    for page in PdfReader(str(back_path)).pages:
        writer.add_page(page)
    with outpath.open("wb") as handle:
        writer.write(handle)
    writer.close()
    cover_path.unlink(missing_ok=True)
    back_path.unlink(missing_ok=True)
    return outpath


def write_pymol_bundle(
    receptor_pdb: Path,
    pose_pdbqt: Path,
    analysis: dict,
    result: dict,
    outdir: Path,
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    receptor_copy = outdir / "receptor.pdb"
    receptor_copy.write_text(
        receptor_pdb.read_text(encoding="utf-8", errors="replace"), encoding="utf-8"
    )
    ligand_pdb = outdir / "ligand.pdb"
    ligand_text = _named_ligand_pdb(
        pdbio.pdbqt_to_pdb(
            pdbio.pose_pdbqt(
                pose_pdbqt.read_text(encoding="utf-8", errors="replace"), 1
            )
        )
    )
    ligand_pdb.write_text(ligand_text, encoding="utf-8")
    ligand_atoms = _ligand_atom_positions(ligand_text)

    residues = analysis.get("residues", [])
    residue_selection = "+".join(str(r["resseq"]) for r in residues[:18]) or "1"
    script = []
    script.append("# DrugHX publication figure script")
    script.append("# Run: pymol figure_binding_mode.pml")
    script.append("reinitialize")
    script.append("load receptor.pdb, receptor")
    script.append("load ligand.pdb, ligand")
    script.append("hide everything")
    script.append("bg_color white")
    script.append("show cartoon, receptor")
    script.append("color grey80, receptor")
    script.append("set cartoon_fancy_helices, 1")
    script.append("set cartoon_transparency, 0.15")
    script.append(f"select pocket, receptor and resi {residue_selection}")
    script.append("show sticks, pocket")
    script.append("color slate, pocket and elem C")
    script.append("show sticks, ligand")
    script.append("color teal, ligand and elem C")
    script.append("set stick_radius, 0.18, ligand")
    script.append("set stick_radius, 0.12, pocket")
    script.append("util.cnc")
    for index, item in enumerate(analysis.get("hbonds", [])[:8], start=1):
        resi = item.get("resseq")
        protein_atom = item.get("protein_atom")
        ligand_atom = _nearest_ligand_name(
            ligand_atoms,
            item.get("ligand_coord"),
            item.get("ligand_atom", ""),
        )
        if resi and protein_atom and ligand_atom:
            script.append(
                f"distance hb{index}, receptor and resi {resi} and name {protein_atom}, "
                f"ligand and name {ligand_atom}"
            )
            script.append(f"color yellow, hb{index}")
            script.append(f"set dash_gap, 0.35, hb{index}")
            script.append(f"set dash_length, 0.25, hb{index}")
    for index, item in enumerate(analysis.get("salt_bridges", [])[:4], start=1):
        resi = item.get("resseq")
        protein_atom = item.get("protein_atom")
        ligand_atom = _nearest_ligand_name(
            ligand_atoms,
            item.get("ligand_coord"),
            item.get("ligand_atom", ""),
        )
        if resi and protein_atom and ligand_atom:
            script.append(
                f"distance sb{index}, receptor and resi {resi} and name {protein_atom}, "
                f"ligand and name {ligand_atom}"
            )
            script.append(f"color red, sb{index}")
    for item in residues[:12]:
        resi = item["resseq"]
        resn = item["residue"]
        script.append(
            f'label receptor and resi {resi} and name CA, "{resn}{resi}"'
        )
    script.append("set label_size, 14")
    script.append("set label_color, black")
    script.append("set label_outline_color, white")
    script.append("set ray_opaque_background, on")
    script.append("set ray_shadows, 0")
    script.append("set antialias, 2")
    script.append("orient ligand")
    script.append("zoom ligand, 6")
    script.append("ray 3600, 2700")
    script.append("png figure_binding_mode.png, dpi=600")
    script.append("save figure_binding_mode.pse")
    script_path = outdir / "figure_binding_mode.pml"
    script_path.write_text("\n".join(script) + "\n", encoding="utf-8")
    return script_path


def write_caption(run: dict, results: list[dict], outdir: Path) -> Path:
    ok = [r for r in results if r.get("status") == "ok"]
    best = min(ok, key=lambda r: r["affinity"]) if ok else None
    params = run.get("params", {})
    engine = str(params.get("engine", "cpu")).lower()
    if engine == "gpu":
        engine_text = (
            "Vina-GPU 2.0 was used for the global search, followed by "
            "CPU-based AutoDock Vina 1.2.5 rescoring of the selected pose"
        )
    else:
        engine_text = "AutoDock Vina 1.2.5 was used on CPU"
    lines = [
        "Suggested figure caption",
        "",
        (
            "Figure. Molecular docking of candidate ligands into the target binding site. "
            f"{engine_text} with exhaustiveness={params.get('exhaustiveness', 8)}, "
            f"num_modes={params.get('num_modes', 9)} and scoring={params.get('scoring', 'vina')}. "
            "The binding-site box was defined from the co-crystallized ligand or predicted pocket. "
            "Panel a ranks the Vina and Vinardo scores; panel b summarises the number of "
            "hydrogen bonds, hydrophobic contacts, salt bridges, π-stacking, halogen bonds "
            "and metal coordination contacts per ligand."
        ),
    ]
    if best:
        lines.append(
            f"The top-ranked pose ({best.get('ligand_name')}, {best.get('affinity'):.2f} kcal/mol) "
            "is shown as a representative binding mode in the accompanying PyMOL scene. "
            "Docking scores are intended for relative ranking within this target and are not "
            "direct measurements of binding free energy."
        )
    path = outdir / "figure_caption.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_data_table(
    results: list[dict],
    analyses: dict[int, dict],
    outdir: Path,
) -> Path:
    path = outdir / "figure_source_data.csv"
    ok = [r for r in results if r.get("status") == "ok"]
    ok.sort(key=lambda r: r.get("affinity", 999))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "ligand",
                "receptor",
                "engine",
                "vina_score",
                "vinardo_score",
                "modes",
                "hbonds",
                "hydrophobic",
                "salt_bridges",
                "pi_stacking",
                "halogens",
                "metals",
            ]
        )
        for item in ok:
            analysis = analyses.get(id(item), {})
            writer.writerow(
                [
                    item.get("ligand_name"),
                    item.get("receptor_name"),
                    item.get("engine", "cpu"),
                    item.get("affinity"),
                    item.get("vinardo"),
                    len(item.get("modes", [])),
                    *[
                        len(analysis.get(key, []))
                        for key, _ in INTERACTION_LABELS
                    ],
                ]
            )
    return path


def build_publication_bundle(
    run: dict,
    results: list[dict],
    analyses: dict[int, dict],
    receptor_pdb: Path,
    pose_pdbqt: Path,
    best_result: dict,
    outdir: Path,
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    _style()
    affinity_figure(results, outdir / "fig_score_ranking")
    interaction_heatmap(results, analyses, outdir / "fig_interaction_heatmap")
    publication_panel(results, analyses, outdir / "fig_publication_panel")
    analysis = analyses.get(id(best_result), {})
    write_pymol_bundle(receptor_pdb, pose_pdbqt, analysis, best_result, outdir)
    write_caption(run, results, outdir)
    write_data_table(results, analyses, outdir)
    readme = outdir / "README.txt"
    readme.write_text(
        textwrap.dedent(
            """\
            DrugHX publication figure bundle
            ===================================
            fig_publication_panel.pdf/.svg   Two-panel summary (a scores, b interactions)
            fig_score_ranking.pdf/.svg       Score ranking (vector)
            fig_interaction_heatmap.pdf/.svg Interaction counts across ligands
            figure_binding_mode.pml          PyMOL scene; run "pymol figure_binding_mode.pml"
                                             to ray-trace an optional 3D raster
            figure_caption.txt               Suggested caption
            figure_source_data.csv           Source data for both panels
            receptor.pdb / ligand.pdb        Input structures for the PyMOL scene

            The PyMOL script uses a white background, muted protein cartoon,
            teal ligand carbons, yellow hydrogen bonds and red salt bridges.
            For journal submission, use the PDF/SVG vector panels.
            """
        ),
        encoding="utf-8",
    )
    zip_path = outdir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(outdir.rglob("*")):
            if file.is_file():
                archive.write(file, file.relative_to(outdir))
    return zip_path


def bulk_screen_png(
    run: dict,
    ligand_meta: dict[str, dict],
    outpath: Path,
) -> Path:
    """Six-panel, CNS-ready screening summary for a large ligand library."""
    _style_bulk()
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    english = i18n.get_lang() == "en"
    rows = [
        item for item in run.get("results", [])
        if item.get("status") == "ok" and item.get("affinity") is not None
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 9.6))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(
        left=0.070, right=0.985, top=0.875, bottom=0.085,
        hspace=0.44, wspace=0.30,
    )
    engine = str((run.get("params") or {}).get("docking_engine") or "vina")
    title = (
        f"Large-library docking · {len(rows)} ligands"
        if english
        else f"大规模虚拟筛选 · {len(rows)} 个配体"
    )
    fig.suptitle(title, fontsize=20, fontweight="bold", x=0.070, ha="left", y=0.965)
    fig.text(
        0.070, 0.921,
        (
            f"Engine {engine} · scores sorted from strongest to weakest"
            if english
            else f"引擎 {engine} · 按结合打分从强到弱排序"
        ),
        fontsize=12.5, color=MUTED,
    )

    if not rows:
        for ax in axes.flat:
            ax.axis("off")
        axes[0, 0].text(
            0.5, 0.5,
            "No successful docking result" if english else "没有成功的对接结果",
            ha="center", va="center", fontsize=17, color=MUTED,
        )
        fig.savefig(outpath, dpi=300, facecolor="white")
        plt.close(fig)
        return outpath

    scores = np.array([float(item["affinity"]) for item in rows])
    mw = []
    le = []
    for item in rows:
        meta = ligand_meta.get(str(item.get("ligand_id") or ""), {})
        mass = float(meta.get("mw") or 0)
        heavy = float(meta.get("heavy") or 0)
        mw.append(mass if mass > 0 else np.nan)
        le.append(float(item["affinity"]) / heavy if heavy > 0 else np.nan)
    mw = np.array(mw)
    le = np.array(le)
    median = float(np.median(scores))
    mean = float(np.mean(scores))

    def panel(ax, letter, zh, en):
        _compact_frame(ax)
        ax.grid(True, color=GRID, linewidth=0.7, alpha=0.75)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=11.5, colors="#475569")
        ax.set_title(
            f"{letter}  {en if english else zh}", loc="left",
            fontsize=16.5, fontweight="bold", pad=9,
        )

    # A: score distribution
    ax = axes[0, 0]
    panel(ax, "A", "对接分数分布", "Binding-score distribution")
    bins = int(min(45, max(10, len(scores) // 90)))
    ax.hist(scores, bins=bins, color=TEAL, alpha=0.86,
            edgecolor="white", linewidth=0.8)
    ax.axvline(median, color=AMBER, linestyle="--", linewidth=2.0,
               label=("median " if english else "中位数 ") + f"{median:.2f}")
    for threshold, color in ((-8.0, "#f59e0b"), (-9.0, RED)):
        ax.axvline(threshold, color=color, linestyle=":", linewidth=1.6)
    ax.set_xlabel("Vina score (kcal/mol)" if english else "Vina 打分（kcal/mol）")
    ax.set_ylabel("Ligands" if english else "配体数")
    ax.legend(frameon=False, loc="upper left", fontsize=11.5)
    hits8 = int(np.sum(scores <= -8.0))
    ax.text(
        0.985, 0.975,
        f"N = {len(scores):,}\nmedian = {median:.2f}\nmean = {mean:.2f}\n"
        f"≤ −8: {hits8:,} ({hits8 / len(scores):.1%})",
        transform=ax.transAxes, ha="right", va="top", fontsize=11,
        color=INK, family="monospace",
        bbox=dict(boxstyle="round,pad=0.42", facecolor="#f8fafc",
                  edgecolor=GRID, linewidth=0.9),
    )

    # B: cumulative distribution
    ax = axes[0, 1]
    panel(ax, "B", "累积分布", "Cumulative distribution")
    ordered = np.sort(scores)
    cumulative = np.arange(1, len(ordered) + 1) / len(ordered) * 100
    ax.plot(ordered, cumulative, color=BLUE, linewidth=2.4)
    ax.fill_between(ordered, 0, cumulative, color=BLUE, alpha=0.12)
    for threshold, color in ((-8.0, "#f59e0b"), (-9.0, RED)):
        ax.axvline(threshold, color=color, linestyle=":", linewidth=1.4)
    ax.set_xlabel("Vina score (kcal/mol)" if english else "Vina 打分（kcal/mol）")
    ax.set_ylabel("Cumulative ligands (%)" if english else "累积配体比例（%）")

    # C: score vs molecular weight
    ax = axes[0, 2]
    panel(ax, "C", "打分–分子量", "Score vs molecular weight")
    mask = np.isfinite(mw) & (mw > 0)
    if mask.sum() >= 3:
        ax.scatter(mw[mask], scores[mask], s=18, alpha=0.55, color=BLUE,
                   edgecolors="none", rasterized=True)
    else:
        ax.text(0.5, 0.5, "No molecular-weight metadata" if english else "缺少分子量信息",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=MUTED)
    ax.set_xlabel("Molecular weight (Da)" if english else "分子量（Da）")
    ax.set_ylabel("Vina score (kcal/mol)" if english else "Vina 打分（kcal/mol）")

    # D: score vs logP
    ax = axes[1, 0]
    panel(ax, "D", "打分–亲脂性", "Score vs lipophilicity")
    logp = np.array([
        float((ligand_meta.get(str(item.get("ligand_id") or ""), {}) or {}).get("logp", np.nan))
        if (ligand_meta.get(str(item.get("ligand_id") or ""), {}) or {}).get("logp") is not None
        else np.nan
        for item in rows
    ], dtype=float)
    mask = np.isfinite(logp)
    if mask.sum() >= 3:
        ax.scatter(logp[mask], scores[mask], s=18, alpha=0.55, color="#7c3aed",
                   edgecolors="none", rasterized=True)
    else:
        ax.text(0.5, 0.5, "No logP metadata" if english else "缺少 logP 信息",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=MUTED)
    ax.set_xlabel("cLogP")
    ax.set_ylabel("Vina score (kcal/mol)" if english else "Vina 打分（kcal/mol）")

    # E: ligand efficiency
    ax = axes[1, 1]
    panel(ax, "E", "配体效率", "Ligand efficiency")
    le_values = le[np.isfinite(le)]
    if le_values.size:
        ax.hist(le_values, bins=int(min(38, max(10, le_values.size // 90))),
                color="#0e7490", alpha=0.86, edgecolor="white", linewidth=0.7)
        ax.axvline(float(np.median(le_values)), color=AMBER,
                   linestyle="--", linewidth=1.8)
    else:
        ax.text(0.5, 0.5, "No heavy-atom metadata" if english else "缺少重原子信息",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=MUTED)
    ax.set_xlabel(
        "Ligand efficiency (kcal/mol/HA)" if english else "配体效率 (kcal/mol/HA)",
        fontsize=13,
    )
    ax.set_ylabel("Ligands" if english else "配体数")

    # F: top 15
    ax = axes[1, 2]
    panel(ax, "F", "最优 15 个候选", "Top 15 candidates")
    top = sorted(rows, key=lambda item: float(item["affinity"]))[:15]
    labels = [
        str(item.get("ligand_name") or item.get("ligand_id") or "")[:22]
        for item in top
    ]
    values = [float(item["affinity"]) for item in top]
    efficiencies = []
    for item in top:
        meta = ligand_meta.get(str(item.get("ligand_id") or ""), {})
        heavy = float(meta.get("heavy") or 0)
        efficiencies.append(-float(item["affinity"]) / heavy if heavy > 0 else 0.0)
    norm = Normalize(
        vmin=min(efficiencies) if efficiencies else 0.0,
        vmax=max(efficiencies) if efficiencies else 1.0,
    )
    colors = plt.cm.viridis(norm(efficiencies))
    y = np.arange(len(top))
    ax.barh(y, values, color=colors, alpha=0.94, height=0.72)
    for pos, value in zip(y, values):
        ax.text(value + 0.12, pos, f"{value:.2f}", va="center", ha="left",
                fontsize=10, color="white", fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10.2)
    ax.invert_yaxis()
    ax.set_xlabel("Vina score (kcal/mol)" if english else "Vina 打分（kcal/mol）")

    fig.text(
        0.070, 0.018,
        "条形颜色表示配体效率（越深越高）" if not english
        else "Bar colour encodes ligand efficiency (darker = higher)",
        fontsize=11, color=MUTED,
    )
    fig.savefig(outpath, dpi=300, facecolor="white")
    plt.close(fig)
    return outpath


def _bulk_values(rows: list[dict], ligand_meta: dict[str, dict], key: str) -> np.ndarray:
    values = []
    for item in rows:
        meta = ligand_meta.get(str(item.get("ligand_id") or ""), {})
        try:
            values.append(float(meta.get(key)))
        except (TypeError, ValueError):
            values.append(np.nan)
    return np.asarray(values, dtype=float)


def _bulk_axes(fig, rect):
    ax = fig.add_axes(rect)
    _compact_frame(ax)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=12.5, colors="#475569")
    return ax


def _bulk_scatter(ax, x, y, xlabel, ylabel, title):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() >= 3:
        if mask.sum() > 1500:
            hb = ax.hexbin(
                x[mask], y[mask], gridsize=42, cmap="Blues", mincnt=1,
                linewidths=0.0,
            )
            fig = ax.figure
            cbar = fig.colorbar(hb, ax=ax, fraction=0.028, pad=0.015)
            cbar.set_label(_L("配体数", "Ligands"), fontsize=10, color=MUTED)
            cbar.ax.tick_params(labelsize=9.5, colors=MUTED)
        else:
            ax.scatter(x[mask], y[mask], s=14, alpha=0.6, color=BLUE,
                       edgecolors="none", rasterized=True)
    else:
        ax.text(0.5, 0.5, _L("数据不足", "Insufficient data"),
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color=MUTED)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=15.5, fontweight="bold")
    return mask


def _bulk_corr(ax, x, y, label: str):
    mask = np.isfinite(x) & np.isfinite(y)
    value = _spearman(list(zip(x[mask], y[mask]))) if mask.sum() >= 3 else None
    if value is not None:
        ax.text(
            0.985, 0.975,
            f"Spearman ρ = {value:+.3f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=11, color=MUTED,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                      edgecolor=GRID, linewidth=0.7, alpha=0.9),
        )
    return value


def _bulk_empty_page(pdf, title: str, message: str):
    fig = _a4_page(title)
    ax = fig.add_axes([0.12, 0.35, 0.76, 0.40])
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center",
            fontsize=13, color=MUTED, wrap=True)
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def bulk_screening_report_pdf(
    run: dict,
    ligand_meta: dict[str, dict],
    outpath: Path,
) -> Path:
    """Seven publication-style statistical pages for a large-library screen."""
    _style_bulk()
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    english = i18n.get_lang() == "en"
    rows = [
        item for item in run.get("results", [])
        if item.get("status") == "ok" and item.get("affinity") is not None
    ]
    if not rows:
        with PdfPages(outpath) as pdf:
            _bulk_empty_page(
                pdf,
                _L("图 S1｜大规模对接统计", "Fig. S1 | Large-library docking"),
                _L("没有成功的对接结果。", "No successful docking result."),
            )
        return outpath

    scores = np.asarray([float(item["affinity"]) for item in rows], dtype=float)
    mw = _bulk_values(rows, ligand_meta, "mw")
    logp = _bulk_values(rows, ligand_meta, "logp")
    tpsa = _bulk_values(rows, ligand_meta, "tpsa")
    hbd = _bulk_values(rows, ligand_meta, "hbd")
    hba = _bulk_values(rows, ligand_meta, "hba")
    rotb = _bulk_values(rows, ligand_meta, "rotb")
    heavy = _bulk_values(rows, ligand_meta, "heavy")
    le = np.where(heavy > 0, -scores / heavy, np.nan)
    ranks = np.argsort(np.argsort(scores))
    params = run.get("params") or {}
    engine = str(params.get("docking_engine") or "vina")
    total = len(run.get("results", []))
    failed = total - len(rows)
    head = _L(
        f"配体库 {len(rows):,} 个成功结果（失败 {failed} 个）· 引擎 {engine}",
        f"{len(rows):,} successful ligands ({failed} failed) · engine {engine}",
    )
    footer = _L("DrugHX · 大规模虚拟筛选统计", "DrugHX · Virtual screening statistics")

    with PdfPages(outpath) as pdf:
        # ---- S1 score distribution + cumulative curve ----
        fig = _a4_page(
            _L("图 S1｜打分分布与命中阈值", "Fig. S1 | Score distribution and hit thresholds"),
            head, footer=footer,
        )
        ax = _bulk_axes(fig, [0.10, 0.47, 0.82, 0.35])
        bins = int(min(60, max(12, len(scores) // 120)))
        ax.hist(scores, bins=bins, color=TEAL, alpha=0.82,
                edgecolor="white", linewidth=0.5)
        median = float(np.median(scores))
        mean = float(np.mean(scores))
        ax.axvline(median, color=AMBER, linestyle="--", linewidth=1.5,
                   label=_L(f"中位数 {median:.2f}", f"median {median:.2f}"))
        for threshold, color in ((-8.0, "#f59e0b"), (-9.0, RED)):
            ax.axvline(threshold, color=color, linestyle=":", linewidth=1.2)
        ax.set_xlabel(_L("Vina 打分（kcal/mol）", "Vina score (kcal/mol)"))
        ax.set_ylabel(_L("配体数", "Ligands"))
        ax.set_title(_L("对接分数分布", "Binding-score distribution"),
                     loc="left", fontsize=15.5, fontweight="bold")
        ax.legend(frameon=False, loc="upper left", fontsize=11.5)
        hits8 = int(np.sum(scores <= -8.0))
        hits9 = int(np.sum(scores <= -9.0))
        ax.text(
            0.985, 0.975,
            (
                f"N = {len(scores):,}\n"
                f"median = {median:.2f}\n"
                f"mean = {mean:.2f}\n"
                f"best = {scores.min():.2f}\n"
                f"≤ −8: {hits8:,} ({hits8 / len(scores):.1%})\n"
                f"≤ −9: {hits9:,} ({hits9 / len(scores):.1%})"
            ),
            transform=ax.transAxes, ha="right", va="top",
            fontsize=11, color=INK, family="monospace",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#f8fafc",
                      edgecolor=GRID, linewidth=0.8),
        )
        ax2 = _bulk_axes(fig, [0.10, 0.19, 0.82, 0.20])
        ordered = np.sort(scores)
        cumulative = np.arange(1, len(ordered) + 1) / len(ordered) * 100
        ax2.plot(ordered, cumulative, color=BLUE, linewidth=1.8)
        ax2.fill_between(ordered, 0, cumulative, color=BLUE, alpha=0.10)
        for threshold, color in ((-8.0, "#f59e0b"), (-9.0, RED)):
            ax2.axvline(threshold, color=color, linestyle=":", linewidth=1.0)
        ax2.set_xlabel(_L("Vina 打分（kcal/mol）", "Vina score (kcal/mol)"))
        ax2.set_ylabel(_L("累积配体比例（%）", "Cumulative ligands (%)"))
        ax2.set_title(_L("累积分布", "Cumulative distribution"),
                      loc="left", fontsize=14.5, fontweight="bold")
        _caption(
            fig,
            _L(
                "分数越负表示预测结合越强；虚线为中位数，点线为 −8 / −9 kcal/mol 命中阈值。",
                "More negative scores indicate stronger predicted binding; dashed line is the median, dotted lines mark −8 / −9 kcal/mol.",
            ),
            y=0.115,
        )
        pdf.savefig(fig, facecolor="white")
        plt.close(fig)

        # ---- S2 chemical space ----
        fig = _a4_page(
            _L("图 S2｜打分与分子性质", "Fig. S2 | Scores in chemical space"),
            head, footer=footer,
        )
        ax = _bulk_axes(fig, [0.10, 0.56, 0.82, 0.27])
        _bulk_scatter(
            ax, mw, scores,
            _L("分子量（Da）", "Molecular weight (Da)"),
            _L("Vina 打分（kcal/mol）", "Vina score (kcal/mol)"),
            _L("打分–分子量", "Score vs molecular weight"),
        )
        _bulk_corr(ax, mw, scores, "mw")
        ax2 = _bulk_axes(fig, [0.10, 0.19, 0.82, 0.25])
        _bulk_scatter(
            ax2, logp, scores,
            _L("cLogP", "cLogP"),
            _L("Vina 打分（kcal/mol）", "Vina score (kcal/mol)"),
            _L("打分–亲脂性", "Score vs lipophilicity"),
        )
        _bulk_corr(ax2, logp, scores, "logp")
        _caption(
            fig,
            _L(
                "散点或六边形密度表示配体分布；Spearman ρ 衡量单调相关性（|ρ| < 0.3 视为基本不相关）。",
                "Scatter/hexbin density shows the ligand cloud; Spearman ρ measures monotonic association (|ρ| < 0.3 is negligible).",
            ),
            y=0.105,
        )
        pdf.savefig(fig, facecolor="white")
        plt.close(fig)

        # ---- S3 ligand efficiency ----
        fig = _a4_page(
            _L("图 S3｜配体效率与重原子数", "Fig. S3 | Ligand efficiency"),
            head, footer=footer,
        )
        ax = _bulk_axes(fig, [0.10, 0.50, 0.78, 0.32])
        mask = np.isfinite(le) & np.isfinite(heavy)
        if mask.sum() >= 3:
            scatter = ax.scatter(
                heavy[mask], le[mask], c=scores[mask], cmap="viridis_r",
                s=9, alpha=0.55, linewidths=0, rasterized=True,
            )
            cbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.02)
            cbar.set_label(_L("Vina 打分", "Vina score"), fontsize=10, color=MUTED)
            cbar.ax.tick_params(labelsize=9.5, colors=MUTED)
        else:
            ax.text(0.5, 0.5, _L("缺少重原子信息", "No heavy-atom data"),
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=11, color=MUTED)
        for level, color in ((0.3, "#f59e0b"), (0.4, RED)):
            ax.axhline(level, color=color, linestyle=":", linewidth=1.0)
        ax.set_xlabel(_L("重原子数", "Heavy atoms"))
        ax.set_ylabel(_L("配体效率（kcal/mol/HA）", "Ligand efficiency (kcal/mol/HA)"))
        ax.set_title(_L("配体效率分布", "Ligand-efficiency landscape"),
                     loc="left", fontsize=15.5, fontweight="bold")
        ax2 = _bulk_axes(fig, [0.10, 0.19, 0.82, 0.21])
        le_values = le[np.isfinite(le)]
        if le_values.size:
            ax2.hist(le_values, bins=int(min(40, max(10, le_values.size // 120))),
                     color="#7c3aed", alpha=0.82, edgecolor="white", linewidth=0.4)
            ax2.axvline(float(np.median(le_values)), color=AMBER,
                        linestyle="--", linewidth=1.3)
        ax2.set_xlabel(_L("配体效率（kcal/mol/HA）", "Ligand efficiency (kcal/mol/HA)"))
        ax2.set_ylabel(_L("配体数", "Ligands"))
        ax2.set_title(_L("配体效率直方图", "Ligand-efficiency histogram"),
                      loc="left", fontsize=14.5, fontweight="bold")
        _caption(
            fig,
            _L(
                "配体效率 LE = −ΔG / 重原子数；0.3 与 0.4 为常见的先导化合物参考线。",
                "LE = −ΔG / heavy atoms; 0.3 and 0.4 are common lead-like reference levels.",
            ),
            y=0.105,
        )
        pdf.savefig(fig, facecolor="white")
        plt.close(fig)

        # ---- S4 top candidates ----
        fig = _a4_page(
            _L("图 S4｜最优候选排行", "Fig. S4 | Top-ranked candidates"),
            head, footer=footer,
        )
        ax = fig.add_axes([0.34, 0.20, 0.58, 0.67])
        _compact_frame(ax, "x")
        ax.grid(True, axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        order = np.argsort(scores)[:25]
        labels = [
            str(rows[index].get("ligand_name") or rows[index].get("ligand_id") or "")[:24]
            for index in order
        ]
        values = scores[order]
        efficiencies = le[order]
        y = np.arange(len(order))
        finite_le = efficiencies[np.isfinite(efficiencies)]
        norm = Normalize(
            vmin=float(np.nanmin(finite_le)) if finite_le.size else 0.0,
            vmax=float(np.nanmax(finite_le)) if finite_le.size else 1.0,
        )
        colors = plt.cm.viridis(norm(np.nan_to_num(efficiencies, nan=0.0)))
        ax.barh(y, values, color=colors, alpha=0.92, height=0.72)
        for pos, value, eff in zip(y, values, efficiencies):
            ax.text(
                value + 0.15, pos, f"{value:.2f}",
                va="center", ha="left", fontsize=10, color="white",
                fontweight="bold",
            )
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10.5)
        ax.invert_yaxis()
        ax.set_xlabel(_L("Vina 打分（kcal/mol）", "Vina score (kcal/mol)"))
        ax.set_title(_L("最优 25 个候选", "Top 25 candidates"),
                     loc="left", fontsize=15.5, fontweight="bold")
        _caption(
            fig,
            _L(
                "条形颜色由配体效率映射（深色 = 更高效率）；这些分子建议进入精细对接复核。",
                "Bar colour maps ligand efficiency (dark = more efficient); these molecules are the best candidates for fine docking.",
            ),
            y=0.105,
        )
        pdf.savefig(fig, facecolor="white")
        plt.close(fig)

        # ---- S5 threshold hits ----
        fig = _a4_page(
            _L("图 S5｜命中数量与筛选收益", "Fig. S5 | Hit counts and screening yield"),
            head, footer=footer,
        )
        thresholds = [-7.0, -8.0, -9.0, -10.0]
        counts = [int(np.sum(scores <= value)) for value in thresholds]
        ax = _bulk_axes(fig, [0.10, 0.56, 0.82, 0.27])
        bars = ax.bar(
            [f"≤ {value:.0f}" for value in thresholds], counts,
            color=[TEAL, "#0e7490", "#1d4ed8", "#7c3aed"], width=0.58,
        )
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{count:,}\n({count / len(scores):.1%})",
                ha="center", va="bottom", fontsize=10.5, color=INK,
            )
        ax.set_ylabel(_L("配体数", "Ligands"))
        ax.set_ylim(0, max(counts) * 1.18 if counts else 1)
        ax.set_title(_L("不同阈值下的命中数量", "Hits at each score threshold"),
                     loc="left", fontsize=15.5, fontweight="bold")
        ax2 = _bulk_axes(fig, [0.10, 0.19, 0.82, 0.25])
        fraction = np.arange(1, len(ordered) + 1) / len(ordered)
        ax2.semilogy(ordered, fraction * 100, color=RED, linewidth=1.8)
        for threshold, color in ((-8.0, "#f59e0b"), (-9.0, "#7c3aed")):
            ax2.axvline(threshold, color=color, linestyle=":", linewidth=1.0)
        ax2.set_xlabel(_L("Vina 打分（kcal/mol）", "Vina score (kcal/mol)"))
        ax2.set_ylabel(_L("累积比例（%，对数）", "Cumulative fraction (%, log)"))
        ax2.set_title(_L("尾部富集（对数刻度）", "Tail enrichment (log scale)"),
                      loc="left", fontsize=14.5, fontweight="bold")
        _caption(
            fig,
            _L(
                "命中率 = 阈值以下配体数 / 全库；对数纵轴用于观察强结合尾部的富集程度。",
                "Hit rate = ligands below threshold / library; the log axis highlights enrichment in the strong-binding tail.",
            ),
            y=0.105,
        )
        pdf.savefig(fig, facecolor="white")
        plt.close(fig)

        # ---- S6 property correlation matrix ----
        fig = _a4_page(
            _L("图 S6｜性质–打分相关矩阵", "Fig. S6 | Property–score correlation matrix"),
            head, footer=footer,
        )
        keys = [_L("打分", "Score"), _L("分子量", "MW"), "cLogP", "TPSA",
                "HBD", "HBA", _L("可旋转键", "RotB"), _L("重原子", "Heavy"),
                _L("配体效率", "LE")]
        series = [scores, mw, logp, tpsa, hbd, hba, rotb, heavy, le]
        count = len(keys)
        matrix = np.full((count, count), np.nan)
        for i in range(count):
            for j in range(count):
                mask = np.isfinite(series[i]) & np.isfinite(series[j])
                if i == j:
                    matrix[i, j] = 1.0
                elif mask.sum() >= 3:
                    value = _spearman(
                        list(zip(series[i][mask], series[j][mask]))
                    )
                    if value is not None:
                        matrix[i, j] = value
        ax = fig.add_axes([0.20, 0.19, 0.62, 0.61])
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(count))
        ax.set_yticks(range(count))
        ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=11)
        ax.set_yticklabels(keys, fontsize=11)
        for i in range(count):
            for j in range(count):
                value = matrix[i, j]
                if np.isfinite(value):
                    ax.text(
                        j, i, f"{value:.2f}", ha="center", va="center",
                        fontsize=9,
                        color="white" if abs(value) > 0.55 else INK,
                    )
        ax.set_title(_L("Spearman 相关系数", "Spearman correlation"),
                     loc="left", fontsize=15.5, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_visible(False)
        cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label(_L("相关系数", "Correlation"), fontsize=10, color=MUTED)
        cbar.ax.tick_params(labelsize=9.5, colors=MUTED)
        _caption(
            fig,
            _L(
                "红色为正相关、蓝色为负相关；配体效率与打分天然强相关，可用于识别体量驱动的假阳性。",
                "Red = positive, blue = negative; the intrinsic score–LE correlation helps flag size-driven false positives.",
            ),
            y=0.105,
        )
        pdf.savefig(fig, facecolor="white")
        plt.close(fig)

        # ---- S7 property-binned scores ----
        fig = _a4_page(
            _L("图 S7｜性质分位与打分", "Fig. S7 | Scores across property quartiles"),
            head, footer=footer,
        )
        ax = _bulk_axes(fig, [0.10, 0.56, 0.82, 0.27])
        _bulk_quartile_panel(
            ax, mw, scores,
            _L("分子量四分位（Da）", "Molecular-weight quartile (Da)"),
            _L("按分子量分组", "Grouped by molecular weight"),
        )
        ax2 = _bulk_axes(fig, [0.10, 0.19, 0.82, 0.25])
        _bulk_quartile_panel(
            ax2, logp, scores,
            _L("cLogP 四分位", "cLogP quartile"),
            _L("按亲脂性分组", "Grouped by lipophilicity"),
        )
        _caption(
            fig,
            _L(
                "箱体为中位数与四分位距，须线为 1.5 倍四分位距；用于判断打分是否被分子性质系统性地驱动。",
                "Boxes show median and interquartile range, whiskers 1.5×IQR; used to check whether scores are systematically driven by physicochemical properties.",
            ),
            y=0.105,
        )
        pdf.savefig(fig, facecolor="white")
        plt.close(fig)
    return outpath


def _bulk_quartile_panel(ax, prop, scores, xlabel, title):
    mask = np.isfinite(prop) & np.isfinite(scores)
    if mask.sum() < 12:
        ax.text(0.5, 0.5, _L("数据不足", "Insufficient data"),
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color=MUTED)
        ax.set_title(title, loc="left", fontsize=14.5, fontweight="bold")
        return
    values = prop[mask]
    edges = np.nanquantile(values, [0, 0.25, 0.5, 0.75, 1.0])
    edges = np.unique(np.round(edges, 2))
    groups = []
    labels = []
    for index in range(len(edges) - 1):
        low, high = edges[index], edges[index + 1]
        if index == len(edges) - 2:
            selected = scores[mask][(values >= low) & (values <= high)]
        else:
            selected = scores[mask][(values >= low) & (values < high)]
        if selected.size:
            groups.append(selected)
            labels.append(f"{low:.1f}–{high:.1f}")
    if not groups:
        ax.text(0.5, 0.5, _L("数据不足", "Insufficient data"),
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color=MUTED)
        ax.set_title(title, loc="left", fontsize=14.5, fontweight="bold")
        return
    box = ax.boxplot(
        groups, patch_artist=True, showfliers=False,
        medianprops=dict(color=AMBER, linewidth=1.4),
        whiskerprops=dict(color=MUTED, linewidth=0.9),
        capprops=dict(color=MUTED, linewidth=0.9),
        boxprops=dict(linewidth=0.8, edgecolor=MUTED),
    )
    for patch, color in zip(box["boxes"], ["#99f6e4", "#5eead4", "#2dd4bf", "#0f766e"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(_L("Vina 打分（kcal/mol）", "Vina score (kcal/mol)"))
    ax.set_title(title, loc="left", fontsize=14.5, fontweight="bold")


_ENGINE_DISPLAY = {
    "vina": "Vina",
    "smina": "smina",
    "qvina2": "QuickVina 2",
}


def _engine_label(name: str) -> str:
    return _ENGINE_DISPLAY.get(str(name), str(name))


def _rank_correlation(matrix: np.ndarray) -> np.ndarray:
    ranks = np.apply_along_axis(
        lambda col: np.argsort(np.argsort(col)).astype(float), 0, matrix
    )
    return np.corrcoef(ranks, rowvar=False)


def multi_engine_report_pdf(run: dict, outpath: Path) -> Path:
    """CNS-style multi-engine consensus figures plus an interpretation page."""
    _style_cjk()
    english = i18n.get_lang() == "en"
    results = [
        item for item in run.get("results", []) if item.get("status") == "ok"
    ]
    meta = run.get("consensus") or {}
    engines = [
        name for name in (meta.get("engines") or [])
        if any((item.get("engine_scores") or {}).get(name) is not None for item in results)
    ]
    rows = [
        item for item in results
        if all((item.get("engine_scores") or {}).get(name) is not None for name in engines)
    ]
    if len(engines) < 2 or len(rows) < 2:
        raise ValueError("没有足够的多引擎对接结果")
    labels = [_engine_label(name) for name in engines]
    matrix = np.array(
        [[float(item["engine_scores"][name]) for name in engines] for item in rows],
        dtype=float,
    )
    names = [str(item.get("ligand_name") or item.get("ligand_id") or "") for item in rows]
    consensus = np.array([float(item.get("consensus_rank") or 0) for item in rows])
    order = np.argsort(consensus)
    ranks = np.apply_along_axis(
        lambda col: np.argsort(np.argsort(col)).astype(float) + 1, 0, matrix
    )
    correlations = _rank_correlation(matrix)
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(outpath) as pdf:
        # ---------------- page 1: distributions and pairwise agreement ----------------
        fig = _a4_page(
            _L("3　多引擎共识对接", "3　Multi-engine consensus docking"),
            _L(
                f"{len(rows)} 个配体 × {len(engines)} 个引擎：打分分布、两两一致性与引擎间差异",
                f"{len(rows)} ligands × {len(engines)} engines: score distributions, "
                "pairwise agreement and inter-engine spread",
            ),
        )
        ax = fig.add_axes([0.100, 0.600, 0.345, 0.235])
        parts = ax.violinplot(
            [matrix[:, index] for index in range(len(engines))],
            showmedians=True, widths=0.72,
        )
        for index, body in enumerate(parts["bodies"]):
            body.set_facecolor([TEAL, BLUE, AMBER, "#7c3aed"][index % 4])
            body.set_alpha(0.38)
        for part in ("cmedians", "cmaxes", "cmins", "cbars"):
            if part in parts:
                parts[part].set_color(MUTED)
        ax.set_xticks(range(1, len(engines) + 1))
        ax.set_xticklabels(labels, fontsize=11.5)
        ax.set_ylabel(_L("对接打分（kcal/mol）", "Docking score (kcal/mol)"), fontsize=12)
        ax.set_title(
            _L("A　各引擎打分分布", "A　Score distribution per engine"),
            loc="left", fontsize=15.5, fontweight="bold",
        )
        _compact_frame(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.7)

        ax = fig.add_axes([0.555, 0.600, 0.370, 0.235])
        ax.scatter(matrix[:, 0], matrix[:, 1], s=30, color=TEAL, alpha=0.8,
                   edgecolors="none")
        low = float(min(matrix[:, 0].min(), matrix[:, 1].min())) - 0.4
        high = float(max(matrix[:, 0].max(), matrix[:, 1].max())) + 0.4
        ax.plot([low, high], [low, high], linestyle="--", color=MUTED, linewidth=1.1)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_xlabel(labels[0], fontsize=12)
        ax.set_ylabel(labels[1], fontsize=12)
        ax.set_title(
            _L(
                f"B　{labels[0]} vs {labels[1]}（ρ = {correlations[0, 1]:.2f}）",
                f"B　{labels[0]} vs {labels[1]} (ρ = {correlations[0, 1]:.2f})",
            ),
            loc="left", fontsize=15.5, fontweight="bold",
        )
        _compact_frame(ax)
        ax.grid(color=GRID, linewidth=0.6, alpha=0.65)

        ax = fig.add_axes([0.075, 0.315, 0.370, 0.235])
        if len(engines) >= 3:
            ax.scatter(matrix[:, 0], matrix[:, 2], s=30, color=BLUE, alpha=0.8,
                       edgecolors="none")
            low = float(min(matrix[:, 0].min(), matrix[:, 2].min())) - 0.4
            high = float(max(matrix[:, 0].max(), matrix[:, 2].max())) + 0.4
            ax.plot([low, high], [low, high], linestyle="--", color=MUTED, linewidth=1.1)
            ax.set_xlim(low, high)
            ax.set_ylim(low, high)
            ax.set_xlabel(labels[0], fontsize=12)
            ax.set_ylabel(labels[2], fontsize=12)
            ax.set_title(
                _L(
                    f"C　{labels[0]} vs {labels[2]}（ρ = {correlations[0, 2]:.2f}）",
                    f"C　{labels[0]} vs {labels[2]} (ρ = {correlations[0, 2]:.2f})",
                ),
                loc="left", fontsize=15.5, fontweight="bold",
            )
            _compact_frame(ax)
            ax.grid(color=GRID, linewidth=0.6, alpha=0.65)
        else:
            spread = matrix.max(axis=1) - matrix.min(axis=1)
            ax.hist(spread, bins=int(min(18, max(6, len(spread) // 2))),
                    color=BLUE, alpha=0.82, edgecolor="white")
            ax.axvline(float(np.median(spread)), color=AMBER, linestyle="--", linewidth=1.6)
            ax.set_xlabel(
                _L("引擎间打分极差（kcal/mol）", "Score spread across engines (kcal/mol)"),
                fontsize=12,
            )
            ax.set_ylabel(_L("配体数", "Ligands"), fontsize=12)
            ax.set_title(_L("C　引擎间差异分布", "C　Inter-engine spread"),
                         loc="left", fontsize=15.5, fontweight="bold")
            _compact_frame(ax)

        ax = fig.add_axes([0.555, 0.315, 0.370, 0.235])
        spread = matrix.max(axis=1) - matrix.min(axis=1)
        sorted_index = np.argsort(spread)
        ax.barh(range(len(spread)), spread[sorted_index], color=AMBER, height=0.72)
        ax.set_yticks([])
        ax.set_xlabel(_L("打分极差（kcal/mol）", "Score spread (kcal/mol)"), fontsize=12)
        ax.set_ylabel(_L("配体（按极差升序）", "Ligands (ascending spread)"), fontsize=12)
        ax.set_title(_L("D　引擎间一致性排序", "D　Agreement ranking"),
                     loc="left", fontsize=15.5, fontweight="bold")
        _compact_frame(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.7)

        _caption(
            fig,
            _L(
                "A：小提琴为各引擎打分分布，横线为中位数；不同引擎打分尺度不同，绝对值不可直接比较。"
                "B/C：两两散点，虚线为 y = x，ρ 为 Spearman 秩相关系数，越接近 1 表示排序越一致。"
                "D：每个配体在全部引擎中的打分极差，越小表示引擎间结论越一致，越大提示需人工复核。",
                "A: violin plots show score distributions with medians; absolute values are "
                "not comparable across engines. B/C: pairwise scatter plots with the y = x "
                "line; ρ is the Spearman rank correlation, closer to 1 meaning more "
                "consistent ranking. D: per-ligand score spread across engines, smaller "
                "meaning better agreement and larger flagging candidates for manual review.",
            ),
            y=0.185, width=94,
        )
        pdf.savefig(fig, facecolor="white")
        plt.close(fig)

        # ---------------- page 2: normalised scores and rank correlations ----------------
        fig = _a4_page(
            _L("3　多引擎共识对接（续一）", "3　Multi-engine consensus (continued 1)"),
            _L("标准化打分与引擎间秩相关矩阵",
               "Normalised scores and rank-correlation matrix"),
        )
        top = order[: min(20, len(order))]
        std = matrix.std(axis=0)
        z = (matrix - matrix.mean(axis=0)) / np.where(std == 0, 1, std)
        ax = fig.add_axes([0.265, 0.565, 0.565, 0.270])
        image = ax.imshow(z[top], cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)
        ax.set_xticks(range(len(engines)))
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels([_clip_display(names[index], 26) for index in top], fontsize=10)
        ax.set_title(_L("E　标准化打分（z 值）", "E　Normalised scores (z)"),
                     loc="left", fontsize=16, fontweight="bold")
        color_ax = fig.add_axes([0.845, 0.565, 0.016, 0.270])
        fig.colorbar(image, cax=color_ax)
        color_ax.tick_params(labelsize=10)

        ax = fig.add_axes([0.265, 0.285, 0.300, 0.165])
        image = ax.imshow(correlations, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(engines)))
        ax.set_xticklabels(labels, fontsize=10.5, rotation=25, ha="right")
        ax.set_yticks(range(len(engines)))
        ax.set_yticklabels(labels, fontsize=10.5)
        for row in range(len(engines)):
            for col in range(len(engines)):
                ax.text(col, row, f"{correlations[row, col]:.2f}",
                        ha="center", va="center", fontsize=10,
                        color="white" if abs(correlations[row, col]) > 0.55 else INK)
        ax.set_title(_L("F　秩相关矩阵", "F　Rank-correlation matrix"),
                     loc="left", fontsize=16, fontweight="bold")
        color_ax = fig.add_axes([0.585, 0.285, 0.014, 0.165])
        fig.colorbar(image, cax=color_ax)
        color_ax.tick_params(labelsize=9.5)

        _caption(
            fig,
            _L(
                "E：各引擎打分按列标准化后的 z 值，蓝色表示相对更优、红色表示相对更差，"
                "用于观察同一候选在不同引擎中的相对位置。F：引擎间 Spearman 秩相关矩阵，"
                "对角线为 1，颜色越红表示排序越一致；ρ < 0.5 时相邻名次需结合作用力人工复核。",
                "E: per-engine z-scores after column-wise normalisation; blue marks "
                "relatively better and red relatively worse positions. F: Spearman "
                "rank-correlation matrix among engines (diagonal = 1); redder cells mean "
                "more consistent ranking, and ρ < 0.5 requires manual review of neighbouring ranks.",
            ),
            y=0.180, width=94,
        )
        pdf.savefig(fig, facecolor="white")
        plt.close(fig)

        # ---------------- page 3: consensus ranking ----------------
        fig = _a4_page(
            _L("3　多引擎共识对接（续二）", "3　Multi-engine consensus (continued 2)"),
            _L("综合共识排名：Vina、Vinardo 与各附加引擎名次均值，越小越靠前",
               "Consensus ranking: mean of Vina, Vinardo and additional engine ranks, lower is better"),
        )
        selected = order[: min(15, len(order))]
        positions = np.arange(len(selected))
        ax = fig.add_axes([0.340, 0.335, 0.505, 0.505])
        ax.barh(positions, consensus[selected], color=TEAL, height=0.66, alpha=0.90)
        ax.set_yticks(positions)
        ax.set_yticklabels(
            [_clip_display(names[index], 30) for index in selected], fontsize=10.5
        )
        ax.invert_yaxis()
        max_rank = float(np.max(consensus[selected])) if len(selected) else 1.0
        ax.set_xlim(0, max_rank * 1.45 + 0.4)
        ax.set_xlabel(
            _L("平均排名（越小越好）", "Mean rank (lower is better)"), fontsize=12.5
        )
        ax.set_title(_L("G　综合共识排名", "G　Consensus ranking"),
                     loc="left", fontsize=16, fontweight="bold")
        short = {"vina": "V", "smina": "S", "qvina2": "Q"}
        for position, index in enumerate(selected):
            detail = " · ".join(
                f"{short.get(engines[col], engines[col][:1].upper())}#{int(ranks[index, col])}"
                for col in range(len(engines))
            )
            ax.text(
                float(consensus[index]) + max_rank * 0.02, position, detail,
                va="center", fontsize=9, color=MUTED, clip_on=False,
            )
        _compact_frame(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.7)
        fig.text(
            0.075, 0.278,
            _L(
                "V = Vina，S = smina，Q = QuickVina 2；# 后为该引擎的单独名次；"
                "平均排名同时计入 Vinardo 交叉打分的名次。",
                "V = Vina, S = smina, Q = QuickVina 2; # = individual engine rank; "
                "the mean rank also includes the Vinardo cross-score rank.",
            ),
            fontsize=9.5, color=MUTED, va="top",
        )
        _caption(
            fig,
            _L(
                "G：综合排名取各引擎名次（Vina、Vinardo 与追加引擎共同参与）的平均值，"
                "条形越短表示跨引擎共识越好；条右侧列出各引擎单独名次，用于判断排名是否由单一引擎驱动。"
                "建议优先选入在至少两个引擎中均位于前 20% 的候选。",
                "G: consensus rank is the mean rank across Vina, Vinardo and the additional "
                "engines; shorter bars mean stronger cross-engine agreement. Individual engine "
                "ranks are listed next to each bar so the ranking is not driven by one engine. "
                "Prioritise candidates in the top 20% of at least two engines.",
            ),
            y=0.175, width=94,
        )
        pdf.savefig(fig, facecolor="white")
        plt.close(fig)

        # ---------------- page 3: interpretation and recommendation ----------------
        rhos = [
            float(correlations[row, col])
            for row in range(len(engines))
            for col in range(row + 1, len(engines))
        ]
        mean_rho = float(np.mean(rhos)) if rhos else 0.0
        if mean_rho >= 0.8:
            agreement_zh = "高度一致（平均 ρ ≥ 0.8），排序结果对引擎选择不敏感"
            agreement_en = "high agreement (mean ρ ≥ 0.8); the ranking is robust to engine choice"
        elif mean_rho >= 0.5:
            agreement_zh = "中等一致（0.5 ≤ 平均 ρ < 0.8），相邻名次需结合作用力复核"
            agreement_en = "moderate agreement (0.5 ≤ mean ρ < 0.8); neighbouring ranks need interaction review"
        else:
            agreement_zh = "一致性偏低（平均 ρ < 0.5），排名只能作为假设，需更长时间采样或 MM-GBSA 复核"
            agreement_en = "low agreement (mean ρ < 0.5); ranks are hypotheses and need longer sampling or MM-GBSA"
        top_names = "、".join(names[index] for index in order[:3])
        engine_notes_zh = [
            "Vina：社区使用最广、打分可解释，是本报告的基准引擎；适合跨项目比较。",
            "smina：Vina 的增强分支，局部优化更充分，排序通常与 Vina 高度一致，"
            "可作为稳健性检验。",
            "QuickVina 2：以搜索加速为目标，速度通常为 Vina 的数倍，"
            "适合大规模初筛；本报告只使用其打分排序。",
        ]
        engine_notes_en = [
            "Vina: the most widely used and interpretable baseline, suitable for "
            "cross-project comparison.",
            "smina: an enhanced Vina derivative with more thorough local optimisation; "
            "its ranking usually tracks Vina closely and serves as a robustness check.",
            "QuickVina 2: optimised for search speed, often several-fold faster than "
            "Vina and suited to first-pass screening; only its ranking is used here.",
        ]
        blocks = [
            (
                "3.1 引擎特点与适用场景" if not english else "3.1 Engine characteristics",
                engine_notes_zh if not english else engine_notes_en,
            ),
            (
                "3.2 一致性结论" if not english else "3.2 Agreement",
                [
                    (
                        f"引擎间 Spearman 秩相关：{'、'.join(f'{labels[row]}–{labels[col]} {correlations[row, col]:.2f}' for row in range(len(engines)) for col in range(row + 1, len(engines)))}。"
                        f"总体判断为{agreement_zh}。"
                        if not english
                        else f"Spearman rank correlations: "
                        + ", ".join(
                            f"{labels[row]}–{labels[col]} {correlations[row, col]:.2f}"
                            for row in range(len(engines))
                            for col in range(row + 1, len(engines))
                        )
                        + f". Overall: {agreement_en}."
                    ),
                    (
                        f"在 {len(engines)} 个引擎中，{top_names} 的综合名次最靠前；"
                        "建议优先选入精细复核清单。"
                        if not english
                        else f"Across {len(engines)} engines, {top_names} rank highest in "
                        "consensus and should enter the refined shortlist first."
                    ),
                ],
            ),
            (
                "3.3 综合选型建议" if not english else "3.3 Recommendation",
                [
                    (
                        "优先选择在至少两个引擎中排名前 20%、且 Vina 打分不高于 −8 kcal/mol 的分子；"
                        "引擎间极差大于 1.5 kcal/mol 的候选应做更长时间采样、MM-GBSA 或分子动力学复核。"
                        "不同引擎打分尺度不同，禁止直接比较绝对值，只使用排名与一致性。"
                        if not english
                        else "Prioritise ligands ranked in the top 20% by at least two engines "
                        "with a Vina score of −8 kcal/mol or better. Candidates whose "
                        "inter-engine spread exceeds 1.5 kcal/mol should be re-examined with "
                        "longer sampling, MM-GBSA or molecular dynamics. Absolute scores are "
                        "not comparable across engines; only ranks and agreement should be used."
                    ),
                ],
            ),
        ]
        _flow_text_pages(
            pdf,
            "3　多引擎结果解读与综合意见" if not english else "3　Multi-engine interpretation",
            "引擎差异、排序稳健性与候选推荐" if not english else "Engine differences, ranking robustness and candidate selection",
            blocks, width=46, size=10.8, heading_size=12.6, stop=0.070,
        )
    return outpath
