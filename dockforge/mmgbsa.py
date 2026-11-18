"""Prime-style MM-GBSA approximation with per-residue energy decomposition.

This is deliberately a fast, transparent force-field calculation rather than a
full MM-GBSA run: a 12-6 Lennard-Jones term, Coulomb electrostatics with a
protein-like dielectric, a Born-style polar desolvation penalty and an
empirical nonpolar term.  Absolute numbers are approximate; the residue
ranking and the comparison between analogs are the useful output.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from . import pdbio
from .utils import safe_name

# (Rmin/2 in Å, well depth in kcal/mol), Amber-like united-atom values.
_LJ = {
    "C": (1.908, 0.086),
    "N": (1.824, 0.170),
    "O": (1.661, 0.210),
    "S": (2.000, 0.250),
    "P": (2.100, 0.200),
    "F": (1.750, 0.061),
    "Cl": (1.948, 0.265),
    "Br": (2.220, 0.320),
    "I": (2.350, 0.400),
    "H": (1.459, 0.0157),
}

_BACKBONE = {"N": -0.35, "O": -0.50, "C": 0.40, "CA": 0.10}
_POLAR_SCALE = 0.15
_SIDECHAIN = {
    ("ASP", "OD1"): -0.55, ("ASP", "OD2"): -0.55,
    ("GLU", "OE1"): -0.55, ("GLU", "OE2"): -0.55,
    ("LYS", "NZ"): 0.85,
    ("ARG", "NE"): 0.30, ("ARG", "NH1"): 0.45, ("ARG", "NH2"): 0.45,
    ("HIS", "ND1"): 0.25, ("HIS", "NE2"): 0.25,
}

_TERM_LABELS = {
    "vdw": "范德华",
    "elec": "静电",
    "desolv": "去溶剂化",
    "hbond": "氢键",
}


def _element_from_type(atype: str) -> str:
    text = atype.strip()
    if not text:
        return "C"
    if text in ("A", "C", "C1", "C2"):
        return "C"
    if text.startswith("N"):
        return "N"
    if text.startswith("O") or text == "OA":
        return "O"
    if text.startswith("S"):
        return "S"
    if text.startswith("H"):
        return "H"
    if text.startswith("F"):
        return "F"
    if text.upper().startswith("CL"):
        return "Cl"
    if text.upper().startswith("BR"):
        return "Br"
    if text == "I":
        return "I"
    if text.startswith("P"):
        return "P"
    return text[0].upper()


def _protein_charge(atom) -> float:
    if atom.name in _BACKBONE:
        return _BACKBONE[atom.name]
    return _SIDECHAIN.get((atom.resname, atom.name), 0.0)


def _fallback_charge(element: str) -> float:
    return {"N": -0.30, "O": -0.40, "S": -0.15, "C": 0.05, "P": 0.30}.get(element, 0.0)


def decompose(
    pose_path: Path,
    receptor_pdb: Path,
    mode: int = 1,
    interaction: dict | None = None,
    affinity: float | None = None,
) -> dict:
    text = pdbio.normalize_pdbqt(pose_path.read_text(encoding="utf-8", errors="replace"))
    text = pdbio.pose_pdbqt(text, mode)
    ligand = []
    for line in text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        charge, atype = pdbio.pdbqt_atom_fields(line)
        element = _element_from_type(atype)
        ligand.append(
            {
                "coord": np.array(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                ),
                "element": element,
                "charge": float(charge),
            }
        )
    if not ligand:
        raise ValueError("构象中没有可用的配体原子")

    st = pdbio.parse_pdb(receptor_pdb.read_text(encoding="utf-8", errors="replace"))
    protein = [a for a in st.atoms if a.element != "H" and a.element in _LJ]
    if not protein:
        raise ValueError("受体结构为空")
    pcoords = np.array([[a.x, a.y, a.z] for a in protein])
    pcharges = np.array([_protein_charge(a) for a in protein])
    tree = cKDTree(pcoords)

    residue_stats: dict[tuple, dict] = {}
    terms = {"vdw": 0.0, "elec": 0.0, "desolv": 0.0, "hbond": 0.0}
    ligand_heavy = [atom for atom in ligand if atom["element"] != "H"]
    buried = []

    for lig in ligand_heavy:
        neighbours = tree.query_ball_point(lig["coord"], 6.0)
        fraction = min(1.0, len(neighbours) / 26.0)
        buried.append(fraction)
        rmin_i, eps_i = _LJ.get(lig["element"], _LJ["C"])
        # Born-style polar penalty, scaled to the partial-burial regime; the
        # free-ligand self term cancels in the MM-GBSA difference.
        solv_polar = float(_POLAR_SCALE * 164.0 * lig["charge"] ** 2 / rmin_i * fraction)
        carbon_contacts = 0
        for j in neighbours:
            atom = protein[j]
            vec = lig["coord"] - pcoords[j]
            r = float(np.linalg.norm(vec))
            if r < 0.8:
                r = 0.8
            key = (atom.chain, atom.resseq, atom.resname)
            entry = residue_stats.setdefault(
                key,
                {
                    "residue": atom.resname,
                    "chain": atom.chain,
                    "resseq": atom.resseq,
                    "vdw": 0.0,
                    "elec": 0.0,
                    "desolv": 0.0,
                    "hbond": 0.0,
                    "contacts": 0,
                    "min_distance": 99.0,
                },
            )
            entry["contacts"] += 1
            entry["min_distance"] = min(entry["min_distance"], round(r, 2))
            fade = 1.0 if r <= 6.0 else max(0.0, 1.0 - (r - 6.0) / 2.0)
            rmin_j, eps_j = _LJ.get(atom.element, _LJ["C"])
            rmin = rmin_i + rmin_j
            epsilon = (eps_i * eps_j) ** 0.5
            ratio = rmin / r
            lj = float(np.clip(epsilon * (ratio ** 12 - 2.0 * ratio ** 6), -6.0, 60.0)) * fade
            # Distance-dependent dielectric (ε = 4r) as in empirical scoring.
            elec = float(332.0 * lig["charge"] * pcharges[j] / (4.0 * r * r) * fade)
            entry["vdw"] += lj
            entry["elec"] += elec
            terms["vdw"] += lj
            terms["elec"] += elec
            if lig["element"] in ("C", "S") and atom.element in ("C", "S"):
                carbon_contacts += 1
        solv_nonpolar = -0.06 * min(carbon_contacts, 14) * fraction
        solv_total = solv_polar + solv_nonpolar
        terms["desolv"] += solv_total
        for j in neighbours:
            atom = protein[j]
            key = (atom.chain, atom.resseq, atom.resname)
            residue_stats[key]["desolv"] += solv_total / max(1, len(neighbours))

    for item in (interaction or {}).get("hbonds", [])[:60]:
        key = (item.get("chain"), int(item.get("resseq", 0)), item.get("residue"))
        entry = residue_stats.get(key)
        if not entry:
            continue
        if entry["hbond"] > -3.2:
            entry["hbond"] -= 1.6
            terms["hbond"] -= 1.6

    rows = []
    for entry in residue_stats.values():
        total = entry["vdw"] + entry["elec"] + entry["desolv"] + entry["hbond"]
        rows.append(
            {
                "residue": entry["residue"],
                "chain": entry["chain"],
                "resseq": entry["resseq"],
                "vdw": round(entry["vdw"], 2),
                "elec": round(entry["elec"], 2),
                "desolv": round(entry["desolv"], 2),
                "hbond": round(entry["hbond"], 2),
                "total": round(total, 2),
                "contacts": entry["contacts"],
                "min_distance": round(entry["min_distance"], 2),
            }
        )
    rows.sort(key=lambda item: item["total"])

    total = sum(item["total"] for item in rows)
    return {
        "available": True,
        "method": "12-6 LJ + Coulomb(ε=4) + Born 去溶剂化 + 几何氢键（MM-GBSA 近似）",
        "total": round(total, 2),
        "terms": {key: round(value, 2) for key, value in terms.items()},
        "residues": rows,
        "top": rows[:16],
        "buried_fraction": round(float(np.mean(buried)) if buried else 0.0, 2),
        "ligand_atoms": len(ligand_heavy),
        "vina": affinity,
        "note": "绝对值仅为近似；残基排序与类似物间相对比较更可靠。",
    }


def mmgbsa_pdf(data: dict, name: str, receptor: str, outpath: Path) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    from . import figures, i18n

    figures._style_cjk()
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8.27, 11.69))
    grid = fig.add_gridspec(
        4, 2, height_ratios=[0.42, 1.55, 0.10, 0.60],
        width_ratios=[2.2, 1.0],
        hspace=0.34, wspace=0.22,
        left=0.20, right=0.95, top=0.90, bottom=0.07,
    )
    english = i18n.get_lang() == "en"
    fig.text(
        0.07, 0.955,
        "Binding free energy decomposition" if english else "结合自由能分解",
        fontsize=19 if english else 20, fontweight="bold", color=figures.INK,
    )
    fig.text(
        0.07, 0.928,
        f"{name} × {receptor}   ·   残基级 MM-GBSA 近似",
        fontsize=11.5, color=figures.MUTED,
    )

    # ---- summary -----------------------------------------------------------
    ax = fig.add_subplot(grid[0, :])
    ax.axis("off")
    metrics = [
        ("估算 ΔG" if not english else "ΔG (est.)", f"{data['total']:.1f}", "kcal/mol"),
        ("Vina 打分" if not english else "Vina", f"{data['vina']:.2f}" if data.get("vina") is not None else "—", "kcal/mol"),
        ("埋藏比例" if not english else "Buried", f"{data['buried_fraction'] * 100:.0f}", "%"),
        ("接触残基" if not english else "Contacts", f"{len(data['residues'])}", "个" if not english else ""),
    ]
    positions = [0.0, 0.24, 0.48, 0.70]
    for (label, value, unit), x in zip(metrics, positions):
        ax.text(x, 0.95, label, fontsize=10, color=figures.MUTED, va="top")
        ax.text(x, 0.62, value, fontsize=18, fontweight="bold", color=figures.INK, va="top")
        if unit:
            ax.text(x + 0.085, 0.60, unit, fontsize=8.2, color=figures.MUTED, va="top")

    # ---- stacked residue bars ---------------------------------------------
    ax = fig.add_subplot(grid[1, 0])
    rows = list(reversed(data["top"]))  # most favourable at the top
    labels = [f"{item['residue']}{item['resseq']}" for item in rows]
    y = np.arange(len(rows))
    colors = {
        "vdw": figures.TEAL,
        "elec": figures.BLUE,
        "desolv": figures.AMBER,
        "hbond": "#7e22ce",
    }
    pos = np.zeros(len(rows))
    neg = np.zeros(len(rows))
    for key in ("vdw", "elec", "desolv", "hbond"):
        values = np.array([item[key] for item in rows])
        left = np.where(values >= 0, pos, neg)
        ax.barh(y, values, left=left, color=colors[key], height=0.66,
                label=_TERM_LABELS[key], edgecolor="white", linewidth=0.4)
        pos += np.where(values >= 0, values, 0.0)
        neg += np.where(values < 0, values, 0.0)
    ax.axvline(0, color=figures.INK, linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("残基贡献 (kcal/mol)", fontsize=11)
    ax.set_title("残基能量贡献（前 16 位）", fontsize=13, loc="left", pad=10)
    ax.grid(axis="x", color=figures.GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # ---- shared legend -----------------------------------------------------
    ax = fig.add_subplot(grid[2, :])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for index, key in enumerate(("vdw", "elec", "desolv", "hbond")):
        x = 0.02 + index * 0.17
        ax.add_patch(
            plt.Rectangle((x, 0.34), 0.018, 0.30, color=colors[key], transform=ax.transAxes)
        )
        ax.text(x + 0.025, 0.48, _TERM_LABELS[key], fontsize=9.5,
                color=figures.INK, va="center")

    # ---- term totals -------------------------------------------------------
    ax = fig.add_subplot(grid[1, 1])
    keys = ["vdw", "elec", "desolv", "hbond"]
    values = [data["terms"][key] for key in keys]
    short_labels = (
        ["vdW", "Elec", "Desolv", "H-bond"]
        if english else [_TERM_LABELS[key] for key in keys]
    )
    bars = ax.barh(
        short_labels[::-1], values[::-1],
        color=[colors[key] for key in keys][::-1], height=0.55,
    )
    for bar, value in zip(bars, values[::-1]):
        offset = 0.6 if value >= 0 else -0.6
        ax.text(value + offset, bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}", va="center",
                ha="left" if value >= 0 else "right", fontsize=9.2, fontweight="bold")
    span = max(1.0, max(abs(item) for item in values))
    pad = 0.45 * span
    ax.set_xlim(min(min(values) - pad, -pad), max(max(values) + pad, pad))
    ax.axvline(0, color=figures.INK, linewidth=0.9)
    ax.set_title("能量项总计", fontsize=12, loc="left", pad=10)
    ax.set_xlabel("kcal/mol", fontsize=10)
    ax.grid(axis="x", color=figures.GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # ---- table -------------------------------------------------------------
    ax = fig.add_subplot(grid[3, :])
    ax.axis("off")
    header = ["残基", "范德华", "静电", "去溶剂化", "氢键", "合计", "接触", "最近距离 (Å)"]
    if english:
        header = [
            "Residue", "vdW", "Elec", "Desolv", "H-bond", "Total",
            "Contacts", "Min dist (Å)",
        ]
    table_rows = []
    for item in data["top"][:6]:
        table_rows.append(
            [
                f"{item['residue']}{item['resseq']}",
                f"{item['vdw']:.1f}",
                f"{item['elec']:.1f}",
                f"{item['desolv']:.1f}",
                f"{item['hbond']:.1f}",
                f"{item['total']:.1f}",
                str(item["contacts"]),
                f"{item['min_distance']:.2f}",
            ]
        )
    if table_rows:
        table = ax.table(
            cellText=table_rows, colLabels=header,
            colWidths=[0.16, 0.09, 0.09, 0.11, 0.09, 0.09, 0.11, 0.16]
            if english else [0.18, 0.11, 0.10, 0.14, 0.09, 0.11, 0.08, 0.14],
            loc="upper center", cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.2 if english else 9)
        table.scale(1, 1.45)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor(figures.GRID)
            cell.set_linewidth(0.6)
            if row == 0:
                cell.set_facecolor("#f1f5f9")
                cell.set_text_props(fontweight="bold", color=figures.INK)
            elif col == 5:
                cell.set_text_props(fontweight="bold", color=figures.TEAL)

    fig.text(
        0.07, 0.020,
        figures._wrap_cjk(
            "方法：配体采用 PDBQT 部分电荷，蛋白采用模板电荷；ε = 4r，8 Å 截断，"
            "Born 型极性去溶剂化。结果为近似排序，非绝对结合自由能。",
            116,
        ),
        fontsize=8.0, color=figures.MUTED, va="bottom",
    )
    fig.savefig(outpath, dpi=600, facecolor="white")
    plt.close(fig)
    return outpath


def filename(name: str) -> str:
    return f"{safe_name(name, 'ligand')}_mmgbsa.pdf"
