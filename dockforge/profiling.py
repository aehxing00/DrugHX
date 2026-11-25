"""Series-level developability, chemical-space and metabolism profiling.

These panels borrow the most useful lightweight ideas from commercial and
open-source drug-design suites:

* SwissADME / DataWarrior : BOILED-Egg style HIA/BBB discrimination.
* StarDrop / CNS MPO     : multi-parameter CNS developability score.
* ChemAxon               : rule-based pKa / logD(7.4) speciation.
* SwissADME / RDKit      : PAINS, Brenk and NIH structural-alert filters.
* DataWarrior            : Murcko scaffold diversity and chemical-space maps.
* SYNTHIA                : Ertl synthetic-accessibility score (RDKit Contrib).
* StarDrop               : rule-based metabolic soft-spot flags.

All numbers are transparent descriptor rules or published one-line models.
They are intended for triage and relative ranking, not for replacing
experimental measurements.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.Scaffolds import MurckoScaffold

from . import figures, i18n

# ---------------------------------------------------------------------------
# optional Ertl SA score shipped with RDKit Contrib
# ---------------------------------------------------------------------------
_SA_SCORER = None


def _sascorer():
    global _SA_SCORER
    if _SA_SCORER is not None:
        return _SA_SCORER
    try:
        import rdkit

        contrib = Path(rdkit.__file__).resolve().parent / "Contrib" / "SA_Score"
        if str(contrib) not in sys.path:
            sys.path.insert(0, str(contrib))
        import sascorer  # type: ignore

        _SA_SCORER = sascorer
    except Exception:
        _SA_SCORER = False
    return _SA_SCORER


# ---------------------------------------------------------------------------
# rule tables
# ---------------------------------------------------------------------------
_ACID_RULES = [
    ("磺酸", "[SX4](=O)(=O)[OX2H1]", 1.0),
    ("磷酸", "[PX4](=O)([OX2H1])", 2.1),
    ("羧酸", "[CX3](=O)[OX2H1]", 4.5),
    ("四唑", "[nH]1nnnc1", 4.5),
    ("酚羟基", "[OX2H][c]", 10.0),
]

_BASE_RULES = [
    ("胍", "[NX3][CX3](=[NX2])[NX3]", 12.5),
    ("脒", "[CX3](=[NX2])[NX3]", 11.0),
    ("咪唑", "[nX2;r5]", 6.8),
    ("吡啶", "[nX2]", 5.2),
    ("脂肪胺", "[NX3;H2,H1,H0;!$(N-a);!$(N-C=O);!$(N-S);!$(N-C=[O,N,S]);!$([N+])]", 9.5),
    ("芳香胺", "[NX3][c]", 4.6),
]

_SOM_RULES = [
    # label, SMARTS, weight, max sites, index of the atom that is oxidised.
    ("N-脱烷基", "[NX3;!$(N-C=O);!$(N-S(=O)=O)][CH2][CH3]", 1.6, 2, -1),
    ("O-脱烷基", "[OX2][CH3]", 1.5, 2, -1),
    ("硫氧化", "[SX2]", 1.5, 2, 0),
    ("苄位氧化", "[CH2,CH3;!$([CH3][CX3]=O)][c]", 1.3, 2, 0),
    ("烯丙位氧化", "[CH2,CH3][CX3]=[CX3]", 1.2, 2, 0),
    ("酚羟基结合", "[OX2H][c]", 1.0, 2, 0),
    ("芳环氧化", "a1ccccc1", 0.7, 1, 0),
]

# Short English labels for the narrow structure panels; the full names used in
# the report text would overflow the three-column layout.
_SOM_LABEL_EN = {
    "N-脱烷基": "N-dealkyl.",
    "O-脱烷基": "O-dealkyl.",
    "硫氧化": "S-oxidation",
    "苄位氧化": "Benzylic ox.",
    "烯丙位氧化": "Allylic ox.",
    "酚羟基结合": "Phenol conj.",
    "芳环氧化": "Aromatic ox.",
}

# Wager (2010) CNS MPO thresholds.
_CNS_MPO = [
    ("cLogP", "logp", 3.0, 5.0),
    ("cLogD", "logd", 2.0, 4.0),
    ("MW", "mw", 360.0, 500.0),
    ("HBD", "hbd", 0.5, 3.5),
    ("TPSA", "tpsa", 90.0, 120.0),
    ("pKa", "pka", 8.0, 10.0),
]

_CATALOGS = None


def _catalogs() -> dict:
    global _CATALOGS
    if _CATALOGS is not None:
        return _CATALOGS
    catalogs = {}
    for key, catalog in (
        ("PAINS", FilterCatalogParams.FilterCatalogs.PAINS),
        ("Brenk", FilterCatalogParams.FilterCatalogs.BRENK),
        ("NIH", FilterCatalogParams.FilterCatalogs.NIH),
    ):
        params = FilterCatalogParams()
        params.AddCatalog(catalog)
        catalogs[key] = FilterCatalog(params)
    _CATALOGS = catalogs
    return catalogs


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _matches(mol: Chem.Mol, pattern: str) -> list[tuple]:
    query = Chem.MolFromSmarts(pattern)
    if query is None:
        return []
    return [match for match in mol.GetSubstructMatches(query)]


def estimate_pka(mol: Chem.Mol) -> dict:
    """Rule-based most-acidic / most-basic centre.

    Every matching rule is collected: a molecule can contain a pyridine *and*
    an aliphatic amine, and taking only the first match would report the
    pyridine (5.2) instead of the far more basic amine (9.5), which then
    corrupts logD7.4 and the CNS MPO pKa term.
    """
    acids = [
        {"label": label, "pka": pka, "kind": "acid"}
        for label, smarts, pka in _ACID_RULES
        if _matches(mol, smarts)
    ]
    bases = [
        {"label": label, "pka": pka, "kind": "base"}
        for label, smarts, pka in _BASE_RULES
        if _matches(mol, smarts)
    ]
    acid_item = min(acids, key=lambda item: item["pka"], default=None)
    base_item = max(bases, key=lambda item: item["pka"], default=None)
    return {
        "acid": acid_item["pka"] if acid_item else None,
        "base": base_item["pka"] if base_item else None,
        "acid_label": acid_item["label"] if acid_item else "",
        "base_label": base_item["label"] if base_item else "",
    }


def estimate_logd(logp: float, pka: dict, ph: float = 7.4) -> float:
    """Henderson-Hasselbalch speciation, taking the most ionised form."""
    values = [logp]
    if pka.get("acid"):
        values.append(logp - np.log10(1 + 10 ** (ph - pka["acid"])))
    if pka.get("base"):
        values.append(logp - np.log10(1 + 10 ** (pka["base"] - ph)))
    return float(min(values))


def _step_score(value: float, low: float, high: float) -> float:
    """Wager CNS MPO uses a three-level 1 / 0.5 / 0 score per property."""
    if value <= low:
        return 1.0
    if value <= high:
        return 0.5
    return 0.0


def cns_mpo(profile: dict) -> tuple[float, list[dict]]:
    """Wager (2010) CNS MPO (0-6, higher is better).

    Each of the six properties is scored 1 / 0.5 / 0 at the published
    thresholds; interpolating instead would inflate the total (e.g. TPSA
    below 40 Å² and HBD = 1 must not receive a fraction of a point).
    """
    components = []
    for label, key, low, high in _CNS_MPO:
        if key in ("tpsa", "hbd", "pka"):
            if key == "pka":
                # No basic centre is the favourable case in the Wager scheme.
                value = float(profile.get("basic_pka") or 0.0)
            else:
                value = float(profile[key])
            score = _step_score(value, low, high)
        else:
            score = _step_score(float(profile[key]), low, high)
        components.append({"label": label, "score": round(score, 2)})
    return round(sum(item["score"] for item in components), 2), components


def egg_zone(wlogp: float, tpsa: float) -> str:
    """BOILED-Egg style HIA / BBB zone from Egan-type thresholds."""
    if tpsa <= 79 and 0.4 <= wlogp <= 6.0:
        return "BBB+"
    if tpsa <= 131.6 and -0.4 <= wlogp <= 5.88:
        return "HIA+"
    return "低渗透"


def alert_hits(mol: Chem.Mol) -> dict:
    hits = {}
    for key, catalog in _catalogs().items():
        entries = catalog.GetMatches(mol)
        hits[key] = [entry.GetDescription() for entry in entries]
    return hits


def _scaffold(mol: Chem.Mol) -> str:
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ""


def series_profiles(records: list[dict]) -> list[dict]:
    """Compute one profiling block per molecule."""
    profiles = []
    sascorer = _sascorer()
    for record in records:
        smiles = str(record.get("smiles") or "")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        mol = Chem.RemoveHs(mol)
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass
        heavy = max(1, mol.GetNumHeavyAtoms())
        mw = float(Descriptors.MolWt(mol))
        logp = float(Crippen.MolLogP(mol))
        tpsa = float(Descriptors.TPSA(mol))
        hbd = int(rdMolDescriptors.CalcNumHBD(mol))
        hba = int(rdMolDescriptors.CalcNumHBA(mol))
        rotb = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
        fsp3 = float(rdMolDescriptors.CalcFractionCSP3(mol))
        rings = int(rdMolDescriptors.CalcNumRings(mol))
        pka = estimate_pka(mol)
        logd = estimate_logd(logp, pka)
        base = {
            "name": record.get("name") or smiles[:18],
            "smiles": smiles,
            "affinity": record.get("affinity"),
            "mw": mw,
            "logp": logp,
            "tpsa": tpsa,
            "hbd": hbd,
            "hba": hba,
            "rotb": rotb,
            "fsp3": fsp3,
            "rings": rings,
            "heavy": heavy,
            "logd": logd,
            "acid_pka": pka.get("acid"),
            "basic_pka": pka.get("base"),
            "acid_label": pka.get("acid_label"),
            "base_label": pka.get("base_label"),
        }
        score, components = cns_mpo(base)
        if sascorer:
            try:
                sa = float(sascorer.calculateScore(mol))
            except Exception:
                sa = None
        else:
            sa = None
        base.update(
            {
                "sa": sa,
                "cns_mpo": score,
                "cns_components": components,
                "zone": egg_zone(logp, tpsa),
                "alerts": alert_hits(mol),
                "scaffold": _scaffold(mol),
            }
        )
        profiles.append(base)
    return profiles


def _alert_counts(profile: dict) -> dict:
    return {key: len(value) for key, value in profile["alerts"].items()}


def _short_name(name: str, width: int = 9) -> str:
    text = i18n.tr(str(name))
    token = text.split()[0] if text.split() else text
    return token[:width]


# ---------------------------------------------------------------------------
# Figure 11 : developability panel
# ---------------------------------------------------------------------------
def developability_pdf(records: list[dict], outpath: Path) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Ellipse

    figures._style_cjk()
    profiles = series_profiles(records)
    if not profiles:
        raise ValueError("没有可用于可开发性分析的分子")
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    fig = figures._a4_page(
        "图 11　类药性与可开发性",
        "BOILED-Egg 吸收判别、CNS MPO、logD7.4 与结构警示",
        "Fig. 11　Developability profiling",
    )
    grid = fig.add_gridspec(
        2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.0, 1.0],
        hspace=0.42, wspace=0.55,
        left=0.12, right=0.94, top=0.868, bottom=0.225,
    )

    # ---- a BOILED-Egg -----------------------------------------------------
    ax = fig.add_subplot(grid[0, 0])
    ax.add_patch(
        Ellipse((2.74, 65.8), width=6.28, height=131.6,
                facecolor="#fef3c7", edgecolor="#d97706", linewidth=1.0,
                alpha=0.55, zorder=0)
    )
    ax.add_patch(
        Ellipse((3.2, 39.5), width=5.6, height=79.0,
                facecolor="#fde68a", edgecolor="#b45309", linewidth=1.0,
                alpha=0.75, zorder=1)
    )
    zone_style = {
        "BBB+": (figures.TEAL, "o", "BBB+ / HIA+"),
        "HIA+": (figures.AMBER, "s", "HIA+"),
        "低渗透": (figures.RED, "x", "低渗透"),
    }
    for zone, (color, marker, label) in zone_style.items():
        subset = [item for item in profiles if item["zone"] == zone]
        if not subset:
            continue
        ax.scatter(
            [item["logp"] for item in subset],
            [item["tpsa"] for item in subset],
            s=[80 + 35 * max(0.0, (item["mw"] - 250) / 250) for item in subset],
            color=color, marker=marker, linewidth=0.9,
            edgecolors="white" if marker != "x" else None,
            label=label, zorder=4,
        )
    best = min(profiles, key=lambda item: item.get("affinity") or 999)
    ax.annotate(
        _short_name(best["name"]),
        (best["logp"], best["tpsa"]),
        xytext=(-64, 22), textcoords="offset points",
        fontsize=8.4, color=figures.INK,
        arrowprops={"arrowstyle": "-", "color": figures.MUTED, "linewidth": 0.8},
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white",
              "edgecolor": figures.GRID, "alpha": 0.95},
        zorder=6,
    )
    ax.set_xlim(-3.0, 7.0)
    ax.set_ylim(0, 165)
    ax.set_xlabel("WLOGP（Crippen）", fontsize=11)
    ax.set_ylabel("TPSA (Å²)", fontsize=11)
    ax.set_title("BOILED-Egg 风格吸收与入脑判别", loc="left", fontsize=13.5, pad=8)
    ax.grid(color=figures.GRID, linewidth=0.6)
    ax.legend(frameon=True, framealpha=0.92, edgecolor=figures.GRID,
              fontsize=7.8, loc="lower left")
    figures._compact_frame(ax)

    # ---- b CNS MPO --------------------------------------------------------
    ax = fig.add_subplot(grid[0, 1])
    ordered = sorted(profiles, key=lambda item: item["cns_mpo"], reverse=True)[:10]
    names = [_short_name(item["name"]) for item in ordered][::-1]
    scores = [item["cns_mpo"] for item in ordered][::-1]
    colors = [
        figures.TEAL if value >= 4.0 else figures.AMBER if value >= 3.0 else figures.RED
        for value in scores
    ]
    positions = np.arange(len(scores))
    ax.barh(positions, scores, color=colors, height=0.62)
    for index, value in enumerate(scores):
        ax.text(value + 0.08, index, f"{value:.2f}", va="center",
                fontsize=8.6, color=figures.INK, clip_on=True)
    ax.set_yticks(positions)
    ax.set_yticklabels(names)
    ax.set_ylim(-0.6, max(0.6, len(scores) - 0.4))
    ax.set_xlim(0, 6.8)
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6])
    ax.set_xlabel("CNS MPO（0–6，越高越有利）", fontsize=10)
    ax.set_title("CNS 多参数成药性", loc="left", fontsize=13.5, pad=8)
    ax.grid(axis="x", color=figures.GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    figures._compact_frame(ax)
    ax.tick_params(labelsize=8.0)

    # ---- c SAscore / logD window -----------------------------------------
    ax = fig.add_subplot(grid[1, 0])
    points = [item for item in profiles if item["sa"] is not None]
    if points:
        ax.scatter(
            [item["sa"] for item in points],
            [item["logd"] for item in points],
            s=[70 + 30 * item["cns_mpo"] for item in points],
            color=figures.TEAL, edgecolor="white", linewidth=0.9, zorder=3,
        )
        best_point = min(points, key=lambda item: item.get("affinity") or 999)
        ax.annotate(
            _short_name(best_point["name"]),
            (best_point["sa"], best_point["logd"]),
            xytext=(22, -34), textcoords="offset points", fontsize=8.2,
            color=figures.INK,
            arrowprops={"arrowstyle": "-", "color": figures.MUTED, "linewidth": 0.8},
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white",
                  "edgecolor": figures.GRID, "alpha": 0.95},
        )
    ax.set_xlim(0.5, 10.5)
    logd_values = [item["logd"] for item in profiles]
    ax.set_ylim(min(-1.0, min(logd_values) - 0.6), max(6.5, max(logd_values) + 0.6))
    ax.axhline(2.0, color=figures.AMBER, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.axvline(4.0, color=figures.AMBER, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.text(4.12, ax.get_ylim()[1] - 0.1, "SA = 4",
            fontsize=8.4, color=figures.AMBER, va="top")
    ax.text(8.5, 2.12, "logD = 2",
            fontsize=8.4, color=figures.AMBER, va="bottom")
    ax.set_xlabel("合成可及性 SAscore（1 易 – 10 难）", fontsize=11)
    ax.set_ylabel("logD7.4", fontsize=11)
    ax.set_title("可合成性与亲脂性窗口", loc="left", fontsize=13.5, pad=8)
    ax.grid(color=figures.GRID, linewidth=0.6)
    figures._compact_frame(ax)

    # ---- d alerts ---------------------------------------------------------
    ax = fig.add_subplot(grid[1, 1])
    ordered = sorted(
        profiles,
        key=lambda item: sum(_alert_counts(item).values()),
        reverse=True,
    )[:8][::-1]
    names = [_short_name(item["name"]) for item in ordered]
    pains = [_alert_counts(item).get("PAINS", 0) for item in ordered]
    brenk = [_alert_counts(item).get("Brenk", 0) for item in ordered]
    nih = [_alert_counts(item).get("NIH", 0) for item in ordered]
    positions = np.arange(len(ordered))
    ax.barh(positions, pains, color=figures.RED, height=0.60, label="PAINS")
    ax.barh(positions, brenk, left=pains, color=figures.AMBER, height=0.60,
            label="Brenk")
    ax.barh(positions, nih, left=np.array(pains) + np.array(brenk),
            color=figures.MUTED, height=0.60, label="NIH")
    ax.set_yticks(positions)
    ax.set_yticklabels(names)
    ax.set_ylim(-0.6, max(0.6, len(ordered) - 0.4))
    total = np.array(pains) + np.array(brenk) + np.array(nih)
    ax.set_xlim(0, max(1.0, float(total.max()) + 1.1))
    for index, value in enumerate(total):
        if value:
            ax.text(value + 0.12, index, str(int(value)), va="center",
                    fontsize=8.6, color=figures.INK, clip_on=True)
    ax.set_xlabel("结构警示命中数", fontsize=10.5)
    ax.set_title("PAINS / Brenk / NIH 警示", loc="left", fontsize=13.5, pad=8)
    ax.legend(frameon=False, fontsize=8.0, loc="lower right")
    ax.grid(axis="x", color=figures.GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    figures._compact_frame(ax)
    ax.tick_params(labelsize=8.0)

    for ax, letter, x in (
        (fig.axes[0], "a", -0.16),
        (fig.axes[1], "b", -0.16),
        (fig.axes[2], "c", -0.16),
        (fig.axes[3], "d", -0.16),
    ):
        figures._panel_tag(ax, letter, x=x, y=1.12)

    figures._caption(
        fig,
        figures._L(
            "BOILED-Egg 判别：横轴 Crippen WLOGP、纵轴 TPSA，内层椭圆为 BBB 渗透区，"
            "外层为 HIA 吸收区（Egan/Clark 经验阈值）。Wager CNS MPO 六项评分，"
            "4 分以上视为中枢神经系统候选。SAscore 与 logD7.4 虚线为常用可开发性窗口。"
            "PAINS/Brenk/NIH 警示仅提示人工复核，不等于化合物不可用；"
            "全部端点为描述符规则，不能替代实验测定。",
            "BOILED-Egg discrimination: Crippen WLOGP versus TPSA, with the inner "
            "ellipse marking BBB penetration and the outer ellipse HIA absorption "
            "(Egan/Clark rules). Wager CNS MPO scores six properties; 4 or more "
            "indicates a CNS candidate. Dashed lines mark the usual SAscore and "
            "logD7.4 windows. PAINS/Brenk/NIH alerts flag structures for manual "
            "review only; all endpoints are descriptor rules, not experimental "
            "measurements.",
        ),
        y=0.170, width=94, split=False,
    )
    with PdfPages(outpath) as pdf:
        pdf.savefig(fig, facecolor="white")
    plt.close(fig)
    return outpath


# ---------------------------------------------------------------------------
# Figure 12 : chemical space and scaffold diversity
# ---------------------------------------------------------------------------
def _single_ligand_chemspace(profile: dict, outpath: Path) -> Path:
    """Figure 12 in single-ligand mode.

    A clustered similarity heatmap and an MDS projection need at least two
    molecules.  The figure number is kept so that cross-references in the main
    report stay valid, but the panels show what can be computed for one
    molecule: the 2D structure, its Murcko scaffold and the distance of key
    descriptors to the Lipinski/Veber limits.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from rdkit.Chem import AllChem

    en = i18n.get_lang() == "en"
    mol = Chem.MolFromSmiles(profile["smiles"])
    fig = figures._a4_page(
        "图 12　化学空间与骨架多样性",
        (
            "Single-ligand view: scaffold, fingerprint occupancy and rule margins"
            if en
            else "单分子版式：骨架、指纹占用与类药规则边界"
        ),
        "Fig. 12　Chemical space and scaffold diversity (single ligand)",
    )
    grid = fig.add_gridspec(
        2, 2, height_ratios=[1.05, 0.95], hspace=0.40, wspace=0.30,
        left=0.10, right=0.93, top=0.845, bottom=0.300,
    )

    ax_mol = fig.add_subplot(grid[0, 0])
    if mol is not None:
        figures._draw_molecule(ax_mol, mol, compact=True)
    else:
        ax_mol.axis("off")
    ax_mol.set_title("二维结构" if not en else "2D structure",
                     loc="left", fontsize=13.0, pad=8)

    ax_scaffold = fig.add_subplot(grid[0, 1])
    scaffold_smiles = str(profile.get("scaffold") or "")
    scaffold = Chem.MolFromSmiles(scaffold_smiles) if scaffold_smiles else None
    if scaffold is not None and scaffold.GetNumAtoms():
        figures._draw_molecule(ax_scaffold, scaffold, compact=True)
    else:
        ax_scaffold.axis("off")
        ax_scaffold.text(
            0.5, 0.5,
            "未提取到 Murcko 骨架" if not en else "No Murcko scaffold",
            ha="center", va="center", fontsize=11, color=figures.MUTED,
        )
    ax_scaffold.set_title("Murcko 骨架" if not en else "Murcko scaffold",
                          loc="left", fontsize=13.0, pad=8)

    # ---- descriptor margins against Lipinski / Veber limits --------------
    ax_rule = fig.add_subplot(grid[1, :])
    limits = [
        ("MW", profile.get("mw"), 500.0),
        ("cLogP", profile.get("logp"), 5.0),
        ("TPSA", profile.get("tpsa"), 140.0),
        ("HBD", profile.get("hbd"), 5.0),
        ("HBA", profile.get("hba"), 10.0),
        ("RotB", profile.get("rotb"), 10.0),
    ]
    labels = []
    ratios = []
    colors = []
    for name, value, limit in limits:
        value = float(value or 0.0)
        ratio = value / limit if limit else 0.0
        labels.append(name)
        ratios.append(ratio)
        colors.append(figures.RED if ratio > 1.0 else figures.TEAL)
    y = np.arange(len(labels))
    ax_rule.barh(y, ratios, color=colors, height=0.58)
    right_edge = max(1.35, max(ratios) + 0.55)
    for index, (name, value, limit) in enumerate(limits):
        ax_rule.text(
            right_edge - 0.02, index,
            f"{float(value or 0):.2f} / {limit:g}",
            va="center", ha="right",
            fontsize=10.2, color=figures.INK,
        )
    ax_rule.axvline(1.0, color=figures.AMBER, linewidth=1.3,
                    linestyle=(0, (4, 3)))
    ax_rule.set_yticks(y)
    ax_rule.set_yticklabels(labels, fontsize=11)
    ax_rule.invert_yaxis()
    ax_rule.set_xlim(0, right_edge)
    ax_rule.set_xlabel(
        "实测值 / 规则上限" if not en else "Value / rule limit", fontsize=12
    )
    ax_rule.set_title(
        "类药规则边界占用（>1 为超出）" if not en else
        "Distance to drug-likeness limits (>1 exceeds)",
        loc="left", fontsize=13.0, pad=8,
    )
    figures._compact_frame(ax_rule)

    bits = 0
    if mol is not None:
        try:
            generator = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
            bits = generator.GetFingerprint(mol).GetNumOnBits()
        except Exception:
            bits = 0
    stats = (
        f"Morgan 指纹（半径 2，2048 bit）占用 {bits} 个位点 · "
        f"重原子 {profile.get('heavy', 0)} 个 · Fsp3 "
        f"{float(profile.get('fsp3') or 0):.2f}"
        if not en
        else f"Morgan fingerprint (radius 2, 2048 bit): {bits} bits set · "
        f"{profile.get('heavy', 0)} heavy atoms · Fsp3 "
        f"{float(profile.get('fsp3') or 0):.2f}"
    )
    fig.text(0.07, 0.878, stats, fontsize=9.8, color=figures.MUTED,
             va="bottom")
    figures._caption(
        fig,
        "单配体模式：相似性聚类热图与多维尺度投影需要至少两个分子，因此图 12 改为"
        "单分子版式，分别给出二维结构、Murcko 骨架与关键描述符相对 Lipinski/Veber "
        "上限的占用比例。条形超过虚线表示该项超出常规类药空间，应结合配体效率与 QED "
        "判断是否值得继续投入；骨架信息可用于判断后续骨架跃迁的起点。"
        if not en
        else "Single-ligand mode: the clustered similarity heatmap and MDS projection "
        "require at least two molecules, so Figure 12 shows the 2D structure, the "
        "Murcko scaffold and the ratio of key descriptors to the Lipinski/Veber "
        "limits. Bars beyond the dashed line exceed the conventional drug-like "
        "space and should be judged together with ligand efficiency and QED.",
        y=0.170,
    )
    outpath = Path(outpath)
    with PdfPages(outpath) as pdf:
        pdf.savefig(fig, facecolor="white")
    plt.close(fig)
    return outpath


def chemspace_pdf(records: list[dict], outpath: Path) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform
    from rdkit.Chem import AllChem, DataStructs

    figures._style_cjk()
    profiles = series_profiles(records)
    mols = []
    for item in profiles:
        mol = Chem.MolFromSmiles(item["smiles"])
        if mol is not None:
            mols.append((item, mol))
    if len(mols) < 2:
        if not profiles:
            raise ValueError("没有可用于化学空间分析的分子（SMILES 均无法解析）")
        return _single_ligand_chemspace(profiles[0], outpath)
    mols = mols[:18]
    names = [_short_name(item["name"], 10) for item, _mol in mols]
    generator = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = [generator.GetFingerprint(mol) for _item, mol in mols]
    matrix = np.zeros((len(mols), len(mols)))
    for i in range(len(mols)):
        for j in range(len(mols)):
            matrix[i, j] = DataStructs.TanimotoSimilarity(
                fingerprints[i], fingerprints[j]
            )
    distance = np.clip(1.0 - matrix, 0, 1)
    condensed = squareform(distance, checks=False)
    order = dendrogram(linkage(condensed, method="average"),
                       no_plot=True)["leaves"]
    ordered = matrix[np.ix_(order, order)]
    ordered_names = [names[index] for index in order]

    fig = figures._a4_page(
        "图 12　化学空间与骨架多样性",
        "Tanimoto 相似性聚类、Murcko 骨架与多维尺度投影",
        "Fig. 12　Chemical space and scaffold diversity",
    )
    grid = fig.add_gridspec(
        2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.0, 0.58],
        hspace=0.36, wspace=0.34,
        left=0.13, right=0.88, top=0.838, bottom=0.285,
    )

    # ---- a clustered Tanimoto heatmap ------------------------------------
    ax_heat = fig.add_subplot(grid[0, 0])
    image = ax_heat.imshow(ordered, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax_heat.set_xticks(range(len(ordered_names)))
    ax_heat.set_xticklabels(ordered_names, rotation=90, fontsize=8.2)
    ax_heat.set_yticks(range(len(ordered_names)))
    ax_heat.set_yticklabels(ordered_names, fontsize=8.2)
    ax_heat.set_title("Tanimoto 相似性聚类热图", loc="left", fontsize=15, pad=8)
    for i in range(len(ordered_names)):
        for j in range(len(ordered_names)):
            value = ordered[i, j]
            if len(ordered_names) <= 14:
                ax_heat.text(
                    j, i, f"{value:.2f}", ha="center", va="center",
                    fontsize=7.2,
                    color="white" if value > 0.62 else figures.INK,
                )

    # ---- b MDS chemical-space map ---------------------------------------
    ax_mds = fig.add_subplot(grid[0, 1])
    count = len(matrix)
    centering = np.eye(count) - np.ones((count, count)) / count
    gram = -0.5 * centering @ (distance ** 2) @ centering
    values, vectors = np.linalg.eigh(gram)
    idx = np.argsort(values)[::-1]
    coords = vectors[:, idx[:2]] * np.sqrt(np.clip(values[idx[:2]], 0, None))
    affinity = np.array([
        item.get("affinity") if item.get("affinity") is not None else np.nan
        for item, _mol in mols
    ])
    scatter = ax_mds.scatter(
        coords[:, 0], coords[:, 1],
        c=affinity, cmap="viridis_r", s=85, edgecolor="white", linewidth=1.0,
    )
    best_index = int(np.nanargmin(affinity)) if np.isfinite(affinity).any() else 0
    ax_mds.annotate(
        names[best_index], (coords[best_index, 0], coords[best_index, 1]),
        xytext=(8, 10), textcoords="offset points", fontsize=9.5,
        color=figures.INK,
        arrowprops={"arrowstyle": "-", "color": figures.MUTED, "linewidth": 0.8},
    )
    colorbar = fig.colorbar(scatter, ax=ax_mds, fraction=0.045, pad=0.02,
                            shrink=0.82)
    colorbar.set_label("Vina (kcal/mol)", fontsize=9.5)
    colorbar.ax.tick_params(labelsize=8.6)
    ax_mds.set_title("化学空间投影（经典 MDS）", loc="left", fontsize=15, pad=8)
    ax_mds.set_xlabel("MDS 1", fontsize=10.5, color=figures.MUTED)
    ax_mds.set_ylabel("MDS 2", fontsize=10.5, color=figures.MUTED)
    ax_mds.set_xticks([])
    ax_mds.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax_mds.spines[side].set_color(figures.GRID)
    ax_mds.grid(color=figures.GRID, linewidth=0.5, alpha=0.6)

    # ---- c Murcko scaffolds ----------------------------------------------
    scaffold_counts = {}
    scaffold_mol = {}
    for item, mol in mols:
        scaffold = item.get("scaffold") or ""
        if not scaffold:
            continue
        scaffold_counts[scaffold] = scaffold_counts.get(scaffold, 0) + 1
        if scaffold not in scaffold_mol:
            scaffold_mol[scaffold] = Chem.MolFromSmiles(scaffold)
    top_scaffolds = sorted(
        scaffold_counts.items(), key=lambda pair: pair[1], reverse=True
    )[:4]
    sub = grid[1, :].subgridspec(1, max(1, len(top_scaffolds)), wspace=0.20)
    for index, (scaffold, frequency) in enumerate(top_scaffolds):
        sub_ax = fig.add_subplot(sub[0, index])
        mol = scaffold_mol.get(scaffold)
        if mol is not None and mol.GetNumAtoms():
            figures._draw_molecule(sub_ax, mol, compact=True)
        sub_ax.set_title(f"n = {frequency}", fontsize=10.5,
                         color=figures.MUTED, pad=2)
    unique = len(scaffold_counts)
    nearest = []
    for i in range(len(matrix)):
        row = np.delete(matrix[i], i)
        if len(row):
            nearest.append(float(row.max()))
    stats_text = (
        f"{len(mols)} 个分子 · {unique} 个唯一 Murcko 骨架 · "
        f"骨架多样性 {unique / max(len(mols), 1):.2f} · "
        f"平均最近邻相似度 {np.mean(nearest):.2f}"
        if nearest else ""
    )
    fig.text(0.07, 0.883, stats_text, fontsize=10.5, color=figures.MUTED,
             va="bottom")
    figures._caption(
        fig,
        figures._L(
            "Morgan 指纹（半径 2，2048 bit）的 Tanimoto 相似矩阵，按平均连接层次聚类排序，"
            "颜色越深表示越相似。由同一相似矩阵经经典多维尺度（MDS）投影得到的化学空间，"
            "颜色表示 Vina 打分。出现频率最高的四个 Murcko 骨架；骨架多样性 = 唯一骨架数 / 分子数。"
            "该图用于判断候选是集中在单一骨架还是覆盖多个化学型，"
            "为骨架跃迁与专利空间规避提供依据。",
            "Tanimoto similarity matrix of Morgan fingerprints (radius 2, 2048 bit), ordered by "
            "average-linkage hierarchical clustering, with darker colour indicating greater similarity. "
            "Chemical space obtained by classical multidimensional scaling (MDS) of the same matrix, "
            "with colour showing the Vina score. The four most frequent Murcko scaffolds; scaffold "
            "diversity = unique scaffolds / molecules. The figure shows whether candidates concentrate "
            "on a single scaffold or span several chemotypes, providing a basis for scaffold hopping "
            "and patent-space avoidance.",
        ),
        width=104,
        y=0.205,
        split=False,
    )
    figures._panel_tag(ax_heat, "a", x=-0.22, y=1.06)
    figures._panel_tag(ax_mds, "b", x=-0.16, y=1.06)
    if top_scaffolds:
        figures._panel_tag(
            fig.axes[-len(top_scaffolds)], "c", x=-0.10, y=1.18
        )
    with PdfPages(outpath) as pdf:
        pdf.savefig(fig, facecolor="white")
    plt.close(fig)
    return Path(outpath)


# ---------------------------------------------------------------------------
# Figure 13 : metabolic soft spots
# ---------------------------------------------------------------------------
def _soft_spots(mol: Chem.Mol) -> dict:
    atoms: list[int] = []
    labels: list[str] = []
    score = 0.0
    for label, smarts, weight, limit, site in _SOM_RULES:
        matches = _matches(mol, smarts)
        if not matches:
            continue
        labels.append(label)
        score += weight * min(len(matches), limit)
        for match in matches[:limit]:
            atom_index = match[site]
            if atom_index not in atoms:
                atoms.append(atom_index)
    return {
        "atoms": atoms[:8],
        "labels": labels,
        "score": round(min(100.0, 14.0 * score), 1),
    }


def metabolism_rows(records: list[dict]) -> list[dict]:
    """Metabolic soft-spot rows for the expert-analysis panel."""
    rows = []
    for item in series_profiles(records):
        mol = Chem.MolFromSmiles(item.get("smiles") or "")
        if mol is None:
            continue
        spots = _soft_spots(mol)
        rows.append(
            {
                **item,
                "soft_spot_score": spots["score"],
                "soft_spot_labels": spots["labels"],
                "soft_spot_atoms": spots["atoms"],
            }
        )
    rows.sort(key=lambda row: row["soft_spot_score"], reverse=True)
    return rows


def metabolism_pdf(records: list[dict], outpath: Path) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    figures._style_cjk()
    profiles = series_profiles(records)
    entries = []
    for item in profiles:
        mol = Chem.MolFromSmiles(item["smiles"])
        if mol is None:
            continue
        spots = _soft_spots(mol)
        entries.append({"profile": item, "mol": mol, "spots": spots})
    if not entries:
        raise ValueError("没有可用于代谢软点分析的分子")
    entries.sort(key=lambda entry: entry["spots"]["score"], reverse=True)

    fig = figures._a4_page(
        "图 13　代谢软点预测",
        "规则式 CYP 氧化与脱烷基位点标注",
        "Fig. 13　Metabolic soft spots",
    )
    grid = fig.add_gridspec(
        2, 1, height_ratios=[0.46, 1.0], hspace=0.34,
        left=0.17, right=0.94, top=0.862, bottom=0.245,
    )

    # ---- a risk ranking ---------------------------------------------------
    bar_ax = fig.add_subplot(grid[0, 0])
    ax = bar_ax
    top = entries[:10]
    names = [_short_name(entry["profile"]["name"]) for entry in top][::-1]
    scores = [entry["spots"]["score"] for entry in top][::-1]
    colors = [
        figures.RED if value >= 60 else figures.AMBER if value >= 35 else figures.TEAL
        for value in scores
    ]
    ax.barh(names, scores, color=colors, height=0.62)
    for index, value in enumerate(scores):
        ax.text(value + 1.2, index, f"{value:.0f}", va="center",
                fontsize=8.4, color=figures.INK)
    ax.set_xlim(0, 112)
    ax.set_xlabel("代谢软点风险分", fontsize=10.5)
    ax.set_title("系列风险排序", loc="left", fontsize=13.5, pad=8)
    ax.grid(axis="x", color=figures.GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    figures._compact_frame(ax)
    ax.tick_params(labelsize=8.0)

    # ---- b 2x3 structures -------------------------------------------------
    sub = grid[1, 0].subgridspec(2, 3, hspace=0.36, wspace=0.22)
    structure_axes = []
    for index, entry in enumerate(entries[:6]):
        ax = fig.add_subplot(sub[index // 3, index % 3])
        structure_axes.append(ax)
        coords = figures._draw_molecule(ax, entry["mol"], compact=True)
        for atom_index in entry["spots"]["atoms"]:
            point = coords.get(atom_index)
            if point is None:
                continue
            ax.scatter(
                [point[0]], [point[1]], s=110, facecolor="none",
                edgecolor=figures.RED, linewidth=1.7, zorder=6,
            )
        labels_list = entry["spots"]["labels"][:2]
        if i18n.get_lang() == "en":
            labels = ", ".join(
                _SOM_LABEL_EN.get(label, i18n.tr(label)) for label in labels_list
            )
        else:
            labels = "、".join(labels_list)
        title = (
            f"{_short_name(entry['profile']['name'], 10)} · "
            f"风险 {entry['spots']['score']:.0f}"
        )
        if labels:
            title += f"\n{labels}"
        ax.set_title(title, fontsize=8.4, pad=5, color=figures.INK)
    fig.text(
        0.42, 0.862,
        "红圈：规则命中的潜在氧化 / 脱烷基位点",
        fontsize=9.6, color=figures.MUTED, va="top",
    )
    figures._panel_tag(bar_ax, "a", x=-0.14, y=1.10)
    if structure_axes:
        figures._panel_tag(structure_axes[0], "b", x=-0.10, y=1.16)
    figures._caption(
        fig,
        "按规则命中加权得到的代谢软点风险分（0–100）；"
        "风险最高的六个分子的二维结构，红圈标出可能发生 CYP 氧化、N-/O-脱烷基、"
        "硫氧化或酚羟基结合的原子。规则基于常见 CYP 反应类型，"
        "不区分同工酶、不预测代谢速率，只用于优先安排代谢稳定性实验与结构封闭策略。",
        y=0.155,
    )
    with PdfPages(outpath) as pdf:
        pdf.savefig(fig, facecolor="white")
    plt.close(fig)
    return Path(outpath)
