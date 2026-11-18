"""Descriptor-based physicochemical, drug-likeness and ADMET profiling.

The models here are transparent descriptor rules and published one-line
correlations (Delaney ESOL, Clark logBB, ...).  They are meant for triage and
ranking inside a docking campaign, not as a replacement for a validated ADMET
platform.  Every endpoint carries its own confidence note so the UI can label
the result honestly.
"""
from __future__ import annotations

import math
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdMolDescriptors

from . import i18n
from .utils import safe_name

RULE_LABELS = {
    "lipinski": "Lipinski 五规则",
    "veber": "Veber 口服规则",
    "egan": "Egan 吸收规则",
    "leadlike": "类先导规则",
}

_SHORT_NAMES = {
    "人肠吸收 HIA": "HIA 吸收",
    "Caco-2 通透性": "Caco-2",
    "血脑屏障 logBB": "logBB",
    "血浆蛋白结合": "蛋白结合",
    "hERG 抑制风险": "hERG",
    "CYP3A4 抑制风险": "CYP3A4",
    "CYP2D6 抑制风险": "CYP2D6",
    "致突变（Ames）风险": "Ames",
    "肝清除率": "清除率",
    "预测半衰期": "半衰期",
    "P-gp 外排风险": "P-gp",
    "水溶性 logS": "logS",
}

_ALERTS = [
    ("nitro", "硝基芳环", "[$([N+](=O)[O-])]"),
    ("aniline", "芳香伯/仲胺", "[NX3;H2,H1][c]"),
    ("epoxide", "环氧化物", "C1OC1"),
    ("aziridine", "氮丙啶", "C1CN1"),
    ("michael", "Michael 受体", "[CX3]=[CX3][CX3]=[OX1]"),
    ("hydrazine", "肼/腙", "[NX3][NX3]"),
    ("nitroso", "亚硝基", "[NX2]=O"),
    ("aldehyde", "脂肪醛", "[CX3H1](=O)[#6]"),
    ("alkyl_halide", "烷基卤", "[CX4][Cl,Br,I]"),
]

_BASIC_AMINE = Chem.MolFromSmarts(
    "[NX3;H2,H1,H0;!$(N-[!#6]);!$(N-a);!$(N-C=O);!$(N-S);!$(N-C=[O,N,S])]"
)
_ACID = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _verdict(score: float, good: tuple = (70, 45)) -> str:
    if score >= good[0]:
        return "优"
    if score >= good[1]:
        return "中"
    return "差"


def esol(logp: float, mw: float, rotb: int, aromatic_fraction: float) -> float:
    """Delaney (2004) ESOL intrinsic aqueous solubility, log S (mol/L)."""
    return 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rotb - 0.74 * aromatic_fraction


def _solubility_class(logs: float) -> str:
    if logs >= -2:
        return "易溶"
    if logs >= -4:
        return "中等"
    if logs >= -6:
        return "难溶"
    return "极难溶"


def _solubility_verdict(logs: float) -> str:
    """Verdict follows the ESOL solubility class, not an arbitrary mapping."""
    if logs >= -2:
        return "优"
    if logs >= -4:
        return "中"
    return "差"


def _solubility_score(logs: float) -> float:
    """0-100 display score that stays consistent with the ESOL class bands."""
    if logs >= -2:
        return float(_clamp(90 + (logs + 2) * 5, 90, 100))
    if logs >= -4:
        return float(_clamp(46 + (logs + 4) * 11.5, 46, 69))
    if logs >= -6:
        return float(_clamp(25 + (logs + 6) * 10, 25, 45))
    return float(_clamp(20 + (logs + 6) * 8, 5, 20))


def _tox_alerts(mol: Chem.Mol) -> list[str]:
    hits = []
    for _key, label, smarts in _ALERTS:
        query = Chem.MolFromSmarts(smarts)
        if query is not None and mol.HasSubstructMatch(query):
            hits.append(label)
    return hits


def _complexity(mol: Chem.Mol) -> float:
    """Rough synthetic-accessibility index (0 easy, 100 hard)."""
    stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    rings = rdMolDescriptors.CalcNumRings(mol)
    bridge = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    charge = abs(Chem.GetFormalCharge(mol))
    score = 12 * stereo + 8 * rings + 12 * bridge + 14 * spiro + 6 * charge
    return round(_clamp(score, 0, 100), 1)


def profile_mol(mol: Chem.Mol, name: str = "", affinity: float | None = None) -> dict:
    """Full property / ADMET panel for one molecule."""
    mol = Chem.RemoveHs(mol)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("分子为空")
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass

    heavy = int(mol.GetNumHeavyAtoms())
    mw = float(Descriptors.MolWt(mol))
    logp = float(Descriptors.MolLogP(mol))
    tpsa = float(Descriptors.TPSA(mol))
    hbd = int(rdMolDescriptors.CalcNumHBD(mol))
    hba = int(rdMolDescriptors.CalcNumHBA(mol))
    rotb = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
    rings = int(rdMolDescriptors.CalcNumRings(mol))
    aromatic_rings = int(rdMolDescriptors.CalcNumAromaticRings(mol))
    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    aromatic_fraction = aromatic_atoms / heavy if heavy else 0.0
    fsp3 = float(rdMolDescriptors.CalcFractionCSP3(mol))
    charge = int(Chem.GetFormalCharge(mol))
    heavy = max(heavy, 1)

    try:
        formula = rdMolDescriptors.CalcMolFormula(mol)
    except Exception:
        formula = ""
    try:
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        smiles = ""

    logs = esol(logp, mw, rotb, aromatic_fraction)

    # ------------------------------------------------------------------ rules
    lipinski_violations = sum(
        [mw > 500, logp > 5, hbd > 5, hba > 10]
    )
    veber_pass = rotb <= 10 and tpsa <= 140
    egan_pass = tpsa <= 131.6 and logp <= 5.88
    leadlike_pass = 250 <= mw <= 450 and logp <= 3.5 and rotb <= 7
    rules = {
        "lipinski": {
            "pass": lipinski_violations <= 1,
            "violations": lipinski_violations,
            "detail": f"MW {mw:.0f} · logP {logp:.1f} · HBD {hbd} · HBA {hba}",
        },
        "veber": {
            "pass": veber_pass,
            "violations": int(rotb > 10) + int(tpsa > 140),
            "detail": f"可旋转键 {rotb} · TPSA {tpsa:.0f} Å²",
        },
        "egan": {
            "pass": egan_pass,
            "violations": int(tpsa > 131.6) + int(logp > 5.88),
            "detail": f"TPSA {tpsa:.0f} · logP {logp:.1f}",
        },
        "leadlike": {
            "pass": leadlike_pass,
            "violations": 0 if leadlike_pass else 1,
            "detail": f"MW {mw:.0f} · logP {logp:.1f} · 可旋转键 {rotb}",
        },
    }

    try:
        qed = float(QED.qed(mol))
    except Exception:
        qed = 0.0

    # ------------------------------------------------------------------ ADMET
    # Absorption: TPSA/MW penalty, lipophilicity bonus, capped at 99 %.
    hia = _clamp(
        105 - 0.30 * tpsa - 0.010 * max(0.0, mw - 250) + 5.0 * min(max(logp, -2.0), 4.0),
        12, 99,
    )
    # Caco-2 apparent permeability, log(Papp / 10^-6 cm s-1).
    caco_log = _clamp(-4.35 + 0.11 * logp - 0.009 * tpsa - 0.0018 * mw, -6.6, -3.4)
    # Clark (1999) log BB.
    logbb = -0.0148 * tpsa + 0.152 * logp + 0.139
    # Plasma protein binding, fraction bound (%).
    ppb = _clamp(100 * (1 - math.exp(-0.55 * max(0.0, logp - 1.0))) + 8, 5, 99.5)

    basic = bool(_BASIC_AMINE and mol.HasSubstructMatch(_BASIC_AMINE))
    acidic = bool(_ACID and mol.HasSubstructMatch(_ACID))
    alerts = _tox_alerts(mol)

    herg = _clamp(
        max(0.0, logp - 3.0) * 16 + (26 if basic else 0) + max(0.0, mw - 420) * 0.10,
        0, 95,
    )
    cyp3a4 = _clamp(
        max(0.0, logp - 3.2) * 18 + max(0, aromatic_rings - 2) * 10 + (8 if mw > 400 else 0),
        0, 95,
    )
    cyp2d6 = _clamp((30 if basic else 4) + max(0.0, logp - 2.0) * 12, 0, 95)
    ames = _clamp(10 + 26 * len(alerts), 5, 95)
    # Hepatic clearance heuristic, mL min-1 kg-1.
    clearance = _clamp(3.0 + 2.4 * max(0.0, logp) + 0.6 * max(0, aromatic_rings - 1), 1, 30)
    pgp = _clamp(
        18 + (22 if (basic and mw > 380) else 0) + max(0.0, tpsa - 90) * 0.55
        + max(0, aromatic_rings - 2) * 6,
        5, 95,
    )
    half_life = _clamp(0.9 + 0.35 * logp - 0.006 * (hia - 70) - 0.03 * clearance, 0.4, 12)

    admet = [
        {
            "key": "hia",
            "name": "人肠吸收 HIA",
            "value": round(hia, 1),
            "unit": "%",
            "score": round(hia, 1),
            "note": "TPSA、MW 与 logP 的经验吸收模型",
        },
        {
            "key": "caco2",
            "name": "Caco-2 通透性",
            "value": round(caco_log, 2),
            "unit": "log Papp",
            "score": round(_clamp(100 - abs(caco_log + 4.4) * 42, 0, 100), 1),
            "note": "被动扩散通透性，-4.5 以上较理想",
        },
        {
            "key": "logbb",
            "name": "血脑屏障 logBB",
            "value": round(logbb, 2),
            "unit": "",
            "score": round(_clamp(100 - abs(logbb - 0.2) * 45, 0, 100), 1),
            "note": "Clark 方程；> 0.3 提示可入脑",
        },
        {
            "key": "ppb",
            "name": "血浆蛋白结合",
            "value": round(ppb, 1),
            "unit": "%",
            "score": round(_clamp(100 - ppb * 0.85, 0, 100), 1),
            "note": "结合率越低游离药物越多",
        },
        {
            "key": "herg",
            "name": "hERG 抑制风险",
            "value": round(herg, 1),
            "unit": "/100",
            "score": round(100 - herg, 1),
            "note": "亲脂性与碱性胺是主要风险因素（0–100 相对风险评分，非概率）",
        },
        {
            "key": "cyp3a4",
            "name": "CYP3A4 抑制风险",
            "value": round(cyp3a4, 1),
            "unit": "/100",
            "score": round(100 - cyp3a4, 1),
            "note": "大体积亲脂芳环体系风险较高（0–100 相对风险评分，非概率）",
        },
        {
            "key": "cyp2d6",
            "name": "CYP2D6 抑制风险",
            "value": round(cyp2d6, 1),
            "unit": "/100",
            "score": round(100 - cyp2d6, 1),
            "note": "碱性氮与亲脂性共同驱动（0–100 相对风险评分，非概率）",
        },
        {
            "key": "ames",
            "name": "致突变（Ames）风险",
            "value": round(ames, 1),
            "unit": "/100",
            "score": round(100 - ames, 1),
            "note": "基于结构警示子（0–100 相对风险评分，非概率）",
        },
        {
            "key": "clearance",
            "name": "肝清除率",
            "value": round(clearance, 1),
            "unit": "mL/min/kg",
            "score": round(_clamp(100 - abs(clearance - 12) * 5, 0, 100), 1),
            "note": "中等清除更利于维持暴露",
        },
        {
            "key": "half_life",
            "name": "预测半衰期",
            "value": round(half_life, 1),
            "unit": "h",
            "score": round(_clamp(100 - abs(half_life - 6) * 12, 0, 100), 1),
            "note": "经验估算，仅用于排序",
        },
        {
            "key": "pgp",
            "name": "P-gp 外排风险",
            "value": round(pgp, 1),
            "unit": "/100",
            "score": round(100 - pgp, 1),
            "note": "高 TPSA 与碱性氮增加外排概率（0–100 相对风险评分，非概率）",
        },
        {
            "key": "solubility",
            "name": "水溶性 logS",
            "value": round(logs, 2),
            "unit": "log mol/L",
            "score": round(_solubility_score(logs), 1),
            "verdict": _solubility_verdict(logs),
            "note": f"ESOL 预测 · {_solubility_class(logs)}",
        },
    ]
    for item in admet:
        item.setdefault("verdict", _verdict(item["score"]))

    radar = [
        ("吸收", (hia + admet[1]["score"]) / 2),
        ("分布", (admet[3]["score"] + admet[2]["score"]) / 2),
        ("代谢", (admet[5]["score"] + admet[6]["score"]) / 2),
        ("排泄", (admet[8]["score"] + admet[9]["score"]) / 2),
        ("毒性", (admet[4]["score"] + admet[7]["score"]) / 2),
        ("类药性", qed * 100),
    ]
    radar = [{"axis": name, "score": round(float(_clamp(value, 0, 100)), 1)} for name, value in radar]

    physchem = {
        "mw": round(mw, 2),
        "logp": round(logp, 2),
        "logs": round(logs, 2),
        "tpsa": round(tpsa, 1),
        "hbd": hbd,
        "hba": hba,
        "rotb": rotb,
        "rings": rings,
        "aromatic_rings": aromatic_rings,
        "fsp3": round(fsp3, 2),
        "heavy": heavy,
        "mr": round(float(Descriptors.MolMR(mol)), 1),
        "charge": charge,
        "formula": formula,
    }
    druglike = {
        "qed": round(qed, 3),
        "complexity": _complexity(mol),
        "rules": rules,
        "alerts": alerts,
        "basic_center": basic,
        "acidic_center": acidic,
    }

    result = {
        "name": name or (mol.GetProp("_Name") if mol.HasProp("_Name") else "ligand"),
        "smiles": smiles,
        "physchem": physchem,
        "druglike": druglike,
        "admet": admet,
        "radar": radar,
        "method": "RDKit 描述符 + 公开经验相关式（ESOL / Clark）",
    }
    if affinity is not None:
        result["efficiency"] = efficiency(result, float(affinity))
    return result


def efficiency(profile: dict, affinity: float) -> dict:
    """Ligand efficiency metrics derived from a Vina score.

    Vina scores are converted to an approximate pKd with pKd = -ΔG / 1.364,
    which keeps the numbers on the same scale as experimental affinities.
    """
    heavy = max(1, int(profile["physchem"]["heavy"]))
    logp = float(profile["physchem"]["logp"])
    p_kd = -float(affinity) / 1.364
    le = -float(affinity) / heavy
    return {
        "p_kd": round(p_kd, 2),
        "le": round(le, 3),
        "lipe": round(p_kd - logp, 2),
        "bei": round(p_kd / max(1.0, logp), 2),
        "sei": round(p_kd - logp - 0.5 * math.log10(max(1.0, p_kd * 10)), 2),
    }


def profile_smiles(smiles: str, name: str = "", affinity: float | None = None) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"无法解析 SMILES：{smiles[:80]}")
    return profile_mol(mol, name=name or smiles[:24], affinity=affinity)


def compact(profile: dict) -> dict:
    """Small property block used in candidate tables."""
    phys = profile["physchem"]
    drug = profile["druglike"]
    rules = drug["rules"]
    return {
        "mw": phys["mw"],
        "logp": phys["logp"],
        "logs": phys["logs"],
        "tpsa": phys["tpsa"],
        "hbd": phys["hbd"],
        "hba": phys["hba"],
        "rotb": phys["rotb"],
        "qed": drug["qed"],
        "alerts": len(drug["alerts"]),
        "lipinski": rules["lipinski"]["pass"],
        "veber": rules["veber"]["pass"],
        "efficiency": profile.get("efficiency", {}),
    }


# ---------------------------------------------------------------------------
# vector figure
# ---------------------------------------------------------------------------
def admet_figure_pdf(profile: dict, outpath: Path) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    from . import figures

    figures._style_cjk()
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    phys = profile["physchem"]
    drug = profile["druglike"]
    admet = profile["admet"]
    english = i18n.get_lang() == "en"
    radar_labels = {
        "吸收": "Absorption", "分布": "Distribution", "代谢": "Metabolism",
        "排泄": "Excretion", "毒性": "Toxicity", "类药性": "Drug-likeness",
    }
    verdicts = {"优": "excellent", "中": "moderate", "差": "poor"}

    fig = plt.figure(figsize=(8.27, 11.69))
    grid = fig.add_gridspec(
        3, 2, height_ratios=[1.05, 0.95, 0.82],
        hspace=0.45, wspace=0.40,
        left=0.20, right=0.94, top=0.825, bottom=0.07,
    )
    fig.text(
        0.07, 0.955,
        "ADMET and drug-likeness prediction" if english else "ADMET 与类药性预测",
        fontsize=20, fontweight="bold", color=figures.INK,
    )
    alerts_text = (
        f"{len(drug.get('alerts', []))} structural alerts" if english
        else f"{len(drug.get('alerts', []))} 个结构警示"
    )
    fig.text(
        0.07, 0.928,
        f"{profile.get('name', 'ligand')}   ·   {phys.get('formula', '')}   ·   {alerts_text}",
        fontsize=11.5, color=figures.MUTED,
    )

    # ---- radar ---------------------------------------------------------
    ax = fig.add_subplot(grid[0, 0], polar=True)
    labels = [item["axis"] for item in profile["radar"]]
    if english:
        labels = [radar_labels.get(label, label) for label in labels]
    values = [item["score"] for item in profile["radar"]]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    values += values[:1]
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.plot(angles, values, color=figures.TEAL, linewidth=2.1)
    ax.fill(angles, values, color=figures.TEAL, alpha=0.18)
    ax.scatter(angles[:-1], values[:-1], s=38, color=figures.TEAL, zorder=3)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8, color=figures.MUTED)
    ax.set_title("Overall drug-likeness radar" if english else "综合成药性雷达", fontsize=14, pad=22)
    ax.grid(color=figures.GRID)

    # ---- property table --------------------------------------------------
    ax = fig.add_subplot(grid[0, 1])
    ax.axis("off")
    rows = [
        ("Molecular weight" if english else "分子量 MW", f"{phys['mw']:.1f}"),
        ("cLogP", f"{phys['logp']:.2f}"),
        ("ESOL logS", f"{phys['logs']:.2f}"),
        ("TPSA", f"{phys['tpsa']:.1f} Å²"),
        (
            "HBD / HBA" if english else "氢键供体 / 受体",
            f"{phys['hbd']} / {phys['hba']}",
        ),
        ("Rotatable bonds" if english else "可旋转键", f"{phys['rotb']}"),
        ("Rings / aromatic" if english else "环 / 芳环", f"{phys['rings']} / {phys['aromatic_rings']}"),
        ("Fsp³", f"{phys['fsp3']:.2f}"),
        ("Molar refractivity" if english else "摩尔折射率", f"{phys['mr']:.1f}"),
        ("QED", f"{drug['qed']:.3f}"),
        ("Synthetic complexity" if english else "合成复杂度", f"{drug['complexity']:.0f} / 100"),
    ]
    table = ax.table(
        cellText=[[label, value] for label, value in rows],
        colWidths=[0.62, 0.38],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.8)
    table.scale(1, 1.52)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(figures.GRID)
        cell.set_linewidth(0.7)
        if col == 0:
            cell.set_text_props(color=figures.MUTED)
        else:
            cell.set_text_props(color=figures.INK, fontweight="bold")
            cell.set_facecolor("#f8fafc")
    ax.set_title("Physicochemical properties" if english else "理化性质", fontsize=14, pad=12, loc="left")

    # ---- ADMET bars -------------------------------------------------------
    ax = fig.add_subplot(grid[1, :])
    ordered = list(reversed(admet))
    names = [_SHORT_NAMES.get(item["name"], item["name"]) for item in ordered]
    if english:
        names = [
            name.replace("HIA 吸收", "HIA absorption")
            .replace("蛋白结合", "Protein binding")
            .replace("清除率", "Clearance")
            .replace("半衰期", "Half-life")
            for name in names
        ]
    scores = [item["score"] for item in ordered]
    colors = [
        figures.TEAL if value >= 70 else figures.AMBER if value >= 45 else figures.RED
        for value in scores
    ]
    bars = ax.barh(names, scores, color=colors, height=0.62, alpha=0.9)
    for bar, item in zip(bars, ordered):
        value = item["value"]
        unit = item["unit"]
        verdict = verdicts.get(item["verdict"], item["verdict"]) if english else item["verdict"]
        text = f"{value:g}{(' ' + unit) if unit else ''}  ·  {verdict}"
        ax.text(
            min(99, bar.get_width() + 1.4), bar.get_y() + bar.get_height() / 2,
            text, va="center", fontsize=9.5, color=figures.INK,
        )
    ax.set_xlim(0, 150)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Predicted quality score (higher is better)" if english else "预测质量分数（越高越有利）", fontsize=11)
    ax.set_title("ADMET endpoint predictions" if english else "ADMET 端点预测", fontsize=14, loc="left", pad=10)
    ax.grid(axis="x", color=figures.GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=10)

    # ---- rules + alerts ---------------------------------------------------
    ax = fig.add_subplot(grid[2, 0])
    ax.axis("off")
    ax.set_title("Oral and drug-likeness rules" if english else "口服与类药规则", fontsize=14, loc="left", pad=12)
    y = 0.98
    for key, label in RULE_LABELS.items():
        rule = drug["rules"][key]
        ok = rule["pass"]
        if english:
            label = {
                "lipinski": "Lipinski rule of five",
                "veber": "Veber rules",
                "egan": "Egan rules",
                "leadlike": "Lead-like rules",
            }.get(key, label)
        detail = rule["detail"].replace("可旋转键", "Rotatable bonds") if english else rule["detail"]
        ax.text(0.0, y, "●", color=figures.TEAL if ok else figures.RED, fontsize=15, va="top")
        ax.text(0.07, y, label, fontsize=10.2, fontweight="bold", va="top", color=figures.INK)
        ax.text(0.07, y - 0.075, detail, fontsize=9.2, va="top", color=figures.MUTED)
        verdict_text = (
            ("Pass" if ok else f"{rule['violations']} violation(s)") if english
            else ("通过" if ok else f"违反 {rule['violations']} 项")
        )
        ax.text(1.0, y, verdict_text,
                fontsize=9.2, va="top", ha="right",
                color=figures.TEAL if ok else figures.RED, fontweight="bold")
        y -= 0.235
    if drug.get("alerts"):
        prefix = "Structural alerts: " if english else "结构警示："
        joiner = ", " if english else "、"
        ax.text(0.0, 0.02, prefix + joiner.join(drug["alerts"]),
                fontsize=10, color=figures.RED, va="bottom", wrap=True)

    # ---- 2D structure -----------------------------------------------------
    ax = fig.add_subplot(grid[2, 1])
    try:
        mol = Chem.MolFromSmiles(profile["smiles"])
        figures._draw_molecule(ax, mol, compact=False)
    except Exception:
        ax.axis("off")
        ax.text(
            0.5, 0.5,
            "Structure unavailable" if english else "结构不可用",
            ha="center", va="center", color=figures.MUTED,
        )
    ax.set_title("2D structure" if english else "二维结构", fontsize=12, loc="left", pad=8)

    fig.text(
        0.07, 0.024,
        figures._wrap_cjk(
            "Methods: RDKit descriptors + ESOL (Delaney 2004) + Clark logBB; "
            "these values rank compounds for triage and do not replace "
            "experimental ADMET measurement."
            if english else
            "方法：RDKit 描述符 + ESOL（Delaney 2004）+ Clark logBB；"
            "结果为计算排序依据，不替代实验 ADMET。",
            112,
        ),
        fontsize=8.2, color=figures.MUTED,
    )
    fig.savefig(outpath, dpi=600, facecolor="white")
    plt.close(fig)
    return outpath


def admet_filename(name: str) -> str:
    return f"{safe_name(name, 'ligand')}_admet.pdf"
