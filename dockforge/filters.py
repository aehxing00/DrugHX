"""Pre-docking ligand triage: drug-likeness bounds and interference filters.

Docking a molecule that is far outside the oral-drug chemical space wastes
time and produces poses nobody can develop.  This module screens every ligand
*before* 3D embedding / PDBQT generation and returns a transparent verdict:

* hard rejects — PAINS (A/B/C), Brenk/NIH alerts, MW > 500, heavy atoms > 45,
  cLogP > 4.5, TPSA > 140, HBD > 5, HBA > 10, rotatable bonds > 10, rings > 6,
  largest ring >= 8 atoms, and QED < 0.30;
* warnings — MW > 450, cLogP > 4.0, TPSA > 120, rotatable bonds > 8,
  largest ring 7 atoms, QED < 0.50, and the exempted Brenk alert below.

Thresholds follow strict Lipinski / Veber drug-like space plus a QED floor, so
the docking queue stays inside chemistry that medicinal chemists can actually
develop; they are not meant to rank the remaining library.
"""
from __future__ import annotations

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, QED, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

RDLogger.DisableLog("rdApp.*")

# Hard limits: outside this box a molecule is not a developable non-covalent
# small molecule (peptides, PROTAC-like huge molecules, polymers, dyes).
LIMITS = {
    "mw": 500.0,
    "logp": 4.5,
    "tpsa": 140.0,
    "hbd": 5,
    "hba": 10,
    "rotb": 10,
    "heavy": 45,
    "rings": 6,
    "max_ring": 8,
    "qed": 0.30,
}

# Softer limits reported as warnings so the user can still see why a molecule
# looks unusual without losing it from the run.
WARN = {
    "mw": 450.0,
    "logp": 4.0,
    "tpsa": 120.0,
    "rotb": 8,
    "max_ring": 7,
    "qed": 0.50,
}

# Brenk flags some ordinary oral drugs (e.g. gefitinib) with this purely
# topological alert; it is a chain-length heuristic, not a reactivity or
# toxicophore warning, so it stays a warning instead of a hard reject.
EXEMPT_ALERTS = {"Aliphatic_long_chain"}

_CATALOG = None


def _catalog() -> FilterCatalog:
    global _CATALOG
    if _CATALOG is None:
        params = FilterCatalogParams()
        for name in ("PAINS_A", "PAINS_B", "PAINS_C"):
            params.AddCatalog(getattr(FilterCatalogParams.FilterCatalogs, name))
        _CATALOG = FilterCatalog(params)
    return _CATALOG


_ALERT_CATALOG = None


def _alert_catalog() -> FilterCatalog:
    """Brenk/NIH toxicophore lists, used as warnings only."""
    global _ALERT_CATALOG
    if _ALERT_CATALOG is None:
        params = FilterCatalogParams()
        for name in ("BRENK", "NIH"):
            params.AddCatalog(getattr(FilterCatalogParams.FilterCatalogs, name))
        _ALERT_CATALOG = FilterCatalog(params)
    return _ALERT_CATALOG


def properties(mol: Chem.Mol) -> dict:
    heavy = Chem.RemoveHs(mol) if mol.GetNumAtoms() != mol.GetNumHeavyAtoms() else mol
    rings = heavy.GetRingInfo().AtomRings()
    sizes = [len(ring) for ring in rings] or [0]
    try:
        qed = round(float(QED.qed(heavy)), 3)
    except Exception:
        qed = None
    return {
        "mw": round(float(Descriptors.MolWt(heavy)), 2),
        "logp": round(float(Descriptors.MolLogP(heavy)), 2),
        "tpsa": round(float(Descriptors.TPSA(heavy)), 1),
        "hbd": int(rdMolDescriptors.CalcNumHBD(heavy)),
        "hba": int(rdMolDescriptors.CalcNumHBA(heavy)),
        "rotb": int(rdMolDescriptors.CalcNumRotatableBonds(heavy)),
        "heavy": int(heavy.GetNumHeavyAtoms()),
        "rings": len(rings),
        "max_ring": int(max(sizes)),
        "qed": qed,
        "charge": int(Chem.GetFormalCharge(heavy)),
        "formula": rdMolDescriptors.CalcMolFormula(heavy),
    }


def evaluate(mol: Chem.Mol, name: str = "") -> dict:
    """Return ``{passed, reasons, warnings, pains, alerts, props}``."""
    if mol is None:
        return {
            "passed": False,
            "reasons": ["无法解析分子结构"],
            "warnings": [],
            "pains": [],
            "alerts": [],
            "props": {},
        }
    props = properties(mol)
    reasons: list[str] = []
    warnings: list[str] = []

    if props["mw"] > LIMITS["mw"]:
        reasons.append(f"分子量 {props['mw']:.0f} > {LIMITS['mw']:.0f}")
    elif props["mw"] > WARN["mw"]:
        warnings.append(f"分子量偏大（{props['mw']:.0f}）")
    if props["heavy"] > LIMITS["heavy"]:
        reasons.append(f"重原子数 {props['heavy']} > {LIMITS['heavy']}")
    if props["logp"] > LIMITS["logp"]:
        reasons.append(f"cLogP {props['logp']:.1f} > {LIMITS['logp']:.1f}")
    elif props["logp"] > WARN["logp"]:
        warnings.append(f"cLogP 偏高（{props['logp']:.1f}）")
    if props["tpsa"] > LIMITS["tpsa"]:
        reasons.append(f"TPSA {props['tpsa']:.0f} > {LIMITS['tpsa']:.0f}")
    elif props["tpsa"] > WARN["tpsa"]:
        warnings.append(f"TPSA 偏高（{props['tpsa']:.0f}）")
    if props["hbd"] > LIMITS["hbd"]:
        reasons.append(f"氢键供体 {props['hbd']} > {LIMITS['hbd']}")
    if props["hba"] > LIMITS["hba"]:
        reasons.append(f"氢键受体 {props['hba']} > {LIMITS['hba']}")
    if props["rotb"] > LIMITS["rotb"]:
        reasons.append(f"可旋转键 {props['rotb']} > {LIMITS['rotb']}")
    elif props["rotb"] > WARN["rotb"]:
        warnings.append(f"可旋转键偏多（{props['rotb']}）")
    if props["rings"] > LIMITS["rings"]:
        reasons.append(f"环数 {props['rings']} > {LIMITS['rings']}")
    if props["max_ring"] >= LIMITS["max_ring"]:
        reasons.append(f"大环：最大环 {props['max_ring']} 元 ≥ {LIMITS['max_ring']}")
    elif props["max_ring"] >= WARN["max_ring"]:
        warnings.append(f"大环（最大环 {props['max_ring']} 元）")
    if props["qed"] is not None:
        if props["qed"] < LIMITS["qed"]:
            reasons.append(
                f"QED {props['qed']:.2f} < {LIMITS['qed']:.2f}（类药性不足）"
            )
        elif props["qed"] < WARN["qed"]:
            warnings.append(f"QED 偏低（{props['qed']:.2f}）")

    pains: list[str] = []
    try:
        for match in _catalog().GetMatches(mol):
            pains.append(match.GetDescription())
            if len(pains) >= 5:
                break
    except Exception:
        pass
    if pains:
        reasons.append("PAINS：" + "、".join(pains))

    alerts: list[str] = []
    try:
        for match in _alert_catalog().GetMatches(mol):
            alerts.append(match.GetDescription())
            if len(alerts) >= 5:
                break
    except Exception:
        pass
    hard_alerts = [item for item in alerts if item not in EXEMPT_ALERTS]
    if hard_alerts:
        reasons.append("Brenk/NIH 警示：" + "、".join(hard_alerts))
    soft_alerts = [item for item in alerts if item in EXEMPT_ALERTS]
    if soft_alerts:
        warnings.append("结构提示：" + "、".join(soft_alerts))

    return {
        "passed": not reasons,
        "reasons": reasons,
        "warnings": warnings,
        "pains": pains,
        "alerts": alerts,
        "props": props,
        "name": name,
    }


def summary(verdicts: list[dict]) -> dict:
    """Aggregate a batch of verdicts for logs and the UI."""
    rejected = [item for item in verdicts if not item.get("passed")]
    counts: dict[str, int] = {}
    for item in rejected:
        for reason in item.get("reasons") or []:
            key = reason.split("：")[0].split(" ")[0]
            counts[key] = counts.get(key, 0) + 1
    return {
        "total": len(verdicts),
        "passed": len(verdicts) - len(rejected),
        "rejected": len(rejected),
        "reasons": sorted(counts.items(), key=lambda pair: -pair[1]),
    }
