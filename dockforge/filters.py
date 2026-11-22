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

from .config import FILTER_LIMIT_DEFAULTS

RDLogger.DisableLog("rdApp.*")

# Hard limits: outside this box a molecule is not a developable non-covalent
# small molecule (peptides, PROTAC-like huge molecules, polymers, dyes).
# These are defaults only; every value can be overridden per preparation run.
LIMITS = dict(FILTER_LIMIT_DEFAULTS)

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

_INTEGER_KEYS = ("hbd", "hba", "rotb", "heavy", "rings", "max_ring")


def normalize_limits(limits: dict | None = None) -> dict:
    """Merge user overrides into the defaults and clamp to sane ranges."""
    merged = dict(LIMITS)
    for key, value in (limits or {}).items():
        if key not in merged:
            continue
        try:
            merged[key] = float(value)
        except (TypeError, ValueError):
            continue
    for key in _INTEGER_KEYS:
        merged[key] = int(round(merged[key]))
    merged["mw"] = max(100.0, min(1000.0, merged["mw"]))
    merged["logp"] = max(-2.0, min(10.0, merged["logp"]))
    merged["tpsa"] = max(20.0, min(300.0, merged["tpsa"]))
    merged["hbd"] = max(0, min(20, merged["hbd"]))
    merged["hba"] = max(0, min(30, merged["hba"]))
    merged["rotb"] = max(0, min(50, merged["rotb"]))
    merged["heavy"] = max(5, min(100, merged["heavy"]))
    merged["rings"] = max(1, min(20, merged["rings"]))
    merged["max_ring"] = max(3, min(20, merged["max_ring"]))
    merged["qed"] = max(0.0, min(0.90, merged["qed"]))
    return merged


def warning_limits(limits: dict) -> dict:
    """Derive warnings from the effective hard limits (same defaults as WARN)."""
    return {
        "mw": max(100.0, limits["mw"] * 0.90),
        "logp": limits["logp"] - 0.50,
        "tpsa": max(20.0, limits["tpsa"] - 20.0),
        "rotb": max(1, limits["rotb"] - 2),
        "max_ring": max(3, limits["max_ring"] - 1),
        "qed": min(0.95, limits["qed"] + 0.20),
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


def evaluate(mol: Chem.Mol, name: str = "", limits: dict | None = None) -> dict:
    """Return ``{passed, reasons, warnings, pains, alerts, props, limits}``."""
    if mol is None:
        return {
            "passed": False,
            "reasons": ["无法解析分子结构"],
            "warnings": [],
            "pains": [],
            "alerts": [],
            "props": {},
            "limits": normalize_limits(limits),
        }
    limits = normalize_limits(limits)
    warn = warning_limits(limits)
    props = properties(mol)
    reasons: list[str] = []
    warnings: list[str] = []

    if props["mw"] > limits["mw"]:
        reasons.append(f"分子量 {props['mw']:.0f} > {limits['mw']:.0f}")
    elif props["mw"] > warn["mw"]:
        warnings.append(f"分子量偏大（{props['mw']:.0f}）")
    if props["heavy"] > limits["heavy"]:
        reasons.append(f"重原子数 {props['heavy']} > {limits['heavy']}")
    if props["logp"] > limits["logp"]:
        reasons.append(f"cLogP {props['logp']:.1f} > {limits['logp']:.1f}")
    elif props["logp"] > warn["logp"]:
        warnings.append(f"cLogP 偏高（{props['logp']:.1f}）")
    if props["tpsa"] > limits["tpsa"]:
        reasons.append(f"TPSA {props['tpsa']:.0f} > {limits['tpsa']:.0f}")
    elif props["tpsa"] > warn["tpsa"]:
        warnings.append(f"TPSA 偏高（{props['tpsa']:.0f}）")
    if props["hbd"] > limits["hbd"]:
        reasons.append(f"氢键供体 {props['hbd']} > {limits['hbd']}")
    if props["hba"] > limits["hba"]:
        reasons.append(f"氢键受体 {props['hba']} > {limits['hba']}")
    if props["rotb"] > limits["rotb"]:
        reasons.append(f"可旋转键 {props['rotb']} > {limits['rotb']}")
    elif props["rotb"] > warn["rotb"]:
        warnings.append(f"可旋转键偏多（{props['rotb']}）")
    if props["rings"] > limits["rings"]:
        reasons.append(f"环数 {props['rings']} > {limits['rings']}")
    if props["max_ring"] >= limits["max_ring"]:
        reasons.append(f"大环：最大环 {props['max_ring']} 元 ≥ {limits['max_ring']}")
    elif props["max_ring"] >= warn["max_ring"]:
        warnings.append(f"大环（最大环 {props['max_ring']} 元）")
    if props["qed"] is not None:
        if props["qed"] < limits["qed"]:
            reasons.append(
                f"QED {props['qed']:.2f} < {limits['qed']:.2f}（类药性不足）"
            )
        elif props["qed"] < warn["qed"]:
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
        "limits": limits,
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
