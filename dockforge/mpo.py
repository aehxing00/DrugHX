"""Multi-parameter optimisation (MPO), Pareto analysis and dose projection.

Academic screening often stops at "lowest docking score"; industrial
pre-research needs a defensible multi-parameter decision and a first-in-human
dose estimate.  This module turns the existing docking + ADMET data into:

* a 0–100 MPO score with explicit weights (potency, ligand efficiency, QED,
  synthetic accessibility, safety);
* an automatic hit / lead / candidate stage label;
* a Pareto front over affinity, LE, QED, SA and safety;
* a novelty indicator against the built-in FDA drug set;
* a rule-based maintenance/loading dose projection for seven species.

All values are screening-level estimates, not experimental conclusions.
"""
from __future__ import annotations

import math

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from . import admet, decision

RDLogger.DisableLog("rdApp.*")

MPO_WEIGHTS = {
    "potency": 0.30,
    "le": 0.15,
    "qed": 0.20,
    "sa": 0.10,
    "safety": 0.25,
}

STAGE_EN = {
    "起点": "Start",
    "命中": "Hit",
    "先导": "Lead",
    "候选": "Candidate",
}

# Species body weights (kg) used by the dose projection.  They mirror the
# physiology table in ``pharm`` but stay local so this module has no import
# cycle.
SPECIES_BW = {
    "human": 70.0,
    "pig": 75.0,
    "cattle": 500.0,
    "sheep": 60.0,
    "dog": 15.0,
    "cat": 4.0,
    "rat": 0.3,
}

_DRUG_FPS = None


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _linear(value: float, low: float, high: float) -> float:
    """Map ``low → 0`` and ``high → 100`` with clamping."""
    if high == low:
        return 0.0
    return _clamp((value - low) / (high - low) * 100.0)


def _drug_fingerprints():
    global _DRUG_FPS
    if _DRUG_FPS is not None:
        return _DRUG_FPS
    from .drugref import FDA_DRUGS

    entries = []
    for item in FDA_DRUGS:
        mol = Chem.MolFromSmiles(item.get("smiles") or "")
        if mol is None:
            continue
        entries.append(
            (
                item.get("name") or "drug",
                AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048),
            )
        )
    _DRUG_FPS = entries
    return entries


def novelty(smiles: str) -> dict:
    """Tanimoto novelty against marketed FDA drugs (higher = more novel)."""
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return {"available": False, "label": "未知", "max": None, "nearest": []}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    scored = []
    for name, ref in _drug_fingerprints():
        try:
            scored.append((DataStructs.TanimotoSimilarity(fp, ref), name))
        except Exception:
            continue
    if not scored:
        return {"available": False, "label": "未知", "max": None, "nearest": []}
    scored.sort(reverse=True)
    top = scored[:3]
    maximum = float(top[0][0])
    if maximum >= 0.60:
        label, level = "低（与上市药高度相似）", "low"
    elif maximum >= 0.40:
        label, level = "中", "medium"
    else:
        label, level = "高", "high"
    return {
        "available": True,
        "label": label,
        "level": level,
        "max": round(maximum, 3),
        "nearest": [{"name": name, "similarity": round(value, 3)} for value, name in top],
    }


def _safety_score(profile: dict) -> float:
    """Weighted ADMET safety score (0–100, higher is safer)."""
    endpoints = {item.get("key"): item.get("score", 50.0) for item in profile.get("admet") or []}
    weights = {
        "herg": 0.30,
        "ames": 0.30,
        "cyp3a4": 0.15,
        "cyp2d6": 0.10,
        "pgp": 0.10,
        "ppb": 0.05,
    }
    total = 0.0
    used = 0.0
    for key, weight in weights.items():
        if key in endpoints:
            total += float(endpoints[key]) * weight
            used += weight
    return _clamp(total / used) if used else 50.0


def score(
    smiles: str,
    affinity: float | None,
    name: str = "",
    ligand_id: str = "",
) -> dict:
    """Compute the full MPO panel for one molecule."""
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return {
            "name": name,
            "ligand_id": ligand_id,
            "smiles": smiles,
            "available": False,
            "error": "SMILES 无法解析",
        }
    profile = admet.profile_mol(mol, name=name or "ligand", affinity=affinity)
    phys = profile.get("physchem") or {}
    drug = profile.get("druglike") or {}
    efficiency = profile.get("efficiency") or {}
    qed = float(drug.get("qed") or 0.0)
    le_value = efficiency.get("le")
    le = float(le_value) if le_value is not None else None
    sa = decision.sa_score(mol)
    safety = _safety_score(profile)

    potency = _linear(-(affinity if affinity is not None else 0.0), 5.0, 12.0)
    le_score = _linear(le if le is not None else 0.0, 0.20, 0.50)
    qed_score = _clamp(qed * 100.0)
    sa_score = _linear(6.0 - (sa if sa is not None else 6.0), 0.0, 5.0)
    components = {
        "potency": round(potency, 1),
        "le": round(le_score, 1),
        "qed": round(qed_score, 1),
        "sa": round(sa_score, 1),
        "safety": round(safety, 1),
    }
    mpo = sum(components[key] * weight for key, weight in MPO_WEIGHTS.items())

    if (
        mpo >= 70
        and affinity is not None and affinity <= -9.0
        and le is not None and le >= 0.35
        and qed >= 0.60
        and (sa is None or sa <= 4.5)
        and safety >= 65
    ):
        stage = "候选"
    elif (
        mpo >= 55
        and affinity is not None and affinity <= -8.0
        and le is not None and le >= 0.28
        and qed >= 0.50
        and (sa is None or sa <= 5.0)
    ):
        stage = "先导"
    elif affinity is not None and affinity <= -7.0:
        stage = "命中"
    else:
        stage = "起点"

    return {
        "available": True,
        "name": name,
        "ligand_id": ligand_id,
        "smiles": smiles,
        "affinity": affinity,
        "mw": phys.get("mw"),
        "logp": phys.get("logp"),
        "tpsa": phys.get("tpsa"),
        "qed": round(qed, 3),
        "le": None if le is None else round(le, 3),
        "sa": None if sa is None else round(float(sa), 2),
        "safety": round(safety, 1),
        "components": components,
        "mpo": round(mpo, 1),
        "stage": stage,
        "stage_en": STAGE_EN.get(stage, stage),
        "novelty": novelty(smiles),
    }


def _pareto(items: list[dict]) -> None:
    """Mark non-dominated items in place (affinity, LE and safety).

    Five objectives make almost every molecule non-dominated, so the front is
    restricted to the three axes medicinal chemists actually trade off:
    potency, ligand efficiency and safety.
    """
    keys = [
        ("affinity", -1.0),
        ("le", 1.0),
        ("safety", 1.0),
    ]
    for index, item in enumerate(items):
        dominated = False
        for other_index, other in enumerate(items):
            if index == other_index:
                continue
            at_least_one = False
            all_better = True
            for key, sign in keys:
                mine = item.get(key)
                theirs = other.get(key)
                if mine is None or theirs is None:
                    all_better = False
                    break
                mine_value = float(mine) * sign
                their_value = float(theirs) * sign
                if their_value < mine_value:
                    all_better = False
                    break
                if their_value > mine_value:
                    at_least_one = True
            if all_better and at_least_one:
                dominated = True
                break
        item["pareto"] = not dominated


def analyse(records: list[dict]) -> dict:
    """Score a list of docking hits and return the ranked MPO panel."""
    items = []
    for record in records:
        scored = score(
            record.get("smiles") or "",
            record.get("affinity"),
            name=record.get("name") or "",
            ligand_id=record.get("ligand_id") or "",
        )
        if scored.get("available"):
            items.append(scored)
    if not items:
        return {"count": 0, "items": [], "pareto_count": 0, "weights": MPO_WEIGHTS}
    _pareto(items)
    items.sort(key=lambda item: -float(item.get("mpo") or 0))
    for rank, item in enumerate(items, start=1):
        item["rank"] = rank
    stages: dict[str, int] = {}
    for item in items:
        stages[item["stage"]] = stages.get(item["stage"], 0) + 1
    best = items[0]
    return {
        "count": len(items),
        "items": items,
        "pareto_count": sum(1 for item in items if item.get("pareto")),
        "weights": MPO_WEIGHTS,
        "stages": stages,
        "best": {
            "name": best.get("name"),
            "mpo": best.get("mpo"),
            "stage": best.get("stage"),
        },
        "summary": (
            f"共评价 {len(items)} 个分子，Pareto 前沿 {sum(1 for item in items if item.get('pareto'))} 个；"
            f"综合最优为 {best.get('name') or '—'}（MPO {best.get('mpo')}，{best.get('stage')}）。"
        ),
    }


def dose_projection(
    potency_nm: float,
    mw: float,
    species: str = "human",
    f_percent: float = 50.0,
    cl_ml_min_kg: float = 5.0,
    fu: float = 0.20,
    vd_l_kg: float = 1.5,
    target_multiplier: float = 2.0,
) -> dict:
    """Rule-based maintenance/loading dose from target potency.

    ``Ceff`` is the free plasma concentration = ``target_multiplier × potency``;
    the total concentration is divided by the free fraction, then multiplied by
    clearance and divided by oral bioavailability.  Values are screening-level.
    """
    species = species if species in SPECIES_BW else "human"
    body_weight = SPECIES_BW[species]
    potency = max(float(potency_nm or 0.0), 1e-3)
    mw = max(float(mw or 0.0), 1.0)
    fu = min(max(float(fu or 0.0), 0.001), 1.0)
    f_fraction = min(max(float(f_percent or 0.0) / 100.0, 0.01), 1.0)
    cl_l_h = max(float(cl_ml_min_kg or 0.0), 0.1) * body_weight * 60.0 / 1000.0
    vd_l = max(float(vd_l_kg or 0.0), 0.05) * body_weight

    ceff_nm = potency * max(float(target_multiplier or 1.0), 1.0)
    ceff_mg_l = ceff_nm * mw * 1e-6
    total_plasma_mg_l = ceff_mg_l / fu
    dose_day_mg = total_plasma_mg_l * cl_l_h * 24.0 / f_fraction
    loading_mg = total_plasma_mg_l * vd_l / f_fraction
    half_life = 0.693 * vd_l / cl_l_h

    if dose_day_mg <= 100:
        verdict, level = "低剂量可行", "ok"
    elif dose_day_mg <= 1000:
        verdict, level = "剂量可行，需关注溶解度与制剂", "warn"
    else:
        verdict, level = "剂量过高，需提高效力、改善生物利用度或降低清除", "bad"
    return {
        "species": species,
        "body_weight_kg": body_weight,
        "potency_nm": round(potency, 3),
        "target_multiplier": round(float(target_multiplier), 2),
        "ceff_nm": round(ceff_nm, 3),
        "ceff_mg_l": round(ceff_mg_l, 5),
        "mw": round(mw, 1),
        "f_percent": round(f_fraction * 100, 1),
        "fu": round(fu, 3),
        "cl_ml_min_kg": round(float(cl_ml_min_kg), 2),
        "cl_l_h": round(cl_l_h, 3),
        "vd_l_kg": round(float(vd_l_kg), 3),
        "vd_l": round(vd_l, 2),
        "half_life_h": round(half_life, 2),
        "dose_mg_day": round(dose_day_mg, 2),
        "dose_mg_kg": round(dose_day_mg / body_weight, 3),
        "loading_mg": round(loading_mg, 2),
        "verdict": verdict,
        "level": level,
        "note": (
            "按目标游离浓度 = 2× 效价、口服 F、肝清除与 1.5 L/kg 分布容积估算；"
            "仅用于立项前剂量可行性判断，需用药代动力学试验校正。"
        ),
    }
