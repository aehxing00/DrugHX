"""Multi-criteria decision score for docking candidates.

Docking affinity alone is a poor selection criterion: a very negative Vina
score can come from a physically invalid pose, an oversized lipophilic
molecule or a compound that cannot be made.  This module combines the signals
the workbench already computes into one transparent, calibratable score:

    score = 0.45 * affinity + 0.20 * pose validity + 0.12 * ligand efficiency
          + 0.13 * QED + 0.10 * synthetic accessibility
          - penalties for structural alerts

Weights live in ``WEIGHTS`` so a benchmark can refit them without touching the
rest of the code.  Every component is reported separately; the score is a
ranking aid, not a substitute for experimental validation.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

WEIGHTS = {
    "affinity": 0.45,
    "pose": 0.20,
    "efficiency": 0.12,
    "qed": 0.13,
    "sa": 0.10,
}

# Penalties are subtracted after the weighted sum (score points, 0-100).
ALERT_PENALTY = 6.0
WARNING_PENALTY = 2.0
MAX_PENALTY = 25.0

_SA_SCORER = None


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


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


def sa_score(mol: Chem.Mol | None) -> float | None:
    if mol is None:
        return None
    scorer = _sascorer()
    if not scorer:
        return None
    try:
        return round(float(scorer.calculateScore(mol)), 2)
    except Exception:
        return None


def score_candidate(
    affinity: float | None,
    heavy: int | None = None,
    qed: float | None = None,
    sa: float | None = None,
    pose_pass_rate: float | None = None,
    alerts: int = 0,
    warnings: int = 0,
) -> dict:
    """Return ``{score, components, confidence, notes}`` for one candidate."""
    notes: list[str] = []
    components: dict[str, float | None] = {
        "affinity": None,
        "pose": None,
        "efficiency": None,
        "qed": None,
        "sa": None,
    }
    if affinity is None:
        return {
            "score": 0.0,
            "components": components,
            "confidence": "low",
            "notes": ["没有对接打分"],
        }
    affinity = float(affinity)
    components["affinity"] = round(_clamp((-affinity - 4.0) / 6.0), 3)

    if heavy and heavy > 0:
        le = -affinity / float(heavy)
        components["efficiency"] = round(_clamp((le - 0.20) / 0.40), 3)
    else:
        notes.append("缺少重原子数，配体效率未计入")

    if qed is not None:
        components["qed"] = round(_clamp(float(qed)), 3)
    if sa is not None:
        components["sa"] = round(_clamp((10.0 - float(sa)) / 7.0), 3)
    if pose_pass_rate is not None:
        components["pose"] = round(_clamp(float(pose_pass_rate)), 3)
        if pose_pass_rate < 0.8:
            notes.append(f"构象物理校验通过率仅 {pose_pass_rate:.0%}")
    else:
        notes.append("未做构象物理校验，按中性 0.5 计")

    weighted = 0.0
    weight_sum = 0.0
    for key, weight in WEIGHTS.items():
        value = components.get(key)
        if value is None:
            if key == "pose":
                value = 0.5
            else:
                continue
        weighted += weight * value
        weight_sum += weight
    base = 100.0 * weighted / max(weight_sum, 1e-6)

    penalty = min(MAX_PENALTY, ALERT_PENALTY * alerts + WARNING_PENALTY * warnings)
    score = max(0.0, min(100.0, base - penalty))
    if penalty:
        notes.append(f"结构警示扣分 {penalty:.0f}")

    confidence = "high"
    if pose_pass_rate is None or weight_sum < 0.95:
        confidence = "medium"
    if affinity > -5.0:
        confidence = "low"
    return {
        "score": round(score, 1),
        "components": components,
        "penalty": round(penalty, 1),
        "confidence": confidence,
        "notes": notes,
    }


def rank(entries: list[dict], key: str = "decision_score") -> list[dict]:
    """Sort candidate dicts by decision score (descending)."""
    return sorted(
        entries,
        key=lambda item: float(item.get(key) or 0.0),
        reverse=True,
    )


def normalised_entropy(values: list[float]) -> float:
    """Small helper for future weight calibration / diversity checks."""
    total = sum(values)
    if total <= 0 or len(values) < 2:
        return 0.0
    probs = [value / total for value in values if value > 0]
    return -sum(p * math.log(p) for p in probs) / math.log(len(values))
