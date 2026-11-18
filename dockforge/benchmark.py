"""Retrospective calibration: enrichment metrics and data-driven weights.

Wet-lab calibration takes weeks; a retrospective benchmark can be built in
days by docking a mixture of known actives and presumed-inactive decoys in one
run, labelling the actives from a CSV, and measuring how well the ranking
enriches them.

The module computes EF1%, EF5%, AUC and BEDROC (alpha = 20), fits a logistic
model over the existing decision features, and compares its cross-validated AUC
with the expert MPO weights.  When the fitted model does not improve on the
expert weights, the report says so instead of forcing a new weight set.
"""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.ML.Scoring import Scoring
from scipy.optimize import minimize
from scipy.stats import mannwhitneyu

from . import admet, assay, decision, mpo

RDLogger.DisableLog("rdApp.*")

FEATURES = ("potency", "le", "qed", "sa", "safety")
FEATURE_LABELS = {
    "potency": "效力（−Vina）",
    "le": "配体效率",
    "qed": "QED",
    "sa": "可合成性（反向）",
    "safety": "安全性",
}


def _canonical(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles or "")
    return Chem.MolToSmiles(mol) if mol is not None else ""


def _match_actives(rows: list[dict], ligands: list[dict]) -> dict[str, dict]:
    """Map assay rows to ligand ids by SMILES then name."""
    by_smiles = {}
    by_name = {}
    for ligand in ligands:
        smiles = (
            ligand.get("smiles")
            or (ligand.get("meta") or {}).get("smiles")
            or ""
        )
        canonical = _canonical(smiles)
        if canonical:
            by_smiles[canonical] = ligand
        name = str(ligand.get("name") or "").strip().lower()
        if name:
            by_name[name] = ligand
    matched = {}
    for row in rows:
        ligand = None
        if row.get("smiles"):
            ligand = by_smiles.get(_canonical(row["smiles"]))
        if ligand is None and row.get("name"):
            ligand = by_name.get(str(row["name"]).strip().lower())
        if ligand is None:
            continue
        p_activity = assay.to_pactivity(row.get("value") or 0, row.get("unit") or "nM")
        matched[ligand["id"]] = {
            "ligand_id": ligand["id"],
            "name": ligand.get("name") or row.get("name"),
            "p_activity": p_activity,
        }
    return matched


def _enrichment_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    """labels: 1 active, 0 decoy; scores: higher is better."""
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    n_total = len(labels_sorted)
    n_active = int(labels_sorted.sum())
    if n_active == 0 or n_active == n_total:
        return {}
    ranks = np.where(labels_sorted == 1)[0] + 1.0
    top1 = max(1, int(math.ceil(n_total * 0.01)))
    top5 = max(1, int(math.ceil(n_total * 0.05)))
    ef1 = (labels_sorted[:top1].sum() / top1) / (n_active / n_total)
    ef5 = (labels_sorted[:top5].sum() / top5) / (n_active / n_total)
    try:
        auc = float(mannwhitneyu(scores[labels == 1], scores[labels == 0], alternative="greater").statistic)
        auc /= n_active * (n_total - n_active)
    except Exception:
        auc = None
    alpha = 20.0
    try:
        bedroc = float(
            Scoring.CalcBEDROC([[int(value)] for value in labels_sorted], 0, alpha)
        )
    except Exception:
        bedroc = None
    return {
        "n_total": n_total,
        "n_active": n_active,
        "ef1": round(float(ef1), 2),
        "ef5": round(float(ef5), 2),
        "auc": None if auc is None else round(float(auc), 3),
        "bedroc": None if bedroc is None else round(float(bedroc), 3),
        "best_active_rank": int(ranks.min()),
        "median_active_rank": int(np.median(ranks)),
        "top1_percent": top1,
    }


def _bootstrap(labels: np.ndarray, scores: np.ndarray, samples: int = 200, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    aucs = []
    ef1s = []
    for _ in range(max(int(samples), 30)):
        pos = rng.choice(positives, size=len(positives), replace=True)
        neg = rng.choice(negatives, size=len(negatives), replace=True)
        merged_labels = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
        merged_scores = np.concatenate([pos, neg])
        metrics = _enrichment_metrics(merged_labels, merged_scores)
        if metrics.get("auc") is not None:
            aucs.append(metrics["auc"])
        ef1s.append(metrics["ef1"])
    return {
        "auc_ci": [round(float(np.percentile(aucs, 2.5)), 3), round(float(np.percentile(aucs, 97.5)), 3)] if aucs else None,
        "ef1_ci": [round(float(np.percentile(ef1s, 2.5)), 2), round(float(np.percentile(ef1s, 97.5)), 2)] if ef1s else None,
    }


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30, 30)))


def _fit_logistic(matrix: np.ndarray, labels: np.ndarray, l2: float = 1.0) -> np.ndarray:
    def loss(weights):
        logits = matrix @ weights
        probabilities = _sigmoid(logits)
        eps = 1e-9
        return (
            -np.mean(labels * np.log(probabilities + eps) + (1 - labels) * np.log(1 - probabilities + eps))
            + l2 * np.sum(weights[1:] ** 2) / max(len(labels), 1)
        )

    result = minimize(loss, np.zeros(matrix.shape[1]), method="L-BFGS-B")
    return result.x


def _auc_score(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if labels.sum() == 0 or (1 - labels).sum() == 0:
        return None
    try:
        value = float(mannwhitneyu(scores[labels == 1], scores[labels == 0], alternative="greater").statistic)
        return value / (labels.sum() * (1 - labels).sum())
    except Exception:
        return None


def _feature_row(ligand: dict, affinity: float) -> dict | None:
    smiles = ligand.get("smiles") or (ligand.get("meta") or {}).get("smiles") or ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    profile = admet.profile_mol(mol, name=ligand.get("name") or "", affinity=affinity)
    efficiency = profile.get("efficiency") or {}
    druglike = profile.get("druglike") or {}
    safety = mpo._safety_score(profile)  # shared safety definition
    return {
        "potency": -float(affinity),
        "le": float(efficiency.get("le") or 0.0),
        "qed": float(druglike.get("qed") or 0.0),
        "sa": float(decision.sa_score(mol) or 6.0),
        "safety": float(safety),
        "expert": float(mpo.score(smiles, affinity, name=ligand.get("name") or "").get("mpo") or 0.0),
    }


def analyse(run: dict, project_ligands: list[dict], actives_rows: list[dict], seed: int = 42) -> dict:
    """Compute enrichment and weight calibration for one docking run."""
    ligand_index = {item["id"]: item for item in project_ligands}
    actives = _match_actives(actives_rows, project_ligands)
    if len(actives) < 5:
        raise ValueError(
            f"只匹配到 {len(actives)} 个活性物；回溯校准至少需要 5 个，"
            "并建议使用 20 个以上活性物 + 200 个以上诱饵。"
        )
    labels = []
    scores = []
    rows = []
    for result in run.get("results", []):
        if result.get("status") != "ok" or result.get("affinity") is None:
            continue
        ligand = ligand_index.get(result.get("ligand_id"))
        if not ligand:
            continue
        affinity = float(result["affinity"])
        labels.append(1 if ligand["id"] in actives else 0)
        scores.append(-affinity)
        rows.append(
            {
                "ligand_id": ligand["id"],
                "name": ligand.get("name") or result.get("ligand_name") or "",
                "affinity": affinity,
                "active": ligand["id"] in actives,
            }
        )
    if sum(labels) < 5:
        raise ValueError("对接结果中的活性物不足 5 个，无法计算富集指标。")
    if len(labels) - sum(labels) < 20:
        raise ValueError(
            f"本次运行只有 {len(labels) - sum(labels)} 个诱饵；"
            "建议把 20 个以上活性物与 200 个以上诱饵放在同一次对接中。"
        )
    labels_array = np.asarray(labels, dtype=float)
    scores_array = np.asarray(scores, dtype=float)
    metrics = _enrichment_metrics(labels_array, scores_array)
    metrics.update(_bootstrap(labels_array, scores_array, seed=seed))

    # Feature table for the labelled set.
    features = []
    used_labels = []
    for row in rows:
        feature = _feature_row(ligand_index[row["ligand_id"]], row["affinity"])
        if feature is None:
            continue
        features.append(feature)
        used_labels.append(1.0 if row["active"] else 0.0)
    weights = {}
    comparison = {}
    if len(features) >= 40 and sum(used_labels) >= 5:
        matrix = np.array([[item[key] for key in FEATURES] for item in features], dtype=float)
        means = matrix.mean(axis=0)
        stds = matrix.std(axis=0)
        stds[stds == 0] = 1.0
        standardized = (matrix - means) / stds
        labels_array2 = np.asarray(used_labels, dtype=float)
        # 5-fold cross-validated AUC.
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(labels_array2))
        folds = np.array_split(indices, 5)
        predictions = np.zeros(len(labels_array2))
        for fold in folds:
            train_mask = np.ones(len(labels_array2), dtype=bool)
            train_mask[fold] = False
            coefficients = _fit_logistic(standardized[train_mask], labels_array2[train_mask])
            predictions[fold] = standardized[fold] @ coefficients
        cv_auc = _auc_score(labels_array2, predictions)
        expert_scores = np.asarray([item["expert"] for item in features], dtype=float)
        expert_auc = _auc_score(labels_array2, expert_scores)
        coefficients = _fit_logistic(standardized, labels_array2)
        total = float(np.sum(np.abs(coefficients))) or 1.0
        weights = {
            key: round(float(abs(value) / total), 3)
            for key, value in zip(FEATURES, coefficients)
        }
        comparison = {
            "cv_auc": None if cv_auc is None else round(float(cv_auc), 3),
            "expert_auc": None if expert_auc is None else round(float(expert_auc), 3),
            "improvement": None if cv_auc is None or expert_auc is None else round(float(cv_auc - expert_auc), 3),
            "n_features": int(len(features)),
        }
        if comparison["cv_auc"] is not None and comparison["expert_auc"] is not None:
            if comparison["cv_auc"] < 0.60 and comparison["expert_auc"] < 0.60:
                comparison["verdict"] = (
                    "两种权重都接近随机（AUC <0.60），本靶点/库不具备排序能力；"
                    "不建议据此更换权重，应先检查对接盒、活性物质量或诱饵设计"
                )
                comparison["level"] = "bad"
            elif comparison["improvement"] is not None and comparison["improvement"] >= 0.03 and comparison["cv_auc"] >= 0.60:
                comparison["verdict"] = "数据权重优于专家权重，可在本靶点系列中试用"
                comparison["level"] = "ok"
            else:
                comparison["verdict"] = "数据权重未明显优于专家权重，建议保留现有 MPO 权重"
                comparison["level"] = "warn"
    else:
        comparison = {
            "verdict": "带 SMILES 的标注分子不足，无法拟合权重；仅报告富集指标",
            "level": "warn",
        }

    # Enrichment curve (cumulative fraction of actives vs fraction screened).
    order = np.argsort(-scores_array)
    cumulative = np.cumsum(labels_array[order])
    x = np.linspace(0, 1, len(cumulative))
    y = cumulative / max(cumulative[-1], 1)
    step = max(1, len(x) // 120)

    return {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "run_id": run.get("id", ""),
        "actives_input": len(actives_rows),
        "actives_matched": len(actives),
        "metrics": metrics,
        "weights": weights,
        "comparison": comparison,
        "feature_labels": FEATURE_LABELS,
        "curve": {
            "x": [round(float(value), 4) for value in x[::step]],
            "y": [round(float(value), 4) for value in y[::step]],
        },
        "note": (
            "诱饵为同一次对接中未标注为活性物的其他配体；请尽量使用性质匹配、"
            "与靶点无关的分子作为诱饵。EF/AUC/BEDROC 只反映本次库与靶点的排序能力，"
            "不能外推到其他靶点。"
        ),
    }
