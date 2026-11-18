"""Batch active learning for the next round of experiments.

Docking ranks a library, but the real question after an assay round is: which
eight to sixteen compounds should be made or tested next?  This module fits a
Gaussian process on the measured pActivity values using a Morgan/Tanimoto
kernel plus docking-score similarity, then scores every undocked-measured
compound by Expected Improvement and selects a diverse batch.

With fewer than four measurements it falls back to the MPO ranking and says so.
"""
from __future__ import annotations

import math

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm

RDLogger.DisableLog("rdApp.*")

_BITS = 1024


def _fingerprints(smiles_list: list[str]) -> np.ndarray:
    matrix = np.zeros((len(smiles_list), _BITS), dtype=np.float64)
    for index, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles or "")
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=_BITS)
        arr = np.zeros((_BITS,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        matrix[index] = arr
    return matrix


def _tanimoto(matrix: np.ndarray) -> np.ndarray:
    inter = matrix @ matrix.T
    sums = matrix.sum(axis=1)
    denom = sums[:, None] + sums[None, :] - inter
    return inter / np.maximum(denom, 1e-9)


def _kernel(
    fp_a: np.ndarray,
    aff_a: np.ndarray,
    fp_b: np.ndarray | None = None,
    aff_b: np.ndarray | None = None,
) -> np.ndarray:
    if fp_b is None:
        fp_b = fp_a
    if aff_b is None:
        aff_b = aff_a
    tanimoto = _tanimoto(fp_a) if fp_b is fp_a else _cross_tanimoto(fp_a, fp_b)
    delta = aff_a[:, None] - aff_b[None, :]
    affinity_kernel = np.exp(-(delta ** 2) / (2 * 0.75 ** 2))
    return 0.75 * tanimoto + 0.25 * affinity_kernel


def _cross_tanimoto(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    inter = a @ b.T
    denom = a.sum(axis=1)[:, None] + b.sum(axis=1)[None, :] - inter
    return inter / np.maximum(denom, 1e-9)


def recommend(
    records: list[dict],
    assay_payload: dict | None = None,
    batch: int = 12,
    diversity: float = 0.85,
) -> dict:
    """Rank candidates by Expected Improvement and return a diverse batch."""
    measured = {
        row.get("ligand_id"): float(row["p_activity"])
        for row in ((assay_payload or {}).get("rows") or [])
        if row.get("ligand_id") and row.get("p_activity") is not None
    }
    rows = [
        record for record in records
        if record.get("smiles") and record.get("affinity") is not None
    ]
    train = [record for record in rows if record.get("ligand_id") in measured]
    pool = [record for record in rows if record.get("ligand_id") not in measured]
    if not pool:
        return {
            "mode": "exhausted",
            "train": len(train),
            "pool": 0,
            "items": [],
            "note": (
                "当前对接集中的分子都已有实测数据；请先对接新一批候选，"
                "或等待新实验数据后再运行主动学习。"
            ),
        }
    if len(train) < 4:
        fallback = sorted(pool, key=lambda item: float(item["affinity"]))[:batch]
        for item in fallback:
            item["predicted_pactivity"] = None
            item["sd"] = None
            item["ei"] = None
            item["reason"] = "实测点不足 4 个，按对接打分排序（先积累实验数据）"
        return {
            "mode": "fallback",
            "train": len(train),
            "pool": len(pool),
            "items": fallback,
            "note": "至少需要 4 个实测 pActivity 才能启用主动学习；当前返回对接最优未测候选。",
        }

    train_smiles = [item["smiles"] for item in train]
    pool_smiles = [item["smiles"] for item in pool]
    fp_train = _fingerprints(train_smiles)
    fp_pool = _fingerprints(pool_smiles)
    aff_train = np.array([float(item["affinity"]) for item in train])
    aff_pool = np.array([float(item["affinity"]) for item in pool])
    y = np.array([measured[item["ligand_id"]] for item in train])
    mean_y = float(np.mean(y))
    std_y = float(np.std(y)) or 1.0
    y_norm = (y - mean_y) / std_y

    kernel = _kernel(fp_train, aff_train)
    kernel += (0.12 ** 2) * np.eye(len(train))
    factor = cho_factor(kernel + 1e-9 * np.eye(len(train)))
    alpha = cho_solve(factor, y_norm)
    cross = _kernel(fp_pool, aff_pool, fp_train, aff_train)
    mu = cross @ alpha
    solved = np.linalg.solve(np.linalg.cholesky(kernel), cross.T)
    variance = np.maximum(1.0 - np.sum(solved ** 2, axis=0), 1e-6)
    sd = np.sqrt(variance)

    best = float(np.max(y_norm))
    xi = 0.10
    z = (mu - best - xi) / sd
    ei = sd * (z * norm.cdf(z) + norm.pdf(z))

    ranked = []
    for index, item in enumerate(pool):
        ranked.append(
            {
                **item,
                "predicted_pactivity": round(float(mu[index] * std_y + mean_y), 2),
                "sd": round(float(sd[index] * std_y), 2),
                "ei": round(float(ei[index]), 3),
                "reason": "EI 高且与已选分子结构差异足够",
            }
        )
    ranked.sort(key=lambda item: -float(item["ei"]))
    chosen = []
    fingerprint_index = {item["ligand_id"]: index for index, item in enumerate(pool)}
    pool_tanimoto = _tanimoto(fp_pool)
    for item in ranked:
        if len(chosen) >= batch:
            break
        row = fingerprint_index[item["ligand_id"]]
        too_close = any(
            pool_tanimoto[row, fingerprint_index[picked["ligand_id"]]] > diversity
            for picked in chosen
        )
        if not too_close:
            chosen.append(item)
    if len(chosen) < batch:
        for item in ranked:
            if len(chosen) >= batch:
                break
            if all(item["ligand_id"] != picked["ligand_id"] for picked in chosen):
                item["reason"] = "EI 补充（多样性约束下不足批次）"
                chosen.append(item)
    return {
        "mode": "active",
        "train": len(train),
        "pool": len(pool),
        "batch": len(chosen),
        "items": chosen,
        "note": (
            "高斯过程：0.75×Morgan/Tanimoto 核 + 0.25×对接打分核，噪声 0.12；"
            "EI 采用当前最优实测值 + 0.1 探索项；批次选择加入 Tanimoto ≤ 0.85 的多样性约束。"
            "预测用于下一轮实验排序，不是活性结论。"
        ),
    }
