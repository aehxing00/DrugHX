"""Decision-confidence layer: structure quality, domain, consensus and assay.

A docking score without context invites over-interpretation.  This module
combines four independent lines of evidence into a 0–100 confidence score and
an explicit "abstain" flag:

* receptor structure quality (resolution/R-free, missing residues, altlocs or
  AlphaFold pLDDT);
* applicability domain of the rule-based models (nearest FDA drug similarity);
* engine/pose consensus spread (disagreement between Vina, Vinardo, smina and
  QuickVina 2, or seed replicates);
* experimental calibration (rank correlation with measured potency).

When the score is low the report and UI say so instead of presenting the
ranking as decision-ready.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

from . import pharm


def _resolution(text: str) -> float | None:
    match = re.search(r"REMARK\s+2\s+RESOLUTION\.\s+([\d.]+)", text)
    return float(match.group(1)) if match else None


def _r_free(text: str) -> float | None:
    match = re.search(r"FREE R VALUE\s*:\s*([\d.]+)", text)
    if match:
        return float(match.group(1))
    match = re.search(r"R VALUE\s+\(FREE SET\)\s*:\s*([\d.]+)", text)
    return float(match.group(1)) if match else None


def _missing_count(text: str) -> int:
    count = 0
    for line in text.splitlines():
        if line.startswith("REMARK 465") and re.search(r"\d+", line[10:]):
            count += 1
    return count


def _altloc_fraction(text: str) -> float:
    total = 0
    alternate = 0
    for line in text.splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            total += 1
            altloc = line[16:17]
            if altloc not in (" ", "", "A"):
                alternate += 1
    return alternate / total if total else 0.0


def _plddt(text: str) -> float | None:
    if "AlphaFold" not in text and "pLDDT" not in text:
        return None
    values = []
    for line in text.splitlines():
        if line.startswith("ATOM  ") and line[12:16].strip() == "CA":
            try:
                values.append(float(line[60:66]))
            except ValueError:
                continue
    if not values:
        return None
    return sum(values) / len(values)


def structure_quality(pdb_path: Path | None, meta: dict | None = None) -> dict:
    meta = meta or {}
    if not pdb_path or not Path(pdb_path).is_file():
        return {
            "score": 50.0,
            "available": False,
            "note": "缺少可解析的受体 PDB，结构质量按中性 50 分处理。",
            "resolution": None,
            "r_free": None,
            "missing": None,
            "altloc": None,
            "plddt": None,
        }
    text = Path(pdb_path).read_text(encoding="utf-8", errors="replace")
    resolution = _resolution(text)
    if resolution is None and meta.get("resolution") is not None:
        try:
            resolution = float(meta["resolution"])
        except (TypeError, ValueError):
            resolution = None
    r_free = _r_free(text)
    missing = _missing_count(text)
    altloc = _altloc_fraction(text)
    plddt = _plddt(text)
    if plddt is not None:
        score = max(0.0, min(100.0, plddt))
        note = f"AlphaFold 模型，平均 pLDDT {plddt:.0f}。"
    else:
        parts = []
        if resolution is not None:
            if resolution <= 1.5:
                parts.append(100.0)
            elif resolution <= 3.5:
                parts.append(100.0 - (resolution - 1.5) / 2.0 * 65.0)
            else:
                parts.append(25.0)
        if r_free is not None:
            if r_free <= 0.20:
                parts.append(100.0)
            elif r_free <= 0.30:
                parts.append(100.0 - (r_free - 0.20) / 0.10 * 45.0)
            else:
                parts.append(35.0)
        if parts:
            score = sum(parts) / len(parts)
        else:
            score = 50.0
        score -= min(25.0, missing * 2.0)
        score -= min(20.0, altloc * 100.0)
        score = max(0.0, min(100.0, score))
        note = (
            "实验结构："
            + (f"分辨率 {resolution:.2f} Å；" if resolution is not None else "分辨率未知；")
            + (f"R-free {r_free:.3f}；" if r_free is not None else "R-free 未知；")
            + f"缺失残基记录 {missing} 条；altloc 比例 {altloc * 100:.1f}%。"
        )
    return {
        "score": round(score, 1),
        "available": True,
        "note": note,
        "resolution": resolution,
        "r_free": r_free,
        "missing": missing,
        "altloc": round(altloc, 4),
        "plddt": None if plddt is None else round(plddt, 1),
    }


def domain_score(smiles: str) -> dict:
    if not smiles:
        return {"score": 50.0, "available": False, "note": "缺少 SMILES，适用域按中性 50 分处理。"}
    try:
        payload = pharm.applicability_block(smiles)
    except Exception as exc:
        return {"score": 50.0, "available": False, "note": f"适用域计算失败：{exc}"}
    if not payload.get("available"):
        return {"score": 50.0, "available": False, "note": "参考集不可用，按中性 50 分处理。"}
    maximum = float(payload.get("max_similarity") or 0.0)
    if maximum >= 0.55:
        score = 90.0
    elif maximum >= 0.40:
        score = 70.0
    elif maximum >= 0.25:
        score = 50.0
    else:
        score = 30.0
    return {
        "score": score,
        "available": True,
        "note": (
            f"与最近上市药 {payload.get('nearest', '—')} 的 Tanimoto 相似度 "
            f"{maximum:.2f}（{payload.get('level', '—')}）。"
        ),
        "max_similarity": maximum,
    }


def consensus_score(results: list[dict]) -> dict:
    rows = [
        item for item in results
        if item.get("status") == "ok" and item.get("consensus_spread") is not None
    ]
    if rows:
        spreads = [float(item["consensus_spread"]) for item in rows]
        mean_spread = sum(spreads) / len(spreads)
        score = max(0.0, min(100.0, 100.0 - mean_spread * 30.0))
        low = sum(1 for item in rows if item.get("low_confidence"))
        note = f"{len(rows)} 个配体参与共识，平均排名跨度 {mean_spread:.2f}；低置信度 {low} 个。"
        return {"score": round(score, 1), "available": True, "note": note}
    if len(results) >= 2:
        return {
            "score": 50.0,
            "available": False,
            "note": "未启用多引擎共识，引擎一致性按中性 50 分处理。",
        }
    return {"score": 50.0, "available": False, "note": "成功结果不足，无法评估共识。"}


def calibration_score(assay_payload: dict | None) -> dict:
    summary = (assay_payload or {}).get("summary") or {}
    matched = int(summary.get("matched") or 0)
    rho = summary.get("spearman")
    if matched < 4 or rho is None:
        return {
            "score": 50.0,
            "available": False,
            "note": "尚无足够实测数据（≥4 条）校准，按中性 50 分处理。",
        }
    rho = float(rho)
    if rho <= -0.60:
        score, note = 95.0, f"实测校准 ρ={rho:.2f}，排序与实验同向。"
    elif rho <= -0.30:
        score, note = 70.0, f"实测校准 ρ={rho:.2f}，仅支持粗排。"
    elif rho >= 0.30:
        score, note = 15.0, f"实测校准 ρ={rho:.2f}，方向相反，需检查数据对应。"
    else:
        score, note = 30.0, f"实测校准 ρ={rho:.2f}，相关性弱。"
    return {"score": score, "available": True, "note": note}


_WEIGHTS = {
    "structure": 0.30,
    "domain": 0.20,
    "consensus": 0.25,
    "calibration": 0.15,
    "pose": 0.10,
}


def assess(
    pdb_path: Path | None,
    smiles: str,
    results: list[dict],
    assay_payload: dict | None = None,
    pose_pass_rate: float | None = None,
    receptor_meta: dict | None = None,
) -> dict:
    """Combine the evidence into a 0–100 confidence score."""
    structure = structure_quality(pdb_path, receptor_meta)
    domain = domain_score(smiles)
    consensus = consensus_score(results)
    calibration = calibration_score(assay_payload)
    if pose_pass_rate is None:
        pose = {"score": 50.0, "available": False, "note": "未运行 PoseBusters，按中性 50 分处理。"}
    else:
        score = max(0.0, min(100.0, float(pose_pass_rate) * 100.0))
        pose = {
            "score": round(score, 1),
            "available": True,
            "note": f"PoseBusters 通过率 {score:.0f}%。",
        }
    components = [
        {"key": "structure", "label": "受体结构质量", "weight": _WEIGHTS["structure"], **structure},
        {"key": "domain", "label": "模型适用域", "weight": _WEIGHTS["domain"], **domain},
        {"key": "consensus", "label": "引擎/构象一致性", "weight": _WEIGHTS["consensus"], **consensus},
        {"key": "calibration", "label": "实验校准", "weight": _WEIGHTS["calibration"], **calibration},
        {"key": "pose", "label": "构象物理有效性", "weight": _WEIGHTS["pose"], **pose},
    ]
    total = sum(item["score"] * item["weight"] for item in components)
    if total >= 75:
        level, color = "高", "ok"
    elif total >= 55:
        level, color = "中", "warn"
    else:
        level, color = "低", "err"
    abstain = total < 45
    return {
        "score": round(total, 1),
        "level": level,
        "color": color,
        "abstain": abstain,
        "components": components,
        "advice": (
            "可信度低：当前排序不建议直接用于立项或合成决策，"
            "请先补充实验校准、换用更高质量结构或增加引擎/种子重复。"
            if abstain
            else "可信度尚可：可用于候选排序与下一轮实验设计，但结论仍需实测确认。"
        ),
    }
