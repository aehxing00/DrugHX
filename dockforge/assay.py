"""Experimental assay import and docking-score calibration.

A docking ranking is only useful when it agrees with measured potency.  This
module reads a CSV/TSV export from an assay plate (name or SMILES plus IC50 /
Kd / EC50 / Ki), matches it to the project ligands, converts values to
pActivity, and reports rank correlation with the docking scores.

The expected sign is negative: a stronger binder has a more negative Vina
affinity and a higher pActivity.  A positive or near-zero correlation means the
ranking cannot be used for this series.
"""
from __future__ import annotations

import csv
import io
import math
import re
from datetime import datetime

_NAME_KEYS = {
    "name", "compound", "compound_id", "compoundid", "id", "ligand",
    "分子", "化合物", "名称", "编号",
}
_SMILES_KEYS = {"smiles", "canonical_smiles", "structure", "结构", "smiles字符串"}
_VALUE_KEYS = {
    "ic50", "kd", "ec50", "ki", "activity", "value", "potency", "result",
    "活性", "数值", "测定值", "结果",
}
_UNIT_KEYS = {"unit", "units", "单位"}

_UNIT_FACTORS = {
    "m": 1e9,
    "mol/l": 1e9,
    "mm": 1e6,
    "mmol/l": 1e6,
    "um": 1e3,
    "μm": 1e3,
    "µm": 1e3,
    "umol/l": 1e3,
    "μm/l": 1e3,
    "nm": 1.0,
    "nmol/l": 1.0,
    "pm": 1e-3,
    "pmol/l": 1e-3,
}


def _norm(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip().lower()


def _pick(header: list[str], keys: set[str]) -> int | None:
    for index, cell in enumerate(header):
        if _norm(cell) in keys:
            return index
    return None


def _number(raw: str) -> float | None:
    text = str(raw or "").strip().replace(",", "")
    text = text.replace("＜", "<").replace("＞", ">").replace("≤", "<").replace("≥", ">")
    text = text.lstrip("<>~≈= ").strip()
    match = re.search(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def parse_table(text: str) -> list[dict]:
    """Parse a CSV/TSV assay table into ``{name, smiles, value, unit}`` rows."""
    sample = (text or "").lstrip("\ufeff")
    if not sample.strip():
        return []
    first = sample.splitlines()[0]
    delimiter = "\t" if first.count("\t") > first.count(",") else ","
    reader = csv.reader(io.StringIO(sample), delimiter=delimiter)
    rows = list(reader)
    if not rows:
        return []
    header = [str(cell).strip() for cell in rows[0]]
    name_col = _pick(header, _NAME_KEYS)
    smiles_col = _pick(header, _SMILES_KEYS)
    value_col = _pick(header, _VALUE_KEYS)
    unit_col = _pick(header, _UNIT_KEYS)
    body = rows[1:]
    if value_col is None:
        # No recognisable header: assume name, smiles, value, unit.
        name_col, smiles_col, value_col, unit_col = 0, 1, 2, 3
        body = rows
    parsed = []
    for row in body:
        if not row:
            continue

        def cell(index: int | None) -> str:
            if index is None or index >= len(row):
                return ""
            return str(row[index]).strip()

        value = _number(cell(value_col))
        if value is None:
            continue
        item = {
            "name": cell(name_col),
            "smiles": cell(smiles_col),
            "value": value,
            "unit": cell(unit_col) or "nM",
        }
        parsed.append(item)
    return parsed


def to_pactivity(value: float, unit: str) -> float | None:
    """Convert IC50/Kd/EC50 to pActivity (pIC50 scale)."""
    key = _norm(unit).replace("μ", "u").replace("µ", "u")
    if key in ("p", "pic50", "pactivity", "pkd", "pec50", "pki"):
        return float(value)
    factor = _UNIT_FACTORS.get(key)
    if factor is None:
        # Unknown unit: assume nM (the most common assay unit).
        factor = 1.0
    nm = float(value) * factor
    if nm <= 0:
        return None
    return 9.0 - math.log10(nm)


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index
        while end + 1 < len(order) and values[order[end + 1]] == values[order[index]]:
            end += 1
        average = (index + end) / 2.0 + 1.0
        for position in range(index, end + 1):
            ranks[order[position]] = average
        index = end + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0 or var_y <= 0:
        return None
    return cov / math.sqrt(var_x * var_y)


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    return _pearson(_ranks(xs), _ranks(ys))


def _canonical(smiles: str) -> str:
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles or "")
        return Chem.MolToSmiles(mol) if mol is not None else ""
    except Exception:
        return ""


def calibrate(
    rows: list[dict],
    ligands: list[dict],
    results: list[dict],
) -> dict:
    """Match assay rows to docked ligands and quantify ranking agreement."""
    name_map = {}
    smiles_map = {}
    for ligand in ligands:
        name = str(ligand.get("name") or "").strip().lower()
        if name:
            name_map[name] = ligand
        smiles = (
            ligand.get("smiles")
            or (ligand.get("meta") or {}).get("smiles")
            or ""
        )
        canonical = _canonical(smiles)
        if canonical:
            smiles_map[canonical] = ligand

    affinity_map = {}
    for result in results:
        if result.get("status") != "ok" or result.get("affinity") is None:
            continue
        affinity_map[result.get("ligand_id")] = float(result["affinity"])

    matched = []
    unmatched = 0
    for row in rows:
        ligand = None
        matched_by = ""
        if row.get("smiles"):
            ligand = smiles_map.get(_canonical(row["smiles"]))
            if ligand:
                matched_by = "SMILES"
        if ligand is None and row.get("name"):
            ligand = name_map.get(str(row["name"]).strip().lower())
            if ligand:
                matched_by = "名称"
        affinity = affinity_map.get(ligand.get("id")) if ligand else None
        p_activity = to_pactivity(row["value"], row.get("unit") or "nM")
        if ligand is None or affinity is None or p_activity is None:
            unmatched += 1
            continue
        matched.append(
            {
                "name": ligand.get("name") or row.get("name"),
                "ligand_id": ligand.get("id"),
                "smiles": ligand.get("smiles") or (ligand.get("meta") or {}).get("smiles") or row.get("smiles") or "",
                "value": row["value"],
                "unit": row.get("unit") or "nM",
                "p_activity": round(p_activity, 3),
                "affinity": affinity,
                "matched_by": matched_by,
            }
        )

    summary = {
        "total": len(rows),
        "matched": len(matched),
        "unmatched": unmatched,
        "spearman": None,
        "pearson": None,
        "slope": None,
        "intercept": None,
        "verdict": "样本不足（至少需要 4 个可匹配的实测点）",
        "level": "warn",
    }
    if len(matched) >= 4:
        xs = [item["affinity"] for item in matched]
        ys = [item["p_activity"] for item in matched]
        spearman = _spearman(xs, ys)
        pearson = _pearson(xs, ys)
        slope = intercept = None
        if pearson is not None:
            mean_x = sum(xs) / len(xs)
            mean_y = sum(ys) / len(ys)
            var_x = sum((x - mean_x) ** 2 for x in xs)
            if var_x > 0:
                slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / var_x
                intercept = mean_y - slope * mean_x
        summary.update(
            {
                "spearman": None if spearman is None else round(spearman, 3),
                "pearson": None if pearson is None else round(pearson, 3),
                "slope": None if slope is None else round(slope, 4),
                "intercept": None if intercept is None else round(intercept, 3),
            }
        )
        rho = spearman if spearman is not None else 0.0
        if rho <= -0.60:
            summary["verdict"] = "对接打分与实测活性同向且相关较强，可用于本系列排序"
            summary["level"] = "ok"
        elif rho <= -0.30:
            summary["verdict"] = "相关性中等，只能用于粗排；建议增加实测点并复核质子化状态与结合盒"
            summary["level"] = "warn"
        elif rho >= 0.30:
            summary["verdict"] = "相关性方向相反，请检查名称/SMILES 是否对应同一分子，或确认实验读出不是效价"
            summary["level"] = "bad"
        else:
            summary["verdict"] = "相关性弱，当前打分不能用于本系列排序；建议改用 MM-GBSA 或自由能微扰复核"
            summary["level"] = "bad"
    matched.sort(key=lambda item: -item["p_activity"])
    return {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "rows": matched,
        "summary": summary,
        "note": (
            "pActivity = 9 − log10(IC50/nM)。预期相关性为负：对接打分越负、pActivity 越高。"
            "相关性只反映本系列内的排序能力，不代表绝对结合自由能。"
        ),
    }
