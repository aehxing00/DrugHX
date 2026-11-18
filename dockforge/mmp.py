"""Matched molecular pair (MMP) analysis for project-specific SAR rules.

Generic transformation libraries are useful, but a project's own assay and
docking data contain stronger evidence: "adding fluorine at this position
improves potency by 0.4 log units in this series".  This module fragments every
ligand at single non-ring bonds, groups molecules that share the same core, and
aggregates the substituent changes into directional rules with sample size and
spread.
"""
from __future__ import annotations

import math

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMMPA

RDLogger.DisableLog("rdApp.*")

_MAX_RULES = 40


def _canonical(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles or "")
    return Chem.MolToSmiles(mol) if mol is not None else ""


def fragments(smiles: str) -> list[tuple[str, str]]:
    """Return ``(core, substituent)`` pairs from single-bond cuts."""
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None or mol.GetNumHeavyAtoms() < 6:
        return []
    pairs = []
    try:
        cuts = rdMMPA.FragmentMol(mol, maxCuts=1, resultsAsMols=False)
    except Exception:
        return []
    for first, second in cuts:
        parts = [
            part for part in f"{first}.{second}".split(".") if part
        ]
        if len(parts) != 2:
            continue
        core, side = parts
        core_mol = Chem.MolFromSmiles(core)
        side_mol = Chem.MolFromSmiles(side)
        if core_mol is None or side_mol is None:
            continue
        if side_mol.GetNumHeavyAtoms() > core_mol.GetNumHeavyAtoms():
            core, side = side, core
        core_canonical = _canonical(core)
        side_canonical = _canonical(side)
        if core_canonical and side_canonical:
            pairs.append((core_canonical, side_canonical))
    return pairs


def _value(record: dict) -> tuple[float | None, str]:
    if record.get("p_activity") is not None:
        return float(record["p_activity"]), "实测 pActivity"
    if record.get("affinity") is not None:
        return -float(record["affinity"]), "对接打分（-Vina）"
    return None, ""


def mine(
    ligands: list[dict],
    results: list[dict],
    assay_payload: dict | None = None,
) -> dict:
    """Mine directional SAR rules from docking and/or measured activity."""
    ligand_index = {item.get("id"): item for item in ligands}
    affinity = {
        item.get("ligand_id"): float(item["affinity"])
        for item in results
        if item.get("status") == "ok" and item.get("affinity") is not None
    }
    measured = {
        row.get("ligand_id"): float(row["p_activity"])
        for row in ((assay_payload or {}).get("rows") or [])
        if row.get("ligand_id") and row.get("p_activity") is not None
    }
    records = []
    for ligand_id, ligand in ligand_index.items():
        smiles = (
            ligand.get("smiles")
            or (ligand.get("meta") or {}).get("smiles")
            or ""
        )
        if not smiles:
            continue
        if ligand_id not in affinity and ligand_id not in measured:
            continue
        records.append(
            {
                "ligand_id": ligand_id,
                "name": ligand.get("name") or ligand_id,
                "smiles": smiles,
                "affinity": affinity.get(ligand_id),
                "p_activity": measured.get(ligand_id),
            }
        )

    groups: dict[str, list[dict]] = {}
    for record in records:
        for core, side in fragments(record["smiles"]):
            groups.setdefault(core, []).append({**record, "side": side})

    rules: dict[str, dict] = {}
    pairs = 0
    for core, members in groups.items():
        if len(members) < 2:
            continue
        for index in range(len(members)):
            for other in range(index + 1, len(members)):
                first, second = members[index], members[other]
                if first["side"] == second["side"]:
                    continue
                value_a, source = _value(first)
                value_b, _source_b = _value(second)
                if value_a is None or value_b is None:
                    continue
                pairs += 1
                delta = value_b - value_a
                key = f"{first['side']}>>{second['side']}"
                entry = rules.setdefault(
                    key,
                    {
                        "from": first["side"],
                        "to": second["side"],
                        "core": core,
                        "deltas": [],
                        "examples": [],
                        "source": source,
                    },
                )
                entry["deltas"].append(delta)
                if len(entry["examples"]) < 3:
                    entry["examples"].append(
                        {
                            "from": first["name"],
                            "to": second["name"],
                            "delta": round(delta, 3),
                        }
                    )

    output = []
    for entry in rules.values():
        deltas = entry["deltas"]
        n = len(deltas)
        mean = sum(deltas) / n
        if n > 1:
            variance = sum((value - mean) ** 2 for value in deltas) / (n - 1)
            sd = math.sqrt(variance)
            se = sd / math.sqrt(n)
        else:
            sd = 0.0
            se = 0.0
        confidence = "高" if n >= 3 and abs(mean) >= 0.30 else "中" if n >= 2 else "低"
        output.append(
            {
                "from": entry["from"],
                "to": entry["to"],
                "core": entry["core"],
                "n": n,
                "mean_delta": round(mean, 3),
                "sd": round(sd, 3),
                "se": round(se, 3),
                "direction": "改善" if mean > 0 else "降低",
                "confidence": confidence,
                "source": entry["source"],
                "examples": entry["examples"],
            }
        )
    output.sort(key=lambda item: (-abs(item["mean_delta"]), -item["n"]))
    return {
        "molecules": len(records),
        "pairs": pairs,
        "rules": output[:_MAX_RULES],
        "rule_count": len(output),
        "note": (
            "规则来自项目内匹配分子对：Δ 为“后者 − 前者”，实测优先，否则用 −Vina 打分。"
            "n 为支持配对数；n=1 的规则只能作为假设，需实验验证。"
        ),
    }


def apply_rules(smiles: str, rules: list[dict], limit: int = 5) -> list[dict]:
    """Find rules whose ``from`` substituent matches the query molecule."""
    fragments_of_query = [side for _core, side in fragments(smiles)]
    hits = []
    for rule in rules:
        if rule.get("from") in fragments_of_query:
            hits.append(rule)
        if len(hits) >= limit:
            break
    return hits
