"""Diversity-aware batch selection for analogue design.

Enumerating analogues is cheap; preparing 3D conformers and docking them is
not.  Instead of docking every enumerated candidate, this module picks a
batch that balances predicted quality against chemical diversity, which is the
acquisition step of a simple active-learning loop.

The quality term is the desirability score already computed by the design
module; diversity is the minimum Tanimoto distance to the already selected
batch (MaxMin).  A benchmark can replace the weights without changing callers.
"""
from __future__ import annotations

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

QUALITY_WEIGHT = 0.6
DIVERSITY_WEIGHT = 0.4


def _fingerprint(smiles: str):
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def select_diverse(
    entries: list[dict],
    batch_size: int,
    quality_key: str = "score",
    quality_weight: float = QUALITY_WEIGHT,
) -> list[dict]:
    """Greedy MaxMin selection with a quality bonus.

    ``entries`` need ``smiles`` and optionally ``score``.  Selection stops at
    ``batch_size``; entries without a parsable SMILES are skipped.
    """
    if batch_size <= 0 or not entries:
        return []
    prepared = []
    for item in entries:
        fp = _fingerprint(item.get("smiles") or "")
        if fp is None:
            continue
        prepared.append((item, fp))
    if len(prepared) <= batch_size:
        return [item for item, _ in prepared]

    qualities = [float(item.get(quality_key) or 0.0) for item, _ in prepared]
    low, high = min(qualities), max(qualities)
    span = (high - low) or 1.0

    first_index = max(range(len(prepared)), key=lambda index: qualities[index])
    chosen = [first_index]
    remaining = set(range(len(prepared))) - {first_index}
    while len(chosen) < batch_size and remaining:
        best_index = None
        best_value = -1e9
        for index in remaining:
            fp = prepared[index][1]
            similarity = max(
                DataStructs.TanimotoSimilarity(fp, prepared[other][1])
                for other in chosen
            )
            quality = (qualities[index] - low) / span
            value = (
                quality_weight * quality
                + (1.0 - quality_weight) * (1.0 - similarity)
            )
            if value > best_value:
                best_value = value
                best_index = index
        if best_index is None:
            break
        chosen.append(best_index)
        remaining.discard(best_index)
    return [prepared[index][0] for index in chosen]
