"""Synthesizability assessment and template-based retrosynthetic routing.

The module deliberately stays transparent:

* synthetic accessibility uses the published Ertl/Schuffenhauer SA score;
* routes come from a small, curated library of common medicinal-chemistry
  disconnections (amide, ester, Suzuki, reductive amination, ...);
* yields and step difficulties are heuristics used for ranking, not
  experimentally validated predictions.

Every route is therefore a *hypothesis* that must be checked against literature
and supplier catalogues before wet-lab work.
"""
from __future__ import annotations

import itertools
import math
import sys
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdDepictor, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")

from . import figures, i18n
from .project import Project
from .utils import now_iso, safe_name, write_json

# ---------------------------------------------------------------------------
# SA score (RDKit Contrib)
# ---------------------------------------------------------------------------
_SA_SCORER = None


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


def _sa_score(mol: Chem.Mol) -> float | None:
    scorer = _sascorer()
    if not scorer:
        return None
    try:
        return round(float(scorer.calculateScore(mol)), 2)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# curated retro-template library
# ---------------------------------------------------------------------------
_TEMPLATES = [
    {
        "key": "amide",
        "label": "酰胺缩合",
        "smarts": "[C:1](=[O:2])[N:3]>>[C:1](=[O:2])[OH].[N:3]",
        "yield": 0.78,
        "difficulty": 0.15,
        "conditions": "HATU / DIPEA / DMF，室温",
        "note": "酸与胺缩合；多官能底物需检查化学选择性。",
    },
    {
        "key": "ester",
        "label": "酯化",
        "smarts": "[C:1](=[O:2])[O:3]>>[C:1](=[O:2])[OH].[O:3]",
        "yield": 0.72,
        "difficulty": 0.20,
        "conditions": "酸催化或 EDC / DMAP",
        "note": "酸与醇酯化；位阻大时产率下降。",
    },
    {
        "key": "sulfonamide",
        "label": "磺酰胺",
        "smarts": "[S:1](=[O:2])(=[O:3])[N:4]>>[S:1](=[O:2])(=[O:3])Cl.[N:4]",
        "yield": 0.70,
        "difficulty": 0.25,
        "conditions": "磺酰氯 + 胺，碱中和",
        "note": "磺酰氯对水敏感，需无水条件。",
    },
    {
        "key": "urea",
        "label": "脲",
        "smarts": "[N:1][C:2](=[O:3])[N:4]>>[N:1][C:2](=[O:3])Cl.[N:4]",
        "yield": 0.68,
        "difficulty": 0.30,
        "conditions": "氨基甲酰氯 + 胺，或 CDI 活化",
        "note": "也可用异氰酸酯路线，需注意放热与选择性。",
    },
    {
        "key": "ether",
        "label": "芳醚（Williamson）",
        "smarts": "[c:1][O:2][CH2:3][c:4]>>[c:1][O:2][H].[CH2:3](Br)[c:4]",
        "yield": 0.65,
        "difficulty": 0.25,
        "conditions": "酚 + 苄溴，K2CO3 / DMF",
        "note": "酚氧负离子与亲电试剂；注意 O/N 竞争烷基化。",
    },
    {
        "key": "suzuki",
        "label": "Suzuki 偶联",
        "smarts": "[c:1]-[c:2]>>[c:1]Br.[c:2]B(O)O",
        "yield": 0.66,
        "difficulty": 0.30,
        "conditions": "Pd 催化，芳卤 + 芳硼酸，碱 / 水-有机相",
        "note": "需预先制备或采购卤代物与硼酸；注意杂环配位。",
    },
    {
        "key": "reductive_amination",
        "label": "还原胺化",
        "smarts": "[c:1][CH2:2][N:3]>>[c:1][CH:2]=O.[N:3]",
        "yield": 0.72,
        "difficulty": 0.20,
        "conditions": "醛 + 胺，NaBH3CN / MeOH",
        "note": "苄位醛与胺还原胺化；需控制亚胺还原选择性。",
    },
    {
        "key": "ketone_reductive",
        "label": "酮还原胺化",
        "smarts": "[C:1]([C:2])([C:3])[N:4]>>[C:1]([C:2])([C:3])=O.[N:4]",
        "yield": 0.70,
        "difficulty": 0.25,
        "conditions": "酮 + 胺，NaBH3CN 或 Ru 催化",
        "note": "空间位阻较大时需筛选还原剂与溶剂。",
    },
    {
        "key": "snar",
        "label": "芳香亲核取代",
        "smarts": "[c:1][N:2]>>[c:1][F].[N:2]",
        "yield": 0.62,
        "difficulty": 0.30,
        "conditions": "缺电子芳卤 + 胺，碱 / 加热",
        "note": "仅适用于邻/对位有吸电子基的杂芳环。",
    },
]

_COMMON_BUILDING_BLOCKS = {
    "O=C(O)c1ccccc1",
    "Nc1ccccc1",
    "Oc1ccccc1",
    "c1ccncc1",
    "c1ccccc1",
    "O=C(O)C",
    "NCC",
    "CN",
    "CCO",
    "O=C(O)C(F)(F)F",
    "B(O)(O)c1ccccc1",
    "Brc1ccccc1",
    "Clc1ccccc1",
    "O=C(Cl)c1ccccc1",
    "NCCN",
    "O=C1CCCN1",
}

_REACTIVE_PATTERNS = [
    ("酰氯", "[CX3](=O)[Cl,Br]"),
    ("叠氮", "[N-]=[N+]=N"),
    ("重氮", "[N+]#N"),
    ("环氧化物", "C1OC1"),
    ("醛", "[CX3H1](=O)"),
    ("Michael 受体", "C=CC=O"),
    ("硝基", "[N+](=O)[O-]"),
    ("硼酸", "B(O)O"),
    ("烷基卤", "[CX4][Cl,Br,I]"),
]

_CHIRAL_HARD = 2


def _canonical(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol else ""


def _matches(mol: Chem.Mol, smarts: str) -> list[tuple]:
    query = Chem.MolFromSmarts(smarts)
    if query is None:
        return []
    return list(mol.GetSubstructMatches(query))


def _template_reaction(template: dict):
    return AllChem.ReactionFromSmarts(template["smarts"])


def _precursor_record(mol: Chem.Mol) -> dict:
    smiles = Chem.MolToSmiles(mol)
    heavy = int(mol.GetNumHeavyAtoms())
    sa = _sa_score(mol)
    return {
        "smiles": smiles,
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "heavy": heavy,
        "sa": sa,
        "common": smiles in _COMMON_BUILDING_BLOCKS
        or (heavy <= 8 and (sa is not None and sa <= 3.2)),
    }


def _cap_dummies(mol: Chem.Mol) -> Chem.Mol | None:
    """Replace BRICS attachment dummies with hydrogen and return a clean mol."""
    try:
        rw = Chem.RWMol(mol)
        for atom in rw.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atom.SetAtomicNum(1)
                atom.SetIsotope(0)
                atom.SetAtomMapNum(0)
        Chem.SanitizeMol(rw)
        return Chem.RemoveHs(rw.GetMol())
    except Exception:
        return None


def _brics_candidate(mol: Chem.Mol) -> list[dict]:
    """Fallback disconnection using RDKit BRICS when no template matches."""
    try:
        from rdkit.Chem import BRICS

        fragments = BRICS.BRICSDecompose(mol)
    except Exception:
        return []
    precursors = []
    for fragment in fragments:
        fmol = Chem.MolFromSmiles(fragment)
        if fmol is None:
            continue
        capped = _cap_dummies(fmol)
        if capped is None or capped.GetNumHeavyAtoms() < 2:
            continue
        if any(atom.GetNumRadicalElectrons() for atom in capped.GetAtoms()):
            continue
        precursors.append(_precursor_record(capped))
    if len(precursors) < 2:
        return []
    unique = {}
    for item in precursors:
        unique[item["smiles"]] = item
    precursors = list(unique.values())
    return [
        {
            "reactants": precursors,
            "precursors": [item["smiles"] for item in precursors],
            "reaction": "BRICS 拆解",
            "template": "brics",
            "yield": 0.55,
            "difficulty": 0.40,
            "conditions": "按 BRICS 断键选择商品化砌块，再匹配偶联/缩合条件",
            "note": "仅给出断键方案，不代表已找到可行反应；需人工匹配条件。",
            "product": Chem.MolToSmiles(mol),
        }
    ]


def _apply_templates(mol: Chem.Mol, max_candidates: int = 6) -> list[dict]:
    parent_smiles = Chem.MolToSmiles(mol)
    parent_heavy = int(mol.GetNumHeavyAtoms())
    found: dict[tuple[str, ...], dict] = {}
    for template in _TEMPLATES:
        try:
            reaction = _template_reaction(template)
            product_sets = reaction.RunReactants((mol,))
        except Exception:
            continue
        for product_set in product_sets:
            precursor_mols = []
            valid = True
            for product in product_set:
                try:
                    from rdkit import RDLogger

                    # Reaction templates can generate a chemically invalid
                    # product; reject it quietly instead of logging RDKit's
                    # valence error for every enumeration attempt.
                    with RDLogger.BlockLogs():
                        Chem.SanitizeMol(product)
                except Exception:
                    valid = False
                    break
                if product.GetNumHeavyAtoms() < 2:
                    valid = False
                    break
                if any(atom.GetNumRadicalElectrons() for atom in product.GetAtoms()):
                    valid = False
                    break
                if any(atom.GetAtomicNum() == 0 for atom in product.GetAtoms()):
                    valid = False
                    break
                precursor_mols.append(product)
            if not valid or not precursor_mols:
                continue
            precursor_mols = [
                item for item in precursor_mols
                if Chem.MolToSmiles(item) != parent_smiles
            ]
            if len(precursor_mols) < 1:
                continue
            total_heavy = sum(item.GetNumHeavyAtoms() for item in precursor_mols)
            if total_heavy > parent_heavy + 5:
                continue
            if total_heavy < max(4, int(parent_heavy * 0.45)):
                continue
            precursors = [
                _precursor_record(item) for item in precursor_mols
            ]
            key = tuple(sorted(item["smiles"] for item in precursors))
            if key in found:
                continue
            found[key] = {
                "reactants": precursors,
                "precursors": [item["smiles"] for item in precursors],
                "reaction": template["label"],
                "template": template["key"],
                "yield": template["yield"],
                "difficulty": template["difficulty"],
                "conditions": template["conditions"],
                "note": template["note"],
                "product": parent_smiles,
            }
    candidates = list(found.values())
    if not candidates:
        candidates = _brics_candidate(mol)
    candidates.sort(key=lambda item: (-item["yield"], item["difficulty"]))
    return candidates[:max_candidates]


def _expand(
    mol: Chem.Mol,
    depth: int,
    seen: set[str],
    branch: int = 4,
) -> list[dict]:
    smiles = Chem.MolToSmiles(mol)
    leaf = _precursor_record(mol)
    if depth <= 0:
        return [{"steps": [], "leaves": [leaf]}]
    candidates = _apply_templates(mol, max_candidates=branch)
    if not candidates:
        return [{"steps": [], "leaves": [leaf]}]
    routes = []
    for candidate in candidates:
        child_lists = []
        for precursor in candidate["reactants"]:
            p_smiles = precursor["smiles"]
            p_mol = Chem.MolFromSmiles(p_smiles)
            if p_mol is None:
                child_lists.append([])
                continue
            if (
                p_smiles in seen
                or precursor["common"]
                or depth <= 1
            ):
                child_lists.append([{"steps": [], "leaves": [precursor]}])
                continue
            children = _expand(p_mol, depth - 1, seen | {p_smiles}, branch=2)
            if not children:
                children = [{"steps": [], "leaves": [precursor]}]
            children.sort(
                key=lambda item: -sum(
                    step["yield"] for step in item["steps"]
                )
            )
            child_lists.append(children[:2])
        if not child_lists or any(not items for items in child_lists):
            continue
        for combination in itertools.product(*child_lists):
            steps = [candidate] + [
                step for child in combination for step in child["steps"]
            ]
            leaves = [
                leaf_item for child in combination for leaf_item in child["leaves"]
            ]
            routes.append({"steps": steps, "leaves": leaves})
    return routes


def _route_score(steps: list[dict], leaves: list[dict], parent_sa: float | None) -> float:
    if not steps:
        return 0.0
    overall_yield = float(np.prod([step["yield"] for step in steps]))
    leaf_scores = [
        float(leaf["sa"]) for leaf in leaves if leaf.get("sa") is not None
    ]
    mean_leaf = float(np.mean(leaf_scores)) if leaf_scores else (parent_sa or 6.0)
    sa_component = max(0.0, min(1.0, (9.0 - mean_leaf) / 6.5))
    step_penalty = 0.08 * max(0, len(steps) - 1)
    hard_penalty = 0.10 * sum(float(step["difficulty"]) for step in steps)
    score = 100 * (
        0.38 * overall_yield
        + 0.34 * sa_component
        + 0.18 * max(0.0, 1.0 - step_penalty)
        + 0.10 * max(0.0, 1.0 - hard_penalty)
    )
    return round(max(0.0, min(100.0, score)), 1)


def _route_label(steps: list[dict]) -> str:
    if not steps:
        return "—"
    return " → ".join(step["reaction"] for step in reversed(steps))


def _dedupe_routes(routes: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for route in routes:
        key = tuple(
            (step["template"], tuple(sorted(step["precursors"])))
            for step in route["steps"]
        ) + tuple(sorted(leaf["smiles"] for leaf in route["leaves"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(route)
    return unique


def _flags(mol: Chem.Mol, routes: list[dict]) -> list[str]:
    flags = []
    stereocenters = Chem.FindMolChiralCenters(
        mol, includeUnassigned=True, useLegacyImplementation=False
    )
    if len(stereocenters) >= _CHIRAL_HARD:
        if i18n.get_lang() == "en":
            flags.append(
                f"{len(stereocenters)} stereocentres: chiral synthesis, enzymatic "
                "resolution or chiral chromatography is required and may lengthen the route."
            )
        else:
            flags.append(
                f"含 {len(stereocenters)} 个手性中心：需手性合成、酶拆分或手性色谱，"
                "是路线中最可能拉长步骤的环节。"
            )
    ring_info = mol.GetRingInfo()
    macro = [
        len(ring) for ring in ring_info.AtomRings() if len(ring) >= 12
    ]
    if macro:
        if i18n.get_lang() == "en":
            flags.append(
                f"{len(macro)} macrocycle(s) (largest {max(macro)}-membered): "
                "ring closure usually requires high-dilution conditions."
            )
        else:
            flags.append(
                f"含 {len(macro)} 个大环（最大 {max(macro)} 元）：成环步骤通常需要高稀释条件。"
            )
    for label, smarts in _REACTIVE_PATTERNS:
        hits = _matches(mol, smarts)
        if hits:
            if i18n.get_lang() == "en":
                flags.append(
                    f"Detected {label} ({len(hits)} site(s)): stability and compatibility "
                    "must be assessed separately when used as a building block or intermediate."
                )
            else:
                flags.append(
                    f"检测到 {label}（{len(hits)} 处）：作为砌块或中间体时需要单独评估稳定性与兼容性。"
                )
    reactions = {step["template"] for route in routes for step in route["steps"]}
    if "amide" in reactions and (
        _matches(mol, "[OX2H]") or len(_matches(mol, "[NX3;H2,H1]")) > 1
    ):
        if i18n.get_lang() == "en":
            flags.append(
                "The amide route contains a free hydroxyl or several amines: protecting "
                "groups or orthogonal activation conditions may be required."
            )
        else:
            flags.append(
                "酰胺路线中存在游离羟基或多个氨基：可能需要保护基或选择正交活化条件。"
            )
    if "suzuki" in reactions:
        if i18n.get_lang() == "en":
            flags.append(
                "The Suzuki route requires both a halide and a boronic acid building block; "
                "with heterocycles, check Pd coordination and dehalogenation side reactions."
            )
        else:
            flags.append(
                "Suzuki 路线需要卤代物与硼酸两个砌块；杂环存在时需检查 Pd 配位与脱卤副反应。"
            )
    return flags[:6]


def _recommendations(metrics: dict, routes: list[dict], flags: list[str]) -> list[str]:
    advice = []
    if metrics.get("sa", 0) and float(metrics["sa"]) > 4.5:
        if i18n.get_lang() == "en":
            advice.append(
                "High SAscore: reduce unnecessary stereocentres, avoid macrocycles, "
                "or split complex fragments into commercial building blocks."
            )
        else:
            advice.append(
                "SAscore 偏高：优先减少非必要手性中心、避免大环，或把复杂片段拆成商品化砌块。"
            )
    if routes:
        best = routes[0]
        if i18n.get_lang() == "en":
            step_count = len(best["steps"])
            advice.append(
                f"The preferred route is {best['label']}, with an estimated "
                f"{step_count} step{'s' if step_count != 1 else ''} and an overall yield of about "
                f"{best['overall_yield']:.0f}%; confirm the first step on a small scale."
            )
        else:
            advice.append(
                f"首选路线为 {best['label']}，预计 {len(best['steps'])} 步、"
                f"总产率约 {best['overall_yield']:.0f}%；建议先做小试确认第一步。"
            )
        if any(step["template"] == "suzuki" for step in best["steps"]):
            if i18n.get_lang() == "en":
                advice.append(
                    "For the Suzuki coupling, confirm the availability of the halide and "
                    "boronic acid, and keep a non-Pd alternative route ready."
                )
            else:
                advice.append(
                    "若采用 Suzuki 偶联，先确认卤代物与硼酸的可得性，并准备备用非 Pd 偶联路线。"
                )
        if any(step["template"] == "amide" for step in best["steps"]):
            if i18n.get_lang() == "en":
                advice.append(
                    "For the amide coupling, screen both HATU and EDC conditions and check "
                    "competing reactions from free OH/NH groups."
                )
            else:
                advice.append(
                    "酰胺缩合步骤建议筛选 HATU/EDC 两套条件，并检查游离 OH/NH 的竞争反应。"
                )
    if flags:
        if i18n.get_lang() == "en":
            advice.append(
                "Hard steps are flagged on the left; check the literature or obtain "
                "supplier quotes for each of them."
            )
        else:
            advice.append("路线中的硬步骤已在左侧标注，建议逐条做文献检索或供应商询价。")
    if i18n.get_lang() == "en":
        advice.append(
            "These routes come from rule-based templates rather than a SYNTHIA-scale "
            "retrosynthesis search; yields are empirical estimates and must be verified experimentally."
        )
    else:
        advice.append(
            "以上路线来自规则模板，不是 SYNTHIA 级全库逆合成；产率为经验估计，必须经实验验证。"
        )
    return advice[:5]


_VERDICT_EN = {
    "未安装 SAscore，无法判定": "SAscore unavailable",
    "易合成": "Easy to synthesize",
    "中等": "Moderate",
    "偏难": "Difficult",
    "困难": "Very difficult",
}


def _verdict_for_sa(sa: float | None) -> str:
    """Language-aware synthesizability verdict derived from the SAscore.

    The English words are kept local because the shared translation table maps
    "中等" to "Moderately soluble" for the solubility module, which would be
    wrong as a synthesizability verdict.
    """
    if sa is None:
        zh = "未安装 SAscore，无法判定"
    elif sa <= 3.0:
        zh = "易合成"
    elif sa <= 4.0:
        zh = "中等"
    elif sa <= 5.0:
        zh = "偏难"
    else:
        zh = "困难"
    return _VERDICT_EN[zh] if i18n.get_lang() == "en" else zh


def assess(smiles: str, name: str = "") -> dict:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        raise ValueError("SMILES 无法解析，请检查输入")
    mol = Chem.RemoveHs(mol)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    canonical = Chem.MolToSmiles(mol)
    parent_sa = _sa_score(mol)
    heavy = int(mol.GetNumHeavyAtoms())
    rings = int(rdMolDescriptors.CalcNumRings(mol))
    aromatic_rings = int(rdMolDescriptors.CalcNumAromaticRings(mol))
    rotb = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
    stereocenters = len(
        Chem.FindMolChiralCenters(
            mol, includeUnassigned=True, useLegacyImplementation=False
        )
    )
    metrics = {
        "mw": round(float(Descriptors.MolWt(mol)), 2),
        "heavy": heavy,
        "rings": rings,
        "aromatic_rings": aromatic_rings,
        "rotb": rotb,
        "stereocenters": stereocenters,
        "fsp3": round(float(rdMolDescriptors.CalcFractionCSP3(mol)), 3),
        "bertz": round(float(Descriptors.BertzCT(mol)), 1),
        "sa": parent_sa,
    }
    routes = _dedupe_routes(_expand(mol, depth=2, seen={canonical}, branch=5))
    for route in routes:
        route["score"] = _route_score(route["steps"], route["leaves"], parent_sa)
        route["overall_yield"] = round(
            float(np.prod([step["yield"] for step in route["steps"]])) * 100, 1
        ) if route["steps"] else 0.0
        route["label"] = _route_label(route["steps"])
        route["step_count"] = len(route["steps"])
    routes.sort(
        key=lambda item: (
            -item["score"],
            -item["overall_yield"],
            item["step_count"],
        )
    )
    for index, route in enumerate(routes[:8], start=1):
        route["rank"] = index
    routes = routes[:8]
    verdict = _verdict_for_sa(parent_sa)
    flags = _flags(mol, routes)
    advice = _recommendations(metrics, routes, flags)
    return {
        "id": f"synth_{uuid.uuid4().hex[:8]}",
        "created": now_iso(),
        "compound": {"name": name or "候选分子", "smiles": canonical},
        "metrics": metrics,
        "sa": parent_sa,
        "verdict": verdict,
        "routes": routes,
        "best_route": routes[0] if routes else None,
        "flags": flags,
        "recommendations": advice,
        "method": "template-based retrosynthesis (heuristic)",
    }


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------
def synthesis_dir(project: Project, record_id: str) -> Path:
    return project.root / "exports" / "synthesis" / record_id


def save_synthesis(project: Project, record: dict) -> dict:
    path = synthesis_dir(project, record["id"])
    path.mkdir(parents=True, exist_ok=True)
    write_json(path / "synthesis.json", record)
    project.data.setdefault("synthesis", [])
    project.data["synthesis"] = [
        item for item in project.data["synthesis"] if item.get("id") != record["id"]
    ]
    project.data["synthesis"].insert(0, record)
    project.data["synthesis"] = project.data["synthesis"][:20]
    project.save()
    return record


def load_synthesis(project: Project, record_id: str = "") -> dict:
    records = project.data.get("synthesis") or []
    if record_id:
        record = next(
            (item for item in records if item.get("id") == record_id), None
        )
        if record:
            path = synthesis_dir(project, record_id) / "synthesis.json"
            if path.is_file():
                try:
                    import json

                    return json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            return record
    if records:
        return records[0]
    return {}


# ---------------------------------------------------------------------------
# figure helpers
# ---------------------------------------------------------------------------
def _mol_image(smiles: str, width: int = 360, height: int = 240):
    from PIL import Image
    import io

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize

        mol = rdMolStandardize.Uncharger().uncharge(mol)
    except Exception:
        pass
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    options = drawer.drawOptions()
    options.padding = 0.04
    options.bondLineWidth = 2.0
    options.fixedFontSize = 17
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB")


def _gauge(ax, sa: float | None, verdict: str):
    import textwrap

    from matplotlib.patches import Wedge

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.40, 1.15)
    segments = [
        (1.0, 3.0, "#0f766e"),
        (3.0, 5.0, "#b45309"),
        (5.0, 10.0, "#b91c1c"),
    ]
    for low, high, color in segments:
        theta1 = 180 - (high - 1) / 9 * 180
        theta2 = 180 - (low - 1) / 9 * 180
        ax.add_patch(
            Wedge((0, 0), 1.0, theta1, theta2, width=0.26, facecolor=color,
                  edgecolor="white", linewidth=1.2)
        )
    for value in (1, 3, 5, 7, 10):
        theta = math.radians(180 - (value - 1) / 9 * 180)
        ax.text(
            1.19 * math.cos(theta), 1.19 * math.sin(theta), str(value),
            ha="center", va="center", fontsize=9.5, color="#64748b",
        )
    if sa is not None:
        theta = math.radians(180 - (min(10, max(1, sa)) - 1) / 9 * 180)
        ax.scatter(
            [0.87 * math.cos(theta)], [0.87 * math.sin(theta)],
            s=120, color="#0f172a", edgecolor="white", linewidth=1.6,
            zorder=7,
        )
        ax.text(
            0, 0.46, f"{sa:.2f}", ha="center", va="center",
            fontsize=20, fontweight="bold", color="#0f172a", zorder=10,
        )
    verdict_text = textwrap.fill(
        str(verdict or "—"), 16 if i18n.get_lang() == "en" else 5
    )
    ax.text(
        0, 0.06, verdict_text, ha="center", va="center",
        fontsize=10.5, fontweight="bold", color="#0f172a", zorder=10,
        linespacing=1.25,
    )
    ax.text(
        0, -0.17, "SAscore", ha="center", va="center",
        fontsize=9.5, color="#64748b", zorder=10,
    )


_REACTION_SHORT = {
    "酰胺缩合": "酰胺",
    "酯化": "酯化",
    "磺酰胺": "磺酰胺",
    "脲": "脲",
    "芳醚（Williamson）": "芳醚",
    "Suzuki 偶联": "Suzuki",
    "还原胺化": "还原胺化",
    "酮还原胺化": "酮还原胺化",
    "芳香亲核取代": "SNAr",
    "BRICS 拆解": "BRICS",
}


def _short_route_label(route: dict) -> str:
    labels = [
        _REACTION_SHORT.get(step["reaction"], step["reaction"])
        for step in reversed(route.get("steps") or [])
    ]
    text = " → ".join(labels) if labels else "砌块建议"
    text = i18n.tr(text)
    if len(text) > 22:
        text = text[:21] + "…"
    return text


def overview_pdf(data: dict, outpath: Path) -> Path:
    import textwrap

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    figures._style_cjk()
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    routes = data.get("routes") or []
    metrics = data.get("metrics") or {}
    leaves = []
    for route in routes[:6]:
        for leaf in route.get("leaves") or []:
            leaves.append(leaf)
    fig = figures._a4_page(
        "图 16　候选分子可合成性画像",
        f"{data.get('compound', {}).get('name', '候选分子')} 的 SAscore、路线评分与砌块复杂度",
        "Fig. 16　Synthesizability profile",
    )
    grid = fig.add_gridspec(
        2, 2, width_ratios=[1.0, 1.15], height_ratios=[1.0, 1.0],
        wspace=0.62, hspace=0.42,
        left=0.10, right=0.93, top=0.845, bottom=0.235,
    )
    ax1 = fig.add_subplot(grid[0, 0])
    _gauge(ax1, data.get("sa"), _verdict_for_sa(data.get("sa")))
    ax1.set_title("合成可及性", loc="left", fontsize=15, pad=12)
    figures._panel_tag(ax1, "a", x=-0.18, y=1.08)
    ax1.text(
        0.0, -0.36,
        f"MW {metrics.get('mw', 0):.0f} · 重原子 {metrics.get('heavy', 0)}\n"
        f"环 {metrics.get('rings', 0)} · 手性 {metrics.get('stereocenters', 0)}",
        ha="center", va="top", fontsize=9.4, color="#64748b",
        linespacing=1.30,
    )

    ax2 = fig.add_subplot(grid[0, 1])
    shown = routes[:5]
    if shown:
        labels = [
            f"R{item['rank']} {_short_route_label(item)}" for item in shown
        ]
        scores = [item["score"] for item in shown]
        y = np.arange(len(shown))
        ax2.barh(y, scores, color="#0f766e", height=0.58, edgecolor="none")
        for index, item in enumerate(shown):
            ax2.text(
                item["score"] + 1.5, index,
                f"{item['score']:.1f} · {item['step_count']} 步 · "
                f"{item['overall_yield']:.0f}%",
                va="center", fontsize=9.6, color="#0f172a",
            )
        ax2.set_yticks(y)
        ax2.set_yticklabels(labels, fontsize=9.6)
        ax2.set_ylim(max(len(shown) - 0.4, 0.9), -0.7)
        ax2.set_xlim(0, 108)
        ax2.set_xlabel("路线评分（0–100）", fontsize=12)
        ax2.set_title("候选路线排序", loc="left", fontsize=15, pad=12)
        figures._compact_frame(ax2)
    else:
        ax2.axis("off")
        ax2.text(0.5, 0.5, "没有匹配到逆合成模板", ha="center", va="center",
                 fontsize=11, color="#64748b")
    figures._panel_tag(ax2, "b", x=-0.22, y=1.08)

    ax3 = fig.add_subplot(grid[1, 0])
    unique_leaves = {}
    for leaf in leaves:
        unique_leaves[leaf["smiles"]] = leaf
    if unique_leaves:
        xs = [item["heavy"] for item in unique_leaves.values()]
        ys = [item["sa"] if item["sa"] is not None else 6.0
              for item in unique_leaves.values()]
        common = [item.get("common") for item in unique_leaves.values()]
        for flag, color, label in (
            (True, "#0f766e", "常见砌块（启发式）"),
            (False, "#94a3b8", "需进一步拆解"),
        ):
            indices = [i for i, value in enumerate(common) if value is flag]
            if indices:
                ax3.scatter(
                    [xs[i] for i in indices], [ys[i] for i in indices],
                    s=70, color=color, edgecolor="white", linewidth=0.8,
                    label=label, zorder=3,
                )
        grouped: dict[tuple[int, int], list[str]] = {}
        for index, leaf in enumerate(unique_leaves.values()):
            key = (int(xs[index]), int(round(float(ys[index]))))
            grouped.setdefault(key, []).append(leaf.get("formula", ""))
        for offset_index, ((x_value, y_value), formulas) in enumerate(grouped.items()):
            offset = (6, 6) if offset_index % 2 == 0 else (-6, -12)
            ha = "left" if offset_index % 2 == 0 else "right"
            ax3.annotate(
                " / ".join(item for item in formulas if item),
                (x_value, y_value),
                xytext=offset, textcoords="offset points",
                ha=ha, fontsize=8.6, color="#64748b",
            )
        ax3.axvline(12, color="#cbd5e1", linewidth=0.8, linestyle="--")
        ax3.axhline(3.5, color="#cbd5e1", linewidth=0.8, linestyle="--")
        ax3.set_xlabel("砌块重原子数", fontsize=12)
        ax3.set_ylabel("砌块 SAscore", fontsize=12)
        ax3.set_title("末端砌块复杂度", loc="left", fontsize=15, pad=12)
        ax3.margins(x=0.16, y=0.22)
        ax3.legend(frameon=False, fontsize=9.6, loc="lower right")
        figures._compact_frame(ax3, "both")
    else:
        ax3.axis("off")
        ax3.text(0.5, 0.5, "没有末端砌块记录", ha="center", va="center",
                 fontsize=11, color="#64748b")
    figures._panel_tag(ax3, "c", x=-0.20, y=1.08)

    ax4 = fig.add_subplot(grid[1, 1])
    ax4.axis("off")
    ax4.set_title("硬步骤与建议", loc="left", fontsize=15, pad=12)
    english = i18n.get_lang() == "en"
    mol = Chem.MolFromSmiles((data.get("compound") or {}).get("smiles") or "")
    flags = _flags(mol, routes) if mol is not None else (data.get("flags") or [])
    advice_list = (
        _recommendations(metrics, routes, flags)
        if mol is not None
        else (data.get("recommendations") or [])
    )
    lines = []
    for flag in flags[:3 if english else 4]:
        lines.append(
            textwrap.fill("• " + figures._trunc(flag, 220), 38 if english else 26)
        )
    for advice in advice_list[:2]:
        lines.append(
            textwrap.fill("→ " + figures._trunc(advice, 260), 38 if english else 26)
        )
    if not lines:
        lines.append("未发现明显硬步骤；仍建议逐条核对文献与供应商目录。")
    ax4.text(
        0.0, 0.98, "\n\n".join(lines), ha="left", va="top",
        fontsize=8.6 if english else 8.4, color="#334155",
        linespacing=1.35,
    )
    figures._panel_tag(ax4, "d", x=-0.12, y=1.08)
    figures._caption(
        fig,
        f"Ertl/Schuffenhauer SAscore（1 易、10 难）与分子复杂度；"
        "模板逆合成路线按产率、砌块复杂度与步骤数综合评分；"
        "末端砌块的 SAscore 与重原子数，绿色为启发式常见砌块；"
        "（d）手性、大环、活性官能团与保护基风险。"
        "路线来自有限反应模板库，不代表 SYNTHIA 级全库搜索；产率与条件仅为经验估计，需实验验证。",
    )
    with PdfPages(outpath) as pdf:
        pdf.savefig(fig, facecolor="white")
    plt.close(fig)
    return outpath


def route_pdf(data: dict, outpath: Path) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import FancyArrowPatch

    figures._style_cjk()
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    route = (data.get("routes") or [None])[0]
    fig = figures._a4_page(
        "图 17　最佳合成路线",
        f"{data.get('compound', {}).get('name', '候选分子')} 的模板逆合成路线与步骤表",
        "Fig. 17　Best synthetic route",
    )
    if not route:
        fig.text(0.5, 0.5, "没有匹配到可展示的逆合成路线", ha="center",
                 va="center", fontsize=14, color="#64748b")
        with PdfPages(outpath) as pdf:
            pdf.savefig(fig, facecolor="white")
        plt.close(fig)
        return outpath

    fig.text(
        0.07, 0.875,
        f"路线 {route['rank']} · 评分 {route['score']:.1f}/100 · "
        f"{route['step_count']} 步 · 估计总产率 {route['overall_yield']:.0f}%",
        fontsize=11.5, color="#0f172a",
    )
    # forward order: last retrosynthetic step is the first forward step
    steps = list(reversed(route["steps"]))
    top = 0.815
    bottom = 0.335
    row_height = min(0.245, (top - bottom) / max(1, len(steps)))
    block_height = row_height * max(1, len(steps))
    row_start = (top + bottom) / 2 + block_height / 2
    for index, step in enumerate(steps):
        y = row_start - row_height * (index + 1)
        height = row_height * 0.82
        # precursors
        reactants = step["reactants"]
        count = max(1, len(reactants))
        left_width = 0.34
        for j, reactant in enumerate(reactants):
            width = left_width / count
            ax = fig.add_axes([0.07 + j * width, y + 0.01, width * 0.92, height * 0.82])
            image = _mol_image(reactant["smiles"])
            if image is not None:
                ax.imshow(image)
            ax.axis("off")
        # arrow / conditions
        arrow_x = 0.45
        fig.add_artist(
            FancyArrowPatch(
                (arrow_x, y + height * 0.45),
                (0.69, y + height * 0.45),
                transform=fig.transFigure,
                arrowstyle="-|>", mutation_scale=16,
                color="#0f766e", linewidth=1.8,
            )
        )
        fig.text(
            0.57, y + height * 0.58, step["reaction"],
            ha="center", va="bottom", fontsize=9.8, color="#0f766e",
            fontweight="bold",
        )
        condition_text = (
            f"{i18n.tr(step['conditions'])} · "
            f"{'estimated yield' if i18n.get_lang() == 'en' else '估计产率'} "
            f"{step['yield'] * 100:.0f}%"
        )
        condition_size = 7.8
        if i18n.get_lang() == "en":
            import textwrap

            condition_text = textwrap.fill(
                f"{i18n.tr(step['reaction'])} · {step['yield'] * 100:.0f}%", 18
            )
            condition_size = 6.8
        elif len(condition_text) > 30:
            import textwrap

            condition_text = textwrap.fill(condition_text, 20)
            condition_size = 7.2
        fig.text(
            0.57, y + height * 0.40, condition_text,
            ha="center", va="top", fontsize=condition_size, color="#64748b",
        )
        # product
        ax = fig.add_axes([0.69, y + 0.01, 0.26, height * 0.82])
        image = _mol_image(step["product"])
        if image is not None:
            ax.imshow(image)
        ax.axis("off")
        fig.text(
            0.82, y - 0.004, f"中间体 {index + 1}" if index < len(steps) - 1 else "目标分子",
            ha="center", va="top", fontsize=8.2, color="#64748b",
        )
    # step table
    table_top = 0.295
    rows = [["步骤", "反应类型", "条件", "估计产率", "备注"]]
    english = i18n.get_lang() == "en"
    for index, step in enumerate(steps, start=1):
        rows.append(
            [
                str(index),
                figures._wrap_cjk(i18n.tr(step["reaction"]), 16) if english
                else step["reaction"],
                figures._wrap_cjk(i18n.tr(step["conditions"]), 34) if english
                else (step["conditions"][:16] + "…"
                      if len(step["conditions"]) > 17 else step["conditions"]),
                f"{step['yield'] * 100:.0f}%",
                figures._wrap_cjk(i18n.tr(step["note"]), 44) if english
                else (step["note"][:19] + "…"
                      if len(step["note"]) > 20 else step["note"]),
            ]
        )
    figures._table(
        fig, table_top, rows,
        [0.05, 0.13, 0.24, 0.08, 0.39] if english else [0.06, 0.15, 0.27, 0.09, 0.32],
        fontsize=7.0 if english else 8.0,
        row_height=0.052 if english else 0.032,
    )
    figures._caption(
        fig,
        "路线按正向顺序展示：左侧为砌块/中间体，箭头上方为反应类型、下方为常见条件与经验产率，右侧为产物。"
        "末端的“常见砌块”仅为启发式标记，不代表已核实库存或价格；所有反应条件与产率均为规则估计，"
        "必须通过文献检索、小试与供应商询价验证。",
    )
    with PdfPages(outpath) as pdf:
        pdf.savefig(fig, facecolor="white")
    plt.close(fig)
    return outpath


def synthesis_report_pdf(data: dict, outpath: Path) -> Path:
    """Merge the two synthesis pages into one download/report file."""
    from pypdf import PdfWriter

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    overview = outpath.with_name("_overview.pdf")
    route = outpath.with_name("_route.pdf")
    overview_pdf(data, overview)
    route_pdf(data, route)
    writer = PdfWriter()
    for path in (overview, route):
        writer.append(str(path))
    with open(outpath, "wb") as handle:
        writer.write(handle)
    writer.close()
    for path in (overview, route):
        try:
            path.unlink()
        except Exception:
            pass
    return outpath


# ---------------------------------------------------------------------------
# project entry point
# ---------------------------------------------------------------------------
def run_synthesis(
    project: Project,
    smiles: str,
    name: str = "",
    options: dict | None = None,
) -> dict:
    data = assess(smiles, name)
    save_synthesis(project, data)
    outdir = synthesis_dir(project, data["id"])
    report = outdir / f"{filename(name)}_synthesis.pdf"
    data["report_pdf"] = str(report)
    try:
        synthesis_report_pdf(data, report)
    except Exception:
        data["report_pdf"] = ""
    write_json(outdir / "synthesis.json", data)
    return data


def filename(name: str) -> str:
    return safe_name(name, "synthesis")
