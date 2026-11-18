"""Medicinal-chemistry improvement: scaffold hopping, residue advice and re-scoring."""
from __future__ import annotations

import copy
from pathlib import Path

from rdkit import Chem, DataStructs
from rdkit.Chem import (
    AllChem,
    Descriptors,
    QED,
    rdFingerprintGenerator,
    rdMolDescriptors,
)

from . import admet, analysis, docking, figures, i18n, ligands as ligmod
from .config import DEFAULT_DOCK_PARAMS
from .project import Project


def _residue_names(analysis_data: dict) -> dict:
    groups = {"hbonds": [], "hydrophobic": [], "salt": [], "pi": []}
    for item in analysis_data.get("summary", []):
        label = f"{item.get('residue')}{item.get('resseq')}"
        types = item.get("types", [])
        if "氢键" in types and label not in groups["hbonds"]:
            groups["hbonds"].append(label)
        if "疏水" in types and label not in groups["hydrophobic"]:
            groups["hydrophobic"].append(label)
        if "盐桥" in types and label not in groups["salt"]:
            groups["salt"].append(label)
        if "π-堆积" in types and label not in groups["pi"]:
            groups["pi"].append(label)
    return groups


def residue_suggestions(analysis_data: dict) -> list[dict]:
    """Turn the interaction profile into residue-level medicinal chemistry advice."""
    groups = _residue_names(analysis_data)
    rows = []
    for item in analysis_data.get("summary", [])[:12]:
        types = item.get("types", [])
        residue = f"{item.get('residue')}{item.get('resseq')}"
        interaction = "、".join(types) or "邻近"
        if "盐桥" in types:
            strategy = (
                f"保留或强化与 {residue} 的静电作用：保持相反电荷，"
                "可用四唑/磺酰胺等生物电子等排体调节 pKa 与电荷分布。"
            )
        elif "氢键" in types:
            strategy = (
                f"该残基提供关键氢键。优先保留配体对应供体/受体；"
                "若需提高亲和力，可在邻近位置引入羟基、氨基或酰胺。"
            )
        elif "π-堆积" in types:
            strategy = (
                f"与 {residue} 存在 π-堆积。可引入缺电子/富电子芳环或杂环，"
                "增强静电互补，同时避免过度增大 logP。"
            )
        elif "疏水" in types:
            strategy = (
                f"该残基构成疏水口袋。可在朝向它的位置引入卤素、甲基或环丙基，"
                "填充空腔；若 logP 偏高，优先用吡啶/氟代平衡极性。"
            )
        else:
            strategy = (
                f"该残基位于配体邻近区域，可尝试在对应方向增加极性基团或"
                "调整环系取代位置，形成新的远端接触。"
            )
        rows.append(
            {
                "residue": residue,
                "chain": item.get("chain", ""),
                "interaction": interaction,
                "distance": item.get("distance"),
                "strategy": strategy,
            }
        )
    if not rows:
        rows.append(
            {
                "residue": "整体",
                "chain": "",
                "interaction": "未检测到强作用",
                "distance": None,
                "strategy": (
                    "建议先做骨架跃迁（苯环→吡啶/嘧啶、五元杂环）与极性替换，"
                    "再根据新构象重新分析残基接触。"
                ),
            }
        )
    return rows


def _valid(mol: Chem.Mol | None, parent: Chem.Mol | None = None) -> bool:
    if mol is None or mol.GetNumAtoms() == 0:
        return False
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return False
    # Reject chemically unstable motifs that frequently arise from naive
    # linker replacement (hemiaminal/aminal/hemiacetal and peroxides).
    for smarts in (
        "[CX4H1]([NX3,O,S])[NX3,O,S]",
        "[OX2][OX2]",
    ):
        query = Chem.MolFromSmarts(smarts)
        if query is not None and mol.HasSubstructMatch(query):
            return False
    # Large natural products/macrocycles are valid optimization starting
    # points.  Allow candidates that stay close to the parent size instead of
    # rejecting everything above the generic drug-like cut-offs.
    parent_heavy = parent.GetNumHeavyAtoms() if parent is not None else 0
    parent_mw = float(Descriptors.MolWt(parent)) if parent is not None else 0.0
    if mol.GetNumHeavyAtoms() > max(50, parent_heavy + 2):
        return False
    # Reject fragment deletions: a nitrile/amide swap that cuts the molecule in
    # half is not an analogue.  Legitimate transformations change only a few
    # atoms, so keeping at least 75% of the parent heavy atoms is generous.
    if parent_heavy >= 12:
        if mol.GetNumHeavyAtoms() < 0.75 * parent_heavy:
            return False
        if Descriptors.MolWt(mol) < 0.75 * parent_mw:
            return False
    if Descriptors.MolWt(mol) > max(700.0, parent_mw + 120.0):
        return False
    if Descriptors.MolLogP(mol) > 6.5:
        return False
    return True


def _canonical(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def _add_aromatic_nitrogen(mol: Chem.Mol, atom_index: int) -> Chem.Mol | None:
    rw = Chem.RWMol(mol)
    atom = rw.GetAtomWithIdx(atom_index)
    atom.SetAtomicNum(7)
    atom.SetNoImplicit(True)
    product = rw.GetMol()
    try:
        Chem.SanitizeMol(product)
        return product
    except Exception:
        return None


def _add_fluorine(mol: Chem.Mol, atom_index: int) -> Chem.Mol | None:
    rw = Chem.RWMol(mol)
    fluorine = rw.AddAtom(Chem.Atom(9))
    rw.AddBond(atom_index, fluorine, Chem.BondType.SINGLE)
    product = rw.GetMol()
    try:
        Chem.SanitizeMol(product)
        return product
    except Exception:
        return None


def _add_hydroxyl(mol: Chem.Mol, atom_index: int) -> Chem.Mol | None:
    rw = Chem.RWMol(mol)
    oxygen = rw.AddAtom(Chem.Atom(8))
    rw.AddBond(atom_index, oxygen, Chem.BondType.SINGLE)
    product = rw.GetMol()
    try:
        Chem.SanitizeMol(product)
        return product
    except Exception:
        return None


def _methyl_to_cf3(mol: Chem.Mol, atom_index: int) -> Chem.Mol | None:
    rw = Chem.RWMol(mol)
    atom = rw.GetAtomWithIdx(atom_index)
    atom.SetNoImplicit(True)
    atom.SetNumExplicitHs(0)
    for _ in range(3):
        fluorine = rw.AddAtom(Chem.Atom(9))
        rw.AddBond(atom_index, fluorine, Chem.BondType.SINGLE)
    product = rw.GetMol()
    try:
        Chem.SanitizeMol(product)
        return product
    except Exception:
        return None


def _ester_to_amide(mol: Chem.Mol, atom_index: int) -> Chem.Mol | None:
    rw = Chem.RWMol(mol)
    atom = rw.GetAtomWithIdx(atom_index)
    atom.SetAtomicNum(7)
    atom.SetNoImplicit(False)
    product = rw.GetMol()
    try:
        Chem.SanitizeMol(product)
        return product
    except Exception:
        return None


def _run_reaction(mol: Chem.Mol, smarts: str, limit: int = 16) -> list[Chem.Mol]:
    """Run one SMARTS transformation and return sanitized unique products."""
    try:
        reaction = AllChem.ReactionFromSmarts(smarts)
    except Exception:
        return []
    try:
        outcomes = reaction.RunReactants((mol,))
    except Exception:
        return []
    products: list[Chem.Mol] = []
    seen: set[str] = set()
    for reactants in outcomes:
        candidate = reactants[0]
        try:
            Chem.SanitizeMol(candidate)
        except Exception:
            continue
        canonical = Chem.MolToSmiles(candidate, isomericSmiles=True)
        if canonical in seen:
            continue
        seen.add(canonical)
        products.append(candidate)
        if len(products) >= limit:
            break
    return products


def _attach_fragment(mol: Chem.Mol, atom_index: int, fragment_smiles: str) -> Chem.Mol | None:
    """Attach a fragment through its first atom to an aromatic carbon."""
    fragment = Chem.MolFromSmiles(fragment_smiles)
    if fragment is None or fragment.GetNumAtoms() == 0:
        return None
    rw = Chem.RWMol(mol)
    mapping = {}
    for atom in fragment.GetAtoms():
        mapping[atom.GetIdx()] = rw.AddAtom(atom)
    for bond in fragment.GetBonds():
        rw.AddBond(
            mapping[bond.GetBeginAtomIdx()],
            mapping[bond.GetEndAtomIdx()],
            bond.GetBondType(),
        )
    rw.AddBond(atom_index, mapping[0], Chem.BondType.SINGLE)
    product = rw.GetMol()
    try:
        Chem.SanitizeMol(product)
        return product
    except Exception:
        return None


def _swap_atom(mol: Chem.Mol, atom_index: int, atomic_num: int) -> Chem.Mol | None:
    rw = Chem.RWMol(mol)
    atom = rw.GetAtomWithIdx(atom_index)
    atom.SetAtomicNum(atomic_num)
    atom.SetNoImplicit(True)
    product = rw.GetMol()
    try:
        Chem.SanitizeMol(product)
        return product
    except Exception:
        return None


def _desirability(mol: Chem.Mol) -> float:
    """Cheap property prior so promising analogues survive the candidate cap."""
    mw = float(Descriptors.MolWt(mol))
    logp = float(Descriptors.MolLogP(mol))
    tpsa = float(Descriptors.TPSA(mol))
    rotb = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
    try:
        qed = float(QED.qed(mol))
    except Exception:
        qed = 0.5
    score = 0.0
    score -= abs(mw - 360.0) / 260.0
    score -= abs(logp - 2.6) / 3.8
    score -= abs(tpsa - 82.0) / 130.0
    score -= 0.05 * max(0, rotb - 6)
    score += 0.7 * qed
    return score


def _aromatic_ch_atoms(mol: Chem.Mol) -> list[int]:
    return [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetIsAromatic()
        and atom.GetAtomicNum() == 6
        and atom.GetTotalNumHs() == 1
    ]


def _combination_candidates(
    base: Chem.Mol,
    base_label: str,
    index: int,
    analysis_data: dict,
) -> list[tuple[Chem.Mol | None, str, str, str]]:
    """Second-generation ideas that combine two complementary changes."""
    products: list[tuple[Chem.Mol | None, str, str, str]] = []
    positions = _aromatic_ch_atoms(base)
    if not positions:
        return products
    choices = [
        ("氟代", "F", "在组合骨架上再引入氟，填充小疏水腔并封闭代谢软点。"),
        ("甲氧基", "OC", "叠加甲氧基以调节极性与受体方向，同时保留骨架跃迁带来的新接触。"),
        ("甲基", "C", "叠加甲基填充疏水空腔，检验骨架跃迁后的局部空间是否仍可利用。"),
        ("羟基", "O", "叠加羟基增加供体/受体，改善极性互补并降低 logP。"),
    ]
    name, fragment, rationale = choices[index % len(choices)]
    position = positions[index % len(positions)]
    candidate = _attach_fragment(base, position, fragment)
    products.append((candidate, f"组合：{base_label}+{name}", rationale, "组合改造"))
    if index % 3 == 0:
        nitrogen = _add_aromatic_nitrogen(base, position)
        products.append(
            (
                nitrogen,
                f"组合：{base_label}+氮杂环",
                "在组合骨架上再做一次芳环氮替换，探索骨架跃迁与取代基改造的叠加效应。",
                "组合改造",
            )
        )
    return products



def _carboxyl_to_tetrazole(mol: Chem.Mol) -> list[Chem.Mol]:
    reaction = AllChem.ReactionFromSmarts(
        "[CX3:1](=[OX1:2])[OX2H1,OX1-:3] >> [c:1]1[nH]nnn1"
    )
    products = []
    for reactants in reaction.RunReactants((mol,)):
        candidate = reactants[0]
        try:
            Chem.SanitizeMol(candidate)
            products.append(candidate)
        except Exception:
            continue
    return products


def propose_analogs(
    smiles: str,
    analysis_data: dict,
    max_candidates: int = 24,
) -> list[dict]:
    """Generate a broad, category-balanced analogue library.

    The library deliberately covers several medicinal-chemistry directions:
    bioisosteric replacement, aromatic scaffold hopping, heteroatom walking,
    substituent walking, linker modification and second-generation
    combinations.  A property prior and a fingerprint diversity filter keep the
    final set explorable without collapsing into near-duplicates.
    """
    parent = Chem.MolFromSmiles(smiles)
    if parent is None:
        raise ValueError("母体 SMILES 无法解析")
    groups = _residue_names(analysis_data)
    hbonds = groups["hbonds"] or ["极性残基"]
    hydrophobic = groups["hydrophobic"] or ["疏水残基"]
    logp = float(Descriptors.MolLogP(parent))
    parent_canonical = _canonical(parent)
    raw: list[dict] = []
    seen: set[str] = set()

    def add(
        mol: Chem.Mol | None,
        label: str,
        modification: str,
        priority: float,
        rationale: str,
        category: str | None = None,
    ):
        if not _valid(mol, parent):
            return
        canonical = _canonical(mol)
        if canonical in seen or canonical == parent_canonical:
            return
        seen.add(canonical)
        raw.append(
            {
                "smiles": canonical,
                "label": label,
                "modification": modification,
                "category": category or modification,
                "priority": float(priority),
                "score": float(priority) + _desirability(mol),
                "rationale": rationale,
                "targets": (
                    hbonds[:3]
                    if "氢键" in rationale or "盐桥" in rationale
                    else hydrophobic[:3]
                ),
                "generation": 1,
            }
        )

    # ------------------------------------------------------------------
    # 1. Bioisosteric replacement of acidic / amide / ester groups.
    # ------------------------------------------------------------------
    carboxyl = Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]")
    if parent.HasSubstructMatch(carboxyl):
        for product in _carboxyl_to_tetrazole(parent):
            add(
                product,
                "羧酸→四唑",
                "生物电子等排",
                10,
                f"羧酸根主要与 {'、'.join(hbonds[:3])} 形成氢键/盐桥；"
                "四唑是经典羧酸生物电子等排体，可增强氢键网络并改善代谢稳定性。",
                "生物电子等排",
            )
        for product in _run_reaction(
            parent,
            "[CX3:1](=[OX1:2])[OX2H1,OX1-:3] >> [CX3:1](=[OX1:2])[NH]S(=O)(=O)C",
        ):
            add(
                product,
                "羧酸→酰基磺酰胺",
                "生物电子等排",
                9.5,
                f"酰基磺酰胺可保持酸性并与 {'、'.join(hbonds[:3])} 形成方向性氢键，"
                "同时改变 pKa 与亲脂性，是羧酸的常用替代基团。",
                "生物电子等排",
            )
        for product in _run_reaction(
            parent,
            "[CX3:1](=[OX1:2])[OX2H1,OX1-:3] >> [CX3:1](=[OX1:2])[NH2]",
        ):
            add(
                product,
                "羧酸→伯酰胺",
                "生物电子等排",
                8.0,
                "伯酰胺去除负电荷、降低极性，可作为羧酸的弱化型等排体，"
                "用于检验静电作用是否过强。",
                "生物电子等排",
            )
        for product in _run_reaction(
            parent,
            "[CX3:1](=[OX1:2])[OX2H1,OX1-:3] >> [CX3:1](=[OX1:2])[NH]O",
        ):
            add(
                product,
                "羧酸→异羟肟酸",
                "生物电子等排",
                8.5,
                "异羟肟酸保留金属配位与氢键能力，可用于金属酶或需要双齿螯合的位点。",
                "生物电子等排",
            )

    amide_smarts = "[CX3:1](=[OX1:2])[NX3H2,NX3H1:3]"
    if parent.HasSubstructMatch(Chem.MolFromSmarts(amide_smarts)):
        for product in _run_reaction(parent, amide_smarts + " >> [CX2:1]#N"):
            add(
                product,
                "酰胺→腈",
                "生物电子等排",
                7.5,
                "腈可去除酰胺的强氢键供体并降低极性，同时保留线性几何，"
                "适合检验该酰胺是否必须作为供体。",
                "生物电子等排",
            )
        for product in _run_reaction(
            parent, amide_smarts + " >> [CX3:1](=[OX1:2])[NX3:3]C"
        ):
            add(
                product,
                "酰胺 N-甲基化",
                "代谢稳定化",
                7.5,
                "N-甲基化可阻断酰胺氮上的代谢与分子间氢键，改变构象偏好，"
                "并可能提高细胞渗透性。",
                "代谢稳定化",
            )

    # 2. Ester replacement / hydrolysis.
    ester_done = False
    for atom in parent.GetAtoms():
        if atom.GetAtomicNum() != 8 or atom.GetDegree() != 2:
            continue
        neighbors = list(atom.GetNeighbors())
        if any(
            neighbor.GetAtomicNum() == 6
            and neighbor.GetTotalNumHs() == 0
            and any(
                bond.GetBondType() == Chem.BondType.DOUBLE
                and bond.GetOtherAtom(neighbor).GetAtomicNum() == 8
                for bond in neighbor.GetBonds()
            )
            for neighbor in neighbors
        ):
            if not ester_done:
                add(
                    _ester_to_amide(parent, atom.GetIdx()),
                    "酯→酰胺",
                    "稳定性改造",
                    8.0,
                    "酯基在血浆中易水解；改为酰胺可提高稳定性，并保留氢键受体/供体方向。",
                    "代谢稳定化",
                )
                ester_done = True
            break
    for product in _run_reaction(
        parent,
        "[CX3:1](=[OX1:2])[OX2:3][#6:4] >> [CX3:1](=[OX1:2])[OX2H1:3]",
        limit=4,
    ):
        add(
            product,
            "酯→羧酸",
            "代谢稳定化",
            7.0,
            "将酯水解为羧酸可消除血浆酯酶软点，同时显著提高极性；"
            "适合局部作用或需要降低透膜的候选。",
            "代谢稳定化",
        )

    # 3. Nitrile -> tetrazole.
    if parent.HasSubstructMatch(Chem.MolFromSmarts("[CX2]#N")):
        for product in _run_reaction(parent, "[CX2:1]#N >> [c:1]1[nH]nnn1"):
            add(
                product,
                "腈→四唑",
                "生物电子等排",
                8.5,
                "腈→四唑可在保持线性连接的同时引入可电离四唑，"
                "增加氢键网络并改变分布体积。",
                "生物电子等排",
            )

    # 4. Sulfonamide N-methylation.
    for product in _run_reaction(
        parent,
        "[S:1](=[OX1])(=[OX1])[NX3H2,NX3H1:2] >> [S:1](=[OX1])(=[OX1])[NX3:2]C",
        limit=4,
    ):
        add(
            product,
            "磺酰胺 N-甲基化",
            "代谢稳定化",
            7.0,
            "磺酰胺 N-甲基化可调节酸性、增加亲脂性并阻断氮上的代谢，"
            "常用于优化磺酰胺类药物的暴露。",
            "代谢稳定化",
        )

    # ------------------------------------------------------------------
    # 5. Aromatic ring nitrogen walking and heteroatom swapping.
    # ------------------------------------------------------------------
    ring_n = 0
    for ring in parent.GetRingInfo().AtomRings():
        if not all(parent.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            continue
        carbons = [
            i
            for i in ring
            if parent.GetAtomWithIdx(i).GetAtomicNum() == 6
            and parent.GetAtomWithIdx(i).GetTotalNumHs() == 1
        ]
        if not carbons:
            continue
        all_carbon = all(
            parent.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in ring
        )
        for atom_index in carbons:
            if ring_n >= 14:
                break
            product = _add_aromatic_nitrogen(parent, atom_index)
            if all_carbon:
                label = "苯环→吡啶" if len(ring) == 6 else "芳环→氮杂芳环"
                priority = 10.0 if logp > 2.0 else 9.0
                rationale = (
                    f"该芳环与 {'、'.join(hydrophobic[:3])} 存在疏水/π 接触；"
                    "将环碳替换为氮可降低 logP、增加极性并保持环几何，是最直接的骨架跃迁。"
                )
            else:
                label = "单氮芳环→二氮芳环" if len(ring) == 6 else "五元杂环→唑/咪唑"
                priority = 8.5
                rationale = (
                    "在已有杂芳环上再引入一个氮，可微调 π 电子密度、氢键方向与 pKa，"
                    "同时保持骨架形状。"
                )
            add(product, label, "骨架跃迁", priority, rationale, "骨架跃迁")
            ring_n += 1

    swap_count = 0
    table = Chem.GetPeriodicTable()
    for ring in parent.GetRingInfo().AtomRings():
        if len(ring) != 5:
            continue
        if not all(parent.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            continue
        for atom_index in ring:
            atomic_num = parent.GetAtomWithIdx(atom_index).GetAtomicNum()
            if atomic_num == 16:
                targets = [8, 7]
            elif atomic_num == 8:
                targets = [16, 7]
            elif atomic_num == 7:
                targets = [16, 8]
            else:
                continue
            for target in targets:
                if swap_count >= 10:
                    break
                product = _swap_atom(parent, atom_index, target)
                old = table.GetElementSymbol(atomic_num)
                new = table.GetElementSymbol(target)
                add(
                    product,
                    f"五元环 {old}→{new}",
                    "骨架跃迁",
                    8.0,
                    f"将五元芳环中的 {old} 替换为 {new}，可改变氢键方向、"
                    "芳香性与电子密度，是常用的五元杂环骨架跃迁。",
                    "骨架跃迁",
                )
                swap_count += 1

    # ------------------------------------------------------------------
    # 6. Substituent walking (multiple positions, not a single token edit).
    # ------------------------------------------------------------------
    walks = [
        ("F", "芳环氟代", 7.0, 6),
        ("Cl", "芳环氯代", 6.0, 4),
        ("C", "芳环甲基化", 6.0, 4),
        ("OC", "芳环甲氧基化", 6.0, 4),
        ("O", "芳环羟基化", 6.0, 4),
        ("N", "芳环氨基化", 5.5, 3),
        ("C#N", "芳环氰基化", 6.0, 3),
    ]
    positions = _aromatic_ch_atoms(parent)
    for fragment, label, priority, cap in walks:
        count = 0
        for atom_index in positions:
            if count >= cap:
                break
            product = _attach_fragment(parent, atom_index, fragment)
            add(
                product,
                label,
                "取代基行走",
                priority,
                f"在朝向 {'、'.join(hydrophobic[:3])} 的不同芳环位置系统引入 {label[2:]}，"
                "用于扫描取代基位置对疏水填充、偶极与代谢的影响。",
                "取代基行走",
            )
            count += 1

    # 7. Terminal methyl -> CF3 at every available methyl.
    count = 0
    for atom in parent.GetAtoms():
        if (
            atom.GetAtomicNum() == 6
            and atom.GetTotalNumHs() == 3
            and atom.GetDegree() == 1
            and count < 4
        ):
            add(
                _methyl_to_cf3(parent, atom.GetIdx()),
                "甲基→CF3",
                "代谢稳定化",
                7.0,
                "末端甲基是弱疏水点且易被代谢；CF3 可填充疏水腔并阻断代谢软点。",
                "代谢稳定化",
            )
            count += 1

    # ------------------------------------------------------------------
    # 8. Linker modifications: methylene -> ether / amine / carbonyl.
    # ------------------------------------------------------------------
    linker_rules = [
        (
            "[#6:1][CH2:2][#6:3] >> [#6:1][OX2:2][#6:3]",
            "亚甲基→醚",
            "连接子改造",
            8.5,
            "把连接亚甲基改为醚氧，可保持几何方向同时降低 logP 并增加受体。",
        ),
        (
            "[#6:1][CH2:2][#6:3] >> [#6:1][NH:2][#6:3]",
            "亚甲基→胺",
            "连接子改造",
            8.0,
            "把连接亚甲基改为胺，可引入可电离中心与氢键供体，改变分布与亲和力。",
        ),
        (
            "[#6:1][CH2:2][#6:3] >> [#6:1][CX3:2](=O)[#6:3]",
            "亚甲基→羰基",
            "连接子改造",
            8.0,
            "把连接亚甲基改为羰基，可增加极性并引入新的氢键受体，"
            "同时改变相邻环系的相对取向。",
        ),
    ]
    for smarts, label, category, priority, rationale in linker_rules:
        for product in _run_reaction(parent, smarts, limit=4):
            add(product, label, category, priority, rationale, category)

    # ------------------------------------------------------------------
    # 9. Second-generation combinations of complementary changes.
    # ------------------------------------------------------------------
    first_generation = sorted(
        [item for item in raw if item.get("generation") == 1],
        key=lambda item: item["score"],
        reverse=True,
    )
    for index, base_item in enumerate(first_generation[:12]):
        base_mol = Chem.MolFromSmiles(base_item["smiles"])
        if base_mol is None:
            continue
        for product, label, rationale, category in _combination_candidates(
            base_mol, base_item["label"], index, analysis_data
        ):
            add(product, label, "组合改造", 9.0, rationale, category)

    # ------------------------------------------------------------------
    # 10. Category-balanced selection followed by a diversity filter.
    # ------------------------------------------------------------------
    by_category: dict[str, list[dict]] = {}
    for item in sorted(raw, key=lambda entry: entry["score"], reverse=True):
        by_category.setdefault(item["category"], []).append(item)
    categories = sorted(
        by_category,
        key=lambda name: max(entry["score"] for entry in by_category[name]),
        reverse=True,
    )
    limit_per_category = max(4, max_candidates // max(len(categories), 1) + 3)
    pool: list[dict] = []
    for name in categories:
        pool.extend(by_category[name][:limit_per_category])
    pool.sort(key=lambda entry: entry["score"], reverse=True)

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    selected: list[dict] = []
    fingerprints = []
    selected_ids: set[int] = set()
    for item in pool:
        mol = Chem.MolFromSmiles(item["smiles"])
        if mol is None:
            continue
        fp = generator.GetFingerprint(mol)
        if any(DataStructs.TanimotoSimilarity(fp, old) > 0.93 for old in fingerprints):
            continue
        selected.append(item)
        selected_ids.add(id(item))
        fingerprints.append(fp)
        if len(selected) >= max_candidates:
            break
    if len(selected) < max_candidates:
        for item in sorted(raw, key=lambda entry: entry["score"], reverse=True):
            if id(item) in selected_ids:
                continue
            mol = Chem.MolFromSmiles(item["smiles"])
            if mol is None:
                continue
            fp = generator.GetFingerprint(mol)
            if any(DataStructs.TanimotoSimilarity(fp, old) > 0.95 for old in fingerprints):
                continue
            selected.append(item)
            selected_ids.add(id(item))
            fingerprints.append(fp)
            if len(selected) >= max_candidates:
                break
    return selected[:max_candidates]



def _category_from_label(label: str) -> str:
    text = str(label or "")
    if "组合" in text:
        return "组合改造"
    if any(key in text for key in ("苯环→", "芳环→", "五元环", "骨架跃迁", "氮杂")):
        return "骨架跃迁"
    if "亚甲基" in text:
        return "连接子改造"
    if any(key in text for key in ("CF3", "N-甲基", "酰胺→腈")):
        return "代谢稳定化"
    if any(key in text for key in ("羧酸", "酰胺", "酯→", "腈→", "磺酰胺")):
        return "生物电子等排"
    if any(key in text for key in ("氟", "氯", "甲基", "甲氧基", "羟基", "氨基", "氰基")):
        return "取代基行走"
    return "结构修饰"


def rebuild_report(project: Project, optimization: dict, outpath: Path) -> Path:
    """Rebuild the optimisation PDF from stored run data (no re-docking)."""
    parent_run_id = optimization.get("parent_run", "")
    parent_run = project.get_run(parent_run_id)
    if not parent_run:
        raise ValueError("找不到优化母体对接记录")
    parent_results = [
        item for item in parent_run.get("results", [])
        if item.get("status") == "ok"
    ]
    try:
        parent_index = int(optimization.get("parent_index") or 0)
    except (TypeError, ValueError):
        parent_index = 0
    if not 0 <= parent_index < len(parent_results):
        raise ValueError("优化母体结果编号无效")
    parent_result = parent_results[parent_index]
    parent_smiles = _parent_smiles(project, parent_result)
    parent_pose = project.root / parent_result["pose"]
    parent_receptor = _receptor_for_result(project, parent_result)
    if not parent_receptor:
        raise ValueError("找不到优化母体受体")
    parent_analysis = analysis.analyse_interactions(
        (project.root / parent_receptor["pdb"]).read_text(
            encoding="utf-8", errors="replace"
        ),
        parent_pose,
        1,
        receptor_pdbqt=analysis.receptor_pdbqt_text(
            project.root / parent_receptor["pdb"]
        ),
    )
    parent_pose_mol, _ = analysis._load_pose_mol(parent_pose, 1)
    parent_item = {
        **parent_result,
        "result": parent_result,
        "analysis": parent_analysis,
        "mol": Chem.MolFromSmiles(parent_smiles),
        "pose_mol": parent_pose_mol,
        "label": parent_result.get("ligand_name", "母体"),
        "ligand_name": parent_result.get("ligand_name", "母体"),
    }
    opt_run = project.get_run(optimization.get("run_id", ""))
    if not opt_run:
        raise ValueError("找不到优化运行记录")
    candidates = []
    for result in opt_run.get("results", []):
        if result.get("status") != "ok":
            continue
        ligand = project.get_ligand(result.get("ligand_id") or "")
        smiles = ""
        if ligand:
            smiles = ligand.get("smiles") or (ligand.get("meta") or {}).get("smiles") or ""
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if not _valid(mol, parent_item["mol"]):
            # Stored runs may predate the current size/stability guards.
            continue
        pose = project.root / result.get("pose", "")
        receptor_path = (
            project.root / result["receptor_pdb"]
            if result.get("receptor_pdb")
            else project.root / parent_receptor["pdb"]
        )
        try:
            interaction = analysis.analyse_interactions(
                receptor_path.read_text(encoding="utf-8", errors="replace"),
                pose,
                1,
                receptor_pdbqt=analysis.receptor_pdbqt_text(receptor_path),
            )
            pose_mol, _ = analysis._load_pose_mol(pose, 1)
        except Exception:
            interaction = {}
            pose_mol = mol
        full_name = str(result.get("ligand_name") or "")
        label = full_name.split(" ", 1)[-1] if " " in full_name else full_name
        candidates.append(
            {
                "label": label,
                "modification": label,
                "category": _category_from_label(label),
                "rationale": "",
                "targets": [],
                "result": result,
                "analysis": interaction,
                "mol": mol,
                "pose_mol": pose_mol,
                "delta": (
                    float(result["affinity"]) - float(parent_result["affinity"])
                    if result.get("affinity") is not None
                    and parent_result.get("affinity") is not None
                    else None
                ),
            }
        )
    figures.optimization_report_pdf(
        parent_item,
        candidates,
        optimization.get("suggestions") or [],
        Path(outpath),
    )
    return Path(outpath)


def _parent_smiles(project: Project, result: dict) -> str:
    ligand = project.get_ligand(result.get("ligand_id"))
    if not ligand:
        ligand = next(
            (
                item for item in project.data.get("ligands", [])
                if item.get("name") == result.get("ligand_name")
            ),
            None,
        )
    smiles = ""
    if ligand:
        smiles = ligand.get("smiles") or ligand.get("meta", {}).get("smiles") or ""
        if not smiles and ligand.get("sdf"):
            sdf_path = project.root / ligand["sdf"]
            if sdf_path.exists():
                molecules = ligmod.read_ligand_file(sdf_path)
                if molecules:
                    smiles = Chem.MolToSmiles(molecules[0][1])
    if not smiles and result.get("pose"):
        # The ligand record may have been removed from the project. Rebuild the
        # parent from the docked pose so optimization can still proceed.
        pose_path = project.root / result["pose"]
        if pose_path.exists():
            try:
                mol, _ = analysis._load_pose_mol(pose_path, 1)
                smiles = Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)
            except Exception:
                smiles = ""
    if not smiles:
        raise ValueError(
            "母体配体没有 SMILES/键级信息，无法自动生成类似物。"
            "请使用 SDF 或 SMILES 准备的配体后重试。"
        )
    return smiles


def _receptor_for_result(project: Project, result: dict) -> dict | None:
    receptor = project.get_receptor(result.get("receptor_id"))
    if receptor:
        return receptor
    for item in project.data.get("receptors", []):
        if item.get("name") == result.get("receptor_name") and item.get("pdbqt"):
            return item
    return None


def run_optimization(
    project: Project,
    parent_run: dict,
    parent_index: int,
    options: dict,
    job,
) -> dict:
    """Generate analogues, dock them and build a vector PDF improvement report."""
    results = parent_run.get("results", [])
    if parent_index < 0 or parent_index >= len(results):
        raise ValueError("无效的母体结果编号")
    parent_result = copy.deepcopy(results[parent_index])
    if parent_result.get("status") != "ok":
        raise ValueError("母体结果未成功，无法进行改进")
    receptor = _receptor_for_result(project, parent_result)
    if not receptor:
        raise ValueError("找不到母体对应的受体，请重新准备受体")
    receptor_pdb = project.root / receptor["pdb"]
    parent_pose = project.root / parent_result["pose"]
    parent_analysis = analysis.analyse_interactions(
        receptor_pdb.read_text(encoding="utf-8", errors="replace"),
        parent_pose,
        1,
        receptor_pdbqt=analysis.receptor_pdbqt_text(receptor_pdb),
    )
    suggestions = residue_suggestions(parent_analysis)
    parent_smiles = _parent_smiles(project, parent_result)
    max_candidates = max(3, min(int(options.get("max_candidates", 16)), 60))
    job.log(f"[优化] 母体 {parent_result.get('ligand_name')}，SMILES {parent_smiles}")
    job.log(f"[优化] 基于 {len(suggestions)} 个残基接触点生成改进方案")
    candidates = propose_analogs(parent_smiles, parent_analysis, max_candidates)
    if not candidates:
        raise ValueError("未能生成有效的改进候选，请尝试提供更完整的母体结构")
    categories: dict[str, int] = {}
    for item in candidates:
        name = str(item.get("category") or item.get("modification") or "其他")
        categories[name] = categories.get(name, 0) + 1
    scaffold_hops = sum(
        count for name, count in categories.items() if "骨架" in name
    )
    job.log(
        f"[优化] 生成 {len(candidates)} 个候选，覆盖 {len(categories)} 类改造"
        f"（骨架跃迁 {scaffold_hops} 个）"
    )
    for item in candidates:
        job.log(f"  · {item['label']}（{item['modification']}）")

    # Prepare candidate ligands in the project.
    prepared = []
    total = max(1, len(candidates))
    for index, candidate in enumerate(candidates, start=1):
        if job.cancel_event.is_set():
            break
        job.set_progress(0.05 + 0.20 * index / total, f"准备候选 {index}/{total}")
        mol = Chem.MolFromSmiles(candidate["smiles"])
        if mol is None:
            continue
        try:
            mol3d, pdbqt, meta = ligmod.prepare_mol_for_docking(
                mol,
                protonation=str(options.get("protonation", "ph7.4")),
                minimize=True,
                seed=int(options.get("seed", 42)) + index,
            )
        except Exception as exc:
            job.log(f"  ! 候选 {candidate['label']} 准备失败：{exc}")
            continue
        name = f"OPT{index:02d} {candidate['label']}"
        record = project.new_ligand(name)
        lig_dir = project.ligand_dir(record)
        pdbqt_path = lig_dir / "ligand.pdbqt"
        sdf_path = lig_dir / "ligand.sdf"
        pdbqt_path.write_text(pdbqt, encoding="utf-8")
        ligmod.write_sdf(mol3d, sdf_path, name)
        record.update(
            {
                "status": "ready",
                "message": "药物优化候选",
                "format": "smi",
                "smiles": candidate["smiles"],
                "pdbqt": str(pdbqt_path.relative_to(project.root)),
                "sdf": str(sdf_path.relative_to(project.root)),
                "meta": meta,
            }
        )
        candidate = {**candidate, "ligand_id": record["id"], "ligand_name": name}
        prepared.append(candidate)
    project.save()

    if not prepared:
        raise ValueError("所有候选均准备失败，请检查母体结构")

    params = dict(DEFAULT_DOCK_PARAMS)
    params.update(
        {
            "engine": str(options.get("engine", "cpu")).lower(),
            "exhaustiveness": int(options.get("exhaustiveness", 4)),
            "num_modes": int(options.get("num_modes", 3)),
            "energy_range": 3.0,
            "cpu": int(options.get("cpu", 0) or 0),
            "seed": int(options.get("seed", 42)),
            "parallel_jobs": int(options.get("parallel_jobs", 2) or 2),
            "rescore": True,
            "scoring": "vina",
            "gpu_replicas": int(options.get("gpu_replicas", 4) or 4),
        }
    )
    run = project.new_run(params)
    run["kind"] = "optimization"
    run["optimization"] = {
        "parent_run": parent_run["id"],
        "parent_index": parent_index,
        "parent_ligand": parent_result.get("ligand_name"),
        "parent_smiles": parent_smiles,
        "suggestions": suggestions,
        "categories": categories,
        "scaffold_hops": scaffold_hops,
        "generated": len(candidates),
    }
    project.save()
    job.log(f"[优化] 开始对接 {len(prepared)} 个候选，引擎 {params['engine'].upper()}")
    docking.run_docking(
        project,
        run,
        [receptor["id"]],
        [item["ligand_id"] for item in prepared],
        params,
        job,
    )

    by_ligand = {item.get("ligand_id"): item for item in run.get("results", [])}
    index_by_ligand = {
        item.get("ligand_id"): index
        for index, item in enumerate(run.get("results", []))
    }
    out_candidates = []
    for candidate in prepared:
        result = by_ligand.get(candidate["ligand_id"])
        if not result or result.get("status") != "ok":
            continue
        result_copy = copy.deepcopy(result)
        pose = project.root / result_copy["pose"]
        receptor_path = (
            project.root / result_copy["receptor_pdb"]
            if result_copy.get("receptor_pdb")
            else receptor_pdb
        )
        try:
            interaction = analysis.analyse_interactions(
                receptor_path.read_text(encoding="utf-8", errors="replace"),
                pose,
                1,
                receptor_pdbqt=analysis.receptor_pdbqt_text(receptor_path),
            )
            pose_mol, _ = analysis._load_pose_mol(pose, 1)
        except Exception:
            interaction = {}
            pose_mol = Chem.MolFromSmiles(candidate["smiles"])
        item = {
            **candidate,
            "result": result_copy,
            "analysis": interaction,
            "mol": Chem.MolFromSmiles(candidate["smiles"]),
            "pose_mol": pose_mol,
            "index": index_by_ligand.get(candidate["ligand_id"], 0),
            "delta": (
                float(result_copy["affinity"]) - float(parent_result["affinity"])
                if result_copy.get("affinity") is not None
                and parent_result.get("affinity") is not None
                else None
            ),
        }
        out_candidates.append(item)

    parent_pose_mol, _ = analysis._load_pose_mol(parent_pose, 1)
    parent_item = {
        **parent_result,
        "result": parent_result,
        "analysis": parent_analysis,
        "mol": Chem.MolFromSmiles(parent_smiles),
        "pose_mol": parent_pose_mol,
        "label": parent_result.get("ligand_name", "母体"),
        "ligand_name": parent_result.get("ligand_name", "母体"),
    }
    report_dir = project.root / "exports" / f"optimization_{run['id']}"
    report_path = report_dir / "optimization_report.pdf"
    figures.optimization_report_pdf(parent_item, out_candidates, suggestions, report_path)
    report_rel = str(report_path.relative_to(project.root))
    report_en = report_dir / "optimization_report_en.pdf"
    report_en_rel = ""
    try:
        i18n.set_lang("en")
        figures.optimization_report_pdf(parent_item, out_candidates, suggestions, report_en)
        report_en_rel = str(report_en.relative_to(project.root))
    except Exception:
        report_en_rel = ""
    finally:
        i18n.set_lang("zh")

    out_candidates.sort(key=lambda item: item["result"].get("affinity", 999))
    best = out_candidates[0] if out_candidates else None
    best_props = None
    if best is not None:
        try:
            best_props = admet.compact(
                admet.profile_mol(best["mol"], affinity=best["result"].get("affinity"))
            )
        except Exception:
            best_props = None
    summary = {
        "id": run["id"],
        "run_id": run["id"],
        "created": run.get("time"),
        "parent_run": parent_run["id"],
        "parent_index": parent_index,
        "parent_ligand": parent_result.get("ligand_name"),
        "report": report_rel,
        "report_en": report_en_rel,
        "suggestions": suggestions,
        "count": len(out_candidates),
        "categories": categories,
        "scaffold_hops": scaffold_hops,
        "generated": len(candidates),
        "top": [
            {
                "label": item["label"],
                "modification": item["modification"],
                "category": item.get("category", ""),
                "affinity": item["result"].get("affinity"),
                "delta": item.get("delta"),
            }
            for item in out_candidates[:8]
        ],
        "best": (
            {
                "ligand_name": best["result"].get("ligand_name"),
                "label": best["label"],
                "modification": best["modification"],
                "category": best.get("category", ""),
                "affinity": best["result"].get("affinity"),
                "delta": best.get("delta"),
                "index": best.get("index"),
                "counts": (best.get("analysis") or {}).get("counts", {}),
                "props": best_props,
            }
            if best
            else None
        ),
    }
    run["optimization"]["report"] = report_rel
    run["optimization"]["best"] = summary["best"]
    project.data.setdefault("optimizations", [])
    project.data["optimizations"] = [
        item for item in project.data["optimizations"] if item.get("run_id") != run["id"]
    ]
    project.data["optimizations"].insert(0, summary)
    project.save()

    condensed = []
    for item in out_candidates:
        counts = (item.get("analysis") or {}).get("counts", {})
        try:
            props = admet.compact(
                admet.profile_mol(item["mol"], affinity=item["result"].get("affinity"))
            )
        except Exception:
            props = None
        condensed.append(
            {
                "ligand_id": item["ligand_id"],
                "ligand_name": item["result"].get("ligand_name"),
                "label": item["label"],
                "modification": item["modification"],
                "category": item.get("category", ""),
                "generation": item.get("generation", 1),
                "rationale": item["rationale"],
                "targets": item.get("targets", []),
                "affinity": item["result"].get("affinity"),
                "vinardo": item["result"].get("vinardo"),
                "delta": item.get("delta"),
                "counts": counts,
                "index": item.get("index"),
                "pose": item["result"].get("pose"),
                "props": props,
            }
        )
    job.result = {
        "kind": "optimization",
        "run_id": run["id"],
        "parent": {
            "ligand_name": parent_result.get("ligand_name"),
            "affinity": parent_result.get("affinity"),
            "vinardo": parent_result.get("vinardo"),
            "smiles": parent_smiles,
        },
        "report": report_rel,
        "report_en": report_en_rel,
        "suggestions": suggestions,
        "categories": categories,
        "scaffold_hops": scaffold_hops,
        "generated": len(candidates),
        "candidates": condensed,
        "best": summary["best"],
    }
    job.set_progress(1.0, "药物优化完成")
    return job.result
