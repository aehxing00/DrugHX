"""Target-focused de novo design and local compound-database similarity search."""
from __future__ import annotations

import copy
import random
from pathlib import Path

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors

from . import active, admet, analysis, docking, figures, i18n, ligands as ligmod
from . import optimization as optmod
from .config import DEFAULT_DOCK_PARAMS
from .project import Project

# Core rings used for fragment-based design.  Every core is a valid aromatic
# scaffold; substituents are attached to aromatic C-H positions.
CORES = [
    ("苯", "c1ccccc1"),
    ("吡啶", "c1ccncc1"),
    ("嘧啶", "c1cncnc1"),
    ("吡嗪", "c1cnccn1"),
    ("哒嗪", "c1ccnnc1"),
    ("三嗪", "c1ncncn1"),
    ("噻吩", "c1ccsc1"),
    ("呋喃", "c1ccoc1"),
    ("吡咯", "c1cc[nH]c1"),
    ("咪唑", "c1cnc[nH]1"),
    ("吡唑", "c1cc[nH]n1"),
    ("噁唑", "c1occn1"),
    ("异噁唑", "c1oncc1"),
    ("噻唑", "c1scnc1"),
    ("异噻唑", "c1sncc1"),
    ("萘", "c1ccc2ccccc2c1"),
    ("喹啉", "c1ccc2ncccc2c1"),
    ("异喹啉", "c1ccc2ccncc2c1"),
    ("喹唑啉", "c1ccc2ncncc2c1"),
    ("喹喔啉", "c1ccc2nccnc2c1"),
    ("吲哚", "c1ccc2[nH]ccc2c1"),
    ("苯并咪唑", "c1ccc2[nH]cnc2c1"),
    ("吲唑", "c1ccc2[nH]ncc2c1"),
    ("苯并呋喃", "c1ccc2occc2c1"),
    ("苯并噻吩", "c1ccc2sccc2c1"),
    ("苯并噁唑", "c1ccc2ocnc2c1"),
    ("苯并噻唑", "c1ccc2scnc2c1"),
    ("7-氮杂吲哚", "c1cnc2[nH]ccc2c1"),
    ("萘啶", "c1cnc2ncccc2c1"),
]

# First atom of each fragment is the attachment point.
SUBSTITUENTS = [
    ("氟", "F"),
    ("氯", "Cl"),
    ("溴", "Br"),
    ("羟基", "O"),
    ("氨基", "N"),
    ("甲基", "C"),
    ("乙基", "CC"),
    ("三氟甲基", "C(F)(F)F"),
    ("三氟甲氧基", "OC(F)(F)F"),
    ("甲氧基", "OC"),
    ("乙氧基", "OCC"),
    ("氰基", "C#N"),
    ("酰胺", "C(=O)N"),
    ("N-甲基酰胺", "C(=O)N(C)C"),
    ("羧酸", "C(=O)O"),
    ("磺酰胺", "S(=O)(=O)N"),
    ("甲磺酰基", "S(=O)(=O)C"),
    ("乙酰基", "C(=O)C"),
    ("甲氨基", "NC"),
    ("二甲氨基", "N(C)C"),
    ("环丙基", "C1CC1"),
    ("甲氧基乙基", "OCCOC"),
]


def _attach_fragment(core: Chem.Mol, core_index: int, fragment_smiles: str) -> Chem.Mol | None:
    fragment = Chem.MolFromSmiles(fragment_smiles)
    if fragment is None or fragment.GetNumAtoms() == 0:
        return None
    rw = Chem.RWMol(core)
    mapping = {}
    for atom in fragment.GetAtoms():
        mapping[atom.GetIdx()] = rw.AddAtom(atom)
    for bond in fragment.GetBonds():
        rw.AddBond(
            mapping[bond.GetBeginAtomIdx()],
            mapping[bond.GetEndAtomIdx()],
            bond.GetBondType(),
        )
    rw.AddBond(core_index, mapping[0], Chem.BondType.SINGLE)
    product = rw.GetMol()
    try:
        Chem.SanitizeMol(product)
    except Exception:
        return None
    return product


def _aromatic_ch_positions(mol: Chem.Mol) -> list[int]:
    return [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetIsAromatic()
        and atom.GetAtomicNum() == 6
        and atom.GetTotalNumHs() == 1
    ]


def _druglike(mol: Chem.Mol) -> bool:
    if mol is None or mol.GetNumHeavyAtoms() < 8:
        return False
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return False
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    if not (140 <= mw <= 520):
        return False
    if not (-2.0 <= logp <= 5.0):
        return False
    if rdMolDescriptors.CalcNumHBD(mol) > 5:
        return False
    if rdMolDescriptors.CalcNumHBA(mol) > 10:
        return False
    if rdMolDescriptors.CalcNumRotatableBonds(mol) > 8:
        return False
    return True


def _design_score(mol: Chem.Mol) -> float:
    """Heuristic pre-score used before expensive docking."""
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    score = 0.0
    score -= abs(mw - 340) / 180
    score -= abs(logp - 2.4) / 3.0
    score -= abs(tpsa - 75) / 80
    score += 0.25 * min(hbd, 3) + 0.18 * min(hba, 6)
    score -= 0.12 * max(0, rotb - 4)
    return score


def propose_designs(
    max_designs: int = 20,
    seed: int = 42,
    reference_smiles: str = "",
    mode: str = "auto",
) -> list[dict]:
    """Enumerate core/substituent combinations plus reference-based scaffold hops.

    ``mode`` can be ``fragment`` (core enumeration only), ``scaffold``
    (reference-based scaffold hopping only) or ``auto``/``hybrid`` (both).
    """
    mode = str(mode or "auto").lower()
    rng = random.Random(seed)
    seen: set[str] = set()
    candidates: list[dict] = []
    for core_name, core_smiles in CORES:
        core = Chem.MolFromSmiles(core_smiles)
        if core is None:
            continue
        positions = _aromatic_ch_positions(core)
        if not positions:
            continue
        # Single substituents
        for sub_name, sub_smiles in SUBSTITUENTS:
            for position in positions:
                product = _attach_fragment(core, position, sub_smiles)
                if not _druglike(product):
                    continue
                canonical = Chem.MolToSmiles(product, isomericSmiles=True)
                if canonical in seen:
                    continue
                seen.add(canonical)
                candidates.append(
                    {
                        "smiles": canonical,
                        "core": core_name,
                        "substituents": [sub_name],
                        "label": f"{core_name}-{sub_name}",
                        "score": _design_score(product),
                        "mode": "fragment",
                    }
                )
        # Two substituents: sample combinations to keep enumeration bounded.
        if len(positions) >= 2:
            for _ in range(70):
                sub_a = rng.choice(SUBSTITUENTS)
                sub_b = rng.choice(SUBSTITUENTS)
                if sub_a[0] == sub_b[0]:
                    continue
                first, second = rng.sample(positions, 2)
                product = _attach_fragment(core, first, sub_a[1])
                if product is None:
                    continue
                product = _attach_fragment(product, second, sub_b[1])
                if not _druglike(product):
                    continue
                canonical = Chem.MolToSmiles(product, isomericSmiles=True)
                if canonical in seen:
                    continue
                seen.add(canonical)
                candidates.append(
                    {
                        "smiles": canonical,
                        "core": core_name,
                        "substituents": [sub_a[0], sub_b[0]],
                        "label": f"{core_name}-{sub_a[0]}-{sub_b[0]}",
                        "score": _design_score(product) + 0.15,
                        "mode": "fragment",
                    }
                )
        # Three substituents: broader exploration of substitution patterns.
        if len(positions) >= 3:
            for _ in range(45):
                picked = [rng.choice(SUBSTITUENTS) for _ in range(3)]
                if len({item[0] for item in picked}) < 3:
                    continue
                first, second, third = rng.sample(positions, 3)
                product = _attach_fragment(core, first, picked[0][1])
                if product is None:
                    continue
                product = _attach_fragment(product, second, picked[1][1])
                if product is None:
                    continue
                product = _attach_fragment(product, third, picked[2][1])
                if not _druglike(product):
                    continue
                canonical = Chem.MolToSmiles(product, isomericSmiles=True)
                if canonical in seen:
                    continue
                seen.add(canonical)
                candidates.append(
                    {
                        "smiles": canonical,
                        "core": core_name,
                        "substituents": [item[0] for item in picked],
                        "label": f"{core_name}-{picked[0][0]}-{picked[1][0]}-{picked[2][0]}",
                        "score": _design_score(product) + 0.30,
                        "mode": "fragment",
                    }
                )

    # Reference-based scaffold hopping: reuse the analogue rules on a known
    # ligand so the design run also explores ring replacements and linkers.
    reference = str(reference_smiles or "").strip()
    use_scaffold = mode in ("auto", "hybrid", "scaffold")
    if reference and use_scaffold:
        try:
            hops = optmod.propose_analogs(
                reference, {}, max(8, max_designs // 3)
            )
        except Exception:
            hops = []
        for hop in hops:
            mol = Chem.MolFromSmiles(hop.get("smiles") or "")
            if mol is None or not _druglike(mol):
                continue
            canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
            if canonical in seen:
                continue
            seen.add(canonical)
            candidates.append(
                {
                    "smiles": canonical,
                    "core": "骨架跃迁",
                    "substituents": [hop.get("modification") or "骨架跃迁"],
                    "label": f"骨架跃迁-{hop.get('label') or '候选'}",
                    "score": _design_score(mol) + 0.35,
                    "mode": "scaffold_hop",
                }
            )

    if mode == "fragment":
        candidates = [item for item in candidates if item.get("mode") != "scaffold_hop"]
    elif mode == "scaffold":
        candidates = [item for item in candidates if item.get("mode") == "scaffold_hop"]
    candidates.sort(key=lambda item: item["score"], reverse=True)

    # Keep a dedicated quota for scaffold hops so they are not crowded out by
    # higher-scoring but structurally similar fragment analogues.
    scaffold_pool = [
        item for item in candidates if item.get("mode") == "scaffold_hop"
    ]
    fragment_pool = [
        item for item in candidates if item.get("mode") != "scaffold_hop"
    ]
    scaffold_quota = min(len(scaffold_pool), max(3, max_designs // 5))
    ordered = scaffold_pool[:scaffold_quota] + fragment_pool + scaffold_pool[scaffold_quota:]

    # Diversity filter: avoid near-duplicate fingerprints.
    selected = []
    fingerprints = []
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    for item in ordered:
        mol = Chem.MolFromSmiles(item["smiles"])
        if mol is None:
            continue
        fp = generator.GetFingerprint(mol)
        if any(DataStructs.TanimotoSimilarity(fp, old) > 0.91 for old in fingerprints):
            continue
        selected.append(item)
        fingerprints.append(fp)
        if len(selected) >= max_designs:
            break
    for index, item in enumerate(selected, start=1):
        item["name"] = f"DESIGN{index:02d} {item['label']}"
    return selected



# A small reference set used when the user has not imported a local database.
BUILTIN_DB = [
    ("阿司匹林", "CC(=O)Oc1ccccc1C(=O)O"),
    ("咖啡因", "Cn1c(=O)c2c(ncn2C)n(C)c1=O"),
    ("布洛芬", "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O"),
    ("对乙酰氨基酚", "CC(=O)Nc1ccc(O)cc1"),
    ("华法林", "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O"),
    ("伊马替尼", "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"),
    ("吉非替尼", "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCN1CCOCC1"),
    ("厄洛替尼", "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1"),
    ("索拉非尼", "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3cc(C(F)(F)F)ccc3Cl)cc2)ccn1"),
    ("苯甲酸", "O=C(O)c1ccccc1"),
    ("苯酚", "Oc1ccccc1"),
    ("苯胺", "Nc1ccccc1"),
    ("吡啶", "c1ccncc1"),
    ("吲哚", "c1ccc2[nH]ccc2c1"),
    ("喹啉", "c1ccc2ncccc2c1"),
]


def database_path(project: Project) -> Path:
    return project.root / "design_database.smi"


def save_database(project: Project, text: str) -> int:
    path = database_path(project)
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        smiles = parts[0]
        name = " ".join(parts[1:]) if len(parts) > 1 else f"DB{len(rows)+1:04d}"
        if Chem.MolFromSmiles(smiles) is None:
            continue
        rows.append(f"{smiles} {name}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return len(rows)


def load_database(project: Project) -> list[tuple[str, str]]:
    path = database_path(project)
    rows: list[tuple[str, str]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if Chem.MolFromSmiles(parts[0]) is None:
                continue
            rows.append((parts[0], " ".join(parts[1:]) if len(parts) > 1 else f"DB{len(rows)+1:04d}"))
    if not rows:
        rows.extend((smiles, name) for name, smiles in BUILTIN_DB)
        # Fall back to the project ligand collection as well.
        for ligand in project.data.get("ligands", []):
            if str(ligand.get("name", "")).upper().startswith("DESIGN"):
                continue
            smiles = ligand.get("smiles") or ligand.get("meta", {}).get("smiles")
            if smiles:
                rows.append((smiles, ligand.get("name", "ligand")))
    # Deduplicate by canonical SMILES while keeping the first name.
    unique = []
    seen_smiles = set()
    for smiles, name in rows:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        canonical = Chem.MolToSmiles(mol)
        if canonical in seen_smiles:
            continue
        seen_smiles.add(canonical)
        unique.append((smiles, name))
    return unique


def similarity_hits(smiles: str, database: list[tuple[str, str]], limit: int = 3) -> list[dict]:
    query = Chem.MolFromSmiles(smiles)
    if query is None or not database:
        return []
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    query_fp = generator.GetFingerprint(query)
    hits = []
    for db_smiles, name in database:
        mol = Chem.MolFromSmiles(db_smiles)
        if mol is None:
            continue
        fp = generator.GetFingerprint(mol)
        score = float(DataStructs.TanimotoSimilarity(query_fp, fp))
        if score >= 0.999:
            continue
        hits.append({"name": name, "smiles": db_smiles, "tanimoto": round(score, 3)})
    hits.sort(key=lambda item: item["tanimoto"], reverse=True)
    return hits[:limit]


def rebuild_report(project: Project, design: dict, outpath: Path) -> Path:
    """Rebuild the design PDF from stored run data (no re-docking)."""
    run = project.get_run(design.get("run_id", ""))
    if not run:
        raise ValueError("找不到设计运行记录")
    database = [
        item for item in load_database(project)
        if not str(item[1]).startswith(("DESIGN", "OPT"))
    ]
    output = []
    design_meta = run.get("design") or {}
    core_map = design_meta.get("core_map") or {}
    core_counts = design_meta.get("core_counts") or {}
    for result in run.get("results", []):
        if result.get("status") != "ok":
            continue
        ligand = project.get_ligand(result.get("ligand_id") or "")
        smiles = ""
        if ligand:
            smiles = ligand.get("smiles") or (ligand.get("meta") or {}).get("smiles") or ""
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        pose = project.root / result.get("pose", "")
        receptor_path = (
            project.root / result["receptor_pdb"]
            if result.get("receptor_pdb")
            else None
        )
        try:
            interaction = analysis.analyse_interactions(
                receptor_path.read_text(encoding="utf-8", errors="replace"),
                pose,
                1,
                receptor_pdbqt=analysis.receptor_pdbqt_text(receptor_path),
            ) if receptor_path and receptor_path.exists() else {}
            pose_mol, _ = analysis._load_pose_mol(pose, 1)
        except Exception:
            interaction = {}
            pose_mol = mol
        full_name = str(result.get("ligand_name") or "")
        label = full_name.split(" ", 1)[-1] if " " in full_name else full_name
        core_name = (
            core_map.get(full_name)
            or core_map.get(result.get("ligand_id") or "")
            or ""
        )
        if not core_name:
            for candidate in core_counts:
                if candidate and candidate in full_name:
                    core_name = candidate
                    break
        hop = "骨架跃迁" in label or "骨架跃迁" in core_name
        if core_name:
            core_label = core_name
        else:
            core_label = (
                ("scaffold" if hop else "fragment")
                if i18n.get_lang() == "en"
                else ("骨架跃迁" if hop else "片段生长")
            )
        output.append(
            {
                "name": full_name,
                "label": label,
                "core": core_label,
                "substituents": [label],
                "mode": "scaffold_hop" if hop else "fragment",
                "result": result,
                "analysis": interaction,
                "mol": mol,
                "pose_mol": pose_mol,
                "database_hits": similarity_hits(smiles, database, 3) if smiles else [],
            }
        )
    output.sort(key=lambda item: item["result"].get("affinity", 999))
    figures.design_report_pdf(design.get("receptor_name") or "target", output, Path(outpath))
    return Path(outpath)


def run_design(project: Project, receptor_id: str, options: dict, job) -> dict:
    receptor = project.get_receptor(receptor_id)
    if not receptor:
        raise ValueError("找不到受体")
    if not receptor.get("box"):
        raise ValueError("请先设置结合位点盒子")
    max_designs = max(4, min(int(options.get("max_designs", 20)), 60))
    active_mode = bool(options.get("active"))
    pool_size = min(60, max_designs * 2) if active_mode else max_designs
    reference_smiles = str(options.get("reference_smiles") or "").strip()
    mode = str(options.get("mode") or "auto").lower()
    if mode not in ("auto", "fragment", "scaffold", "hybrid"):
        mode = "auto"
    if mode == "scaffold" and not reference_smiles:
        mode = "auto"
        job.log("[设计] 未提供参考配体，骨架跃迁模式自动切换为片段生长 + 骨架跃迁")
    job.log(
        f"[设计] 靶点 {receptor['name']}，生成 {pool_size} 个候选"
        f"（模式：{mode}{'，主动选择' if active_mode else ''}）"
    )
    job.set_progress(0.05, "生成候选分子")
    designs = propose_designs(
        pool_size,
        seed=int(options.get("seed", 42)),
        reference_smiles=reference_smiles,
        mode=mode,
    )
    if not designs:
        raise ValueError("未能生成有效设计分子")
    if active_mode and len(designs) > max_designs:
        selected = active.select_diverse(designs, max_designs)
        job.log(
            f"[设计] 主动选择：{len(designs)} 个候选 → 对接 {len(selected)} 个"
            "（质量 60% + 多样性 40%）"
        )
        designs = selected
    core_counts: dict[str, int] = {}
    for item in designs:
        core = str(item.get("core") or "其他")
        core_counts[core] = core_counts.get(core, 0) + 1
    scaffold_hops = sum(
        count for name, count in core_counts.items() if "骨架" in name
    )
    job.log(
        f"[设计] 生成 {len(designs)} 个类药候选，覆盖 {len(core_counts)} 个核心"
        f"（骨架跃迁 {scaffold_hops} 个）"
    )

    prepared = []
    for index, item in enumerate(designs, start=1):
        if job.cancel_event.is_set():
            break
        job.set_progress(0.08 + 0.17 * index / max(1, len(designs)), f"准备设计分子 {index}/{len(designs)}")
        mol = Chem.MolFromSmiles(item["smiles"])
        if mol is None:
            continue
        try:
            mol3d, pdbqt, meta = ligmod.prepare_mol_for_docking(
                mol,
                protonation="ph7.4",
                minimize=True,
                seed=int(options.get("seed", 42)) + index,
            )
        except Exception as exc:
            job.log(f"  ! {item['label']} 准备失败：{exc}")
            continue
        record = project.new_ligand(item["name"])
        lig_dir = project.ligand_dir(record)
        pdbqt_path = lig_dir / "ligand.pdbqt"
        sdf_path = lig_dir / "ligand.sdf"
        pdbqt_path.write_text(pdbqt, encoding="utf-8")
        ligmod.write_sdf(mol3d, sdf_path, item["name"])
        record.update(
            {
                "status": "ready",
                "message": "分子设计候选",
                "format": "smi",
                "smiles": item["smiles"],
                "pdbqt": str(pdbqt_path.relative_to(project.root)),
                "sdf": str(sdf_path.relative_to(project.root)),
                "meta": meta,
            }
        )
        prepared.append({**item, "ligand_id": record["id"], "ligand_name": item["name"]})
    project.save()
    if not prepared:
        raise ValueError("所有设计分子准备失败")

    params = dict(DEFAULT_DOCK_PARAMS)
    params.update(
        {
            "engine": str(options.get("engine", "cpu")).lower(),
            "exhaustiveness": int(options.get("exhaustiveness", 2)),
            "num_modes": 3,
            "energy_range": 3.0,
            "cpu": int(options.get("cpu", 0) or 0),
            "seed": int(options.get("seed", 42)),
            "parallel_jobs": int(options.get("parallel_jobs", 2) or 2),
            "rescore": True,
            "scoring": "vina",
        }
    )
    run = project.new_run(params)
    run["kind"] = "design"
    run["design"] = {
        "receptor_id": receptor_id,
        "receptor_name": receptor.get("name"),
        "box": receptor.get("box"),
        "candidates": len(prepared),
        "mode": mode,
        "reference_smiles": reference_smiles,
        "core_counts": core_counts,
        "scaffold_hops": scaffold_hops,
        "generated": len(designs),
    }
    project.save()
    job.log(f"[设计] 开始对接 {len(prepared)} 个候选")
    docking.run_docking(
        project, run, [receptor_id], [item["ligand_id"] for item in prepared], params, job
    )

    by_ligand = {item.get("ligand_id"): item for item in run.get("results", [])}
    index_by_ligand = {
        item.get("ligand_id"): index for index, item in enumerate(run.get("results", []))
    }
    database = [
        item for item in load_database(project)
        if not str(item[1]).startswith(("DESIGN", "OPT"))
    ]
    job.log(f"[设计] 数据库相似性检索：{len(database)} 条记录")
    output = []
    for item in prepared:
        result = by_ligand.get(item["ligand_id"])
        if not result or result.get("status") != "ok":
            continue
        result_copy = copy.deepcopy(result)
        pose = project.root / result_copy["pose"]
        receptor_path = (
            project.root / result_copy["receptor_pdb"]
            if result_copy.get("receptor_pdb")
            else project.root / receptor["pdb"]
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
            pose_mol = Chem.MolFromSmiles(item["smiles"])
        output.append(
            {
                **item,
                "result": result_copy,
                "analysis": interaction,
                "mol": Chem.MolFromSmiles(item["smiles"]),
                "pose_mol": pose_mol,
                "index": index_by_ligand.get(item["ligand_id"], 0),
                "database_hits": similarity_hits(item["smiles"], database, 3),
            }
        )
    output.sort(key=lambda item: item["result"].get("affinity", 999))
    run["design"]["core_map"] = {
        item["ligand_name"]: item.get("core", "") for item in output
    }
    project.save()
    report_dir = project.root / "exports" / f"design_{run['id']}"
    report_path = report_dir / "design_report.pdf"
    figures.design_report_pdf(receptor.get("name", "target"), output, report_path)
    report_rel = str(report_path.relative_to(project.root))
    report_en = report_dir / "design_report_en.pdf"
    report_en_rel = ""
    try:
        i18n.set_lang("en")
        figures.design_report_pdf(receptor.get("name", "target"), output, report_en)
        report_en_rel = str(report_en.relative_to(project.root))
    except Exception:
        report_en_rel = ""
    finally:
        i18n.set_lang("zh")
    run["design"]["report"] = report_rel
    run["design"]["report_en"] = report_en_rel
    best_props = None
    if output:
        try:
            best_props = admet.compact(
                admet.profile_mol(
                    output[0]["mol"], affinity=output[0]["result"].get("affinity")
                )
            )
        except Exception:
            best_props = None
    run["design"]["best"] = (
        {
            "label": output[0]["label"],
            "affinity": output[0]["result"].get("affinity"),
            "props": best_props,
        }
        if output
        else None
    )
    project.data.setdefault("designs", [])
    summary = {
        "id": run["id"],
        "run_id": run["id"],
        "created": run.get("time"),
        "receptor_name": receptor.get("name"),
        "report": report_rel,
        "report_en": report_en_rel,
        "best": run["design"]["best"],
        "count": len(output),
        "mode": mode,
        "core_counts": core_counts,
        "scaffold_hops": scaffold_hops,
        "generated": len(designs),
        "top": [
            {
                "label": item["label"],
                "core": item["core"],
                "substituents": item["substituents"],
                "mode": item.get("mode", "fragment"),
                "affinity": item["result"].get("affinity"),
            }
            for item in output[:8]
        ],
    }
    project.data["designs"] = [
        item for item in project.data["designs"] if item.get("run_id") != run["id"]
    ]
    project.data["designs"].insert(0, summary)
    project.save()

    condensed = []
    for item in output:
        try:
            props = admet.compact(
                admet.profile_mol(item["mol"], affinity=item["result"].get("affinity"))
            )
        except Exception:
            props = None
        condensed.append(
            {
                "ligand_name": item["ligand_name"],
                "name": item.get("name", item.get("label")),
                "label": item["label"],
                "core": item["core"],
                "substituents": item["substituents"],
                "mode": item.get("mode", "fragment"),
                "smiles": item["smiles"],
                "affinity": item["result"].get("affinity"),
                "vinardo": item["result"].get("vinardo"),
                "index": item["index"],
                "database_hits": item.get("database_hits", []),
                "props": props,
            }
        )
    job.result = {
        "kind": "design",
        "run_id": run["id"],
        "receptor": receptor.get("name"),
        "report": report_rel,
        "report_en": report_en_rel,
        "mode": mode,
        "core_counts": core_counts,
        "scaffold_hops": scaffold_hops,
        "generated": len(designs),
        "designs": condensed,
        "best": run["design"]["best"],
    }
    job.set_progress(1.0, "分子设计完成")
    return job.result
