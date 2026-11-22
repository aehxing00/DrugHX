"""Ligand input, standardisation, 3D embedding and PDBQT preparation."""
from __future__ import annotations

import csv
import io
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize as mstd

RDLogger.DisableLog("rdApp.*")

_ACID_PATTERNS = [
    ("[CX3](=O)[OX2H1]", 1),          # carboxylic acid → carboxylate
    ("[SX4](=O)(=O)[OX2H1]", 1),      # sulfonic acid → sulfonate
    ("[PX4](=O)([OX2H1])[OX2H1]", 2),  # phosphate: pKa1/2 both below 7.4
    ("[nH]1nnnc1", 1),                # tetrazole → tetrazolate
]

# Strong bases that are protonated at pH ~7.4. Anilines, amides,
# sulfonamides and aromatic nitrogens are deliberately excluded.
_BASE_PATTERNS = [
    "[NX2;H1]=[CX3]-[NX3]",          # amidine / guanidine imine nitrogen
    "[NX3;H2,H1,H0;!$(N-[!#6]);!$(N-a);!$(N-C=O);!$(N-S);!$(N-C=[O,N,S]);!$([N+])]",
]


def mol_from_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"无法解析 SMILES: {smiles[:80]}")
    return mol


def _looks_like_smiles(token: str) -> bool:
    if not token or len(token) > 500:
        return False
    RDLogger.DisableLog("rdApp.*")
    try:
        return Chem.MolFromSmiles(token) is not None
    except Exception:
        return False


def read_smiles_lines(text: str) -> list[tuple[str, str]]:
    """Parse 'SMILES name' or 'name SMILES' lines, auto-detecting the columns.

    Everything after the first whitespace is kept as the name, so names with
    spaces (for example ``Salvianolic acid G``) survive intact.  A name-first
    line is still accepted when the last whitespace-separated token is a valid
    SMILES.
    """
    out = []
    for i, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        if ("," in line or ";" in line) and "\t" not in line:
            parts = [
                item.strip()
                for item in line.replace(";", ",").split(",")
                if item.strip()
            ]
            if len(parts) >= 2 and _looks_like_smiles(parts[0]):
                out.append((", ".join(parts[1:]) or f"ligand_{i}", parts[0]))
                continue
            if len(parts) >= 2 and _looks_like_smiles(parts[1]):
                out.append((parts[0] or f"ligand_{i}", parts[1]))
                continue
        if "\t" in line:
            head, sep, tail = line.partition("\t")
        elif " " in line:
            head, tail = line.split(None, 1)
            sep = " "
        else:
            head, tail, sep = line, "", ""
        head = head.strip()
        tail = tail.strip()
        if head and _looks_like_smiles(head):
            out.append((tail or f"ligand_{i}", head))
            continue
        if tail:
            matched = False
            for separator_char in ("\t", " "):
                name_part, last_sep, last = line.rpartition(separator_char)
                if last_sep and _looks_like_smiles(last.strip()):
                    out.append(
                        (name_part.strip() or f"ligand_{i}", last.strip())
                    )
                    matched = True
                    break
            if matched:
                continue
        raise ValueError(f"第 {i} 行无法识别 SMILES：{line[:80]}")
    return out


def read_ligand_file(path: Path) -> list[tuple[str, Chem.Mol]]:
    """Read molecules from disk using byte buffers.

    RDKit's file-based readers do not reliably accept non-ASCII Windows
    paths, so everything is read as bytes and parsed from blocks instead.
    """
    suffix = path.suffix.lower()
    data = path.read_bytes()
    text = data.decode("utf-8", errors="replace")
    mols: list[tuple[str, Chem.Mol]] = []

    if suffix in (".sdf", ".sd"):
        supplier = Chem.ForwardSDMolSupplier(
            io.BytesIO(data), removeHs=False, sanitize=True
        )
        for i, mol in enumerate(supplier, start=1):
            if mol is None:
                continue
            name = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else ""
            mols.append((name or f"{path.stem}_{i}", mol))
    elif suffix in (".mol", ".mdl"):
        mol = Chem.MolFromMolBlock(text, removeHs=False, sanitize=True)
        if mol is None:
            mol = Chem.MolFromMolBlock(text, removeHs=False, sanitize=False)
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    mol = None
        if mol is not None:
            mols.append((path.stem, mol))
    elif suffix == ".mol2":
        mol = Chem.MolFromMol2Block(text, removeHs=False, sanitize=True)
        if mol is not None:
            mols.append((path.stem, mol))
    elif suffix in (".pdb", ".ent"):
        mol = Chem.MolFromPDBBlock(
            text, removeHs=False, sanitize=False, proximityBonding=True
        )
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                mol = None
        if mol is not None:
            mols.append((path.stem, mol))
    elif suffix in (".smi", ".txt"):
        for name, smi in read_smiles_lines(text):
            mols.append((name, mol_from_smiles(smi)))
    elif suffix in (".csv", ".tsv"):
        delimiter = "," if suffix == ".csv" else "\t"
        rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
        if not rows:
            raise ValueError(f"文件为空: {path.name}")
        header = [cell.strip().lower() for cell in rows[0]]
        smiles_idx = next(
            (i for i, cell in enumerate(header)
             if cell in {"smiles", "canonical_smiles", "structure", "smi"}),
            None,
        )
        name_idx = next(
            (i for i, cell in enumerate(header)
             if cell in {"name", "id", "compound", "compound_id", "title"}),
            None,
        )
        start = 0
        if smiles_idx is None:
            smiles_idx = 0
            name_idx = 1 if len(header) > 1 else None
        else:
            start = 1
        for row_index, row in enumerate(rows[start:], start=start + 1):
            if not row or smiles_idx >= len(row):
                continue
            smiles = row[smiles_idx].strip()
            if not smiles:
                continue
            name = (
                row[name_idx].strip()
                if name_idx is not None and name_idx < len(row) and row[name_idx].strip()
                else f"{path.stem}_{row_index}"
            )
            mols.append((name, mol_from_smiles(smiles)))
    else:
        raise ValueError(f"不支持的配体格式: {suffix}")

    if not mols:
        raise ValueError(f"未能从文件中读取任何分子: {path.name}")
    return mols


def _deprotonate_acids(mol: Chem.Mol) -> Chem.Mol:
    """Remove the acidic proton of common groups at pH ~7.4."""
    for smarts, max_remove in _ACID_PATTERNS:
        query = Chem.MolFromSmarts(smarts)
        if query is None or not mol.HasSubstructMatch(query):
            continue
        removed = 0
        for match in mol.GetSubstructMatches(query):
            if removed >= max_remove:
                break
            acidic = None
            for idx in match:
                atom = mol.GetAtomWithIdx(idx)
                # RDKit reports aromatic nitrogen as "N"; the tetrazole
                # pattern only matches ring nitrogen, so this is safe.
                if atom.GetSymbol() in ("O", "S", "N") and atom.GetTotalNumHs() > 0:
                    acidic = atom
                    break
            if acidic is None:
                continue
            acidic.SetFormalCharge(acidic.GetFormalCharge() - 1)
            acidic.SetNumExplicitHs(0)
            acidic.SetNoImplicit(True)
            removed += 1
        try:
            Chem.SanitizeMol(mol)
        except Exception as exc:
            raise ValueError(f"酸性质子处理失败: {exc}") from exc
    return mol


def _protonate_bases(mol: Chem.Mol) -> Chem.Mol:
    """Protonate the strongly basic nitrogen that dominates at pH ~7.4.

    Treating every aliphatic amine as protonated overcharges diamines and
    piperazines (e.g. piperazine has pKa1 ≈ 9.8 but pKa2 ≈ 5.6).  Basic
    nitrogens separated by at most three bonds are therefore treated as one
    coupled acid–base cluster and only the most basic centre is protonated.
    Well-separated amines remain independently protonated, which matches the
    behaviour of polyamines such as spermine.
    """
    strong: set[int] = set()
    candidates: set[int] = set()
    for position, smarts in enumerate(_BASE_PATTERNS):
        query = Chem.MolFromSmarts(smarts)
        if query is None or not mol.HasSubstructMatch(query):
            continue
        for match in mol.GetSubstructMatches(query):
            idx = match[0]
            if mol.GetAtomWithIdx(idx).GetFormalCharge() != 0:
                continue
            candidates.add(idx)
            if position == 0:  # amidine / guanidine
                strong.add(idx)
    if not candidates:
        return mol
    distance = Chem.GetDistanceMatrix(mol)
    pending = set(candidates)
    keep: list[int] = []
    while pending:
        seed = pending.pop()
        cluster = {seed}
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            close = {
                other
                for other in pending
                if distance[current][other] <= 3.0
            }
            pending -= close
            cluster |= close
            frontier.extend(close)
        keep.append(
            max(
                cluster,
                key=lambda idx: (
                    1 if idx in strong else 0,
                    mol.GetAtomWithIdx(idx).GetDegree(),
                    -idx,
                ),
            )
        )
    for idx in keep:
        mol.GetAtomWithIdx(idx).SetFormalCharge(1)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    return mol


def standardize_mol(mol: Chem.Mol, protonation: str = "ph7.4") -> Chem.Mol:
    mol = Chem.RemoveHs(mol)
    try:
        mol = mstd.LargestFragmentChooser().choose(mol)
    except Exception:
        pass
    try:
        mol = mstd.Cleanup(mol)
    except Exception:
        pass
    try:
        mol = mstd.Normalize(mol)
    except Exception:
        pass

    if protonation == "ph7.4":
        mol = _deprotonate_acids(mol)
        mol = _protonate_bases(mol)
    elif protonation == "neutral":
        try:
            mol = mstd.Uncharger().uncharge(mol)
        except Exception:
            pass

    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:
        raise ValueError(f"分子结构标准化失败: {exc}") from exc
    return Chem.AddHs(mol, addCoords=True)


def ensure_3d(mol: Chem.Mol, minimize: bool = True, seed: int = 42) -> Chem.Mol:
    conf = mol.GetConformer() if mol.GetNumConformers() else None
    need_embed = conf is None or not conf.Is3D()
    if need_embed:
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed)
        params.useSmallRingTorsions = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            params.useRandomCoords = True
            params.randomSeed = int(seed) + 1
            if AllChem.EmbedMolecule(mol, params) != 0:
                raise ValueError("3D 构象生成失败")
    if minimize:
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol, maxIters=800)
            else:
                AllChem.UFFOptimizeMolecule(mol, maxIters=800)
        except Exception:
            pass
    return mol


def ligand_metadata(mol: Chem.Mol) -> dict:
    heavy = Chem.RemoveHs(mol)
    try:
        formula = rdMolDescriptors.CalcMolFormula(heavy)
    except Exception:
        formula = ""
    return {
        "mw": round(float(Descriptors.MolWt(heavy)), 2),
        "logp": round(float(Descriptors.MolLogP(heavy)), 2),
        "tpsa": round(float(Descriptors.TPSA(heavy)), 1),
        "hbd": int(rdMolDescriptors.CalcNumHBD(heavy)),
        "hba": int(rdMolDescriptors.CalcNumHBA(heavy)),
        "rotb": int(rdMolDescriptors.CalcNumRotatableBonds(heavy)),
        "heavy": int(heavy.GetNumHeavyAtoms()),
        "formula": formula,
        "charge": int(Chem.GetFormalCharge(heavy)),
        "smiles": Chem.MolToSmiles(heavy, isomericSmiles=True),
        "name": mol.GetProp("_Name").strip() if mol.HasProp("_Name") else "",
    }


def prepare_mol_for_docking(
    mol: Chem.Mol,
    protonation: str = "ph7.4",
    minimize: bool = True,
    seed: int = 42,
) -> tuple[Chem.Mol, str, dict]:
    """Return (3D mol with Hs, PDBQT string, metadata)."""
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    mol = standardize_mol(mol, protonation=protonation)
    mol = ensure_3d(mol, minimize=minimize, seed=seed)

    # Keep macrocycles rigid instead of adding AutoDock "G0"/"CG0" glue atoms.
    # Vina-GPU 2.0 and some Vina builds reject the three-character glue atom
    # types, which otherwise causes parse errors or native memory faults.
    prep = MoleculePreparation(rigid_macrocycles=True)
    try:
        setups = prep.prepare(mol)
    except Exception as exc:
        raise ValueError(f"Meeko 配体准备失败: {exc}") from exc
    if not setups:
        raise ValueError("Meeko 未生成可用的配体构象")
    pdbqt, ok, err = PDBQTWriterLegacy.write_string(setups[0])
    if not ok or not pdbqt:
        raise ValueError(f"PDBQT 写入失败: {err}")
    return mol, pdbqt, ligand_metadata(mol)


def write_sdf(mol: Chem.Mol, path: Path, name: str = "ligand") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mol = Chem.Mol(mol)
    mol.SetProp("_Name", name)
    block = Chem.MolToMolBlock(mol, kekulize=True)
    path.write_text(block + "$$$$\n", encoding="utf-8")
