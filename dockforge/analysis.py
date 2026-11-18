"""Protein–ligand interaction analysis and pose RMSD."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .config import ION_NAMES
from .pdbio import extract_residue_pdb, normalize_pdbqt, parse_pdb, pose_pdbqt, pdbqt_atom_fields, pdbqt_to_pdb

RDLogger.DisableLog("rdApp.*")

_AROMATIC_RINGS = {
    "PHE": [("CG", "CD1", "CD2", "CE1", "CE2", "CZ")],
    "TYR": [("CG", "CD1", "CD2", "CE1", "CE2", "CZ")],
    "TRP": [
        ("CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
        ("CG", "CD1", "NE1", "CE2", "CD2"),
    ],
    "HIS": [("CG", "ND1", "CD2", "CE1", "NE2")],
}

_METALS = {"ZN", "MG", "CA", "MN", "FE", "FE2", "CU", "CU1", "NI", "CO", "CD", "HG"}


def _is_metal(atom) -> bool:
    return atom.resname.upper() in _METALS or atom.element.upper() in _METALS


def _role(atom) -> dict:
    name = atom.name.upper()
    res = atom.resname.upper()
    donor = False
    acceptor = False
    charge = 0

    if atom.record == "ATOM":
        if name == "N" and res != "PRO":
            donor = True
        if name in ("O", "OXT"):
            acceptor = True
        if res == "ARG" and name in ("NE", "NH1", "NH2"):
            donor = True
            charge = 1
        elif res == "LYS" and name == "NZ":
            donor = True
            charge = 1
        elif res == "ASP" and name in ("OD1", "OD2"):
            acceptor = True
            charge = -1
        elif res == "GLU" and name in ("OE1", "OE2"):
            acceptor = True
            charge = -1
        elif res == "ASN":
            if name == "OD1":
                acceptor = True
            elif name == "ND2":
                donor = True
        elif res == "GLN":
            if name == "OE1":
                acceptor = True
            elif name == "NE2":
                donor = True
        elif res == "HIS" and name in ("ND1", "NE2"):
            donor = True
            acceptor = True
        elif res == "SER" and name == "OG":
            donor = True
            acceptor = True
        elif res == "THR" and name == "OG1":
            donor = True
            acceptor = True
        elif res == "TYR" and name == "OH":
            donor = True
            acceptor = True
        elif res == "TRP" and name == "NE1":
            donor = True
        elif res == "CYS" and name == "SG":
            donor = True
            acceptor = True
        elif res == "MET" and name == "SD":
            acceptor = True
    return {"donor": donor, "acceptor": acceptor, "charge": charge}


def _ligand_flags(mol: Chem.Mol, conf_id: int) -> tuple[np.ndarray, list[dict]]:
    conf = mol.GetConformer(conf_id)
    coords = conf.GetPositions()
    fallback = mol.HasProp("_dockforge_fallback") and mol.GetBoolProp("_dockforge_fallback")
    atoms = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        idx = atom.GetIdx()
        nbrs = [n.GetIdx() for n in atom.GetNeighbors()]
        element = atom.GetSymbol()
        h_neighbors = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 1)
        atype = atom.GetProp("pdbqt_type") if atom.HasProp("pdbqt_type") else ""
        is_amide_n = False
        is_sulfonamide_n = False
        if element == "N":
            for nbr in atom.GetNeighbors():
                if nbr.GetSymbol() not in ("C", "S"):
                    continue
                for bond in nbr.GetBonds():
                    other = bond.GetOtherAtom(nbr)
                    if other.GetSymbol() != "O" or bond.GetBondType() != Chem.BondType.DOUBLE:
                        continue
                    if nbr.GetSymbol() == "C":
                        is_amide_n = True
                    else:
                        is_sulfonamide_n = True
        if fallback:
            donor = h_neighbors > 0
            acceptor = element in ("O", "S") or (element == "N" and atype == "NA")
        else:
            donor = (h_neighbors + atom.GetTotalNumHs()) > 0
            # Amide/sulfonamide N and pyrrole-type aromatic N-H are donors only.
            pyrrole_n = atom.GetIsAromatic() and atom.GetTotalNumHs() > 0
            acceptor = (
                element in ("O", "S")
                or (
                    element == "N"
                    and atom.GetFormalCharge() <= 0
                    and not is_amide_n
                    and not is_sulfonamide_n
                    and not pyrrole_n
                )
            )
        charge = atom.GetFormalCharge()
        if fallback and charge == 0:
            if element in ("O", "S") and atom.GetDegree() == 1 and h_neighbors == 0:
                charge = -1
            elif element == "N" and atom.GetDegree() >= 4:
                charge = 1
            elif element in ("Cl", "Br", "I") and atom.GetDegree() == 0:
                charge = -1
        atoms.append(
            {
                "idx": idx,
                "element": element,
                "coord": coords[idx],
                "charge": charge,
                "aromatic": atom.GetIsAromatic(),
                "donor": donor,
                "acceptor": acceptor,
                "nbrs": nbrs,
            }
        )
    return coords, atoms


def _load_pose_mol(pose_path: Path, mode: int) -> tuple[Chem.Mol, int]:
    """Reconstruct one pose, with a geometry fallback for third-party PDBQT.

    Meeko can rebuild bond orders only when the PDBQT carries SMILES
    metadata. Uploaded PDBQT files often do not, and some use non-numeric
    residue IDs. We normalize the PDBQT, read one model at a time, and fall
    back to a proximity-bonded molecule (no sanitization) if necessary.
    """
    text = normalize_pdbqt(pose_path.read_text(encoding="utf-8", errors="replace"))
    one = pose_pdbqt(text, mode)
    try:
        from meeko import PDBQTMolecule, RDKitMolCreate

        pm = PDBQTMolecule(one, skip_typing=True)
        mols = RDKitMolCreate.from_pdbqt_mol(pm)
        if mols and mols[0] is not None and mols[0].GetNumAtoms():
            mol = mols[0]
            mol.SetBoolProp("_dockforge_fallback", False)
            return mol, 0
    except Exception:
        pass
    return _fallback_pose_mol(one), 0


def _fallback_pose_mol(pdbqt: str) -> Chem.Mol:
    atom_rows = []
    for line in pdbqt.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        charge, atype = pdbqt_atom_fields(line)
        atom_rows.append((atype, charge))

    pdb_block = pdbqt_to_pdb(pdbqt)
    mol = Chem.MolFromPDBBlock(
        pdb_block, removeHs=False, sanitize=False, proximityBonding=True
    )
    if mol is None:
        raise ValueError("无法从该 PDBQT 重建分子（缺少坐标或原子信息）")
    for atom, (atype, charge) in zip(mol.GetAtoms(), atom_rows):
        atom.SetProp("pdbqt_type", atype)
        atom.SetDoubleProp("pdbqt_charge", float(charge))
        atom.SetNoImplicit(True)
    try:
        from rdkit import RDLogger

        # Third-party PDBQT occasionally yields a hypervalent atom; the
        # geometry is still usable for interaction analysis, so skip the
        # valence/property stage rather than aborting or spamming the log.
        flags = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
        with RDLogger.BlockLogs():
            Chem.SanitizeMol(mol, flags)
    except Exception:
        pass
    try:
        Chem.GetSymmSSSR(mol)
    except Exception:
        pass
    mol.SetBoolProp("_dockforge_fallback", True)
    return mol


def _strip_hydrogens(mol: Chem.Mol) -> Chem.Mol:
    """Return a heavy-atom copy, dropping hydrogens RDKit keeps on purpose.

    ``Chem.RemoveHs`` retains hydrogens that define double-bond stereo, carry a
    charge or sit next to a dummy atom.  PDBQT-rebuilt poses regularly contain
    such an N-H on an imine, which breaks the atom-count match used by the
    graph RMSD.  Remove them explicitly after the standard pass.
    """
    heavy = Chem.RemoveHs(mol)
    if all(atom.GetAtomicNum() != 1 for atom in heavy.GetAtoms()):
        return heavy
    editable = Chem.RWMol(heavy)
    for atom in sorted(heavy.GetAtoms(), key=lambda item: item.GetIdx(), reverse=True):
        if atom.GetAtomicNum() == 1:
            editable.RemoveAtom(atom.GetIdx())
    stripped = editable.GetMol()
    try:
        Chem.SanitizeMol(stripped)
    except Exception:
        pass
    return stripped


def receptor_pdbqt_text(pdb_path: str | Path | None) -> str | None:
    """Read the PDBQT that belongs to a receptor PDB, if present."""
    if not pdb_path:
        return None
    path = Path(pdb_path)
    for candidate in (path.with_suffix(".pdbqt"), path.with_name("receptor.pdbqt")):
        try:
            if candidate.is_file():
                return candidate.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
    return None


def _protein_hydrogen_map(pdbqt_text: str) -> dict[tuple[str, int, str], list[np.ndarray]]:
    """Map protein heavy atoms to attached polar hydrogens from receptor PDBQT.

    Meeko writes polar hydrogens into the receptor PDBQT while the cleaned PDB
    stays heavy-atom only.  Hydrogens are assigned to the nearest heavy atom of
    the same residue (≤1.35 Å), so the mapping does not depend on naming
    conventions used by different preparation tools.
    """
    if not pdbqt_text:
        return {}
    heavy: dict[tuple[str, int], list[tuple[str, np.ndarray]]] = {}
    hydrogens: list[tuple[str, int, np.ndarray]] = []
    for line in pdbqt_text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            name = line[12:16].strip()
            chain = line[21:22].strip() or "A"
            resseq = int(line[22:26])
            xyz = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            )
        except (ValueError, IndexError):
            continue
        element = (line[76:78].strip() or name[:1]).upper()
        if element == "H" or name.upper().startswith("H"):
            hydrogens.append((chain, resseq, xyz))
        else:
            heavy.setdefault((chain, resseq), []).append((name, xyz))
    if not hydrogens:
        return {}
    mapping: dict[tuple[str, int, str], list[np.ndarray]] = {}
    for chain, resseq, xyz in hydrogens:
        best_name = None
        best_dist = 1.35
        for name, coord in heavy.get((chain, resseq), ()):
            dist = float(np.linalg.norm(coord - xyz))
            if dist < best_dist:
                best_dist = dist
                best_name = name
        if best_name is not None:
            mapping.setdefault((chain, resseq, best_name), []).append(xyz)
    return mapping


def _attached_hydrogens(
    mol: Chem.Mol, atom_index: int, conf_id: int
) -> list[np.ndarray]:
    """Coordinates of explicit hydrogens bonded to a ligand heavy atom."""
    conf = mol.GetConformer(conf_id)
    return [
        np.array(list(conf.GetAtomPosition(neighbor.GetIdx())))
        for neighbor in mol.GetAtomWithIdx(atom_index).GetNeighbors()
        if neighbor.GetAtomicNum() == 1
    ]


def _best_hbond_angle(
    donor: np.ndarray, hydrogens: list[np.ndarray], acceptor: np.ndarray
) -> float | None:
    """Largest D–H···A angle (degrees) over the available hydrogens."""
    if not hydrogens:
        return None
    return max(
        _angle(donor - hydrogen, acceptor - hydrogen) for hydrogen in hydrogens
    )


def analyse_interactions(
    receptor_pdb: str,
    pose_path: Path,
    mode: int = 1,
    max_hydrophobic_residues: int = 40,
    receptor_pdbqt: str | None = None,
) -> dict:
    mol, conf_id = _load_pose_mol(pose_path, mode)
    fallback = mol.HasProp("_dockforge_fallback") and mol.GetBoolProp("_dockforge_fallback")
    lig_coords, lig_atoms = _ligand_flags(mol, conf_id)
    lig_heavy = np.array([a["coord"] for a in lig_atoms])
    lig_tree = cKDTree(lig_heavy)

    st = parse_pdb(receptor_pdb)
    protein = [a for a in st.atoms if a.element != "H"]
    protein_coords = np.array([[a.x, a.y, a.z] for a in protein])
    if len(protein_coords) == 0:
        raise ValueError("受体结构为空")
    prot_tree = cKDTree(protein_coords)

    hbonds = []
    hydrogen_map = _protein_hydrogen_map(receptor_pdbqt or "")
    for i, atom in enumerate(protein):
        role = _role(atom)
        if not (role["donor"] or role["acceptor"]) or _is_metal(atom):
            continue
        for j in lig_tree.query_ball_point(protein_coords[i], 4.0):
            lig = lig_atoms[j]
            if not lig["element"] in ("N", "O", "S"):
                continue
            distance = float(np.linalg.norm(protein_coords[i] - lig["coord"]))
            # N/O hydrogen bonds are only credible up to 3.5 Å; sulfur is
            # larger and legitimately reaches ~4.0 Å (e.g. Met SD / Cys SG).
            cutoff = 4.0 if "S" in (atom.element, lig["element"]) else 3.5
            if distance > cutoff:
                continue
            angle = None
            if role["donor"] and lig["acceptor"]:
                angle = _best_hbond_angle(
                    protein_coords[i],
                    hydrogen_map.get((atom.chain, atom.resseq, atom.name), []),
                    lig["coord"],
                )
                if angle is not None and angle < 120.0:
                    continue
                hbonds.append(
                    {
                        "residue": atom.resname,
                        "chain": atom.chain,
                        "resseq": atom.resseq,
                        "protein_atom": atom.name,
                        "ligand_atom": _atom_label(lig),
                        "distance": round(distance, 2),
                        "angle": round(angle, 1) if angle is not None else None,
                        "direction": "protein→ligand",
                        "protein_coord": [round(float(v), 3) for v in protein_coords[i]],
                        "ligand_coord": [round(float(v), 3) for v in lig["coord"]],
                    }
                )
            elif role["acceptor"] and lig["donor"]:
                angle = _best_hbond_angle(
                    lig["coord"],
                    _attached_hydrogens(mol, lig["idx"], conf_id),
                    protein_coords[i],
                )
                if angle is not None and angle < 120.0:
                    continue
                hbonds.append(
                    {
                        "residue": atom.resname,
                        "chain": atom.chain,
                        "resseq": atom.resseq,
                        "protein_atom": atom.name,
                        "ligand_atom": _atom_label(lig),
                        "distance": round(distance, 2),
                        "angle": round(angle, 1) if angle is not None else None,
                        "direction": "ligand→protein",
                        "protein_coord": [round(float(v), 3) for v in protein_coords[i]],
                        "ligand_coord": [round(float(v), 3) for v in lig["coord"]],
                    }
                )
    hbonds = _dedupe(hbonds, ("residue", "resseq", "protein_atom", "ligand_atom"))

    salt = []
    charged = [a for a in protein if _role(a)["charge"] != 0]
    for atom in charged:
        role = _role(atom)
        for j in lig_tree.query_ball_point(np.array([atom.x, atom.y, atom.z]), 4.0):
            lig = lig_atoms[j]
            if lig["charge"] == 0 or lig["charge"] * role["charge"] > 0:
                continue
            salt.append(
                {
                    "residue": atom.resname,
                    "chain": atom.chain,
                    "resseq": atom.resseq,
                    "protein_atom": atom.name,
                    "ligand_atom": _atom_label(lig),
                    "distance": round(float(np.linalg.norm(np.array([atom.x, atom.y, atom.z]) - lig["coord"])), 2),
                    "protein_coord": [round(float(atom.x), 3), round(float(atom.y), 3), round(float(atom.z), 3)],
                    "ligand_coord": [round(float(v), 3) for v in lig["coord"]],
                }
            )
    # Report one salt bridge per protein residue (the shortest contact), so a
    # bidentate Arg/Asp pair is not counted twice merely because it exposes two
    # charged atoms.
    salt.sort(key=lambda item: item["distance"])
    salt = _dedupe(salt, ("chain", "resseq", "residue"))

    hydrophobic_map: dict[tuple, dict] = {}
    for i, atom in enumerate(protein):
        if atom.element != "C":
            continue
        for j in lig_tree.query_ball_point(protein_coords[i], 4.2):
            lig = lig_atoms[j]
            if lig["element"] != "C":
                continue
            key = (atom.chain, atom.resseq, atom.resname)
            dist = float(np.linalg.norm(protein_coords[i] - lig["coord"]))
            item = hydrophobic_map.get(key)
            if item is None:
                hydrophobic_map[key] = {
                    "residue": atom.resname,
                    "chain": atom.chain,
                    "resseq": atom.resseq,
                    "contacts": 1,
                    "min_distance": round(dist, 2),
                    "_ligand_atoms": {j},
                }
            else:
                item["_ligand_atoms"].add(j)
                item["min_distance"] = min(item["min_distance"], round(dist, 2))
    # Count distinct ligand atoms per residue, not protein–ligand atom pairs:
    # pair counting would rank large residues (Phe/Trp/Arg) higher merely
    # because they contribute more atoms.
    for item in hydrophobic_map.values():
        item["contacts"] = len(item.pop("_ligand_atoms"))
    hydrophobic = sorted(
        hydrophobic_map.values(), key=lambda x: (-x["contacts"], x["min_distance"])
    )[:max_hydrophobic_residues]

    pi = _pi_stacking(protein, lig_atoms, mol, conf_id, fallback)

    halogens = []
    for lig in lig_atoms:
        if lig["element"] not in ("Cl", "Br", "I"):
            continue
        for j in prot_tree.query_ball_point(lig["coord"], 3.9):
            atom = protein[j]
            role = _role(atom)
            if not role["acceptor"] or atom.element not in ("O", "N", "S"):
                continue
            angle = 90.0
            if lig["nbrs"]:
                c_idx = lig["nbrs"][0]
                carbon = mol.GetConformer(conf_id).GetAtomPosition(c_idx)
                v1 = np.array([carbon.x, carbon.y, carbon.z]) - lig["coord"]
                v2 = protein_coords[j] - lig["coord"]
                angle = _angle(v1, v2)
            if angle < 135:
                continue
            halogens.append(
                {
                    "residue": atom.resname,
                    "chain": atom.chain,
                    "resseq": atom.resseq,
                    "protein_atom": atom.name,
                    "ligand_atom": lig["element"],
                    "distance": round(float(np.linalg.norm(lig["coord"] - protein_coords[j])), 2),
                    "angle": round(angle, 1),
                    "protein_coord": [round(float(v), 3) for v in protein_coords[j]],
                    "ligand_coord": [round(float(v), 3) for v in lig["coord"]],
                }
            )

    metals = []
    for atom in protein:
        if not _is_metal(atom):
            continue
        pos = np.array([atom.x, atom.y, atom.z])
        for j in lig_tree.query_ball_point(pos, 3.1):
            lig = lig_atoms[j]
            if lig["element"] not in ("N", "O", "S"):
                continue
            metals.append(
                {
                    "residue": atom.resname,
                    "chain": atom.chain,
                    "resseq": atom.resseq,
                    "protein_atom": atom.name,
                    "ligand_atom": _atom_label(lig),
                    "distance": round(float(np.linalg.norm(pos - lig["coord"])), 2),
                    "protein_coord": [round(float(v), 3) for v in pos],
                    "ligand_coord": [round(float(v), 3) for v in lig["coord"]],
                }
            )

    summary: dict[tuple, dict] = {}

    def add_summary(item, kind):
        key = (item["chain"], item["resseq"], item["residue"])
        entry = summary.setdefault(
            key,
            {
                "residue": item["residue"],
                "chain": item["chain"],
                "resseq": item["resseq"],
                "types": [],
                "distance": item.get("distance", item.get("min_distance", 99)),
            },
        )
        if kind not in entry["types"]:
            entry["types"].append(kind)
        entry["distance"] = min(entry["distance"], item.get("distance", item.get("min_distance", 99)))

    for item in hbonds:
        add_summary(item, "氢键")
    for item in salt:
        add_summary(item, "盐桥")
    for item in hydrophobic:
        add_summary(item, "疏水")
    for item in pi:
        add_summary(item, "π-堆积")
    for item in halogens:
        add_summary(item, "卤键")
    for item in metals:
        add_summary(item, "金属配位")

    summary_list = sorted(
        summary.values(), key=lambda x: (-len(x["types"]), x["distance"])
    )

    residues = [
        {
            "chain": item["chain"],
            "resseq": item["resseq"],
            "residue": item["residue"],
        }
        for item in summary_list
    ]

    return {
        "available": True,
        "method": "geometry" if fallback else "meeko",
        "mode": mode,
        "hbonds": hbonds,
        "salt_bridges": salt,
        "hydrophobic": hydrophobic,
        "pi_stacking": pi,
        "halogens": halogens,
        "metals": metals,
        "summary": summary_list,
        "residues": residues,
        "counts": {
            "hbonds": len(hbonds),
            "salt_bridges": len(salt),
            "hydrophobic": len(hydrophobic),
            "pi_stacking": len(pi),
            "halogens": len(halogens),
            "metals": len(metals),
            "residues": len(summary_list),
        },
    }


def _atom_label(atom: dict) -> str:
    return f"{atom['element']}{atom['idx']}"


def _angle(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos = float(np.dot(v1, v2) / (n1 * n2))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _dedupe(items: list[dict], keys: tuple) -> list[dict]:
    seen = set()
    out = []
    for item in items:
        key = tuple(item.get(k) for k in keys)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _pi_stacking(protein, lig_atoms, mol: Chem.Mol, conf_id: int, fallback: bool = False) -> list[dict]:
    conf = mol.GetConformer(conf_id)
    ring_info = mol.GetRingInfo()
    lig_rings = []
    for ring in ring_info.AtomRings():
        if len(ring) not in (5, 6):
            continue
        ring_atoms = [mol.GetAtomWithIdx(i) for i in ring]
        if fallback:
            if not all(a.GetSymbol() in ("C", "N", "O", "S") for a in ring_atoms):
                continue
            coords = np.array([list(conf.GetAtomPosition(i)) for i in ring])
            if not _planar_aromatic_ring(coords):
                continue
        else:
            if not all(a.GetIsAromatic() for a in ring_atoms):
                continue
            coords = np.array([list(conf.GetAtomPosition(i)) for i in ring])
        lig_rings.append(_ring_geometry(coords))
    if not lig_rings:
        return []

    # residue -> ring index -> {atom name: coordinate}
    protein_rings: dict[tuple, dict[int, dict[str, np.ndarray]]] = {}
    for atom in protein:
        if atom.resname not in _AROMATIC_RINGS:
            continue
        for ring_idx, names in enumerate(_AROMATIC_RINGS[atom.resname]):
            if atom.name in names:
                key = (atom.chain, atom.resseq, atom.resname)
                protein_rings.setdefault(key, {}).setdefault(ring_idx, {})[
                    atom.name
                ] = np.array([atom.x, atom.y, atom.z])

    results = []
    for (chain, resseq, resname), rings in protein_rings.items():
        for ring_idx, coords_by_name in rings.items():
            names = _AROMATIC_RINGS[resname][ring_idx]
            if not all(n in coords_by_name for n in names):
                continue
            ring = _ring_geometry(np.array([coords_by_name[n] for n in names]))
            for lring in lig_rings:
                delta = lring["center"] - ring["center"]
                dist = float(np.linalg.norm(delta))
                if dist > 6.0:
                    continue
                cos = abs(float(np.dot(ring["normal"], lring["normal"])))
                angle = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
                if angle <= 35:
                    kind = "平行"
                    # Parallel stacking also needs face-to-face contact, not
                    # two parallel rings displaced sideways by many Å.
                    vertical = abs(float(np.dot(delta, ring["normal"])))
                    lateral_vec = delta - float(np.dot(delta, ring["normal"])) * ring["normal"]
                    if vertical > 4.0 or float(np.linalg.norm(lateral_vec)) > 2.5:
                        continue
                elif angle >= 55:
                    kind = "T 型"
                else:
                    continue
                results.append(
                    {
                        "residue": resname,
                        "chain": chain,
                        "resseq": resseq,
                        "distance": round(dist, 2),
                        "angle": round(angle, 1),
                        "type": kind,
                        "protein_center": [round(float(v), 3) for v in ring["center"]],
                        "ligand_center": [round(float(v), 3) for v in lring["center"]],
                    }
                )
    return _dedupe(results, ("residue", "resseq", "distance"))


def _planar_aromatic_ring(coords: np.ndarray) -> bool:
    """Approximate aromatic-ring detection for PDBQT without bond orders."""
    if len(coords) < 5:
        return False
    centered = coords - coords.mean(0)
    try:
        singular = np.linalg.svd(centered, compute_uv=False)
    except np.linalg.LinAlgError:
        return False
    if singular[0] <= 1e-6 or singular[-1] / singular[0] > 0.16:
        return False
    distances = [
        float(np.linalg.norm(coords[i] - coords[(i + 1) % len(coords)]))
        for i in range(len(coords))
    ]
    return float(np.mean(distances)) < 1.47


def _ring_geometry(coords: np.ndarray) -> dict:
    center = coords.mean(0)
    centered = coords - center
    _, _, vt = np.linalg.svd(centered)
    normal = vt[-1]
    return {"center": center, "normal": normal}


def rmsd_to_reference(
    receptor_pdb: str,
    ref: dict,
    pose_path: Path,
    mode: int = 1,
) -> dict:
    """Symmetry-corrected heavy-atom RMSD between a pose and a reference ligand."""
    pdb_block = extract_residue_pdb(
        receptor_pdb, ref["chain"], int(ref["resseq"]), ref.get("icode", ""), ref["name"]
    )
    if not pdb_block:
        raise ValueError("未能从受体中提取参考配体")
    ref_mol = Chem.MolFromPDBBlock(pdb_block, removeHs=False, sanitize=False)
    if ref_mol is None:
        raise ValueError("参考配体无法解析")
    try:
        Chem.SanitizeMol(ref_mol)
    except Exception:
        pass

    pose_mol, conf_id = _load_pose_mol(pose_path, mode)
    fallback = pose_mol.HasProp("_dockforge_fallback") and pose_mol.GetBoolProp("_dockforge_fallback")
    if fallback:
        return _rmsd_by_element(pose_mol, conf_id, ref_mol, mode, ref)

    pose_heavy = _strip_hydrogens(pose_mol)
    ref_heavy = _strip_hydrogens(ref_mol)
    if pose_heavy.GetNumAtoms() != ref_heavy.GetNumAtoms():
        return {
            "available": False,
            "error": f"原子数不一致（对接构象 {pose_heavy.GetNumAtoms()}，参考配体 {ref_heavy.GetNumAtoms()}）",
        }
    try:
        ref_matched = AllChem.AssignBondOrdersFromTemplate(pose_heavy, ref_heavy)
    except Exception:
        ref_matched = ref_heavy
    try:
        value = AllChem.GetBestRMS(pose_heavy, ref_matched, prbId=conf_id, refId=0)
        return {
            "available": True,
            "method": "graph",
            "rmsd": round(float(value), 3),
            "mode": mode,
            "reference": ref,
        }
    except Exception:
        return _rmsd_by_element(pose_mol, conf_id, ref_mol, mode, ref)


def _rmsd_by_element(
    pose_mol: Chem.Mol,
    conf_id: int,
    ref_mol: Chem.Mol,
    mode: int,
    ref: dict,
) -> dict:
    """Geometry fallback: element-constrained Hungarian atom matching."""
    if ref_mol.GetNumConformers() == 0:
        return {"available": False, "error": "参考配体缺少三维坐标"}
    pose_atoms = [a for a in pose_mol.GetAtoms() if a.GetAtomicNum() > 1]
    ref_atoms = [a for a in ref_mol.GetAtoms() if a.GetAtomicNum() > 1]
    if len(pose_atoms) != len(ref_atoms):
        return {
            "available": False,
            "error": f"原子数不一致（对接构象 {len(pose_atoms)}，参考配体 {len(ref_atoms)}）",
        }
    pose_conf = pose_mol.GetConformer(conf_id)
    ref_conf = ref_mol.GetConformer(0)
    pose_coords = np.array([list(pose_conf.GetAtomPosition(a.GetIdx())) for a in pose_atoms])
    ref_coords = np.array([list(ref_conf.GetAtomPosition(a.GetIdx())) for a in ref_atoms])
    pose_elem = [a.GetSymbol() for a in pose_atoms]
    ref_elem = [a.GetSymbol() for a in ref_atoms]

    diff = pose_coords[:, None, :] - ref_coords[None, :, :]
    distances = np.linalg.norm(diff, axis=2)
    cost = np.where(
        np.array(pose_elem)[:, None] == np.array(ref_elem)[None, :],
        distances,
        1e6,
    )
    rows, cols = linear_sum_assignment(cost)
    if np.any(cost[rows, cols] >= 1e5):
        return {"available": False, "error": "构象与参考配体的元素组成不匹配"}
    rmsd = math.sqrt(float(np.mean(distances[rows, cols] ** 2)))
    return {
        "available": True,
        "method": "geometry",
        "rmsd": round(rmsd, 3),
        "mode": mode,
        "reference": ref,
    }
