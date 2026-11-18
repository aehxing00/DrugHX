"""PDB / PDBQT parsing, cleaning and conversion utilities."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import ION_NAMES, STANDARD_RESIDUES, WATER_NAMES


@dataclass
class PDBAtom:
    record: str
    serial: int
    name: str
    altloc: str
    resname: str
    chain: str
    resseq: int
    icode: str
    x: float
    y: float
    z: float
    occupancy: float
    bfactor: float
    element: str
    charge: str
    line: str

    @property
    def key(self) -> tuple:
        return (self.chain, self.resseq, self.icode, self.resname, self.name)

    @property
    def coord(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)


@dataclass
class PDBStructure:
    atoms: list[PDBAtom] = field(default_factory=list)
    title: str = ""
    header: str = ""
    resolution: float | None = None
    hetnam: dict[str, str] = field(default_factory=dict)
    raw: str = ""

    @property
    def chains(self) -> list[str]:
        return sorted({a.chain for a in self.atoms if a.chain})

    @property
    def residues(self) -> int:
        return len({(a.chain, a.resseq, a.icode) for a in self.atoms})


def _to_float(text: str, default: float = 0.0) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def _to_int(text: str, default: int = 0) -> int:
    try:
        return int(text)
    except (TypeError, ValueError):
        return default


def parse_pdb(text: str) -> PDBStructure:
    structure = PDBStructure(raw=text)
    for line in text.splitlines():
        rec = line[:6].strip()
        if rec == "TITLE":
            structure.title += " " + line[10:].strip()
        elif rec == "HEADER":
            structure.header = line[10:50].strip()
        elif rec == "REMARK" and line.startswith("REMARK   2 RESOLUTION."):
            m = re.search(r"RESOLUTION\.\s+([\d.]+)", line)
            if m:
                structure.resolution = _to_float(m.group(1))
        elif rec == "HETNAM":
            resname = line[11:14].strip()
            structure.hetnam[resname] = line[15:].strip()
        elif rec in ("ATOM", "HETATM"):
            element = line[76:78].strip()
            name = line[12:16].strip()
            if not element:
                element = re.sub(r"[^A-Za-z]", "", name)[:1].upper()
            structure.atoms.append(
                PDBAtom(
                    record=rec,
                    serial=_to_int(line[6:11].strip()),
                    name=name,
                    altloc=line[16:17].strip(),
                    resname=line[17:20].strip(),
                    chain=line[21:22].strip(),
                    resseq=_to_int(line[22:26].strip()),
                    icode=line[26:27].strip(),
                    x=_to_float(line[30:38]),
                    y=_to_float(line[38:46]),
                    z=_to_float(line[46:54]),
                    occupancy=_to_float(line[54:60], 1.0),
                    bfactor=_to_float(line[60:66]),
                    element=element.capitalize() if len(element) > 1 else element.upper(),
                    charge=line[78:80].strip(),
                    line=line,
                )
            )
    structure.title = re.sub(r"\s+", " ", structure.title).strip()
    return structure


def pdb_info(text: str) -> dict:
    st = parse_pdb(text)
    return {
        "title": st.title,
        "header": st.header,
        "resolution": st.resolution,
        "chains": st.chains,
        "residues": st.residues,
        "atoms": len(st.atoms),
    }


def hetero_ligands(text: str, min_heavy: int = 4, max_atoms: int = 400) -> list[dict]:
    """List plausible small-molecule HETATM groups (waters/ions excluded)."""
    st = parse_pdb(text)
    groups: dict[tuple, list[PDBAtom]] = {}
    for atom in st.atoms:
        if atom.record != "HETATM":
            continue
        if atom.resname in WATER_NAMES or atom.resname in ION_NAMES:
            continue
        key = (atom.chain, atom.resseq, atom.icode, atom.resname)
        groups.setdefault(key, []).append(atom)

    result = []
    for (chain, resseq, icode, resname), atoms in groups.items():
        heavy = [a for a in atoms if a.element != "H"]
        if len(heavy) < min_heavy or len(atoms) > max_atoms:
            continue
        coords = np.array([[a.x, a.y, a.z] for a in atoms])
        size = (coords.max(0) - coords.min(0)).round(2).tolist()
        result.append(
            {
                "key": f"{chain}:{resseq}{icode}:{resname}",
                "name": resname,
                "chain": chain,
                "resseq": resseq,
                "icode": icode,
                "atoms": len(atoms),
                "heavy": len(heavy),
                "center": coords.mean(0).round(3).tolist(),
                "size": size,
                "label": st.hetnam.get(resname, ""),
            }
        )
    result.sort(key=lambda x: (-x["heavy"], x["key"]))
    return result


def _normalized_pdb_line(atom: PDBAtom) -> str:
    """Keep PDB columns intact, clear altloc and populate the element field."""
    line = atom.line
    if len(line) < 78:
        line = line.ljust(78)
    # Altloc selection has already happened, so write the chosen conformer as
    # the blank/default one.  Meeko refuses structures that still carry an
    # altloc character in column 17.
    if len(line) >= 17:
        line = line[:16] + " " + line[17:]
    element = (atom.element or "").rjust(2)
    return line[:76] + element


def clean_pdb(
    text: str,
    chains: list[str] | None = None,
    remove_water: bool = True,
    remove_hetero: bool = True,
    keep_ions: bool = False,
    altloc: str = "A",
) -> dict:
    """Return cleaned PDB text plus statistics.

    Altloc handling keeps blank or the requested altloc, then de-duplicates
    atoms preferring the blank conformer / highest occupancy.
    """
    st = parse_pdb(text)
    chain_set = {c for c in (chains or []) if c}
    chosen: dict[tuple, PDBAtom] = {}

    for atom in st.atoms:
        if chain_set and atom.chain not in chain_set:
            continue
        if atom.record == "HETATM":
            if atom.resname in WATER_NAMES:
                if remove_water:
                    continue
            elif atom.resname in ION_NAMES:
                if not keep_ions:
                    continue
            elif remove_hetero:
                continue
        if atom.altloc not in ("", altloc):
            continue
        prev = chosen.get(atom.key)
        if prev is None:
            chosen[atom.key] = atom
            continue
        # Prefer blank altloc, then higher occupancy.
        prev_score = (1 if prev.altloc == "" else 0, prev.occupancy)
        new_score = (1 if atom.altloc == "" else 0, atom.occupancy)
        if new_score > prev_score:
            chosen[atom.key] = atom

    atoms = sorted(chosen.values(), key=lambda a: (a.chain, a.resseq, a.icode, a.serial))
    lines = []
    last_chain = None
    for atom in atoms:
        if last_chain is not None and atom.chain != last_chain:
            lines.append("TER")
        lines.append(_normalized_pdb_line(atom))
        last_chain = atom.chain
    lines.append("TER")
    lines.append("END")
    return {
        "pdb": "\n".join(lines) + "\n",
        "atoms": len(atoms),
        "residues": len({(a.chain, a.resseq, a.icode) for a in atoms}),
        "chains": sorted({a.chain for a in atoms if a.chain}),
        "removed_water": remove_water,
        "kept_ions": keep_ions and not remove_hetero,
    }


def extract_residue_pdb(text: str, chain: str, resseq: int, icode: str, resname: str) -> str:
    st = parse_pdb(text)
    lines = []
    for atom in st.atoms:
        if (
            atom.chain == chain
            and atom.resseq == resseq
            and atom.icode == icode
            and atom.resname == resname
        ):
            lines.append(_normalized_pdb_line(atom))
    if not lines:
        return ""
    lines.append("END")
    return "\n".join(lines) + "\n"


def extract_ions(text: str) -> list[PDBAtom]:
    st = parse_pdb(text)
    return [
        a
        for a in st.atoms
        if a.record == "HETATM" and a.resname in ION_NAMES and a.element != "H"
    ]


# --------------------------------------------------------------------------
# PDBQT
# --------------------------------------------------------------------------

_PDBQT_TYPE_TO_ELEMENT = {
    "HD": "H", "HS": "H", "H": "H",
    "OA": "O", "OS": "O", "O": "O",
    "NA": "N", "NS": "N", "N": "N",
    "SA": "S", "S": "S",
    "A": "C", "C": "C",
    "F": "F", "Cl": "Cl", "CL": "Cl", "Br": "Br", "BR": "Br",
    "I": "I", "P": "P", "B": "B", "Si": "Si", "SE": "Se",
}


def pdbqt_atom_fields(line: str) -> tuple[float, str]:
    """Return (charge, autodock_type) from a PDBQT atom line."""
    charge = _to_float(line[70:76].strip(), 0.0)
    atype = line[77:79].strip()
    if not atype:
        parts = line.split()
        atype = parts[-1] if parts else "C"
        if len(parts) >= 2 and _looks_float(parts[-2]):
            charge = _to_float(parts[-2])
    return charge, atype


def _looks_float(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def normalize_pdbqt(
    pdbqt: str,
    resseq: int = 1,
    chain: str = "L",
    resname: str | None = None,
) -> str:
    """Normalize atom serials, chain and residue numbers in a PDBQT string.

    Some third-party ligand PDBQT files use non-numeric residue identifiers
    (e.g. ``C000``), which Vina accepts but Meeko's parser rejects with
    ``invalid literal for int()``. Rewriting these columns fixes analysis,
    RMSD and pose reconstruction without touching coordinates or torsions.
    """
    lines = []
    serial = 0
    for line in pdbqt.splitlines():
        rec = line[:6].strip()
        if rec in ("ATOM", "HETATM"):
            serial += 1
            padded = line.ljust(80)
            name_block = padded[11:21]
            if resname:
                name_block = name_block[:6] + f"{resname[:3]:>3s}" + name_block[9:]
            lines.append(
                padded[:6]
                + f"{serial:5d}"
                + name_block
                + (chain or "L")[:1]
                + f"{resseq:4d}"
                + " "
                + padded[27:]
            )
        else:
            lines.append(line)
    return "\n".join(lines) + "\n"


def pdbqt_to_pdb(pdbqt: str) -> str:
    lines = []
    for line in pdbqt.splitlines():
        rec = line[:6].strip()
        if rec in ("ATOM", "HETATM"):
            base = line[:66]
            if len(base) < 66:
                base = base.ljust(66)
            _, atype = pdbqt_atom_fields(line)
            element = _PDBQT_TYPE_TO_ELEMENT.get(atype, atype[:1])
            lines.append(base + "    " + element.rjust(2))
        elif rec in ("TER", "END"):
            lines.append(rec)
        elif rec in ("MODEL", "ENDMDL"):
            lines.append(rec)
    return "\n".join(lines) + "\n"


def split_pdbqt_models(text: str) -> list[dict]:
    """Split a multi-model Vina output into pose dicts."""
    models: list[dict] = []
    current: list[str] = []
    remark = ""
    for line in text.splitlines():
        if line.startswith("MODEL"):
            current = []
            remark = ""
            continue
        if line.startswith("ENDMDL"):
            if current:
                models.append({"remark": remark, "lines": current})
            current = []
            continue
        if line.startswith("REMARK VINA RESULT:"):
            remark = line
        if line.startswith(("ATOM", "HETATM", "ROOT", "ENDROOT", "BRANCH", "ENDBRANCH",
                            "TORSDOF", "REMARK")):
            current.append(line)
    if current and not models:
        models.append({"remark": remark, "lines": current})
    return models


def pose_pdbqt(text: str, mode: int = 1) -> str:
    models = split_pdbqt_models(text)
    if not models:
        return text
    idx = max(0, min(mode - 1, len(models) - 1))
    return "\n".join(models[idx]["lines"]) + "\n"


def pdbqt_stats(text: str) -> dict:
    atoms = [l for l in text.splitlines() if l.startswith(("ATOM", "HETATM"))]
    types = {}
    for line in atoms:
        _, atype = pdbqt_atom_fields(line)
        types[atype] = types.get(atype, 0) + 1
    return {"atoms": len(atoms), "types": types}


def format_ion_pdbqt(atom: PDBAtom) -> str:
    from .config import ION_CHARGES

    charge = ION_CHARGES.get(atom.resname, 0.0)
    atype = atom.element or atom.resname[:2].capitalize()
    name = atom.name[:4]
    return (
        f"HETATM{atom.serial:5d} {name:<4s}{atom.resname:>4s} {atom.chain:1s}"
        f"{atom.resseq:4d}{atom.icode:1s}   "
        f"{atom.x:8.3f}{atom.y:8.3f}{atom.z:8.3f}"
        f"{1.0:6.2f}{0.0:6.2f}    {charge:+6.3f} {atype:<2s}"
    )


def complex_pdb(receptor_pdb: str, ligand_pdb: str, name: str = "DrugHX") -> str:
    receptor_lines = [
        line for line in receptor_pdb.splitlines() if line.strip() not in ("END", "ENDMDL")
    ]
    ligand_lines = [
        line for line in ligand_pdb.splitlines() if line.strip() not in ("END", "ENDMDL")
    ]
    lines = [f"REMARK   DrugHX complex: {name}", *receptor_lines]
    lines.append("TER")
    lines.extend(ligand_lines)
    lines.append("END")
    return "\n".join(lines) + "\n"
