"""PoseBusters physical-validity checks for docked poses.

Vina's scoring function does not penalise physically impossible geometry
(internal clashes, distorted bond lengths/angles, strained rings) or a pose
that never touches the protein.  PoseBusters complements docking scores with
the same rule set used in the PoseBusters benchmark (Buttenschoen et al.,
*Chem. Sci.* 2024, DOI 10.1039/D3SC04185A), so a report can state whether a
binding mode is physically plausible rather than only "well scored".

The wrapper is intentionally thin: it only translates the PoseBusters output
table into bilingual pass/fail records and never raises into the API layer.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

# PoseBusters column -> (中文标签, English label, group)
CHECKS: dict[str, tuple[str, str, str]] = {
    "sanitization": ("价键可标准化", "Sanitization", "structure"),
    "inchi_convertible": ("可转换为 InChI", "InChI convertible", "structure"),
    "all_atoms_connected": ("全部原子连通", "All atoms connected", "structure"),
    "no_radicals": ("无自由基", "No radicals", "structure"),
    "bond_lengths": ("键长合理", "Bond lengths", "structure"),
    "bond_angles": ("键角合理", "Bond angles", "structure"),
    "internal_steric_clash": ("无内部原子冲突", "No internal steric clash", "structure"),
    "aromatic_ring_flatness": ("芳环共平面", "Aromatic ring flatness", "structure"),
    "non-aromatic_ring_non-flatness": (
        "非芳环未过度平面化",
        "Non-aromatic ring non-flatness",
        "structure",
    ),
    "double_bond_flatness": ("双键共平面", "Double bond flatness", "structure"),
    "internal_energy": ("内部应变能合理", "Internal energy", "structure"),
    "protein-ligand_maximum_distance": (
        "配体位于蛋白附近",
        "Ligand within protein",
        "complex",
    ),
    "minimum_distance_to_protein": (
        "与蛋白原子无过近接触",
        "Minimum distance to protein",
        "complex",
    ),
    "minimum_distance_to_organic_cofactors": (
        "与有机辅因子无过近接触",
        "Minimum distance to organic cofactors",
        "complex",
    ),
    "minimum_distance_to_inorganic_cofactors": (
        "与无机离子无过近接触",
        "Minimum distance to inorganic cofactors",
        "complex",
    ),
    "minimum_distance_to_waters": (
        "与水分子无过近接触",
        "Minimum distance to waters",
        "complex",
    ),
    "volume_overlap_with_protein": (
        "与蛋白无体积重叠",
        "Volume overlap with protein",
        "complex",
    ),
    "volume_overlap_with_organic_cofactors": (
        "与有机辅因子无体积重叠",
        "Volume overlap with organic cofactors",
        "complex",
    ),
    "volume_overlap_with_inorganic_cofactors": (
        "与无机离子无体积重叠",
        "Volume overlap with inorganic cofactors",
        "complex",
    ),
    "volume_overlap_with_waters": (
        "与水分子无体积重叠",
        "Volume overlap with waters",
        "complex",
    ),
}

GROUP_LABELS = {
    "structure": ("分子内几何", "Intramolecular"),
    "complex": ("蛋白–配体界面", "Protein–ligand interface"),
}

# PoseBusters reports two loading flags that are preconditions, not checks.
_PRECONDITIONS = ("mol_pred_loaded", "mol_cond_loaded")


def available() -> bool:
    """Whether the optional ``posebusters`` package can be imported."""
    return importlib.util.find_spec("posebusters") is not None


def module_version() -> str:
    try:
        import posebusters

        return getattr(posebusters, "__version__", "?")
    except Exception:
        return "未安装"


def _buster():
    from posebusters import PoseBusters

    # max_workers=0 keeps PoseBusters in-process.  The default (0 means "no
    # parallelisation") avoids Windows ``spawn`` child processes, which are
    # illegal inside the uvicorn worker thread and would re-import the server.
    return PoseBusters(config="dock", max_workers=0)


def _record(key: str, passed: bool) -> dict:
    label_zh, label_en, group = CHECKS.get(key, (key, key, "structure"))
    return {
        "key": key,
        "label_zh": label_zh,
        "label_en": label_en,
        "group": group,
        "passed": bool(passed),
    }


def _summarise(checks: list[dict]) -> dict:
    failed = [item for item in checks if not item["passed"]]
    return {
        "checks": checks,
        "passed": len(checks) - len(failed),
        "total": len(checks),
        "all_passed": not failed,
        "failed": [item["key"] for item in failed],
        "failed_labels": [item["label_zh"] for item in failed],
    }


def run(pdb_path: Path, mol) -> dict:
    """Check one docked pose against its receptor.

    ``mol`` must carry 3D coordinates and bond orders (the caller rebuilds it
    with :func:`analysis._load_pose_mol`).  Returns ``{"available": False,
    "error": ...}`` instead of raising so callers can keep the rest of the
    analysis pipeline intact.
    """
    if not available():
        return {
            "available": False,
            "error": "未安装 PoseBusters，请运行 pip install posebusters",
            "module_version": "未安装",
        }
    try:
        frame = _buster().bust(mol_pred=mol, mol_cond=str(pdb_path))
    except Exception as exc:  # noqa: BLE001 - reported to the user verbatim
        return {
            "available": False,
            "error": f"PoseBusters 运行失败：{exc}",
            "module_version": module_version(),
        }
    if frame is None or frame.empty:
        return {
            "available": False,
            "error": "PoseBusters 未返回结果",
            "module_version": module_version(),
        }
    row = frame.iloc[0]
    checks = [
        _record(key, bool(row[key]))
        for key in CHECKS
        if key in frame.columns
    ]
    result = {
        "available": True,
        "config": "dock",
        "module_version": module_version(),
        "preconditions": {
            key: bool(row[key]) for key in _PRECONDITIONS if key in frame.columns
        },
    }
    result.update(_summarise(checks))
    return result


def run_batch(pdb_path: Path, molecules: list[tuple[str, object]]) -> dict:
    """Check several poses in one PoseBusters call (protein parsed once)."""
    if not available():
        return {
            "available": False,
            "error": "未安装 PoseBusters，请运行 pip install posebusters",
            "rows": [],
        }
    entries = [(name, mol) for name, mol in molecules if mol is not None]
    if not entries:
        return {"available": True, "rows": [], "total": 0, "module_version": module_version()}
    names = [name for name, _ in entries]
    mols = [mol for _, mol in entries]
    try:
        frame = _buster().bust(mol_pred=mols, mol_cond=str(pdb_path))
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "error": f"PoseBusters 批量运行失败：{exc}",
            "rows": [],
            "module_version": module_version(),
        }
    rows: list[dict] = []
    for position, (name, mol) in enumerate(entries):
        if frame is None or position >= len(frame):
            rows.append(
                {
                    "name": name,
                    "available": False,
                    "error": "PoseBusters 未返回该行结果",
                    "passed": 0,
                    "total": len(CHECKS),
                    "all_passed": False,
                    "failed_labels": [],
                }
            )
            continue
        row = frame.iloc[position]
        checks = [_record(key, bool(row[key])) for key in CHECKS if key in frame.columns]
        summary = _summarise(checks)
        rows.append({"name": name, **summary})
    passed_rows = sum(1 for row in rows if row.get("all_passed"))
    return {
        "available": True,
        "config": "dock",
        "module_version": module_version(),
        "rows": rows,
        "total": len(rows),
        "all_passed": passed_rows,
    }
