"""Binding-site box helpers and a lightweight LIGSITE-style pocket finder."""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .pdbio import parse_pdb


def suggest_box(
    box: dict,
    spans: list[list[float]] | None = None,
    padding: float = 4.0,
    max_size: float = 40.0,
) -> dict:
    """Expand a docking box so the largest ligand still fits with padding.

    ``spans`` are per-ligand ``[dx, dy, dz]`` extents in Å.  The box is only
    grown, never shrunk, and capped at ``max_size`` to keep the search space
    bounded.
    """
    center = [float(value) for value in (box.get("center") or [0, 0, 0])]
    size = [float(value) for value in (box.get("size") or [20, 20, 20])]
    original = list(size)
    for span in spans or []:
        for axis in range(3):
            wanted = float(span[axis]) + 2.0 * padding
            size[axis] = min(max_size, max(size[axis], wanted))
    return {
        "center": center,
        "size": [round(value, 2) for value in size],
        "expanded": any(
            size[axis] > original[axis] + 1e-6 for axis in range(3)
        ),
        "original_size": [round(value, 2) for value in original],
    }


def box_volume(size: list[float] | tuple[float, float, float]) -> float:
    return float(size[0] * size[1] * size[2])


def box_from_coords(coords, padding: float = 5.0, min_size: float = 18.0) -> dict:
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or len(arr) == 0:
        raise ValueError("没有可用的坐标")
    lo = arr.min(0)
    hi = arr.max(0)
    size = np.maximum(hi - lo + 2 * padding, min_size)
    center = (lo + hi) / 2.0
    return {
        "center": [round(float(v), 3) for v in center],
        "size": [round(float(v), 2) for v in size],
        "source": "ligand",
    }


def _grid_for_atoms(coords: np.ndarray):
    spacing = 0.9
    pad = 6.0
    lo = coords.min(0) - pad
    hi = coords.max(0) + pad
    while True:
        axes = [np.arange(lo[i], hi[i] + spacing, spacing) for i in range(3)]
        shape = tuple(len(a) for a in axes)
        if shape[0] * shape[1] * shape[2] <= 3_500_000 or spacing > 2.0:
            return spacing, lo, axes, shape
        spacing *= 1.25


HYDROPHOBIC_RESIDUES = {
    "ALA", "VAL", "LEU", "ILE", "PHE", "TRP", "MET", "PRO", "TYR", "CYS",
}


def detect_pockets(pdb_text: str, top: int = 5) -> list[dict]:
    """Detect buried cavities.

    This is a fast heuristic inspired by LIGSITE: a grid point is a pocket
    candidate when it is outside the van der Waals surface and protein atoms
    are found in both directions along at least four of nine scan axes.
    It is not a replacement for fpocket, but needs no external binary and
    runs in well under a second for typical receptors.
    """
    st = parse_pdb(pdb_text)
    protein_atoms = [
        a for a in st.atoms if a.record == "ATOM" and a.element != "H"
    ]
    coords = np.array([[a.x, a.y, a.z] for a in protein_atoms])
    hydrophobic = np.array(
        [1.0 if a.resname in HYDROPHOBIC_RESIDUES else 0.0 for a in protein_atoms]
    )
    if len(coords) < 80:
        return []

    spacing, lo, axes, shape = _grid_for_atoms(coords)
    mesh = np.meshgrid(*axes, indexing="ij")
    points = np.c_[mesh[0].ravel(), mesh[1].ravel(), mesh[2].ravel()]
    tree = cKDTree(coords)
    dist, _ = tree.query(points, workers=-1)

    protein = (dist < 2.0).reshape(shape)
    free = (dist > 2.7).reshape(shape)

    directions = []
    for vec in [
        (1, 0, 0), (0, 1, 0), (0, 0, 1),
        (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1),
        (0, 1, 1), (0, 1, -1),
    ]:
        arr = np.asarray(vec, dtype=float)
        directions.append(arr / np.linalg.norm(arr))

    scan_steps = 10

    def hit_both(mask: np.ndarray, vec: np.ndarray) -> np.ndarray:
        plus = np.zeros_like(mask)
        minus = np.zeros_like(mask)
        for k in range(1, scan_steps + 1):
            shift = [int(round(-k * vec[i])) for i in range(3)]
            plus |= np.roll(mask, shift, axis=(0, 1, 2))
            minus |= np.roll(mask, [-s for s in shift], axis=(0, 1, 2))
        return plus & minus

    buriedness = np.zeros(shape, dtype=np.int8)
    for vec in directions:
        buriedness += hit_both(protein, vec).astype(np.int8)

    pocket = free & (buriedness >= 4)
    labels, count = ndimage.label(pocket, structure=np.ones((3, 3, 3)))
    if count == 0:
        return []

    sizes = ndimage.sum(pocket, labels, range(1, count + 1))
    boundary = np.zeros(shape, dtype=bool)
    boundary[0, :, :] = boundary[-1, :, :] = True
    boundary[:, 0, :] = boundary[:, -1, :] = True
    boundary[:, :, 0] = boundary[:, :, -1] = True

    candidates = []
    for idx in range(1, count + 1):
        voxels = int(sizes[idx - 1])
        if voxels < 20:
            continue
        mask = labels == idx
        if np.any(mask & boundary):
            continue
        positions = np.argwhere(mask)
        coords_pocket = lo + positions * spacing
        lo_p = coords_pocket.min(0)
        hi_p = coords_pocket.max(0)
        center = (lo_p + hi_p) / 2.0
        ext = hi_p - lo_p
        size = np.clip(ext + 10.0, 18.0, 28.0)
        mean_buried = float(buriedness[mask].mean()) if voxels else 0.0
        score = float(voxels * (mean_buried / 9.0))
        # Ligandability: a large, enclosed, hydrophobic cavity is more likely
        # to bind a drug-like molecule than a shallow polar groove.
        near = np.linalg.norm(coords - center, axis=1) <= 8.0
        hydrophobic_fraction = (
            float(hydrophobic[near].mean()) if np.any(near) else 0.0
        )
        volume_score = min(1.0, max(0.0, np.log10(max(voxels, 1)) / 3.0))
        druggability = 100.0 * (
            0.45 * volume_score
            + 0.30 * (mean_buried / 9.0)
            + 0.25 * hydrophobic_fraction
        )
        candidates.append(
            {
                "center": [round(float(v), 3) for v in center],
                "size": [round(float(v), 2) for v in size],
                "voxels": voxels,
                "buriedness": round(mean_buried, 2),
                "hydrophobic_fraction": round(float(hydrophobic_fraction), 3),
                "druggability": round(float(druggability), 1),
                "score": round(score, 1),
                "source": "pocket",
            }
        )

    candidates.sort(
        key=lambda x: (-x.get("druggability", 0), -x.get("score", 0))
    )
    candidates = candidates[:top]
    for rank, item in enumerate(candidates, start=1):
        item["rank"] = rank
        item["id"] = f"pocket_{rank}"
        item["label"] = f"预测口袋 {rank}"
    return candidates
