"""pymatgen.Structure → kgcnn input-dict conversion.

kgcnn 4.x crystal models (SchNet, MEGNet, ...) take a padded input dict with
keys like:
  - node_number       : (N_atoms,)        int64
  - node_coordinates  : (N_atoms, 3)      float32
  - edge_indices      : (N_edges, 2)      int64
  - edge_image        : (N_edges, 3)      int64   (periodic image offsets)
  - graph_lattice     : (3, 3)            float32
  - total_nodes       : ()                int64
  - total_edges       : ()                int64

This module converts a list of pymatgen Structures into a batched padded dict
with a leading batch dim, suitable for keras.Model.fit / predict.

Cutoff and max-neighbor settings are kept identical to the kgcnn MatBench
configs (8 Å, max 12 neighbors) so reproducing leaderboard numbers later
won't require re-tuning preprocessing.
"""
from typing import Dict, List, Optional

import numpy as np


CUTOFF = 8.0
MAX_NEIGHBORS = 12


def _structure_to_arrays(structure, cutoff: float = CUTOFF,
                         max_neighbors: int = MAX_NEIGHBORS):
    """Single-structure conversion using pymatgen's neighbor list.

    Returns a dict of single-structure arrays (no batch dim)."""
    n = len(structure)
    coords = np.array(structure.cart_coords, dtype=np.float32)
    atomic_numbers = np.array([site.specie.Z for site in structure], dtype=np.int64)
    lattice = np.array(structure.lattice.matrix, dtype=np.float32)

    # pymatgen neighbor list — returns (center_indices, neighbor_indices,
    # image_offsets, distances) for all pairs within cutoff.
    c_idx, n_idx, offsets, dists = structure.get_neighbor_list(r=cutoff)

    # Trim to max_neighbors per center atom by ranking by distance.
    if max_neighbors is not None and len(c_idx) > 0:
        keep = []
        for atom_i in range(n):
            mask = c_idx == atom_i
            if mask.sum() <= max_neighbors:
                keep.extend(np.where(mask)[0].tolist())
            else:
                local_idx = np.where(mask)[0]
                local_dists = dists[mask]
                order = np.argsort(local_dists)[:max_neighbors]
                keep.extend(local_idx[order].tolist())
        keep = np.array(sorted(keep), dtype=np.int64)
        c_idx = c_idx[keep]
        n_idx = n_idx[keep]
        offsets = offsets[keep]

    edge_indices = np.stack([c_idx, n_idx], axis=1).astype(np.int64)
    edge_image = offsets.astype(np.int64)

    return {
        "node_number": atomic_numbers,
        "node_coordinates": coords,
        "edge_indices": edge_indices,
        "edge_image": edge_image,
        "graph_lattice": lattice,
    }


def structures_to_padded_batch(structures: List, targets: Optional[List] = None,
                               cutoff: float = CUTOFF,
                               max_neighbors: int = MAX_NEIGHBORS) -> Dict[str, np.ndarray]:
    """Convert a list of N pymatgen Structures to a padded batch dict.

    Returns a dict with leading batch dim:
      node_number       : (N_struct, max_atoms)
      node_coordinates  : (N_struct, max_atoms, 3)
      edge_indices      : (N_struct, max_edges, 2)
      edge_image        : (N_struct, max_edges, 3)
      graph_lattice     : (N_struct, 3, 3)
      total_nodes       : (N_struct,)
      total_edges       : (N_struct,)
      targets           : (N_struct, 1) if targets provided, else absent
    """
    per_struct = [_structure_to_arrays(s, cutoff, max_neighbors)
                  for s in structures]
    n_struct = len(per_struct)
    max_atoms = max(p["node_number"].shape[0] for p in per_struct)
    max_edges = max(p["edge_indices"].shape[0] for p in per_struct) or 1

    out = {
        "node_number": np.zeros((n_struct, max_atoms), dtype=np.int64),
        "node_coordinates": np.zeros((n_struct, max_atoms, 3), dtype=np.float32),
        "edge_indices": np.zeros((n_struct, max_edges, 2), dtype=np.int64),
        "edge_image": np.zeros((n_struct, max_edges, 3), dtype=np.int64),
        "graph_lattice": np.zeros((n_struct, 3, 3), dtype=np.float32),
        "total_nodes": np.zeros((n_struct,), dtype=np.int64),
        "total_edges": np.zeros((n_struct,), dtype=np.int64),
    }
    for i, p in enumerate(per_struct):
        n_a = p["node_number"].shape[0]
        n_e = p["edge_indices"].shape[0]
        out["node_number"][i, :n_a] = p["node_number"]
        out["node_coordinates"][i, :n_a] = p["node_coordinates"]
        out["edge_indices"][i, :n_e] = p["edge_indices"]
        out["edge_image"][i, :n_e] = p["edge_image"]
        out["graph_lattice"][i] = p["graph_lattice"]
        out["total_nodes"][i] = n_a
        out["total_edges"][i] = n_e

    if targets is not None:
        out["targets"] = np.asarray(targets, dtype=np.float32).reshape(-1, 1)

    return out


def split_inputs_targets(batch: Dict[str, np.ndarray]):
    """Separate model inputs from training targets in a batch dict."""
    targets = batch.pop("targets") if "targets" in batch else None
    return batch, targets


def edge_pairs_sharing_center(edge_indices: np.ndarray) -> np.ndarray:
    """For each pair of edges (i, j) where edges share the same source atom,
    return [i, j] index pairs. Used as DimeNet++'s `angle_indices`.

    edge_indices: (E, 2) where col 0 is source atom, col 1 is target.
    Returns (P, 2) int64 array of (edge_i, edge_j) pairs (i != j).
    """
    if len(edge_indices) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    src = edge_indices[:, 0]
    pairs = []
    # Group edges by source atom
    order = np.argsort(src, kind="stable")
    sorted_src = src[order]
    starts = np.searchsorted(sorted_src, np.arange(sorted_src.max() + 1), side="left")
    ends = np.searchsorted(sorted_src, np.arange(sorted_src.max() + 1), side="right")
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        edges = order[s:e]
        # all ordered (i, j), i != j pairs of edges sharing this source
        ii, jj = np.meshgrid(edges, edges, indexing="ij")
        mask = ii != jj
        pairs.append(np.stack([ii[mask], jj[mask]], axis=1))
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64)
    return np.concatenate(pairs, axis=0).astype(np.int64)


def structures_to_padded_batch_with_angles(
    structures: List, targets=None,
    cutoff: float = CUTOFF, max_neighbors: int = MAX_NEIGHBORS,
) -> Dict[str, np.ndarray]:
    """Same as structures_to_padded_batch but also computes angle_indices
    (DimeNet++ needs these in addition to edge_indices)."""
    base = structures_to_padded_batch(structures, targets, cutoff, max_neighbors)
    n_struct = base["edge_indices"].shape[0]
    per_angles = []
    for i in range(n_struct):
        ne = int(base["total_edges"][i])
        ang = edge_pairs_sharing_center(base["edge_indices"][i, :ne])
        per_angles.append(ang)
    max_a = max((a.shape[0] for a in per_angles), default=1) or 1
    angle_indices = np.zeros((n_struct, max_a, 2), dtype=np.int64)
    total_angles = np.zeros((n_struct,), dtype=np.int64)
    for i, a in enumerate(per_angles):
        angle_indices[i, :a.shape[0]] = a
        total_angles[i] = a.shape[0]
    base["angle_indices"] = angle_indices
    base["total_angles"] = total_angles
    return base
