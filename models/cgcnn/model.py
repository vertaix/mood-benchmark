"""CGCNN wrapper for Graph-OOD-Bench.

Uses the official CGCNN implementation from external/cgcnn/ (Xie & Grossman, PRL 2018).
Provides the same BaseModel interface as ALIGNNWrapper.

Embedding = output of the hidden FC layer(s) after graph pooling, before fc_out.
Typically 128-dim with default settings.
"""
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Any, List, Tuple
from models.base import BaseModel

# Add external cgcnn to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'external', 'cgcnn'))
from cgcnn.model import CrystalGraphConvNet


# ---------------------------------------------------------------------------
# Gaussian distance expansion (from original CGCNN data.py)
# ---------------------------------------------------------------------------

class GaussianDistance:
    """Expand distances into Gaussian basis."""
    def __init__(self, dmin=0.0, dmax=8.0, step=0.2):
        self.filter = np.arange(dmin, dmax + step, step)

    def expand(self, distances):
        return np.exp(-(distances[..., np.newaxis] - self.filter) ** 2
                      / self.filter[1] ** 2)

    @property
    def nbr_fea_len(self):
        return len(self.filter)


# ---------------------------------------------------------------------------
# Graph construction from pymatgen Structure
# ---------------------------------------------------------------------------

def structure_to_cgcnn_input(structure, max_nbr=12, radius=8.0, gauss=None):
    """Convert pymatgen Structure to CGCNN input tensors.

    Returns:
        atom_fea: (n_atoms, n_elem) one-hot element features
        nbr_fea: (n_atoms, max_nbr, nbr_fea_len) Gaussian-expanded distances
        nbr_idx: (n_atoms, max_nbr) neighbor indices
    """
    n_elem = 100
    if gauss is None:
        gauss = GaussianDistance()

    # One-hot atom features by atomic number
    atom_fea = np.zeros((len(structure), n_elem), dtype=np.float32)
    for i, site in enumerate(structure):
        z = min(site.specie.Z, n_elem) - 1
        atom_fea[i, z] = 1.0

    # Get neighbors
    all_nbrs = structure.get_all_neighbors(radius, include_index=True)

    nbr_idx = np.zeros((len(structure), max_nbr), dtype=np.int64)
    nbr_dist = np.zeros((len(structure), max_nbr), dtype=np.float32)

    for i, nbrs in enumerate(all_nbrs):
        if len(nbrs) == 0:
            nbr_idx[i] = i
            nbr_dist[i] = radius
            continue
        nbrs = sorted(nbrs, key=lambda x: x[1])
        if len(nbrs) < max_nbr:
            while len(nbrs) < max_nbr:
                nbrs.append(nbrs[0])
        for j in range(max_nbr):
            nbr_idx[i, j] = nbrs[j][2]
            nbr_dist[i, j] = nbrs[j][1]

    nbr_fea = gauss.expand(nbr_dist)

    return (torch.FloatTensor(atom_fea),
            torch.FloatTensor(nbr_fea),
            torch.LongTensor(nbr_idx))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CGCNNDataset(Dataset):
    """Dataset that converts structures to CGCNN graph format."""

    def __init__(self, structures, targets, max_nbr=12, radius=8.0):
        self.structures = structures
        self.targets = targets
        self.max_nbr = max_nbr
        self.radius = radius
        self.gauss = GaussianDistance()

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        struct = self.structures[idx]
        target = self.targets[idx]
        atom_fea, nbr_fea, nbr_idx = structure_to_cgcnn_input(
            struct, self.max_nbr, self.radius, self.gauss
        )
        return (atom_fea, nbr_fea, nbr_idx), target


def cgcnn_collate_fn(batch):
    """Collate CGCNN graphs into a batch."""
    graphs, targets = zip(*batch)
    batch_atom_fea = []
    batch_nbr_fea = []
    batch_nbr_idx = []
    crystal_atom_idx = []
    offset = 0

    for (atom_fea, nbr_fea, nbr_idx) in graphs:
        n_atoms = atom_fea.shape[0]
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_idx.append(nbr_idx + offset)
        crystal_atom_idx.append(torch.arange(offset, offset + n_atoms))
        offset += n_atoms

    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_idx, dim=0),
            crystal_atom_idx), \
           torch.tensor(targets, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class CGCNNWrapper(BaseModel):
    """CGCNN wrapper using the official implementation."""

    def __init__(self, atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 max_nbr=12, radius=8.0):
        super().__init__()
        gauss = GaussianDistance()
        self.net = CrystalGraphConvNet(
            orig_atom_fea_len=100,
            nbr_fea_len=gauss.nbr_fea_len,
            atom_fea_len=atom_fea_len,
            n_conv=n_conv,
            h_fea_len=h_fea_len,
            n_h=n_h,
            classification=False,
        )
        self.embedding_dim = h_fea_len
        self.max_nbr = max_nbr
        self.radius = radius

    def _to_device(self, batch):
        atom_fea, nbr_fea, nbr_idx, crystal_atom_idx = batch
        device = next(self.parameters()).device
        return (atom_fea.to(device),
                nbr_fea.to(device),
                nbr_idx.to(device),
                [idx.to(device) for idx in crystal_atom_idx])

    def forward(self, batch) -> torch.Tensor:
        atom_fea, nbr_fea, nbr_idx, crystal_atom_idx = self._to_device(batch)
        return self.net(atom_fea, nbr_fea, nbr_idx, crystal_atom_idx)

    def _get_embedding(self, atom_fea, nbr_fea, nbr_idx, crystal_atom_idx):
        """Extract crystal embedding (before fc_out)."""
        x = self.net.embedding(atom_fea)
        for conv in self.net.convs:
            x = conv(x, nbr_fea, nbr_idx)
        crys_fea = self.net.pooling(x, crystal_atom_idx)
        crys_fea = self.net.conv_to_fc(self.net.conv_to_fc_softplus(crys_fea))
        crys_fea = self.net.conv_to_fc_softplus(crys_fea)
        if hasattr(self.net, 'fcs') and hasattr(self.net, 'softpluses'):
            for fc, sp in zip(self.net.fcs, self.net.softpluses):
                crys_fea = sp(fc(crys_fea))
        return crys_fea

    def get_embeddings(self, batch) -> torch.Tensor:
        atom_fea, nbr_fea, nbr_idx, crystal_atom_idx = self._to_device(batch)
        with torch.no_grad():
            return self._get_embedding(atom_fea, nbr_fea, nbr_idx,
                                       crystal_atom_idx)

    def forward_with_embeddings(self, batch):
        atom_fea, nbr_fea, nbr_idx, crystal_atom_idx = self._to_device(batch)
        emb = self._get_embedding(atom_fea, nbr_fea, nbr_idx,
                                  crystal_atom_idx)
        out = self.net.fc_out(emb)
        return out, emb

    def make_dataset(self, structures, targets):
        return CGCNNDataset(structures, targets, self.max_nbr, self.radius)

    def collate_fn(self, batch):
        return cgcnn_collate_fn(batch)
