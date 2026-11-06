# models/alignn/model.py
from typing import Any, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from models.base import BaseModel

try:
    from alignn.models.alignn import ALIGNN, ALIGNNConfig
except ImportError:
    ALIGNN, ALIGNNConfig = None, None

try:
    from jarvis.core.atoms import Atoms as JAtoms
    from alignn.graphs import Graph
except ImportError:
    JAtoms, Graph = None, None


def _structure_to_dgl_graph(structure, cutoff: float = 8.0, max_nbr: int = 12):
    """Convert pymatgen Structure to (g, lg) DGL graph pair for ALIGNN."""
    jarvis_atoms = JAtoms(
        lattice_mat=structure.lattice.matrix,
        elements=[str(site.specie) for site in structure],
        coords=[site.frac_coords for site in structure],
        cartesian=False,
    )
    g, lg = Graph.atom_dgl_multigraph(
        jarvis_atoms, cutoff=cutoff, max_neighbors=max_nbr
    )
    return g, lg


class _ALIGNNDataset(Dataset):
    def __init__(self, structures, targets, cutoff, max_nbr):
        self.data = [
            (_structure_to_dgl_graph(s, cutoff, max_nbr), t)
            for s, t in zip(structures, targets)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ALIGNNWrapper(BaseModel):
    def __init__(self, config=None, embedding_dim: int = 256,
                 cutoff: float = 8.0, max_nbr: int = 12):
        super().__init__()
        cfg = config or ALIGNNConfig(name='alignn', alignn_layers=4, gcn_layers=4,
                                      atom_input_features=92, edge_input_features=80,
                                      triplet_input_features=40, embedding_features=64,
                                      hidden_features=256, output_features=1)
        self.net = ALIGNN(cfg)
        self.embedding_dim = embedding_dim
        self.cutoff = cutoff
        self.max_nbr = max_nbr

    def forward(self, batch: Tuple) -> torch.Tensor:
        g, lg = batch
        return self.net([g, lg, None])

    def _extract_embeddings(self, g, lg) -> torch.Tensor:
        """Tap mean-pooled node features before the final linear layer.

        Replicates the ALIGNN forward pass (atom_embedding -> edge_embedding ->
        angle_embedding -> alignn_layers -> gcn_layers -> readout) but stops
        before the fc head.
        """
        device = next(self.parameters()).device
        g = g.to(device)
        lg = lg.to(device)
        lg = lg.local_var()

        # Angle features
        z = self.net.angle_embedding(lg.edata.pop('h'))

        g = g.local_var()
        # Node features
        x = self.net.atom_embedding(g.ndata.pop('atom_features'))
        # Edge features
        bondlength = torch.norm(g.edata.pop('r'), dim=1)
        y = self.net.edge_embedding(bondlength)

        # ALIGNN updates
        for alignn_layer in self.net.alignn_layers:
            x, y, z = alignn_layer(g, lg, x, y, z)

        # Gated GCN updates
        for gcn_layer in self.net.gcn_layers:
            x, y = gcn_layer(g, x, y)

        # Readout (mean pooling)
        h = self.net.readout(g, x)
        return h

    def get_embeddings(self, batch: Tuple) -> torch.Tensor:
        g, lg = batch
        with torch.no_grad():
            return self._extract_embeddings(g, lg)

    def forward_with_embeddings(self, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        g, lg = batch
        emb = self._extract_embeddings(g, lg)
        out = self.net.fc(emb)
        return out, emb

    def make_dataset(self, structures: List, targets: List) -> Dataset:
        return _ALIGNNDataset(structures, targets, self.cutoff, self.max_nbr)

    def collate_fn(self, batch: List) -> Tuple:
        import dgl
        graphs, targets = zip(*batch)
        g_list, lg_list = zip(*graphs)
        bg = dgl.batch(g_list)
        blg = dgl.batch(lg_list)
        return (bg, blg), torch.tensor(targets, dtype=torch.float32)
