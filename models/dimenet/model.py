"""DimeNet++ wrapper (kgcnn 4.0.2 backend).

Reference: Klicpera et al. 2020, "Fast and Uncertainty-Aware Directional Message
Passing for Non-Equilibrium Molecules" (NeurIPS 2020 ML4Phys).
arXiv: https://arxiv.org/abs/2011.14115

Backend: kgcnn (https://github.com/aimat-lab/gcnn_keras), Keras 3 + TF.
Run from the kgcnn-bench env (see scripts/setup_kgcnn_env.sh).

DimeNet++ extends SchNet by including angular (3-body) features. The kgcnn
input is the SchNet input set plus `angle_indices` — pairs (edge_i, edge_j)
where the two edges share the same source atom.
"""
from typing import Any, Dict, List, Tuple

import numpy as np

from models._kgcnn_base import KGCNNModelBase
from models._kgcnn_data import structures_to_padded_batch_with_angles


EMBEDDING_LAYER = "pooling_nodes"


class DimeNetPPWrapper(KGCNNModelBase):
    embedding_dim = 256  # out_emb_size in kgcnn's default DimeNet++ config

    # kgcnn MatBench-leaderboard DimeNet++ recipe: 5 Å cutoff, 17 neighbors
    # (capped for memory because of the 3-body angle term).
    graph_cutoff = 5.0
    graph_max_neighbors = 17

    def __init__(self):
        from kgcnn.literature.DimeNetPP import make_crystal_model, model_crystal_default
        import keras
        import copy

        # DimeNet++ defaults to ragged inputs; switch to padded to match the
        # SchNet path (so the same _kgcnn_data converter works for both).
        cfg = copy.deepcopy(model_crystal_default)
        cfg["input_tensor_type"] = "padded"
        for inp in cfg["inputs"]:
            inp.pop("ragged", None)
        # Padded inputs need explicit total_nodes / total_edges / total_angles
        # placeholders so the cast layer knows segment sizes.
        existing = {i["name"] for i in cfg["inputs"]}
        for extra in ("total_nodes", "total_edges", "total_angles"):
            if extra not in existing:
                cfg["inputs"].append(
                    {"shape": (), "name": extra, "dtype": "int64"})
        # Default config is multi-target (12 outputs for QM9-style); override
        # to a single regression output for materials property prediction.
        cfg["output_mlp"] = {
            "use_bias": [True, False],
            "units": [64, 1],
            "activation": ["swish", "linear"],
        }

        self.net = make_crystal_model(**cfg)
        try:
            embedding_layer = self.net.get_layer(EMBEDDING_LAYER)
            self._emb_model = keras.Model(
                inputs=self.net.inputs, outputs=embedding_layer.output)
        except (ValueError, KeyError):
            self._emb_model = keras.Model(
                inputs=self.net.inputs, outputs=self.net.layers[-2].output)
        try:
            self.embedding_dim = int(self._emb_model.output_shape[-1])
        except Exception:
            pass

    def forward(self, batch: Dict[str, np.ndarray]) -> np.ndarray:
        x = self._batch_to_inputs(batch)
        import tensorflow as tf
        tf.config.run_functions_eagerly(True)
        return np.asarray(self.net(x, training=False))

    def get_embeddings(self, batch: Dict[str, np.ndarray]) -> np.ndarray:
        x = self._batch_to_inputs(batch)
        import tensorflow as tf
        tf.config.run_functions_eagerly(True)
        return np.asarray(self._emb_model(x, training=False))

    def forward_with_embeddings(
        self, batch: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.forward(batch), self.get_embeddings(batch)

    def make_dataset(self, structures: List, targets: List) -> Dict[str, np.ndarray]:
        return structures_to_padded_batch_with_angles(
            structures, targets,
            cutoff=self.graph_cutoff, max_neighbors=self.graph_max_neighbors)

    def collate_fn(self, batch: Any) -> Any:
        return batch

    def save_checkpoint(self, path: str) -> None:
        self.net.save(path)

    def load_checkpoint(self, path: str) -> None:
        # Full-model load via keras.models.load_model fails because kgcnn's
        # custom layers aren't registered with @keras.saving.register_keras_serializable.
        # Load weights into the freshly-built net from __init__ instead.
        self.net.load_weights(path)

    def _batch_to_inputs(self, batch: Dict[str, np.ndarray]):
        keys = [inp.name for inp in self.net.inputs] \
            if hasattr(self.net.inputs[0], "name") else None
        if keys:
            return [batch[k] for k in keys if k in batch]
        return batch
