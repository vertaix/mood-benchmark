"""SchNet wrapper (kgcnn 4.0.2 backend).

Reference: Schütt et al. 2017, "SchNet: A continuous-filter convolutional neural
network for modeling quantum interactions" (NeurIPS 2017).
arXiv: https://arxiv.org/abs/1706.08566

Backend: kgcnn (https://github.com/aimat-lab/gcnn_keras), Keras 3 + TF.
Run from the kgcnn-bench env (see scripts/setup_kgcnn_env.sh).
"""
from typing import Any, Dict, List, Tuple

import numpy as np

from models._kgcnn_base import KGCNNModelBase
from models._kgcnn_data import structures_to_padded_batch


# Embedding tap point: kgcnn's SchNet has a `pooling_nodes` layer that
# produces graph-level features just before the regression-head MLP.
EMBEDDING_LAYER = "pooling_nodes"


class SchNetWrapper(KGCNNModelBase):
    embedding_dim = 64

    # kgcnn MatBench-leaderboard SchNet recipe: 5 Å cutoff, 12 neighbors,
    # RBF range matched to cutoff. The kgcnn 4.0.2 default leaves
    # gauss_args.distance at 4 Å, which clips the RBF before the cutoff;
    # we override to 5 Å with finer binning.
    graph_cutoff = 5.0
    graph_max_neighbors = 12

    def __init__(self):
        # Lazy-import kgcnn so importing this module doesn't drag TF in
        # if we just want the class for type checks elsewhere.
        from kgcnn.literature.Schnet import make_crystal_model, model_crystal_default
        import keras
        import copy

        cfg = copy.deepcopy(model_crystal_default)
        gauss = dict(cfg.get("gauss_args", {}))
        gauss["distance"] = self.graph_cutoff
        gauss["bins"] = 25
        gauss["sigma"] = 0.4
        cfg["gauss_args"] = gauss
        self.net = make_crystal_model(**cfg)
        # Build a sub-model that returns the post-pooling, pre-output-MLP
        # graph-level embedding. Keras 3 lets us index by layer name.
        try:
            embedding_layer = self.net.get_layer(EMBEDDING_LAYER)
            self._emb_model = keras.Model(
                inputs=self.net.inputs, outputs=embedding_layer.output)
        except (ValueError, KeyError):
            # If layer naming has shifted, fall back to second-to-last.
            self._emb_model = keras.Model(
                inputs=self.net.inputs, outputs=self.net.layers[-2].output)

    # ─── KGCNNModelBase implementation ────────────────────────────────
    def forward(self, batch: Dict[str, np.ndarray]) -> np.ndarray:
        x = self._batch_to_inputs(batch)
        return np.asarray(self.net.predict(x, verbose=0))

    def get_embeddings(self, batch: Dict[str, np.ndarray]) -> np.ndarray:
        x = self._batch_to_inputs(batch)
        return np.asarray(self._emb_model.predict(x, verbose=0))

    def forward_with_embeddings(
        self, batch: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Single sub-model with two outputs would be nicer; for now do two
        # forward passes (acceptable cost for inference-time embedding extraction).
        return self.forward(batch), self.get_embeddings(batch)

    def make_dataset(self, structures: List, targets: List) -> Dict[str, np.ndarray]:
        return structures_to_padded_batch(
            structures, targets,
            cutoff=self.graph_cutoff, max_neighbors=self.graph_max_neighbors)

    def collate_fn(self, batch: Any) -> Any:
        return batch  # already batched by make_dataset

    def save_checkpoint(self, path: str) -> None:
        self.net.save(path)

    def load_checkpoint(self, path: str) -> None:
        # Full-model load via keras.models.load_model fails because kgcnn's
        # custom layers aren't registered with @keras.saving.register_keras_serializable.
        # Load weights into the freshly-built net from __init__ instead.
        self.net.load_weights(path)

    # ─── Internal ─────────────────────────────────────────────────────
    def _batch_to_inputs(self, batch: Dict[str, np.ndarray]):
        keys = [inp.name for inp in self.net.inputs] \
            if hasattr(self.net.inputs[0], "name") else None
        if keys:
            return [batch[k] for k in keys if k in batch]
        return batch
