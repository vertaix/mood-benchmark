"""Shared base for Keras/kgcnn-backed model wrappers (SchNet, MEGNet).

The PyTorch-based BaseModel in models/base.py is unsuitable for kgcnn models
(they're Keras, and inheriting torch.nn.Module would create a confusing dual
identity). This base class mirrors BaseModel's interface but for Keras
backends, so scripts/train_model.py can dispatch by architecture.

Use this from a fresh env: `conda activate kgcnn-bench` (see
scripts/setup_kgcnn_env.sh). PyTorch is NOT a runtime dep here — kgcnn is
pure Keras + TF.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np


class KGCNNModelBase(ABC):
    """Interface mirroring models/base.py:BaseModel for Keras-backed models.

    Subclasses hold a kgcnn-built keras.Model in self.net and expose the
    same forward / get_embeddings / make_dataset / collate_fn surface so
    train_model.py can drive them.
    """

    embedding_dim: int = 0  # filled by subclass

    @abstractmethod
    def forward(self, batch: Dict[str, np.ndarray]) -> np.ndarray:
        """Predict on one batched input dict. Returns shape (B, 1)."""

    @abstractmethod
    def get_embeddings(self, batch: Dict[str, np.ndarray]) -> np.ndarray:
        """Graph-level embeddings (B, embedding_dim) tapped at the post-readout,
        pre-output-MLP position."""

    @abstractmethod
    def forward_with_embeddings(
        self, batch: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (predictions (B,1), embeddings (B,d)) in one pass."""

    @abstractmethod
    def make_dataset(self, structures: List, targets: List) -> Dict[str, np.ndarray]:
        """Convert a list of pymatgen Structures + targets into the padded
        input-dict format kgcnn expects. Returned dict contains:
          - all the input arrays the kgcnn model was built with
          - 'targets': shape (N, 1) — passed to model.fit as y."""

    @abstractmethod
    def collate_fn(self, batch: List) -> Any:
        """No-op for kgcnn — included only so the same code-path that calls
        make_dataset(...) followed by collate(...) doesn't have to special-case
        the backend. Returns batch unchanged."""

    # Saving / loading: subclasses should set the model checkpoint format.
    @abstractmethod
    def save_checkpoint(self, path: str) -> None: ...

    @abstractmethod
    def load_checkpoint(self, path: str) -> None: ...
