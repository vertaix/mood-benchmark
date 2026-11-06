# models/base.py
from abc import ABC, abstractmethod
from typing import Any, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BaseModel(nn.Module, ABC):
    """Shared interface for all model wrappers."""

    @abstractmethod
    def forward(self, batch: Any) -> torch.Tensor:
        """Return predictions of shape (B, 1)."""

    @abstractmethod
    def get_embeddings(self, batch: Any) -> torch.Tensor:
        """Return crystal-level embeddings of shape (B, d)."""

    @abstractmethod
    def forward_with_embeddings(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (predictions (B,1), embeddings (B,d)) with gradients."""

    @abstractmethod
    def make_dataset(self, structures: List, targets: List) -> Dataset:
        """Convert structures + targets to a model-specific Dataset."""

    @abstractmethod
    def collate_fn(self, batch: List) -> Any:
        """Custom collate function for use with DataLoader."""
