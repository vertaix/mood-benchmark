"""MatBench data loader.

Loads a MatBench task for one cross-validation fold and returns
PyTorch-compatible datasets for train, val, and test splits.

MatBench tasks: https://matbench.materialsproject.org/
Structure tasks return pymatgen Structure objects as inputs.
Composition tasks return pymatgen Composition objects.

Usage:
    train_ds, val_ds, test_ds = load_matbench_fold(
        task_name="matbench_phonons",
        fold=0,
        val_fraction=0.1,
        seed=42,
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              collate_fn=your_model_collate_fn)
"""
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple


class MatbenchDataset(Dataset):
    """Thin Dataset wrapper around a list of (input, target) pairs."""

    def __init__(self, inputs: list, targets: list):
        assert len(inputs) == len(targets)
        self._inputs = inputs
        self._targets = targets

    def __len__(self) -> int:
        return len(self._inputs)

    def __getitem__(self, idx: int) -> Tuple:
        return (self._inputs[idx], self._targets[idx])


def _load_task(task_name: str):
    """Load and return a MatbenchTask. Separated for easy mocking in tests."""
    from matbench.bench import MatbenchBenchmark
    mb = MatbenchBenchmark(autoload=False, subset=[task_name])
    mb.load()
    return list(mb.tasks)[0]


def load_matbench_fold(
    task_name: str,
    fold: int,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[MatbenchDataset, MatbenchDataset, MatbenchDataset]:
    """Load one fold of a MatBench task.

    Args:
        task_name: MatBench task name, e.g. "matbench_phonons".
        fold: Cross-validation fold index (0-4).
        val_fraction: Fraction of train+val data held out for validation.
        seed: Random seed for the train/val split.

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    task = _load_task(task_name)

    train_val_data = task.get_train_and_val_data(fold)
    inputs_tv, targets_tv = train_val_data[0], train_val_data[1]

    # get_test_data with include_target=True returns (inputs, targets)
    inputs_test, targets_test = task.get_test_data(fold, include_target=True)

    # Shuffle and split train+val
    n = len(inputs_tv)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = int(val_fraction * n)

    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    train_ds = MatbenchDataset(
        inputs=[inputs_tv.iloc[i] for i in train_idx],
        targets=[targets_tv.iloc[i] for i in train_idx],
    )
    val_ds = MatbenchDataset(
        inputs=[inputs_tv.iloc[i] for i in val_idx],
        targets=[targets_tv.iloc[i] for i in val_idx],
    )
    test_ds = MatbenchDataset(
        inputs=list(inputs_test),
        targets=list(targets_test),
    )

    return train_ds, val_ds, test_ds
