"""OOD split construction for Graph-OOD-Bench.

Split types:
  Physics-driven (1 fold each, single AUROC per setting):
    Compositional:
      comp_oxide          — OOD = contains oxygen
      comp_halide         — OOD = contains halogens (F, Cl, Br, I)
      comp_chalcogenide   — OOD = contains S/Se/Te but no O
      comp_nitride        — OOD = contains N but no O
      comp_carbide        — OOD = contains C but no O or N
      comp_intermetallic  — OOD = pure metals only (no common anions)
    Structural:
      struct_cubic           — OOD = cubic crystal system
      struct_perovskite      — OOD = ABX3 perovskite stoichiometry
      struct_hexagonal       — OOD = hexagonal crystal system
      struct_triclinic       — OOD = triclinic crystal system
      struct_monoclinic      — OOD = monoclinic crystal system
      struct_tetragonal      — OOD = tetragonal crystal system
      struct_orthorhombic    — OOD = orthorhombic crystal system
      struct_high_spacegroup — OOD = spacegroup >= 195
      struct_low_spacegroup  — OOD = spacegroup <= 15

  Data-driven / Omee splits (50 folds each, mean AUROC ± std):
    LOCO             — leave one OFM cluster out
    SparseXcluster   — cluster in sparse feature space region
    SparseYcluster   — cluster with extreme property values
    Source: https://github.com/sadmanomee/OOD_Materials_Benchmark

ID is always the majority class. At test time, the ID test set is
subsampled to match the OOD test set size (50:50 ratio) for fair
AUROC comparison across splits.

The model is trained on the FULL training set (no split awareness).
At test time, we split the test set into ID-test vs OOD-test based on
the criterion. This reflects realistic deployment: model trained normally,
detector must flag OOD at inference time.
"""
import json
import numpy as np
from typing import Tuple, Optional
from data.matbench_loader import load_matbench_fold


# ─── Element sets ────────────────────────────────────────────────────────
HALOGENS = {'F', 'Cl', 'Br', 'I'}
CHALCOGENS = {'S', 'Se', 'Te'}
ANIONS = {'O', 'N', 'C', 'S', 'Se', 'Te', 'F', 'Cl', 'Br', 'I'}

MIN_OOD_SAMPLES = 50


def _get_elements(structure) -> set:
    """Extract element symbols from a pymatgen Structure."""
    return {str(sp) for sp in structure.composition.elements}


def _get_crystal_system(structure) -> str:
    """Get crystal system string from a pymatgen Structure."""
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    try:
        sga = SpacegroupAnalyzer(structure, symprec=0.1)
        return sga.get_crystal_system()
    except Exception:
        return "unknown"


def _get_spacegroup_number(structure) -> int:
    """Get spacegroup number from a pymatgen Structure."""
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    try:
        sga = SpacegroupAnalyzer(structure, symprec=0.1)
        return sga.get_space_group_number()
    except Exception:
        return -1


def _is_perovskite(structure) -> bool:
    """Check if structure has perovskite-like prototype (ABX3 stoichiometry)."""
    comp = structure.composition.reduced_composition
    if len(comp.elements) != 3:
        return False
    amounts = sorted(comp.get_el_amt_dict().values())
    return amounts == [1.0, 1.0, 3.0]


# ─── Split classifiers ──────────────────────────────────────────────────
# Each returns True for the "special" (potentially OOD) group.
SPLIT_CLASSIFIERS = {
    # compositional
    'comp_oxide':         lambda s: 'O' in _get_elements(s),
    'comp_halide':        lambda s: bool(_get_elements(s) & HALOGENS),
    'comp_chalcogenide':  lambda s: bool(_get_elements(s) & CHALCOGENS) and 'O' not in _get_elements(s),
    'comp_nitride':       lambda s: 'N' in _get_elements(s) and 'O' not in _get_elements(s),
    'comp_carbide':       lambda s: 'C' in _get_elements(s) and not _get_elements(s) & {'O', 'N'},
    'comp_intermetallic': lambda s: not bool(_get_elements(s) & ANIONS),
    # structural
    'struct_cubic':           lambda s: _get_crystal_system(s) == 'cubic',
    'struct_hexagonal':       lambda s: _get_crystal_system(s) == 'hexagonal',
    'struct_triclinic':       lambda s: _get_crystal_system(s) == 'triclinic',
    'struct_monoclinic':      lambda s: _get_crystal_system(s) == 'monoclinic',
    'struct_tetragonal':      lambda s: _get_crystal_system(s) == 'tetragonal',
    'struct_orthorhombic':    lambda s: _get_crystal_system(s) == 'orthorhombic',
    'struct_high_spacegroup': lambda s: _get_spacegroup_number(s) >= 195,
    'struct_low_spacegroup':  lambda s: 0 < _get_spacegroup_number(s) <= 15,
    'struct_perovskite':      lambda s: _is_perovskite(s),
}


def _load_test_structures(task_name: str, fold: int = 0, seed: int = 42):
    """Load test structures for a task."""
    _, _, test_ds = load_matbench_fold(task_name, fold=fold, seed=seed)
    return test_ds._inputs


def balance_split(
    id_indices: np.ndarray,
    ood_indices: np.ndarray,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Subsample ID to match OOD size for 50:50 ratio.

    ID must be the majority class (len(id) >= len(ood)).
    Subsamples ID indices randomly to len(ood).
    """
    n_ood = len(ood_indices)
    n_id = len(id_indices)

    if n_id <= n_ood:
        return id_indices, ood_indices

    rng = np.random.default_rng(seed)
    sampled_id = rng.choice(id_indices, size=n_ood, replace=False)
    return sampled_id, ood_indices


# ─── Physics-driven splits (1 fold) ─────────────────────────────────────

def physics_split(
    task_name: str,
    split_type: str,
    fold: int = 0,
    seed: int = 42,
    balanced: bool = True,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Physics-driven OOD split on the test set.

    ID is always the majority class. OOD is the minority (distinctive) class.

    Args:
        task_name: MatBench task name
        split_type: One of 'comp_oxide', 'comp_halide',
                    'struct_cubic', 'struct_perovskite'
        balanced: If True, subsample ID to match OOD for 50:50 ratio

    Returns:
        (id_indices, ood_indices) into the test set, or None if too few OOD
    """
    if split_type not in SPLIT_CLASSIFIERS:
        raise ValueError(f"Unknown split type: {split_type}. "
                         f"Available: {list(SPLIT_CLASSIFIERS.keys())}")

    test_structures = _load_test_structures(task_name, fold, seed)
    classifier = SPLIT_CLASSIFIERS[split_type]

    # Classify each test sample into group A (special) or group B (general)
    group_a = []
    group_b = []

    for i, struct in enumerate(test_structures):
        if classifier(struct):
            group_a.append(i)
        else:
            group_b.append(i)

    group_a = np.array(group_a)
    group_b = np.array(group_b)

    # ID = majority, OOD = minority
    if len(group_a) >= len(group_b):
        id_indices, ood_indices = group_a, group_b
    else:
        id_indices, ood_indices = group_b, group_a

    if len(ood_indices) < MIN_OOD_SAMPLES:
        return None

    if balanced:
        id_indices, ood_indices = balance_split(id_indices, ood_indices, seed)

    return id_indices, ood_indices


# ─── Data-driven / Omee splits (5 folds) ────────────────────────────────

def load_omee_split(
    task_name: str,
    split_type: str,
    fold: int,
    omee_dir: str = 'splits/omee',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a single Omee fold.

    Args:
        task_name: 'matbench_dielectric' or 'matbench_log_gvrh'
        split_type: 'LOCO', 'SparseXcluster', or 'SparseYcluster'
        fold: Fold index (0-4)
        omee_dir: Path to directory containing Omee JSON files

    Returns:
        (train_indices, id_test_indices, ood_test_indices)
        These index into the full dataset array.
    """
    path = f"{omee_dir}/{task_name}/{split_type}/fold_{fold}.json"
    with open(path) as f:
        fold_data = json.load(f)

    train_idx = np.array(fold_data['train_indices'])
    id_test_idx = np.array(fold_data['id_test_indices'])
    ood_test_idx = np.array(fold_data['ood_test_indices'])

    return train_idx, id_test_idx, ood_test_idx


# ─── Registry ───────────────────────────────────────────────────────────

PHYSICS_SPLITS = list(SPLIT_CLASSIFIERS.keys())
OMEE_SPLITS = ['LOCO', 'SparseXcluster', 'SparseYcluster']
OMEE_N_FOLDS = 50

SPLIT_REGISTRY = {
    **{name: ('physics', name) for name in PHYSICS_SPLITS},
    # Data-driven / Omee (50 folds each)
    'LOCO': ('omee', 'LOCO'),
    'SparseXcluster': ('omee', 'SparseXcluster'),
    'SparseYcluster': ('omee', 'SparseYcluster'),
}


def get_split(
    task_name: str,
    split_name: str,
    fold: int = 0,
    seed: int = 42,
    omee_dir: str = 'splits/omee',
    balanced: bool = True,
) -> Optional[Tuple[np.ndarray, ...]]:
    """Unified interface to get any split by name.

    For physics splits: returns (id_indices, ood_indices) into the test set.
        ID is always the majority class. If balanced=True, ID is subsampled
        to match OOD size (50:50).
    For Omee splits: returns (train_indices, id_test_indices, ood_test_indices)
        into the full dataset.
    """
    if split_name not in SPLIT_REGISTRY:
        raise ValueError(
            f"Unknown split: {split_name}. "
            f"Available: {list(SPLIT_REGISTRY.keys())}"
        )

    split_family, split_arg = SPLIT_REGISTRY[split_name]

    if split_family == 'physics':
        return physics_split(task_name, split_arg, fold, seed, balanced)
    elif split_family == 'omee':
        return load_omee_split(task_name, split_arg, fold, omee_dir)
    else:
        raise ValueError(f"Unknown split family: {split_family}")
