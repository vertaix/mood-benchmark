"""Generate physics-driven OOD splits from scratch.

For each (dataset, split_type):
  1. Load ALL structures from the MatBench dataset
  2. Classify each structure using the split classifier
  3. Majority class = ID, minority class = OOD
  4. 90/10 split of ID -> train_idx, id_test_idx
  5. All of OOD -> ood_test_idx
  6. Balance id_test and ood_test to min(n_id_test, n_ood_test)
  7. Save .npy files + stats.json

Usage:
    python scripts/generate_physics_splits.py
    python scripts/generate_physics_splits.py --splits comp_chalcogenide comp_nitride
    python scripts/generate_physics_splits.py --check  # dry run
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mood import paths as _P
from data.ood_splits import SPLIT_CLASSIFIERS, MIN_OOD_SAMPLES

DATASETS = [
    "matbench_dielectric",
    "matbench_log_gvrh",
    "matbench_log_kvrh",
    "matbench_phonons",
    "matbench_mp_gap",
]
TRAIN_FRACTION = 0.90
SEED = 42


def load_all_structures(task_name):
    """Load all structures from a MatBench task."""
    from matbench.bench import MatbenchBenchmark
    mb = MatbenchBenchmark(autoload=False, subset=[task_name])
    mb.load()
    task = list(mb.tasks)[0]
    # Get the full dataset (inputs column = structures)
    df = task.df
    return list(df.iloc[:, 0])


def generate_split(dataset_name, split_type, structures, write=True):
    """Generate a single physics split."""
    classifier = SPLIT_CLASSIFIERS[split_type]
    out = _P.splits / dataset_name / split_type

    # Classify all structures
    special = []  # matches criterion
    general = []  # does not match
    for i, struct in enumerate(structures):
        if classifier(struct):
            special.append(i)
        else:
            general.append(i)

    special = np.array(special)
    general = np.array(general)

    # ID = majority, OOD = minority
    if len(general) >= len(special):
        id_pool, ood_pool = general, special
    else:
        id_pool, ood_pool = special, general

    n_id, n_ood = len(id_pool), len(ood_pool)
    n_total = n_id + n_ood

    print(f"\n[{dataset_name} / {split_type}]")
    print(f"  special={len(special)}, general={len(general)}, total={n_total}")
    print(f"  ID (majority)={n_id}, OOD (minority)={n_ood}")

    if n_ood < MIN_OOD_SAMPLES:
        print(f"  SKIPPED: only {n_ood} OOD samples (need >= {MIN_OOD_SAMPLES})")
        return None

    # 90/10 split of ID -> train + id_test
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_id)
    n_train = int(round(TRAIN_FRACTION * n_id))
    train_idx = np.sort(id_pool[perm[:n_train]])
    id_test_idx = np.sort(id_pool[perm[n_train:]])
    ood_test_idx = np.sort(ood_pool)

    # Balance
    n_balanced = min(len(id_test_idx), len(ood_test_idx))
    bal_id = np.sort(rng.choice(id_test_idx, size=n_balanced, replace=False))
    bal_ood = np.sort(rng.choice(ood_test_idx, size=n_balanced, replace=False))

    stats = {
        "n_total": int(n_total),
        "n_id": int(n_id),
        "n_ood": int(n_ood),
        "n_train": int(len(train_idx)),
        "n_id_test": int(len(id_test_idx)),
        "n_ood_test": int(len(ood_test_idx)),
        "id_ood_ratio": f"{n_id}:{n_ood}",
        "n_balanced": int(n_balanced),
    }
    print(f"  -> train={stats['n_train']}, id_test={stats['n_id_test']}, "
          f"ood_test={stats['n_ood_test']}, balanced={stats['n_balanced']}")

    if not write:
        return stats

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "train_idx.npy", train_idx)
    np.save(out / "id_test_idx.npy", id_test_idx)
    np.save(out / "ood_test_idx.npy", ood_test_idx)
    np.save(out / "balanced_id_test_idx.npy", bal_id)
    np.save(out / "balanced_ood_test_idx.npy", bal_ood)
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"  wrote {out}")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--splits", nargs="+", default=list(SPLIT_CLASSIFIERS.keys()),
                    choices=list(SPLIT_CLASSIFIERS.keys()))
    ap.add_argument("--check", action="store_true",
                    help="Print sizes without writing")
    args = ap.parse_args()

    print(f"Datasets: {args.datasets}")
    print(f"Splits:   {args.splits}")
    print(f"Mode:     {'CHECK (dry run)' if args.check else 'WRITE'}\n")

    for dataset in args.datasets:
        print(f"{'='*60}")
        print(f"Loading {dataset}...")
        structures = load_all_structures(dataset)
        print(f"Loaded {len(structures)} structures")

        for split in args.splits:
            generate_split(dataset, split, structures, write=not args.check)


if __name__ == "__main__":
    main()
