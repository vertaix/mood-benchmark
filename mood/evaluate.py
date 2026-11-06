"""Evaluate a post-hoc OOD detector on every cell of the MOOD embeddings.

    from mood.evaluate import evaluate_detector
    evaluate_detector(score_fn=my_fn, embeddings_root="data/embeddings/",
                      output="results/my_detector.csv")

``score_fn(z_train, z_test)`` returns one score per row of ``z_test``;
**higher = more in-distribution**. Per cell, the ID-test and OOD-test points are
scored together (one fit per cell), the majority class is subsampled to the
minority size with a fixed seed, and AUROC is computed with ID=1, OOD=0 --
the same protocol as the paper's results.
"""
import csv
import os
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

_FIELDS = ["dataset", "arch", "split", "fold", "seed", "n_id", "n_ood", "auroc"]


def balanced_auroc(id_scores, ood_scores, seed=42):
    """AUROC with ID=1/OOD=0 after subsampling the larger class (same RNG
    draw order as scripts/run_detectors.py)."""
    rng = np.random.RandomState(seed)
    n_min = min(len(id_scores), len(ood_scores))
    if n_min < 2:
        return float("nan")
    id_sub = rng.choice(id_scores, size=n_min, replace=False)
    ood_sub = rng.choice(ood_scores, size=n_min, replace=False)
    labels = np.concatenate([np.ones(n_min), np.zeros(n_min)])
    return float(roc_auc_score(labels, np.concatenate([id_sub, ood_sub])))


def find_cells(embeddings_root):
    """Yield (dataset, arch, split, fold, seed, dir) for every directory that
    holds train.npy / id_test.npy / ood_test.npy. Layout:
    <root>/<dataset>/<arch>/<split>/fold_<f>[/seed_<s>]  (no seed dir = seed 42;
    <root> may also be a single dataset directory)."""
    root = Path(embeddings_root)
    for tr in sorted(root.rglob("train.npy")):
        d = tr.parent
        if not (d / "id_test.npy").exists() or not (d / "ood_test.npy").exists():
            continue
        parts = d.relative_to(root).parts
        if not parts:
            continue
        m = re.fullmatch(r"seed_(\d+)", parts[-1])
        seed = int(m[1]) if m else 42
        core = parts[:-1] if m else parts
        if len(core) == 3:  # root is itself a dataset dir, e.g. data/embeddings/matbench_phonons
            core = (root.name,) + tuple(core)
        if len(core) != 4:
            continue
        ds, arch, split, fold = core
        yield ds, arch, split, int(fold.split("_")[1]), seed, d


def evaluate_detector(score_fn, embeddings_root="data/embeddings/", output="results/detector.csv"):
    """Run ``score_fn`` on every cell under ``embeddings_root`` and write a CSV
    of AUROCs to ``output``. Returns the list of row dicts."""
    rows = []
    for ds, arch, split, fold, seed, d in find_cells(embeddings_root):
        z_tr, z_id, z_ood = (np.load(d / f"{n}.npy") for n in ("train", "id_test", "ood_test"))
        if min(z_tr.var(), z_id.var(), z_ood.var()) < 1e-10:  # degenerate (collapsed) encoder
            continue
        s = np.asarray(score_fn(z_tr, np.concatenate([z_id, z_ood])), dtype=float)
        if s.shape != (len(z_id) + len(z_ood),):
            raise ValueError(f"score_fn must return one score per test point; got shape {s.shape}")
        auroc = balanced_auroc(s[: len(z_id)], s[len(z_id):])
        rows.append(dict(dataset=ds, arch=arch, split=split, fold=fold, seed=seed,
                         n_id=len(z_id), n_ood=len(z_ood), auroc=auroc))
    if not rows:
        raise FileNotFoundError(
            f"no usable cells under {embeddings_root!r}; expected "
            "<root>/<dataset>/<arch>/<split>/fold_<f>[/seed_<s>]/{train,id_test,ood_test}.npy")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows
