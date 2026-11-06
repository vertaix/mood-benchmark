#!/usr/bin/env python3
"""TASK 2 step B: per-seed permutation test on the linear probe.

For every valid (dataset,arch,split,seed) cell: B=100 label shuffles under the
same 5-fold protocol as the canonical probe. p = (1+#{null>=real})/(B+1),
z = (real-null_mean)/null_std. Parallel across cells (inner n_jobs=1).

Writes outputs/perm_per_seed.csv.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cr_common as C

ROOT = C.ROOT
B = 100
N_WORKERS = int(os.environ.get("CR_WORKERS", "32"))


def one_cell(args):
    ds, ar, sp, seed = args
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, permutation_test_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    id_e, ood_e = C.load_idood(ds, ar, sp, seed)
    base = dict(dataset=ds, arch=ar, split=sp, seed=seed)
    if id_e is None:
        return {**base, "real_auroc": np.nan, "null_mean": np.nan,
                "null_std": np.nan, "pvalue": np.nan, "z_score": np.nan, "n_per_class": 0}
    X, y = C.balanced_xy(id_e, ood_e, seed_sub=42)
    if X is None:
        return {**base, "real_auroc": np.nan, "null_mean": np.nan,
                "null_std": np.nan, "pvalue": np.nan, "z_score": np.nan, "n_per_class": 0}
    pipe = Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                                max_iter=1000, random_state=42))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score, perm, pvalue = permutation_test_score(
        pipe, X, y, scoring="roc_auc", cv=cv, n_permutations=B, n_jobs=1, random_state=42)
    nm, nsd = float(np.mean(perm)), float(np.std(perm))
    z = (score - nm) / nsd if nsd > 0 else np.nan
    return {**base, "real_auroc": float(score), "null_mean": nm, "null_std": nsd,
            "pvalue": float(pvalue), "z_score": float(z), "n_per_class": len(y) // 2}


def main():
    det = pd.read_csv(os.path.join(C.RESULTS, "all_detectors_seeded.csv"))
    cells = list(det[["dataset", "arch", "split", "seed"]].drop_duplicates()
                 .itertuples(index=False, name=None))
    print(f"permutation test (B={B}) on {len(cells)} cells, {N_WORKERS} workers...")
    rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for i, r in enumerate(ex.map(one_cell, cells, chunksize=2)):
            rows.append(r)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(cells)}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(C.OUT, "perm_per_seed.csv"), index=False)
    print("wrote outputs/perm_per_seed.csv  shape", out.shape)
    print("NaN (degenerate) cells:", int(out.pvalue.isna().sum()))


if __name__ == "__main__":
    main()
