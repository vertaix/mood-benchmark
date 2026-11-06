#!/usr/bin/env python3
"""TASK 2 step A: per-(setting,seed) cell table.

For every valid (dataset,arch,split,seed) cell in all_detectors_seeded.csv:
  - 15 canonical detector AUROCs (from the CSV; verified for the 8 recovered cells),
  - linear + MLP probe AUROC recomputed via the CANONICAL pipeline (cr_common),
  - MAE_ID, MAE_OOD from stored predictions.

Writes outputs/per_seed_cells.csv. Probes parallelized across cells.
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
N_WORKERS = int(os.environ.get("CR_WORKERS", "32"))


def one_cell(args):
    ds, ar, sp, seed = args
    id_e, ood_e = C.load_idood(ds, ar, sp, seed)
    if id_e is None:
        lin = mlp = float("nan")
    else:
        lin = C.linear_probe_auroc(id_e, ood_e, seed_sub=42)
        mlp = C.mlp_probe_auroc(id_e, ood_e, seed_sub=42)
    mae_id, mae_ood = C.mae_id_ood(ds, ar, sp, seed)
    return dict(dataset=ds, arch=ar, split=sp, seed=seed,
                linear_probe=lin, mlp_probe=mlp,
                mae_id=mae_id, mae_ood=mae_ood)


def main():
    det = pd.read_csv(os.path.join(C.RESULTS, "all_detectors_seeded.csv"))
    # keep only the 15 canonical detectors, pivot to wide per (setting,seed)
    sub = det[det.detector.isin(C.UNSUP_KEYS)].copy()
    sub["cell"] = sub.dataset + "|" + sub.arch + "|" + sub.split + "|" + sub.seed.astype(str)
    wide = sub.pivot_table(index=["dataset", "arch", "split", "seed"],
                           columns="detector", values="auroc", aggfunc="first").reset_index()
    # valid cells = those present in the detector CSV
    cells = list(wide[["dataset", "arch", "split", "seed"]].itertuples(index=False, name=None))
    print(f"recomputing probes + MAE for {len(cells)} cells with {N_WORKERS} workers...")

    results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for i, r in enumerate(ex.map(one_cell, cells, chunksize=4)):
            results.append(r)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(cells)}", flush=True)
    probe = pd.DataFrame(results)

    out = wide.merge(probe, on=["dataset", "arch", "split", "seed"], how="left")
    out["mae_gap_log10"] = np.log10(out.mae_ood / out.mae_id)
    out["degraded"] = out.mae_ood > out.mae_id
    out.to_csv(os.path.join(C.OUT, "per_seed_cells.csv"), index=False)
    print("wrote outputs/per_seed_cells.csv  shape", out.shape)
    # quick sanity
    print("NaN probe cells:", int(out.linear_probe.isna().sum()),
          "| NaN mae cells:", int(out.mae_id.isna().sum()))
    print("degraded (MAE_OOD>MAE_ID) cell count:", int(out.degraded.sum()))


if __name__ == "__main__":
    main()
