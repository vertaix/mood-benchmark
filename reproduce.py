#!/usr/bin/env python3
"""Regenerate the main-paper Tables 1-3 and Figures 2-5.

Default (fast, no GPU, no embeddings): start from the aggregated per-setting
CSVs released with the dataset (`per_seed_cells.csv`, `perm_per_seed.csv`),
expected in `results` (see configs/paths.yaml), and write everything to `outputs`.

    python reproduce.py                    # tables + figures
    python reproduce.py --tables           # Tables 1-3 (+ Table 4) only
    python reproduce.py --figures          # Figures 2-5 only
    python reproduce.py --from-embeddings  # recompute the two CSVs from raw
                                           # embeddings first (slow: all 3.7 GB)
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from mood import paths

SCRIPTS = Path(__file__).resolve().parent / "scripts"
AGG_INPUTS = ["per_seed_cells.csv", "perm_per_seed.csv"]


def run(script):
    print(f"\n>>> {script}", flush=True)
    subprocess.run([sys.executable, str(SCRIPTS / script)], check=True)


def stage_inputs():
    paths.outputs.mkdir(parents=True, exist_ok=True)
    for name in AGG_INPUTS:
        src = paths.results / name
        if not src.exists():
            sys.exit(f"missing {src}\nDownload the dataset (README: 'Data download') "
                     f"or set `results:` in configs/paths.yaml.")
        shutil.copy(src, paths.outputs / name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", action="store_true", help="Tables 1-3 only")
    ap.add_argument("--figures", action="store_true", help="Figures 2-5 only")
    ap.add_argument("--from-embeddings", action="store_true",
                    help="recompute per_seed_cells.csv / perm_per_seed.csv from raw embeddings (slow)")
    a = ap.parse_args()
    want_t, want_f = a.tables or not a.figures, a.figures or not a.tables

    if a.from_embeddings:
        run("cr_build_cells.py")   # detector AUROCs + probes + MAE per (setting, seed)
        run("cr_permutation.py")   # permutation test, B=100
    else:
        stage_inputs()
    run("cr_numbers.py")           # per-seed statistics -> mean +/- std (agg_*.csv)
    run("cr_perm_numbers.py")
    if want_t:
        run("cr_build_tables.py")  # Tables 1-3 (+ Table 4) -> outputs/tables/
    if want_f:
        run("cr_figures.py")       # Figures 2-5 -> outputs/figures/
    print(f"\nDone. See {paths.outputs}/tables and {paths.outputs}/figures")


if __name__ == "__main__":
    main()
