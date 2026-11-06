"""
magpie_analysis.py
==================

Compute composition-only Magpie shift between ID and OOD pools for every
(dataset, architecture, split) setting, and produce the scatter of linear-probe
AUROC vs. Magpie shift used in Appendix Figure `magpie_vs_probe`.

Magpie shift definition
-----------------------
Let X_id, X_ood be the matminer ElementProperty (preset='magpie') fingerprints
for the *balanced* ID and OOD test pools used in the AUROC evaluation.
Standardize per-feature on the ID pool (mean mu, std sigma + 1e-8). The shift is:

    shift = || mean_standardized(X_id) - mean_standardized(X_ood) ||_2

This is composition-only: no structural or learned information enters.

Inputs (expected on disk)
-------------------------
- A per-setting metadata CSV with columns:
    dataset, architecture, split, id_indices_path, ood_indices_path,
    linear_probe_auroc
  These are produced by the existing MOOD evaluation pipeline; the indices
  point into the original MatBench pymatgen Structure list for `dataset`.
- The MatBench structures themselves, loaded via matminer's MatbenchBenchmark
  or a cached pickle of compositions per dataset.

Output
------
- magpie_shift_per_setting.csv with one row per setting, adding `magpie_shift`.
- outputs/figures/magpie_vs_probe.png: faceted scatter, color by dataset.

Dependencies: matminer, pymatgen, scikit-learn, pandas, matplotlib, numpy.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Structure
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mood import paths as _P

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("magpie_analysis")


# ---------------------------------------------------------------------------
# Featurization
# ---------------------------------------------------------------------------

def build_featurizer() -> ElementProperty:
    """Magpie preset: 22 elemental properties x {mean, std, min, max, range} = 132 dims."""
    f = ElementProperty.from_preset("magpie")
    return f


def featurize_compositions(compositions, featurizer: ElementProperty) -> np.ndarray:
    """
    Featurize a list of pymatgen Composition objects.

    Returns a (N, 132) float array. Rows that fail featurization (e.g., elements
    Magpie does not cover) are dropped with a warning; this is rare on MatBench.
    """
    rows = []
    failed = 0
    for comp in compositions:
        try:
            rows.append(featurizer.featurize(comp))
        except Exception as e:  # noqa: BLE001
            failed += 1
            log.debug("Featurization failed for %s: %s", comp, e)
    if failed:
        log.warning("Magpie featurization failed on %d / %d compositions",
                    failed, len(compositions))
    return np.asarray(rows, dtype=float)


# ---------------------------------------------------------------------------
# Magpie shift
# ---------------------------------------------------------------------------

def magpie_shift(X_id: np.ndarray, X_ood: np.ndarray,
                 sigma_floor: float = 1e-2) -> float:
    """
    L2 distance between mean standardized fingerprints.

    Standardization is fit on the ID pool only (per the appendix definition).
    Magpie includes a number of features that can be near-constant on a small
    ID pool — e.g. ``MinimumAtomicNumber`` for an oxide-only chalcogenide
    split — and their standardized OOD values blow up if we just divide by
    sigma + tiny epsilon. We therefore clip very small sigmas (< 1e-2 in raw
    feature units) to 1.0, so a feature with no ID-pool variance contributes
    as a raw mean difference rather than a divide-by-near-zero amplification.
    """
    mu = X_id.mean(axis=0)
    sigma = X_id.std(axis=0)
    sigma = np.where(sigma < sigma_floor, 1.0, sigma)
    z_id = (X_id - mu) / sigma
    z_ood = (X_ood - mu) / sigma
    return float(np.linalg.norm(z_id.mean(axis=0) - z_ood.mean(axis=0)))


# ---------------------------------------------------------------------------
# Per-setting driver
# ---------------------------------------------------------------------------

def compute_for_setting(
    structures: list[Structure],
    id_idx: np.ndarray,
    ood_idx: np.ndarray,
    featurizer: ElementProperty,
) -> float:
    comps_id = [structures[i].composition for i in id_idx]
    comps_ood = [structures[i].composition for i in ood_idx]
    X_id = featurize_compositions(comps_id, featurizer)
    X_ood = featurize_compositions(comps_ood, featurizer)
    if X_id.size == 0 or X_ood.size == 0:
        return float("nan")
    return magpie_shift(X_id, X_ood)


def run(metadata_csv: Path, structures_dir: Path, out_csv: Path) -> pd.DataFrame:
    """
    metadata_csv: one row per setting, with index files and probe AUROC.
    structures_dir: per-dataset cached pickle of List[Structure].
    """
    df = pd.read_csv(metadata_csv)
    featurizer = build_featurizer()

    # Cache structures per dataset to avoid reloading.
    struct_cache: dict[str, list[Structure]] = {}

    shifts = []
    for _, row in df.iterrows():
        ds = row["dataset"]
        if ds not in struct_cache:
            path = structures_dir / f"{ds}.pkl"
            log.info("Loading structures: %s", path)
            struct_cache[ds] = pd.read_pickle(path)

        id_idx = np.load(row["id_indices_path"])
        ood_idx = np.load(row["ood_indices_path"])
        try:
            s = compute_for_setting(struct_cache[ds], id_idx, ood_idx, featurizer)
        except Exception as e:  # noqa: BLE001
            log.warning("Setting failed (%s, %s, %s): %s",
                        ds, row.get("architecture"), row.get("split"), e)
            s = float("nan")
        shifts.append(s)

    df["magpie_shift"] = shifts
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    log.info("Wrote %s", out_csv)
    return df


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_probe_vs_shift(df: pd.DataFrame, out_path: Path) -> None:
    """Faceted scatter of linear-probe AUROC vs. Magpie shift, color by dataset."""
    df = df.dropna(subset=["magpie_shift", "linear_probe_auroc"]).copy()

    method_bn = {"dielectric", "log_gvrh"}
    df["regime"] = df["dataset"].apply(
        lambda d: "method-bottleneck" if d in method_bn else "encoder-bottleneck"
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    palette = {
        "dielectric": "#1f77b4",
        "log_gvrh":   "#2ca02c",
        "log_kvrh":   "#d62728",
        "perovskites": "#9467bd",
        "phonons":    "#ff7f0e",
    }

    for ax, regime in zip(axes, ["method-bottleneck", "encoder-bottleneck"]):
        sub = df[df["regime"] == regime]
        for ds, g in sub.groupby("dataset"):
            ax.scatter(
                g["magpie_shift"], g["linear_probe_auroc"],
                s=22, alpha=0.6, label=ds, color=palette.get(ds, "grey"),
            )

        # OLS line
        if len(sub) >= 5:
            x = sub["magpie_shift"].to_numpy()
            y = sub["linear_probe_auroc"].to_numpy()
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, slope * xs + intercept, color="black", lw=1.2,
                    label=f"OLS slope = {slope:.3f}")

        ax.axhline(0.5, color="grey", lw=0.8, ls="--")
        ax.set_xlabel("Magpie shift  (||mean ID - mean OOD||₂, standardized)")
        ax.legend(fontsize=8, loc="best")

    axes[0].set_ylabel("Linear probe AUROC")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    log.info("Wrote %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata-csv", type=Path, required=True,
                    help="Per-setting CSV with id_indices_path, ood_indices_path, "
                         "linear_probe_auroc columns.")
    ap.add_argument("--structures-dir", type=Path, required=True,
                    help="Directory with {dataset}.pkl files of List[Structure].")
    ap.add_argument("--out-csv", type=Path,
                    default=_P.outputs / "magpie_shift_per_setting.csv")
    ap.add_argument("--out-fig", type=Path,
                    default=_P.outputs / "figures" / "magpie_vs_probe.png")
    args = ap.parse_args()

    df = run(args.metadata_csv, args.structures_dir, args.out_csv)
    plot_probe_vs_shift(df, args.out_fig)


if __name__ == "__main__":
    main()
