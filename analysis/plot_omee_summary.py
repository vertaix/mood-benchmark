"""Plot held-out ID vs OOD MAE for OMEE splits (mean +/- sd over 50 folds)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mood import paths as _P
SRC_CSV = _P.results / "omee_splits_summary.csv"
OUT_FIG = _P.outputs / "figures" / "fig_mae_id_vs_ood_omee.png"

SPLIT_ORDER = ["LOCO", "SparseXcluster", "SparseYcluster"]
ARCH_ORDER = ["alignn", "cgcnn"]
DATASET_ORDER = ["dielectric", "log_gvrh"]


def main():
    agg = pd.read_csv(SRC_CSV).rename(columns={
        "id_mae_mean": "id_mean", "id_mae_std": "id_sd",
        "ood_mae_mean": "ood_mean", "ood_mae_std": "ood_sd",
    })
    agg["split"] = pd.Categorical(agg["split"], categories=SPLIT_ORDER, ordered=True)
    agg["architecture"] = pd.Categorical(agg["architecture"], categories=ARCH_ORDER, ordered=True)
    agg["dataset"] = pd.Categorical(agg["dataset"], categories=DATASET_ORDER, ordered=True)
    agg = agg.sort_values(["dataset", "architecture", "split"]).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    bar_w = 0.38

    for ax, dset in zip(axes, DATASET_ORDER):
        sub = agg[agg["dataset"] == dset].reset_index(drop=True)
        n = len(sub)
        x = np.arange(n)

        ax.bar(x - bar_w/2, sub["id_mean"],  yerr=sub["id_sd"],  width=bar_w,
               color="#4c9aff", label="held-out ID", capsize=6,
               error_kw={"lw": 1.6, "ecolor": "black", "capthick": 1.6})
        ax.bar(x + bar_w/2, sub["ood_mean"], yerr=sub["ood_sd"], width=bar_w,
               color="#e8553c", label="held-out OOD", capsize=6,
               error_kw={"lw": 1.6, "ecolor": "black", "capthick": 1.6})

        labels = [f"{a}\n{s}" for a, s in zip(sub["architecture"], sub["split"])]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.axvline(2.5, color="black", lw=0.8, alpha=0.4)
        ymax = (sub[["id_mean", "ood_mean"]].to_numpy().max()
                + sub[["id_sd", "ood_sd"]].to_numpy().max()) * 1.15
        ax.set_ylim(0, ymax)
        ax.text(1.0, ymax * 0.97, "alignn", ha="center", va="top",
                fontsize=9, color="dimgray", fontweight="bold")
        ax.text(4.0, ymax * 0.97, "cgcnn", ha="center", va="top",
                fontsize=9, color="dimgray", fontweight="bold")
        ax.set_title(dset)
        ax.set_ylabel("MAE  (mean ± sd across 50 folds)")
        ax.grid(axis="y", alpha=0.3)

    axes[1].legend(loc="upper right", frameon=True)
    fig.suptitle("Held-out ID vs held-out OOD MAE  (OMEE splits: LOCO, SparseX, SparseY)",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
