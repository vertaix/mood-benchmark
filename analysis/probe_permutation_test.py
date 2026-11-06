"""Permutation test for the linear probe.

For each (dataset, arch, split) combo:
  - Real statistic: 5-fold CV mean AUROC of an L2 logistic-regression probe
    on the balanced ID/OOD test embeddings (same protocol as
    run_linear_probe.py).
  - Null distribution: same protocol, but with the labels shuffled. Repeated
    B=100 times.
  - Permutation p-value: (1 + #{null >= real}) / (B + 1).
  - z-score: (real - null_mean) / null_std.

Output:
  results/probe_permutation_test.csv
  paper/tables/table5_probe_pvalues.{csv,tex}
  figures/fig6_probe_significance.png
"""
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, permutation_test_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
from mood import paths as _P
EMB_ROOT = _P.embeddings
PROBE_ROOT = _P.linear_probes
OUT_CSV = _P.results / "probe_permutation_test.csv"
TABLE_DIR = ROOT / "paper" / "tables"
FIG_DIR = ROOT / "figures"

B = 100   # number of permutations
N_JOBS = -1


def load_balanced(ds, arch, split, seed=42):
    p = EMB_ROOT / f"matbench_{ds}" / arch / split / "fold_0"
    if not (p / "id_test.npy").exists():
        return None, None
    id_e = np.load(p / "id_test.npy"); ood_e = np.load(p / "ood_test.npy")
    if id_e.var() < 1e-10 or ood_e.var() < 1e-10:
        return None, None
    rng = np.random.RandomState(seed)
    n = min(len(id_e), len(ood_e))
    if n < 5:
        return None, None
    X = np.concatenate([id_e[rng.choice(len(id_e), n, replace=False)],
                        ood_e[rng.choice(len(ood_e), n, replace=False)]])
    y = np.concatenate([np.ones(n), np.zeros(n)])
    return X, y


def discover_combos():
    out = []
    for ds_dir in sorted(EMB_ROOT.glob("matbench_*")):
        for arch_dir in sorted(ds_dir.iterdir()):
            if not arch_dir.is_dir():
                continue
            for split_dir in sorted(arch_dir.iterdir()):
                if not split_dir.name.startswith(("comp_", "struct_")):
                    continue
                if not (split_dir / "fold_0" / "id_test.npy").exists():
                    continue
                out.append((ds_dir.name.replace("matbench_", ""),
                            arch_dir.name, split_dir.name))
    return out


def main():
    combos = discover_combos()
    print(f"running permutation test on {len(combos)} combos, B={B}")
    pipe = Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(penalty="l2", C=1.0,
                                                solver="lbfgs", max_iter=1000,
                                                random_state=42))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    t0 = time.time()
    for i, (ds, arch, split) in enumerate(combos):
        X, y = load_balanced(ds, arch, split)
        if X is None:
            rows.append({"dataset": ds, "arch": arch, "split": split,
                         "real_auroc": np.nan, "null_mean": np.nan,
                         "null_std": np.nan, "pvalue": np.nan,
                         "z_score": np.nan, "n_per_class": 0,
                         "skipped": "degenerate or too few samples"})
            continue
        score, perm_scores, pvalue = permutation_test_score(
            pipe, X, y, scoring="roc_auc", cv=cv,
            n_permutations=B, n_jobs=N_JOBS, random_state=42)
        null_mean = float(np.mean(perm_scores))
        null_std = float(np.std(perm_scores))
        z = (score - null_mean) / null_std if null_std > 0 else np.nan
        rows.append({"dataset": ds, "arch": arch, "split": split,
                     "real_auroc": float(score), "null_mean": null_mean,
                     "null_std": null_std, "pvalue": float(pvalue),
                     "z_score": float(z), "n_per_class": len(y) // 2,
                     "skipped": ""})
        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(combos) - i - 1) / rate
            print(f"  [{i+1:>3}/{len(combos)}] {ds}/{arch}/{split:<22} "
                  f"real={score:.3f} null={null_mean:.3f} p={pvalue:.3f} z={z:+.2f}  "
                  f"({rate:.1f} combos/s, ETA {eta/60:.1f} min)", flush=True)
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, float_format="%.4f")
    print(f"\nwrote {OUT_CSV}  ({len(df)} rows)")
    return df


def aggregate_and_plot(df):
    df = df.dropna(subset=["pvalue"]).copy()
    df["sig_05"] = (df["pvalue"] < 0.05).astype(int)
    df["sig_01"] = (df["pvalue"] < 0.01).astype(int)

    print("\n=== Per-dataset significance ===")
    g = df.groupby("dataset").agg(
        n=("real_auroc", "size"),
        mean_real=("real_auroc", "mean"),
        mean_null=("null_mean", "mean"),
        mean_z=("z_score", "mean"),
        frac_p05=("sig_05", "mean"),
        frac_p01=("sig_01", "mean"),
    ).round(3)
    print(g.to_string())

    # Table 5: per-dataset summary
    out_csv = TABLE_DIR / "table5_probe_pvalues.csv"
    out_tex = TABLE_DIR / "table5_probe_pvalues.tex"
    g.to_csv(out_csv, float_format="%.3f")
    print(f"\nwrote {out_csv}")
    lines = [
        r"\begin{table}[h]\centering",
        r"\caption{Permutation test on the linear probe ($B{=}100$ label "
        r"shuffles per combo, same 5-fold CV protocol). $z$-score is "
        r"$(AUROC_{real}-\overline{AUROC_{null}})/\sigma_{null}$. "
        r"Frac($p<0.05$): fraction of (arch $\times$ split) combos in the "
        r"dataset that reject the null at the 0.05 level.}",
        r"\label{tab:probe_significance}",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"dataset & $n$ & real AUROC & null AUROC & mean $z$ & frac($p{<}0.05$) & frac($p{<}0.01$) \\",
        r"\midrule",
    ]
    for ds, r in g.iterrows():
        lines.append(f"{ds} & {int(r['n'])} & {r['mean_real']:.3f} & "
                     f"{r['mean_null']:.3f} & {r['mean_z']:+.2f} & "
                     f"{r['frac_p05']:.2f} & {r['frac_p01']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out_tex.write_text("\n".join(lines))
    print(f"wrote {out_tex}")

    # Fig 6: significance overview
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: scatter of real AUROC vs z-score, colored by dataset
    ax = axes[0]
    cmap = plt.get_cmap("tab10")
    for i, ds in enumerate(sorted(df["dataset"].unique())):
        sub = df[df["dataset"] == ds]
        ax.scatter(sub["real_auroc"], sub["z_score"], s=28,
                   color=cmap(i), label=ds, alpha=0.8,
                   edgecolor="black", lw=0.3)
    ax.axhline(1.96, color="red", ls="--", lw=0.8, alpha=0.6, label="$z=1.96$ (p<0.05)")
    ax.axhline(2.58, color="darkred", ls=":", lw=0.8, alpha=0.6, label="$z=2.58$ (p<0.01)")
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel("Real probe AUROC")
    ax.set_ylabel("$z$-score = (real $-$ null) / null$\\,\\sigma$")
    ax.set_title("Effect size vs. statistical significance")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")

    # Panel B: p-value histogram
    ax = axes[1]
    ax.hist(df["pvalue"], bins=20, color="#888", edgecolor="black", alpha=0.85)
    ax.axvline(0.05, color="red", ls="--", lw=1.0, label="p=0.05")
    ax.axvline(0.01, color="darkred", ls=":", lw=1.0, label="p=0.01")
    ax.set_xlabel("permutation p-value")
    ax.set_ylabel("count of combos")
    ax.set_title(f"p-value distribution\n"
                 f"{(df['pvalue']<0.05).sum()}/{len(df)} combos significant at p<0.05  |  "
                 f"{(df['pvalue']<0.01).sum()}/{len(df)} at p<0.01")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Panel C: per-dataset bar chart of fraction significant
    ax = axes[2]
    datasets = list(g.index)
    x = np.arange(len(datasets))
    ax.bar(x - 0.2, g["frac_p05"], width=0.4, color="#1f77b4",
           label="p < 0.05", alpha=0.85, edgecolor="black", lw=0.5)
    ax.bar(x + 0.2, g["frac_p01"], width=0.4, color="#d62728",
           label="p < 0.01", alpha=0.85, edgecolor="black", lw=0.5)
    for xi, ds in zip(x, datasets):
        ax.text(xi - 0.2, g["frac_p05"][ds] + 0.02,
                f"{g['frac_p05'][ds]:.2f}", ha="center", fontsize=8)
        ax.text(xi + 0.2, g["frac_p01"][ds] + 0.02,
                f"{g['frac_p01'][ds]:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(datasets, rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("fraction of combos with significant probe")
    ax.set_title("Per-dataset rejection rate")
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=9)

    fig.suptitle("Fig. 6 — Linear-probe permutation test "
                 "($B{=}100$ shuffles per combo, 5-fold CV)", y=1.02, fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "fig6_probe_significance.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    if OUT_CSV.exists():
        print(f"resuming from {OUT_CSV} (delete to recompute)")
        df = pd.read_csv(OUT_CSV)
    else:
        df = main()
    aggregate_and_plot(df)
