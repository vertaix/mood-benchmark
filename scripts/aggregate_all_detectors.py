"""Comprehensive collection: every (dataset, arch, split) x detector AUROC.

Sources:
  - results/<ds>/<arch>/<split>/fold_0/auroc.json   — one-class detectors
  - flow_models/<ds>/<arch>/<split>/fold_0/seed_42/results.json  — flow
  - linear_probes/.../results.json                  — supervised linear probe
  - linear_probes_mlp/.../results.json              — supervised MLP probe

Outputs:
  results/all_detectors.csv          — long form, one row per (cell, detector)
  paper/tables/table_all_detectors_summary.{csv,tex}
                                       — wide summary, one row per detector
  paper/figures/fig_all_detectors_summary.png
                                       — sorted bar chart of mean AUROC
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
from mood import paths as _P
RESULTS = _P.results
FLOW = _P.flow_models
PROBE = _P.linear_probes
PROBE_MLP = _P.linear_probes_mlp
EMB = _P.embeddings
TABLE_DIR = ROOT / "paper" / "tables"
FIG_DIR = ROOT / "paper" / "figures"

# (display, family, supervision, getter-from-auroc.json or special tag)
ONECLASS = [
    ("MDS-legacy",     "density",   "mahalanobis"),
    ("MDS-BIC",        "density",   "mahalanobis_bic"),
    ("MDS-AIC",        "density",   "mahalanobis_aic"),
    ("KDE",            "density",   "kde"),
    ("kPCA",           "density",   "kpca"),
    ("KNN-legacy",     "distance",  "knn"),
    ("KNN k=5 norm",   "distance",  "knn_sweep:k5_norm1"),
    ("KNN k=10 norm",  "distance",  "knn_sweep:k10_norm1"),
    ("KNN k=20 norm",  "distance",  "knn_sweep:k20_norm1"),
    ("KNN k=50 norm",  "distance",  "knn_sweep:k50_norm1"),
    ("KNN k=5 raw",    "distance",  "knn_sweep:k5_norm0"),
    ("KNN k=10 raw",   "distance",  "knn_sweep:k10_norm0"),
    ("KNN k=20 raw",   "distance",  "knn_sweep:k20_norm0"),
    ("KNN k=50 raw",   "distance",  "knn_sweep:k50_norm0"),
    ("Cosine",         "distance",  "cosine"),
    ("Norm",           "distance",  "norm"),
    ("ViM",            "subspace",  "vim"),
    ("LOF",            "local",     "lof"),
    ("OCSVM",          "boundary",  "ocsvm"),
    ("IForest",        "isolation", "iforest"),
    ("ADERH",          "isolation", "aderh"),
    ("ReAct + MDS",    "rectified", "react_mds"),
    ("ASH + KNN",      "rectified", "ash_knn"),
]


def lookup(combo, key):
    if key.startswith("knn_sweep:"):
        sub = key.split(":", 1)[1]
        e = combo.get("knn_sweep", {}).get(sub, {})
    else:
        e = combo.get(key, {})
    if isinstance(e, dict) and "auroc" in e:
        a = e.get("auroc")
        return float(a) if a is not None and not (isinstance(a, float) and np.isnan(a)) else None
    return None


def is_degenerate(ds, arch, sp, seed=42):
    """Check if embeddings have zero variance (collapsed encoder).

    Returns True only if embeddings exist AND are degenerate.
    Missing embeddings are NOT degenerate — the auroc.json may have been
    computed from embeddings on a different filesystem (e.g. scratch)."""
    base = EMB / f"matbench_{ds}" / arch / sp / "fold_0"
    if seed != 42:
        base = base / f"seed_{seed}"
    for n in ["id_test.npy", "ood_test.npy"]:
        p = base / n
        if not p.exists():
            continue  # missing ≠ degenerate
        if np.load(p).var() < 1e-10:
            return True
    return False


def gather_long():
    """Original single-seed (seed-42 only) gather for backward compat."""
    rows = []
    for r in sorted(RESULTS.glob("matbench_*/*/*/fold_0/auroc.json")):
        parts = r.parts
        ds = parts[-5].replace("matbench_", "")
        arch, sp = parts[-4], parts[-3]
        if is_degenerate(ds, arch, sp, seed=42):
            continue
        combo = json.loads(r.read_text())
        for label, fam, key in ONECLASS:
            v = lookup(combo, key)
            if v is None:
                continue
            rows.append({
                "dataset": ds, "arch": arch, "split": sp,
                "detector": label, "family": fam,
                "supervision": "one-class", "auroc": v,
            })
        fp = FLOW / f"matbench_{ds}" / arch / sp / "fold_0" / "seed_42" / "results.json"
        if fp.exists():
            d = json.loads(fp.read_text())
            v = d.get("auroc")
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                rows.append({
                    "dataset": ds, "arch": arch, "split": sp,
                    "detector": "Flow (NF)", "family": "density",
                    "supervision": "one-class", "auroc": float(v),
                })
        for path, label in [(PROBE, "Linear probe"), (PROBE_MLP, "MLP probe")]:
            pp = path / f"matbench_{ds}" / arch / sp / "fold_0" / "results.json"
            if pp.exists():
                d = json.loads(pp.read_text())
                v = d.get("mean_auroc")
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    rows.append({
                        "dataset": ds, "arch": arch, "split": sp,
                        "detector": label, "family": "supervised",
                        "supervision": "supervised", "auroc": float(v),
                    })
    return pd.DataFrame(rows)


def gather_long_seeded():
    """Multi-seed gather: seed 42 + seeds 123/456.

    Seed 42: uses the committed all_detectors.csv as the authoritative source,
    since some cells' auroc.json lived on scratch and were never committed to the
    repo (gitignored).  This gives the full 257 valid seed-42 cells (264 minus
    the 7 degenerate phonons/cgcnn).

    Seeds 123, 456: from fold_0/seed_<s>/auroc.json on disk.

    Returns long-form DataFrame with a 'seed' column.
    Only includes one-class detectors (no supervised probes)."""
    rows = []

    # Seed 42: read from the committed all_detectors.csv (authoritative source).
    # Some cells' auroc.json lived on scratch and were never committed (gitignored),
    # but the CSV was committed and is the canonical seed-42 record.
    # Filter to physics splits only (Omee splits were not reseeded).
    csv_path = _P.results / "all_detectors.csv"
    if csv_path.exists():
        s42 = pd.read_csv(csv_path)
        s42 = s42[s42["supervision"] == "one-class"].copy()
        s42 = s42[s42["split"].str.startswith(("comp_", "struct_"))]
        s42["seed"] = 42
        rows.append(s42)

    # Seeds 123, 456: fold_0/seed_<s>/auroc.json
    seed_rows = []
    for r in sorted(RESULTS.glob("matbench_*/*/*/fold_0/seed_*/auroc.json")):
        parts = r.parts
        seed = int(parts[-2].split("_")[1])
        if seed == 42:
            continue
        ds = parts[-6].replace("matbench_", "")
        arch, sp = parts[-5], parts[-4]
        if is_degenerate(ds, arch, sp, seed=seed):
            continue
        _emit_detectors(seed_rows, r, ds, arch, sp, seed=seed)

    if seed_rows:
        rows.append(pd.DataFrame(seed_rows))

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _emit_detectors(rows, auroc_path, ds, arch, sp, seed):
    """Emit one-class detector rows from a single auroc.json."""
    combo = json.loads(auroc_path.read_text())
    for label, fam, key in ONECLASS:
        v = lookup(combo, key)
        if v is None:
            continue
        rows.append({
            "dataset": ds, "arch": arch, "split": sp, "seed": seed,
            "detector": label, "family": fam,
            "supervision": "one-class", "auroc": v,
        })
    # Flow
    fp = FLOW / f"matbench_{ds}" / arch / sp / "fold_0" / f"seed_{seed}" / "results.json"
    if fp.exists():
        d = json.loads(fp.read_text())
        v = d.get("auroc")
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            rows.append({
                "dataset": ds, "arch": arch, "split": sp, "seed": seed,
                "detector": "Flow (NF)", "family": "density",
                "supervision": "one-class", "auroc": float(v),
            })


def make_summary(long_df):
    summary = (long_df.groupby(["detector", "family", "supervision"])
                       .agg(n=("auroc", "size"),
                            mean_auroc=("auroc", "mean"),
                            std_auroc=("auroc", "std"),
                            median_auroc=("auroc", "median"))
                       .reset_index())
    summary["rank_within_supervision"] = (
        summary.groupby("supervision")["mean_auroc"].rank(ascending=False, method="min").astype(int))
    summary = summary.sort_values(["supervision", "mean_auroc"],
                                   ascending=[True, False]).reset_index(drop=True)
    return summary


def make_table(summary):
    out_csv = TABLE_DIR / "table_all_detectors_summary.csv"
    summary.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"  wrote {out_csv}")

    lines = [
        r"\begin{table}[h]\centering",
        r"\caption{All detectors compared across the benchmark. One-class "
        r"detectors are deployable; supervised probes (linear, MLP) see OOD "
        r"labels and are diagnostic only.}",
        r"\label{tab:all_detectors}",
        r"\small",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"detector & family & $n$ & mean AUROC & std & median \\",
        r"\midrule",
    ]
    last_super = None
    for _, r in summary.iterrows():
        if r["supervision"] != last_super:
            if last_super is not None: lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{r['supervision']} detectors}}}} \\")
            lines.append(r"\midrule")
            last_super = r["supervision"]
        lines.append(f"{r['detector']} & {r['family']} & {int(r['n'])} & "
                     f"{r['mean_auroc']:.3f} & {r['std_auroc']:.3f} & {r['median_auroc']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out_tex = TABLE_DIR / "table_all_detectors_summary.tex"
    out_tex.write_text("\n".join(lines))
    print(f"  wrote {out_tex}")


def make_figure(summary):
    fam_color = {"density": "#1f77b4", "distance": "#d62728",
                 "boundary": "#2ca02c", "isolation": "#9467bd",
                 "subspace": "#bcbd22", "rectified": "#17becf",
                 "local": "#ff7f0e", "supervised": "#666666"}
    fig, axes = plt.subplots(2, 1, figsize=(13, 9),
                              gridspec_kw={"height_ratios": [4, 1]})
    for ax, super_ in zip(axes, ["one-class", "supervised"]):
        sub = summary[summary["supervision"] == super_]
        x = np.arange(len(sub))
        bars = ax.bar(x, sub["mean_auroc"], yerr=sub["std_auroc"],
                      color=[fam_color.get(f, "#888") for f in sub["family"]],
                      edgecolor="black", lw=0.4, alpha=0.85,
                      error_kw=dict(ecolor="black", elinewidth=0.7, capsize=3))
        for i, (m, s) in enumerate(zip(sub["mean_auroc"], sub["std_auroc"])):
            ax.text(i, m + s + 0.01, f"{m:.3f}", ha="center", fontsize=7)
        ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["detector"], rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("mean AUROC")
        ax.set_title(f"{super_} detectors  (sorted by mean AUROC)", fontsize=10)
        ax.set_ylim(0.40, 1.05 if super_ == "supervised" else 0.75)
        ax.grid(axis="y", alpha=0.3)
        if super_ == "one-class":
            handles = [plt.Rectangle((0, 0), 1, 1, color=c) for f, c in fam_color.items() if f != "supervised"]
            labels = [f for f in fam_color if f != "supervised"]
            ax.legend(handles, labels, loc="upper right", fontsize=8, ncol=4)
    fig.suptitle(f"All detectors x cells summary  "
                 f"(whiskers = $\\pm$1 std across cells)", fontsize=12, y=1.01)
    fig.tight_layout()
    out = FIG_DIR / "fig_all_detectors_summary.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    # NOTE: all_detectors.csv is the committed seed-42 record. Do NOT overwrite it
    # from gather_long() — some cells' auroc.json only existed on scratch.
    # The seeded gather reads FROM it as seed-42 source of truth.
    print("=== seed-42 baseline (committed all_detectors.csv) ===")
    csv_path = _P.results / "all_detectors.csv"
    long_df = pd.read_csv(csv_path)
    print(f"  {len(long_df)} rows  ({long_df['detector'].nunique()} detectors, "
          f"{long_df.groupby(['dataset','arch','split']).ngroups} cells)")

    print("\n=== summary table ===")
    summary = make_summary(long_df)
    print(summary.round(4).to_string(index=False))
    make_table(summary)
    print("\n=== figure ===")
    make_figure(summary)

    # ── Multi-seed aggregation ──────────────────────────────────────────
    print("\n=== gather seeded (cell × seed × detector) long ===")
    seeded_df = gather_long_seeded()
    n_cell_seeds = seeded_df.groupby(["dataset", "arch", "split", "seed"]).ngroups
    print(f"  {len(seeded_df)} rows  ({seeded_df['detector'].nunique()} detectors, "
          f"{n_cell_seeds} cell-seeds)")

    out_seeded = _P.results / "all_detectors_seeded.csv"
    seeded_df.to_csv(out_seeded, index=False, float_format="%.4f")
    print(f"  wrote {out_seeded}")

    # Summary: mean / std / n_seeds per (dataset, arch, split, detector)
    summary_seeded = (
        seeded_df.groupby(["dataset", "arch", "split", "detector", "family"])
        .agg(
            mean_auroc=("auroc", "mean"),
            std_auroc=("auroc", "std"),
            n_seeds=("auroc", "size"),
        )
        .reset_index()
    )
    out_summary = _P.results / "all_detectors_summary.csv"
    summary_seeded.to_csv(out_summary, index=False, float_format="%.4f")
    print(f"  wrote {out_summary}")

    # Report cells where n_seeds != 3
    cell_seeds = seeded_df.groupby(["dataset", "arch", "split"])["seed"].nunique()
    short = cell_seeds[cell_seeds != 3]
    print(f"\n  Cells with n_seeds != 3: {len(short)}")
    for (ds, arch, sp), ns in short.items():
        print(f"    {ds}/{arch}/{sp}: {ns} seeds")


if __name__ == "__main__":
    main()
