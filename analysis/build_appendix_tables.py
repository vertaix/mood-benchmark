"""Build appendix tables for the MOOD paper.

Produces four tables in outputs/appendix_tables/:
  appendix_per_arch_summary.{csv,tex}   — Table 1 broken down by architecture
                                          (supports the 'invariant to arch' claim)
  appendix_per_split_type.{csv,tex}     — comp vs struct breakdown of detector
                                          performance (E.5 robustness)
  appendix_dataset_stats.{csv,tex}      — per-dataset setting counts, MatBench size,
                                          mean ID/OOD test size (§C)
  appendix_id_mae_per_arch.{csv,tex}    — per architecture × dataset, mean ID
                                          MAE (sanity check for §D)

Sources:
  results/all_detectors.csv           — per-setting × per-detector AUROC
  results/physics_splits_summary.csv  — per (arch, dataset, split) ID/OOD MAE,
                                        train/test sizes

Usage: python3 scripts/build_appendix_tables.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mood import paths as _P
RESULTS = _P.results
TEX_DIR = _P.outputs / "appendix_tables"      # \input'd by appendix.tex
CSV_DIR = _P.outputs / "appendix_other"       # data sources
TEX_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

# Same 15 unsupervised detectors used in the main paper.
UNSUP_DETECTORS = [
    "KNN k=20 norm", "KNN k=50 norm", "Norm", "Cosine", "LOF", "ASH + KNN",
    "MDS-BIC", "KDE", "Flow (NF)", "kPCA", "ReAct + MDS",
    "IForest", "ADERH",
    "OCSVM", "ViM",
]
DISPLAY = {
    "KNN k=20 norm": "KNN-k20", "KNN k=50 norm": "KNN-k50",
    "Norm": "Norm", "Cosine": "Cosine", "LOF": "LOF",
    "ASH + KNN": "ASH+KNN", "MDS-BIC": "MDS-BIC", "KDE": "KDE",
    "Flow (NF)": "Flow", "kPCA": "kPCA", "ReAct + MDS": "ReAct+MDS",
    "IForest": "IForest", "ADERH": "ADERH",
    "OCSVM": "OCSVM", "ViM": "ViM",
}
ARCHS = ["alignn", "cgcnn", "schnet", "dimenet"]
DATASETS = ["dielectric", "log_gvrh", "log_kvrh", "perovskites", "phonons"]
DATASETS_TEX = [d.replace("_", r"\_") for d in DATASETS]   # safe in tabular cells


# ─── Column-wise highlighting (best / 2nd-best) ──────────────────────────────
HEX_GREEN = "1E8449"   # column-wise max (or min when lower-is-better)
HEX_BLUE  = "2874A6"   # column-wise 2nd best


def _ranks(values, mode="max"):
    """Return rank list: 0 = best, 1 = 2nd, -1 = other / NaN."""
    indexed = [(i, v) for i, v in enumerate(values) if v == v]
    indexed.sort(key=lambda x: -x[1] if mode == "max" else x[1])
    rank = [-1] * len(values)
    for r, (i, _) in enumerate(indexed[:2]):
        rank[i] = r
    return rank


def _fmt_cell(value, rank, fmt=".3f", missing="—"):
    """Format a numeric tabular cell, colouring rank 0 (green-bold) and 1 (blue)."""
    if value != value:  # NaN
        return missing
    s = f"{value:{fmt}}"
    if rank == 0:
        return r"\textcolor[HTML]{" + HEX_GREEN + r"}{\textbf{" + s + "}}"
    if rank == 1:
        return r"\textcolor[HTML]{" + HEX_BLUE + r"}{" + s + "}"
    return s


COLOR_CODING_NOTE = (
    r"\textcolor[HTML]{" + HEX_GREEN + r"}{\textbf{Bold green}} marks the "
    r"column-wise maximum among unsupervised detectors; "
    r"\textcolor[HTML]{" + HEX_BLUE + r"}{blue} marks the second-best."
)
COLOR_CODING_NOTE_MIN = (
    r"\textcolor[HTML]{" + HEX_GREEN + r"}{\textbf{Bold green}} marks the "
    r"column-wise minimum (lower is better); "
    r"\textcolor[HTML]{" + HEX_BLUE + r"}{blue} marks the second-best."
)


# ─── Table A: per-architecture detector summary ───────────────────────────────
def per_arch_summary():
    df = pd.read_csv(RESULTS / "all_detectors.csv")
    sub = df[df["detector"].isin(UNSUP_DETECTORS)].copy()
    sub["display"] = sub["detector"].map(DISPLAY)

    rows = []
    for det in [DISPLAY[k] for k in UNSUP_DETECTORS]:
        row = {"detector": det}
        for arch in ARCHS:
            vals = sub.loc[(sub["display"] == det) & (sub["arch"] == arch),
                           "auroc"].values
            row[f"{arch}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
            row[f"{arch}_n"] = int(len(vals))
        rows.append(row)
    out = pd.DataFrame(rows)

    csv = CSV_DIR / "appendix_per_arch_summary.csv"
    disp = out.copy()
    for arch in ARCHS:
        disp[f"{arch}_mean"] = disp[f"{arch}_mean"].round(3)
    disp.to_csv(csv, index=False)
    print(f"wrote {csv}")

    # Per-column ranks (max=best across the 15 unsupervised detectors).
    col_ranks = {arch: _ranks(out[f"{arch}_mean"].tolist(), mode="max")
                 for arch in ARCHS}

    lines = [
        r"% Requires: \usepackage{booktabs} \usepackage[table]{xcolor}",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{\textbf{Per-architecture detector summary.} Mean AUROC for "
        r"each unsupervised detector under each GNN backbone, computed across "
        r"all evaluation settings available for that (arch, detector) pair. "
        r"Bottleneck patterns are stable across architectures: no detector "
        r"shifts above 0.55 mean AUROC for any architecture, supporting the "
        r"main paper's `invariant to architecture' claim. " + COLOR_CODING_NOTE + "}",
        r"\label{tab:per_arch_summary}",
        r"\begin{tabular}{l" + "r" * len(ARCHS) + r"}",
        r"\toprule",
        r"Detector & " + " & ".join(a.upper() if a in ("knn",) else a.capitalize()
                                     for a in ARCHS) + r" \\",
        r"\midrule",
    ]
    for i, (_, row) in enumerate(out.iterrows()):
        cells = [row["detector"].replace("_", r"\_")]
        for arch in ARCHS:
            cells.append(_fmt_cell(row[f"{arch}_mean"], col_ranks[arch][i]))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    # Old standalone-arch table — superseded by per_arch_split_combined; archive.
    tex = CSV_DIR / "appendix_per_arch_summary.tex"
    tex.write_text("\n".join(lines) + "\n")
    print(f"wrote {tex}")


# ─── Table B: per split type (compositional vs structural) ────────────────────
def per_split_type():
    df = pd.read_csv(RESULTS / "all_detectors.csv")
    sub = df[df["detector"].isin(UNSUP_DETECTORS)].copy()
    sub["display"] = sub["detector"].map(DISPLAY)
    sub["split_type"] = sub["split"].apply(
        lambda s: "Compositional" if s.startswith("comp_")
        else ("Structural" if s.startswith("struct_") else "Other"))

    rows = []
    for det in [DISPLAY[k] for k in UNSUP_DETECTORS]:
        row = {"detector": det}
        for stype in ("Compositional", "Structural"):
            vals = sub.loc[(sub["display"] == det)
                           & (sub["split_type"] == stype), "auroc"].values
            row[f"{stype}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
            row[f"{stype}_n"] = int(len(vals))
        rows.append(row)
    out = pd.DataFrame(rows)

    csv = CSV_DIR / "appendix_per_split_type.csv"
    disp = out.copy()
    for s in ("Compositional", "Structural"):
        disp[f"{s}_mean"] = disp[f"{s}_mean"].round(3)
    disp.to_csv(csv, index=False)
    print(f"wrote {csv}")

    n_comp = int(out["Compositional_n"].max())
    n_struct = int(out["Structural_n"].max())
    comp_ranks = _ranks(out["Compositional_mean"].tolist(), mode="max")
    struct_ranks = _ranks(out["Structural_mean"].tolist(), mode="max")

    lines = [
        r"% Requires: \usepackage{booktabs} \usepackage[table]{xcolor}",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\caption{\textbf{Compositional vs structural split breakdown.} Mean "
        r"AUROC for each unsupervised detector restricted to compositional "
        rf"splits ($N{{=}}{n_comp}$ settings) and to structural splits "
        rf"($N{{=}}{n_struct}$ settings). The same near-chance ceiling holds "
        r"under either split type, indicating the dichotomy is not specific "
        r"to one shift family. " + COLOR_CODING_NOTE + "}",
        r"\label{tab:per_split_type}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Detector & Compositional & Structural \\",
        r"\midrule",
    ]
    for i, (_, row) in enumerate(out.iterrows()):
        cells = [row["detector"].replace("_", r"\_"),
                 _fmt_cell(row["Compositional_mean"], comp_ranks[i]),
                 _fmt_cell(row["Structural_mean"], struct_ranks[i])]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    # Old standalone-split-type table — superseded by per_arch_split_combined; archive.
    tex = CSV_DIR / "appendix_per_split_type.tex"
    tex.write_text("\n".join(lines) + "\n")
    print(f"wrote {tex}")


# ─── Table C: per-dataset stats (setting counts + MatBench size) ─────────────────
def dataset_stats():
    df = pd.read_csv(RESULTS / "all_detectors.csv")
    sub = df[df["detector"].isin(UNSUP_DETECTORS)].copy()
    sub["setting"] = sub["dataset"] + "/" + sub["arch"] + "/" + sub["split"]

    physics = pd.read_csv(RESULTS / "physics_splits_summary.csv")

    # MatBench per-task sizes (hard-coded; these match the front-matter).
    matbench_n = {
        "dielectric":  4764,
        "log_gvrh":    10987,
        "log_kvrh":    10987,
        "perovskites": 18928,
        "phonons":     1265,
    }
    target = {
        "dielectric":  "refractive index",
        "log_gvrh":    "log shear modulus",
        "log_kvrh":    "log bulk modulus",
        "perovskites": "formation energy",
        "phonons":     "phonon frequency",
    }

    rows = []
    for ds in DATASETS:
        sub_ds = sub[sub["dataset"] == ds]
        n_cells = sub_ds.drop_duplicates(["arch", "split"])
        n_splits = sub_ds["split"].nunique()
        n_archs = sub_ds["arch"].nunique()
        # Average ID/OOD test size (from physics summary, balanced sets are
        # reported as min(n_id_test, n_ood_test); show n_id_test and n_ood_test
        # raw means).
        ph = physics[physics["dataset"] == ds]
        rows.append({
            "dataset": ds,
            "target": target[ds],
            "matbench_n": matbench_n[ds],
            "n_splits_kept": int(n_splits),
            "n_archs": int(n_archs),
            "n_cells": int(len(n_cells)),
            "mean_n_id_test":  int(round(ph["n_id_test"].mean())) if len(ph) else 0,
            "mean_n_ood_test": int(round(ph["n_ood_test"].mean())) if len(ph) else 0,
        })
    out = pd.DataFrame(rows)
    csv = CSV_DIR / "appendix_dataset_stats.csv"
    out.to_csv(csv, index=False)
    print(f"wrote {csv}")

    lines = [
        r"% Requires: \usepackage{booktabs}",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{\textbf{Per-dataset benchmark statistics.} For each "
        r"MatBench task we report the prediction target, total task size, "
        r"number of OOD splits retained after the $\geq 50$ minority-class "
        r"filter, number of architectures evaluated, total number of settings "
        r"(arch $\times$ split), and mean unbalanced ID/OOD test sizes "
        r"across settings.}",
        r"\label{tab:dataset_stats}",
        r"\begin{tabular}{lllrrrrrr}",
        r"\toprule",
        r"Dataset & Target & MatBench $n$ & Splits & Archs & Cells & "
        r"Mean $|\mathrm{ID}_\mathrm{test}|$ & Mean $|\mathrm{OOD}_\mathrm{test}|$ \\",
        r"\midrule",
    ]
    for _, row in out.iterrows():
        ds = row["dataset"].replace("_", r"\_")
        lines.append(
            f"{ds} & {row['target']} & {row['matbench_n']:,} & "
            f"{row['n_splits_kept']} & {row['n_archs']} & {row['n_cells']} & "
            f"{row['mean_n_id_test']:,} & {row['mean_n_ood_test']:,} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = TEX_DIR / "appendix_dataset_stats.tex"
    tex.write_text("\n".join(lines) + "\n")
    print(f"wrote {tex}")


# ─── Table D: ID MAE per arch × dataset ───────────────────────────────────────
def id_mae_per_arch():
    physics = pd.read_csv(RESULTS / "physics_splits_summary.csv")

    rows = []
    for arch in ARCHS:
        row = {"arch": arch}
        for ds in DATASETS:
            sub = physics[(physics["architecture"] == arch)
                          & (physics["dataset"] == ds)]
            row[f"{ds}_id_mae"] = (float(sub["id_mae"].mean())
                                    if len(sub) else np.nan)
            row[f"{ds}_ood_mae"] = (float(sub["ood_mae"].mean())
                                     if len(sub) else np.nan)
        rows.append(row)
    out = pd.DataFrame(rows)

    csv = CSV_DIR / "appendix_id_mae_per_arch.csv"
    disp = out.copy()
    for ds in DATASETS:
        disp[f"{ds}_id_mae"] = disp[f"{ds}_id_mae"].round(4)
        disp[f"{ds}_ood_mae"] = disp[f"{ds}_ood_mae"].round(4)
    disp.to_csv(csv, index=False)
    print(f"wrote {csv}")

    lines = [
        r"% Requires: \usepackage{booktabs}",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{\textbf{Mean ID and OOD MAE per architecture and dataset.} "
        r"Each setting is the average across the OOD splits retained for that "
        r"dataset. Lower is better. The OOD/ID gap captures the magnitude of "
        r"distribution shift each architecture experiences and is the "
        r"independent variable in Figure~\ref{fig:auroc_vs_mae_gap} and "
        r"Table~\ref{tab:regression_auroc_mae_gap}.}",
        r"\label{tab:id_mae_per_arch}",
        r"\begin{tabular}{l" + "rr" * len(DATASETS) + r"}",
        r"\toprule",
        r" & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{d}}}"
                              for d in DATASETS_TEX) + r" \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}"
        r"\cmidrule(lr){8-9}\cmidrule(lr){10-11}",
        r"Arch & " + " & ".join("ID & OOD" for _ in DATASETS) + r" \\",
        r"\midrule",
    ]
    for _, row in out.iterrows():
        cells = [row["arch"]]
        for ds in DATASETS:
            id_v = row[f"{ds}_id_mae"]
            ood_v = row[f"{ds}_ood_mae"]
            cells.append(f"{id_v:.3f}" if not np.isnan(id_v) else "—")
            cells.append(f"{ood_v:.3f}" if not np.isnan(ood_v) else "—")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = TEX_DIR / "appendix_id_mae_per_arch.tex"
    tex.write_text("\n".join(lines) + "\n")
    print(f"wrote {tex}")


# ─── Table E: FPR95 per detector (only detectors that record FPR95) ──────────
def fpr95_per_detector():
    """Aggregate FPR95 from auroc.json files. Covers 16 of the 17 reported
    detectors; the only one without FPR95 in auroc.json is Flow, whose
    FPR95 lives in flow_models/.../results.json (deferred to v2)."""
    import json

    # Top-level auroc.json keys → display name.
    KEY_TO_DISPLAY = {
        "mahalanobis_bic": "MDS-BIC",
        "knn":             "KNN-k10",   # auroc.json's "knn" is the k=10 default
        "kpca":            "kPCA",
        "norm":            "Norm",
        "lof":             "LOF",
        "ocsvm":           "OCSVM",
        "iforest":         "IForest",
        "aderh":           "ADERH",
        "kde":             "KDE",
        "vim":             "ViM",
        "cosine":          "Cosine",
        "react_mds":       "ReAct+MDS",
        "ash_knn":         "ASH+KNN",
    }
    # Nested keys under auroc.json's "knn_sweep" block.
    KNN_SWEEP_KEYS = {
        "k20_norm1": "KNN-k20-norm",
        "k50_norm1": "KNN-k50-norm",
    }

    rows = []
    for f in (RESULTS).glob("matbench_*/*/*/fold_0/auroc.json"):
        d = json.loads(f.read_text())
        ds = d["dataset"].replace("matbench_", "")
        arch = d["architecture"]
        split = d["split"]
        for k, disp in KEY_TO_DISPLAY.items():
            entry = d.get(k)
            if isinstance(entry, dict) and "fpr95" in entry:
                rows.append({"dataset": ds, "arch": arch, "split": split,
                             "detector": disp, "fpr95": float(entry["fpr95"])})
        sweep = d.get("knn_sweep", {})
        if isinstance(sweep, dict):
            for sk, disp in KNN_SWEEP_KEYS.items():
                entry = sweep.get(sk)
                if isinstance(entry, dict) and "fpr95" in entry:
                    rows.append({"dataset": ds, "arch": arch, "split": split,
                                 "detector": disp, "fpr95": float(entry["fpr95"])})

    long_df = pd.DataFrame(rows)
    if long_df.empty:
        print("no FPR95 data found")
        return

    # Restrict to the 257 retained settings (drop degenerate / minority<50).
    cells = pd.read_csv(RESULTS / "all_cells_summary.csv")
    cells = cells[~cells["degenerate"]].copy()
    keep = set(zip(cells["dataset"], cells["arch"], cells["split"]))
    long_df = long_df[long_df.apply(
        lambda r: (r["dataset"], r["arch"], r["split"]) in keep, axis=1)]

    # Combined order: top-level keys + knn_sweep keys.
    DETECTOR_ORDER = list(KEY_TO_DISPLAY.values()) + list(KNN_SWEEP_KEYS.values())

    # Per-detector summary: mean and median FPR95 across all cells.
    per_det = long_df.groupby("detector")["fpr95"].agg(
        mean_fpr95="mean", median_fpr95="median").reset_index()
    per_det = per_det.set_index("detector").reindex(
        DETECTOR_ORDER).reset_index()

    csv = CSV_DIR / "appendix_fpr95_per_detector.csv"
    per_det_disp = per_det.copy()
    per_det_disp["mean_fpr95"] = per_det_disp["mean_fpr95"].round(3)
    per_det_disp["median_fpr95"] = per_det_disp["median_fpr95"].round(3)
    per_det_disp.to_csv(csv, index=False)
    print(f"wrote {csv}")

    # Per-dataset × detector mean FPR95.
    per_ds = (long_df.groupby(["dataset", "detector"])["fpr95"]
                       .mean().reset_index())
    per_ds_pivot = per_ds.pivot(index="detector", columns="dataset",
                                 values="fpr95")
    per_ds_pivot = per_ds_pivot.reindex(DETECTOR_ORDER)
    per_ds_pivot = per_ds_pivot[DATASETS]
    csv2 = CSV_DIR / "appendix_fpr95_per_dataset.csv"
    per_ds_pivot.round(3).to_csv(csv2)
    print(f"wrote {csv2}")

    # LaTeX combined table: per-detector + per-dataset means.
    n_cells = long_df.drop_duplicates(["dataset", "arch", "split"]).shape[0]
    detectors_in_order = DETECTOR_ORDER
    all_cells_col = [
        per_det.loc[per_det["detector"] == det, "mean_fpr95"].iat[0]
        for det in detectors_in_order
    ]
    per_ds_cols = {
        ds: [per_ds_pivot.loc[det, ds] if det in per_ds_pivot.index else np.nan
             for det in detectors_in_order]
        for ds in DATASETS
    }
    # Lower-is-better: rank by min.
    rank_all = _ranks(all_cells_col, mode="min")
    rank_ds  = {ds: _ranks(per_ds_cols[ds], mode="min") for ds in DATASETS}

    lines = [
        r"% Requires: \usepackage{booktabs} \usepackage[table]{xcolor}",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{\textbf{FPR95 per detector and per dataset} ($N{=}"
        + str(n_cells) +
        r"$ settings aggregated from per-setting \texttt{auroc.json} files). "
        r"FPR95 is the false-positive rate when the true-positive rate is "
        r"95\%; lower is better. Covers 14 of the 15 unsupervised detectors "
        r"reported in Table~\ref{tab:per_detector_summary}; the lone "
        r"omission is Flow (NF), whose FPR95 lives in \texttt{flow\_models/.../results.json} "
        r"on the released dataset and is deferred to v2 of this aggregation. "
        r"Mean FPR95 across all settings $\geq 0.92$ for every detector "
        r"is consistent with the near-chance AUROC story in Table~1. "
        + COLOR_CODING_NOTE_MIN + "}",
        r"\label{tab:fpr95_summary}",
        r"\begin{tabular}{lr" + "r" * len(DATASETS) + r"}",
        r"\toprule",
        r"Detector & All settings & " + " & ".join(DATASETS_TEX) + r" \\",
        r"\midrule",
    ]
    for i, det in enumerate(detectors_in_order):
        cells = [det.replace("_", r"\_"),
                 _fmt_cell(all_cells_col[i], rank_all[i])]
        for ds in DATASETS:
            cells.append(_fmt_cell(per_ds_cols[ds][i], rank_ds[ds][i]))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    # Full per-detector × per-dataset table is the appendix-bound version.
    tex = TEX_DIR / "appendix_fpr95_per_detector.tex"
    tex.write_text("\n".join(lines) + "\n")
    print(f"wrote {tex}")


# ─── Table F: OMee LOCO cross-check ───────────────────────────────────────────
def omee_cross_check():
    src = RESULTS / "omee_splits_summary.csv"
    if not src.exists():
        print(f"missing {src} — skipping OMee table")
        return
    df = pd.read_csv(src)

    # Aggregate per (dataset, split) across architectures for a clean summary.
    agg = (df.groupby(["dataset", "split"])
             .agg(n_folds=("n_folds", "sum"),
                  mean_id_mae=("id_mae_mean", "mean"),
                  mean_ood_mae=("ood_mae_mean", "mean"),
                  mean_mae_gap=("mae_gap_mean", "mean"),
                  mean_maha_auroc=("maha_auroc_mean", "mean"),
                  mean_knn_auroc=("knn_auroc_mean", "mean"))
             .reset_index())

    csv = CSV_DIR / "appendix_omee_cross_check.csv"
    agg_disp = agg.copy()
    for c in ("mean_id_mae", "mean_ood_mae", "mean_mae_gap",
              "mean_maha_auroc", "mean_knn_auroc"):
        agg_disp[c] = agg_disp[c].round(3)
    agg_disp.to_csv(csv, index=False)
    print(f"wrote {csv}")

    lines = [
        r"% Requires: \usepackage{booktabs}",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{\textbf{OMee distribution-shift cross-check.} The "
        r"OMee suite (\citet{omee2024structurebased}) defines three "
        r"alternative split protocols: \textbf{LOCO} (leave-one-cluster-"
        r"out), \textbf{SparseXcluster} (sparse covariate clusters), and "
        r"\textbf{SparseYcluster} (sparse target clusters). We rerun the "
        r"two detectors with directly-comparable implementations "
        r"(Mahalanobis and KNN) on these splits using the same ALIGNN "
        r"and CGCNN embeddings as the main paper, 50 folds per split. "
        r"Mean AUROC remains within $\sim$0.05 of chance, consistent "
        r"with the near-chance story under physics-motivated splits.}",
        r"\label{tab:omee_cross_check}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Dataset & Split & ID MAE & OOD MAE & log gap & "
        r"Maha AUROC & KNN AUROC \\",
        r"\midrule",
    ]
    prev_ds = None
    for _, row in agg.iterrows():
        ds = row["dataset"].replace("_", r"\_") if row["dataset"] != prev_ds else ""
        if prev_ds is not None and row["dataset"] != prev_ds:
            lines.append(r"\addlinespace[2pt]")
        lines.append(
            f"{ds} & {row['split']} & "
            f"{row['mean_id_mae']:.3f} & {row['mean_ood_mae']:.3f} & "
            f"{row['mean_mae_gap']:+.3f} & "
            f"{row['mean_maha_auroc']:.3f} & {row['mean_knn_auroc']:.3f} \\\\"
        )
        prev_ds = row["dataset"]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = TEX_DIR / "appendix_omee_cross_check.tex"
    tex.write_text("\n".join(lines) + "\n")
    print(f"wrote {tex}")


# ─── Table G: per-setting viability matrix ──────────────────────────────────────
def per_cell_viability():
    """Enumerate every (dataset, arch, split) ∈ {5} × {4} × {15} and label
    each with `kept`, `dropped (no split)` (when the predicate didn't yield
    ≥50 OOD samples for that dataset), or `dropped (degenerate)` (CGCNN
    on phonons collapse)."""
    df = pd.read_csv(RESULTS / "all_detectors.csv")
    sub = df[df["detector"].isin(UNSUP_DETECTORS)].copy()
    present = set(zip(sub["dataset"], sub["arch"], sub["split"]))

    # Splits per dataset that exist at all (even if some archs fail).
    splits_per_dataset = {ds: set(sub.loc[sub["dataset"] == ds, "split"].unique())
                          for ds in DATASETS}

    # Canonical 15 splits.
    ALL_SPLITS = [
        "comp_carbide", "comp_chalcogenide", "comp_halide",
        "comp_intermetallic", "comp_nitride", "comp_oxide",
        "struct_cubic", "struct_hexagonal", "struct_tetragonal",
        "struct_orthorhombic", "struct_monoclinic", "struct_triclinic",
        "struct_high_spacegroup", "struct_low_spacegroup", "struct_perovskite",
    ]

    rows = []
    for ds in DATASETS:
        for split in ALL_SPLITS:
            split_exists_for_ds = split in splits_per_dataset[ds]
            for arch in ARCHS:
                if (ds, arch, split) in present:
                    status = "kept"
                    reason = ""
                elif not split_exists_for_ds:
                    status = "dropped"
                    reason = "minority class < 50 samples"
                else:
                    status = "dropped"
                    reason = "training collapsed (all-zero embeddings)"
                rows.append({"dataset": ds, "arch": arch, "split": split,
                             "status": status, "reason": reason})
    out = pd.DataFrame(rows)
    csv = CSV_DIR / "appendix_per_cell_viability.csv"
    out.to_csv(csv, index=False)
    print(f"wrote {csv}  ({len(out)} rows; "
          f"{(out['status'] == 'kept').sum()} kept)")

    # Compact LaTeX summary: one row per (dataset, split) showing count of
    # archs kept (out of 4) plus dropping reason if any are missing.
    summary = (out.assign(kept=lambda d: d["status"] == "kept")
                  .groupby(["dataset", "split"])
                  .agg(n_kept=("kept", "sum"),
                       reasons=("reason",
                                lambda s: ";".join(sorted(set(r for r in s if r)))))
                  .reset_index())

    n_total_settings = len(summary)
    n_kept_total = int(summary["n_kept"].sum())

    lines = [
        r"% Requires: \usepackage{booktabs}",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{\textbf{Per-setting viability matrix.} For each "
        r"(dataset, split) in the canonical $5 \times 15 = 75$ pair grid, "
        r"the number of architectures (out of 4: ALIGNN, CGCNN, SchNet, "
        r"DimeNet++) that produced a valid embedding and an evaluable "
        r"AUROC. Cells are dropped at construction if the minority "
        r"predicate-class has $< 50$ samples in the dataset, or excluded "
        r"after training if the encoder collapsed (all-zero embeddings). "
        rf"Total: ${n_kept_total}$ of ${4 * n_total_settings}$ candidate settings "
        r"retained.}",
        r"\label{tab:per_cell_viability}",
        r"\begin{tabular}{llcl}",
        r"\toprule",
        r"Dataset & Split & Archs kept & Reason for drops \\",
        r"\midrule",
    ]
    prev_ds = None
    for _, row in summary.iterrows():
        ds = (row["dataset"].replace("_", r"\_")
              if row["dataset"] != prev_ds else "")
        if prev_ds is not None and row["dataset"] != prev_ds:
            lines.append(r"\addlinespace[2pt]")
        n_kept = int(row["n_kept"])
        reasons = row["reasons"].replace("_", r"\_") if row["reasons"] else "—"
        if n_kept == 4:
            kept_disp = "4"
        else:
            kept_disp = f"\\textbf{{{n_kept}}}"
        split_disp = row["split"].replace("_", r"\_")
        lines.append(f"{ds} & {split_disp} & {kept_disp} & {reasons} \\\\")
        prev_ds = row["dataset"]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    # Per-cell viability table is superseded by the dropped_splits table (which
    # focuses on the rows that were excluded). Archive in CSV_DIR.
    tex = CSV_DIR / "appendix_per_cell_viability.tex"
    tex.write_text("\n".join(lines) + "\n")
    print(f"wrote {tex}")


# ─── Table H: Per-setting ID/OOD MAE (every retained (dataset, split, arch)) ─────
def per_split_mae():
    src = RESULTS / "all_cells_summary.csv"
    if not src.exists():
        print(f"missing {src} — skipping per-split MAE table")
        return
    df = pd.read_csv(src)
    df = df[~df["degenerate"]].copy()
    df["log_mae_gap"] = np.log10(df["ood_mae"] / df["id_mae"])
    df = df.sort_values(["dataset", "split", "arch"]).reset_index(drop=True)

    csv = CSV_DIR / "appendix_per_setting_mae.csv"
    df[["dataset", "arch", "split", "id_mae", "ood_mae",
        "log_mae_gap"]].to_csv(csv, index=False)
    print(f"wrote {csv}")

    n = len(df)
    lines = [
        r"% Requires: \usepackage{booktabs} \usepackage{longtable}",
        # Caption spans full text width (longtable's default is the natural
        # column-set width, which is narrow with single-digit numerics).
        r"{\setlength{\LTcapwidth}{\linewidth}",
        # @{\extracolsep{\fill}} stretches columns to fill available width.
        r"\begin{longtable}{@{\extracolsep{\fill}}llrrrr@{}}",
        r"\caption{\textbf{Per-setting held-out ID and OOD MAE.} Each row is one "
        r"retained (dataset, split, arch) setting ($N=" + str(n) + r"$ after "
        r"excluding the seven CGCNN/phonons settings with collapsed embeddings "
        r"and the splits with $<$50 minority-class samples). $\log_{10}$ "
        r"gap $= \log_{10}(\text{OOD MAE} / \text{ID MAE})$ is the "
        r"independent variable in Figure~\ref{fig:auroc_vs_mae_gap_full} "
        r"and Table~\ref{tab:regression_auroc_mae_gap}. Positive values "
        r"indicate the encoder regresses worse on OOD inputs (the viable "
        r"regime); negative values are sub-shift settings (OOD MAE lower "
        r"than ID MAE).}",
        r"\label{tab:per_setting_mae} \\",
        r"\toprule",
        r"Dataset & Split & Arch & ID MAE & OOD MAE & $\log_{10}$ gap \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{6}{l}{\small\itshape (continued)} \\",
        r"\toprule",
        r"Dataset & Split & Arch & ID MAE & OOD MAE & $\log_{10}$ gap \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{6}{r}{\small\itshape (continued on next page)} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    prev_ds, prev_split = None, None
    for _, row in df.iterrows():
        if prev_ds is not None and row["dataset"] != prev_ds:
            lines.append(r"\addlinespace[2pt]")
        ds = row["dataset"] if row["dataset"] != prev_ds else ""
        sp = row["split"] if (row["dataset"], row["split"]) != (prev_ds, prev_split) else ""
        ds_disp = ds.replace("_", r"\_")
        sp_disp = sp.replace("_", r"\_")
        gap = row["log_mae_gap"]
        gap_str = f"{gap:+.3f}" if not np.isnan(gap) else "—"
        lines.append(
            f"{ds_disp} & {sp_disp} & {row['arch']} & "
            f"{row['id_mae']:.4f} & {row['ood_mae']:.4f} & {gap_str} \\\\"
        )
        prev_ds, prev_split = row["dataset"], row["split"]
    lines.append(r"\end{longtable}}")
    # Full 257-row per-setting MAE table is the appendix-bound version: the
    # data input that justifies N=257 in the central regression.
    tex = TEX_DIR / "appendix_per_setting_mae.tex"
    tex.write_text("\n".join(lines) + "\n")
    print(f"wrote {tex}")


# ─── Table I: Dropped splits / settings (focused list of exclusions) ─────────────
def dropped_splits():
    """List splits dropped at construction (minority < 50) and settings dropped
    after training (encoder collapse). Source: per_cell_viability matrix."""
    # Pull the structured data from the per_cell_viability summary used elsewhere.
    # We replicate the same logic locally so this function is self-contained.
    src = RESULTS / "physics_splits_summary.csv"
    if not src.exists():
        print(f"missing {src} — skipping dropped-splits table")
        return
    df = pd.read_csv(src)

    minority_dropped = []   # (dataset, split) → minority_n
    if "minority_n" in df.columns:
        # If the source has it, use it; otherwise fall back.
        for (ds, sp), grp in df.groupby(["dataset", "split"]):
            mn = int(grp["minority_n"].iat[0]) if pd.notna(grp["minority_n"].iat[0]) else None
            if mn is not None and mn < 50:
                minority_dropped.append((ds, sp, mn))
    # Hard-coded fallback (matches per_cell_viability): the 9 dropped splits.
    if not minority_dropped:
        minority_dropped = [
            ("dielectric",  "comp_carbide", None),
            ("perovskites", "comp_carbide", None),
            ("perovskites", "comp_intermetallic", None),
            ("perovskites", "struct_hexagonal", None),
            ("perovskites", "struct_low_spacegroup", None),
            ("perovskites", "struct_monoclinic", None),
            ("perovskites", "struct_triclinic", None),
            ("phonons",     "comp_carbide", None),
            ("phonons",     "struct_triclinic", None),
        ]

    # Cells dropped post-training (encoder collapsed; CGCNN on phonons).
    collapsed_cells = []
    sm = pd.read_csv(RESULTS / "all_cells_summary.csv")
    if "degenerate" in sm.columns:
        for _, row in sm[sm["degenerate"]].iterrows():
            collapsed_cells.append(
                (row["dataset"], row["split"], row["arch"]))

    csv = CSV_DIR / "appendix_dropped_splits.csv"
    rows = []
    for (ds, sp, mn) in minority_dropped:
        rows.append({"dataset": ds, "split": sp, "arch": "—",
                     "reason": "minority class < 50",
                     "details": f"minority n={mn}" if mn is not None else "",
                     "scope": "entire split (all 4 archs)"})
    for (ds, sp, arch) in collapsed_cells:
        rows.append({"dataset": ds, "split": sp, "arch": arch,
                     "reason": "encoder collapsed",
                     "details": "var(z) < 1e-10",
                     "scope": "single (arch, dataset, split) setting"})
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"wrote {csv}")

    n_minority = len(minority_dropped)
    n_collapsed = len(collapsed_cells)

    lines = [
        r"% Requires: \usepackage{booktabs}",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{8pt}",
        r"\caption{\textbf{Excluded splits and settings.} Two filters reduce "
        rf"the candidate $5 \times 15 \times 4 = 300$ settings to the $257$ "
        r"reported in the main paper. \textbf{(top)} Splits dropped at "
        r"construction because the minority predicate-class has fewer "
        rf"than 50 samples in the source dataset ($n={n_minority}$ splits, "
        r"all 4 architectures). \textbf{(bottom)} Cells dropped after "
        r"training because CGCNN failed to converge on phonons and "
        rf"produced all-zero embeddings ($n={n_collapsed}$ settings, "
        r"single-arch).}",
        r"\label{tab:dropped_splits}",
        r"\begin{tabular}{lllll}",
        r"\toprule",
        r"Dataset & Split & Arch & Reason & Details \\",
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{Splits dropped at construction (minority class $<$ 50):}} \\",
    ]
    for (ds, sp, mn) in minority_dropped:
        details = f"$n={mn}$" if mn is not None else "$n<50$"
        lines.append(
            f"{ds.replace('_', chr(92)+'_')} & "
            f"{sp.replace('_', chr(92)+'_')} & all & "
            f"minority class $<$ 50 & {details} \\\\"
        )
    if collapsed_cells:
        lines.append(r"\addlinespace[2pt]")
        lines.append(r"\multicolumn{5}{l}{\textit{Cells dropped post-training (encoder collapse):}} \\")
        for (ds, sp, arch) in collapsed_cells:
            lines.append(
                f"{ds.replace('_', chr(92)+'_')} & "
                f"{sp.replace('_', chr(92)+'_')} & "
                f"{arch} & "
                f"encoder collapsed & $\\mathrm{{var}}(z) < 10^{{-10}}$ \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = TEX_DIR / "appendix_dropped_splits.tex"
    tex.write_text("\n".join(lines) + "\n")
    print(f"wrote {tex}")


# ─── Table J: Combined per-arch + per-split-type (the appendix-bound version) ──
def per_arch_split_combined():
    """Single wider table: detector × {ALIGNN, CGCNN, SchNet, DimeNet++,
    Compositional, Structural} mean AUROC, with arch and split-type as nested
    column-group headers.
    """
    df = pd.read_csv(RESULTS / "all_detectors.csv")
    sub = df[df["detector"].isin(UNSUP_DETECTORS)].copy()
    sub["display"] = sub["detector"].map(DISPLAY)
    sub["split_type"] = sub["split"].apply(
        lambda s: "Compositional" if s.startswith("comp_")
        else ("Structural" if s.startswith("struct_") else "Other"))

    rows = []
    for det in [DISPLAY[k] for k in UNSUP_DETECTORS]:
        row = {"detector": det}
        for arch in ARCHS:
            vals = sub.loc[(sub["display"] == det) & (sub["arch"] == arch),
                           "auroc"].values
            row[f"arch_{arch}"] = float(np.mean(vals)) if len(vals) else np.nan
        for stype in ("Compositional", "Structural"):
            vals = sub.loc[(sub["display"] == det) & (sub["split_type"] == stype),
                           "auroc"].values
            row[f"split_{stype}"] = float(np.mean(vals)) if len(vals) else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)

    csv = CSV_DIR / "appendix_per_arch_split_combined.csv"
    out_round = out.copy()
    for c in out.columns:
        if c != "detector":
            out_round[c] = out_round[c].round(3)
    out_round.to_csv(csv, index=False)
    print(f"wrote {csv}")

    n_comp = int(((sub["split_type"] == "Compositional")).sum() // len(UNSUP_DETECTORS))
    n_struct = int(((sub["split_type"] == "Structural")).sum() // len(UNSUP_DETECTORS))

    # Per-column ranks (max=best across the 15 unsupervised detectors).
    arch_cols = [f"arch_{a}" for a in ARCHS]
    split_cols = [f"split_{s}" for s in ("Compositional", "Structural")]
    col_ranks = {c: _ranks(out[c].tolist(), mode="max")
                 for c in arch_cols + split_cols}

    arch_disp = [a.upper() if a in ("knn",) else a.capitalize() for a in ARCHS]
    n_arch = len(ARCHS)
    n_split = 2

    lines = [
        r"% Requires: \usepackage{booktabs} \usepackage[table]{xcolor}",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{\textbf{Per-detector AUROC, stratified by architecture and "
        r"by split type.} Each row is one unsupervised detector; the four "
        r"left numeric columns report mean AUROC under each GNN backbone, the "
        rf"two right columns report mean AUROC restricted to compositional "
        rf"splits ($N={n_comp}$ settings) and structural splits "
        rf"($N={n_struct}$ settings). No detector exceeds $0.55$ mean AUROC in "
        r"any of the six columns: the near-chance ceiling is invariant to "
        r"both architecture and split family. " + COLOR_CODING_NOTE + "}",
        r"\label{tab:per_arch_split_combined}",
        r"\begin{tabular}{l" + "r" * (n_arch + n_split) + r"}",
        r"\toprule",
        r" & \multicolumn{" + str(n_arch) + r"}{c}{Architecture} & "
        r"\multicolumn{" + str(n_split) + r"}{c}{Split type} \\",
        rf"\cmidrule(lr){{2-{1+n_arch}}}\cmidrule(lr){{{2+n_arch}-{1+n_arch+n_split}}}",
        r"Detector & " + " & ".join(arch_disp + ["Compositional", "Structural"]) + r" \\",
        r"\midrule",
    ]
    for i, (_, row) in enumerate(out.iterrows()):
        cells = [row["detector"].replace("_", r"\_")]
        for c in arch_cols + split_cols:
            cells.append(_fmt_cell(row[c], col_ranks[c][i]))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = TEX_DIR / "appendix_per_arch_split_combined.tex"
    tex.write_text("\n".join(lines) + "\n")
    print(f"wrote {tex}")


# ─── Table K: Detector-family × dataset mean AUROC (GeSS-style summary) ──────
def family_by_dataset():
    """Mean AUROC per (detector family, dataset). Rows are families
    (Distance, Density, Isolation, Boundary/Hybrid, Supervised);
    columns are the 5 datasets plus an "All" column. Highlighting is
    column-wise max + 2nd-best among unsupervised families only;
    Supervised is reported as a diagnostic ceiling and excluded from
    the rankings.
    """
    DETECTOR_TO_FAMILY = {
        # Distance
        "KNN k=20 norm": "Distance",
        "KNN k=50 norm": "Distance",
        "Norm":          "Distance",
        "Cosine":        "Distance",
        "LOF":           "Distance",
        "ASH + KNN":     "Distance",
        # Density
        "MDS-BIC":       "Density",
        "KDE":           "Density",
        "Flow (NF)":     "Density",
        "kPCA":          "Density",
        "ReAct + MDS":   "Density",
        # Isolation
        "IForest":       "Isolation",
        "ADERH":         "Isolation",
        # Boundary / Hybrid
        "OCSVM":         "Boundary / Hybrid",
        "ViM":           "Boundary / Hybrid",
        # Supervised diagnostic probes
        "Linear probe":  "Supervised",
        "MLP probe":     "Supervised",
    }
    FAMILY_ORDER = ["Distance", "Density", "Isolation",
                    "Boundary / Hybrid", "Supervised"]

    df = pd.read_csv(RESULTS / "all_detectors.csv")
    sub = df[df["detector"].isin(DETECTOR_TO_FAMILY)].copy()
    sub["family"] = sub["detector"].map(DETECTOR_TO_FAMILY)

    rows = []
    for fam in FAMILY_ORDER:
        row = {"family": fam}
        # All-settings mean
        all_vals = sub.loc[sub["family"] == fam, "auroc"].values
        row["all"] = float(np.mean(all_vals)) if len(all_vals) else np.nan
        for ds in DATASETS:
            vals = sub.loc[(sub["family"] == fam) & (sub["dataset"] == ds),
                           "auroc"].values
            row[ds] = float(np.mean(vals)) if len(vals) else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)

    csv = CSV_DIR / "appendix_family_by_dataset.csv"
    disp = out.copy()
    for c in ["all"] + DATASETS:
        disp[c] = disp[c].round(3)
    disp.to_csv(csv, index=False)
    print(f"wrote {csv}")

    # Per-column ranks restricted to UNsupervised families.
    unsup_idx = [i for i, fam in enumerate(out["family"]) if fam != "Supervised"]
    cols_to_rank = ["all"] + DATASETS
    col_ranks = {c: [-1] * len(out) for c in cols_to_rank}
    for c in cols_to_rank:
        unsup_pairs = [(i, out[c].iat[i]) for i in unsup_idx]
        unsup_pairs.sort(key=lambda t: -t[1] if t[1] == t[1] else float("inf"))
        for r, (i, _) in enumerate(unsup_pairs[:2]):
            col_ranks[c][i] = r

    lines = [
        r"% Requires: \usepackage{booktabs} \usepackage[table]{xcolor}",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\caption{\textbf{Mean AUROC by detector family and dataset.} Each "
        r"row aggregates the per-setting AUROCs of every detector in that "
        r"family; columns are the 5 datasets plus an ``All settings'' "
        r"mean. The supervised row (linear and MLP probes) is reported as a "
        r"diagnostic ceiling and is excluded from the column rankings. "
        r"No unsupervised family reaches mean AUROC $\geq 0.55$ on any "
        r"individual dataset; the cross-dataset spread tracks the "
        r"bottleneck dichotomy of "
        r"Section~\ref{sec:bottleneck_classification}. " + COLOR_CODING_NOTE + "}",
        r"\label{tab:family_by_dataset}",
        r"\begin{tabular}{l" + "r" * (1 + len(DATASETS)) + r"}",
        r"\toprule",
        r"Family & All settings & " + " & ".join(DATASETS_TEX) + r" \\",
        r"\midrule",
    ]
    for i, (_, row) in enumerate(out.iterrows()):
        # Add a midrule separator before the Supervised row to mark it as
        # the ceiling reference rather than an apples-to-apples comparison.
        if row["family"] == "Supervised":
            lines.append(r"\midrule")
        cells = [row["family"]]
        for c in cols_to_rank:
            cells.append(_fmt_cell(row[c], col_ranks[c][i]))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = TEX_DIR / "appendix_family_by_dataset.tex"
    tex.write_text("\n".join(lines) + "\n")
    print(f"wrote {tex}")


def main():
    per_arch_summary()           # writes archived .tex to CSV_DIR
    per_split_type()             # writes archived .tex to CSV_DIR
    per_arch_split_combined()    # the appendix-bound replacement
    family_by_dataset()          # GeSS-style family × dataset summary
    dataset_stats()
    id_mae_per_arch()
    fpr95_per_detector()         # writes both archived full .tex and compressed appendix .tex
    omee_cross_check()
    per_cell_viability()
    per_split_mae()              # CSV in code release; .tex archived in CSV_DIR
    dropped_splits()


if __name__ == "__main__":
    main()
