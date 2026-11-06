#!/usr/bin/env python3
"""TASK 3: generate camera-ready LaTeX tables (booktabs; mean $\\pm$ std) into
outputs/tables/. Reads the agg_*.csv produced by cr_numbers.py /
cr_perm_numbers.py and per_seed_cells.csv.

Tables:
  per_detector_summary.tex   (Table 1, 15 detectors + 2 probes, mean+-std)
  bottleneck_summary.tex     (Table 2, + best-detector gap + label)
  per_arch_probe.tex         (\\label{tab:per_arch_probe})
  table13_probe_column.tex   (permutation table, probe col = canonical pipeline)
  regression_compacted.tex   (\\label{tab:regression_compacted})
  regression_auroc_mae_gap.tex (full OLS grid, \\label{tab:regression_auroc_mae_gap})
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cr_common as C
ROOT = C.ROOT
CO = C.OUT
TB = os.path.join(CO, "tables"); os.makedirs(TB, exist_ok=True)

det = pd.read_csv(os.path.join(CO, "agg_per_detector.csv"))
pdp = pd.read_csv(os.path.join(CO, "agg_per_dataset_probe.csv"))
ols = pd.read_csv(os.path.join(CO, "agg_ols_grid.csv"))
perm = pd.read_csv(os.path.join(CO, "agg_permutation.csv"))
cells = pd.read_csv(os.path.join(CO, "per_seed_cells.csv"))

def esc(s): return s.replace("_", r"\_")
def pm(m, s, d=3): return f"{m:.{d}f} $\\pm$ {s:.{d}f}"
def pm1(m, s): return f"{m:.1f} $\\pm$ {s:.1f}"

FAM_DISPLAY = {"Distance":"Distance","Density":"Density","Isolation":"Isolation",
               "Boundary-Hybrid":"BH","Supervised":"Supervised"}
FAM_ORDER = ["Distance","Density","Isolation","Boundary-Hybrid","Supervised"]

# ---------------- Table 1: per_detector_summary ----------------
def t_per_detector():
    det["family"] = pd.Categorical(det.family, FAM_ORDER, ordered=True)
    d = det.sort_values("family", kind="stable")
    L = [r"% Requires: \usepackage{booktabs} \usepackage{multirow}",
         r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
         r"\caption{\textbf{Per-detector performance summary across $N_{all}=264$ settings "
         r"(mean $\pm$ std over three encoder training seeds).} Win count splits ties equally; "
         r"\% settings $>0.55$ uses the above-chance threshold of \S\ref{sec:design}. Supervised "
         r"probes (bottom two rows) are diagnostic ceilings, not deployable detectors.}",
         r"\label{tab:per_detector_summary}",
         r"\resizebox{\linewidth}{!}{%",
         r"\begin{tabular}{llcccc}", r"\toprule",
         r"Family & Detector & Mean AUROC & Median AUROC & Win count & \% settings $>0.55$ \\",
         r"\midrule"]
    fams = [f for f in FAM_ORDER if (d.family == f).any()]
    for fi, fam in enumerate(fams):
        sub = d[d.family == fam]
        nrow = len(sub)
        for i, (_, r) in enumerate(sub.iterrows()):
            fcell = (r"\multirow{%d}{*}{%s}" % (nrow, FAM_DISPLAY[fam])) if i == 0 else ""
            win = "--" if r.family == "Supervised" or not np.isfinite(r.win_count) else pm(r.win_count, r.win_count_sd, 1)
            L.append(f"{fcell} & {esc(r.detector)} & {pm(r['mean'], r.mean_sd)} & "
                     f"{pm(r['median'], r.median_sd)} & {win} & {pm1(r.pct, r.pct_sd)}\\% \\\\")
        if fam == "Boundary-Hybrid":
            L.append(r"\midrule")
        elif fi < len(fams) - 1:
            L.append(r"\cmidrule(lr){1-6}")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(L) + "\n"

# ---------------- Table 2: bottleneck_summary ----------------
def t_bottleneck():
    L = [r"% Requires: \usepackage{booktabs} \usepackage[table]{xcolor}",
         r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{5pt}",
         r"\caption{\textbf{Bottleneck dichotomy by task (mean $\pm$ std over three "
         r"encoder training seeds).} Linear/MLP: supervised probe AUROCs. Mean/Best "
         r"detector: mean across all 15 unsupervised detectors and Cosine (highest-mean "
         r"detector). Gap (mean) $=$ Linear $-$ Mean detector (sets the primary Bottleneck "
         r"label); Gap (best) $=$ Linear $-$ Best detector (sets Label (best)). Under Gap "
         r"(best), dielectric and log\_gvrh fall below the 0.05 cutoff and are Mixed. "
         r"Method rows shaded yellow, Encoder rows blue.}",
         r"\label{tab:bottleneck_summary}",
         r"\resizebox{\linewidth}{!}{%",
         r"\begin{tabular}{lrrrrrrrll}", r"\toprule",
         r"Dataset & $N$ & Linear & MLP & Mean det. & Best det. & Gap (mean) & Gap (best) & "
         r"Bottleneck & Label (best) \\", r"\midrule"]
    color = {"Method":"yellow!20","Encoder":"blue!12","Mixed":"gray!15"}
    for _, r in pdp.iterrows():
        rc = color.get(r.label_mean)
        if rc: L.append(r"\rowcolor{" + rc + "}")
        L.append(f"{esc(r.dataset)} & {int(r.N)} & {pm(r.linear,r.linear_sd)} & {pm(r.mlp,r.mlp_sd)} & "
                 f"{pm(r.mean_det,r.mean_det_sd)} & {pm(r.cosine,r.cosine_sd)} & "
                 f"{r.gap_mean:+.3f} $\\pm$ {r.gap_mean_sd:.3f} & {r.gap_best:+.3f} $\\pm$ {r.gap_best_sd:.3f} & "
                 f"{r.label_mean} & {r.label_best} \\\\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(L) + "\n"

# ---------------- per_arch_probe ----------------
def t_per_arch_probe():
    labels = dict(zip(pdp.dataset, pdp.label_mean))
    archs = ["alignn","cgcnn","schnet","dimenet"]
    disp = {"alignn":"ALIGNN","cgcnn":"CGCNN","schnet":"SchNet","dimenet":"DimeNet++"}
    # per-seed (dataset,arch) mean, then mean/std across seeds
    g = cells.groupby(["dataset","arch","seed"]).linear_probe.mean().reset_index()
    rows = {}
    for ds in C.DATASETS:
        rows[ds] = {}
        for ar in archs:
            v = g[(g.dataset==ds)&(g.arch==ar)].linear_probe.values
            rows[ds][ar] = (float(np.mean(v)), float(np.std(v, ddof=1)))
    L = [r"% Requires: \usepackage{booktabs}",
         r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
         r"\caption{\textbf{Linear-probe AUROC per dataset $\times$ architecture "
         r"(mean $\pm$ std over three encoder training seeds).} Every one of the 20 cells "
         r"falls on the same side of the 0.55 threshold as its dataset-level bottleneck "
         r"label: Method datasets are above, Encoder datasets below. Spread $=$ max $-$ min "
         r"across architectures.}",
         r"\label{tab:per_arch_probe}",
         r"\resizebox{\linewidth}{!}{%",
         r"\begin{tabular}{l" + "c"*len(archs) + r"cl}", r"\toprule",
         "Dataset & " + " & ".join(disp[a] for a in archs) + r" & Spread & Label \\", r"\midrule"]
    for ds in C.DATASETS:
        means = [rows[ds][a][0] for a in archs]
        spread = max(means) - min(means)
        cellvals = " & ".join(pm(rows[ds][a][0], rows[ds][a][1]) for a in archs)
        L.append(f"{esc(ds)} & {cellvals} & {spread:.3f} & {labels[ds]} \\\\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(L) + "\n"

# ---------------- Table 13: permutation w/ canonical probe column ----------------
def t_perm_table():
    probe = dict(zip(pdp.dataset, zip(pdp.linear, pdp.linear_sd)))
    lab = dict(zip(pdp.dataset, pdp.label_mean))
    color = {"Method":"yellow!20","Encoder":"blue!12","Mixed":"gray!15"}
    L = [r"% Requires: \usepackage{booktabs} \usepackage[table]{xcolor}",
         r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
         r"\caption{\textbf{Per-setting permutation test of the linear probe "
         r"($B{=}100$, same 5-fold protocol), per encoder seed.} $p=(1+\#\{\mathrm{null}\ge"
         r"\mathrm{real}\})/(B+1)$. Rejection rates are mean $\pm$ std across seeds; per-seed "
         r"counts given as $c_{42}/c_{123}/c_{456}$ out of $N$. The linear-probe column is from "
         r"the canonical probe pipeline. Under the null, $\sim$1\% of settings reach $p<0.01$ "
         r"by chance.}",
         r"\label{tab:permutation_test}",
         r"\begin{tabular}{lrcrrl}", r"\toprule",
         r"Dataset & $N$ & Mean linear probe & \% $p<0.01$ & \% $p<0.05$ & Bottleneck \\",
         r"\midrule"]
    for _, r in perm.iterrows():
        ds = r.dataset
        rc = color.get(lab[ds])
        if rc: L.append(r"\rowcolor{" + rc + "}")
        pmv = pm(*probe[ds])
        c01 = r.counts_p01
        c05 = r.counts_p05
        L.append(f"{esc(ds)} & {int(r.n_settings)} & {pmv} & "
                 f"{r.rate_p01_mean:.1f} $\\pm$ {r.rate_p01_std:.1f} ({c01}) & "
                 f"{r.rate_p05_mean:.1f} $\\pm$ {r.rate_p05_std:.1f} ({c05}) & {lab[ds]} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"

# ---------------- regression tables ----------------
REP = {"Cosine":"Cosine (Distance)","MDS-BIC":"MDS-BIC (Density)",
       "IForest":"IForest (Isolation)","ViM":"ViM (Boundary / Hybrid)",
       "Linear probe":"Linear probe (Supervised)","MLP probe":"MLP probe (Supervised)"}
def _ols_row(r):
    return (f"{r.beta_all:+.3f} $\\pm$ {r.beta_all_sd:.3f} & {r.r2_all:.2f} $\\pm$ {r.r2_all_sd:.2f} & "
            f"{r.beta_deg:+.3f} $\\pm$ {r.beta_deg_sd:.3f} & {r.r2_deg:.2f} $\\pm$ {r.r2_deg_sd:.2f}")
def t_regression_compacted():
    # Main-text table: single OLS fit on the SEED-AVERAGED per-setting points
    # (degraded defined on seed-mean MAE, N_deg=153).
    sa = pd.read_csv(os.path.join(CO, "agg_ols_seedavg.csv")).set_index("detector")
    L = [r"\begin{table}[h]", r"\centering",
         r"\caption{\textbf{Regression of detector AUROC against $\boldsymbol{\log_{10}"
         r"(\mathrm{MAE}_{\mathrm{OOD}}/\mathrm{MAE}_{\mathrm{ID}})}$.} OLS slope $\beta$ and "
         r"$R^2$ for one representative detector per family plus the supervised probes, fit on "
         r"the seed-averaged per-setting points (AUROC and MAE ratio averaged over the three "
         r"encoder seeds) for all $N_{all}=264$ settings and the degraded subset "
         r"$N_{deg}=153$. Per-seed fits ($\beta\pm$std) are in Appendix "
         r"Table~\ref{tab:regression_auroc_mae_gap}.}",
         r"\label{tab:regression_compacted}", r"\small",
         r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Detector & $\beta$ (all) & $R^2$ (all) & $\beta$ (degraded) & $R^2$ (degraded) \\",
         r"\midrule"]
    for disp in ["Cosine","MDS-BIC","IForest","ViM","Linear probe","MLP probe"]:
        r = sa.loc[disp]
        L.append(f"{REP[disp]} & ${r.beta_all:+.3f}$ & {r.r2_all:.2f} & "
                 f"${r.beta_deg:+.3f}$ & {r.r2_deg:.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"
def t_regression_full():
    order = [C.DISPLAY[k] for k in C.UNSUP_KEYS] + ["Linear probe","MLP probe"]
    L = [r"\begin{table}[h]", r"\centering", r"\small",
         r"\caption{\textbf{Full per-detector OLS of AUROC against "
         r"$\log_{10}(\mathrm{MAE}_{\mathrm{OOD}}/\mathrm{MAE}_{\mathrm{ID}})$} "
         r"(mean $\pm$ std over three encoder seeds), on all $N_{all}=264$ settings and the "
         r"degraded subset.}",
         r"\label{tab:regression_auroc_mae_gap}",
         r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Detector & $\beta$ (all) & $R^2$ (all) & $\beta$ (degraded) & $R^2$ (degraded) \\",
         r"\midrule"]
    for disp in order:
        r = ols[ols.detector == disp].iloc[0]
        L.append(f"{esc(disp)} & {_ols_row(r)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"

def t_per_detector_full():
    deg = pd.read_csv(os.path.join(CO, "agg_detector_degraded.csv")).set_index("detector")
    det["family"] = pd.Categorical(det.family, FAM_ORDER, ordered=True)
    d = det.sort_values("family", kind="stable")
    uns = d[d.family != "Supervised"]
    BEST, SECOND = "1E8449", "2874A6"
    def rank(series, mode="max"):
        vals = sorted({round(float(v), 3) for v in series}, reverse=(mode == "max"))
        return (vals[0], vals[1] if len(vals) > 1 else None)
    cols = {"mean": rank(uns["mean"]), "median": rank(uns["median"]),
            "degr": rank([deg.loc[r.detector, "mean_deg"] for _, r in uns.iterrows()]),
            "gap": rank([deg.loc[r.detector, "probe_gap"] for _, r in uns.iterrows()], "min"),
            "win": rank(uns["win_count"]), "pct": rank(uns["pct"])}
    def col(v, key, fmt):
        b, s = cols[key]; r = round(float(v), 3); txt = format(v, fmt)
        if abs(r - b) < 1e-9: return r"\textcolor[HTML]{"+BEST+r"}{\textbf{"+txt+"}}"
        if s is not None and abs(r - s) < 1e-9: return r"\textcolor[HTML]{"+SECOND+"}{"+txt+"}"
        return txt
    L = [r"% Requires: \usepackage{xcolor} \usepackage{multirow}",
         r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
         r"\caption{\textbf{Per-detector performance summary} (full version with degraded-subset "
         r"and probe-gap columns; the main paper Table~\ref{tab:per_detector_summary} omits these "
         r"for width; central values are means over three encoder seeds). Mean / Median AUROC: "
         r"across all $N_{all}=264$ settings. Mean AUROC (degraded): restricted to the "
         r"$N_{deg}=153$ settings where seed-mean OOD MAE $>$ seed-mean ID MAE. Probe gap: mean "
         r"per-setting (linear probe AUROC $-$ this AUROC); smaller is better. Win count: ties "
         r"split equally. \textcolor[HTML]{1E8449}{\textbf{Bold green}}: best among unsupervised; "
         r"\textcolor[HTML]{2874A6}{blue}: second.}",
         r"\label{tab:per_detector_summary_full}",
         r"\begin{tabular}{llcccccc}", r"\toprule",
         r"Family & Detector & Mean & Median & Mean (degraded) & Probe gap & Wins & \%$>$0.55 \\",
         r"\midrule"]
    fams = [f for f in FAM_ORDER if (d.family == f).any()]
    for fi, fam in enumerate(fams):
        sub = d[d.family == fam]; n = len(sub)
        for i, (_, r) in enumerate(sub.iterrows()):
            fc = (r"\multirow{%d}{*}{%s}" % (n, FAM_DISPLAY[fam])) if i == 0 else ""
            if fam == "Supervised":
                dm = deg_sup(r.detector);
                L.append(f"{fc} & {esc(r.detector)} & {r['mean']:.3f} & {r['median']:.3f} & "
                         f"{dm:.3f} & -- & -- & {r.pct:.1f}\\% \\\\")
            else:
                dm = deg.loc[r.detector, "mean_deg"]; gp = deg.loc[r.detector, "probe_gap"]
                L.append(f"{fc} & {esc(r.detector)} & {col(r['mean'],'mean','.3f')} & "
                         f"{col(r['median'],'median','.3f')} & {col(dm,'degr','.3f')} & "
                         f"+{col(gp,'gap','.3f')} & {col(r.win_count,'win','.1f')} & "
                         f"{col(r.pct,'pct','.1f')}\\% \\\\")
        L.append(r"\midrule" if fam == "Boundary-Hybrid" else (r"\cmidrule(lr){1-8}" if fi < len(fams)-1 else ""))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"

def deg_sup(name):
    # supervised degraded-subset mean from seed-averaged cells
    c = pd.read_csv(os.path.join(CO, "per_seed_cells.csv"))
    S = c.groupby(["dataset","arch","split"]).agg({"linear_probe":"mean","mlp_probe":"mean","mae_id":"mean","mae_ood":"mean"}).reset_index()
    S = S[S.mae_ood > S.mae_id]
    return S.linear_probe.mean() if name == "Linear probe" else S.mlp_probe.mean()

for fn, gen in [("per_detector_summary.tex", t_per_detector),
                ("appendix_per_detector_summary_full.tex", t_per_detector_full),
                ("bottleneck_summary.tex", t_bottleneck),
                ("per_arch_probe.tex", t_per_arch_probe),
                ("table13_probe_column.tex", t_perm_table),
                ("regression_compacted.tex", t_regression_compacted),
                ("regression_auroc_mae_gap.tex", t_regression_full)]:
    try:
        text = gen()
    except FileNotFoundError as e:  # an input that is not part of this release
        print(f"SKIPPED tables/{fn}: missing input {os.path.basename(e.filename)}")
        continue
    open(os.path.join(TB, fn), "w").write(text)
    print("wrote tables/", fn)
# also CSV of perm probe column for convenience
perm.merge(pdp[["dataset","linear","linear_sd","label_mean"]], on="dataset").to_csv(
    os.path.join(TB, "table13_probe_column.csv"), index=False)
print("wrote tables/table13_probe_column.csv")
