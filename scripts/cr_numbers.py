#!/usr/bin/env python3
"""TASK 2 step C: aggregate per-seed cells + permutation results into every
reported statistic as mean +/- sample std across seeds, and write numbers.md
plus supporting CSVs for the TASK 3 tables.

Aggregation rule: compute each statistic PER SEED, then mean +/- sample std
(ddof=1) across seeds. The 7 collapsed CGCNN/phonons settings contribute only
seeds 123 & 456 (they have no seed-42 cell).
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cr_common as C
ROOT = C.ROOT
CO = C.OUT

DETS = C.SELECTION                      # (csvkey, display, family)
DET_KEYS = [k for k, *_ in DETS]
FAM_OF = {k: f for k, _, f in DETS}
DISP = {k: d for k, d, _ in DETS}
METHOD_DS = ["dielectric", "log_gvrh"]
ENCODER_DS = ["log_kvrh", "perovskites", "phonons"]

cells = pd.read_csv(os.path.join(CO, "per_seed_cells.csv"))
SEEDS = [int(s) for s in sorted(cells.seed.unique())]
cells["det_mean15"] = cells[DET_KEYS].mean(axis=1)   # per-setting mean of 15 detectors


def msd(vals):
    vals = [v for v in vals if v == v]
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)


def per_seed(fn, seeds=SEEDS):
    """fn(df_one_seed)->scalar; return (mean,std,[per-seed vals])."""
    vals = []
    for s in seeds:
        sub = cells[cells.seed == s]
        vals.append(fn(sub))
    m, sd = msd(vals)
    return m, sd, vals


def ols(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.ptp(x) == 0:
        return float("nan"), float("nan"), len(x)
    b1, b0 = np.polyfit(x, y, 1)
    yhat = b0 + b1 * x
    ss_res = ((y - yhat) ** 2).sum(); ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(b1), float(r2), len(x)


MD = []  # master table rows: (quantity, old, new_mean, new_std)

def add(q, old, m, sd=None):
    MD.append((q, old, m, sd))

# ---------------- Block 1: grid counts ----------------
add("N_all (settings)", "264", 264, 0.0)
ndeg_m, ndeg_sd, ndeg_vals = per_seed(lambda d: int(d.degraded.sum()))
add("N_deg (MAE_OOD>MAE_ID), per seed", "[ ]", ndeg_m, ndeg_sd)
# AUROC count = #valid settings that seed * 17 methods
auroc_counts = {s: int((cells.seed == s).sum()) * 17 for s in SEEDS}
add("AUROC count per seed (settings x 17)",
    "4488/seed", float(np.mean(list(auroc_counts.values()))),
    float(np.std(list(auroc_counts.values()), ddof=1)))
TOTAL_AUROC = sum(auroc_counts.values())

# ---------------- Block 2: per-detector ----------------
OLD_DET = {  # from per_detector_summary.tex (seed-42, N=264): mean, median, pct>0.55
 "KNN-k20-norm":(0.512,0.504,10.1),"KNN-k50-norm":(0.508,0.503,8.2),"Norm":(0.500,0.501,5.1),
 "Cosine":(0.531,0.508,14.0),"LOF":(0.526,0.506,13.2),"ASH+KNN":(0.518,0.506,12.1),
 "MDS-BIC":(0.513,0.504,7.4),"KDE":(0.517,0.505,10.1),"Flow":(0.524,0.503,8.9),
 "kPCA":(0.512,0.503,7.8),"ReAct+MDS":(0.512,0.503,6.6),"IForest":(0.501,0.500,7.4),
 "ADERH":(0.495,0.500,3.9),"OCSVM":(0.495,0.498,7.0),"ViM":(0.508,0.502,7.4)}
det_rows = []
for k in DET_KEYS:
    disp = DISP[k]
    mmean = per_seed(lambda d, k=k: d[k].mean())
    mmed = per_seed(lambda d, k=k: d[k].median())
    mpct = per_seed(lambda d, k=k: 100.0 * (d[k] > C.ABOVE).mean())
    o = OLD_DET[disp]
    add(f"[det] {disp} mean AUROC", f"{o[0]:.3f}", mmean[0], mmean[1])
    add(f"[det] {disp} median AUROC", f"{o[1]:.3f}", mmed[0], mmed[1])
    add(f"[det] {disp} %>0.55", f"{o[2]:.1f}", mpct[0], mpct[1])
    det_rows.append(dict(detector=disp, family=FAM_OF[k],
        mean=mmean[0], mean_sd=mmean[1], median=mmed[0], median_sd=mmed[1],
        pct=mpct[0], pct_sd=mpct[1]))
# probes
for col, name, old in [("linear_probe","Linear probe",(0.546,0.501,17.1)),
                       ("mlp_probe","MLP probe",(0.542,0.493,14.8))]:
    mmean = per_seed(lambda d, c=col: d[c].mean())
    mmed = per_seed(lambda d, c=col: d[c].median())
    mpct = per_seed(lambda d, c=col: 100.0 * (d[c] > C.ABOVE).mean())
    add(f"[probe] {name} mean AUROC", f"{old[0]:.3f}", mmean[0], mmean[1])
    add(f"[probe] {name} median AUROC", f"{old[1]:.3f}", mmed[0], mmed[1])
    add(f"[probe] {name} %>0.55", f"{old[2]:.1f}", mpct[0], mpct[1])
    det_rows.append(dict(detector=name, family="Supervised",
        mean=mmean[0], mean_sd=mmean[1], median=mmed[0], median_sd=mmed[1],
        pct=mpct[0], pct_sd=mpct[1]))
pd.DataFrame(det_rows).to_csv(os.path.join(CO, "agg_per_detector.csv"), index=False)

# ---------------- Block 3: family-max + win rates ----------------
def family_cols(fam):
    return [k for k in DET_KEYS if FAM_OF[k] == fam]
OLD_FMAX = {"Distance":0.554}
for fam in C.FAMILIES:
    cols = family_cols(fam)
    fm = per_seed(lambda d, cols=cols: d[cols].max(axis=1).mean())
    old = f"{OLD_FMAX.get(fam,'?')}" if fam in OLD_FMAX else "--"
    add(f"[family-max] {fam} mean", old, fm[0], fm[1])
sup = per_seed(lambda d: d[["linear_probe","mlp_probe"]].max(axis=1).mean())
add("[family-max] Supervised mean", "0.557", sup[0], sup[1])

# Family win RATE (matches plot_family_analysis.py): per cell take each family's
# family-max (the 4 detector families PLUS Supervised=max(linear,mlp)); the winner
# is the argmax family. Win rate = fraction of cells. 5 families -> 20% baseline.
WR_FAMS = C.FAMILIES + ["Supervised"]
def fammax_matrix(d):
    cols = {}
    for fam in C.FAMILIES:
        cols[fam] = d[[k for k in DET_KEYS if FAM_OF[k] == fam]].max(axis=1)
    cols["Supervised"] = d[["linear_probe", "mlp_probe"]].max(axis=1)
    return pd.DataFrame(cols, index=d.index)[WR_FAMS]
def winrates(d):
    fm = fammax_matrix(d)
    winners = fm.idxmax(axis=1)
    vc = winners.value_counts()
    n = len(d)
    return {f: 100.0 * vc.get(f, 0) / n for f in WR_FAMS}
wr_seed = {s: winrates(cells[cells.seed == s]) for s in SEEDS}
OLD_WR = {"Distance":41}
for fam in WR_FAMS:
    vals = [wr_seed[s][fam] for s in SEEDS]
    m, sd = msd(vals)
    add(f"[win-rate] {fam}", f"{OLD_WR.get(fam,'?')}" if fam in OLD_WR else "--", m, sd)

# Per-detector win COUNT among the 15 unsupervised detectors (1/n tie split),
# for Table 1. Per seed then mean+/-std.
def det_wincounts(d):
    arr = d[DET_KEYS].values
    mx = arr.max(axis=1, keepdims=True)
    w = (arr == mx).astype(float)
    w = w / w.sum(axis=1, keepdims=True)
    return dict(zip([DISP[k] for k in DET_KEYS], w.sum(axis=0)))
wc_seed = {s: det_wincounts(cells[cells.seed == s]) for s in SEEDS}
for r in det_rows:
    if r["family"] == "Supervised":
        r["win_count"], r["win_count_sd"] = float("nan"), float("nan")
    else:
        vals = [wc_seed[s][r["detector"]] for s in SEEDS]
        r["win_count"], r["win_count_sd"] = msd(vals)
pd.DataFrame(det_rows).to_csv(os.path.join(CO, "agg_per_detector.csv"), index=False)

# ---------------- Block 4: per-dataset probe + gap + label ----------------
OLD_PD = {"dielectric":(0.601,0.080),"log_gvrh":(0.608,0.083),
          "log_kvrh":(0.493,None),"perovskites":(0.507,None),"phonons":(0.498,None)}
def label(probe, gap):
    if probe >= 0.55 and gap >= 0.05: return "Method"
    if probe < 0.55: return "Encoder"
    return "Mixed"
pd_rows = []
for ds in C.DATASETS:
    def dstat(fn, ds=ds):
        return per_seed(lambda d, ds=ds, fn=fn: fn(d[d.dataset == ds]))
    lin = dstat(lambda x: x.linear_probe.mean())
    mlp = dstat(lambda x: x.mlp_probe.mean())
    mdet = dstat(lambda x: x.det_mean15.mean())
    cos = dstat(lambda x: x["Cosine"].mean())
    gap_mean = per_seed(lambda d, ds=ds: d[d.dataset==ds].linear_probe.mean() - d[d.dataset==ds].det_mean15.mean())
    gap_best = per_seed(lambda d, ds=ds: d[d.dataset==ds].linear_probe.mean() - d[d.dataset==ds]["Cosine"].mean())
    lab_mean = label(lin[0], gap_mean[0]); lab_best = label(lin[0], gap_best[0])
    o = OLD_PD[ds]
    add(f"[dataset] {ds} linear probe", f"{o[0]:.3f}", lin[0], lin[1])
    add(f"[dataset] {ds} gap(mean-det)", f"+{o[1]:.3f}" if o[1] else "--", gap_mean[0], gap_mean[1])
    add(f"[dataset] {ds} gap(best=Cosine)", "--", gap_best[0], gap_best[1])
    pd_rows.append(dict(dataset=ds, N=int((cells.dataset==ds).drop_duplicates().shape[0]),
        linear=lin[0], linear_sd=lin[1], mlp=mlp[0], mlp_sd=mlp[1],
        mean_det=mdet[0], mean_det_sd=mdet[1], cosine=cos[0], cosine_sd=cos[1],
        gap_mean=gap_mean[0], gap_mean_sd=gap_mean[1],
        gap_best=gap_best[0], gap_best_sd=gap_best[1],
        label_mean=lab_mean, label_best=lab_best))
pdf = pd.DataFrame(pd_rows)
# fix N per dataset
Nmap = cells[["dataset","arch","split"]].drop_duplicates().dataset.value_counts().to_dict()
pdf["N"] = pdf.dataset.map(Nmap)
pdf.to_csv(os.path.join(CO, "agg_per_dataset_probe.csv"), index=False)

# ---------------- Block 5: MLP vs linear ----------------
from scipy.stats import pearsonr
dlt = per_seed(lambda d: (d.linear_probe - d.mlp_probe).abs().mean())
rr = per_seed(lambda d: pearsonr(d.linear_probe, d.mlp_probe)[0])
add("[MLP-lin] mean |delta|", "0.027", dlt[0], dlt[1])
add("[MLP-lin] pearson r", "0.960", rr[0], rr[1])
enc_mask = cells.dataset.isin(ENCODER_DS)
maxmlp = per_seed(lambda d: d[d.dataset.isin(ENCODER_DS)].mlp_probe.max())
add("[MLP-lin] max MLP on encoder-bottleneck (per-seed max)", "0.601", maxmlp[0], maxmlp[1])
add("[MLP-lin] max MLP on encoder-bottleneck (overall)", "0.601",
    float(cells[enc_mask].mlp_probe.max()), None)

# ---------------- Block 7: OLS ----------------
ols_rows = []
OLD_OLS = {"Cosine":(0.279,0.23,0.559,0.41),"MDS-BIC":(0.196,0.18,0.271,0.16),
           "IForest":(0.127,0.14,0.152,0.10),"ViM":(0.158,0.14,0.186,0.09),
           "Linear probe":(0.282,0.11,0.756,0.40),"MLP probe":(0.299,0.11,0.765,0.38)}
for k in DET_KEYS + ["linear_probe","mlp_probe"]:
    disp = DISP.get(k, {"linear_probe":"Linear probe","mlp_probe":"MLP probe"}.get(k,k))
    ba = per_seed(lambda d, k=k: ols(d.mae_gap_log10, d[k])[0])
    ra = per_seed(lambda d, k=k: ols(d.mae_gap_log10, d[k])[1])
    bd = per_seed(lambda d, k=k: ols(d[d.degraded].mae_gap_log10, d[d.degraded][k])[0])
    rd = per_seed(lambda d, k=k: ols(d[d.degraded].mae_gap_log10, d[d.degraded][k])[1])
    ols_rows.append(dict(detector=disp, beta_all=ba[0], beta_all_sd=ba[1], r2_all=ra[0], r2_all_sd=ra[1],
                         beta_deg=bd[0], beta_deg_sd=bd[1], r2_deg=rd[0], r2_deg_sd=rd[1]))
pd.DataFrame(ols_rows).to_csv(os.path.join(CO, "agg_ols_grid.csv"), index=False)
cos_ols = [r for r in ols_rows if r["detector"]=="Cosine"][0]
add("[OLS] Cosine beta (degraded)", "+0.559", cos_ols["beta_deg"], cos_ols["beta_deg_sd"])
add("[OLS] Cosine R2 (degraded)", "0.41", cos_ols["r2_deg"], cos_ols["r2_deg_sd"])

# ---------------- Block 8: seed stability ----------------
# per-setting probe mean/std across its valid seeds
g = cells.groupby(["dataset","arch","split"])
setting_probe = g.linear_probe.agg(["mean","std","count"]).reset_index()
# regime separation per seed: method-regime mean probe - encoder-regime mean probe
sep_vals = []
for s in SEEDS:
    sub = cells[cells.seed==s]
    sep_vals.append(sub[sub.dataset.isin(METHOD_DS)].linear_probe.mean() -
                    sub[sub.dataset.isin(ENCODER_DS)].linear_probe.mean())
# also per-dataset mean probe across seeds and seed-std
ds_probe = cells.groupby(["dataset","seed"]).linear_probe.mean().unstack()
ds_seedstd = ds_probe.std(axis=1, ddof=1)
# settings crossing 0.55
cross = setting_probe.copy()
cross_counts = cells.assign(above=cells.linear_probe>0.55).groupby(["dataset","arch","split"]).above.sum()
cross_valid = cells.groupby(["dataset","arch","split"]).seed.count()
cross_df = pd.DataFrame({"n_above":cross_counts, "n_seeds":cross_valid}).reset_index()
one_seed = cross_df[cross_df.n_above==1]
two_seed = cross_df[cross_df.n_above==2]
all_seed = cross_df[(cross_df.n_above==cross_df.n_seeds)]
# per-arch: settings with every seed below 0.55
per_arch_below = {}
for ar in C.ARCHS:
    sub = cross_df[cross_df.apply(lambda r: r.arch==ar, axis=1)]
    per_arch_below[ar] = int((sub.n_above==0).sum())
seed_std_setting = setting_probe["std"].dropna()
stab = dict(
    regime_sep_min=float(np.min(sep_vals)), regime_sep_max=float(np.max(sep_vals)),
    seed_std_dataset_min=float(ds_seedstd.min()), seed_std_dataset_max=float(ds_seedstd.max()),
    seed_std_setting_min=float(seed_std_setting.min()), seed_std_setting_max=float(seed_std_setting.max()),
    n_cross_one=int(len(one_seed)), n_cross_two=int(len(two_seed)), n_cross_all=int(len(all_seed)),
    max_probe_cell=float(cells.linear_probe.max()),
    per_arch_all_below_055=per_arch_below,
)
import json
with open(os.path.join(CO,"agg_seed_stability.json"),"w") as f: json.dump(
    {**stab,
     "one_seed_settings":[f"{r.dataset}/{r.arch}/{r.split}" for r in one_seed.itertuples()],
     "two_seed_settings":[f"{r.dataset}/{r.arch}/{r.split}" for r in two_seed.itertuples()]}, f, indent=2)
add("[stability] regime separation (min over seeds)","[ ]",stab["regime_sep_min"],None)
add("[stability] regime separation (max over seeds)","[ ]",stab["regime_sep_max"],None)
add("[stability] seed-to-seed std, per-dataset mean probe (min)","[ ]",stab["seed_std_dataset_min"],None)
add("[stability] seed-to-seed std, per-dataset mean probe (max)","[ ]",stab["seed_std_dataset_max"],None)
add("[stability] #settings crossing 0.55 on exactly one seed","[ ]",stab["n_cross_one"],None)
add("[stability] #settings crossing 0.55 on exactly two seeds","[ ]",stab["n_cross_two"],None)
add("[stability] max probe AUROC in any cell","[ ]",stab["max_probe_cell"],None)

# method-bottleneck subset size
mb = int(pdf[pdf.label_mean=="Method"].N.sum())
add("method-bottleneck subset size (settings)","116",mb,None)

# ---------------- write master table ----------------
def fmt(m, sd):
    if m != m: return "nan"
    if sd is None: return f"{m:.4f}".rstrip('0').rstrip('.') if m==int(m) else f"{m:.4f}"
    return f"{m:.4f} ± {sd:.4f}"
lines = ["# MOOD camera-ready — recomputed numbers (mean ± sample std across seeds)",
         "",
         f"- Seeds: {SEEDS}; 7 collapsed CGCNN/phonons use seeds 123 & 456 only.",
         f"- Total AUROC measurements (17 methods × valid settings, summed over seeds): **{TOTAL_AUROC}**  "
         f"(per seed: {auroc_counts}).",
         "",
         "| quantity | old | new mean | new std |","|---|---|---|---|"]
for q, old, m, sd in MD:
    ms = f"{m:.4f}" if isinstance(m,(int,float)) and m==m else str(m)
    sds = "" if sd is None else f"{sd:.4f}"
    lines.append(f"| {q} | {old} | {ms} | {sds} |")
open(os.path.join(CO,"numbers.md"),"w").write("\n".join(lines)+"\n")
print("wrote outputs/numbers.md and agg_*.csv")
print("method-bottleneck subset size:", mb)
print("per-dataset labels (mean gap / best gap):")
print(pdf[["dataset","linear","gap_mean","gap_best","label_mean","label_best"]].to_string(index=False))
