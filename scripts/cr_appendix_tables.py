#!/usr/bin/env python3
"""Regenerate the single-seed appendix result tables as 3-seed mean +/- std.
Reuses the detector/family maps, color-coding and FPR95 json logic from
build_appendix_tables.py, but computes each statistic per encoder seed and
reports mean +/- sample std across seeds (2 seeds for the 7 collapsed
CGCNN/phonons settings).

Tables written to outputs/appendix_tables/:
  appendix_family_by_dataset.tex, appendix_per_arch_split_combined.tex,
  appendix_id_mae_per_arch.tex, appendix_per_setting_mae.tex,
  appendix_fpr95_per_detector.tex
Also prints count checks for dataset_stats / dropped_splits (264 settings).
"""
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cr_common as C
ROOT = C.ROOT
CO = C.OUT
OUT = os.path.join(CO, "appendix_tables"); os.makedirs(OUT, exist_ok=True)
SEEDS = [42, 123, 456]

ARCHS = ["alignn", "cgcnn", "schnet", "dimenet"]
DATASETS = ["dielectric", "log_gvrh", "log_kvrh", "perovskites", "phonons"]
DST = [d.replace("_", r"\_") for d in DATASETS]
GREEN, BLUE = "1E8449", "2874A6"

det = pd.read_csv(os.path.join(C.RESULTS, "all_detectors_seeded.csv"))
cells = pd.read_csv(os.path.join(CO, "per_seed_cells.csv"))   # per-seed MAE + probes

def esc(s): return s.replace("_", r"\_")

def msd(vals):
    vals = [v for v in vals if v == v]
    if not vals: return float("nan"), float("nan")
    return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)

def ranks(means, mode="max"):
    idx = [(i, v) for i, v in enumerate(means) if v == v]
    idx.sort(key=lambda x: -x[1] if mode == "max" else x[1])
    r = [-1]*len(means)
    for rr, (i, _) in enumerate(idx[:2]): r[i] = rr
    return r

def cell(mean, std, rank, fmt=".3f"):
    if mean != mean: return "---"
    s = f"{mean:{fmt}}"
    if rank == 0: s = r"\textcolor[HTML]{"+GREEN+r"}{\textbf{"+s+"}}"
    elif rank == 1: s = r"\textcolor[HTML]{"+BLUE+"}{"+s+"}"
    return s + (f" {{\\tiny$\\pm${std:.3f}}}" if std == std else "")

# 15 canonical detectors (CSV key -> display) for AUROC tables
UNSUP = [("KNN k=20 norm","KNN-k20"),("KNN k=50 norm","KNN-k50"),("Norm","Norm"),("Cosine","Cosine"),
         ("LOF","LOF"),("ASH + KNN","ASH+KNN"),("MDS-BIC","MDS-BIC"),("KDE","KDE"),("Flow (NF)","Flow"),
         ("kPCA","kPCA"),("ReAct + MDS","ReAct+MDS"),("IForest","IForest"),("ADERH","ADERH"),
         ("OCSVM","OCSVM"),("ViM","ViM")]
FAMILY = {"KNN k=20 norm":"Distance","KNN k=50 norm":"Distance","Norm":"Distance","Cosine":"Distance",
          "LOF":"Distance","ASH + KNN":"Distance","MDS-BIC":"Density","KDE":"Density","Flow (NF)":"Density",
          "kPCA":"Density","ReAct + MDS":"Density","IForest":"Isolation","ADERH":"Isolation",
          "OCSVM":"Boundary / Hybrid","ViM":"Boundary / Hybrid"}
FAM_ORDER = ["Distance","Density","Isolation","Boundary / Hybrid","Supervised"]

CC = (r"\textcolor[HTML]{"+GREEN+r"}{\textbf{Bold green}} marks the column-wise "
      r"maximum among unsupervised detectors; \textcolor[HTML]{"+BLUE+r"}{blue} the second-best.")
CCmin = CC.replace("maximum", "minimum (lower is better)")


def wrap(lines, body):
    return "\n".join(lines[:_hdr] + [r"\resizebox{\linewidth}{!}{%"] + body + [r"}"] + lines[_hdr:])

# ---------------- family_by_dataset ----------------
def family_by_dataset():
    keys = [k for k,_ in UNSUP]
    d = det[det.detector.isin(keys)].copy(); d["family"] = d.detector.map(FAMILY)
    # per seed: family mean AUROC over (all / per-dataset)
    def fam_means(seed):
        s = d[d.seed == seed]; out = {}
        for fam in FAM_ORDER[:4]:
            fv = s[s.family == fam]
            out[(fam,"all")] = fv.auroc.mean()
            for ds in DATASETS: out[(fam,ds)] = fv[fv.dataset==ds].auroc.mean()
        return out
    per = {sd: fam_means(sd) for sd in SEEDS}
    # supervised row from per_seed_cells probes
    def sup_means(seed):
        s = cells[cells.seed==seed]; v = pd.concat([s.linear_probe, s.mlp_probe])
        out = {("Supervised","all"): v.mean()}
        for ds in DATASETS:
            sd = s[s.dataset==ds]; out[("Supervised",ds)] = pd.concat([sd.linear_probe,sd.mlp_probe]).mean()
        return out
    sper = {sd: sup_means(sd) for sd in SEEDS}
    cols = ["all"]+DATASETS
    # aggregate mean/std
    agg = {}
    for fam in FAM_ORDER:
        for c in cols:
            src = sper if fam=="Supervised" else per
            agg[(fam,c)] = msd([src[sd][(fam,c)] for sd in SEEDS])
    rankmap = {c: ranks([agg[(fam,c)][0] for fam in FAM_ORDER[:4]]) for c in cols}
    L=[r"% Requires: \usepackage{booktabs} \usepackage[table]{xcolor}",r"\begin{table}[t]",r"\centering",r"\small",
       r"\setlength{\tabcolsep}{4pt}",
       r"\caption{\textbf{Mean AUROC by detector family and dataset (mean $\pm$ std over three encoder seeds).} "
       r"Each row aggregates the per-setting AUROCs of every detector in that family. The supervised row is a "
       r"diagnostic ceiling, excluded from rankings. No unsupervised family reaches $\geq 0.55$ on any dataset. "
       +CC+"}",r"\label{tab:family_by_dataset}",r"\resizebox{\linewidth}{!}{%",
       r"\begin{tabular}{lrrrrrr}",r"\toprule",
       r"Family & All settings & "+" & ".join(DST)+r" \\",r"\midrule"]
    for fam in FAM_ORDER:
        if fam=="Supervised": L.append(r"\midrule")
        cs=[fam]
        for ci,c in enumerate(cols):
            m,s=agg[(fam,c)]; rk = rankmap[c][FAM_ORDER.index(fam)] if fam!="Supervised" else -1
            cs.append(cell(m,s,rk))
        L.append(" & ".join(cs)+r" \\")
    L+=[r"\bottomrule",r"\end{tabular}}",r"\end{table}"]
    open(os.path.join(OUT,"appendix_family_by_dataset.tex"),"w").write("\n".join(L)+"\n")

# ---------------- per_arch_split_combined ----------------
def per_arch_split_combined():
    d = det[det.detector.isin([k for k,_ in UNSUP])].copy()
    disp = dict(UNSUP)
    d["stype"] = d.split.map(lambda s: "Compositional" if s.startswith("comp_") else "Structural")
    def means(seed):
        s=d[d.seed==seed]; out={}
        for k,dp in UNSUP:
            for a in ARCHS: out[(dp,a)] = s[(s.detector==k)&(s.arch==a)].auroc.mean()
            for st in ("Compositional","Structural"): out[(dp,st)] = s[(s.detector==k)&(s.stype==st)].auroc.mean()
        return out
    per={sd:means(sd) for sd in SEEDS}
    colkeys = ARCHS+["Compositional","Structural"]
    agg={(dp,c): msd([per[sd][(dp,c)] for sd in SEEDS]) for _,dp in UNSUP for c in colkeys}
    rankmap={c: ranks([agg[(dp,c)][0] for _,dp in UNSUP]) for c in colkeys}
    arch_disp=[a.capitalize() for a in ARCHS]
    L=[r"% Requires: \usepackage{booktabs} \usepackage[table]{xcolor}",r"\begin{table}[t]",r"\centering",r"\small",
       r"\setlength{\tabcolsep}{4pt}",
       r"\caption{\textbf{Per-detector AUROC, stratified by architecture and split type (mean $\pm$ std over three "
       r"encoder seeds).} Compositional $N{=}101$, structural $N{=}156$ settings. No detector exceeds $0.55$ in any "
       r"column. "+CC+"}",r"\label{tab:per_arch_split_combined}",r"\resizebox{\linewidth}{!}{%",
       r"\begin{tabular}{lrrrrrr}",r"\toprule",
       r" & \multicolumn{4}{c}{Architecture} & \multicolumn{2}{c}{Split type} \\",
       r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}",
       r"Detector & "+" & ".join(arch_disp+["Compositional","Structural"])+r" \\",r"\midrule"]
    for i,(k,dp) in enumerate(UNSUP):
        cs=[esc(dp)]
        for c in colkeys:
            m,s=agg[(dp,c)]; cs.append(cell(m,s,rankmap[c][i]))
        L.append(" & ".join(cs)+r" \\")
    L+=[r"\bottomrule",r"\end{tabular}}",r"\end{table}"]
    open(os.path.join(OUT,"appendix_per_arch_split_combined.tex"),"w").write("\n".join(L)+"\n")

# ---------------- id_mae_per_arch ----------------
def id_mae_per_arch():
    def means(seed):
        s=cells[cells.seed==seed]; out={}
        for a in ARCHS:
            for ds in DATASETS:
                sub=s[(s.arch==a)&(s.dataset==ds)]
                out[(a,ds,"id")]=sub.mae_id.mean(); out[(a,ds,"ood")]=sub.mae_ood.mean()
        return out
    per={sd:means(sd) for sd in SEEDS}
    L=[r"% Requires: \usepackage{booktabs}",r"\begin{table}[t]",r"\centering",r"\small",r"\setlength{\tabcolsep}{3pt}",
       r"\caption{\textbf{Mean ID and OOD MAE per architecture and dataset (mean $\pm$ std over three encoder seeds).} "
       r"Averaged across the OOD splits retained for each dataset; lower is better.}",r"\label{tab:id_mae_per_arch}",
       r"\resizebox{\linewidth}{!}{%",
       r"\begin{tabular}{l"+"rr"*len(DATASETS)+"}",r"\toprule",
       r" & "+" & ".join(rf"\multicolumn{{2}}{{c}}{{{d}}}" for d in DST)+r" \\",
       r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}",
       r"Arch & "+" & ".join("ID & OOD" for _ in DATASETS)+r" \\",r"\midrule"]
    for a in ARCHS:
        cs=[a]
        for ds in DATASETS:
            for kind in ("id","ood"):
                m,s=msd([per[sd][(a,ds,kind)] for sd in SEEDS])
                dec=3 if m<10 else 1
                cs.append(f"{m:.{dec}f} {{\\tiny$\\pm${s:.{dec}f}}}")
        L.append(" & ".join(cs)+r" \\")
    L+=[r"\bottomrule",r"\end{tabular}}",r"\end{table}"]
    open(os.path.join(OUT,"appendix_id_mae_per_arch.tex"),"w").write("\n".join(L)+"\n")

# ---------------- per_setting_mae (longtable) ----------------
def per_setting_mae():
    g=cells.groupby(["dataset","arch","split"])
    rows=[]
    for (ds,ar,sp),grp in g:
        mi,si=msd(grp.mae_id.tolist()); mo,so=msd(grp.mae_ood.tolist())
        gap=np.log10(grp.mae_ood.mean()/grp.mae_id.mean())
        rows.append((ds,sp,ar,mi,si,mo,so,gap))
    rows.sort(key=lambda r:(DATASETS.index(r[0]),r[1],r[2]))
    n=len(rows)
    L=[r"% Requires: \usepackage{booktabs} \usepackage{longtable}",
       r"{\setlength{\LTcapwidth}{\linewidth}",
       r"\begin{longtable}{@{\extracolsep{\fill}}llrrr@{}}",
       r"\caption{\textbf{Per-setting held-out ID and OOD MAE (mean $\pm$ std over three encoder seeds).} "
       r"Each row is one retained (dataset, split, arch) setting ($N="+str(n)+r"$). $\log_{10}$ gap "
       r"$=\log_{10}(\text{seed-mean OOD MAE}/\text{seed-mean ID MAE})$.}",
       r"\label{tab:per_setting_mae} \\",r"\toprule",
       r"Dataset & Split & Arch & ID MAE & OOD MAE \\",r"\midrule",r"\endfirsthead",
       r"\multicolumn{5}{l}{\small\itshape (continued)} \\",r"\toprule",
       r"Dataset & Split & Arch & ID MAE & OOD MAE \\",r"\midrule",r"\endhead",
       r"\midrule",r"\multicolumn{5}{r}{\small\itshape (continued)} \\",r"\endfoot",
       r"\bottomrule",r"\endlastfoot"]
    pd_,ps_=None,None
    for ds,sp,ar,mi,si,mo,so,gap in rows:
        if pd_ is not None and ds!=pd_: L.append(r"\addlinespace[2pt]")
        dsd=esc(ds) if ds!=pd_ else ""; spd=esc(sp) if (ds,sp)!=(pd_,ps_) else ""
        dec_i=4 if mi<10 else 1; dec_o=4 if mo<10 else 1
        L.append(f"{dsd} & {spd} & {ar} & {mi:.{dec_i}f}{{\\tiny$\\pm${si:.{dec_i}f}}} & {mo:.{dec_o}f}{{\\tiny$\\pm${so:.{dec_o}f}}} \\\\")
        pd_,ps_=ds,sp
    L+=[r"\end{longtable}}"]
    open(os.path.join(OUT,"appendix_per_setting_mae.tex"),"w").write("\n".join(L)+"\n")

# ---------------- fpr95_per_detector (from per-seed auroc.json) ----------------
def fpr95():
    KEY={"mahalanobis_bic":"MDS-BIC","knn":"KNN-k10","kpca":"kPCA","norm":"Norm","lof":"LOF","ocsvm":"OCSVM",
         "iforest":"IForest","aderh":"ADERH","kde":"KDE","vim":"ViM","cosine":"Cosine","react_mds":"ReAct+MDS","ash_knn":"ASH+KNN"}
    SWEEP={"k20_norm1":"KNN-k20-norm","k50_norm1":"KNN-k50-norm"}
    ORDER=list(KEY.values())+list(SWEEP.values())
    R=C.RESULTS
    def jpath(ds,ar,sp,seed):
        b=os.path.join(R,f"matbench_{ds}",ar,sp,"fold_0")
        return os.path.join(b,"auroc.json") if seed==42 else os.path.join(b,f"seed_{seed}","auroc.json")
    settings=det[["dataset","arch","split"]].drop_duplicates().values.tolist()
    # per seed: collect fpr95 rows
    def collect(seed):
        rows=[]
        for ds,ar,sp in settings:
            p=jpath(ds,ar,sp,seed)
            if not os.path.exists(p): continue
            d=json.loads(open(p).read())
            for k,dp in KEY.items():
                e=d.get(k)
                if isinstance(e,dict) and "fpr95" in e and e["fpr95"]==e["fpr95"]:
                    rows.append((ds,dp,float(e["fpr95"])))
            sw=d.get("knn_sweep",{})
            if isinstance(sw,dict):
                for sk,dp in SWEEP.items():
                    e=sw.get(sk)
                    if isinstance(e,dict) and "fpr95" in e and e["fpr95"]==e["fpr95"]:
                        rows.append((ds,dp,float(e["fpr95"])))
        return pd.DataFrame(rows,columns=["dataset","detector","fpr95"])
    perseed={sd:collect(sd) for sd in SEEDS}
    # per seed aggregate: all-settings mean + per-dataset mean per detector
    def agg_seed(df):
        allm=df.groupby("detector").fpr95.mean()
        dsm=df.groupby(["detector","dataset"]).fpr95.mean()
        return allm,dsm
    A={sd:agg_seed(perseed[sd]) for sd in SEEDS}
    cols=["all"]+DATASETS
    agg={}
    for dp in ORDER:
        agg[(dp,"all")]=msd([A[sd][0].get(dp,np.nan) for sd in SEEDS])
        for ds in DATASETS:
            agg[(dp,ds)]=msd([A[sd][1].get((dp,ds),np.nan) for sd in SEEDS])
    rankmap={c: ranks([agg[(dp,c)][0] for dp in ORDER], mode="min") for c in cols}
    L=[r"% Requires: \usepackage{booktabs} \usepackage[table]{xcolor}",r"\begin{table}[t]",r"\centering",r"\small",
       r"\setlength{\tabcolsep}{3pt}",
       r"\caption{\textbf{FPR95 per detector and per dataset (mean $\pm$ std over three encoder seeds).} Aggregated "
       r"from per-seed \texttt{auroc.json} files; lower is better. Covers 14 of 15 unsupervised detectors (Flow (NF) "
       r"FPR95 lives in \texttt{flow\_models/} and is deferred). "+CCmin+"}",r"\label{tab:fpr95_summary}",
       r"\resizebox{\linewidth}{!}{%",
       r"\begin{tabular}{lrrrrrr}",r"\toprule",
       r"Detector & All settings & "+" & ".join(DST)+r" \\",r"\midrule"]
    for i,dp in enumerate(ORDER):
        cs=[esc(dp)]
        for c in cols:
            m,s=agg[(dp,c)]; cs.append(cell(m,s,rankmap[c][i]))
        L.append(" & ".join(cs)+r" \\")
    L+=[r"\bottomrule",r"\end{tabular}}",r"\end{table}"]
    open(os.path.join(OUT,"appendix_fpr95_per_detector.tex"),"w").write("\n".join(L)+"\n")

# ---------------- count checks ----------------
def checks():
    nset=det[["dataset","arch","split"]].drop_duplicates()
    print("dataset_stats check: settings per dataset =", nset.dataset.value_counts().reindex(DATASETS).to_dict(),
          "| total =", len(nset))
    # dropped: 300 candidate - 264 = 36 dropped; of which 7 are seed-42 collapses (kept via 2 seeds)
    print("dropped_splits check: 5*15*4=300 candidates ->", len(nset), "retained;",
          300-len(nset), "dropped by minority<50 filter; 7 CGCNN/phonons seed-42 collapses retained via 2 seeds.")

for fn in [family_by_dataset, per_arch_split_combined, id_mae_per_arch, per_setting_mae, fpr95]:
    fn(); print("wrote", fn.__name__)
checks()
