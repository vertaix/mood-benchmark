#!/usr/bin/env python3
"""TASK 4: regenerate the 5 data-driven figures from the 3-seed, 264-setting
results, authored at NeurIPS single-column print size so all axis/tick/panel/
legend text is >= 9pt when placed at \\linewidth. Outputs PDF (vector) + 300-dpi
PNG into outputs/figures/, same basenames as the paper.

Data: outputs/per_seed_cells.csv (seed-averaged per setting) and
outputs/agg_ols_grid.csv (per-seed-mean OLS beta/R^2).
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cr_common as C
ROOT = C.ROOT
CO = C.OUT
FIG = os.path.join(CO, "figures"); os.makedirs(FIG, exist_ok=True)
LINEWIDTH_IN = 5.5   # NeurIPS single-column text width

# >=9pt at print size: author at native width == LINEWIDTH so pt==effective pt.
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.dpi": 200, "savefig.dpi": 300, "axes.linewidth": 0.8,
})

DET_DISPLAY = {"MDS-BIC":"MDS-BIC","KDE":"KDE","Flow (NF)":"Flow","kPCA":"kPCA",
    "ReAct + MDS":"ReAct+MDS","KNN k=20 norm":"KNN k=20","KNN k=50 norm":"KNN k=50",
    "Norm":"Norm","Cosine":"Cosine","LOF":"LOF","ASH + KNN":"ASH+KNN",
    "IForest":"IForest","ADERH":"ADERH","OCSVM":"OCSVM","ViM":"ViM"}
FAM = {"MDS-BIC":"Density","KDE":"Density","Flow":"Density","kPCA":"Density","ReAct+MDS":"Density",
    "KNN k=20":"Distance","KNN k=50":"Distance","Norm":"Distance","Cosine":"Distance","LOF":"Distance","ASH+KNN":"Distance",
    "IForest":"Isolation","ADERH":"Isolation","OCSVM":"Boundary / Hybrid","ViM":"Boundary / Hybrid",
    "Linear probe":"Supervised","MLP probe":"Supervised"}
FAM_EDGE = {"Density":"#1f77b4","Distance":"#2ca02c","Isolation":"#17becf","Boundary / Hybrid":"#f1c40f","Supervised":"#9467bd"}
FAM_FILL = {"Density":"#eaf3fa","Distance":"#ebf5eb","Isolation":"#e0f5f7","Boundary / Hybrid":"#fff7d6","Supervised":"#f1ebf6"}
DS_COLOR = {"dielectric":"#E63946","log_gvrh":"#F77F00","log_kvrh":"#06A77D","perovskites":"#118AB2","phonons":"#7209B7"}
DS_ORDER = ["dielectric","log_gvrh","log_kvrh","perovskites","phonons"]
INK = "#222222"; CHANCE="blue"

# ---- seed-averaged per-setting frame ----
cells = pd.read_csv(os.path.join(CO, "per_seed_cells.csv"))
detcols = list(DET_DISPLAY.keys())
agg = {c: "mean" for c in detcols + ["linear_probe","mlp_probe","mae_id","mae_ood"]}
S = cells.groupby(["dataset","arch","split"]).agg(agg).reset_index()
S = S.rename(columns={**DET_DISPLAY, "linear_probe":"Linear probe","mlp_probe":"MLP probe"})
S["mae_gap"] = np.log10(S.mae_ood / S.mae_id)
S["degraded"] = S.mae_ood > S.mae_id
ols = pd.read_csv(os.path.join(CO, "agg_ols_grid.csv")).set_index("detector")
NALL = len(S)
print("seed-averaged settings:", NALL)

def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", dpi=300, facecolor="white")
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name + ".{png,pdf}")

# =================== Fig 2: auroc_distribution ===================
def fig_auroc_distribution():
    order = ["MDS-BIC","KDE","Flow","kPCA","ReAct+MDS","KNN k=20","KNN k=50","Norm",
             "Cosine","LOF","ASH+KNN","IForest","ADERH","OCSVM","ViM","Linear probe","MLP probe"]
    data = [S[d].values for d in order]
    fig, ax = plt.subplots(figsize=(LINEWIDTH_IN, 3.1))
    bp = ax.boxplot(data, positions=range(len(order)), widths=0.6, patch_artist=True,
        showfliers=True, flierprops=dict(marker="o", markersize=1.6, markerfacecolor="#555",
        markeredgecolor="none", alpha=0.5), medianprops=dict(color=INK, linewidth=1.1),
        boxprops=dict(linewidth=0.7), whiskerprops=dict(linewidth=0.7, color="#555"),
        capprops=dict(linewidth=0.7, color="#555"), zorder=3)
    for patch, det in zip(bp["boxes"], order):
        f = FAM[det]; patch.set_facecolor(FAM_FILL[f]); patch.set_edgecolor(FAM_EDGE[f]); patch.set_linewidth(0.9)
    ax.axhline(0.5, ls="--", color=CHANCE, lw=0.8, alpha=0.6, zorder=10)
    ax.text(len(order)-0.4, 0.5, "chance", ha="left", va="center", fontsize=8, color=CHANCE)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("AUROC"); ax.set_ylim(0, 1.0)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#eee", lw=0.5, zorder=0)
    pi = [order.index("Linear probe"), order.index("MLP probe")]
    ax.add_patch(Ellipse((sum(pi)/2, 0.99), width=(max(pi)-min(pi))+1.0, height=0.06,
        facecolor="none", edgecolor="#d62728", lw=1.2, zorder=12, clip_on=False))
    handles = [Patch(facecolor=FAM_FILL[f], edgecolor=FAM_EDGE[f], label=f)
               for f in ["Density","Distance","Isolation","Boundary / Hybrid","Supervised"]]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5,-0.42), ncol=3,
              fontsize=8, frameon=False, handlelength=1.3)
    save(fig, "auroc_distribution")

# =================== Fig 3: family_analysis ===================
def fig_family_analysis():
    # Computed PER SEED then mean +/- std across seeds, to match numbers.md / tables
    # (family-max and win-rate are nonlinear, so seed-averaging first would differ).
    fams = ["Distance","Density","Isolation","Boundary / Hybrid","Supervised"]
    fam_dets = {"Distance":["KNN k=20 norm","KNN k=50 norm","Norm","Cosine","LOF","ASH + KNN"],
                "Density":["MDS-BIC","KDE","Flow (NF)","kPCA","ReAct + MDS"],
                "Isolation":["IForest","ADERH"],"Boundary / Hybrid":["OCSVM","ViM"],
                "Supervised":["linear_probe","mlp_probe"]}
    seeds = sorted(cells.seed.unique())
    ps_fmax = {f: [] for f in fams}; ps_wr = {f: [] for f in fams}
    for s in seeds:
        cs = cells[cells.seed == s]
        fmaxcol = {f: cs[fam_dets[f]].max(axis=1) for f in fams}
        for f in fams:
            ps_fmax[f].append(fmaxcol[f].mean())
        fm = pd.DataFrame(fmaxcol); win = fm.idxmax(axis=1).value_counts()/len(cs)*100
        for f in fams:
            ps_wr[f].append(float(win.get(f, 0)))
    means = {f: float(np.mean(ps_fmax[f])) for f in fams}
    sems = {f: float(np.std(ps_fmax[f], ddof=1)) for f in fams}
    wr = pd.Series({f: float(np.mean(ps_wr[f])) for f in fams})
    wr_sd = {f: float(np.std(ps_wr[f], ddof=1)) for f in fams}
    order = sorted(fams, key=lambda f: -means[f])
    fig, axes = plt.subplots(1, 2, figsize=(LINEWIDTH_IN, 3.3))
    lab = [f.replace("Boundary / Hybrid","Bnd/Hyb").replace("Supervised","Superv.") for f in order]
    ax = axes[0]
    ax.bar(range(len(order)), [means[f] for f in order], yerr=[sems[f] for f in order],
           color=[FAM_FILL[f] for f in order], edgecolor=[FAM_EDGE[f] for f in order],
           linewidth=1.0, capsize=2.5, error_kw=dict(elinewidth=0.7))
    for i,f in enumerate(order):
        ax.text(i, means[f]+sems[f]+0.004, f"{means[f]:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.axhline(0.5, ls="--", color=CHANCE, lw=0.8, alpha=0.6)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(lab, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("Mean family-max AUROC", fontsize=10); ax.set_ylim(0.48, max(means.values())+0.015)
    ax.set_title("(A) Family-max AUROC", fontsize=10); ax.grid(axis="y", color="#eee", lw=0.5)
    ax.tick_params(axis="y", labelsize=9)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    ax = axes[1]
    ax.bar(range(len(order)), [wr.get(f,0) for f in order], yerr=[wr_sd[f] for f in order],
           color=[FAM_FILL[f] for f in order], edgecolor=[FAM_EDGE[f] for f in order],
           linewidth=1.0, capsize=2.5, error_kw=dict(elinewidth=0.7))
    for i,f in enumerate(order):
        ax.text(i, wr.get(f,0)+wr_sd[f]+1.0, f"{wr.get(f,0):.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.axhline(20, ls=":", color="#777", lw=0.8); ax.text(len(order)-0.4, 21.5, "chance 20%", ha="right", fontsize=9, color="#777")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(lab, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("Win rate (%)", fontsize=10); ax.set_ylim(0, float(wr.max())+11)
    ax.set_title("(B) Win rate", fontsize=10); ax.grid(axis="y", color="#eee", lw=0.5)
    ax.tick_params(axis="y", labelsize=9)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    fig.tight_layout(w_pad=1.4)
    save(fig, "family_analysis")

# =================== Fig 4: bottleneck_dichotomy ===================
def fig_bottleneck_dichotomy():
    fig, axes = plt.subplots(1, 3, figsize=(LINEWIDTH_IN, 2.5))
    short = {"dielectric":"diel","log_gvrh":"gvrh","log_kvrh":"kvrh","perovskites":"perov","phonons":"phon"}
    # A: linear vs mlp
    ax = axes[0]
    for ds in DS_ORDER:
        d = S[S.dataset==ds]
        ax.scatter(d["Linear probe"], d["MLP probe"], s=6, color=DS_COLOR[ds], alpha=0.75, edgecolor="none", label=short[ds])
    ax.plot([0.3,1],[0.3,1], ls="--", color="#999", lw=0.7)
    ax.set_xlabel("Linear AUROC", fontsize=9); ax.set_ylabel("MLP AUROC", fontsize=9)
    ax.set_title("(A) Linear vs MLP", fontsize=9.5); ax.set_xlim(0.3,1.0); ax.set_ylim(0.3,1.0)
    ax.tick_params(labelsize=9); ax.locator_params(nbins=4)
    # B: linear vs cosine
    ax = axes[1]
    for ds in DS_ORDER:
        d = S[S.dataset==ds]
        ax.scatter(d["Linear probe"], d["Cosine"], s=6, color=DS_COLOR[ds], alpha=0.75, edgecolor="none")
    ax.axhline(0.5, ls="--", color=CHANCE, lw=0.7, alpha=0.6); ax.axvline(0.55, ls=":", color="#777", lw=0.7)
    ax.set_xlabel("Linear AUROC", fontsize=9); ax.set_ylabel("Cosine AUROC", fontsize=9)
    ax.set_title("(B) Probe vs best det.", fontsize=9.5); ax.set_xlim(0.3,1.0); ax.set_ylim(0.4,0.8)
    ax.tick_params(labelsize=9); ax.locator_params(nbins=4)
    # C: linear probe by dataset (violin+strip)
    ax = axes[2]
    data = [S[S.dataset==ds]["Linear probe"].values for ds in DS_ORDER]
    parts = ax.violinplot(data, positions=range(len(DS_ORDER)), showextrema=False, widths=0.8)
    for i,b in enumerate(parts['bodies']): b.set_facecolor(DS_COLOR[DS_ORDER[i]]); b.set_alpha(0.3); b.set_edgecolor("none")
    for i,ds in enumerate(DS_ORDER):
        y = S[S.dataset==ds]["Linear probe"].values
        ax.scatter(np.full(len(y), i)+np.random.RandomState(0).uniform(-0.12,0.12,len(y)), y, s=3, color=DS_COLOR[ds], alpha=0.6, edgecolor="none")
    ax.axhline(0.55, ls=":", color="#777", lw=0.7); ax.axhline(0.5, ls="--", color=CHANCE, lw=0.7, alpha=0.6)
    ax.set_xticks(range(len(DS_ORDER))); ax.set_xticklabels([short[d] for d in DS_ORDER], rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Linear AUROC", fontsize=9); ax.set_title("(C) Probe by task", fontsize=9.5); ax.tick_params(labelsize=9)
    axes[0].legend(loc="lower right", fontsize=9, frameon=True, markerscale=1.4, handletextpad=0.2, labelspacing=0.2, borderpad=0.25)
    for ax in axes:
        for sp in ("top","right"): ax.spines[sp].set_visible(False)
    fig.tight_layout(w_pad=1.1)
    save(fig, "bottleneck_dichotomy")

# =================== Fig 5: auroc_vs_mae_gap_compacted ===================
ARCH_COLOR = {"alignn":"#1f77b4","cgcnn":"#d62728","schnet":"#2ca02c","dimenet":"#9467bd"}
DS_MARK = {"dielectric":"o","log_gvrh":"^","log_kvrh":"s","perovskites":"D","phonons":"v"}
def _fit(sub, col):
    x = sub.mae_gap.values; y = sub[col].values
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    b1, b0 = np.polyfit(x, y, 1)
    return b1, b0, x.min(), x.max()
def _panel(ax, col, title, fam):
    for ds in DS_ORDER:
        for ar in ["alignn","cgcnn","schnet","dimenet"]:
            d = S[(S.dataset==ds)&(S.arch==ar)]
            ax.scatter(d.mae_gap, d[col], s=5, marker=DS_MARK[ds], color=ARCH_COLOR[ar], alpha=0.6, edgecolor="none")
    # blue=all (N=264), red=degraded (N=153); OLS fit on the seed-averaged points.
    for sub, c in [(S, CHANCE), (S[S.degraded], "#d62728")]:
        b1, b0, lo, hi = _fit(sub, col)
        x0 = np.linspace(lo, hi, 50); ax.plot(x0, b1*x0+b0, color=c, lw=1.3)
    ax.axhline(0.5, ls="--", color="#999", lw=0.6); ax.axvline(0, ls=":", color="#999", lw=0.6)
    ax.set_title(f"{title}", fontsize=8.5, color=FAM_EDGE[fam], fontweight="bold")
    ax.tick_params(labelsize=7.5); ax.set_ylim(0.3, 0.95)
def fig_mae_gap_compacted():
    fig, axes = plt.subplots(1, 4, figsize=(LINEWIDTH_IN, 1.9))
    _panel(axes[0], "Cosine", "Cosine (Distance)", "Distance")
    _panel(axes[1], "MDS-BIC", "MDS-BIC (Density)", "Density")
    _panel(axes[2], "IForest", "IForest (Isolation)", "Isolation")
    _panel(axes[3], "ViM", "ViM (Bnd/Hyb)", "Boundary / Hybrid")
    axes[0].set_ylabel("AUROC", fontsize=8.5)
    for ax in axes: ax.set_xlabel(r"$\log_{10}$(OOD/ID MAE)", fontsize=7.5)
    from matplotlib.lines import Line2D
    leg = [Line2D([0],[0],color=CHANCE,lw=1.3,label=f"all ($N{{=}}{NALL}$)"),
           Line2D([0],[0],color="#d62728",lw=1.3,label=f"degraded ($N{{=}}{int(S.degraded.sum())}$)")]
    axes[3].legend(handles=leg, loc="lower right", fontsize=6, frameon=True)
    for ax in axes:
        for sp in ("top","right"): ax.spines[sp].set_visible(False)
    fig.tight_layout(w_pad=0.6)
    save(fig, "auroc_vs_mae_gap_compacted")

# =================== Appendix: auroc_vs_mae_gap_full (3x5 grid + probes) ===================
def fig_mae_gap_full():
    order = [("Cosine","Distance"),("KNN k=20","Distance"),("KNN k=50","Distance"),("Norm","Distance"),
             ("LOF","Distance"),("ASH+KNN","Distance"),("MDS-BIC","Density"),("KDE","Density"),
             ("Flow","Density"),("kPCA","Density"),("ReAct+MDS","Density"),("IForest","Isolation"),
             ("ADERH","Isolation"),("OCSVM","Boundary / Hybrid"),("ViM","Boundary / Hybrid"),
             ("Linear probe","Supervised"),("MLP probe","Supervised")]
    # map display -> ols index
    OLSKEY = {"Cosine":"Cosine","KNN k=20":"KNN-k20-norm","KNN k=50":"KNN-k50-norm","Norm":"Norm",
              "LOF":"LOF","ASH+KNN":"ASH+KNN","MDS-BIC":"MDS-BIC","KDE":"KDE","Flow":"Flow","kPCA":"kPCA",
              "ReAct+MDS":"ReAct+MDS","IForest":"IForest","ADERH":"ADERH","OCSVM":"OCSVM","ViM":"ViM",
              "Linear probe":"Linear probe","MLP probe":"MLP probe"}
    ncol=4; nrow=5
    fig, axes = plt.subplots(nrow, ncol, figsize=(LINEWIDTH_IN, LINEWIDTH_IN*1.15))
    axes=axes.ravel()
    for i,(col,fam) in enumerate(order):
        ax=axes[i]
        for ds in DS_ORDER:
            for ar in ARCH_COLOR:
                d=S[(S.dataset==ds)&(S.arch==ar)]
                ax.scatter(d.mae_gap, d[col], s=3, marker=DS_MARK[ds], color=ARCH_COLOR[ar], alpha=0.55, edgecolor="none")
        for sub,c in [(S,CHANCE),(S[S.degraded],"#d62728")]:
            b1,b0,lo,hi=_fit(sub,col); x0=np.linspace(lo,hi,30); ax.plot(x0,b1*x0+b0,color=c,lw=1.0)
        ax.axhline(0.5,ls="--",color="#999",lw=0.5); ax.axvline(0,ls=":",color="#999",lw=0.5)
        ax.set_title(col, fontsize=7, color=FAM_EDGE[fam], fontweight="bold", pad=2)
        ax.tick_params(labelsize=6); ax.set_ylim(0.25,1.0)
    for j in range(len(order), nrow*ncol): axes[j].set_visible(False)
    fig.supxlabel(r"$\log_{10}$(OOD MAE / ID MAE)", fontsize=9); fig.supylabel("AUROC", fontsize=9)
    fig.tight_layout()
    save(fig, "auroc_vs_mae_gap_full")

fig_auroc_distribution(); fig_family_analysis(); fig_bottleneck_dichotomy()
fig_mae_gap_compacted(); fig_mae_gap_full()
print("done")
