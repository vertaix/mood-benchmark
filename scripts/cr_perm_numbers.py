#!/usr/bin/env python3
"""Aggregate the per-seed permutation test into camera-ready statistics.

Per dataset, per seed: count settings with p<0.01 and p<0.05 (denominator =
valid settings that dataset/seed), z stats. Report rejection rates and mean z
as mean +/- std across seeds; keep per-seed counts for the appendix. Flags any
encoder-bottlenecked dataset that rejects at p<0.01 under any seed.

Appends a permutation section to numbers.md; writes agg_permutation.csv and
agg_permutation_per_seed.csv.
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cr_common as C
ROOT = C.ROOT
CO = C.OUT
ENCODER_DS = ["log_kvrh", "perovskites", "phonons"]
OLD = {"dielectric": "16/56=28.6%, z=+3.49", "log_gvrh": "16/60=26.7%, z=+5.65"}

p = pd.read_csv(os.path.join(CO, "perm_per_seed.csv"))
p = p.dropna(subset=["pvalue"]).copy()
SEEDS = sorted(p.seed.unique())
# dataset total setting count (for the N column; per-seed denominators live in the counts)
_cells = pd.read_csv(os.path.join(CO, "per_seed_cells.csv"))
NSET = _cells[["dataset", "arch", "split"]].drop_duplicates().dataset.value_counts().to_dict()

rows, perseed = [], []
flags = []
for ds in C.DATASETS:
    rate01, rate05, meanz, counts01, counts05, maxz_all = [], [], [], [], [], []
    for s in SEEDS:
        sub = p[(p.dataset == ds) & (p.seed == s)]
        if len(sub) == 0:
            continue
        n = len(sub)
        c01 = int((sub.pvalue < 0.01).sum()); c05 = int((sub.pvalue < 0.05).sum())
        rate01.append(100.0 * c01 / n); rate05.append(100.0 * c05 / n)
        meanz.append(float(sub.z_score.mean())); maxz_all.append(float(sub.z_score.max()))
        counts01.append((s, c01, n)); counts05.append((s, c05, n))
        perseed.append(dict(dataset=ds, seed=s, n=n, n_p01=c01, n_p05=c05,
                            rate01=100.0*c01/n, rate05=100.0*c05/n,
                            mean_z=float(sub.z_score.mean()), max_z=float(sub.z_score.max())))
        if ds in ENCODER_DS and c01 > 0:
            flags.append(f"{ds} seed {s}: {c01}/{n} reject at p<0.01")
    def ms(v):
        return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
    r01m, r01s = ms(rate01); r05m, r05s = ms(rate05); zm, zs = ms(meanz)
    rows.append(dict(dataset=ds, n_settings=NSET.get(ds, counts01[0][2] if counts01 else 0),
        rate_p01_mean=r01m, rate_p01_std=r01s, rate_p05_mean=r05m, rate_p05_std=r05s,
        mean_z_mean=zm, mean_z_std=zs, max_z=max(maxz_all) if maxz_all else float("nan"),
        counts_p01=", ".join(f"{c}/{n}" for _, c, n in counts01),
        counts_p05=", ".join(f"{c}/{n}" for _, c, n in counts05)))
agg = pd.DataFrame(rows)
agg.to_csv(os.path.join(CO, "agg_permutation.csv"), index=False)
pd.DataFrame(perseed).to_csv(os.path.join(CO, "agg_permutation_per_seed.csv"), index=False)

lines = ["", "## Permutation test (B=100, per-seed, mean ± std across seeds)", "",
         "| dataset | N | p<0.01 rate | p<0.05 rate | mean z | max z | per-seed counts p<0.01 | old |",
         "|---|---|---|---|---|---|---|---|"]
for _, r in agg.iterrows():
    old = OLD.get(r.dataset, "--")
    lines.append(f"| {r.dataset} | {int(r.n_settings)} | {r.rate_p01_mean:.1f}% ± {r.rate_p01_std:.1f} | "
                 f"{r.rate_p05_mean:.1f}% ± {r.rate_p05_std:.1f} | {r.mean_z_mean:+.2f} ± {r.mean_z_std:.2f} | "
                 f"{r.max_z:+.1f} | {r.counts_p01} | {old} |")
lines.append("")
if flags:
    lines.append("**FLAG — encoder-bottlenecked dataset rejects at p<0.01 under some seed:**")
    for f in flags:
        lines.append(f"- {f}")
else:
    lines.append("**No encoder-bottlenecked dataset rejects at p<0.01 under any seed.**")
lines.append("")
with open(os.path.join(CO, "numbers.md"), "a") as f:
    f.write("\n".join(lines) + "\n")
print("appended permutation section to numbers.md; wrote agg_permutation*.csv")
print(agg[["dataset","rate_p01_mean","rate_p01_std","mean_z_mean","max_z","counts_p01"]].to_string(index=False))
print("FLAGS:", flags if flags else "none")
