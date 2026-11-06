"""Risk-coverage analysis (Jaeger et al. ICLR 2023).

For each (dataset, arch, split) combo and each detector, computes:
  - per-sample detector scores (higher = more ID-like)
  - per-sample prediction errors |y_pred - y_true|
  - risk-coverage curve and AURC (area under it)
  - oracle / random / constant baselines
  - SPE = (AURC_random - AURC) / (AURC_random - AURC_oracle)
    SPE = 1.0 matches oracle, 0.0 matches random, < 0 worse than random.

Detectors covered: MDS-BIC, KNN k=10 normalized, LOF, kPCA, OCSVM, IForest,
Norm, Flow (NF). Skips ADERH (external dependency, slow).

Output: <out_dir>/<dataset>/<arch>/<split>/fold_<f>/<detector>/results.json
        <out_dir>/aggregate.csv
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import auc
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.svm import OneClassSVM

ROOT = Path(__file__).resolve().parent.parent
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mood import paths as _P
EMB_ROOT = _P.embeddings
FLOW_ROOT = _P.flow_models
OUT_ROOT = _P.outputs / "risk_coverage"


def detector_mds_bic(train, id_test, ood_test):
    best_K, best_bic, best_gmm = 1, np.inf, None
    for K in [1, 2, 4, 8, 16, 32, 64]:
        if K >= len(train):
            continue
        try:
            g = GaussianMixture(n_components=K, covariance_type="tied",
                                random_state=42, max_iter=200, reg_covar=1e-4)
            g.fit(train)
            b = g.bic(train)
            if b < best_bic:
                best_bic, best_K, best_gmm = b, K, g
        except Exception:
            continue
    if best_gmm is None:
        return None, None
    means = best_gmm.means_
    cov = best_gmm.covariances_
    cov_inv = np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))

    def score(X):
        out = np.zeros(len(X))
        for i, z in enumerate(X):
            out[i] = -min(np.sqrt(max((z - mu) @ cov_inv @ (z - mu), 0))
                          for mu in means)
        return out
    return score(id_test), score(ood_test)


def detector_knn(train, id_test, ood_test, k=10):
    tr = normalize(train).astype(np.float32)
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(tr)

    def score(X):
        Xn = normalize(X).astype(np.float32)
        d, _ = nn.kneighbors(Xn)
        return -d.mean(axis=1)  # higher = closer to train = more ID-like
    return score(id_test), score(ood_test)


def detector_lof(train, id_test, ood_test, k=20):
    lof = LocalOutlierFactor(n_neighbors=k, novelty=True).fit(train)
    return lof.decision_function(id_test), lof.decision_function(ood_test)


def detector_kpca(train, id_test, ood_test, n_components=64, gamma=None):
    n_components = min(n_components, len(train) - 1, train.shape[1])
    if gamma is None:
        gamma = 1.0 / train.shape[1]
    kpca = KernelPCA(n_components=n_components, kernel="rbf", gamma=gamma,
                     fit_inverse_transform=True, random_state=42)
    kpca.fit(train)

    def score(X):
        Z = kpca.transform(X)
        Xrec = kpca.inverse_transform(Z)
        return -np.linalg.norm(X - Xrec, axis=1)
    return score(id_test), score(ood_test)


def detector_ocsvm(train, id_test, ood_test):
    sc = StandardScaler().fit(train)
    tr = sc.transform(train)
    svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1).fit(tr)
    return svm.decision_function(sc.transform(id_test)), \
           svm.decision_function(sc.transform(ood_test))


def detector_iforest(train, id_test, ood_test):
    f = IsolationForest(n_estimators=100, contamination=0.1,
                        random_state=42).fit(train)
    return f.decision_function(id_test), f.decision_function(ood_test)


def detector_norm(train, id_test, ood_test):
    return -np.linalg.norm(id_test, axis=1), -np.linalg.norm(ood_test, axis=1)


def detector_flow(combo_dir):
    """Flow scores are already saved per-sample (log p(x))."""
    s = combo_dir / "fold_0" / "seed_42"
    id_lp = s / "id_log_prob.npy"
    ood_lp = s / "ood_log_prob.npy"
    if not (id_lp.exists() and ood_lp.exists()):
        return None, None
    return np.load(id_lp), np.load(ood_lp)


DETECTORS = [
    ("mds_bic",  detector_mds_bic),
    ("knn_k10",  detector_knn),
    ("lof",      detector_lof),
    ("kpca",     detector_kpca),
    ("ocsvm",    detector_ocsvm),
    ("iforest",  detector_iforest),
    ("norm",     detector_norm),
    # flow handled separately below
]


def compute_risk_coverage(scores, errors, coverages=None):
    """Higher score = more ID-like = keep first."""
    if coverages is None:
        coverages = np.linspace(0.01, 1.0, 100)
    n = len(scores)
    order = np.argsort(-scores)  # descending
    sorted_err = errors[order]
    risks = np.array([sorted_err[: max(1, int(np.ceil(c * n)))].mean()
                      for c in coverages])
    return coverages, risks, float(auc(coverages, risks))


def compute_aurc_set(scores, errors, n_random=20):
    coverages = np.linspace(0.01, 1.0, 100)
    _, risks_actual, aurc_actual = compute_risk_coverage(scores, errors, coverages)
    _, _, aurc_oracle = compute_risk_coverage(-errors, errors, coverages)

    rng = np.random.RandomState(42)
    rand_aurcs = []
    for _ in range(n_random):
        _, _, a = compute_risk_coverage(rng.permutation(scores), errors, coverages)
        rand_aurcs.append(a)
    aurc_random = float(np.mean(rand_aurcs))
    aurc_constant = float(errors.mean() * (coverages.max() - coverages.min()))
    spe = ((aurc_random - aurc_actual) / (aurc_random - aurc_oracle)
           if aurc_random != aurc_oracle else 0.0)
    return {
        "aurc_actual": float(aurc_actual),
        "aurc_oracle": float(aurc_oracle),
        "aurc_random": aurc_random,
        "aurc_constant": aurc_constant,
        "spe": float(spe),
        "coverages": coverages.tolist(),
        "risks": risks_actual.tolist(),
    }


def discover_combos():
    out = []
    for ds in sorted(EMB_ROOT.glob("matbench_*")):
        for arch in sorted(ds.iterdir()):
            if not arch.is_dir():
                continue
            for sp in sorted(arch.iterdir()):
                if not sp.name.startswith(("comp_", "struct_")):
                    continue
                f = sp / "fold_0"
                needed = ["id_test.npy", "ood_test.npy", "train.npy",
                          "id_test_pred.npy", "ood_test_pred.npy",
                          "id_test_targets.npy", "ood_test_targets.npy"]
                if all((f / n).exists() for n in needed):
                    out.append((ds.name.replace("matbench_", ""),
                                arch.name, sp.name))
    return out


def is_degenerate(train, id_test, ood_test):
    return (train.var() < 1e-10 or id_test.var() < 1e-10
            or ood_test.var() < 1e-10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(OUT_ROOT))
    ap.add_argument("--limit", type=int, default=None,
                    help="run on first N combos for testing")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="restrict to these datasets")
    args = ap.parse_args()
    out_root = Path(args.out_dir)

    combos = discover_combos()
    if args.datasets:
        combos = [c for c in combos if c[0] in args.datasets]
    if args.limit:
        combos = combos[: args.limit]
    print(f"discovered {len(combos)} combos")

    agg_rows = []
    t0 = time.time()
    for i, (ds, arch, split) in enumerate(combos):
        emb_dir = EMB_ROOT / f"matbench_{ds}" / arch / split / "fold_0"
        train = np.load(emb_dir / "train.npy")
        id_emb = np.load(emb_dir / "id_test.npy")
        ood_emb = np.load(emb_dir / "ood_test.npy")
        id_pred = np.load(emb_dir / "id_test_pred.npy").ravel()
        ood_pred = np.load(emb_dir / "ood_test_pred.npy").ravel()
        id_tgt = np.load(emb_dir / "id_test_targets.npy").ravel()
        ood_tgt = np.load(emb_dir / "ood_test_targets.npy").ravel()

        if is_degenerate(train, id_emb, ood_emb):
            continue

        id_err = np.abs(id_pred - id_tgt)
        ood_err = np.abs(ood_pred - ood_tgt)
        all_err = np.concatenate([id_err, ood_err])

        for det_name, det_fn in DETECTORS:
            try:
                id_s, ood_s = det_fn(train, id_emb, ood_emb)
            except Exception as e:
                print(f"  [{ds}/{arch}/{split}] {det_name} failed: {e}")
                continue
            if id_s is None:
                continue
            all_s = np.concatenate([id_s, ood_s])
            res = compute_aurc_set(all_s, all_err)
            out_dir = out_root / ds / arch / split / "fold_0" / det_name
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "results.json").write_text(json.dumps(res, indent=2))
            agg_rows.append({
                "dataset": ds, "arch": arch, "split": split, "detector": det_name,
                "aurc_actual": res["aurc_actual"], "aurc_oracle": res["aurc_oracle"],
                "aurc_random": res["aurc_random"], "spe": res["spe"],
                "id_mae": float(id_err.mean()), "ood_mae": float(ood_err.mean()),
            })

        # Flow (use cached log probs)
        flow_dir = FLOW_ROOT / f"matbench_{ds}" / arch / split
        id_s, ood_s = detector_flow(flow_dir)
        if id_s is not None:
            all_s = np.concatenate([id_s, ood_s])
            res = compute_aurc_set(all_s, all_err)
            out_dir = out_root / ds / arch / split / "fold_0" / "flow"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "results.json").write_text(json.dumps(res, indent=2))
            agg_rows.append({
                "dataset": ds, "arch": arch, "split": split, "detector": "flow",
                "aurc_actual": res["aurc_actual"], "aurc_oracle": res["aurc_oracle"],
                "aurc_random": res["aurc_random"], "spe": res["spe"],
                "id_mae": float(id_err.mean()), "ood_mae": float(ood_err.mean()),
            })

        if (i + 1) % 5 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(combos) - i - 1) / rate
            print(f"  [{i+1}/{len(combos)}] {ds}/{arch}/{split} "
                  f"({rate:.1f} combos/s, ETA {eta/60:.1f} min)", flush=True)

    import pandas as pd
    df = pd.DataFrame(agg_rows)
    out_csv = ROOT / "results" / "risk_coverage_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"\nwrote {out_csv}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
