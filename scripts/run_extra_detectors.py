"""Extra OOD detectors that augment an existing auroc.json.

Five additional detectors, all post-hoc on frozen embeddings:

  kde          — KernelDensity on PCA(32)-reduced embeddings
                 (sklearn.neighbors.KernelDensity, gaussian, Scott bandwidth)
  vim          — regression-adapted Virtual-logit Matching: residual norm in
                 the orthogonal complement of training PCA subspace
                 (Wang et al. CVPR 2022, classifier-free formulation)
  cosine       — max cosine similarity to any training embedding
                 (CSI-style baseline; Tack et al. NeurIPS 2020)
  react_mds    — ReAct preprocessing (clip at p90 per-feature) then
                 Mahalanobis-BIC (Sun et al. NeurIPS 2021)
  ash_knn      — ASH-S preprocessing (keep top 10% activations, rescale)
                 then KNN k=20 with L2-normalized embeddings
                 (Djurisic et al. ICLR 2023)

Each cell's auroc.json is read, augmented with these 5 keys, and rewritten.
Existing keys are not modified.

Usage:
  python scripts/run_extra_detectors.py \\
      --dataset matbench_dielectric --architecture alignn --split comp_oxide --fold 0
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
from mood import paths as _P
EMB_DIR = str(_P.embeddings)
RESULTS_DIR = str(_P.results)


# ─── Common scoring helpers (same protocol as run_detectors.py) ─────────
def fpr_at_tpr(id_scores, ood_scores, tpr_threshold=0.95):
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.searchsorted(tpr, tpr_threshold)
    return float(fpr[min(idx, len(fpr) - 1)])


def compute_auroc_balanced(id_scores, ood_scores, seed=42):
    rng = np.random.RandomState(seed)
    n_min = min(len(id_scores), len(ood_scores))
    if n_min < 2:
        return float("nan"), float("nan")
    id_sub = rng.choice(id_scores, n_min, replace=False)
    ood_sub = rng.choice(ood_scores, n_min, replace=False)
    scores = np.concatenate([id_sub, ood_sub])
    labels = np.concatenate([np.ones(n_min), np.zeros(n_min)])
    return float(roc_auc_score(labels, scores)), fpr_at_tpr(id_sub, ood_sub)


# ─── KDE on PCA-reduced embeddings ──────────────────────────────────────
class KDEDetector:
    def __init__(self, n_components=32, bandwidth="scott"):
        self.n_components = n_components
        self.bandwidth = bandwidth

    def fit(self, train_emb):
        self.scaler = StandardScaler().fit(train_emb)
        scaled = self.scaler.transform(train_emb)
        self.pca = PCA(n_components=min(self.n_components, scaled.shape[1])).fit(scaled)
        reduced = self.pca.transform(scaled)
        n, d = reduced.shape
        if self.bandwidth == "scott":
            bw = n ** (-1.0 / (d + 4))
        elif self.bandwidth == "silverman":
            bw = (n * (d + 2) / 4.0) ** (-1.0 / (d + 4))
        else:
            bw = float(self.bandwidth)
        self.kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(reduced)
        return self

    def score(self, emb):
        return self.kde.score_samples(self.pca.transform(self.scaler.transform(emb)))


# ─── ViM (regression-adapted: residual norm in PCA-orthogonal complement) ─
class ViMDetector:
    def __init__(self, variance_threshold=0.95):
        self.variance_threshold = variance_threshold

    def fit(self, train_emb):
        self.mean = train_emb.mean(axis=0)
        centered = train_emb - self.mean
        full = PCA(n_components=None).fit(centered)
        cumvar = np.cumsum(full.explained_variance_ratio_)
        self.n_keep = int(np.searchsorted(cumvar, self.variance_threshold)) + 1
        self.principal = full.components_[: self.n_keep]
        return self

    def score(self, emb):
        centered = emb - self.mean
        proj = centered @ self.principal.T
        recon = proj @ self.principal
        residual = centered - recon
        return -np.linalg.norm(residual, axis=1)  # higher = more ID


# ─── Cosine similarity to nearest training neighbor ─────────────────────
class CosineDetector:
    def __init__(self, k=1):
        self.k = k

    def fit(self, train_emb):
        self.train_norm = normalize(train_emb, axis=1)
        return self

    def score(self, emb):
        test_norm = normalize(emb, axis=1)
        cos = test_norm @ self.train_norm.T
        if self.k == 1:
            return cos.max(axis=1)
        topk = np.partition(cos, -self.k, axis=1)[:, -self.k:]
        return topk.mean(axis=1)


# ─── ReAct wrapper: clip at percentile, then run base detector ──────────
class ReActWrapper:
    def __init__(self, base, percentile=90):
        self.base = base
        self.percentile = percentile

    def fit(self, train_emb):
        self.threshold = np.percentile(train_emb, self.percentile, axis=0)
        rectified = np.minimum(train_emb, self.threshold)
        self.base.fit(rectified)
        return self

    def score(self, emb):
        return self.base.score(np.minimum(emb, self.threshold))


# ─── ASH-S wrapper: prune small activations, rescale to preserve sum ────
class ASHWrapper:
    def __init__(self, base, keep_fraction=0.1):
        self.base = base
        self.keep_fraction = keep_fraction

    def _ash_s(self, x):
        n_features = x.shape[1]
        n_keep = max(1, int(n_features * self.keep_fraction))
        abs_x = np.abs(x)
        # threshold per-row = the (n_features - n_keep)-th smallest |val|
        threshold = np.partition(abs_x, n_features - n_keep, axis=1)[
            :, n_features - n_keep : n_features - n_keep + 1
        ]
        mask = abs_x >= threshold
        original_sum = x.sum(axis=1, keepdims=True)
        pruned = x * mask
        pruned_sum = pruned.sum(axis=1, keepdims=True)
        rescale = np.where(np.abs(pruned_sum) > 1e-10, original_sum / pruned_sum, 1.0)
        return pruned * rescale

    def fit(self, train_emb):
        self.base.fit(self._ash_s(train_emb))
        return self

    def score(self, emb):
        return self.base.score(self._ash_s(emb))


# ─── Bare-bones MDS-BIC and KNN-k20-normalized for use as ReAct/ASH bases ─
class MDSBicDetector:
    """Mahalanobis with K chosen by BIC. Lifted from run_detectors.py to keep
    this module self-contained."""

    def fit(self, train_emb):
        best_K, best_bic, best_gmm = 1, np.inf, None
        for K in [1, 2, 4, 8, 16, 32, 64, 128]:
            if K >= len(train_emb):
                continue
            try:
                g = GaussianMixture(
                    n_components=K, covariance_type="tied",
                    random_state=42, max_iter=200, reg_covar=1e-4,
                ).fit(train_emb)
                b = g.bic(train_emb)
                if b < best_bic:
                    best_bic, best_K, best_gmm = b, K, g
            except Exception:
                continue
        if best_gmm is None:
            self.means = train_emb.mean(0, keepdims=True)
            self.cov_inv = np.linalg.inv(np.cov(train_emb.T) + 1e-6 * np.eye(train_emb.shape[1]))
        else:
            self.means = best_gmm.means_
            cov = best_gmm.covariances_
            self.cov_inv = np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))
        return self

    def score(self, emb):
        out = np.zeros(len(emb))
        for i, z in enumerate(emb):
            dists = []
            for mu in self.means:
                diff = z - mu
                dists.append(np.sqrt(max(diff @ self.cov_inv @ diff, 0)))
            out[i] = -min(dists)
        return out


class KNNk20NormDetector:
    def __init__(self, k=20):
        self.k = k

    def fit(self, train_emb):
        tr = normalize(train_emb).astype(np.float32)
        self.nn = NearestNeighbors(n_neighbors=self.k, metric="euclidean").fit(tr)
        return self

    def score(self, emb):
        x = normalize(emb).astype(np.float32)
        d, _ = self.nn.kneighbors(x)
        return -d.mean(axis=1)


# ─── Driver: load embeddings, run 5 detectors, augment auroc.json ──────
def run(args):
    emb_dir = Path(args.emb_dir) / args.dataset / args.architecture / args.split / f"fold_{args.fold}"
    if args.seed != 42:
        emb_dir = emb_dir / f"seed_{args.seed}"
    train = np.load(emb_dir / "train.npy")
    id_test = np.load(emb_dir / "id_test.npy")
    ood_test = np.load(emb_dir / "ood_test.npy")

    print(f"=== {args.dataset} / {args.architecture} / {args.split} ===")
    print(f"  train {train.shape}  id {id_test.shape}  ood {ood_test.shape}")

    if train.var() < 1e-10 or id_test.var() < 1e-10 or ood_test.var() < 1e-10:
        print("  SKIPPED: degenerate embeddings (zero variance)")
        return

    detectors = {
        "kde":        KDEDetector(n_components=32, bandwidth="scott"),
        "vim":        ViMDetector(variance_threshold=0.95),
        "cosine":     CosineDetector(k=1),
        "react_mds":  ReActWrapper(MDSBicDetector(), percentile=90),
        "ash_knn":    ASHWrapper(KNNk20NormDetector(k=20), keep_fraction=0.1),
    }

    new_entries = {}
    for name, det in detectors.items():
        try:
            det.fit(train)
            id_s = det.score(id_test)
            ood_s = det.score(ood_test)
            auroc, fpr95 = compute_auroc_balanced(id_s, ood_s)
            new_entries[name] = {
                "auroc": auroc, "fpr95": fpr95,
                "n_id": int(len(id_s)), "n_ood": int(len(ood_s)),
            }
            print(f"  {name:12s}  AUROC={auroc:.4f}  FPR95={fpr95:.4f}")
        except Exception as e:
            print(f"  {name:12s}  FAILED: {type(e).__name__}: {e}")
            new_entries[name] = {"auroc": float("nan"), "error": str(e)[:200]}

    # Augment auroc.json (preserving existing keys)
    out_dir = Path(args.results_dir) / args.dataset / args.architecture / args.split / f"fold_{args.fold}"
    if args.seed != 42:
        out_dir = out_dir / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "auroc.json"
    if out_path.exists():
        existing = json.loads(out_path.read_text())
    else:
        existing = {"dataset": args.dataset, "architecture": args.architecture,
                    "split": args.split, "fold": args.fold}
    existing.update(new_entries)
    out_path.write_text(json.dumps(existing, indent=2))
    print(f"  -> wrote {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--architecture", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--emb_dir", default=EMB_DIR)
    p.add_argument("--results_dir", default=RESULTS_DIR)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
