"""Run Mahalanobis (Ren et al. 2021), KNN (Sun et al. 2022), kPCA / CoRP
(Fang et al., ood-kernel-pca), embedding-norm, LOF (Breunig et al. 2000),
One-Class SVM (Schoelkopf et al. 1999), and Isolation Forest (Liu et al. 2008)
detectors.

Mahalanobis: GMM with tied covariance, K tuned on ID-only validation set.
Mahalanobis-BIC: same GMM, K chosen by BIC on full train (no val split).
KNN: L2 on normalized embeddings, k tuned on ID-only validation set.
KNN sweep: results for each (k in {5,10,20,50}) x (normalize? in {True, False}).
kPCA / CoRP: L2-normalize -> RFF mapping for Gaussian kernel -> linear PCA;
score = -reconstruction error in the RFF space.
Norm: training-free; score = -||z||_2 (higher norm -> more OOD).
LOF: sklearn LocalOutlierFactor in novelty mode on L2-normalized embeddings.
OCSVM: sklearn OneClassSVM (RBF, nu=0.1, gamma='scale'); subsamples train
to 10k if it exceeds 20k for tractability.
IForest: sklearn IsolationForest (n_estimators=100); substitute for ADERH
since no reference implementation of ADERH is available.
All use 50:50 balanced AUROC (single deterministic subsample).

Usage:
    python scripts/run_detectors.py --dataset matbench_dielectric \
        --architecture alignn --split comp_oxide --fold 0
"""
import argparse
import builtins
import json
import os
import sys
import numpy as np
import faiss
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

_original_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    _original_print(*args, **kwargs)

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
from mood import paths as _P
EMB_DIR = str(_P.embeddings)
RESULTS_DIR = str(_P.results)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, required=True)
    p.add_argument('--architecture', type=str, required=True)
    p.add_argument('--split', type=str, required=True)
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + 1e-10)


def compute_fpr_at_tpr(id_scores, ood_scores, tpr_threshold=0.95):
    """Compute FPR at given TPR threshold.
    Convention: ID=positive (label=1), OOD=negative (label=0).
    Higher score = more ID-like."""
    from sklearn.metrics import roc_curve
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.searchsorted(tpr, tpr_threshold)
    return float(fpr[min(idx, len(fpr) - 1)])


def compute_auroc_balanced(id_scores, ood_scores, seed=42):
    """Subsample majority class to match minority (single deterministic draw).
    Returns auroc, fpr95."""
    rng = np.random.RandomState(seed)
    n_min = min(len(id_scores), len(ood_scores))
    if n_min < 2:
        return float('nan'), float('nan')
    id_sub = rng.choice(id_scores, size=n_min, replace=False)
    ood_sub = rng.choice(ood_scores, size=n_min, replace=False)
    scores = np.concatenate([id_sub, ood_sub])
    labels = np.concatenate([np.ones(n_min), np.zeros(n_min)])  # 1=ID, 0=OOD
    auroc = float(roc_auc_score(labels, scores))
    fpr95 = float(compute_fpr_at_tpr(id_sub, ood_sub, 0.95))
    return auroc, fpr95


# ─── Mahalanobis (Ren et al. 2021) ───────────────────────────────────────
def run_mahalanobis(train_emb, id_test_emb, ood_test_emb):
    """GMM with tied covariance, K tuned on ID-only validation."""
    print("  Running Mahalanobis (Ren et al. 2021)...")

    # Split train into fit/val for K tuning
    train_fit, train_val = train_test_split(
        train_emb, test_size=0.1, random_state=42)

    # Tune K on ID validation log-likelihood
    best_K, best_score = 1, -np.inf
    for K in [1, 2, 4, 8, 16, 32, 64, 128]:
        if K >= len(train_fit):
            continue
        try:
            gmm = GaussianMixture(
                n_components=K, covariance_type='tied',
                random_state=42, max_iter=200, reg_covar=1e-4)
            gmm.fit(train_fit)
            score = gmm.score(train_val)
            if score > best_score:
                best_score, best_K = score, K
        except Exception as e:
            print(f"    K={K} failed: {e}")
            continue

    print(f"    Best K={best_K} (val score={best_score:.4f})")

    # Fit final model on full training set, with fallback to smaller K on
    # numerical failure (collapsed components, ill-defined covariance).
    final_K = best_K
    while final_K >= 1:
        try:
            gmm = GaussianMixture(
                n_components=final_K, covariance_type='tied',
                random_state=42, max_iter=200, reg_covar=1e-4)
            gmm.fit(train_emb)
            break
        except Exception as e:
            print(f"    final fit K={final_K} failed: {e}")
            final_K = final_K // 2 if final_K > 1 else 0
    if final_K < 1:
        raise RuntimeError("MDS final-fit failed at all K values")
    if final_K != best_K:
        print(f"    fell back to K={final_K}")
        best_K = final_K
    means = gmm.means_
    cov = gmm.covariances_
    cov_inv = np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))

    # Score: -min_k MD(z, mu_k). Higher = more ID.
    def maha_score_batch(embeddings):
        scores = np.zeros(len(embeddings))
        for i, z in enumerate(embeddings):
            dists = []
            for mu_k in means:
                diff = z - mu_k
                dist = np.sqrt(np.maximum(diff @ cov_inv @ diff, 0))
                dists.append(dist)
            scores[i] = -min(dists)
        return scores

    id_scores = maha_score_batch(id_test_emb)
    ood_scores = maha_score_batch(ood_test_emb)
    auroc, fpr95 = compute_auroc_balanced(id_scores, ood_scores)

    print(f"    AUROC: {auroc:.4f}  FPR95: {fpr95:.4f}")
    return {
        'auroc': auroc, 'fpr95': fpr95, 'best_K': best_K,
        'n_id': len(id_scores), 'n_ood': len(ood_scores),
    }


# ─── Mahalanobis with BIC (no val set) ───────────────────────────────────
def run_mahalanobis_bic(train_emb, id_test_emb, ood_test_emb, criterion='bic'):
    """GMM with tied covariance, K chosen by BIC (or AIC) on full train set.

    No validation split — BIC penalizes complexity directly:
      BIC = -2 * log L + k * log(n)
    where k is the number of free parameters and n is the sample size.
    AIC uses 2*k instead of k*log(n) (less conservative, prefers larger K).
    """
    print(f"  Running Mahalanobis with {criterion.upper()} (no val set)...")

    best_K, best_score = 1, np.inf
    fitted = {}
    for K in [1, 2, 4, 8, 16, 32, 64, 128]:
        if K >= len(train_emb):
            continue
        try:
            gmm = GaussianMixture(
                n_components=K, covariance_type='tied',
                random_state=42, max_iter=200, reg_covar=1e-4)
            gmm.fit(train_emb)
            score = gmm.bic(train_emb) if criterion == 'bic' else gmm.aic(train_emb)
            fitted[K] = (gmm, score)
            if score < best_score:
                best_score, best_K = score, K
        except Exception as e:
            print(f"    K={K} failed: {e}")
            continue

    print(f"    Best K={best_K}  ({criterion}={best_score:.2f})")

    gmm = fitted[best_K][0]
    means = gmm.means_
    cov = gmm.covariances_
    cov_inv = np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))

    def maha_score_batch(embeddings):
        scores = np.zeros(len(embeddings))
        for i, z in enumerate(embeddings):
            dists = []
            for mu_k in means:
                diff = z - mu_k
                dist = np.sqrt(np.maximum(diff @ cov_inv @ diff, 0))
                dists.append(dist)
            scores[i] = -min(dists)
        return scores

    id_scores = maha_score_batch(id_test_emb)
    ood_scores = maha_score_batch(ood_test_emb)
    auroc, fpr95 = compute_auroc_balanced(id_scores, ood_scores)

    print(f"    AUROC: {auroc:.4f}  FPR95: {fpr95:.4f}")
    return {
        'auroc': auroc, 'fpr95': fpr95, 'best_K': best_K,
        'criterion': criterion,
        'criterion_value': float(best_score),
        'n_id': len(id_scores), 'n_ood': len(ood_scores),
    }


# ─── KNN (Sun et al. 2022) ───────────────────────────────────────────────
def run_knn(train_emb, id_test_emb, ood_test_emb):
    """L2 distance on normalized embeddings, k tuned on ID-only validation."""
    print("  Running KNN (Sun et al. 2022)...")

    # Normalize
    train_norm = normalize(train_emb).astype(np.float32)
    id_test_norm = normalize(id_test_emb).astype(np.float32)
    ood_test_norm = normalize(ood_test_emb).astype(np.float32)

    # Split for k tuning
    train_fit, train_val = train_test_split(
        train_norm, test_size=0.1, random_state=42)

    # Build FAISS index on fit set for tuning
    d = train_fit.shape[1]
    index_fit = faiss.IndexFlatL2(d)
    index_fit.add(train_fit)

    # Tune k: pick k that minimizes coefficient of variation on val
    best_k, best_cv = 1, np.inf
    for k in [1, 5, 10, 20, 50, 100]:
        if k >= len(train_fit):
            continue
        distances, _ = index_fit.search(train_val, k)
        kth_dist = distances[:, -1]
        cv = np.std(kth_dist) / (np.mean(kth_dist) + 1e-10)
        if cv < best_cv:
            best_cv, best_k = cv, k

    print(f"    Best k={best_k}")

    # Build full index and score
    index_full = faiss.IndexFlatL2(d)
    index_full.add(train_norm)

    id_dist, _ = index_full.search(id_test_norm, best_k)
    ood_dist, _ = index_full.search(ood_test_norm, best_k)
    id_scores = -id_dist[:, -1]   # negative so higher = more ID
    ood_scores = -ood_dist[:, -1]

    auroc, fpr95 = compute_auroc_balanced(id_scores, ood_scores)

    print(f"    AUROC: {auroc:.4f}  FPR95: {fpr95:.4f}")
    return {
        'auroc': auroc, 'fpr95': fpr95, 'best_k': best_k,
        'n_id': len(id_scores), 'n_ood': len(ood_scores),
    }


# ─── KNN sweep across (k, normalize) ─────────────────────────────────────
def run_knn_sweep(train_emb, id_test_emb, ood_test_emb,
                  ks=(5, 10, 20, 50), normalize_options=(True, False)):
    """Run KNN detector for each k x normalize? combination.

    Returns a dict keyed by 'k{K}_norm{0|1}' with auroc/fpr95 per setting.
    """
    print(f"  Running KNN sweep (k in {ks}, normalize in {normalize_options})...")
    results = {}

    for do_norm in normalize_options:
        if do_norm:
            tr = normalize(train_emb).astype(np.float32)
            id_t = normalize(id_test_emb).astype(np.float32)
            ood_t = normalize(ood_test_emb).astype(np.float32)
        else:
            tr = train_emb.astype(np.float32)
            id_t = id_test_emb.astype(np.float32)
            ood_t = ood_test_emb.astype(np.float32)

        d = tr.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(tr)

        max_k = max(ks)
        # one search up to max_k, then slice
        id_dist, _ = index.search(id_t, min(max_k, len(tr)))
        ood_dist, _ = index.search(ood_t, min(max_k, len(tr)))

        for k in ks:
            if k >= len(tr):
                continue
            id_scores = -id_dist[:, k - 1]
            ood_scores = -ood_dist[:, k - 1]
            auroc, fpr95 = compute_auroc_balanced(id_scores, ood_scores)
            key = f"k{k}_norm{int(do_norm)}"
            results[key] = {
                'k': k, 'normalize': bool(do_norm),
                'auroc': auroc, 'fpr95': fpr95,
                'n_id': len(id_scores), 'n_ood': len(ood_scores),
            }
            print(f"    k={k:>3}  normalize={int(do_norm)}  "
                  f"AUROC={auroc:.4f}  FPR95={fpr95:.4f}")

    return results


# ─── kPCA / CoRP (Fang et al., ood-kernel-pca) ───────────────────────────
def run_kpca(train_emb, id_test_emb, ood_test_emb,
             gamma=2.0, M=2048, exp_var_ratio=0.8, seed=42):
    """Cosine + RBF Random Features PCA (CoRP).

    1. L2-normalize features (cosine kernel mapping).
    2. RFF mapping for the RBF kernel:
           phi(x) = sqrt(2/M) * cos(W x + u),
           W ~ N(0, 2*gamma * I),  u ~ U[0, 2*pi].
    3. Centralize using the training mean.
    4. Linear PCA via SVD of K = X_train^T X_train  (M x M).
    5. Pick q so cumulative explained variance >= exp_var_ratio.
    6. Score = -|| z - U_q U_q^T z ||_2  (higher = more ID).

    Defaults gamma=2, M=2048, exp_var_ratio=0.8 are the CIFAR10-R18-CE row of
    fanghenshaometeor/ood-kernel-pca's run_detection.sh (the closest analogue
    to our 128-d non-contrastive backbones).
    """
    print(f"  Running kPCA / CoRP (gamma={gamma}, M={M}, exp_var={exp_var_ratio})...")
    rng = np.random.RandomState(seed)

    tr = normalize(train_emb).astype(np.float32)
    id_t = normalize(id_test_emb).astype(np.float32)
    ood_t = normalize(ood_test_emb).astype(np.float32)

    d = tr.shape[1]
    w = (np.sqrt(2.0 * gamma) * rng.normal(size=(M, d))).astype(np.float32)
    u = (2.0 * np.pi * rng.rand(M)).astype(np.float32)
    scale = np.float32(np.sqrt(2.0 / M))

    def rff(x):
        return scale * np.cos(x @ w.T + u[None, :])

    tr = rff(tr)
    id_t = rff(id_t)
    ood_t = rff(ood_t)

    mu = tr.mean(axis=0)
    tr -= mu
    id_t -= mu
    ood_t -= mu

    # K is M x M; SVD on it yields the same left singular vectors as on tr.
    K = tr.T @ tr
    u_full, s, _ = np.linalg.svd(K)

    s_total = float(s.sum())
    cum = np.cumsum(s) / max(s_total, 1e-12)
    q = int(np.searchsorted(cum, exp_var_ratio) + 1)
    q = min(max(q, 1), M)
    print(f"    q = {q} / {M}  (cum_evr_q = {cum[q-1]:.4f})")

    u_q = u_full[:, :q]

    def recon_score(x):
        proj = x @ u_q          # (n, q)
        recon = proj @ u_q.T    # (n, M)
        return -np.linalg.norm(x - recon, ord=2, axis=1)

    id_scores = recon_score(id_t)
    ood_scores = recon_score(ood_t)
    auroc, fpr95 = compute_auroc_balanced(id_scores, ood_scores)

    print(f"    AUROC: {auroc:.4f}  FPR95: {fpr95:.4f}")
    return {
        'auroc': auroc, 'fpr95': fpr95,
        'gamma': float(gamma), 'M': int(M),
        'exp_var_ratio': float(exp_var_ratio), 'q': int(q),
        'n_id': len(id_scores), 'n_ood': len(ood_scores),
    }


# ─── Embedding norm ──────────────────────────────────────────────────────
def run_norm(train_emb, id_test_emb, ood_test_emb):
    """Sort by raw L2 norm of the embedding. Higher norm = more OOD,
    so the score (higher = more ID) is the negated norm.

    Train embeddings are unused — this is a training-free detector that
    relies entirely on the magnitude of the test-time embedding.
    """
    print("  Running Norm (||z||_2)...")
    id_scores = -np.linalg.norm(id_test_emb, ord=2, axis=1)
    ood_scores = -np.linalg.norm(ood_test_emb, ord=2, axis=1)
    auroc, fpr95 = compute_auroc_balanced(id_scores, ood_scores)
    print(f"    AUROC: {auroc:.4f}  FPR95: {fpr95:.4f}")
    return {
        'auroc': auroc, 'fpr95': fpr95,
        'n_id': len(id_scores), 'n_ood': len(ood_scores),
    }


# ─── LOF (Breunig et al. 2000) ───────────────────────────────────────────
def run_lof(train_emb, id_test_emb, ood_test_emb, n_neighbors=20):
    """sklearn LocalOutlierFactor in novelty mode on L2-normalized embeddings.
    score_samples returns negative LOF (higher = more ID).
    n_neighbors=20 is sklearn's default and the standard in OOD literature.
    """
    print(f"  Running LOF (n_neighbors={n_neighbors})...")
    tr = normalize(train_emb).astype(np.float32)
    id_t = normalize(id_test_emb).astype(np.float32)
    ood_t = normalize(ood_test_emb).astype(np.float32)

    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors, novelty=True,
        contamination='auto', metric='euclidean', n_jobs=-1)
    lof.fit(tr)

    id_scores = lof.score_samples(id_t)
    ood_scores = lof.score_samples(ood_t)
    auroc, fpr95 = compute_auroc_balanced(id_scores, ood_scores)

    print(f"    AUROC: {auroc:.4f}  FPR95: {fpr95:.4f}")
    return {
        'auroc': auroc, 'fpr95': fpr95, 'n_neighbors': int(n_neighbors),
        'n_id': len(id_scores), 'n_ood': len(ood_scores),
    }


# ─── One-Class SVM (Schoelkopf et al. 1999) ──────────────────────────────
def run_ocsvm(train_emb, id_test_emb, ood_test_emb,
              nu=0.1, gamma='scale', subsample_threshold=20000,
              subsample_size=10000, seed=42):
    """sklearn OneClassSVM with RBF kernel on L2-normalized embeddings.
    decision_function returns higher = more ID.
    Subsamples train to subsample_size if it exceeds subsample_threshold,
    since RBF OCSVM scales O(n_train^2).
    """
    print(f"  Running OCSVM (nu={nu}, gamma={gamma})...")
    tr = normalize(train_emb).astype(np.float32)
    id_t = normalize(id_test_emb).astype(np.float32)
    ood_t = normalize(ood_test_emb).astype(np.float32)

    n_fit = len(tr)
    if n_fit > subsample_threshold:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n_fit, subsample_size, replace=False)
        tr = tr[idx]
        print(f"    subsampled train: {n_fit} -> {len(tr)}")
        n_fit = len(tr)

    ocsvm = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma, cache_size=2000)
    ocsvm.fit(tr)

    id_scores = ocsvm.decision_function(id_t)
    ood_scores = ocsvm.decision_function(ood_t)
    auroc, fpr95 = compute_auroc_balanced(id_scores, ood_scores)

    print(f"    AUROC: {auroc:.4f}  FPR95: {fpr95:.4f}")
    return {
        'auroc': auroc, 'fpr95': fpr95,
        'nu': float(nu), 'gamma': str(gamma), 'n_fit': int(n_fit),
        'n_id': len(id_scores), 'n_ood': len(ood_scores),
    }


# ─── Isolation Forest (Liu et al. 2008) ──────────────────────────────────
def run_iforest(train_emb, id_test_emb, ood_test_emb,
                n_estimators=100, seed=42):
    """sklearn IsolationForest on L2-normalized embeddings.
    Substitute for ADERH (NeurIPS 2025) since no reference implementation
    is available — IForest is the canonical isolation-based detector ADERH
    extends.
    """
    print(f"  Running IsolationForest (n_estimators={n_estimators})...")
    tr = normalize(train_emb).astype(np.float32)
    id_t = normalize(id_test_emb).astype(np.float32)
    ood_t = normalize(ood_test_emb).astype(np.float32)

    iforest = IsolationForest(
        n_estimators=n_estimators, contamination='auto',
        random_state=seed, n_jobs=-1)
    iforest.fit(tr)

    id_scores = iforest.score_samples(id_t)
    ood_scores = iforest.score_samples(ood_t)
    auroc, fpr95 = compute_auroc_balanced(id_scores, ood_scores)

    print(f"    AUROC: {auroc:.4f}  FPR95: {fpr95:.4f}")
    return {
        'auroc': auroc, 'fpr95': fpr95, 'n_estimators': int(n_estimators),
        'n_id': len(id_scores), 'n_ood': len(ood_scores),
    }


# ─── ADERH (Walid 2024+, ensemble of random hypersphere pairs) ────────────
def run_aderh(train_emb, id_test_emb, ood_test_emb,
              n_estimators=256, n=18, seed=42):
    """ADERH OOD detector using the upstream reference implementation.

    Reference: https://github.com/Walid10010/ADERH
    Input is MinMaxScaler-normalized to [0, 1] per the upstream's eval
    protocol (run_experiment.py applies MinMaxScaler before fit/score).
    Score returned is "higher = more anomalous" (sklearn convention);
    we negate so higher = more ID-like to match the rest of this module.
    """
    print(f"  Running ADERH (n_estimators={n_estimators}, n={n})...")
    import sys, os
    aderh_path = str(_P.aderh)
    if aderh_path not in sys.path:
        sys.path.insert(0, aderh_path)
    from ADERH import ADERH
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    tr = scaler.fit_transform(train_emb).astype(np.float32)
    id_t = scaler.transform(id_test_emb).astype(np.float32)
    ood_t = scaler.transform(ood_test_emb).astype(np.float32)

    algo = ADERH(n_estimators=n_estimators, n=n, random_state=seed)
    algo.fit(tr)

    # decision_function returns positive = more anomalous; negate.
    id_scores = -np.asarray(algo.decision_function(id_t)).ravel()
    ood_scores = -np.asarray(algo.decision_function(ood_t)).ravel()
    auroc, fpr95 = compute_auroc_balanced(id_scores, ood_scores)

    print(f"    AUROC: {auroc:.4f}  FPR95: {fpr95:.4f}")
    return {
        'auroc': auroc, 'fpr95': fpr95,
        'n_estimators': int(n_estimators), 'n': int(n),
        'preprocessing': 'MinMaxScaler',
        'n_id': len(id_scores), 'n_ood': len(ood_scores),
    }


# ─── Main ─────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    emb_path = os.path.join(EMB_DIR, args.dataset, args.architecture,
                            args.split, f'fold_{args.fold}')
    if args.seed != 42:
        emb_path = os.path.join(emb_path, f'seed_{args.seed}')

    print(f"{'='*60}")
    print(f"  {args.architecture.upper()} / {args.dataset} / {args.split} / fold {args.fold}")
    print(f"{'='*60}")

    # Load embeddings
    train_emb = np.load(os.path.join(emb_path, 'train.npy'))
    id_test_emb = np.load(os.path.join(emb_path, 'id_test.npy'))
    ood_test_emb = np.load(os.path.join(emb_path, 'ood_test.npy'))

    print(f"  Train: {train_emb.shape}, ID test: {id_test_emb.shape}, "
          f"OOD test: {ood_test_emb.shape}")

    if len(ood_test_emb) < 2:
        print(f"  WARNING: OOD test has only {len(ood_test_emb)} samples — skipping")
        return

    out_dir = os.path.join(RESULTS_DIR, args.dataset, args.architecture,
                           args.split, f'fold_{args.fold}')
    if args.seed != 42:
        out_dir = os.path.join(out_dir, f'seed_{args.seed}')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'auroc.json')

    # Merge with any existing auroc.json so prior results are not deleted.
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
    else:
        results = {}

    results.update({
        'dataset': args.dataset,
        'architecture': args.architecture,
        'split': args.split,
        'fold': args.fold,
        'n_train': len(train_emb),
        'n_id_test': len(id_test_emb),
        'n_ood_test': len(ood_test_emb),
        'embedding_dim': train_emb.shape[1],
    })

    # Re-run legacy detectors (preserves the existing 'mahalanobis' / 'knn' keys
    # with up-to-date values) plus the new BIC/AIC and KNN sweep variants.
    results['mahalanobis'] = run_mahalanobis(train_emb, id_test_emb, ood_test_emb)
    results['mahalanobis_bic'] = run_mahalanobis_bic(
        train_emb, id_test_emb, ood_test_emb, criterion='bic')
    results['mahalanobis_aic'] = run_mahalanobis_bic(
        train_emb, id_test_emb, ood_test_emb, criterion='aic')
    results['knn'] = run_knn(train_emb, id_test_emb, ood_test_emb)
    results['knn_sweep'] = run_knn_sweep(train_emb, id_test_emb, ood_test_emb)
    results['kpca'] = run_kpca(train_emb, id_test_emb, ood_test_emb)
    results['norm'] = run_norm(train_emb, id_test_emb, ood_test_emb)
    results['lof'] = run_lof(train_emb, id_test_emb, ood_test_emb)
    results['ocsvm'] = run_ocsvm(train_emb, id_test_emb, ood_test_emb)
    results['iforest'] = run_iforest(train_emb, id_test_emb, ood_test_emb)
    results['aderh'] = run_aderh(train_emb, id_test_emb, ood_test_emb)

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Saved to {out_path}")
    print("  DONE")


if __name__ == '__main__':
    main()
