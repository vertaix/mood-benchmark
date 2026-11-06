"""Shared definitions for the MOOD camera-ready recompute.

Single source of truth for: the canonical 15 unsupervised detectors + 2 probes,
the 4-class family taxonomy, paths, seeds, and the balanced-probe pipeline
(identical protocol to scripts/run_linear_probe.py / run_mlp_probe.py).
"""
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys as _sys
_sys.path.insert(0, ROOT)
from mood import paths as _P
EMB = str(_P.embeddings)
OUT = str(_P.outputs)
RESULTS = str(_P.results)
SEEDS = [42, 123, 456]

# Canonical selection (matches scripts/build_per_detector_summary.py SELECTION).
# (csv_detector_key, display_name, family)
SELECTION = [
    ("KNN k=20 norm", "KNN-k20-norm", "Distance"),
    ("KNN k=50 norm", "KNN-k50-norm", "Distance"),
    ("Norm",          "Norm",         "Distance"),
    ("Cosine",        "Cosine",       "Distance"),
    ("LOF",           "LOF",          "Distance"),
    ("ASH + KNN",     "ASH+KNN",      "Distance"),
    ("MDS-BIC",       "MDS-BIC",      "Density"),
    ("KDE",           "KDE",          "Density"),
    ("Flow (NF)",     "Flow",         "Density"),
    ("kPCA",          "kPCA",         "Density"),
    ("ReAct + MDS",   "ReAct+MDS",    "Density"),
    ("IForest",       "IForest",      "Isolation"),
    ("ADERH",         "ADERH",        "Isolation"),
    ("OCSVM",         "OCSVM",        "Boundary-Hybrid"),
    ("ViM",           "ViM",          "Boundary-Hybrid"),
]
UNSUP_KEYS = [k for k, *_ in SELECTION]            # 15 unsupervised detector CSV keys
DISPLAY = {k: d for k, d, _ in SELECTION}
FAMILY = {k: f for k, _, f in SELECTION}
FAMILIES = ["Distance", "Density", "Isolation", "Boundary-Hybrid"]

ABOVE = 0.55   # "% of settings with AUROC > 0.55"

DATASETS = ["dielectric", "log_gvrh", "log_kvrh", "perovskites", "phonons"]
ARCHS = ["alignn", "cgcnn", "dimenet", "schnet"]


def cell_dir(dataset, arch, split, seed):
    base = os.path.join(EMB, f"matbench_{dataset}", arch, split, "fold_0")
    return base if seed == 42 else os.path.join(base, f"seed_{seed}")


def load_idood(dataset, arch, split, seed):
    """Return (id_emb, ood_emb) or (None, None) if missing/degenerate."""
    d = cell_dir(dataset, arch, split, seed)
    fid, food = os.path.join(d, "id_test.npy"), os.path.join(d, "ood_test.npy")
    if not (os.path.exists(fid) and os.path.exists(food)):
        return None, None
    id_e, ood_e = np.load(fid), np.load(food)
    if id_e.var() < 1e-10 or ood_e.var() < 1e-10:
        return None, None
    return id_e, ood_e


def balanced_xy(id_e, ood_e, seed_sub=42):
    rng = np.random.RandomState(seed_sub)
    n = min(len(id_e), len(ood_e))
    if n < 5:
        return None, None
    X = np.concatenate([id_e[rng.choice(len(id_e), n, replace=False)],
                        ood_e[rng.choice(len(ood_e), n, replace=False)]])
    y = np.concatenate([np.ones(n), np.zeros(n)])
    return X, y


def linear_probe_auroc(id_e, ood_e, seed_sub=42, n_folds=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    X, y = balanced_xy(id_e, ood_e, seed_sub)
    if X is None:
        return float("nan")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed_sub)
    au = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        Xtr, Xte = sc.fit_transform(X[tr]), sc.transform(X[te])
        clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                 max_iter=1000, random_state=seed_sub)
        clf.fit(Xtr, y[tr])
        au.append(roc_auc_score(y[te], clf.predict_proba(Xte)[:, 1]))
    return float(np.mean(au))


def mlp_probe_auroc(id_e, ood_e, seed_sub=42, n_folds=5):
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    X, y = balanced_xy(id_e, ood_e, seed_sub)
    if X is None:
        return float("nan")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed_sub)
    au = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        Xtr, Xte = sc.fit_transform(X[tr]), sc.transform(X[te])
        clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500,
                            alpha=1.0, early_stopping=True, random_state=seed_sub,
                            validation_fraction=0.1, n_iter_no_change=20)
        clf.fit(Xtr, y[tr])
        au.append(roc_auc_score(y[te], clf.predict_proba(Xte)[:, 1]))
    return float(np.mean(au))


def mae_id_ood(dataset, arch, split, seed):
    """Return (mae_id, mae_ood) from stored predictions, or (nan, nan)."""
    d = cell_dir(dataset, arch, split, seed)
    try:
        idp = np.load(os.path.join(d, "id_test_pred.npy")).ravel()
        idt = np.load(os.path.join(d, "id_test_targets.npy")).ravel()
        oop = np.load(os.path.join(d, "ood_test_pred.npy")).ravel()
        oot = np.load(os.path.join(d, "ood_test_targets.npy")).ravel()
    except FileNotFoundError:
        return float("nan"), float("nan")
    return float(np.abs(idp - idt).mean()), float(np.abs(oop - oot).mean())
