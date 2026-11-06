"""MLP probe — non-linear analog of run_linear_probe.py.

Uses the exact same balanced-ID/OOD setup and 5-fold CV split as the linear
probe. The only change is the classifier: a 2-hidden-layer MLP with early
stopping. Purpose: verify that the linear-vs-no-signal dichotomy holds for
non-linear classifiers too. If MLP gets high AUROC where the linear probe
got 0.50, the dichotomy breaks (signal exists, just non-linearly).

L2 regularization: `alpha=1.0` matches the linear probe's `C=1.0`
(L2 strength = 1.0) for an apples-to-apples comparison.

Output: linear_probes_mlp/<dataset>/<arch>/<split>/fold_<f>/results.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
from mood import paths as _P
DEFAULT_EMB_DIR = str(_P.embeddings)
DEFAULT_OUT_DIR = str(_P.linear_probes_mlp)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--architecture", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--n_cv_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--emb_dir", default=DEFAULT_EMB_DIR)
    p.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--alpha", type=float, default=1.0,
                   help="L2 regularization strength (default 1.0 to match "
                        "linear probe C=1.0)")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    emb = Path(a.emb_dir) / a.dataset / a.architecture / a.split / f"fold_{a.fold}"
    id_e = np.load(emb / "id_test.npy"); ood_e = np.load(emb / "ood_test.npy")
    if not a.quiet:
        print(f"=== MLP probe / {a.dataset} / {a.architecture} / {a.split} / fold {a.fold} ===")
        print(f"  id: {id_e.shape}  ood: {ood_e.shape}")

    n_min = min(len(id_e), len(ood_e))
    out_dir = Path(a.out_dir) / a.dataset / a.architecture / a.split / f"fold_{a.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if n_min < a.n_cv_folds:
        (out_dir / "results.json").write_text(json.dumps(
            {"mean_auroc": float("nan"), "std_auroc": float("nan"),
             "skipped": "fewer samples than CV folds"}, indent=2))
        if not a.quiet:
            print(f"  SKIPPED: only {n_min} per class")
        return
    if id_e.var() < 1e-10 or ood_e.var() < 1e-10:
        (out_dir / "results.json").write_text(json.dumps(
            {"mean_auroc": float("nan"), "std_auroc": float("nan"),
             "skipped": "degenerate embeddings"}, indent=2))
        if not a.quiet:
            print(f"  SKIPPED: degenerate embeddings")
        return

    rng = np.random.RandomState(a.seed)
    id_sub = id_e[rng.choice(len(id_e), n_min, replace=False)]
    ood_sub = ood_e[rng.choice(len(ood_e), n_min, replace=False)]
    X = np.concatenate([id_sub, ood_sub], axis=0)
    y = np.concatenate([np.ones(n_min), np.zeros(n_min)])

    skf = StratifiedKFold(n_splits=a.n_cv_folds, shuffle=True, random_state=a.seed)
    aurocs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500,
                            alpha=a.alpha,
                            early_stopping=True, random_state=a.seed,
                            validation_fraction=0.1, n_iter_no_change=20)
        clf.fit(Xtr, y[tr])
        scores = clf.predict_proba(Xte)[:, 1]
        aurocs.append(float(roc_auc_score(y[te], scores)))

    mean_a = float(np.mean(aurocs)); std_a = float(np.std(aurocs))
    (out_dir / "results.json").write_text(json.dumps({
        "mean_auroc": mean_a, "std_auroc": std_a, "cv_aurocs": aurocs,
        "n_id_used": int(n_min), "n_ood_used": int(n_min),
        "config": vars(a),
    }, indent=2))
    if not a.quiet:
        print(f"  AUROC: {mean_a:.4f} ± {std_a:.4f}  (n_per_class={n_min})")


if __name__ == "__main__":
    main()
