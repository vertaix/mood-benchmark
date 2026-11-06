"""Linear probe diagnostic — can a logistic regression separate ID from OOD
using only the GNN's frozen test-set embeddings?

This is *not* a fair comparison to one-class detectors: the probe sees OOD
labels at training time. It's an upper-bound diagnostic. High probe AUROC
together with low practical-detector AUROC means the information is
linearly available in the embeddings but inaccessible to post-hoc methods.

Methodology:
  1. Build labeled set X = [id_test_emb; ood_test_emb], y = [1...; 0...]
  2. Subsample to balanced n_min:n_min ID/OOD (matches main benchmark protocol).
  3. 5-fold stratified CV: in each fold, fit LogisticRegression(L2, C=1.0) on
     the train slice, score on the held-out slice. Report mean ± std AUROC.
  4. The GNN's training-set embeddings are NEVER used. The probe trains and
     tests on test-set embeddings only — otherwise the probe could just learn
     the physical criterion the GNN already organized in its training space.

Output:
  linear_probes/<dataset>/<arch>/<split>/fold_<f>/results.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
from mood import paths as _P
DEFAULT_EMB_DIR = str(_P.embeddings)
DEFAULT_OUT_DIR = str(_P.linear_probes)


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
    p.add_argument("--model_seed", type=int, default=42,
                   help="Training seed of the GNN whose embeddings to probe. "
                        "When != 42, looks in .../fold_<f>/seed_<s>/ subdirs.")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    emb_path = Path(args.emb_dir) / args.dataset / args.architecture / args.split / f"fold_{args.fold}"
    if args.model_seed != 42:
        emb_path = emb_path / f"seed_{args.model_seed}"
    id_emb = np.load(emb_path / "id_test.npy")
    ood_emb = np.load(emb_path / "ood_test.npy")

    if not args.quiet:
        print(f"=== probe / {args.dataset} / {args.architecture} / {args.split} / fold {args.fold} ===")
        print(f"  id: {id_emb.shape}  ood: {ood_emb.shape}")

    # --- balanced n_min:n_min subsample (matches benchmark AUROC protocol) ---
    n_min = min(len(id_emb), len(ood_emb))
    if n_min < args.n_cv_folds:
        result = {"mean_auroc": float("nan"), "std_auroc": float("nan"),
                  "cv_aurocs": [], "n_id_used": int(n_min), "n_ood_used": int(n_min),
                  "skipped": "fewer samples than CV folds"}
        out_dir = Path(args.out_dir) / args.dataset / args.architecture / args.split / f"fold_{args.fold}"
        if args.model_seed != 42:
            out_dir = out_dir / f"seed_{args.model_seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "results.json").write_text(json.dumps(result, indent=2))
        if not args.quiet:
            print(f"  SKIPPED: only {n_min} per class")
        return

    rng = np.random.RandomState(args.seed)
    id_sub = id_emb[rng.choice(len(id_emb), n_min, replace=False)]
    ood_sub = ood_emb[rng.choice(len(ood_emb), n_min, replace=False)]
    X = np.concatenate([id_sub, ood_sub], axis=0)
    y = np.concatenate([np.ones(n_min), np.zeros(n_min)])

    # --- 5-fold stratified CV ---
    skf = StratifiedKFold(n_splits=args.n_cv_folds, shuffle=True, random_state=args.seed)
    aurocs = []
    for fold_idx, (tr, te) in enumerate(skf.split(X, y)):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                 max_iter=1000, random_state=args.seed)
        clf.fit(Xtr, y[tr])
        scores = clf.predict_proba(Xte)[:, 1]
        aurocs.append(float(roc_auc_score(y[te], scores)))

    mean_auroc = float(np.mean(aurocs))
    std_auroc = float(np.std(aurocs))

    out_dir = Path(args.out_dir) / args.dataset / args.architecture / args.split / f"fold_{args.fold}"
    if args.model_seed != 42:
        out_dir = out_dir / f"seed_{args.model_seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps({
        "mean_auroc": mean_auroc, "std_auroc": std_auroc,
        "cv_aurocs": aurocs, "n_id_used": int(n_min), "n_ood_used": int(n_min),
        "config": vars(args),
    }, indent=2))

    if not args.quiet:
        print(f"  AUROC: {mean_auroc:.4f} ± {std_auroc:.4f}  (n_per_class={n_min})")


if __name__ == "__main__":
    main()
