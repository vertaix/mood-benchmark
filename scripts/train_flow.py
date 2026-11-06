"""Train a normalizing flow on training embeddings as a density-based
one-class OOD detector. Treated as one detector in the density family
alongside Mahalanobis variants and kPCA.

The flow learns p_train(z) over GNN embeddings; OOD scores are log p_train,
where higher = more ID-like.

Architecture: zuko's Neural Spline Flow (NSF) with rational quadratic
splines — standard for moderate-dim density estimation.

Output goes to:
  flow_models/<dataset>/<arch>/<split>/fold_<f>/seed_<s>/{flow_state.pt,
    id_log_prob.npy, ood_log_prob.npy, train_nll_history.npy,
    val_nll_history.npy, results.json}
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
from mood import paths as _P
DEFAULT_EMB_DIR = str(_P.embeddings)
DEFAULT_OUT_DIR = str(_P.flow_models)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, required=True)
    p.add_argument('--architecture', type=str, required=True)
    p.add_argument('--split', type=str, required=True)
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--n_layers', type=int, default=8)
    p.add_argument('--hidden_dim', type=int, default=128)
    p.add_argument('--n_bins', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--emb_dir', type=str, default=DEFAULT_EMB_DIR)
    p.add_argument('--out_dir', type=str, default=DEFAULT_OUT_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== flow / {args.dataset} / {args.architecture} / {args.split} "
          f"/ fold {args.fold} / seed {args.seed} ===", flush=True)
    print(f"  device: {device}", flush=True)

    # --- load embeddings ---
    emb_path = Path(args.emb_dir) / args.dataset / args.architecture / args.split / f"fold_{args.fold}"
    if args.seed != 42:
        emb_path = emb_path / f"seed_{args.seed}"
    train_emb = np.load(emb_path / "train.npy")
    id_test_emb = np.load(emb_path / "id_test.npy")
    ood_test_emb = np.load(emb_path / "ood_test.npy")
    print(f"  train: {train_emb.shape}  id: {id_test_emb.shape}  ood: {ood_test_emb.shape}",
          flush=True)

    # --- standardize per-dim using train stats ---
    mu = train_emb.mean(axis=0)
    sigma = train_emb.std(axis=0) + 1e-8
    train_emb = (train_emb - mu) / sigma
    id_test_emb = (id_test_emb - mu) / sigma
    ood_test_emb = (ood_test_emb - mu) / sigma

    # --- 90/10 fit/val split for monitoring ---
    n_train = len(train_emb)
    perm = np.random.RandomState(args.seed).permutation(n_train)
    n_fit = max(1, int(0.9 * n_train))
    fit_emb = train_emb[perm[:n_fit]]
    val_emb = train_emb[perm[n_fit:]] if n_train - n_fit > 0 else fit_emb[:1]

    fit_t = torch.tensor(fit_emb, dtype=torch.float32, device=device)
    val_t = torch.tensor(val_emb, dtype=torch.float32, device=device)
    id_test_t = torch.tensor(id_test_emb, dtype=torch.float32, device=device)
    ood_test_t = torch.tensor(ood_test_emb, dtype=torch.float32, device=device)
    d = train_emb.shape[1]

    # --- build NSF flow ---
    import zuko
    flow = zuko.flows.NSF(
        features=d,
        transforms=args.n_layers,
        hidden_features=[args.hidden_dim] * 2,
        bins=args.n_bins,
    ).to(device)
    n_params = sum(p.numel() for p in flow.parameters())
    print(f"  flow params: {n_params:,}", flush=True)

    opt = torch.optim.Adam(flow.parameters(), lr=args.lr)
    loader = DataLoader(TensorDataset(fit_t), batch_size=args.batch_size,
                        shuffle=True, drop_last=False)

    train_nll_history, val_nll_history = [], []
    best_val_nll = float('inf')
    best_state = None
    t0 = time.time()

    for epoch in range(args.epochs):
        flow.train()
        losses = []
        for (batch,) in loader:
            opt.zero_grad()
            loss = -flow().log_prob(batch).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        flow.eval()
        with torch.no_grad():
            val_nll = -flow().log_prob(val_t).mean().item()
            train_nll = float(np.mean(losses))
        train_nll_history.append(train_nll)
        val_nll_history.append(val_nll)

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_state = {k: v.detach().cpu().clone() for k, v in flow.state_dict().items()}

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:>3}/{args.epochs} | train NLL={train_nll:.4f} "
                  f"| val NLL={val_nll:.4f} | best={best_val_nll:.4f} | {elapsed:.0f}s",
                  flush=True)

    # --- score test sets with the best model ---
    flow.load_state_dict(best_state)
    flow.eval()
    with torch.no_grad():
        # Score in chunks to avoid OOM on large test sets.
        def score(t, chunk=512):
            out = []
            for i in range(0, len(t), chunk):
                out.append(flow().log_prob(t[i:i+chunk]).cpu().numpy())
            return np.concatenate(out)
        id_log_prob = score(id_test_t)
        ood_log_prob = score(ood_test_t)

    # --- balanced AUROC ---
    from sklearn.metrics import roc_auc_score, roc_curve

    def fpr_at_tpr(id_scores, ood_scores, tpr_threshold=0.95):
        labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
        scores = np.concatenate([id_scores, ood_scores])
        fpr, tpr, _ = roc_curve(labels, scores)
        idx = np.searchsorted(tpr, tpr_threshold)
        return float(fpr[min(idx, len(fpr) - 1)])

    def compute_auroc_balanced(id_scores, ood_scores, n_repeats=5, seed=42):
        rng = np.random.RandomState(seed)
        n_min = min(len(id_scores), len(ood_scores))
        if n_min < 2:
            return float('nan'), float('nan'), float('nan')
        aurocs, fpr95s = [], []
        for _ in range(n_repeats):
            id_sub = rng.choice(id_scores, n_min, replace=False)
            ood_sub = rng.choice(ood_scores, n_min, replace=False)
            scores = np.concatenate([id_sub, ood_sub])
            labels = np.concatenate([np.ones(n_min), np.zeros(n_min)])
            aurocs.append(roc_auc_score(labels, scores))
            fpr95s.append(fpr_at_tpr(id_sub, ood_sub))
        return float(np.mean(aurocs)), float(np.std(aurocs)), float(np.mean(fpr95s))

    mean_auroc, std_auroc, mean_fpr95 = compute_auroc_balanced(id_log_prob, ood_log_prob)

    # --- save ---
    out_dir = Path(args.out_dir) / args.dataset / args.architecture / args.split / \
        f"fold_{args.fold}" / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_dir / "flow_state.pt")
    np.save(out_dir / "id_log_prob.npy", id_log_prob)
    np.save(out_dir / "ood_log_prob.npy", ood_log_prob)
    np.save(out_dir / "train_nll_history.npy", np.asarray(train_nll_history))
    np.save(out_dir / "val_nll_history.npy", np.asarray(val_nll_history))
    with open(out_dir / "results.json", "w") as f:
        json.dump({
            "auroc": mean_auroc, "auroc_std": std_auroc, "fpr95": mean_fpr95,
            "best_val_nll": best_val_nll,
            "final_train_nll": train_nll_history[-1],
            "n_params": n_params,
            "n_id_test": int(len(id_log_prob)),
            "n_ood_test": int(len(ood_log_prob)),
            "config": vars(args),
        }, f, indent=2)
    print(f"\n  AUROC: {mean_auroc:.4f} ± {std_auroc:.4f}  FPR95: {mean_fpr95:.4f}", flush=True)
    print(f"  best val NLL: {best_val_nll:.4f}  saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
