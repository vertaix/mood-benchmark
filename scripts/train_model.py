"""Unified training script for ALIGNN and CGCNN on ID-only subsets.

Supports physics splits (single fold) and Omee splits (LOCO/SparseX/SparseY).
Trains on ID-only training data, extracts embeddings for all sets.

Usage:
    # Physics split
    python scripts/train_model.py --dataset matbench_dielectric \
        --architecture alignn --split_type comp_oxide --fold 0

    # Omee split
    python scripts/train_model.py --dataset matbench_dielectric \
        --architecture cgcnn --split_type LOCO --fold 3
"""
import argparse
import builtins
import json
import os
import sys
import time
import yaml
import numpy as np

# Torch is only required for the PyTorch-backed archs (ALIGNN, CGCNN).
# kgcnn-bench env doesn't ship torch, so make this import optional —
# kgcnn architectures (schnet, dimenet) skip the torch path entirely.
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    DataLoader = None
    HAS_TORCH = False

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    tf = None
    HAS_TF = False

_original_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    _original_print(*args, **kwargs)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DIR = os.path.join(BASE_DIR, 'splits')
OMEE_FOLDS_DIR = os.path.join(BASE_DIR, 'omee_folds', 'folds')
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints')
EMB_DIR = os.path.join(BASE_DIR, 'embeddings')

PHYSICS_SPLITS = [
    'comp_oxide', 'comp_halide', 'comp_chalcogenide', 'comp_nitride',
    'comp_carbide', 'comp_intermetallic',
    'struct_cubic', 'struct_perovskite', 'struct_hexagonal', 'struct_triclinic',
    'struct_monoclinic', 'struct_tetragonal', 'struct_orthorhombic',
    'struct_high_spacegroup', 'struct_low_spacegroup',
]
OMEE_SPLITS = ['LOCO', 'SparseXcluster', 'SparseYcluster']

# Map dataset names to Omee folder names
OMEE_DATASET_MAP = {
    'matbench_dielectric': 'dielectric',
    'matbench_log_gvrh': 'elasticity',
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, required=True,
                   choices=['matbench_dielectric', 'matbench_log_gvrh',
                            'matbench_log_kvrh', 'matbench_perovskites',
                            'matbench_phonons', 'matbench_mp_gap'])
    p.add_argument('--architecture', type=str, required=True,
                   choices=['alignn', 'cgcnn', 'schnet', 'dimenet'])
    p.add_argument('--split_type', type=str, required=True)
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=None,
                   help='Override default epochs (ALIGNN=300, CGCNN=200)')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--reuse_ckpt', action='store_true',
                   help='kgcnn only: skip training and load existing '
                        'best_model.keras + target_scaler.json (recovery mode '
                        'for cells where prior training timed out before '
                        'extraction). Caller is responsible for ensuring the '
                        'on-disk checkpoint corresponds to the desired config.')
    p.add_argument('--device', type=str,
                   default='cuda' if (HAS_TORCH and torch.cuda.is_available()) else 'cpu')
    return p.parse_args()


def load_full_dataset(dataset_name):
    """Load the full MatBench dataset (all structures + targets)."""
    from matbench.bench import MatbenchBenchmark
    mb = MatbenchBenchmark(autoload=False, subset=[dataset_name])
    mb.load()
    task = list(mb.tasks)[0]

    import pandas as pd
    inputs_train, targets_train = task.get_train_and_val_data(0)
    inputs_test, targets_test = task.get_test_data(0, include_target=True)

    all_inputs = pd.concat([inputs_train, inputs_test]).reset_index(drop=True)
    all_targets = pd.concat([targets_train, targets_test]).reset_index(drop=True)

    return list(all_inputs), list(all_targets)


def load_split_indices(dataset_name, split_type, fold):
    """Load train/id_test/ood_test indices for given split."""

    if split_type in PHYSICS_SPLITS:
        split_dir = os.path.join(SPLIT_DIR, dataset_name, split_type)
        train_idx = np.load(os.path.join(split_dir, 'train_idx.npy'))
        id_test_idx = np.load(os.path.join(split_dir, 'id_test_idx.npy'))
        ood_test_idx = np.load(os.path.join(split_dir, 'ood_test_idx.npy'))
        return train_idx, id_test_idx, ood_test_idx

    elif split_type in OMEE_SPLITS:
        omee_name = OMEE_DATASET_MAP[dataset_name]
        folds_dir = os.path.join(OMEE_FOLDS_DIR, f'{omee_name}_folds')

        train_file = os.path.join(folds_dir, 'train',
                                  f'OFM_{omee_name}_{split_type}_target_clusters50_train.json')
        test_file = os.path.join(folds_dir, 'test',
                                 f'OFM_{omee_name}_{split_type}_target_clusters50_test.json')
        val_file = os.path.join(folds_dir, 'val',
                                f'OFM_{omee_name}_{split_type}_target_clusters50_val.json')

        with open(train_file) as f:
            train_folds = json.load(f)
        with open(test_file) as f:
            test_folds = json.load(f)
        with open(val_file) as f:
            val_folds = json.load(f)

        fold_key = str(fold)
        train_idx = np.array(train_folds[fold_key])
        ood_test_idx = np.array(test_folds[fold_key])
        id_test_idx = np.array(val_folds[fold_key])

        return train_idx, id_test_idx, ood_test_idx

    else:
        raise ValueError(f"Unknown split_type: {split_type}")


# The kgcnn-backed (Keras/TF) architectures need a separate training driver;
# the existing models/alignn/train.py loop is PyTorch-only.
KGCNN_ARCHS = {'schnet', 'dimenet'}


def build_model(architecture):
    """Build model wrapper."""
    if architecture == 'alignn':
        from models.alignn.model import ALIGNNWrapper
        return ALIGNNWrapper()
    elif architecture == 'cgcnn':
        from models.cgcnn.model import CGCNNWrapper
        return CGCNNWrapper()
    elif architecture == 'schnet':
        from models.schnet.model import SchNetWrapper
        return SchNetWrapper()
    elif architecture == 'dimenet':
        from models.dimenet.model import DimeNetPPWrapper
        return DimeNetPPWrapper()
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def train_alignn(model, train_structures, train_targets, val_structures,
                 val_targets, args, ckpt_dir):
    """Train ALIGNN with cosine LR + SWA."""
    from models.alignn.train import train_epoch, eval_epoch

    epochs = args.epochs or 300
    lr = 1e-3

    train_ds = model.make_dataset(train_structures, train_targets)
    val_ds = model.make_dataset(val_structures, val_targets)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=model.collate_fn,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, collate_fn=model.collate_fn, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr / 20.0)

    swa_state = None
    swa_n = 0
    swa_start = int(epochs * 0.6)
    best_val_mae = float('inf')
    start = time.time()

    for epoch in range(epochs):
        loss = train_epoch(model, train_loader, optimizer, args.device)
        val_mae = eval_epoch(model, val_loader, args.device)
        scheduler.step()

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_model.pt'))

        if epoch >= swa_start:
            if swa_state is None:
                swa_state = {k: v.clone().float() for k, v in model.state_dict().items()}
                swa_n = 1
            else:
                swa_n += 1
                for k, v in model.state_dict().items():
                    swa_state[k] += (v.float() - swa_state[k]) / swa_n

        if epoch % 20 == 0 or epoch == epochs - 1:
            elapsed = time.time() - start
            print(f"  Epoch {epoch:>3}/{epochs} | loss={loss:.4f} | "
                  f"val={val_mae:.4f} | best={best_val_mae:.4f} | {elapsed:.0f}s")

    if swa_state is not None:
        swa_load = {k: v.to(model.state_dict()[k].dtype) for k, v in swa_state.items()}
        torch.save(swa_load, os.path.join(ckpt_dir, 'swa_model.pt'))
        model.load_state_dict(swa_load)

    return best_val_mae


def train_cgcnn(model, train_structures, train_targets, val_structures,
                val_targets, args, ckpt_dir):
    """Train CGCNN with SGD + MultiStepLR.

    Standardizes targets to (target - mean) / std before training. This is
    necessary for datasets where target magnitude is large (e.g. matbench
    phonons targets ~700 cm^-1) — without it, MSE blows up to NaN under
    momentum-SGD. The scaler is saved to ckpt_dir/target_scaler.json so
    the embedding-extraction step can unscale predictions back to original
    units."""
    from models.cgcnn.model import cgcnn_collate_fn

    epochs = args.epochs or 200
    lr = 1e-3

    # Target standardization
    target_mean = float(np.mean(train_targets))
    target_std = float(np.std(train_targets) + 1e-9)
    print(f"  target scaling: mean={target_mean:.4f}, std={target_std:.4f}")
    with open(os.path.join(ckpt_dir, 'target_scaler.json'), 'w') as f:
        json.dump({'mean': target_mean, 'std': target_std}, f)

    scaled_train_targets = [(t - target_mean) / target_std for t in train_targets]
    scaled_val_targets = [(t - target_mean) / target_std for t in val_targets]

    train_ds = model.make_dataset(train_structures, scaled_train_targets)
    val_ds = model.make_dataset(val_structures, scaled_val_targets)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=cgcnn_collate_fn,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, collate_fn=cgcnn_collate_fn, num_workers=0)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(epochs * 0.5), int(epochs * 0.75)], gamma=0.1)

    criterion = nn.MSELoss()
    best_val_mae = float('inf')
    start = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss, n = 0, 0
        for batch, targets in train_loader:
            targets = targets.to(args.device)
            preds = model(batch).squeeze(-1)
            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(targets)
            n += len(targets)
        scheduler.step()

        # Validate — compute MAE in ORIGINAL (unscaled) units so reported
        # numbers stay comparable across datasets.
        model.eval()
        val_mae, vn = 0, 0
        with torch.no_grad():
            for batch, targets in val_loader:
                targets = targets.to(args.device)
                preds = model(batch).squeeze(-1)
                preds_unscaled = preds * target_std + target_mean
                targets_unscaled = targets * target_std + target_mean
                val_mae += torch.abs(preds_unscaled - targets_unscaled).sum().item()
                vn += len(targets)
        val_mae /= vn

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_model.pt'))

        if epoch % 20 == 0 or epoch == epochs - 1:
            elapsed = time.time() - start
            print(f"  Epoch {epoch:>3}/{epochs} | loss={total_loss/n:.4f} | "
                  f"val={val_mae:.4f} | best={best_val_mae:.4f} | {elapsed:.0f}s")

    return best_val_mae


def extract_embeddings(model, structures, targets, device, batch_size=64,
                       target_scaler=None):
    """Extract penultimate layer embeddings.

    target_scaler: optional (mean, std) tuple. If provided, predictions are
    unscaled to original units before being returned (so saved
    *_pred.npy files are comparable to *_targets.npy)."""
    dataset = model.make_dataset(structures, targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=model.collate_fn, num_workers=0)

    all_emb, all_pred = [], []
    model.eval()
    with torch.no_grad():
        for batch, _ in loader:
            preds, emb = model.forward_with_embeddings(batch)
            all_emb.append(emb.cpu().numpy())
            all_pred.append(preds.squeeze(-1).cpu().numpy())

    pred = np.concatenate(all_pred)
    if target_scaler is not None:
        mean, std = target_scaler
        pred = pred * std + mean
    return np.concatenate(all_emb), pred


def _kgcnn_extract_and_save_embeddings(model, all_structures, all_targets,
                                       train_idx, id_test_idx, ood_test_idx,
                                       args, emb_root, scaler):
    """Extract embeddings + predictions for kgcnn-backed models and save them
    in the same on-disk layout the PyTorch path uses, so the detector pipeline
    works unchanged."""
    emb_dir = os.path.join(emb_root, args.dataset, args.architecture,
                           args.split_type, f'fold_{args.fold}')
    if args.seed != 42:
        emb_dir = os.path.join(emb_dir, f'seed_{args.seed}')
    os.makedirs(emb_dir, exist_ok=True)
    mean, std = scaler

    def _process(idx, name, chunk_size=256):
        structs = [all_structures[i] for i in idx]
        targets = list(all_targets[idx])
        all_embs, all_preds = [], []
        for start in range(0, len(structs), chunk_size):
            end = min(start + chunk_size, len(structs))
            chunk_batch = model.make_dataset(structs[start:end],
                                             targets[start:end])
            p, e = model.forward_with_embeddings(chunk_batch)
            all_preds.append(p)
            all_embs.append(e)
            print(f"  {name}: chunk {start}-{end}/{len(structs)} done")
        embs = np.concatenate(all_embs, axis=0)
        preds_scaled = np.concatenate(all_preds, axis=0)
        preds = preds_scaled.ravel() * std + mean  # unscale
        np.save(os.path.join(emb_dir, f'{name}.npy'), embs)
        np.save(os.path.join(emb_dir, f'{name}_pred.npy'), preds)
        np.save(os.path.join(emb_dir, f'{name}_targets.npy'),
                np.asarray(targets, dtype=np.float32))
        print(f"  {name}: emb {embs.shape}, pred {preds.shape}")

    _process(train_idx, 'train')
    _process(id_test_idx, 'id_test')
    _process(ood_test_idx, 'ood_test')


def main():
    args = parse_args()
    np.random.seed(args.seed)
    if HAS_TORCH:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    if HAS_TF:
        tf.random.set_seed(args.seed)

    print(f"{'='*60}")
    print(f"  {args.architecture.upper()} / {args.dataset} / {args.split_type} / fold {args.fold}")
    print(f"  Device: {args.device}")
    print(f"{'='*60}")

    # Load full dataset
    print("\nLoading dataset...")
    all_structures, all_targets = load_full_dataset(args.dataset)
    all_targets = np.array(all_targets)
    print(f"  Total: {len(all_structures)} structures")

    # Load split indices
    print("Loading split indices...")
    train_idx, id_test_idx, ood_test_idx = load_split_indices(
        args.dataset, args.split_type, args.fold)
    print(f"  Train: {len(train_idx)}, ID test: {len(id_test_idx)}, "
          f"OOD test: {len(ood_test_idx)}")

    # Split train into train + val (90/10)
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(train_idx))
    n_val = max(1, int(0.1 * len(train_idx)))
    val_local_idx = perm[:n_val]
    train_local_idx = perm[n_val:]

    actual_train_idx = train_idx[train_local_idx]
    actual_val_idx = train_idx[val_local_idx]

    train_structures = [all_structures[i] for i in actual_train_idx]
    train_targets_arr = all_targets[actual_train_idx]
    val_structures = [all_structures[i] for i in actual_val_idx]
    val_targets_arr = all_targets[actual_val_idx]

    print(f"  After val split: train={len(train_structures)}, val={len(val_structures)}")

    # Build model
    model = build_model(args.architecture)

    # Checkpoint dir (seed != 42 gets a seed_<s> subdirectory to avoid
    # overwriting existing paper results produced with the default seed=42)
    ckpt_dir = os.path.join(CKPT_DIR, args.dataset, args.architecture,
                            args.split_type, f'fold_{args.fold}')
    if args.seed != 42:
        ckpt_dir = os.path.join(ckpt_dir, f'seed_{args.seed}')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Dispatch by architecture family. PyTorch backends share the existing
    # train_alignn / train_cgcnn drivers; Keras (kgcnn) backends route to
    # the shared driver in models/_kgcnn_train.py.
    if args.architecture in KGCNN_ARCHS:
        from models._kgcnn_train import train as kgcnn_train
        import json as _json
        ckpt_path = os.path.join(ckpt_dir, "best_model.keras")
        scaler_path = os.path.join(ckpt_dir, "target_scaler.json")

        # Recovery branch: if a fresh best_model.keras + target_scaler.json
        # already exist (e.g., a prior run timed out before extraction), skip
        # the costly training and go straight to embedding extraction.
        # `--reuse_ckpt` opt-in to avoid silently picking up stale checkpoints.
        reuse_ckpt = bool(getattr(args, 'reuse_ckpt', False))
        if reuse_ckpt and os.path.exists(ckpt_path) and os.path.exists(scaler_path):
            print(f"\n  REUSE CKPT: loading {ckpt_path} (skip training)")
            model.load_checkpoint(ckpt_path)
            with open(scaler_path) as f:
                s = _json.load(f)
            scaler = (s["mean"], s["std"])
        else:
            train_batch = model.make_dataset(train_structures, list(train_targets_arr))
            val_batch = model.make_dataset(val_structures, list(val_targets_arr))
            # kgcnn fix: MAE loss + kgcnn warmup-exp schedule, 500 epochs with
            # 50-epoch early stopping. Validated on perovskites/comp_oxide
            # where this combo dropped SchNet from 5.5x to 1.3x leaderboard.
            epochs = args.epochs or 500
            best = kgcnn_train(model, train_batch, val_batch,
                               epochs=epochs, batch_size=args.batch_size,
                               lr=1e-3,
                               loss_kind="mae", schedule_kind="kgcnn", patience=50,
                               ckpt_path=ckpt_path)
            best_val = best["val_mae"]
            with open(scaler_path, "w") as f:
                _json.dump({"mean": best["target_mean"], "std": best["target_std"]}, f)
            print(f"\n  Best val MAE: {best_val:.4f}")
            scaler = (best["target_mean"], best["target_std"])

        _kgcnn_extract_and_save_embeddings(
            model, all_structures, all_targets,
            train_idx, id_test_idx, ood_test_idx, args, EMB_DIR,
            scaler=scaler)
        return

    # PyTorch path (ALIGNN / CGCNN)
    model = model.to(args.device)
    print(f"\nTraining {args.architecture.upper()}...")
    if args.architecture == 'alignn':
        best_val = train_alignn(
            model, train_structures, list(train_targets_arr),
            val_structures, list(val_targets_arr), args, ckpt_dir)
    else:
        best_val = train_cgcnn(
            model, train_structures, list(train_targets_arr),
            val_structures, list(val_targets_arr), args, ckpt_dir)

    print(f"\n  Best val MAE: {best_val:.4f}")

    # Load best checkpoint for embedding extraction
    best_ckpt = os.path.join(ckpt_dir, 'swa_model.pt') \
        if os.path.exists(os.path.join(ckpt_dir, 'swa_model.pt')) \
        else os.path.join(ckpt_dir, 'best_model.pt')
    model.load_state_dict(torch.load(best_ckpt, map_location=args.device,
                                      weights_only=True))
    model.eval()
    print(f"  Loaded: {best_ckpt}")

    # Load target scaler if one was saved during training (CGCNN path).
    # Predictions get unscaled before saving so id/ood_test_pred.npy files
    # are in the same units as targets.
    target_scaler = None
    scaler_path = os.path.join(ckpt_dir, 'target_scaler.json')
    if os.path.exists(scaler_path):
        with open(scaler_path) as f:
            s = json.load(f)
        target_scaler = (s['mean'], s['std'])
        print(f"  Loaded target scaler: mean={s['mean']:.4f}, std={s['std']:.4f}")

    # Extract embeddings
    print("\nExtracting embeddings...")
    emb_dir = os.path.join(EMB_DIR, args.dataset, args.architecture,
                           args.split_type, f'fold_{args.fold}')
    if args.seed != 42:
        emb_dir = os.path.join(emb_dir, f'seed_{args.seed}')
    os.makedirs(emb_dir, exist_ok=True)

    # Train embeddings (full train set, not split)
    train_structs_full = [all_structures[i] for i in train_idx]
    train_targets_full = list(all_targets[train_idx])
    train_emb, train_pred = extract_embeddings(
        model, train_structs_full, train_targets_full, args.device,
        target_scaler=target_scaler)
    np.save(os.path.join(emb_dir, 'train.npy'), train_emb)
    np.save(os.path.join(emb_dir, 'train_pred.npy'), train_pred)
    np.save(os.path.join(emb_dir, 'train_targets.npy'), all_targets[train_idx])
    print(f"  Train: {train_emb.shape}")

    # ID test embeddings
    id_structs = [all_structures[i] for i in id_test_idx]
    id_targets = list(all_targets[id_test_idx])
    id_emb, id_pred = extract_embeddings(
        model, id_structs, id_targets, args.device, target_scaler=target_scaler)
    np.save(os.path.join(emb_dir, 'id_test.npy'), id_emb)
    np.save(os.path.join(emb_dir, 'id_test_pred.npy'), id_pred)
    np.save(os.path.join(emb_dir, 'id_test_targets.npy'), all_targets[id_test_idx])
    print(f"  ID test: {id_emb.shape}")

    # OOD test embeddings
    ood_structs = [all_structures[i] for i in ood_test_idx]
    ood_targets = list(all_targets[ood_test_idx])
    ood_emb, ood_pred = extract_embeddings(
        model, ood_structs, ood_targets, args.device, target_scaler=target_scaler)
    np.save(os.path.join(emb_dir, 'ood_test.npy'), ood_emb)
    np.save(os.path.join(emb_dir, 'ood_test_pred.npy'), ood_pred)
    np.save(os.path.join(emb_dir, 'ood_test_targets.npy'), all_targets[ood_test_idx])
    print(f"  OOD test: {ood_emb.shape}")

    # Save config
    config = {
        'dataset': args.dataset,
        'architecture': args.architecture,
        'split_type': args.split_type,
        'fold': args.fold,
        'seed': args.seed,
        'n_train': len(train_idx),
        'n_id_test': len(id_test_idx),
        'n_ood_test': len(ood_test_idx),
        'best_val_mae': float(best_val),
        'embedding_dim': int(train_emb.shape[1]),
    }
    with open(os.path.join(ckpt_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    print(f"\n  Checkpoint: {ckpt_dir}/")
    print(f"  Embeddings: {emb_dir}/")
    print("  DONE")


if __name__ == '__main__':
    main()
