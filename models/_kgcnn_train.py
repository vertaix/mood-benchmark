"""Shared Keras training driver for kgcnn-backed models.

train() and eval() mirror the call signature of models/alignn/train.py so
scripts/train_model.py can dispatch to either backend by architecture.

Notes:
  - kgcnn 4.0.2 uses Keras 3 with TF backend. This driver uses keras directly
    (not tf.keras) per kgcnn's own conventions.
  - Loss = MSE on the (scaled) target; reported metric = MAE in original units.
    Standard scaling is fit on training targets and unwound at evaluation.
  - SWA / cosine schedules from the ALIGNN driver aren't replicated here;
    Keras's CosineDecay is used as a simpler equivalent.
"""
from typing import Dict, Tuple

import numpy as np


def _sanitize_inputs(batch: Dict[str, np.ndarray], input_names):
    """Filter batch dict to the inputs the model expects, in order."""
    return [batch[name] for name in input_names]


def fit_scaler(targets: np.ndarray):
    mean = float(np.mean(targets))
    std = float(np.std(targets) + 1e-9)
    return mean, std


def scale(targets: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (targets - mean) / std


def unscale(scaled: np.ndarray, mean: float, std: float) -> np.ndarray:
    return scaled * std + mean


def train(model_wrapper, train_batch, val_batch, *,
          epochs: int = 100, batch_size: int = 32,
          lr: float = 1e-3, ckpt_path: str = None,
          loss_kind: str = "mse",
          schedule_kind: str = "cosine",
          patience: int = 0):
    """Train a KGCNNModelBase wrapper.

    train_batch and val_batch are padded-batch dicts produced by the wrapper's
    make_dataset(...). Each contains a 'targets' key.
    """
    import keras
    from keras import losses, metrics, callbacks
    # CosineDecay lives at different paths in Keras 2 (megnet-bench, TF 2.10)
    # vs Keras 3 (kgcnn-bench, TF 2.16). Probe both.
    try:
        from keras.optimizers.schedules import CosineDecay
        from keras.optimizers import Adam
    except (ImportError, AttributeError):
        from tensorflow.keras.optimizers.schedules import CosineDecay
        from tensorflow.keras.optimizers import Adam

    train_inputs, train_targets = (
        {k: v for k, v in train_batch.items() if k != "targets"},
        train_batch["targets"],
    )
    val_inputs, val_targets = (
        {k: v for k, v in val_batch.items() if k != "targets"},
        val_batch["targets"],
    )

    mean, std = fit_scaler(train_targets)
    print(f"  target scaling: mean={mean:.4f}, std={std:.4f}")

    keras_model = model_wrapper.net
    input_names = [inp.name for inp in keras_model.inputs] \
        if hasattr(keras_model.inputs[0], "name") else None

    # Compile. Two schedule options:
    #  cosine: keras CosineDecay from `lr` to 0 over `epochs` steps.
    #  kgcnn:  LinearWarmupExponentialLRScheduler — kgcnn's matbench
    #          leaderboard recipe (warmup 10 ep + exponential decay
    #          5e-4 -> 5e-6 over 400 ep).
    train_loss = losses.MeanAbsoluteError() if loss_kind == "mae" \
                 else losses.MeanSquaredError()
    print(f"  loss: {loss_kind.upper()}  schedule: {schedule_kind}", flush=True)
    cbs_extra = []
    if schedule_kind == "kgcnn":
        from kgcnn.training.scheduler import LinearWarmupExponentialLearningRateScheduler
        keras_model.compile(
            optimizer=Adam(learning_rate=5e-4),
            loss=train_loss,
            metrics=[metrics.MeanAbsoluteError(name="mae")],
            jit_compile=False,
        )
        cbs_extra.append(LinearWarmupExponentialLearningRateScheduler(
            lr_start=5e-4, decay_lifetime=400, lr_min=5e-6, epo_warmup=10))
    else:
        schedule = CosineDecay(lr, decay_steps=epochs)
        keras_model.compile(
            optimizer=Adam(learning_rate=schedule),
            loss=train_loss,
            metrics=[metrics.MeanAbsoluteError(name="mae")],
            jit_compile=False,
        )

    cbs = list(cbs_extra)
    if ckpt_path:
        cbs.append(callbacks.ModelCheckpoint(
            filepath=ckpt_path, save_best_only=True, monitor="val_mae",
            mode="min", save_weights_only=False))
    if patience > 0:
        cbs.append(callbacks.EarlyStopping(
            monitor="val_mae", mode="min", patience=patience,
            restore_best_weights=True, verbose=1))
        print(f"  early stopping: patience={patience} on val_mae", flush=True)

    # If model expects a list, convert dicts → list in input order
    if input_names:
        x_train = [train_inputs[n] for n in input_names if n in train_inputs]
        x_val   = [val_inputs[n]   for n in input_names if n in val_inputs]
    else:
        x_train, x_val = train_inputs, val_inputs

    keras_model.fit(
        x_train, scale(train_targets, mean, std),
        validation_data=(x_val, scale(val_targets, mean, std)),
        epochs=epochs, batch_size=batch_size, callbacks=cbs, verbose=2,
    )

    # Unscaled val MAE for the final epoch
    val_pred_scaled = keras_model.predict(x_val, batch_size=batch_size, verbose=0)
    val_pred = unscale(val_pred_scaled, mean, std)
    val_mae = float(np.mean(np.abs(val_pred.ravel() - val_targets.ravel())))
    print(f"  final val MAE (unscaled): {val_mae:.4f}")
    return {"target_mean": mean, "target_std": std, "val_mae": val_mae}


def predict(model_wrapper, batch: Dict[str, np.ndarray],
            scaler: Tuple[float, float], batch_size: int = 32):
    """Return predictions in original units."""
    keras_model = model_wrapper.net
    inputs = {k: v for k, v in batch.items() if k != "targets"}
    input_names = [inp.name for inp in keras_model.inputs] \
        if hasattr(keras_model.inputs[0], "name") else None
    x = [inputs[n] for n in input_names if n in inputs] if input_names else inputs
    p = keras_model.predict(x, batch_size=batch_size, verbose=0)
    mean, std = scaler
    return unscale(p, mean, std)
