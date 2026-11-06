"""Evaluate your own post-hoc OOD detector on MOOD in a few lines.

A detector is any function ``score_fn(z_train, z_test) -> scores`` returning one
float per row of ``z_test``. **Higher score = more in-distribution (ID-like).**
If your method scores anomalousness (a distance, an outlier score), return its
negative.

    python examples/add_my_detector.py                       # uses data/embeddings/
    python examples/add_my_detector.py --embeddings data/embeddings/matbench_phonons
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mood.evaluate import evaluate_detector  # noqa: E402


def mahalanobis_to_mean(z_train, z_test, reg=1e-3):
    """Toy detector: negative Mahalanobis distance to the training mean."""
    mu = z_train.mean(axis=0)
    cov = np.cov(z_train, rowvar=False) + reg * np.eye(z_train.shape[1])
    delta = z_test - mu
    d2 = np.einsum("ij,jk,ik->i", delta, np.linalg.inv(cov), delta)
    return -d2  # negate: large distance = OOD = low score


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", default="data/embeddings/")
    ap.add_argument("--output", default="results/my_detector.csv")
    a = ap.parse_args()
    rows = evaluate_detector(score_fn=mahalanobis_to_mean, embeddings_root=a.embeddings, output=a.output)
    mean = np.nanmean([r["auroc"] for r in rows]) if rows else float("nan")
    print(f"{len(rows)} cells, mean AUROC = {mean:.4f}  ->  {a.output}")
