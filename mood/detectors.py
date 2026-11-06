"""Reference detectors as plain callables.

Every detector has the signature ``score_fn(z_train, z_test) -> scores`` and
returns one float per row of ``z_test``. **Convention: higher score = more
in-distribution (ID-like).** If your method naturally scores anomalousness
(e.g. a distance), return its negative.

``cosine`` and ``knn_k20`` reproduce the "Cosine" and "KNN k=20 norm" rows of the
released per-setting results; the full set of 15 detectors lives in
``scripts/run_detectors.py`` and ``scripts/run_extra_detectors.py``.
"""
import numpy as np

from mood.evaluate import evaluate_detector  # noqa: F401  (convenience re-export)


def cosine(z_train, z_test):
    """Max cosine similarity to any training embedding (Cosine, k=1)."""
    from scripts.run_extra_detectors import CosineDetector

    return CosineDetector(k=1).fit(z_train).score(z_test)


def knn_k20(z_train, z_test, k=20):
    """Negative squared L2 distance to the k-th nearest training neighbour,
    on L2-normalised embeddings ("KNN k=20 norm")."""
    import faiss

    def _normalize(x):  # identical to scripts/run_detectors.py::normalize
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)

    tr = _normalize(z_train).astype(np.float32)
    te = _normalize(z_test).astype(np.float32)
    index = faiss.IndexFlatL2(tr.shape[1])
    index.add(tr)
    dist, _ = index.search(te, min(k, len(tr)))
    return -dist[:, k - 1]


DETECTORS = {"cosine": cosine, "knn_k20": knn_k20}
