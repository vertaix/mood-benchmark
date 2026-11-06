"""Single source of truth for filesystem paths (see configs/paths.yaml).

Relative entries resolve against the repository root. Set the environment
variable MOOD_PATHS to use a different yaml file.
"""
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
_CFG = Path(os.environ.get("MOOD_PATHS", REPO_ROOT / "configs" / "paths.yaml"))


def _load():
    with open(_CFG) as f:
        raw = yaml.safe_load(f)
    return {k: (Path(v) if Path(v).is_absolute() else REPO_ROOT / v) for k, v in raw.items()}


_P = _load()
embeddings = _P["embeddings"]
results = _P["results"]
splits = _P["splits"]
outputs = _P["outputs"]
flow_models = _P["flow_models"]
linear_probes = _P["linear_probes"]
linear_probes_mlp = _P["linear_probes_mlp"]
aderh = _P["aderh"]
