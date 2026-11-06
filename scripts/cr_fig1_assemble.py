#!/usr/bin/env python3
"""Reassemble Figure 1 (dataset_split_overview) from the corrected, equal-aspect
thumbnails in outputs/fig1_structs/, using the repo's layout. Forces an
isotropic main axes so square thumbnails are placed in square physical boxes
(no stretch). Writes outputs/figures/dataset_split_overview.{png,pdf}.
"""
import os, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from mood import paths as _P
sys.path.insert(0, HERE)
import plot_dataset_split_overview as A

# redirect inputs/outputs to the camera-ready locations
A.STRUCT_DIR = Path(_P.outputs) / "fig1_structs"
A.OUT_DIR = Path(_P.outputs) / "figures"
A.OUT_DIR.mkdir(parents=True, exist_ok=True)

# Patch inset creation so every icon keeps a 1:1 physical aspect regardless of
# the figure's width:height, and imshow never stretches the thumbnail.
import matplotlib.pyplot as plt
_orig_inset = plt.Axes.inset_axes
def _square_inset(self, bounds, **kw):
    ax = _orig_inset(self, bounds, **kw)
    ax.set_aspect("equal", adjustable="box")
    return ax
plt.Axes.inset_axes = _square_inset

# Make the main canvas isotropic (28x28) so (2*half)/100 fractions map to equal
# physical inches on both axes.
import types
src = A.main.__code__
A.main.__globals__.setdefault("_patched", True)

if __name__ == "__main__":
    # monkeypatch figsize inside main by wrapping plt.subplots
    _orig_subplots = plt.subplots
    def _sub(*a, **k):
        if "figsize" in k:
            k["figsize"] = (28, 28)
        return _orig_subplots(*a, **k)
    plt.subplots = _sub
    A.main()
    print("reassembled -> outputs/figures/dataset_split_overview.{png,pdf}")
