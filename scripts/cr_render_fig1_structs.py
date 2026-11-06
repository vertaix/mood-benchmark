#!/usr/bin/env python3
"""Re-render the Figure 1 crystal thumbnails with CORRECT aspect ratios using
ASE's matplotlib backend (no POV-Ray). Equal axis scaling guarantees cubic cells
(NaCl, diamond Si, SrTiO3) look cubic. Reuses the exact structure builders and
slot mapping from scripts/plot_dataset_renders.py.

Run with an env that has ase, e.g.:
  python scripts/cr_render_fig1_structs.py

Writes square thumbnails to outputs/fig1_structs/{top,mid,bot}_<key>.png
and a contact-sheet outputs/fig1_structs/_contact_sheet.png.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from mood import paths as _P
sys.path.insert(0, HERE)
# import structure builders + slot lists + periodic padding from the repo script
import plot_dataset_renders as R

OUT = str(_P.outputs / "fig1_structs")
os.makedirs(OUT, exist_ok=True)
ROTATION = R.ROTATION   # "30x,15y,0z"


def render(atoms, out_png, scale_px=560):
    """Equal-aspect matplotlib render of a padded unit cell + atoms."""
    padded = R.pad_periodic_images(atoms)
    fig, ax = plt.subplots(figsize=(2.8, 2.8), dpi=scale_px / 2.8)
    plot_atoms(padded, ax, rotation=ROTATION, radii=0.5, show_unit_cell=2)
    ax.set_aspect("equal")          # <- cubic looks cubic
    ax.set_axis_off()
    fig.savefig(out_png, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    jobs = []
    for key, _sub, atoms in R.DATASET_STRUCTS:
        jobs.append((f"top_{key}", atoms))
    for key, atoms in R.COMPOSITIONAL_STRUCTS:
        jobs.append((f"mid_{key}", atoms))
    for key, atoms in R.STRUCTURAL_STRUCTS:
        jobs.append((f"bot_{key}", atoms))
    for name, atoms in jobs:
        render(atoms, os.path.join(OUT, name + ".png"))
        print("rendered", name)

    # contact sheet for a quick cubic-looks-cubic sanity check
    from PIL import Image
    cols = 5
    rows = (len(jobs) + cols - 1) // cols
    thumbs = [(n, Image.open(os.path.join(OUT, n + ".png")).convert("RGBA")) for n, _ in jobs]
    cw = max(t.size[0] for _, t in thumbs); ch = max(t.size[1] for _, t in thumbs)
    sheet = Image.new("RGBA", (cols * cw, rows * ch), (255, 255, 255, 255))
    for i, (n, t) in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet.paste(t, (c * cw + (cw - t.size[0]) // 2, r * ch + (ch - t.size[1]) // 2), t)
    sheet.convert("RGB").save(os.path.join(OUT, "_contact_sheet.png"))
    print("wrote _contact_sheet.png")


if __name__ == "__main__":
    main()
