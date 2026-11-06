"""Figure 1 — Option B (revised): 5 datasets (left column) + 15 split icons in a 3×5 grid (right).

No bounding boxes — icons sit directly on the page with labels. Dataset
icons are slightly larger than split icons. The first 2 rows of the 3×5
grid are compositional splits (6); the bottom 3 rows are structural (9).

Output: paper/submission/dataset_split_overview.{png,pdf}
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
STRUCT_DIR = ROOT / "paper" / "submission" / "_fig6_structs"
OUT_DIR = ROOT / "paper" / "submission"

INK = "#222222"

DATASETS = [
    ("matbench_dielectric",  "dielectric",   "4,764"),
    ("matbench_log_gvrh",    "log_gvrh",     "10,987"),
    ("matbench_log_kvrh",    "log_kvrh",     "10,987"),
    ("matbench_perovskites", "perovskites",  "18,928"),
    ("matbench_phonons",     "phonons",      "1,265"),
]

# 3×5 grid of splits: rows 1–2 = compositional (6 cells, last cell of row 2
# leaves room for 9 structural in rows 3–5). Order: row-major reading.
SPLITS_GRID = [
    # row 0 (compositional)
    [("comp_carbide",       "carbide",       "mid"),
     ("comp_chalcogenide",  "chalcogenide",  "mid"),
     ("comp_halide",        "halide",        "mid")],
    # row 1 (compositional)
    [("comp_intermetallic", "intermetallic", "mid"),
     ("comp_nitride",       "nitride",       "mid"),
     ("comp_oxide",         "oxide",         "mid")],
    # row 2 (structural)
    [("struct_cubic",       "cubic",         "bot"),
     ("struct_hexagonal",   "hexagonal",     "bot"),
     ("struct_tetragonal",  "tetragonal",    "bot")],
    # row 3 (structural)
    [("struct_orthorhombic","orthorhombic",  "bot"),
     ("struct_monoclinic",  "monoclinic",    "bot"),
     ("struct_triclinic",   "triclinic",     "bot")],
    # row 4 (structural)
    [("struct_high_spacegroup","high-spgrp", "bot"),
     ("struct_low_spacegroup", "low-spgrp",  "bot"),
     ("struct_perovskite",  "perovskite",    "bot")],
]


def main():
    # Figsize 28×28 (was 20×20) so each data unit = 0.28 inch in both axes.
    # ds_icon_half = 7.5 → icon 15 data units = 4.2 inches (matches Option A's
    # ~4.4" top/middle rows). sp_icon_half = 6.25 → icon ≈ 3.5" (matches A's
    # ~4" bottom row).
    fig, ax = plt.subplots(figsize=(28, 25), dpi=200)
    XR = 100   # x-range for axes-relative fraction conversion
    YR = 100   # y-range — column titles sit just above the boxes
    ax.set_xlim(0, XR)
    ax.set_ylim(0, YR)
    ax.set_axis_off()

    # ─── Column titles, sitting just above the box / dataset content
    ax.text(10.5, 96.5, "5 Tasks",
            ha="center", va="bottom",
            fontsize=28, fontweight="bold", color=INK)
    ax.text((29.5 + 96.5) / 2, 96.5, "15 Splits",
            ha="center", va="bottom",
            fontsize=28, fontweight="bold", color=INK)

    # ─── Datasets column (5 rows) ──────────────────────────────────────────
    # row_top chosen so the first row's label top sits just below the column
    # title, aligned with comp_y_top (95) on the splits side.
    ds_x_center = 10.5   # x-center of dataset column (shifted slightly so
                         # left + right margins of the figure are symmetric)
    ds_icon_half = 7.0
    row_top = 95.8
    row_bot = 0.8
    n_ds = 5
    ds_row_h = (row_top - row_bot) / n_ds   # = 19
    ds_cy = [row_top - (i + 0.5) * ds_row_h for i in range(n_ds)]
    for i, (key, name, sub) in enumerate(DATASETS):
        cy = ds_cy[i]
        png = STRUCT_DIR / f"top_{key}.png"
        if png.exists():
            inset = ax.inset_axes([
                (ds_x_center - ds_icon_half) / XR,
                (cy - ds_icon_half) / YR,
                (2 * ds_icon_half) / XR,
                (2 * ds_icon_half) / YR,
            ])
            inset.imshow(Image.open(png))
            inset.set_axis_off()
        # Single-line label: "name (N = ...)" — same font as dataset name.
        ax.text(ds_x_center, cy + ds_icon_half + 0.5,
                f"{name}  ({sub})",
                ha="center", va="bottom",
                fontsize=24, fontweight="bold", color=INK)

    # ─── Splits 3×5 grid with green (compositional) and purple (structural)
    # background regions. Titles sit at the top of each region, inside the
    # box, above the icons.
    # Splits box: same width as before, shifted right by 1.5 data units so
    # left and right margins of the figure are symmetric.
    sp_x0 = 29.5
    sp_x1 = 96.5
    n_cols = 3
    col_w = (sp_x1 - sp_x0) / n_cols           # ≈ 22.3
    sp_icon_half = 6.25

    COMP_FILL = "#d9eed4"      # soft green
    COMP_EDGE = "#5a9c52"
    STRUCT_FILL = "#ebd9ee"    # soft purple
    STRUCT_EDGE = "#8a5a9c"

    # Region boundaries — content vertically centered within each rect.
    comp_y_top    = 95
    comp_y_bot    = 56
    struct_y_top  = 54
    struct_y_bot  = 1

    # Icon-row cy values — chosen so icons + labels sit cleanly inside each
    # rect without overlapping each other or the region title.
    sp_cy = [83,         # row 0 (compositional)
             67,         # row 1 (compositional)
             42,         # row 2 (structural)
             27,         # row 3 (structural)
             12]         # row 4 (structural)

    # Draw colored backgrounds (zorder=0).
    ax.add_patch(Rectangle((sp_x0, comp_y_bot),
                           sp_x1 - sp_x0, comp_y_top - comp_y_bot,
                           facecolor=COMP_FILL, edgecolor=COMP_EDGE,
                           linewidth=1.4, zorder=0))
    ax.add_patch(Rectangle((sp_x0, struct_y_bot),
                           sp_x1 - sp_x0, struct_y_top - struct_y_bot,
                           facecolor=STRUCT_FILL, edgecolor=STRUCT_EDGE,
                           linewidth=1.4, zorder=0))

    # ─── Arrows: each dataset → left-center of each colored region ─────────
    src_x = ds_x_center + ds_icon_half + 0.5     # right edge of dataset icon
    comp_target = (sp_x0 - 0.3, (comp_y_top + comp_y_bot) / 2)
    struct_target = (sp_x0 - 0.3, (struct_y_top + struct_y_bot) / 2)
    for cy in ds_cy:
        for target in (comp_target, struct_target):
            ax.add_patch(FancyArrowPatch(
                (src_x, cy), target,
                arrowstyle="-|>,head_length=8,head_width=5",
                color="black", lw=2.4, mutation_scale=1, alpha=0.95,
                zorder=1))

    # Region titles centered horizontally at the top of each box.
    ax.text((sp_x0 + sp_x1) / 2, comp_y_top - 1.0,
            "Compositional (6)", ha="center", va="top",
            fontsize=26, fontweight="bold", color="#2f5a2a")
    ax.text((sp_x0 + sp_x1) / 2, struct_y_top - 1.0,
            "Structural (9)", ha="center", va="top",
            fontsize=26, fontweight="bold", color="#5a2a6b")

    # Draw the 15 split icons.
    for r, row in enumerate(SPLITS_GRID):
        cy = sp_cy[r]
        for c, (key, label, prefix) in enumerate(row):
            cx = sp_x0 + (c + 0.5) * col_w
            png = STRUCT_DIR / f"{prefix}_{key}.png"
            if png.exists():
                inset = ax.inset_axes([
                    (cx - sp_icon_half) / XR,
                    (cy - sp_icon_half) / YR,
                    (2 * sp_icon_half) / XR,
                    (2 * sp_icon_half) / YR,
                ])
                inset.imshow(Image.open(png))
                inset.set_axis_off()
            ax.text(cx, cy - sp_icon_half - 1.0, label,
                    ha="center", va="top",
                    fontsize=22, fontweight="bold", color=INK)

    out_png = OUT_DIR / "dataset_split_overview.png"
    out_pdf = OUT_DIR / "dataset_split_overview.pdf"
    fig.savefig(out_png, bbox_inches="tight", dpi=300, facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
