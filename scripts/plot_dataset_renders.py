"""Figure 6 (paper Figure 2) — Material datasets and split criteria.

Three-row figure of canonical crystal structures rendered via POV-Ray for
publication quality. Top row: 5 dataset representatives. Middle row:
6 compositional split representatives. Bottom row: 9 structural split
representatives.

Output: paper/submission/dataset_renders.{png,pdf}
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import itertools

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
import numpy as np

from ase import Atoms
from ase.io import write as ase_write
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.spacegroup import crystal

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "paper" / "submission"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INK = "#222222"
ROTATION = "30x,15y,0z"

# Per-slot overrides passed as kwargs to render_pov(). Slot keys match labels
# in DATASET_STRUCTS, COMPOSITIONAL_STRUCTS, and STRUCTURAL_STRUCTS.
#
# Ionic structures: covalent-radius cutoffs capture spurious cation-cation
# contacts. List the element pairs that are NOT chemical bonds in the
# structure so they get dropped.
_IONIC_AGGREGATE = [("Ca", "Ca"), ("C", "C"), ("O", "O"), ("Ca", "C")]
BOND_OVERRIDES = {
    # Rutile TiO2: only Ti-O bonds.
    "struct_tetragonal":    dict(exclude_pairs=[("Ti", "Ti")]),
    "matbench_dielectric":  dict(exclude_pairs=[("Ti", "Ti")]),
    # SrTiO3 perovskite: only Ti-O bonds (the octahedron view).
    "struct_perovskite":     dict(exclude_pairs=[("Sr", "Sr"), ("Sr", "Ti")]),
    "matbench_perovskites":  dict(exclude_pairs=[("Sr", "Sr"), ("Sr", "Ti")]),
    # Aragonite CaCO3: only Ca-O and C-O are chemical bonds.
    "struct_orthorhombic":   dict(exclude_pairs=_IONIC_AGGREGATE),
    # Gypsum CaSO4: only Ca-O and S-O are chemical bonds.
    "struct_monoclinic":     dict(exclude_pairs=[("Ca", "Ca"), ("Ca", "S")]),
    # CaF2 fluorite: only Ca-F bonds (Ca-Ca at 3.86 A is spurious).
    "comp_halide":           dict(exclude_pairs=[("Ca", "Ca")]),
    # Al2O3 corundum: only Al-O bonds (Al-Al at ~2.65 A is spurious).
    "comp_oxide":            dict(exclude_pairs=[("Al", "Al")]),
}


def pad_periodic_images(atoms, eps=0.02):
    """Duplicate atoms near cell faces to all symmetry-equivalent positions.

    For each atom, look at fractional coordinates along each axis. If the
    coord is within `eps` of 0, schedule a `+1` shift; if within `eps` of 1,
    schedule a `-1` shift. Take the cartesian product over the three axes
    and emit one copy per non-trivial shift combination.

    A corner atom at (0, 0, 0) produces 7 extra copies (all 8 corners).
    A face atom at (0, 0.5, 0.5) produces 1 extra copy.
    An interior atom is unchanged.
    """
    frac = atoms.get_scaled_positions(wrap=False)
    cell = atoms.get_cell()
    symbols = atoms.get_chemical_symbols()

    new_symbols = list(symbols)
    new_cart = list(atoms.get_positions())

    for i in range(len(atoms)):
        # Per-axis shift candidates (always include 0 = no shift)
        candidates = []
        for axis in range(3):
            shifts = [0]
            if frac[i, axis] < eps:
                shifts.append(1)
            elif frac[i, axis] > 1 - eps:
                shifts.append(-1)
            candidates.append(shifts)

        for shift in itertools.product(*candidates):
            if shift == (0, 0, 0):
                continue
            new_frac = frac[i] + np.array(shift)
            new_symbols.append(symbols[i])
            new_cart.append(new_frac @ cell)

    out = Atoms(symbols=new_symbols,
                positions=np.array(new_cart),
                cell=cell,
                pbc=True)
    return out


def compute_bondatoms(atoms, scale=1.15, exclude_pairs=()):
    """Return list of (i, j) index pairs to draw as bonds.

    `exclude_pairs` is an iterable of (element_a, element_b) tuples (e.g.
    [("Sr", "Sr"), ("Sr", "Ti")]) — bonds where the two atoms match a pair
    in any order are dropped. Used for ionic structures where covalent
    radius cutoffs capture non-bonded cation-cation contacts.

    Caller must already have padded boundary images. PBC is forced off
    on a copy here to prevent double-counting bonds against ASE's
    implicit periodic images.
    """
    atoms = atoms.copy()
    atoms.set_pbc(False)

    cutoffs = natural_cutoffs(atoms, mult=scale)
    # skin=0.0: ASE's default 0.3 Å skin pushes the Na-Na capture radius above
    # the 3.99 Å rocksalt Na-Na distance, producing false homoatomic bonds.
    nl = NeighborList(cutoffs, skin=0.0, self_interaction=False, bothways=False)
    nl.update(atoms)

    excluded = {tuple(sorted(p)) for p in exclude_pairs}
    symbols = atoms.get_chemical_symbols()
    pairs = []
    for i in range(len(atoms)):
        indices, _ = nl.get_neighbors(i)
        for j in indices:
            if i < j:
                if tuple(sorted([symbols[i], symbols[j]])) in excluded:
                    continue
                pairs.append((int(i), int(j)))
    return pairs


# ─── Structures (real cell parameters from the literature) ──────────────
def NaCl():
    return crystal(["Na", "Cl"],
                   basis=[(0, 0, 0), (0.5, 0.5, 0.5)],
                   spacegroup=225,
                   cellpar=[5.6402, 5.6402, 5.6402, 90, 90, 90])


def rutile_TiO2():
    return crystal(["Ti", "O"],
                   basis=[(0, 0, 0), (0.305, 0.305, 0)],
                   spacegroup=136,
                   cellpar=[4.594, 4.594, 2.959, 90, 90, 90])


def SrTiO3():
    return crystal(["Sr", "Ti", "O"],
                   basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
                   spacegroup=221,
                   cellpar=[3.905, 3.905, 3.905, 90, 90, 90])


def Si_diamond():
    return crystal(["Si"],
                   basis=[(0, 0, 0)],
                   spacegroup=227,
                   cellpar=[5.431, 5.431, 5.431, 90, 90, 90])


def ZnS_wurtzite():
    return crystal(["Zn", "S"],
                   basis=[(1 / 3, 2 / 3, 0), (1 / 3, 2 / 3, 0.375)],
                   spacegroup=186,
                   cellpar=[3.811, 3.811, 6.234, 90, 90, 120])


def aragonite():
    """Aragonite CaCO3 — orthorhombic Pmcn, simpler alternative to olivine."""
    return crystal(
        ["Ca", "C", "O", "O"],
        basis=[(0.25, 0.4151, 0.7591),
               (0.25, 0.7621, 0.0832),
               (0.25, 0.9224, 0.0961),
               (0.4729, 0.6810, 0.0858)],
        spacegroup=62,
        cellpar=[4.961, 7.967, 5.741, 90, 90, 90],
    )


def gypsum():
    """Gypsum CaSO4·2H2O — monoclinic C2/c. Skip H atoms for clarity."""
    return crystal(
        ["Ca", "S", "O", "O", "O"],
        basis=[(0.0, 0.0794, 0.25),
               (0.0, 0.0769, 0.75),
               (0.9743, 0.1305, 0.5524),
               (0.7569, 0.0226, 0.7491),
               (0.3819, 0.1846, 0.4561)],
        spacegroup=15,
        cellpar=[6.284, 15.197, 5.681, 90, 114.09, 90],
    )


def albite():
    """Albite NaAlSi3O8 — triclinic C-1, real low-symmetry feldspar."""
    return crystal(
        ["Na", "Al", "Si", "Si", "O", "O", "O", "O"],
        basis=[(0.268,  0.005, 0.139),    # Na
               (0.009,  0.166, 0.208),    # Al (T1o)
               (0.003,  0.819, 0.235),    # Si (T1m)
               (0.690,  0.110, 0.317),    # Si (T2o)
               (0.005,  0.131, 0.967),    # O_A1
               (0.591,  0.998, 0.279),    # O_A2
               (0.812,  0.110, 0.193),    # O_BO
               (0.018,  0.310, 0.260)],   # O_C0
        spacegroup=2,                       # P-1
        cellpar=[8.144, 12.787, 7.160,
                 94.26, 116.58, 87.71],
    )


def SiC_zincblende():
    """3C-SiC zincblende, F-43m (#216)."""
    return crystal(["Si", "C"],
                   basis=[(0, 0, 0), (0.25, 0.25, 0.25)],
                   spacegroup=216,
                   cellpar=[4.3596, 4.3596, 4.3596, 90, 90, 90])


def MoS2_2H():
    """2H-MoS2, P6_3/mmc (#194). Layered chalcogenide."""
    return crystal(["Mo", "S"],
                   basis=[(1/3, 2/3, 0.25), (1/3, 2/3, 0.621)],
                   spacegroup=194,
                   cellpar=[3.16, 3.16, 12.295, 90, 90, 120])


def CaF2_fluorite():
    """CaF2 fluorite, Fm-3m (#225). Classic ionic halide."""
    return crystal(["Ca", "F"],
                   basis=[(0, 0, 0), (0.25, 0.25, 0.25)],
                   spacegroup=225,
                   cellpar=[5.4626, 5.4626, 5.4626, 90, 90, 90])


def Cu3Au_L12():
    """Cu3Au L1_2 ordered intermetallic, Pm-3m (#221).
    Au at corners, Cu at face centers — visualises 3:1 compositional order."""
    return crystal(["Au", "Cu"],
                   basis=[(0, 0, 0), (0, 0.5, 0.5)],
                   spacegroup=221,
                   cellpar=[3.748, 3.748, 3.748, 90, 90, 90])


def GaN_wurtzite():
    """GaN wurtzite, P6_3mc (#186). LED nitride."""
    return crystal(["Ga", "N"],
                   basis=[(1/3, 2/3, 0), (1/3, 2/3, 0.376)],
                   spacegroup=186,
                   cellpar=[3.189, 3.189, 5.185, 90, 90, 120])


def Al2O3_corundum():
    """Al2O3 corundum (sapphire), R-3c (#167) hexagonal setting."""
    return crystal(["Al", "O"],
                   basis=[(0, 0, 0.3523), (0.3062, 0, 0.25)],
                   spacegroup=167,
                   cellpar=[4.759, 4.759, 12.991, 90, 90, 120])


def MgB2():
    """MgB2, P6/mmm (#191). Layered diboride: B honeycomb sheets with
    Mg in between; canonical anisotropic-shear reference."""
    return crystal(["Mg", "B"],
                   basis=[(0, 0, 0), (1/3, 2/3, 0.5)],
                   spacegroup=191,
                   cellpar=[3.086, 3.086, 3.524, 90, 90, 120])


# ─── Mapping of slots to structures ─────────────────────────────────────
DATASET_STRUCTS = [
    ("matbench_dielectric",    "n = 4,764\nrefractive index", rutile_TiO2()),
    ("matbench_log_gvrh",      "n = 10,987\nlog shear (GPa)", MgB2()),
    ("matbench_log_kvrh",      "n = 10,987\nlog bulk (GPa)",  NaCl()),
    ("matbench_perovskites",   "n = 18,928\nformation (eV)",  SrTiO3()),
    ("matbench_phonons",       "n = 1,265\nphonon ω (1/cm)",  Si_diamond()),
]

COMPOSITIONAL_STRUCTS = [
    ("comp_carbide",       SiC_zincblende()),
    ("comp_chalcogenide",  MoS2_2H()),
    ("comp_halide",        CaF2_fluorite()),
    ("comp_intermetallic", Cu3Au_L12()),
    ("comp_nitride",       GaN_wurtzite()),
    ("comp_oxide",         Al2O3_corundum()),
]

STRUCTURAL_STRUCTS = [
    ("struct_cubic",            NaCl()),
    ("struct_hexagonal",        ZnS_wurtzite()),
    ("struct_tetragonal",       rutile_TiO2()),
    ("struct_orthorhombic",     aragonite()),
    ("struct_monoclinic",       gypsum()),
    ("struct_triclinic",        albite()),
    ("struct_high_spacegroup",  Si_diamond()),
    ("struct_low_spacegroup",   albite()),
    ("struct_perovskite",       SrTiO3()),
]


# ─── POV-Ray rendering ──────────────────────────────────────────────────
def render_pov(atoms, out_png, size_px=520, label=None, scale=1.15,
               exclude_pairs=()):
    """Render atoms to a PNG via POV-Ray, with periodic padding + bonds."""
    padded = pad_periodic_images(atoms)
    bondatoms = compute_bondatoms(padded, scale=scale,
                                  exclude_pairs=exclude_pairs)
    if label is not None:
        print(f"    {label}: {len(padded)} atoms, {len(bondatoms)} bonds")

    tmpdir = Path(tempfile.mkdtemp(prefix="ase_pov_"))
    try:
        pov_path = tmpdir / "struct.pov"
        png_path = tmpdir / "struct.png"

        ase_write(
            str(pov_path), padded,
            format="pov",
            rotation=ROTATION,
            povray_settings=dict(
                canvas_width=size_px,
                transparent=True,
                background="White",
                camera_dist=18.0,
                camera_type="orthographic",
                point_lights=[
                    [(2.0, 3.0, 40.0), "White"],
                ],
                area_light=[(2.0, 3.0, 40.0), "White", 0.7, 0.7, 3, 3],
                pause=False,
                exportconstraints=False,
                celllinewidth=0.05,
                bondlinewidth=0.10,
                bondatoms=bondatoms,
            ),
            radii=0.55,
            colors=None,
            show_unit_cell=2,
        )
        subprocess.run(
            ["povray", "+I" + str(pov_path),
             "+O" + str(png_path),
             "+W" + str(size_px), "+H" + str(size_px),
             "+A0.1", "+UA", "Quality=9",
             "Display=Off", "Verbose=Off"],
            check=True, capture_output=True,
        )
        shutil.copy(png_path, out_png)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    structs_dir = OUT_DIR / "_fig6_structs"
    structs_dir.mkdir(exist_ok=True)

    print("Rendering top row …")
    top_pngs = []
    for name, sub, atoms in DATASET_STRUCTS:
        out = structs_dir / f"top_{name}.png"
        if not out.exists():
            print(f"  {name} -> {out.name}")
            override = BOND_OVERRIDES.get(name, {})
            render_pov(atoms, out, label=name, **override)
        top_pngs.append((name, sub, out))

    print("Rendering middle row (compositional splits) …")
    mid_pngs = []
    for name, atoms in COMPOSITIONAL_STRUCTS:
        out = structs_dir / f"mid_{name}.png"
        if not out.exists():
            print(f"  {name} -> {out.name}")
            override = BOND_OVERRIDES.get(name, {})
            render_pov(atoms, out, label=name, **override)
        mid_pngs.append((name, out))

    print("Rendering bottom row (structural splits) …")
    bot_pngs = []
    for name, atoms in STRUCTURAL_STRUCTS:
        out = structs_dir / f"bot_{name}.png"
        if not out.exists():
            print(f"  {name} -> {out.name}")
            override = BOND_OVERRIDES.get(name, {})
            render_pov(atoms, out, label=name, **override)
        bot_pngs.append((name, out))

    # ─── Compose the figure ───
    fig = plt.figure(figsize=(36, 21), dpi=200)
    gs = GridSpec(3, 90, figure=fig,
                  height_ratios=[1, 1, 1],
                  hspace=0.0, wspace=0.0,
                  left=0.05, right=0.99, top=0.95, bottom=0.02)

    # Strip prefix for display only (data files keep full slot names).
    def short(name):
        for prefix in ("matbench_", "comp_", "struct_"):
            if name.startswith(prefix):
                return name[len(prefix):]
        return name

    # Top row: 5 cells, centered. Each cell spans 10 cols, starting at col 20.
    for i, (name, sub, png) in enumerate(top_pngs):
        c0 = 20 + i * 10
        ax = fig.add_subplot(gs[0, c0:c0 + 10])
        img = Image.open(png)
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(f"{short(name)}\n{sub}",
                     fontsize=28, color=INK, pad=12, linespacing=1.2)

    # Middle row: 6 cells, centered. Each cell spans 10 cols, starting at col 15.
    for i, (name, png) in enumerate(mid_pngs):
        c0 = 15 + i * 10
        ax = fig.add_subplot(gs[1, c0:c0 + 10])
        img = Image.open(png)
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(short(name), fontsize=30, color=INK,
                     fontweight="bold", pad=12)

    # Bottom row: 9 cells, full width. Each cell spans 10 cols, starting at col 0.
    for i, (name, png) in enumerate(bot_pngs):
        c0 = i * 10
        ax = fig.add_subplot(gs[2, c0:c0 + 10])
        img = Image.open(png)
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(short(name), fontsize=28, color=INK,
                     fontweight="bold", pad=12)

    # Row labels on the left, in the middle of each row.
    fig.text(0.012, 0.79, "Datasets",
             fontsize=36, fontweight="bold", color=INK,
             rotation=90, ha="center", va="center")
    fig.text(0.034, 0.79, "(prediction targets)",
             fontsize=28, color="#555555",
             rotation=90, ha="center", va="center")

    fig.text(0.012, 0.49, "Compositional splits",
             fontsize=36, fontweight="bold", color=INK,
             rotation=90, ha="center", va="center")
    fig.text(0.034, 0.49, "(by element family)",
             fontsize=28, color="#555555",
             rotation=90, ha="center", va="center")

    fig.text(0.012, 0.18, "Structural splits",
             fontsize=36, fontweight="bold", color=INK,
             rotation=90, ha="center", va="center")
    fig.text(0.034, 0.18, "(by lattice / symmetry)",
             fontsize=28, color="#555555",
             rotation=90, ha="center", va="center")

    fig.suptitle("Material Datasets And Split Criteria",
                 fontsize=44, fontweight="bold", color=INK, y=0.985)

    out_png = OUT_DIR / "dataset_renders.png"
    out_pdf = OUT_DIR / "dataset_renders.pdf"
    fig.savefig(out_png, bbox_inches="tight", dpi=300, facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


def _smoke_test():
    # Single atom at origin in a 1×1×1 unit cell → expect 8 atoms after padding
    a = Atoms("H", positions=[(0, 0, 0)],
              cell=[1, 1, 1], pbc=True)
    padded = pad_periodic_images(a)
    assert len(padded) == 8, f"corner atom: expected 8, got {len(padded)}"

    # Atom at (0.999, 0.999, 0.999) (floating-point noise) → also expect 8
    a2 = Atoms("H", positions=[(0.999, 0.999, 0.999)],
               cell=[1, 1, 1], pbc=True)
    padded2 = pad_periodic_images(a2)
    assert len(padded2) == 8, f"near-1 corner: expected 8, got {len(padded2)}"

    # Atom at (0.5, 0.5, 0.5) (interior) → expect 1
    a3 = Atoms("H", positions=[(0.5, 0.5, 0.5)],
               cell=[1, 1, 1], pbc=True)
    padded3 = pad_periodic_images(a3)
    assert len(padded3) == 1, f"interior: expected 1, got {len(padded3)}"

    # Face atom at (0, 0.5, 0.5) → expect 2 (this face + opposite face)
    a4 = Atoms("H", positions=[(0, 0.5, 0.5)],
               cell=[1, 1, 1], pbc=True)
    padded4 = pad_periodic_images(a4)
    assert len(padded4) == 2, f"face: expected 2, got {len(padded4)}"

    print("pad_periodic_images smoke test: PASS")

    # Bond test: NaCl primitive should have Na-Cl pairs after padding.
    nacl = NaCl()
    padded_nacl = pad_periodic_images(nacl)
    bonds_nacl = compute_bondatoms(padded_nacl)
    assert len(bonds_nacl) > 0, "NaCl: no bonds detected"
    # All bonds must be Na-Cl (no Na-Na or Cl-Cl in rocksalt at standard cutoff)
    syms = padded_nacl.get_chemical_symbols()
    for i, j in bonds_nacl:
        pair = tuple(sorted([syms[i], syms[j]]))
        assert pair == ("Cl", "Na"), f"NaCl: unexpected bond {pair}"
    # Sanity range — NaCl unpadded has 4 Na + 4 Cl. After padding, ~14 atoms.
    # Each interior Na has 6 Cl neighbors; rocksalt should give 24 < N < 60.
    assert 12 <= len(bonds_nacl) <= 60, \
        f"NaCl bond count {len(bonds_nacl)} out of expected range"
    print(f"compute_bondatoms NaCl: {len(bonds_nacl)} Na-Cl bonds (PASS)")


if __name__ == "__main__":
    import sys
    if "--smoke" in sys.argv:
        _smoke_test()
    else:
        main()
