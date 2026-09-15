#!/usr/bin/env python3
"""
Generate only the B2 structures that were previously skipped:

- B2_TERN_SOL_078–107: Ru on A-sublattice; (Hf/Ti/Zr) + (Nb/Mo/Ta/W/V) on B.
- B2_MULTI_163,164,167,168,169,170:
    A-sublattice: Ru + Al
    B-sublattice: (Hf/Ti/Zr) + Al

This script does NOT use icet. It just constructs a 2x2x2 B2 supercell
and decorates it randomly but exactly matching the requested composition.

It assumes the same directory layout as the original script:

DFT_Runs/
  05_B2_Ternaries_Solubility/B2_TERN_SOL_078_...
  07_B2_HighOrder/B2_MULTI_163_...
"""

import os
import random
import numpy as np
from ase import Atoms
from ase.io.vasp import write_vasp

ROOT_DIR = "DFT_Runs"
A0 = 3.1  # lattice constant (same as main script)
SUPERCELL = (2, 2, 2)  # 2x2x2 B2 → 16 atoms total (8 A, 8 B)

random.seed(42)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_b2_supercell(a_species, b_species, a0=A0, supercell=SUPERCELL) -> Atoms:
    """
    Build a 2x2x2 B2 supercell with:
        - A sites at (i/nx, j/ny, k/nz)
        - B sites at ((i+0.5)/nx, (j+0.5)/ny, (k+0.5)/nz)

    Parameters
    ----------
    a_species : list[str]
        Length nx*ny*nz species list for A-sublattice.
    b_species : list[str]
        Length nx*ny*nz species list for B-sublattice.
    """
    nx, ny, nz = supercell
    n_sites = nx * ny * nz
    assert len(a_species) == n_sites
    assert len(b_species) == n_sites

    # Supercell cell
    cell = np.diag([a0 * nx, a0 * ny, a0 * nz])

    symbols = []
    scaled_positions = []

    # A-sublattice
    idx = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                x = i / nx
                y = j / ny
                z = k / nz
                symbols.append(a_species[idx])
                scaled_positions.append((x, y, z))
                idx += 1

    # B-sublattice
    idx = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                x = (i + 0.5) / nx
                y = (j + 0.5) / ny
                z = (k + 0.5) / nz
                symbols.append(b_species[idx])
                scaled_positions.append((x, y, z))
                idx += 1

    atoms = Atoms(
        symbols=symbols,
        scaled_positions=scaled_positions,
        cell=cell,
        pbc=True,
    )
    return atoms


def write_poscar(path: str, atoms: Atoms) -> None:
    """
    Write VASP5 POSCAR with species grouped by element.
    """
    symbols = atoms.get_chemical_symbols()
    unique = []
    for s in symbols:
        if s not in unique:
            unique.append(s)

    # reorder atoms so that same species are contiguous
    indices = [i for sym in unique for i, s in enumerate(symbols) if s == sym]
    atoms_sorted = atoms[indices]

    write_vasp(
        os.path.join(path, "POSCAR"),
        atoms_sorted,
        vasp5=True,
        direct=False,
    )


def generate_b2_ternary_solubility():
    """
    Generate the 30 B2_TERN_SOL_078–107 structures:

    For each partner in [Hf, Ti, Zr]
    and matrix in [Nb, Mo, Ta, W, V]:
        - B-site: 7 partner + 1 matrix  → ~87/12 (label 87/12)
        - B-site: 6 partner + 2 matrix  → 75/25 (label 75/25)
    A-site: Ru (8 atoms)
    """
    partners = ["Hf", "Ti", "Zr"]
    matrices = ["Nb", "Mo", "Ta", "W", "V"]

    # (n_partner, n_matrix, label_partner_pct, label_matrix_pct)
    comp_specs = [
        (7, 1, 87, 12),
        (6, 2, 75, 25),
    ]

    index = 78  # match original numbering from your log

    base_dir = os.path.join(ROOT_DIR, "05_B2_Ternaries_Solubility")
    for partner in partners:
        for matrix in matrices:
            for n_partner, n_matrix, pct_partner, pct_matrix in comp_specs:
                # A-sublattice: 8 Ru
                a_species = ["Ru"] * 8

                # B-sublattice: n_partner of partner, n_matrix of matrix
                b_list = [partner] * n_partner + [matrix] * n_matrix
                assert len(b_list) == 8
                random.shuffle(b_list)

                label = f"B2_TERN_SOL_{index:03d}_Ru_{partner}{pct_partner}_{matrix}{pct_matrix}"
                path = os.path.join(base_dir, label)
                ensure_dir(path)

                atoms = build_b2_supercell(a_species, b_list)
                write_poscar(path, atoms)

                print(f"[OK] Wrote {label}/POSCAR")
                index += 1


def generate_b2_multi_highorder():
    """
    Generate the 6 problematic B2_MULTI structures:

    B2_MULTI_163_A_Ru87_Al12_B_HfAl
    B2_MULTI_164_A_Ru75_Al25_B_HfAl
    B2_MULTI_167_A_Ru87_Al12_B_TiAl
    B2_MULTI_168_A_Ru75_Al25_B_TiAl
    B2_MULTI_169_A_Ru87_Al12_B_ZrAl
    B2_MULTI_170_A_Ru75_Al25_B_ZrAl

    A-sublattice (8 sites): Ru + Al
    B-sublattice (8 sites): partner (Hf/Ti/Zr) + Al (4+4)
    """
    base_dir = os.path.join(ROOT_DIR, "07_B2_HighOrder")

    cases = [
        (163, 0.875, "Hf"),
        (164, 0.75, "Hf"),
        (167, 0.875, "Ti"),
        (168, 0.75, "Ti"),
        (169, 0.875, "Zr"),
        (170, 0.75, "Zr"),
    ]

    for idx, c_ru, partner in cases:
        n_ru = int(round(c_ru * 8))
        n_al_a = 8 - n_ru
        # B-sublattice: 4 partner + 4 Al
        n_partner_b = 4
        n_al_b = 4

        # Build species lists
        a_species = ["Ru"] * n_ru + ["Al"] * n_al_a
        b_species = [partner] * n_partner_b + ["Al"] * n_al_b

        assert len(a_species) == 8
        assert len(b_species) == 8

        random.shuffle(a_species)
        random.shuffle(b_species)

        pct_ru = int(c_ru * 100)
        pct_al = 100 - pct_ru  # just for label consistency

        label = f"B2_MULTI_{idx:03d}_A_Ru{pct_ru}_Al{pct_al}_B_{partner}Al"
        path = os.path.join(base_dir, label)
        ensure_dir(path)

        atoms = build_b2_supercell(a_species, b_species)
        write_poscar(path, atoms)

        print(f"[OK] Wrote {label}/POSCAR")


def main():
    print(f"Generating missing B2 structures under '{ROOT_DIR}'")
    generate_b2_ternary_solubility()
    generate_b2_multi_highorder()
    print("Done.")


if __name__ == "__main__":
    main()
