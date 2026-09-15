#!/usr/bin/env python3
"""
Fill missing POSCARs using automatic supercell scaling if ratios are not integer.

This script:
- Reads dirs.txt
- Processes only leaf folders (those meant to contain POSCAR)
- Parses composition from folder name
- Finds the SMALLEST supercell where composition gives integer site counts
- Builds the B2 supercell and writes POSCAR

Supports:
- 02_Binary_B2
- 05_B2_Ternaries_Solubility
- 06_B2_Penalty_RuSubstitution
- 07_B2_HighOrder
"""

import os
import re
import random
import numpy as np
from ase import Atoms
from ase.io.vasp import write_vasp

DIRS_FILE = "dirs.txt"
A0 = 3.1

random.seed(42)

SUPER_OPTIONS = [
    (2,2,2),   # 8 sites per sublattice
    (2,2,3),   # 12 sites
    (3,2,2),   # 12 sites
    (3,2,3),   # 18 sites
    (3,3,2),   # 18 sites
    (3,3,3),   # 27 sites
    (4,2,2),   # 16 sites
    (4,2,3),   # 24 sites
    (4,3,3),   # 36 sites
]


def is_leaf_folder(path):
    """Folder is leaf if it contains no subdirectories."""
    for _, dirs, files in os.walk(path):
        return len(dirs) == 0
    return False


# -----------------------------------------------------------
# B2 SUPERCELL GENERATOR WITH AUTO-SCALING
# -----------------------------------------------------------

def build_b2_supercell_auto(a_comp, b_comp):
    """
    a_comp, b_comp = dict of element → fraction (sum to 1.0)
    Example: {"Ru":0.875, "Al":0.125}

    We try available supercells until counts become integer.
    """
    for (nx, ny, nz) in SUPER_OPTIONS:
        n_sites = nx * ny * nz

        # desired integer counts
        a_counts = {el: round(frac * n_sites) for el, frac in a_comp.items()}
        b_counts = {el: round(frac * n_sites) for el, frac in b_comp.items()}

        # check integer validity
        if sum(a_counts.values()) != n_sites:
            continue
        if sum(b_counts.values()) != n_sites:
            continue

        # Build lists
        a_list = []
        for el, cnt in a_counts.items():
            a_list.extend([el] * cnt)
        b_list = []
        for el, cnt in b_counts.items():
            b_list.extend([el] * cnt)

        random.shuffle(a_list)
        random.shuffle(b_list)

        # Coordinates
        positions = []
        symbols = []
        cell = np.diag([A0 * nx, A0 * ny, A0 * nz])

        # A sublattice
        idx = 0
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    positions.append([i/nx, j/ny, k/nz])
                    symbols.append(a_list[idx])
                    idx += 1

        # B sublattice
        idx = 0
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    positions.append([(i+0.5)/nx, (j+0.5)/ny, (k+0.5)/nz])
                    symbols.append(b_list[idx])
                    idx += 1

        atoms = Atoms(symbols=symbols, scaled_positions=positions, cell=cell, pbc=True)
        return atoms

    raise RuntimeError("Could not find any supercell size with integer composition.")


# -----------------------------------------------------------
# PARSERS
# -----------------------------------------------------------

def parse_b2_binary(folder):
    name = os.path.basename(folder)
    m = re.search(r"Ru([A-Za-z]+)", name)
    partner = m.group(1)
    return {"Ru":1.0}, {partner:1.0}


def parse_b2_ternary_sol(folder):
    name = os.path.basename(folder)
    m = re.search(r"Ru_([A-Za-z]+)(\d+)_([A-Za-z]+)(\d+)", name)
    partner, pct_p, matrix, pct_m = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
    total = pct_p + pct_m
    return {"Ru":1.0}, {
        partner: pct_p/total,
        matrix: pct_m/total,
    }


def parse_b2_penalty(folder):
    name = os.path.basename(folder)

    # RuXXDopYY
    m = re.search(r"Ru(\d+)([A-Za-z]+)(\d+)", name)
    if m:
        pct_ru = int(m.group(1))
        dop = m.group(2)
        pct_dop = int(m.group(3))
        total = pct_ru + pct_dop
        a = {
            "Ru": pct_ru/total,
            dop: pct_dop/total,
        }
    else:
        # Ru100X0 pure case
        m = re.search(r"Ru100([A-Za-z]+)0", name)
        dop = m.group(1)
        a = {"Ru":1.0}

    # partner on B sublattice
    m2 = re.search(r"_([A-Za-z]+)$", name)
    partner = m2.group(1)

    return a, {partner:1.0}


def parse_b2_multi(folder):
    name = os.path.basename(folder)

    # A_RuXX_AlYY
    m = re.search(r"A_Ru(\d+)_Al(\d+)", name)
    pct_ru = int(m.group(1))
    pct_al = int(m.group(2))
    total = pct_ru + pct_al
    a = {
        "Ru": pct_ru/total,
        "Al": pct_al/total,
    }

    # B_partnerAl (50/50)
    m2 = re.search(r"B_([A-Za-z]+)Al", name)
    partner = m2.group(1)

    b = {partner:0.5, "Al":0.5}
    return a, b


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------

def main():
    print("Reading dirs.txt…")

    with open(DIRS_FILE, "r") as f:
        lines = f.readlines()

    missing = [l.split()[1] for l in lines if l.startswith("MISS")]

    print(f"Found {len(missing)} missing folders.")
    for folder in missing:
        if not os.path.isdir(folder):
            continue
        if not is_leaf_folder(folder):
            continue

        name = folder.replace("\\", "/")

        # Parse based on folder type
        if "/02_Binary_B2/" in name:
            a,b = parse_b2_binary(folder)
        elif "/05_B2_Ternaries_Solubility/" in name:
            a,b = parse_b2_ternary_sol(folder)
        elif "/06_B2_Penalty_RuSubstitution/" in name:
            a,b = parse_b2_penalty(folder)
        elif "/07_B2_HighOrder/" in name:
            a,b = parse_b2_multi(folder)
        else:
            print(f"[SKIP] Unknown folder type: {folder}")
            continue

        try:
            atoms = build_b2_supercell_auto(a, b)
        except Exception as e:
            print(f"[ERR] Failed {folder}: {e}")
            continue

        out = os.path.join(folder, "POSCAR")
        write_vasp(out, atoms, vasp5=True, direct=False)
        print(f"[OK] Wrote POSCAR → {folder}")


if __name__ == "__main__":
    main()
