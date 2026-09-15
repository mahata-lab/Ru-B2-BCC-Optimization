#!/usr/bin/env python3
import os

ROOT = "DFT_Runs"

# Set of elements you actually use
ELEMENTS = {"Al", "Cr", "Cu", "Hf", "Mo", "Nb", "Ru",
            "Si", "Ta", "Ti", "V", "W", "Zr"}


def read_species_from_poscar(pos):
    with open(pos) as f:
        lines = f.readlines()
    # VASP5: line 6 is element symbols
    species = lines[5].split()
    return species


def read_species_from_potcar(pot):
    species = []
    with open(pot) as f:
        for line in f:
            if "TITEL" not in line:
                continue
            # Example: "TITEL  = PAW_PBE Ru 06Sep2000"
            tokens = line.split()
            # look for the token that looks like one of our elements
            el_found = None
            for tok in tokens:
                base = tok.split("_")[0]   # handle "Hf_pv", "Zr_sv", etc.
                if base in ELEMENTS:
                    el_found = base
                    break
            if el_found is None:
                # fall back just in case
                el_found = tokens[-1].split("_")[0]
            species.append(el_found)
    return species


for root, dirs, files in os.walk(ROOT):
    if "POSCAR" in files and "POTCAR" in files:
        pos = os.path.join(root, "POSCAR")
        pot = os.path.join(root, "POTCAR")

        pos_species = read_species_from_poscar(pos)
        pot_species = read_species_from_potcar(pot)

        if pos_species != pot_species:
            print(f"[ERROR] MISMATCH in {root}: "
                  f"POSCAR={pos_species}, POTCAR={pot_species}")
        else:
            print(f"[OK] {root}")
