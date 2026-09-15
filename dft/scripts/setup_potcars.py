#!/usr/bin/env python3
import os
import sys
import shutil

ROOT_DIR = "DFT_Runs"
TEMPLATE_DIR = "TEMPLATE"

def read_species_from_poscar(poscar_path):
    """
    Parse a VASP5-style POSCAR to get element symbols from line 6.
    Assumes:
      line 1: comment
      line 2: scaling
      lines 3-5: lattice vectors
      line 6: element symbols
      line 7: counts
    """
    with open(poscar_path, "r") as f:
        lines = f.readlines()

    if len(lines) < 7:
        raise ValueError(f"POSCAR too short: {poscar_path}")

    species = lines[5].split()
    if not species:
        raise ValueError(f"Could not read species line in {poscar_path}")

    return species

def build_potcar_for_dir(job_dir):
    poscar_path = os.path.join(job_dir, "POSCAR")
    if not os.path.isfile(poscar_path):
        return  # nothing to do

    species = read_species_from_poscar(poscar_path)

    out_path = os.path.join(job_dir, "POTCAR")
    # Overwrite existing POTCAR to keep things consistent
    with open(out_path, "wb") as fout:
        for el in species:
            potcar_name = f"POTCAR_{el}"
            potcar_src = os.path.join(TEMPLATE_DIR, potcar_name)
            if not os.path.isfile(potcar_src):
                raise FileNotFoundError(
                    f"Missing {potcar_src} required for {job_dir}"
                )
            with open(potcar_src, "rb") as fin:
                fout.write(fin.read())

    print(f"[OK] POTCAR written in {job_dir} (elements: {', '.join(species)})")

def main():
    if not os.path.isdir(ROOT_DIR):
        print(f"ERROR: {ROOT_DIR} directory not found.", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(TEMPLATE_DIR):
        print(f"ERROR: {TEMPLATE_DIR} directory not found.", file=sys.stderr)
        sys.exit(1)

    for root, dirs, files in os.walk(ROOT_DIR):
        if "POSCAR" in files:
            build_potcar_for_dir(root)

    print("Done building POTCARs for all POSCAR-containing directories.")

if __name__ == "__main__":
    main()
