#!/usr/bin/env python3
"""
Generate DFT input structures (POSCARs) for:
- Baseline pure elements (BCC)
- Binary Ru–X B2 (crystalline)
- Binary + ternary BCC matrices (SQS)
- B2 ternaries (Ru + partner + matrix) (SQS)
- B2 Ru→X substitution series (SQS)
- Higher-order B2 (multi-partner on B-site, 2-element mix) (SQS)

All POSCARs are written in proper VASP5 format with
elements grouped on the species line, e.g.:

Mo Ta Nb
6 8 4

Requires:
    ase
    icet
"""

import os
import itertools
import numpy as np

from ase import Atoms
from ase.build import bulk, make_supercell
from ase.io.vasp import write_vasp  # for proper POSCAR writing

from icet import ClusterSpace
from icet.tools.structure_generation import generate_sqs_from_supercells


# ============================================================
# CONFIG
# ============================================================

ROOT_DIR = "DFT_Runs"

# Cluster-space cutoffs: list[float], one per cluster order (pair, triplet, quadruplet, ...)
SQS_CUTOFFS = [5.0, 5.0, 5.0]

SUPERCELL_SIZE_B2 = [2, 2, 2]       # 16 atoms total (8 A, 8 B)
SUPERCELL_SIZE_BCC_BIN = [2, 2, 2]  # 16 atoms for binary BCC
SUPERCELL_SIZE_BCC_TERN = [3, 3, 1] # 18 atoms for ternary BCC (≤ 20–30 target)

NUM_SQS_STEPS = 10_000
RANDOM_SEED = 42

# Element sets
RU = "Ru"
REFRACTORY = ["Nb", "Mo", "Ta", "W", "V"]
B2_PRIMARY = ["Hf", "Ti", "Zr"]         # main Ru–B2 partners
B2_SECONDARY = ["Al"]                   # optional extra B2 partner
SUBSTITUTES = ["Al", "Cr", "Cu", "Si"]  # Ru-site dopants

# Track uniqueness: one structure per (phase, composition) combo
UNIQUE_COMPOSITIONS = set()  # entries: (phase_tag, tuple(sorted((el, frac), ...)))


# ============================================================
# HELPERS
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def create_directory(group_name: str, label: str) -> str:
    """Create directory ROOT_DIR/group_name/label and return its path."""
    path = os.path.join(ROOT_DIR, group_name, label)
    ensure_dir(path)
    return path


def write_poscar(filename: str, atoms: Atoms) -> None:
    """
    Write a VASP5 POSCAR with species grouped, e.g.:

    Mo Ta Nb
    6  8  4

    ASE's write_vasp will deduce species and counts from the Atoms object.
    We explicitly reorder atoms so that all atoms of the same element
    are contiguous, and each element appears only once in the header.
    """
    symbols = atoms.get_chemical_symbols()
    # Unique symbols in order of first appearance
    unique = []
    for s in symbols:
        if s not in unique:
            unique.append(s)

    # Reorder atoms so they are grouped by species
    indices = [i for sym in unique for i, s in enumerate(symbols) if s == sym]
    atoms_sorted = atoms[indices]

    # Write in VASP5 format
    write_vasp(
        filename,
        atoms_sorted,
        vasp5=True,
        direct=False  # set True if you prefer Direct coordinates
    )


def build_b2_primitive(site_a_elements, site_b_elements, a0=3.1) -> Atoms:
    """
    Build a primitive B2 (CsCl-type) cell.

    For multicomponent A or B, this is just a structural prototype;
    actual random mixing is handled via SQS.
    """
    # Two atoms: one at origin (A sublattice), one at body center (B sublattice)
    a_el = site_a_elements[0]
    b_el = site_b_elements[0]
    positions = [(0, 0, 0), (0.5, 0.5, 0.5)]
    cell = np.eye(3) * a0
    symbols = [a_el, b_el]
    atoms = Atoms(symbols=symbols, scaled_positions=positions, cell=cell, pbc=True)
    return atoms


def build_bcc_primitive(element: str, a0=3.1) -> Atoms:
    """Simple BCC primitive cell for a single element."""
    return bulk(element, "bcc", a=a0)


def get_b2_clusterspace(site_a_elements, site_b_elements, a0=3.1):
    """
    Build a ClusterSpace for B2 (two sublattices):
        - sublattice A: site_a_elements
        - sublattice B: site_b_elements

    icet expects chemical_symbols as a list of lists, one per site
    in the primitive structure.
    """
    prim = build_b2_primitive(site_a_elements, site_b_elements, a0=a0)
    chemical_symbols = [site_a_elements, site_b_elements]  # site 0 = A, site 1 = B
    cs = ClusterSpace(
        prim,
        cutoffs=SQS_CUTOFFS,
        chemical_symbols=chemical_symbols,
    )
    return cs, prim


def get_bcc_clusterspace(elements, a0=3.1):
    """
    Build a ClusterSpace for disordered BCC (single-sublattice) with given elements.
    """
    prim = build_bcc_primitive(elements[0], a0=a0)
    cs = ClusterSpace(
        prim,
        cutoffs=SQS_CUTOFFS,
        chemical_symbols=list(sorted(elements)),
    )
    return cs, prim


def generate_sqs_structure(
    cs: ClusterSpace,
    primitive: Atoms,
    concentrations: dict,
    supercell_size,
    num_steps: int = NUM_SQS_STEPS,
    random_seed: int = RANDOM_SEED,
) -> Atoms:
    """
    Wrapper for icet.tools.structure_generation.generate_sqs_from_supercells.

    Parameters
    ----------
    cs : ClusterSpace
        ClusterSpace for this system (B2 or BCC).
    primitive : Atoms
        Primitive structure (B2 or BCC).
    concentrations : dict
        Target concentrations. For single-sublattice systems this is
        a global dict like {'Nb': 0.5, 'Mo': 0.5}. For systems with
        sublattices (B2) this must be per-sublattice, e.g.
        {'A': {'Ru': 1.0},
         'B': {'Hf': 0.875, 'Nb': 0.125}}.
    supercell_size : list[int]
        [nx, ny, nz] supercell multipliers.
    """

    # Build supercell from primitive
    P = np.diag(supercell_size)
    supercell = make_supercell(primitive, P)

    # generate_sqs_from_supercells returns a single Atoms object
    sqs_atoms = generate_sqs_from_supercells(
        cluster_space=cs,
        supercells=[supercell],
        target_concentrations=concentrations,
        n_steps=num_steps,
        random_seed=random_seed,
    )

    return sqs_atoms


def composition_key(phase_tag: str, concs: dict):
    """Create a hashable key (phase, sorted composition) for uniqueness tracking."""
    items = tuple(sorted((el, float(f"{frac:.6f}")) for el, frac in concs.items() if frac > 1e-6))
    return (phase_tag, items)


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_dir(ROOT_DIR)
    run_counter = 1
    print(f"Generating SQS/crystalline structures under '{ROOT_DIR}'")

    # -------------------------------
    # Group 1: Baselines (pure BCC)
    # -------------------------------
    all_elements = sorted(set([RU] + REFRACTORY + B2_PRIMARY + B2_SECONDARY + SUBSTITUTES))
    for el in all_elements:
        path = create_directory("01_Baselines", f"BASE_{run_counter:03d}_{el}")
        atoms = build_bcc_primitive(el, a0=3.1)
        write_poscar(os.path.join(path, "POSCAR"), atoms)
        run_counter += 1

    # -----------------------------------
    # Group 2: Binary Ru–X B2 (crystalline)
    # -----------------------------------
    for partner in B2_PRIMARY + B2_SECONDARY:
        path = create_directory("02_Binary_B2", f"B2_{run_counter:03d}_Ru{partner}")
        atoms = build_b2_primitive([RU], [partner], a0=3.1)
        write_poscar(os.path.join(path, "POSCAR"), atoms)
        run_counter += 1

    # -------------------------------------------------
    # Group 3: Binary BCC matrices (SQS)
    # -------------------------------------------------
    matrix_binary_combos = list(itertools.combinations(REFRACTORY, 2))
    matrix_binary_concs = [0.125, 0.25, 0.5, 0.75, 0.875]

    for (el1, el2) in matrix_binary_combos:
        for c1 in matrix_binary_concs:
            c2 = 1.0 - c1
            concs = {el1: c1, el2: c2}
            key = composition_key("BCC", concs)
            if key in UNIQUE_COMPOSITIONS:
                print(f"[INFO] Skipping duplicate BCC composition {key}")
                continue
            UNIQUE_COMPOSITIONS.add(key)

            label = f"MATRIX_BIN_{run_counter:03d}_{el1}{int(c1*100)}_{el2}{int(c2*100)}"
            path = create_directory("03_Matrix_Binaries", label)

            cs, prim = get_bcc_clusterspace([el1, el2])
            try:
                sqs = generate_sqs_structure(
                    cs,
                    prim,
                    concs,
                    supercell_size=SUPERCELL_SIZE_BCC_BIN,
                )
                write_poscar(os.path.join(path, "POSCAR"), sqs)
            except Exception as e:
                print(f"[WARN] Skipping {label}: {e}")
            run_counter += 1

    # -------------------------------------------------
    # Group 4: Ternary BCC matrices (SQS, equiatomic, 3x3x1 → 18 atoms)
    # -------------------------------------------------
    matrix_ternary_combos = list(itertools.combinations(REFRACTORY, 3))
    matrix_ternary_concs = [(1/3, 1/3, 1/3)]  # exactly commensurate with 18 atoms

    for (el1, el2, el3) in matrix_ternary_combos:
        for (c1, c2, c3) in matrix_ternary_concs:
            concs = {el1: c1, el2: c2, el3: c3}
            key = composition_key("BCC", concs)
            if key in UNIQUE_COMPOSITIONS:
                print(f"[INFO] Skipping duplicate BCC ternary composition {key}")
                continue
            UNIQUE_COMPOSITIONS.add(key)

            label = (
                f"MATRIX_TERN_{run_counter:03d}_"
                f"{el1}{int(c1*100)}_{el2}{int(c2*100)}_{el3}{int(c3*100)}"
            )
            path = create_directory("04_Matrix_Ternaries", label)

            cs, prim = get_bcc_clusterspace([el1, el2, el3])
            try:
                sqs = generate_sqs_structure(
                    cs,
                    prim,
                    concs,
                    supercell_size=SUPERCELL_SIZE_BCC_TERN,
                )
                write_poscar(os.path.join(path, "POSCAR"), sqs)
            except Exception as e:
                print(f"[WARN] Skipping {label}: {e}")
            run_counter += 1

    # -------------------------------------------------
    # Group 5: B2 ternaries (Ru + partner + matrix) – solubility
    # -------------------------------------------------
    solubility_concs_b = [0.125, 0.25]  # matrix fraction on B-sublattice (8 B sites)

    for partner in B2_PRIMARY:
        for matrix_el in REFRACTORY:
            for c_mat in solubility_concs_b:
                c_partner = 1.0 - c_mat
                concs_a = {RU: 1.0}
                concs_b = {partner: c_partner, matrix_el: c_mat}

                # For uniqueness we still track a "global" composition
                global_concs = dict(concs_a)
                for k, v in concs_b.items():
                    global_concs[k] = global_concs.get(k, 0.0) + v

                key = composition_key("B2", global_concs)
                if key in UNIQUE_COMPOSITIONS:
                    print(f"[INFO] Skipping duplicate B2 ternary composition {key}")
                    continue
                UNIQUE_COMPOSITIONS.add(key)

                label = (
                    f"B2_TERN_SOL_{run_counter:03d}_"
                    f"Ru_{partner}{int(c_partner*100)}_{matrix_el}{int(c_mat*100)}"
                )
                path = create_directory("05_B2_Ternaries_Solubility", label)

                site_a_els = list(concs_a.keys())
                site_b_els = list(concs_b.keys())
                cs, prim = get_b2_clusterspace(site_a_els, site_b_els)

                # Proper per-sublattice target concentrations for icet
                target_concs = {"A": concs_a, "B": concs_b}

                try:
                    sqs = generate_sqs_structure(
                        cs,
                        prim,
                        target_concs,
                        supercell_size=SUPERCELL_SIZE_B2,
                    )
                    write_poscar(os.path.join(path, "POSCAR"), sqs)
                except Exception as e:
                    print(f"[WARN] Skipping {label}: {e}")
                run_counter += 1

    # -------------------------------------------------
    # Group 6: B2 Ru→X substitution (penalty study)
    # -------------------------------------------------
    # A-site: Ru + dopant; B-site: single B2 partner
    # 8 A-site atoms → use Ru fractions with denominator 8:
    # (8,0), (7,1), (6,2), (5,3), (4,4)
    penalty_concs_a = [1.0, 0.875, 0.75, 0.625, 0.5]

    for partner in B2_PRIMARY:
        for dop in SUBSTITUTES:
            if dop == RU:
                continue
            for c_ru in penalty_concs_a:
                c_dop = 1.0 - c_ru
                concs_a = {RU: c_ru}
                if c_dop > 1e-6:
                    concs_a[dop] = c_dop
                concs_b = {partner: 1.0}

                # "Global" composition for uniqueness tracking only
                global_concs = dict(concs_a)
                for k, v in concs_b.items():
                    global_concs[k] = global_concs.get(k, 0.0) + v

                key = composition_key("B2", global_concs)
                if key in UNIQUE_COMPOSITIONS:
                    print(f"[INFO] Skipping duplicate B2 penalty composition {key}")
                    continue
                UNIQUE_COMPOSITIONS.add(key)

                label = (
                    f"B2_PENALTY_{run_counter:03d}_"
                    f"Ru{int(c_ru*100)}"
                    f"{dop}{int(c_dop*100) if c_dop>1e-6 else 0}_{partner}"
                )
                path = create_directory("06_B2_Penalty_RuSubstitution", label)

                site_a_els = list(concs_a.keys())
                site_b_els = list(concs_b.keys())

                # If there is no disorder on either sublattice (pure Ru–partner B2),
                # icet's ClusterSpace will raise "No active sites found".
                # In that case just build an ordered 2x2x2 B2 supercell.
                if len(site_a_els) == 1 and len(site_b_els) == 1:
                    prim = build_b2_primitive(site_a_els, site_b_els, a0=3.1)
                    P = np.diag(SUPERCELL_SIZE_B2)
                    supercell = make_supercell(prim, P)
                    write_poscar(os.path.join(path, "POSCAR"), supercell)
                    run_counter += 1
                    continue

                cs, prim = get_b2_clusterspace(site_a_els, site_b_els)
                target_concs = {"A": concs_a, "B": concs_b}

                try:
                    sqs = generate_sqs_structure(
                        cs,
                        prim,
                        target_concs,
                        supercell_size=SUPERCELL_SIZE_B2,
                    )
                    write_poscar(os.path.join(path, "POSCAR"), sqs)
                except Exception as e:
                    print(f"[WARN] Skipping {label}: {e}")
                run_counter += 1

    # -------------------------------------------------
    # Group 7: Higher-order B2 (multi-partner on B-site, 2-element mix)
    # -------------------------------------------------
    # B-site: mix of 2 elements from B2_PRIMARY + B2_SECONDARY, 50/50 (4+4 on B)
    b2_multi_pool = B2_PRIMARY + B2_SECONDARY  # e.g., Hf, Ti, Zr, Al

    for b_elems in itertools.combinations(b2_multi_pool, 2):
        concs_b = {b_elems[0]: 0.5, b_elems[1]: 0.5}

        # Ru fractions commensurate with 8 A sites: 7/8 (12.5% dop), 6/8 (25% dop)
        for c_ru in [0.875, 0.75]:
            c_rest = 1.0 - c_ru
            if c_rest > 1e-6:
                dop = "Al" if "Al" in SUBSTITUTES else "Ti"
                concs_a = {RU: c_ru, dop: c_rest}
            else:
                concs_a = {RU: 1.0}

            global_concs = dict(concs_a)
            for k2, v2 in concs_b.items():
                global_concs[k2] = global_concs.get(k2, 0.0) + v2

            key = composition_key("B2", global_concs)
            if key in UNIQUE_COMPOSITIONS:
                print(f"[INFO] Skipping duplicate B2 high-order composition {key}")
                continue
            UNIQUE_COMPOSITIONS.add(key)

            dop_label = ""
            if len(concs_a) > 1:
                dop_name = [x for x in concs_a.keys() if x != RU][0]
                dop_label = f"_{dop_name}{int(c_rest*100)}"

            label = (
                f"B2_MULTI_{run_counter:03d}_"
                f"A_Ru{int(c_ru*100)}{dop_label}_"
                f"B_{''.join(b_elems)}"
            )
            path = create_directory("07_B2_HighOrder", label)

            site_a_els = list(concs_a.keys())
            site_b_els = list(concs_b.keys())
            cs, prim = get_b2_clusterspace(site_a_els, site_b_els)

            target_concs = {"A": concs_a, "B": concs_b}

            try:
                sqs = generate_sqs_structure(
                    cs,
                    prim,
                    target_concs,
                    supercell_size=SUPERCELL_SIZE_B2,
                )
                write_poscar(os.path.join(path, "POSCAR"), sqs)
            except Exception as e:
                print(f"[WARN] Skipping {label}: {e}")
            run_counter += 1

    print(f"Done. Generated {run_counter - 1} structures in '{ROOT_DIR}'.")


if __name__ == "__main__":
    main()
