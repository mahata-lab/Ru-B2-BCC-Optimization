import os
import random
import itertools
import numpy as np
from ase import Atoms
from ase.io import write
from ase.build import bulk, make_supercell

# ==========================================
# CONFIGURATION
# ==========================================
ROOT_DIR = "DFT_Runs"
SUPERCELL_SIZE = [2, 2, 2]  # 16 atoms for BCC/B2

# Element Lists
REFRACTORY = ['Nb', 'Mo', 'Ta', 'W', 'V']
B2_PRIMARY = ['Hf', 'Ti', 'Zr', 'Al']
SUBSTITUTES = ['Al', 'Cr', 'Cu', 'Si']

# ==========================================
# GENERATION FUNCTIONS
# ==========================================

def create_directory(group, run_id):
    """Creates directory: DFT_Runs/Group/Run_ID"""
    path = os.path.join(ROOT_DIR, group, run_id)
    os.makedirs(path, exist_ok=True)
    return path

def get_atom_counts(target_concs, total_atoms):
    """Converts percentage (0.5) to integer count (8)."""
    counts = {}
    current_total = 0
    
    # Sort by concentration to handle rounding cleanly
    sorted_items = sorted(target_concs.items(), key=lambda x: x[1])
    
    for el, conc in sorted_items[:-1]:
        count = int(round(conc * total_atoms))
        counts[el] = count
        current_total += count
    
    # Force conservation of atoms
    last_el = sorted_items[-1][0]
    counts[last_el] = total_atoms - current_total
    return counts

def generate_random_bcc_sqs(elements, concentrations, supercell_dim):
    """Generates a Disordered BCC Matrix (Randomized)"""
    # 1. Base Structure
    prim = bulk('Mo', 'bcc', a=3.15)
    supercell = make_supercell(prim, np.diag(supercell_dim))
    total_atoms = len(supercell)
    
    # 2. Assign Species
    counts = get_atom_counts(dict(zip(elements, concentrations)), total_atoms)
    symbols = []
    for el, count in counts.items():
        symbols.extend([el] * count)
    
    random.shuffle(symbols)
    supercell.set_chemical_symbols(symbols)
    return supercell

def generate_sublattice_sqs(site_a_dict, site_b_dict, supercell_dim):
    """
    Generates B2 with separate mixing on Site A and Site B.
    Site A (Corner): Indices 0, 2, 4...
    Site B (Center): Indices 1, 3, 5...
    """
    # 1. Base B2 Structure (CsCl type)
    prim = Atoms('AlNi', scaled_positions=[(0,0,0), (0.5,0.5,0.5)], cell=[3.1, 3.1, 3.1], pbc=True)
    # Using simple cubic expansion preserves A-B-A-B ordering
    supercell = make_supercell(prim, np.diag(supercell_dim))
    
    # 2. Identify Sites (Even=A, Odd=B)
    site_a_indices = [i for i in range(len(supercell)) if i % 2 == 0]
    site_b_indices = [i for i in range(len(supercell)) if i % 2 != 0]
    
    # 3. Generate Site A Content
    counts_a = get_atom_counts(site_a_dict, len(site_a_indices))
    symbols_a = []
    for el, count in counts_a.items():
        symbols_a.extend([el] * count)
    random.shuffle(symbols_a)
    
    # 4. Generate Site B Content
    counts_b = get_atom_counts(site_b_dict, len(site_b_indices))
    symbols_b = []
    for el, count in counts_b.items():
        symbols_b.extend([el] * count)
    random.shuffle(symbols_b)
    
    # 5. Apply to Supercell
    new_symbols = supercell.get_chemical_symbols()
    for i, idx in enumerate(site_a_indices):
        new_symbols[idx] = symbols_a[i]
    for i, idx in enumerate(site_b_indices):
        new_symbols[idx] = symbols_b[i]
        
    supercell.set_chemical_symbols(new_symbols)
    return supercell

# ==========================================
# MAIN EXECUTION
# ==========================================
print(f"Generating POSCARs in {ROOT_DIR} using Pure ASE...")
run_counter = 1

# --- 1. BASELINES (Pure Elements) ---
all_els = sorted(list(set(['Ru'] + REFRACTORY + B2_PRIMARY + SUBSTITUTES)))
for el in all_els:
    path = create_directory("01_Baselines", f"BASE_{run_counter:03d}_{el}")
    atoms = bulk(el, 'bcc', a=3.1)
    write(os.path.join(path, "POSCAR"), atoms)
    run_counter += 1

# --- 2. BINARY B2s (Pure) ---
for partner in B2_PRIMARY:
    path = create_directory("02_Binaries", f"BIN_{run_counter:03d}_Ru{partner}")
    atoms = Atoms(f'Ru{partner}', scaled_positions=[(0,0,0), (0.5,0.5,0.5)], cell=[3.1, 3.1, 3.1], pbc=True)
    write(os.path.join(path, "POSCAR"), atoms)
    run_counter += 1

# --- 3. KUBE GRID (Solubility) ---
# Mixing on Refractory Sublattice (Site B)
for partner in B2_PRIMARY:
    for matrix_el in REFRACTORY:
        for solubility, label in [(0.125, "12.5"), (0.25, "25.0")]:
            path = create_directory("03_Kube_Solubility", f"KUBE_{run_counter:03d}_Ru{partner}-{matrix_el}{label}")
            
            # Site A: Ru 100%
            # Site B: Partner (1-x), Matrix (x)
            site_a = {'Ru': 1.0}
            site_b = {partner: 1.0 - solubility, matrix_el: solubility}
            
            atoms = generate_sublattice_sqs(site_a, site_b, SUPERCELL_SIZE)
            write(os.path.join(path, "POSCAR"), atoms)
            run_counter += 1

# --- 4. PENALTY STUDY (Reviewer Request) ---
# Mixing on Ru Sublattice (Site A)
for partner in ['Hf', 'Ti', 'Zr']: 
    for dopant in SUBSTITUTES:
        for conc, label in [(0.125, "12.5"), (0.25, "25.0"), (0.375, "37.5")]:
            path = create_directory("04_Penalty_Study", f"PEN_{run_counter:03d}_Ru{partner}-{dopant}{label}")
            
            # Site A: Ru (1-x), Dopant (x)
            # Site B: Partner 100%
            site_a = {'Ru': 1.0 - conc, dopant: conc}
            site_b = {partner: 1.0}
            
            atoms = generate_sublattice_sqs(site_a, site_b, SUPERCELL_SIZE)
            write(os.path.join(path, "POSCAR"), atoms)
            run_counter += 1

# --- 5. MATRIX (Disordered BCC) ---
matrix_pairs = list(itertools.combinations(REFRACTORY, 2))
for m1, m2 in matrix_pairs:
    for conc_m1 in [0.125, 0.25, 0.5, 0.75, 0.875]:
        conc_m2 = 1.0 - conc_m1
        label = f"{int(conc_m1*100)}-{int(conc_m2*100)}"
        path = create_directory("05_Matrix", f"MAT_{run_counter:03d}_{m1}{m2}_{label}")
        
        atoms = generate_random_bcc_sqs([m1, m2], [conc_m1, conc_m2], SUPERCELL_SIZE)
        write(os.path.join(path, "POSCAR"), atoms)
        run_counter += 1

# --- 6. FREY VALIDATION (Specific Complex Alloys) ---
validation_set = [
    ("RuHf_TiMix", {'Ru':1.0}, {'Hf':0.5, 'Ti':0.5}),
    ("RuHf_ZrMix", {'Ru':1.0}, {'Hf':0.5, 'Zr':0.5}),
    ("RuAl_HfMix", {'Ru':0.5, 'Al':0.5}, {'Hf':1.0}),
    ("Complex_1", {'Ru':0.9, 'Al':0.1}, {'Hf':0.9, 'Nb':0.1}),
]

for name, site_a, site_b in validation_set:
    path = create_directory("06_Validation", f"VAL_{run_counter:03d}_{name}")
    atoms = generate_sublattice_sqs(site_a, site_b, SUPERCELL_SIZE)
    write(os.path.join(path, "POSCAR"), atoms)
    run_counter += 1

print(f"Done! Generated {run_counter-1} structures in '{ROOT_DIR}'.")