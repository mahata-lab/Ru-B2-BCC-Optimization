import pandas as pd
import re

# ==========================================
# 1. LOAD AND CLEAN B2 DATA (CREATE MASTER B2)
# ==========================================
b2_data = pd.read_csv('B2-Data.csv')
b2_rom = pd.read_csv('B2_runs_summary_with_rom.csv')

# Merge on 'run_folder' to get all properties for the same alloy
# suffixes handles duplicate column names
b2_master = pd.merge(b2_data, b2_rom, on='run_folder', suffixes=('', '_rom'))

# Define columns to keep and their new clean names for ML
b2_cols_map = {
    'alloy_name': 'B2_Name',
    'lattice_parameter': 'a_B2',
    'formation_energy_per_atom_eV': 'B2_Formation_Energy',
    'VEC': 'B2_VEC',
    'delta_r': 'B2_delta_r',
    'chi_avg': 'B2_chi_avg',
    'delta_chi': 'B2_delta_chi',
    'Hmix_kJmol': 'B2_Hmix',
    # Rename compositions to B2_Element to distinguish from BCC matrix later
    'X_Ru': 'B2_Ru', 'X_Hf': 'B2_Hf', 'X_Ti': 'B2_Ti', 'X_Zr': 'B2_Zr',
    'X_Al': 'B2_Al', 'X_Mo': 'B2_Mo', 'X_Nb': 'B2_Nb', 'X_Ta': 'B2_Ta',
    'X_W': 'B2_W',   'X_V': 'B2_V',   'X_Cr': 'B2_Cr', 'X_Cu': 'B2_Cu', 'X_Si': 'B2_Si'
}

# Select only available columns and rename
existing_b2_cols = [c for c in b2_cols_map.keys() if c in b2_master.columns]
b2_clean = b2_master[existing_b2_cols].rename(columns=b2_cols_map)

# Fill NaN compositions with 0
b2_comp_cols = [c for c in b2_clean.columns if c.startswith('B2_') and c != 'B2_VEC']
b2_clean[b2_comp_cols] = b2_clean[b2_comp_cols].fillna(0)

# Save the intermediate Master B2 CSV if needed
b2_clean.to_csv('Master_B2_Cleaned.csv', index=False)

# ==========================================
# 2. PROCESS BCC DATA
# ==========================================
bcc_df = pd.read_csv('BCC_parent_lattice_parameters.csv')

def parse_bcc_composition(folder_name):
    # Parses strings like "Nb75-Mo25" into {'BCC_Nb': 0.75, 'BCC_Mo': 0.25}
    elements = re.findall(r'([A-Z][a-z]*)(\d+)', str(folder_name))
    comp = {}
    for el, fraction in elements:
        comp[f"BCC_{el}"] = float(fraction) / 100.0
    return pd.Series(comp)

# Extract compositions
bcc_comps = bcc_df['folder'].apply(parse_bcc_composition).fillna(0)
# Combine with lattice parameter
bcc_clean = pd.concat([bcc_df[['folder', 'a_from_volume_A']], bcc_comps], axis=1)
bcc_clean = bcc_clean.rename(columns={'folder': 'BCC_Name', 'a_from_volume_A': 'a_BCC'})

# ==========================================
# 3. COMBINE & CALCULATE MISFIT
# ==========================================
# Create Cartesian Product (Every B2 paired with Every BCC)
b2_clean['key'] = 1
bcc_clean['key'] = 1
final_df = pd.merge(bcc_clean, b2_clean, on='key').drop('key', axis=1)

# Calculate Misfit Equation: 2 * (a_B2 - a_BCC) / (a_B2 + a_BCC)
final_df['Misfit'] = 2 * (final_df['a_B2'] - final_df['a_BCC']) / (final_df['a_B2'] + final_df['a_BCC'])

# ==========================================
# 4. FINALIZE AND SAVE
# ==========================================
# Reorder columns: Identifiers -> Target (Misfit) -> Features
id_cols = ['BCC_Name', 'B2_Name']
target_col = ['Misfit']
lattice_cols = ['a_BCC', 'a_B2']
b2_props = ['B2_Formation_Energy', 'B2_VEC', 'B2_delta_r', 'B2_chi_avg', 'B2_delta_chi', 'B2_Hmix']
# Get all composition columns dynamically
comp_cols = [c for c in final_df.columns if c.startswith('BCC_') or (c.startswith('B2_') and c not in b2_props)]

# Assemble final column order
final_cols = id_cols + target_col + lattice_cols + b2_props + comp_cols
final_df = final_df[final_cols].fillna(0)

# WRITE THE FILE
final_df.to_csv('Final_BCC_B2_Misfit_Data.csv', index=False)

print("Files 'Master_B2_Cleaned.csv' and 'Final_BCC_B2_Misfit_Data.csv' have been generated.")
print(f"Final Dataset Shape: {final_df.shape}")
print("-" * 30)
print("First 5 rows of the Final CSV:")
print(final_df.head().to_markdown(index=False))