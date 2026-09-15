import pandas as pd
import re

# Load the datasets
bcc_df = pd.read_csv('BCC_parent_lattice_parameters.csv')
b2_df = pd.read_csv('B2-Data.csv')

# --- Process BCC Data ---
# We need to parse the 'folder' column to get compositions (e.g., "Nb75-Mo25")
def parse_bcc_composition(folder_name):
    # Extracts pairs like ('Nb', '75'), ('Mo', '25')
    elements = re.findall(r'([A-Z][a-z]*)(\d+)', folder_name)
    comp = {}
    for el, fraction in elements:
        comp[f"BCC_{el}"] = float(fraction) / 100.0
    return pd.Series(comp)

# Extract compositions and fill NaNs with 0
bcc_comps = bcc_df['folder'].apply(parse_bcc_composition).fillna(0)
# Combine with the lattice parameter
bcc_clean = pd.concat([bcc_df[['folder', 'a_from_volume_A']], bcc_comps], axis=1)
bcc_clean = bcc_clean.rename(columns={'folder': 'BCC_Name', 'a_from_volume_A': 'a_BCC'})

# --- Process B2 Data ---
# Select relevant columns for ML
# Dropping alpha, beta, gamma, volume, lattice vectors as requested to keep it clean
b2_cols_to_keep = [
    'alloy_name', 'lattice_parameter', 'formation_energy_per_atom_eV', 
    'VEC', 'X_Ru', 'X_Hf', 'X_Ti', 'X_Zr', 'X_Al', 'X_Mo', 'X_Nb', 
    'X_Ta', 'X_W', 'X_V', 'X_Cr', 'X_Cu', 'X_Si'
]

# Check if columns exist before selecting (handle potential missing columns gracefully)
existing_cols = [c for c in b2_cols_to_keep if c in b2_df.columns]
b2_clean = b2_df[existing_cols].copy()

# Rename columns to distinguish B2 features from BCC features
rename_dict = {
    'alloy_name': 'B2_Name',
    'lattice_parameter': 'a_B2',
    'formation_energy_per_atom_eV': 'B2_Formation_Energy_eV',
    'VEC': 'B2_VEC'
}
# Rename composition columns (X_Ru -> B2_Ru)
for col in b2_clean.columns:
    if col.startswith('X_'):
        rename_dict[col] = 'B2_' + col.split('_')[1]

b2_clean = b2_clean.rename(columns=rename_dict)

# --- Create Cartesian Product (160 * 18 rows) ---
# Add a temporary key for cross join
bcc_clean['key'] = 1
b2_clean['key'] = 1
merged_df = pd.merge(bcc_clean, b2_clean, on='key').drop('key', axis=1)

# --- Calculate Misfit ---
# Formula: delta = 2 * (a_B2 - a_BCC) / (a_B2 + a_BCC)
merged_df['Misfit'] = 2 * (merged_df['a_B2'] - merged_df['a_BCC']) / (merged_df['a_B2'] + merged_df['a_BCC'])

# --- Final Cleanup ---
# Reorder columns for readability: Names -> Misfit -> Lattice Params -> B2 Props -> Compositions
cols = list(merged_df.columns)
# Move key columns to front
front_cols = ['BCC_Name', 'B2_Name', 'Misfit', 'a_BCC', 'a_B2', 'B2_Formation_Energy_eV', 'B2_VEC']
# Append the rest (compositions)
composition_cols = [c for c in cols if c not in front_cols]
final_df = merged_df[front_cols + composition_cols]

# Fill any remaining NaNs (e.g., if a BCC lacked 'BCC_Al' column but B2 had 'B2_Al') with 0
final_df = final_df.fillna(0)

# Display first 5 rows
print(final_df.head().to_markdown(index=False, numalign="left", stralign="left"))