#!/usr/bin/env python3
"""
build_B2_BCC_cartesian_ML_ready_v2.py

Produces: B2_BCC_cartesian_ML_ready.csv

Key behavior (explicit):
 - Standardize fraction columns to B2_X_<El> and BCC_X_<El> across the merged table.
 - Copy all numeric B2 property columns (ROM-style) except geometry/angles and fraction columns,
   prefixing them with B2_ in the merged output.
 - Ensure BCC fraction columns exist for the same element set (fill zeros where missing).
 - Compute symmetric misfit: delta = 2*(a_b2 - a_bcc) / (a_b2 + a_bcc)
 - Compute driving_force_eV_per_atom when both formation energies exist.
"""

import pandas as pd
import numpy as np
import re
import sys

B2_CSV = "B2-Data.csv"
BCC_CSV = "BCC_parent_lattice_parameters.csv"
OUT_CSV = "B2_BCC_cartesian_ML_ready.csv"

# ---------- helper functions ----------
def load_csv(name):
    try:
        df = pd.read_csv(name)
    except FileNotFoundError:
        raise SystemExit(f"ERROR: {name} not found in current directory.")
    df.columns = [c.strip() for c in df.columns]
    return df

def find_lattice_col(df, prefer=None):
    """Find best lattice column name (returns column or None)."""
    if prefer is None: prefer = []
    candidates = prefer + [
        'lattice_parameter','a_b2_A','a_bcc_A','a_from_volume_A',
        'lattice_parameter_A','a_A','a_cell','a'
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # fuzzy search
    for c in df.columns:
        lc = c.lower()
        if 'lattice' in lc and ('param' in lc or 'a' in lc):
            return c
    for c in df.columns:
        if c.lower().startswith('a_') and pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None

def detect_fraction_columns(df):
    """Return dict original_col -> element_symbol for columns that look like element fractions.
    Recognize patterns: X_Ru, Ru_frac, Ru, Ru_fraction.
    """
    mapping = {}
    # element regex: capital letter + optional lowercase (Ru, Hf, Ti, ...)
    el_re = re.compile(r'^(X[_\-\s])?([A-Z][a-z]?)($|_|-| )', re.I)
    for c in df.columns:
        cn = c.strip()
        # pattern X_Ru or X-Ru
        if cn.startswith('X_') or cn.startswith('X-') or cn.startswith('X '):
            el = cn.split('_',1)[1] if '_' in cn else cn[2:]
            mapping[c] = el.strip()
            continue
        # pattern Ru_frac, Ru_fraction
        if cn.lower().endswith('_frac') or cn.lower().endswith('_fraction'):
            el = cn.rsplit('_',1)[0]
            mapping[c] = el.strip()
            continue
        # column named exactly 'Ru' or 'Nb' etc (single element)
        if re.fullmatch(r'[A-Z][a-z]?$', cn):
            mapping[c] = cn
            continue
        # fuzzy: contains known element tokens
        found = re.findall(r'(Ru|Hf|Ti|Zr|Al|Mo|Nb|Ta|W|V|Cr|Cu|Si|Fe|Co|Ni)', cn, flags=re.I)
        if found:
            mapping[c] = found[0].title()
    return mapping  # original_col -> element symbol

def standardize_frac_columns(df, mapping, prefix):
    """Rename fraction columns to prefix + 'X_<El>' and return new df and list of new columns."""
    ren = {}
    newcols = []
    for orig, el in mapping.items():
        el_clean = el.strip().replace(' ', '')
        new = f"{prefix}X_{el_clean}"
        ren[orig] = new
        newcols.append(new)
    df = df.rename(columns=ren)
    return df, newcols

def pick_formation_energy_col(df):
    """Return the most plausible formation-energy column name or None."""
    priority = [
        'formation_energy_per_atom_eV','formation_energy_eV_per_atom','formation_energy_eV',
        'formation_energy_per_atom','formation_energy'
    ]
    for c in priority:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            return c
    # fuzzy search for 'formation' + 'energy'
    for c in df.columns:
        lc = c.lower()
        if 'formation' in lc and 'energy' in lc and pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None

# ---------- load inputs ----------
b2 = load_csv(B2_CSV)
bcc = load_csv(BCC_CSV)

# ---------- find lattice columns and standardize names ----------
b2_lat = find_lattice_col(b2, prefer=['lattice_parameter','a_b2_A'])
bcc_lat = find_lattice_col(bcc, prefer=['a_from_volume_A','a_bcc_A'])

if b2_lat is None:
    raise SystemExit("ERROR: cannot detect B2 lattice parameter column in B2-Data.csv (look for 'lattice_parameter' or similar).")
if bcc_lat is None:
    raise SystemExit("ERROR: cannot detect BCC lattice parameter column in BCC_parent_lattice_parameters.csv (look for 'a_from_volume_A' or similar).")

b2 = b2.rename(columns={b2_lat: 'a_b2_A'})
bcc = bcc.rename(columns={bcc_lat: 'a_bcc_A'})

# ---------- detect and standardize fraction columns ----------
b2_frac_map = detect_fraction_columns(b2)
bcc_frac_map = detect_fraction_columns(bcc)

b2, b2_frac_cols = standardize_frac_columns(b2, b2_frac_map, 'B2_')
bcc, bcc_frac_cols = standardize_frac_columns(bcc, bcc_frac_map, 'BCC_')

# convert fraction columns to numeric, fill NaN->0
for c in b2_frac_cols:
    b2[c] = pd.to_numeric(b2[c], errors='coerce').fillna(0.0)
for c in bcc_frac_cols:
    bcc[c] = pd.to_numeric(bcc[c], errors='coerce').fillna(0.0)

# ---------- build union of elements and ensure both B2 and BCC have the same set of fraction columns ----------
elements = set()
# extract element symbols from B2_X_* and BCC_X_* names
for c in b2_frac_cols:
    m = re.match(r'B2_X_([A-Za-z0-9]+)', c)
    if m:
        elements.add(m.group(1))
for c in bcc_frac_cols:
    m = re.match(r'BCC_X_([A-Za-z0-9]+)', c)
    if m:
        elements.add(m.group(1))

elements = sorted(list(elements))

# ensure all B2_X_<El> present in b2 (if missing, add zero col)
for el in elements:
    col = f"B2_X_{el}"
    if col not in b2.columns:
        b2[col] = 0.0
# ensure all BCC_X_<El> present in bcc (if missing, add zero col)
for el in elements:
    col = f"BCC_X_{el}"
    if col not in bcc.columns:
        bcc[col] = 0.0

# refresh lists
b2_frac_cols = [f"B2_X_{el}" for el in elements]
bcc_frac_cols = [f"BCC_X_{el}" for el in elements]

# ---------- pick B2 ROM / numeric properties to copy ----------
# Exclude geometry / angles / raw lattice vectors & fraction cols & ID-like strings
geometry_patterns = ['alpha','beta','gamma','volume','a_A','b_A','c_A','cell','vec_x','vec_y','vec_z','lattice_vectors','lattice_vector']
exclude_cols = set(b2_frac_cols) | set(['a_b2_A'])
keep_b2_props = []
for c in b2.columns:
    if c in exclude_cols: 
        continue
    lc = c.lower()
    if any(g in lc for g in geometry_patterns):
        continue
    # skip obvious non-numeric metadata
    if pd.api.types.is_numeric_dtype(b2[c]) and len(c) > 0:
        keep_b2_props.append(c)

# prefix these columns with 'B2_' in the merged file to avoid collisions
b2_prop_renames = {}
for col in keep_b2_props:
    # avoid double-prefix if already looks prefixed
    if not col.startswith('B2_'):
        b2_prop_renames[col] = 'B2_' + col
    else:
        b2_prop_renames[col] = col

b2 = b2.rename(columns=b2_prop_renames)

# ---------- ensure BCC id columns exist (for identification) ----------
bcc_id_cols = [c for c in bcc.columns if c.lower() in ('bcc_id','id','run_folder','name','parent','alloy_name')]
if not bcc_id_cols:
    # if nothing, create a simple id from index
    bcc.insert(0, 'bcc_id', ['BCC_'+str(i+1) for i in range(len(bcc))])
    bcc_id_cols = ['bcc_id']

b2_id_cols = [c for c in b2.columns if c.lower() in ('b2_id','id','run_folder','alloy_name','name')]
if not b2_id_cols:
    b2.insert(0, 'b2_id', ['B2_'+str(i+1) for i in range(len(b2))])
    b2_id_cols = ['b2_id']

# ---------- Cartesian product ----------
b2['_tmp'] = 1
bcc['_tmp'] = 1
merged = pd.merge(b2, bcc, on='_tmp', suffixes=('_B2','_BCC')).drop(columns=['_tmp'])

# ---------- compute symmetric misfit (explicit) ----------
merged['a_b2_A'] = pd.to_numeric(merged['a_b2_A'], errors='coerce')
merged['a_bcc_A'] = pd.to_numeric(merged['a_bcc_A'], errors='coerce')
merged['misfit_delta'] = 2.0 * (merged['a_b2_A'] - merged['a_bcc_A']) / (merged['a_b2_A'] + merged['a_bcc_A'])

# ---------- compute driving force if formation energies present ----------
b2_form_col = pick_formation_energy_col(b2)
bcc_form_col = pick_formation_energy_col(bcc)

# after merge, columns may have suffixes; attempt to find merged names
def find_col_in_merged(base, df_merged):
    if base is None: 
        return None
    # try exact base (if prefixing earlier changed names, handle)
    if base in df_merged.columns:
        return base
    # look for columns that contain the base token
    for c in df_merged.columns:
        if base.lower() in c.lower() and pd.api.types.is_numeric_dtype(df_merged[c]):
            return c
    return None

b2_form_in_merged = find_col_in_merged(b2_form_col, merged)
bcc_form_in_merged = find_col_in_merged(bcc_form_col, merged)

if b2_form_in_merged and bcc_form_in_merged:
    merged['driving_force_eV_per_atom'] = pd.to_numeric(merged[b2_form_in_merged], errors='coerce') - pd.to_numeric(merged[bcc_form_in_merged], errors='coerce')
else:
    merged['driving_force_eV_per_atom'] = np.nan

# ---------- finalize column ordering ----------
final_cols = []

# IDs: B2 id then BCC id
final_cols.append(b2_id_cols[0])
final_cols.append(bcc_id_cols[0])

# lattice/misfit
final_cols += ['a_b2_A', 'a_bcc_A', 'misfit_delta']

# add B2 numeric properties (prefixed)
b2_prefixed_props = [c for c in merged.columns if c.startswith('B2_') and c not in b2_frac_cols]
# keep formation energy if prefixed present or merged
if b2_form_in_merged and b2_form_in_merged not in final_cols:
    final_cols.append(b2_form_in_merged)
# append B2 prefixed props (unique)
for c in b2_prefixed_props:
    if c not in final_cols:
        final_cols.append(c)

# add driving force
final_cols.append('driving_force_eV_per_atom')

# add VEC if present (it may be in prefixed props)
vec_cols = [c for c in merged.columns if c.lower().endswith('vec') or c.lower()=='vec']
for c in vec_cols:
    if c not in final_cols:
        final_cols.append(c)

# add B2 fraction columns (in canonical order)
for el in elements:
    col = f"B2_X_{el}"
    final_cols.append(col)
# add BCC fraction columns
for el in elements:
    col = f"BCC_X_{el}"
    final_cols.append(col)

# optionally keep natoms/n_formula_units/multiplicity if present
for c in ['natoms','n_formula_units','multiplicity']:
    if c in merged.columns and c not in final_cols:
        final_cols.append(c)

# ensure final_cols exist in merged, filter
final_cols = [c for c in final_cols if c in merged.columns]

# ---------- write out ----------
merged[final_cols].to_csv(OUT_CSV, index=False)
print(f"WROTE {OUT_CSV} with {len(merged)} rows and {len(final_cols)} columns.")
print("Columns written (head):", final_cols[:30])
