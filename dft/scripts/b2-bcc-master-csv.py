#!/usr/bin/env python3
"""
b2-bcc-master-csv_fixed.py
Robust builder for B2-BCC ML CSV. See inline comments.
"""
import pandas as pd
import numpy as np
import re
import sys
from collections import OrderedDict

B2_CSV = "B2-Data.csv"
BCC_CSV = "BCC_parent_lattice_parameters.csv"
OUT_CSV = "B2_BCC_cartesian_ML_ready.csv"

GEOMETRY_TOKENS = ['alpha', 'beta', 'gamma', 'volume', 'vec', 'vector', 'a_from', 'a_from_volume', 'a_cell', 'lattice_vectors', 'cartesian', 'direct']
ELEMENT_TOKENS = ['Ru','Hf','Ti','Zr','Al','Mo','Nb','Ta','W','V','Cr','Cu','Si','Fe','Co','Ni']

# ---- Helpers ----
def load_csv(path):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise SystemExit(f"ERROR: file not found: {path}")
    df.columns = [str(c).strip() for c in df.columns]
    return df

def find_lattice_col(df, prefer=None):
    if prefer is None: prefer = []
    candidates = prefer + ['lattice_parameter','lattice_parameter_A','a_b2_A','a_bcc_A','a_from_volume_A','a_A','a_cell','a']
    for c in candidates:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            return c
    # fuzzy
    for c in df.columns:
        lc = c.lower()
        if 'lattice' in lc and ('param' in lc or 'a' in lc):
            return c
        if c.lower().startswith('a_') and pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None

def detect_fraction_columns(df):
    mapping = OrderedDict()
    for c in df.columns:
        cn = c.strip()
        # X_Ru or X-Ru
        m = re.match(r'^[Xx][ _-]?([A-Za-z]{1,3})$', cn)
        if m:
            mapping[c] = m.group(1).title()
            continue
        # Ru_frac
        m = re.match(r'^([A-Za-z]{1,3})_(?:frac|fraction|fr)$', cn, flags=re.I)
        if m:
            mapping[c] = m.group(1).title()
            continue
        # single element name 'Ru'
        if re.fullmatch(r'[A-Z][a-z]?$', cn):
            mapping[c] = cn
            continue
        # fuzzy contains known element tokens
        found = re.findall(r'(Ru|Hf|Ti|Zr|Al|Mo|Nb|Ta|W|V|Cr|Cu|Si|Fe|Co|Ni)', cn, flags=re.I)
        if found:
            mapping[c] = found[0].title()
            continue
    return mapping

def standardize_fraction_columns(df, mapping, prefix):
    ren = {}
    newcols = []
    for orig, el in mapping.items():
        el_clean = re.sub(r'\s+', '', el)
        newname = f"{prefix}X_{el_clean}"
        ren[orig] = newname
        newcols.append(newname)
    df = df.rename(columns=ren)
    return df, newcols

def safe_to_numeric_series(s, colname=None):
    # If s is DataFrame (multiple columns), try to reduce or raise a helpful error
    if isinstance(s, pd.DataFrame):
        if s.shape[1] == 1:
            s = s.iloc[:,0]
        else:
            raise TypeError(f"Expected a single Series for column conversion but got DataFrame with columns {list(s.columns)} (colname hint: {colname})")
    # Now s should be a Series
    # Convert to str, remove commas, keep digits, signs, decimal, exponent
    try:
        s2 = s.astype(str).str.replace(',', '', regex=False)
        s3 = s2.str.replace(r'[^0-9eE+\-\.]', '', regex=True)
        return pd.to_numeric(s3, errors='coerce')
    except Exception as e:
        # Last-resort: coerce with pandas directly
        try:
            return pd.to_numeric(s, errors='coerce')
        except Exception as e2:
            raise TypeError(f"Could not coerce column {colname} to numeric: {e}; secondary: {e2}")

def pick_formation_energy_col(df):
    priority = ['formation_energy_per_atom_eV','formation_energy_eV_per_atom','formation_energy_eV','formation_energy_per_atom','formation_energy']
    for p in priority:
        if p in df.columns and pd.api.types.is_numeric_dtype(df[p]):
            return p
    for c in df.columns:
        lc = c.lower()
        if 'formation' in lc and 'energy' in lc and pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None

# ---- Load ----
b2 = load_csv(B2_CSV)
bcc = load_csv(BCC_CSV)
print(f"Loaded: B2 rows={len(b2)}, BCC rows={len(bcc)}")

# ---- Lattice columns ----
b2_lat = find_lattice_col(b2, prefer=['lattice_parameter','a_b2_A'])
bcc_lat = find_lattice_col(bcc, prefer=['a_from_volume_A','a_bcc_A'])
if b2_lat is None or bcc_lat is None:
    raise SystemExit(f"ERROR: missing lattice columns. b2_lat={b2_lat}, bcc_lat={bcc_lat}")
b2 = b2.rename(columns={b2_lat:'a_b2_A'})
bcc = bcc.rename(columns={bcc_lat:'a_bcc_A'})
b2['a_b2_A'] = pd.to_numeric(b2['a_b2_A'], errors='coerce')
bcc['a_bcc_A'] = pd.to_numeric(bcc['a_bcc_A'], errors='coerce')

# ---- Fraction detection and standardization ----
b2_frac_map = detect_fraction_columns(b2)
bcc_frac_map = detect_fraction_columns(bcc)

b2, b2_frac_cols = standardize_fraction_columns(b2, b2_frac_map, 'B2_')
bcc, bcc_frac_cols = standardize_fraction_columns(bcc, bcc_frac_map, 'BCC_')

# Validate that new names are strings and unique
b2_frac_cols = [str(c) for c in b2_frac_cols]
bcc_frac_cols = [str(c) for c in bcc_frac_cols]

# Convert each fraction col individually with robust checks
for c in b2_frac_cols:
    if c not in b2.columns:
        print(f"WARNING: expected B2 fraction column {c} missing -> creating zero column.")
        b2[c] = 0.0
        continue
    # ensure we pass a Series, not DataFrame
    series_obj = b2[c]
    if isinstance(series_obj, pd.DataFrame):
        # should not happen since we renamed unique columns, but guard
        if series_obj.shape[1] == 1:
            series_obj = series_obj.iloc[:,0]
        else:
            raise SystemExit(f"ERROR: after renaming B2 fraction column {c} resolved to multiple columns: {list(series_obj.columns)}")
    b2[c] = safe_to_numeric_series(series_obj, colname=c).fillna(0.0)

for c in bcc_frac_cols:
    if c not in bcc.columns:
        print(f"WARNING: expected BCC fraction column {c} missing -> creating zero column.")
        bcc[c] = 0.0
        continue
    series_obj = bcc[c]
    if isinstance(series_obj, pd.DataFrame):
        if series_obj.shape[1] == 1:
            series_obj = series_obj.iloc[:,0]
        else:
            raise SystemExit(f"ERROR: after renaming BCC fraction column {c} resolved to multiple columns: {list(series_obj.columns)}")
    bcc[c] = safe_to_numeric_series(series_obj, colname=c).fillna(0.0)

# ---- Build union of elements ----
elements = []
for c in b2_frac_cols + bcc_frac_cols:
    m = re.match(r'.*X_([A-Za-z0-9]+)$', c)
    if m:
        el = m.group(1)
        if el not in elements:
            elements.append(el)
print("Detected element union:", elements)

# Ensure all B2_X_<el> and BCC_X_<el> exist
for el in elements:
    bcol = f"B2_X_{el}"
    ccol = f"BCC_X_{el}"
    if bcol not in b2.columns:
        b2[bcol] = 0.0
    if ccol not in bcc.columns:
        bcc[ccol] = 0.0

# ---- Select B2 numeric ROM properties (exclude fractions, lattice, geometry) ----
exclude = set([f"B2_X_{el}" for el in elements] + ['a_b2_A'])
b2_numeric = []
for c in b2.columns:
    if c in exclude:
        continue
    lc = c.lower()
    if any(tok in lc for tok in GEOMETRY_TOKENS):
        continue
    if pd.api.types.is_numeric_dtype(b2[c]):
        b2_numeric.append(c)

# Prefix numeric properties with B2_ if not already
b2_rename_map = {}
for col in b2_numeric:
    if not col.startswith('B2_'):
        b2_rename_map[col] = 'B2_' + col
b2 = b2.rename(columns=b2_rename_map)

# ---- Ensure ID columns ----
def ensure_id(df, prefer_list, default_prefix):
    for p in prefer_list:
        if p in df.columns:
            return p
    new = default_prefix + '_id'
    if new not in df.columns:
        df.insert(0, new, [f"{default_prefix}_{i+1}" for i in range(len(df))])
    return new

b2_id = ensure_id(b2, ['b2_id','alloy_name','run_folder','name','id'], 'B2')
bcc_id = ensure_id(bcc, ['bcc_id','alloy_name','run_folder','name','id','parent'], 'BCC')

# ---- Cartesian product ----
b2['_tmp'] = 1
bcc['_tmp'] = 1
merged = pd.merge(b2, bcc, on='_tmp', suffixes=('_B2','_BCC')).drop(columns=['_tmp'])
print("Merged rows:", len(merged))

# ---- Symmetric misfit ----
merged['a_b2_A'] = pd.to_numeric(merged['a_b2_A'], errors='coerce')
merged['a_bcc_A'] = pd.to_numeric(merged['a_bcc_A'], errors='coerce')
merged['misfit_delta'] = 2.0 * (merged['a_b2_A'] - merged['a_bcc_A']) / (merged['a_b2_A'] + merged['a_bcc_A'])

# ---- Driving force (formation energies) ----
b2_form = pick_formation_energy_col(b2)
bcc_form = pick_formation_energy_col(bcc)

def find_merged_col(merged_df, base):
    if base is None:
        return None
    if base in merged_df.columns:
        return base
    # check prefixed with B2_
    if ('B2_' + base) in merged_df.columns:
        return 'B2_' + base
    for c in merged_df.columns:
        if base.lower() in c.lower() and pd.api.types.is_numeric_dtype(merged_df[c]):
            return c
    return None

b2f = find_merged_col(merged, b2_form)
bccf = find_merged_col(merged, bcc_form)

if b2f and bccf:
    merged['driving_force_eV_per_atom'] = pd.to_numeric(merged[b2f], errors='coerce') - pd.to_numeric(merged[bccf], errors='coerce')
else:
    merged['driving_force_eV_per_atom'] = np.nan
    if not b2f:
        print("WARNING: no B2 formation-energy column detected for driving force computation.")
    if not bccf:
        print("WARNING: no BCC formation-energy column detected for driving force computation.")

# ---- Build final column list ----
final = []
final.append(b2_id)
final.append(bcc_id)
final += ['a_b2_A','a_bcc_A','misfit_delta']

# add B2 numeric props (prefixed)
for c in sorted([c for c in merged.columns if c.startswith('B2_') and pd.api.types.is_numeric_dtype(merged[c])]):
    if c not in final:
        final.append(c)

final.append('driving_force_eV_per_atom')

# add fraction cols in canonical element order
for el in elements:
    final.append(f"B2_X_{el}")
for el in elements:
    final.append(f"BCC_X_{el}")

# optional natoms etc
for opt in ['natoms','n_formula_units','multiplicity']:
    if opt in merged.columns and opt not in final:
        final.append(opt)

# filter final to present
final = [c for c in final if c in merged.columns]

# ---- write out ----
merged[final].to_csv(OUT_CSV, index=False)
print(f"WROTE {OUT_CSV}, rows={len(merged)}, cols={len(final)}")

# ---- quick checks ----
ele_sample = elements[:20]
print("Elements:", elements)
print("Expected merged rows (B2*BCC):", len(b2)*len(bcc))
print("Sample B2 fraction sums (first 5):", (merged[[f"B2_X_{el}" for el in elements]].head(5).sum(axis=1)).tolist())
print("Sample BCC fraction sums (first 5):", (merged[[f"BCC_X_{el}" for el in elements]].head(5).sum(axis=1)).tolist())
print("misfit stats:\n", merged['misfit_delta'].describe())
