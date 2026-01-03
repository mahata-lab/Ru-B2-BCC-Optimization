#!/usr/bin/env python3
"""
solvus_sr_pipeline_v10.py

GOAL: "Guided Freedom" Search for Solvus Physics.
METHOD: Allow non-linear Misfit terms (d^2, d^3) and interaction terms.

1. EXACT ENTROPY: Uses V9 exact mixing calculation.
2. FEATURE ENGINEERING: Pre-computes physics blocks (Misfit^2, Misfit^3).
3. MASSIVE SEARCH: 5000 Iterations, saving Top 5 equations.
"""
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Use PySR
try:
    from pysr import PySRRegressor
except ImportError:
    print("CRITICAL: PySR not installed. Please run: pip install pysr")
    sys.exit(1)

# Constants
kB_eV_per_K = 8.617333262145e-5
BCC_ELEMENTS = ['Mo', 'Ta', 'V', 'Nb', 'W', 'Ti', 'Zr', 'Hf', 'Cr', 'Al']

def calculate_s_mix_exact(row, cols):
    """Calculates Ideal Mixing Entropy from composition columns"""
    s = 0
    total = 0
    for col in cols:
        c = row.get(col, 0)
        if c > 1e-9: 
            s += c * np.log(c)
    return -1 * kB_eV_per_K * s

def run_pipeline(args):
    OUTDIR = Path(args.outdir)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input)
    
    # ==========================================
    # 1. Exact Entropy & Thermo Base
    # ==========================================
    print("--- 1. Calculating Exact Entropy ---")
    valid_bcc_cols = [f"BCC_{el}" for el in BCC_ELEMENTS if f"BCC_{el}" in df.columns]
    df['DeltaS_Exact_eV_K'] = df.apply(lambda row: calculate_s_mix_exact(row, valid_bcc_cols), axis=1)
    df['DeltaS_Exact_eV_K'] = df['DeltaS_Exact_eV_K'].clip(lower=1e-6)

    # Enthalpy
    if 'BCC_formation_energy_per_atom_from_BCC' in df.columns:
        df['DeltaH_eV'] = (df['BCC_formation_energy_per_atom_from_BCC'] - df['B2_Formation_Energy']).abs()
    else:
        df['DeltaH_eV'] = df['DeltaE_B2_minus_BCC_eV_per_atom_fixed'].abs()

    # Base Thermodynamic Ratio
    df['Term_Thermo_Ratio'] = df['DeltaH_eV'] / df['DeltaS_Exact_eV_K']
    
    # ==========================================
    # 2. Physics-Informed Target
    # ==========================================
    # We maintain the "Penalty" in the target to GUIDE the AI, but we verify it later.
    # Calibrate Base
    mask_ru = (df['B2_Ru'] > 0)
    median_ratio = df.loc[mask_ru, 'Term_Thermo_Ratio'].median()
    base_target_K = 1600 + 273.15 
    calib_C1 = base_target_K / median_ratio
    
    df['T_Base_K'] = df['Term_Thermo_Ratio'] * calib_C1
    
    # Apply Penalty for Training Target
    PENALTY_COEFF = 20000 
    misfit_col = 'lattice_misfit_delta_fixed'
    if misfit_col not in df.columns: misfit_col = 'Misfit'
    df['Misfit_Abs'] = df[misfit_col].abs()
    
    df['Ts_Target_K'] = df['T_Base_K'] - (PENALTY_COEFF * df['Misfit_Abs'])
    df['Ts_Target_K'] = df['Ts_Target_K'].clip(lower=300)
    df['Ts_Target_C'] = df['Ts_Target_K'] - 273.15
    
    # ==========================================
    # 3. Non-Linear Feature Engineering
    # ==========================================
    # Provide higher-order terms directly so AI doesn't have to "invent" them
    df['Misfit_Sq'] = df['Misfit_Abs'] ** 2
    df['Misfit_Cub'] = df['Misfit_Abs'] ** 3
    df['Strain_Energy_Term'] = (df['Misfit_Abs']**2) / df['DeltaS_Exact_eV_K']
    
    features = [
        'Term_Thermo_Ratio', 
        'Misfit_Abs', 
        'Misfit_Sq', 
        'Strain_Energy_Term'
    ]
    
    # Training Data
    mask_train = df['Ts_Target_C'].between(500, 3500)
    df_train = df[mask_train].copy()
    print(f"Training on {len(df_train)} systems.")

    # ==========================================
    # 4. Massive Symbolic Regression
    # ==========================================
    print(f"--- 3. Starting Massive Search ({args.niter} iterations) ---")
    
    if args.procs > 1 or args.procs == 0:
        p_type = args.parallelism
        deter = False
        seed = None
    else:
        p_type = 'serial'
        deter = True
        seed = args.seed

    model = PySRRegressor(
        niterations=args.niter,
        populations=args.pop,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square", "cube"], # Explicitly allow AI to square/cube things
        maxsize=args.maxsize,
        parsimony=args.parsimony,
        elementwise_loss="loss(x, y) = (x - y)^2",
        random_state=seed,
        deterministic=deter,
        procs=args.procs,
        parallelism=p_type
    )

    model.fit(df_train[features], df_train['Ts_Target_K'])

    # ==========================================
    # 5. Output Multiple Candidates
    # ==========================================
    print("\n--- Top Equations ---")
    # We iterate through the Hall of Fame to save the best 5 unique equations
    hof = model.get_hof()
    
    # Save the absolute best one for the CSV
    df['Predicted_Solvus_K'] = model.predict(df[features])
    df['Predicted_Solvus_C'] = df['Predicted_Solvus_K'] - 273.15
    
    # Filter Ideal Candidates
    df_ideal = df[
        (df['B2_Ru'] > 0) & 
        (df['Misfit_Abs'] <= args.misfit_limit) &
        (df['Predicted_Solvus_C'] >= args.ts_low_C) &
        (df['Predicted_Solvus_C'] <= args.ts_high_C)
    ].sort_values(by='Predicted_Solvus_C', ascending=False)

    print(f"\nFound {len(df_ideal)} Ideal Candidates (Best Model).")
    
    df.to_csv(OUTDIR / 'All_Predictions_V10.csv', index=False)
    df_ideal.to_csv(OUTDIR / 'Ideal_Candidates_V10.csv', index=False)
    
    # Save Top 5 Equations to Text File
    with open(OUTDIR / 'top_5_formulas.txt', 'w') as f:
        f.write("Top Equations found by Symbolic Regression:\n")
        f.write("-" * 50 + "\n")
        # PySR HOF is a pandas dataframe
        for index, row in hof.head(10).iterrows():
            f.write(f"Complexity: {row['complexity']}, Loss: {row['loss']:.4f}, R2: {row['score']:.4f}\n")
            f.write(f"Eq: {row['equation']}\n\n")

    print(f"Results in {OUTDIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--outdir', default='sr_out_v10')
    # INCREASED SEARCH PARAMETERS
    parser.add_argument('--niter', type=int, default=5000) 
    parser.add_argument('--pop', type=int, default=100)
    parser.add_argument('--maxsize', type=int, default=35)
    parser.add_argument('--parsimony', type=float, default=0.0001) # Very low penalty to allow complex terms
    
    parser.add_argument('--misfit_limit', type=float, default=0.02)
    parser.add_argument('--ts_low_C', type=float, default=1200)
    parser.add_argument('--ts_high_C', type=float, default=1900)
    parser.add_argument('--s_scale', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--procs', type=int, default=0)
    parser.add_argument('--parallelism', type=str, default='multiprocessing')
    
    args = parser.parse_args()
    if args.procs == 0:
        import multiprocessing
        args.procs = multiprocessing.cpu_count()
    run_pipeline(args)
