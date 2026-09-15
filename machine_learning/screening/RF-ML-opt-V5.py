#!/usr/bin/env python3
"""
RF-ML-opt-A.py

Purpose: Run Random Forest training and produce primary/secondary RF-window selections,
branch-specific CSVs for RuTi/RuHf/RuAl, save models (joblib) and hyperparameters (json),
and regenerate figures (Fig1, Fig4, Fig7). SHAP plotting (Fig5) optional if shap is installed.

Usage: place Final_BCC_B2_Misfit_Data.csv in same directory and run:
    python RF-ML-opt-A.py
"""

import os
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

# -----------------------
# User-configurable parameters
# -----------------------
DPI = 300
FIG_FORMAT = 'png'
TITLE_FS = 18
LABEL_FS = 16
TICK_FS = 14
LEGEND_FS = 13
POINT_SIZE = 80

# RF hyperparameters (baseline)
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "random_state": 42,
    "n_jobs": -1
}

# Design window thresholds
PRIMARY_MISFIT = 0.01   # ±1%
SECONDARY_MISFIT = 0.02 # ±2%
ENERGY_CUTOFF = -0.5    # predicted formation energy (eV/atom) threshold

# Data and outputs
DATA_CSV = 'Final_BCC_B2_Misfit_Data.csv'
OUTPUT_DIR = Path('.').resolve()
TOP_N_PER_BRANCH = 10

# Branch definitions used to select rows (B2_Name substring or fallback on B2_{Element} columns)
BRANCHES = {
    'RuTi': ['ruti', 'ru-ti', 'ru ti', 'ru_ti', 'rut', 'ti'],
    'RuHf': ['ruhf', 'ru-hf', 'ru hf', 'ru_hf', 'ruhf', 'hf'],
    'RuAl': ['rual', 'ru-al', 'ru al', 'ru_al', 'rual', 'al']
}

# Attached replacement figure (optional)
ATTACHED_FIG2_A = Path('/mnt/data/rmpea_Eform_vs_RuFrac.png')
ATTACHED_FIG2_B = Path('/mnt/data/rmpea_Eform_vs_B2frac.png')
FIG2_TARGET = OUTPUT_DIR / 'Fig2_Stability_Trend.png'

# -----------------------
# Matplotlib global styles
# -----------------------
plt.rcParams.update({
    'figure.dpi': DPI,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': TICK_FS,
    'axes.titlesize': TITLE_FS,
    'axes.labelsize': LABEL_FS,
    'xtick.labelsize': TICK_FS,
    'ytick.labelsize': TICK_FS,
    'legend.fontsize': LEGEND_FS,
    'lines.linewidth': 1.2,
    'savefig.bbox': 'tight'
})

# -----------------------
# Helper functions
# -----------------------
def load_csv_or_raise(fp):
    p = Path(fp)
    if not p.exists():
        raise FileNotFoundError(f"Required CSV not found: {fp}")
    return pd.read_csv(p)

def branch_filter(df_all, substrings):
    mask = pd.Series(False, index=df_all.index)
    if 'B2_Name' in df_all.columns:
        for s in substrings:
            mask = mask | df_all['B2_Name'].str.contains(s, case=False, na=False)
    # fallback to composition columns like B2_Ti, B2_Hf, B2_Al
    for s in substrings:
        col = f'B2_{s.strip().capitalize()}'
        if col in df_all.columns:
            mask = mask | (df_all[col].fillna(0.0) > 0.01)
    return df_all[mask].copy()

def save_df(df, fname, cols=None):
    outp = OUTPUT_DIR / fname
    if cols:
        df.to_csv(outp, index=False, columns=[c for c in cols if c in df.columns])
    else:
        df.to_csv(outp, index=False)
    return outp

# -----------------------
# Load data
# -----------------------
print("Loading data:", DATA_CSV)
df = load_csv_or_raise(DATA_CSV)

# pick candidate features present in df
candidate_features = [
    'B2_VEC', 'B2_delta_r', 'B2_chi_avg', 'B2_delta_chi', 'B2_Hmix',
    'BCC_Mo', 'BCC_Ta', 'BCC_V', 'BCC_Nb',
    'B2_Ru', 'B2_Hf', 'B2_Ti', 'B2_Zr', 'B2_Al',
    'B2_Mo', 'B2_Nb', 'B2_Ta', 'B2_W', 'B2_V', 'B2_Cr', 'B2_Cu', 'B2_Si'
]
features = [f for f in candidate_features if f in df.columns]
if len(features) == 0:
    raise RuntimeError("No feature columns found in CSV. Check file.")

# targets
if 'Misfit' not in df.columns or 'B2_Formation_Energy' not in df.columns:
    raise RuntimeError("CSV must contain 'Misfit' and 'B2_Formation_Energy' columns.")
X = df[features].fillna(0.0)
y_misfit = df['Misfit'].astype(float)
y_energy = df['B2_Formation_Energy'].astype(float)

# -----------------------
# Train/test split and Random Forest models
# -----------------------
print("Training Random Forest (misfit + energy) ...")
X_train, X_test, y_m_train, y_m_test, y_e_train, y_e_test = train_test_split(
    X, y_misfit, y_energy, test_size=0.2, random_state=RF_PARAMS['random_state']
)

rf_misfit = RandomForestRegressor(**RF_PARAMS)
rf_energy = RandomForestRegressor(**RF_PARAMS)

rf_misfit.fit(X_train, y_m_train)
rf_energy.fit(X_train, y_e_train)

# save models and hyperparams
joblib.dump(rf_misfit, OUTPUT_DIR / 'rf_misfit.joblib')
joblib.dump(rf_energy, OUTPUT_DIR / 'rf_energy.joblib')
with open(OUTPUT_DIR / 'rf_hyperparams.json', 'w') as fh:
    json.dump(RF_PARAMS, fh, indent=2)

# predictions
df['Pred_Misfit'] = rf_misfit.predict(X)
df['Pred_Energy'] = rf_energy.predict(X)

# metrics
m_r2 = r2_score(y_m_test, rf_misfit.predict(X_test))
m_mae = mean_absolute_error(y_m_test, rf_misfit.predict(X_test))
e_r2 = r2_score(y_e_test, rf_energy.predict(X_test))
e_mae = mean_absolute_error(y_e_test, rf_energy.predict(X_test))

print(f"Misfit RF -> R2={m_r2:.3f}, MAE={m_mae:.4f}")
print(f"Energy RF -> R2={e_r2:.3f}, MAE={e_mae:.4f}")

# -----------------------
# Apply design windows
# -----------------------
print("Applying primary and secondary RF windows (±1% & ±2%) with energy cutoff ...")
primary_df = df[(df['Pred_Misfit'].abs() <= PRIMARY_MISFIT) & (df['Pred_Energy'] <= ENERGY_CUTOFF)].copy()
secondary_df = df[(df['Pred_Misfit'].abs() <= SECONDARY_MISFIT) & (df['Pred_Energy'] <= ENERGY_CUTOFF)].copy()

primary_df = primary_df.sort_values('Pred_Energy')
secondary_df = secondary_df.sort_values('Pred_Energy')

# pick columns for journal CSV outputs
journal_cols = [
    'B2_Name', 'a_B2', 'a_BCC', 'Pred_Misfit', 'Pred_Energy', 'Misfit',
    'B2_Ru', 'B2_Hf', 'B2_Ti', 'B2_Al', 'B2_Zr', 'B2_Mo',
    'BCC_Mo', 'BCC_Ta', 'BCC_V', 'BCC_Nb'
]
journal_cols = [c for c in journal_cols if c in df.columns]

# save
save_df(primary_df, 'BCC_B2_matrices_primary_1pct.csv', cols=journal_cols)
save_df(secondary_df, 'BCC_B2_matrices_secondary_2pct.csv', cols=journal_cols)

# combined with flag
df_comb = df.copy()
df_comb['Design_Window'] = 'none'
df_comb.loc[df_comb.index.isin(primary_df.index), 'Design_Window'] = 'primary_1pct'
df_comb.loc[df_comb.index.isin(secondary_df.index), 'Design_Window'] = 'secondary_2pct'
save_df(df_comb, 'BCC_B2_matrices_for_journal.csv', cols=journal_cols + ['Design_Window'])

print(f"Primary candidates saved: {len(primary_df)} rows")
print(f"Secondary candidates saved: {len(secondary_df)} rows")

# -----------------------
# Branch-specific CSVs for RuTi, RuHf, RuAl
# -----------------------
print("Generating branch-specific CSVs (RuTi, RuHf, RuAl)...")
branch_summary = {}
for branch, substrs in BRANCHES.items():
    b_all = branch_filter(df, substrs)
    b_primary = b_all[b_all.index.isin(primary_df.index)]
    b_secondary = b_all[b_all.index.isin(secondary_df.index)]
    fn_prim = f'{branch}_primary_1pct.csv'
    fn_sec = f'{branch}_secondary_2pct.csv'
    fn_comb = f'{branch}_combined_for_journal.csv'
    save_df(b_primary, fn_prim, cols=journal_cols)
    save_df(b_secondary, fn_sec, cols=journal_cols)
    b_all['Design_Window'] = 'none'
    b_all.loc[b_all.index.isin(b_primary.index), 'Design_Window'] = 'primary_1pct'
    b_all.loc[b_all.index.isin(b_secondary.index), 'Design_Window'] = 'secondary_2pct'
    save_df(b_all, fn_comb, cols=journal_cols + ['Design_Window'])
    branch_summary[branch] = {
        'n_all': len(b_all),
        'n_primary': len(b_primary),
        'n_secondary': len(b_secondary),
        'files': {'primary': fn_prim, 'secondary': fn_sec, 'combined': fn_comb}
    }
    print(f"Branch {branch}: total={len(b_all)}, primary={len(b_primary)}, secondary={len(b_secondary)}")

# Save top N primary per branch summary
top_rows = []
for branch in branch_summary.keys():
    combf = OUTPUT_DIR / branch_summary[branch]['files']['combined']
    if combf.exists():
        bdf = pd.read_csv(combf)
        prim = bdf[bdf['Design_Window'] == 'primary_1pct'].sort_values('Pred_Energy').head(TOP_N_PER_BRANCH)
        if not prim.empty:
            prim['Branch'] = branch
            top_rows.append(prim)
if top_rows:
    topdf = pd.concat(top_rows, ignore_index=True, sort=False)
    save_df(topdf, 'Top_primary_candidates_by_branch.csv')
    print(f"Saved Top_primary_candidates_by_branch.csv ({len(topdf)} rows)")

# -----------------------
# Replace Fig2 with attached if present
# -----------------------
if ATTACHED_FIG2_A.exists():
    shutil.copy(ATTACHED_FIG2_A, FIG2_TARGET)
    print(f"Replaced Fig2 with attached image: {ATTACHED_FIG2_A}")
elif ATTACHED_FIG2_B.exists():
    shutil.copy(ATTACHED_FIG2_B, FIG2_TARGET)
    print(f"Replaced Fig2 with attached image: {ATTACHED_FIG2_B}")
else:
    print("No attached Fig2 found; skipping Fig2 replacement.")

# -----------------------
# Figures: Fig1 (validation), Fig4 (design space), Fig7 (branch candidates)
# -----------------------
print("Generating figures (Fig1, Fig4, Fig7) ...")

# Fig1: validation
try:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter(y_m_test, rf_misfit.predict(X_test), s=POINT_SIZE/3, alpha=0.8, edgecolors='k', linewidth=0.25)
    axes[0].plot([y_misfit.min(), y_misfit.max()], [y_misfit.min(), y_misfit.max()], 'k--', lw=1.2)
    axes[0].set_xlabel('DFT Misfit (δ)')
    axes[0].set_ylabel('RF Predicted Misfit (δ)')
    axes[0].set_title('(a) Lattice Misfit Validation')
    axes[0].text(0.05, 0.9, f'R²={m_r2:.2f}\nMAE={m_mae:.3f}', transform=axes[0].transAxes,
                 bbox=dict(facecolor='white', alpha=0.8), fontsize=12)

    axes[1].scatter(y_e_test, rf_energy.predict(X_test), s=POINT_SIZE/3, alpha=0.8, edgecolors='k', linewidth=0.25)
    axes[1].plot([y_energy.min(), y_energy.max()], [y_energy.min(), y_energy.max()], 'k--', lw=1.2)
    axes[1].set_xlabel('DFT Formation Energy (eV/atom)')
    axes[1].set_ylabel('RF Predicted Energy (eV/atom)')
    axes[1].set_title('(b) Formation Energy Validation')
    axes[1].text(0.05, 0.9, f'R²={e_r2:.2f}\nMAE={e_mae:.3f} eV', transform=axes[1].transAxes,
                 bbox=dict(facecolor='white', alpha=0.8), fontsize=12)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'Fig1_Validation.{FIG_FORMAT}', dpi=DPI)
    plt.close(fig)
    print("Saved Fig1_Validation")
except Exception as e:
    warnings.warn(f"Fig1 generation failed: {e}")

# Fig4: design space with vertical ±1% and ±2% lines, hollow secondary points
try:
    fig = plt.figure(figsize=(10, 8))
    cm = plt.cm.get_cmap('viridis')
    sc = plt.scatter(df['Pred_Misfit'], df['Pred_Energy'], c=df.get('B2_Ru', 0.0), cmap=cm, alpha=0.6, s=30)
    plt.scatter(secondary_df['Pred_Misfit'], secondary_df['Pred_Energy'],
                facecolors='none', edgecolors='orange', linewidths=1.6, s=120, label='Secondary (2%)')
    plt.scatter(primary_df['Pred_Misfit'], primary_df['Pred_Energy'],
                color='red', marker='*', s=160, edgecolors='k', label='Primary (1%)')

    plt.axvline(x=PRIMARY_MISFIT, color='red', linestyle=':', linewidth=1.8)
    plt.axvline(x=-PRIMARY_MISFIT, color='red', linestyle=':', linewidth=1.8)
    plt.axvline(x=SECONDARY_MISFIT, color='orange', linestyle=':', linewidth=1.8)
    plt.axvline(x=-SECONDARY_MISFIT, color='orange', linestyle=':', linewidth=1.8)

    plt.xlim(-0.15, 0.15)
    plt.ylim(-1.6, -0.4)
    plt.xlabel('Predicted Misfit (δ)')
    plt.ylabel('Predicted Formation Energy (eV/atom)')
    plt.title('Alloy Design Space: Primary + Secondary RF Windows')
    plt.gca().invert_yaxis()
    cbar = plt.colorbar(sc)
    cbar.set_label('Ru content (atomic fraction)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'Fig4_Design_Space.{FIG_FORMAT}', dpi=DPI)
    plt.close(fig)
    print("Saved Fig4_Design_Space")
except Exception as e:
    warnings.warn(f"Fig4 generation failed: {e}")

# Fig7: branch candidates plot
try:
    fig = plt.figure(figsize=(10, 8))
    colors = {'RuTi': '#1f77b4', 'RuHf': '#2ca02c', 'RuAl': '#ff7f0e'}
    for branch in BRANCHES.keys():
        combf = OUTPUT_DIR / f'{branch}_combined_for_journal.csv'
        if not combf.exists():
            continue
        bdf = pd.read_csv(combf)
        sec = bdf[bdf['Design_Window'] == 'secondary_2pct']
        prim = bdf[bdf['Design_Window'] == 'primary_1pct']
        if not sec.empty:
            plt.scatter(sec['Pred_Misfit'], sec['Pred_Energy'], facecolors='none',
                        edgecolors=colors[branch], s=120, label=f'{branch} (2%)')
        if not prim.empty:
            plt.scatter(prim['Pred_Misfit'], prim['Pred_Energy'], marker='*',
                        c=colors[branch], s=160, edgecolors='k', label=f'{branch} (1%)')
    plt.xlim(-0.06, 0.06)
    plt.ylim(-1.4, -0.6)
    plt.xlabel('Predicted Misfit (δ)')
    plt.ylabel('Predicted Formation Energy (eV/atom)')
    plt.title('Fig7 — Branch Candidates: RuTi vs RuHf vs RuAl')
    plt.gca().invert_yaxis()
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'Fig7_Branch_Candidates.{FIG_FORMAT}', dpi=DPI)
    plt.close(fig)
    print("Saved Fig7_Branch_Candidates")
except Exception as e:
    warnings.warn(f"Fig7 generation failed: {e}")

# -----------------------
# Optional SHAP (Fig5): top 10 features with LaTeX labels
# -----------------------
print("Attempting SHAP (optional) ...")
try:
    import shap
    explainer = shap.TreeExplainer(rf_misfit)
    shap_values = explainer.shap_values(X_test)
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    top_idx = np.argsort(mean_abs)[-10:][::-1]
    top_features = list(X_test.columns[top_idx])

    latex_map = {
        'B2_VEC': r'$\mathrm{VEC}_{\mathrm{B2}}$',
        'B2_delta_r': r'$\Delta r_{\mathrm{B2}}$',
        'B2_chi_avg': r'$\bar{\chi}_{\mathrm{B2}}$',
        'B2_delta_chi': r'$\Delta\chi_{\mathrm{B2}}$',
        'B2_Hmix': r'$H_{\mathrm{mix,B2}}$',
        'B2_Ru': r'$x_{\mathrm{Ru,B2}}$',
        'B2_Hf': r'$x_{\mathrm{Hf,B2}}$',
        'B2_Ti': r'$x_{\mathrm{Ti,B2}}$',
        'B2_Al': r'$x_{\mathrm{Al,B2}}$'
    }
    top_feature_names_latex = [latex_map.get(f, f) for f in top_features]
    feats_top_df = X_test[top_features].copy()
    feats_top_df.columns = top_feature_names_latex
    shap_top = shap_values[:, top_idx]

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_top, feats_top_df, show=False, max_display=10)
    plt.title('SHAP: Drivers of Predicted Misfit', fontsize=TITLE_FS)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'Fig5_SHAP_Physics.{FIG_FORMAT}', dpi=DPI)
    plt.close()
    # save numeric table
    pd.Series(mean_abs[top_idx], index=top_feature_names_latex).sort_values(ascending=False).to_csv(
        OUTPUT_DIR / 'Top10_SHAP_values.csv', header=['mean_abs_shap'])
    print("Saved Fig5_SHAP_Physics and Top10_SHAP_values.csv")
except Exception as e:
    warnings.warn(f"SHAP skipped or failed: {e}")

# -----------------------
# Summary of outputs
# -----------------------
print("\nSaved outputs (check current directory):")
expected = [
    'rf_misfit.joblib', 'rf_energy.joblib', 'rf_hyperparams.json',
    'BCC_B2_matrices_primary_1pct.csv', 'BCC_B2_matrices_secondary_2pct.csv',
    'BCC_B2_matrices_for_journal.csv', 'Top_primary_candidates_by_branch.csv',
    'Fig1_Validation.' + FIG_FORMAT, 'Fig4_Design_Space.' + FIG_FORMAT,
    'Fig7_Branch_Candidates.' + FIG_FORMAT, 'Fig5_SHAP_Physics.' + FIG_FORMAT,
    'Top10_SHAP_values.csv', str(FIG2_TARGET)
]
for fn in expected:
    p = OUTPUT_DIR / fn
    if p.exists():
        print(" -", p.name)

print("\nOption A complete: RF models trained and candidate CSVs + figures saved.")
print("Wait to tell me when to proceed with Option B (hyperparameter search / additional modeling).")
