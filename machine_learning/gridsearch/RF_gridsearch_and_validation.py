#!/usr/bin/env python3
"""
RF_gridsearch_and_validation_clean.py

Performs randomized hyperparameter search for RandomForest models (misfit and formation energy),
saves tuned models, CV results CSVs, and generates two publication-quality figures and a short summary CSV.

This variant explicitly removes all plot grid lines (no seaborn grids) while preserving fonts/DPI.
Requires:
 - Python 3.8+
 - numpy, pandas, matplotlib, scikit-learn, joblib, seaborn
 - Final_BCC_B2_Misfit_Data.csv in the working directory
"""

import os
from pathlib import Path
import json
import warnings
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, RepeatedKFold
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

# -----------------------
# User settings (tweak here)
# -----------------------
DATA_CSV = 'Final_BCC_B2_Misfit_Data.csv'
OUTPUT_DIR = Path('.').resolve()
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_JOBS = -1
N_ITER = 60           # number of parameter settings sampled by RandomizedSearchCV
CV_REPEATS = 3
CV_SPLITS = 5
DPI = 300
FIG_FORMAT = 'png'
FONT_SIZE = 12

# baseline features (same as used previously)
candidate_features = [
    'B2_VEC', 'B2_delta_r', 'B2_chi_avg', 'B2_delta_chi', 'B2_Hmix',
    'BCC_Mo', 'BCC_Ta', 'BCC_V', 'BCC_Nb',
    'B2_Ru', 'B2_Hf', 'B2_Ti', 'B2_Zr', 'B2_Al',
    'B2_Mo', 'B2_Nb', 'B2_Ta', 'B2_W', 'B2_V', 'B2_Cr', 'B2_Cu', 'B2_Si'
]

# parameter search space for RF
param_dist = {
    'n_estimators': [100, 200, 400, 600, 1000],
    'max_depth': [None, 8, 12, 16, 20, 28],
    'min_samples_leaf': [1, 2, 4, 8],
    'min_samples_split': [2, 4, 8],
    'max_features': ['auto', 'sqrt', 0.3, 0.5],
    'bootstrap': [True, False]
}

# which target to optimize in the script: 'misfit' or 'energy' or both
TARGETS = ['misfit', 'energy']  # will run separate searches for each

# -----------------------
# Prepare plotting defaults (NO grids)
# -----------------------
plt.rcParams.update({
    'figure.dpi': DPI,
    'font.size': FONT_SIZE,
    'axes.titlesize': FONT_SIZE + 2,
    'axes.labelsize': FONT_SIZE,
    'xtick.labelsize': FONT_SIZE - 1,
    'ytick.labelsize': FONT_SIZE - 1,
    'legend.fontsize': FONT_SIZE - 1,
    'axes.grid': False,       # ensure no grids by default
    'savefig.bbox': 'tight'
})
# Use seaborn white style (no grid)
sns.set_style('white')


# -----------------------
# Helper functions
# -----------------------
def load_data(csv_path):
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return pd.read_csv(p)


def prepare_features(df):
    features = [f for f in candidate_features if f in df.columns]
    if len(features) == 0:
        raise RuntimeError("No features found in CSV. Edit candidate_features.")
    X = df[features].fillna(0.0)
    return X, features


def run_randomized_search(X_train, y_train, base_estimator, param_dist, n_iter=60, cv=None, scoring='neg_mean_absolute_error', n_jobs=-1, random_state=RANDOM_STATE):
    rnd = RandomizedSearchCV(
        estimator=base_estimator,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        refit=True,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=2,
        return_train_score=True
    )
    t0 = time.time()
    rnd.fit(X_train, y_train)
    t1 = time.time()
    print(f"RandomizedSearchCV finished in {t1 - t0:.1f}s; best score (CV) = {rnd.best_score_:.4f}")
    return rnd


# -----------------------
# Main
# -----------------------
def main():
    print("Loading data...")
    df = load_data(DATA_CSV)

    # targets
    if 'Misfit' not in df.columns or 'B2_Formation_Energy' not in df.columns:
        raise RuntimeError("CSV must contain 'Misfit' and 'B2_Formation_Energy' columns.")

    X_all, feature_list = prepare_features(df)
    y_all_misfit = df['Misfit'].astype(float)
    y_all_energy = df['B2_Formation_Energy'].astype(float)

    # split once for fair test
    X_train, X_test, y_m_train, y_m_test, y_e_train, y_e_test = train_test_split(
        X_all, y_all_misfit, y_all_energy, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    cv = RepeatedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE)

    summary_rows = []

    for target in TARGETS:
        if target == 'misfit':
            y_train = y_m_train
            y_test = y_m_test
            baseline_model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=RANDOM_STATE, n_jobs=N_JOBS)
        elif target == 'energy':
            y_train = y_e_train
            y_test = y_e_test
            baseline_model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=RANDOM_STATE, n_jobs=N_JOBS)
        else:
            raise RuntimeError("Unknown target")

        print(f"\n=== Target: {target} ===")
        # baseline fit (for reference)
        baseline_model.fit(X_train, y_train)
        y_pred_test_baseline = baseline_model.predict(X_test)
        base_r2 = r2_score(y_test, y_pred_test_baseline)
        base_mae = mean_absolute_error(y_test, y_pred_test_baseline)
        print(f"Baseline test R2={base_r2:.4f}, MAE={base_mae:.4f}")

        # Randomized search (optimize MAE)
        print("Running randomized search (optimizing CV neg_mean_absolute_error)...")
        rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=N_JOBS)
        rnd_search = run_randomized_search(X_train, y_train, rf, param_dist, n_iter=N_ITER, cv=cv, scoring='neg_mean_absolute_error', n_jobs=N_JOBS)

        # Save cv results (DataFrame)
        cvres = pd.DataFrame(rnd_search.cv_results_)
        cvres_path = OUTPUT_DIR / f'cv_results_{target}.csv'
        cvres.to_csv(cvres_path, index=False)
        print("Saved CV results to:", cvres_path)

        # Best estimator metrics on held-out test
        best = rnd_search.best_estimator_
        y_pred_test = best.predict(X_test)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        print(f"Tuned RF test R2={test_r2:.4f}, MAE={test_mae:.4f}")

        # Save best model
        model_fname = OUTPUT_DIR / f'rf_{target}_tuned.joblib'
        joblib.dump(best, model_fname)
        print("Saved tuned model to:", model_fname)

        # Save summary row
        summary_rows.append({
            'target': target,
            'baseline_r2': base_r2,
            'baseline_mae': base_mae,
            'tuned_r2': test_r2,
            'tuned_mae': test_mae,
            'best_params': json.dumps(rnd_search.best_params_)
        })

        # Visual: plot CV sampled configurations colored by mean_test_score and annotate best
        if 'mean_test_score' in cvres.columns:
            cvres_plot = cvres.copy()
            # mean_test_score is negative MAE (since we optimized neg_mean_absolute_error)
            cvres_plot['mean_cv_MAE'] = -cvres_plot['mean_test_score']
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            ax.grid(False)
            ax.scatter(range(len(cvres_plot)), cvres_plot['mean_cv_MAE'], s=40, alpha=0.8)
            best_idx = rnd_search.best_index_
            ax.scatter(best_idx, cvres_plot.loc[best_idx, 'mean_cv_MAE'], color='red', s=120, label='best (CV)')
            ax.set_xlabel('Randomized sample index (arbitrary)')
            ax.set_ylabel('CV mean MAE')
            ax.set_title(f'Grid-search samples: CV mean MAE ({target})')
            ax.legend()
            figpath = OUTPUT_DIR / f'Fig_grid_search_scores_{target}.{FIG_FORMAT}'
            plt.savefig(figpath, dpi=DPI)
            plt.close(fig)
            print("Saved grid-search scores figure:", figpath)

        # Parity plot comparing DFT (y_test) vs tuned RF predictions
        fig, ax = plt.subplots(figsize=(6.5, 6))
        ax.grid(False)
        ax.scatter(y_test, y_pred_test, s=40, alpha=0.8, edgecolor='k')
        mn = min(y_test.min(), y_pred_test.min())
        mx = max(y_test.max(), y_pred_test.max())
        ax.plot([mn, mx], [mn, mx], 'k--', linewidth=1)
        ax.set_xlabel('DFT target')
        ax.set_ylabel('RF tuned prediction')
        ax.set_title(f'Parity: DFT vs RF ({target})\nR²={test_r2:.3f}, MAE={test_mae:.3f}')
        figpath = OUTPUT_DIR / f'Fig_tuned_parity_{target}.{FIG_FORMAT}'
        plt.savefig(figpath, dpi=DPI)
        plt.close(fig)
        print("Saved parity figure:", figpath)

    # Save top-level summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / 'RF_tuned_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print("Saved RF tuned summary to", summary_path)

    print("\nAll done. Produced:")
    for p in OUTPUT_DIR.glob('cv_results_*.csv'):
        print(" -", p.name)
    for p in OUTPUT_DIR.glob('Fig_grid_search_scores_*.png'):
        print(" -", p.name)
    for p in OUTPUT_DIR.glob('Fig_tuned_parity_*.png'):
        print(" -", p.name)
    print(" - RF_tuned_summary.csv")
    for p in OUTPUT_DIR.glob('rf_*_tuned.joblib'):
        print(" -", p.name)


if __name__ == '__main__':
    main()
