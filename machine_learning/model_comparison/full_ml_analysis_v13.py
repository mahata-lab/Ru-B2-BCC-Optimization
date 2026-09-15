#!/usr/bin/env python3
"""
full_ml_analysis_v13.py

Upgraded pipeline (v13):
 - Forced target: E_form_per_atom_eV
 - Models: RF, SVR, GPR (standard), GPR_highnoise (higher WhiteKernel floor)
 - Outputs in results_v13/ with per-model folders (RF, SVR, GPR, GPR_highnoise)
 - Keeps raw UQ plots and adds normalized-uncertainty plots (std / (|pred| + eps))
 - Leave-one-group-out supported via --leave_out_col
"""
import argparse, json, math, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib as mpl
import seaborn as sns

sns.set_context(
    "talk",   # or "paper", "notebook", "poster"
    rc={
        "xtick.labelsize": 21,
        "ytick.labelsize": 21,
        "axes.labelsize": 21,
        "axes.titlesize": 21,
        "legend.fontsize": 21
    }
)


# ------------- USER CONFIG -------------
TARGET_COL = "E_form_per_atom_eV"
RANDOM_STATE = 2026
TEST_SIZE = 0.30
BOOTSTRAP_N = 120
BOOTSTRAP_SAMPLE_FRAC = 0.8
CV = 4
N_ITER = 48
N_JOBS = -1
VERBOSE = 1
OUTROOT_DEFAULT = Path("./results_v13")
PREFERRED_ROM = [
    "Ru_at_frac", "mean_group", "radius_avg", "volume_per_atom_A3",
    "delta_r", "delta_a", "chi_avg", "Hmix_kJmol"
]
GROUP_COL_CANDIDATES = ["group_key", "B2_partner", "matrix_family", "matrix_group"]
EPS = 1e-8   # for normalized uncertainty denominator
# ---------------------------------------

def makedir(p): Path(p).mkdir(parents=True, exist_ok=True)
def safe_write_json(obj, fp): Path(fp).write_text(json.dumps(obj, indent=2))
def compute_metrics(y_true, y_pred):
    return {"r2": float(r2_score(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred)))}

# ---------- plotting utilities ----------
def parity_plot(y_true, y_pred, outfp, title=None):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.scatter(y_true, y_pred, s=50, alpha=0.9, edgecolor='k', linewidth=0.2)
    mn = min(min(y_true), min(y_pred)); mx = max(max(y_true), max(y_pred))
    pad = 0.05*(mx-mn) if mx>mn else 0.1
    ax.plot([mn-pad, mx+pad],[mn-pad, mx+pad], '--', color='gray', linewidth=1.2)
    ax.set_xlabel(r"DFT formation energy, $E_{\mathrm{form}}$ (eV/atom)")
    ax.set_ylabel (r"Predicted formation energy, $E_{\mathrm{form}}$ (eV/atom)")
    if title:
        ax.set_title(title)
    m = compute_metrics(y_true, y_pred)
    ax.text(0.02, 0.95, f"r² = {m['r2']:.3f}\nMAE = {m['mae']:.3f} eV/atom",
            transform=ax.transAxes, fontsize=24, va='top', ha='left',
            bbox=dict(facecolor='white', edgecolor='k', linewidth=0.5, alpha=0.9))
    fig.tight_layout(); fig.savefig(outfp, dpi=600, bbox_inches="tight"); plt.close(fig)

def uq_scatter_plot(pred_std, abs_err, outfp, title=None):
    fig, ax = plt.subplots(figsize=(8,8))
    sc = ax.scatter(pred_std, abs_err, c=abs_err, cmap='viridis', s=70, edgecolor='k', linewidth=0.2)
    ax.set_xlabel (r"Predicted uncertainty, $\sigma(E_{\mathrm{form}})$ (eV/atom)")
    ax.set_ylabel (r"Absolute error, $|\Delta E_{\mathrm{form}}|$ (eV/atom)")
    if title:
        ax.set_title(title)
    cb = plt.colorbar(sc, ax=ax); cb.set_label("Absolute error (eV/atom)")
    fig.tight_layout(); fig.savefig(outfp, dpi=600, bbox_inches="tight"); plt.close(fig)

def uq_scatter_plot_normalized(pred_std_norm, abs_err, outfp, title=None):
    fig, ax = plt.subplots(figsize=(8,8))
    sc = ax.scatter(pred_std_norm, abs_err, c=abs_err, cmap='viridis', s=70, edgecolor='k', linewidth=0.2)
    ax.set_xlabel(r"Normalized uncertainty, $\sigma(E_{\mathrm{form}})/|E_{\mathrm{form}}|$")
    ax.set_ylabel (r"Absolute error, $|\Delta E_{\mathrm{form}}|$ (eV/atom)")
    if title:
        ax.set_title(title)
    cb = plt.colorbar(sc, ax=ax); cb.set_label("Absolute error (eV/atom)")
    fig.tight_layout(); fig.savefig(outfp, dpi=600, bbox_inches="tight"); plt.close(fig)

def make_timeseries_uq_plot(y_true, pred_ensemble, outfp, ci=(5,95), title="Uncertainty Quantification"):
    mean_pred = np.mean(pred_ensemble, axis=0)
    p_lo = np.percentile(pred_ensemble, ci[0], axis=0)
    p_hi = np.percentile(pred_ensemble, ci[1], axis=0)
    order = np.argsort(y_true)
    x = np.arange(len(y_true))
    fig, ax = plt.subplots(figsize=(14,8))
    ax.plot(x, y_true[order], color='k', linewidth=2, label='DFT (True)')
    ax.plot(x, mean_pred[order], linestyle='--', color='red', linewidth=1.4, label='Prediction')
    ax.fill_between(x, p_lo[order], p_hi[order], color='red', alpha=0.25, label=f"{ci[1]-ci[0]}% CI")
    ax.set_xlabel("Sample Index (Sorted by Target)", fontsize=24)
    ax.set_ylabel (r"Formation energy, $E_{\mathrm{form}}$ (eV/atom)", fontsize=24)
    ax.set_title(title, fontsize=24); ax.legend(fontsize=24); 
    fig.tight_layout(); fig.savefig(outfp, dpi=600, bbox_inches="tight"); plt.close(fig)

def make_timeseries_uq_plot_normalized(y_true, pred_ensemble, outfp, ci=(5,95), title="Normalized UQ"):
    # pred_ensemble shape: (n_ensemble, n_samples)
    mean_pred = np.mean(pred_ensemble, axis=0)
    p_lo = np.percentile(pred_ensemble, ci[0], axis=0)
    p_hi = np.percentile(pred_ensemble, ci[1], axis=0)
    std_pred = np.std(pred_ensemble, axis=0)
    norm_std = std_pred / (np.abs(mean_pred) + EPS)
    order = np.argsort(y_true)
    x = np.arange(len(y_true))
    fig, ax = plt.subplots(figsize=(14,8))
    ax.plot(x, norm_std[order], color='tab:purple', linewidth=2, label='Normalized std')
    ax.set_xlabel("Sample Index (Sorted by Target)", fontsize=24)
    ax.set_ylabel("Normalized std (std / (|pred|+eps))", fontsize=24)
    ax.set_title(title, fontsize=24); ax.legend(fontsize=24);
    fig.tight_layout(); fig.savefig(outfp, dpi=600, bbox_inches="tight"); plt.close(fig)

# ---------- feature selection & leakage protection ----------
def detect_group_column(df):
    for c in GROUP_COL_CANDIDATES:
        if c in df.columns: return c
    for c in df.columns:
        if isinstance(c, str) and ('partner' in c or 'matrix' in c):
            return c
    return None

def prepare_feature_list(df, target_col):
    target_lower = str(target_col).lower()
    LEAKY = set([
        target_col,
        "deltae", "delta_e", "delta_r", "delta_a",
        "e_form_per_atom_eV".lower(), "e_per_atom_eV".lower(),
        "e_total_eV".lower(), "formation_energy".lower(), "E_form_matrix".lower()
    ])
    comp_cols = sorted([c for c in df.columns if str(c).startswith("x_")])
    rom_present = [c for c in PREFERRED_ROM if c in df.columns]
    numeric_cols = []
    for c in df.select_dtypes(include=[np.number]).columns:
        if c in comp_cols or c in rom_present: continue
        if str(c).lower() in LEAKY or target_lower in str(c).lower(): continue
        if str(c).lower() in ("b2_label","b2label","label"): continue
        numeric_cols.append(c)
    final = comp_cols + rom_present + numeric_cols[:16]
    if len(final) == 0:
        raise RuntimeError("No non-leaky features found. Inspect CSV columns.")
    return final

# ----------------- model training helpers -----------------
def rf_search_and_train(X_train, y_train):
    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1)
    pipeline = Pipeline([("scaler", StandardScaler()), ("rf", rf)])
    param_distributions = {
        "rf__n_estimators": [200,400,800],
        "rf__max_depth": [None, 6, 12],
        "rf__min_samples_split": [2,4,8],
        "rf__min_samples_leaf": [1,2,4],
        "rf__max_features": ["sqrt", 0.4, 0.7]
    }
    search = RandomizedSearchCV(pipeline, param_distributions, n_iter=N_ITER,
                                scoring="neg_mean_absolute_error", cv=KFold(n_splits=CV, shuffle=True, random_state=RANDOM_STATE),
                                random_state=RANDOM_STATE, n_jobs=N_JOBS, verbose=VERBOSE)
    search.fit(X_train, y_train)
    return search.best_estimator_, search

def svr_search_and_train(X_train, y_train):
    svr = SVR()
    pipeline = Pipeline([("scaler", StandardScaler()), ("svr", svr)])
    param_distributions = {
        "svr__C": [0.1,1,10,50,100],
        "svr__epsilon": [1e-3,1e-2,1e-1,0.2],
        "svr__kernel": ["rbf","linear","poly"],
        "svr__gamma": ["scale","auto"]
    }
    search = RandomizedSearchCV(pipeline, param_distributions, n_iter=min(40, N_ITER),
                                scoring="neg_mean_absolute_error", cv=KFold(n_splits=CV, shuffle=True, random_state=RANDOM_STATE),
                                random_state=RANDOM_STATE, n_jobs=N_JOBS, verbose=VERBOSE)
    search.fit(X_train, y_train)
    return search.best_estimator_, search

def gpr_train_predict(X_train, y_train, X_test, noise_level=1e-6):
    # kernel: constant * RBF + white noise
    kernel = C(1.0, (1e-3,1e3)) * RBF(length_scale=np.ones(max(1, X_train.shape[1])), length_scale_bounds=(1e-3, 1e3)) \
             + WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-10, 1e1))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=6, normalize_y=True, random_state=RANDOM_STATE)
    gpr.fit(X_train, y_train)
    y_mean, y_std = gpr.predict(X_test, return_std=True)
    return gpr, y_mean, y_std

def bootstrap_predict(pipeline, X_train, y_train, X_test, n_boot=100, frac=0.8):
    from sklearn.base import clone
    rng = np.random.RandomState(RANDOM_STATE + 777)
    n_train = X_train.shape[0]; n_test = X_test.shape[0]
    boot_preds = np.zeros((n_boot, n_test))
    for b in range(n_boot):
        idx = rng.randint(0, n_train, size=int(max(2, frac*n_train)))
        Xb = X_train[idx]; yb = y_train[idx]
        model_b = clone(pipeline)
        model_b.fit(Xb, yb)
        boot_preds[b,:] = model_b.predict(X_test)
    return boot_preds

# -------------- data splitting utilities ----------------
def grouped_leave_one_out(df, group_col):
    groups = df[group_col].unique().tolist()
    for g in groups:
        test_idx = df.index[df[group_col] == g].tolist()
        train_idx = df.index.difference(test_idx).tolist()
        yield g, train_idx, test_idx

# ----------------- run single split -----------------
def run_single_split(df, features, target_col, train_idx, test_idx, outroot, tag):
    X_train = df.loc[train_idx, features].values
    X_test  = df.loc[test_idx, features].values
    y_train = df.loc[train_idx, target_col].values
    y_test  = df.loc[test_idx, target_col].values

    makedir(outroot)

    # --- RF ---
    RF_DIR = outroot / "RF"; makedir(RF_DIR)
    print(f"[{tag}] RF: search/train ...")
    rf_model, rf_search = rf_search_and_train(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    metrics_rf = compute_metrics(y_test, y_pred_rf)
    joblib.dump(rf_model, RF_DIR / f"RF_model_{tag}.joblib")
    pd.DataFrame({"y_test": y_test, "y_pred": y_pred_rf, "residual": y_test - y_pred_rf}).to_csv(RF_DIR / f"RF_preds_{tag}.csv", index=False)
    parity_plot(y_test, y_pred_rf, RF_DIR / f"RF_parity_{tag}.png", title=f"RF Parity ({tag})")
    # RF bootstrap UQ
    boot_rf = bootstrap_predict(rf_model, X_train, y_train, X_test, n_boot=BOOTSTRAP_N, frac=BOOTSTRAP_SAMPLE_FRAC)
    rf_mean = boot_rf.mean(axis=0); rf_std = boot_rf.std(axis=0)
    pd.DataFrame(boot_rf.T, columns=[f"boot_{i}" for i in range(boot_rf.shape[0])]).assign(y_test=y_test, y_pred_mean=rf_mean, y_pred_std=rf_std).to_csv(RF_DIR / f"RF_bootstrap_{tag}.csv", index=False)
    uq_scatter_plot(rf_std, np.abs(y_test - rf_mean), RF_DIR / f"RF_uq_scatter_{tag}.png", title=f"RF UQ scatter ({tag})")
    # normalized scatter
    rf_std_norm = rf_std / (np.abs(rf_mean) + EPS)
    uq_scatter_plot_normalized(rf_std_norm, np.abs(y_test - rf_mean), RF_DIR / f"RF_uq_scatter_norm_{tag}.png", title=f"RF UQ scatter normalized ({tag})")
    make_timeseries_uq_plot(y_test, boot_rf, RF_DIR / f"RF_uq_timeseries_{tag}.png", title=f"RF UQ ({tag})")
    make_timeseries_uq_plot_normalized(y_test, boot_rf, RF_DIR / f"RF_uq_timeseries_norm_{tag}.png", title=f"RF normalized UQ ({tag})")
    safe_write_json({"model":"RF","best_params": rf_search.best_params_, "test_metrics": metrics_rf}, RF_DIR / f"RF_summary_{tag}.json")

    # --- SVR ---
    SVR_DIR = outroot / "SVR"; makedir(SVR_DIR)
    print(f"[{tag}] SVR: search/train ...")
    svr_model, svr_search = svr_search_and_train(X_train, y_train)
    y_pred_svr = svr_model.predict(X_test)
    metrics_svr = compute_metrics(y_test, y_pred_svr)
    joblib.dump(svr_model, SVR_DIR / f"SVR_model_{tag}.joblib")
    pd.DataFrame({"y_test": y_test, "y_pred": y_pred_svr, "residual": y_test - y_pred_svr}).to_csv(SVR_DIR / f"SVR_preds_{tag}.csv", index=False)
    parity_plot(y_test, y_pred_svr, SVR_DIR / f"SVR_parity_{tag}.png", title=f"SVR Parity ({tag})")
    # SVR bootstrap UQ
    boot_svr = bootstrap_predict(svr_model, X_train, y_train, X_test, n_boot=BOOTSTRAP_N, frac=BOOTSTRAP_SAMPLE_FRAC)
    svr_mean = boot_svr.mean(axis=0); svr_std = boot_svr.std(axis=0)
    pd.DataFrame(boot_svr.T, columns=[f"boot_{i}" for i in range(boot_svr.shape[0])]).assign(y_test=y_test, y_pred_mean=svr_mean, y_pred_std=svr_std).to_csv(SVR_DIR / f"SVR_bootstrap_{tag}.csv", index=False)
    uq_scatter_plot(svr_std, np.abs(y_test - svr_mean), SVR_DIR / f"SVR_uq_scatter_{tag}.png", title=f"SVR UQ scatter ({tag})")
    svr_std_norm = svr_std / (np.abs(svr_mean) + EPS)
    uq_scatter_plot_normalized(svr_std_norm, np.abs(y_test - svr_mean), SVR_DIR / f"SVR_uq_scatter_norm_{tag}.png", title=f"SVR UQ scatter normalized ({tag})")
    make_timeseries_uq_plot(y_test, boot_svr, SVR_DIR / f"SVR_uq_timeseries_{tag}.png", title=f"SVR UQ ({tag})")
    make_timeseries_uq_plot_normalized(y_test, boot_svr, SVR_DIR / f"SVR_uq_timeseries_norm_{tag}.png", title=f"SVR normalized UQ ({tag})")
    safe_write_json({"model":"SVR","best_params": svr_search.best_params_, "test_metrics": metrics_svr}, SVR_DIR / f"SVR_summary_{tag}.json")

    # --- GPR (standard) ---
    GPR_DIR = outroot / "GPR"; makedir(GPR_DIR)
    print(f"[{tag}] GPR: train/predict (standard noise) ...")
    try:
        gpr_model, gpr_mean, gpr_std = gpr_train_predict(X_train, y_train, X_test, noise_level=1e-6)
        metrics_gpr = compute_metrics(y_test, gpr_mean)
        joblib.dump(gpr_model, GPR_DIR / f"GPR_model_{tag}.joblib")
        pd.DataFrame({"y_test": y_test, "y_pred_mean": gpr_mean, "y_pred_std": gpr_std, "residual": y_test - gpr_mean}).to_csv(GPR_DIR / f"GPR_preds_{tag}.csv", index=False)
        parity_plot(y_test, gpr_mean, GPR_DIR / f"GPR_parity_{tag}.png", title=f"GPR Parity ({tag})")
        uq_scatter_plot(gpr_std, np.abs(y_test - gpr_mean), GPR_DIR / f"GPR_uq_scatter_{tag}.png", title=f"GPR UQ scatter ({tag})")
        # normalized versions
        gpr_std_norm = gpr_std / (np.abs(gpr_mean) + EPS)
        uq_scatter_plot_normalized(gpr_std_norm, np.abs(y_test - gpr_mean), GPR_DIR / f"GPR_uq_scatter_norm_{tag}.png", title=f"GPR UQ scatter normalized ({tag})")
        # timeseries
        bs = (np.random.randn(100, len(gpr_mean)) * gpr_std) + gpr_mean
        make_timeseries_uq_plot(y_test, bs, GPR_DIR / f"GPR_uq_timeseries_{tag}.png", title=f"GPR UQ ({tag})")
        make_timeseries_uq_plot_normalized(y_test, bs, GPR_DIR / f"GPR_uq_timeseries_norm_{tag}.png", title=f"GPR normalized UQ ({tag})")
        # coverage
        z95 = 1.96; z50 = 0.67
        covered95 = float(((y_test >= (gpr_mean - z95*gpr_std)) & (y_test <= (gpr_mean + z95*gpr_std))).mean())
        covered50 = float(((y_test >= (gpr_mean - z50*gpr_std)) & (y_test <= (gpr_mean + z50*gpr_std))).mean())
        safe_write_json({"model":"GPR", "test_metrics": metrics_gpr, "coverage_95": covered95, "coverage_50": covered50, "kernel": str(gpr_model.kernel_)}, GPR_DIR / f"GPR_summary_{tag}.json")
    except Exception as e:
        print("[error] GPR (standard) failed for tag", tag, ":", e)
        metrics_gpr = None

    # --- GPR (high noise floor) ---
    GPR_H_DIR = outroot / "GPR_highnoise"; makedir(GPR_H_DIR)
    print(f"[{tag}] GPR_highnoise: train/predict (higher noise floor) ...")
    try:
        gpr_h_model, gpr_h_mean, gpr_h_std = gpr_train_predict(X_train, y_train, X_test, noise_level=1e-3)
        metrics_gpr_h = compute_metrics(y_test, gpr_h_mean)
        joblib.dump(gpr_h_model, GPR_H_DIR / f"GPR_highnoise_model_{tag}.joblib")
        pd.DataFrame({"y_test": y_test, "y_pred_mean": gpr_h_mean, "y_pred_std": gpr_h_std, "residual": y_test - gpr_h_mean}).to_csv(GPR_H_DIR / f"GPR_highnoise_preds_{tag}.csv", index=False)
        parity_plot(y_test, gpr_h_mean, GPR_H_DIR / f"GPR_highnoise_parity_{tag}.png", title=f"GPR_highnoise Parity ({tag})")
        uq_scatter_plot(gpr_h_std, np.abs(y_test - gpr_h_mean), GPR_H_DIR / f"GPR_highnoise_uq_scatter_{tag}.png", title=f"GPR_highnoise UQ scatter ({tag})")
        gpr_h_std_norm = gpr_h_std / (np.abs(gpr_h_mean) + EPS)
        uq_scatter_plot_normalized(gpr_h_std_norm, np.abs(y_test - gpr_h_mean), GPR_H_DIR / f"GPR_highnoise_uq_scatter_norm_{tag}.png", title=f"GPR_highnoise UQ scatter normalized ({tag})")
        # timeseries
        bs_h = (np.random.randn(100, len(gpr_h_mean)) * gpr_h_std) + gpr_h_mean
        make_timeseries_uq_plot(y_test, bs_h, GPR_H_DIR / f"GPR_highnoise_uq_timeseries_{tag}.png", title=f"GPR_highnoise UQ ({tag})")
        make_timeseries_uq_plot_normalized(y_test, bs_h, GPR_H_DIR / f"GPR_highnoise_uq_timeseries_norm_{tag}.png", title=f"GPR_highnoise normalized UQ ({tag})")
        # coverage
        covered95_h = float(((y_test >= (gpr_h_mean - z95*gpr_h_std)) & (y_test <= (gpr_h_mean + z95*gpr_h_std))).mean())
        covered50_h = float(((y_test >= (gpr_h_mean - z50*gpr_h_std)) & (y_test <= (gpr_h_mean + z50*gpr_h_std))).mean())
        safe_write_json({"model":"GPR_highnoise", "test_metrics": metrics_gpr_h, "coverage_95": covered95_h, "coverage_50": covered50_h, "kernel": str(gpr_h_model.kernel_)}, GPR_H_DIR / f"GPR_highnoise_summary_{tag}.json")
    except Exception as e:
        print("[error] GPR_highnoise failed for tag", tag, ":", e)
        metrics_gpr_h = None

    return {"RF": metrics_rf, "SVR": metrics_svr, "GPR": metrics_gpr, "GPR_highnoise": metrics_gpr_h}

# ----------------- main -----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to CSV")
    p.add_argument("--outroot", default=str(OUTROOT_DEFAULT), help="Results directory")
    p.add_argument("--leave_out_col", default=None, help="Column name to perform leave-one-group-out (optional)")
    args = p.parse_args()

    csvp = Path(args.csv)
    if not csvp.exists():
        print("CSV not found:", csvp); sys.exit(2)
    outroot = Path(args.outroot); makedir(outroot)

    df = pd.read_csv(csvp)
    print(f"[info] Loaded CSV with {len(df)} rows, {len(df.columns)} columns.")
    if TARGET_COL not in df.columns:
        print(f"ERROR: target column '{TARGET_COL}' not found. Available columns:", df.columns.tolist()); sys.exit(2)
    print(f"[info] Using forced target: {TARGET_COL}")

    before = len(df)
    df = df[~df[TARGET_COL].isna()].copy()
    print(f"[info] Dropped {before - len(df)} rows with NaN target. Remaining: {len(df)}")
    if len(df) < 8:
        print("Not enough samples after cleaning."); sys.exit(2)

    if "B2_label" not in df.columns:
        df["B2_label"] = (df[TARGET_COL] < 0.0).astype(int)

    features = prepare_feature_list(df, TARGET_COL)
    print(f"[info] Features used ({len(features)}): {features}")

    if args.leave_out_col:
        col = args.leave_out_col
        if col not in df.columns:
            print(f"ERROR: leave_out_col '{col}' not in CSV columns."); sys.exit(2)
        print(f"[info] Running leave-one-group-out on column: {col}")
        all_metrics = {"RF": [], "SVR": [], "GPR": [], "GPR_highnoise": []}
        for grp, train_idx, test_idx in grouped_leave_one_out(df, col):
            tag = f"leaveout_{col}_{str(grp)}"
            print(f"\n--- fold: {tag} (train {len(train_idx)} / test {len(test_idx)}) ---")
            metrics = run_single_split(df, features, TARGET_COL, train_idx, test_idx, outroot, tag)
            for m in all_metrics.keys():
                all_metrics[m].append({"group": grp, "metrics": metrics.get(m)})
        safe_write_json(all_metrics, outroot / "leave_one_group_metrics.json")
        print("[done] leave-one-group-out finished. Results:", (outroot / "leave_one_group_metrics.json").resolve())
        return

    # random 70/30 stratified by B2_label if possible
    stratify_col = df["B2_label"] if df["B2_label"].nunique() > 1 else None
    if stratify_col is not None:
        indices = np.arange(len(df))
        tr, te = train_test_split(indices, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["B2_label"])
        train_idx, test_idx = tr.tolist(), te.tolist()
    else:
        tr, te = train_test_split(df.index.tolist(), test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)
        train_idx, test_idx = tr, te

    print(f"[info] Random split: train {len(train_idx)} / test {len(test_idx)}")
    metrics = run_single_split(df, features, TARGET_COL, train_idx, test_idx, outroot, tag="random70_30")
    safe_write_json({"random70_30": metrics}, outroot / "training_summary.json")
    print("[done] Results written to", outroot.resolve())

if __name__ == "__main__":
    main()
