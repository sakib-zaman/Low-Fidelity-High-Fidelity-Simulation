# analysis.py
# =============================================================================
# Boundary Analysis, SHAP Feature Importance, and All Paper Figures
#
# This script produces every figure referenced in the paper outline:
#   1. Force–time curves comparison (LF vs HF)
#   2. Force discrepancy heatmaps
#   3. Fidelity boundary contour plots
#   4. Progressive failure sequence visualization
#   5. SHAP feature importance
#   6. Training-size ablation curve
# =============================================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import h5py
import shap

import matplotlib.patches as mpatches

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from xgboost import XGBClassifier

from selector import FidelitySelector, FEATURE_COLS
from experiment import S_MIN, S_MAX, MU_MIN, MU_MAX, DELTA_MIN, DELTA_MAX

RANDOM_SEED = 42
os.makedirs("figures", exist_ok=True)


# =============================================================================
# Figure 1: LF vs HF Force–Time Curves for 4 Representative Cases
# =============================================================================

def plot_force_time_curves(dataset_path, hdf5_path, n_cases=4):
    """
    Select 4 representative cases and plot LF vs HF force-time curves.

    Cases selected:
    1. Easy removal (low S, low mu, zero delta) — LF and HF agree
    2. Moderate (mid values) — slight discrepancy
    3. Hard removal (high S, high mu) — large discrepancy
    4. Failed case (safety violation or no separation)
    """
    df = pd.read_csv(dataset_path)

    # Select representative cases from test grid
    test = df[df["phase"] == "test_grid"].copy()

    # Case 1: Low discrepancy (LF adequate)
    case1 = test[test["adequate_label"] == 1].nsmallest(1, "force_discrepancy")
    # Case 2: Moderate discrepancy
    mid_disc = test["force_discrepancy"].median()
    case2 = test.iloc[(test["force_discrepancy"] - mid_disc).abs().argsort()[:1]]
    # Case 3: High discrepancy (LF inadequate)
    case3 = test[test["adequate_label"] == 0].nlargest(1, "force_discrepancy")
    # Case 4: Safety violation
    case4 = test[test["hf_safety_violated"] == 1]
    if len(case4) == 0:
        case4 = test.nlargest(1, "hf_max_rotation_deg")

    cases  = [case1, case2, case3, case4]
    labels = ["(a) Low Discrepancy", "(b) Moderate Discrepancy",
              "(c) High Discrepancy", "(d) Safety/Failure Case"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    with h5py.File(hdf5_path, "r") as f:
        for i, (case_df, label) in enumerate(zip(cases, labels)):
            cid = int(case_df["case_id"].values[0])
            grp_key = f"case_{cid:05d}"

            if grp_key not in f:
                axes[i].set_title(f"{label}\n(no time series)")
                continue

            grp      = f[grp_key]
            lf_force = grp["lf_force_series"][:] if "lf_force_series" in grp else None
            hf_force = grp["hf_force_series"][:] if "hf_force_series" in grp else None
            lf_time  = grp["lf_time_series"][:]  if "lf_time_series"  in grp else None
            hf_time  = grp["hf_time_series"][:]  if "hf_time_series"  in grp else None

            ax = axes[i]
            if lf_force is not None:
                ax.plot(lf_time, lf_force, "b-", linewidth=2,
                        label="LF (1 connector)", alpha=0.85)
            if hf_force is not None:
                ax.plot(hf_time, hf_force, "r-", linewidth=2,
                        label="HF (8 connectors)", alpha=0.85)

            # Annotate with key metrics
            row = case_df.iloc[0]
            ax.set_title(
                f"{label}\nS={row.S:.0f}N, μ={row.mu:.2f}, δ={row.delta*1000:.1f}mm\n"
                f"Discrepancy={row.force_discrepancy*100:.1f}%",
                fontsize=9
            )
            ax.set_xlabel("Time (s)", fontsize=10)
            ax.set_ylabel("Force (N)", fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

    plt.suptitle("LF vs. HF Force–Time Curves: Representative Cases",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/force_time_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: figures/force_time_curves.png")


# =============================================================================
# Figure 2: Force Discrepancy Heatmaps
# =============================================================================

def plot_discrepancy_heatmaps(dataset_path):
    """
    2D heatmaps of force discrepancy in each pair of parameter dimensions,
    with the third variable fixed at its median.
    """
    df   = pd.read_csv(dataset_path)
    data = df[df["phase"].isin(["lhs_main", "boundary"])].copy()

    # Fit a GP to interpolate discrepancy surface
    X = data[FEATURE_COLS].values
    y = data["force_discrepancy"].values

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    kernel = 1.0 * RBF(length_scale=[1.0, 1.0, 1.0]) + WhiteKernel(noise_level=0.01)
    gp = GaussianProcessRegressor(kernel=kernel, random_state=RANDOM_SEED,
                                   n_restarts_optimizer=3)
    gp.fit(X_s, y)

    n_grid  = 40
    S_grid  = np.linspace(S_MIN,  S_MAX,  n_grid)
    mu_grid = np.linspace(MU_MIN, MU_MAX, n_grid)
    d_grid  = np.linspace(DELTA_MIN, DELTA_MAX, n_grid)

    # Fixed values (medians)
    S_fix  = float(np.median(data["S"]))
    mu_fix = float(np.median(data["mu"]))
    d_fix  = float(np.median(data["delta"]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    panels = [
        ("S vs μ",   S_grid, mu_grid, "S (N)", "μ",
         [(s, m, d_fix) for s in S_grid for m in mu_grid],
         S_grid, mu_grid),
        ("S vs δ",   S_grid, d_grid, "S (N)", "δ (mm)",
         [(s, mu_fix, d) for s in S_grid for d in d_grid],
         S_grid, d_grid * 1000),
        ("μ vs δ",   mu_grid, d_grid, "μ", "δ (mm)",
         [(S_fix, m, d) for m in mu_grid for d in d_grid],
         mu_grid, d_grid * 1000),
    ]

    for ax, (title, xv, yv, xlabel, ylabel, pts, xplot, yplot) in \
            zip(axes, panels):
        pts_arr = np.array(pts)
        pts_s   = scaler.transform(pts_arr)
        preds, stds = gp.predict(pts_s, return_std=True)
        preds = preds.reshape(n_grid, n_grid)

        norm  = TwoSlopeNorm(vmin=0, vcenter=0.10, vmax=preds.max())
        im = ax.contourf(xplot, yplot, preds.T,
                         levels=20, cmap="RdYlGn_r", norm=norm)
        ax.contour(xplot, yplot, preds.T,
                   levels=[0.10], colors="black",
                   linewidths=2, linestyles="--")
        plt.colorbar(im, ax=ax, label="Force Discrepancy")

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{title}\n(dashed = 10% boundary)", fontsize=11)

    plt.suptitle("Force Discrepancy Between LF and HF Simulations",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/discrepancy_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: figures/discrepancy_heatmaps.png")


# =============================================================================
# Figure 3: Fidelity Boundary Contour Plots (P=0.95)
# =============================================================================

def plot_fidelity_boundary(dataset_path, model_dir):
    """
    Plot the P(adequate) = 0.95 boundary from the AI selector in
    each pair of dimensions, with third variable at 3 fixed levels.
    """
    selector = FidelitySelector.load(model_dir)
    df = pd.read_csv(dataset_path)
    test = df[df["phase"] == "test_grid"].copy()

    n_grid  = 60
    S_arr   = np.linspace(S_MIN,  S_MAX,  n_grid)
    mu_arr  = np.linspace(MU_MIN, MU_MAX, n_grid)
    d_arr   = np.linspace(DELTA_MIN, DELTA_MAX, n_grid)

    # Three fixed levels for the third variable
    S_levels  = [S_MIN  + (S_MAX  - S_MIN)  * f for f in [0.1, 0.5, 0.9]]
    mu_levels = [MU_MIN + (MU_MAX - MU_MIN) * f for f in [0.1, 0.5, 0.9]]
    d_levels  = [DELTA_MIN + (DELTA_MAX - DELTA_MIN) * f
                 for f in [0.0, 0.5, 1.0]]

    fig = plt.figure(figsize=(15, 12))
    gs  = gridspec.GridSpec(3, 3, hspace=0.4, wspace=0.35)

    level_labels = ["Low", "Mid", "High"]

    for row_i, (fix_d, d_label) in enumerate(zip(d_levels, level_labels)):
        for col_j, (fix_S, S_label) in enumerate(zip(S_levels, level_labels)):
            ax = fig.add_subplot(gs[row_i, col_j])

            # Build grid: S vs mu, fixed delta=fix_d
            SS, MU = np.meshgrid(S_arr, mu_arr)
            grid_pts = np.column_stack([
                SS.ravel(),
                MU.ravel(),
                np.full(SS.size, fix_d)
            ])
            grid_df = pd.DataFrame(grid_pts, columns=FEATURE_COLS)
            _, probs = selector.predict_batch(grid_df)
            Z = probs.reshape(SS.shape)

            # Background: probability colormap
            im = ax.contourf(S_arr, mu_arr, Z,
                             levels=np.linspace(0, 1, 21),
                             cmap="RdYlGn", alpha=0.75,
                             vmin=0, vmax=1)
            # Boundary contour at tau
            ax.contour(S_arr, mu_arr, Z,
                       levels=[selector.threshold],
                       colors="black", linewidths=2.0)

            # Overlay test points
            subset = test[
                (test["delta"] >= fix_d - 0.002) &
                (test["delta"] <= fix_d + 0.002)
            ]
            if len(subset) > 0:
                for _, row in subset.iterrows():
                    color = "#27ae60" if row["adequate_label"] == 1 else "#e74c3c"
                    ax.scatter(row["S"], row["mu"], c=color,
                               s=25, zorder=5, edgecolors="white",
                               linewidths=0.5)

            ax.set_xlabel("S (N)", fontsize=9)
            ax.set_ylabel("μ", fontsize=9)
            ax.set_title(f"δ={fix_d*1000:.1f}mm ({d_label})",
                         fontsize=10)

    # Shared colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.70])
    sm = plt.cm.ScalarMappable(cmap="RdYlGn",
                                norm=plt.Normalize(vmin=0, vmax=1))
    fig.colorbar(sm, cax=cbar_ax, label="P(LF Adequate)")

    fig.suptitle(
        f"Fidelity Boundary (P={selector.threshold:.2f} contour = black)\n"
        "Green=adequate, Red=inadequate (test points)",
        fontsize=13, fontweight="bold"
    )
    plt.savefig("figures/fidelity_boundary_contours.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: figures/fidelity_boundary_contours.png")


# =============================================================================
# Figure 4: HF Progressive Failure Sequence
# =============================================================================

def plot_failure_sequence(dataset_path, n_cases=6):
    """
    Bar chart of connector failure order for high-discrepancy cases.
    Shows which connectors fail first and the time gaps between them.
    """
    df = pd.read_csv(dataset_path)
    # Select cases where all 8 connectors failed and discrepancy is high
    high_disc = df[
        (df["hf_n_failed"] == 8) &
        (df["force_discrepancy"] > 0.15)
    ].nlargest(n_cases, "force_discrepancy")

    if len(high_disc) == 0:
        print("No complete failure cases found for failure sequence plot.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    connector_names = [
        "C0\n(FL)", "C1\n(FC)", "C2\n(FR)",
        "C3\n(ML)", "C4\n(MR)",
        "C5\n(BL)", "C6\n(BC)", "C7\n(BR)"
    ]
    # Color connectors by position: front=blue, mid=green, back=orange
    conn_colors = ["#3498db", "#3498db", "#3498db",
                   "#27ae60", "#27ae60",
                   "#e67e22", "#e67e22", "#e67e22"]

    for i, (_, row) in enumerate(high_disc.iterrows()):
        if i >= len(axes):
            break
        ax = axes[i]

        try:
            seq = json.loads(row["hf_failure_sequence"])
        except (json.JSONDecodeError, TypeError):
            ax.set_title("No sequence data")
            continue

        # seq is list of [connector_id, failure_time]
        conn_ids   = [s[0] for s in seq if s[1] is not None]
        fail_times = [s[1] for s in seq if s[1] is not None]

        if len(conn_ids) == 0:
            continue

        # Bar chart: connector ID vs failure time
        bars = ax.bar(
            range(len(conn_ids)),
            fail_times,
            color=[conn_colors[c] for c in conn_ids],
            edgecolor="white", linewidth=0.5
        )
        ax.set_xticks(range(len(conn_ids)))
        ax.set_xticklabels(
            [connector_names[c] for c in conn_ids], fontsize=8
        )
        ax.set_ylabel("Failure Time (s)", fontsize=9)
        ax.set_title(
            f"S={row.S:.0f}N, μ={row.mu:.2f}, "
            f"δ={row.delta*1000:.1f}mm\n"
            f"Discrepancy={row.force_discrepancy*100:.1f}%",
            fontsize=9
        )
        ax.grid(True, axis="y", alpha=0.3)

        # Time gap annotation between first and last failure
        if len(fail_times) >= 2:
            gap = fail_times[-1] - fail_times[0]
            ax.annotate(
                f"Δt={gap:.3f}s",
                xy=(len(conn_ids) - 1, fail_times[-1]),
                xytext=(-30, 10), textcoords="offset points",
                fontsize=8, color="red",
                arrowprops=dict(arrowstyle="->", color="red")
            )

    # Legend
    patches = [
        mpatches.Patch(color="#3498db", label="Front connectors"),
        mpatches.Patch(color="#27ae60", label="Mid connectors"),
        mpatches.Patch(color="#e67e22", label="Back connectors"),
    ]
    fig.legend(handles=patches, loc="lower center",
               ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle("HF Progressive Connector Failure Sequence\n"
                 "(Order reveals edge-loading under misalignment)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/failure_sequence.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: figures/failure_sequence.png")


# =============================================================================
# Figure 5: SHAP Feature Importance
# =============================================================================

def plot_shap_importance(dataset_path, model_dir):
    """
    SHAP summary plot for the XGBoost selector.
    Shows which input variable most drives the LF/HF decision.
    Expected result: delta (misalignment) has highest importance.
    """
    df = pd.read_csv(dataset_path)
    lhs = df[df["phase"] == "lhs_main"].copy()

    selector = FidelitySelector.load(model_dir)
    X = lhs[FEATURE_COLS].values
    X_s = selector.scaler.transform(X)

    # Extract the base XGBoost from the calibrated wrapper
    base_xgb = selector.calibrated_model.calibrated_classifiers_[0].estimator

    explainer   = shap.TreeExplainer(base_xgb)
    shap_values = explainer.shap_values(X_s)

    # For binary classification, shap_values may be shape (n, 3) or list
    if isinstance(shap_values, list):
        sv = shap_values[1]   # class 1 (adequate) SHAP values
    else:
        sv = shap_values

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Bar plot: mean |SHAP|
    mean_shap = np.abs(sv).mean(axis=0)
    feature_labels = ["Attachment\nStrength (S)", "Friction (μ)",
                      "Misalignment (δ)"]
    colors = ["#3498db", "#27ae60", "#e74c3c"]

    axes[0].barh(feature_labels, mean_shap, color=colors, edgecolor="white")
    axes[0].set_xlabel("Mean |SHAP Value|", fontsize=12)
    axes[0].set_title("Feature Importance\n(Mean |SHAP|)", fontsize=12)
    axes[0].grid(True, axis="x", alpha=0.3)

    # Scatter: SHAP value vs feature value for each variable
    for idx, (label, color) in enumerate(zip(feature_labels, colors)):
        axes[1].scatter(
            X[:, idx] / X[:, idx].max(),   # normalize to [0,1] for overlay
            sv[:, idx],
            alpha=0.3, s=15, c=color, label=label
        )
    axes[1].axhline(0, color="black", linewidth=1, linestyle="--")
    axes[1].set_xlabel("Normalized Feature Value", fontsize=12)
    axes[1].set_ylabel("SHAP Value (impact on P(adequate))", fontsize=12)
    axes[1].set_title("SHAP Values vs. Feature Values", fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("SHAP Analysis: What Drives the Fidelity Selection?",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: figures/shap_importance.png")


# =============================================================================
# Figure 6: Training-Size Ablation Curve
# =============================================================================

def plot_training_size_curve(dataset_path):
    """
    Train XGBoost selectors on progressively larger training sets.
    Plot Force MAE vs. number of training samples.
    This shows how much data is needed to achieve near-asymptotic performance.
    """
    df   = pd.read_csv(dataset_path)
    lhs  = df[df["phase"] == "lhs_main"].sort_values("case_id").reset_index(drop=True)
    test = df[df["phase"] == "test_grid"].copy()

    n_val    = int(len(lhs) * 0.15)
    val_df   = lhs.iloc[:n_val]
    train_df = lhs.iloc[n_val:]

    train_sizes = [50, 100, 150, 200, 280, 350]
    maes_mean   = []
    maes_std    = []

    # Reference: fixed HF MAE = 0 by definition
    ref_force = test["hf_peak_force"].values
    lf_force  = test["lf_peak_force"].values

    for n in train_sizes:
        if n > len(train_df):
            break

        fold_maes = []
        for rep in range(5):   # 5 repetitions with different subsets
            rng     = np.random.default_rng(RANDOM_SEED + rep)
            indices = rng.choice(len(train_df), n, replace=False)
            sub     = train_df.iloc[indices]

            X_tr = sub[FEATURE_COLS].values
            y_tr = sub["adequate_label"].values

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_v_s  = scaler.transform(val_df[FEATURE_COLS].values)
            X_te_s = scaler.transform(test[FEATURE_COLS].values)

            model = XGBClassifier(
                n_estimators=200, max_depth=4,
                random_state=RANDOM_SEED + rep,
                eval_metric="logloss", verbosity=0
            )
            model.fit(X_tr_s, y_tr)

            # Tune threshold quickly
            probs_val = model.predict_proba(X_v_s)[:, 1]
            best_tau  = 0.95
            for tau in np.linspace(0.5, 0.99, 100):
                lf_mask   = probs_val >= tau
                if lf_mask.sum() == 0:
                    continue
                inc_rate  = ((val_df["adequate_label"].values == 0) &
                              lf_mask).sum() / lf_mask.sum()
                if inc_rate <= 0.05:
                    best_tau = tau
                    break

            # Evaluate on test
            probs_te  = model.predict_proba(X_te_s)[:, 1]
            decisions = np.where(probs_te >= best_tau, "LF", "HF")
            pred_force = np.where(
                decisions == "LF", lf_force, ref_force
            )
            mae = mean_absolute_error(ref_force, pred_force)
            fold_maes.append(mae)

        maes_mean.append(np.mean(fold_maes))
        maes_std.append(np.std(fold_maes))
        print(f"  n={n:4d}: MAE={np.mean(fold_maes):.3f} ± {np.std(fold_maes):.3f}")

    # Fixed LF MAE as horizontal reference
    fl_mae = mean_absolute_error(ref_force, lf_force)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        train_sizes[:len(maes_mean)],
        maes_mean, yerr=maes_std,
        fmt="o-", color="#27ae60", linewidth=2,
        capsize=5, markersize=8, label="AI Selector"
    )
    ax.axhline(0.0, color="#2c3e50", linewidth=2,
               linestyle="--", label="Fixed HF (reference)")
    ax.axhline(fl_mae, color="#e74c3c", linewidth=2,
               linestyle="--", label="Fixed LF")
    ax.set_xlabel("Number of Training Samples", fontsize=12)
    ax.set_ylabel("Force MAE vs. Fixed HF (N)", fontsize=12)
    ax.set_title("Training-Size Ablation: Data Efficiency of AI Selector",
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/training_size_curve.png", dpi=150)
    plt.close()
    print("Saved: figures/training_size_curve.png")


# =============================================================================
# Main
# =============================================================================

def main():
    dataset_path = "data/results/dataset.csv"
    hdf5_path    = "data/results/timeseries.h5"
    model_dir    = "data/models"

    print("Generating all analysis figures...")
    print("\n[1/6] Force–time curves")
    plot_force_time_curves(dataset_path, hdf5_path)

    print("\n[2/6] Discrepancy heatmaps")
    plot_discrepancy_heatmaps(dataset_path)

    print("\n[3/6] Fidelity boundary contours")
    plot_fidelity_boundary(dataset_path, model_dir)

    print("\n[4/6] Progressive failure sequence")
    plot_failure_sequence(dataset_path)

    print("\n[5/6] SHAP feature importance")
    plot_shap_importance(dataset_path, model_dir)

    print("\n[6/6] Training-size ablation curve")
    plot_training_size_curve(dataset_path)

    print("\nAll figures saved to figures/")


if __name__ == "__main__":
    main()