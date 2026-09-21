# evaluate.py
# =============================================================================
# Four-Strategy Evaluation on the Hold-Out Test Grid
#
# Strategies evaluated:
#   FL  — Fixed Low Fidelity  (always use LF result)
#   FH  — Fixed High Fidelity (always use HF result — reference)
#   RB  — Rule-Based Selection (normalized uncertainty score threshold)
#   AI  — AI-Guided Selection  (calibrated XGBoost)
#
# All strategies are evaluated on the SAME 125 test conditions.
# The test grid results are already in the dataset CSV from Phase 3.
# No new simulations are run here — only the recorded LF/HF results
# are re-used, and the strategy determines which one to report.
#
# Bootstrap confidence intervals are computed for all metrics.
# =============================================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

from selector import FidelitySelector
from experiment import (
    S_MIN, S_MAX,
    MU_MIN, MU_MAX,
    DELTA_MIN, DELTA_MAX,
)

RANDOM_SEED     = 42
N_BOOTSTRAP     = 2000
EPSILON_FORCE   = 1.0


# =============================================================================
# Rule-Based Selector
# =============================================================================

class RuleBasedSelector:
    """
    Simple rule: compute a normalized uncertainty score U and select HF
    when U > threshold. Threshold is tuned on the validation split
    using the same risk constraint as the AI selector.

    U = (1/3)[(S-Smin)/(Smax-Smin) + (mu-mumin)/(mumax-mumin)
              + (delta-deltamin)/(deltamax-deltamin)]

    This is intentionally simple. Its weakness is that it treats all
    three variables as equally important, which may not match reality.
    """

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def score(self, S, mu, delta):
        s_norm = (S     - S_MIN)     / (S_MAX     - S_MIN)
        m_norm = (mu    - MU_MIN)    / (MU_MAX    - MU_MIN)
        d_norm = (delta - DELTA_MIN) / (DELTA_MAX - DELTA_MIN)
        return (s_norm + m_norm + d_norm) / 3.0

    def predict(self, S, mu, delta):
        u = self.score(S, mu, delta)
        return "LF" if u <= self.threshold else "HF", u

    def predict_batch(self, df):
        scores = df.apply(
            lambda r: self.score(r["S"], r["mu"], r["delta"]), axis=1
        ).values
        decisions = np.where(scores <= self.threshold, "LF", "HF")
        return decisions, scores

    def tune_threshold(self, val_df, max_incorrect_lf=0.05):
        """Tune threshold on validation data using same risk constraint as AI."""
        thresholds = np.linspace(0.0, 1.0, 200)
        best_threshold = -1.0  # No U in [0,1] can satisfy U <= -1: always select HF
        best_lf_rate = 0.0

        scores = val_df.apply(
            lambda r: self.score(r["S"], r["mu"], r["delta"]), axis=1
        ).values
        y_val = val_df["adequate_label"].values

        for tau in thresholds:
            assigned_lf = scores <= tau
            n_lf = assigned_lf.sum()
            if n_lf == 0:
                continue
            incorrect = ((y_val == 0) & assigned_lf).sum() / n_lf
            if incorrect <= max_incorrect_lf:
                lf_rate = n_lf / len(y_val)
                if lf_rate > best_lf_rate:
                    best_lf_rate   = lf_rate
                    best_threshold = tau

        self.threshold = float(best_threshold)
        print(f"Rule-based threshold tuned: U* = {self.threshold:.3f} "
              f"(LF rate: {best_lf_rate*100:.1f}%)")
        return self.threshold


# =============================================================================
# Metric computation
# =============================================================================

def compute_metrics(test_df, decisions):
    """
    Compute all evaluation metrics for one strategy.

    For each case, the strategy either chose "LF" or "HF".
    If "LF" was chosen, use lf_peak_force, lf_success, lf_safety_violated.
    If "HF" was chosen, use hf_peak_force, hf_success, hf_safety_violated.
    HF values are ALWAYS the reference for agreement computation.

    Parameters
    ----------
    test_df   : pd.DataFrame  Test grid results (125 rows)
    decisions : array-like    "LF" or "HF" for each row

    Returns
    -------
    dict   All metrics for this strategy
    """
    decisions = np.array(decisions)
    n = len(test_df)

    # Predicted values based on strategy decision
    pred_force   = np.where(
        decisions == "LF",
        test_df["lf_peak_force"].values,
        test_df["hf_peak_force"].values
    )
    pred_success = np.where(
        decisions == "LF",
        test_df["lf_success"].values,
        test_df["hf_success"].values
    )
    pred_safety  = np.where(
        decisions == "LF",
        test_df["lf_safety_violated"].values,
        test_df["hf_safety_violated"].values
    )

    # Reference: always HF
    ref_force   = test_df["hf_peak_force"].values
    ref_success = test_df["hf_success"].values
    ref_safety  = test_df["hf_safety_violated"].values

    # Normalized force error per case
    force_errors = np.abs(pred_force - ref_force) / np.maximum(ref_force, EPSILON_FORCE)

    # Metrics
    mae           = float(np.mean(np.abs(pred_force - ref_force)))
    force_range   = ref_force.max() - ref_force.min()
    nrmse         = float(np.sqrt(np.mean((pred_force - ref_force)**2)) /
                          max(force_range, EPSILON_FORCE))
    success_agree = float(np.mean(pred_success == ref_success))
    safety_agree  = float(np.mean(pred_safety  == ref_safety))

    n_lf          = int((decisions == "LF").sum())
    n_hf          = int((decisions == "HF").sum())
    lf_rate       = n_lf / n

    # Incorrect LF: cases assigned to LF that were truly inadequate
    truly_inadequate = (test_df["adequate_label"].values == 0)
    assigned_lf_mask = (decisions == "LF")
    n_incorrect_lf   = int((truly_inadequate & assigned_lf_mask).sum())
    incorrect_lf_rate = (n_incorrect_lf / max(n_lf, 1))

    # Computational time
    lf_times = test_df["lf_wall_time"].values
    hf_times = test_df["hf_wall_time"].values
    total_time = np.sum(
        np.where(decisions == "LF", lf_times, hf_times)
    )
    fixed_hf_time = np.sum(hf_times)
    speedup = fixed_hf_time / max(total_time, 1e-6)

    return {
        "n_lf":             n_lf,
        "n_hf":             n_hf,
        "lf_rate":          lf_rate,
        "incorrect_lf_rate": incorrect_lf_rate,
        "n_incorrect_lf":   n_incorrect_lf,
        "force_mae":        mae,
        "force_nrmse":      nrmse,
        "success_agree":    success_agree,
        "safety_agree":     safety_agree,
        "total_time":       total_time,
        "speedup":          speedup,
        "force_errors":     force_errors,   # per-case, for bootstrap
        "pred_success":     pred_success,   # for bootstrap
        "pred_safety":      pred_safety,
    }


def bootstrap_ci(data, stat_fn=np.mean, n_boot=N_BOOTSTRAP,
                 alpha=0.05, seed=RANDOM_SEED):
    """
    Non-parametric bootstrap confidence interval.

    Parameters
    ----------
    data    : array-like   Observed values
    stat_fn : callable     Statistic to compute (default: mean)
    n_boot  : int          Number of bootstrap samples
    alpha   : float        Significance level (default: 0.05 → 95% CI)

    Returns
    -------
    (float, float)   (lower, upper) confidence interval
    """
    rng    = np.random.default_rng(seed)
    stats_ = [
        stat_fn(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_boot)
    ]
    lower = float(np.percentile(stats_, 100 * alpha / 2))
    upper = float(np.percentile(stats_, 100 * (1 - alpha / 2)))
    return lower, upper


# =============================================================================
# Main evaluation
# =============================================================================

def evaluate_all_strategies(
    dataset_path   = "data/results/dataset.csv",
    model_dir      = "data/models",
    output_dir     = "data/results",
    figures_dir    = "figures"
):
    os.makedirs(figures_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    df = pd.read_csv(dataset_path)
    test_df  = df[df["phase"] == "test_grid"].copy().reset_index(drop=True)
    val_df   = df[df["phase"] == "lhs_main"].copy()
    n_val    = int(len(val_df) * 0.15)
    val_df   = val_df.sort_values("case_id").iloc[:n_val].reset_index(drop=True)

    print(f"Test grid: {len(test_df)} cases")
    print(f"Validation (for RB tuning): {len(val_df)} cases")

    # ------------------------------------------------------------------
    # Strategy 1: Fixed Low Fidelity
    # ------------------------------------------------------------------
    fl_decisions = ["LF"] * len(test_df)

    # ------------------------------------------------------------------
    # Strategy 2: Fixed High Fidelity (reference)
    # ------------------------------------------------------------------
    fh_decisions = ["HF"] * len(test_df)

    # ------------------------------------------------------------------
    # Strategy 3: Rule-Based Selection
    # ------------------------------------------------------------------
    rb_selector = RuleBasedSelector()
    rb_selector.tune_threshold(val_df, max_incorrect_lf=0.05)
    rb_decisions, _ = rb_selector.predict_batch(test_df)

    # ------------------------------------------------------------------
    # Strategy 4: AI-Guided Selection
    # ------------------------------------------------------------------
    ai_selector = FidelitySelector.load(model_dir)
    ai_decisions, ai_probs = ai_selector.predict_batch(test_df)

    # ------------------------------------------------------------------
    # Compute metrics for all strategies
    # ------------------------------------------------------------------
    strategies = {
        "Fixed LF":  fl_decisions,
        "Fixed HF":  fh_decisions,
        "Rule-Based": rb_decisions,
        "AI-Guided": ai_decisions,
    }

    results = {}
    for name, decisions in strategies.items():
        m = compute_metrics(test_df, decisions)
        results[name] = m
        print(f"\n{'='*50}")
        print(f"Strategy: {name}")
        print(f"  LF rate:            {m['lf_rate']*100:.1f}%")
        print(f"  Incorrect LF rate:  {m['incorrect_lf_rate']*100:.1f}%")
        print(f"  Force MAE:          {m['force_mae']:.2f} N")
        print(f"  Force NRMSE:        {m['force_nrmse']*100:.2f}%")
        print(f"  Success agreement:  {m['success_agree']*100:.1f}%")
        print(f"  Safety agreement:   {m['safety_agree']*100:.1f}%")
        print(f"  Speedup vs FH:      {m['speedup']:.2f}×")

    # ------------------------------------------------------------------
    # Bootstrap confidence intervals
    # ------------------------------------------------------------------
    print("\n--- Bootstrap 95% CIs (Force MAE) ---")
    ci_results = {}
    ref_force = test_df["hf_peak_force"].values
    for name, m in results.items():
        abs_errors = np.abs(
            np.where(
                np.array(strategies[name]) == "LF",
                test_df["lf_peak_force"].values,
                ref_force
            ) - ref_force
        )
        lo, hi = bootstrap_ci(abs_errors, np.mean)
        ci_results[name] = (lo, hi)
        print(f"  {name:15s}  MAE={np.mean(abs_errors):.2f} N  "
              f"95% CI [{lo:.2f}, {hi:.2f}]")

    # ------------------------------------------------------------------
    # Summary table (saved as CSV)
    # ------------------------------------------------------------------
    rows = []
    for name, m in results.items():
        lo, hi = ci_results[name]
        rows.append({
            "Strategy":           name,
            "LF_Rate_%":          round(m["lf_rate"] * 100, 1),
            "Incorrect_LF_%":     round(m["incorrect_lf_rate"] * 100, 1),
            "Force_MAE_N":        round(m["force_mae"], 2),
            "Force_MAE_CI_lo":    round(lo, 2),
            "Force_MAE_CI_hi":    round(hi, 2),
            "Force_NRMSE_%":      round(m["force_nrmse"] * 100, 2),
            "Success_Agree_%":    round(m["success_agree"] * 100, 1),
            "Safety_Agree_%":     round(m["safety_agree"] * 100, 1),
            "Speedup_vs_FH":      round(m["speedup"], 2),
            "Total_Time_s":       round(m["total_time"], 1),
        })
    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(output_dir, "strategy_comparison.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nStrategy comparison saved: {summary_path}")
    print(summary_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Figure: Cost–Accuracy tradeoff (the paper's centerpiece figure)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    colors  = {"Fixed LF": "#e74c3c", "Fixed HF": "#2c3e50",
               "Rule-Based": "#f39c12", "AI-Guided": "#27ae60"}
    markers = {"Fixed LF": "s", "Fixed HF": "D",
               "Rule-Based": "^", "AI-Guided": "o"}

    for name, m in results.items():
        lo, hi = ci_results[name]
        mae    = m["force_mae"]
        cost   = m["total_time"]

        ax.errorbar(cost, mae,
                    yerr=[[mae - lo], [hi - mae]],
                    fmt=markers[name],
                    color=colors[name],
                    markersize=12,
                    capsize=5,
                    linewidth=2,
                    label=name)
        ax.annotate(name,
                    xy=(cost, mae),
                    xytext=(8, 4),
                    textcoords="offset points",
                    fontsize=10,
                    color=colors[name])

    ax.set_xlabel("Total Simulation Time (seconds)", fontsize=13)
    ax.set_ylabel("Force MAE vs. Fixed HF (N)", fontsize=13)
    ax.set_title("Accuracy–Cost Tradeoff: Four Fidelity Strategies\n"
                 "(Lower-left is better)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "cost_accuracy_tradeoff.png"),
                dpi=150)
    plt.close()
    print("Saved: figures/cost_accuracy_tradeoff.png")

    # ------------------------------------------------------------------
    # Figure: AI confusion matrix on test set
    # ------------------------------------------------------------------
    y_true = test_df["adequate_label"].values
    y_pred_ai = (ai_decisions == "LF").astype(int)

    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred_ai):
        cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: HF", "Pred: LF"], fontsize=11)
    ax.set_yticklabels(["True: Inadequate", "True: Adequate"], fontsize=11)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    fontsize=16, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title("AI Selector Confusion Matrix\n(Test Grid)", fontsize=13)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "ai_confusion_matrix.png"), dpi=150)
    plt.close()
    print("Saved: figures/ai_confusion_matrix.png")

    return results, summary_df


if __name__ == "__main__":
    evaluate_all_strategies()