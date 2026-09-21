# experiment.py
# =============================================================================
# Experiment orchestration: pilot study + main LHS dataset + boundary phase.
#
# Run this script directly to execute all three experiment phases in order.
# Each phase saves its own records and can be re-run independently.
#
# Usage:
#   python experiment.py --phase pilot
#   python experiment.py --phase main
#   python experiment.py --phase boundary
#   python experiment.py --phase testgrid
#   python experiment.py --phase all
# =============================================================================

import argparse
import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube, scale
import joblib
import os
import sys

from runner import ExperimentRunner

# =============================================================================
# Parameter bounds — defined once here, used everywhere
# =============================================================================
S_MIN,     S_MAX     = 20.0,  120.0   # Attachment strength (N)
MU_MIN,    MU_MAX    = 0.10,  0.80    # Friction coefficient
DELTA_MIN, DELTA_MAX = 0.000, 0.008   # Misalignment (m)

BOUNDS_LOW  = [S_MIN,  MU_MIN,  DELTA_MIN]
BOUNDS_HIGH = [S_MAX,  MU_MAX,  DELTA_MAX]

RANDOM_SEED = 42   # global seed for full reproducibility


# =============================================================================
# Phase 0 — Pilot Study (3 x 3 x 3 = 27 conditions)
# =============================================================================

def generate_pilot_conditions():
    """
    Full factorial 3-level design.
    Returns list of condition dicts.
    """
    S_levels     = [30.0, 70.0, 110.0]
    mu_levels    = [0.15, 0.45, 0.75]
    delta_levels = [0.000, 0.004, 0.008]

    conditions = []
    case_id = 0
    for S in S_levels:
        for mu in mu_levels:
            for delta in delta_levels:
                conditions.append({
                    "case_id": case_id,
                    "S":       S,
                    "mu":      mu,
                    "delta":   delta,
                    "seed":    RANDOM_SEED,
                })
                case_id += 1
    return conditions


def run_pilot(runner):
    print("=" * 60)
    print("PHASE 0: PILOT STUDY (27 conditions)")
    print("=" * 60)
    conditions = generate_pilot_conditions()
    df = runner.run_batch(conditions, phase_label="pilot",
                          save_timeseries=True)

    # -------------------------------------------------------------------------
    # Pilot acceptance criteria — print pass/fail for each
    # -------------------------------------------------------------------------
    print("\n--- Pilot Acceptance Criteria ---")

    success_rate = df["hf_success"].mean()
    print(f"[{'PASS' if 0.20 <= success_rate <= 0.80 else 'FAIL'}] "
          f"HF success rate: {success_rate*100:.1f}% (target: 20–80%)")

    mean_speedup = df["speedup_ratio"].mean()
    print(f"[{'PASS' if mean_speedup >= 3.0 else 'FAIL'}] "
          f"Mean HF/LF speedup: {mean_speedup:.2f}x (target: >=3x)")

    high_disc = (df["force_discrepancy"] > 0.10).sum()
    print(f"[{'PASS' if high_disc >= 8 else 'FAIL'}] "
          f"Conditions with >10% force discrepancy: {high_disc} (target: >=8)")

    safety_count = df["hf_safety_violated"].sum()
    print(f"[{'PASS' if safety_count >= 5 else 'FAIL'}] "
          f"Safety violations (HF): {safety_count} (target: >=5)")

    adequate_rate = df["adequate_label"].mean()
    print(f"[INFO] Adequate rate: {adequate_rate*100:.1f}%")
    print(f"[INFO] Force discrepancy: "
          f"mean={df['force_discrepancy'].mean():.3f}, "
          f"max={df['force_discrepancy'].max():.3f}")

    return df


# =============================================================================
# Phase 1 — Main LHS Dataset (500 conditions)
# =============================================================================

def generate_lhs_conditions(n=500, start_id=100):
    """
    Latin Hypercube Sampling over the three-dimensional parameter space.

    LHS ensures uniform coverage of the space — much better than random
    uniform sampling, which tends to cluster. With 500 samples and 3
    variables, LHS guarantees good coverage of all marginal distributions.
    """
    sampler = LatinHypercube(d=3, seed=RANDOM_SEED)
    sample  = sampler.random(n=n)   # shape (n, 3), values in [0, 1]

    # Scale from [0,1]^3 to actual parameter ranges
    scaled  = scale(sample, BOUNDS_LOW, BOUNDS_HIGH)

    conditions = []
    for i, (S, mu, delta) in enumerate(scaled):
        conditions.append({
            "case_id": start_id + i,
            "S":       float(S),
            "mu":      float(mu),
            "delta":   float(delta),
            "seed":    RANDOM_SEED + i,
        })
    return conditions


def run_main(runner, n_workers=1):
    print("=" * 60)
    print("PHASE 1: MAIN LHS DATASET (500 conditions)")
    print("=" * 60)
    conditions = generate_lhs_conditions(n=500, start_id=100)
    df = runner.run_batch(conditions, phase_label="lhs_main",
                          save_timeseries=True)
    return df


# =============================================================================
# Phase 2 — Adaptive Boundary Enrichment (200 conditions)
#
# Strategy:
#   1. Train a preliminary XGBoost on Phase 1 data
#   2. Predict P(adequate) on a large candidate pool
#   3. Keep only candidates where 0.15 <= P <= 0.85 (boundary band)
#   4. Sample 200 from this band
#
# This concentrates samples near the fidelity boundary, sharpening it.
# =============================================================================

def generate_boundary_conditions(lhs_df, n=200, start_id=700):
    """
    Adaptively sample near the predicted fidelity boundary.
    Requires the Phase 1 dataset to already exist.
    """
    try:
        from xgboost import XGBClassifier
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("XGBoost not installed. Skipping boundary enrichment.")
        return []

    # Train quick preliminary model
    features = ["S", "mu", "delta"]
    X = lhs_df[features].values
    y = lhs_df["adequate_label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    prelim_model = XGBClassifier(
        n_estimators=100, max_depth=3,
        random_state=RANDOM_SEED, eval_metric="logloss",
        verbosity=0
    )
    prelim_model.fit(X_scaled, y)

    # Generate large candidate pool
    rng = np.random.default_rng(RANDOM_SEED + 999)
    n_candidates = 5000
    candidates = rng.uniform(
        low=BOUNDS_LOW, high=BOUNDS_HIGH,
        size=(n_candidates, 3)
    )

    # Predict probabilities
    probs = prelim_model.predict_proba(
        scaler.transform(candidates)
    )[:, 1]

    # Keep boundary band: 0.15 <= P <= 0.85
    boundary_mask = (probs >= 0.15) & (probs <= 0.85)
    boundary_candidates = candidates[boundary_mask]

    print(f"  Boundary band: {boundary_mask.sum()} candidates from {n_candidates}")

    if len(boundary_candidates) < n:
        print(f"  Warning: only {len(boundary_candidates)} boundary candidates. "
              f"Using all of them.")
        selected = boundary_candidates
    else:
        idx = rng.choice(len(boundary_candidates), n, replace=False)
        selected = boundary_candidates[idx]

    conditions = []
    for i, (S, mu, delta) in enumerate(selected):
        conditions.append({
            "case_id": start_id + i,
            "S":       float(S),
            "mu":      float(mu),
            "delta":   float(delta),
            "seed":    RANDOM_SEED + start_id + i,
        })
    return conditions


def run_boundary(runner, lhs_df, n_workers=1):
    print("=" * 60)
    print("PHASE 2: BOUNDARY ENRICHMENT (up to 200 conditions)")
    print("=" * 60)
    conditions = generate_boundary_conditions(lhs_df, n=200, start_id=700)
    if not conditions:
        return pd.DataFrame()
    df = runner.run_batch(conditions, phase_label="boundary",
                          save_timeseries=False)
    return df


# =============================================================================
# Phase 3 — Hold-out Test Grid (5 x 5 x 5 = 125 conditions)
#
# IMPORTANT: This grid is generated with a different seed and NEVER used
# during training or threshold tuning. It is only loaded during evaluate.py.
# Generate it first, save it, then leave it untouched.
# =============================================================================

def generate_test_grid(start_id=1000):
    """
    Regular 5x5x5 grid for interpretable boundary visualization.
    Generated once and saved as a CSV. Never modified after creation.
    """
    S_levels     = np.linspace(S_MIN,     S_MAX,     5)
    mu_levels    = np.linspace(MU_MIN,    MU_MAX,    5)
    delta_levels = np.linspace(DELTA_MIN, DELTA_MAX, 5)

    conditions = []
    case_id = start_id
    for S in S_levels:
        for mu in mu_levels:
            for delta in delta_levels:
                conditions.append({
                    "case_id": case_id,
                    "S":       float(S),
                    "mu":      float(mu),
                    "delta":   float(delta),
                    "seed":    RANDOM_SEED * 2,   # different seed from training
                })
                case_id += 1
    return conditions


def run_testgrid(runner):
    print("=" * 60)
    print("PHASE 3: HOLD-OUT TEST GRID (125 conditions)")
    print("=" * 60)
    conditions = generate_test_grid(start_id=1000)
    df = runner.run_batch(conditions, phase_label="test_grid",
                          save_timeseries=True)
    return df


# =============================================================================
# Main entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run HDD multi-fidelity experiment phases."
    )
    parser.add_argument(
        "--phase", type=str, default="all",
        choices=["pilot", "main", "boundary", "testgrid", "all"],
        help="Which experiment phase to run."
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel worker processes."
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/results",
        help="Directory for output files."
    )
    args = parser.parse_args()

    runner = ExperimentRunner(
        output_dir=args.output_dir,
        n_workers=args.workers
    )

    if args.phase in ("pilot", "all"):
        run_pilot(runner)

    lhs_df = None
    if args.phase in ("main", "all"):
        lhs_df = run_main(runner, args.workers)

    if args.phase in ("boundary", "all"):
        if lhs_df is None:
            df_all = runner.load_dataset()
            lhs_df = df_all[df_all["phase"] == "lhs_main"]
        run_boundary(runner, lhs_df, args.workers)

    if args.phase in ("testgrid", "all"):
        run_testgrid(runner)

    print("\nAll requested phases complete.")
    print(f"Dataset saved to: {runner.csv_path}")
    print(f"Time series saved to: {runner.hdf5_path}")


if __name__ == "__main__":
    main()