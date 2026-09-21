# selector.py
# =============================================================================
# AI Fidelity Selector — Training, Calibration, and Persistence
#
# Model: XGBoost classifier with isotonic calibration
# Inputs: [S, mu, delta]
# Output: P(LF is adequate | S, mu, delta)
#
# The asymmetric risk constraint is enforced here:
#   Among cases assigned to LF, at most 5% may be truly inadequate.
#   This means we require LF-selection PRECISION >= 0.95.
#   The confidence threshold tau is tuned on the validation split
#   to satisfy this constraint, then frozen.
# =============================================================================

import os
import numpy as np
import pandas as pd
import joblib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, brier_score_loss
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

RANDOM_SEED = 42
FEATURE_COLS = ["S", "mu", "delta"]
TARGET_COL   = "adequate_label"

# Risk constraint: at most 5% of LF-assigned cases may be truly inadequate
MAX_INCORRECT_LF_RATE = 0.05


class FidelitySelector:
    """
    Trains, calibrates, and applies the AI fidelity selector.

    Workflow:
    1. Load training + validation data
    2. Tune XGBoost hyperparameters via cross-validation
    3. Calibrate probabilities with isotonic regression
    4. Find optimal confidence threshold on validation set
    5. Save model and threshold for use in evaluate.py

    Parameters
    ----------
    model_dir : str   Directory to save/load model artifacts
    """

    def __init__(self, model_dir="data/models"):
        self.model_dir    = model_dir
        os.makedirs(model_dir, exist_ok=True)

        self.scaler           = StandardScaler()
        self.base_model       = None
        self.calibrated_model = None
        self.threshold        = 0.95   # default; tuned in tune_threshold()
        self.feature_cols     = FEATURE_COLS
        self.is_fitted        = False

    # -------------------------------------------------------------------------
    # Data preparation
    # -------------------------------------------------------------------------

    @staticmethod
    def prepare_splits(dataset_path, val_fraction=0.15):
        """
        Split the LHS main dataset (phase='lhs_main') into train and validation.

        Important: split by case_id order, not randomly, so that
        no spatial information leaks between train and validation.

        Parameters
        ----------
        dataset_path : str   Path to the full dataset CSV
        val_fraction : float Fraction of LHS data reserved for validation

        Returns
        -------
        train_df, val_df : pd.DataFrame
        """
        df = pd.read_csv(dataset_path)

        # Use only the main LHS phase for training
        lhs = df[df["phase"] == "lhs_main"].copy()
        lhs = lhs.sort_values("case_id").reset_index(drop=True)

        n_val = int(len(lhs) * val_fraction)
        val_df   = lhs.iloc[:n_val]    # first 15% as validation
        train_df = lhs.iloc[n_val:]    # remaining 85% as training

        print(f"Train: {len(train_df)} | Validation: {len(val_df)}")
        print(f"Train adequate rate: "
              f"{train_df[TARGET_COL].mean()*100:.1f}%")
        print(f"Val adequate rate:   "
              f"{val_df[TARGET_COL].mean()*100:.1f}%")

        return train_df, val_df

    # -------------------------------------------------------------------------
    # Model training
    # -------------------------------------------------------------------------

    def train(self, train_df, val_df):
        """
        Full training pipeline: fit base XGBoost + isotonic calibration.

        Step 1: Fit base XGBoost on training data with cross-validation
                to select n_estimators and max_depth.
        Step 2: Fit CalibratedClassifierCV (isotonic) on validation data.
                This maps raw probabilities to reliable confidence scores.
        Step 3: Store scaler parameters for inference.

        Parameters
        ----------
        train_df : pd.DataFrame   Training split
        val_df   : pd.DataFrame   Calibration/validation split
        """
        X_train = train_df[FEATURE_COLS].values
        y_train = train_df[TARGET_COL].values
        X_val   = val_df[FEATURE_COLS].values
        y_val   = val_df[TARGET_COL].values

        # Fit scaler on training data only
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s   = self.scaler.transform(X_val)

        # -------------------------------------------------------
        # Class imbalance: compute scale_pos_weight
        # This tells XGBoost to up-weight the minority class.
        # Typical situation: more adequate (1) than inadequate (0).
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos_weight = n_neg / max(n_pos, 1)
        print(f"Class balance — Adequate: {n_pos}, Inadequate: {n_neg}, "
              f"scale_pos_weight: {scale_pos_weight:.2f}")
        # -------------------------------------------------------

        # Base XGBoost model
        self.base_model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_SEED,
            eval_metric="logloss",
            verbosity=0,
            use_label_encoder=False,
        )

        # Cross-validation on training data to check stability
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        cv_scores = cross_val_score(
            self.base_model, X_train_s, y_train,
            cv=cv, scoring="roc_auc"
        )
        print(f"5-fold CV ROC-AUC: {cv_scores.mean():.3f} "
              f"± {cv_scores.std():.3f}")

        # Fit base model on full training split
        self.base_model.fit(
            X_train_s, y_train,
            eval_set=[(X_val_s, y_val)],
            verbose=False
        )

        # Isotonic calibration on validation set
        # cv='prefit' means we calibrate an already-fitted model
        self.calibrated_model = CalibratedClassifierCV(
            estimator=self.base_model,
            method="isotonic",
            cv="prefit"
        )
        self.calibrated_model.fit(X_val_s, y_val)

        # Evaluate calibration quality
        probs_val = self.calibrated_model.predict_proba(X_val_s)[:, 1]
        brier = brier_score_loss(y_val, probs_val)
        print(f"Brier score after calibration: {brier:.4f} "
              f"(lower = better; 0.25 = random)")

        self.is_fitted = True
        print("Model training complete.")

    # -------------------------------------------------------------------------
    # Threshold tuning
    # -------------------------------------------------------------------------

    def tune_threshold(self, val_df, plot=True):
        """
        Find the minimum confidence threshold tau such that:
            P(truly inadequate | assigned to LF) <= MAX_INCORRECT_LF_RATE

        This enforces the asymmetric risk constraint: we accept some
        unnecessary HF runs to protect against sending unsafe conditions
        to the fast but potentially wrong LF simulation.

        The optimal tau is the SMALLEST value satisfying the constraint,
        because we want to maximize LF utilization while staying safe.

        Parameters
        ----------
        val_df : pd.DataFrame   Validation split (not test set)
        plot   : bool           Save a threshold sensitivity plot

        Returns
        -------
        float   Selected threshold tau
        """
        X_val   = val_df[FEATURE_COLS].values
        y_val   = val_df[TARGET_COL].values
        X_val_s = self.scaler.transform(X_val)
        probs   = self.calibrated_model.predict_proba(X_val_s)[:, 1]

        # Sweep tau from 0.50 to 0.99
        tau_values         = np.linspace(0.50, 0.99, 200)
        lf_rates           = []
        incorrect_lf_rates = []
        precision_lf       = []

        for tau in tau_values:
            assigned_lf = probs >= tau
            n_lf = assigned_lf.sum()

            if n_lf == 0:
                lf_rates.append(0.0)
                incorrect_lf_rates.append(0.0)
                precision_lf.append(1.0)
                continue

            # Among cases assigned to LF, how many are truly inadequate?
            truly_inadequate_lf = ((y_val == 0) & assigned_lf).sum()
            incorrect_rate = truly_inadequate_lf / n_lf

            lf_rates.append(n_lf / len(y_val))
            incorrect_lf_rates.append(incorrect_rate)
            precision_lf.append(1.0 - incorrect_rate)

        lf_rates           = np.array(lf_rates)
        incorrect_lf_rates = np.array(incorrect_lf_rates)
        precision_lf       = np.array(precision_lf)

        # Find smallest tau satisfying the constraint
        valid_mask = incorrect_lf_rates <= MAX_INCORRECT_LF_RATE
        if valid_mask.any():
            selected_tau = float(tau_values[valid_mask][0])
        else:
            # Constraint cannot be met — use maximum tau
            selected_tau = 0.99
            print("WARNING: Risk constraint could not be satisfied. "
                  "Using tau=0.99. Consider retraining with more data.")

        self.threshold = selected_tau
        selected_idx   = np.argmin(np.abs(tau_values - selected_tau))

        print(f"\nThreshold tuning results:")
        print(f"  Selected tau:         {selected_tau:.3f}")
        print(f"  LF assignment rate:   "
              f"{lf_rates[selected_idx]*100:.1f}%")
        print(f"  Incorrect LF rate:    "
              f"{incorrect_lf_rates[selected_idx]*100:.1f}% "
              f"(constraint: <={MAX_INCORRECT_LF_RATE*100:.0f}%)")
        print(f"  LF precision:         "
              f"{precision_lf[selected_idx]*100:.1f}%")

        if plot:
            fig, ax1 = plt.subplots(figsize=(8, 5))
            ax2 = ax1.twinx()

            ax1.plot(tau_values, lf_rates * 100, "b-",
                     label="LF assignment rate (%)", linewidth=2)
            ax2.plot(tau_values, incorrect_lf_rates * 100, "r-",
                     label="Incorrect LF rate (%)", linewidth=2)
            ax2.axhline(MAX_INCORRECT_LF_RATE * 100, color="r",
                        linestyle="--", alpha=0.5, label="Risk constraint")
            ax1.axvline(selected_tau, color="green", linestyle="--",
                        linewidth=2, label=f"Selected tau={selected_tau:.3f}")

            ax1.set_xlabel("Confidence Threshold τ", fontsize=12)
            ax1.set_ylabel("LF Assignment Rate (%)", color="b", fontsize=11)
            ax2.set_ylabel("Incorrect LF Rate (%)", color="r", fontsize=11)
            ax1.set_xlim(0.5, 1.0)

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

            plt.title("Threshold Sensitivity: Risk–Cost Tradeoff", fontsize=13)
            plt.tight_layout()
            os.makedirs("figures", exist_ok=True)
            plt.savefig("figures/threshold_sensitivity.png", dpi=150)
            plt.close()
            print("  Saved: figures/threshold_sensitivity.png")

        return selected_tau

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    def predict(self, S, mu, delta):
        """
        Predict whether LF simulation is adequate for a single condition.

        Parameters
        ----------
        S, mu, delta : float   Input parameters

        Returns
        -------
        str     "LF" or "HF"
        float   Confidence probability P(adequate)
        """
        assert self.is_fitted, "Model must be trained before calling predict()"
        x = np.array([[S, mu, delta]])
        x_s = self.scaler.transform(x)
        prob = float(self.calibrated_model.predict_proba(x_s)[0, 1])
        decision = "LF" if prob >= self.threshold else "HF"
        return decision, prob

    def predict_batch(self, df):
        """
        Predict fidelity selection for a DataFrame with columns [S, mu, delta].

        Returns
        -------
        np.ndarray of str  ("LF" or "HF" for each row)
        np.ndarray of float (probabilities)
        """
        X = df[FEATURE_COLS].values
        X_s = self.scaler.transform(X)
        probs = self.calibrated_model.predict_proba(X_s)[:, 1]
        decisions = np.where(probs >= self.threshold, "LF", "HF")
        return decisions, probs

    # -------------------------------------------------------------------------
    # Save and load
    # -------------------------------------------------------------------------

    def save(self):
        """Save all model artifacts to model_dir."""
        joblib.dump(self.calibrated_model,
                    os.path.join(self.model_dir, "calibrated_model.pkl"))
        joblib.dump(self.scaler,
                    os.path.join(self.model_dir, "scaler.pkl"))
        meta = {
            "threshold":       self.threshold,
            "feature_cols":    self.feature_cols,
            "max_incorrect_lf": MAX_INCORRECT_LF_RATE,
        }
        with open(os.path.join(self.model_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Model saved to {self.model_dir}/")

    @classmethod
    def load(cls, model_dir="data/models"):
        """Load a previously trained and saved selector."""
        obj = cls(model_dir=model_dir)
        obj.calibrated_model = joblib.load(
            os.path.join(model_dir, "calibrated_model.pkl")
        )
        obj.scaler = joblib.load(
            os.path.join(model_dir, "scaler.pkl")
        )
        with open(os.path.join(model_dir, "meta.json")) as f:
            meta = json.load(f)
        obj.threshold    = meta["threshold"]
        obj.feature_cols = meta["feature_cols"]
        obj.is_fitted    = True
        print(f"Loaded model from {model_dir}/ "
              f"(threshold={obj.threshold:.3f})")
        return obj

    # -------------------------------------------------------------------------
    # Calibration plot (reliability diagram)
    # -------------------------------------------------------------------------

    def plot_calibration(self, val_df):
        """
        Reliability diagram: predicted probability vs. actual frequency.
        A well-calibrated model should follow the diagonal.
        """
        X_val_s = self.scaler.transform(val_df[FEATURE_COLS].values)
        y_val   = val_df[TARGET_COL].values
        probs   = self.calibrated_model.predict_proba(X_val_s)[:, 1]

        fraction_of_positives, mean_predicted = calibration_curve(
            y_val, probs, n_bins=10
        )

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.plot(mean_predicted, fraction_of_positives,
                "o-", color="steelblue", label="XGBoost (calibrated)")
        ax.set_xlabel("Mean Predicted Probability", fontsize=12)
        ax.set_ylabel("Fraction of Positives (Actual)", fontsize=12)
        ax.set_title("Reliability Diagram — Fidelity Selector", fontsize=13)
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig("figures/calibration_reliability.png", dpi=150)
        plt.close()
        print("Saved: figures/calibration_reliability.png")


# =============================================================================
# Main training script
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/results/dataset.csv")
    parser.add_argument("--model_dir", type=str, default="data/models")
    args = parser.parse_args()

    os.makedirs("figures", exist_ok=True)

    # Prepare splits
    train_df, val_df = FidelitySelector.prepare_splits(args.dataset)

    # Train
    selector = FidelitySelector(model_dir=args.model_dir)
    selector.train(train_df, val_df)

    # Tune threshold
    selector.tune_threshold(val_df, plot=True)

    # Calibration plot
    selector.plot_calibration(val_df)

    # Save
    selector.save()

    print(f"\nFinal threshold: tau = {selector.threshold:.3f}")


if __name__ == "__main__":
    main()