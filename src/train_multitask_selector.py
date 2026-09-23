import json
import os

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier


DATASET = "data/multitask_dataset.csv"
MODEL_DIR = "data/multitask_model"


def main():
    data = pd.read_csv(DATASET)

    features = [
        "task",
        "strength",
        "friction",
        "misalignment",
    ]

    X = data[features]
    y = data["adequate"]

    X_train, X_temporary, y_train, y_temporary = (
        train_test_split(
            X,
            y,
            test_size=0.30,
            stratify=y,
            random_state=42,
        )
    )

    X_validation, X_test, y_validation, y_test = (
        train_test_split(
            X_temporary,
            y_temporary,
            test_size=0.50,
            stratify=y_temporary,
            random_state=42,
        )
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "task",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                ["task"],
            ),
            (
                "physical",
                StandardScaler(),
                [
                    "strength",
                    "friction",
                    "misalignment",
                ],
            ),
        ]
    )

    base_classifier = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", base_classifier),
    ])

    pipeline.fit(X_train, y_train)

    calibrated = CalibratedClassifierCV(
        estimator=pipeline,
        method="sigmoid",
        cv="prefit",
    )
    calibrated.fit(X_validation, y_validation)

    validation_probability = (
        calibrated.predict_proba(X_validation)[:, 1]
    )

    selected_threshold = 0.99

    for threshold_integer in range(50, 100):
        threshold = threshold_integer / 100.0
        select_lf = (
            validation_probability >= threshold
        )

        if select_lf.sum() == 0:
            continue

        incorrect_lf_rate = (
            (
                (y_validation.to_numpy() == 0)
                & select_lf
            ).sum()
            / select_lf.sum()
        )

        if incorrect_lf_rate <= 0.05:
            selected_threshold = threshold
            break

    test_probability = (
        calibrated.predict_proba(X_test)[:, 1]
    )
    test_prediction = (
        test_probability >= selected_threshold
    ).astype(int)

    print(
        classification_report(
            y_test,
            test_prediction,
        )
    )
    print(
        "Test ROC-AUC:",
        roc_auc_score(
            y_test,
            test_probability,
        ),
    )
    print(
        "Selected LF threshold:",
        selected_threshold,
    )

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(
        calibrated,
        os.path.join(
            MODEL_DIR,
            "selector.joblib",
        ),
    )

    with open(
        os.path.join(
            MODEL_DIR,
            "metadata.json",
        ),
        "w",
    ) as metadata_file:
        json.dump(
            {
                "threshold": selected_threshold,
                "features": features,
            },
            metadata_file,
            indent=2,
        )


if __name__ == "__main__":
    main()