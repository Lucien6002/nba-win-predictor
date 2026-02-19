from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

DATA_PROCESSED = Path("data/processed")
FEATURES_PATH = DATA_PROCESSED / "features_game.csv"
OUTPUT_DIR = Path("reports")
SPLIT_DATE = pd.Timestamp("2022-07-01")


def build_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[("num", numeric_pipeline, feature_cols)],
        remainder="drop",
    )


def evaluate_model(name: str, y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FEATURES_PATH, parse_dates=["game_date"])

    target = "home_team_win"
    feature_cols = [col for col in df.columns if col.startswith("diff_")]

    train_df = df[df["game_date"] < SPLIT_DATE].copy()
    test_df = df[df["game_date"] >= SPLIT_DATE].copy()

    x_train = train_df[feature_cols]
    y_train = train_df[target]
    x_test = test_df[feature_cols]
    y_test = test_df[target]

    results = []

    baseline_pred = np.ones_like(y_test)
    baseline_proba = np.full_like(y_test, y_test.mean(), dtype=float)
    results.append(evaluate_model("baseline_home_win", y_test, baseline_pred, baseline_proba))

    preprocessor = build_preprocessor(feature_cols)

    log_reg = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=1000, n_jobs=None)),
        ]
    )
    log_reg.fit(x_train, y_train)
    log_pred = log_reg.predict(x_test)
    log_proba = log_reg.predict_proba(x_test)[:, 1]
    results.append(evaluate_model("logistic_regression", y_test, log_pred, log_proba))

    rf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_split=2,
                    random_state=42,
                ),
            ),
        ]
    )
    rf.fit(x_train, y_train)
    rf_pred = rf.predict(x_test)
    rf_proba = rf.predict_proba(x_test)[:, 1]
    results.append(evaluate_model("random_forest", y_test, rf_pred, rf_proba))

    gb = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", GradientBoostingClassifier(random_state=42)),
        ]
    )
    gb.fit(x_train, y_train)
    gb_pred = gb.predict(x_test)
    gb_proba = gb.predict_proba(x_test)[:, 1]
    results.append(evaluate_model("gradient_boosting", y_test, gb_pred, gb_proba))

    results_df = pd.DataFrame(results).sort_values(by="roc_auc", ascending=False)
    results_df.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

    conf_matrix = confusion_matrix(y_test, log_pred)
    conf_df = pd.DataFrame(conf_matrix, index=["actual_0", "actual_1"], columns=["pred_0", "pred_1"])
    conf_df.to_csv(OUTPUT_DIR / "logreg_confusion_matrix.csv")

    report = classification_report(y_test, log_pred, zero_division=0, output_dict=False)
    with open(OUTPUT_DIR / "logreg_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print("Modelisation terminee.")
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    print(f"Metrics: {OUTPUT_DIR / 'model_metrics.csv'}")
    print(f"Confusion matrix: {OUTPUT_DIR / 'logreg_confusion_matrix.csv'}")
    print(f"Classification report: {OUTPUT_DIR / 'logreg_classification_report.txt'}")


if __name__ == "__main__":
    main()
