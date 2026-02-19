from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
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


def save_logreg_coefficients(model: Pipeline, feature_cols: list[str]) -> None:
    coefs = model.named_steps["model"].coef_.ravel()
    coef_df = pd.DataFrame({"feature": feature_cols, "coef": coefs})
    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df = coef_df.sort_values(by="abs_coef", ascending=False)
    coef_df.to_csv(OUTPUT_DIR / "logreg_coefficients.csv", index=False)


def save_permutation_importance(
    name: str, model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series, feature_cols: list[str]
) -> None:
    result = permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="roc_auc",
    )
    importances = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values(by="importance_mean", ascending=False)
    importances.to_csv(OUTPUT_DIR / f"{name}_permutation_importance.csv", index=False)


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

    preprocessor = build_preprocessor(feature_cols)

    log_reg = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=1000, n_jobs=None)),
        ]
    )
    log_reg.fit(x_train, y_train)
    save_logreg_coefficients(log_reg, feature_cols)
    save_permutation_importance("logreg", log_reg, x_test, y_test, feature_cols)

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
    save_permutation_importance("random_forest", rf, x_test, y_test, feature_cols)

    gb = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", GradientBoostingClassifier(random_state=42)),
        ]
    )
    gb.fit(x_train, y_train)
    save_permutation_importance("gradient_boosting", gb, x_test, y_test, feature_cols)

    print("Interpretabilite terminee.")
    print(f"LogReg coefficients: {OUTPUT_DIR / 'logreg_coefficients.csv'}")
    print(f"LogReg permutation: {OUTPUT_DIR / 'logreg_permutation_importance.csv'}")
    print(f"RF permutation: {OUTPUT_DIR / 'random_forest_permutation_importance.csv'}")
    print(f"GB permutation: {OUTPUT_DIR / 'gradient_boosting_permutation_importance.csv'}")


if __name__ == "__main__":
    main()
