from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PROCESSED = Path("data/processed")
FEATURES_PATH = DATA_PROCESSED / "features_game.csv"


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


def load_model(feature_cols: list[str]) -> Pipeline:
    df = pd.read_csv(FEATURES_PATH, parse_dates=["game_date"])
    target = "home_team_win"
    model = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(feature_cols)),
            ("model", LogisticRegression(max_iter=1000, n_jobs=None)),
        ]
    )
    model.fit(df[feature_cols], df[target])
    return model


def get_latest_team_features() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_PATH, parse_dates=["game_date"])

    exclude_home = {"team_id_home"}
    exclude_away = {"team_id_away"}
    home_cols = [
        col for col in df.columns if col.endswith("_home") and col not in exclude_home
    ]
    away_cols = [
        col for col in df.columns if col.endswith("_away") and col not in exclude_away
    ]
    home_map = {col: col[:-5] for col in home_cols}
    away_map = {col: col[:-5] for col in away_cols}

    home_df = df[["game_id", "game_date", "team_id_home", "team_id_away"] + home_cols].copy()
    home_df = home_df.rename(
        columns={
            "team_id_home": "team_id",
            "team_id_away": "opponent_id",
            **home_map,
        }
    )
    home_df["is_home"] = 1

    away_df = df[["game_id", "game_date", "team_id_away", "team_id_home"] + away_cols].copy()
    away_df = away_df.rename(
        columns={
            "team_id_away": "team_id",
            "team_id_home": "opponent_id",
            **away_map,
        }
    )
    away_df["is_home"] = 0

    long_df = pd.concat([home_df, away_df], ignore_index=True)
    long_df = long_df.sort_values(["team_id", "game_date", "game_id"])
    latest = long_df.groupby("team_id").tail(1)
    return latest


def build_matchup_features(
    home_row: pd.Series,
    away_row: pd.Series,
    feature_cols: list[str],
) -> pd.DataFrame:
    feature_values = {}
    for col in feature_cols:
        base = col.replace("diff_", "")
        feature_values[col] = home_row[base] - away_row[base]
    return pd.DataFrame([feature_values])


st.set_page_config(page_title="NBA Win Predictor", layout="wide")

st.title("NBA Win Predictor")
st.write("Select two teams to estimate the home win probability.")

latest = get_latest_team_features()
teams = latest[["team_id"]].drop_duplicates()
team_ids = teams["team_id"].astype(int).tolist()

home_team_id = st.selectbox("Home team", team_ids, format_func=lambda x: str(x))
away_team_id = st.selectbox("Away team", team_ids, format_func=lambda x: str(x))

if home_team_id == away_team_id:
    st.warning("Home and away teams must be different.")
    st.stop()

home_row = latest.loc[latest["team_id"] == home_team_id].iloc[0]
away_row = latest.loc[latest["team_id"] == away_team_id].iloc[0]

feature_cols = [
    col for col in pd.read_csv(FEATURES_PATH, nrows=1).columns if col.startswith("diff_")
]
model = load_model(feature_cols)
match_features = build_matchup_features(home_row, away_row, feature_cols)
proba = model.predict_proba(match_features)[0][1]

st.subheader("Prediction")
st.metric("Home win probability", f"{proba:.1%}")

show_details = st.checkbox("Show team features")
if show_details:
    st.subheader("Latest rolling features")
    st.dataframe(
        pd.DataFrame(
            {
                "home_team_id": [home_team_id],
                "away_team_id": [away_team_id],
                "home_elo": [home_row["elo_pre"]],
                "away_elo": [away_row["elo_pre"]],
                "home_rest_days": [home_row["rest_days"]],
                "away_rest_days": [away_row["rest_days"]],
            }
        )
    )

st.subheader("Top feature impact (LogReg coef)")
coeffs = pd.read_csv(Path("reports") / "logreg_coefficients.csv")
st.dataframe(coeffs.head(10))
