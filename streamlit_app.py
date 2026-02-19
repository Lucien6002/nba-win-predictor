from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from test_data import (
    add_elo_features,
    add_rolling_features,
    build_long_table,
)

DATA_RAW = Path("data/raw")
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
    games = pd.read_csv(DATA_RAW / "game.csv")
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")
    games = games.dropna(subset=["game_date"])

    base_stats = [
        "pts",
        "reb",
        "ast",
        "tov",
        "fg_pct",
        "fg3_pct",
        "ft_pct",
        "oreb",
        "dreb",
        "fgm",
        "fga",
        "fg3m",
        "fg3a",
        "ftm",
        "fta",
        "stl",
        "blk",
        "pf",
    ]
    base_stats = [
        stat
        for stat in base_stats
        if f"{stat}_home" in games.columns and f"{stat}_away" in games.columns
    ]

    games = add_elo_features(games)
    extra_cols = [("elo_home_pre", "elo_away_pre", "elo_pre")]
    long_df = build_long_table(games, base_stats, extra_cols=extra_cols)
    long_df = add_rolling_features(long_df, base_stats)

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
team_lookup = (
    latest[["team_id", "opponent_id"]]
    .assign(team_id=latest["team_id"].astype(int))
    .drop_duplicates()
)

teams = latest[["team_id"]].drop_duplicates()
team_ids = teams["team_id"].astype(int).tolist()
team_labels = [str(team_id) for team_id in team_ids]

home_team_id = st.selectbox("Home team", team_ids, format_func=lambda x: str(x))
away_team_id = st.selectbox("Away team", team_ids, format_func=lambda x: str(x))

if home_team_id == away_team_id:
    st.warning("Home and away teams must be different.")
    st.stop()

home_row = latest.loc[latest["team_id"] == home_team_id].iloc[0]
away_row = latest.loc[latest["team_id"] == away_team_id].iloc[0]

feature_cols = [col for col in pd.read_csv(FEATURES_PATH, nrows=1).columns if col.startswith("diff_")]
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
