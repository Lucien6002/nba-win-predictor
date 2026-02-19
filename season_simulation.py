from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")
FEATURES_PATH = DATA_PROCESSED / "features_game.csv"
OUTPUT_DIR = Path("reports")
TARGET_SEASON_END_YEAR = 2023
RANDOM_SEED = 42
MONTE_CARLO_RUNS = 100


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


def season_end_year(date: pd.Timestamp) -> int:
    if date.month >= 10:
        return date.year + 1
    return date.year


def load_team_lookup() -> dict[int, str]:
    games = pd.read_csv(DATA_RAW / "game.csv")
    columns = ["team_id_home", "team_abbreviation_home"]
    lookup = games[columns].dropna().drop_duplicates()
    return dict(zip(lookup["team_id_home"].astype(int), lookup["team_abbreviation_home"]))


def load_real_standings() -> pd.DataFrame:
    data = [
        {"team": "MIL", "wins": 58, "losses": 24},
        {"team": "BOS", "wins": 57, "losses": 25},
        {"team": "PHI", "wins": 54, "losses": 28},
        {"team": "CLE", "wins": 51, "losses": 31},
        {"team": "NYK", "wins": 47, "losses": 35},
        {"team": "BKN", "wins": 45, "losses": 37},
        {"team": "MIA", "wins": 44, "losses": 38},
        {"team": "ATL", "wins": 41, "losses": 41},
        {"team": "TOR", "wins": 41, "losses": 41},
        {"team": "CHI", "wins": 40, "losses": 42},
        {"team": "IND", "wins": 35, "losses": 47},
        {"team": "WAS", "wins": 35, "losses": 47},
        {"team": "ORL", "wins": 34, "losses": 48},
        {"team": "CHO", "wins": 27, "losses": 55},
        {"team": "DET", "wins": 17, "losses": 65},
        {"team": "DEN", "wins": 53, "losses": 29},
        {"team": "MEM", "wins": 51, "losses": 31},
        {"team": "SAC", "wins": 48, "losses": 34},
        {"team": "PHX", "wins": 45, "losses": 37},
        {"team": "LAC", "wins": 44, "losses": 38},
        {"team": "GSW", "wins": 44, "losses": 38},
        {"team": "LAL", "wins": 43, "losses": 39},
        {"team": "MIN", "wins": 42, "losses": 40},
        {"team": "NOP", "wins": 42, "losses": 40},
        {"team": "OKC", "wins": 40, "losses": 42},
        {"team": "DAL", "wins": 38, "losses": 44},
        {"team": "UTA", "wins": 37, "losses": 45},
        {"team": "POR", "wins": 33, "losses": 49},
        {"team": "HOU", "wins": 22, "losses": 60},
        {"team": "SAS", "wins": 22, "losses": 60},
    ]
    return pd.DataFrame(data)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(RANDOM_SEED)

    df = pd.read_csv(FEATURES_PATH, parse_dates=["game_date"])
    df["season_end_year"] = df["game_date"].apply(season_end_year)

    target = "home_team_win"
    feature_cols = [col for col in df.columns if col.startswith("diff_")]

    train_df = df[df["season_end_year"] < TARGET_SEASON_END_YEAR].copy()
    season_df = df[df["season_end_year"] == TARGET_SEASON_END_YEAR].copy()
    season_df = season_df.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    if season_df.empty:
        raise ValueError("No games found for the requested season.")

    preprocessor = build_preprocessor(feature_cols)
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=1000, n_jobs=None)),
        ]
    )
    model.fit(train_df[feature_cols], train_df[target])

    team_lookup = load_team_lookup()
    season_df["home_team"] = season_df["team_id_home"].map(team_lookup)
    season_df["away_team"] = season_df["team_id_away"].map(team_lookup)

    win_proba = model.predict_proba(season_df[feature_cols])[:, 1]
    season_df["home_win_proba"] = win_proba

    standings_runs = []
    games_snapshot = None
    for run_idx in range(MONTE_CARLO_RUNS):
        random_draws = np.random.rand(len(season_df))
        simulated_home_win = (random_draws < win_proba).astype(int)

        if run_idx == 0:
            games_snapshot = season_df.copy()
            games_snapshot["sim_home_win"] = simulated_home_win

        standings = {}
        for _, row in season_df.iterrows():
            home_id = int(row["team_id_home"])
            away_id = int(row["team_id_away"])
            standings.setdefault(home_id, {"wins": 0, "losses": 0})
            standings.setdefault(away_id, {"wins": 0, "losses": 0})

            home_win = simulated_home_win[row.name]
            if home_win == 1:
                standings[home_id]["wins"] += 1
                standings[away_id]["losses"] += 1
            else:
                standings[home_id]["losses"] += 1
                standings[away_id]["wins"] += 1

        standings_df = pd.DataFrame(
            [
                {
                    "team_id": team_id,
                    "team": team_lookup.get(team_id, str(team_id)),
                    "wins": record["wins"],
                    "losses": record["losses"],
                }
                for team_id, record in standings.items()
            ]
        )
        standings_df["run"] = run_idx + 1
        standings_runs.append(standings_df)

    all_runs = pd.concat(standings_runs, ignore_index=True)
    standings_summary = (
        all_runs.groupby(["team_id", "team"])[["wins", "losses"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    standings_summary.columns = [
        "team_id",
        "team",
        "wins_mean",
        "wins_std",
        "losses_mean",
        "losses_std",
    ]
    standings_summary = standings_summary.sort_values(
        ["wins_mean", "losses_mean"], ascending=[False, True]
    )

    real_df = load_real_standings()
    comparison = standings_summary.merge(real_df, on="team", how="left")
    comparison["win_diff"] = comparison["wins_mean"] - comparison["wins"]

    mae = float(np.mean(np.abs(comparison["win_diff"])))
    rmse = float(np.sqrt(np.mean(comparison["win_diff"] ** 2)))

    games_out = OUTPUT_DIR / f"season_simulation_{TARGET_SEASON_END_YEAR}_games.csv"
    standings_out = OUTPUT_DIR / f"season_simulation_{TARGET_SEASON_END_YEAR}_standings.csv"
    comparison_out = OUTPUT_DIR / f"season_simulation_{TARGET_SEASON_END_YEAR}_comparison.csv"
    metrics_out = OUTPUT_DIR / f"season_simulation_{TARGET_SEASON_END_YEAR}_metrics.csv"

    if games_snapshot is not None:
        games_snapshot[
            [
                "game_date",
                "game_id",
                "home_team",
                "away_team",
                "home_win_proba",
                "sim_home_win",
            ]
        ].to_csv(games_out, index=False)

    standings_summary.to_csv(standings_out, index=False)
    comparison.to_csv(comparison_out, index=False)

    pd.DataFrame(
        [{"metric": "mae", "value": mae}, {"metric": "rmse", "value": rmse}]
    ).to_csv(metrics_out, index=False)

    print("Season simulation terminee.")
    print(f"Games: {games_out}")
    print(f"Standings: {standings_out}")
    print(f"Comparison: {comparison_out}")
    print(f"Metrics: {metrics_out}")


if __name__ == "__main__":
    main()
