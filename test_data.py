
from pathlib import Path

import numpy as np
import pandas as pd

DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")
MIN_DATE = pd.Timestamp("2021-01-01")
ELO_K = 20
ELO_HOME_BONUS = 65


def compute_win_streak(wins: pd.Series) -> pd.Series:
	streaks = np.zeros(len(wins), dtype=int)
	current = 0
	for i, win in enumerate(wins):
		streaks[i] = current
		current = current + 1 if win == 1 else 0
	return pd.Series(streaks, index=wins.index)


def add_elo_features(games: pd.DataFrame) -> pd.DataFrame:
	games = games.sort_values(["game_date", "game_id"]).copy()
	elo_ratings: dict[int, float] = {}
	elo_home_pre = []
	elo_away_pre = []

	for _, row in games.iterrows():
		home_id = int(row["team_id_home"])
		away_id = int(row["team_id_away"])
		home_elo = elo_ratings.get(home_id, 1500.0)
		away_elo = elo_ratings.get(away_id, 1500.0)

		home_expected = 1 / (1 + 10 ** (((away_elo) - (home_elo + ELO_HOME_BONUS)) / 400))
		away_expected = 1 - home_expected
		home_win = 1 if row["wl_home"] == "W" else 0

		home_elo_new = home_elo + ELO_K * (home_win - home_expected)
		away_elo_new = away_elo + ELO_K * ((1 - home_win) - away_expected)

		elo_home_pre.append(home_elo)
		elo_away_pre.append(away_elo)

		elo_ratings[home_id] = home_elo_new
		elo_ratings[away_id] = away_elo_new

	games["elo_home_pre"] = elo_home_pre
	games["elo_away_pre"] = elo_away_pre
	return games


def build_long_table(
	games: pd.DataFrame,
	base_stats: list[str],
	extra_cols: list[tuple[str, str, str]] | None = None,
) -> pd.DataFrame:
	home_cols = [f"{stat}_home" for stat in base_stats]
	away_cols = [f"{stat}_away" for stat in base_stats]

	extra_cols = extra_cols or []
	home_extra = [home_col for home_col, _, _ in extra_cols]
	away_extra = [away_col for _, away_col, _ in extra_cols]

	home_df = games[
		[
			"game_id",
			"game_date",
			"season_id",
			"season_type",
			"team_id_home",
			"team_id_away",
			"wl_home",
		]
		+ home_cols
		+ home_extra
	].copy()
	home_df = home_df.rename(
		columns={
			"team_id_home": "team_id",
			"team_id_away": "opponent_id",
			"wl_home": "wl",
		}
	)
	home_df["is_home"] = 1
	home_df = home_df.rename(columns={f"{stat}_home": stat for stat in base_stats})
	for home_col, _, base_name in extra_cols:
		home_df = home_df.rename(columns={home_col: base_name})

	away_df = games[
		[
			"game_id",
			"game_date",
			"season_id",
			"season_type",
			"team_id_away",
			"team_id_home",
			"wl_away",
		]
		+ away_cols
		+ away_extra
	].copy()
	away_df = away_df.rename(
		columns={
			"team_id_away": "team_id",
			"team_id_home": "opponent_id",
			"wl_away": "wl",
		}
	)
	away_df["is_home"] = 0
	away_df = away_df.rename(columns={f"{stat}_away": stat for stat in base_stats})
	for _, away_col, base_name in extra_cols:
		away_df = away_df.rename(columns={away_col: base_name})

	return pd.concat([home_df, away_df], ignore_index=True)


def add_rolling_features(long_df: pd.DataFrame, base_stats: list[str]) -> pd.DataFrame:
	long_df = long_df.sort_values(["team_id", "game_date", "game_id"]).reset_index(drop=True)
	long_df["win"] = (long_df["wl"] == "W").astype(int)

	for stat in base_stats:
		long_df[f"{stat}_rolling5"] = (
			long_df.groupby("team_id")[stat]
			.transform(lambda s: s.shift(1).rolling(5, min_periods=5).mean())
		)

	long_df["win_rate_5"] = (
		long_df.groupby("team_id")["win"]
		.transform(lambda s: s.shift(1).rolling(5, min_periods=5).mean())
	)
	long_df["win_streak"] = long_df.groupby("team_id")["win"].transform(compute_win_streak)

	long_df["rest_days"] = long_df.groupby("team_id")["game_date"].diff().dt.days
	long_df["back_to_back"] = (long_df["rest_days"] == 1).astype(int)
	return long_df



def build_match_level_features(long_df: pd.DataFrame, base_stats: list[str]) -> pd.DataFrame:
	rolling_cols = [f"{stat}_rolling5" for stat in base_stats]
	context_cols = ["win_rate_5", "win_streak", "rest_days", "back_to_back", "elo_pre"]
	feature_cols = rolling_cols + context_cols

	home_df = long_df[long_df["is_home"] == 1].copy()
	away_df = long_df[long_df["is_home"] == 0].copy()

	merged = home_df.merge(
		away_df,
		on="game_id",
		suffixes=("_home", "_away"),
		how="inner",
	)

	merged["home_team_win"] = (merged["wl_home"] == "W").astype(int)
	merged["home_advantage"] = 1

	for col in feature_cols:
		merged[f"diff_{col}"] = merged[f"{col}_home"] - merged[f"{col}_away"]

	keep_cols = [
		"game_id",
		"game_date_home",
		"season_id_home",
		"team_id_home",
		"team_id_away",
		"home_team_win",
		"home_advantage",
	]
	keep_cols += [f"{col}_home" for col in feature_cols]
	keep_cols += [f"{col}_away" for col in feature_cols]
	keep_cols += [f"diff_{col}" for col in feature_cols]

	final_df = merged[keep_cols].rename(
		columns={
			"game_date_home": "game_date",
			"season_id_home": "season_id",
		}
	)
	return final_df


def main() -> None:
	DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

	games = pd.read_csv(DATA_RAW / "game.csv")
	games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")
	games = games.dropna(subset=["game_date"])
	games = games[games["game_date"] >= MIN_DATE].copy()

	if "season_type" in games.columns:
		games = games[games["season_type"] == "Regular Season"].copy()

	base_stats = ["pts","reb","ast","tov","fg_pct","fg3_pct","ft_pct","oreb",
		"dreb","fgm","fga","fg3m","fg3a","ftm","fta","stl","blk","pf",]
	base_stats = [
		stat
		for stat in base_stats
		if f"{stat}_home" in games.columns and f"{stat}_away" in games.columns
	]

	games = add_elo_features(games)
	extra_cols = [("elo_home_pre", "elo_away_pre", "elo_pre")]
	long_df = build_long_table(games, base_stats, extra_cols=extra_cols)
	long_df = add_rolling_features(long_df, base_stats)

	match_df = build_match_level_features(long_df, base_stats)

	key_feature = "pts_rolling5_home"
	match_df = match_df.dropna(subset=[key_feature])

	output_path = DATA_PROCESSED / "features_game.csv"
	match_df.to_csv(output_path, index=False)

	print("Pipeline terminee.")
	print(f"Fichier genere : {output_path}")
	print(f"Lignes : {match_df.shape[0]}, Colonnes : {match_df.shape[1]}")


if __name__ == "__main__":
	main()