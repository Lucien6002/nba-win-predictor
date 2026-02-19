
from pathlib import Path

import numpy as np
import pandas as pd

try:
	from scipy.stats import ttest_ind
except ImportError:
	ttest_ind = None

DATA_PROCESSED = Path("data/processed")
FEATURES_PATH = DATA_PROCESSED / "features_game.csv"
OUTPUT_DIR = Path("reports")


def safe_ttest(group_a: pd.Series, group_b: pd.Series) -> dict:
	if ttest_ind is None:
		return {"t_stat": np.nan, "p_value": np.nan}
	t_stat, p_value = ttest_ind(group_a, group_b, nan_policy="omit", equal_var=False)
	return {"t_stat": float(t_stat), "p_value": float(p_value)}


def main() -> None:
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	df = pd.read_csv(FEATURES_PATH, parse_dates=["game_date"])

	target = "home_team_win"
	feature_cols = [col for col in df.columns if col.startswith("diff_")]
	context_cols = [
		"diff_rest_days",
		"diff_win_rate_5",
		"diff_win_streak",
		"diff_back_to_back",
	]

	summary = {
		"rows": int(df.shape[0]),
		"cols": int(df.shape[1]),
		"date_min": str(df["game_date"].min()),
		"date_max": str(df["game_date"].max()),
		"home_win_rate": float(df[target].mean()),
	}
	pd.Series(summary).to_csv(OUTPUT_DIR / "eda_summary.csv", header=["value"])

	corr = df[feature_cols + [target]].corr(numeric_only=True)[target].sort_values(ascending=False)
	corr.to_csv(OUTPUT_DIR / "eda_correlations.csv", header=["corr_with_target"])

	ttests = []
	for col in context_cols:
		if col not in df.columns:
			continue
		home_win = df.loc[df[target] == 1, col]
		home_loss = df.loc[df[target] == 0, col]
		stats = safe_ttest(home_win, home_loss)
		ttests.append({"feature": col, **stats})

	pd.DataFrame(ttests).to_csv(OUTPUT_DIR / "eda_ttests.csv", index=False)

	print("EDA terminee.")
	print(f"Summary: {OUTPUT_DIR / 'eda_summary.csv'}")
	print(f"Correlations: {OUTPUT_DIR / 'eda_correlations.csv'}")
	print(f"T-tests: {OUTPUT_DIR / 'eda_ttests.csv'}")


if __name__ == "__main__":
	main()

