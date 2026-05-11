"""Clean WNBA game logs and write matchup features."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


TEAM_ALIASES = {
    "SAS": "LVA",
    "LAS": "LVA",
    "San Antonio": "LVA",
    "San Antonio Stars": "LVA",
    "TUL": "DAL",
    "Tulsa": "DAL",
    "Tulsa Shock": "DAL",
}


def data_dir() -> Path:
    default_dir = Path(__file__).resolve().parents[2] / "sample_data"
    return Path(os.environ.get("WNBA_DATA_DIR", default_dir))


def find_input_file(base_dir: Path) -> Path:
    candidates = [
        base_dir / "raw" / "wnba_gamelogs_2015_2025.csv",
        base_dir / "wnba_gamelogs_2015_2025.csv",
        base_dir / "sample_games.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("No game-log CSV found. Set WNBA_DATA_DIR or add sample_games.csv.")


def normalize_team(value: object) -> object:
    if pd.isna(value):
        return value
    text = str(value).strip()
    return TEAM_ALIASES.get(text, text)


def clean_gamelogs(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean.columns = [str(col).strip() for col in clean.columns]

    for col in ["Team", "team", "home_team", "away_team", "Opp", "opponent"]:
        if col in clean.columns:
            clean[col] = clean[col].map(normalize_team)

    for col in ["Date", "date", "game_date"]:
        if col in clean.columns:
            clean[col] = pd.to_datetime(clean[col], errors="coerce")

    return clean


def add_match_result(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    score_pairs = [
        ("home_score", "away_score"),
        ("HomeScore", "AwayScore"),
        ("PTS", "OppPTS"),
    ]
    for home_col, away_col in score_pairs:
        if home_col in clean.columns and away_col in clean.columns:
            clean["home_win"] = clean[home_col] > clean[away_col]
            break
    return clean


def main() -> None:
    base_dir = data_dir()
    input_file = find_input_file(base_dir)
    output_dir = base_dir / "processed" / "clean"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_file)
    clean = add_match_result(clean_gamelogs(df))
    clean.to_csv(output_dir / "wnba_gamelogs_clean.csv", index=False)

    print(f"Read {input_file}")
    print(f"Wrote {output_dir / 'wnba_gamelogs_clean.csv'}")


if __name__ == "__main__":
    main()
