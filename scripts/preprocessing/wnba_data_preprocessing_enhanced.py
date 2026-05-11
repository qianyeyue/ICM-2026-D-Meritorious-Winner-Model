"""Small preprocessing runner for public data releases."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def data_dir() -> Path:
    default_dir = Path(__file__).resolve().parents[2] / "sample_data"
    return Path(os.environ.get("WNBA_DATA_DIR", default_dir))


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean.columns = [str(col).strip().lower().replace(" ", "_") for col in clean.columns]
    return clean


def process_known_samples(base_dir: Path) -> None:
    output_dir = base_dir / "processed" / "clean"
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = {
        "sample_games.csv": "games_clean.csv",
        "sample_players.csv": "players_clean.csv",
        "sample_teams.csv": "teams_clean.csv",
    }

    wrote_any = False
    for source_name, output_name in samples.items():
        df = read_csv_if_exists(base_dir / source_name)
        if df is None:
            continue
        normalize_columns(df).to_csv(output_dir / output_name, index=False)
        print(f"Wrote {output_dir / output_name}")
        wrote_any = True

    if not wrote_any:
        raise FileNotFoundError("No sample CSV files found. Set WNBA_DATA_DIR or add sample data.")


def main() -> None:
    process_known_samples(data_dir())


if __name__ == "__main__":
    main()
