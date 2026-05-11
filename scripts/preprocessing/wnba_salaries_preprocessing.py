"""Clean WNBA salary data and write processed salary tables."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd


def data_dir() -> Path:
    default_dir = Path(__file__).resolve().parents[2] / "sample_data"
    return Path(os.environ.get("WNBA_DATA_DIR", default_dir))


def find_input_file(base_dir: Path) -> Path:
    candidates = [
        base_dir / "raw" / "wnba_salaries.csv",
        base_dir / "wnba_salaries.csv",
        base_dir / "sample_players.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("No salary CSV found. Set WNBA_DATA_DIR or add sample_players.csv.")


def parse_money(value: object) -> float | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = re.sub(r"[^0-9.\-]", "", str(value))
    if not text:
        return None
    return float(text)


def clean_salaries(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean.columns = [str(col).strip().lower().replace(" ", "_") for col in clean.columns]

    salary_cols = [col for col in clean.columns if "salary" in col]
    for col in salary_cols:
        clean[col] = clean[col].map(parse_money)

    if "team" in clean.columns and "team_code" not in clean.columns:
        clean = clean.rename(columns={"team": "team_code"})

    return clean


def main() -> None:
    base_dir = data_dir()
    input_file = find_input_file(base_dir)
    output_dir = base_dir / "processed" / "clean"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_file)
    clean = clean_salaries(df)
    clean.to_csv(output_dir / "salaries_clean.csv", index=False)

    print(f"Read {input_file}")
    print(f"Wrote {output_dir / 'salaries_clean.csv'}")


if __name__ == "__main__":
    main()
