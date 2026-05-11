"""Clean manually collected auxiliary WNBA data files."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd


def data_dir() -> Path:
    default_dir = Path(__file__).resolve().parents[2] / "sample_data"
    return Path(os.environ.get("WNBA_DATA_DIR", default_dir))


def clean_column_name(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "column"


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def clean_table(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean.columns = [clean_column_name(col) for col in clean.columns]
    return clean.dropna(how="all")


def main() -> None:
    base_dir = data_dir()
    input_dir = base_dir / "other"
    if not input_dir.exists():
        input_dir = base_dir

    output_dir = base_dir / "processed" / "other_clean"
    output_dir.mkdir(parents=True, exist_ok=True)

    files = [
        path for path in input_dir.iterdir()
        if path.suffix.lower() in {".csv", ".xls", ".xlsx"}
    ]
    if not files:
        raise FileNotFoundError("No auxiliary CSV/XLSX files found.")

    for path in files:
        clean = clean_table(read_table(path))
        output_file = output_dir / f"{path.stem}_clean.csv"
        clean.to_csv(output_file, index=False)
        print(f"Wrote {output_file}")


if __name__ == "__main__":
    main()
