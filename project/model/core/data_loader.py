"""
Centralized data loader for WNBA model modules.

This module provides a unified interface for loading data files with
consistent error handling and validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd

from .utils import load_csv_with_fallback, resolve_path


class DataLoader:
    """
    Centralized data loader with consistent interface.

    Handles path resolution, CSV loading, and data validation for all
    model modules.

    Parameters
    ----------
    root_dir : Path, optional
        Root directory for relative paths (defaults to repo root)
    processed_dir : Path, optional
        Processed data directory (defaults to project/data/processed)

    Examples
    --------
    >>> loader = DataLoader()
    >>> elo_ratings = loader.load_elo_ratings("elo_final_ratings.csv")
    >>> brand_b0 = loader.load_brand_b0("brand_b0.csv")
    """

    def __init__(
        self,
        root_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
    ):
        if root_dir is None:
            root_dir = Path(__file__).resolve().parents[3]
        self.root_dir = root_dir

        if processed_dir is None:
            processed_dir = root_dir / "project" / "data" / "processed"
        self.processed_dir = processed_dir

        self.data_dir = root_dir / "project" / "data"
        self.raw_dir = self.data_dir / "raw"
        self.external_dir = self.data_dir / "external"
        self.config_dir = processed_dir / "config"
        self.elo_dir = processed_dir / "elo"
        self.simulation_dir = processed_dir / "simulation"
        self.mpc_dir = processed_dir / "mpc"
        self.financial_dir = processed_dir / "financial"
        self.players_dir = processed_dir / "players"
        self.clean_dir = processed_dir / "clean"
        self.other_clean_dir = processed_dir / "other_clean"

    def load_elo_ratings(
        self,
        path: str | Path,
        teams: Optional[Sequence[str]] = None,
        base_elo: float = 1500.0,
    ) -> Dict[str, float]:
        """
        Load Elo ratings from CSV.

        Parameters
        ----------
        path : str | Path
            Path to elo_final_ratings.csv
        teams : Sequence[str], optional
            List of teams to load (if None, loads all)
        base_elo : float, default=1500.0
            Default Elo for missing teams

        Returns
        -------
        Dict[str, float]
            Mapping of team code to Elo rating

        Examples
        --------
        >>> loader = DataLoader()
        >>> elo_ratings = loader.load_elo_ratings("elo_final_ratings.csv")
        >>> elo_ratings["LVA"]
        1661.95...
        """
        df = load_csv_with_fallback(
            path,
            fallback_dirs=[self.elo_dir, self.processed_dir],
            root_dir=self.root_dir,
            required_columns=["team", "elo"],
        )

        df["team"] = df["team"].astype(str).str.strip()
        df["elo"] = pd.to_numeric(df["elo"], errors="coerce").fillna(base_elo)

        elo_dict = dict(zip(df["team"], df["elo"]))

        if teams is not None:
            # Ensure all requested teams are present
            return {t: float(elo_dict.get(t, base_elo)) for t in teams}

        return {k: float(v) for k, v in elo_dict.items()}

    def load_elo_config(self, path: str | Path) -> Dict[str, float]:
        """
        Load Elo configuration parameters from CSV.

        Parameters
        ----------
        path : str | Path
            Path to elo_config.csv

        Returns
        -------
        Dict[str, float]
            Configuration parameters (k, base_elo, home_advantage, etc.)

        Examples
        --------
        >>> loader = DataLoader()
        >>> config = loader.load_elo_config("elo_config.csv")
        >>> config["k"]
        20.0
        """
        df = load_csv_with_fallback(
            path,
            fallback_dirs=[self.config_dir, self.processed_dir],
            root_dir=self.root_dir,
        )

        if df.empty:
            return {}

        row = df.iloc[0].to_dict()
        out = {}

        for k in ("base_elo", "home_advantage", "k", "season_carryover", "home_win_prob"):
            if k in row and pd.notna(row[k]):
                out[k] = float(row[k])

        return out

    def load_brand_b0(self, path: str | Path) -> pd.DataFrame:
        """
        Load brand B0 values from CSV.

        Parameters
        ----------
        path : str | Path
            Path to brand_b0.csv

        Returns
        -------
        pd.DataFrame
            Brand data with columns: team, B0, and indicator columns

        Examples
        --------
        >>> loader = DataLoader()
        >>> brand_b0 = loader.load_brand_b0("brand_b0.csv")
        >>> brand_b0[["team", "B0"]].head()
        """
        df = load_csv_with_fallback(
            path,
            fallback_dirs=[self.config_dir, self.processed_dir],
            root_dir=self.root_dir,
            required_columns=["team"],
        )

        df["team"] = df["team"].astype(str).str.strip()

        return df

    def load_game_schedule(self, path: str | Path) -> pd.DataFrame:
        """
        Load game schedule from CSV.

        Parameters
        ----------
        path : str | Path
            Path to schedule CSV

        Returns
        -------
        pd.DataFrame
            Schedule with columns: game_number, date, home_team, away_team

        Examples
        --------
        >>> loader = DataLoader()
        >>> schedule = loader.load_game_schedule("2026_schedule.csv")
        """
        df = load_csv_with_fallback(
            path,
            fallback_dirs=[self.processed_dir],
            root_dir=self.root_dir,
            required_columns=["home_team", "away_team"],
        )

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        return df

    def load_team_gamelogs(self, path: str | Path) -> pd.DataFrame:
        """
        Load team-level game logs from CSV.

        Parameters
        ----------
        path : str | Path
            Path to game logs CSV

        Returns
        -------
        pd.DataFrame
            Game logs with columns: Season, Team, Date, Home, Opp, W/L, etc.

        Examples
        --------
        >>> loader = DataLoader()
        >>> gamelogs = loader.load_team_gamelogs("wnba_gamelogs_2015_2025.csv")
        """
        df = load_csv_with_fallback(
            path,
            fallback_dirs=[self.data_dir, self.processed_dir],
            root_dir=self.root_dir,
            required_columns=["Season", "Team", "Date", "Home", "Opp", "W/L"],
        )

        return df

    def load_player_data(self, path: str | Path) -> pd.DataFrame:
        """
        Load player data from CSV.

        Parameters
        ----------
        path : str | Path
            Path to player data CSV

        Returns
        -------
        pd.DataFrame
            Player data

        Examples
        --------
        >>> loader = DataLoader()
        >>> players = loader.load_player_data("players_pcv.csv")
        """
        df = load_csv_with_fallback(
            path,
            fallback_dirs=[self.processed_dir],
            root_dir=self.root_dir,
        )

        return df

    def load_salary_data(self, path: str | Path, year: int = 2025) -> pd.DataFrame:
        """
        Load salary data from CSV.

        Parameters
        ----------
        path : str | Path
            Path to salary CSV
        year : int, default=2025
            Year to filter (if Year column exists)

        Returns
        -------
        pd.DataFrame
            Salary data for specified year

        Examples
        --------
        >>> loader = DataLoader()
        >>> salaries = loader.load_salary_data("average-player-salary-in-the-wnba-by-team-2025_clean.csv")
        """
        df = load_csv_with_fallback(
            path,
            fallback_dirs=[self.other_clean_dir, self.processed_dir],
            root_dir=self.root_dir,
        )

        if "Year" in df.columns:
            df = df[df["Year"] == year].copy()

        return df

    def load_attendance_data(self, path: str | Path, year: int = 2025) -> pd.DataFrame:
        """
        Load attendance data from CSV.

        Parameters
        ----------
        path : str | Path
            Path to attendance CSV
        year : int, default=2025
            Year to filter

        Returns
        -------
        pd.DataFrame
            Attendance data for specified year

        Examples
        --------
        >>> loader = DataLoader()
        >>> attendance = loader.load_attendance_data("attendance_clean.csv")
        """
        df = load_csv_with_fallback(
            path,
            fallback_dirs=[self.clean_dir, self.processed_dir],
            root_dir=self.root_dir,
        )

        if "Year" in df.columns:
            df = df[df["Year"] == year].copy()

        return df

    def load_city_distances(self, path: str | Path) -> pd.DataFrame:
        """
        Load city distance matrix from CSV.

        Parameters
        ----------
        path : str | Path
            Path to city distances CSV

        Returns
        -------
        pd.DataFrame
            Distance matrix with columns: from_city, to_city, distance_km

        Examples
        --------
        >>> loader = DataLoader()
        >>> distances = loader.load_city_distances("wnba_city_distances.csv")
        >>> distances[distances["from_city"] == "Atlanta"]
        """
        df = load_csv_with_fallback(
            path,
            fallback_dirs=[self.processed_dir],
            root_dir=self.root_dir,
            required_columns=["from_city", "to_city", "distance_km"],
        )

        return df

    def load_simulation_results(self, path: str | Path) -> pd.DataFrame:
        """
        Load simulation results from CSV.

        Parameters
        ----------
        path : str | Path
            Path to simulation results CSV

        Returns
        -------
        pd.DataFrame
            Simulation results

        Examples
        --------
        >>> loader = DataLoader()
        >>> results = loader.load_simulation_results("2026_team_statistics.csv")
        """
        df = load_csv_with_fallback(
            path,
            fallback_dirs=[self.processed_dir],
            root_dir=self.root_dir,
        )

        return df

    def save_results(
        self,
        df: pd.DataFrame,
        filename: str,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """
        Save results to CSV.

        Parameters
        ----------
        df : pd.DataFrame
            Data to save
        filename : str
            Output filename
        output_dir : Path, optional
            Output directory (defaults to processed_dir)

        Returns
        -------
        Path
            Path to saved file

        Examples
        --------
        >>> loader = DataLoader()
        >>> path = loader.save_results(df, "my_results.csv")
        """
        if output_dir is None:
            output_dir = self.processed_dir

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / filename
        df.to_csv(output_path, index=False)

        return output_path


# Convenience function for quick loading
def load_elo_ratings(
    path: str | Path = "elo_final_ratings.csv",
    teams: Optional[Sequence[str]] = None,
    base_elo: float = 1500.0,
) -> Dict[str, float]:
    """
    Quick load Elo ratings (convenience function).

    Parameters
    ----------
    path : str | Path, default="elo_final_ratings.csv"
        Path to Elo ratings CSV
    teams : Sequence[str], optional
        List of teams to load
    base_elo : float, default=1500.0
        Default Elo for missing teams

    Returns
    -------
    Dict[str, float]
        Team to Elo mapping

    Examples
    --------
    >>> elo_ratings = load_elo_ratings()
    >>> elo_ratings["LVA"]
    1661.95...
    """
    loader = DataLoader()
    return loader.load_elo_ratings(path, teams=teams, base_elo=base_elo)


def load_elo_config(path: str | Path = "elo_config.csv") -> Dict[str, float]:
    """
    Quick load Elo config (convenience function).

    Parameters
    ----------
    path : str | Path, default="elo_config.csv"
        Path to Elo config CSV

    Returns
    -------
    Dict[str, float]
        Configuration parameters

    Examples
    --------
    >>> config = load_elo_config()
    >>> config["k"]
    20.0
    """
    loader = DataLoader()
    return loader.load_elo_config(path)


def load_brand_b0(path: str | Path = "brand_b0.csv") -> pd.DataFrame:
    """
    Quick load brand B0 (convenience function).

    Parameters
    ----------
    path : str | Path, default="brand_b0.csv"
        Path to brand B0 CSV

    Returns
    -------
    pd.DataFrame
        Brand data

    Examples
    --------
    >>> brand_b0 = load_brand_b0()
    >>> brand_b0[["team", "B0"]].head()
    """
    loader = DataLoader()
    return loader.load_brand_b0(path)
