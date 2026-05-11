"""
Load real data from project datasets for LVA analysis.

This module loads:
1. Real Elo ratings from game history
2. Real attendance data
3. Real revenue data
"""

from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Union

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "project" / "data"
PROCESSED_DIR = DATA_DIR / "processed" / "other_clean"


def _read_csv(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    path = Path(path)
    with path.open("r", encoding=kwargs.pop("encoding", "utf-8-sig"), newline="") as f:
        return pd.read_csv(f, **kwargs)


def load_real_elo_ratings(team_code: str = "LVA", season: int = 2024) -> float:
    """
    Load real Elo rating from game history.

    Args:
        team_code: Team abbreviation (e.g., "LVA")
        season: Season year

    Returns:
        Team's Elo rating
    """
    try:
        import sys
        from pathlib import Path

        # Add project root to path
        project_root = Path(__file__).resolve().parents[3]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from project.model.core.elo import EloModel, EloConfig, load_team_gamelogs, build_unique_games_from_gamelogs

        # Load game logs - try multiple paths
        gamelogs_paths = [
            DATA_DIR / "wnba_gamelogs_2015_2025.csv",
            DATA_DIR / "processed" / "clean" / "wnba_gamelogs_clean.csv",
            DATA_DIR / "wnba_gamelogs_clean.csv"
        ]

        gamelogs_path = None
        for path in gamelogs_paths:
            if path.exists():
                gamelogs_path = path
                break

        if gamelogs_path is None:
            print(f"[WARNING] Game logs not found, tried: {[str(p) for p in gamelogs_paths]}")
            return None

        # Load and process
        team_logs = load_team_gamelogs(gamelogs_path)
        games = build_unique_games_from_gamelogs(team_logs)

        # Filter to desired season
        games_season = games[games['season'] == season].copy()

        if games_season.empty:
            print(f"[WARNING] No games found for season {season}, using all available data")
            games_season = games.copy()

        # Fit Elo model
        config = EloConfig(
            k=20.0,
            base_elo=1500.0,
            season_carryover=0.75,
            home_advantage=100.0
        )

        model = EloModel(config)

        # Process games chronologically
        for _, game in games_season.iterrows():
            home_team = game['home_team']
            away_team = game['away_team']
            home_win = game['home_win']

            # Get current ratings
            home_elo = model._get_rating(home_team)
            away_elo = model._get_rating(away_team)

            # Predict
            home_prob = model.predict_home_win_prob(home_elo, away_elo)

            # Update
            home_new = home_elo + config.k * (home_win - home_prob)
            away_new = away_elo + config.k * ((1 - home_win) - (1 - home_prob))

            model._set_rating(home_team, home_new)
            model._set_rating(away_team, away_new)

        # Get team rating
        team_elo = model._get_rating(team_code)

        print(f"[OK] Loaded real Elo for {team_code}: {team_elo:.1f}")
        return float(team_elo)

    except Exception as e:
        print(f"[ERROR] Failed to load real Elo: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_real_attendance(team_code: str = "LVA", year: int = 2024) -> Optional[float]:
    """
    Load real attendance data.

    Args:
        team_code: Team abbreviation
        year: Year

    Returns:
        Average attendance
    """
    try:
        # Try detailed attendance file first
        detailed_path = DATA_DIR / "processed" / "clean" / "attendance_clean.csv"

        if detailed_path.exists():
            df = _read_csv(detailed_path, encoding='utf-8-sig')

            # Map team codes to full names
            team_name_map = {
                "LVA": "Las Vegas Aces",
                "NYL": "New York Liberty",
                "SEA": "Seattle Storm",
                "CON": "Connecticut Sun",
                "CHI": "Chicago Sky",
                "IND": "Indiana Fever",
                "ATL": "Atlanta Dream",
                "DAL": "Dallas Wings",
                "PHO": "Phoenix Mercury",
                "LAS": "Los Angeles Sparks",
                "MIN": "Minnesota Lynx",
                "WAS": "Washington Mystics",
            }

            team_name = team_name_map.get(team_code, team_code)
            team_data = df[(df['Team'] == team_name) & (df['Year'] == year)]

            if not team_data.empty and 'Average' in team_data.columns:
                attendance = float(team_data['Average'].values[0])
                print(f"[OK] Loaded real attendance for {team_code} ({year}): {attendance:.0f}")
                return attendance

        # Fallback to summary file
        attendance_path = PROCESSED_DIR / "wnba-average-regular-season-attendance-1997-2025_clean.csv"

        if not attendance_path.exists():
            print(f"[WARNING] Attendance data not found")
            return None

        df = _read_csv(attendance_path, encoding='utf-8-sig')

        if 'Year' in df.columns and 'AvgAttendance' in df.columns:
            year_data = df[df['Year'] == year]
            if not year_data.empty:
                attendance = float(year_data['AvgAttendance'].values[0])
                print(f"[OK] Loaded league average attendance for {year}: {attendance:.0f}")
                return attendance

        print(f"[WARNING] No attendance data found for {team_code} in {year}")
        return None

    except Exception as e:
        print(f"[ERROR] Failed to load attendance: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_real_revenue(team_code: str = "LVA") -> Optional[float]:
    """
    Load real revenue data.

    Args:
        team_code: Team abbreviation

    Returns:
        Team revenue in millions USD
    """
    try:
        revenue_path = PROCESSED_DIR / "wnba-teams-with-the-highest-revenue-2024_clean.csv"

        if not revenue_path.exists():
            print(f"[WARNING] Revenue data not found at {revenue_path}")
            return None

        df = _read_csv(revenue_path, encoding='utf-8-sig')

        # Try to find team
        if 'team_abbr' in df.columns:
            team_data = df[df['team_abbr'] == team_code]
        elif 'Team' in df.columns:
            # Map team names
            team_name_map = {
                "LVA": "Las Vegas Aces",
                "NYL": "New York Liberty",
                "SEA": "Seattle Storm",
                "CON": "Connecticut Sun",
                "CHI": "Chicago Sky",
                "IND": "Indiana Fever",
            }
            if team_code in team_name_map:
                team_data = df[df['Team'] == team_name_map[team_code]]
            else:
                team_data = pd.DataFrame()
        else:
            team_data = pd.DataFrame()

        if not team_data.empty and 'TeamRevenueUSD' in team_data.columns:
            revenue_usd = float(team_data['TeamRevenueUSD'].values[0])
            revenue_m = revenue_usd / 1e6  # Convert to millions
            print(f"[OK] Loaded real revenue for {team_code}: ${revenue_m:.2f}M")
            return revenue_m
        else:
            print(f"[WARNING] No revenue data found for {team_code}")
            return None

    except Exception as e:
        print(f"[ERROR] Failed to load revenue: {e}")
        return None


def load_lva_real_data() -> Dict[str, float]:
    """
    Load all real data for LVA.

    Returns:
        Dictionary with real data values
    """
    print("\n" + "="*80)
    print("Loading Real Data for LVA")
    print("="*80)

    data = {}

    # Load Elo
    elo = load_real_elo_ratings("LVA", season=2024)
    if elo is not None:
        data['elo'] = elo
    else:
        data['elo'] = 1620  # Fallback estimate
        print(f"[FALLBACK] Using estimated Elo: {data['elo']}")

    # Load attendance
    attendance = load_real_attendance("LVA", year=2024)
    if attendance is not None:
        data['attendance'] = attendance
    else:
        data['attendance'] = 9500  # Fallback estimate
        print(f"[FALLBACK] Using estimated attendance: {data['attendance']}")

    # Load revenue
    revenue = load_real_revenue("LVA")
    if revenue is not None:
        data['revenue'] = revenue
    else:
        data['revenue'] = 15.0  # Fallback estimate
        print(f"[FALLBACK] Using estimated revenue: ${data['revenue']:.2f}M")

    print("="*80)
    print()

    return data


if __name__ == "__main__":
    # Test loading real data
    data = load_lva_real_data()

    print("Loaded Data Summary:")
    print(f"  Elo Rating:  {data['elo']:.1f}")
    print(f"  Attendance:  {data['attendance']:.0f}")
    print(f"  Revenue:     ${data['revenue']:.2f}M")
