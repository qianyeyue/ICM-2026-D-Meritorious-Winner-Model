"""
Load and prepare real WNBA data for expansion analysis.
"""

from pathlib import Path
from typing import Dict, Tuple, Union
import pandas as pd
import numpy as np


def _read_csv(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    path = Path(path)
    with path.open("r", encoding=kwargs.pop("encoding", "utf-8-sig"), newline="") as f:
        return pd.read_csv(f, **kwargs)


# WNBA team locations (lat, lon) - current 12 teams as of 2024
# NOTE: Use the same team codes as Task1 (see project/model/brands.py).
WNBA_TEAM_LOCATIONS = {
    "ATL": (33.7573, -84.3963),  # Atlanta Dream - State Farm Arena
    "CHI": (41.8807, -87.6742),  # Chicago Sky - Wintrust Arena
    "CON": (41.5623, -72.6506),  # Connecticut Sun - Mohegan Sun Arena
    "DAL": (32.7905, -96.8103),  # Dallas Wings - College Park Center
    "IND": (39.7640, -86.1555),  # Indiana Fever - Gainbridge Fieldhouse
    "LVA": (36.0909, -115.1833), # Las Vegas Aces - Michelob ULTRA Arena
    "LAS": (34.0430, -118.2673), # Los Angeles Sparks - Crypto.com Arena
    "MIN": (44.9795, -93.2760),  # Minnesota Lynx - Target Center
    "NYL": (40.6826, -73.9754),  # New York Liberty - Barclays Center
    "PHO": (33.4457, -112.0712), # Phoenix Mercury - Footprint Center
    "SEA": (47.6221, -122.3540), # Seattle Storm - Climate Pledge Arena
    "WAS": (38.8981, -77.0209),  # Washington Mystics - Entertainment & Sports Arena
}

# Potential expansion cities
EXPANSION_CITIES = {
    "Toronto": (43.6435, -79.3791),      # Scotiabank Arena
    "Bay Area": (37.7503, -122.2028),    # Oakland/San Francisco
    "Portland": (45.5316, -122.6668),    # Moda Center
    "Nashville": (36.1591, -86.7784),    # Bridgestone Arena
    "Philadelphia": (39.9012, -75.1720), # Wells Fargo Center
    "Houston": (29.7508, -95.3621),      # Toyota Center
}

# Team abbreviation mapping
TEAM_NAME_MAPPING = {
    "Atlanta Dream": "ATL",
    "Chicago Sky": "CHI",
    "Connecticut Sun": "CON",
    "Dallas Wings": "DAL",
    "Indiana Fever": "IND",
    "Las Vegas Aces": "LVA",
    "Los Angeles Sparks": "LAS",
    "Minnesota Lynx": "MIN",
    "New York Liberty": "NYL",
    "Phoenix Mercury": "PHO",
    "Seattle Storm": "SEA",
    "Washington Mystics": "WAS",
}


def get_project_root() -> Path:
    """Get project root directory."""
    # From task3/data_loader.py -> task3 -> model -> project -> repository root
    return Path(__file__).resolve().parents[3]


def load_team_revenue_data() -> pd.DataFrame:
    """Load WNBA team revenue data."""
    data_path = get_project_root() / "project" / "data" / "processed" / "other_clean" / "wnba-teams-with-the-highest-revenue-2024_clean.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Revenue data not found at {data_path}")

    df = _read_csv(data_path, encoding='utf-8-sig')

    # Map team names to abbreviations
    df['team_abbr'] = df['Team'].map(TEAM_NAME_MAPPING)

    return df


def load_attendance_data() -> pd.DataFrame:
    """Load WNBA average attendance data."""
    data_path = get_project_root() / "project" / "data" / "processed" / "other_clean" / "wnba-average-regular-season-attendance-1997-2025_clean.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Attendance data not found at {data_path}")

    df = _read_csv(data_path, encoding='utf-8-sig')

    return df


def load_game_logs() -> pd.DataFrame:
    """Load WNBA game logs."""
    data_root = get_project_root() / "project" / "data"
    candidates = [
        data_root / "raw" / "wnba_gamelogs_2015_2025.csv",
        data_root / "processed" / "clean" / "wnba_gamelogs_clean.csv",
        data_root / "wnba_gamelogs_2015_2025.csv",
    ]
    data_path = next((path for path in candidates if path.exists()), None)

    if data_path is None:
        tried = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Game logs not found. Tried: {tried}")

    df = _read_csv(data_path)

    return df


def calculate_team_attendance_by_season(game_logs: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate per-team attendance estimates from game logs.

    Note: Game logs don't have attendance data, so we'll use
    league averages and adjust by team market size/revenue.
    """
    # Get home games only
    home_games = game_logs[game_logs['Home'] == 1].copy()

    # Count games per team per season
    team_games = home_games.groupby(['Season', 'Team']).size().reset_index(name='home_games')

    return team_games


def load_wnba_expansion_data() -> Dict:
    """
    Load all WNBA data needed for expansion analysis.

    Returns:
        Dictionary containing:
        - team_locations: Dict[str, Tuple[float, float]]
        - expansion_cities: Dict[str, Tuple[float, float]]
        - team_revenue: pd.DataFrame
        - attendance_history: pd.DataFrame
        - team_games: pd.DataFrame
    """
    # Load data
    revenue_df = load_team_revenue_data()
    attendance_df = load_attendance_data()
    game_logs = load_game_logs()
    team_games = calculate_team_attendance_by_season(game_logs)

    # Calculate team-specific attendance estimates
    # Use 2024 revenue as proxy for market size
    revenue_2024 = revenue_df[revenue_df['Year'] == 2024].copy()

    # Get 2024 league average attendance
    avg_2024 = attendance_df[attendance_df['Year'] == 2024]['AvgAttendance'].values[0]

    # Estimate team attendance based on revenue share
    total_revenue = revenue_2024['TeamRevenueUSD'].sum()
    revenue_2024['revenue_share'] = revenue_2024['TeamRevenueUSD'] / total_revenue

    # Assume attendance roughly proportional to revenue (with some variance)
    # Top teams get more, bottom teams get less
    revenue_2024['estimated_attendance'] = avg_2024 * (0.5 + revenue_2024['revenue_share'] * 12)

    # Add locations
    revenue_2024['location'] = revenue_2024['team_abbr'].map(WNBA_TEAM_LOCATIONS)

    return {
        'team_locations': WNBA_TEAM_LOCATIONS,
        'expansion_cities': EXPANSION_CITIES,
        'team_revenue': revenue_2024,
        'attendance_history': attendance_df,
        'team_games': team_games,
        'game_logs': game_logs,
    }


def get_team_market_size(team_abbr: str, revenue_df: pd.DataFrame) -> str:
    """
    Classify team as large/medium/small market based on revenue.

    Args:
        team_abbr: Team abbreviation
        revenue_df: Revenue dataframe

    Returns:
        'large', 'medium', or 'small'
    """
    team_data = revenue_df[revenue_df['team_abbr'] == team_abbr]

    if team_data.empty:
        return 'medium'

    revenue = team_data['TeamRevenueUSD'].values[0]

    # Thresholds based on 2024 data
    if revenue >= 22_000_000:  # Top tier (IND, NYL, PHO, LVA, SEA)
        return 'large'
    elif revenue >= 15_000_000:  # Mid tier
        return 'medium'
    else:  # Bottom tier
        return 'small'


def estimate_expansion_city_revenue(city_name: str) -> float:
    """
    Estimate potential revenue for expansion city based on market characteristics.

    Args:
        city_name: Name of expansion city

    Returns:
        Estimated annual revenue in USD
    """
    # Market size estimates based on metro population, sports culture, etc.
    market_estimates = {
        "Toronto": 20_000_000,      # Large international market
        "Bay Area": 22_000_000,     # Large tech market
        "Portland": 16_000_000,     # Medium market, strong sports culture
        "Nashville": 15_000_000,    # Growing market
        "Philadelphia": 18_000_000, # Large East Coast market
        "Houston": 19_000_000,      # Large Texas market
    }

    return market_estimates.get(city_name, 15_000_000)


def estimate_expansion_city_attendance(city_name: str) -> float:
    """
    Estimate potential attendance for expansion city.

    Args:
        city_name: Name of expansion city

    Returns:
        Estimated average attendance
    """
    # Attendance estimates based on market size and sports culture
    attendance_estimates = {
        "Toronto": 8500,   # Strong basketball market
        "Bay Area": 9000,  # Tech money, Warriors fans
        "Portland": 7500,  # Strong local support
        "Nashville": 7000, # Growing market
        "Philadelphia": 8000,  # Large market
        "Houston": 7800,   # Large market
    }

    return attendance_estimates.get(city_name, 7000)
