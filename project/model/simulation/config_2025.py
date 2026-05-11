"""
2025 Season Prediction Configuration for LVA (Las Vegas Aces)

This module defines the parameters and settings for predicting the 2025 WNBA season.
Key changes from 2024:
1. League expansion: 13 teams (+ Golden State Valkyries)
2. Market growth factor: +15% due to continued Caitlin Clark effect
3. Updated roster (some players transferred)
4. More games: 44 games per team (up from 40)

Usage:
    python -m project.model.simulation.predict_2025 --team LVA
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class Season2025Config:
    """Configuration for 2025 season prediction."""
    
    # Season parameters
    season_year: int = 2025
    num_teams: int = 13  # 12 + Golden State Valkyries
    games_per_team: int = 44  # Increased from 40
    periods: int = 11  # Same structure as 2024
    
    # Market growth factors (relative to 2024 baseline)
    market_growth_factor: float = 1.15  # +15% overall market growth
    caitlin_clark_effect: float = 1.10  # Additional boost for games vs Indiana
    
    # Demand model parameters (calibrated from 2024)
    beta0_2025: float = 10.61 + 0.14  # Adjusted for market growth: ln(1.15) ≈ 0.14
    epsilon: float = 0.50  # Price elasticity (unchanged)
    
    # Stage multipliers (seasonal demand variation)
    stage_multipliers: Tuple[float, ...] = (1.0, 1.0, 1.1, 1.1, 1.2, 1.2, 1.3, 1.3, 1.4, 1.4, 1.5)
    
    # Initial cash (carry over from 2024 or reset)
    initial_cash: float = 5_000_000  # Conservative: assume fresh start
    
    # New team info
    expansion_team: str = "GSV"  # Golden State Valkyries
    expansion_team_elo: float = 1450  # Starting Elo for new team


# 2025 WNBA Teams (13 teams)
TEAMS_2025 = {
    "ATL": "Atlanta Dream",
    "CHI": "Chicago Sky", 
    "CON": "Connecticut Sun",
    "DAL": "Dallas Wings",
    "IND": "Indiana Fever",  # Caitlin Clark
    "LVA": "Las Vegas Aces",
    "LAX": "Los Angeles Sparks",
    "MIN": "Minnesota Lynx",
    "NYL": "New York Liberty",
    "PHO": "Phoenix Mercury",
    "SEA": "Seattle Storm",
    "WAS": "Washington Mystics",
    "GSV": "Golden State Valkyries",  # NEW in 2025
}


# LVA roster changes for 2025 (estimated based on available data)
LVA_ROSTER_2025_CHANGES = """
Departures (transferred or retired):
- Kate Martin -> Golden State Valkyries (expansion draft)
- Tiffany Hayes -> Golden State Valkyries (free agency)

Remaining core:
- A'ja Wilson (MVP candidate)
- Chelsea Gray
- Kelsey Plum  
- Jackie Young
- Kiah Stokes
- Megan Gustafson
- Alysha Clark
- Sydney Colson
- Tiffany Mitchell

New additions (hypothetical):
- Draft pick / free agent signings TBD
"""


# Elo ratings entering 2025 (estimated based on 2024 performance + carryover)
ELO_2025_INITIAL = {
    "NYL": 1650,  # Strong contender
    "LVA": 1620,  # Defending strength
    "CON": 1600,  # Playoff team
    "MIN": 1580,  # Consistent
    "SEA": 1560,  # Rebuilding champion
    "IND": 1540,  # Rising with Caitlin Clark
    "PHO": 1520,  # Veteran team
    "CHI": 1500,  # Middle tier
    "ATL": 1480,  # Developing
    "WAS": 1460,  # Rebuilding
    "DAL": 1450,  # Young team
    "LAX": 1440,  # Rebuilding
    "GSV": 1450,  # Expansion team (slightly below average)
}


# Venue capacities for 2025
VENUE_CAPACITY_2025 = {
    "ATL": 4500,   # Gateway Center Arena
    "CHI": 10387,  # Wintrust Arena
    "CON": 10000,  # Mohegan Sun Arena
    "DAL": 6800,   # College Park Center
    "IND": 18165,  # Gainbridge Fieldhouse (increased for Clark)
    "LVA": 12000,  # Michelob Ultra Arena
    "LAX": 10000,  # Crypto.com Arena (downsized section)
    "MIN": 12000,  # Target Center
    "NYL": 17732,  # Barclays Center
    "PHO": 14870,  # Footprint Center
    "SEA": 18100,  # Climate Pledge Arena
    "WAS": 4200,   # Entertainment & Sports Arena
    "GSV": 18064,  # Chase Center (NEW - shares with Warriors)
}


# Base ticket prices for 2025 (estimated +10% from 2024)
BASE_TICKET_PRICE_2025 = {
    "ATL": 45,
    "CHI": 55,
    "CON": 50,
    "DAL": 40,
    "IND": 85,   # Premium due to Caitlin Clark
    "LVA": 141,  # $128 * 1.10
    "LAX": 60,
    "MIN": 55,
    "NYL": 95,   # Premium market
    "PHO": 50,
    "SEA": 65,
    "WAS": 45,
    "GSV": 100,  # Bay Area premium
}


def get_2025_schedule_pattern(team: str, num_periods: int = 11) -> List[Dict]:
    """
    Generate a representative 2025 schedule pattern.
    
    With 13 teams and 44 games:
    - 12 opponents × 3.5 games avg = 42 games + 2 extra rivalry games
    - Roughly 22 home, 22 away
    """
    import random
    random.seed(2025 + hash(team))  # Reproducible per team
    
    games_per_period = 44 // num_periods  # ~4 games per period
    
    schedule = []
    opponents = [t for t in TEAMS_2025.keys() if t != team]
    
    for period in range(1, num_periods + 1):
        # Distribute home/away roughly evenly
        home_games = games_per_period // 2 + (1 if period % 2 == 0 else 0)
        away_games = games_per_period - home_games
        
        # Select random opponents for this period
        period_opponents_home = random.sample(opponents, min(home_games, len(opponents)))
        period_opponents_away = random.sample(opponents, min(away_games, len(opponents)))
        
        schedule.append({
            "period": period,
            "home_games": home_games,
            "away_games": away_games,
            "total_games": home_games + away_games,
            "home_opponents": period_opponents_home,
            "away_opponents": period_opponents_away,
        })
    
    return schedule


if __name__ == "__main__":
    # Print 2025 season configuration
    config = Season2025Config()
    print("=" * 60)
    print("2025 WNBA Season Prediction Configuration")
    print("=" * 60)
    print(f"Teams: {config.num_teams}")
    print(f"Games per team: {config.games_per_team}")
    print(f"Market growth: +{(config.market_growth_factor - 1) * 100:.0f}%")
    print(f"β₀ (2025): {config.beta0_2025:.2f}")
    print(f"New team: {config.expansion_team} ({TEAMS_2025[config.expansion_team]})")
    
    print("\n" + "=" * 60)
    print("LVA 2025 Schedule Pattern")
    print("=" * 60)
    schedule = get_2025_schedule_pattern("LVA")
    for p in schedule:
        print(f"Period {p['period']:2d}: {p['home_games']} home, {p['away_games']} away")
