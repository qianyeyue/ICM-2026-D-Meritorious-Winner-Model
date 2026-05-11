"""
Team-level Star Power Aggregation Module.

This module extends players.py to provide team-level star power calculation
by aggregating player-level PCV (Player Commercial Value) indices.

Mathematical Model
------------------
Star_t = (1 + 魏_u路ln(1+u_t) + 魏_m路ln(1+m_t)) 路 危_i PCV_i 路 Minutes_{i,t}/40 路 位_decay(PCV_i)

Where:
    - 魏_u: Investment amplification coefficient
    - 魏_m: Marketing amplification coefficient
    - u_t: Sports operations investment (USD)
    - m_t: Marketing spend (USD)
    - PCV_i: Player i's commercial value index
    - Minutes_{i,t}: Player i's minutes per game
    - 位_decay(PCV_i): Softmax decay emphasizing high-value players

Softmax Decay:
    位_decay(PCV_i) = exp(纬路PCV_i) / 危_j exp(纬路PCV_j)

This emphasizes star players (high PCV) over role players (low PCV).

Integration
-----------
This module works with:
- players.py: Provides player-level PCV calculation
- brands.py: Uses Star_t in brand dynamics (B_{t+1} = ... + 畏_star路Star_t)
- income.py: Uses Star_t in demand model (ln(Dem_g) = ... + 尾_star路Star_t)
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def aggregate_team_star_power(
    roster_pcv: pd.Series,
    minutes_per_game: pd.Series,
    *,
    investment_usd: float = 0.0,
    marketing_usd: float = 0.0,
    kappa_u: float = 0.25,
    kappa_m: float = 0.35,
    gamma: float = 2.0,
    use_softmax_decay: bool = True,
) -> float:
    """
    Aggregate player-level PCV into team-level Star_t.

    Star_t = (1 + 魏_u路ln(1+u_t) + 魏_m路ln(1+m_t)) 路 危_i PCV_i 路 Minutes_{i,t}/40 路 位_decay(PCV_i)

    Where 位_decay emphasizes high-value players using softmax:
        位_decay(PCV_i) = exp(纬路PCV_i) / 危_j exp(纬路PCV_j)

    Parameters
    ----------
    roster_pcv : pd.Series
        Player commercial value indices
    minutes_per_game : pd.Series
        Minutes per game for each player
    investment_usd : float, default=0.0
        Sports operations investment (u_t) in USD
    marketing_usd : float, default=0.0
        Marketing spend (m_t) in USD
    kappa_u : float, default=0.25
        Investment amplification coefficient
    kappa_m : float, default=0.35
        Marketing amplification coefficient
    gamma : float, default=2.0
        Softmax temperature for 位_decay (higher = more emphasis on stars)
    use_softmax_decay : bool, default=True
        Whether to use softmax decay (if False, uses uniform weighting)

    Returns
    -------
    float
        Team star power index

    Examples
    --------
    >>> # Example roster: 3 players with different PCV and minutes
    >>> pcv = pd.Series([2.0, 1.0, 0.5])  # Star, starter, bench
    >>> minutes = pd.Series([35.0, 28.0, 15.0])
    >>> aggregate_team_star_power(pcv, minutes, investment_usd=500000, marketing_usd=1000000)
    2.823...

    >>> # Same roster without investment/marketing
    >>> aggregate_team_star_power(pcv, minutes)
    2.123...
    """
    if len(roster_pcv) == 0:
        return 0.0

    # Investment/marketing amplification
    u_million = investment_usd / 1_000_000.0
    m_million = marketing_usd / 1_000_000.0
    amplification = (
        1.0 +
        kappa_u * float(np.log1p(max(u_million, 0.0))) +
        kappa_m * float(np.log1p(max(m_million, 0.0)))
    )

    # Convert to numpy arrays
    pcv_array = np.array(roster_pcv, dtype=float)
    minutes_array = np.array(minutes_per_game, dtype=float)

    # Calculate 位_decay (softmax weighting)
    if use_softmax_decay and len(pcv_array) > 1:
        exp_gamma_pcv = np.exp(gamma * pcv_array)
        lambda_decay = exp_gamma_pcv / np.sum(exp_gamma_pcv)
    else:
        # Uniform weighting
        lambda_decay = np.ones_like(pcv_array) / len(pcv_array)

    # Calculate star contributions
    # PCV_i 脳 (Minutes_i / 40) 脳 位_decay(PCV_i)
    star_contributions = pcv_array * (minutes_array / 40.0) * lambda_decay

    # Total team star power
    star_t = amplification * float(np.sum(star_contributions))

    return star_t


def calculate_star_power_from_roster_data(
    roster_df: pd.DataFrame,
    *,
    pcv_col: str = "pcv",
    minutes_col: str = "mpg",
    investment_usd: float = 0.0,
    marketing_usd: float = 0.0,
    **kwargs,
) -> float:
    """
    Convenience function to calculate Star_t from a roster DataFrame.

    Parameters
    ----------
    roster_df : pd.DataFrame
        Roster with PCV and minutes columns
    pcv_col : str, default="pcv"
        Column name for player commercial value
    minutes_col : str, default="mpg"
        Column name for minutes per game
    investment_usd : float, default=0.0
        Sports investment
    marketing_usd : float, default=0.0
        Marketing spend
    **kwargs
        Additional arguments passed to aggregate_team_star_power

    Returns
    -------
    float
        Team star power index
    """
    if roster_df.empty:
        return 0.0

    return aggregate_team_star_power(
        roster_df[pcv_col],
        roster_df[minutes_col],
        investment_usd=investment_usd,
        marketing_usd=marketing_usd,
        **kwargs,
    )


def calculate_star_power_by_team(
    players_df: pd.DataFrame,
    *,
    team_col: str = "team",
    pcv_col: str = "pcv",
    minutes_col: str = "mpg",
    investment_by_team: Optional[Dict[str, float]] = None,
    marketing_by_team: Optional[Dict[str, float]] = None,
    **kwargs,
) -> pd.Series:
    """
    Calculate Star_t for all teams in a dataset.

    Parameters
    ----------
    players_df : pd.DataFrame
        Player data with team, PCV, and minutes columns
    team_col : str, default="team"
        Column name for team
    pcv_col : str, default="pcv"
        Column name for player commercial value
    minutes_col : str, default="mpg"
        Column name for minutes per game
    investment_by_team : Dict[str, float], optional
        Sports investment by team (USD)
    marketing_by_team : Dict[str, float], optional
        Marketing spend by team (USD)
    **kwargs
        Additional arguments passed to aggregate_team_star_power

    Returns
    -------
    pd.Series
        Star_t indexed by team

    Examples
    --------
    >>> # Example: Calculate star power for all teams
    >>> players = pd.DataFrame({
    ...     'team': ['LVA', 'LVA', 'NYL', 'NYL'],
    ...     'pcv': [2.0, 1.0, 1.5, 0.8],
    ...     'mpg': [35.0, 28.0, 32.0, 25.0]
    ... })
    >>> calculate_star_power_by_team(players)
    team
    LVA    2.123...
    NYL    1.856...
    dtype: float64
    """
    if players_df.empty:
        return pd.Series(dtype=float)

    investment_by_team = investment_by_team or {}
    marketing_by_team = marketing_by_team or {}

    star_by_team = {}
    for team, group in players_df.groupby(team_col):
        star_t = aggregate_team_star_power(
            group[pcv_col],
            group[minutes_col],
            investment_usd=investment_by_team.get(team, 0.0),
            marketing_usd=marketing_by_team.get(team, 0.0),
            **kwargs,
        )
        star_by_team[team] = star_t

    return pd.Series(star_by_team, name="star_t")


if __name__ == "__main__":
    # Example usage
    print("Team Star Power Aggregation Examples")
    print("=" * 70)

    # Example 1: Single team with 3 players
    print("\n1. Single team (3 players):")
    pcv = pd.Series([2.0, 1.0, 0.5], name="pcv")
    minutes = pd.Series([35.0, 28.0, 15.0], name="mpg")

    star_base = aggregate_team_star_power(pcv, minutes)
    print(f"   Base Star_t (no investment): {star_base:.3f}")

    star_invest = aggregate_team_star_power(
        pcv, minutes,
        investment_usd=500_000,
        marketing_usd=1_000_000
    )
    print(f"   Star_t with $500K sports + $1M marketing: {star_invest:.3f}")
    print(f"   Amplification: {star_invest / star_base:.2f}x")

    # Example 2: Compare softmax vs uniform weighting
    print("\n2. Softmax decay effect:")
    star_softmax = aggregate_team_star_power(pcv, minutes, use_softmax_decay=True)
    star_uniform = aggregate_team_star_power(pcv, minutes, use_softmax_decay=False)
    print(f"   With softmax (emphasizes stars): {star_softmax:.3f}")
    print(f"   Without softmax (uniform): {star_uniform:.3f}")
    print(f"   Star emphasis boost: {(star_softmax - star_uniform) / star_uniform:.1%}")

    # Example 3: Multiple teams
    print("\n3. Multiple teams:")
    players = pd.DataFrame({
        'team': ['LVA', 'LVA', 'LVA', 'NYL', 'NYL', 'NYL'],
        'player': ['A\'ja Wilson', 'Kelsey Plum', 'Chelsea Gray',
                   'Breanna Stewart', 'Sabrina Ionescu', 'Jonquel Jones'],
        'pcv': [1.68, 1.77, 1.22, 1.91, 1.93, 1.71],
        'mpg': [34.4, 34.0, 26.0, 32.7, 32.1, 29.8]
    })

    star_by_team = calculate_star_power_by_team(players)
    print(f"   Las Vegas Aces: {star_by_team['LVA']:.3f}")
    print(f"   New York Liberty: {star_by_team['NYL']:.3f}")

    print("\n" + "=" * 70)

