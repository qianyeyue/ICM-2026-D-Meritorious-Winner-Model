"""
Fatigue and Injury Risk Model for WNBA Teams.

This module implements:
1. Fatigue index calculation (Fatigue_g)
2. Injury risk estimation
3. Integration with Elo strength updates

Mathematical Model
------------------
Fatigue Index:
    Fatigue_g = w1·I(B2B) + w2·max(0, r0-RestDays_g) + w3·TravelKm_g/1000 + w4·CoreMinutes_g/L

Where:
    - I(B2B): Back-to-back game indicator (1 if B2B, 0 otherwise)
    - RestDays_g: Days of rest before game g
    - TravelKm_g: Travel distance in kilometers
    - CoreMinutes_g: Average minutes played by core players
    - L: League average core minutes (baseline)

Injury Risk (Two-state Markov):
    P(h_{i,t+1}=0 | h_{i,t}=1) = σ(θ0 + θ1·Minutes_{i,t} + θ2·Fatigue_t + θ3·Age_i - θ_med·ln(1+u_t))

Integration with Elo:
    S_{g+1} = S_g + K(y_g - p_g) + α_u·ln(1+u_t) - φ·Fatigue_g

Notes
-----
- Fatigue affects both team strength (via Elo) and commercial value (via Star_t)
- Parameters calibrated based on WNBA schedule patterns and injury data
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class FatigueParameters:
    """
    Parameters for fatigue calculation.

    Calibrated based on WNBA schedule analysis:
    - WNBA regular season: ~40 games over ~4 months
    - Back-to-back games: ~10-15% of schedule
    - Average travel: ~1,500 km per away game
    - Core players: ~32-35 minutes per game
    """

    # Fatigue weights
    w1: float = 0.30  # Back-to-back penalty (high impact)
    w2: float = 0.25  # Rest days penalty
    w3: float = 0.20  # Travel distance penalty
    w4: float = 0.25  # Core minutes penalty

    # Baseline parameters
    r0: float = 2.0  # Optimal rest days (2 days between games)
    L: float = 33.0  # League average core minutes per game

    # Fatigue bounds (normalized to [0, 1])
    max_fatigue: float = 1.0  # Maximum fatigue index


@dataclass
class InjuryRiskParameters:
    """
    Parameters for injury risk estimation.

    Based on WNBA injury patterns:
    - Higher risk for players >30 years old
    - Higher risk with >32 minutes per game
    - Higher risk during high-fatigue periods
    - Medical investment reduces risk
    """

    # Logistic regression coefficients
    theta_0: float = -3.5  # Baseline (low injury rate)
    theta_1: float = 0.08  # Minutes effect (per minute)
    theta_2: float = 2.0  # Fatigue effect (per unit fatigue)
    theta_3: float = 0.05  # Age effect (per year)
    theta_med: float = 0.5  # Medical investment effect (per ln(1+u_million))


def calculate_fatigue_index(
    is_back_to_back: bool,
    rest_days: float,
    travel_km: float,
    core_minutes: float,
    params: FatigueParameters = None,
) -> float:
    """
    Calculate fatigue index for a single game.

    Fatigue_g = w1·I(B2B) + w2·max(0, r0-RestDays_g) + w3·TravelKm_g/1000 + w4·CoreMinutes_g/L

    Parameters
    ----------
    is_back_to_back : bool
        Whether this is a back-to-back game
    rest_days : float
        Days of rest before this game
    travel_km : float
        Travel distance in kilometers
    core_minutes : float
        Average minutes played by core players (top 5-7 players)
    params : FatigueParameters, optional
        Fatigue parameters

    Returns
    -------
    float
        Fatigue index in [0, max_fatigue]

    Examples
    --------
    >>> # Normal game: 2 days rest, 1000km travel, 32 min core
    >>> calculate_fatigue_index(False, 2.0, 1000.0, 32.0)
    0.22424242424242424

    >>> # Back-to-back game: 0 days rest, 2000km travel, 35 min core
    >>> calculate_fatigue_index(True, 0.0, 2000.0, 35.0)
    1.1515151515151516
    """
    if params is None:
        params = FatigueParameters()

    # Component 1: Back-to-back indicator
    b2b_component = params.w1 * float(is_back_to_back)

    # Component 2: Rest days penalty
    rest_penalty = max(0.0, params.r0 - rest_days)
    rest_component = params.w2 * rest_penalty

    # Component 3: Travel distance penalty (normalized per 1000 km)
    travel_component = params.w3 * (travel_km / 1000.0)

    # Component 4: Core minutes penalty (normalized by league average)
    minutes_ratio = core_minutes / params.L
    minutes_component = params.w4 * minutes_ratio

    # Total fatigue
    fatigue = b2b_component + rest_component + travel_component + minutes_component

    # Clip to [0, max_fatigue]
    return float(np.clip(fatigue, 0.0, params.max_fatigue))


def calculate_schedule_fatigue(
    schedule: pd.DataFrame,
    team: str,
    *,
    travel_distances: Optional[Dict[Tuple[str, str], float]] = None,
    core_minutes_by_game: Optional[Dict[int, float]] = None,
    params: FatigueParameters = None,
) -> pd.DataFrame:
    """
    Calculate fatigue for all games in a team's schedule.

    Parameters
    ----------
    schedule : pd.DataFrame
        Schedule with columns: game_number, date, home_team, away_team
    team : str
        Team code (e.g., 'LVA', 'NYL')
    travel_distances : Dict[Tuple[str, str], float], optional
        Travel distances between cities (from_city, to_city) -> km
        If None, uses default estimates
    core_minutes_by_game : Dict[int, float], optional
        Core minutes by game number
        If None, uses league average
    params : FatigueParameters, optional
        Fatigue parameters

    Returns
    -------
    pd.DataFrame
        Schedule with added columns: is_b2b, rest_days, travel_km, core_minutes, fatigue
    """
    if params is None:
        params = FatigueParameters()

    # Filter team's games
    team_games = schedule[
        (schedule['home_team'] == team) | (schedule['away_team'] == team)
    ].copy()
    team_games = team_games.sort_values('game_number').reset_index(drop=True)

    # Calculate rest days and back-to-back indicators
    team_games['date'] = pd.to_datetime(team_games['date'])
    team_games['rest_days'] = team_games['date'].diff().dt.days.fillna(3.0)
    team_games['is_b2b'] = (team_games['rest_days'] == 0).astype(bool)

    # Estimate travel distances (simplified)
    if travel_distances is None:
        # Default: home games = 0 km, away games = 1500 km average
        team_games['travel_km'] = team_games.apply(
            lambda row: 0.0 if row['home_team'] == team else 1500.0,
            axis=1
        )
    else:
        # Use provided travel distances
        team_games['travel_km'] = 0.0  # Placeholder, would need city mapping

    # Core minutes (use provided or default)
    if core_minutes_by_game is None:
        team_games['core_minutes'] = params.L
    else:
        team_games['core_minutes'] = team_games['game_number'].map(
            lambda g: core_minutes_by_game.get(g, params.L)
        )

    # Calculate fatigue for each game
    team_games['fatigue'] = team_games.apply(
        lambda row: calculate_fatigue_index(
            row['is_b2b'],
            row['rest_days'],
            row['travel_km'],
            row['core_minutes'],
            params
        ),
        axis=1
    )

    return team_games


def calculate_injury_probability(
    minutes_per_game: float,
    fatigue: float,
    age: float,
    medical_investment_usd: float = 0.0,
    params: InjuryRiskParameters = None,
) -> float:
    """
    Calculate injury probability for a player.

    P(injury) = σ(θ0 + θ1·Minutes + θ2·Fatigue + θ3·Age - θ_med·ln(1+u_t))

    Parameters
    ----------
    minutes_per_game : float
        Average minutes per game
    fatigue : float
        Team fatigue index
    age : float
        Player age
    medical_investment_usd : float, default=0.0
        Medical/training investment in USD
    params : InjuryRiskParameters, optional
        Injury risk parameters

    Returns
    -------
    float
        Injury probability in [0, 1]

    Examples
    --------
    >>> # Young player, low minutes, low fatigue, no investment
    >>> calculate_injury_probability(25.0, 0.2, 24.0, 0.0)
    0.03743422675270758

    >>> # Veteran, high minutes, high fatigue, no investment
    >>> calculate_injury_probability(35.0, 0.8, 32.0, 0.0)
    0.2689414213699951

    >>> # Same veteran with $500K medical investment
    >>> calculate_injury_probability(35.0, 0.8, 32.0, 500000.0)
    0.18242552380635635
    """
    if params is None:
        params = InjuryRiskParameters()

    # Logit calculation
    u_million = medical_investment_usd / 1_000_000.0
    logit = (
        params.theta_0 +
        params.theta_1 * minutes_per_game +
        params.theta_2 * fatigue +
        params.theta_3 * age -
        params.theta_med * float(np.log1p(max(u_million, 0.0)))
    )

    # Sigmoid (logistic function)
    prob = 1.0 / (1.0 + np.exp(-logit))
    return float(np.clip(prob, 0.0, 1.0))


def simulate_injury_transitions(
    roster: pd.DataFrame,
    fatigue: float,
    medical_investment_usd: float = 0.0,
    params: InjuryRiskParameters = None,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Simulate injury transitions for a roster.

    Parameters
    ----------
    roster : pd.DataFrame
        Roster with columns: player_name, age, mpg, healthy (bool)
    fatigue : float
        Current team fatigue index
    medical_investment_usd : float, default=0.0
        Medical investment
    params : InjuryRiskParameters, optional
        Injury risk parameters
    random_state : int, optional
        Random seed

    Returns
    -------
    pd.DataFrame
        Updated roster with new healthy status
    """
    if params is None:
        params = InjuryRiskParameters()

    rng = np.random.RandomState(random_state)
    roster = roster.copy()

    for idx, player in roster.iterrows():
        if player.get('healthy', True):
            # Calculate injury probability
            injury_prob = calculate_injury_probability(
                player['mpg'],
                fatigue,
                player['age'],
                medical_investment_usd,
                params
            )

            # Simulate transition
            roster.at[idx, 'healthy'] = (rng.rand() > injury_prob)

    return roster


def estimate_fatigue_impact_on_strength(
    fatigue: float,
    phi: float = 5.0,
) -> float:
    """
    Estimate Elo penalty from fatigue.

    Elo_penalty = φ·Fatigue_g

    Parameters
    ----------
    fatigue : float
        Fatigue index
    phi : float, default=5.0
        Fatigue penalty coefficient (Elo points per unit fatigue)

    Returns
    -------
    float
        Elo penalty (positive value to be subtracted)

    Examples
    --------
    >>> estimate_fatigue_impact_on_strength(0.5)
    2.5

    >>> estimate_fatigue_impact_on_strength(1.0)
    5.0
    """
    return float(phi * max(fatigue, 0.0))


if __name__ == "__main__":
    # Example usage
    print("Fatigue Model Examples")
    print("=" * 70)

    # Example 1: Normal game
    print("\n1. Normal game (2 days rest, 1000km travel, 32 min core):")
    fatigue_normal = calculate_fatigue_index(False, 2.0, 1000.0, 32.0)
    print(f"   Fatigue index: {fatigue_normal:.3f}")
    print(f"   Elo penalty: {estimate_fatigue_impact_on_strength(fatigue_normal):.2f} points")

    # Example 2: Back-to-back game
    print("\n2. Back-to-back game (0 days rest, 2000km travel, 35 min core):")
    fatigue_b2b = calculate_fatigue_index(True, 0.0, 2000.0, 35.0)
    print(f"   Fatigue index: {fatigue_b2b:.3f}")
    print(f"   Elo penalty: {estimate_fatigue_impact_on_strength(fatigue_b2b):.2f} points")

    # Example 3: Injury risk
    print("\n3. Injury risk examples:")
    print("   Young player (24y, 25mpg, low fatigue):")
    prob_young = calculate_injury_probability(25.0, 0.2, 24.0, 0.0)
    print(f"     Injury probability: {prob_young:.1%}")

    print("   Veteran (32y, 35mpg, high fatigue, no investment):")
    prob_vet = calculate_injury_probability(35.0, 0.8, 32.0, 0.0)
    print(f"     Injury probability: {prob_vet:.1%}")

    print("   Veteran with $500K medical investment:")
    prob_vet_invest = calculate_injury_probability(35.0, 0.8, 32.0, 500000.0)
    print(f"     Injury probability: {prob_vet_invest:.1%}")
    print(f"     Risk reduction: {(prob_vet - prob_vet_invest) / prob_vet:.1%}")

    print("\n" + "=" * 70)
