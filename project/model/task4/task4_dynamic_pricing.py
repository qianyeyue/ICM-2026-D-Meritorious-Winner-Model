"""
Task 4: Dynamic Ticket Pricing Model

This module implements dynamic ticket pricing strategies including:
1. Single-game pricing with time-varying price elasticity
2. Season ticket pricing
3. Attendance-dependent home advantage
4. Cross-period demand coupling through brand/attendance feedback

Mathematical Model
------------------
Price Elasticity Decay:
    ε_t = ε_0 * exp(-λ * t)

Demand Function (Single Game):
    ln(Dem_t) = β_0 - ε_t * ln(τ_t) + β^T * x_t

Demand Function (Season Ticket):
    ln(Dem^ST) = β_0^ST - ε_ST * ln(τ^ST) + β_ST^T * x_t

Modified Home Advantage:
    Δ_g = (S_t - S_t^opp) + H * (2 * Ω_g - 1)
    where Ω_g = Dem_g / Capacity (attendance rate)

Revenue:
    Rev^ticket_t = Σ_g (τ_t * p_0 * Dem_g) + Rev^ST

Constraints:
    - Dem_g ≤ Capacity (venue capacity)
    - Dem^ST ≤ 0.8 * Capacity (season ticket cap)
    - τ_t ∈ [τ_min, τ_max] (price bounds)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class DynamicPricingParameters:
    """Parameters for dynamic pricing model."""

    # Price elasticity parameters
    epsilon_0: float = 0.8  # Initial price elasticity (early season)
    lambda_decay: float = 0.15  # Elasticity decay rate
    epsilon_ST: float = 0.6  # Season ticket price elasticity

    # Demand function parameters
    beta_0: float = 9.31  # Baseline demand, log scale (about 11,000 seats)
    beta_S: float = 0.0015  # Elo coefficient, relative to S_ref
    beta_star: float = 0.004  # Star power coefficient, relative to Star_ref
    beta_opp: float = 0.0008  # Opponent quality coefficient, relative to S_opp_ref
    beta_league: float = 0.30  # League popularity coefficient, relative to league_pop_ref

    # Reference values used to keep the log-demand scale calibrated
    S_ref: float = 1600.0
    Star_ref: float = 50.0
    S_opp_ref: float = 1550.0
    league_pop_ref: float = 1.0

    # Season ticket parameters
    beta_0_ST: float = 8.60  # ST baseline demand, log scale
    ST_capacity_ratio: float = 0.8  # Max ST as fraction of capacity

    # Venue parameters
    capacity: float = 12000.0  # Venue capacity
    base_price: float = 50.0  # Base ticket price ($)

    # Home advantage parameters
    H_base: float = 100.0  # Base home advantage (Elo points)
    H_attendance_coef: float = 0.5  # Attendance effect on home advantage

    # Price bounds
    tau_min: float = 0.7  # Minimum price multiplier
    tau_max: float = 1.5  # Maximum price multiplier
    tau_ST_min: float = 0.6  # Minimum ST price multiplier
    tau_ST_max: float = 1.2  # Maximum ST price multiplier


def calculate_price_elasticity(
    period: int,
    params: DynamicPricingParameters,
) -> float:
    """
    Calculate time-varying price elasticity.

    ε_t = ε_0 * exp(-λ * t)

    Early season: high elasticity (price-sensitive)
    Late season: low elasticity (less price-sensitive)

    Parameters
    ----------
    period : int
        Current period (0-indexed)
    params : DynamicPricingParameters
        Pricing parameters

    Returns
    -------
    float
        Price elasticity for period t

    Examples
    --------
    >>> params = DynamicPricingParameters()
    >>> calculate_price_elasticity(0, params)  # Early season
    0.8
    >>> calculate_price_elasticity(2, params)  # Late season
    0.59...
    """
    return params.epsilon_0 * np.exp(-params.lambda_decay * period)


def calculate_single_game_demand(
    tau: float,
    S_t: float,
    Star_t: float,
    S_opp: float,
    league_pop: float,
    period: int,
    params: DynamicPricingParameters,
) -> float:
    """
    Calculate single-game ticket demand.

    ln(Dem_t) = β_0 - ε_t * ln(τ_t) + β_S * S_t + β_star * Star_t
                + β_opp * S_opp + β_league * L_t

    Parameters
    ----------
    tau : float
        Price multiplier
    S_t : float
        Home team Elo rating
    Star_t : float
        Home team star power
    S_opp : float
        Opponent Elo rating
    league_pop : float
        League popularity index
    period : int
        Current period
    params : DynamicPricingParameters
        Pricing parameters

    Returns
    -------
    float
        Expected attendance (demand)
    """
    # Calculate time-varying elasticity
    epsilon_t = calculate_price_elasticity(period, params)

    # Log-linear demand function
    ln_demand = (
        params.beta_0
        - epsilon_t * np.log(tau)
        + params.beta_S * (S_t - params.S_ref)
        + params.beta_star * (Star_t - params.Star_ref)
        + params.beta_opp * (S_opp - params.S_opp_ref)
        + params.beta_league * (league_pop - params.league_pop_ref)
    )

    # Convert to demand level
    demand = np.exp(ln_demand)

    # Apply capacity constraint
    demand = min(demand, params.capacity)

    return demand


def calculate_season_ticket_demand(
    tau_ST: float,
    S_0: float,
    Star_0: float,
    league_pop: float,
    params: DynamicPricingParameters,
) -> float:
    """
    Calculate season ticket demand.

    ln(Dem^ST) = β_0^ST - ε_ST * ln(τ^ST) + β_ST^T * x_0

    Parameters
    ----------
    tau_ST : float
        Season ticket price multiplier
    S_0 : float
        Team Elo at season start
    Star_0 : float
        Team star power at season start
    league_pop : float
        League popularity
    params : DynamicPricingParameters
        Pricing parameters

    Returns
    -------
    float
        Season ticket demand
    """
    # Log-linear demand for season tickets
    ln_demand_ST = (
        params.beta_0_ST
        - params.epsilon_ST * np.log(tau_ST)
        + params.beta_S * (S_0 - params.S_ref)
        + params.beta_star * (Star_0 - params.Star_ref)
        + params.beta_league * (league_pop - params.league_pop_ref)
    )

    # Convert to demand level
    demand_ST = np.exp(ln_demand_ST)

    # Apply season ticket capacity constraint (80% of capacity)
    max_ST = params.ST_capacity_ratio * params.capacity
    demand_ST = min(demand_ST, max_ST)

    return demand_ST


def calculate_attendance_rate(
    demand: float,
    params: DynamicPricingParameters,
) -> float:
    """
    Calculate attendance rate (occupancy).

    Ω_g = Dem_g / Capacity

    Parameters
    ----------
    demand : float
        Ticket demand
    params : DynamicPricingParameters
        Pricing parameters

    Returns
    -------
    float
        Attendance rate in [0, 1]
    """
    return min(demand / params.capacity, 1.0)


def calculate_modified_home_advantage(
    attendance_rate: float,
    params: DynamicPricingParameters,
) -> float:
    """
    Calculate attendance-dependent home advantage.

    H_effective = H_base * (2 * Ω_g - 1)

    When Ω_g = 1 (full): H_effective = H_base
    When Ω_g = 0.5: H_effective = 0
    When Ω_g = 0: H_effective = -H_base

    Parameters
    ----------
    attendance_rate : float
        Attendance rate Ω_g ∈ [0, 1]
    params : DynamicPricingParameters
        Pricing parameters

    Returns
    -------
    float
        Effective home advantage (Elo points)
    """
    return params.H_base * (2 * attendance_rate - 1)


def calculate_win_probability_with_attendance(
    S_home: float,
    S_away: float,
    attendance_rate: float,
    params: DynamicPricingParameters,
) -> float:
    """
    Calculate win probability with attendance-dependent home advantage.

    Δ_g = (S_home - S_away) + H * (2 * Ω_g - 1)
    p_g = 1 / (1 + 10^(-Δ_g / 400))

    Parameters
    ----------
    S_home : float
        Home team Elo
    S_away : float
        Away team Elo
    attendance_rate : float
        Attendance rate
    params : DynamicPricingParameters
        Pricing parameters

    Returns
    -------
    float
        Win probability
    """
    # Calculate home advantage
    H_effective = calculate_modified_home_advantage(attendance_rate, params)

    # Calculate advantage
    delta = (S_home - S_away) + H_effective

    # Convert to win probability
    win_prob = 1.0 / (1.0 + 10 ** (-delta / 400))

    return win_prob


def calculate_period_ticket_revenue(
    tau_t: float,
    games: List[Dict],
    S_t: float,
    Star_t: float,
    league_pop: float,
    period: int,
    params: DynamicPricingParameters,
) -> Tuple[float, List[float]]:
    """
    Calculate ticket revenue for a period.

    Rev^ticket_t = Σ_g (τ_t * p_0 * Dem_g)

    Parameters
    ----------
    tau_t : float
        Price multiplier for period t
    games : List[Dict]
        List of home games with opponent info
    S_t : float
        Team Elo
    Star_t : float
        Team star power
    league_pop : float
        League popularity
    period : int
        Current period
    params : DynamicPricingParameters
        Pricing parameters

    Returns
    -------
    Tuple[float, List[float]]
        (Total revenue, List of demands per game)
    """
    total_revenue = 0.0
    demands = []

    for game in games:
        # Calculate demand for this game
        demand = calculate_single_game_demand(
            tau=tau_t,
            S_t=S_t,
            Star_t=Star_t,
            S_opp=game['opponent_elo'],
            league_pop=league_pop,
            period=period,
            params=params,
        )

        # Calculate revenue
        price = tau_t * params.base_price
        revenue = price * demand

        total_revenue += revenue
        demands.append(demand)

    return total_revenue, demands


def optimize_single_period_pricing(
    games: List[Dict],
    S_t: float,
    Star_t: float,
    league_pop: float,
    period: int,
    params: DynamicPricingParameters,
    fixed_u: float = 0.0,
    fixed_m: float = 0.0,
) -> Dict[str, float]:
    """
    Optimize ticket price for a single period.

    max_τ  Rev^ticket_t(τ_t)
    s.t.   τ_min ≤ τ_t ≤ τ_max
           Dem_g ≤ Capacity

    Parameters
    ----------
    games : List[Dict]
        Home games in this period
    S_t : float
        Team Elo
    Star_t : float
        Team star power
    league_pop : float
        League popularity
    period : int
        Current period
    params : DynamicPricingParameters
        Pricing parameters
    fixed_u : float
        Fixed sports investment (for consistency)
    fixed_m : float
        Fixed marketing investment (for consistency)

    Returns
    -------
    Dict[str, float]
        Optimization results with keys:
        - optimal_tau: optimal price multiplier
        - revenue: expected revenue
        - avg_demand: average demand per game
        - avg_attendance_rate: average attendance rate
    """
    # Objective: maximize revenue (minimize negative revenue)
    def objective(tau):
        revenue, _ = calculate_period_ticket_revenue(
            tau_t=tau[0],
            games=games,
            S_t=S_t,
            Star_t=Star_t,
            league_pop=league_pop,
            period=period,
            params=params,
        )
        return -revenue  # Minimize negative revenue

    # Bounds
    bounds = [(params.tau_min, params.tau_max)]

    # Initial guess (start at 1.0)
    x0 = [1.0]

    # Optimize
    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
    )

    # Extract results
    optimal_tau = result.x[0]
    revenue, demands = calculate_period_ticket_revenue(
        tau_t=optimal_tau,
        games=games,
        S_t=S_t,
        Star_t=Star_t,
        league_pop=league_pop,
        period=period,
        params=params,
    )

    avg_demand = np.mean(demands)
    avg_attendance_rate = avg_demand / params.capacity

    return {
        'optimal_tau': optimal_tau,
        'revenue': revenue,
        'avg_demand': avg_demand,
        'avg_attendance_rate': avg_attendance_rate,
        'demands': demands,
    }


def optimize_season_ticket_pricing(
    S_0: float,
    Star_0: float,
    league_pop: float,
    num_home_games: int,
    params: DynamicPricingParameters,
) -> Dict[str, float]:
    """
    Optimize season ticket pricing.

    max_τ^ST  Rev^ST = τ^ST * p_0 * num_games * Dem^ST
    s.t.      τ_ST_min ≤ τ^ST ≤ τ_ST_max
              Dem^ST ≤ 0.8 * Capacity

    Parameters
    ----------
    S_0 : float
        Team Elo at season start
    Star_0 : float
        Team star power at season start
    league_pop : float
        League popularity
    num_home_games : int
        Number of home games in season
    params : DynamicPricingParameters
        Pricing parameters

    Returns
    -------
    Dict[str, float]
        Optimization results
    """
    # Objective: maximize ST revenue
    def objective(tau_ST):
        demand_ST = calculate_season_ticket_demand(
            tau_ST=tau_ST[0],
            S_0=S_0,
            Star_0=Star_0,
            league_pop=league_pop,
            params=params,
        )
        # ST revenue = price * demand * num_games
        price_ST = tau_ST[0] * params.base_price * num_home_games
        revenue_ST = price_ST * demand_ST
        return -revenue_ST

    # Bounds
    bounds = [(params.tau_ST_min, params.tau_ST_max)]

    # Initial guess
    x0 = [0.9]  # ST typically discounted

    # Optimize
    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
    )

    # Extract results
    optimal_tau_ST = result.x[0]
    demand_ST = calculate_season_ticket_demand(
        tau_ST=optimal_tau_ST,
        S_0=S_0,
        Star_0=Star_0,
        league_pop=league_pop,
        params=params,
    )
    price_ST = optimal_tau_ST * params.base_price * num_home_games
    revenue_ST = price_ST * demand_ST

    return {
        'optimal_tau_ST': optimal_tau_ST,
        'demand_ST': demand_ST,
        'revenue_ST': revenue_ST,
        'ST_rate': demand_ST / params.capacity,
    }


def simulate_dynamic_pricing_season(
    schedule: pd.DataFrame,
    S_trajectory: List[float],
    Star_trajectory: List[float],
    league_pop: float,
    params: DynamicPricingParameters,
) -> pd.DataFrame:
    """
    Simulate full season with dynamic pricing.

    Parameters
    ----------
    schedule : pd.DataFrame
        Game schedule with columns: period, opponent_elo, is_home
    S_trajectory : List[float]
        Team Elo by period
    Star_trajectory : List[float]
        Team star power by period
    league_pop : float
        League popularity
    params : DynamicPricingParameters
        Pricing parameters

    Returns
    -------
    pd.DataFrame
        Results by period with optimal pricing decisions
    """
    # Group games by period
    periods = schedule['period'].unique()
    results = []

    for t, period in enumerate(periods):
        # Get home games for this period
        period_games = schedule[
            (schedule['period'] == period) & (schedule['is_home'] == True)
        ]

        if len(period_games) == 0:
            continue

        # Convert to list of dicts
        games = period_games[['opponent_elo']].to_dict('records')

        # Optimize pricing for this period
        result = optimize_single_period_pricing(
            games=games,
            S_t=S_trajectory[t],
            Star_t=Star_trajectory[t],
            league_pop=league_pop,
            period=t,
            params=params,
        )

        # Calculate elasticity for this period
        epsilon_t = calculate_price_elasticity(t, params)

        # Store results
        results.append({
            'period': period,
            'optimal_tau': result['optimal_tau'],
            'revenue': result['revenue'],
            'avg_demand': result['avg_demand'],
            'avg_attendance_rate': result['avg_attendance_rate'],
            'price_elasticity': epsilon_t,
            'num_games': len(games),
        })

    return pd.DataFrame(results)


def main() -> int:
    params = DynamicPricingParameters()

    print("=" * 80)
    print("Task 4: Dynamic Ticket Pricing Model - Example")
    print("=" * 80)

    print("\n1. Price Elasticity Decay Over Season:")
    print("-" * 80)
    for period in range(3):
        epsilon = calculate_price_elasticity(period, params)
        print(f"Period {period}: epsilon_t = {epsilon:.3f}")

    print("\n2. Single Game Demand at Different Prices:")
    print("-" * 80)
    for tau in [1.0, 1.1, 1.2]:
        demand = calculate_single_game_demand(
            tau=tau,
            S_t=1600,
            Star_t=50,
            S_opp=1550,
            league_pop=1.0,
            period=0,
            params=params,
        )
        revenue = tau * params.base_price * demand
        print(f"tau = {tau:.1f}: Demand = {demand:,.0f}, Revenue = ${revenue:,.0f}")

    print("\n3. Season Ticket Optimization:")
    print("-" * 80)
    ST_result = optimize_season_ticket_pricing(
        S_0=1600,
        Star_0=50,
        league_pop=1.0,
        num_home_games=20,
        params=params,
    )
    print(f"Optimal tau_ST: {ST_result['optimal_tau_ST']:.3f}")
    print(f"ST Demand: {ST_result['demand_ST']:,.0f}")
    print(f"ST Revenue: ${ST_result['revenue_ST']:,.0f}")
    print(f"ST Rate: {ST_result['ST_rate']:.1%}")

    print("\n4. Attendance-Dependent Home Advantage:")
    print("-" * 80)
    for attendance_rate in [0.5, 0.75, 1.0]:
        H_eff = calculate_modified_home_advantage(attendance_rate, params)
        win_prob = calculate_win_probability_with_attendance(
            S_home=1600,
            S_away=1550,
            attendance_rate=attendance_rate,
            params=params,
        )
        print(f"Attendance {attendance_rate:.0%}: H_eff = {H_eff:+.1f}, Win Prob = {win_prob:.1%}")

    print("\n" + "=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    # Example usage
    print("=" * 80)
    print("Task 4: Dynamic Ticket Pricing Model - Example")
    print("=" * 80)

    params = DynamicPricingParameters()

    # Example 1: Price elasticity decay
    print("\n1. Price Elasticity Decay Over Season:")
    print("-" * 80)
    for period in range(3):
        epsilon = calculate_price_elasticity(period, params)
        print(f"Period {period}: ε_t = {epsilon:.3f}")

    # Example 2: Single game demand
    print("\n2. Single Game Demand at Different Prices:")
    print("-" * 80)
    S_t = 1600
    Star_t = 50
    S_opp = 1550
    league_pop = 1.0
    period = 0

    for tau in [0.8, 1.0, 1.2]:
        demand = calculate_single_game_demand(
            tau=tau,
            S_t=S_t,
            Star_t=Star_t,
            S_opp=S_opp,
            league_pop=league_pop,
            period=period,
            params=params,
        )
        revenue = tau * params.base_price * demand
        print(f"τ = {tau:.1f}: Demand = {demand:,.0f}, Revenue = ${revenue:,.0f}")

    # Example 3: Season ticket optimization
    print("\n3. Season Ticket Optimization:")
    print("-" * 80)
    ST_result = optimize_season_ticket_pricing(
        S_0=1600,
        Star_0=50,
        league_pop=1.0,
        num_home_games=20,
        params=params,
    )
    print(f"Optimal τ^ST: {ST_result['optimal_tau_ST']:.3f}")
    print(f"ST Demand: {ST_result['demand_ST']:,.0f}")
    print(f"ST Revenue: ${ST_result['revenue_ST']:,.0f}")
    print(f"ST Rate: {ST_result['ST_rate']:.1%}")

    # Example 4: Attendance-dependent home advantage
    print("\n4. Attendance-Dependent Home Advantage:")
    print("-" * 80)
    for attendance_rate in [0.5, 0.75, 1.0]:
        H_eff = calculate_modified_home_advantage(attendance_rate, params)
        win_prob = calculate_win_probability_with_attendance(
            S_home=1600,
            S_away=1550,
            attendance_rate=attendance_rate,
            params=params,
        )
        print(f"Attendance {attendance_rate:.0%}: H_eff = {H_eff:+.1f}, Win Prob = {win_prob:.1%}")

    print("\n" + "=" * 80)
