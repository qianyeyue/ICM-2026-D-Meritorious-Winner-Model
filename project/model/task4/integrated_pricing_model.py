"""
Task 4: Integrated Dynamic Pricing Model

This module integrates dynamic pricing into the main MPC framework.
It extends the base model by:
1. Making τ_t a period-specific decision variable (not season-wide)
2. Coupling attendance rate to home advantage
3. Adding season ticket revenue stream
4. Implementing time-varying price elasticity

Integration with Main Model
----------------------------
- Replaces fixed τ with dynamic τ_t optimization
- Modifies win probability calculation to include attendance feedback
- Adds season ticket revenue to total revenue
- Maintains compatibility with SA+MC optimization framework
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Use relative imports when imported as package, absolute when run directly
try:
    from .task4_dynamic_pricing import (
        DynamicPricingParameters,
        calculate_attendance_rate,
        calculate_modified_home_advantage,
        calculate_period_ticket_revenue,
        calculate_price_elasticity,
        calculate_season_ticket_demand,
        calculate_win_probability_with_attendance,
    )
except ImportError:
    from task4_dynamic_pricing import (
        DynamicPricingParameters,
        calculate_attendance_rate,
        calculate_modified_home_advantage,
        calculate_period_ticket_revenue,
        calculate_price_elasticity,
        calculate_season_ticket_demand,
        calculate_win_probability_with_attendance,
    )


class IntegratedPricingModel:
    """
    Integrated model combining dynamic pricing with MPC framework.

    This class wraps the dynamic pricing functions and provides
    interfaces compatible with the main simulation loop.
    """

    def __init__(
        self,
        pricing_params: Optional[DynamicPricingParameters] = None,
    ):
        """
        Initialize integrated pricing model.

        Parameters
        ----------
        pricing_params : DynamicPricingParameters, optional
            Pricing parameters (uses defaults if not provided)
        """
        self.pricing_params = pricing_params or DynamicPricingParameters()

        # Cache for season ticket revenue (calculated once at season start)
        self.season_ticket_revenue: Optional[float] = None
        self.season_ticket_demand: Optional[float] = None

    def initialize_season_tickets(
        self,
        S_0: float,
        Star_0: float,
        league_pop: float,
        num_home_games: int,
        tau_ST: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Initialize season ticket sales at season start.

        Parameters
        ----------
        S_0 : float
            Team Elo at season start
        Star_0 : float
            Team star power at season start
        league_pop : float
            League popularity
        num_home_games : int
            Total home games in season
        tau_ST : float, optional
            Season ticket price multiplier (optimized if not provided)

        Returns
        -------
        Dict[str, float]
            Season ticket results
        """
        if tau_ST is None:
            # Optimize season ticket price
            try:
                from .task4_dynamic_pricing import optimize_season_ticket_pricing
            except ImportError:
                from task4_dynamic_pricing import optimize_season_ticket_pricing

            result = optimize_season_ticket_pricing(
                S_0=S_0,
                Star_0=Star_0,
                league_pop=league_pop,
                num_home_games=num_home_games,
                params=self.pricing_params,
            )
            tau_ST = result['optimal_tau_ST']
            self.season_ticket_demand = result['demand_ST']
            self.season_ticket_revenue = result['revenue_ST']
        else:
            # Use provided price
            self.season_ticket_demand = calculate_season_ticket_demand(
                tau_ST=tau_ST,
                S_0=S_0,
                Star_0=Star_0,
                league_pop=league_pop,
                params=self.pricing_params,
            )
            price_ST = tau_ST * self.pricing_params.base_price * num_home_games
            self.season_ticket_revenue = price_ST * self.season_ticket_demand

        return {
            'tau_ST': tau_ST,
            'demand_ST': self.season_ticket_demand,
            'revenue_ST': self.season_ticket_revenue,
            'ST_rate': self.season_ticket_demand / self.pricing_params.capacity,
        }

    def calculate_period_revenue_with_pricing(
        self,
        tau_t: float,
        games: List[Dict],
        S_t: float,
        Star_t: float,
        league_pop: float,
        period: int,
    ) -> Dict[str, float]:
        """
        Calculate period revenue with dynamic pricing.

        This replaces the standard ticket revenue calculation in the main model.

        Parameters
        ----------
        tau_t : float
            Price multiplier for this period
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

        Returns
        -------
        Dict[str, float]
            Revenue breakdown with keys:
            - ticket_revenue: single-game ticket revenue
            - ST_revenue_allocated: season ticket revenue allocated to this period
            - total_ticket_revenue: total ticket revenue
            - avg_attendance_rate: average attendance rate
            - demands: list of demands per game
        """
        # Calculate single-game ticket revenue
        ticket_revenue, demands = calculate_period_ticket_revenue(
            tau_t=tau_t,
            games=games,
            S_t=S_t,
            Star_t=Star_t,
            league_pop=league_pop,
            period=period,
            params=self.pricing_params,
        )

        # Allocate season ticket revenue to this period
        # (proportional to number of games)
        if self.season_ticket_revenue is not None:
            # Assume season ticket revenue is spread across all periods
            # Here we allocate based on number of games in this period
            ST_revenue_allocated = self.season_ticket_revenue * len(games) / 20  # Assume 20 home games total
        else:
            ST_revenue_allocated = 0.0

        # Calculate average attendance rate
        avg_demand = np.mean(demands) if demands else 0.0
        avg_attendance_rate = calculate_attendance_rate(
            demand=avg_demand,
            params=self.pricing_params,
        )

        return {
            'ticket_revenue': ticket_revenue,
            'ST_revenue_allocated': ST_revenue_allocated,
            'total_ticket_revenue': ticket_revenue + ST_revenue_allocated,
            'avg_attendance_rate': avg_attendance_rate,
            'demands': demands,
        }

    def calculate_win_probability_with_pricing(
        self,
        S_home: float,
        S_away: float,
        demand: float,
    ) -> float:
        """
        Calculate win probability with attendance-dependent home advantage.

        This replaces the standard Elo win probability calculation.

        Parameters
        ----------
        S_home : float
            Home team Elo
        S_away : float
            Away team Elo
        demand : float
            Expected attendance

        Returns
        -------
        float
            Win probability
        """
        # Calculate attendance rate
        attendance_rate = calculate_attendance_rate(
            demand=demand,
            params=self.pricing_params,
        )

        # Calculate win probability with attendance effect
        win_prob = calculate_win_probability_with_attendance(
            S_home=S_home,
            S_away=S_away,
            attendance_rate=attendance_rate,
            params=self.pricing_params,
        )

        return win_prob

    def get_price_elasticity(self, period: int) -> float:
        """
        Get price elasticity for a given period.

        Parameters
        ----------
        period : int
            Current period

        Returns
        -------
        float
            Price elasticity
        """
        return calculate_price_elasticity(period, self.pricing_params)

    def optimize_period_pricing(
        self,
        games: List[Dict],
        S_t: float,
        Star_t: float,
        league_pop: float,
        period: int,
    ) -> Dict[str, float]:
        """
        Optimize pricing for a single period.

        This can be called by the SA optimizer to find optimal τ_t.

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

        Returns
        -------
        Dict[str, float]
            Optimization results
        """
        try:
            from .task4_dynamic_pricing import optimize_single_period_pricing
        except ImportError:
            from task4_dynamic_pricing import optimize_single_period_pricing

        return optimize_single_period_pricing(
            games=games,
            S_t=S_t,
            Star_t=Star_t,
            league_pop=league_pop,
            period=period,
            params=self.pricing_params,
        )


def integrate_pricing_into_mpc(
    state: Dict[str, float],
    schedule: pd.DataFrame,
    period: int,
    tau_t: float,
    pricing_model: IntegratedPricingModel,
) -> Dict[str, float]:
    """
    Integration function for MPC loop.

    This function is called within the MPC optimization loop to calculate
    revenue and win probabilities with dynamic pricing.

    Parameters
    ----------
    state : Dict[str, float]
        Current state with keys: S_t, B_t, Star_t, Cash_t, D_t
    schedule : pd.DataFrame
        Game schedule
    period : int
        Current period
    tau_t : float
        Price multiplier decision variable
    pricing_model : IntegratedPricingModel
        Pricing model instance

    Returns
    -------
    Dict[str, float]
        Updated revenue and performance metrics
    """
    # Extract state variables
    S_t = state['S_t']
    Star_t = state['Star_t']
    league_pop = state.get('league_pop', 1.0)

    # Get home games for this period
    period_games = schedule[
        (schedule['period'] == period) & (schedule['is_home'] == True)
    ]

    if len(period_games) == 0:
        return {
            'ticket_revenue': 0.0,
            'total_ticket_revenue': 0.0,
            'avg_attendance_rate': 0.0,
            'expected_wins': 0.0,
        }

    # Convert to list of dicts
    games = period_games[['opponent_elo']].to_dict('records')

    # Calculate revenue with pricing
    revenue_result = pricing_model.calculate_period_revenue_with_pricing(
        tau_t=tau_t,
        games=games,
        S_t=S_t,
        Star_t=Star_t,
        league_pop=league_pop,
        period=period,
    )

    # Calculate expected wins with attendance effect
    expected_wins = 0.0
    for i, (_, game) in enumerate(period_games.iterrows()):
        demand = revenue_result['demands'][i]
        win_prob = pricing_model.calculate_win_probability_with_pricing(
            S_home=S_t,
            S_away=game['opponent_elo'],
            demand=demand,
        )
        expected_wins += win_prob

    return {
        'ticket_revenue': revenue_result['ticket_revenue'],
        'total_ticket_revenue': revenue_result['total_ticket_revenue'],
        'avg_attendance_rate': revenue_result['avg_attendance_rate'],
        'expected_wins': expected_wins,
        'demands': revenue_result['demands'],
    }


if __name__ == "__main__":
    # Example: Integration with MPC
    print("=" * 80)
    print("Task 4: Integrated Pricing Model - Example")
    print("=" * 80)

    # Initialize pricing model
    pricing_model = IntegratedPricingModel()

    # Example state
    state = {
        'S_t': 1600,
        'B_t': 1.5,
        'Star_t': 50,
        'Cash_t': 10_000_000,
        'D_t': 0,
        'league_pop': 1.0,
    }

    # Example schedule
    schedule = pd.DataFrame({
        'period': [0, 0, 0, 1, 1, 1, 2, 2, 2],
        'opponent_elo': [1550, 1580, 1520, 1600, 1540, 1590, 1610, 1530, 1570],
        'is_home': [True] * 9,
    })

    # Initialize season tickets
    print("\n1. Season Ticket Initialization:")
    print("-" * 80)
    ST_result = pricing_model.initialize_season_tickets(
        S_0=state['S_t'],
        Star_0=state['Star_t'],
        league_pop=state['league_pop'],
        num_home_games=20,
    )
    print(f"Optimal τ^ST: {ST_result['tau_ST']:.3f}")
    print(f"ST Demand: {ST_result['demand_ST']:,.0f}")
    print(f"ST Revenue: ${ST_result['revenue_ST']:,.0f}")

    # Simulate pricing across periods
    print("\n2. Dynamic Pricing Across Periods:")
    print("-" * 80)
    print(f"{'Period':<8} {'τ_t':<8} {'Revenue':<15} {'Attendance':<12} {'Elasticity':<12}")
    print("-" * 80)

    for period in range(3):
        # Optimize pricing for this period
        period_games = schedule[schedule['period'] == period]
        games = period_games[['opponent_elo']].to_dict('records')

        result = pricing_model.optimize_period_pricing(
            games=games,
            S_t=state['S_t'],
            Star_t=state['Star_t'],
            league_pop=state['league_pop'],
            period=period,
        )

        elasticity = pricing_model.get_price_elasticity(period)

        print(f"{period:<8} {result['optimal_tau']:<8.3f} "
              f"${result['revenue']:<14,.0f} {result['avg_attendance_rate']:<12.1%} "
              f"{elasticity:<12.3f}")

    # Compare with fixed pricing
    print("\n3. Comparison: Dynamic vs Fixed Pricing:")
    print("-" * 80)

    total_revenue_dynamic = 0.0
    total_revenue_fixed = 0.0

    for period in range(3):
        period_games = schedule[schedule['period'] == period]
        games = period_games[['opponent_elo']].to_dict('records')

        # Dynamic pricing
        result_dynamic = pricing_model.optimize_period_pricing(
            games=games,
            S_t=state['S_t'],
            Star_t=state['Star_t'],
            league_pop=state['league_pop'],
            period=period,
        )
        total_revenue_dynamic += result_dynamic['revenue']

        # Fixed pricing (τ = 1.0)
        result_fixed = pricing_model.calculate_period_revenue_with_pricing(
            tau_t=1.0,
            games=games,
            S_t=state['S_t'],
            Star_t=state['Star_t'],
            league_pop=state['league_pop'],
            period=period,
        )
        total_revenue_fixed += result_fixed['ticket_revenue']

    print(f"Dynamic Pricing Total: ${total_revenue_dynamic:,.0f}")
    print(f"Fixed Pricing Total:   ${total_revenue_fixed:,.0f}")
    print(f"Improvement:           ${total_revenue_dynamic - total_revenue_fixed:,.0f} "
          f"({(total_revenue_dynamic / total_revenue_fixed - 1) * 100:+.1f}%)")

    # Demonstrate attendance effect on win probability
    print("\n4. Attendance Effect on Win Probability:")
    print("-" * 80)
    print(f"{'Demand':<12} {'Attendance':<12} {'Win Prob':<12} {'Home Adv':<12}")
    print("-" * 80)

    for demand in [6000, 9000, 12000]:
        attendance_rate = calculate_attendance_rate(
            demand=demand,
            params=pricing_model.pricing_params,
        )
        win_prob = pricing_model.calculate_win_probability_with_pricing(
            S_home=1600,
            S_away=1550,
            demand=demand,
        )
        H_eff = calculate_modified_home_advantage(
            attendance_rate=attendance_rate,
            params=pricing_model.pricing_params,
        )
        print(f"{demand:<12,.0f} {attendance_rate:<12.1%} {win_prob:<12.1%} {H_eff:<12.1f}")

    print("\n" + "=" * 80)
