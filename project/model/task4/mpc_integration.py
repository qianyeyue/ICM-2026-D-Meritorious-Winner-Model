"""
Task 4: MPC Integration Module

This module provides the interface to integrate Task 4 dynamic pricing
into the main MPC simulation framework from Task 1.

Key Integration Points:
1. Replace fixed 蟿 with period-specific 蟿_t optimization
2. Modify revenue calculation to use dynamic pricing
3. Update win probability to include attendance feedback
4. Add season ticket revenue stream
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from task4.task4_dynamic_pricing import (
    DynamicPricingParameters,
    calculate_period_ticket_revenue,
    calculate_price_elasticity,
    calculate_season_ticket_demand,
    calculate_win_probability_with_attendance,
)


class Task4MPCIntegration:
    """
    Integration layer between Task 4 pricing model and Task 1 MPC framework.

    This class provides methods that can be directly called from the main
    MPC simulation loop, replacing the original fixed-price calculations.
    """

    def __init__(
        self,
        pricing_params: Optional[DynamicPricingParameters] = None,
        enable_season_tickets: bool = True,
        enable_attendance_feedback: bool = True,
    ):
        """
        Initialize Task 4 integration.

        Parameters
        ----------
        pricing_params : DynamicPricingParameters, optional
            Pricing parameters
        enable_season_tickets : bool, default=True
            Whether to include season ticket revenue
        enable_attendance_feedback : bool, default=True
            Whether to use attendance-dependent home advantage
        """
        self.pricing_params = pricing_params or DynamicPricingParameters()
        self.enable_season_tickets = enable_season_tickets
        self.enable_attendance_feedback = enable_attendance_feedback

        # Season ticket cache
        self.ST_revenue_total: Optional[float] = None
        self.ST_demand: Optional[float] = None
        self.num_home_games_total: Optional[int] = None

    def initialize_season(
        self,
        S_0: float,
        Star_0: float,
        league_pop: float,
        num_home_games: int,
        optimize_ST: bool = True,
    ) -> Dict[str, float]:
        """
        Initialize season-level decisions (season tickets).

        This should be called once at the beginning of each season.

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
        optimize_ST : bool, default=True
            Whether to optimize ST price (vs using default)

        Returns
        -------
        Dict[str, float]
            Season ticket results
        """
        self.num_home_games_total = num_home_games

        if not self.enable_season_tickets:
            self.ST_revenue_total = 0.0
            self.ST_demand = 0.0
            return {
                'tau_ST': 0.0,
                'demand_ST': 0.0,
                'revenue_ST': 0.0,
                'ST_rate': 0.0,
            }

        if optimize_ST:
            # Optimize season ticket price
            from task4.task4_dynamic_pricing import optimize_season_ticket_pricing

            result = optimize_season_ticket_pricing(
                S_0=S_0,
                Star_0=Star_0,
                league_pop=league_pop,
                num_home_games=num_home_games,
                params=self.pricing_params,
            )
            tau_ST = result['optimal_tau_ST']
            self.ST_demand = result['demand_ST']
            self.ST_revenue_total = result['revenue_ST']
        else:
            # Use default ST price (10% discount)
            tau_ST = 0.9
            self.ST_demand = calculate_season_ticket_demand(
                tau_ST=tau_ST,
                S_0=S_0,
                Star_0=Star_0,
                league_pop=league_pop,
                params=self.pricing_params,
            )
            price_ST = tau_ST * self.pricing_params.base_price * num_home_games
            self.ST_revenue_total = price_ST * self.ST_demand

        return {
            'tau_ST': tau_ST,
            'demand_ST': self.ST_demand,
            'revenue_ST': self.ST_revenue_total,
            'ST_rate': self.ST_demand / self.pricing_params.capacity,
        }

    def calculate_period_revenue(
        self,
        tau_t: float,
        home_games: List[Dict],
        S_t: float,
        Star_t: float,
        league_pop: float,
        period: int,
    ) -> Dict[str, float]:
        """
        Calculate ticket revenue for a period with dynamic pricing.

        This replaces the original ticket revenue calculation in MPC.

        Parameters
        ----------
        tau_t : float
            Price multiplier for this period (decision variable)
        home_games : List[Dict]
            Home games with 'opponent_elo' key
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
            - single_game_revenue: revenue from single-game tickets
            - ST_revenue_allocated: season ticket revenue for this period
            - total_ticket_revenue: total ticket revenue
            - total_attendance: total attendance across games
            - avg_attendance_rate: average attendance rate
            - demands: list of demands per game
        """
        # Calculate single-game ticket revenue
        single_game_revenue, demands = calculate_period_ticket_revenue(
            tau_t=tau_t,
            games=home_games,
            S_t=S_t,
            Star_t=Star_t,
            league_pop=league_pop,
            period=period,
            params=self.pricing_params,
        )

        # Allocate season ticket revenue to this period
        if self.enable_season_tickets and self.ST_revenue_total is not None:
            # Proportional allocation based on number of games
            ST_revenue_allocated = (
                self.ST_revenue_total * len(home_games) / self.num_home_games_total
            )
        else:
            ST_revenue_allocated = 0.0

        # Calculate attendance metrics
        total_attendance = sum(demands)
        avg_attendance_rate = np.mean(demands) / self.pricing_params.capacity if demands else 0.0

        return {
            'single_game_revenue': single_game_revenue,
            'ST_revenue_allocated': ST_revenue_allocated,
            'total_ticket_revenue': single_game_revenue + ST_revenue_allocated,
            'total_attendance': total_attendance,
            'avg_attendance_rate': avg_attendance_rate,
            'demands': demands,
        }

    def calculate_game_win_probability(
        self,
        S_home: float,
        S_away: float,
        demand: float,
    ) -> float:
        """
        Calculate win probability with attendance feedback.

        This replaces the original Elo win probability calculation.

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
        if self.enable_attendance_feedback:
            # Use attendance-dependent home advantage
            attendance_rate = min(demand / self.pricing_params.capacity, 1.0)
            win_prob = calculate_win_probability_with_attendance(
                S_home=S_home,
                S_away=S_away,
                attendance_rate=attendance_rate,
                params=self.pricing_params,
            )
        else:
            # Use standard Elo (fixed home advantage)
            H = self.pricing_params.H_base
            delta = (S_home - S_away) + H
            win_prob = 1.0 / (1.0 + 10 ** (-delta / 400))

        return win_prob

    def simulate_period_with_pricing(
        self,
        tau_t: float,
        home_games: List[Dict],
        S_t: float,
        Star_t: float,
        league_pop: float,
        period: int,
        simulate_games: bool = True,
    ) -> Dict[str, float]:
        """
        Simulate a full period with dynamic pricing.

        This is the main interface for MPC integration.

        Parameters
        ----------
        tau_t : float
            Price multiplier decision
        home_games : List[Dict]
            Home games with 'opponent_elo' key
        S_t : float
            Team Elo
        Star_t : float
            Team star power
        league_pop : float
            League popularity
        period : int
            Current period
        simulate_games : bool, default=True
            Whether to simulate game outcomes (vs just expected values)

        Returns
        -------
        Dict[str, float]
            Period results with keys:
            - ticket_revenue: total ticket revenue
            - total_attendance: total attendance
            - avg_attendance_rate: average attendance rate
            - wins: actual wins (if simulated) or expected wins
            - win_probs: list of win probabilities
            - demands: list of demands per game
        """
        # Calculate revenue
        revenue_result = self.calculate_period_revenue(
            tau_t=tau_t,
            home_games=home_games,
            S_t=S_t,
            Star_t=Star_t,
            league_pop=league_pop,
            period=period,
        )

        # Calculate win probabilities and simulate games
        win_probs = []
        wins = 0

        for i, game in enumerate(home_games):
            demand = revenue_result['demands'][i]

            # Calculate win probability
            win_prob = self.calculate_game_win_probability(
                S_home=S_t,
                S_away=game['opponent_elo'],
                demand=demand,
            )
            win_probs.append(win_prob)

            # Simulate game outcome
            if simulate_games:
                outcome = np.random.binomial(1, win_prob)
                wins += outcome

        # If not simulating, use expected wins
        if not simulate_games:
            wins = sum(win_probs)

        return {
            'ticket_revenue': revenue_result['total_ticket_revenue'],
            'single_game_revenue': revenue_result['single_game_revenue'],
            'ST_revenue_allocated': revenue_result['ST_revenue_allocated'],
            'total_attendance': revenue_result['total_attendance'],
            'avg_attendance_rate': revenue_result['avg_attendance_rate'],
            'wins': wins,
            'expected_wins': sum(win_probs),
            'win_probs': win_probs,
            'demands': revenue_result['demands'],
        }

    def get_elasticity_for_period(self, period: int) -> float:
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


def example_mpc_integration():
    """
    Example demonstrating how to integrate Task 4 into MPC loop.
    """
    print("=" * 80)
    print("Task 4: MPC Integration Example")
    print("=" * 80)
    print()

    # Initialize integration
    task4 = Task4MPCIntegration()

    # Season initialization
    print("1. Season Initialization:")
    print("-" * 80)

    S_0 = 1600
    Star_0 = 50
    league_pop = 1.0
    num_home_games = 21

    ST_result = task4.initialize_season(
        S_0=S_0,
        Star_0=Star_0,
        league_pop=league_pop,
        num_home_games=num_home_games,
    )

    print(f"Season Ticket Price Multiplier: {ST_result['tau_ST']:.3f}")
    print(f"Season Ticket Revenue:          ${ST_result['revenue_ST']:,.0f}")
    print(f"Season Ticket Demand:           {ST_result['demand_ST']:,.0f} seats")
    print()

    # Simulate 3 periods
    print("2. Period-by-Period Simulation:")
    print("-" * 80)
    print(f"{'Period':<8} {'蟿_t':<8} {'Revenue':<15} {'Attendance':<12} {'Wins':<8} {'E[Wins]':<10}")
    print("-" * 80)

    # Example state trajectory
    S_trajectory = [1600, 1610, 1620]
    Star_trajectory = [50, 52, 54]

    total_revenue = ST_result['revenue_ST']
    total_wins = 0
    total_expected_wins = 0

    for period in range(3):
        # Example home games for this period
        home_games = [
            {'opponent_elo': 1550},
            {'opponent_elo': 1580},
            {'opponent_elo': 1520},
            {'opponent_elo': 1600},
            {'opponent_elo': 1540},
            {'opponent_elo': 1590},
            {'opponent_elo': 1560},
        ]

        # Optimal price for this period (simplified - in real MPC, this is optimized)
        if period == 0:
            tau_t = 0.95  # Early season discount
        elif period == 1:
            tau_t = 1.00  # Mid season baseline
        else:
            tau_t = 1.10  # Late season premium

        # Simulate period
        result = task4.simulate_period_with_pricing(
            tau_t=tau_t,
            home_games=home_games,
            S_t=S_trajectory[period],
            Star_t=Star_trajectory[period],
            league_pop=league_pop,
            period=period,
            simulate_games=True,
        )

        print(f"{period:<8} {tau_t:<8.2f} ${result['ticket_revenue']:<14,.0f} "
              f"{result['avg_attendance_rate']:<12.1%} {result['wins']:<8.0f} "
              f"{result['expected_wins']:<10.1f}")

        total_revenue += result['ticket_revenue']
        total_wins += result['wins']
        total_expected_wins += result['expected_wins']

    print("-" * 80)
    print(f"{'Total':<8} {'':<8} ${total_revenue:<14,.0f} {'':<12} {total_wins:<8.0f} {total_expected_wins:<10.1f}")
    print()

    # Summary
    print("3. Integration Summary:")
    print("-" * 80)
    print(f"Total Revenue (including ST):    ${total_revenue:,.0f}")
    print(f"Total Wins:                      {total_wins}")
    print(f"Expected Wins:                   {total_expected_wins:.1f}")
    print(f"Win Rate:                        {total_wins / 21:.1%}")
    print()

    print("Key Features Demonstrated:")
    print("  [OK]Season ticket revenue calculated at season start")
    print("  [OK]Period-specific price multipliers (蟿_t)")
    print("  [OK]Time-varying price elasticity")
    print("  [OK]Attendance-dependent home advantage")
    print("  [OK]Seamless integration with MPC state variables")
    print()


def create_mpc_wrapper_functions():
    """
    Create wrapper functions for easy MPC integration.

    These functions can be directly imported and used in the main MPC code.
    """
    print("=" * 80)
    print("MPC Wrapper Functions")
    print("=" * 80)
    print()

    print("Example wrapper functions for MPC integration:")
    print()

    print("""
# In your main MPC simulation file:

from task4.mpc_integration import Task4MPCIntegration

# Initialize at start of simulation
task4_integration = Task4MPCIntegration()

# At season start
ST_result = task4_integration.initialize_season(
    S_0=initial_state['S_0'],
    Star_0=initial_state['Star_0'],
    league_pop=initial_state['league_pop'],
    num_home_games=20
)

# In each period of MPC loop
for period in range(num_periods):
    # Get current state
    state = get_current_state(period)

    # Get home games for this period
    home_games = get_period_home_games(period)

    # Simulate period with pricing decision tau_t
    result = task4_integration.simulate_period_with_pricing(
        tau_t=tau_t,  # This is now a decision variable
        home_games=home_games,
        S_t=state['S_t'],
        Star_t=state['Star_t'],
        league_pop=state['league_pop'],
        period=period,
        simulate_games=True
    )

    # Extract results
    ticket_revenue = result['ticket_revenue']
    wins = result['wins']
    attendance = result['total_attendance']

    # Continue with rest of MPC logic...
    """)


def demonstrate_sa_integration():
    """
    Demonstrate how Task 4 integrates with SA optimization.
    """
    print("=" * 80)
    print("Task 4 + Simulated Annealing Integration")
    print("=" * 80)
    print()

    print("In the SA optimization loop, 蟿_t becomes a decision variable:")
    print()

    print("""
# SA optimization for period t
def evaluate_strategy(tau_t, u_t, m_t, d_t, state, period):
    '''
    Evaluate a strategy with Task 4 pricing.

    Parameters
    ----------
    tau_t : float
        Price multiplier (NEW: period-specific)
    u_t : float
        Sports investment
    m_t : float
        Marketing investment
    d_t : float
        Borrowing decision
    state : Dict
        Current state
    period : int
        Current period

    Returns
    -------
    float
        Objective value J
    '''
    # Calculate revenue with Task 4 pricing
    revenue_result = task4_integration.simulate_period_with_pricing(
        tau_t=tau_t,
        home_games=get_period_games(period),
        S_t=state['S_t'],
        Star_t=Star_t,
        league_pop=state['league_pop'],
        period=period,
        simulate_games=False  # Use expected values in SA
    )

    # Extract metrics
    ticket_revenue = revenue_result['ticket_revenue']
    expected_wins = revenue_result['expected_wins']

    # Calculate other revenues (merch, sponsorship, etc.)
    merch_revenue = calculate_merch_revenue(revenue_result['total_attendance'], Star_t)
    spon_revenue = calculate_spon_revenue(state['B_t'], Star_t, league_pop)

    total_revenue = ticket_revenue + merch_revenue + spon_revenue

    # Calculate costs
    total_cost = calculate_total_cost(u_t, m_t, d_t, state)

    # Calculate profit
    profit = total_revenue - total_cost

    # Calculate objective
    J = profit + lambda_W * expected_wins - lambda_risk * calculate_risk(state)

    return J


# SA loop
current_strategy = {'tau_t': 1.0, 'u_t': 1.5e6, 'm_t': 1.3e6, 'd_t': 0}
temperature = 1000

for iteration in range(max_iterations):
    # Propose new strategy
    new_strategy = perturb_strategy(current_strategy)

    # Evaluate both strategies
    J_current = evaluate_strategy(**current_strategy, state=state, period=period)
    J_new = evaluate_strategy(**new_strategy, state=state, period=period)

    # Accept or reject
    if accept_move(J_current, J_new, temperature):
        current_strategy = new_strategy

    # Cool down
    temperature *= cooling_rate
    """)

    print()
    print("Key Changes from Base Model:")
    print("  1. 蟿 is now 蟿_t (period-specific decision variable)")
    print("  2. Revenue calculation uses Task 4 pricing module")
    print("  3. Win probability includes attendance feedback")
    print("  4. Season ticket revenue added to total revenue")
    print()


if __name__ == "__main__":
    print("\n")
    print("*" * 80)
    print("TASK 4: MPC INTEGRATION EXAMPLES")
    print("*" * 80)
    print("\n")

    example_mpc_integration()
    print("\n")

    create_mpc_wrapper_functions()
    print("\n")

    demonstrate_sa_integration()

    print("\n")
    print("*" * 80)
    print("INTEGRATION EXAMPLES COMPLETED")
    print("*" * 80)
    print()

