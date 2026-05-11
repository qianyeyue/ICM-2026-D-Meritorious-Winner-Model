"""
Task 4: Dynamic Ticket Pricing Model

This package implements dynamic ticket pricing strategies for WNBA team management,
including time-varying price elasticity, season ticket optimization, and
attendance-dependent home advantage.

Key Components
--------------
1. task4_dynamic_pricing: Core pricing model with demand functions
2. integrated_pricing_model: Integration with MPC framework
3. mpc_integration: Full MPC loop integration

Quick Start
-----------
>>> from task4 import DynamicPricingParameters, IntegratedPricingModel
>>>
>>> # Initialize model
>>> params = DynamicPricingParameters()
>>> pricing_model = IntegratedPricingModel(params)
>>>
>>> # Optimize season tickets
>>> ST_result = pricing_model.initialize_season_tickets(
...     S_0=1600, Star_0=50, league_pop=1.0, num_home_games=20
... )
>>>
>>> # Optimize period pricing
>>> games = [{'opponent_elo': 1550}, {'opponent_elo': 1580}]
>>> result = pricing_model.optimize_period_pricing(
...     games=games, S_t=1600, Star_t=50, league_pop=1.0, period=0
... )

Mathematical Model
------------------
Price Elasticity Decay:
    ε_t = ε_0 * exp(-λ * t)

Demand Function:
    ln(Dem_t) = β_0 - ε_t * ln(τ_t) + β^T * x_t

Modified Home Advantage:
    Δ_g = (S_home - S_away) + H * (2 * Ω_g - 1)
    where Ω_g = Dem_g / Capacity

Revenue:
    Rev^ticket_t = Σ_g (τ_t * p_0 * Dem_g) + Rev^ST

"""

from .task4_dynamic_pricing import (
    DynamicPricingParameters,
    calculate_attendance_rate,
    calculate_modified_home_advantage,
    calculate_period_ticket_revenue,
    calculate_price_elasticity,
    calculate_season_ticket_demand,
    calculate_single_game_demand,
    calculate_win_probability_with_attendance,
    optimize_season_ticket_pricing,
    optimize_single_period_pricing,
    simulate_dynamic_pricing_season,
)

from .integrated_pricing_model import (
    IntegratedPricingModel,
    integrate_pricing_into_mpc,
)

__all__ = [
    # Parameters
    'DynamicPricingParameters',

    # Core pricing functions
    'calculate_price_elasticity',
    'calculate_single_game_demand',
    'calculate_season_ticket_demand',
    'calculate_attendance_rate',
    'calculate_modified_home_advantage',
    'calculate_win_probability_with_attendance',
    'calculate_period_ticket_revenue',

    # Optimization functions
    'optimize_single_period_pricing',
    'optimize_season_ticket_pricing',
    'simulate_dynamic_pricing_season',

    # Integration classes
    'IntegratedPricingModel',
    'integrate_pricing_into_mpc',
]

__version__ = '1.0.0'
__description__ = 'Dynamic Ticket Pricing Model for WNBA Team Management'
