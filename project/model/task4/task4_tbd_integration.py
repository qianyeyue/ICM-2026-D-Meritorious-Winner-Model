"""
Task 4: TBD Model Integration for Dynamic Pricing

This module implements the missing components from the paper:
1. State updates with fixed u_t and m_t (competitive/marketing investment)
2. Cross-period demand coupling through brand feedback
3. Full objective function (profit + wins + playoff + terminal value - risk)
4. Simulated Annealing + Monte Carlo optimization

References: Paper lines 1037-1113, Task 1 objective function lines 237-243
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.special import expit

from task4_dynamic_pricing import (
    DynamicPricingParameters,
    calculate_single_game_demand,
    calculate_attendance_rate,
    calculate_win_probability_with_attendance,
)


@dataclass
class TBDIntegrationParameters:
    """Parameters for TBD model integration."""

    # State update parameters (from paper Task 1)
    alpha_u: float = 0.05  # Investment effect on Elo
    rho: float = 0.85  # Brand persistence
    eta_W: float = 0.02  # Wins effect on brand
    eta_m: float = 0.15  # Marketing effect on brand
    eta_star: float = 0.1  # Star effect on brand
    eta_attendance: float = 0.05  # Attendance effect on brand

    kappa_u: float = 0.08  # Investment effect on star value
    kappa_m: float = 0.12  # Marketing effect on star value

    # Objective function weights
    lambda_W: float = 20000.0  # Value per win ($)
    lambda_risk: float = 0.1  # Risk aversion coefficient
    delta: float = 0.95  # Discount factor

    # Playoff parameters
    playoff_threshold: int = 8  # Top 8 teams make playoffs
    playoff_bonus_base: float = 1000000.0  # Base playoff revenue ($)

    # Valuation parameters
    omega_ebitda: float = 8.0  # EBITDA multiple
    omega_brand: float = 500000.0  # Brand value coefficient

    # Elo update parameters
    K_factor: float = 32.0  # Elo K-factor
    elo_carryover: float = 0.75  # Season-to-season carryover


def update_elo_with_investment(
    S_t: float,
    game_results: List[int],
    expected_probs: List[float],
    u_t: float,
    params: TBDIntegrationParameters,
) -> float:
    """
    Update Elo rating with game results and investment.

    S_{t+1} = S_t + K * 危(y_g - p_g) + 伪_u * ln(1 + u_t)

    Parameters
    ----------
    S_t : float
        Current Elo rating
    game_results : List[int]
        Game outcomes (1=win, 0=loss)
    expected_probs : List[float]
        Expected win probabilities
    u_t : float
        Competitive investment ($)
    params : TBDIntegrationParameters
        Model parameters

    Returns
    -------
    float
        Updated Elo rating
    """
    # Elo update from game results
    elo_change = params.K_factor * sum(
        result - prob for result, prob in zip(game_results, expected_probs)
    )

    # Investment effect (diminishing returns)
    investment_effect = params.alpha_u * np.log(1 + u_t / 1e6)  # Normalize to millions

    S_next = S_t + elo_change + investment_effect

    return S_next


def update_brand_with_feedback(
    B_t: float,
    W_t: int,
    avg_attendance_rate: float,
    Star_t: float,
    m_t: float,
    params: TBDIntegrationParameters,
) -> float:
    """
    Update brand value with cross-period feedback.

    B_{t+1} = 蟻 * B_t + 畏_W * W_t + 畏_m * ln(1 + m_t)
              + 畏_star * Star_t + 畏_attendance * 惟_t

    This implements the cross-period coupling mentioned in paper line 1043.

    Parameters
    ----------
    B_t : float
        Current brand index
    W_t : int
        Wins in period
    avg_attendance_rate : float
        Average attendance rate in period
    Star_t : float
        Star power value
    m_t : float
        Marketing investment ($)
    params : TBDIntegrationParameters
        Model parameters

    Returns
    -------
    float
        Updated brand index
    """
    B_next = (
        params.rho * B_t
        + params.eta_W * W_t
        + params.eta_m * np.log(1 + m_t / 1e6)
        + params.eta_star * Star_t / 100  # Normalize
        + params.eta_attendance * avg_attendance_rate
    )

    return B_next


def update_star_value_with_investment(
    Star_t: float,
    u_t: float,
    m_t: float,
    params: TBDIntegrationParameters,
) -> float:
    """
    Update star value with investment effects.

    Star_{t+1} = Star_t * (1 + 魏_u * ln(1 + u_t) + 魏_m * ln(1 + m_t))

    Parameters
    ----------
    Star_t : float
        Current star value
    u_t : float
        Competitive investment ($)
    m_t : float
        Marketing investment ($)
    params : TBDIntegrationParameters
        Model parameters

    Returns
    -------
    float
        Updated star value
    """
    multiplier = (
        1.0
        + params.kappa_u * np.log(1 + u_t / 1e6)
        + params.kappa_m * np.log(1 + m_t / 1e6)
    )

    return Star_t * multiplier


def calculate_playoff_probability(
    expected_wins: float,
    league_standings: List[float],
    params: TBDIntegrationParameters,
) -> float:
    """
    Calculate probability of making playoffs.

    Uses logistic function based on expected wins relative to threshold.

    Parameters
    ----------
    expected_wins : float
        Expected total wins
    league_standings : List[float]
        Expected wins of all teams (sorted)
    params : TBDIntegrationParameters
        Model parameters

    Returns
    -------
    float
        Playoff probability in [0, 1]
    """
    # Find 8th place threshold
    sorted_standings = sorted(league_standings, reverse=True)
    threshold_wins = sorted_standings[params.playoff_threshold - 1]

    # Logistic function: P(playoff) = 蟽(尾 * (W - threshold))
    beta = 0.5  # Steepness parameter
    playoff_prob = expit(beta * (expected_wins - threshold_wins))

    return playoff_prob


def calculate_playoff_revenue(
    playoff_prob: float,
    team_strength: float,
    params: TBDIntegrationParameters,
) -> float:
    """
    Calculate expected playoff revenue.

    E[螤^PO] = P(PO) * E[螤^PO | PO]

    Parameters
    ----------
    playoff_prob : float
        Probability of making playoffs
    team_strength : float
        Team Elo rating (affects playoff performance)
    params : TBDIntegrationParameters
        Model parameters

    Returns
    -------
    float
        Expected playoff revenue ($)
    """
    # Base playoff revenue
    base_revenue = params.playoff_bonus_base

    # Strength multiplier (stronger teams go deeper)
    strength_multiplier = 1.0 + (team_strength - 1500) / 1000

    expected_revenue = playoff_prob * base_revenue * strength_multiplier

    return expected_revenue


def calculate_terminal_value(
    cumulative_ebitda: float,
    final_brand: float,
    league_pop: float,
    params: TBDIntegrationParameters,
) -> float:
    """
    Calculate franchise terminal value.

    V_end = 蠅 * EBITDA + 蠅_B * B_T * L_T

    From paper line 311-314.

    Parameters
    ----------
    cumulative_ebitda : float
        Cumulative EBITDA over horizon ($)
    final_brand : float
        Brand index at end of horizon
    league_pop : float
        League popularity index
    params : TBDIntegrationParameters
        Model parameters

    Returns
    -------
    float
        Terminal franchise value ($)
    """
    ebitda_value = params.omega_ebitda * cumulative_ebitda
    brand_value = params.omega_brand * final_brand * league_pop

    return ebitda_value + brand_value


def calculate_cvar_risk(
    profit_samples: np.ndarray,
    alpha: float = 0.95,
) -> float:
    """
    Calculate Conditional Value-at-Risk (CVaR).

    CVaR_伪 = E[-蟺 | -蟺 鈮?VaR_伪]

    From paper lines 342-348.

    Parameters
    ----------
    profit_samples : np.ndarray
        Monte Carlo profit samples
    alpha : float
        Confidence level (default 0.95)

    Returns
    -------
    float
        CVaR (expected loss in worst 伪% scenarios)
    """
    losses = -profit_samples
    var_threshold = np.percentile(losses, alpha * 100)
    cvar = np.mean(losses[losses >= var_threshold])

    return cvar


def calculate_full_objective(
    tau_trajectory: List[float],
    schedule: pd.DataFrame,
    initial_state: Dict[str, float],
    fixed_u: float,
    fixed_m: float,
    pricing_params: DynamicPricingParameters,
    tbd_params: TBDIntegrationParameters,
    n_mc_samples: int = 1000,
) -> float:
    """
    Calculate full TBD objective function with dynamic pricing.

    J = E[危 未^t(蟺_t + 位_W * W_t)] + 未^T * E[螤^PO]
        + 未^T * E[V_end] - 位_risk * Risk

    From paper lines 237-243.

    Parameters
    ----------
    tau_trajectory : List[float]
        Price multipliers by period
    schedule : pd.DataFrame
        Game schedule
    initial_state : Dict[str, float]
        Initial state (S_0, B_0, Star_0, Cash_0)
    fixed_u : float
        Fixed competitive investment per period ($)
    fixed_m : float
        Fixed marketing investment per period ($)
    pricing_params : DynamicPricingParameters
        Pricing model parameters
    tbd_params : TBDIntegrationParameters
        TBD model parameters
    n_mc_samples : int
        Number of Monte Carlo samples

    Returns
    -------
    float
        Expected objective value
    """
    import pandas as pd

    mc_objectives = []

    for _ in range(n_mc_samples):
        # Initialize state
        S_t = initial_state['S_0']
        B_t = initial_state['B_0']
        Star_t = initial_state['Star_0']
        Cash_t = initial_state['Cash_0']

        cumulative_profit = 0.0
        cumulative_wins = 0
        cumulative_ebitda = 0.0

        periods = schedule['period'].unique()

        for t, period in enumerate(periods):
            # Get period games
            period_games = schedule[schedule['period'] == period]
            home_games = period_games[period_games['is_home'] == True]

            if len(home_games) == 0:
                continue

            # Calculate demand and revenue for each home game
            period_revenue = 0.0
            period_attendance_rates = []
            game_results = []
            expected_probs = []

            for _, game in home_games.iterrows():
                # Calculate demand
                demand = calculate_single_game_demand(
                    tau=tau_trajectory[t],
                    S_t=S_t,
                    Star_t=Star_t,
                    S_opp=game['opponent_elo'],
                    league_pop=initial_state.get('league_pop', 1.0),
                    period=t,
                    params=pricing_params,
                )

                # Calculate revenue
                price = tau_trajectory[t] * pricing_params.base_price
                period_revenue += price * demand

                # Track attendance
                attendance_rate = calculate_attendance_rate(demand, pricing_params)
                period_attendance_rates.append(attendance_rate)

                # Simulate game outcome
                win_prob = calculate_win_probability_with_attendance(
                    S_home=S_t,
                    S_away=game['opponent_elo'],
                    attendance_rate=attendance_rate,
                    params=pricing_params,
                )

                result = np.random.binomial(1, win_prob)
                game_results.append(result)
                expected_probs.append(win_prob)

            # Calculate period profit (simplified)
            period_cost = fixed_u + fixed_m + 500000  # Base operating cost
            period_profit = period_revenue - period_cost

            # Update cumulative metrics
            period_wins = sum(game_results)
            cumulative_profit += tbd_params.delta ** t * period_profit
            cumulative_wins += period_wins
            cumulative_ebitda += period_profit  # Simplified

            # Update state for next period
            avg_attendance = np.mean(period_attendance_rates)

            S_t = update_elo_with_investment(
                S_t, game_results, expected_probs, fixed_u, tbd_params
            )

            B_t = update_brand_with_feedback(
                B_t, period_wins, avg_attendance, Star_t, fixed_m, tbd_params
            )

            Star_t = update_star_value_with_investment(
                Star_t, fixed_u, fixed_m, tbd_params
            )

        # Calculate playoff component
        league_standings = [cumulative_wins] + list(
            np.random.normal(20, 5, 12)
        )  # Simplified
        playoff_prob = calculate_playoff_probability(
            cumulative_wins, league_standings, tbd_params
        )
        playoff_revenue = calculate_playoff_revenue(playoff_prob, S_t, tbd_params)

        # Calculate terminal value
        terminal_value = calculate_terminal_value(
            cumulative_ebitda,
            B_t,
            initial_state.get('league_pop', 1.0),
            tbd_params,
        )

        # Calculate objective for this sample
        objective = (
            cumulative_profit
            + tbd_params.lambda_W * cumulative_wins
            + tbd_params.delta ** len(periods) * playoff_revenue
            + tbd_params.delta ** len(periods) * terminal_value
        )

        mc_objectives.append(objective)

    # Calculate expected objective
    mc_objectives = np.array(mc_objectives)
    expected_objective = np.mean(mc_objectives)

    # Subtract risk penalty
    cvar_risk = calculate_cvar_risk(mc_objectives)
    expected_objective -= tbd_params.lambda_risk * cvar_risk

    return expected_objective


def simulated_annealing_optimization(
    schedule: pd.DataFrame,
    initial_state: Dict[str, float],
    fixed_u: float,
    fixed_m: float,
    pricing_params: DynamicPricingParameters,
    tbd_params: TBDIntegrationParameters,
    n_periods: int,
    T_init: float = 1.0,
    T_min: float = 0.01,
    alpha: float = 0.95,
    max_iter: int = 1000,
    n_mc_samples: int = 100,
) -> Dict[str, any]:
    """
    Optimize pricing strategy using Simulated Annealing + Monte Carlo.

    This implements the SA+MC approach mentioned in paper line 1102.

    Parameters
    ----------
    schedule : pd.DataFrame
        Game schedule
    initial_state : Dict[str, float]
        Initial team state
    fixed_u : float
        Fixed competitive investment ($)
    fixed_m : float
        Fixed marketing investment ($)
    pricing_params : DynamicPricingParameters
        Pricing parameters
    tbd_params : TBDIntegrationParameters
        TBD parameters
    n_periods : int
        Number of decision periods
    T_init : float
        Initial temperature
    T_min : float
        Minimum temperature
    alpha : float
        Cooling rate
    max_iter : int
        Maximum iterations
    n_mc_samples : int
        MC samples per evaluation

    Returns
    -------
    Dict[str, any]
        Optimization results with optimal tau trajectory
    """
    import pandas as pd

    # Initialize with baseline pricing
    current_tau = [1.0] * n_periods
    current_objective = calculate_full_objective(
        current_tau,
        schedule,
        initial_state,
        fixed_u,
        fixed_m,
        pricing_params,
        tbd_params,
        n_mc_samples,
    )

    best_tau = current_tau.copy()
    best_objective = current_objective

    T = T_init
    iteration = 0

    objective_history = [current_objective]

    while T > T_min and iteration < max_iter:
        # Generate neighbor solution
        neighbor_tau = current_tau.copy()

        # Perturb random period
        period_to_change = np.random.randint(n_periods)
        perturbation = np.random.normal(0, 0.1 * T)  # Temperature-dependent
        neighbor_tau[period_to_change] = np.clip(
            neighbor_tau[period_to_change] + perturbation,
            pricing_params.tau_min,
            pricing_params.tau_max,
        )

        # Evaluate neighbor
        neighbor_objective = calculate_full_objective(
            neighbor_tau,
            schedule,
            initial_state,
            fixed_u,
            fixed_m,
            pricing_params,
            tbd_params,
            n_mc_samples,
        )

        # Acceptance criterion
        delta = neighbor_objective - current_objective

        if delta > 0 or np.random.random() < np.exp(delta / T):
            current_tau = neighbor_tau
            current_objective = neighbor_objective

            if current_objective > best_objective:
                best_tau = current_tau.copy()
                best_objective = current_objective

        # Cool down
        T *= alpha
        iteration += 1
        objective_history.append(current_objective)

        if iteration % 100 == 0:
            print(f"Iteration {iteration}: T={T:.4f}, Best J={best_objective:.2e}")

    return {
        'optimal_tau': best_tau,
        'optimal_objective': best_objective,
        'objective_history': objective_history,
        'iterations': iteration,
    }


if __name__ == "__main__":
    import pandas as pd

    print("=" * 80)
    print("Task 4: TBD Integration - Example")
    print("=" * 80)

    # Example: State updates
    print("\n1. State Update with Investment:")
    print("-" * 80)

    tbd_params = TBDIntegrationParameters()

    S_t = 1600
    B_t = 1.5
    Star_t = 50
    u_t = 1500000  # $1.5M investment
    m_t = 1000000  # $1M marketing

    # Simulate period
    game_results = [1, 1, 0, 1, 0]  # 3 wins, 2 losses
    expected_probs = [0.6, 0.7, 0.5, 0.65, 0.55]
    avg_attendance = 0.85

    S_next = update_elo_with_investment(S_t, game_results, expected_probs, u_t, tbd_params)
    B_next = update_brand_with_feedback(B_t, 3, avg_attendance, Star_t, m_t, tbd_params)
    Star_next = update_star_value_with_investment(Star_t, u_t, m_t, tbd_params)

    print(f"Elo: {S_t:.1f} ->{S_next:.1f} (螖={S_next-S_t:+.1f})")
    print(f"Brand: {B_t:.3f} ->{B_next:.3f} (螖={B_next-B_t:+.3f})")
    print(f"Star: {Star_t:.1f} ->{Star_next:.1f} (螖={Star_next-Star_t:+.1f})")

    # Example: Playoff probability
    print("\n2. Playoff Probability Calculation:")
    print("-" * 80)

    expected_wins = 24
    league_standings = [26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15]

    playoff_prob = calculate_playoff_probability(expected_wins, league_standings, tbd_params)
    playoff_rev = calculate_playoff_revenue(playoff_prob, S_next, tbd_params)

    print(f"Expected Wins: {expected_wins}")
    print(f"Playoff Probability: {playoff_prob:.1%}")
    print(f"Expected Playoff Revenue: ${playoff_rev:,.0f}")

    # Example: Terminal value
    print("\n3. Terminal Value Calculation:")
    print("-" * 80)

    cumulative_ebitda = 40000000  # $40M
    final_brand = 2.0
    league_pop = 1.2

    terminal_value = calculate_terminal_value(
        cumulative_ebitda, final_brand, league_pop, tbd_params
    )

    print(f"Cumulative EBITDA: ${cumulative_ebitda/1e6:.1f}M")
    print(f"Final Brand Index: {final_brand:.2f}")
    print(f"Terminal Value: ${terminal_value/1e6:.1f}M")

    # Example: Risk calculation
    print("\n4. CVaR Risk Calculation:")
    print("-" * 80)

    profit_samples = np.random.normal(8000000, 2000000, 1000)
    cvar = calculate_cvar_risk(profit_samples, alpha=0.95)

    print(f"Mean Profit: ${np.mean(profit_samples)/1e6:.2f}M")
    print(f"CVaR (95%): ${cvar/1e6:.2f}M")
    print(f"Risk Penalty: ${tbd_params.lambda_risk * cvar/1e6:.2f}M")

    print("\n" + "=" * 80)

