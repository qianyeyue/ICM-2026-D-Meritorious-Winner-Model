"""
Cost Model for WNBA Teams.

This module implements the cost terms used by the model:
1. Salary costs (player contracts with equity subsidies)
2. Venue and operations costs
3. General & Administrative (G&A) and marketing costs
4. Sports operations costs
5. Financing costs (debt service)

Mathematical Model
------------------
Total Cost:
    Cost_t = Cost^salary_t + Cost^venue_t + Cost^GA_t + Cost^sport_t + Cost^fin_t

1. Salary Costs (contract rigidity + equity subsidies):
    Cost^salary_t = Σ_i x_i(1-θ_i)Salary_i
    where x_i = 1 if player i is on roster, θ_i = equity subsidy ratio

2. Venue & Operations:
    Cost^venue_t = F^venue + v_att·Σ(Dem_g) (home games only)
    where F^venue = fixed venue costs, v_att = variable cost per attendee

3. General & Administrative + Marketing:
    Cost^GA_t = F^GA + m_t + γ_tax·Rev_t
    where F^GA = fixed G&A, m_t = marketing spend, γ_tax = tax rate

4. Sports Operations:
    Cost^sport_t = u_t
    where u_t = training, travel, equipment costs

5. Financing Costs:
    Cost^fin_t = r·D_t
    D_{t+1} = D_t + d_t
    where r = interest rate, D_t = debt balance, d_t = new borrowing

Profit & Cash Flow
------------------
Regular Season Profit:
    π_t = Rev_t - Cost_t

Cash Flow State Equation:
    Cash_{t+1} = Cash_t + π_t + d_t - CapEx_t - DebtService_t

Risk Metrics:
    - Bankruptcy Probability: P(min_t Cash_t < 0)
    - CVaR: CVaR_α(-Cash_T) or CVaR_α(-Σ_t π_t)

Integration with income.py
--------------------------
This module uses revenue calculations from income.py:
    - Rev_t from calculate_period_revenue()
    - Dem_g (attendance) for variable venue costs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SalaryCostParameters:
    """Parameters for salary cost calculation."""

    # Average salary per player (WNBA 2024)
    avg_salary: float = 120000.0  # $120K average
    min_salary: float = 64000.0  # $64K minimum
    max_salary: float = 250000.0  # $250K supermax

    # Roster size
    roster_size: int = 12  # Standard WNBA roster

    # Equity subsidy (league revenue sharing reduces effective salary cost)
    avg_equity_subsidy_ratio: float = 0.0  # θ_i, typically 0 for WNBA


@dataclass
class VenueCostParameters:
    """Parameters for venue and operations costs."""

    # Fixed venue costs (annual lease, maintenance, utilities)
    fixed_venue_cost: float = 2000000.0  # $2M per season

    # Variable cost per attendee (security, cleaning, concessions staff)
    variable_cost_per_attendee: float = 8.0  # $8 per person


@dataclass
class GeneralAdminParameters:
    """Parameters for G&A and marketing costs."""

    # Fixed G&A (front office, admin staff, insurance)
    fixed_ga_cost: float = 3000000.0  # $3M per season

    # Marketing spend (variable, decision variable in optimization)
    base_marketing_spend: float = 500000.0  # $500K baseline

    # Tax rate on revenue (state/local taxes, league fees)
    tax_rate: float = 0.08  # 8% of revenue


@dataclass
class SportsOperationsParameters:
    """Parameters for sports operations costs."""

    # Training facilities, equipment, medical staff, travel
    base_sports_ops_cost: float = 2500000.0  # $2.5M per season

    # Variable travel cost (depends on schedule, assume fixed for simplicity)
    travel_cost_per_game: float = 15000.0  # $15K per away game


@dataclass
class FinancingParameters:
    """Parameters for financing costs."""

    # Interest rate on debt
    interest_rate: float = 0.06  # 6% annual rate

    # Initial debt balance
    initial_debt: float = 0.0  # $0 (assume no initial debt)

    # Debt service (principal repayment, if any)
    debt_service_ratio: float = 0.10  # 10% of debt balance per period


@dataclass
class CashFlowParameters:
    """Parameters for cash flow management."""

    # Minimum cash balance (liquidity requirement)
    min_cash_balance: float = 1000000.0  # $1M minimum

    # Initial cash balance
    initial_cash: float = 5000000.0  # $5M starting cash

    # Capital expenditures (facility upgrades, etc.)
    annual_capex: float = 500000.0  # $500K per year


def update_cash_flow_state(
    current_cash: float,
    profit: float,
    new_debt: float = 0.0,
    capex: float = 0.0,
    debt_service: float = 0.0,
    *,
    min_cash: float = 1000000.0,
    allow_negative: bool = False,
) -> float:
    """
    Update cash flow state equation.

    Cash_{t+1} = Cash_t + π_t + d_t - CapEx_t - DebtService_t

    Parameters
    ----------
    current_cash : float
        Current cash balance (Cash_t)
    profit : float
        Period profit (π_t = Rev_t - Cost_t)
    new_debt : float, default=0.0
        New borrowing (d_t)
    capex : float, default=0.0
        Capital expenditures (CapEx_t)
    debt_service : float, default=0.0
        Debt service payment (principal + interest)
    min_cash : float, default=1000000.0
        Minimum cash balance requirement
    allow_negative : bool, default=False
        Whether to allow negative cash (bankruptcy)

    Returns
    -------
    float
        Updated cash balance (Cash_{t+1})

    Raises
    ------
    ValueError
        If cash becomes negative and allow_negative=False

    Examples
    --------
    >>> # Profitable period with no debt
    >>> update_cash_flow_state(5_000_000, 1_000_000, capex=200_000)
    5800000.0

    >>> # Loss period requiring borrowing
    >>> update_cash_flow_state(2_000_000, -500_000, new_debt=1_000_000, capex=100_000)
    2400000.0
    """
    # Calculate new cash balance
    cash_new = (
        current_cash +
        profit +
        new_debt -
        capex -
        debt_service
    )

    # Check bankruptcy condition
    if not allow_negative and cash_new < min_cash:
        raise ValueError(
            f"Bankruptcy: Cash balance would be {cash_new:.2f} "
            f"(below minimum {min_cash:.2f})"
        )

    return float(cash_new)


def calculate_debt_service(
    debt_balance: float,
    interest_rate: float = 0.06,
    principal_payment_ratio: float = 0.10,
) -> float:
    """
    Calculate debt service payment.

    DebtService_t = r·D_t + principal_payment

    Parameters
    ----------
    debt_balance : float
        Current debt balance (D_t)
    interest_rate : float, default=0.06
        Annual interest rate
    principal_payment_ratio : float, default=0.10
        Ratio of debt balance to pay as principal

    Returns
    -------
    float
        Total debt service payment

    Examples
    --------
    >>> calculate_debt_service(10_000_000, 0.06, 0.10)
    1600000.0
    """
    interest_payment = debt_balance * interest_rate
    principal_payment = debt_balance * principal_payment_ratio
    return float(interest_payment + principal_payment)


def update_debt_balance(
    current_debt: float,
    new_borrowing: float,
    principal_payment: float,
) -> float:
    """
    Update debt balance.

    D_{t+1} = D_t + d_t - principal_payment

    Parameters
    ----------
    current_debt : float
        Current debt balance (D_t)
    new_borrowing : float
        New borrowing (d_t)
    principal_payment : float
        Principal payment

    Returns
    -------
    float
        Updated debt balance (D_{t+1})
    """
    return float(max(0.0, current_debt + new_borrowing - principal_payment))


def calculate_risk_metrics(
    cash_history: np.ndarray,
    profit_history: np.ndarray,
    *,
    alpha: float = 0.05,
    min_cash_threshold: float = 0.0,
) -> Dict[str, float]:
    """
    Calculate risk metrics from simulation results.

    Metrics:
    - Bankruptcy probability: P(min_t Cash_t < threshold)
    - Value at Risk (VaR): α-quantile of losses
    - Conditional Value at Risk (CVaR): Expected loss beyond VaR
    - Profit volatility and downside risk

    Parameters
    ----------
    cash_history : np.ndarray
        Cash balance history, shape (n_simulations, n_periods)
    profit_history : np.ndarray
        Profit history, shape (n_simulations, n_periods)
    alpha : float, default=0.05
        Confidence level for VaR/CVaR (e.g., 0.05 = 95% confidence)
    min_cash_threshold : float, default=0.0
        Bankruptcy threshold

    Returns
    -------
    Dict[str, float]
        Risk metrics dictionary with keys:
        - bankruptcy_prob: Probability of bankruptcy
        - var: Value at Risk (terminal cash)
        - cvar: Conditional Value at Risk
        - min_cash_mean: Average minimum cash across simulations
        - min_cash_std: Std dev of minimum cash
        - profit_mean: Average cumulative profit
        - profit_volatility: Std dev of cumulative profit
        - loss_periods_prob: Probability of loss periods

    Examples
    --------
    >>> # Simulate 1000 scenarios
    >>> cash = np.random.randn(1000, 10).cumsum(axis=1) + 5_000_000
    >>> profit = np.random.randn(1000, 10) * 500_000
    >>> metrics = calculate_risk_metrics(cash, profit)
    >>> metrics['bankruptcy_prob']  # doctest: +SKIP
    0.023
    """
    # Bankruptcy probability
    min_cash_per_sim = np.min(cash_history, axis=1)
    bankruptcy_prob = float(np.mean(min_cash_per_sim < min_cash_threshold))

    # Terminal cash statistics
    terminal_cash = cash_history[:, -1]
    losses = -terminal_cash  # Convert to losses (negative cash = loss)

    # VaR: α-quantile of losses
    var = float(np.quantile(losses, 1 - alpha))

    # CVaR: Expected loss beyond VaR
    tail_losses = losses[losses >= var]
    cvar = float(np.mean(tail_losses)) if len(tail_losses) > 0 else var

    # Minimum cash statistics
    min_cash_mean = float(np.mean(min_cash_per_sim))
    min_cash_std = float(np.std(min_cash_per_sim))

    # Profit statistics
    cumulative_profit = np.sum(profit_history, axis=1)
    profit_mean = float(np.mean(cumulative_profit))
    profit_volatility = float(np.std(cumulative_profit))

    # Loss periods probability
    loss_periods = (profit_history < 0).astype(float)
    loss_periods_prob = float(np.mean(loss_periods))

    return {
        'bankruptcy_prob': bankruptcy_prob,
        'var': var,
        'cvar': cvar,
        'min_cash_mean': min_cash_mean,
        'min_cash_std': min_cash_std,
        'terminal_cash_mean': float(np.mean(terminal_cash)),
        'terminal_cash_std': float(np.std(terminal_cash)),
        'profit_mean': profit_mean,
        'profit_volatility': profit_volatility,
        'loss_periods_prob': loss_periods_prob,
    }


def calculate_salary_cost(
    roster: List[Dict[str, float]],
    params: SalaryCostParameters = None,
) -> float:
    """
    Calculate total salary costs.

    Cost^salary_t = Σ_i x_i(1-θ_i)Salary_i

    Parameters
    ----------
    roster : List[Dict[str, float]]
        List of player contracts, each with keys:
        - 'salary': player salary
        - 'on_roster': 1 if active, 0 if not (x_i)
        - 'equity_subsidy': equity subsidy ratio (θ_i), default 0
    params : SalaryCostParameters, optional
        Salary cost parameters

    Returns
    -------
    float
        Total salary cost
    """
    if params is None:
        params = SalaryCostParameters()

    total_cost = 0.0
    for player in roster:
        x_i = player.get('on_roster', 1)  # Default: on roster
        salary_i = player.get('salary', params.avg_salary)
        theta_i = player.get('equity_subsidy', params.avg_equity_subsidy_ratio)

        # Cost = x_i * (1 - θ_i) * Salary_i
        total_cost += x_i * (1 - theta_i) * salary_i

    return total_cost


def calculate_venue_cost(
    total_attendance: float,
    num_home_games: int,
    params: VenueCostParameters = None,
) -> float:
    """
    Calculate venue and operations costs.

    Cost^venue_t = F^venue + v_att·Σ(Dem_g)

    Parameters
    ----------
    total_attendance : float
        Total attendance across all home games (Σ Dem_g)
    num_home_games : int
        Number of home games
    params : VenueCostParameters, optional
        Venue cost parameters

    Returns
    -------
    float
        Total venue cost
    """
    if params is None:
        params = VenueCostParameters()

    # Fixed cost (prorated by number of games if needed)
    fixed_cost = params.fixed_venue_cost

    # Variable cost
    variable_cost = params.variable_cost_per_attendee * total_attendance

    return fixed_cost + variable_cost


def calculate_ga_cost(
    revenue: float,
    marketing_spend: float = None,
    params: GeneralAdminParameters = None,
) -> float:
    """
    Calculate G&A and marketing costs.

    Cost^GA_t = F^GA + m_t + γ_tax·Rev_t

    Parameters
    ----------
    revenue : float
        Total revenue (Rev_t)
    marketing_spend : float, optional
        Marketing spend (m_t), uses base if not provided
    params : GeneralAdminParameters, optional
        G&A cost parameters

    Returns
    -------
    float
        Total G&A cost
    """
    if params is None:
        params = GeneralAdminParameters()

    if marketing_spend is None:
        marketing_spend = params.base_marketing_spend

    # Fixed G&A
    fixed_ga = params.fixed_ga_cost

    # Marketing spend
    marketing = marketing_spend

    # Tax on revenue
    tax = params.tax_rate * revenue

    return fixed_ga + marketing + tax


def calculate_sports_ops_cost(
    num_games: int,
    num_away_games: int = None,
    params: SportsOperationsParameters = None,
) -> float:
    """
    Calculate sports operations costs.

    Cost^sport_t = u_t (base + travel)

    Parameters
    ----------
    num_games : int
        Total number of games
    num_away_games : int, optional
        Number of away games (for travel costs)
    params : SportsOperationsParameters, optional
        Sports ops cost parameters

    Returns
    -------
    float
        Total sports operations cost
    """
    if params is None:
        params = SportsOperationsParameters()

    # Base sports ops cost
    base_cost = params.base_sports_ops_cost

    # Travel costs (if away games specified)
    if num_away_games is None:
        num_away_games = num_games // 2  # Assume half are away games

    travel_cost = params.travel_cost_per_game * num_away_games

    return base_cost + travel_cost


def calculate_financing_cost(
    debt_balance: float,
    params: FinancingParameters = None,
) -> float:
    """
    Calculate financing costs (interest on debt).

    Cost^fin_t = r·D_t

    Parameters
    ----------
    debt_balance : float
        Current debt balance (D_t)
    params : FinancingParameters, optional
        Financing parameters

    Returns
    -------
    float
        Interest cost for the period
    """
    if params is None:
        params = FinancingParameters()

    return params.interest_rate * debt_balance


def calculate_total_cost(
    roster: List[Dict[str, float]],
    total_attendance: float,
    num_home_games: int,
    revenue: float,
    debt_balance: float = 0.0,
    marketing_spend: float = None,
    num_away_games: int = None,
    salary_params: SalaryCostParameters = None,
    venue_params: VenueCostParameters = None,
    ga_params: GeneralAdminParameters = None,
    sports_params: SportsOperationsParameters = None,
    finance_params: FinancingParameters = None,
) -> Dict[str, float]:
    """
    Calculate total costs for a period.

    Returns breakdown of all cost components.

    Parameters
    ----------
    roster : List[Dict[str, float]]
        Player roster with salaries
    total_attendance : float
        Total home game attendance
    num_home_games : int
        Number of home games
    revenue : float
        Total revenue (from income.py)
    debt_balance : float, default=0.0
        Current debt balance
    marketing_spend : float, optional
        Marketing spend (decision variable)
    num_away_games : int, optional
        Number of away games
    salary_params : SalaryCostParameters, optional
    venue_params : VenueCostParameters, optional
    ga_params : GeneralAdminParameters, optional
    sports_params : SportsOperationsParameters, optional
    finance_params : FinancingParameters, optional

    Returns
    -------
    Dict[str, float]
        Dictionary with cost breakdown:
        - salary_cost
        - venue_cost
        - ga_cost
        - sports_ops_cost
        - financing_cost
        - total_cost
    """
    # Calculate each cost component
    salary_cost = calculate_salary_cost(roster, salary_params)

    venue_cost = calculate_venue_cost(
        total_attendance, num_home_games, venue_params
    )

    ga_cost = calculate_ga_cost(revenue, marketing_spend, ga_params)

    sports_ops_cost = calculate_sports_ops_cost(
        num_home_games + (num_away_games or num_home_games),
        num_away_games,
        sports_params,
    )

    financing_cost = calculate_financing_cost(debt_balance, finance_params)

    # Total cost
    total_cost = (
        salary_cost
        + venue_cost
        + ga_cost
        + sports_ops_cost
        + financing_cost
    )

    return {
        'salary_cost': salary_cost,
        'venue_cost': venue_cost,
        'ga_cost': ga_cost,
        'sports_ops_cost': sports_ops_cost,
        'financing_cost': financing_cost,
        'total_cost': total_cost,
    }


def calculate_profit(
    revenue: float,
    costs: Dict[str, float],
) -> float:
    """
    Calculate regular season profit.

    π_t = Rev_t - Cost_t

    Parameters
    ----------
    revenue : float
        Total revenue
    costs : Dict[str, float]
        Cost breakdown from calculate_total_cost()

    Returns
    -------
    float
        Profit (can be negative)
    """
    return revenue - costs['total_cost']


def update_cash_flow(
    cash_balance: float,
    profit: float,
    new_borrowing: float = 0.0,
    capex: float = 0.0,
    debt_service: float = 0.0,
) -> float:
    """
    Update cash balance using cash flow state equation.

    Cash_{t+1} = Cash_t + π_t + d_t - CapEx_t - DebtService_t

    Parameters
    ----------
    cash_balance : float
        Current cash balance (Cash_t)
    profit : float
        Period profit (π_t)
    new_borrowing : float, default=0.0
        New debt issued (d_t)
    capex : float, default=0.0
        Capital expenditures (CapEx_t)
    debt_service : float, default=0.0
        Debt principal repayment (DebtService_t)

    Returns
    -------
    float
        Updated cash balance (Cash_{t+1})
    """
    return cash_balance + profit + new_borrowing - capex - debt_service


def update_debt_balance(
    debt_balance: float,
    new_borrowing: float = 0.0,
    debt_service: float = 0.0,
) -> float:
    """
    Update debt balance.

    D_{t+1} = D_t + d_t - DebtService_t

    Parameters
    ----------
    debt_balance : float
        Current debt balance (D_t)
    new_borrowing : float, default=0.0
        New borrowing (d_t)
    debt_service : float, default=0.0
        Principal repayment

    Returns
    -------
    float
        Updated debt balance (D_{t+1})
    """
    return debt_balance + new_borrowing - debt_service


def simulate_season_financials(
    revenue_by_period: List[float],
    attendance_by_period: List[float],
    roster: List[Dict[str, float]],
    num_home_games_per_period: List[int],
    initial_cash: float = 5000000.0,
    initial_debt: float = 0.0,
    marketing_spend_per_period: List[float] = None,
    capex_per_period: List[float] = None,
    salary_params: SalaryCostParameters = None,
    venue_params: VenueCostParameters = None,
    ga_params: GeneralAdminParameters = None,
    sports_params: SportsOperationsParameters = None,
    finance_params: FinancingParameters = None,
    cash_params: CashFlowParameters = None,
) -> pd.DataFrame:
    """
    Simulate multi-period financial performance.

    Parameters
    ----------
    revenue_by_period : List[float]
        Revenue for each period (from income.py)
    attendance_by_period : List[float]
        Total attendance for each period
    roster : List[Dict[str, float]]
        Player roster
    num_home_games_per_period : List[int]
        Number of home games per period
    initial_cash : float, default=5000000.0
        Starting cash balance
    initial_debt : float, default=0.0
        Starting debt balance
    marketing_spend_per_period : List[float], optional
        Marketing spend per period
    capex_per_period : List[float], optional
        CapEx per period
    salary_params : SalaryCostParameters, optional
    venue_params : VenueCostParameters, optional
    ga_params : GeneralAdminParameters, optional
    sports_params : SportsOperationsParameters, optional
    finance_params : FinancingParameters, optional
    cash_params : CashFlowParameters, optional

    Returns
    -------
    pd.DataFrame
        Financial results by period with columns:
        - period, revenue, total_cost, profit, cash_balance, debt_balance, etc.
    """
    if finance_params is None:
        finance_params = FinancingParameters()
    if cash_params is None:
        cash_params = CashFlowParameters()

    num_periods = len(revenue_by_period)

    # Initialize
    cash_balance = initial_cash
    debt_balance = initial_debt

    results = []

    for t in range(num_periods):
        revenue = revenue_by_period[t]
        attendance = attendance_by_period[t]
        num_home_games = num_home_games_per_period[t]

        marketing_spend = (
            marketing_spend_per_period[t]
            if marketing_spend_per_period
            else None
        )

        capex = (
            capex_per_period[t]
            if capex_per_period
            else cash_params.annual_capex / num_periods
        )

        # Calculate costs
        costs = calculate_total_cost(
            roster=roster,
            total_attendance=attendance,
            num_home_games=num_home_games,
            revenue=revenue,
            debt_balance=debt_balance,
            marketing_spend=marketing_spend,
            salary_params=salary_params,
            venue_params=venue_params,
            ga_params=ga_params,
            sports_params=sports_params,
            finance_params=finance_params,
        )

        # Calculate profit
        profit = calculate_profit(revenue, costs)

        # Debt service (principal repayment)
        debt_service = finance_params.debt_service_ratio * debt_balance

        # Check if need to borrow (if cash would go negative)
        projected_cash = cash_balance + profit - capex - debt_service
        new_borrowing = 0.0
        if projected_cash < cash_params.min_cash_balance:
            new_borrowing = cash_params.min_cash_balance - projected_cash

        # Update cash and debt
        cash_balance = update_cash_flow(
            cash_balance, profit, new_borrowing, capex, debt_service
        )
        debt_balance = update_debt_balance(
            debt_balance, new_borrowing, debt_service
        )

        # Record results
        results.append({
            'period': t + 1,
            'revenue': revenue,
            'salary_cost': costs['salary_cost'],
            'venue_cost': costs['venue_cost'],
            'ga_cost': costs['ga_cost'],
            'sports_ops_cost': costs['sports_ops_cost'],
            'financing_cost': costs['financing_cost'],
            'total_cost': costs['total_cost'],
            'profit': profit,
            'capex': capex,
            'debt_service': debt_service,
            'new_borrowing': new_borrowing,
            'cash_balance': cash_balance,
            'debt_balance': debt_balance,
            'attendance': attendance,
            'num_home_games': num_home_games,
        })

    return pd.DataFrame(results)


def calculate_bankruptcy_probability(
    cash_balances: np.ndarray,
    min_cash_threshold: float = 0.0,
) -> float:
    """
    Calculate bankruptcy probability.

    Risk = P(min_t Cash_t < threshold)

    For deterministic case, returns 1.0 if any period has cash < threshold, else 0.0.
    For stochastic simulation, would calculate fraction of scenarios with bankruptcy.

    Parameters
    ----------
    cash_balances : np.ndarray
        Cash balance trajectory
    min_cash_threshold : float, default=0.0
        Bankruptcy threshold

    Returns
    -------
    float
        Bankruptcy probability (0.0 or 1.0 for deterministic case)
    """
    min_cash = np.min(cash_balances)
    return 1.0 if min_cash < min_cash_threshold else 0.0


def calculate_cvar(
    outcomes: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """
    Calculate Conditional Value at Risk (CVaR).

    CVaR_α(X) = E[X | X ≤ VaR_α(X)]

    For losses, use negative values (e.g., -Cash_T or -Σπ_t).

    Parameters
    ----------
    outcomes : np.ndarray
        Distribution of outcomes (e.g., final cash or cumulative profit)
        Should be negative for losses
    alpha : float, default=0.05
        Confidence level (e.g., 0.05 for 5% worst cases)

    Returns
    -------
    float
        CVaR (expected value in worst α% of cases)
    """
    var = np.quantile(outcomes, alpha)
    cvar = np.mean(outcomes[outcomes <= var])
    return cvar


if __name__ == "__main__":
    # Example usage: simulate a season with costs
    print("=" * 80)
    print("WNBA Team Cost Model - Example Simulation")
    print("=" * 80)

    # Example roster (12 players)
    roster = [
        {'salary': 250000, 'on_roster': 1, 'equity_subsidy': 0.0},  # Supermax
        {'salary': 200000, 'on_roster': 1, 'equity_subsidy': 0.0},  # Star
        {'salary': 150000, 'on_roster': 1, 'equity_subsidy': 0.0},  # Veteran
        {'salary': 120000, 'on_roster': 1, 'equity_subsidy': 0.0},  # Average
        {'salary': 120000, 'on_roster': 1, 'equity_subsidy': 0.0},
        {'salary': 100000, 'on_roster': 1, 'equity_subsidy': 0.0},
        {'salary': 90000, 'on_roster': 1, 'equity_subsidy': 0.0},
        {'salary': 80000, 'on_roster': 1, 'equity_subsidy': 0.0},
        {'salary': 70000, 'on_roster': 1, 'equity_subsidy': 0.0},
        {'salary': 64000, 'on_roster': 1, 'equity_subsidy': 0.0},  # Minimum
        {'salary': 64000, 'on_roster': 1, 'equity_subsidy': 0.0},
        {'salary': 64000, 'on_roster': 1, 'equity_subsidy': 0.0},
    ]

    # Example: 3 periods (early, mid, late season)
    # Revenue from income.py simulation (example values)
    revenue_by_period = [18_000_000, 30_000_000, 24_000_000]  # Total: $72M
    attendance_by_period = [105_000, 185_000, 141_000]  # Total: 431K
    num_home_games_per_period = [10, 10, 10]

    # Run simulation
    financials = simulate_season_financials(
        revenue_by_period=revenue_by_period,
        attendance_by_period=attendance_by_period,
        roster=roster,
        num_home_games_per_period=num_home_games_per_period,
        initial_cash=5_000_000,
        initial_debt=0,
    )

    print("\nFinancial Performance by Period:")
    print(financials.to_string(index=False))

    # Summary statistics
    print("\n" + "=" * 80)
    print("Season Summary:")
    print("=" * 80)
    print(f"Total Revenue: ${financials['revenue'].sum():,.0f}")
    print(f"Total Costs: ${financials['total_cost'].sum():,.0f}")
    print(f"  - Salary: ${financials['salary_cost'].sum():,.0f}")
    print(f"  - Venue: ${financials['venue_cost'].sum():,.0f}")
    print(f"  - G&A: ${financials['ga_cost'].sum():,.0f}")
    print(f"  - Sports Ops: ${financials['sports_ops_cost'].sum():,.0f}")
    print(f"  - Financing: ${financials['financing_cost'].sum():,.0f}")
    print(f"Total Profit: ${financials['profit'].sum():,.0f}")
    print(f"Final Cash Balance: ${financials['cash_balance'].iloc[-1]:,.0f}")
    print(f"Final Debt Balance: ${financials['debt_balance'].iloc[-1]:,.0f}")

    # Risk metrics
    bankruptcy_prob = calculate_bankruptcy_probability(
        financials['cash_balance'].values
    )
    print(f"\nBankruptcy Probability: {bankruptcy_prob:.1%}")

    print("\n" + "=" * 80)
