"""
Mitigation Strategies for Injury Risk

Implements the three mitigation strategies from paper Task 5:
1. Load management (aggressive rest)
2. Medical investment (increase u_t)
3. Network repair signing (high flow centrality players)

Decision Framework
------------------
Default mitigation options:

**Strategy 1: Load Management**
- Trade-off: Sacrifice 1 game win probability (-20%) to reduce injury risk
- Effect: Injury probability 15% ->2% (-13pp)
- Expected value: Positive if single-game value < 1.3 脳 expected miss value

**Strategy 2: Medical Investment**
- Increase u_t from $1.0M to $1.5M (+50%)
- Effect: Injury probability 17.4% ->13.6% (-3.8pp)
- Break-even: Value per game missed 鈮?$1.32M

**Strategy 3: Network Repair Signing**
- Sign high flow centrality player to repair network
- Effect: Win probability 55% ->65% during injury (+10pp)
- Cost threshold: If 1 win value 鈮?signing cost, execute

Three-Tier Warning System
--------------------------
- Green (Fatigue < 1.2): Normal operations
- Yellow (1.2 鈮?Fatigue < 1.6): Increase u_t +20%, reduce minutes
- Red (Fatigue 鈮?1.6): Mandatory rest, activate network repair
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class MitigationStrategy:
    """
    Mitigation strategy parameters and evaluation.

    Attributes
    ----------
    strategy_name : str
        Strategy name
    cost : float
        Implementation cost ($M)
    injury_risk_reduction : float
        Reduction in injury probability (pp)
    performance_impact : float
        Impact on win probability (can be negative for load management)
    """
    strategy_name: str
    cost: float
    injury_risk_reduction: float
    performance_impact: float


def evaluate_load_management(
    baseline_injury_prob: float,
    games_remaining: int,
    win_prob_sacrifice: float = 0.20,
    injury_reduction: float = 0.13,
    value_per_win: float = 0.5,
    expected_games_missed_if_injured: float = 10.0,
) -> Dict[str, float]:
    """
    Evaluate load management strategy.

    Trade-off: Rest core player for 1 game to reduce injury risk.

    Default assumption:
    - Sacrifice: 1 game at -20% win probability
    - Benefit: Injury probability 15% ->2% (-13pp)
    - Decision: Execute if expected value > 0

    Parameters
    ----------
    baseline_injury_prob : float
        Current injury probability
    games_remaining : int
        Games remaining in season
    win_prob_sacrifice : float, default=0.20
        Win probability sacrifice for rest game
    injury_reduction : float, default=0.13
        Injury probability reduction (pp)
    value_per_win : float, default=0.5
        Value per win ($M)
    expected_games_missed_if_injured : float, default=10.0
        Expected games missed if injury occurs

    Returns
    -------
    Dict[str, float]
        - immediate_cost: Cost of resting (lost win probability)
        - expected_benefit: Expected benefit from injury reduction
        - net_value: Net expected value
        - recommendation: 1 if execute, 0 if not

    Examples
    --------
    >>> result = evaluate_load_management(0.15, 20)
    >>> result['net_value'] > 0  # Should be positive
    True
    >>> result['recommendation']
    1.0
    """
    # Immediate cost: Lost win probability in rest game
    immediate_cost = -win_prob_sacrifice * value_per_win

    # Expected benefit: Reduced injury risk 脳 value of avoiding injury
    # Value of avoiding injury = expected games missed 脳 win prob 脳 value per win
    value_per_game_missed = 0.75 * value_per_win  # Assume 75% win prob when healthy
    value_of_avoiding_injury = expected_games_missed_if_injured * value_per_game_missed

    expected_benefit = injury_reduction * value_of_avoiding_injury

    net_value = expected_benefit + immediate_cost  # immediate_cost is negative

    recommendation = 1.0 if net_value > 0 else 0.0

    return {
        'immediate_cost': float(immediate_cost),
        'expected_benefit': float(expected_benefit),
        'net_value': float(net_value),
        'recommendation': float(recommendation),
        'break_even_injury_reduction': float(-immediate_cost / value_of_avoiding_injury),
    }


def evaluate_medical_investment(
    current_investment: float,
    proposed_investment: float,
    baseline_injury_prob: float,
    beta_med: float = 0.5,
    games_remaining: int = 20,
    expected_games_missed_if_injured: float = 10.0,
    value_per_win: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate medical investment increase.

    Default assumption:
    - Increase u_t from $1.0M to $1.5M (+50%)
    - Effect: Injury probability 17.4% ->13.6% (-3.8pp)
    - Break-even: Value per game missed 鈮?$1.32M

    Parameters
    ----------
    current_investment : float
        Current medical investment ($M)
    proposed_investment : float
        Proposed medical investment ($M)
    baseline_injury_prob : float
        Current injury probability
    beta_med : float, default=0.5
        Medical investment effect coefficient
    games_remaining : int, default=20
        Games remaining in season
    expected_games_missed_if_injured : float, default=10.0
        Expected games missed if injury occurs
    value_per_win : float, default=0.5
        Value per win ($M)

    Returns
    -------
    Dict[str, float]
        - investment_increase: Additional investment
        - injury_prob_reduction: Reduction in injury probability
        - expected_benefit: Expected benefit from risk reduction
        - net_value: Net expected value
        - recommendation: 1 if execute, 0 if not

    Examples
    --------
    >>> result = evaluate_medical_investment(1.0, 1.5, 0.174)
    >>> result['injury_prob_reduction'] > 0
    True
    >>> result['net_value'] > 0
    True
    """
    investment_increase = proposed_investment - current_investment

    # Calculate injury probability reduction using logistic model
    # p_inj = 蟽(... - 尾_med路ln(1+u))
    # Reduction 鈮?尾_med 脳 [ln(1+u_new) - ln(1+u_old)] 脳 p 脳 (1-p)
    log_diff = np.log1p(proposed_investment) - np.log1p(current_investment)

    # Approximate marginal effect (derivative of sigmoid)
    marginal_effect = baseline_injury_prob * (1 - baseline_injury_prob)
    injury_prob_reduction = beta_med * log_diff * marginal_effect

    # Expected benefit over remaining games
    value_per_game_missed = 0.75 * value_per_win
    value_of_avoiding_injury = expected_games_missed_if_injured * value_per_game_missed

    # Expected benefit = reduction 脳 games 脳 value
    expected_benefit = injury_prob_reduction * games_remaining * value_of_avoiding_injury

    net_value = expected_benefit - investment_increase

    recommendation = 1.0 if net_value > 0 else 0.0

    return {
        'investment_increase': float(investment_increase),
        'injury_prob_reduction': float(injury_prob_reduction),
        'expected_benefit': float(expected_benefit),
        'net_value': float(net_value),
        'recommendation': float(recommendation),
        'break_even_value_per_game': float(investment_increase / (injury_prob_reduction * games_remaining)) if injury_prob_reduction > 0 else float('inf'),
    }


def evaluate_network_repair_signing(
    injured_player_centrality: float,
    candidate_centrality: float,
    signing_cost: float,
    games_remaining: int,
    baseline_win_prob_injured: float = 0.55,
    value_per_win: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate network repair signing strategy.

    Default assumption:
    - Sign high flow centrality player to repair network
    - Effect: Win probability 55% ->65% (+10pp)
    - Decision: Execute if win value 鈮?signing cost

    Parameters
    ----------
    injured_player_centrality : float
        Flow centrality of injured player (0-1)
    candidate_centrality : float
        Flow centrality of signing candidate (0-1)
    signing_cost : float
        Cost to sign candidate ($M)
    games_remaining : int
        Games remaining in season
    baseline_win_prob_injured : float, default=0.55
        Win probability with injury, no signing
    value_per_win : float, default=0.5
        Value per win ($M)

    Returns
    -------
    Dict[str, float]
        - centrality_recovery: Fraction of centrality recovered
        - win_prob_improvement: Improvement in win probability
        - expected_wins_gained: Expected additional wins
        - expected_benefit: Expected benefit ($M)
        - net_value: Net expected value
        - recommendation: 1 if execute, 0 if not

    Examples
    --------
    >>> result = evaluate_network_repair_signing(0.85, 0.60, 0.35, 10)
    >>> result['centrality_recovery'] > 0
    True
    >>> result['net_value'] > 0
    True
    """
    # Centrality recovery (fraction of lost centrality restored)
    centrality_recovery = candidate_centrality / injured_player_centrality if injured_player_centrality > 0 else 0.0
    centrality_recovery = min(centrality_recovery, 1.0)  # Cap at 100%

    # Win probability improvement (proportional to centrality recovery)
    # Full recovery would restore 75% - 55% = 20pp.
    max_improvement = 0.20
    win_prob_improvement = centrality_recovery * max_improvement

    # Expected wins gained
    expected_wins_gained = win_prob_improvement * games_remaining

    # Expected benefit
    expected_benefit = expected_wins_gained * value_per_win

    net_value = expected_benefit - signing_cost

    recommendation = 1.0 if net_value > 0 else 0.0

    return {
        'centrality_recovery': float(centrality_recovery),
        'win_prob_improvement': float(win_prob_improvement),
        'expected_wins_gained': float(expected_wins_gained),
        'expected_benefit': float(expected_benefit),
        'signing_cost': float(signing_cost),
        'net_value': float(net_value),
        'recommendation': float(recommendation),
        'break_even_cost': float(expected_benefit),
    }


def create_fatigue_warning_system(
    fatigue_level: float,
    injury_prob: float,
) -> Dict[str, any]:
    """
    Create three-tier fatigue warning system.

    Default scenario:
    - Green (Fatigue < 1.2): Normal operations
    - Yellow (1.2 鈮?Fatigue < 1.6): Increase u_t +20%, reduce minutes
    - Red (Fatigue 鈮?1.6): Mandatory rest, activate network repair

    Parameters
    ----------
    fatigue_level : float
        Current fatigue index
    injury_prob : float
        Current injury probability

    Returns
    -------
    Dict
        - alert_level: "green", "yellow", or "red"
        - recommended_actions: List of actions
        - urgency_score: 0-10 scale

    Examples
    --------
    >>> result = create_fatigue_warning_system(1.0, 0.08)
    >>> result['alert_level']
    'green'

    >>> result = create_fatigue_warning_system(1.8, 0.18)
    >>> result['alert_level']
    'red'
    """
    if fatigue_level < 1.2 and injury_prob < 0.10:
        alert_level = "green"
        actions = [
            "Normal rotation",
            "Standard medical protocols",
            "Continue monitoring",
        ]
        urgency_score = 2.0

    elif fatigue_level < 1.6 and injury_prob < 0.15:
        alert_level = "yellow"
        actions = [
            "Increase medical investment u_t by +20%",
            "Reduce core player minutes by 10%",
            "Enhanced recovery protocols (ice baths, massage)",
            "Monitor daily fatigue markers",
        ]
        urgency_score = 5.0

    else:
        alert_level = "red"
        actions = [
            "MANDATORY REST for core players",
            "Activate network repair signing protocol",
            "Increase medical investment u_t by +50%",
            "Emergency load management",
            "Daily medical assessments",
        ]
        urgency_score = 9.0

    return {
        'alert_level': alert_level,
        'fatigue_level': float(fatigue_level),
        'injury_prob': float(injury_prob),
        'recommended_actions': actions,
        'urgency_score': float(urgency_score),
    }


def optimize_mitigation_portfolio(
    current_state: Dict[str, float],
    available_strategies: List[MitigationStrategy],
    budget_constraint: float,
) -> Dict[str, any]:
    """
    Optimize portfolio of mitigation strategies under budget constraint.

    Parameters
    ----------
    current_state : Dict
        Current state with keys: fatigue, injury_prob, games_remaining
    available_strategies : List[MitigationStrategy]
        Available mitigation strategies
    budget_constraint : float
        Maximum budget for mitigation ($M)

    Returns
    -------
    Dict
        - selected_strategies: List of selected strategies
        - total_cost: Total cost
        - total_benefit: Total expected benefit
        - net_value: Net expected value

    Examples
    --------
    >>> state = {'fatigue': 1.5, 'injury_prob': 0.15, 'games_remaining': 20}
    >>> strategies = [
    ...     MitigationStrategy("Load Management", 0.1, 0.13, -0.20),
    ...     MitigationStrategy("Medical Investment", 0.5, 0.038, 0.0),
    ... ]
    >>> result = optimize_mitigation_portfolio(state, strategies, 1.0)
    >>> len(result['selected_strategies']) > 0
    True
    """
    # Simple greedy optimization: sort by benefit/cost ratio
    strategy_values = []

    for strategy in available_strategies:
        # Estimate benefit (simplified)
        injury_reduction_benefit = strategy.injury_risk_reduction * 10 * 0.75 * 0.5  # games 脳 win_prob 脳 value
        performance_benefit = strategy.performance_impact * current_state['games_remaining'] * 0.5

        total_benefit = injury_reduction_benefit + performance_benefit
        net_value = total_benefit - strategy.cost

        strategy_values.append({
            'strategy': strategy,
            'benefit': total_benefit,
            'cost': strategy.cost,
            'net_value': net_value,
            'benefit_cost_ratio': total_benefit / strategy.cost if strategy.cost > 0 else float('inf'),
        })

    # Sort by benefit/cost ratio
    strategy_values.sort(key=lambda x: x['benefit_cost_ratio'], reverse=True)

    # Greedy selection under budget constraint
    selected = []
    total_cost = 0.0
    total_benefit = 0.0

    for sv in strategy_values:
        if total_cost + sv['cost'] <= budget_constraint and sv['net_value'] > 0:
            selected.append(sv['strategy'])
            total_cost += sv['cost']
            total_benefit += sv['benefit']

    return {
        'selected_strategies': selected,
        'total_cost': float(total_cost),
        'total_benefit': float(total_benefit),
        'net_value': float(total_benefit - total_cost),
        'budget_utilization': float(total_cost / budget_constraint) if budget_constraint > 0 else 0.0,
    }


def create_mitigation_dashboard(
    fatigue_level: float,
    injury_prob: float,
    games_remaining: int,
    current_medical_investment: float = 1.0,
) -> pd.DataFrame:
    """
    Create mitigation strategy dashboard.

    Parameters
    ----------
    fatigue_level : float
        Current fatigue index
    injury_prob : float
        Current injury probability
    games_remaining : int
        Games remaining in season
    current_medical_investment : float, default=1.0
        Current medical investment ($M)

    Returns
    -------
    pd.DataFrame
        Dashboard with columns: strategy, cost, benefit, net_value, recommendation
    """
    # Evaluate each strategy
    load_mgmt = evaluate_load_management(injury_prob, games_remaining)
    med_invest = evaluate_medical_investment(
        current_medical_investment,
        current_medical_investment * 1.5,
        injury_prob,
        games_remaining=games_remaining,
    )

    # Network repair (assume typical values)
    network_repair = evaluate_network_repair_signing(
        injured_player_centrality=0.85,
        candidate_centrality=0.60,
        signing_cost=0.35,
        games_remaining=games_remaining,
    )

    # Warning system
    warning = create_fatigue_warning_system(fatigue_level, injury_prob)

    data = [
        {
            'strategy': 'Load Management',
            'cost': -load_mgmt['immediate_cost'],
            'benefit': load_mgmt['expected_benefit'],
            'net_value': load_mgmt['net_value'],
            'recommendation': 'Execute' if load_mgmt['recommendation'] > 0 else 'Skip',
        },
        {
            'strategy': 'Medical Investment (+50%)',
            'cost': med_invest['investment_increase'],
            'benefit': med_invest['expected_benefit'],
            'net_value': med_invest['net_value'],
            'recommendation': 'Execute' if med_invest['recommendation'] > 0 else 'Skip',
        },
        {
            'strategy': 'Network Repair Signing',
            'cost': network_repair['signing_cost'],
            'benefit': network_repair['expected_benefit'],
            'net_value': network_repair['net_value'],
            'recommendation': 'Execute' if network_repair['recommendation'] > 0 else 'Skip',
        },
    ]

    df = pd.DataFrame(data)
    df['alert_level'] = warning['alert_level']

    return df

