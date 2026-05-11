"""
Injury impact calculations for Task 5.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class InjuryImpactAnalyzer:
    """
    Analyzer for quantifying injury shock impacts.

    Parameters
    ----------
    lambda_W : float
        Weight of wins in utility function ($/win)
    lambda_risk : float
        Risk aversion coefficient
    playoff_revenue_per_round : float
        Revenue per playoff round ($M)
    valuation_multiplier : float
        EBITDA multiplier for valuation
    """
    lambda_W: float = 0.5  # $0.5M per win
    lambda_risk: float = 0.1
    playoff_revenue_per_round: float = 1.0  # $1M per round
    valuation_multiplier: float = 3.34


def quantify_win_loss(
    baseline_win_prob: float,
    injured_win_prob: float,
    games_missed: int,
    lambda_W: float = 0.5,
) -> Dict[str, float]:
    """
    Quantify win loss impact from injury.

    Default scenario: win probability changes from 75% to 55%.

    Parameters
    ----------
    baseline_win_prob : float
        Win probability when healthy (e.g., 0.75)
    injured_win_prob : float
        Win probability when injured (e.g., 0.55)
    games_missed : int
        Number of games missed (e.g., 10)
    lambda_W : float, default=0.5
        Value per win in millions USD

    Returns
    -------
    Dict[str, float]
        - expected_wins_baseline: Expected wins if healthy
        - expected_wins_injured: Expected wins with injury
        - win_loss: 螖W (negative)
        - value_loss: 螖J_W = 位_W 路 螖W (negative)

    Examples
    --------
    >>> result = quantify_win_loss(0.75, 0.55, 10, lambda_W=0.5)
    >>> result['win_loss']
    -2.0
    >>> result['value_loss']
    -1.0
    """
    expected_wins_baseline = baseline_win_prob * games_missed
    expected_wins_injured = injured_win_prob * games_missed

    win_loss = expected_wins_injured - expected_wins_baseline
    value_loss = lambda_W * win_loss

    return {
        'expected_wins_baseline': float(expected_wins_baseline),
        'expected_wins_injured': float(expected_wins_injured),
        'win_loss': float(win_loss),
        'value_loss': float(value_loss),
    }


def quantify_playoff_impact(
    baseline_playoff_prob: float,
    injured_playoff_prob: float,
    playoff_revenue: float = 4.0,
) -> Dict[str, float]:
    """
    Quantify playoff impact from injury.

    Default scenario: playoff probability changes from 90% to 60%.

    Parameters
    ----------
    baseline_playoff_prob : float
        Playoff probability when healthy (e.g., 0.90)
    injured_playoff_prob : float
        Playoff probability when injured (e.g., 0.60)
    playoff_revenue : float, default=4.0
        Expected playoff revenue in millions USD

    Returns
    -------
    Dict[str, float]
        - baseline_playoff_prob: Baseline probability
        - injured_playoff_prob: Injured probability
        - playoff_prob_change: Change in probability (pp)
        - expected_revenue_loss: 螖J_PO (negative)

    Examples
    --------
    >>> result = quantify_playoff_impact(0.90, 0.60, playoff_revenue=4.0)
    >>> result['playoff_prob_change']
    -0.3
    >>> result['expected_revenue_loss']
    -1.2
    """
    playoff_prob_change = injured_playoff_prob - baseline_playoff_prob

    # Expected revenue loss = 螖P(PO) 脳 E[Revenue | PO]
    expected_revenue_loss = playoff_prob_change * playoff_revenue

    return {
        'baseline_playoff_prob': float(baseline_playoff_prob),
        'injured_playoff_prob': float(injured_playoff_prob),
        'playoff_prob_change': float(playoff_prob_change),
        'expected_revenue_loss': float(expected_revenue_loss),
    }


def quantify_revenue_loss(
    games_missed: int,
    home_games_missed: int,
    attendance_drop_pct: float = 0.125,
    avg_ticket_revenue_per_game: float = 0.375,
    sponsorship_drop_pct: float = 0.05,
    total_sponsorship: float = 15.0,
) -> Dict[str, float]:
    """
    Quantify regular season revenue loss.

    Default scenario:
    - Attendance drops 10-15% for home games
    - Sponsorship drops ~5% due to reduced Star_t

    Parameters
    ----------
    games_missed : int
        Total games missed
    home_games_missed : int
        Home games missed (typically ~50% of total)
    attendance_drop_pct : float, default=0.125
        Attendance drop percentage (12.5% midpoint)
    avg_ticket_revenue_per_game : float, default=0.375
        Average ticket revenue per home game ($M)
    sponsorship_drop_pct : float, default=0.05
        Sponsorship drop percentage
    total_sponsorship : float, default=15.0
        Total annual sponsorship ($M)

    Returns
    -------
    Dict[str, float]
        - ticket_revenue_loss: Loss from reduced attendance
        - sponsorship_loss: Loss from reduced Star_t
        - total_revenue_loss: 螖J_蟺 (negative)

    Examples
    --------
    >>> result = quantify_revenue_loss(10, 5)
    >>> result['ticket_revenue_loss']
    -0.234375
    >>> result['total_revenue_loss'] < 0
    True
    """
    # Ticket revenue loss (only affects home games)
    ticket_revenue_loss = -home_games_missed * avg_ticket_revenue_per_game * attendance_drop_pct

    # Sponsorship loss (affects entire season proportionally)
    # Assume injury affects (games_missed / 40) of season
    season_fraction = games_missed / 40.0
    sponsorship_loss = -total_sponsorship * sponsorship_drop_pct * season_fraction

    total_revenue_loss = ticket_revenue_loss + sponsorship_loss

    return {
        'ticket_revenue_loss': float(ticket_revenue_loss),
        'sponsorship_loss': float(sponsorship_loss),
        'total_revenue_loss': float(total_revenue_loss),
    }


def quantify_valuation_impact(
    missed_championship_prob: float,
    baseline_valuation: float = 150.0,
    championship_premium: float = 0.20,
    use_worst_case: bool = True,
) -> Dict[str, float]:
    """
    Quantify valuation impact from missed championship window.

    In the stress case, if an injury closes the championship window,
    valuation drops by ~20% (-$30M on $150M baseline).

    Parameters
    ----------
    missed_championship_prob : float
        Probability of missing championship due to injury
    baseline_valuation : float, default=150.0
        Baseline team valuation ($M)
    championship_premium : float, default=0.20
        Valuation premium for championship contender
    use_worst_case : bool, default=True
        If True, use worst-case scenario (100% championship miss) as in paper.
        If False, use expected value (probability-weighted).

    Returns
    -------
    Dict[str, float]
        - baseline_valuation: Baseline valuation
        - valuation_loss: 螖J_V (negative)
        - worst_case_loss: Maximum loss if championship missed
        - expected_loss: Expected loss (probability-weighted)

    Examples
    --------
    >>> # Paper approach: worst-case scenario
    >>> result = quantify_valuation_impact(0.30, baseline_valuation=150.0, use_worst_case=True)
    >>> result['valuation_loss']
    -30.0

    >>> # Expected value approach
    >>> result = quantify_valuation_impact(0.30, baseline_valuation=150.0, use_worst_case=False)
    >>> result['valuation_loss']
    -9.0
    """
    worst_case_loss = -baseline_valuation * championship_premium
    expected_loss = missed_championship_prob * worst_case_loss

    # Stress mode assumes the championship window is missed.
    if use_worst_case:
        valuation_loss = worst_case_loss
    else:
        valuation_loss = expected_loss

    return {
        'baseline_valuation': float(baseline_valuation),
        'valuation_loss': float(valuation_loss),
        'worst_case_loss': float(worst_case_loss),
        'expected_loss': float(expected_loss),
    }


def calculate_total_impact(
    win_loss_impact: Dict[str, float],
    playoff_impact: Dict[str, float],
    revenue_impact: Dict[str, float],
    valuation_impact: Dict[str, float],
    risk_change: float = 0.0,
    lambda_risk: float = 0.1,
) -> Dict[str, float]:
    """
    Calculate total impact on objective function J.

    螖J = 螖J_W + 螖J_PO + 螖J_蟺 + 螖J_V - 位_risk路螖Risk

    Parameters
    ----------
    win_loss_impact : Dict
        From quantify_win_loss
    playoff_impact : Dict
        From quantify_playoff_impact
    revenue_impact : Dict
        From quantify_revenue_loss
    valuation_impact : Dict
        From quantify_valuation_impact
    risk_change : float, default=0.0
        Change in risk metric (CVaR, etc.)
    lambda_risk : float, default=0.1
        Risk aversion coefficient

    Returns
    -------
    Dict[str, float]
        - delta_J_W: Win loss component
        - delta_J_PO: Playoff component
        - delta_J_pi: Revenue component
        - delta_J_V: Valuation component
        - risk_penalty: Risk component
        - total_impact: Total 螖J

    Examples
    --------
    >>> win = quantify_win_loss(0.75, 0.55, 10)
    >>> playoff = quantify_playoff_impact(0.90, 0.60)
    >>> revenue = quantify_revenue_loss(10, 5)
    >>> valuation = quantify_valuation_impact(0.30)
    >>> total = calculate_total_impact(win, playoff, revenue, valuation)
    >>> total['total_impact'] < 0
    True
    """
    delta_J_W = win_loss_impact['value_loss']
    delta_J_PO = playoff_impact['expected_revenue_loss']
    delta_J_pi = revenue_impact['total_revenue_loss']
    delta_J_V = valuation_impact['valuation_loss']
    risk_penalty = -lambda_risk * risk_change

    total_impact = delta_J_W + delta_J_PO + delta_J_pi + delta_J_V + risk_penalty

    return {
        'delta_J_W': float(delta_J_W),
        'delta_J_PO': float(delta_J_PO),
        'delta_J_pi': float(delta_J_pi),
        'delta_J_V': float(delta_J_V),
        'risk_penalty': float(risk_penalty),
        'total_impact': float(total_impact),
    }


def create_impact_waterfall(
    total_impact: Dict[str, float],
) -> pd.DataFrame:
    """
    Create waterfall chart data for impact decomposition.

    Parameters
    ----------
    total_impact : Dict
        From calculate_total_impact

    Returns
    -------
    pd.DataFrame
        Waterfall data with columns: component, value, cumulative

    Examples
    --------
    >>> win = quantify_win_loss(0.75, 0.55, 10)
    >>> playoff = quantify_playoff_impact(0.90, 0.60)
    >>> revenue = quantify_revenue_loss(10, 5)
    >>> valuation = quantify_valuation_impact(0.30)
    >>> total = calculate_total_impact(win, playoff, revenue, valuation)
    >>> df = create_impact_waterfall(total)
    >>> df['component'].tolist()
    ['Win Loss', 'Playoff Impact', 'Revenue Loss', 'Valuation Loss', 'Risk Penalty', 'Total Impact']
    """
    components = [
        ('Win Loss', total_impact['delta_J_W']),
        ('Playoff Impact', total_impact['delta_J_PO']),
        ('Revenue Loss', total_impact['delta_J_pi']),
        ('Valuation Loss', total_impact['delta_J_V']),
        ('Risk Penalty', total_impact['risk_penalty']),
        ('Total Impact', total_impact['total_impact']),
    ]

    cumulative = 0.0
    data = []

    for component, value in components:
        if component != 'Total Impact':
            cumulative += value
            data.append({
                'component': component,
                'value': value,
                'cumulative': cumulative,
            })
        else:
            data.append({
                'component': component,
                'value': value,
                'cumulative': value,
            })

    return pd.DataFrame(data)


def simulate_injury_scenario(
    player_name: str,
    games_missed: int = 10,
    baseline_win_prob: float = 0.75,
    injured_win_prob: float = 0.55,
    baseline_playoff_prob: float = 0.90,
    injured_playoff_prob: float = 0.60,
    baseline_valuation: float = 150.0,
    use_worst_case_valuation: bool = True,
) -> Dict[str, Any]:
    """
    Simulate a complete injury scenario.

    Default parameters are a small, reproducible stress scenario.

    Parameters
    ----------
    player_name : str
        Player name
    games_missed : int, default=10
        Games missed
    baseline_win_prob : float, default=0.75
        Healthy win probability
    injured_win_prob : float, default=0.55
        Injured win probability
    baseline_playoff_prob : float, default=0.90
        Healthy playoff probability
    injured_playoff_prob : float, default=0.60
        Injured playoff probability
    baseline_valuation : float, default=150.0
        Baseline valuation ($M)
    use_worst_case_valuation : bool, default=True
        If True, use the stress-case valuation loss (-$30M).
        If False, use expected value (-$9M).

    Returns
    -------
    Dict
        Complete impact analysis with all components

    Examples
    --------
    >>> # Stress case
    >>> result = simulate_injury_scenario("A'ja Wilson", use_worst_case_valuation=True)
    >>> result['summary']['total_impact']
    -32.621875

    >>> # Expected value approach
    >>> result = simulate_injury_scenario("A'ja Wilson", use_worst_case_valuation=False)
    >>> result['summary']['total_impact']
    -11.621875
    """
    # Calculate each component
    win_impact = quantify_win_loss(baseline_win_prob, injured_win_prob, games_missed)

    playoff_impact = quantify_playoff_impact(
        baseline_playoff_prob,
        injured_playoff_prob,
        playoff_revenue=4.0,
    )

    home_games = games_missed // 2  # Assume 50% home games
    revenue_impact = quantify_revenue_loss(games_missed, home_games)

    # Probability of missing championship = playoff prob drop
    missed_champ_prob = baseline_playoff_prob - injured_playoff_prob
    valuation_impact = quantify_valuation_impact(
        missed_champ_prob,
        baseline_valuation=baseline_valuation,
        use_worst_case=use_worst_case_valuation,
    )

    # Calculate total
    total = calculate_total_impact(
        win_impact,
        playoff_impact,
        revenue_impact,
        valuation_impact,
    )

    # Create waterfall
    waterfall = create_impact_waterfall(total)

    return {
        'player_name': player_name,
        'games_missed': games_missed,
        'win_impact': win_impact,
        'playoff_impact': playoff_impact,
        'revenue_impact': revenue_impact,
        'valuation_impact': valuation_impact,
        'total_impact': total,
        'waterfall': waterfall,
        'summary': {
            'win_loss': win_impact['win_loss'],
            'playoff_prob_change': playoff_impact['playoff_prob_change'],
            'revenue_loss': revenue_impact['total_revenue_loss'],
            'valuation_loss': valuation_impact['valuation_loss'],
            'total_impact': total['total_impact'],
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a small Task 5 injury-impact smoke test.",
    )
    parser.add_argument(
        "--player-name",
        default="Sample Player",
        help="Name used in the printed scenario label.",
    )
    parser.add_argument(
        "--games-missed",
        type=int,
        default=10,
        help="Number of games missed in the injury scenario.",
    )
    parser.add_argument(
        "--baseline-win-prob",
        type=float,
        default=0.75,
        help="Win probability before the injury shock.",
    )
    parser.add_argument(
        "--injured-win-prob",
        type=float,
        default=0.55,
        help="Win probability during the injury shock.",
    )
    parser.add_argument(
        "--baseline-playoff-prob",
        type=float,
        default=0.90,
        help="Playoff probability before the injury shock.",
    )
    parser.add_argument(
        "--injured-playoff-prob",
        type=float,
        default=0.60,
        help="Playoff probability during the injury shock.",
    )
    parser.add_argument(
        "--baseline-valuation",
        type=float,
        default=150.0,
        help="Baseline franchise valuation in millions of USD.",
    )
    parser.add_argument(
        "--expected-value",
        action="store_true",
        help="Use probability-weighted valuation loss instead of the stress case.",
    )
    return parser


def _format_summary(result: Dict[str, Any], valuation_mode: str) -> str:
    summary = result["summary"]
    waterfall = result["waterfall"][["component", "value", "cumulative"]].copy()
    waterfall["value"] = waterfall["value"].map(lambda value: f"{value:,.3f}")
    waterfall["cumulative"] = waterfall["cumulative"].map(lambda value: f"{value:,.3f}")

    lines = [
        "Task 5 injury-impact smoke run",
        f"Player: {result['player_name']}",
        f"Games missed: {result['games_missed']}",
        f"Valuation mode: {valuation_mode}",
        "",
        "Summary (USD millions unless noted)",
        f"Win change: {summary['win_loss']:.2f} wins",
        f"Playoff probability change: {summary['playoff_prob_change']:.1%}",
        f"Revenue loss: {summary['revenue_loss']:.3f}",
        f"Valuation loss: {summary['valuation_loss']:.3f}",
        f"Total impact: {summary['total_impact']:.3f}",
        "",
        "Breakdown",
        waterfall.to_string(index=False),
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    use_worst_case = not args.expected_value

    result = simulate_injury_scenario(
        player_name=args.player_name,
        games_missed=args.games_missed,
        baseline_win_prob=args.baseline_win_prob,
        injured_win_prob=args.injured_win_prob,
        baseline_playoff_prob=args.baseline_playoff_prob,
        injured_playoff_prob=args.injured_playoff_prob,
        baseline_valuation=args.baseline_valuation,
        use_worst_case_valuation=use_worst_case,
    )
    valuation_mode = "stress case" if use_worst_case else "expected value"
    print(_format_summary(result, valuation_mode))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

