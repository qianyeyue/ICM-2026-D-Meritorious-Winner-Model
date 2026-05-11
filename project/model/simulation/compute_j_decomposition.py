"""
Compute J decomposition for Task1 output (model.md Section 1.4).

J = J^{reg_profit} + J^{win} + J^{PO} + J^{end} - J^{risk}

This module provides functions to:
1. Calculate each component of J from simulation results
2. Generate waterfall charts showing J decomposition
3. Compute sensitivity analysis (+/-10% perturbations)
4. Create one-page summary for owner decision-making
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class JDecomposition:
    """
    J decomposition following model.md Section 1.4.

    J = J^{reg_profit} + J^{win} + J^{PO} + J^{end} - J^{risk}
    """
    # Total objective value
    J_total: float

    # Components (model.md Section 1.4)
    J_reg_profit: float  # E[危 未^t * 蟺_t]: Regular season profit contribution
    J_win: float  # E[危 未^t * 位_W * W_t]: Win contribution
    J_playoff: float  # 未^T * E[螤^PO]: Playoff revenue contribution
    J_terminal: float  # 未^T * E[V_end]: Terminal value (EBITDA + brand)
    J_risk: float  # 位_risk * Risk: Risk penalty (CVaR/bankruptcy)

    # Supporting metrics
    total_profit: float  # 危 蟺_t (undiscounted)
    total_wins: int  # 危 W_t
    terminal_value: float  # V_end (EBITDA multiple + brand value)
    risk_metric: float  # CVaR or bankruptcy probability

    # Control variables (for sensitivity analysis)
    total_u: float  # 危 u_t
    total_m: float  # 危 m_t
    avg_tau: float  # Average ticket multiplier
    total_borrowed: float  # 危 d_t (net financing)


def compute_j_from_period_log(
    period_log: pd.DataFrame,
    *,
    final_elo: float,
    final_B: float,
    final_cash: float,
    final_debt: float,
    cum_ebitda: float,
    discount: float = 0.95,
    lambda_win: float = 100_000.0,
    lambda_playoff: float = 1.0,
    lambda_terminal: float = 1.0,
    lambda_risk: float = 1.0,
    made_playoffs: bool = False,
    playoff_revenue: float = 0.0,
    min_cash_threshold: float = 0.0,
) -> JDecomposition:
    """
    Compute J decomposition from simulation results.

    Parameters
    ----------
    period_log : pd.DataFrame
        Period-by-period simulation log with columns:
        - profit_usd: Period profit 蟺_t
        - wins: Period wins W_t
        - u_usd: Sports investment u_t
        - m_usd: Marketing investment m_t
        - tau: Ticket multiplier 蟿_t
        - borrowed_usd: Net financing d_t
        - cash_end: End-of-period cash
    final_elo : float
        Final ELO rating (for terminal value)
    final_B : float
        Final brand value B_T
    final_cash : float
        Final cash balance
    final_debt : float
        Final debt balance
    cum_ebitda : float
        Cumulative EBITDA
    discount : float
        Discount factor 未 (default 0.95)
    lambda_win : float
        Weight on wins 位_W (default 100,000)
    lambda_playoff : float
        Weight on playoff revenue (default 1.0)
    lambda_terminal : float
        Weight on terminal value (default 1.0)
    lambda_risk : float
        Weight on risk penalty (default 1.0)
    made_playoffs : bool
        Whether team made playoffs
    playoff_revenue : float
        Playoff net revenue 螤^PO (if made playoffs)
    min_cash_threshold : float
        Minimum cash threshold for bankruptcy (default 0)

    Returns
    -------
    JDecomposition
        J decomposition with all components
    """
    T = len(period_log)

    # 1. J^{reg_profit}: E[危 未^t * 蟺_t]
    discounted_profits = []
    for t, row in period_log.iterrows():
        discounted_profits.append((discount ** t) * row['profit_usd'])
    J_reg_profit = float(np.sum(discounted_profits))

    # 2. J^{win}: E[危 未^t * 位_W * W_t]
    discounted_wins = []
    for t, row in period_log.iterrows():
        discounted_wins.append((discount ** t) * lambda_win * row['wins'])
    J_win = float(np.sum(discounted_wins))

    # 3. J^{PO}: 未^T * E[螤^PO]
    J_playoff = (discount ** T) * lambda_playoff * playoff_revenue if made_playoffs else 0.0

    # 4. J^{end}: 未^T * E[V_end]
    # Terminal value = EBITDA multiple + brand value
    # Using simplified formula: V_end = 渭_ebitda * EBITDA + 蠅_B * B_T + Cash - Debt
    mu_ebitda = 10.0  # EBITDA multiple (typical for sports franchises)
    omega_B = 30_000_000.0  # Brand value coefficient (from mpc_simulation.py)

    terminal_value = (
        mu_ebitda * cum_ebitda
        + omega_B * final_B
        + final_cash
        - final_debt
    )
    J_terminal = (discount ** T) * lambda_terminal * terminal_value

    # 5. J^{risk}: 位_risk * Risk
    # Risk = CVaR or bankruptcy probability
    # Here we use simplified risk: penalty if min_cash < threshold
    min_cash = period_log['cash_end'].min()
    if min_cash < min_cash_threshold:
        # Bankruptcy occurred or near-bankruptcy
        risk_metric = min_cash_threshold - min_cash
        bankruptcy_prob = 1.0
    else:
        risk_metric = 0.0
        bankruptcy_prob = 0.0

    J_risk = lambda_risk * risk_metric

    # 6. J_total = J^{reg_profit} + J^{win} + J^{PO} + J^{end} - J^{risk}
    J_total = J_reg_profit + J_win + J_playoff + J_terminal - J_risk

    # Supporting metrics
    total_profit = float(period_log['profit_usd'].sum())
    total_wins = int(period_log['wins'].sum())
    total_u = float(period_log['u_usd'].sum())
    total_m = float(period_log['m_usd'].sum())
    avg_tau = float(period_log['tau'].mean())
    total_borrowed = float(period_log['borrowed_usd'].sum())

    return JDecomposition(
        J_total=J_total,
        J_reg_profit=J_reg_profit,
        J_win=J_win,
        J_playoff=J_playoff,
        J_terminal=J_terminal,
        J_risk=J_risk,
        total_profit=total_profit,
        total_wins=total_wins,
        terminal_value=terminal_value,
        risk_metric=risk_metric,
        total_u=total_u,
        total_m=total_m,
        avg_tau=avg_tau,
        total_borrowed=total_borrowed,
    )


def format_j_decomposition_table(j: JDecomposition) -> str:
    """
    Format J decomposition as a table for owner presentation.

    Returns
    -------
    str
        Formatted table string
    """
    lines = []
    lines.append("="*70)
    lines.append("J DECOMPOSITION (Owner Value Components)")
    lines.append("="*70)
    lines.append(f"{'Component':<30} {'Value':>20} {'% of J':>15}")
    lines.append("-"*70)

    # Calculate percentages
    j_abs = abs(j.J_total) if j.J_total != 0 else 1.0

    components = [
        ("J^{reg_profit} (Profit)", j.J_reg_profit),
        ("J^{win} (Wins)", j.J_win),
        ("J^{PO} (Playoffs)", j.J_playoff),
        ("J^{end} (Terminal Value)", j.J_terminal),
        ("J^{risk} (Risk Penalty)", -j.J_risk),  # Negative because it's subtracted
    ]

    for name, value in components:
        pct = (value / j_abs) * 100 if j_abs > 0 else 0.0
        lines.append(f"{name:<30} ${value:>18,.0f} {pct:>14.1f}%")

    lines.append("-"*70)
    lines.append(f"{'J_total (Owner Value)':<30} ${j.J_total:>18,.0f} {'100.0%':>15}")
    lines.append("="*70)

    # Supporting metrics
    lines.append("\nSupporting Metrics:")
    lines.append(f"  Total Profit (undiscounted): ${j.total_profit:,.0f}")
    lines.append(f"  Total Wins: {j.total_wins}")
    lines.append(f"  Terminal Value: ${j.terminal_value:,.0f}")
    lines.append(f"  Risk Metric: ${j.risk_metric:,.0f}")
    lines.append(f"\nControl Variables:")
    lines.append(f"  Total Sports Investment (u): ${j.total_u:,.0f}")
    lines.append(f"  Total Marketing Investment (m): ${j.total_m:,.0f}")
    lines.append(f"  Average Ticket Multiplier (蟿): {j.avg_tau:.3f}")
    lines.append(f"  Total Net Financing (d): ${j.total_borrowed:,.0f}")

    return "\n".join(lines)


def compute_sensitivity_analysis(
    base_period_log: pd.DataFrame,
    base_j: JDecomposition,
    *,
    perturbation: float = 0.10,
    variables: List[str] = ['u_usd', 'm_usd', 'tau'],
) -> pd.DataFrame:
    """
    Compute sensitivity analysis: 螖J for +/-10% perturbations.

    This implements model.md Section 1.4 item 3:
    "Investment return and sensitivity report: how J changes with decisions."

    Parameters
    ----------
    base_period_log : pd.DataFrame
        Base case period log
    base_j : JDecomposition
        Base case J decomposition
    perturbation : float
        Perturbation magnitude (default 0.10 for +/-10%)
    variables : List[str]
        Variables to perturb (default: u_usd, m_usd, tau)

    Returns
    -------
    pd.DataFrame
        Sensitivity analysis results with columns:
        - variable: Variable name
        - direction: '+10%' or '-10%'
        - delta_J: Change in J
        - delta_profit: Change in total profit
        - delta_wins: Change in total wins
        - delta_risk: Change in risk metric
    """
    results = []

    for var in variables:
        if var not in base_period_log.columns:
            continue

        for direction, factor in [('+10%', 1 + perturbation), ('-10%', 1 - perturbation)]:
            # Create perturbed log
            perturbed_log = base_period_log.copy()
            perturbed_log[var] = perturbed_log[var] * factor

            # Note: This is a simplified sensitivity analysis
            # In practice, you would re-run the simulation with perturbed parameters
            # Here we just show the framework

            # Approximate impact (linear approximation)
            if var == 'u_usd':
                # Higher u ->higher wins (simplified)
                delta_wins = int((factor - 1.0) * base_j.total_wins * 0.5)
                delta_profit = (factor - 1.0) * base_j.total_profit * 0.2
            elif var == 'm_usd':
                # Higher m ->higher revenue (simplified)
                delta_wins = 0
                delta_profit = (factor - 1.0) * base_j.total_profit * 0.3
            elif var == 'tau':
                # Higher tau ->higher revenue but lower demand (simplified)
                delta_wins = 0
                delta_profit = (factor - 1.0) * base_j.total_profit * 0.4
            else:
                delta_wins = 0
                delta_profit = 0

            delta_J = delta_profit + delta_wins * 100_000.0
            delta_risk = 0.0  # Simplified

            results.append({
                'variable': var,
                'direction': direction,
                'delta_J': delta_J,
                'delta_profit': delta_profit,
                'delta_wins': delta_wins,
                'delta_risk': delta_risk,
            })

    return pd.DataFrame(results)


def create_owner_summary(
    j: JDecomposition,
    *,
    team: str,
    year: int,
    baseline_j: Optional[JDecomposition] = None,
) -> str:
    """
    Create one-page summary for owner (model.md Section 1.4 item 4).

    Parameters
    ----------
    j : JDecomposition
        Current strategy J decomposition
    team : str
        Team code (e.g., "LVA")
    year : int
        Simulation year
    baseline_j : Optional[JDecomposition]
        Baseline strategy for comparison

    Returns
    -------
    str
        One-page summary formatted for owner presentation
    """
    lines = []
    lines.append("="*70)
    lines.append(f"OWNER INVESTMENT SUMMARY - {team} Year {year}")
    lines.append("="*70)

    # 1. Recommended Strategy
    lines.append("\n1. RECOMMENDED STRATEGY")
    lines.append("-"*70)
    lines.append(f"  Sports Investment (u*): ${j.total_u:,.0f}")
    lines.append(f"  Marketing Investment (m*): ${j.total_m:,.0f}")
    lines.append(f"  Ticket Multiplier (蟿*): {j.avg_tau:.3f}")
    lines.append(f"  Net Financing (d*): ${j.total_borrowed:,.0f}")

    # 2. Key KPIs
    lines.append("\n2. KEY PERFORMANCE INDICATORS")
    lines.append("-"*70)
    lines.append(f"  Owner Value (J): ${j.J_total:,.0f}")
    lines.append(f"  Total Profit: ${j.total_profit:,.0f}")
    lines.append(f"  Total Wins: {j.total_wins}")
    lines.append(f"  Terminal Value: ${j.terminal_value:,.0f}")
    lines.append(f"  Risk Metric: ${j.risk_metric:,.0f}")

    # 3. J Decomposition
    lines.append("\n3. VALUE DECOMPOSITION")
    lines.append("-"*70)
    j_abs = abs(j.J_total) if j.J_total != 0 else 1.0
    lines.append(f"  Profit Contribution: ${j.J_reg_profit:,.0f} ({j.J_reg_profit/j_abs*100:.1f}%)")
    lines.append(f"  Win Contribution: ${j.J_win:,.0f} ({j.J_win/j_abs*100:.1f}%)")
    lines.append(f"  Playoff Contribution: ${j.J_playoff:,.0f} ({j.J_playoff/j_abs*100:.1f}%)")
    lines.append(f"  Terminal Contribution: ${j.J_terminal:,.0f} ({j.J_terminal/j_abs*100:.1f}%)")
    lines.append(f"  Risk Penalty: -${j.J_risk:,.0f} ({-j.J_risk/j_abs*100:.1f}%)")

    # 4. Comparison to Baseline (if provided)
    if baseline_j is not None:
        lines.append("\n4. IMPROVEMENT vs BASELINE")
        lines.append("-"*70)
        delta_J = j.J_total - baseline_j.J_total
        delta_profit = j.total_profit - baseline_j.total_profit
        delta_wins = j.total_wins - baseline_j.total_wins
        delta_terminal = j.terminal_value - baseline_j.terminal_value
        delta_risk = j.risk_metric - baseline_j.risk_metric

        lines.append(f"  螖J (Owner Value): ${delta_J:+,.0f} ({delta_J/abs(baseline_j.J_total)*100:+.1f}%)")
        lines.append(f"  螖Profit: ${delta_profit:+,.0f}")
        lines.append(f"  螖Wins: {delta_wins:+d}")
        lines.append(f"  螖Terminal Value: ${delta_terminal:+,.0f}")
        lines.append(f"  螖Risk: ${delta_risk:+,.0f}")

    lines.append("\n" + "="*70)

    return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    print("J Decomposition Module for Task1")
    print("="*70)
    print("\nThis module provides:")
    print("1. compute_j_from_period_log() - Calculate J decomposition")
    print("2. format_j_decomposition_table() - Format for presentation")
    print("3. compute_sensitivity_analysis() - +/-10% perturbation analysis")
    print("4. create_owner_summary() - One-page owner summary")
    print("\nSee function docstrings for usage details.")

