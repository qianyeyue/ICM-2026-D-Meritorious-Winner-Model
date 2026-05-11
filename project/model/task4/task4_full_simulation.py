"""
Task 4: Full Season Simulation with Period-Based Pricing

This script implements the complete Task 4 simulation as specified in paper lines 1046-1047:
- Fix all decisions except ticket price tau_t
- Make tau_t a period-level decision (4 games per period)
- Use full TBD objective function from Task 1
- Simulate complete season with state updates

Usage:
    python task4_full_simulation.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from task4_dynamic_pricing import (
    DynamicPricingParameters,
    calculate_single_game_demand,
    calculate_attendance_rate,
    calculate_win_probability_with_attendance,
)
from task4_tbd_integration import (
    TBDIntegrationParameters,
    update_elo_with_investment,
    update_brand_with_feedback,
    update_star_value_with_investment,
    calculate_playoff_probability,
    calculate_playoff_revenue,
    calculate_terminal_value,
    calculate_cvar_risk,
    simulated_annealing_optimization,
)


def create_full_season_schedule(
    num_home_games: int = 20,
    games_per_period: int = 4,
) -> pd.DataFrame:
    """
    Create full season schedule with period structure.

    Parameters
    ----------
    num_home_games : int
        Total home games in season (WNBA: ~20)
    games_per_period : int
        Games per decision period (default: 4)

    Returns
    -------
    pd.DataFrame
        Schedule with columns: game_id, period, is_home, opponent_elo, opponent_name
    """
    schedule = []
    game_id = 0

    # WNBA teams and approximate Elo ratings
    opponents = [
        ('New York Liberty', 1620),
        ('Connecticut Sun', 1600),
        ('Chicago Sky', 1580),
        ('Indiana Fever', 1520),
        ('Atlanta Dream', 1500),
        ('Washington Mystics', 1540),
        ('Minnesota Lynx', 1590),
        ('Phoenix Mercury', 1570),
        ('Seattle Storm', 1610),
        ('Los Angeles Sparks', 1560),
        ('Dallas Wings', 1530),
        ('Las Vegas Aces', 1650),  # Self (for away games)
    ]

    # Generate home games
    for game_num in range(num_home_games):
        period = game_num // games_per_period

        # Select opponent (cycle through teams)
        opponent_idx = game_num % len(opponents)
        opponent_name, opponent_elo = opponents[opponent_idx]

        # Add some randomness to opponent strength
        opponent_elo += np.random.normal(0, 20)

        schedule.append({
            'game_id': game_id,
            'period': period,
            'is_home': True,
            'opponent_elo': opponent_elo,
            'opponent_name': opponent_name,
        })
        game_id += 1

    return pd.DataFrame(schedule)


def simulate_season_with_fixed_decisions(
    schedule: pd.DataFrame,
    tau_trajectory: List[float],
    initial_state: Dict[str, float],
    fixed_u: float,
    fixed_m: float,
    pricing_params: DynamicPricingParameters,
    tbd_params: TBDIntegrationParameters,
    verbose: bool = True,
) -> Dict:
    """
    Simulate full season with fixed u_t, m_t and period-based tau_t.

    This implements paper lines 1046-1047:
    "固定除了票价 tau_t 以外的决策"

    Parameters
    ----------
    schedule : pd.DataFrame
        Game schedule
    tau_trajectory : List[float]
        Price multipliers by period
    initial_state : Dict[str, float]
        Initial team state
    fixed_u : float
        Fixed competitive investment per period ($)
    fixed_m : float
        Fixed marketing investment per period ($)
    pricing_params : DynamicPricingParameters
        Pricing parameters
    tbd_params : TBDIntegrationParameters
        TBD parameters
    verbose : bool
        Print detailed output

    Returns
    -------
    Dict
        Simulation results with full trajectory
    """
    # Initialize state
    S_t = initial_state['S_0']
    B_t = initial_state['B_0']
    Star_t = initial_state['Star_0']
    Cash_t = initial_state['Cash_0']
    league_pop = initial_state.get('league_pop', 1.0)

    # Storage for results
    results = {
        'periods': [],
        'states': [],
        'decisions': [],
        'revenues': [],
        'costs': [],
        'profits': [],
        'wins': [],
        'attendance_rates': [],
        'game_details': [],
    }

    periods = schedule['period'].unique()
    cumulative_wins = 0
    cumulative_profit = 0
    cumulative_ebitda = 0

    if verbose:
        print("=" * 80)
        print("TASK 4: FULL SEASON SIMULATION")
        print("=" * 80)
        print(f"\nInitial State:")
        print(f"  Elo (S_0): {S_t:.1f}")
        print(f"  Brand (B_0): {B_t:.3f}")
        print(f"  Star (Star_0): {Star_t:.1f}")
        print(f"  Cash: ${Cash_t/1e6:.1f}M")
        print(f"\nFixed Decisions:")
        print(f"  Competitive Investment (u_t): ${fixed_u/1e6:.2f}M per period")
        print(f"  Marketing Investment (m_t): ${fixed_m/1e6:.2f}M per period")
        print(f"\nPeriod Structure:")
        print(f"  Total Periods: {len(periods)}")
        print(f"  Games per Period: {len(schedule[schedule['period']==0])}")
        print(f"  Total Home Games: {len(schedule)}")
        print("=" * 80)

    # Simulate each period
    for t, period in enumerate(periods):
        if verbose:
            print(f"\n{'='*80}")
            print(f"PERIOD {period} (Games {period*4+1}-{(period+1)*4})")
            print(f"{'='*80}")

        # Get period games
        period_games = schedule[schedule['period'] == period]

        # Current period pricing decision
        tau_t = tau_trajectory[t]

        if verbose:
            print(f"\nDecision: tau_t = {tau_t:.3f}")
            print(f"\nState at Period Start:")
            print(f"  Elo: {S_t:.1f}")
            print(f"  Brand: {B_t:.3f}")
            print(f"  Star: {Star_t:.1f}")
            print(f"  Cash: ${Cash_t/1e6:.2f}M")

        # Simulate games in this period
        period_revenue = 0.0
        period_attendance_rates = []
        game_results = []
        expected_probs = []
        game_details = []

        for idx, game in period_games.iterrows():
            # Calculate demand
            demand = calculate_single_game_demand(
                tau=tau_t,
                S_t=S_t,
                Star_t=Star_t,
                S_opp=game['opponent_elo'],
                league_pop=league_pop,
                period=t,
                params=pricing_params,
            )

            # Calculate revenue
            price = tau_t * pricing_params.base_price
            game_revenue = price * demand
            period_revenue += game_revenue

            # Track attendance
            attendance_rate = calculate_attendance_rate(demand, pricing_params)
            period_attendance_rates.append(attendance_rate)

            # Calculate win probability with attendance effect
            win_prob = calculate_win_probability_with_attendance(
                S_home=S_t,
                S_away=game['opponent_elo'],
                attendance_rate=attendance_rate,
                params=pricing_params,
            )

            # Simulate game outcome
            result = np.random.binomial(1, win_prob)
            game_results.append(result)
            expected_probs.append(win_prob)

            # Store game details
            game_details.append({
                'game_id': game['game_id'],
                'period': period,
                'opponent': game['opponent_name'],
                'opponent_elo': game['opponent_elo'],
                'tau': tau_t,
                'price': price,
                'demand': demand,
                'attendance_rate': attendance_rate,
                'win_prob': win_prob,
                'result': result,
                'revenue': game_revenue,
            })

        # Period metrics
        period_wins = sum(game_results)
        avg_attendance = np.mean(period_attendance_rates)

        # Calculate period costs
        period_cost = fixed_u + fixed_m + 500000  # Base operating cost
        period_profit = period_revenue - period_cost

        # Update cash
        Cash_t += period_profit

        # Update cumulative metrics
        cumulative_wins += period_wins
        cumulative_profit += tbd_params.delta ** t * period_profit
        cumulative_ebitda += period_profit

        if verbose:
            print(f"\nPeriod Results:")
            print(f"  Games: {len(period_games)}")
            print(f"  Wins: {period_wins}/{len(period_games)} ({period_wins/len(period_games):.1%})")
            print(f"  Avg Attendance: {avg_attendance:.1%}")
            print(f"  Revenue: ${period_revenue/1000:.1f}K")
            print(f"  Costs: ${period_cost/1000:.1f}K")
            print(f"  Profit: ${period_profit/1000:.1f}K")
            print(f"  Cash: ${Cash_t/1e6:.2f}M")

        # Store period results
        results['periods'].append(period)
        results['states'].append({
            'S_t': S_t,
            'B_t': B_t,
            'Star_t': Star_t,
            'Cash_t': Cash_t,
        })
        results['decisions'].append({
            'tau_t': tau_t,
            'u_t': fixed_u,
            'm_t': fixed_m,
        })
        results['revenues'].append(period_revenue)
        results['costs'].append(period_cost)
        results['profits'].append(period_profit)
        results['wins'].append(period_wins)
        results['attendance_rates'].append(avg_attendance)
        results['game_details'].extend(game_details)

        # Update state for next period
        S_t = update_elo_with_investment(
            S_t, game_results, expected_probs, fixed_u, tbd_params
        )

        B_t = update_brand_with_feedback(
            B_t, period_wins, avg_attendance, Star_t, fixed_m, tbd_params
        )

        Star_t = update_star_value_with_investment(
            Star_t, fixed_u, fixed_m, tbd_params
        )

        if verbose:
            print(f"\nState Updates:")
            print(f"  Elo: {S_t:.1f} (Δ={S_t - results['states'][-1]['S_t']:+.1f})")
            print(f"  Brand: {B_t:.3f} (Δ={B_t - results['states'][-1]['B_t']:+.3f})")
            print(f"  Star: {Star_t:.1f} (Δ={Star_t - results['states'][-1]['Star_t']:+.1f})")

    # Calculate playoff component
    league_standings = [cumulative_wins] + list(np.random.normal(20, 5, 12))
    playoff_prob = calculate_playoff_probability(
        cumulative_wins, league_standings, tbd_params
    )
    playoff_revenue = calculate_playoff_revenue(playoff_prob, S_t, tbd_params)

    # Calculate terminal value
    terminal_value = calculate_terminal_value(
        cumulative_ebitda, B_t, league_pop, tbd_params
    )

    # Calculate total objective
    total_objective = (
        cumulative_profit
        + tbd_params.lambda_W * cumulative_wins
        + tbd_params.delta ** len(periods) * playoff_revenue
        + tbd_params.delta ** len(periods) * terminal_value
    )

    # Risk calculation (simplified - would need MC for full CVaR)
    profit_samples = np.array(results['profits'])
    risk_penalty = tbd_params.lambda_risk * np.std(profit_samples)

    total_objective -= risk_penalty

    if verbose:
        print(f"\n{'='*80}")
        print("SEASON SUMMARY")
        print(f"{'='*80}")
        print(f"\nFinal State:")
        print(f"  Elo: {S_t:.1f}")
        print(f"  Brand: {B_t:.3f}")
        print(f"  Star: {Star_t:.1f}")
        print(f"  Cash: ${Cash_t/1e6:.2f}M")
        print(f"\nSeason Performance:")
        print(f"  Total Wins: {cumulative_wins}/{len(schedule)} ({cumulative_wins/len(schedule):.1%})")
        print(f"  Avg Attendance: {np.mean(results['attendance_rates']):.1%}")
        print(f"  Total Revenue: ${sum(results['revenues'])/1e6:.2f}M")
        print(f"  Total Costs: ${sum(results['costs'])/1e6:.2f}M")
        print(f"  Total Profit: ${sum(results['profits'])/1e6:.2f}M")
        print(f"\nPlayoff:")
        print(f"  Probability: {playoff_prob:.1%}")
        print(f"  Expected Revenue: ${playoff_revenue/1e6:.2f}M")
        print(f"\nValuation:")
        print(f"  Terminal Value: ${terminal_value/1e6:.2f}M")
        print(f"\nObjective Function:")
        print(f"  Regular Season: ${cumulative_profit/1e6:.2f}M")
        print(f"  Wins Value: ${tbd_params.lambda_W * cumulative_wins/1e6:.2f}M")
        print(f"  Playoff Value: ${playoff_revenue/1e6:.2f}M")
        print(f"  Terminal Value: ${terminal_value/1e6:.2f}M")
        print(f"  Risk Penalty: ${-risk_penalty/1e6:.2f}M")
        print(f"  Total J: ${total_objective/1e6:.2f}M")
        print(f"{'='*80}")

    # Store summary
    results['summary'] = {
        'cumulative_wins': cumulative_wins,
        'cumulative_profit': cumulative_profit,
        'cumulative_ebitda': cumulative_ebitda,
        'playoff_prob': playoff_prob,
        'playoff_revenue': playoff_revenue,
        'terminal_value': terminal_value,
        'risk_penalty': risk_penalty,
        'total_objective': total_objective,
        'final_state': {
            'S_T': S_t,
            'B_T': B_t,
            'Star_T': Star_t,
            'Cash_T': Cash_t,
        },
    }

    return results


def plot_simulation_results(results: Dict, save_path: str = None):
    """
    Visualize simulation results.

    Parameters
    ----------
    results : Dict
        Simulation results from simulate_season_with_fixed_decisions()
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))

    periods = results['periods']

    # Plot 1: State Evolution
    ax = axes[0, 0]
    states = results['states']
    S_trajectory = [s['S_t'] for s in states]
    B_trajectory = [s['B_t'] for s in states]

    ax2 = ax.twinx()
    line1 = ax.plot(periods, S_trajectory, 'o-', color='#2E86AB', label='Elo', linewidth=2)
    line2 = ax2.plot(periods, B_trajectory, 's-', color='#F18F01', label='Brand', linewidth=2)

    ax.set_xlabel('Period', fontsize=11)
    ax.set_ylabel('Elo Rating', fontsize=11, color='#2E86AB')
    ax2.set_ylabel('Brand Index', fontsize=11, color='#F18F01')
    ax.set_title('State Evolution', fontsize=12, fontweight='bold')
    ax.tick_params(axis='y', labelcolor='#2E86AB')
    ax2.tick_params(axis='y', labelcolor='#F18F01')
    ax.grid(True, alpha=0.3)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left')

    # Plot 2: Pricing Decisions
    ax = axes[0, 1]
    decisions = results['decisions']
    tau_trajectory = [d['tau_t'] for d in decisions]

    ax.bar(periods, tau_trajectory, color='#A23B72', alpha=0.7, edgecolor='black')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Base Price')
    ax.set_xlabel('Period', fontsize=11)
    ax.set_ylabel('Price Multiplier (τ)', fontsize=11)
    ax.set_title('Pricing Decisions', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Revenue & Profit
    ax = axes[1, 0]
    revenues = np.array(results['revenues']) / 1000
    profits = np.array(results['profits']) / 1000

    x = np.arange(len(periods))
    width = 0.35

    ax.bar(x - width/2, revenues, width, label='Revenue', color='#06A77D', alpha=0.7)
    ax.bar(x + width/2, profits, width, label='Profit', color='#2E86AB', alpha=0.7)
    ax.set_xlabel('Period', fontsize=11)
    ax.set_ylabel('Amount ($K)', fontsize=11)
    ax.set_title('Revenue & Profit by Period', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(periods)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Wins & Attendance
    ax = axes[1, 1]
    wins = results['wins']
    attendance_rates = np.array(results['attendance_rates']) * 100

    ax2 = ax.twinx()
    line1 = ax.bar(periods, wins, color='#F18F01', alpha=0.7, label='Wins')
    line2 = ax2.plot(periods, attendance_rates, 'o-', color='#06A77D', linewidth=2, markersize=8, label='Attendance')

    ax.set_xlabel('Period', fontsize=11)
    ax.set_ylabel('Wins', fontsize=11, color='#F18F01')
    ax2.set_ylabel('Attendance Rate (%)', fontsize=11, color='#06A77D')
    ax.set_title('Performance Metrics', fontsize=12, fontweight='bold')
    ax.tick_params(axis='y', labelcolor='#F18F01')
    ax2.tick_params(axis='y', labelcolor='#06A77D')
    ax.grid(True, alpha=0.3)

    # Plot 5: Cumulative Profit
    ax = axes[2, 0]
    cumulative_profits = np.cumsum(results['profits']) / 1000

    ax.plot(periods, cumulative_profits, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax.fill_between(periods, 0, cumulative_profits, alpha=0.2, color='#2E86AB')
    ax.set_xlabel('Period', fontsize=11)
    ax.set_ylabel('Cumulative Profit ($K)', fontsize=11)
    ax.set_title('Cumulative Profit Trajectory', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 6: Objective Components
    ax = axes[2, 1]
    summary = results['summary']

    components = {
        'Regular\nSeason': summary['cumulative_profit'] / 1e6,
        'Wins\nValue': 20000 * summary['cumulative_wins'] / 1e6,
        'Playoff': summary['playoff_revenue'] / 1e6,
        'Terminal\nValue': summary['terminal_value'] / 1e6,
        'Risk\nPenalty': -summary['risk_penalty'] / 1e6,
    }

    colors = ['#2E86AB', '#F18F01', '#06A77D', '#A23B72', '#E63946']
    bars = ax.bar(components.keys(), components.values(), color=colors, alpha=0.7, edgecolor='black')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:.1f}M',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=9, fontweight='bold')

    ax.set_ylabel('Value ($M)', fontsize=11)
    ax.set_title('Objective Function Components', fontsize=12, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved: {save_path}")

    plt.show()


def main():
    """
    Run Task 4 full simulation demonstration.
    """
    print("\n" + "=" * 80)
    print("TASK 4: FULL SEASON SIMULATION WITH PERIOD-BASED PRICING")
    print("=" * 80)
    print("\nPaper Reference: Lines 1046-1047")
    print("  'Fixed all decisions except ticket price tau_t'")
    print("  'tau is now a period-level decision, not season-level'")
    print("=" * 80)

    # Setup
    np.random.seed(42)

    # Adjust parameters to show realistic demand elasticity
    pricing_params = DynamicPricingParameters(
        beta_0=9.2,          # Much lower baseline demand (was 10.5)
        epsilon_0=1.5,       # Much higher initial elasticity (was 0.8)
        lambda_decay=0.08,   # Slower decay (was 0.15)
        capacity=12000,      # Keep capacity
        beta_S=0.0015,       # Lower Elo effect (was 0.002)
        beta_star=0.25,      # Lower star effect (was 0.3)
    )
    tbd_params = TBDIntegrationParameters()

    # Create schedule (20 home games, 4 games per period = 5 periods)
    schedule = create_full_season_schedule(num_home_games=20, games_per_period=4)

    print(f"\nSchedule Created:")
    print(f"  Total Home Games: {len(schedule)}")
    print(f"  Games per Period: 4")
    print(f"  Total Periods: {schedule['period'].nunique()}")

    # Initial state (Las Vegas Aces)
    initial_state = {
        'S_0': 1650,      # Strong team
        'B_0': 1.6,       # Above-average brand
        'Star_0': 55,     # High star power (A'ja Wilson)
        'Cash_0': 10000000,  # $10M cash
        'league_pop': 1.2,   # Growing league
    }

    # Fixed decisions (adjusted to be more realistic)
    fixed_u = 800000   # $0.8M competitive investment per period (was $1.5M)
    fixed_m = 500000   # $0.5M marketing investment per period (was $1.0M)

    # Pricing strategy (wider range to see demand response)
    tau_trajectory = [0.85, 0.95, 1.05, 1.15, 1.25]  # 5 periods

    print(f"\nPricing Strategy (wider range to test demand elasticity):")
    for i, tau in enumerate(tau_trajectory):
        print(f"  Period {i}: tau = {tau:.2f}")

    print(f"\nFixed Decisions (more realistic):")
    print(f"  Competitive Investment: ${fixed_u/1e6:.1f}M per period")
    print(f"  Marketing Investment: ${fixed_m/1e6:.1f}M per period")

    # Run simulation
    results = simulate_season_with_fixed_decisions(
        schedule=schedule,
        tau_trajectory=tau_trajectory,
        initial_state=initial_state,
        fixed_u=fixed_u,
        fixed_m=fixed_m,
        pricing_params=pricing_params,
        tbd_params=tbd_params,
        verbose=True,
    )

    # Visualize
    print(f"\nGenerating visualizations...")
    plot_simulation_results(results, save_path='./task4_results/full_simulation.png')

    # Export game details
    game_df = pd.DataFrame(results['game_details'])
    game_df.to_csv('./task4_results/game_details.csv', index=False)
    print(f"Saved: ./task4_results/game_details.csv")

    # Print demand analysis
    print("\n" + "=" * 80)
    print("DEMAND ELASTICITY ANALYSIS")
    print("=" * 80)
    for i, period in enumerate(results['periods']):
        avg_demand = np.mean([g['demand'] for g in results['game_details'] if g['period'] == period])
        avg_attendance = results['attendance_rates'][i]
        tau = tau_trajectory[i]
        revenue = results['revenues'][i]

        print(f"\nPeriod {period}:")
        print(f"  Price Multiplier: {tau:.2f}")
        print(f"  Avg Demand: {avg_demand:,.0f} tickets")
        print(f"  Attendance Rate: {avg_attendance:.1%}")
        print(f"  Revenue: ${revenue/1000:.1f}K")
        print(f"  Revenue per Ticket: ${revenue/avg_demand/4:.2f}")

    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    import os
    os.makedirs('./task4_results', exist_ok=True)
    main()
