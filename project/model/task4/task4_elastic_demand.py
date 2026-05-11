"""
Task 4: True Elastic Demand Simulation

This version manually calibrates demand to show realistic price elasticity
by adjusting the demand calculation to prevent always hitting capacity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List

from task4_full_simulation import create_full_season_schedule
from task4_tbd_integration import (
    TBDIntegrationParameters,
    update_elo_with_investment,
    update_brand_with_feedback,
    update_star_value_with_investment,
    calculate_playoff_probability,
    calculate_playoff_revenue,
    calculate_terminal_value,
)


def calculate_elastic_demand(
    tau: float,
    S_t: float,
    Star_t: float,
    S_opp: float,
    league_pop: float,
    period: int,
    capacity: float = 12000,
    base_demand: float = 5500,
    epsilon_0: float = 2.5,
    lambda_decay: float = 0.1,
) -> float:
    """
    Calculate demand with true price elasticity.

    Uses a calibrated formula so demand varies with price
    and doesn't always hit capacity.
    """
    # Time-varying elasticity (higher = more price-sensitive)
    epsilon_t = epsilon_0 * np.exp(-lambda_decay * period)

    # Base demand adjusted by team factors
    team_factor = 1.0 + (S_t - 1600) / 1000  # Elo effect
    star_factor = 1.0 + Star_t / 200  # Star effect
    opp_factor = 1.0 + (S_opp - 1550) / 2000  # Opponent effect
    league_factor = league_pop  # League popularity

    # Adjusted base demand
    adjusted_base = base_demand * team_factor * star_factor * opp_factor * league_factor

    # Price effect (power function for elasticity)
    price_effect = tau ** (-epsilon_t)

    # Final demand
    demand = adjusted_base * price_effect

    # Add realistic noise (+/-15%)
    noise_factor = np.random.lognormal(0, 0.15)
    demand *= noise_factor

    # Apply capacity constraint
    demand = min(demand, capacity)

    return demand


def simulate_elastic_season(
    schedule: pd.DataFrame,
    tau_trajectory: List[float],
    initial_state: Dict[str, float],
    fixed_u: float,
    fixed_m: float,
    tbd_params: TBDIntegrationParameters,
    capacity: float = 12000,
    base_price: float = 50,
    base_demand: float = 5500,
    epsilon_0: float = 2.5,
    verbose: bool = True,
) -> Dict:
    """
    Simulate season with elastic demand.
    """
    # Initialize state
    S_t = initial_state['S_0']
    B_t = initial_state['B_0']
    Star_t = initial_state['Star_0']
    Cash_t = initial_state['Cash_0']
    league_pop = initial_state.get('league_pop', 1.0)

    results = {
        'periods': [],
        'states': [],
        'decisions': [],
        'revenues': [],
        'costs': [],
        'profits': [],
        'wins': [],
        'attendance_rates': [],
        'demands': [],
        'game_details': [],
    }

    periods = schedule['period'].unique()
    cumulative_wins = 0
    cumulative_profit = 0
    cumulative_ebitda = 0

    if verbose:
        print("=" * 80)
        print("ELASTIC DEMAND SIMULATION")
        print("=" * 80)

    for t, period in enumerate(periods):
        period_games = schedule[schedule['period'] == period]
        tau_t = tau_trajectory[t]

        if verbose:
            print(f"\nPERIOD {period}: tau = {tau_t:.2f}")

        # Simulate games
        period_revenue = 0.0
        period_demands = []
        period_attendance_rates = []
        game_results = []
        expected_probs = []

        for idx, game in period_games.iterrows():
            # Calculate elastic demand
            demand = calculate_elastic_demand(
                tau=tau_t,
                S_t=S_t,
                Star_t=Star_t,
                S_opp=game['opponent_elo'],
                league_pop=league_pop,
                period=t,
                capacity=capacity,
                base_demand=base_demand,
                epsilon_0=epsilon_0,
            )

            # Revenue
            price = tau_t * base_price
            game_revenue = price * demand
            period_revenue += game_revenue

            # Attendance rate
            attendance_rate = demand / capacity
            period_attendance_rates.append(attendance_rate)
            period_demands.append(demand)

            # Win probability (with attendance effect)
            H_base = 100
            H_effective = H_base * (2 * attendance_rate - 1)
            delta = (S_t - game['opponent_elo']) + H_effective
            win_prob = 1.0 / (1.0 + 10 ** (-delta / 400))

            # Simulate outcome
            result = np.random.binomial(1, win_prob)
            game_results.append(result)
            expected_probs.append(win_prob)

        # Period metrics
        period_wins = sum(game_results)
        avg_attendance = np.mean(period_attendance_rates)
        avg_demand = np.mean(period_demands)

        # Costs and profit
        period_cost = fixed_u + fixed_m + 500000
        period_profit = period_revenue - period_cost
        Cash_t += period_profit

        cumulative_wins += period_wins
        cumulative_profit += tbd_params.delta ** t * period_profit
        cumulative_ebitda += period_profit

        if verbose:
            print(f"  Wins: {period_wins}/4, Avg Demand: {avg_demand:.0f}, Attendance: {avg_attendance:.1%}")
            print(f"  Revenue: ${period_revenue/1000:.0f}K, Profit: ${period_profit/1000:.0f}K")

        # Store results
        results['periods'].append(period)
        results['states'].append({'S_t': S_t, 'B_t': B_t, 'Star_t': Star_t, 'Cash_t': Cash_t})
        results['decisions'].append({'tau_t': tau_t, 'u_t': fixed_u, 'm_t': fixed_m})
        results['revenues'].append(period_revenue)
        results['costs'].append(period_cost)
        results['profits'].append(period_profit)
        results['wins'].append(period_wins)
        results['attendance_rates'].append(avg_attendance)
        results['demands'].append(avg_demand)

        # Update state
        S_t = update_elo_with_investment(S_t, game_results, expected_probs, fixed_u, tbd_params)
        B_t = update_brand_with_feedback(B_t, period_wins, avg_attendance, Star_t, fixed_m, tbd_params)
        Star_t = update_star_value_with_investment(Star_t, fixed_u, fixed_m, tbd_params)

    # Calculate objective
    league_standings = [cumulative_wins] + list(np.random.normal(20, 5, 12))
    playoff_prob = calculate_playoff_probability(cumulative_wins, league_standings, tbd_params)
    playoff_revenue = calculate_playoff_revenue(playoff_prob, S_t, tbd_params)
    terminal_value = calculate_terminal_value(cumulative_ebitda, B_t, league_pop, tbd_params)

    total_objective = (
        cumulative_profit
        + tbd_params.lambda_W * cumulative_wins
        + tbd_params.delta ** len(periods) * playoff_revenue
        + tbd_params.delta ** len(periods) * terminal_value
    )

    results['summary'] = {
        'cumulative_wins': cumulative_wins,
        'cumulative_profit': cumulative_profit,
        'playoff_prob': playoff_prob,
        'playoff_revenue': playoff_revenue,
        'terminal_value': terminal_value,
        'total_objective': total_objective,
        'final_state': {'S_T': S_t, 'B_T': B_t, 'Star_T': Star_t, 'Cash_T': Cash_t},
    }

    if verbose:
        print(f"\n{'='*80}")
        print("SEASON SUMMARY")
        print(f"{'='*80}")
        print(f"Total Wins: {cumulative_wins}/20 ({cumulative_wins/20:.1%})")
        print(f"Avg Attendance: {np.mean(results['attendance_rates']):.1%}")
        print(f"Total Revenue: ${sum(results['revenues'])/1e6:.2f}M")
        print(f"Total Profit: ${sum(results['profits'])/1e6:.2f}M")
        print(f"Objective J: ${total_objective/1e6:.2f}M")

    return results


def main():
    """
    Run elastic demand simulation with multiple strategies.
    """
    print("\n" + "=" * 80)
    print("TASK 4: TRUE ELASTIC DEMAND SIMULATION")
    print("=" * 80)
    print("\nDemand now responds realistically to price changes")
    print("=" * 80)

    np.random.seed(42)

    tbd_params = TBDIntegrationParameters()
    schedule = create_full_season_schedule(num_home_games=20, games_per_period=4)

    initial_state = {
        'S_0': 1650,
        'B_0': 1.6,
        'Star_0': 55,
        'Cash_0': 10000000,
        'league_pop': 1.2,
    }

    fixed_u = 800000
    fixed_m = 500000

    # Test strategies with wider price range
    strategies = {
        'Low Price (High Volume)': [0.70, 0.75, 0.80, 0.85, 0.90],
        'Moderate Increase': [0.85, 0.95, 1.05, 1.15, 1.25],
        'High Price (Low Volume)': [1.20, 1.25, 1.30, 1.35, 1.40],
        'Optimal Dynamic': [0.80, 0.90, 1.00, 1.10, 1.20],
    }

    results_comparison = []

    for strategy_name, tau_trajectory in strategies.items():
        print(f"\n{'='*80}")
        print(f"STRATEGY: {strategy_name}")
        print(f"Pricing: {tau_trajectory}")
        print(f"{'='*80}")

        results = simulate_elastic_season(
            schedule=schedule,
            tau_trajectory=tau_trajectory,
            initial_state=initial_state.copy(),
            fixed_u=fixed_u,
            fixed_m=fixed_m,
            tbd_params=tbd_params,
            verbose=False,
        )

        results_comparison.append({
            'strategy': strategy_name,
            'tau_trajectory': tau_trajectory,
            'avg_tau': np.mean(tau_trajectory),
            'avg_attendance': np.mean(results['attendance_rates']),
            'avg_demand': np.mean(results['demands']),
            'total_revenue': sum(results['revenues']),
            'total_profit': sum(results['profits']),
            'total_wins': results['summary']['cumulative_wins'],
            'objective': results['summary']['total_objective'],
        })

        print(f"\nResults:")
        print(f"  Avg Price Multiplier: {np.mean(tau_trajectory):.2f}")
        print(f"  Avg Demand: {np.mean(results['demands']):.0f} tickets/game")
        print(f"  Avg Attendance: {np.mean(results['attendance_rates']):.1%}")
        print(f"  Total Revenue: ${sum(results['revenues'])/1e6:.2f}M")
        print(f"  Total Profit: ${sum(results['profits'])/1e6:.2f}M")
        print(f"  Total Wins: {results['summary']['cumulative_wins']}/20")
        print(f"  Objective J: ${results['summary']['total_objective']/1e6:.2f}M")

    # Create comparison
    comparison_df = pd.DataFrame(results_comparison)

    print(f"\n{'='*80}")
    print("STRATEGY COMPARISON")
    print(f"{'='*80}")
    print(comparison_df[['strategy', 'avg_tau', 'avg_demand', 'avg_attendance', 'total_revenue', 'objective']].to_string(index=False))

    # Find best
    best_idx = comparison_df['objective'].idxmax()
    best_strategy = comparison_df.iloc[best_idx]

    print(f"\n{'='*80}")
    print("BEST STRATEGY")
    print(f"{'='*80}")
    print(f"Strategy: {best_strategy['strategy']}")
    print(f"Avg Price: {best_strategy['avg_tau']:.2f}x")
    print(f"Avg Demand: {best_strategy['avg_demand']:.0f} tickets")
    print(f"Avg Attendance: {best_strategy['avg_attendance']:.1%}")
    print(f"Total Revenue: ${best_strategy['total_revenue']/1e6:.2f}M")
    print(f"Objective J: ${best_strategy['objective']/1e6:.2f}M")

    # Detailed simulation for best
    print(f"\n{'='*80}")
    print("DETAILED SIMULATION FOR BEST STRATEGY")
    print(f"{'='*80}")

    best_results = simulate_elastic_season(
        schedule=schedule,
        tau_trajectory=best_strategy['tau_trajectory'],
        initial_state=initial_state.copy(),
        fixed_u=fixed_u,
        fixed_m=fixed_m,
        tbd_params=tbd_params,
        verbose=True,
    )

    # Visualize
    plot_elastic_results(results_comparison, best_results)

    # Export
    comparison_df.to_csv('./task4_results/elastic_demand_comparison.csv', index=False)
    print(f"\nSaved: ./task4_results/elastic_demand_comparison.csv")

    print(f"\n{'='*80}")
    print("SIMULATION COMPLETE")
    print(f"{'='*80}")


def plot_elastic_results(results_comparison: List[Dict], best_results: Dict):
    """
    Visualize elastic demand results.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    strategies = [r['strategy'] for r in results_comparison]
    colors = ['#2E86AB', '#F18F01', '#06A77D', '#A23B72']

    # Plot 1: Price vs Attendance
    ax = axes[0, 0]
    avg_taus = [r['avg_tau'] for r in results_comparison]
    avg_attendances = [r['avg_attendance'] * 100 for r in results_comparison]

    for i, (tau, att, strat) in enumerate(zip(avg_taus, avg_attendances, strategies)):
        ax.scatter(tau, att, s=300, color=colors[i], alpha=0.7, edgecolor='black', linewidth=2, label=strat)

    ax.set_xlabel('Average Price Multiplier (蟿)', fontsize=11)
    ax.set_ylabel('Average Attendance Rate (%)', fontsize=11)
    ax.set_title('Price-Attendance Relationship', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=100, color='red', linestyle='--', alpha=0.3, label='Capacity')

    # Plot 2: Demand by Strategy
    ax = axes[0, 1]
    avg_demands = [r['avg_demand'] for r in results_comparison]
    bars = ax.bar(range(len(strategies)), avg_demands, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=12000, color='red', linestyle='--', alpha=0.5, label='Capacity')
    ax.set_ylabel('Avg Demand (tickets/game)', fontsize=11)
    ax.set_title('Average Demand by Strategy', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=15, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Revenue Comparison
    ax = axes[0, 2]
    revenues = [r['total_revenue'] / 1e6 for r in results_comparison]
    bars = ax.bar(range(len(strategies)), revenues, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Total Revenue ($M)', fontsize=11)
    ax.set_title('Total Revenue by Strategy', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=15, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Objective Function
    ax = axes[1, 0]
    objectives = [r['objective'] / 1e6 for r in results_comparison]
    bars = ax.bar(range(len(strategies)), objectives, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Objective Value J ($M)', fontsize=11)
    ax.set_title('Total Objective Function', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=15, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Highlight best
    best_idx = np.argmax(objectives)
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(3)

    # Plot 5: Best Strategy - Period Trajectory
    ax = axes[1, 1]
    periods = best_results['periods']
    demands = best_results['demands']
    attendance_rates = [r * 100 for r in best_results['attendance_rates']]

    ax2 = ax.twinx()
    line1 = ax.plot(periods, demands, 'o-', color='#2E86AB', linewidth=2, markersize=8, label='Demand')
    line2 = ax2.plot(periods, attendance_rates, 's-', color='#F18F01', linewidth=2, markersize=8, label='Attendance %')

    ax.set_xlabel('Period', fontsize=11)
    ax.set_ylabel('Demand (tickets)', fontsize=11, color='#2E86AB')
    ax2.set_ylabel('Attendance Rate (%)', fontsize=11, color='#F18F01')
    ax.set_title('Best Strategy: Demand Trajectory', fontsize=12, fontweight='bold')
    ax.tick_params(axis='y', labelcolor='#2E86AB')
    ax2.tick_params(axis='y', labelcolor='#F18F01')
    ax.grid(True, alpha=0.3)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left')

    # Plot 6: Revenue-Profit Trade-off
    ax = axes[1, 2]
    revenues = [r['total_revenue'] / 1e6 for r in results_comparison]
    profits = [r['total_profit'] / 1e6 for r in results_comparison]

    for i, (rev, prof, strat) in enumerate(zip(revenues, profits, strategies)):
        ax.scatter(rev, prof, s=300, color=colors[i], alpha=0.7, edgecolor='black', linewidth=2, label=strat)

    ax.set_xlabel('Total Revenue ($M)', fontsize=11)
    ax.set_ylabel('Total Profit ($M)', fontsize=11)
    ax.set_title('Revenue-Profit Trade-off', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = '../../pic/task4/elastic_demand_analysis.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    import os
    os.makedirs('./task4_results', exist_ok=True)
    os.makedirs('./pic/task4', exist_ok=True)
    main()

