"""
Task 4: 连续优化票价策略
使用优化算法找到最优的连续tau轨迹，而不是预设的离散值
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from typing import Dict, List
import matplotlib.pyplot as plt

from task4_tbd_integration import (
    TBDIntegrationParameters,
    update_elo_with_investment,
    update_brand_with_feedback,
    update_star_value_with_investment,
    calculate_playoff_probability,
    calculate_playoff_revenue,
    calculate_terminal_value,
)
from task4_elastic_demand import calculate_elastic_demand


def simulate_with_continuous_tau(
    schedule: pd.DataFrame,
    tau_trajectory: np.ndarray,
    initial_state: Dict[str, float],
    fixed_u: float,
    fixed_m: float,
    tbd_params: TBDIntegrationParameters,
    capacity: float = 12000,
    base_price: float = 50,
    base_demand: float = 5500,
    epsilon_0: float = 2.5,
) -> Dict:
    """
    模拟赛季，使用连续的tau轨迹
    """
    # 初始化
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
    }

    periods = schedule['period'].unique()
    cumulative_wins = 0
    cumulative_profit = 0
    cumulative_ebitda = 0

    for t, period in enumerate(periods):
        period_games = schedule[schedule['period'] == period]
        tau_t = tau_trajectory[t]

        # 模拟比赛
        period_revenue = 0.0
        period_demands = []
        period_attendance_rates = []
        game_results = []
        expected_probs = []

        for idx, game in period_games.iterrows():
            # 计算需求
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

            # 收入
            price = tau_t * base_price
            game_revenue = price * demand
            period_revenue += game_revenue

            # 上座率
            attendance_rate = demand / capacity
            period_attendance_rates.append(attendance_rate)
            period_demands.append(demand)

            # 胜率
            H_base = 100
            H_effective = H_base * (2 * attendance_rate - 1)
            delta = (S_t - game['opponent_elo']) + H_effective
            win_prob = 1.0 / (1.0 + 10 ** (-delta / 400))

            # 使用期望胜场而不是随机抽样，让wins反映上座率的影响
            result = win_prob  # 期望值
            game_results.append(result)
            expected_probs.append(win_prob)

        # 期间指标
        period_wins = sum(game_results)
        avg_attendance = np.mean(period_attendance_rates)
        avg_demand = np.mean(period_demands)

        period_cost = fixed_u + fixed_m + 500000
        period_profit = period_revenue - period_cost
        Cash_t += period_profit

        cumulative_wins += period_wins
        cumulative_profit += tbd_params.delta ** t * period_profit
        cumulative_ebitda += period_profit

        # 存储结果
        results['periods'].append(period)
        results['states'].append({'S_t': S_t, 'B_t': B_t, 'Star_t': Star_t, 'Cash_t': Cash_t})
        results['decisions'].append({'tau_t': tau_t, 'u_t': fixed_u, 'm_t': fixed_m})
        results['revenues'].append(period_revenue)
        results['costs'].append(period_cost)
        results['profits'].append(period_profit)
        results['wins'].append(period_wins)
        results['attendance_rates'].append(avg_attendance)
        results['demands'].append(avg_demand)

        # 更新状态
        S_t = update_elo_with_investment(S_t, game_results, expected_probs, fixed_u, tbd_params)
        B_t = update_brand_with_feedback(B_t, period_wins, avg_attendance, Star_t, fixed_m, tbd_params)
        Star_t = update_star_value_with_investment(Star_t, fixed_u, fixed_m, tbd_params)

    # 计算目标函数
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

    return results


def objective_function(tau_trajectory, schedule, initial_state, fixed_u, fixed_m, tbd_params,
                      capacity, base_price, base_demand, epsilon_0, n_simulations=5):
    """
    目标函数：返回负的平均目标函数值（用于最小化）
    """
    # 确保tau在合理范围内
    tau_trajectory = np.clip(tau_trajectory, 0.7, 1.5)

    objectives = []
    for _ in range(n_simulations):
        results = simulate_with_continuous_tau(
            schedule=schedule,
            tau_trajectory=tau_trajectory,
            initial_state=initial_state.copy(),
            fixed_u=fixed_u,
            fixed_m=fixed_m,
            tbd_params=tbd_params,
            capacity=capacity,
            base_price=base_price,
            base_demand=base_demand,
            epsilon_0=epsilon_0,
        )
        objectives.append(results['summary']['total_objective'])

    # 返回负值（因为scipy.optimize.minimize是最小化）
    return -np.mean(objectives)


def optimize_continuous_pricing(
    schedule: pd.DataFrame,
    initial_state: Dict[str, float],
    fixed_u: float,
    fixed_m: float,
    tbd_params: TBDIntegrationParameters,
    capacity: float = 12000,
    base_price: float = 50,
    base_demand: float = 5500,
    epsilon_0: float = 2.5,
    method: str = 'differential_evolution',
) -> Dict:
    """
    优化连续的票价轨迹
    """
    n_periods = len(schedule['period'].unique())

    print(f"\n{'='*80}")
    print(f"优化连续票价轨迹（{n_periods}期）")
    print(f"{'='*80}")
    print(f"方法: {method}")
    print(f"搜索空间: tau ∈ [0.7, 1.5]^{n_periods}")

    if method == 'differential_evolution':
        # 使用差分进化算法（全局优化）
        bounds = [(0.7, 1.5)] * n_periods

        result = differential_evolution(
            objective_function,
            bounds=bounds,
            args=(schedule, initial_state, fixed_u, fixed_m, tbd_params,
                  capacity, base_price, base_demand, epsilon_0, 3),
            maxiter=50,
            popsize=10,
            seed=42,
            disp=True,
        )

        optimal_tau = result.x
        optimal_objective = -result.fun

    else:
        # 使用局部优化（从tau=1.0开始）
        initial_tau = np.ones(n_periods)
        bounds = [(0.7, 1.5)] * n_periods

        result = minimize(
            objective_function,
            initial_tau,
            args=(schedule, initial_state, fixed_u, fixed_m, tbd_params,
                  capacity, base_price, base_demand, epsilon_0, 3),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'disp': True},
        )

        optimal_tau = result.x
        optimal_objective = -result.fun

    print(f"\n优化完成:")
    print(f"  最优tau轨迹: {optimal_tau}")
    print(f"  最优目标函数: ${optimal_objective/1e6:.2f}M")

    # 运行最优策略的详细模拟
    optimal_results = simulate_with_continuous_tau(
        schedule=schedule,
        tau_trajectory=optimal_tau,
        initial_state=initial_state.copy(),
        fixed_u=fixed_u,
        fixed_m=fixed_m,
        tbd_params=tbd_params,
        capacity=capacity,
        base_price=base_price,
        base_demand=base_demand,
        epsilon_0=epsilon_0,
    )

    return {
        'optimal_tau': optimal_tau,
        'optimal_objective': optimal_objective,
        'results': optimal_results,
        'optimization_result': result,
    }


def main():
    """
    运行连续优化实验
    """
    print("\n" + "="*80)
    print("TASK 4: 连续票价优化实验")
    print("="*80)
    print("\n目标: 找到最优的连续tau轨迹（非预设离散值）")
    print("="*80)

    np.random.seed(42)

    # 创建赛程
    real_teams = {
        'New York Liberty': 1640, 'Connecticut Sun': 1620, 'Minnesota Lynx': 1610,
        'Indiana Fever': 1540, 'Phoenix Mercury': 1580, 'Seattle Storm': 1600,
        'Los Angeles Sparks': 1570, 'Chicago Sky': 1560, 'Washington Mystics': 1520,
        'Atlanta Dream': 1510, 'Dallas Wings': 1530,
    }

    schedule = []
    for game_num in range(20):
        period = game_num // 4
        opponent = list(real_teams.keys())[game_num % len(real_teams)]
        opponent_elo = real_teams[opponent] + np.random.normal(0, 15)

        schedule.append({
            'game_id': game_num,
            'period': period,
            'is_home': True,
            'opponent_name': opponent,
            'opponent_elo': opponent_elo,
        })

    schedule = pd.DataFrame(schedule)

    # 初始状态
    initial_state = {
        'S_0': 1650,
        'B_0': 1.7,
        'Star_0': 65,
        'Cash_0': 10000000,
        'league_pop': 1.3,
    }

    fixed_u = 800000
    fixed_m = 500000
    tbd_params = TBDIntegrationParameters()

    # 参数
    capacity = 12000
    base_price = 50
    calibrated_base_demand = 5500
    epsilon_0 = 2.5

    # 运行优化
    print("\n开始优化...")
    optimization_result = optimize_continuous_pricing(
        schedule=schedule,
        initial_state=initial_state,
        fixed_u=fixed_u,
        fixed_m=fixed_m,
        tbd_params=tbd_params,
        capacity=capacity,
        base_price=base_price,
        base_demand=calibrated_base_demand,
        epsilon_0=epsilon_0,
        method='differential_evolution',
    )

    optimal_tau = optimization_result['optimal_tau']

    # 重新评估：使用相同的随机种子进行公平对比
    print(f"\n{'='*80}")
    print("对比分析：连续优化 vs 基准策略")
    print(f"{'='*80}")
    print("\n使用固定随机种子重新评估所有策略...")

    # 基准策略
    np.random.seed(42)
    baseline_tau = np.ones(5)
    baseline_results = simulate_with_continuous_tau(
        schedule=schedule,
        tau_trajectory=baseline_tau,
        initial_state=initial_state.copy(),
        fixed_u=fixed_u,
        fixed_m=fixed_m,
        tbd_params=tbd_params,
        capacity=capacity,
        base_price=base_price,
        base_demand=calibrated_base_demand,
        epsilon_0=epsilon_0,
    )

    # 优化策略（重置种子）
    np.random.seed(42)
    optimal_results = simulate_with_continuous_tau(
        schedule=schedule,
        tau_trajectory=optimal_tau,
        initial_state=initial_state.copy(),
        fixed_u=fixed_u,
        fixed_m=fixed_m,
        tbd_params=tbd_params,
        capacity=capacity,
        base_price=base_price,
        base_demand=calibrated_base_demand,
        epsilon_0=epsilon_0,
    )

    print(f"\n基准策略 (tau=1.0):")
    print(f"  tau轨迹: {baseline_tau}")
    print(f"  平均需求: {np.mean(baseline_results['demands']):,.0f}")
    print(f"  平均上座率: {np.mean(baseline_results['attendance_rates']):.1%}")
    print(f"  总收入: ${sum(baseline_results['revenues'])/1e6:.2f}M")
    print(f"  总利润: ${sum(baseline_results['profits'])/1e6:.2f}M")
    print(f"  总胜场: {baseline_results['summary']['cumulative_wins']}/20")
    print(f"  目标函数: ${baseline_results['summary']['total_objective']/1e6:.2f}M")

    print(f"\n连续优化策略:")
    print(f"  tau轨迹: {optimal_tau}")
    print(f"  平均需求: {np.mean(optimal_results['demands']):,.0f}")
    print(f"  平均上座率: {np.mean(optimal_results['attendance_rates']):.1%}")
    print(f"  总收入: ${sum(optimal_results['revenues'])/1e6:.2f}M")
    print(f"  总利润: ${sum(optimal_results['profits'])/1e6:.2f}M")
    print(f"  总胜场: {optimal_results['summary']['cumulative_wins']}/20")
    print(f"  目标函数: ${optimal_results['summary']['total_objective']/1e6:.2f}M")

    print(f"\n改进:")
    print(f"  ΔJ: ${(optimal_results['summary']['total_objective'] - baseline_results['summary']['total_objective'])/1e6:.2f}M")
    print(f"  Δ利润: ${(sum(optimal_results['profits']) - sum(baseline_results['profits']))/1e6:.2f}M")
    print(f"  Δ胜场: {optimal_results['summary']['cumulative_wins'] - baseline_results['summary']['cumulative_wins']}")

    # 可视化
    plot_continuous_optimization_results(optimal_tau, optimal_results, baseline_tau, baseline_results)

    # 保存结果
    results_df = pd.DataFrame({
        'period': range(5),
        'optimal_tau': optimal_tau,
        'baseline_tau': baseline_tau,
        'optimal_demand': optimal_results['demands'],
        'baseline_demand': baseline_results['demands'],
        'optimal_attendance': optimal_results['attendance_rates'],
        'baseline_attendance': baseline_results['attendance_rates'],
        'optimal_revenue': optimal_results['revenues'],
        'baseline_revenue': baseline_results['revenues'],
    })

    results_df.to_csv('./task4_results/continuous_optimization_results.csv', index=False)
    print(f"\n保存: ./task4_results/continuous_optimization_results.csv")

    print(f"\n{'='*80}")
    print("连续优化实验完成")
    print(f"{'='*80}")

    return optimization_result


def plot_continuous_optimization_results(optimal_tau, optimal_results, baseline_tau, baseline_results):
    """
    可视化连续优化结果
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    periods = range(5)

    # 图1: tau轨迹对比
    ax = axes[0, 0]
    ax.plot(periods, optimal_tau, 'o-', linewidth=3, markersize=12,
            color='#2E86AB', label='Continuous Optimal', alpha=0.8)
    ax.plot(periods, baseline_tau, 's--', linewidth=2, markersize=10,
            color='#95B8D1', label='Baseline (tau=1.0)', alpha=0.6)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Period', fontsize=11, fontweight='bold')
    ax.set_ylabel('Price Multiplier τ', fontsize=11, fontweight='bold')
    ax.set_title('Pricing Trajectory Comparison', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 图2: 需求对比
    ax = axes[0, 1]
    ax.plot(periods, optimal_results['demands'], 'o-', linewidth=3, markersize=12,
            color='#F18F01', label='Continuous Optimal', alpha=0.8)
    ax.plot(periods, baseline_results['demands'], 's--', linewidth=2, markersize=10,
            color='#FDB462', label='Baseline', alpha=0.6)
    ax.axhline(y=12000, color='red', linestyle='--', alpha=0.5, label='Capacity')
    ax.set_xlabel('Period', fontsize=11, fontweight='bold')
    ax.set_ylabel('Avg Demand (persons/game)', fontsize=11, fontweight='bold')
    ax.set_title('Demand Trajectory Comparison', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 图3: 上座率对比
    ax = axes[0, 2]
    optimal_att = [r*100 for r in optimal_results['attendance_rates']]
    baseline_att = [r*100 for r in baseline_results['attendance_rates']]
    ax.plot(periods, optimal_att, 'o-', linewidth=3, markersize=12,
            color='#06A77D', label='Continuous Optimal', alpha=0.8)
    ax.plot(periods, baseline_att, 's--', linewidth=2, markersize=10,
            color='#8DD3C7', label='Baseline', alpha=0.6)
    ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Full Capacity')
    ax.set_xlabel('Period', fontsize=11, fontweight='bold')
    ax.set_ylabel('Attendance Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title('Attendance Rate Comparison', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 图4: 收入对比
    ax = axes[1, 0]
    optimal_rev = [r/1e6 for r in optimal_results['revenues']]
    baseline_rev = [r/1e6 for r in baseline_results['revenues']]
    ax.plot(periods, optimal_rev, 'o-', linewidth=3, markersize=12,
            color='#A23B72', label='Continuous Optimal', alpha=0.8)
    ax.plot(periods, baseline_rev, 's--', linewidth=2, markersize=10,
            color='#D4A5C7', label='Baseline', alpha=0.6)
    ax.set_xlabel('Period', fontsize=11, fontweight='bold')
    ax.set_ylabel('Revenue ($M)', fontsize=11, fontweight='bold')
    ax.set_title('Revenue Trajectory Comparison', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 图5: 利润对比
    ax = axes[1, 1]
    optimal_prof = [p/1e6 for p in optimal_results['profits']]
    baseline_prof = [p/1e6 for p in baseline_results['profits']]
    ax.plot(periods, optimal_prof, 'o-', linewidth=3, markersize=12,
            color='#E63946', label='Continuous Optimal', alpha=0.8)
    ax.plot(periods, baseline_prof, 's--', linewidth=2, markersize=10,
            color='#F4A5AE', label='Baseline', alpha=0.6)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Period', fontsize=11, fontweight='bold')
    ax.set_ylabel('Profit ($M)', fontsize=11, fontweight='bold')
    ax.set_title('Profit Trajectory Comparison', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 图6: 累计指标对比
    ax = axes[1, 2]
    metrics = ['Objective\nJ', 'Revenue', 'Profit', 'Wins']
    optimal_vals = [
        optimal_results['summary']['total_objective']/1e6,
        sum(optimal_results['revenues'])/1e6,
        sum(optimal_results['profits'])/1e6,
        optimal_results['summary']['cumulative_wins']/2,  # 缩放到相同量级
    ]
    baseline_vals = [
        baseline_results['summary']['total_objective']/1e6,
        sum(baseline_results['revenues'])/1e6,
        sum(baseline_results['profits'])/1e6,
        baseline_results['summary']['cumulative_wins']/2,
    ]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, optimal_vals, width, label='Continuous Optimal',
                   color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, baseline_vals, width, label='Baseline',
                   color='#95B8D1', alpha=0.6, edgecolor='black', linewidth=1.5)

    ax.set_ylabel('Value ($M or Wins/2)', fontsize=11, fontweight='bold')
    ax.set_title('Cumulative Metrics Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    output_path = '../../pic/task4/continuous_optimization_analysis.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"保存: {output_path}")
    plt.close()


if __name__ == "__main__":
    import os
    os.makedirs('./task4_results', exist_ok=True)
    main()
