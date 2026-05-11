"""
Task2 端到端Pipeline

功能:
- 整合所有模块
- 提供完整的球员获取策略优化流程
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import json

from .attributes import load_player_base_data, add_player_attributes, load_task2_candidates_from_task1_inputs
from .forecast import forecast_performance, check_growth_constraints
from .marginal_value import compute_marginal_value
from .optimizer import optimize_roster, export_roster_summary
from .initializer import initialize_season_state, export_season_state
from .task1_adapter import export_task1_initial_state


def run_task2_pipeline(
    horizon: int = 3,
    salary_cap: float = 1500000,
    roster_size: int = 12,
    roster_min: int = 11,
    position_quotas: dict = None,
    transfer_budget: float = 500000,
    team_code: str = "LVA",
    mv_weights: dict = None,
    g_min: float = 0.2,
    min_rookies: int = 0,
    max_veterans: int = None,
    output_dir: str = None,
    team_filter: str = None,  # 新增: 球队过滤
    *,
    mode: str = "paper",
    offseason_year: int = 2025,
    lambda_R: float = 120000,
    penalty_pos: float | None = None,
    roster_target: int | None = None,
    position_targets: dict | None = None,
) -> dict:
    """
    Task2完整流程: 球员获取策略优化

    流程:
    1. 加载player.py输出的基础数据
    2. 扩展球员属性 (降维到4类)
    3. 预测未来表现 (增长/下降)
    4. 计算边际价值 MV
    5. MILP优化选择球员
    6. 生成赛季初始状态

    Args:
        horizon: 预测年数 (default: 3)
        salary_cap: 工资帽上限 (default: $1.5M)
        roster_size: 阵容人数 (WNBA规定: 11-12人, default: 11)
        position_quotas: 位置配额 (default: {'G':4, 'F':4, 'C':3} for 11人)
        transfer_budget: 转会预算 (default: $500K)
        team_code: 球队代码/全名(用于读取TOPSIS品牌初值B0; default: LVA)
        mv_weights: MV权重字典 (default: None, 使用优化后的默认值)
        g_min: 新秀最小增长率 (default: 2%, 已优化)
        min_rookies: 最少新秀数量 (default: 0)
        max_veterans: 最多老将数量 (default: None)
        output_dir: 输出目录 (default: project/data/processed/task2)
        team_filter: 球队名称过滤 (default: None, 例如: "Phoenix Mercury")

    Returns:
        {
            'selected_roster': DataFrame,
            'season_state': dict,
            'all_candidates': DataFrame,
            'optimization_metrics': dict
        }
    """

    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[3] / "project" / "data" / "processed" / "task2"
    else:
        output_dir = Path(output_dir)

    roster_min = int(roster_min)
    if roster_min < 0:
        raise ValueError("roster_min must be >= 0")
    if roster_min > int(roster_size):
        raise ValueError(f"roster_min ({roster_min}) must be <= roster_size ({int(roster_size)})")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Task2: 球员获取策略优化")
    print("=" * 80)

    paper_cfg = None
    position_targets_used = None
    penalty_pos_used = penalty_pos
    roster_target_used = roster_target

    if mode == "paper":
        print("\n[1/3] 加载论文一致的Task1预处理工件...")
        df, paper_cfg = load_task2_candidates_from_task1_inputs(offseason_year=offseason_year)

        position_targets_used = position_targets or (paper_cfg.get("position_targets") if paper_cfg else None)
        roster_target_used = roster_target if roster_target is not None else int(paper_cfg.get("roster_target", roster_size))
        if penalty_pos_used is None:
            penalty_pos_used = float(paper_cfg.get("penalty_pos", 1000.0))

        print("\n[2/3] 计算边际价值 MV (paper)...")
        df = compute_marginal_value(df, weights=mv_weights, g_min=g_min, mode="paper")
    else:
        # ========================================================================
        # 步骤1: 加载球员基础数据 (legacy)
        # ========================================================================
        print("\n[1/6] 加载球员基础数据...")
        df = load_player_base_data(team_filter=team_filter)

        # ========================================================================
        # 步骤2: 扩展球员属性 (legacy)
        # ========================================================================
        print("\n[2/6] 计算球员属性 (降维到4类)...")
        df = add_player_attributes(df)

        # ========================================================================
        # 步骤3: 表现预测 (legacy)
        # ========================================================================
        print(f"\n[3/6] 预测未来{horizon}年表现...")
        df = forecast_performance(df, horizon=horizon)
        df = check_growth_constraints(df, g_min=g_min, horizon=horizon)

        # ========================================================================
        # 步骤4: 计算边际价值 (legacy)
        # ========================================================================
        print("\n[4/6] 计算边际价值 MV...")
        df = compute_marginal_value(df, weights=mv_weights, g_min=g_min)

    # 保存所有候选球员数据
    candidates_file = output_dir / "all_candidates.csv"
    df.to_csv(candidates_file, index=False)
    print(f"  [OK] 候选球员数据已保存: {candidates_file}")

    # ========================================================================
    # 优化选人
    # ========================================================================
    if mode == "paper":
        print("\n[3/3] MILP优化球员选择 (paper)...")
    else:
        print("\n[5/6] MILP优化球员选择...")

    selected_roster = optimize_roster(
        df,
        salary_cap=salary_cap,
        roster_size=roster_size,
        position_quotas=position_targets_used or position_quotas,
        transfer_budget=transfer_budget,
        min_rookies=min_rookies,
        max_veterans=max_veterans,
        mode=mode,
        roster_target=roster_target_used,
        roster_min=roster_min,
        position_targets=position_targets_used,
        penalty_pos=float(penalty_pos_used) if penalty_pos_used is not None else 1000.0,
        lambda_R=lambda_R,
    )

    # 导出阵容摘要
    roster_file = output_dir / "selected_roster.csv"
    export_roster_summary(selected_roster, output_path=roster_file)

    # ========================================================================
    # 步骤6: 生成赛季初始状态
    # ========================================================================
    print("\n[6/6] 生成赛季初始状态...")
    season_state = initialize_season_state(
        selected_roster,
        salary_cap=salary_cap,
        transfer_budget=transfer_budget,
        team_code=team_code,
    )

    # 导出赛季状态
    state_file = output_dir / "season_initial_state.json"
    export_season_state(season_state, output_path=state_file)
    task1_state_file = output_dir / "task1_initial_state.json"
    export_task1_initial_state(season_state, output_path=task1_state_file)
    print(f"  [OK] Task1 初值(elo/B/cash)已保存: {task1_state_file}")

    # ========================================================================
    # 汇总优化指标
    # ========================================================================
    optimization_metrics = {
        'total_mv': float(selected_roster['MV_i'].sum()),
        'total_salary': float(selected_roster['salary'].sum()),
        'total_transfer_fee': float(selected_roster['transfer_fee'].sum()),
        'avg_court_impact': float(selected_roster['court_impact'].mean()),
        'avg_brand_impact': float(selected_roster['brand_impact'].mean()),
        'avg_injury_risk': float(selected_roster['injury_risk'].mean()),
        'avg_age': float(selected_roster['age'].mean()),
        'salary_cap_utilization': float(selected_roster['salary'].sum() / salary_cap),
        'transfer_budget_utilization': float(selected_roster['transfer_fee'].sum() / transfer_budget),
        'n_high_mv_players': int((selected_roster['MV_i'] > 0).sum()),
        'n_star_players': int((selected_roster['pcv'] > 1.5).sum())
    }

    if mode == "paper":
        roster_target_eff = int(roster_target_used) if roster_target_used is not None else roster_size
        pos_targets_eff = position_targets_used or {}
        pos_counts = selected_roster["assigned_position"].value_counts().to_dict()
        pos_dev = {p: abs(int(pos_counts.get(p, 0)) - int(pos_targets_eff.get(p, 0))) for p in pos_targets_eff.keys()}

        optimization_metrics.update({
            "mode": "paper",
            "offseason_year": int(offseason_year),
            "g_min": float(g_min),
            "roster_min": int(roster_min),
            "roster_max": int(roster_size),
            "roster_target": int(roster_target_eff),
            "roster_shortage_u": int(max(0, roster_target_eff - len(selected_roster))),
            "lambda_R": float(lambda_R),
            "penalty_pos": float(penalty_pos_used) if penalty_pos_used is not None else None,
            "position_targets": pos_targets_eff,
            "position_deviation": int(sum(pos_dev.values())),
        })

    # 保存优化指标
    metrics_file = output_dir / "optimization_metrics.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(optimization_metrics, f, indent=2, ensure_ascii=False)
    print(f"  [OK] 优化指标已保存: {metrics_file}")

    # ========================================================================
    # 生成报告
    # ========================================================================
    report_file = output_dir / "task2_report.txt"
    generate_report(
        selected_roster=selected_roster,
        season_state=season_state,
        optimization_metrics=optimization_metrics,
        output_path=report_file
    )

    print("\n" + "=" * 80)
    print("Task2 完成!")
    print("=" * 80)
    print(f"\n输出文件:")
    print(f"  - 候选球员: {candidates_file}")
    print(f"  - 选定阵容: {roster_file}")
    print(f"  - 赛季状态: {state_file}")
    print(f"  - Task1 初值: {task1_state_file}")
    print(f"  - 优化指标: {metrics_file}")
    print(f"  - 分析报告: {report_file}")

    return {
        'selected_roster': selected_roster,
        'season_state': season_state,
        'all_candidates': df,
        'optimization_metrics': optimization_metrics,
        'output_dir': output_dir,
        'n_selected': len(selected_roster),
        'total_mv': optimization_metrics['total_mv'],
        'salary_cap_utilization': optimization_metrics['salary_cap_utilization']
    }


def generate_report(
    selected_roster: pd.DataFrame,
    season_state: dict,
    optimization_metrics: dict,
    output_path: Path,
) -> None:
    is_paper = optimization_metrics.get("mode") == "paper" or {
        "mu0_perf",
        "mu1_perf",
        "mu1_pop",
        "inj_score",
        "sigma_perf2",
    }.issubset(selected_roster.columns)

    total_salary = float(optimization_metrics.get("total_salary", 0.0))
    util = float(optimization_metrics.get("salary_cap_utilization", 0.0))
    salary_cap = total_salary / util if util > 0 else None

    roster_min = int(optimization_metrics.get("roster_min", 0))
    roster_max = int(optimization_metrics.get("roster_max", optimization_metrics.get("roster_target", len(selected_roster))))
    roster_target = int(optimization_metrics.get("roster_target", len(selected_roster)))
    shortage_u = int(optimization_metrics.get("roster_shortage_u", max(0, roster_target - len(selected_roster))))
    position_targets = optimization_metrics.get("position_targets") or {}

    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("Task 2: Operations in the Transfer Market")
    lines.append("=" * 80)
    lines.append("")

    lines.append("1. Model")
    if is_paper:
        penalty_pos = optimization_metrics.get("penalty_pos")
        lambda_R = optimization_metrics.get("lambda_R")
        lines.append("Objective: max sum(MV_i*x_i) - lambda_R*u - penalty_pos*sum_p abs(N_p - target_p)")
        lines.append(f"Salary cap: ${salary_cap:,.0f}" if salary_cap else "Salary cap: N/A")
        lines.append(f"Roster: {roster_min} <= sum(x_i) <= {roster_max} (target={roster_target}, u=max(0,target-sum(x_i)))")
        if position_targets:
            lines.append(f"Position targets: {position_targets} (penalty_pos={penalty_pos})")
        lines.append(f"Parameters: lambda_R={lambda_R}, g_min={optimization_metrics.get('g_min', 'N/A')}")
    else:
        lines.append("Objective: max sum(MV_i*x_i) (legacy)")
    lines.append("")

    lines.append("2. Results")
    lines.append(f"Total MV: ${float(optimization_metrics.get('total_mv', 0.0)):,.0f}")
    lines.append(f"Total salary: ${total_salary:,.0f}" + (f" / ${salary_cap:,.0f}" if salary_cap else ""))
    if is_paper:
        lines.append(f"Final roster size: {len(selected_roster)} (u={shortage_u})")
        if position_targets:
            lines.append(f"Position deviation: {optimization_metrics.get('position_deviation', 'N/A')}")
    else:
        lines.append(f"Roster size: {len(selected_roster)}")
    lines.append("")

    lines.append("3. Selected Players")
    lines.append("")

    if is_paper:
        header = f"{'Player':<20} {'Pos':<3} {'Type':<10} {'Salary':>10} {'MV':>10} {'mu1_perf':>9} {'mu1_pop':>8}"
        lines.append(header)
        lines.append("-" * len(header))
    else:
        header = f"{'Player':<20} {'Pos':<3} {'Type':<10} {'Salary':>10} {'MV':>10} {'PCV':>6}"
        lines.append(header)
        lines.append("-" * len(header))

    display = selected_roster.copy()
    if 'assigned_position' in display.columns and 'MV_i' in display.columns:
        display = display.sort_values(['assigned_position', 'MV_i'], ascending=[True, False])
    elif 'MV_i' in display.columns:
        display = display.sort_values('MV_i', ascending=False)

    for _, row in display.iterrows():
        name = str(row.get('player_name', ''))[:20]
        pos = str(row.get('assigned_position', ''))[:3]
        ptype = str(row.get('player_type', ''))[:10]
        salary = float(row.get('salary', 0.0))
        mv = float(row.get('MV_i', 0.0))

        if is_paper:
            mu1_perf = float(row.get('mu1_perf', 0.0))
            mu1_pop = float(row.get('mu1_pop', 0.0))
            lines.append(f"{name:<20} {pos:<3} {ptype:<10} {salary:>10,.0f} {mv:>10,.0f} {mu1_perf:>9.2f} {mu1_pop:>8.2f}")
        else:
            pcv = float(row.get('pcv', 0.0))
            lines.append(f"{name:<20} {pos:<3} {ptype:<10} {salary:>10,.0f} {mv:>10,.0f} {pcv:>6.2f}")

    lines.append("")

    lines.append("4. Season Initial State (Task1 Interface)")
    lines.append("")
    if season_state.get('team') is not None:
        lines.append(f"Team: {season_state.get('team')}")
    lines.append(f"S_0: {float(season_state.get('S_0', 0.0)):.2f}")
    lines.append(f"Star_0: {int(season_state.get('Star_0', 0))}")
    lines.append(f"Cash_0: ${float(season_state.get('Cash_0', 0.0)):,.0f}")
    if season_state.get('B_0') is not None:
        lines.append(f"B_0: {float(season_state.get('B_0', 0.0)):.3f}")
    lines.append("")

    lines.append("=" * 80)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [OK] Report written: {output_path}")


if __name__ == '__main__':
    """
    命令行入口: python -m project.model.task2.pipeline
    """
    print("=" * 80)
    print("Task2: 球员获取策略优化 Pipeline")
    print("=" * 80)
    print()

    # 默认参数运行
    result = run_task2_pipeline(
        horizon=3,
        salary_cap=1500000,
        roster_size=12,
        team_code="LVA",
        mode="paper",
        offseason_year=2025,
        output_dir=None  # 自动生成时间戳目录
    )

    print()
    print("=" * 80)
    print("Pipeline 执行完成")
    print("=" * 80)
    print()
    print(f"输出目录: {result['output_dir']}")
    print(f"选定球员数: {result['n_selected']}")
    print(f"总边际价值: ${result['total_mv']:,.0f}")
    print(f"工资帽利用率: {result['salary_cap_utilization']:.1%}")
    print()
    print("生成的文件:")
    print(f"  - 候选球员: all_candidates.csv")
    print(f"  - 选定阵容: selected_roster.csv")
    print(f"  - 赛季状态: season_state.json")
    print(f"  - Task1接口: task1_initial_state.json")
    print(f"  - 分析报告: task2_report.txt")
    print()
