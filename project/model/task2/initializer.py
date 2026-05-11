"""
赛季状态初始化模块

功能:
- 从选定阵容生成赛季初始状态
- 输出到Task1总模型的接口
"""

import pandas as pd
import numpy as np

# Support two execution modes:
# 1) `python project/model/mpc_simulation.py` (model dir on sys.path; imports are top-level)
# 2) `python project/model/task2_main.py` (repo root on sys.path; namespace package `project.model.*`)
try:  # mode 2
    from ..core.brands import FRANCHISE_ALIASES, TEAM_NAME_TO_CODE  # type: ignore
    from ..core.data_loader import DataLoader  # type: ignore
except Exception:  # mode 1
    from core.brands import FRANCHISE_ALIASES, TEAM_NAME_TO_CODE  # type: ignore
    from core.data_loader import DataLoader  # type: ignore


def _normalize_team_code(team_code: object) -> str:
    s = str(team_code).strip()
    if not s:
        return "LVA"

    if s in TEAM_NAME_TO_CODE:
        s = TEAM_NAME_TO_CODE[s]

    s = s.upper()
    return FRANCHISE_ALIASES.get(s, s)


def _load_brand_b0(team_code: str) -> float:
    """Load TOPSIS-derived B0 (mean ~1) from processed brand_b0.csv."""
    loader = DataLoader()
    try:
        df = loader.load_brand_b0("brand_b0.csv")
    except Exception:
        return 1.0

    if df.empty or "B0" not in df.columns:
        return 1.0

    s = pd.to_numeric(df.set_index("team")["B0"], errors="coerce")
    v = float(s.get(team_code, np.nan))
    if np.isfinite(v):
        return v
    return float(s.mean()) if s.notna().any() else 1.0

def initialize_season_state(
    selected_roster: pd.DataFrame,
    salary_cap: float = 1500000,
    transfer_budget: float = 500000,
    team_code: str = "LVA",
) -> dict:
    """
    从选定阵容生成赛季初始状态

    输出到Task1总模型的接口:
    - S_0: Elo初值/阵容强度 [0-100]
    - Star_0: 明星球员数量/质量
    - Cash_0: 赛季开始的现金变化(转会费/签约金等)，负值代表支出
    - B_0: 品牌初值

    Args:
        selected_roster: 选定的球员阵容
        salary_cap: 工资帽上限
        transfer_budget: 转会预算(用于统计剩余额度)
        team_code: 球队代码/全名(用于读取TOPSIS品牌初值B0)

    Returns:
        Dict with season initial state
    """

    # S_0: 阵容强度 (基于court_impact加权平均)
    # 标准化到0-100，其中50是联盟平均水平
    avg_court_impact = selected_roster['court_impact'].mean()
    S_0 = 50 + (avg_court_impact - 10) * 3  # 假设court_impact均值~10
    S_0 = np.clip(S_0, 0, 100)

    # Star_0: 明星球员数量 (PCV > 1.5的球员)
    high_pcv_players = selected_roster[selected_roster['pcv'] > 1.5]
    Star_0 = len(high_pcv_players)

    # 或者用明星质量加权
    star_quality = selected_roster['pcv'].clip(1.0, 3.0).sum()

    team_code = _normalize_team_code(team_code)

    # 成本/预算统计
    total_salary = selected_roster['salary'].sum()
    total_transfer_fee = selected_roster['transfer_fee'].sum()

    # 工资帽剩余(不是现金，只用于约束检查/解释)
    cap_space_0 = salary_cap - total_salary

    # 转会预算剩余(不是现金，只用于约束检查/解释)
    transfer_budget_remaining_0 = transfer_budget - total_transfer_fee

    # Cash_0: 赛季开始的现金变化(这里只把“转会费”作为一次性现金流出代理)
    Cash_0 = -total_transfer_fee

    # B_0: 品牌初值 (来自brands.py的“熵权法+TOPSIS”，均值约为1)
    B_0 = _load_brand_b0(team_code)

    # 阵容自身的“商业影响力”代理(不等同于B_0，更多影响Star_t/赞助)
    B_roster_sum_0 = selected_roster['brand_impact'].sum()
    B_roster_mean_0 = selected_roster['brand_impact'].mean()

    # 额外统计
    avg_age = selected_roster['age'].mean()
    avg_injury_risk = selected_roster['injury_risk'].mean()
    total_mv = selected_roster['MV_i'].sum()

    state = {
        # Task1接口变量
        'team': team_code,
        'S_0': float(S_0),
        'Star_0': int(Star_0),
        'Cash_0': float(Cash_0),
        'B_0': float(B_0),

        # 额外统计信息
        'roster_size': len(selected_roster),
        'avg_age': float(avg_age),
        'avg_injury_risk': float(avg_injury_risk),
        'total_mv': float(total_mv),
        'total_salary': float(total_salary),
        'total_transfer_fee': float(total_transfer_fee),
        'star_quality': float(star_quality),
        'cap_space_0': float(cap_space_0),
        'transfer_budget_remaining_0': float(transfer_budget_remaining_0),
        'B_roster_sum_0': float(B_roster_sum_0),
        'B_roster_mean_0': float(B_roster_mean_0),

        # 位置分布
        'position_distribution': {
            'G': len(selected_roster[selected_roster['assigned_position'] == 'G']),
            'F': len(selected_roster[selected_roster['assigned_position'] == 'F']),
            'C': len(selected_roster[selected_roster['assigned_position'] == 'C'])
        },

        # 球员类型分布
        'player_type_distribution': {
            'rookie': len(selected_roster[selected_roster['player_type'] == 'rookie']),
            'free_agent': len(selected_roster[selected_roster['player_type'] == 'free_agent']),
            'veteran': len(selected_roster[selected_roster['player_type'] == 'veteran'])
        },

        # 阵容详情
        'roster': selected_roster
    }

    print(f"\n  赛季初始状态:")
    print(f"    - Team: {team_code}")
    print(f"    - S_0 (阵容强度): {S_0:.2f}")
    print(f"    - Star_0 (明星数量): {Star_0}")
    print(f"    - Cash_0 (转会现金流, 负为支出): ${Cash_0:,.0f}")
    print(f"    - CapSpace_0 (工资帽剩余): ${cap_space_0:,.0f}")
    print(f"    - TransferFee_0 (转会费): ${total_transfer_fee:,.0f}")
    print(f"    - TransferBudgetRemaining_0: ${transfer_budget_remaining_0:,.0f}")
    print(f"    - B_0 (品牌初值, TOPSIS): {B_0:.3f}")
    print(f"    - 平均年龄: {avg_age:.1f}")
    print(f"    - 平均伤病风险: {avg_injury_risk:.3f}")

    return state


def export_season_state(state: dict, output_path: str) -> None:
    """
    导出赛季状态到JSON文件

    Args:
        state: Season state dict
        output_path: Output JSON file path
    """
    import json
    from pathlib import Path

    # 移除DataFrame (不能序列化)
    state_export = {k: v for k, v in state.items() if k != 'roster'}

    # 添加阵容摘要
    roster = state['roster']
    state_export['roster_summary'] = {
        'player_names': roster['player_name'].tolist(),
        'positions': roster['assigned_position'].tolist(),
        'salaries': roster['salary'].tolist(),
        'pcv_values': roster['pcv'].tolist()
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(state_export, f, indent=2, ensure_ascii=False)

    print(f"\n  [OK] 赛季状态已保存: {output_path}")


def compare_rosters(
    roster1: pd.DataFrame,
    roster2: pd.DataFrame,
    name1: str = "Roster 1",
    name2: str = "Roster 2"
) -> pd.DataFrame:
    """
    比较两个阵容

    Args:
        roster1, roster2: Selected roster DataFrames
        name1, name2: Roster names

    Returns:
        Comparison DataFrame
    """
    state1 = initialize_season_state(roster1)
    state2 = initialize_season_state(roster2)

    comparison = pd.DataFrame({
        'Metric': [
            'S_0 (阵容强度)',
            'Star_0 (明星数量)',
            'Cash_0 (转会现金流)',
            'CapSpace_0 (工资帽剩余)',
            'B_0 (品牌初值, TOPSIS)',
            '总MV',
            '总薪资',
            '总转会费',
            '平均年龄',
            '平均伤病风险'
        ],
        name1: [
            f"{state1['S_0']:.2f}",
            f"{state1['Star_0']}",
            f"${state1['Cash_0']:,.0f}",
            f"${state1.get('cap_space_0', 0.0):,.0f}",
            f"{state1['B_0']:.2f}",
            f"${state1['total_mv']:,.0f}",
            f"${state1['total_salary']:,.0f}",
            f"${state1['total_transfer_fee']:,.0f}",
            f"{state1['avg_age']:.1f}",
            f"{state1['avg_injury_risk']:.3f}"
        ],
        name2: [
            f"{state2['S_0']:.2f}",
            f"{state2['Star_0']}",
            f"${state2['Cash_0']:,.0f}",
            f"${state2.get('cap_space_0', 0.0):,.0f}",
            f"{state2['B_0']:.2f}",
            f"${state2['total_mv']:,.0f}",
            f"${state2['total_salary']:,.0f}",
            f"${state2['total_transfer_fee']:,.0f}",
            f"{state2['avg_age']:.1f}",
            f"{state2['avg_injury_risk']:.3f}"
        ]
    })

    return comparison


def validate_season_state(state: dict) -> dict:
    """
    验证赛季状态的合理性

    Returns:
        Dict with validation results
    """
    validation = {
        'valid': True,
        'warnings': []
    }

    # 检查S_0范围
    if not (0 <= state['S_0'] <= 100):
        validation['warnings'].append(
            f"S_0超出范围: {state['S_0']:.2f} (应在0-100)"
        )

    # 检查工资帽是否超支
    if float(state.get('cap_space_0', 0.0)) < 0:
        validation['valid'] = False
        validation['warnings'].append(
            f"工资帽超支: cap_space_0=${state.get('cap_space_0', 0.0):,.0f}"
        )

    # 检查转会预算是否超支
    if float(state.get('transfer_budget_remaining_0', 0.0)) < 0:
        validation['valid'] = False
        validation['warnings'].append(
            f"转会预算超支: remaining=${state.get('transfer_budget_remaining_0', 0.0):,.0f}"
        )

    # 检查明星数量
    if state['Star_0'] == 0:
        validation['warnings'].append(
            "警告: 没有明星球员 (PCV>1.5)"
        )

    # 检查平均年龄
    if state['avg_age'] > 32:
        validation['warnings'].append(
            f"警告: 平均年龄过高 ({state['avg_age']:.1f}岁)"
        )

    # 检查伤病风险
    if state['avg_injury_risk'] > 0.3:
        validation['warnings'].append(
            f"警告: 平均伤病风险过高 ({state['avg_injury_risk']:.3f})"
        )

    return validation
