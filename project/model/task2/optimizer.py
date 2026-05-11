"""
MILP优化器模块

功能:
- 基于边际价值MV优化球员选择
- 工资帽约束
- 位置配额约束
- 转会预算约束

注: 如果PuLP未安装，使用贪心算法作为备选
"""

from __future__ import annotations

import pandas as pd
import numpy as np

try:
    from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, LpStatus, PULP_CBC_CMD
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    print("警告: PuLP未安装，将使用贪心算法作为备选方案")
    print("安装PuLP以获得最优解: pip install pulp")


def optimize_roster(
    df: pd.DataFrame,
    salary_cap: float = 1500000,
    roster_size: int = 12,
    position_quotas: dict | None = None,
    transfer_budget: float = 500000,
    min_rookies: int = 0,
    max_veterans: int | None = None,
    *,
    mode: str | None = None,
    roster_target: int | None = None,
    roster_min: int | None = None,
    position_targets: dict | None = None,
    penalty_pos: float = 1000,
    lambda_R: float = 120000,
) -> pd.DataFrame:
    """
    MILP优化球员选择 (如果PuLP可用) 或贪心算法 (备选)

    目标函数:
        max Σ MV_i * x_i

    约束:
        1. 工资帽: Σ salary_i * x_i <= salary_cap
        2. 阵容人数: Σ x_i = roster_size
        3. 位置配额: Σ a_{i,p} = quota_p for p in {G,F,C}
        4. 位置可打性: a_{i,p} <= x_i, a_{i,p}=0 if p not in position_set_i
        5. 转会预算: Σ transfer_fee_i * x_i <= transfer_budget
        6. (可选) 新秀最小数量: Σ x_i (rookies) >= min_rookies
        7. (可选) 老将最大数量: Σ x_i (veterans) <= max_veterans

    Args:
        df: DataFrame with MV_i, salary, position_set, transfer_fee
        salary_cap: 工资帽上限
        roster_size: 阵容人数 (WNBA规定: 11-12人)
        position_quotas: 位置配额 (自动调整以匹配roster_size)
        transfer_budget: 转会预算
        min_rookies: 最少新秀数量
        max_veterans: 最多老将数量

    Returns:
        DataFrame with selected players (selected=1, assigned_position)
    """

    if mode == "paper" or {"mu0_perf", "mu1_perf", "mu1_pop", "inj_score", "sigma_perf2"}.issubset(df.columns):
        roster_min_eff = 11 if roster_min is None else int(roster_min)
        return _optimize_paper(
            df,
            salary_cap=salary_cap,
            roster_max=roster_size,
            roster_target=roster_target if roster_target is not None else roster_size,
            roster_min=roster_min_eff,
            position_targets=position_targets or position_quotas,
            penalty_pos=penalty_pos,
            lambda_R=lambda_R,
        )

    # 自动调整位置配额以匹配阵容人数
    if position_quotas is None:
        if roster_size == 11:
            position_quotas = {'G': 4, 'F': 4, 'C': 3}  # 11人配置
        elif roster_size == 12:
            position_quotas = {'G': 4, 'F': 5, 'C': 3}  # 12人配置
        else:
            # 按比例分配 (G:33%, F:42%, C:25%)
            total = roster_size
            position_quotas = {
                'G': max(3, int(total * 0.33)),
                'F': max(3, int(total * 0.42)),
                'C': max(2, int(total * 0.25))
            }
            # 调整使总和等于roster_size
            diff = roster_size - sum(position_quotas.values())
            if diff != 0:
                position_quotas['F'] += diff

    # 验证位置配额总和
    quota_sum = sum(position_quotas.values())
    if quota_sum != roster_size:
        raise ValueError(
            f"位置配额总和 ({quota_sum}) 必须等于阵容人数 ({roster_size})\n"
            f"当前配额: {position_quotas}"
        )

    print(f"  优化参数:")
    print(f"    - 工资帽: ${salary_cap:,.0f}")
    print(f"    - 阵容人数: {roster_size}")
    print(f"    - 位置配额: {position_quotas} (总和: {quota_sum})")
    print(f"    - 转会预算: ${transfer_budget:,.0f}")

    if PULP_AVAILABLE:
        return _optimize_milp(df, salary_cap, roster_size, position_quotas,
                             transfer_budget, min_rookies, max_veterans)
    else:
        return _optimize_greedy(df, salary_cap, roster_size, position_quotas,
                               transfer_budget, min_rookies, max_veterans)


def _optimize_paper(
    df: pd.DataFrame,
    *,
    salary_cap: float,
    roster_max: int,
    roster_target: int,
    roster_min: int,
    position_targets: dict | None,
    penalty_pos: float,
    lambda_R: float,
) -> pd.DataFrame:
    targets = position_targets or {"G": 5, "F": 4, "C": 3}
    positions = [p for p in ["G", "F", "C"] if p in targets] + [p for p in targets.keys() if p not in {"G", "F", "C"}]

    roster_max = int(roster_max)
    roster_target = int(roster_target)
    roster_min = int(roster_min)

    if roster_min < 0:
        raise ValueError("roster_min must be >= 0")
    if roster_min > roster_max:
        raise ValueError(f"roster_min ({roster_min}) must be <= roster_max ({roster_max})")

    _assert_roster_min_feasible(df, roster_min=roster_min, salary_cap=salary_cap)

    print(f"  [Paper] Optimization params:")
    print(f"    - Salary cap: ${salary_cap:,.0f}")
    print(f"    - Roster max: {roster_max} (target: {roster_target})")
    print(f"    - Roster min: {roster_min}")
    print(f"    - Position targets: {targets} (penalty_pos={penalty_pos})")
    print(f"    - Roster shortage penalty: lambda_R={lambda_R}")

    if PULP_AVAILABLE:
        return _optimize_milp_paper(
            df,
            salary_cap=salary_cap,
            roster_max=roster_max,
            roster_target=roster_target,
            roster_min=roster_min,
            positions=positions,
            position_targets=targets,
            penalty_pos=float(penalty_pos),
            lambda_R=float(lambda_R),
        )

    return _optimize_greedy_paper(
        df,
        salary_cap=salary_cap,
        roster_max=roster_max,
        roster_target=roster_target,
        roster_min=roster_min,
        positions=positions,
        position_targets=targets,
        penalty_pos=float(penalty_pos),
        lambda_R=float(lambda_R),
    )


def _assert_roster_min_feasible(df: pd.DataFrame, *, roster_min: int, salary_cap: float) -> None:
    if roster_min <= 0:
        return
    if roster_min > len(df):
        raise ValueError(f"roster_min ({roster_min}) exceeds candidate pool size ({len(df)})")
    if "salary" not in df.columns:
        raise ValueError("Missing salary column for roster feasibility check")

    salaries = pd.to_numeric(df["salary"], errors="coerce").fillna(0.0).clip(lower=0.0).sort_values()
    min_cost = float(salaries.iloc[:roster_min].sum())
    if min_cost <= float(salary_cap):
        return

    cum = salaries.cumsum()
    max_feasible = int((cum <= float(salary_cap)).sum())
    raise ValueError(
        "Infeasible roster size under salary cap with current candidate pool: "
        f"roster_min={int(roster_min)}, salary_cap=${float(salary_cap):,.0f}. "
        f"Cheapest {int(roster_min)} players cost ${min_cost:,.0f} "
        f"(max feasible roster size is {max_feasible})."
    )


def _optimize_milp_paper(
    df: pd.DataFrame,
    *,
    salary_cap: float,
    roster_max: int,
    roster_target: int,
    roster_min: int,
    positions: list[str],
    position_targets: dict,
    penalty_pos: float,
    lambda_R: float,
) -> pd.DataFrame:
    if "MV_i" not in df.columns:
        raise ValueError("Paper optimizer requires MV_i in dataframe (run compute_marginal_value first).")

    prob = LpProblem("Roster_Optimization_Paper", LpMaximize)

    indices = df.index.tolist()
    x = LpVariable.dicts("x", indices, cat=LpBinary)
    a = LpVariable.dicts("a", [(i, p) for i in indices for p in positions], cat=LpBinary)
    u = LpVariable("u", lowBound=0)
    dev = LpVariable.dicts("dev", positions, lowBound=0)

    prob += (
        lpSum([df.loc[i, "MV_i"] * x[i] for i in indices])
        - float(lambda_R) * u
        - float(penalty_pos) * lpSum([dev[p] for p in positions])
    )

    prob += lpSum([df.loc[i, "salary"] * x[i] for i in indices]) <= float(salary_cap), "Salary_Cap"
    prob += lpSum([x[i] for i in indices]) <= int(roster_max), "Roster_Max"
    prob += lpSum([x[i] for i in indices]) >= int(roster_min), "Roster_Min"
    prob += u >= float(roster_target) - lpSum([x[i] for i in indices]), "Roster_Shortage"

    for i in indices:
        pos_set = df.loc[i, "position_set"]
        prob += lpSum([a[i, p] for p in positions]) == x[i], f"Assign_{i}"
        for p in positions:
            if p not in pos_set:
                prob += a[i, p] == 0, f"Elig_{i}_{p}"

    for p in positions:
        count_p = lpSum([a[i, p] for i in indices])
        target = int(position_targets.get(p, 0))
        prob += dev[p] >= count_p - target, f"DevPos_{p}"
        prob += dev[p] >= target - count_p, f"DevNeg_{p}"

    print("\n  Solving MILP (paper mode, PuLP)...")
    status = prob.solve(PULP_CBC_CMD(msg=0))
    if status != 1:
        raise ValueError(f"Optimization failed: {LpStatus[status]}")

    out = df.copy()
    out["selected"] = [x[i].varValue for i in indices]
    out["assigned_position"] = None

    for i in indices:
        if out.loc[i, "selected"] != 1:
            continue
        for p in positions:
            if a[i, p].varValue == 1:
                out.loc[i, "assigned_position"] = p
                break

    selected = out[out["selected"] == 1].copy()
    _print_paper_stats(
        selected,
        salary_cap=salary_cap,
        roster_target=roster_target,
        position_targets=position_targets,
        penalty_pos=penalty_pos,
        lambda_R=lambda_R,
    )
    return selected


def _optimize_greedy_paper(
    df: pd.DataFrame,
    *,
    salary_cap: float,
    roster_max: int,
    roster_target: int,
    roster_min: int,
    positions: list[str],
    position_targets: dict,
    penalty_pos: float,
    lambda_R: float,
) -> pd.DataFrame:
    if "MV_i" not in df.columns:
        raise ValueError("Paper optimizer requires MV_i in dataframe (run compute_marginal_value first).")

    print("\n  Solving greedy (paper mode, fallback)...")

    df = df.copy()
    df["selected"] = 0
    df["assigned_position"] = None

    selected_indices: list[int] = []
    counts = {p: 0 for p in positions}
    total_salary = 0.0

    def total_dev(counts_dict: dict) -> int:
        return int(sum(abs(int(counts_dict.get(p, 0)) - int(position_targets.get(p, 0))) for p in positions))

    if int(roster_min) > 0:
        baseline = df.sort_values("salary", ascending=True).head(int(roster_min))
        for idx, row in baseline.iterrows():
            pos_set = row.get("position_set", [])
            if not isinstance(pos_set, (list, tuple)) or not pos_set:
                pos_set = ["F"]
            eligible = [p for p in pos_set if p in positions] or [positions[0]]

            best_pos = eligible[0]
            best_dev = None
            for p in eligible:
                new_counts = dict(counts)
                new_counts[p] = int(new_counts.get(p, 0)) + 1
                d = total_dev(new_counts)
                if best_dev is None or d < best_dev:
                    best_dev = d
                    best_pos = p

            selected_indices.append(int(idx))
            counts[best_pos] = int(counts.get(best_pos, 0)) + 1
            salary_val = pd.to_numeric(row.get("salary", 0.0), errors="coerce")
            if pd.isna(salary_val):
                salary_val = 0.0
            total_salary += float(salary_val)
            df.loc[idx, "selected"] = 1
            df.loc[idx, "assigned_position"] = best_pos

    while True:
        if len(selected_indices) >= int(roster_max):
            break

        base_dev = total_dev(counts)
        best = None  # (gain, idx, pos)

        for idx, row in df.iterrows():
            if idx in selected_indices:
                continue
            salary = float(row.get("salary", 0.0))
            if total_salary + salary > float(salary_cap):
                continue

            pos_set = row.get("position_set", [])
            if not isinstance(pos_set, (list, tuple)) or not pos_set:
                pos_set = ["F"]

            best_pos = None
            best_gain = None

            for p in pos_set:
                if p not in positions:
                    continue
                new_counts = dict(counts)
                new_counts[p] = int(new_counts.get(p, 0)) + 1
                delta_dev = total_dev(new_counts) - base_dev

                roster_bonus = float(lambda_R) if len(selected_indices) < int(roster_target) else 0.0
                gain = float(row["MV_i"]) + roster_bonus - float(penalty_pos) * float(delta_dev)

                if best_gain is None or gain > best_gain:
                    best_gain = gain
                    best_pos = p

            if best_gain is None or best_pos is None:
                continue

            if best is None or best_gain > best[0]:
                best = (best_gain, idx, best_pos)

        if best is None or best[0] <= 0:
            break

        _, idx, p = best
        selected_indices.append(int(idx))
        counts[p] = int(counts.get(p, 0)) + 1
        total_salary += float(df.loc[idx, "salary"])
        df.loc[idx, "selected"] = 1
        df.loc[idx, "assigned_position"] = p

    selected = df[df["selected"] == 1].copy()
    _print_paper_stats(
        selected,
        salary_cap=salary_cap,
        roster_target=roster_target,
        position_targets=position_targets,
        penalty_pos=penalty_pos,
        lambda_R=lambda_R,
    )
    return selected


def _print_paper_stats(
    selected: pd.DataFrame,
    *,
    salary_cap: float,
    roster_target: int,
    position_targets: dict,
    penalty_pos: float,
    lambda_R: float,
) -> None:
    roster_n = len(selected)
    shortage = max(0, int(roster_target) - int(roster_n))
    total_salary = float(pd.to_numeric(selected.get("salary"), errors="coerce").fillna(0.0).sum())
    total_mv = float(pd.to_numeric(selected.get("MV_i"), errors="coerce").fillna(0.0).sum())

    counts = selected.get("assigned_position", pd.Series([], dtype=str)).value_counts().to_dict()
    dev = {p: abs(int(counts.get(p, 0)) - int(position_targets.get(p, 0))) for p in position_targets.keys()}
    dev_total = int(sum(dev.values()))

    print("\n  [OK] Paper optimization completed")
    print(f"    - Roster size: {roster_n} (shortage u={shortage}, lambda_R={lambda_R})")
    print(f"    - Total salary: ${total_salary:,.0f} / ${float(salary_cap):,.0f}")
    print(f"    - Total MV: ${total_mv:,.0f}")
    print(f"    - Position counts: {counts} (target={position_targets}, dev={dev_total}, penalty_pos={penalty_pos})")

def _optimize_milp(
    df: pd.DataFrame,
    salary_cap: float,
    roster_size: int,
    position_quotas: dict,
    transfer_budget: float,
    min_rookies: int,
    max_veterans: int
) -> pd.DataFrame:
    """
    使用MILP求解最优阵容
    """
    # 创建问题
    prob = LpProblem("Roster_Optimization", LpMaximize)

    # 决策变量
    indices = df.index.tolist()
    x = LpVariable.dicts("x", indices, cat=LpBinary)
    a = LpVariable.dicts("a",
                         [(i, p) for i in indices for p in ['G', 'F', 'C']],
                         cat=LpBinary)

    # 目标函数: 最大化总MV
    prob += lpSum([df.loc[i, 'MV_i'] * x[i] for i in indices]), "Total_MV"

    # 约束1: 工资帽
    prob += (
        lpSum([df.loc[i, 'salary'] * x[i] for i in indices]) <= salary_cap,
        "Salary_Cap"
    )

    # 约束2: 阵容人数
    prob += (
        lpSum([x[i] for i in indices]) == roster_size,
        "Roster_Size"
    )

    # 约束3: 位置配额
    for p in ['G', 'F', 'C']:
        prob += (
            lpSum([a[i, p] for i in indices]) == position_quotas[p],
            f"Position_Quota_{p}"
        )

    # 约束4: 位置可打性
    for i in indices:
        positions = df.loc[i, 'position_set']

        # 每个球员最多分配到一个位置
        prob += (
            lpSum([a[i, p] for p in ['G', 'F', 'C']]) <= x[i],
            f"Position_Assignment_{i}"
        )

        # 只能分配到可打的位置
        for p in ['G', 'F', 'C']:
            if p not in positions:
                prob += (a[i, p] == 0, f"Position_Eligibility_{i}_{p}")
            else:
                prob += (a[i, p] <= x[i], f"Position_Link_{i}_{p}")

    # 约束5: 转会预算
    prob += (
        lpSum([df.loc[i, 'transfer_fee'] * x[i] for i in indices]) <= transfer_budget,
        "Transfer_Budget"
    )

    # 约束6: 新秀最小数量
    if min_rookies > 0:
        rookie_indices = df[df['player_type'] == 'rookie'].index.tolist()
        prob += (
            lpSum([x[i] for i in rookie_indices]) >= min_rookies,
            "Min_Rookies"
        )

    # 约束7: 老将最大数量
    if max_veterans is not None:
        veteran_indices = df[df['player_type'] == 'veteran'].index.tolist()
        prob += (
            lpSum([x[i] for i in veteran_indices]) <= max_veterans,
            "Max_Veterans"
        )

    # 求解
    print(f"\n  求解MILP (使用PuLP)...")
    status = prob.solve(PULP_CBC_CMD(msg=0))

    if status != 1:  # LpStatusOptimal
        raise ValueError(f"优化失败: {LpStatus[status]}")

    # 提取结果
    df = df.copy()
    df['selected'] = [x[i].varValue for i in indices]
    df['assigned_position'] = None

    for i in indices:
        for p in ['G', 'F', 'C']:
            if a[i, p].varValue == 1:
                df.loc[i, 'assigned_position'] = p
                break

    selected = df[df['selected'] == 1].copy()

    # 输出统计
    print(f"\n  [OK] MILP优化成功!")
    _print_roster_stats(selected, salary_cap, transfer_budget)

    return selected


def _optimize_greedy(
    df: pd.DataFrame,
    salary_cap: float,
    roster_size: int,
    position_quotas: dict,
    transfer_budget: float,
    min_rookies: int,
    max_veterans: int
) -> pd.DataFrame:
    """
    使用贪心算法求解近似最优阵容

    策略:
    1. 按MV/cost比率排序
    2. 依次选择球员，满足约束
    3. 优先满足位置配额
    4. 如果位置配额已满但阵容未满，放宽位置约束
    """
    print(f"\n  使用贪心算法求解...")

    df = df.copy()
    df['selected'] = 0
    df['assigned_position'] = None

    # 计算性价比
    df['value_ratio'] = df['MV_i'] / (df['cost_i'] + 1)

    # 按性价比排序
    candidates = df.sort_values('value_ratio', ascending=False).copy()

    selected_indices = []
    total_salary = 0
    total_fee = 0
    position_counts = {'G': 0, 'F': 0, 'C': 0}
    rookie_count = 0
    veteran_count = 0

    def can_still_fill_roster(candidate_idx: int, *, next_salary: float, next_fee: float) -> bool:
        remaining_slots = roster_size - (len(selected_indices) + 1)
        if remaining_slots <= 0:
            return True

        remaining = candidates[~candidates.index.isin(selected_indices + [candidate_idx])]
        if len(remaining) < remaining_slots:
            return False

        salary_budget_left = salary_cap - (total_salary + float(next_salary))
        fee_budget_left = transfer_budget - (total_fee + float(next_fee))

        min_salary_needed = remaining['salary'].nsmallest(remaining_slots).sum()
        min_fee_needed = remaining['transfer_fee'].nsmallest(remaining_slots).sum()

        return (salary_budget_left >= float(min_salary_needed)) and (fee_budget_left >= float(min_fee_needed))

    # 第一轮: 严格按位置配额选择
    for idx, row in candidates.iterrows():
        # 检查是否已满足位置配额
        if all(position_counts[p] >= position_quotas[p] for p in ['G', 'F', 'C']):
            break

        # 检查工资帽
        if total_salary + row['salary'] > salary_cap:
            continue

        if not can_still_fill_roster(int(idx), next_salary=float(row['salary']), next_fee=float(row['transfer_fee'])):
            continue

        # 检查转会预算
        if total_fee + row['transfer_fee'] > transfer_budget:
            continue

        # 检查老将上限
        if max_veterans is not None and row['player_type'] == 'veteran':
            if veteran_count >= max_veterans:
                continue

        # 尝试分配位置
        assigned = False
        for pos in row['position_set']:
            if position_counts[pos] < position_quotas[pos]:
                # 可以分配到这个位置
                selected_indices.append(idx)
                total_salary += row['salary']
                total_fee += row['transfer_fee']
                position_counts[pos] += 1
                df.loc[idx, 'selected'] = 1
                df.loc[idx, 'assigned_position'] = pos

                if row['player_type'] == 'rookie':
                    rookie_count += 1
                elif row['player_type'] == 'veteran':
                    veteran_count += 1

                assigned = True
                break

        if not assigned:
            continue

    # 第二轮: 如果阵容未满，放宽位置约束（允许超配）
    if len(selected_indices) < roster_size:
        print(f"  [INFO] 位置配额已满但阵容未满 ({len(selected_indices)}/{roster_size})，放宽位置约束...")

        for idx, row in candidates.iterrows():
            # 跳过已选择的球员
            if idx in selected_indices:
                continue

            # 检查是否已满
            if len(selected_indices) >= roster_size:
                break

            # 检查工资帽
            if total_salary + row['salary'] > salary_cap:
                continue

            if not can_still_fill_roster(int(idx), next_salary=float(row['salary']), next_fee=float(row['transfer_fee'])):
                continue

            # 检查转会预算
            if total_fee + row['transfer_fee'] > transfer_budget:
                continue

            # 检查老将上限
            if max_veterans is not None and row['player_type'] == 'veteran':
                if veteran_count >= max_veterans:
                    continue

            # 分配到任意可打的位置（优先选择未超配的位置）
            assigned = False
            for pos in row['position_set']:
                selected_indices.append(idx)
                total_salary += row['salary']
                total_fee += row['transfer_fee']
                position_counts[pos] += 1
                df.loc[idx, 'selected'] = 1
                df.loc[idx, 'assigned_position'] = pos

                if row['player_type'] == 'rookie':
                    rookie_count += 1
                elif row['player_type'] == 'veteran':
                    veteran_count += 1

                assigned = True
                break

            if not assigned:
                continue

    # 第三轮: 强制凑满阵容 - 选择低薪球员（即使MV为负）
    if len(selected_indices) < roster_size:
        print(f"  [INFO] 第三轮: 强制凑满阵容 ({len(selected_indices)}/{roster_size})...")
        
        # 按薪资升序排列（选择最便宜的球员）
        remaining = candidates[~candidates.index.isin(selected_indices)].sort_values('salary', ascending=True)
        
        for idx, row in remaining.iterrows():
            if len(selected_indices) >= roster_size:
                break
                
            # 检查工资帽
            if total_salary + row['salary'] > salary_cap:
                continue
                
            # 检查转会预算
            if total_fee + row['transfer_fee'] > transfer_budget:
                continue
            
            # 分配到任意可打的位置
            for pos in row['position_set']:
                selected_indices.append(idx)
                total_salary += row['salary']
                total_fee += row['transfer_fee']
                position_counts[pos] += 1
                df.loc[idx, 'selected'] = 1
                df.loc[idx, 'assigned_position'] = pos
                
                if row['player_type'] == 'rookie':
                    rookie_count += 1
                elif row['player_type'] == 'veteran':
                    veteran_count += 1
                break

    # 检查新秀最小数量
    if rookie_count < min_rookies:
        print(f"  [WARN] 警告: 新秀数量不足 ({rookie_count}/{min_rookies})")

    # 检查阵容人数 - 必须凑满
    if len(selected_indices) < roster_size:
        remaining_slots = roster_size - len(selected_indices)
        remaining_budget = salary_cap - total_salary
        print(f"  [ERROR] 阵容人数不足 ({len(selected_indices)}/{roster_size})")
        print(f"  [ERROR] 剩余工资帽空间: ${remaining_budget:,.0f}")
        print(f"  [ERROR] 需要额外 {remaining_slots} 人，但无法在约束内找到合适球员")
        raise ValueError(
            f"无法凑满阵容: 只选到 {len(selected_indices)}/{roster_size} 人。"
            f"剩余工资帽: ${remaining_budget:,.0f}。"
            f"建议: 1) 提高工资帽 2) 增加候选球员池"
        )

    selected = df[df['selected'] == 1].copy()

    print(f"\n  [OK] 贪心算法完成!")
    _print_roster_stats(selected, salary_cap, transfer_budget)

    return selected


def _print_roster_stats(selected: pd.DataFrame, salary_cap: float, transfer_budget: float):
    """打印阵容统计信息"""
    print(f"    - 总MV: ${selected['MV_i'].sum():,.0f}")
    print(f"    - 总薪资: ${selected['salary'].sum():,.0f} / ${salary_cap:,.0f}")
    print(f"    - 总转会费: ${selected['transfer_fee'].sum():,.0f} / ${transfer_budget:,.0f}")
    print(f"    - 位置分布: G={len(selected[selected['assigned_position']=='G'])}, "
          f"F={len(selected[selected['assigned_position']=='F'])}, "
          f"C={len(selected[selected['assigned_position']=='C'])}")
    print(f"    - 球员类型: 新秀={len(selected[selected['player_type']=='rookie'])}, "
          f"自由球员={len(selected[selected['player_type']=='free_agent'])}, "
          f"老将={len(selected[selected['player_type']=='veteran'])}")


def optimize_with_scenarios(
    df: pd.DataFrame,
    scenarios: list
) -> dict:
    """
    多场景优化 (敏感性分析)

    Args:
        df: DataFrame with player data
        scenarios: List of scenario dicts, e.g.,
            [
                {'name': 'conservative', 'salary_cap': 1200000, 'transfer_budget': 300000},
                {'name': 'aggressive', 'salary_cap': 1800000, 'transfer_budget': 700000}
            ]

    Returns:
        Dict of {scenario_name: selected_roster_df}
    """
    results = {}

    for scenario in scenarios:
        name = scenario.pop('name')
        print(f"\n{'='*80}")
        print(f"场景: {name}")
        print(f"{'='*80}")

        selected = optimize_roster(df, **scenario)
        results[name] = selected

    return results


def validate_roster(df: pd.DataFrame, constraints: dict) -> dict:
    """
    验证阵容是否满足所有约束

    Args:
        df: Selected roster DataFrame
        constraints: Constraint dict (salary_cap, roster_size, etc.)

    Returns:
        Dict of validation results
    """
    validation = {
        'valid': True,
        'violations': []
    }

    # 检查工资帽
    total_salary = df['salary'].sum()
    if total_salary > constraints.get('salary_cap', float('inf')):
        validation['valid'] = False
        validation['violations'].append(
            f"工资帽超支: ${total_salary:,.0f} > ${constraints['salary_cap']:,.0f}"
        )

    # 检查阵容人数
    roster_size = len(df)
    if roster_size != constraints.get('roster_size', roster_size):
        validation['valid'] = False
        validation['violations'].append(
            f"阵容人数不符: {roster_size} != {constraints['roster_size']}"
        )

    # 检查位置配额
    position_quotas = constraints.get('position_quotas', {})
    for p, quota in position_quotas.items():
        actual = len(df[df['assigned_position'] == p])
        if actual != quota:
            validation['valid'] = False
            validation['violations'].append(
                f"位置{p}配额不符: {actual} != {quota}"
            )

    # 检查转会预算
    total_fee = df['transfer_fee'].sum()
    if total_fee > constraints.get('transfer_budget', float('inf')):
        validation['valid'] = False
        validation['violations'].append(
            f"转会预算超支: ${total_fee:,.0f} > ${constraints['transfer_budget']:,.0f}"
        )

    return validation


def export_roster_summary(selected: pd.DataFrame, output_path: str = None) -> pd.DataFrame:
    """
    导出阵容摘要

    Returns:
        Summary DataFrame
    """
    base_cols = ["player_name", "player_type", "age", "assigned_position", "salary", "MV_i"]
    optional_cols = [
        # paper inputs
        "team_id",
        "experience",
        "mu0_perf",
        "mu1_perf",
        "mu0_pop",
        "mu1_pop",
        "inj_score",
        "sigma_perf2",
        "d_i",
        "g_i",
        # legacy inputs
        "perf_i",
        "pop_i",
        "risk_i",
        "cost_i",
        "s_dec_i",
        "s_grow_i",
        # shared
        "transfer_fee",
        "pcv",
        "ppg",
        "apg",
        "rpg",
        "mpg",
        "court_impact",
        "brand_impact",
    ]

    cols = [c for c in base_cols + optional_cols if c in selected.columns]
    summary = selected[cols].copy()

    if "assigned_position" in summary.columns and "MV_i" in summary.columns:
        summary = summary.sort_values(["assigned_position", "MV_i"], ascending=[True, False])
    elif "MV_i" in summary.columns:
        summary = summary.sort_values("MV_i", ascending=False)

    if output_path:
        summary.to_csv(output_path, index=False)
        print(f"\n  [OK] 阵容摘要已保存: {output_path}")

    return summary
