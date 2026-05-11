"""
边际价值计算模块

功能:
- 计算球员净边际价值 MV_i
- 考虑竞技贡献、商业贡献、风险、成本
- 支持自定义权重参数
"""

from __future__ import annotations

import pandas as pd
import numpy as np


PAPER_DEFAULT_WEIGHTS = {
    "v_W": 20000,
    "v_pop": 60000,
    "v_inj": 5000,
    "v_var": 30000,
    "v_dec": 40000,
    "v_grow": 40000,
}


def compute_marginal_value_paper(
    df: pd.DataFrame,
    *,
    weights: dict | None = None,
    g_min: float = 0.2,
) -> pd.DataFrame:
    """
    Paper-consistent MV definition (Task2).
    """
    df = df.copy()
    w = PAPER_DEFAULT_WEIGHTS.copy()
    if weights:
        w.update(weights)

    required = {"mu0_perf", "mu1_perf", "mu1_pop", "salary", "inj_score", "sigma_perf2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for paper MV: {sorted(missing)}")

    mu0 = pd.to_numeric(df["mu0_perf"], errors="coerce").fillna(0.0)
    mu1 = pd.to_numeric(df["mu1_perf"], errors="coerce").fillna(mu0)
    mu_pop = pd.to_numeric(df["mu1_pop"], errors="coerce").fillna(0.0)
    salary = pd.to_numeric(df["salary"], errors="coerce").fillna(0.0)
    inj = pd.to_numeric(df["inj_score"], errors="coerce").fillna(0.0)
    sigma2 = pd.to_numeric(df["sigma_perf2"], errors="coerce").fillna(0.0)

    if "player_type" in df.columns:
        player_type = df["player_type"].astype(str)
    else:
        player_type = pd.Series([""] * len(df), index=df.index, dtype="object")
    is_fa = player_type == "free_agent"
    is_rookie = player_type == "rookie"

    df["d_i"] = 0.0
    df.loc[is_fa, "d_i"] = np.maximum(0.0, mu0[is_fa] - mu1[is_fa])

    df["g_i"] = 0.0
    df.loc[is_rookie, "g_i"] = np.maximum(0.0, float(g_min) - (mu1[is_rookie] - mu0[is_rookie]))

    df["mv_performance"] = w["v_W"] * mu1
    df["mv_commercial"] = w["v_pop"] * mu_pop
    df["mv_salary"] = -salary
    df["mv_injury"] = -w["v_inj"] * inj
    df["mv_variance"] = -w["v_var"] * sigma2
    df["mv_decay_penalty"] = -w["v_dec"] * df["d_i"]
    df["mv_growth_penalty"] = -w["v_grow"] * df["g_i"]

    df["MV_i"] = (
        df["mv_performance"]
        + df["mv_commercial"]
        + df["mv_salary"]
        + df["mv_injury"]
        + df["mv_variance"]
        + df["mv_decay_penalty"]
        + df["mv_growth_penalty"]
    )

    # Backward-compatible aliases for visualization/legacy utilities.
    if "risk_i" not in df.columns:
        if "injury_risk" in df.columns:
            df["risk_i"] = pd.to_numeric(df["injury_risk"], errors="coerce").fillna(0.0)
        else:
            df["risk_i"] = inj
    if "cost_i" not in df.columns:
        df["cost_i"] = salary
    if "E_delta_W" not in df.columns:
        df["E_delta_W"] = mu1
    if "E_delta_pop" not in df.columns:
        df["E_delta_pop"] = mu_pop
    if "s_dec_i" not in df.columns:
        df["s_dec_i"] = df["d_i"]
    if "s_grow_i" not in df.columns:
        df["s_grow_i"] = df["g_i"]
    if "growth_rate" not in df.columns:
        df["growth_rate"] = (mu1 - mu0) / (mu0 + 1e-6)
    if "perf_i" not in df.columns:
        df["perf_i"] = mu1
    if "pop_i" not in df.columns:
        df["pop_i"] = mu_pop

    return df


def compute_marginal_value(
    df: pd.DataFrame,
    weights: dict = None,
    *,
    g_min: float = 0.2,
    mode: str | None = None,
) -> pd.DataFrame:
    """
    计算净边际价值 MV_i

    公式:
    MV_i = v_W * E[ΔW_i] + v_pop * E[Δpop_i]
           - cost_i
           - v_inj * risk_i
           - v_dec * s_dec_i
           - v_grow * s_grow_i

    新增字段:
    - E_delta_W: 期望胜场贡献
    - E_delta_pop: 期望商业贡献
    - MV_i: 净边际价值

    Args:
        df: DataFrame with perf_i, pop_i, risk_i, cost_i, s_dec_i, s_grow_i
        weights: 权重字典 {v_W, v_pop, v_inj, v_dec, v_grow}

    Returns:
        DataFrame with MV_i column
    """
    if mode == "paper" or {"mu0_perf", "mu1_perf", "mu1_pop", "inj_score", "sigma_perf2"}.issubset(df.columns):
        return compute_marginal_value_paper(df, weights=weights, g_min=g_min)

    df = df.copy()

    # 默认权重 (已优化: 降低惩罚权重50%)
    if weights is None:
        weights = {
            'v_W': 1000000,      # $1M/win (胜场价值)
            'v_pop': 500000,     # 商业价值系数
            'v_inj': 100000,     # 风险惩罚 (降低50%)
            'v_dec': 150000,     # 下降惩罚 (降低50%)
            'v_grow': 100000     # 增长不足惩罚 (降低50%)
        }

    # 期望胜场贡献 (简化模型: perf_i * 系数)
    # 假设top球员(perf_i=1.0)贡献0.1胜场
    df['E_delta_W'] = df['perf_i'] * 0.1

    # 期望商业贡献 (相对于平均水平)
    # pop_i=1.0是平均水平, >1.0是正贡献
    df['E_delta_pop'] = df['pop_i'] - 1.0

    # 计算MV
    df['MV_i'] = (
        weights['v_W'] * df['E_delta_W'] +
        weights['v_pop'] * df['E_delta_pop'] -
        df['cost_i'] -
        weights['v_inj'] * df['risk_i'] -
        weights['v_dec'] * df['s_dec_i'] -
        weights['v_grow'] * df['s_grow_i']
    )

    # 统计
    print(f"  [OK] 平均MV: ${df['MV_i'].mean():,.0f}")
    print(f"  [OK] MV范围: [${df['MV_i'].min():,.0f}, ${df['MV_i'].max():,.0f}]")
    print(f"  [OK] 正MV球员: {(df['MV_i'] > 0).sum()}/{len(df)} 人")

    return df


def compute_mv_components(df: pd.DataFrame, weights: dict = None) -> pd.DataFrame:
    """
    分解MV各组成部分 (用于分析)

    新增字段:
    - mv_performance: 竞技贡献部分
    - mv_commercial: 商业贡献部分
    - mv_cost: 成本部分 (负值)
    - mv_risk: 风险惩罚部分 (负值)
    - mv_constraints: 约束惩罚部分 (负值)

    Returns:
        DataFrame with MV component columns
    """
    df = df.copy()

    if weights is None:
        weights = {
            'v_W': 1000000,
            'v_pop': 500000,
            'v_inj': 200000,
            'v_dec': 300000,
            'v_grow': 200000
        }

    # 计算各组成部分
    df['mv_performance'] = weights['v_W'] * df['E_delta_W']
    df['mv_commercial'] = weights['v_pop'] * df['E_delta_pop']
    df['mv_cost'] = -df['cost_i']
    df['mv_risk'] = -weights['v_inj'] * df['risk_i']
    df['mv_constraints'] = -(
        weights['v_dec'] * df['s_dec_i'] +
        weights['v_grow'] * df['s_grow_i']
    )

    return df


def rank_players_by_mv(
    df: pd.DataFrame,
    top_n: int = 20,
    position_filter: str = None
) -> pd.DataFrame:
    """
    按MV排序球员

    Args:
        df: DataFrame with MV_i
        top_n: 返回前N名
        position_filter: 位置过滤 ('G', 'F', 'C', or None)

    Returns:
        Sorted DataFrame with top_n players
    """
    result = df.copy()

    # 位置过滤
    if position_filter:
        result = result[result['position_set'].apply(
            lambda x: position_filter in x
        )]

    # 排序
    result = result.sort_values('MV_i', ascending=False).head(top_n)

    return result[[
        'player_name', 'player_type', 'age', 'position_set',
        'perf_i', 'pop_i', 'risk_i', 'cost_i',
        'MV_i', 'salary', 'transfer_fee'
    ]]


def sensitivity_analysis(
    df: pd.DataFrame,
    param_name: str,
    param_range: list
) -> pd.DataFrame:
    """
    MV参数敏感性分析

    Args:
        df: DataFrame with base attributes
        param_name: 参数名 ('v_W', 'v_pop', 'v_inj', etc.)
        param_range: 参数取值范围 [min, max, step]

    Returns:
        DataFrame with sensitivity results
    """
    base_weights = {
        'v_W': 1000000,
        'v_pop': 500000,
        'v_inj': 200000,
        'v_dec': 300000,
        'v_grow': 200000
    }

    results = []
    param_values = np.arange(*param_range)

    for val in param_values:
        weights = base_weights.copy()
        weights[param_name] = val

        df_temp = compute_marginal_value(df.copy(), weights=weights)

        results.append({
            param_name: val,
            'mean_mv': df_temp['MV_i'].mean(),
            'std_mv': df_temp['MV_i'].std(),
            'positive_mv_count': (df_temp['MV_i'] > 0).sum()
        })

    return pd.DataFrame(results)


def calibrate_weights_from_history(
    historical_data: pd.DataFrame,
    target_col: str = 'actual_wins_contributed'
) -> dict:
    """
    从历史数据校准权重 (可选高级功能)

    使用线性回归拟合:
    actual_outcome ~ v_W * perf + v_pop * pop - v_inj * risk

    Args:
        historical_data: 历史数据 (需包含实际结果)
        target_col: 目标变量列名

    Returns:
        Calibrated weights dict
    """
    from sklearn.linear_model import LinearRegression

    # 准备特征
    X = historical_data[['E_delta_W', 'E_delta_pop', 'risk_i']].values
    y = historical_data[target_col].values

    # 拟合
    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)

    # 提取权重
    weights = {
        'v_W': abs(model.coef_[0]),
        'v_pop': abs(model.coef_[1]),
        'v_inj': abs(model.coef_[2]),
        'v_dec': 300000,  # 保持默认
        'v_grow': 200000  # 保持默认
    }

    print(f"  [OK] 校准权重: v_W=${weights['v_W']:,.0f}, v_pop=${weights['v_pop']:,.0f}")

    return weights
