"""
表现预测模块

功能:
- 基于年龄曲线预测未来H年表现 μ_i,k
- 估算表现方差 σ²_i,k
- 检查增长/下降约束
"""

import numpy as np
import pandas as pd


def forecast_performance(
    df: pd.DataFrame,
    horizon: int = 3
) -> pd.DataFrame:
    """
    预测未来H年表现 μ_i,k 和 σ²_i,k

    基于年龄曲线模型:
    - <23岁: 增长期 (0.85 → 1.0)
    - 23-27岁: 巅峰期 (1.0)
    - >27岁: 衰退期 (1.0 → 0.7)

    新增字段:
    - mu_i_0, mu_i_1, ..., mu_i_H: 各年期望表现
    - sigma2_i_0, ..., sigma2_i_H: 各年方差
    - growth_rate: 增长率 (mu_H - mu_0) / mu_0

    Args:
        df: DataFrame with perf_i, age, performance_variance
        horizon: 预测年数 (default: 3)

    Returns:
        DataFrame with forecast columns
    """
    df = df.copy()

    for k in range(horizon + 1):
        future_age = df['age'] + k

        # 期望表现 (基于年龄曲线)
        age_factor = future_age.apply(age_curve_factor)
        df[f'mu_i_{k}'] = df['perf_i'] * age_factor

        # 方差 (随年龄/时间增加)
        df[f'sigma2_i_{k}'] = df['performance_variance'] * (1 + 0.1 * k)

    # 计算增长率
    df['growth_rate'] = (df[f'mu_i_{horizon}'] - df['mu_i_0']) / \
                        (df['mu_i_0'] + 1e-6)

    # 统计
    print(f"  [OK] 预测期望表现 (mu_i_0): {df['mu_i_0'].mean():.3f} ± {df['mu_i_0'].std():.3f}")
    print(f"  [OK] 预测期望表现 (mu_i_{horizon}): {df[f'mu_i_{horizon}'].mean():.3f} ± {df[f'mu_i_{horizon}'].std():.3f}")
    print(f"  [OK] 平均增长率: {df['growth_rate'].mean():.2%}")

    return df


def age_curve_factor(age: float, peak_age: float = 27) -> float:
    """
    年龄曲线因子

    模型 (基于WNBA球员职业生涯曲线):
    - age < 23: 线性增长 (0.85 → 1.0)
    - 23 <= age <= 27: 巅峰期 (1.0)
    - age > 27: 线性衰退 (1.0 → 0.7, 每年-3%)

    Args:
        age: 球员年龄
        peak_age: 巅峰年龄 (default: 27)

    Returns:
        Performance factor [0.7, 1.0]
    """
    if age < 23:
        # 增长期: 20岁=0.85, 23岁=1.0
        return min(1.0, 0.85 + (age - 20) * 0.05)
    elif age <= peak_age:
        # 巅峰期
        return 1.0
    else:
        # 衰退期: 每年-3%
        return max(0.7, 1.0 - (age - peak_age) * 0.03)


def check_growth_constraints(
    df: pd.DataFrame,
    g_min: float = 0.02,  # 降低新秀增长要求 (原5% → 2%)
    horizon: int = 3
) -> pd.DataFrame:
    """
    检查增长/下降约束

    约束:
    1. 自由球员不下降: s_dec_i = max(0, mu_0 - mu_H - 0.05)  # 允许5%下降
    2. 新秀必须增长: s_grow_i = max(0, g_min - growth_rate)

    新增字段:
    - s_dec_i: 自由球员下降惩罚 (>0表示违反约束)
    - s_grow_i: 新秀增长不足惩罚 (>0表示违反约束)

    Args:
        df: DataFrame with mu_i_k, growth_rate, player_type
        g_min: 新秀最小增长率 (default: 2%, 原5%)
        horizon: 预测期

    Returns:
        DataFrame with constraint violation columns
    """
    df = df.copy()

    # 初始化
    df['s_dec_i'] = 0.0
    df['s_grow_i'] = 0.0

    # 约束1: 自由球员不下降 (允许5%下降空间)
    free_agents = df['player_type'] == 'free_agent'
    if free_agents.sum() > 0:
        df.loc[free_agents, 's_dec_i'] = np.maximum(
            0,
            df.loc[free_agents, 'mu_i_0'] - df.loc[free_agents, f'mu_i_{horizon}'] - 0.05  # 允许5%下降
        )

    # 约束2: 新秀必须增长
    rookies = df['player_type'] == 'rookie'
    if rookies.sum() > 0:
        df.loc[rookies, 's_grow_i'] = np.maximum(
            0,
            g_min - df.loc[rookies, 'growth_rate']
        )

    # 统计
    n_dec_violations = (df['s_dec_i'] > 0).sum()
    n_grow_violations = (df['s_grow_i'] > 0).sum()

    print(f"  [OK] 自由球员下降违规: {n_dec_violations}/{free_agents.sum()} 人")
    print(f"  [OK] 新秀增长不足违规: {n_grow_violations}/{rookies.sum()} 人")

    return df


def get_performance_summary(df: pd.DataFrame, horizon: int = 3) -> pd.DataFrame:
    """
    生成表现预测摘要

    Returns:
        DataFrame with columns:
        - player_name, player_type, age
        - mu_i_0, mu_i_H, growth_rate
        - s_dec_i, s_grow_i
        - forecast_quality: 'excellent'/'good'/'poor'
    """
    summary = df[[
        'player_name', 'player_type', 'age',
        'mu_i_0', f'mu_i_{horizon}', 'growth_rate',
        's_dec_i', 's_grow_i'
    ]].copy()

    # 评估预测质量
    def assess_quality(row):
        if row['player_type'] == 'rookie':
            return 'excellent' if row['growth_rate'] > 0.10 else \
                   'good' if row['growth_rate'] > 0.05 else 'poor'
        elif row['player_type'] == 'free_agent':
            return 'excellent' if row['s_dec_i'] == 0 else 'poor'
        else:  # veteran
            return 'good' if row['growth_rate'] > -0.10 else 'poor'

    summary['forecast_quality'] = summary.apply(assess_quality, axis=1)

    return summary.sort_values('growth_rate', ascending=False)
