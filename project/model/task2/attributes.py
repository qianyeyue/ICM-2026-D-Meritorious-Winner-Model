"""
球员属性降维模块

功能:
- 从player.py输出加载基础数据
- 降维到4类核心属性: perf_i, pop_i, risk_i, cost_i
- 推断位置和球员类型
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


# 配置路径
ROOT_DIR = Path(__file__).resolve().parents[3]  # math_competition_2026/
PCV_FILE = ROOT_DIR / "project" / "data" / "processed" / "players" / "players_pcv.csv"
TASK1_INPUT_DIR = ROOT_DIR / "project" / "model" / "task1_input"
TASK2_CANDIDATES_FILE = TASK1_INPUT_DIR / "candidates_new.csv"
TASK2_PLAYER_INPUTS_FILE = TASK1_INPUT_DIR / "player_inputs_new.csv"
TASK2_POSITION_TARGETS_FILE = TASK1_INPUT_DIR / "position_targets_new.csv"


def _parse_position_set(raw):
    # type: (object) -> list
    s = str(raw).strip() if raw is not None else ""
    if not s:
        return ["F"]

    for delim in ["/", ",", ";", "|"]:
        if delim in s:
            parts = [p.strip() for p in s.replace("|", "/").replace(",", "/").replace(";", "/").split("/") if p.strip()]
            out = []
            for p in parts:
                code = p.strip().upper()
                if code:
                    out.append(code[0])
            return sorted(set(out)) or ["F"]

    s = s.upper()
    return [s[0]]


def load_task2_candidates_from_task1_inputs(
    offseason_year: int = 2025,
) -> tuple[pd.DataFrame, dict]:
    """
    Load Task2 candidate pool and calibrated inputs used by the paper.

    Data sources (preprocessed artifacts):
      - candidates_new.csv: player meta + salary + position + team
      - player_inputs_new.csv: (mu0/mu1 perf & pop, variance, injury score)
      - position_targets_new.csv: target (G,F,C) and penalty_pos for the target team

    Returns:
      (df, cfg) where cfg includes:
        - team_id (paper's team code, e.g. "LAS")
        - penalty_pos
        - position_targets: {"G":5,"F":4,"C":3}
        - roster_target: sum(position_targets.values())
    """
    if not TASK2_CANDIDATES_FILE.exists():
        raise FileNotFoundError(f"Missing Task2 candidates file: {TASK2_CANDIDATES_FILE}")
    if not TASK2_PLAYER_INPUTS_FILE.exists():
        raise FileNotFoundError(f"Missing Task2 player_inputs file: {TASK2_PLAYER_INPUTS_FILE}")
    if not TASK2_POSITION_TARGETS_FILE.exists():
        raise FileNotFoundError(f"Missing Task2 position_targets file: {TASK2_POSITION_TARGETS_FILE}")

    candidates = pd.read_csv(TASK2_CANDIDATES_FILE)
    required_cols = {
        "Player_ID",
        "Player_Name",
        "Position",
        "Age",
        "Experience",
        "Salary",
        "Commercial_Value_PCA",
        "Court_Impact",
        "Injury_Risk",
        "Team",
    }
    missing = required_cols - set(candidates.columns)
    if missing:
        raise ValueError(f"candidates_new.csv missing required columns: {sorted(missing)}")

    candidates = candidates.rename(columns={
        "Player_ID": "player_id",
        "Player_Name": "player_name",
        "Position": "position",
        "Age": "age",
        "Experience": "experience",
        "Salary": "salary",
        "Commercial_Value_PCA": "mu0_pop",
        "Court_Impact": "court_impact",
        "Injury_Risk": "injury_risk",
        "Team": "team_id",
        "PPG": "ppg",
        "RPG": "rpg",
        "APG": "apg",
        "MPG": "mpg",
        "FG_Percent": "fg_pct",
        "TS_Percent": "ts_pct",
    })

    candidates["player_id"] = pd.to_numeric(candidates["player_id"], errors="coerce")
    for col in ["salary", "age", "experience", "mu0_pop", "court_impact", "injury_risk", "ppg", "rpg", "apg", "mpg"]:
        if col in candidates.columns:
            candidates[col] = pd.to_numeric(candidates[col], errors="coerce")

    inputs = pd.read_csv(TASK2_PLAYER_INPUTS_FILE)
    required_inputs = {
        "offseason_year",
        "player_id",
        "mu0_perf",
        "mu1_perf",
        "sigma_perf2",
        "mu0_pop",
        "mu1_pop",
        "inj_score",
    }
    missing_inputs = required_inputs - set(inputs.columns)
    if missing_inputs:
        raise ValueError(f"player_inputs_new.csv missing required columns: {sorted(missing_inputs)}")

    inputs = inputs[inputs["offseason_year"] == offseason_year].copy()
    inputs["player_id"] = pd.to_numeric(inputs["player_id"], errors="coerce")

    df = candidates.merge(
        inputs[[
            "player_id",
            "mu0_perf",
            "mu1_perf",
            "sigma_perf2",
            "mu0_pop",
            "mu1_pop",
            "inj_score",
        ]],
        on="player_id",
        how="left",
        suffixes=("_cand", "_inp"),
    )

    if "mu0_pop_inp" in df.columns:
        df["mu0_pop"] = pd.to_numeric(df["mu0_pop_inp"], errors="coerce").fillna(
            pd.to_numeric(df["mu0_pop_cand"], errors="coerce")
        )
        df = df.drop(columns=["mu0_pop_inp", "mu0_pop_cand"])

    for col in ["mu0_perf", "mu1_perf", "sigma_perf2", "mu0_pop", "mu1_pop", "inj_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["mu1_perf"] = df["mu1_perf"].fillna(df["mu0_perf"])
    df["mu1_pop"] = df["mu1_pop"].fillna(df["mu0_pop"])

    pos_cfg = pd.read_csv(TASK2_POSITION_TARGETS_FILE)
    pos_cfg = pos_cfg[pos_cfg["offseason_year"] == offseason_year].copy()
    if pos_cfg.empty:
        raise ValueError(f"No position_targets row found for offseason_year={offseason_year}")

    cfg_row = pos_cfg.iloc[0]
    cfg_team_id = str(cfg_row.get("team_id", "")).strip() or "LAS"
    penalty_pos = float(cfg_row.get("penalty_pos", 0.0))
    position_targets = {
        "G": int(cfg_row.get("target_G", 0)),
        "F": int(cfg_row.get("target_F", 0)),
        "C": int(cfg_row.get("target_C", 0)),
    }
    roster_target = int(sum(position_targets.values()))

    df["position_set"] = df["position"].apply(_parse_position_set)

    is_rookie_like = df["experience"].isna() | (df["experience"] <= 1)
    is_free_agent = df["team_id"].astype(str) != cfg_team_id
    df["player_type"] = np.select(
        [is_rookie_like, is_free_agent],
        ["rookie", "free_agent"],
        default="retained",
    )

    df["pcv"] = df["mu1_pop"]
    df["brand_impact"] = df["mu1_pop"]
    df["transfer_fee"] = 0.0

    cfg = {
        "offseason_year": int(offseason_year),
        "team_id": cfg_team_id,
        "penalty_pos": float(penalty_pos),
        "position_targets": position_targets,
        "roster_target": roster_target,
    }

    return df, cfg


def load_player_base_data(team_filter: str = None) -> pd.DataFrame:
    """
    从player.py的输出加载基础数据

    可用字段:
    - player_name, salary, pred_salary, pcv
    - ppg, mpg, apg, rpg, FG%, 3P%, FT%
    - court_impact, brand_impact, injury_risk, age
    - fans_w, value_gap, value_ratio

    Args:
        team_filter: 球队名称过滤 (例如: "Phoenix Mercury")

    Returns:
        DataFrame with base player data
    """
    if not PCV_FILE.exists():
        raise FileNotFoundError(
            f"未找到 {PCV_FILE}\n"
            "请先运行: python project/model/players.py\n"
            "生成 players_pcv.csv"
        )

    df = pd.read_csv(PCV_FILE)

    # 验证必需字段
    required = ['player_name', 'court_impact', 'brand_impact',
                'injury_risk', 'salary', 'pcv', 'age']
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"缺少必需字段: {missing}")

    # 尝试加载球队信息
    team_file = ROOT_DIR / "project" / "data" / "team_rosters.csv"
    if team_file.exists():
        teams = pd.read_csv(team_file)
        df = df.merge(teams[['player_name', 'team']], on='player_name', how='left')

        if team_filter:
            original_count = len(df)
            df = df[df['team'] == team_filter].copy()
            print(f"  [OK] 过滤到球队: {team_filter} ({len(df)}/{original_count}名球员)")
    elif team_filter:
        print(f"  [WARN] 未找到team_rosters.csv，无法过滤球队")

    print(f"  [OK] 加载 {len(df)} 名球员")
    print(f"  [OK] 可用字段: {len(df.columns)} 个")

    return df


def add_player_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """
    在player.py输出基础上添加Task2所需属性

    新增字段:
    - perf_i: 竞技贡献 [0-1标准化]
    - pop_i: 商业贡献 [PCV]
    - risk_i: 综合风险 [injury + variance]
    - cost_i: 总成本 [salary + fee]
    - position_set: 位置集合 ['G'], ['F'], ['C'], ['G','F']
    - player_type: 'rookie'/'free_agent'/'veteran'
    - performance_variance: 表现不确定性
    - transfer_fee: 转会费估算

    Returns:
        DataFrame with extended attributes
    """
    df = df.copy()

    # 1. 竞技贡献 (标准化court_impact到0-1)
    court_min = df['court_impact'].min()
    court_max = df['court_impact'].max()
    df['perf_i'] = (df['court_impact'] - court_min) / (court_max - court_min + 1e-6)

    # 2. 商业贡献 (直接用pcv)
    df['pop_i'] = df['pcv']

    # 3. 表现方差估算
    df['performance_variance'] = estimate_variance(df)

    # 4. 综合风险 (injury_risk + performance_variance)
    df['risk_i'] = df['injury_risk'] + 0.3 * df['performance_variance']

    # 5. 转会费估算
    df['transfer_fee'] = estimate_transfer_fee(df)

    # 6. 总成本
    df['cost_i'] = df['salary'] + df['transfer_fee']

    # 7. 位置推断
    df['position_set'] = df.apply(infer_position, axis=1)

    # 8. 球员类型推断
    df['player_type'] = df['age'].apply(classify_player_type)

    print(f"  [OK] 竞技贡献 (perf_i): {df['perf_i'].mean():.3f} ± {df['perf_i'].std():.3f}")
    print(f"  [OK] 商业贡献 (pop_i): {df['pop_i'].mean():.3f} ± {df['pop_i'].std():.3f}")
    print(f"  [OK] 综合风险 (risk_i): {df['risk_i'].mean():.3f} ± {df['risk_i'].std():.3f}")
    print(f"  [OK] 总成本 (cost_i): ${df['cost_i'].mean():,.0f} ± ${df['cost_i'].std():,.0f}")

    return df


def estimate_variance(df: pd.DataFrame) -> pd.Series:
    """
    估算表现不确定性 σ²

    逻辑:
    - 新秀(age<24): 高方差 (0.15-0.25) - 潜力未知
    - 巅峰期(24-30): 低方差 (0.05-0.10) - 稳定表现
    - 老将(>30): 中等方差 (0.10-0.20) - 伤病/衰退风险

    调整因子:
    - 上场时间波动 → 增加方差
    - 得分效率低 → 增加方差

    Returns:
        Series of variance values [0.05, 0.30]
    """
    variance = np.zeros(len(df))

    # 基础方差（基于年龄）
    age = df['age'].values
    variance += np.where(age < 24, 0.20, 0)  # 新秀
    variance += np.where((age >= 24) & (age <= 30), 0.08, 0)  # 巅峰
    variance += np.where(age > 30, 0.15, 0)  # 老将

    # 调整因子1: 上场时间稳定性
    if 'mpg' in df.columns:
        mpg_normalized = (df['mpg'] - df['mpg'].mean()) / (df['mpg'].std() + 1e-6)
        variance += 0.03 * np.abs(mpg_normalized)

    # 调整因子2: 得分效率
    if 'FG%' in df.columns:
        low_efficiency = df['FG%'] < 0.40
        variance += np.where(low_efficiency, 0.05, 0)

    return pd.Series(variance, index=df.index).clip(0.05, 0.30)


def estimate_transfer_fee(df: pd.DataFrame) -> pd.Series:
    """
    估算转会费/交易成本

    逻辑:
    - 自由球员: 0 (无需交易)
    - 高价值球员(pcv>1.5): salary * 0.2 (需要交易筹码)
    - 新秀: salary * 0.1 (签约金)
    - 其他: 0

    Returns:
        Series of transfer fees
    """
    fee = np.zeros(len(df))

    # 高价值球员需要交易成本
    high_value = df['pcv'] > 1.5
    fee = np.where(high_value, df['salary'] * 0.2, 0)

    # 新秀签约金
    rookie = df['age'] < 24
    fee = np.where(rookie & ~high_value, df['salary'] * 0.1, fee)

    return pd.Series(fee, index=df.index)


def infer_position(row: pd.Series) -> list:
    """
    根据统计特征推断位置

    规则:
    - 高apg (>4) & 低rpg (<5) → G (后卫)
    - 高rpg (>7) → C (中锋)
    - 高apg (>3) & 高rpg (>5) → ['G','F'] (全能前锋)
    - 其他 → F (前锋)

    Returns:
        List of positions ['G'], ['F'], ['C'], or ['G','F'], etc.
    """
    apg = row.get('apg', 0)
    rpg = row.get('rpg', 0)
    ppg = row.get('ppg', 0)

    # 后卫: 高助攻低篮板
    if apg > 4 and rpg < 5:
        return ['G']

    # 中锋: 高篮板
    elif rpg > 7:
        return ['C']

    # 全能前锋: 助攻+篮板都不错
    elif apg > 3 and rpg > 5:
        return ['G', 'F']

    # 内线前锋: 高篮板中等得分
    elif rpg > 5 and ppg > 12:
        return ['F', 'C']

    # 默认前锋
    else:
        return ['F']


def classify_player_type(age: float) -> str:
    """
    根据年龄分类球员类型

    - age < 24: rookie (新秀)
    - 24 <= age < 32: free_agent (自由球员/巅峰期)
    - age >= 32: veteran (老将)

    Returns:
        'rookie', 'free_agent', or 'veteran'
    """
    if age < 24:
        return 'rookie'
    elif age < 32:
        return 'free_agent'
    else:
        return 'veteran'
