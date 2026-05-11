"""
历史数据校准模块

功能:
- 从历史多赛季数据校准MV权重
- 使用实际胜场/收入数据拟合权重
- 支持交叉验证
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple


def load_historical_data(
    historical_file: str = "project/data/historical_performance.csv"
) -> pd.DataFrame:
    """
    加载历史数据

    期望格式:
    - player_name, season, team
    - actual_wins_contributed: 实际胜场贡献
    - actual_revenue_generated: 实际收入贡献
    - perf_i, pop_i, risk_i: 球员属性
    - E_delta_W, E_delta_pop: 期望贡献

    Returns:
        DataFrame with historical data
    """
    file_path = Path(historical_file)

    if not file_path.exists():
        print(f"  [WARN] 历史数据文件不存在: {historical_file}")
        print(f"  [INFO] 将使用默认权重")
        return None

    df = pd.read_csv(file_path)

    required = ['player_name', 'season', 'actual_wins_contributed',
                'E_delta_W', 'E_delta_pop', 'risk_i']
    missing = set(required) - set(df.columns)

    if missing:
        print(f"  [WARN] 历史数据缺少字段: {missing}")
        return None

    print(f"  [OK] 加载历史数据: {len(df)}条记录, {df['season'].nunique()}个赛季")
    return df


def calibrate_weights_from_history(
    historical_data: pd.DataFrame,
    target_col: str = 'actual_wins_contributed',
    method: str = 'ridge'
) -> Dict[str, float]:
    """
    从历史数据校准权重

    使用回归拟合:
    actual_outcome ~ v_W * E_delta_W + v_pop * E_delta_pop - v_inj * risk_i

    Args:
        historical_data: 历史数据
        target_col: 目标变量列名
        method: 回归方法 ('ridge', 'lasso', 'linear')

    Returns:
        Calibrated weights dict
    """
    if historical_data is None or len(historical_data) < 20:
        print(f"  [WARN] 历史数据不足 (<20条)，使用默认权重")
        return get_default_weights()

    # 准备特征
    X = historical_data[['E_delta_W', 'E_delta_pop', 'risk_i']].values
    y = historical_data[target_col].values

    # 移除缺失值
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]

    if len(X) < 10:
        print(f"  [WARN] 有效样本不足 (<10条)，使用默认权重")
        return get_default_weights()

    # 选择回归方法
    if method == 'ridge':
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)
    elif method == 'lasso':
        from sklearn.linear_model import Lasso
        model = Lasso(alpha=0.1)
    else:
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()

    # 拟合
    model.fit(X, y)

    # 提取权重 (取绝对值)
    weights = {
        'v_W': abs(model.coef_[0]) * 1000000,  # 缩放到合理范围
        'v_pop': abs(model.coef_[1]) * 500000,
        'v_inj': abs(model.coef_[2]) * 200000,
        'v_dec': 150000,  # 保持优化后的默认值
        'v_grow': 100000  # 保持优化后的默认值
    }

    # 评估
    y_pred = model.predict(X)
    r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)

    print(f"  [OK] 权重校准完成 (R²={r2:.3f})")
    print(f"      v_W=${weights['v_W']:,.0f}, v_pop=${weights['v_pop']:,.0f}, v_inj=${weights['v_inj']:,.0f}")

    return weights


def cross_validate_weights(
    historical_data: pd.DataFrame,
    n_folds: int = 5
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    交叉验证权重校准

    Args:
        historical_data: 历史数据
        n_folds: 折数

    Returns:
        (mean_weights, std_weights)
    """
    from sklearn.model_selection import KFold

    if historical_data is None or len(historical_data) < n_folds * 10:
        print(f"  [WARN] 数据不足以进行{n_folds}折交叉验证")
        return get_default_weights(), {}

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    X = historical_data[['E_delta_W', 'E_delta_pop', 'risk_i']].values
    y = historical_data['actual_wins_contributed'].values

    # 移除缺失值
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]

    weights_list = []
    r2_list = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)

        # 评估
        y_pred = model.predict(X_test)
        r2 = 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2)
        r2_list.append(r2)

        # 记录权重
        weights_list.append({
            'v_W': abs(model.coef_[0]) * 1000000,
            'v_pop': abs(model.coef_[1]) * 500000,
            'v_inj': abs(model.coef_[2]) * 200000
        })

    # 计算平均权重
    mean_weights = {
        'v_W': np.mean([w['v_W'] for w in weights_list]),
        'v_pop': np.mean([w['v_pop'] for w in weights_list]),
        'v_inj': np.mean([w['v_inj'] for w in weights_list]),
        'v_dec': 150000,
        'v_grow': 100000
    }

    # 计算标准差
    std_weights = {
        'v_W': np.std([w['v_W'] for w in weights_list]),
        'v_pop': np.std([w['v_pop'] for w in weights_list]),
        'v_inj': np.std([w['v_inj'] for w in weights_list])
    }

    print(f"  [OK] 交叉验证完成 (平均R²={np.mean(r2_list):.3f}±{np.std(r2_list):.3f})")
    print(f"      v_W=${mean_weights['v_W']:,.0f}±${std_weights['v_W']:,.0f}")
    print(f"      v_pop=${mean_weights['v_pop']:,.0f}±${std_weights['v_pop']:,.0f}")
    print(f"      v_inj=${mean_weights['v_inj']:,.0f}±${std_weights['v_inj']:,.0f}")

    return mean_weights, std_weights


def get_default_weights() -> Dict[str, float]:
    """
    返回优化后的默认权重

    Returns:
        Default weights dict
    """
    return {
        'v_W': 1000000,
        'v_pop': 500000,
        'v_inj': 100000,
        'v_dec': 150000,
        'v_grow': 100000
    }


def generate_synthetic_historical_data(
    current_data: pd.DataFrame,
    n_seasons: int = 3
) -> pd.DataFrame:
    """
    生成合成历史数据 (用于演示/测试)

    基于当前数据生成过去N个赛季的模拟数据

    Args:
        current_data: 当前赛季数据
        n_seasons: 生成赛季数

    Returns:
        Synthetic historical data
    """
    historical = []

    for season in range(2025 - n_seasons, 2025):
        season_data = current_data.copy()
        season_data['season'] = season

        # 模拟实际胜场贡献 (基于E_delta_W + 噪声)
        if 'E_delta_W' in season_data.columns:
            noise = np.random.normal(0, 0.02, len(season_data))
            season_data['actual_wins_contributed'] = season_data['E_delta_W'] + noise
        else:
            season_data['actual_wins_contributed'] = np.random.uniform(0, 0.15, len(season_data))

        # 模拟实际收入贡献
        if 'E_delta_pop' in season_data.columns:
            noise = np.random.normal(0, 0.1, len(season_data))
            season_data['actual_revenue_generated'] = (
                season_data['E_delta_pop'] * 500000 + noise * 100000
            )

        historical.append(season_data)

    df = pd.concat(historical, ignore_index=True)

    print(f"  [OK] 生成合成历史数据: {len(df)}条记录, {n_seasons}个赛季")

    return df


def save_calibrated_weights(
    weights: Dict[str, float],
    output_file: str = "project/model/calibrated_weights.json"
) -> None:
    """
    保存校准后的权重到JSON文件

    Args:
        weights: 权重字典
        output_file: 输出文件路径
    """
    import json
    from datetime import datetime

    config = {
        "calibration_date": datetime.now().strftime("%Y-%m-%d"),
        "method": "historical_data_regression",
        "weights": weights,
        "description": "MV weights calibrated from historical performance data"
    }

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"  [OK] 校准权重已保存: {output_file}")


def load_calibrated_weights(
    weights_file: str = "project/model/calibrated_weights.json"
) -> Dict[str, float]:
    """
    从JSON文件加载校准后的权重

    Args:
        weights_file: 权重文件路径

    Returns:
        Weights dict or None if file doesn't exist
    """
    import json

    weights_path = Path(weights_file)

    if not weights_path.exists():
        return None

    try:
        with open(weights_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        weights = config.get('weights', {})
        print(f"  [OK] 加载校准权重: {weights_file}")
        print(f"      校准日期: {config.get('calibration_date', 'unknown')}")

        return weights

    except Exception as e:
        print(f"  [WARN] 加载权重文件失败: {e}")
        return None
