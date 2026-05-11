"""
Task2: 球员获取策略优化模块

功能:
- 球员属性降维 (竞技/商业/风险/成本)
- 表现预测 (增长/下降)
- 边际价值计算 (MV)
- MILP优化 (工资帽/位置约束)
- 赛季状态初始化
- 历史数据校准
- 可视化分析

使用:
    # 方式1: 直接运行模块
    python -m project.model.task2.pipeline

    # 方式2: 在代码中调用
    from project.model.task2.pipeline import run_task2_pipeline

    result = run_task2_pipeline(
        horizon=3,
        salary_cap=1500000,
        roster_size=12,
        team_filter="Phoenix Mercury"
    )
"""

# 延迟导入，避免 python -m 运行时的循环导入警告
def __getattr__(name):
    if name == 'run_task2_pipeline':
        from .pipeline import run_task2_pipeline
        return run_task2_pipeline
    elif name == 'load_player_base_data':
        from .attributes import load_player_base_data
        return load_player_base_data
    elif name == 'add_player_attributes':
        from .attributes import add_player_attributes
        return add_player_attributes
    elif name == 'forecast_performance':
        from .forecast import forecast_performance
        return forecast_performance
    elif name == 'check_growth_constraints':
        from .forecast import check_growth_constraints
        return check_growth_constraints
    elif name == 'compute_marginal_value':
        from .marginal_value import compute_marginal_value
        return compute_marginal_value
    elif name == 'optimize_roster':
        from .optimizer import optimize_roster
        return optimize_roster
    elif name == 'initialize_season_state':
        from .initializer import initialize_season_state
        return initialize_season_state
    elif name == 'calibrate_weights_from_history':
        from .calibration import calibrate_weights_from_history
        return calibrate_weights_from_history
    elif name == 'cross_validate_weights':
        from .calibration import cross_validate_weights
        return cross_validate_weights
    elif name == 'load_calibrated_weights':
        from .calibration import load_calibrated_weights
        return load_calibrated_weights
    elif name == 'save_calibrated_weights':
        from .calibration import save_calibrated_weights
        return save_calibrated_weights
    elif name == 'generate_synthetic_historical_data':
        from .calibration import generate_synthetic_historical_data
        return generate_synthetic_historical_data
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'run_task2_pipeline',
    'load_player_base_data',
    'add_player_attributes',
    'forecast_performance',
    'check_growth_constraints',
    'compute_marginal_value',
    'optimize_roster',
    'initialize_season_state',
    'calibrate_weights_from_history',
    'cross_validate_weights',
    'load_calibrated_weights',
    'save_calibrated_weights',
    'generate_synthetic_historical_data',
]
