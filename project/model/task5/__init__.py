"""Task 5 injury-risk utilities."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    # Network model
    "PlayerNetwork": "network_model",
    "calculate_flow_centrality": "network_model",
    "calculate_network_entropy": "network_model",
    "simulate_network_disruption": "network_model",
    "create_network_from_data": "network_model",
    "analyze_key_players": "network_model",
    # Injury model
    "InjuryShockParameters": "injury_model",
    "calculate_fatigue_index": "injury_model",
    "calculate_injury_probability": "injury_model",
    "simulate_injury_shock": "injury_model",
    "estimate_recovery_time": "injury_model",
    "calculate_injury_risk_distribution": "injury_model",
    "create_fatigue_warning_thresholds": "injury_model",
    # Markov volatility
    "MarkovRegimeModel": "markov_volatility",
    "fit_regime_model": "markov_volatility",
    "predict_regime_probabilities": "markov_volatility",
    "simulate_regime_path": "markov_volatility",
    "compare_regimes": "markov_volatility",
    "create_default_model": "markov_volatility",
    # Impact analysis
    "InjuryImpactAnalyzer": "impact_analysis",
    "quantify_win_loss": "impact_analysis",
    "quantify_revenue_loss": "impact_analysis",
    "quantify_playoff_impact": "impact_analysis",
    "quantify_valuation_impact": "impact_analysis",
    "simulate_injury_scenario": "impact_analysis",
    "create_impact_waterfall": "impact_analysis",
    # Mitigation strategies
    "MitigationStrategy": "mitigation_strategies",
    "evaluate_load_management": "mitigation_strategies",
    "evaluate_medical_investment": "mitigation_strategies",
    "evaluate_network_repair_signing": "mitigation_strategies",
    "create_fatigue_warning_system": "mitigation_strategies",
    "create_mitigation_dashboard": "mitigation_strategies",
    # Data utilities
    "RealDataLoader": "data_utils",
    "DataValidator": "data_utils",
    "ParameterCalibrator": "data_utils",
    "DataTransformer": "data_utils",
    "quick_load_real_data": "data_utils",
    "validate_and_report": "data_utils",
    "load_existing_project_data": "data_utils",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value
