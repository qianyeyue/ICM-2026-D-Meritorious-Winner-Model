"""
Task 3: League Expansion Impact Analysis (WNBA)

Quantifies expansion effects through three channels:
1. Revenue sharing: Div_t 鈭?1/N_t
2. Schedule/travel: Fatigue ->Win rate/Injury ->Profit/Risk
3. Market competition: Distance decay model for local market competition
"""

from .expansion_model import ExpansionModel, ExpansionConfig
from .data_loader import load_wnba_expansion_data
from .scenarios import calculate_expansion_scenarios

__all__ = [
    "ExpansionModel",
    "ExpansionConfig",
    "load_wnba_expansion_data",
    "calculate_expansion_scenarios",
    "build_task1_exogenous_shock",
    "export_task1_exogenous_shock",
]


def __getattr__(name: str):
    if name in {"build_task1_exogenous_shock", "export_task1_exogenous_shock"}:
        from . import task1_impact

        return getattr(task1_impact, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

