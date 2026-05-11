"""
Core models for WNBA simulation.

This package contains the fundamental building blocks:
- elo: Elo rating system
- brands: Brand value model
- income: Revenue model
- cost: Cost model
- fatigue: Fatigue and injury risk model
- players: Player Commercial Value (PCV) model
- config: Unified system configuration
- utils: Shared utility functions
- data_loader: Data loading utilities
- data_paths: Centralized path configuration
"""

from .elo import EloConfig, EloModel
from .brands import BrandParameters, update_brand, TEAM_NAME_TO_CODE, FRANCHISE_ALIASES
from .income import DemandParameters, RevenueParameters, calculate_period_revenue
from .cost import (
    SalaryCostParameters,
    VenueCostParameters,
    GeneralAdminParameters,
    SportsOperationsParameters,
    FinancingParameters,
    CashFlowParameters,
    calculate_total_cost,
)
from .fatigue import FatigueParameters, InjuryRiskParameters
from .players import estimate_pcv, load_salaries, load_players_fans
from .config import SystemConfig
from .utils import (
    resolve_path,
    load_csv_with_fallback,
    elo_win_probability,
    update_elo_rating,
    min_max_scale,
)
from .data_loader import DataLoader, load_elo_ratings, load_elo_config, load_brand_b0
from .data_paths import (
    ROOT_DIR,
    DATA_DIR,
    RAW_DIR,
    EXTERNAL_DIR,
    PROCESSED_DIR,
    CONFIG_DIR,
    ELO_DIR,
    SIMULATION_DIR,
    MPC_DIR,
    FINANCIAL_DIR,
    PLAYERS_DIR,
    RawFiles,
    ConfigFiles,
    EloFiles,
)

__all__ = [
    # Elo
    "EloConfig",
    "EloModel",
    "elo_win_probability",
    # Brands
    "BrandParameters",
    "update_brand",
    "TEAM_NAME_TO_CODE",
    "FRANCHISE_ALIASES",
    # Income
    "DemandParameters",
    "RevenueParameters",
    "calculate_period_revenue",
    # Cost
    "SalaryCostParameters",
    "VenueCostParameters",
    "GeneralAdminParameters",
    "SportsOperationsParameters",
    "FinancingParameters",
    "CashFlowParameters",
    "calculate_total_cost",
    # Fatigue
    "FatigueParameters",
    "InjuryRiskParameters",
    # Players
    "estimate_pcv",
    "load_salaries",
    "load_players_fans",
    # Config
    "SystemConfig",
    # Utils
    "resolve_path",
    "load_csv_with_fallback",
    "update_elo_rating",
    "min_max_scale",
    # Data Loader
    "DataLoader",
    "load_elo_ratings",
    "load_elo_config",
    "load_brand_b0",
    # Paths
    "ROOT_DIR",
    "DATA_DIR",
    "RAW_DIR",
    "PROCESSED_DIR",
    "CONFIG_DIR",
    "ELO_DIR",
]
