"""
Centralized data path configuration for the WNBA simulation project.

This module defines all data paths in one place, making it easy to:
1. Understand the data folder structure
2. Update paths when reorganizing files
3. Maintain consistency across all modules

Data Folder Structure:
    project/data/
    ├── raw/                    # 原始输入数据 (Raw Input)
    │   ├── wnba_gamelogs_2015_2025.csv
    │   ├── wnba_salaries.csv
    │   ├── players_fans.csv
    │   ├── teams_fans.csv
    │   ├── years.csv
    │   └── multiTimeline.csv
    │
    ├── external/               # 外部数据源 (External Sources)
    │   ├── media_deals.csv
    │   ├── expansion_teams.csv
    │   ├── team_valuations.csv
    │   └── 联盟搜索数据.csv
    │
    ├── processed/              # 处理后数据 (Processed)
    │   ├── config/             # 模型配置参数
    │   │   ├── elo_config.csv
    │   │   ├── brand_weights.csv
    │   │   └── brand_b0.csv
    │   │
    │   ├── clean/              # 清洗后数据
    │   │   ├── wnba_gamelogs_clean.csv
    │   │   ├── salaries_clean.csv
    │   │   └── attendance_clean.csv
    │   │
    │   ├── elo/                # Elo模型输出
    │   │   ├── elo_final_ratings.csv
    │   │   ├── elo_game_history.csv
    │   │   └── elo_team_season_summary.csv
    │   │
    │   ├── simulation/         # 仿真结果
    │   │   ├── 2026_schedule.csv
    │   │   ├── 2026_all_simulations.csv
    │   │   └── 2026_*.csv/png
    │   │
    │   ├── mpc/                # MPC优化输出
    │   │   ├── mpc_summary_*.csv
    │   │   └── mpc_period_log_*.csv
    │   │
    │   ├── financial/          # 财务分析结果
    │   │   ├── financial_summary_with_breakdown.csv
    │   │   └── team_financial_summary_*.csv
    │   │
    │   └── players/            # 球员分析结果
    │       └── players_pcv.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

# =============================================================================
# Root directories
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT_DIR / "project" / "data"

# =============================================================================
# Input data directories
# =============================================================================

# Raw input data (original source files)
RAW_DIR = DATA_DIR / "raw"

# External data sources (manually collected)
EXTERNAL_DIR = DATA_DIR / "external"

# =============================================================================
# Processed data directories  
# =============================================================================

PROCESSED_DIR = DATA_DIR / "processed"

# Configuration files (model parameters)
CONFIG_DIR = PROCESSED_DIR / "config"

# Cleaned data
CLEAN_DIR = PROCESSED_DIR / "clean"
OTHER_CLEAN_DIR = PROCESSED_DIR / "other_clean"

# Model outputs
ELO_DIR = PROCESSED_DIR / "elo"
SIMULATION_DIR = PROCESSED_DIR / "simulation"
MPC_DIR = PROCESSED_DIR / "mpc"
FINANCIAL_DIR = PROCESSED_DIR / "financial"
PLAYERS_DIR = PROCESSED_DIR / "players"

# Visualization outputs
ELO_VIZ_DIR = PROCESSED_DIR / "elo_viz"
PLAYERS_VIZ_DIR = PROCESSED_DIR / "players_viz"

# Task-specific outputs
TASK2_DIR = PROCESSED_DIR / "task2"
TASK2_CHECK_DIR = PROCESSED_DIR / "task2_check"
INTEGRATION_TEST_DIR = PROCESSED_DIR / "integration_test"

# =============================================================================
# Raw input files
# =============================================================================

class RawFiles:
    """Paths to raw input files."""
    GAMELOGS = RAW_DIR / "wnba_gamelogs_2015_2025.csv"
    SALARIES = RAW_DIR / "wnba_salaries.csv"
    PLAYERS_FANS = RAW_DIR / "players_fans.csv"
    TEAMS_FANS = RAW_DIR / "teams_fans.csv"
    YEARS = RAW_DIR / "years.csv"
    MULTI_TIMELINE = RAW_DIR / "multiTimeline.csv"
    ATTENDANCE_NORMALIZED = RAW_DIR / "attendance_normalized.csv"
    SALARIES_NORMALIZED = RAW_DIR / "salaries_normalized.csv"


class ExternalFiles:
    """Paths to external data files."""
    MEDIA_DEALS = EXTERNAL_DIR / "media_deals.csv"
    EXPANSION_TEAMS = EXTERNAL_DIR / "expansion_teams.csv"
    TEAM_VALUATIONS = EXTERNAL_DIR / "team_valuations.csv"
    LEAGUE_SEARCH = EXTERNAL_DIR / "联盟搜索数据.csv"


class ConfigFiles:
    """Paths to configuration files."""
    ELO_CONFIG = CONFIG_DIR / "elo_config.csv"
    BRAND_WEIGHTS = CONFIG_DIR / "brand_weights.csv"
    BRAND_B0 = CONFIG_DIR / "brand_b0.csv"


class EloFiles:
    """Paths to Elo model output files."""
    FINAL_RATINGS = ELO_DIR / "elo_final_ratings.csv"
    FINAL_RATINGS_ADJUSTED = ELO_DIR / "elo_final_ratings_adjusted.csv"
    GAME_HISTORY = ELO_DIR / "elo_game_history.csv"
    TEAM_SEASON_SUMMARY = ELO_DIR / "elo_team_season_summary.csv"


class SimulationFiles:
    """Paths to simulation output files."""
    SCHEDULE_2026 = SIMULATION_DIR / "2026_schedule.csv"
    ALL_SIMULATIONS_2026 = SIMULATION_DIR / "2026_all_simulations.csv"
    GAME_PROBABILITIES_2026 = SIMULATION_DIR / "2026_game_probabilities.csv"
    TEAM_STATISTICS_2026 = SIMULATION_DIR / "2026_team_statistics.csv"
    VALIDATION_METRICS_2026 = SIMULATION_DIR / "2026_validation_metrics.csv"


class CleanFiles:
    """Paths to cleaned data files."""
    GAMELOGS_CLEAN = CLEAN_DIR / "wnba_gamelogs_clean.csv"
    GAMELOGS_FEATURES = CLEAN_DIR / "wnba_gamelogs_features.csv"
    SALARIES_CLEAN = CLEAN_DIR / "salaries_clean.csv"
    SALARIES_CORE = CLEAN_DIR / "salaries_core.csv"
    ATTENDANCE_CLEAN = CLEAN_DIR / "attendance_clean.csv"
    MATCHUP_RESULTS = CLEAN_DIR / "wnba_matchup_results.csv"
    BOX_SCORES_CLEAN = CLEAN_DIR / "box_scores_clean.csv"


class PlayersFiles:
    """Paths to player analysis files."""
    PCV = PLAYERS_DIR / "players_pcv.csv"
    AGES = PROCESSED_DIR / "wnba_player_ages_2025.csv"
    CLEANED_2025 = PROCESSED_DIR / "wnba_players_2025_cleaned.csv"


class FinancialFiles:
    """Paths to financial analysis files."""
    SUMMARY_BREAKDOWN = FINANCIAL_DIR / "financial_summary_with_breakdown.csv"
    TEAM_SUMMARY_FORMATTED = FINANCIAL_DIR / "team_financial_summary_formatted.csv"
    TEAM_SUMMARY_LONG = FINANCIAL_DIR / "team_financial_summary_long_format.csv"


# =============================================================================
# Legacy path mappings (for backward compatibility)
# =============================================================================

# Map old paths to new paths for gradual migration
LEGACY_PATH_MAPPING: Dict[str, Path] = {
    # Raw files that were in data/ root
    "wnba_gamelogs_2015_2025.csv": RawFiles.GAMELOGS,
    "wnba_salaries.csv": RawFiles.SALARIES,
    "players_fans.csv": RawFiles.PLAYERS_FANS,
    "teams_fans.csv": RawFiles.TEAMS_FANS,
    "years.csv": RawFiles.YEARS,
    "multiTimeline.csv": RawFiles.MULTI_TIMELINE,
    "attendance_normalized.csv": RawFiles.ATTENDANCE_NORMALIZED,
    "salaries_normalized.csv": RawFiles.SALARIES_NORMALIZED,
    
    # External files with Chinese names
    "媒体版权合同数据：media_deals.csv": ExternalFiles.MEDIA_DEALS,
    "扩张球队数据：expansion_teams.csv": ExternalFiles.EXPANSION_TEAMS,
    "球队估值与运营数据：team_valuations.csv": ExternalFiles.TEAM_VALUATIONS,
    "联盟搜索数据.csv": ExternalFiles.LEAGUE_SEARCH,
    
    # Config files from processed/ root
    "elo_config.csv": ConfigFiles.ELO_CONFIG,
    "brand_weights.csv": ConfigFiles.BRAND_WEIGHTS,
    "brand_b0.csv": ConfigFiles.BRAND_B0,
    
    # Elo files from processed/ root
    "elo_final_ratings.csv": EloFiles.FINAL_RATINGS,
    "elo_game_history.csv": EloFiles.GAME_HISTORY,
    "elo_team_season_summary.csv": EloFiles.TEAM_SEASON_SUMMARY,
    
    # Players files
    "players_pcv.csv": PlayersFiles.PCV,
}


def resolve_data_path(filename: str, *, check_legacy: bool = True) -> Path:
    """
    Resolve a data file path, checking both new and legacy locations.
    
    Args:
        filename: The filename or relative path to resolve
        check_legacy: Whether to check legacy locations if new path doesn't exist
        
    Returns:
        The resolved Path object
    """
    # Check if it's in the legacy mapping
    if filename in LEGACY_PATH_MAPPING:
        new_path = LEGACY_PATH_MAPPING[filename]
        if new_path.is_file():
            return new_path
    
    # Try the filename directly
    direct_path = Path(filename)
    if direct_path.is_file():
        return direct_path
    
    # Try in DATA_DIR
    data_path = DATA_DIR / filename
    if data_path.is_file():
        return data_path
    
    # Try in PROCESSED_DIR
    processed_path = PROCESSED_DIR / filename
    if processed_path.is_file():
        return processed_path
    
    # Try in RAW_DIR
    raw_path = RAW_DIR / filename
    if raw_path.is_file():
        return raw_path
    
    # Return the expected new path (for creation)
    if filename in LEGACY_PATH_MAPPING:
        return LEGACY_PATH_MAPPING[filename]
    
    return data_path


def ensure_directories() -> None:
    """Create all data directories if they don't exist."""
    directories = [
        RAW_DIR,
        EXTERNAL_DIR,
        CONFIG_DIR,
        CLEAN_DIR,
        OTHER_CLEAN_DIR,
        ELO_DIR,
        SIMULATION_DIR,
        MPC_DIR,
        FINANCIAL_DIR,
        PLAYERS_DIR,
        ELO_VIZ_DIR,
        PLAYERS_VIZ_DIR,
        TASK2_DIR,
        INTEGRATION_TEST_DIR,
    ]
    for d in directories:
        d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Convenience functions for common operations
# =============================================================================

def get_gamelogs_path() -> Path:
    """Get path to WNBA gamelogs CSV (tries new then legacy location)."""
    if RawFiles.GAMELOGS.is_file():
        return RawFiles.GAMELOGS
    legacy = DATA_DIR / "wnba_gamelogs_2015_2025.csv"
    if legacy.is_file():
        return legacy
    return RawFiles.GAMELOGS


def get_salaries_path() -> Path:
    """Get path to WNBA salaries CSV (tries new then legacy location)."""
    if RawFiles.SALARIES.is_file():
        return RawFiles.SALARIES
    legacy = DATA_DIR / "wnba_salaries.csv"
    if legacy.is_file():
        return legacy
    return RawFiles.SALARIES


def get_elo_ratings_path() -> Path:
    """Get path to Elo final ratings CSV (tries new then legacy location)."""
    if EloFiles.FINAL_RATINGS.is_file():
        return EloFiles.FINAL_RATINGS
    legacy = PROCESSED_DIR / "elo_final_ratings.csv"
    if legacy.is_file():
        return legacy
    return EloFiles.FINAL_RATINGS


def get_elo_config_path() -> Path:
    """Get path to Elo config CSV (tries new then legacy location)."""
    if ConfigFiles.ELO_CONFIG.is_file():
        return ConfigFiles.ELO_CONFIG
    legacy = PROCESSED_DIR / "elo_config.csv"
    if legacy.is_file():
        return legacy
    return ConfigFiles.ELO_CONFIG


if __name__ == "__main__":
    # Print directory structure for reference
    print("=== Data Directory Structure ===\n")
    print(f"ROOT_DIR: {ROOT_DIR}")
    print(f"DATA_DIR: {DATA_DIR}")
    print()
    print("Input Directories:")
    print(f"  RAW_DIR: {RAW_DIR}")
    print(f"  EXTERNAL_DIR: {EXTERNAL_DIR}")
    print()
    print("Processed Directories:")
    print(f"  CONFIG_DIR: {CONFIG_DIR}")
    print(f"  CLEAN_DIR: {CLEAN_DIR}")
    print(f"  ELO_DIR: {ELO_DIR}")
    print(f"  SIMULATION_DIR: {SIMULATION_DIR}")
    print(f"  MPC_DIR: {MPC_DIR}")
    print(f"  FINANCIAL_DIR: {FINANCIAL_DIR}")
    print(f"  PLAYERS_DIR: {PLAYERS_DIR}")
