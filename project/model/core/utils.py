"""
Shared utility functions for WNBA model modules.

This module keeps common data and math helpers in one place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


# ============================================================================
# Path Resolution
# ============================================================================

def resolve_path(
    p: str | Path,
    *,
    fallback_dirs: Sequence[Path] = (),
    root_dir: Optional[Path] = None,
) -> Path:
    """
    Resolve a file path with fallback directories.

    Tries to locate the file in the following order:
    1. As-is (if absolute or relative path exists)
    2. Relative to root_dir
    3. In each fallback directory

    Parameters
    ----------
    p : str | Path
        Path to resolve
    fallback_dirs : Sequence[Path], optional
        Additional directories to search
    root_dir : Path, optional
        Root directory for relative paths (defaults to repo root)

    Returns
    -------
    Path
        Resolved path (may not exist if not found)

    Examples
    --------
    >>> resolve_path("data.csv", fallback_dirs=[Path("project/data")])
    """
    p = Path(p)

    # Try as-is
    if p.is_file():
        return p

    # Try relative to root
    if root_dir is None:
        root_dir = Path(__file__).resolve().parents[3]

    if (root_dir / p).is_file():
        return root_dir / p

    # Try fallback directories
    for d in fallback_dirs:
        if (d / p.name).is_file():
            return d / p.name

    # Return original if not found
    return p


# ============================================================================
# Data Scaling & Normalization
# ============================================================================

def min_max_scale(series: pd.Series) -> pd.Series:
    """
    Min-max normalize a pandas Series to [0, 1].

    Handles edge cases:
    - Non-numeric values (coerced to NaN)
    - All NaN (returns 0.5)
    - Constant values (returns 0.0)

    Parameters
    ----------
    series : pd.Series
        Input series

    Returns
    -------
    pd.Series
        Normalized series in [0, 1]

    Examples
    --------
    >>> min_max_scale(pd.Series([1, 2, 3, 4, 5]))
    0    0.00
    1    0.25
    2    0.50
    3    0.75
    4    1.00
    """
    s = pd.to_numeric(series, errors="coerce")

    if s.notna().sum() == 0:
        return pd.Series([0.5] * len(s), index=s.index, dtype=float)

    mn = float(s.min())
    mx = float(s.max())

    if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
        return pd.Series([0.5] * len(s), index=s.index, dtype=float)

    return ((s - mn) / (mx - mn)).clip(0.0, 1.0).astype(float)


def normalize_to_01(values: np.ndarray) -> np.ndarray:
    """
    Min-max normalize a NumPy array to [0, 1].

    Parameters
    ----------
    values : np.ndarray
        Input array

    Returns
    -------
    np.ndarray
        Normalized array in [0, 1]
    """
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return values

    mn = np.nanmin(values)
    mx = np.nanmax(values)

    if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
        return np.full_like(values, 0.5)

    return np.clip((values - mn) / (mx - mn), 0.0, 1.0)


# ============================================================================
# Elo Rating Calculations
# ============================================================================

def elo_win_probability(
    home_elo: float,
    away_elo: float,
    home_advantage: float = 0.0,
    clutch: float = 0.0,
    importance: float = 0.0,
) -> float:
    """
    Calculate home team win probability using Elo formula.

    Formula: p = 1 / (1 + 10^(-delta/400))
    where delta = home_elo - away_elo + home_advantage + clutch*importance

    Parameters
    ----------
    home_elo : float
        Home team Elo rating
    away_elo : float
        Away team Elo rating
    home_advantage : float, default=0.0
        Home court advantage in Elo points
    clutch : float, default=0.0
        Clutch factor (C in formula)
    importance : float, default=0.0
        Game importance (I in formula)

    Returns
    -------
    float
        Win probability in [0, 1]

    Examples
    --------
    >>> elo_win_probability(1600, 1500, home_advantage=24.0)
    0.6498...
    """
    delta = float(home_elo - away_elo + home_advantage + clutch * importance)
    return float(1.0 / (1.0 + 10.0 ** (-delta / 400.0)))


def update_elo_rating(
    current_elo: float,
    actual_result: float,
    predicted_prob: float,
    k: float = 20.0,
) -> float:
    """
    Update Elo rating based on game result.

    Formula: new_elo = current_elo + K * (actual - predicted)

    Parameters
    ----------
    current_elo : float
        Current Elo rating
    actual_result : float
        Actual game result (1.0 for win, 0.0 for loss)
    predicted_prob : float
        Predicted win probability
    k : float, default=20.0
        K-factor (learning rate)

    Returns
    -------
    float
        Updated Elo rating

    Examples
    --------
    >>> update_elo_rating(1500, 1.0, 0.5, k=20.0)
    1510.0
    """
    return float(current_elo + k * (actual_result - predicted_prob))


def strength_from_elo(elo: float, *, base_elo: float = 1500.0) -> float:
    """
    Convert Elo rating to strength index in [0, 1].

    Uses logistic transformation: S = 1 / (1 + 10^(-(elo - base)/400))
    At base_elo, strength = 0.5

    Parameters
    ----------
    elo : float
        Elo rating
    base_elo : float, default=1500.0
        Base Elo rating (maps to strength 0.5)

    Returns
    -------
    float
        Strength index in [0, 1]

    Examples
    --------
    >>> strength_from_elo(1500, base_elo=1500)
    0.5
    >>> strength_from_elo(1600, base_elo=1500)
    0.64...
    """
    return float(1.0 / (1.0 + 10.0 ** (-(float(elo) - float(base_elo)) / 400.0)))


# ============================================================================
# Financial Calculations
# ============================================================================

def calculate_ebitda(
    revenue: float,
    salary_cost: float,
    venue_cost: float,
    sports_ops_cost: float,
    ga_cost: float,
    tax: float = 0.0,
) -> float:
    """
    Calculate EBITDA (Earnings Before Interest, Taxes, Depreciation, Amortization).

    EBITDA = Revenue - Operating Costs (excluding tax)

    Parameters
    ----------
    revenue : float
        Total revenue
    salary_cost : float
        Player salary costs
    venue_cost : float
        Venue operational costs
    sports_ops_cost : float
        Sports operations costs
    ga_cost : float
        General & administrative costs (including marketing)
    tax : float, default=0.0
        Tax amount (excluded from EBITDA)

    Returns
    -------
    float
        EBITDA

    Examples
    --------
    >>> calculate_ebitda(10_000_000, 3_000_000, 1_000_000, 500_000, 1_500_000, 500_000)
    4000000.0
    """
    ga_ex_tax = float(ga_cost) - float(tax)
    return float(
        revenue
        - salary_cost
        - venue_cost
        - sports_ops_cost
        - ga_ex_tax
    )


def calculate_ebitda_from_dict(
    revenue: float,
    costs: Dict[str, float],
) -> float:
    """
    Calculate EBITDA from cost dictionary.

    Convenience wrapper for calculate_ebitda() that accepts a cost dict.

    Parameters
    ----------
    revenue : float
        Total revenue
    costs : Dict[str, float]
        Cost breakdown with keys: salary_cost, venue_cost, sports_ops_cost,
        ga_cost, tax

    Returns
    -------
    float
        EBITDA
    """
    return calculate_ebitda(
        revenue=float(revenue),
        salary_cost=float(costs.get("salary_cost", 0.0)),
        venue_cost=float(costs.get("venue_cost", 0.0)),
        sports_ops_cost=float(costs.get("sports_ops_cost", 0.0)),
        ga_cost=float(costs.get("ga_cost", 0.0)),
        tax=float(costs.get("tax", 0.0)),
    )


# ============================================================================
# Data Loading Helpers
# ============================================================================

def load_csv_with_fallback(
    path: str | Path,
    *,
    fallback_dirs: Sequence[Path] = (),
    root_dir: Optional[Path] = None,
    required_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Load CSV with path resolution and column validation.

    Parameters
    ----------
    path : str | Path
        CSV file path
    fallback_dirs : Sequence[Path], optional
        Fallback directories to search
    root_dir : Path, optional
        Root directory for relative paths
    required_columns : Sequence[str], optional
        Required column names (raises ValueError if missing)

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame

    Raises
    ------
    FileNotFoundError
        If file not found in any location
    ValueError
        If required columns are missing
    """
    resolved_path = resolve_path(path, fallback_dirs=fallback_dirs, root_dir=root_dir)

    if not resolved_path.is_file():
        tried = [str(path)]
        if root_dir:
            tried.append(str(root_dir / path))
        for d in fallback_dirs:
            tried.append(str(d / Path(path).name))
        raise FileNotFoundError(f"CSV not found. Tried: {', '.join(tried)}")

    # Opening the file explicitly avoids pandas path handling issues on some
    # Windows setups when the project path contains non-ASCII characters.
    with resolved_path.open("r", encoding="utf-8-sig", newline="") as f:
        df = pd.read_csv(f)

    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    return df


# ============================================================================
# Numeric Utilities
# ============================================================================

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.

    Parameters
    ----------
    numerator : float
        Numerator
    denominator : float
        Denominator
    default : float, default=0.0
        Value to return if denominator is zero

    Returns
    -------
    float
        Result of division or default
    """
    if denominator == 0.0 or not np.isfinite(denominator):
        return float(default)
    return float(numerator / denominator)


def clip_to_range(value: float, min_val: float, max_val: float) -> float:
    """
    Clip value to [min_val, max_val] range.

    Parameters
    ----------
    value : float
        Value to clip
    min_val : float
        Minimum value
    max_val : float
        Maximum value

    Returns
    -------
    float
        Clipped value
    """
    return float(max(min_val, min(max_val, value)))
