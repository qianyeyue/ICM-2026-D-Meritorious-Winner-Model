"""
Revenue Model for WNBA Teams.

This module implements the revenue terms used by the model:
1. Demand estimation (log-linear model)
2. Ticket revenue
3. Merchandise/food revenue
4. Sponsorship revenue
5. League dividend

Mathematical Model
------------------
Demand (for home game g in period t):
    ln(Dem_g) = beta_0 - epsilon * ln(tau_t) + beta_S * S_t
                + beta_star * Star_t + beta_A * A_g + beta_cal * Z_g + error_g
    where Dem_g <= Cap_g (capacity constraint)

Game Attractiveness:
    A_g = weighted opponent strength, star power, rivalry, and calendar effects

Revenue Components:
    - Ticket: demand times ticket price for home games
    - Merchandise: attendance and star-power terms
    - Sponsorship: baseline, brand, and star-power terms
    - League dividend: league revenue share divided by team count

Total Revenue:
    Rev_t = Rev^ticket_t + Rev^merch_t + Rev^spon_t + Div_t

Notes:
- Only home games generate ticket/merchandise revenue
- Regular season stages have demand multipliers (e.g., 1.1x, 1.2x for later stages)
- Venue is assumed self-operated (no rental costs, only operational costs)
- Ticket revenue formula: capacity * attendance_rate * price = demand * price
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from typing import NamedTuple


@dataclass
class DemandParameters:
    """
    Parameters for the demand model.

    Calibrated based on WNBA actual data:
    - 2023: avg ticket price $53, avg attendance 9,091
    - 2024: avg ticket price $123, avg attendance 10,761
    - 2024: max attendance 18,209

    Calibration approach (updated 2026-02-01):
    - beta_0 = 10.77 calibrated using 2024 WNBA data from 7 uncapped teams
    - epsilon = 0.50 (moderate price elasticity, theoretical value)
    - beta_star = 0.50 captures star effect (Caitlin Clark phenomenon: 1.59x boost)
    - Calibration method: Direct solve from ln(Dem) = 尾鈧€ - 蔚路ln(p) + 尾_S路S + 尾_star路Star + A_g
    
    Calibration details (see project/model/analysis/calibrate_beta0.py):
    - Used 7 teams without capacity constraints: IND, NYL, LAS, SEA, PHO, CHI, MIN
    - 尾鈧€ adjusted to match LVA actual attendance (10,761) at stage_mult=1.0
    - This allows MPC to have pricing strategy space in early season periods
    
    Note: Some teams (ATL, WAS, DAL, CON) have small arenas that cap attendance,
    causing model underprediction. This is expected behavior.
    """

    # Base parameters (calibrated from 2024 WNBA attendance data)
    beta_0: float = 10.61  # Intercept (log-scale) - calibrated to match LVA actual
    epsilon: float = 0.50  # Price elasticity - moderate sensitivity to pricing
    beta_S: float = 0.20  # Team strength coefficient
    beta_star: float = 0.50  # Star power coefficient (Caitlin Clark effect: 1.59x boost)
    beta_cal: float = 0.0  # Calendar/weather effects (not used in task 1)

    # Attractiveness sub-model coefficients
    a1: float = 0.10  # Opponent strength
    a2: float = 0.20  # Opponent star power
    a3: float = 0.35  # Rivalry factor
    a4: float = 0.25  # Game importance

    # Noise/error term standard deviation
    sigma_eta: float = 0.25  # Reflects high variance in attendance across teams


@dataclass
class RevenueParameters:
    """
    Parameters for revenue components.

    Calibrated based on WNBA actual data:
    - 2023: avg ticket price $53, Las Vegas Aces revenue $17.8M
    - 2024: avg ticket price $123 (132% increase), Las Vegas Aces revenue $22M (+23.6%)
    - 2024: avg attendance 10,761 (18.4% increase despite price hike)
    - Las Vegas Aces: 2024 valuation $140M ->2025 valuation $310M (+121%)
    - Arena capacity: 8,600 (Las Vegas Aces)

    Revenue calibration based on Las Vegas Aces (championship team):
    - Assuming 20 home games, avg attendance ~8,000, ticket price ~$137
    - Ticket revenue: $137 脳 8,000 脳 20 = $21.92M 鈮?actual $22M [OK]    """

    # Ticket pricing (using 2024 as baseline)
    p0: float = 123.0  # Base ticket price ($) - 2024 league average

    # Merchandise/food revenue (calibrated to match ~$22M total for top teams)
    kappa_1: float = 25.0  # Per-attendee spending ($) - increased based on actual revenue
    kappa_2: float = 150000.0  # Star power bonus ($) - championship team premium

    # Sponsorship revenue (calibrated to match actual team revenues)
    delta_0: float = 1200000.0  # Base sponsorship ($) - increased to match $17.8M-$22M range
    delta_B: float = 200000.0  # Brand value coefficient ($)
    delta_star: float = 150000.0  # Star power coefficient ($) - championship premium

    # League dividend (media rights distribution)
    theta_rev: float = 0.05  # Share of league revenue distributed (reduced from 0.40 to achieve ~5% of team revenue)


@dataclass
class LeaguePopularityParameters:
    """
    Parameters for league popularity and revenue model.
    
    Exponential growth model:
        L_t = L_0 * x^t

    where x is the annual growth factor.
    
    League popularity L_t affects total league revenue through:
        LeagueRev(L_t) = LeagueRev_0 * (1 + gamma * (L_t - 1))
    
    Dividend distribution:
        Div_t = revenue_share * LeagueRev(L_t) / N_t
    
    Data sources for x estimation:
    - WNBA revenue growth: $60M(2022)->200M(2024) ->~82% CAGR
    - Attendance growth: 2015-2024 average ~8%/yr
    - Social media growth: ~20%/yr
    - Conservative estimate: x ~= 1.10 (10% annual growth)
    """
    
    # ========== Base League Revenue ==========
    league_rev_base: float = 200_000_000.0  # $200M baseline (2024)
    
    # Popularity elasticity (how much revenue changes per unit popularity)
    # Back-solved from 2022 and 2024 revenue/attendance.
    # 2022: Rev=$60M, Att=6490 -> L_2022 = 6490/10761 = 0.603
    # 2024: Rev=$200M, Att=10761 -> L_2024 = 1.0
    # gamma = (60/200 - 1) / (0.603 - 1) = 1.76
    gamma: float = 1.76  # Data-driven estimate from WNBA 2022-2024
    
    # ========== Exponential Growth Parameters ==========
    # L_t = L_0 路 x^t
    L_0: float = 1.0    # Base popularity in 2024 (normalized to 1.0)
    x: float = 1.10     # Annual growth factor (10% per year)
    base_year: int = 2024  # Reference year for t=0
    
    # ========== Legacy Weights (kept for compatibility, not used in new model) ==========
    w_star: float = 0.21
    w_competition: float = 0.52
    w_media: float = 0.12
    w_social: float = 0.15
    star_baseline: float = 12.0
    competition_baseline: float = 0.82
    media_baseline: float = 60.0
    social_baseline: float = 21.8
    star_growth_rate: float = 0.10
    media_growth_rate: float = 0.82
    social_growth_rate: float = 0.20


@dataclass
class SeasonStageMultipliers:
    """Demand multipliers for different regular season stages."""

    early: float = 1.0  # Early season (baseline)
    mid: float = 1.1  # Mid season
    late: float = 1.2  # Late season / playoff push


@dataclass
class SeasonStagePriceMultipliers:
    """Ticket price multipliers for different regular season stages."""

    early: float = 1.0  # Early season (baseline price)
    mid: float = 1.1  # Mid season (10% increase)
    late: float = 1.2  # Late season / playoff push (20% increase)


def calculate_game_attractiveness(
    opponent_strength: float,
    opponent_star_power: float,
    rivalry_factor: float,
    game_importance: float,
    params: DemandParameters,
) -> float:
    """
    Calculate game attractiveness index A_g.

    A_g = a鈧伮稴_g^opp + a鈧偮稴tar_g^opp + a鈧兟稲ivalry_g + a鈧劼稩_g

    Parameters
    ----------
    opponent_strength : float
        Opponent team strength (e.g., Elo rating normalized)
    opponent_star_power : float
        Opponent star power index
    rivalry_factor : float
        Rivalry indicator (0-1, or binary)
    game_importance : float
        Game importance (0-1, higher for playoff implications)
    params : DemandParameters
        Model parameters

    Returns
    -------
    float
        Game attractiveness index
    """
    A_g = (
        params.a1 * opponent_strength
        + params.a2 * opponent_star_power
        + params.a3 * rivalry_factor
        + params.a4 * game_importance
    )
    return A_g


def calculate_demand(
    ticket_price: float,
    team_strength: float,
    team_star_power: float,
    game_attractiveness: float,
    capacity: float,
    stage_multiplier: float = 1.0,
    calendar_effects: float = 0.0,
    params: DemandParameters = None,
    add_noise: bool = False,
    random_state: Optional[int] = None,
) -> float:
    """
    Calculate demand for a single home game using log-linear model.

    ln(Dem_g) = 尾鈧€ - 蔚路ln(蟿_t) + 尾_S路S_t + 尾_star路Star_t + 尾_A路A_g + 尾_cal路Z_g + 畏_g

    Constraint: Dem_g 鈮?Cap_g

    Parameters
    ----------
    ticket_price : float
        Ticket price 蟿_t (must be > 0)
    team_strength : float
        Team strength S_t (e.g., normalized Elo or win rate)
    team_star_power : float
        Team star power index
    game_attractiveness : float
        Game attractiveness A_g (from calculate_game_attractiveness)
    capacity : float
        Venue capacity Cap_g
    stage_multiplier : float, default=1.0
        Season stage multiplier (1.0 early, 1.2 mid, 1.4 late)
    calendar_effects : float, default=0.0
        Calendar/weather effects Z_g (not used in task 1)
    params : DemandParameters, optional
        Model parameters (uses defaults if None)
    add_noise : bool, default=False
        Whether to add random noise 畏_g
    random_state : int, optional
        Random seed for noise generation

    Returns
    -------
    float
        Estimated demand (attendance), capped at capacity
    """
    if params is None:
        params = DemandParameters()

    if ticket_price <= 0:
        raise ValueError("Ticket price must be positive")

    # Log-linear demand model
    log_demand = (
        params.beta_0
        - params.epsilon * np.log(ticket_price)
        + params.beta_S * team_strength
        + params.beta_star * team_star_power
        + game_attractiveness  # 尾_A is implicitly 1.0 in the attractiveness calculation
        + params.beta_cal * calendar_effects
    )

    # Add noise if requested
    if add_noise:
        rng = np.random.RandomState(random_state)
        eta = rng.normal(0, params.sigma_eta)
        log_demand += eta

    # Apply stage multiplier (in log space: ln(multiplier * Dem) = ln(multiplier) + ln(Dem))
    log_demand += np.log(stage_multiplier)

    # Convert to actual demand
    demand = np.exp(log_demand)

    # Apply capacity constraint
    demand = min(demand, capacity)

    return demand


def calculate_ticket_revenue(
    demands: List[float],
    ticket_price: float,
    base_price: float,
    is_home: List[bool],
    capacities: Optional[List[float]] = None,
) -> float:
    """
    Calculate ticket revenue for a period.

    Formula: ticket revenue = sum(demand_g * tau_t * base_price).

    Where:
    - Cap_g: venue capacity for game g
    - Dem_g / Cap_g: attendance rate
    - tau_t * base_price: ticket price

    Note: This simplifies to the same formula as before, but conceptually
    represents: capacity * attendance_rate * price

    Parameters
    ----------
    demands : List[float]
        List of demand (attendance) for each game
    ticket_price : float
        Ticket price multiplier 蟿_t (relative to base)
    base_price : float
        Base ticket price p鈧€
    is_home : List[bool]
        Boolean list indicating which games are home games
    capacities : List[float], optional
        List of venue capacities for each game (not used in calculation,
        kept for API compatibility)

    Returns
    -------
    float
        Total ticket revenue
    """
    # Note: capacities parameter is kept for API compatibility but not used
    # because the formula simplifies: Cap 脳 (Dem/Cap) 脳 Price = Dem 脳 Price
    _ = capacities  # Mark as unused

    revenue = 0.0
    for dem, home in zip(demands, is_home):
        if home:
            # Formula: capacity 脳 attendance_rate 脳 price
            # = capacity 脳 (demand / capacity) 脳 price
            # = demand 脳 price
            revenue += ticket_price * base_price * dem
    return revenue


def calculate_merchandise_revenue(
    demands: List[float],
    team_star_power: float,
    is_home: List[bool],
    params: RevenueParameters = None,
) -> float:
    """
    Calculate merchandise/food revenue for a period.

    Rev^merch_t = 魏鈧伮肺?Dem_g) + 魏鈧偮稴tar_t (sum over home games only)

    Parameters
    ----------
    demands : List[float]
        List of demand (attendance) for each game
    team_star_power : float
        Team star power index
    is_home : List[bool]
        Boolean list indicating which games are home games
    params : RevenueParameters, optional
        Revenue parameters (uses defaults if None)

    Returns
    -------
    float
        Total merchandise/food revenue
    """
    if params is None:
        params = RevenueParameters()

    # Per-attendee spending (home games only)
    total_attendance = sum(dem for dem, home in zip(demands, is_home) if home)
    revenue = params.kappa_1 * total_attendance + params.kappa_2 * team_star_power

    return revenue


def calculate_sponsorship_revenue(
    brand_value: float,
    team_star_power: float,
    params: RevenueParameters = None,
) -> float:
    """
    Calculate sponsorship revenue for a period.

    Rev^spon_t = 未鈧€ + 未_B路B_t + 未_star路Star_t

    Parameters
    ----------
    brand_value : float
        Team brand value B_t (slow-moving variable)
    team_star_power : float
        Team star power index (fast-moving variable)
    params : RevenueParameters, optional
        Revenue parameters (uses defaults if None)

    Returns
    -------
    float
        Sponsorship revenue
    """
    if params is None:
        params = RevenueParameters()

    revenue = (
        params.delta_0
        + params.delta_B * brand_value
        + params.delta_star * team_star_power
    )

    return revenue


def calculate_league_popularity(
    year: int = 2024,
    params: LeaguePopularityParameters = None,
    # Legacy parameters (kept for compatibility, not used in exponential model)
    avg_star_power: float = None,
    competition_index: float = None,
    media_exposure: float = None,
    social_engagement: float = None,
) -> float:
    """
    Calculate league popularity index L_t using exponential growth model.
    
    Formula:
        L_t = L_0 * x^t

    where x is the annual growth factor and t is the number of years from the
    base year.
    
    Parameters
    ----------
    year : int
        Target year for popularity calculation
        
    params : LeaguePopularityParameters, optional
        Model parameters containing L_0, x, base_year
        
    Returns
    -------
    float
        League popularity index L_t
        
    Examples
    --------
    >>> # 2024 baseline
    >>> calculate_league_popularity(year=2024)
    1.0
    
    >>> # 2025 (1 year growth at 10%)
    >>> calculate_league_popularity(year=2025)
    1.10
    
    >>> # 2026 (2 years growth at 10%)
    >>> calculate_league_popularity(year=2026)
    1.21
    
    >>> # 2030 (6 years growth at 10%)
    >>> calculate_league_popularity(year=2030)
    1.77
    """
    if params is None:
        params = LeaguePopularityParameters()
    
    # Calculate t = years from base year
    t = max(0, year - params.base_year)
    
    # L_t = L_0 路 x^t
    L_t = params.L_0 * (params.x ** t)
    
    return L_t


def load_league_popularity_from_data(year: int = 2024) -> dict:
    """
    Load L_t component values from project data files.
    
    Returns a dict with values suitable for calculate_league_popularity():
    - avg_star_power: from players_pcv.csv (avg court_impact + brand_impact)
    - competition_index: from elo_team_season_summary.csv (1/std(win_pct))
    - media_exposure: from media_deals.csv (annual_value_M)
    - social_engagement: from instagram_followers data (sum top 10)
    
    Parameters
    ----------
    year : int
        Target year for data lookup
        
    Returns
    -------
    dict
        Keys: 'avg_star_power', 'competition_index', 'media_exposure', 
              'social_engagement', 'data_sources'
              
    Examples
    --------
    >>> data = load_league_popularity_from_data(2024)
    >>> L_t = calculate_league_popularity(**{k: v for k, v in data.items() if k != 'data_sources'})
    
    Note: This function requires pandas and access to project data files.
    """
    import pandas as pd
    from pathlib import Path
    
    # Find project root
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    data_dir = project_root / "project" / "data"
    
    result = {
        'avg_star_power': 20.0,  # default
        'competition_index': 1.0,  # default
        'media_exposure': 60.0,  # default
        'social_engagement': 20.0,  # default
        'year': year,
        'data_sources': {}
    }
    
    # 1. Load Star Power from players_pcv.csv
    pcv_path = data_dir / "processed" / "players" / "players_pcv.csv"
    if pcv_path.exists():
        try:
            df_pcv = pd.read_csv(pcv_path)
            if 'court_impact' in df_pcv.columns and 'brand_impact' in df_pcv.columns:
                # Average of (court_impact + brand_impact) for all players
                total_impact = df_pcv['court_impact'] + df_pcv['brand_impact']
                result['avg_star_power'] = float(total_impact.mean())
                result['data_sources']['star'] = str(pcv_path)
        except Exception as e:
            result['data_sources']['star_error'] = str(e)
    
    # 2. Load Competition Index from elo_team_season_summary.csv
    elo_path = data_dir / "processed" / "elo" / "elo_team_season_summary.csv"
    if elo_path.exists():
        try:
            df_elo = pd.read_csv(elo_path)
            # Filter to target year
            df_year = df_elo[df_elo['season'] == year]
            if len(df_year) > 0 and 'win_pct' in df_year.columns:
                win_pct_std = df_year['win_pct'].std()
                if win_pct_std > 0:
                    # Lower std = more competitive balance = higher index
                    # Normalize: 2024 std 鈮?0.17 ->index = 1.0
                    result['competition_index'] = 0.17 / win_pct_std
                result['data_sources']['competition'] = str(elo_path)
        except Exception as e:
            result['data_sources']['competition_error'] = str(e)
    
    # 3. Load Media Exposure from media_deals.csv
    media_path = data_dir / "external" / "media_deals.csv"
    if media_path.exists():
        try:
            df_media = pd.read_csv(media_path)
            # Find deal active in target year
            for _, row in df_media.iterrows():
                if row['start_year'] <= year <= row['end_year']:
                    result['media_exposure'] = float(row['annual_value_M'])
                    result['data_sources']['media'] = str(media_path)
                    break
        except Exception as e:
            result['data_sources']['media_error'] = str(e)
    
    # 4. Load Social Engagement from instagram followers
    ig_path = data_dir / "processed" / "other_clean" / "womens-basketball-players-with-the-most-instagram-followers-2025_clean.csv"
    if ig_path.exists():
        try:
            df_ig = pd.read_csv(ig_path)
            if 'IGFollowers' in df_ig.columns:
                # Sum top 10 players' followers (in millions)
                top_10_followers = df_ig.nlargest(10, 'IGFollowers')['IGFollowers'].sum()
                result['social_engagement'] = float(top_10_followers / 1_000_000)
                result['data_sources']['social'] = str(ig_path)
        except Exception as e:
            result['data_sources']['social_error'] = str(e)
    
    return result


def calculate_league_revenue_from_popularity(
    popularity: float,
    params: LeaguePopularityParameters = None,
) -> float:
    """
    Calculate total league revenue as a function of popularity.
    
    LeagueRev(L_t) = LeagueRev_0 路 (1 + 纬 路 (L_t - 1))
    
    When L_t = 1.0 (baseline), LeagueRev = LeagueRev_0
    When L_t > 1.0 (increased popularity), LeagueRev > LeagueRev_0
    
    Parameters
    ----------
    popularity : float
        League popularity index L_t
    params : LeaguePopularityParameters, optional
        Model parameters
        
    Returns
    -------
    float
        Total league revenue in dollars
        
    Examples
    --------
    >>> # Baseline popularity
    >>> calculate_league_revenue_from_popularity(1.0)
    200000000.0
    
    >>> # 20% increase in popularity
    >>> calculate_league_revenue_from_popularity(1.2)
    212000000.0  # $200M 脳 (1 + 0.3 脳 0.2) = $212M
    """
    if params is None:
        params = LeaguePopularityParameters()
    
    # LeagueRev = LeagueRev_0 脳 (1 + 纬 脳 (L_t - 1))
    league_revenue = params.league_rev_base * (1 + params.gamma * (popularity - 1))
    
    return max(league_revenue, 0.0)


def calculate_league_dividend(
    total_league_revenue: float,
    num_teams: int,
    params: RevenueParameters = None,
) -> float:
    """
    Calculate league dividend (media rights distribution).

    Div_t = 胃_rev 路 LeagueRev(L_t) / N_t

    Assumes equal distribution among all teams (can be modified for task 3
    to account for large/small market shares).

    Parameters
    ----------
    total_league_revenue : float
        Total league revenue from media rights, etc.
    num_teams : int
        Number of teams in the league
    params : RevenueParameters, optional
        Revenue parameters (uses defaults if None)

    Returns
    -------
    float
        League dividend for this team
    """
    if params is None:
        params = RevenueParameters()

    if num_teams <= 0:
        raise ValueError("Number of teams must be positive")

    dividend = params.theta_rev * total_league_revenue / num_teams

    return dividend


def calculate_league_dividend_with_popularity(
    year: int = 2024,
    num_teams: int = 13,
    popularity_params: LeaguePopularityParameters = None,
    revenue_params: RevenueParameters = None,
) -> dict:
    """
    Calculate league dividend considering popularity dynamics.
    
    浣跨敤鎸囨暟澧為暱妯″瀷:
        L_t = L_0 路 x^t
        LeagueRev(L_t) = LeagueRev_0 路 (1 + 纬 路 (L_t - 1))
        Div_t = 胃_rev 路 LeagueRev(L_t) / N_t
    
    Parameters
    ----------
    year : int
        Current year (used to calculate t = year - base_year)
    num_teams : int
        Number of teams in league (N_t)
    popularity_params : LeaguePopularityParameters, optional
        Popularity model parameters (L_0, x, base_year)
    revenue_params : RevenueParameters, optional
        Revenue model parameters (theta_rev)
        
    Returns
    -------
    dict
        Dictionary with:
        - popularity: L_t value
        - league_revenue: LeagueRev(L_t)
        - dividend_per_team: Div_t
        - total_dividend_pool: 胃_rev 脳 LeagueRev
        - t: years from base year
        
    Examples
    --------
    >>> # 2024 baseline
    >>> result = calculate_league_dividend_with_popularity(year=2024)
    >>> result['popularity']  # L_t = 1.0
    1.0
    >>> result['dividend_per_team']  # $200M 脳 0.05 / 13 = $769K
    769230.77
    
    >>> # 2026 with 10% annual growth
    >>> result = calculate_league_dividend_with_popularity(year=2026)
    >>> result['popularity']  # L_t = 1.0 脳 1.10^2 = 1.21
    1.21
    """
    if popularity_params is None:
        popularity_params = LeaguePopularityParameters()
    if revenue_params is None:
        revenue_params = RevenueParameters()
    
    # Step 1: Calculate popularity using exponential model L_t = L_0 路 x^t
    L_t = calculate_league_popularity(year=year, params=popularity_params)
    
    # Step 2: Calculate league revenue
    league_revenue = calculate_league_revenue_from_popularity(L_t, popularity_params)
    
    # Step 3: Calculate dividend pool
    total_dividend_pool = revenue_params.theta_rev * league_revenue
    
    # Step 4: Calculate per-team dividend
    dividend_per_team = total_dividend_pool / num_teams
    
    return {
        "popularity": L_t,
        "league_revenue": league_revenue,
        "dividend_per_team": dividend_per_team,
        "total_dividend_pool": total_dividend_pool,
        "t": year - popularity_params.base_year,
    }


def calculate_period_revenue(
    games: pd.DataFrame,
    ticket_price: float,
    team_strength: float,
    team_star_power: float,
    brand_value: float,
    capacity: float,
    stage_multiplier: float = 1.0,
    total_league_revenue: float = 0.0,
    num_teams: int = 12,
    demand_params: DemandParameters = None,
    revenue_params: RevenueParameters = None,
    add_noise: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """
    Calculate total revenue for a period (e.g., a season or stage).

    Parameters
    ----------
    games : pd.DataFrame
        DataFrame with columns:
        - is_home: bool, whether game is home
        - opponent_strength: float
        - opponent_star_power: float
        - rivalry_factor: float
        - game_importance: float
    ticket_price : float
        Ticket price multiplier 蟿_t
    team_strength : float
        Team strength S_t
    team_star_power : float
        Team star power index
    brand_value : float
        Team brand value B_t
    capacity : float
        Venue capacity
    stage_multiplier : float, default=1.0
        Season stage multiplier
    total_league_revenue : float, default=0.0
        Total league revenue for dividend calculation
    num_teams : int, default=12
        Number of teams in league
    demand_params : DemandParameters, optional
        Demand model parameters
    revenue_params : RevenueParameters, optional
        Revenue model parameters
    add_noise : bool, default=False
        Whether to add random noise to demand
    random_state : int, optional
        Random seed

    Returns
    -------
    Dict[str, float]
        Dictionary with keys:
        - ticket_revenue
        - merchandise_revenue
        - sponsorship_revenue
        - league_dividend
        - total_revenue
        - total_attendance (home games only)
        - avg_attendance (home games only)
    """
    if demand_params is None:
        demand_params = DemandParameters()
    if revenue_params is None:
        revenue_params = RevenueParameters()

    # Calculate demand for each game
    demands = []
    for _, game in games.iterrows():
        attractiveness = calculate_game_attractiveness(
            opponent_strength=game["opponent_strength"],
            opponent_star_power=game["opponent_star_power"],
            rivalry_factor=game["rivalry_factor"],
            game_importance=game["game_importance"],
            params=demand_params,
        )

        demand = calculate_demand(
            ticket_price=ticket_price * revenue_params.p0,
            team_strength=team_strength,
            team_star_power=team_star_power,
            game_attractiveness=attractiveness,
            capacity=capacity,
            stage_multiplier=stage_multiplier,
            params=demand_params,
            add_noise=add_noise,
            random_state=random_state,
        )

        demands.append(demand)

    is_home = games["is_home"].tolist()

    # Calculate revenue components
    ticket_rev = calculate_ticket_revenue(
        demands=demands,
        ticket_price=ticket_price,
        base_price=revenue_params.p0,
        is_home=is_home,
    )

    merch_rev = calculate_merchandise_revenue(
        demands=demands,
        team_star_power=team_star_power,
        is_home=is_home,
        params=revenue_params,
    )

    spon_rev = calculate_sponsorship_revenue(
        brand_value=brand_value,
        team_star_power=team_star_power,
        params=revenue_params,
    )

    dividend = calculate_league_dividend(
        total_league_revenue=total_league_revenue,
        num_teams=num_teams,
        params=revenue_params,
    )

    # Calculate attendance statistics (home games only)
    home_demands = [dem for dem, home in zip(demands, is_home) if home]
    total_attendance = sum(home_demands)
    avg_attendance = total_attendance / len(home_demands) if home_demands else 0.0

    return {
        "ticket_revenue": ticket_rev,
        "merchandise_revenue": merch_rev,
        "sponsorship_revenue": spon_rev,
        "league_dividend": dividend,
        "total_revenue": ticket_rev + merch_rev + spon_rev + dividend,
        "total_attendance": total_attendance,
        "avg_attendance": avg_attendance,
    }


def simulate_season_revenue(
    num_home_games_per_stage: Dict[str, int],
    ticket_price: float,
    team_strength: float,
    team_star_power: float,
    brand_value: float,
    capacity: float,
    opponent_strength_mean: float = 0.5,
    opponent_star_mean: float = 0.5,
    rivalry_rate: float = 0.15,
    total_league_revenue: float = 50_000_000.0,
    num_teams: int = 12,
    demand_params: DemandParameters = None,
    revenue_params: RevenueParameters = None,
    stage_multipliers: SeasonStageMultipliers = None,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Simulate revenue for a full season with multiple stages.

    Parameters
    ----------
    num_home_games_per_stage : Dict[str, int]
        Number of home games in each stage, e.g., {"early": 10, "mid": 10, "late": 10}
    ticket_price : float
        Ticket price multiplier
    team_strength : float
        Team strength
    team_star_power : float
        Team star power
    brand_value : float
        Team brand value
    capacity : float
        Venue capacity
    opponent_strength_mean : float, default=0.5
        Mean opponent strength (for random generation)
    opponent_star_mean : float, default=0.5
        Mean opponent star power (for random generation)
    rivalry_rate : float, default=0.15
        Proportion of games that are rivalries
    total_league_revenue : float, default=50_000_000
        Total league revenue for dividend
    num_teams : int, default=12
        Number of teams
    demand_params : DemandParameters, optional
        Demand parameters
    revenue_params : RevenueParameters, optional
        Revenue parameters
    stage_multipliers : SeasonStageMultipliers, optional
        Stage multipliers
    random_state : int, optional
        Random seed

    Returns
    -------
    pd.DataFrame
        Revenue breakdown by stage and totals
    """
    if stage_multipliers is None:
        stage_multipliers = SeasonStageMultipliers()

    rng = np.random.RandomState(random_state)

    results = []

    for stage_name, num_games in num_home_games_per_stage.items():
        # Get stage multiplier
        multiplier = getattr(stage_multipliers, stage_name, 1.0)

        # Generate synthetic game data
        games_data = {
            "is_home": [True] * num_games,
            "opponent_strength": rng.normal(opponent_strength_mean, 0.15, num_games).clip(0, 1),
            "opponent_star_power": rng.normal(opponent_star_mean, 0.2, num_games).clip(0, 1),
            "rivalry_factor": rng.binomial(1, rivalry_rate, num_games).astype(float),
            "game_importance": rng.uniform(0.3, 0.9, num_games),
        }
        games_df = pd.DataFrame(games_data)

        # Calculate revenue for this stage
        stage_revenue = calculate_period_revenue(
            games=games_df,
            ticket_price=ticket_price,
            team_strength=team_strength,
            team_star_power=team_star_power,
            brand_value=brand_value,
            capacity=capacity,
            stage_multiplier=multiplier,
            total_league_revenue=total_league_revenue,
            num_teams=num_teams,
            demand_params=demand_params,
            revenue_params=revenue_params,
            add_noise=True,
            random_state=rng.randint(0, 1_000_000),
        )

        stage_revenue["stage"] = stage_name
        stage_revenue["num_games"] = num_games
        stage_revenue["stage_multiplier"] = multiplier
        results.append(stage_revenue)

    df_results = pd.DataFrame(results)

    # Add totals row
    totals = {
        "stage": "TOTAL",
        "num_games": df_results["num_games"].sum(),
        "stage_multiplier": np.nan,
        "ticket_revenue": df_results["ticket_revenue"].sum(),
        "merchandise_revenue": df_results["merchandise_revenue"].sum(),
        "sponsorship_revenue": df_results["sponsorship_revenue"].sum(),
        "league_dividend": df_results["league_dividend"].sum(),
        "total_revenue": df_results["total_revenue"].sum(),
        "total_attendance": df_results["total_attendance"].sum(),
        "avg_attendance": df_results["total_attendance"].sum() / df_results["num_games"].sum(),
    }
    df_results = pd.concat([df_results, pd.DataFrame([totals])], ignore_index=True)

    return df_results


class RegressionResults(NamedTuple):
    """
    Results from demand regression analysis.

    Contains fitted parameters and statistical diagnostics.
    """
    # Fitted parameters
    beta_0: float  # Intercept
    epsilon: float  # Price elasticity
    beta_S: float  # Team strength coefficient
    beta_star: float  # Star power coefficient
    beta_A: float  # Attractiveness coefficient (implicitly in A_g calculation)
    beta_cal: float  # Calendar effects coefficient

    # Statistical diagnostics
    r_squared: float  # R虏 goodness of fit
    adj_r_squared: float  # Adjusted R虏
    rmse: float  # Root mean squared error
    n_obs: int  # Number of observations

    # Standard errors
    se_beta_0: float
    se_epsilon: float
    se_beta_S: float
    se_beta_star: float
    se_beta_A: float
    se_beta_cal: float

    # t-statistics
    t_beta_0: float
    t_epsilon: float
    t_beta_S: float
    t_beta_star: float
    t_beta_A: float
    t_beta_cal: float

    # p-values
    p_beta_0: float
    p_epsilon: float
    p_beta_S: float
    p_beta_star: float
    p_beta_A: float
    p_beta_cal: float


def fit_demand_model(
    data: pd.DataFrame,
    include_calendar: bool = False,
) -> RegressionResults:
    """
    Fit demand model parameters using OLS regression on historical data.

    Estimates the log-linear demand model:
    ln(Dem_g) = 尾鈧€ - 蔚路ln(蟿_t) + 尾_S路S_t + 尾_star路Star_t + 尾_A路A_g + 尾_cal路Z_g + 畏_g

    where Dem_g 鈮?Cap_g (capacity constraint)

    Parameters
    ----------
    data : pd.DataFrame
        Historical game data with columns:
        - attendance: actual attendance (Dem_g), must be > 0
        - ticket_price: ticket price (蟿_t), must be > 0
        - team_strength: team strength (S_t)
        - team_star_power: star power index (Star_t)
        - game_attractiveness: attractiveness index (A_g)
        - calendar_effects: calendar/weather effects (Z_g), optional
        - capacity: venue capacity (Cap_g)

        Note: Observations where attendance >= capacity should be handled carefully
        as they represent censored data (demand may exceed observed attendance).

    include_calendar : bool, default=False
        Whether to include calendar effects in the regression

    Returns
    -------
    RegressionResults
        Fitted parameters and regression diagnostics

    Raises
    ------
    ValueError
        If required columns are missing or data contains invalid values

    Notes
    -----
    - The function performs OLS regression on the log-transformed model
    - Observations at capacity are included but may bias estimates downward
    - For better estimates with censored data, consider Tobit regression
    - Stage multipliers should be applied to attendance before fitting
    """
    # Validate required columns
    required_cols = [
        "attendance",
        "ticket_price",
        "team_strength",
        "team_star_power",
        "game_attractiveness",
    ]

    if include_calendar:
        required_cols.append("calendar_effects")

    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Create working copy and remove invalid observations
    df = data.copy()

    # Remove observations with non-positive attendance or price
    df = df[(df["attendance"] > 0) & (df["ticket_price"] > 0)]

    if len(df) == 0:
        raise ValueError("No valid observations after filtering")

    # Prepare regression variables
    # Dependent variable: ln(Dem_g)
    y = np.log(df["attendance"].values)

    # Independent variables
    X_cols = [
        np.ones(len(df)),  # intercept
        -np.log(df["ticket_price"].values),  # -ln(price) for -蔚路ln(蟿)
        df["team_strength"].values,
        df["team_star_power"].values,
        df["game_attractiveness"].values,
    ]

    if include_calendar:
        X_cols.append(df["calendar_effects"].values)

    # Create design matrix
    X = np.column_stack(X_cols)

    # Perform OLS regression: 尾 = (X'X)^(-1) X'y
    XtX = X.T @ X
    Xty = X.T @ y

    try:
        beta = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        raise ValueError("Singular matrix - check for multicollinearity in data")

    # Extract coefficients
    beta_0 = beta[0]
    epsilon = beta[1]  # Coefficient on -ln(price) is epsilon (already positive)
    beta_S = beta[2]
    beta_star = beta[3]
    beta_A = beta[4]
    beta_cal = beta[5] if include_calendar else 0.0

    # Calculate fitted values and residuals
    y_pred = X @ beta
    residuals = y - y_pred

    # Calculate R虏 and adjusted R虏
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)

    n = len(df)
    k = X.shape[1] - 1  # Number of predictors (excluding intercept)
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)

    # Calculate RMSE (in log space)
    rmse = np.sqrt(ss_res / n)

    # Calculate standard errors
    # Var(尾) = 蟽虏 (X'X)^(-1), where 蟽虏 = RSS / (n - k - 1)
    sigma_squared = ss_res / (n - k - 1)
    var_beta = sigma_squared * np.linalg.inv(XtX)
    se = np.sqrt(np.diag(var_beta))

    se_beta_0 = se[0]
    se_epsilon = se[1]  # Standard error for -蔚 coefficient
    se_beta_S = se[2]
    se_beta_star = se[3]
    se_beta_A = se[4]
    se_beta_cal = se[5] if include_calendar else 0.0

    # Calculate t-statistics
    t_beta_0 = beta_0 / se_beta_0
    t_epsilon = epsilon / se_epsilon  # Use positive epsilon
    t_beta_S = beta_S / se_beta_S
    t_beta_star = beta_star / se_beta_star
    t_beta_A = beta_A / se_beta_A
    t_beta_cal = beta_cal / se_beta_cal if include_calendar and se_beta_cal > 0 else 0.0

    # Calculate p-values (two-tailed test)
    df_resid = n - k - 1
    p_beta_0 = 2 * (1 - stats.t.cdf(abs(t_beta_0), df_resid))
    p_epsilon = 2 * (1 - stats.t.cdf(abs(t_epsilon), df_resid))
    p_beta_S = 2 * (1 - stats.t.cdf(abs(t_beta_S), df_resid))
    p_beta_star = 2 * (1 - stats.t.cdf(abs(t_beta_star), df_resid))
    p_beta_A = 2 * (1 - stats.t.cdf(abs(t_beta_A), df_resid))
    p_beta_cal = 2 * (1 - stats.t.cdf(abs(t_beta_cal), df_resid)) if include_calendar else 1.0

    return RegressionResults(
        beta_0=beta_0,
        epsilon=epsilon,
        beta_S=beta_S,
        beta_star=beta_star,
        beta_A=beta_A,
        beta_cal=beta_cal,
        r_squared=r_squared,
        adj_r_squared=adj_r_squared,
        rmse=rmse,
        n_obs=n,
        se_beta_0=se_beta_0,
        se_epsilon=se_epsilon,
        se_beta_S=se_beta_S,
        se_beta_star=se_beta_star,
        se_beta_A=se_beta_A,
        se_beta_cal=se_beta_cal,
        t_beta_0=t_beta_0,
        t_epsilon=t_epsilon,
        t_beta_S=t_beta_S,
        t_beta_star=t_beta_star,
        t_beta_A=t_beta_A,
        t_beta_cal=t_beta_cal,
        p_beta_0=p_beta_0,
        p_epsilon=p_epsilon,
        p_beta_S=p_beta_S,
        p_beta_star=p_beta_star,
        p_beta_A=p_beta_A,
        p_beta_cal=p_beta_cal,
    )


def print_regression_summary(results: RegressionResults) -> None:
    """
    Print a formatted summary of regression results.

    Parameters
    ----------
    results : RegressionResults
        Regression results to display
    """
    print("=" * 80)
    print("Demand Model Regression Results")
    print("=" * 80)
    print(f"\nModel: ln(Dem_g) = beta_0 - epsilon*ln(tau_t) + beta_S*S_t + beta_star*Star_t + beta_A*A_g + beta_cal*Z_g + eta_g")
    print(f"\nNumber of observations: {results.n_obs}")
    print(f"R-squared: {results.r_squared:.4f}")
    print(f"Adjusted R-squared: {results.adj_r_squared:.4f}")
    print(f"RMSE (log space): {results.rmse:.4f}")

    print("\n" + "-" * 80)
    print(f"{'Parameter':<20} {'Estimate':>12} {'Std Error':>12} {'t-stat':>10} {'p-value':>10}")
    print("-" * 80)

    params = [
        ("beta_0 (Intercept)", results.beta_0, results.se_beta_0, results.t_beta_0, results.p_beta_0),
        ("epsilon (Price)", results.epsilon, results.se_epsilon, results.t_epsilon, results.p_epsilon),
        ("beta_S (Strength)", results.beta_S, results.se_beta_S, results.t_beta_S, results.p_beta_S),
        ("beta_star (Star)", results.beta_star, results.se_beta_star, results.t_beta_star, results.p_beta_star),
        ("beta_A (Attract.)", results.beta_A, results.se_beta_A, results.t_beta_A, results.p_beta_A),
        ("beta_cal (Calendar)", results.beta_cal, results.se_beta_cal, results.t_beta_cal, results.p_beta_cal),
    ]

    for name, est, se, t, p in params:
        sig = ""
        if p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "**"
        elif p < 0.05:
            sig = "*"
        elif p < 0.1:
            sig = "."

        print(f"{name:<20} {est:>12.4f} {se:>12.4f} {t:>10.3f} {p:>10.4f} {sig}")

    print("-" * 80)
    print("Significance codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
    print("=" * 80)


def create_demand_parameters_from_regression(
    results: RegressionResults,
    a1: float = 0.10,
    a2: float = 0.20,
    a3: float = 0.35,
    a4: float = 0.25,
    sigma_eta: float = 0.25,
) -> DemandParameters:
    """
    Create DemandParameters object from regression results.

    Parameters
    ----------
    results : RegressionResults
        Fitted regression results
    a1, a2, a3, a4 : float
        Attractiveness sub-model coefficients (not estimated in main regression)
    sigma_eta : float
        Error term standard deviation (can be estimated from RMSE)

    Returns
    -------
    DemandParameters
        Parameters object for use in demand calculations
    """
    return DemandParameters(
        beta_0=results.beta_0,
        epsilon=results.epsilon,
        beta_S=results.beta_S,
        beta_star=results.beta_star,
        beta_cal=results.beta_cal,
        a1=a1,
        a2=a2,
        a3=a3,
        a4=a4,
        sigma_eta=sigma_eta,
    )


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("WNBA Team Revenue Model - Example Simulation")
    print("=" * 80)

    # Example 1: Regression Analysis with Synthetic Data
    print("\n" + "=" * 80)
    print("Example 1: Demand Model Regression Analysis")
    print("=" * 80)

    # Generate synthetic historical data that follows the demand model
    np.random.seed(42)
    n_games = 200

    # True parameters (what we want to recover)
    true_beta_0 = 8.36
    true_epsilon = 0.35
    true_beta_S = 0.20
    true_beta_star = 0.50
    true_beta_A = 1.0
    true_sigma = 0.25

    # Generate independent variables
    ticket_prices = np.random.uniform(80, 150, n_games)
    team_strengths = np.random.uniform(0.3, 0.8, n_games)
    team_star_powers = np.random.uniform(0.2, 1.0, n_games)
    game_attractiveness = np.random.uniform(0.2, 0.6, n_games)
    capacities = np.full(n_games, 12000)

    # Generate attendance following the model: ln(Dem) = 尾鈧€ - 蔚路ln(蟿) + 尾_S路S + 尾_star路Star + 尾_A路A + 畏
    log_attendance = (
        true_beta_0
        - true_epsilon * np.log(ticket_prices)
        + true_beta_S * team_strengths
        + true_beta_star * team_star_powers
        + true_beta_A * game_attractiveness
        + np.random.normal(0, true_sigma, n_games)
    )

    # Convert to actual attendance and apply capacity constraint
    attendance = np.exp(log_attendance)
    attendance = np.minimum(attendance, capacities)

    # Create synthetic game data
    synthetic_data = pd.DataFrame({
        "attendance": attendance,
        "ticket_price": ticket_prices,
        "team_strength": team_strengths,
        "team_star_power": team_star_powers,
        "game_attractiveness": game_attractiveness,
        "calendar_effects": np.zeros(n_games),
        "capacity": capacities,
    })

    # Fit the demand model
    print("\nFitting demand model to synthetic data...")
    print(f"True parameters: beta_0={true_beta_0:.2f}, epsilon={true_epsilon:.2f}, beta_S={true_beta_S:.2f}, beta_star={true_beta_star:.2f}, beta_A={true_beta_A:.2f}")

    regression_results = fit_demand_model(synthetic_data, include_calendar=False)

    # Print regression summary
    print_regression_summary(regression_results)

    # Create DemandParameters from regression results
    fitted_params = create_demand_parameters_from_regression(regression_results)
    print("\nFitted DemandParameters object:")
    print(fitted_params)

    # Example 2: Revenue Simulation with Default Parameters
    print("\n" + "=" * 80)
    print("Example 2: Season Revenue Simulation")
    print("=" * 80)

    # Example: simulate a season with 3 stages
    # Using realistic WNBA parameters (2024 season)
    season_revenue = simulate_season_revenue(
        num_home_games_per_stage={"early": 10, "mid": 10, "late": 10},
        ticket_price=1.0,  # No price adjustment (base=$123)
        team_strength=0.6,  # Above-average team
        team_star_power=0.8,  # Strong star power (e.g., team with star player)
        brand_value=1.2,  # Above-average brand
        capacity=12000,  # Las Vegas Aces arena capacity (Michelob ULTRA Arena)
        random_state=42,
    )

    print("\nRevenue by Season Stage:")
    print(season_revenue.to_string(index=False))

    print("\n" + "=" * 80)
    print("Model Parameters:")
    print("=" * 80)
    print("\nDefault Demand Parameters:")
    print(DemandParameters())
    print("\nRevenue Parameters:")
    print(RevenueParameters())
    print("\nStage Multipliers:")
    print(SeasonStageMultipliers())

    # Example 3: Compare predictions with different parameters
    print("\n" + "=" * 80)
    print("Example 3: Demand Prediction Comparison")
    print("=" * 80)

    # Test case: predict attendance for a specific game
    test_game = {
        "ticket_price": 123.0,
        "team_strength": 0.6,
        "team_star_power": 0.8,
        "game_attractiveness": 0.4,
        "capacity": 12000,
    }

    default_params = DemandParameters()
    default_demand = calculate_demand(
        ticket_price=test_game["ticket_price"],
        team_strength=test_game["team_strength"],
        team_star_power=test_game["team_star_power"],
        game_attractiveness=test_game["game_attractiveness"],
        capacity=test_game["capacity"],
        params=default_params,
    )

    print(f"\nTest game parameters:")
    print(f"  Ticket price: ${test_game['ticket_price']:.2f}")
    print(f"  Team strength: {test_game['team_strength']:.2f}")
    print(f"  Star power: {test_game['team_star_power']:.2f}")
    print(f"  Attractiveness: {test_game['game_attractiveness']:.2f}")
    print(f"  Capacity: {test_game['capacity']:,.0f}")

    print(f"\nPredicted attendance (default parameters): {default_demand:,.0f}")
    print(f"Capacity utilization: {default_demand / test_game['capacity'] * 100:.1f}%")

