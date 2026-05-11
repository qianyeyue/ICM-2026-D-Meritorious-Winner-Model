"""
Task 1 MPC-style dynamic system (lightweight, grid-search, runnable).

Goal
----
Close the loop:
  (cash/brand/strength) -> decisions (tau_t, m_t, u_t, borrowing) -> games -> revenue/cost
  -> update (S_t, B_t, Star_t, Cash_t, D_t) -> repeat.

Notes
-----
- No heavy solver; we use small discrete action grids + Monte Carlo lookahead (MPC).
- Uses only repo data / runnable outputs:
  - Elo: project/data/processed/elo_final_ratings.csv, elo_config.csv
  - Brand B0: project/data/processed/brand_b0.csv (and its indicators)
  - Cost params: project/model/cost.py (numbers only)
  - Revenue model: project/model/income.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Import from core modules
from ..core.brands import BrandParameters, TEAM_NAME_TO_CODE, update_brand
from ..core.cost import (
    CashFlowParameters,
    FinancingParameters,
    GeneralAdminParameters,
    SalaryCostParameters,
    SportsOperationsParameters,
    VenueCostParameters,
    calculate_risk_metrics,
)
from ..core.income import DemandParameters, RevenueParameters

# Import shared utilities
from ..core.utils import (
    elo_win_probability,
    strength_from_elo,
    calculate_ebitda_from_dict,
)
from ..core.data_loader import DataLoader
from ..task2.task1_adapter import load_json as load_task2_json
from ..task2.task1_adapter import map_task2_state_to_task1_initial_state


ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT_DIR / "project" / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
CONFIG_DIR = PROCESSED_DIR / "config"
ELO_DIR = PROCESSED_DIR / "elo"
SIMULATION_DIR = PROCESSED_DIR / "simulation"
MPC_DIR = PROCESSED_DIR / "mpc"
PLAYERS_DIR = PROCESSED_DIR / "players"


def _read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return pd.read_csv(f)


SALARY_PARAMS = SalaryCostParameters()
VENUE_PARAMS = VenueCostParameters()
GA_PARAMS = GeneralAdminParameters()
SPORTS_PARAMS = SportsOperationsParameters()
FINANCE_PARAMS = FinancingParameters()
CASH_PARAMS = CashFlowParameters()


@dataclass(frozen=True)
class ControlAction:
    tau: float  # ticket multiplier (tau_t)
    m: float  # marketing spend (USD) (m_t)
    u: float  # incremental sports investment (USD) (u_t)


@dataclass(frozen=True)
class MPCConfig:
    periods: int = 3
    horizon: int = 4  # [OPTIMIZED: was 2] Extended planning horizon
    n_mc: int = 10  # Reduced from 20 for 20s target (was 100 originally)
    discount: float = 0.95  # [OPTIMIZED: was 0.98] Value future more

    # Strength / star dynamics
    alpha_u: float = 5.0  # [UPDATED] Increased from 2.0 to 5.0 - investment should meaningfully boost Elo
    phi: float = 1.5  # [UPDATED] Reduced from 3.0 to 1.5 - lower fatigue penalty to balance u_boost
    kappa_u: float = 0.25  # Star amplification via sports spend
    kappa_m: float = 0.35  # Star amplification via marketing spend

    # Constraints / penalties
    min_cash: float = CashFlowParameters().min_cash_balance
    max_debt: float = 5_000_000.0  # [UPDATED] Reduced from 30M to 5M for moderate financing
    debt_penalty: float = 0.02  # weight on debt in objective

    # Objective function weights (from model.md Section 1.1)
    # J = E[Σ δ^t(π_t + λ_W*W_t)] + δ^T*E[Π^PO] + δ^T*E[V_end] - λ_risk*Risk
    lambda_win: float = 100_000.0  # λ_W: Weight on wins in objective function
    lambda_playoff: float = 1.0  # δ^T coefficient for playoff revenue
    lambda_terminal: float = 1.0  # δ^T coefficient for terminal value
    lambda_risk: float = 1.0  # λ_risk: Weight on risk penalty (CVaR/bankruptcy)

    # ELO-triggered investment boost
    elo_trigger_threshold: float = 1550.0  # [NEW] When ELO drops below this, boost u by 20%
    elo_trigger_boost: float = 0.30  # [FIXED] Reduced from 0.50 to 0.30 for moderate response

    # Season context (we use a simple 2*(N-1) schedule from teams in brand_b0.csv)
    league_revenue_year: float = 200_000_000.0  # from project/data/processed/other_clean/revenue-of-the-wnba-2022-2024_clean.csv
    stage_multipliers: Tuple[float, ...] = (1.0, 1.2, 1.4)
    stage_importance: Tuple[float, ...] = (0.40, 0.60, 0.80)

    # Control mode
    # - "policy": tau is discrete and automatically varies by stage; m and u are continuous controls updated from record.
    # - "grid": original grid-search MPC on (tau, m, u).
    control_mode: str = "policy"

    # Discrete ticket multiplier schedule (3 stages by default).
    tau_by_stage: Tuple[float, ...] = (0.90, 1.00, 1.10)

    # Continuous control policy bounds (USD)
    m_min: float = 200_000.0
    m_max: float = 800_000.0  # [FIXED] Reduced from 1M to 800K to control spending
    u_min: float = 100_000.0  # [FIXED] Reduced from 500K to 100K for lower baseline
    u_max: float = 2_500_000.0  # [FIXED] Reduced from 6M to 2.5M to prevent over-investment

    # Record-based policy parameters
    target_win_rate: float = 0.55
    # Gains map win-rate gap to where we are between min/max.
    # u increases when win_rate < target; m increases when win_rate > target.
    u_gain: float = 1.0  # [FIXED] Reduced from 2.0 to 1.0 for more conservative response
    m_gain: float = 0.8  # [FIXED] Reduced from 1.0 to 0.8 to control marketing spend
    # If cash margin above min_cash is small, damp spending toward mins.
    cash_buffer: float = 5_000_000.0
    # Smooth updates across periods (0=no change, 1=fully jump to new value)
    policy_smoothing: float = 0.70

    # Task3 exogenous shocks (optional, default = no shock)
    task3_revenue_sharing_ratio: float = 1.0  # multiplies league dividend only
    task3_competition_intensity: float = 0.0  # recorded for reporting/debugging
    task3_competition_elasticity: float = 0.0  # demand shock per unit competition_intensity
    task3_travel_fatigue_delta: float = 0.0  # added to away-game fatigue


@dataclass(frozen=True)
class MultiYearConfig:
    """Configuration for multi-year prediction."""
    n_years: int = 1
    n_simulations_per_year: int = 20
    elo_carryover: float = 0.90  # [UPDATED] Increased from 0.75 to 0.90 for slower Elo decay
    save_yearly_details: bool = True


def _load_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_task3_exogenous_shock(cfg: MPCConfig, *, shock: Dict[str, Any]) -> MPCConfig:
    ratio = float(shock.get("revenue_sharing_ratio", 1.0))
    competition_intensity = float(shock.get("competition_intensity", 0.0))
    competition_elasticity = float(shock.get("competition_elasticity", 0.0))
    travel_fatigue_delta = float(shock.get("travel_fatigue_delta", 0.0))

    if ratio <= 0.0:
        ratio = 1.0

    return replace(
        cfg,
        task3_revenue_sharing_ratio=float(ratio),
        task3_competition_intensity=float(competition_intensity),
        task3_competition_elasticity=float(competition_elasticity),
        task3_travel_fatigue_delta=float(travel_fatigue_delta),
    )


@dataclass
class SimulationResults:
    """
    Comprehensive simulation results with J decomposition.

    Implements model.md Section 1.4: Owner Value decomposition
    J = J^{reg_profit} + J^{win} + J^{PO} + J^{end} - J^{risk}
    """
    # Period-by-period data
    period_log: pd.DataFrame

    # Final state
    final_elo: float
    final_B: float
    final_cash: float
    final_debt: float
    cum_profit: float
    cum_ebitda: float
    total_wins: int

    # J decomposition (Section 1.4)
    J_total: float
    J_reg_profit: float  # Regular season profit contribution
    J_win: float  # Win contribution (λ_W * W_t)
    J_playoff: float  # Playoff revenue contribution
    J_terminal: float  # Terminal value contribution
    J_risk: float  # Risk penalty

    # Control variables
    total_u_usd: float
    total_m_usd: float
    avg_tau: float
    total_borrowed: float

    # Risk metrics
    min_cash: float
    bankruptcy_occurred: bool

    # Playoff qualification
    made_playoffs: bool
    playoff_revenue: float = 0.0


@dataclass
class TeamState:
    elo: float
    B: float
    cash: float
    debt: float
    cum_profit: float = 0.0
    cum_ebitda: float = 0.0
    cum_wins: int = 0  # [NEW] Track cumulative wins for J decomposition


# Removed: _resolve_path, load_brand_b0, load_elo_ratings, load_elo_config
# Now using DataLoader from data_loader.py

# Removed: min_max_scale
# Now using min_max_scale from utils.py

# Removed: strength_from_elo
# Now using strength_from_elo from utils.py


def elo_home_win_prob(home_elo: float, away_elo: float, *, home_advantage: float) -> float:
    """
    Calculate home team win probability using Elo formula.

    Note: This is a wrapper for utils.elo_win_probability for backward compatibility.
    Consider using utils.elo_win_probability directly.
    """
    return elo_win_probability(home_elo, away_elo, home_advantage=home_advantage)



def elo_home_win_prob(home_elo: float, away_elo: float, *, home_advantage: float) -> float:
    delta = (float(home_elo) - float(away_elo)) + float(home_advantage)
    return float(1.0 / (1.0 + 10.0 ** (-delta / 400.0)))


def build_double_round_robin_indices(num_teams: int) -> Tuple[np.ndarray, np.ndarray]:
    home = []
    away = []
    for i in range(num_teams):
        for j in range(num_teams):
            if i == j:
                continue
            home.append(i)
            away.append(j)
    return np.asarray(home, dtype=int), np.asarray(away, dtype=int)


def build_action_grid() -> List[ControlAction]:
    tau_grid = [0.90, 1.00, 1.10]
    m_grid = [200_000.0, 500_000.0, 1_000_000.0]
    u_grid = [500_000.0, 1_000_000.0, 2_000_000.0, 4_000_000.0, 6_000_000.0]  # [UPDATED] Increased baseline and max to match new u_min/u_max
    return [ControlAction(tau=t, m=m, u=u) for t, m, u in itertools.product(tau_grid, m_grid, u_grid)]


def tau_for_stage(cfg: MPCConfig, *, period_num: int) -> float:
    if not cfg.tau_by_stage:
        return 1.0
    i = int(max(period_num, 0))
    i = min(i, len(cfg.tau_by_stage) - 1)
    return float(cfg.tau_by_stage[i])


def _expected_win_rate_prior(elo: float, *, league_mean_elo: float) -> float:
    # Prior win-rate estimate when no games have been played yet.
    # Use Elo vs league mean as a single-match proxy.
    p = float(1.0 / (1.0 + 10.0 ** (-(float(elo) - float(league_mean_elo)) / 400.0)))
    return float(np.clip(p, 0.10, 0.90))


def continuous_controls_from_record(
    cfg: MPCConfig,
    *,
    win_rate: float,
    cash: float,
    min_cash: float,
    prev_m: Optional[float] = None,
    prev_u: Optional[float] = None,
) -> Tuple[float, float]:
    """Record-feedback continuous policy for (m, u). [DEPRECATED: use SA optimizer instead]

    - If win_rate < target: increase u toward u_max.
    - If win_rate > target: increase m toward m_max.
    - If cash is tight: damp both toward mins.
    """
    wr = float(np.clip(win_rate, 0.0, 1.0))

    u_span = float(max(cfg.u_max - cfg.u_min, 0.0))
    m_span = float(max(cfg.m_max - cfg.m_min, 0.0))

    gap = float(cfg.target_win_rate) - wr
    u_frac = float(np.clip(cfg.u_gain * max(gap, 0.0) / max(cfg.target_win_rate, 1e-6), 0.0, 1.0))
    m_frac = float(np.clip(cfg.m_gain * max(-gap, 0.0) / max(1.0 - cfg.target_win_rate, 1e-6), 0.0, 1.0))

    u_raw = float(cfg.u_min) + u_frac * u_span
    m_raw = float(cfg.m_min) + m_frac * m_span

    # Cash safety damping: move toward mins when cash margin is low.
    margin = float(cash) - float(min_cash)
    if float(cfg.cash_buffer) > 0:
        scale = float(np.clip(margin / float(cfg.cash_buffer), 0.0, 1.0))
        u_raw = float(cfg.u_min) + scale * (u_raw - float(cfg.u_min))
        m_raw = float(cfg.m_min) + scale * (m_raw - float(cfg.m_min))

    # Smooth changes to avoid extreme jumps.
    s = float(np.clip(cfg.policy_smoothing, 0.0, 1.0))
    if prev_m is not None:
        m_raw = (1.0 - s) * float(prev_m) + s * float(m_raw)
    if prev_u is not None:
        u_raw = (1.0 - s) * float(prev_u) + s * float(u_raw)

    m_out = float(np.clip(m_raw, float(cfg.m_min), float(cfg.m_max)))
    u_out = float(np.clip(u_raw, float(cfg.u_min), float(cfg.u_max)))
    return m_out, u_out


# =============================================================================
# SA + MC Optimizer for continuous (m, u) with stage-based discrete tau
# =============================================================================

def _rollout_with_continuous_controls(
    start_state: TeamState,
    *,
    m_seq: Sequence[float],
    u_seq: Sequence[float],
    tau_seq: Sequence[float],
    period_schedules: Sequence[Tuple[np.ndarray, np.ndarray]],
    t0: int,
    focus_team: str,
    focus_idx: int,
    teams: Sequence[str],
    elos_start: np.ndarray,
    B_start: float,
    ticket_base_mult: pd.Series,
    star_base_by_team: pd.Series,
    capacity_by_team: pd.Series,
    avg_salary_usd: float,
    base_elo: float,
    home_advantage: float,
    k: float,
    league_home_games_total: int,
    cfg: MPCConfig,
    rng: np.random.RandomState,
    lambda_win: float = 100_000.0,
) -> float:
    """
    Rollout objective for SA optimizer with continuous (m, u) and discrete tau.
    
    Objective: maximize expected (profit + lambda_win * wins) - risk penalty
    """
    elos = elos_start.copy()
    elos[focus_idx] = float(start_state.elo)
    B = float(B_start)

    s = TeamState(
        elo=float(start_state.elo),
        B=float(B_start),
        cash=float(start_state.cash),
        debt=float(start_state.debt),
        cum_profit=float(start_state.cum_profit),
        cum_ebitda=float(start_state.cum_ebitda),
    )

    total_profit = 0.0
    total_wins = 0.0
    min_cash_seen = float(s.cash)
    horizon = min(len(m_seq), len(u_seq), len(tau_seq))

    for h in range(horizon):
        t = int(t0 + h)
        if t >= len(period_schedules):
            break

        action = ControlAction(
            tau=float(tau_seq[h]),
            m=float(m_seq[h]),
            u=float(u_seq[h]),
        )

        stage_mult = float(cfg.stage_multipliers[min(t, len(cfg.stage_multipliers) - 1)])
        stage_imp = float(cfg.stage_importance[min(t, len(cfg.stage_importance) - 1)])

        home_idx, away_idx = period_schedules[t]
        elos_pre = elos.copy()

        elos, wins, games_cnt, home_games, away_games, opps_home, _ = simulate_period_games(
            home_idx,
            away_idx,
            elos,
            focus_idx=focus_idx,
            focus_action=action,
            cfg=cfg,
            k=k,
            base_elo=base_elo,
            home_advantage=home_advantage,
            rng=rng,
            period_num=t,
        )

        star_t = star_from_controls(float(star_base_by_team.get(focus_team, 0.6)), action=action, cfg=cfg)
        rev = calculate_period_revenue_for_team(
            team=focus_team,
            team_idx=focus_idx,
            opp_indices_home=opps_home,
            elos_pre=elos_pre,
            teams=teams,
            B_t=B,
            star_t=star_t,
            action=action,
            cfg=cfg,
            stage_multiplier=stage_mult,
            stage_importance=stage_imp,
            ticket_base_mult=ticket_base_mult,
            star_base_by_team=star_base_by_team,
            capacity_by_team=capacity_by_team,
            base_elo=base_elo,
            league_home_games_total=league_home_games_total,
        )

        costs, s, _borrowed = calculate_period_costs_and_cash_update(
            revenue=float(rev["total_revenue"]),
            attendance=float(rev["total_attendance"]),
            away_games=int(away_games),
            avg_salary_usd=float(avg_salary_usd),
            action=action,
            state=s,
            cfg=cfg,
        )

        B = update_brand(
            B_t=float(B),
            W_t=float(wins),
            m_t=float(action.m) / 1_000_000.0,
            Star_t=float(star_t),
            params=BrandParameters(),
        )

        total_profit += float(costs["profit"])
        total_wins += float(wins)
        min_cash_seen = min(min_cash_seen, float(s.cash))

    # Terminal value contribution
    terminal_val = terminal_proxy_value(s)
    
    # Risk penalty: penalize if cash ever dropped near or below minimum
    cash_risk_penalty = 0.0
    if min_cash_seen < float(cfg.min_cash) * 1.5:
        cash_risk_penalty = float(cfg.min_cash) * 1.5 - min_cash_seen

    # Objective: profit + win bonus + discounted terminal - risk
    objective = (
        total_profit
        + lambda_win * total_wins
        + float(cfg.lambda_terminal) * terminal_val
        - float(cfg.debt_penalty) * float(s.debt)
        - cash_risk_penalty
    )
    return float(objective)


def _mc_evaluate_controls(
    m_seq: Sequence[float],
    u_seq: Sequence[float],
    tau_seq: Sequence[float],
    *,
    n_mc: int,
    seed: int,
    start_state: TeamState,
    period_schedules: Sequence[Tuple[np.ndarray, np.ndarray]],
    t0: int,
    focus_team: str,
    focus_idx: int,
    teams: Sequence[str],
    elos_start: np.ndarray,
    B_start: float,
    ticket_base_mult: pd.Series,
    star_base_by_team: pd.Series,
    capacity_by_team: pd.Series,
    avg_salary_usd: float,
    base_elo: float,
    home_advantage: float,
    k: float,
    league_home_games_total: int,
    cfg: MPCConfig,
    lambda_win: float = 100_000.0,
) -> float:
    """Monte Carlo evaluation of a control sequence."""
    scores = []
    for mc_i in range(n_mc):
        rng = np.random.RandomState(int(seed + 10000 * mc_i))
        score = _rollout_with_continuous_controls(
            start_state,
            m_seq=m_seq,
            u_seq=u_seq,
            tau_seq=tau_seq,
            period_schedules=period_schedules,
            t0=t0,
            focus_team=focus_team,
            focus_idx=focus_idx,
            teams=teams,
            elos_start=elos_start,
            B_start=B_start,
            ticket_base_mult=ticket_base_mult,
            star_base_by_team=star_base_by_team,
            capacity_by_team=capacity_by_team,
            avg_salary_usd=avg_salary_usd,
            base_elo=base_elo,
            home_advantage=home_advantage,
            k=k,
            league_home_games_total=league_home_games_total,
            cfg=cfg,
            rng=rng,
            lambda_win=lambda_win,
        )
        scores.append(score)
    return float(np.mean(scores)) if scores else 0.0


def sa_optimize_continuous_controls(
    start_state: TeamState,
    *,
    t0: int,
    horizon: int,
    focus_team: str,
    focus_idx: int,
    teams: Sequence[str],
    elos_start: np.ndarray,
    B_start: float,
    period_schedules: Sequence[Tuple[np.ndarray, np.ndarray]],
    ticket_base_mult: pd.Series,
    star_base_by_team: pd.Series,
    capacity_by_team: pd.Series,
    avg_salary_usd: float,
    base_elo: float,
    home_advantage: float,
    k: float,
    league_home_games_total: int,
    cfg: MPCConfig,
    seed: int,
    n_mc_coarse: int = 5,
    n_mc_fine: int = 15,
    sa_iterations: int = 50,
    initial_temp: float = 1_000_000.0,
    cooling_rate: float = 0.92,
    lambda_win: float = 100_000.0,
) -> Tuple[float, float]:
    """
    Simulated Annealing optimizer for continuous (m, u) with MC evaluation.

    tau is determined by stage (not optimized here).

    Returns: (optimal_m, optimal_u) for the current period.
    """
    rng = np.random.RandomState(int(seed))

    # Build tau sequence from stage schedule
    tau_seq = [tau_for_stage(cfg, period_num=t0 + h) for h in range(horizon)]

    # Initialize with midpoint
    m_init = (float(cfg.m_min) + float(cfg.m_max)) / 2.0
    u_init = (float(cfg.u_min) + float(cfg.u_max)) / 2.0

    # [NEW] ELO-triggered investment boost: if ELO < threshold, boost u_max by 20%
    u_max_adjusted = float(cfg.u_max)
    if float(start_state.elo) < float(cfg.elo_trigger_threshold):
        u_max_adjusted = float(cfg.u_max) * (1.0 + float(cfg.elo_trigger_boost))
        u_init = min(u_init * 1.2, u_max_adjusted)  # Also boost initial guess
    
    # For simplicity, optimize a single (m, u) applied to all periods in horizon
    # (This is "piecewise constant" as recommended in doc 2.10.1)
    best_m = m_init
    best_u = u_init

    # Evaluate initial solution
    m_seq = [best_m] * horizon
    u_seq = [best_u] * horizon
    best_score = _mc_evaluate_controls(
        m_seq, u_seq, tau_seq,
        n_mc=n_mc_coarse,
        seed=seed,
        start_state=start_state,
        period_schedules=period_schedules,
        t0=t0,
        focus_team=focus_team,
        focus_idx=focus_idx,
        teams=teams,
        elos_start=elos_start,
        B_start=B_start,
        ticket_base_mult=ticket_base_mult,
        star_base_by_team=star_base_by_team,
        capacity_by_team=capacity_by_team,
        avg_salary_usd=avg_salary_usd,
        base_elo=base_elo,
        home_advantage=home_advantage,
        k=k,
        league_home_games_total=league_home_games_total,
        cfg=cfg,
        lambda_win=lambda_win,
    )
    
    current_m = best_m
    current_u = best_u
    current_score = best_score
    temp = initial_temp

    m_span = float(cfg.m_max - cfg.m_min)
    u_span = float(u_max_adjusted - cfg.u_min)  # [UPDATED] Use adjusted u_max for span calculation
    
    for iteration in range(sa_iterations):
        # Generate neighbor by perturbing (m, u) with Gaussian noise
        # Scale perturbation by temperature
        scale = temp / initial_temp
        delta_m = rng.normal(0, 0.15 * m_span * scale)
        delta_u = rng.normal(0, 0.15 * u_span * scale)
        
        new_m = float(np.clip(current_m + delta_m, cfg.m_min, cfg.m_max))
        new_u = float(np.clip(current_u + delta_u, cfg.u_min, u_max_adjusted))  # [UPDATED] Use adjusted u_max
        
        # Evaluate neighbor
        m_seq = [new_m] * horizon
        u_seq = [new_u] * horizon
        new_score = _mc_evaluate_controls(
            m_seq, u_seq, tau_seq,
            n_mc=n_mc_coarse,
            seed=seed + iteration,
            start_state=start_state,
            period_schedules=period_schedules,
            t0=t0,
            focus_team=focus_team,
            focus_idx=focus_idx,
            teams=teams,
            elos_start=elos_start,
            B_start=B_start,
            ticket_base_mult=ticket_base_mult,
            star_base_by_team=star_base_by_team,
            capacity_by_team=capacity_by_team,
            avg_salary_usd=avg_salary_usd,
            base_elo=base_elo,
            home_advantage=home_advantage,
            k=k,
            league_home_games_total=league_home_games_total,
            cfg=cfg,
            lambda_win=lambda_win,
        )
        
        # SA acceptance criterion
        delta = new_score - current_score
        if delta > 0:
            # Accept improvement
            current_m = new_m
            current_u = new_u
            current_score = new_score
            if new_score > best_score:
                best_m = new_m
                best_u = new_u
                best_score = new_score
        else:
            # Accept worse solution with probability exp(delta/temp)
            accept_prob = float(np.exp(delta / max(temp, 1e-10)))
            if rng.rand() < accept_prob:
                current_m = new_m
                current_u = new_u
                current_score = new_score
        
        # Cool down
        temp *= cooling_rate
    
    # Fine evaluation of best solution
    m_seq = [best_m] * horizon
    u_seq = [best_u] * horizon
    fine_score = _mc_evaluate_controls(
        m_seq, u_seq, tau_seq,
        n_mc=n_mc_fine,
        seed=seed + 999999,
        start_state=start_state,
        period_schedules=period_schedules,
        t0=t0,
        focus_team=focus_team,
        focus_idx=focus_idx,
        teams=teams,
        elos_start=elos_start,
        B_start=B_start,
        ticket_base_mult=ticket_base_mult,
        star_base_by_team=star_base_by_team,
        capacity_by_team=capacity_by_team,
        avg_salary_usd=avg_salary_usd,
        base_elo=base_elo,
        home_advantage=home_advantage,
        k=k,
        league_home_games_total=league_home_games_total,
        cfg=cfg,
        lambda_win=lambda_win,
    )
    
    return float(best_m), float(best_u)


def build_ticket_base_multiplier(brand_b0: pd.DataFrame, *, rev_params: RevenueParameters) -> pd.Series:
    b = brand_b0.set_index("team")
    ticket_usd = pd.to_numeric(b["avg_ticket_price_usd"], errors="coerce")
    league_ticket = float(ticket_usd.mean()) if ticket_usd.notna().any() else float(rev_params.p0)
    ticket_index = (ticket_usd / league_ticket).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    shrink = 0.25
    return (1.0 + shrink * (ticket_index - 1.0)).clip(0.80, 1.20).astype(float)


def build_star_base(brand_b0: pd.DataFrame) -> pd.Series:
    from ..core.utils import min_max_scale
    b = brand_b0.set_index("team")
    ig = pd.to_numeric(b.get("ig_followers", pd.Series(index=b.index, dtype=float)), errors="coerce").fillna(0.0)
    x01 = min_max_scale(np.log1p(ig))
    return (0.2 + 1.0 * x01).astype(float)


def star_from_controls(star_base: float, *, action: ControlAction, cfg: MPCConfig) -> float:
    u_m = float(action.u) / 1_000_000.0
    m_m = float(action.m) / 1_000_000.0
    amp = 1.0 + cfg.kappa_u * float(np.log1p(max(u_m, 0.0))) + cfg.kappa_m * float(np.log1p(max(m_m, 0.0)))
    return float(max(0.0, star_base * amp))


def simulate_period_games(
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    elos: np.ndarray,
    *,
    focus_idx: int,
    focus_action: ControlAction,
    cfg: MPCConfig,
    k: float,
    base_elo: float,
    home_advantage: float,
    rng: np.random.RandomState,
    period_num: int = 0,
) -> Tuple[np.ndarray, int, int, int, int, List[int], float]:
    """
    Simulate all games in a period, updating all teams' Elo with fatigue effects.

    Returns:
      - updated elos
      - focus wins in this period (W_t)
      - focus games, focus home games, focus away games
      - list of opponents for focus home games (for revenue)
      - average fatigue for focus team
    """
    elos_out = elos.copy()
    focus_wins = 0
    focus_games = 0
    focus_home = 0
    focus_away = 0
    focus_home_opps: List[int] = []

    u_m = float(focus_action.u) / 1_000_000.0
    u_boost_per_game = 0.0
    focus_games_in_period = int(np.sum((home_idx == focus_idx) | (away_idx == focus_idx)))
    if focus_games_in_period > 0:
        u_boost_per_game = cfg.alpha_u * float(np.log1p(max(u_m, 0.0))) / float(focus_games_in_period)

    # Fatigue calculation: simplified model based on period intensity
    # Early period: lower fatigue, late period: higher fatigue
    base_fatigue = 0.15 + 0.10 * float(period_num)  # 0.15 -> 0.25 -> 0.35

    # Back-to-back probability increases in later periods (schedule compression)
    b2b_prob = 0.10 + 0.05 * float(period_num)  # 10% -> 15% -> 20%

    focus_fatigue_sum = 0.0
    focus_game_count = 0

    for h, a in zip(home_idx.tolist(), away_idx.tolist()):
        # Calculate fatigue for this game (focus team only)
        game_fatigue = base_fatigue
        if h == focus_idx or a == focus_idx:
            is_b2b = rng.rand() < b2b_prob
            is_away = (a == focus_idx)

            # Fatigue components (simplified from fatigue.py model)
            fatigue_b2b = 0.30 if is_b2b else 0.0
            fatigue_travel = max(0.0, 0.20 + float(cfg.task3_travel_fatigue_delta)) if is_away else 0.0
            fatigue_minutes = 0.10  # Assume average core minutes

            game_fatigue = fatigue_b2b + fatigue_travel + fatigue_minutes
            game_fatigue = min(game_fatigue, 1.0)  # Cap at 1.0

            focus_fatigue_sum += game_fatigue
            focus_game_count += 1

        p_home = elo_home_win_prob(elos_out[h], elos_out[a], home_advantage=home_advantage)
        y = 1.0 if rng.rand() < p_home else 0.0

        delta = float(k) * (y - p_home)
        elos_out[h] += delta
        elos_out[a] -= delta

        # Add u_t -> strength drift and subtract fatigue penalty for focus team
        if h == focus_idx:
            fatigue_penalty = cfg.phi * game_fatigue
            elos_out[h] += u_boost_per_game - fatigue_penalty
            focus_games += 1
            focus_home += 1
            focus_wins += int(y == 1.0)
            focus_home_opps.append(a)
        elif a == focus_idx:
            fatigue_penalty = cfg.phi * game_fatigue
            elos_out[a] += u_boost_per_game - fatigue_penalty
            focus_games += 1
            focus_away += 1
            focus_wins += int(y == 0.0)

    avg_fatigue = focus_fatigue_sum / float(focus_game_count) if focus_game_count > 0 else 0.0
    return elos_out, int(focus_wins), int(focus_games), int(focus_home), int(focus_away), focus_home_opps, float(avg_fatigue)


def calculate_period_revenue_for_team(
    *,
    team: str,
    team_idx: int,
    opp_indices_home: Sequence[int],
    elos_pre: np.ndarray,
    teams: Sequence[str],
    B_t: float,
    star_t: float,
    action: ControlAction,
    cfg: MPCConfig,
    stage_multiplier: float,
    stage_importance: float,
    ticket_base_mult: pd.Series,
    star_base_by_team: pd.Series,
    capacity_by_team: pd.Series,
    base_elo: float,
    league_home_games_total: int,
) -> Dict[str, float]:
    if len(opp_indices_home) == 0:
        return {
            "ticket_revenue": 0.0,
            "merchandise_revenue": 0.0,
            "sponsorship_revenue": 0.0,
            "league_dividend": 0.0,
            "total_revenue": 0.0,
            "total_attendance": 0.0,
            "avg_attendance": 0.0,
        }

    demand_params = DemandParameters()
    rev_params = RevenueParameters()

    n_home = int(len(opp_indices_home))
    f = float(n_home) / float(max(int(league_home_games_total), 1))

    ticket_mult = float(ticket_base_mult.get(team, 1.0)) * float(action.tau)
    ticket_price_usd = ticket_mult * float(rev_params.p0)

    strength_t = strength_from_elo(float(elos_pre[team_idx]), base_elo=base_elo)
    cap = float(capacity_by_team.get(team, float(capacity_by_team.median()) if len(capacity_by_team) else 10000.0))

    opp_strength = np.empty(n_home, dtype=float)
    opp_star = np.empty(n_home, dtype=float)
    for i, j in enumerate(opp_indices_home):
        opp = teams[int(j)]
        opp_strength[i] = strength_from_elo(elos_pre[int(j)], base_elo=base_elo)
        opp_star[i] = float(star_base_by_team.get(opp, float(star_base_by_team.mean()) if len(star_base_by_team) else 0.6))

    attractiveness = (
        demand_params.a1 * opp_strength
        + demand_params.a2 * opp_star
        + demand_params.a3 * 0.15
        + demand_params.a4 * float(stage_importance)
    )

    log_demand = (
        demand_params.beta_0
        - demand_params.epsilon * np.log(ticket_price_usd)
        + demand_params.beta_S * float(strength_t)
        + demand_params.beta_star * float(star_t)
        + attractiveness
        + float(np.log(stage_multiplier))
    )
    demand = np.exp(log_demand)
    competition_mult = 1.0 - float(cfg.task3_competition_elasticity) * float(cfg.task3_competition_intensity)
    demand = demand * float(np.clip(competition_mult, 0.50, 1.50))
    demand = np.minimum(demand, cap)

    total_att = float(demand.sum())
    avg_att = float(total_att / n_home) if n_home > 0 else 0.0

    ticket_rev = float(ticket_mult * rev_params.p0 * total_att)
    merch_rev = float(rev_params.kappa_1 * total_att + (rev_params.kappa_2 * float(star_t) * f))

    sponsor_full = float(rev_params.delta_0 + rev_params.delta_B * float(B_t) + rev_params.delta_star * float(star_t))
    sponsor = sponsor_full * f

    dividend = float(
        rev_params.theta_rev
        * float(cfg.league_revenue_year)
        * float(cfg.task3_revenue_sharing_ratio)
        * f
        / float(max(len(teams), 1))
    )

    total_rev = ticket_rev + merch_rev + sponsor + dividend

    return {
        "ticket_revenue": float(ticket_rev),
        "merchandise_revenue": float(merch_rev),
        "sponsorship_revenue": float(sponsor),
        "league_dividend": float(dividend),
        "total_revenue": float(total_rev),
        "total_attendance": float(total_att),
        "avg_attendance": float(avg_att),
    }


def calculate_period_costs_and_cash_update(
    *,
    revenue: float,
    attendance: float,
    away_games: int,
    avg_salary_usd: float,
    action: ControlAction,
    state: TeamState,
    cfg: MPCConfig,
) -> Tuple[Dict[str, float], TeamState, float]:
    """
    Returns (cost_breakdown, updated_state, borrowed_amount).
    """
    # Fixed costs should be split evenly by period (not scaled by games),
    # otherwise a "light schedule period" looks unrealistically profitable.
    periods = float(max(int(cfg.periods), 1))

    roster_salary_season = float(avg_salary_usd) * float(SALARY_PARAMS.roster_size)
    salary_cost = roster_salary_season / periods

    venue_fixed = float(VENUE_PARAMS.fixed_venue_cost) / periods
    venue_var = float(VENUE_PARAMS.variable_cost_per_attendee) * float(attendance)
    venue_cost = venue_fixed + venue_var

    tax = float(GA_PARAMS.tax_rate) * float(revenue)
    ga_fixed = float(GA_PARAMS.fixed_ga_cost) / periods
    ga_cost = ga_fixed + float(action.m) + tax

    sports_fixed = float(SPORTS_PARAMS.base_sports_ops_cost) / periods
    travel_cost = float(SPORTS_PARAMS.travel_cost_per_game) * float(away_games)
    sports_cost = sports_fixed + travel_cost + float(action.u)

    interest = float(FINANCE_PARAMS.interest_rate) * float(state.debt) / periods

    total_cost = salary_cost + venue_cost + ga_cost + sports_cost + interest
    profit = float(revenue) - total_cost

    capex = float(CASH_PARAMS.annual_capex) / periods
    cash_next = float(state.cash) + profit - capex

    # [UPDATED] Moderate financing: allow borrowing up to max_debt limit
    borrowed = 0.0
    debt_next = float(state.debt)
    if cash_next < float(cfg.min_cash):
        needed = float(cfg.min_cash) - cash_next
        # Check if we can borrow (within debt limit)
        available_credit = float(cfg.max_debt) - debt_next
        borrowed = min(needed, available_credit)
        if borrowed > 0:
            debt_next += borrowed
            cash_next += borrowed

    out = {
        "salary_cost": float(salary_cost),
        "venue_cost": float(venue_cost),
        "ga_cost": float(ga_cost),
        "sports_ops_cost": float(sports_cost),
        "financing_cost": float(interest),
        "capex": float(capex),
        "total_cost": float(total_cost),
        "profit": float(profit),
        "tax": float(tax),
    }

    new_state = TeamState(
        elo=float(state.elo),
        B=float(state.B),
        cash=float(cash_next),
        debt=float(debt_next),
        cum_profit=float(state.cum_profit) + float(profit),
        cum_ebitda=float(state.cum_ebitda) + calculate_ebitda_from_dict(revenue=float(revenue), costs=out),
    )
    return out, new_state, float(borrowed)


# Removed: estimate_ebitda
# Now using calculate_ebitda_from_dict from utils.py


def terminal_proxy_value(state: TeamState, *, mu_brand: float = 30_000_000.0) -> float:
    # Light proxy: brand asset value + cash - debt.
    return float(mu_brand) * float(state.B) + float(state.cash) - float(state.debt)


def apply_season_carryover(
    elo: float,
    *,
    base_elo: float = 1500.0,
    carryover: float = 0.90,  # [UPDATED] Increased from 0.75 to 0.90 for slower Elo decay
    shock_std: float = 0.0,
    rng: Optional[np.random.Generator] = None
) -> float:
    """
    Apply season carryover to Elo rating with optional random shock.

    Formula: elo_new = base_elo + (elo_old - base_elo) * carryover + shock

    This regresses ratings toward the mean between seasons and adds random variation
    to simulate roster changes, injuries, coaching changes, etc.

    Args:
        elo: Current Elo rating
        base_elo: Base Elo rating (default 1500)
        carryover: Carryover coefficient (default 0.75)
        shock_std: Standard deviation of random shock (default 0.0, no shock)
        rng: Random number generator (if None, no shock applied)

    Returns:
        New Elo rating after carryover and shock
    """
    carried_elo = base_elo + (elo - base_elo) * carryover

    # Add random shock if enabled
    if shock_std > 0.0 and rng is not None:
        shock = rng.normal(0.0, shock_std)
        carried_elo += shock

    return float(carried_elo)


def rollout_objective(
    start_state: TeamState,
    *,
    actions: Sequence[ControlAction],
    period_schedules: Sequence[Tuple[np.ndarray, np.ndarray]],
    t0: int,
    focus_team: str,
    focus_idx: int,
    teams: Sequence[str],
    elos_start: np.ndarray,
    B_start: float,
    ticket_base_mult: pd.Series,
    star_base_by_team: pd.Series,
    capacity_by_team: pd.Series,
    avg_salary_usd: float,
    base_elo: float,
    home_advantage: float,
    k: float,
    league_home_games_total: int,
    cfg: MPCConfig,
    rng: np.random.RandomState,
) -> float:
    # Copy per-rollout mutable state
    elos = elos_start.copy()
    elos[focus_idx] = float(start_state.elo)
    B = float(B_start)

    s = TeamState(
        elo=float(start_state.elo),
        B=float(B_start),
        cash=float(start_state.cash),
        debt=float(start_state.debt),
        cum_profit=float(start_state.cum_profit),
        cum_ebitda=float(start_state.cum_ebitda),
    )

    total = 0.0

    for h, action in enumerate(actions):
        t = int(t0 + h)
        if t >= len(period_schedules):
            break

        stage_mult = float(cfg.stage_multipliers[min(t, len(cfg.stage_multipliers) - 1)])
        stage_imp = float(cfg.stage_importance[min(t, len(cfg.stage_importance) - 1)])

        home_idx, away_idx = period_schedules[t]
        elos_pre = elos.copy()

        elos, wins, games_cnt, home_games, away_games, opps_home, _ = simulate_period_games(
            home_idx,
            away_idx,
            elos,
            focus_idx=focus_idx,
            focus_action=action,
            cfg=cfg,
            k=k,
            base_elo=base_elo,
            home_advantage=home_advantage,
            rng=rng,
            period_num=t,
        )

        star_t = star_from_controls(float(star_base_by_team.get(focus_team, 0.6)), action=action, cfg=cfg)
        rev = calculate_period_revenue_for_team(
            team=focus_team,
            team_idx=focus_idx,
            opp_indices_home=opps_home,
            elos_pre=elos_pre,
            teams=teams,
            B_t=B,
            star_t=star_t,
            action=action,
            cfg=cfg,
            stage_multiplier=stage_mult,
            stage_importance=stage_imp,
            ticket_base_mult=ticket_base_mult,
            star_base_by_team=star_base_by_team,
            capacity_by_team=capacity_by_team,
            base_elo=base_elo,
            league_home_games_total=league_home_games_total,
        )

        costs, s, _borrowed = calculate_period_costs_and_cash_update(
            revenue=float(rev["total_revenue"]),
            attendance=float(rev["total_attendance"]),
            away_games=int(away_games),
            avg_salary_usd=float(avg_salary_usd),
            action=action,
            state=s,
            cfg=cfg,
        )

        # Update brand (slow variable)
        B = update_brand(
            B_t=float(B),
            W_t=float(wins),
            m_t=float(action.m) / 1_000_000.0,
            Star_t=float(star_t),
            params=BrandParameters(),
        )

        s.elo = float(elos[focus_idx])
        s.B = float(B)

        if s.debt > float(cfg.max_debt):
            return float("-inf")

        total += (float(cfg.discount) ** h) * float(costs["profit"])

    total += float(cfg.lambda_terminal) * terminal_proxy_value(s) - float(cfg.debt_penalty) * float(s.debt)
    return float(total)


def _rollout_objective_wrapper(
    *,
    actions: Sequence[ControlAction],
    seed: int,
    start_state: TeamState,
    period_schedules: Sequence[Tuple[np.ndarray, np.ndarray]],
    t0: int,
    focus_team: str,
    focus_idx: int,
    teams: Sequence[str],
    elos_start: np.ndarray,
    B_start: float,
    ticket_base_mult: pd.Series,
    star_base_by_team: pd.Series,
    capacity_by_team: pd.Series,
    avg_salary_usd: float,
    base_elo: float,
    home_advantage: float,
    k: float,
    league_home_games_total: int,
    cfg: MPCConfig,
    **_kwargs,
) -> float:
    """
    Picklable wrapper for parallel rollouts.

    Important: this must be defined at module scope (not inside choose_action_mpc),
    otherwise ProcessPoolExecutor on Windows will fail to pickle it.
    """
    rng = np.random.RandomState(int(seed))
    return rollout_objective(
        start_state,
        actions=actions,
        period_schedules=period_schedules,
        t0=int(t0),
        focus_team=str(focus_team),
        focus_idx=int(focus_idx),
        teams=teams,
        elos_start=elos_start,
        B_start=float(B_start),
        ticket_base_mult=ticket_base_mult,
        star_base_by_team=star_base_by_team,
        capacity_by_team=capacity_by_team,
        avg_salary_usd=float(avg_salary_usd),
        base_elo=float(base_elo),
        home_advantage=float(home_advantage),
        k=float(k),
        league_home_games_total=int(league_home_games_total),
        cfg=cfg,
        rng=rng,
    )


def choose_action_mpc(
    state: TeamState,
    *,
    t: int,
    focus_team: str,
    focus_idx: int,
    teams: Sequence[str],
    elos_start: np.ndarray,
    B_start: float,
    period_schedules: Sequence[Tuple[np.ndarray, np.ndarray]],
    ticket_base_mult: pd.Series,
    star_base_by_team: pd.Series,
    capacity_by_team: pd.Series,
    avg_salary_usd: float,
    base_elo: float,
    home_advantage: float,
    k: float,
    league_home_games_total: int,
    actions: Sequence[ControlAction],
    cfg: MPCConfig,
    seed: int,
    use_parallel: bool = True,
) -> ControlAction:
    remaining = len(period_schedules) - int(t)
    horizon = int(min(max(cfg.horizon, 1), max(remaining, 1)))

    sequences = list(itertools.product(actions, repeat=horizon))

    if use_parallel:
        from .parallel_mpc import parallel_mc_evaluation

        # Parallel evaluation
        # NOTE: rollout function must be picklable for ProcessPoolExecutor on Windows,
        # so we use a module-level wrapper (not a nested function).
        results = parallel_mc_evaluation(
            sequences,
            _rollout_objective_wrapper,
            n_mc=int(cfg.n_mc),
            seed=seed,
            t=t,
            start_state=state,
            period_schedules=period_schedules,
            t0=int(t),
            focus_team=focus_team,
            focus_idx=focus_idx,
            teams=teams,
            elos_start=elos_start,
            B_start=float(B_start),
            ticket_base_mult=ticket_base_mult,
            star_base_by_team=star_base_by_team,
            capacity_by_team=capacity_by_team,
            avg_salary_usd=avg_salary_usd,
            base_elo=base_elo,
            home_advantage=home_advantage,
            k=k,
            league_home_games_total=league_home_games_total,
            cfg=cfg,
        )

        # Find best sequence
        best_first: Optional[ControlAction] = None
        best_score = float("-inf")

        for seq_i, scores in results.items():
            if scores:
                score = float(np.mean(scores))
                if score > best_score:
                    best_score = score
                    best_first = sequences[seq_i][0]
    else:
        # Sequential evaluation (original implementation)
        best_first: Optional[ControlAction] = None
        best_score = float("-inf")

        for seq_i, seq in enumerate(sequences):
            scores = []
            for mc_i in range(int(cfg.n_mc)):
                rng = np.random.RandomState(int(seed + 100000 * t + 1000 * seq_i + mc_i))
                scores.append(
                    rollout_objective(
                        state,
                        actions=seq,
                        period_schedules=period_schedules,
                        t0=int(t),
                        focus_team=focus_team,
                        focus_idx=focus_idx,
                        teams=teams,
                        elos_start=elos_start,
                        B_start=float(B_start),
                        ticket_base_mult=ticket_base_mult,
                        star_base_by_team=star_base_by_team,
                        capacity_by_team=capacity_by_team,
                        avg_salary_usd=avg_salary_usd,
                        base_elo=base_elo,
                        home_advantage=home_advantage,
                        k=k,
                        league_home_games_total=league_home_games_total,
                        cfg=cfg,
                        rng=rng,
                    )
                )
            score = float(np.mean(scores)) if scores else float("-inf")
            if score > best_score:
                best_score = score
                best_first = seq[0]

    return best_first if best_first is not None else actions[0]


def split_schedule_into_periods(
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    *,
    periods: int,
    rng: np.random.RandomState,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    n = len(home_idx)
    order = rng.permutation(n)
    chunk = int(n // periods)
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    start = 0
    for p in range(int(periods)):
        end = start + chunk if p < periods - 1 else n
        idx = order[start:end]
        out.append((home_idx[idx], away_idx[idx]))
        start = end
    return out


def run_mpc_simulation(
    *,
    focus_team: str,
    brand_b0_path: str | Path = CONFIG_DIR / "brand_b0.csv",
    elo_ratings_path: str | Path = ELO_DIR / "elo_final_ratings.csv",
    elo_config_path: str | Path = CONFIG_DIR / "elo_config.csv",
    init_state_path: Optional[str | Path] = None,
    task3_impact_path: Optional[str | Path] = None,
    seed: int = 42,
    cfg: MPCConfig = MPCConfig(),
    out_dir: str | Path = MPC_DIR,
    write: bool = True,
    use_parallel: bool = True,
    decision_every: int = 1,
    fixed_action: Optional[ControlAction] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Load data using DataLoader
    loader = DataLoader()
    brand_b0 = loader.load_brand_b0(brand_b0_path)
    teams = brand_b0["team"].astype(str).str.strip().tolist()
    if focus_team not in teams:
        raise ValueError(f"focus_team {focus_team!r} not in brand_b0 teams: {teams}")

    elo_cfg = loader.load_elo_config(elo_config_path)
    base_elo = float(elo_cfg.get("base_elo", 1500.0))
    k = float(elo_cfg.get("k", 20.0))
    home_advantage = float(elo_cfg.get("home_advantage", 0.0))

    elo_by_team = loader.load_elo_ratings(elo_ratings_path, teams=teams, base_elo=base_elo)
    focus_idx = int(teams.index(focus_team))

    b = brand_b0.set_index("team")
    capacity_by_team = pd.to_numeric(b.get("arena_capacity", pd.Series(index=b.index, dtype=float)), errors="coerce").fillna(10000.0)
    ticket_base_mult = build_ticket_base_multiplier(brand_b0, rev_params=RevenueParameters())
    star_base_by_team = build_star_base(brand_b0)

    # Costs need a salary proxy; use team-average salary if present, else league mean.
    avg_salary_path = PROCESSED_DIR / "other_clean" / "average-player-salary-in-the-wnba-by-team-2025_clean.csv"
    if avg_salary_path.is_file():
        df_sal = _read_csv(avg_salary_path)
        df_sal = df_sal[df_sal.get("Year", 2025) == 2025].copy()
        df_sal["team"] = df_sal["Team"].astype(str).str.strip().map(lambda s: TEAM_NAME_TO_CODE.get(s))
        avg_salary_by_team = (
            df_sal.assign(avg_salary_usd=pd.to_numeric(df_sal["AvgSalaryUSD"], errors="coerce"))
            .dropna(subset=["team", "avg_salary_usd"])
            .set_index("team")["avg_salary_usd"]
        )
    else:
        avg_salary_by_team = pd.Series(dtype=float)

    avg_salary_usd = float(avg_salary_by_team.get(focus_team, float(avg_salary_by_team.mean()) if len(avg_salary_by_team) else 120000.0))

    # Build league schedule (double round robin). This defines the whole season.
    home_idx, away_idx = build_double_round_robin_indices(len(teams))
    rng_real = np.random.RandomState(int(seed))
    period_schedules = split_schedule_into_periods(home_idx, away_idx, periods=int(cfg.periods), rng=rng_real)

    # For per-period prorating, compute season home games for focus team (exact from schedule).
    focus_home_games_total = int(np.sum(home_idx == focus_idx))
    if focus_home_games_total <= 0:
        focus_home_games_total = max(int((len(teams) - 1)), 1)

    actions: List[ControlAction] = []
    if str(cfg.control_mode).lower().strip() == "grid":
        actions = build_action_grid()

    # Initialize state for the controlled team
    init_state = TeamState(
        elo=float(elo_by_team[focus_team]),
        B=float(pd.to_numeric(b.loc[focus_team, "B0"], errors="coerce")) if focus_team in b.index else 1.0,
        cash=float(CashFlowParameters().initial_cash),
        debt=float(FinancingParameters().initial_debt),
    )

    if init_state_path:
        raw = load_task2_json(Path(init_state_path))
        if all(k in raw for k in ("elo", "B", "cash")):
            init_state = TeamState(
                elo=float(raw["elo"]),
                B=float(raw["B"]),
                cash=float(raw["cash"]),
                debt=float(raw.get("debt", FinancingParameters().initial_debt)),
            )
            if float(raw.get("avg_salary_usd", 0.0)) > 0:
                avg_salary_usd = float(raw["avg_salary_usd"])
        elif any(k in raw for k in ("S_0", "B_0", "Cash_0")):
            elo_std = float(np.std([float(elo_by_team[t]) for t in teams])) if len(teams) else 0.0
            elo_per_s = (elo_std / 10.0) if elo_std > 0 else 10.0
            mapped = map_task2_state_to_task1_initial_state(
                raw,
                base_elo=float(base_elo),
                elo_per_s_point=float(elo_per_s),
                initial_cash=float(CashFlowParameters().initial_cash),
                pcv_path=PLAYERS_DIR / "players_pcv.csv",
                roster_size_default=12,
            )
            init_state = TeamState(
                elo=float(mapped["elo"]),
                B=float(mapped["B"]),
                cash=float(mapped["cash"]),
                debt=float(FinancingParameters().initial_debt),
            )
            if float(mapped.get("avg_salary_usd", 0.0)) > 0:
                avg_salary_usd = float(mapped["avg_salary_usd"])

    if task3_impact_path:
        task3_shock = _load_json(task3_impact_path)
        cfg = apply_task3_exogenous_shock(cfg, shock=task3_shock)

    # We maintain full league Elo array, but only optimize for one team.
    elos = np.asarray([float(elo_by_team[t]) for t in teams], dtype=float)
    elos[focus_idx] = float(init_state.elo)
    B_t = float(init_state.B)
    s = init_state

    rows: List[Dict[str, float]] = []

    decision_every_n = int(decision_every) if int(decision_every) > 0 else 1
    cached_action: Optional[ControlAction] = None

    # Record feedback for continuous policy
    cum_wins = 0
    cum_games = 0
    cached_m: Optional[float] = None
    cached_u: Optional[float] = None
    league_mean_elo = float(np.mean(elos)) if len(elos) else float(base_elo)

    for t in range(int(cfg.periods)):
        control_mode = str(cfg.control_mode).lower().strip()
        if fixed_action is not None:
            # User-forced fixed action overrides both modes.
            action = fixed_action
        elif control_mode == "grid":
            if not actions:
                actions = build_action_grid()
            if cached_action is None or (t % decision_every_n == 0):
                cached_action = choose_action_mpc(
                    s,
                    t=int(t),
                    focus_team=focus_team,
                    focus_idx=focus_idx,
                    teams=teams,
                    elos_start=elos,
                    B_start=float(B_t),
                    period_schedules=period_schedules,
                    ticket_base_mult=ticket_base_mult,
                    star_base_by_team=star_base_by_team,
                    capacity_by_team=capacity_by_team,
                    avg_salary_usd=avg_salary_usd,
                    base_elo=base_elo,
                    home_advantage=home_advantage,
                    k=k,
                    league_home_games_total=focus_home_games_total,
                    actions=actions,
                    cfg=cfg,
                    seed=int(seed),
                    use_parallel=use_parallel,
                )
            action = cached_action if cached_action is not None else actions[0]
        else:
            # Policy mode with SA + MC optimization:
            # - tau is discrete and varies by stage/period (not optimized).
            # - m and u are continuous, optimized via Simulated Annealing + Monte Carlo.
            tau_t = tau_for_stage(cfg, period_num=int(t))

            # Determine remaining horizon for lookahead
            remaining = len(period_schedules) - int(t)
            horizon = int(min(max(cfg.horizon, 1), max(remaining, 1)))

            if cached_m is None or cached_u is None or (t % decision_every_n == 0):
                # Run SA + MC optimization for continuous (m, u)
                cached_m, cached_u = sa_optimize_continuous_controls(
                    s,
                    t0=int(t),
                    horizon=horizon,
                    focus_team=focus_team,
                    focus_idx=focus_idx,
                    teams=teams,
                    elos_start=elos,
                    B_start=float(B_t),
                    period_schedules=period_schedules,
                    ticket_base_mult=ticket_base_mult,
                    star_base_by_team=star_base_by_team,
                    capacity_by_team=capacity_by_team,
                    avg_salary_usd=avg_salary_usd,
                    base_elo=base_elo,
                    home_advantage=home_advantage,
                    k=k,
                    league_home_games_total=focus_home_games_total,
                    cfg=cfg,
                    seed=int(seed) + int(t) * 12345,
                    n_mc_coarse=max(3, int(cfg.n_mc) // 2),
                    n_mc_fine=int(cfg.n_mc),
                    sa_iterations=40,
                    initial_temp=1_000_000.0,
                    cooling_rate=0.90,
                    lambda_win=100_000.0,
                )

            action = ControlAction(tau=float(tau_t), m=float(cached_m), u=float(cached_u))

        home_p, away_p = period_schedules[t]
        elos_pre = elos.copy()

        elos, wins, games_cnt, home_games, away_games, opps_home, avg_fatigue = simulate_period_games(
            home_p,
            away_p,
            elos,
            focus_idx=focus_idx,
            focus_action=action,
            cfg=cfg,
            k=k,
            base_elo=base_elo,
            home_advantage=home_advantage,
            rng=rng_real,
            period_num=t,
        )

        # Update record feedback for next decision.
        cum_wins += int(wins)
        cum_games += int(games_cnt)

        stage_mult = float(cfg.stage_multipliers[min(t, len(cfg.stage_multipliers) - 1)])
        stage_imp = float(cfg.stage_importance[min(t, len(cfg.stage_importance) - 1)])

        star_t = star_from_controls(float(star_base_by_team.get(focus_team, 0.6)), action=action, cfg=cfg)
        rev = calculate_period_revenue_for_team(
            team=focus_team,
            team_idx=focus_idx,
            opp_indices_home=opps_home,
            elos_pre=elos_pre,
            teams=teams,
            B_t=B_t,
            star_t=star_t,
            action=action,
            cfg=cfg,
            stage_multiplier=stage_mult,
            stage_importance=stage_imp,
            ticket_base_mult=ticket_base_mult,
            star_base_by_team=star_base_by_team,
            capacity_by_team=capacity_by_team,
            base_elo=base_elo,
            league_home_games_total=focus_home_games_total,
        )

        costs, s2, borrowed = calculate_period_costs_and_cash_update(
            revenue=float(rev["total_revenue"]),
            attendance=float(rev["total_attendance"]),
            away_games=int(away_games),
            avg_salary_usd=float(avg_salary_usd),
            action=action,
            state=s,
            cfg=cfg,
        )

        B_next = update_brand(
            B_t=float(B_t),
            W_t=float(wins),
            m_t=float(action.m) / 1_000_000.0,
            Star_t=float(star_t),
            params=BrandParameters(),
        )

        row = {
            "period": int(t + 1),
            "team": focus_team,
            "tau": float(action.tau),
            "m_usd": float(action.m),
            "u_usd": float(action.u),
            "elo_start": float(elos_pre[focus_idx]),
            "elo_end": float(elos[focus_idx]),
            "B_start": float(B_t),
            "B_end": float(B_next),
            "Star_t": float(star_t),
            "avg_fatigue": float(avg_fatigue),
            "wins": float(wins),
            "games": float(games_cnt),
            "home_games": float(home_games),
            "away_games": float(away_games),
            "total_attendance": float(rev["total_attendance"]),
            "total_revenue_usd": float(rev["total_revenue"]),
            "ticket_revenue_usd": float(rev["ticket_revenue"]),
            "merch_revenue_usd": float(rev["merchandise_revenue"]),
            "sponsorship_revenue_usd": float(rev["sponsorship_revenue"]),
            "league_dividend_usd": float(rev["league_dividend"]),
            "total_cost_usd": float(costs["total_cost"]),
            "profit_usd": float(costs["profit"]),
            "ebitda_usd": float(calculate_ebitda_from_dict(revenue=float(rev["total_revenue"]), costs=costs)),
            "cash_start": float(s.cash),
            "cash_end": float(s2.cash),
            "debt_start": float(s.debt),
            "debt_end": float(s2.debt),
            "borrowed_usd": float(borrowed),
        }
        rows.append(row)

        # Commit updates
        B_t = float(B_next)
        s = s2
        s.elo = float(elos[focus_idx])
        s.B = float(B_t)

    period_log = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "team": focus_team,
                "final_elo": float(s.elo),
                "final_B": float(s.B),
                "final_cash": float(s.cash),
                "final_debt": float(s.debt),
                "cum_profit": float(s.cum_profit),
                "cum_ebitda": float(s.cum_ebitda),
                "terminal_proxy": float(terminal_proxy_value(s)),
                "task3_revenue_sharing_ratio": float(cfg.task3_revenue_sharing_ratio),
                "task3_competition_intensity": float(cfg.task3_competition_intensity),
                "task3_competition_elasticity": float(cfg.task3_competition_elasticity),
                "task3_travel_fatigue_delta": float(cfg.task3_travel_fatigue_delta),
            }
        ]
    )

    if write:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        period_log.to_csv(out_dir / f"mpc_period_log_{focus_team}.csv", index=False)
        summary.to_csv(out_dir / f"mpc_summary_{focus_team}.csv", index=False)

    return period_log, summary


def run_multi_year_mpc_simulation(
    *,
    focus_team: str,
    n_years: int,
    n_simulations_per_year: int = 10,  # Reduced from 20 for 20s target
    brand_b0_path: str | Path = CONFIG_DIR / "brand_b0.csv",
    elo_ratings_path: str | Path = ELO_DIR / "elo_final_ratings.csv",
    elo_config_path: str | Path = CONFIG_DIR / "elo_config.csv",
    init_state_path: Optional[str | Path] = None,
    task3_impact_path: Optional[str | Path] = None,
    seed: int = 42,
    cfg: MPCConfig = MPCConfig(),
    out_dir: str | Path = MPC_DIR,
    write: bool = True,
    use_parallel: bool = True,
    elo_shock_std: float = 50.0,  # [UPDATED] Increased from 30.0 to 50.0 for more Elo randomness
    decision_every: int = 1,
    fixed_action: Optional[ControlAction] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run multi-year MPC simulation with Monte Carlo sampling.

    Each year runs n_simulations_per_year independent simulations.
    Elo ratings carry over between years with regression to mean.

    Parameters
    ----------
    focus_team : str
        Team code (e.g., "LVA")
    n_years : int
        Number of years to simulate
    n_simulations_per_year : int
        Monte Carlo simulations per year (default: 20)
    brand_b0_path : str | Path
        Path to brand_b0.csv
    elo_ratings_path : str | Path
        Path to elo_final_ratings.csv
    elo_config_path : str | Path
        Path to elo_config.csv
    init_state_path : Optional[str | Path]
        Path to initial state JSON (optional)
    task3_impact_path : Optional[str | Path]
        Path to task3 impact JSON (optional)
    seed : int
        Random seed
    cfg : MPCConfig
        MPC configuration
    out_dir : str | Path
        Output directory
    write : bool
        Whether to write results to CSV
    use_parallel : bool
        Whether to use parallel processing

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (yearly_summary, aggregate_stats)
        - yearly_summary: Detailed results for each year and simulation
        - aggregate_stats: Mean/std/quantiles aggregated by year
    """
    # Load initial data
    loader = DataLoader()
    brand_b0 = loader.load_brand_b0(brand_b0_path)
    teams = brand_b0["team"].astype(str).str.strip().tolist()

    elo_cfg = loader.load_elo_config(elo_config_path)
    base_elo = float(elo_cfg.get("base_elo", 1500.0))
    elo_carryover = float(elo_cfg.get("season_carryover", 0.90))  # [UPDATED] Default 0.90 (slower decay)

    # Load initial Elo ratings for ALL teams
    initial_elo_by_team = loader.load_elo_ratings(elo_ratings_path, teams=teams, base_elo=base_elo)

    # Load initial state if provided
    if init_state_path is not None:
        init_state_data = _load_json(init_state_path)
    else:
        init_state_data = None

    # Initialize storage
    all_results = []

    print(f"\nRunning {n_years}-year simulation with {n_simulations_per_year} samples per year...")
    print(f"Focus team: {focus_team}")
    print(f"Base Elo: {base_elo}, Carryover: {elo_carryover}")
    print(f"Random shock std: {elo_shock_std} Elo points")
    print(f"Applying carryover to ALL {len(teams)} teams between seasons")
    print("=" * 70)

    # Outer loop: simulations
    for sim_id in range(n_simulations_per_year):
        if (sim_id + 1) % 20 == 0:
            print(f"  Simulation {sim_id + 1}/{n_simulations_per_year}")

        # Initialize Elo ratings for all teams for this simulation
        current_elo_by_team = initial_elo_by_team.copy()

        # Initialize state for this simulation
        current_init_state_path = init_state_path

        # Create RNG for this simulation (for random shocks)
        sim_rng = np.random.default_rng(seed + sim_id * 1000000)

        # Create temporary Elo ratings file for this simulation
        import tempfile
        import json
        import csv

        temp_elo_fd, temp_elo_path = tempfile.mkstemp(suffix='_elo.csv', text=True)
        # Close the file descriptor immediately, we'll use the path
        os.close(temp_elo_fd)

        # Inner loop: years
        for year in range(1, n_years + 1):
            # Write current Elo ratings to temporary file
            with open(temp_elo_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['team', 'elo'])
                for team in teams:
                    writer.writerow([team, current_elo_by_team[team]])

            # Run single-year MPC simulation with updated Elo ratings
            period_log, summary = run_mpc_simulation(
                focus_team=focus_team,
                brand_b0_path=brand_b0_path,
                elo_ratings_path=temp_elo_path,  # Use temporary Elo file
                elo_config_path=elo_config_path,
                init_state_path=current_init_state_path,
                task3_impact_path=task3_impact_path,
                seed=seed + sim_id * 10000 + year * 100,
                cfg=cfg,
                out_dir=out_dir,
                write=False,  # Don't write individual runs
                use_parallel=use_parallel,
                decision_every=decision_every,
                fixed_action=fixed_action,
            )

            # Extract final state
            final_elo = float(summary['final_elo'].iloc[0])
            final_B = float(summary['final_B'].iloc[0])
            final_cash = float(summary['final_cash'].iloc[0])
            final_debt = float(summary['final_debt'].iloc[0])
            cum_profit = float(summary['cum_profit'].iloc[0])
            cum_ebitda = float(summary['cum_ebitda'].iloc[0])

            # Calculate total wins and games from period_log
            total_wins = int(period_log['wins'].sum())
            total_games = int(period_log['games'].sum())

            # Extract control variables (tau, m, u) from period_log
            avg_tau = float(period_log['tau'].mean()) if 'tau' in period_log.columns else 1.0
            total_m = float(period_log['m_usd'].sum()) if 'm_usd' in period_log.columns else 0.0
            total_u = float(period_log['u_usd'].sum()) if 'u_usd' in period_log.columns else 0.0
            avg_m = float(period_log['m_usd'].mean()) if 'm_usd' in period_log.columns else 0.0
            avg_u = float(period_log['u_usd'].mean()) if 'u_usd' in period_log.columns else 0.0

            # Record results
            all_results.append({
                'year': year,
                'simulation_id': sim_id,
                'final_elo': final_elo,
                'final_B': final_B,
                'final_cash': final_cash,
                'final_debt': final_debt,
                'cum_profit': cum_profit,
                'cum_ebitda': cum_ebitda,
                'total_wins': total_wins,
                'total_games': total_games,
                'avg_tau': avg_tau,
                'total_m_usd': total_m,
                'total_u_usd': total_u,
                'avg_m_usd': avg_m,
                'avg_u_usd': avg_u,
            })

            # Apply carryover for next year to ALL teams
            if year < n_years:
                # Apply season carryover to ALL teams' Elo ratings with random shocks
                for team in teams:
                    if team == focus_team:
                        # Use the simulated final Elo for focus team
                        current_elo_by_team[team] = apply_season_carryover(
                            final_elo,
                            base_elo=base_elo,
                            carryover=elo_carryover,
                            shock_std=elo_shock_std,
                            rng=sim_rng
                        )
                    else:
                        # Apply carryover to other teams' Elo with random shocks
                        current_elo_by_team[team] = apply_season_carryover(
                            current_elo_by_team[team],
                            base_elo=base_elo,
                            carryover=elo_carryover,
                            shock_std=elo_shock_std,
                            rng=sim_rng
                        )

                # Update focus team's Elo with carried over value
                carried_elo = current_elo_by_team[focus_team]

                # Create temporary state file for next year
                temp_state = {
                    "elo": carried_elo,
                    "B": final_B,
                    "cash": final_cash,
                    "debt": final_debt,
                }

                # Write to temporary file
                temp_state_fd, temp_state_path = tempfile.mkstemp(suffix='_state.json', text=True)
                with os.fdopen(temp_state_fd, 'w') as f:
                    json.dump(temp_state, f)

                current_init_state_path = temp_state_path

        # Clean up temporary Elo file after all years for this simulation
        try:
            os.unlink(temp_elo_path)
        except:
            pass

    print(f"  Simulation {n_simulations_per_year}/{n_simulations_per_year}")
    print("=" * 70)
    print("Computing aggregate statistics...")

    # Create DataFrames
    yearly_summary = pd.DataFrame(all_results)

    # Compute aggregate statistics
    def q25(x):
        return x.quantile(0.25)

    def q75(x):
        return x.quantile(0.75)

    q25.__name__ = 'q25'
    q75.__name__ = 'q75'

    aggregate_stats = yearly_summary.groupby('year').agg({
        'final_elo': ['mean', 'std', q25, q75],
        'cum_profit': ['mean', 'std', q25, q75],
        'cum_ebitda': ['mean', 'std'],
        'total_wins': ['mean', 'std'],
        'final_cash': ['mean', 'std'],
        'final_debt': ['mean', 'std'],
        'final_B': ['mean', 'std'],
        'avg_tau': ['mean', 'std'],
        'total_m_usd': ['mean', 'std'],
        'total_u_usd': ['mean', 'std'],
        'avg_m_usd': ['mean', 'std'],
        'avg_u_usd': ['mean', 'std'],
    }).reset_index()

    # ========== Control Variables Summary ==========
    print("\n" + "="*70)
    print("CONTROL VARIABLES (Decision Variables per Year)")
    print("="*70)
    for year in range(1, n_years + 1):
        year_data = yearly_summary[yearly_summary['year'] == year]
        if len(year_data) == 0:
            continue
        print(f"Year {year}:")
        print(f"  τ (ticket multiplier):  μ = {year_data['avg_tau'].mean():.3f}, σ = {year_data['avg_tau'].std():.3f}")
        print(f"  m (marketing, total):   μ = ${year_data['total_m_usd'].mean():>12,.0f}, σ = ${year_data['total_m_usd'].std():>10,.0f}")
        print(f"  u (sports inv, total):  μ = ${year_data['total_u_usd'].mean():>12,.0f}, σ = ${year_data['total_u_usd'].std():>10,.0f}")
        print(f"  m (per period avg):     μ = ${year_data['avg_m_usd'].mean():>12,.0f}")
        print(f"  u (per period avg):     μ = ${year_data['avg_u_usd'].mean():>12,.0f}")
        print()

    # ========== 2.9 CVaR Risk Metrics ==========
    # Calculate risk metrics across all simulations for each year
    risk_metrics_by_year = []
    for year in range(1, n_years + 1):
        year_data = yearly_summary[yearly_summary['year'] == year]
        if len(year_data) < 2:
            continue

        # Construct cash and profit arrays (each simulation is a row)
        n_sims = len(year_data)
        cash_history = year_data['final_cash'].values.reshape(n_sims, 1)
        profit_history = year_data['cum_profit'].values.reshape(n_sims, 1)

        # Calculate risk metrics using existing function
        metrics = calculate_risk_metrics(
            cash_history=cash_history,
            profit_history=profit_history,
            alpha=0.05,  # 95% confidence
            min_cash_threshold=0.0,
        )

        # Add year identifier
        metrics['year'] = year

        # Additional calculations: min_cash across entire simulation path
        # (approximated by final_cash since we only have end-of-year snapshots)
        min_cash_values = year_data['final_cash'].values
        metrics['min_cash_p5'] = float(np.percentile(min_cash_values, 5))
        metrics['min_cash_p10'] = float(np.percentile(min_cash_values, 10))

        risk_metrics_by_year.append(metrics)

    risk_df = pd.DataFrame(risk_metrics_by_year)
    if len(risk_df) > 0:
        print("\n" + "="*70)
        print("RISK METRICS (Section 2.9)")
        print("="*70)
        print(f"Alpha = 0.05 (95% confidence level)")
        print(f"CVaR = Conditional Value at Risk = E[Loss | Loss > VaR]")
        print("-"*70)
        for _, row in risk_df.iterrows():
            print(f"Year {int(row['year'])}:")
            print(f"  Bankruptcy Prob:   {row['bankruptcy_prob']:.2%}")
            print(f"  VaR (95%):         ${row['var']:>12,.0f}")
            print(f"  CVaR (95%):        ${row['cvar']:>12,.0f}")
            print(f"  Terminal Cash μ:   ${row['terminal_cash_mean']:>12,.0f}")
            print(f"  Terminal Cash σ:   ${row['terminal_cash_std']:>12,.0f}")
            print(f"  Min Cash (P5):     ${row['min_cash_p5']:>12,.0f}")
            print(f"  Profit μ:          ${row['profit_mean']:>12,.0f}")
            print(f"  Profit σ:          ${row['profit_volatility']:>12,.0f}")
            print()

    # Flatten column names
    aggregate_stats.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                                for col in aggregate_stats.columns.values]

    # Save results
    if write:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        yearly_summary.to_csv(
            out_dir / f"multi_year_summary_{focus_team}_{n_years}y_{n_simulations_per_year}sims.csv",
            index=False
        )
        aggregate_stats.to_csv(
            out_dir / f"multi_year_aggregate_{focus_team}_{n_years}y_{n_simulations_per_year}sims.csv",
            index=False
        )
        print(f"\nResults saved to {out_dir}")

    return yearly_summary, aggregate_stats


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Task1: MPC-style dynamic system simulation (grid-search).")
    parser.add_argument("--team", type=str, default="LVA", help="Team code, e.g. LVA/NYL/IND")
    parser.add_argument(
        "--control-mode",
        type=str,
        default="policy",
        choices=["policy", "grid"],
        help="Control mode: 'policy' uses continuous (m,u) adjusted by record and discrete stage-based tau; 'grid' uses original grid-search MPC on (tau,m,u).",
    )
    parser.add_argument("--n-mc", type=int, default=10)  # Reduced from 20 for 20s target
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--periods", type=int, default=3)
    parser.add_argument(
        "--decision-every",
        type=int,
        default=1,
        help="Re-optimize (tau,m,u) every N periods. Use N>=periods for one-time decision with per-period state updates.",
    )
    parser.add_argument(
        "--fixed-tau",
        type=float,
        default=None,
        help="If set, use a fixed tau across all periods (skips MPC optimization).",
    )
    parser.add_argument(
        "--fixed-m",
        type=float,
        default=None,
        help="If set, use a fixed marketing spend per period (USD) across all periods (skips MPC optimization).",
    )
    parser.add_argument(
        "--fixed-u",
        type=float,
        default=None,
        help="If set, use a fixed sports investment per period (USD) across all periods (skips MPC optimization).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--task2-state",
        type=str,
        default=None,
        help="Path to Task2 season_initial_state.json or task1_initial_state.json (elo/B/cash).",
    )
    parser.add_argument(
        "--task3-impact",
        type=str,
        default=None,
        help="Path to Task3->Task1 exogenous shock JSON (from project/model/task3/task1_impact.py).",
    )
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel MC sampling")

    # Multi-year arguments
    parser.add_argument("--multi-year", action="store_true", help="Enable multi-year prediction mode")
    parser.add_argument("--n-years", type=int, default=1, help="Number of years to predict")
    parser.add_argument("--n-sims-per-year", type=int, default=10, help="MC simulations per year (default: 10 for 20s target)")
    parser.add_argument("--elo-shock-std", type=float, default=30.0, help="Std dev of random Elo shock between seasons (default: 30.0)")

    args = parser.parse_args(argv)

    cfg = MPCConfig(
        periods=int(args.periods),
        horizon=int(args.horizon),
        n_mc=int(args.n_mc),
        control_mode=str(args.control_mode).strip().lower(),
    )
    fixed_action: Optional[ControlAction] = None
    if args.fixed_tau is not None or args.fixed_m is not None or args.fixed_u is not None:
        fixed_action = ControlAction(
            tau=float(args.fixed_tau if args.fixed_tau is not None else 1.0),
            m=float(args.fixed_m if args.fixed_m is not None else 0.0),
            u=float(args.fixed_u if args.fixed_u is not None else 0.0),
        )

    if args.multi_year:
        # Multi-year mode
        yearly_summary, aggregate_stats = run_multi_year_mpc_simulation(
            focus_team=str(args.team).strip().upper(),
            n_years=int(args.n_years),
            n_simulations_per_year=int(args.n_sims_per_year),
            seed=int(args.seed),
            cfg=cfg,
            write=(not args.no_write),
            use_parallel=(not args.no_parallel),
            init_state_path=args.task2_state,
            task3_impact_path=args.task3_impact,
            elo_shock_std=float(args.elo_shock_std),
            decision_every=int(args.decision_every),
            fixed_action=fixed_action,
        )

        print("\n" + "="*70)
        print(f"Multi-Year Simulation Results ({args.n_years} years, {args.n_sims_per_year} sims/year)")
        print("="*70)
        print("\nAggregate Statistics by Year:")
        print(aggregate_stats.to_string(index=False))
    else:
        # Single-year mode (existing logic)
        period_log, summary = run_mpc_simulation(
            focus_team=str(args.team).strip().upper(),
            seed=int(args.seed),
            cfg=cfg,
            write=(not args.no_write),
            use_parallel=(not args.no_parallel),
            init_state_path=args.task2_state,
            task3_impact_path=args.task3_impact,
            decision_every=int(args.decision_every),
            fixed_action=fixed_action,
        )

        print(period_log.to_string(index=False))
        print("")
        print(summary.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
