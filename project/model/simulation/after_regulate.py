"""
After-regular-season model: playoffs + terminal franchise value.

Implements the ideas in `project/model.md`:
- Use Elo to Monte-Carlo simulate a season, then simulate playoffs (best-of-3 + best-of-5).
- Use repo data (brand_b0.csv, avg salary by team, 2024 revenue/valuation) to estimate:
  regular season + playoff incremental revenue, costs, EBITDA, brand end-state, and terminal value.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Import from core modules
from ..core.brands import BrandParameters, TEAM_NAME_TO_CODE, update_brand
from ..core.cost import GeneralAdminParameters, calculate_total_cost
from ..core.income import RevenueParameters, calculate_period_revenue


ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT_DIR / "project" / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CONFIG_DIR = PROCESSED_DIR / "config"
ELO_DIR = PROCESSED_DIR / "elo"
MPC_DIR = PROCESSED_DIR / "mpc"
FINANCIAL_DIR = PROCESSED_DIR / "financial"
OTHER_CLEAN_DIR = PROCESSED_DIR / "other_clean"


@dataclass(frozen=True)
class PlayoffParams:
    best_of_first: int = 3
    best_of_semis: int = 5
    best_of_finals: int = 5
    ticket_multiplier: float = 1.25  # playoffs ticket premium (multiplier on team avg ticket price)
    stage_multiplier: float = 1.35  # playoffs demand multiplier
    rivalry_factor: float = 0.20
    importance: float = 0.90


@dataclass(frozen=True)
class SeasonParams:
    games_per_team: int = 40
    home_games_per_team: int = 20


@dataclass(frozen=True)
class TerminalValueParams:
    mu_ebitda: float
    mu_brand: float
    intercept: float = 0.0


def _team_name_to_code(name: object) -> Optional[str]:
    s = str(name).strip()
    if not s or s.lower() == "league average":
        return None
    if s in TEAM_NAME_TO_CODE:
        return TEAM_NAME_TO_CODE[s]
    if "(" in s and ")" in s:
        inner = s[s.find("(") + 1 : s.rfind(")")].strip()
        if inner in TEAM_NAME_TO_CODE:
            return TEAM_NAME_TO_CODE[inner]
    if len(s) in (2, 3, 4) and s.isupper():
        return s
    return None


def _read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.is_file():
        return pd.read_csv(p)
    if (ROOT_DIR / p).is_file():
        return pd.read_csv(ROOT_DIR / p)
    raise FileNotFoundError(f"CSV not found: {path}")


def load_brand_b0(path: str | Path) -> pd.DataFrame:
    df = _read_csv(path)
    df["team"] = df["team"].astype(str).str.strip()
    return df


def load_elo_ratings(path: str | Path, teams: List[str], *, base_elo: float) -> Dict[str, float]:
    df = _read_csv(path)
    df["team"] = df["team"].astype(str).str.strip()
    d = dict(zip(df["team"], pd.to_numeric(df["elo"], errors="coerce").fillna(base_elo)))
    return {t: float(d.get(t, base_elo)) for t in teams}


def load_elo_config(path: str | Path) -> Dict[str, float]:
    cfg = _read_csv(path).iloc[0].to_dict()
    return {k: float(cfg[k]) for k in ("base_elo", "home_advantage", "k", "season_carryover") if k in cfg}


def load_team_revenue_2024() -> pd.Series:
    p = OTHER_CLEAN_DIR / "wnba-teams-with-the-highest-revenue-2024_clean.csv"
    if not p.is_file():
        return pd.Series(dtype=float)
    df = pd.read_csv(p)
    df = df[df["Year"] == 2024].copy()
    df["team"] = df["Team"].map(_team_name_to_code)
    df["team_revenue_usd"] = pd.to_numeric(df["TeamRevenueUSD"], errors="coerce")
    return df.dropna(subset=["team", "team_revenue_usd"]).set_index("team")["team_revenue_usd"]


def load_team_valuation_2024() -> pd.Series:
    p = OTHER_CLEAN_DIR / "value-of-teams-in-the-wnba-2024_clean.csv"
    if not p.is_file():
        return pd.Series(dtype=float)
    df = pd.read_csv(p)
    df = df[df["Year"] == 2024].copy()
    df["team"] = df["Team"].map(_team_name_to_code)
    df["team_value_usd"] = pd.to_numeric(df["TeamValueUSD"], errors="coerce")
    return df.dropna(subset=["team", "team_value_usd"]).set_index("team")["team_value_usd"]


def load_avg_salary_by_team_2025() -> pd.Series:
    p = OTHER_CLEAN_DIR / "average-player-salary-in-the-wnba-by-team-2025_clean.csv"
    if not p.is_file():
        return pd.Series(dtype=float)
    df = pd.read_csv(p)
    df = df[df["Year"] == 2025].copy()
    df["team"] = df["Team"].map(_team_name_to_code)
    df["avg_salary_usd"] = pd.to_numeric(df["AvgSalaryUSD"], errors="coerce")
    return df.dropna(subset=["team", "avg_salary_usd"]).set_index("team")["avg_salary_usd"]


def min_max_scale(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mn = float(s.min())
    mx = float(s.max())
    if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
        return pd.Series([0.5] * len(s), index=s.index, dtype=float)
    return ((s - mn) / (mx - mn)).clip(0.0, 1.0).astype(float)


def elo_win_prob(home_elo: float, away_elo: float, home_advantage: float) -> float:
    delta = (home_elo - away_elo) + home_advantage
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


def simulate_regular_season(
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    elos_init: np.ndarray,
    *,
    k: float,
    home_advantage: float,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray]:
    elos = elos_init.copy()
    wins = np.zeros(len(elos), dtype=float)
    order = rng.permutation(len(home_idx))

    for g in order:
        h = int(home_idx[g])
        a = int(away_idx[g])

        p_home = elo_win_prob(elos[h], elos[a], home_advantage)
        y = 1.0 if rng.rand() < p_home else 0.0

        wins[h] += y
        wins[a] += 1.0 - y

        delta = k * (y - p_home)
        elos[h] += delta
        elos[a] -= delta

    return wins, elos


def rank_teams(teams: List[str], wins: np.ndarray, elos: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"team": teams, "wins": wins, "elo": elos})
    df = df.sort_values(["wins", "elo"], ascending=[False, False]).reset_index(drop=True)
    df["seed"] = np.arange(1, len(df) + 1)
    return df


def _home_pattern(best_of: int) -> List[bool]:
    if best_of == 3:
        return [True, True, False]  # 2-1
    if best_of == 5:
        return [True, True, False, False, True]  # 2-2-1
    raise ValueError(f"Unsupported best_of: {best_of}")


def simulate_series(
    higher_idx: int,
    lower_idx: int,
    elos: np.ndarray,
    *,
    k: float,
    home_advantage: float,
    best_of: int,
    rng: np.random.RandomState,
    playoff_wins: np.ndarray,
    playoff_games: np.ndarray,
    playoff_home_games: np.ndarray,
    playoff_away_games: np.ndarray,
) -> int:
    need = best_of // 2 + 1
    w_hi = 0
    w_lo = 0

    for hi_home in _home_pattern(best_of):
        home = higher_idx if hi_home else lower_idx
        away = lower_idx if hi_home else higher_idx

        p_home = elo_win_prob(elos[home], elos[away], home_advantage)
        y = 1.0 if rng.rand() < p_home else 0.0
        winner = home if y == 1.0 else away

        playoff_games[home] += 1.0
        playoff_games[away] += 1.0
        playoff_home_games[home] += 1.0
        playoff_away_games[away] += 1.0
        playoff_wins[winner] += 1.0

        delta = k * (y - p_home)
        elos[home] += delta
        elos[away] -= delta

        if winner == higher_idx:
            w_hi += 1
        else:
            w_lo += 1
        if w_hi >= need or w_lo >= need:
            break

    return higher_idx if w_hi >= need else lower_idx


def simulate_playoffs(
    seed_order: List[int],
    seeds_by_team: np.ndarray,
    elos: np.ndarray,
    *,
    k: float,
    home_advantage: float,
    playoff_params: PlayoffParams,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    n = len(elos)
    made_playoffs = np.zeros(n, dtype=float)
    playoff_wins = np.zeros(n, dtype=float)
    playoff_games = np.zeros(n, dtype=float)
    playoff_home_games = np.zeros(n, dtype=float)
    playoff_away_games = np.zeros(n, dtype=float)

    for idx in seed_order:
        made_playoffs[idx] = 1.0

    # Round 1: 1v8, 4v5, 2v7, 3v6
    s = seed_order
    r1_pairs = [(s[0], s[7]), (s[3], s[4]), (s[1], s[6]), (s[2], s[5])]
    r1_winners = []
    for hi, lo in r1_pairs:
        r1_winners.append(
            simulate_series(
                hi,
                lo,
                elos,
                k=k,
                home_advantage=home_advantage,
                best_of=playoff_params.best_of_first,
                rng=rng,
                playoff_wins=playoff_wins,
                playoff_games=playoff_games,
                playoff_home_games=playoff_home_games,
                playoff_away_games=playoff_away_games,
            )
        )

    # Semis
    semi_pairs = [(r1_winners[0], r1_winners[1]), (r1_winners[2], r1_winners[3])]
    semi_winners = []
    for t1, t2 in semi_pairs:
        hi = t1 if seeds_by_team[t1] < seeds_by_team[t2] else t2
        lo = t2 if hi == t1 else t1
        semi_winners.append(
            simulate_series(
                hi,
                lo,
                elos,
                k=k,
                home_advantage=home_advantage,
                best_of=playoff_params.best_of_semis,
                rng=rng,
                playoff_wins=playoff_wins,
                playoff_games=playoff_games,
                playoff_home_games=playoff_home_games,
                playoff_away_games=playoff_away_games,
            )
        )

    # Finals
    t1, t2 = semi_winners
    hi = t1 if seeds_by_team[t1] < seeds_by_team[t2] else t2
    lo = t2 if hi == t1 else t1
    champion = simulate_series(
        hi,
        lo,
        elos,
        k=k,
        home_advantage=home_advantage,
        best_of=playoff_params.best_of_finals,
        rng=rng,
        playoff_wins=playoff_wins,
        playoff_games=playoff_games,
        playoff_home_games=playoff_home_games,
        playoff_away_games=playoff_away_games,
    )

    return made_playoffs, playoff_wins, playoff_games, playoff_home_games, playoff_away_games, int(champion)


def run_monte_carlo_season_and_playoffs(
    teams: List[str],
    *,
    elo_by_team: Dict[str, float],
    k: float,
    home_advantage: float,
    n_sims: int,
    seed: int,
    playoff_params: PlayoffParams,
) -> pd.DataFrame:
    n = len(teams)
    team_to_idx = {t: i for i, t in enumerate(teams)}
    elos_init = np.asarray([elo_by_team[t] for t in teams], dtype=float)

    home_idx, away_idx = build_double_round_robin_indices(n)
    games_per_team_sim = int((n - 1) * 2)

    reg_wins_sum = np.zeros(n, dtype=float)
    reg_wins_sq = np.zeros(n, dtype=float)
    made_playoffs_sum = np.zeros(n, dtype=float)
    playoff_wins_sum = np.zeros(n, dtype=float)
    playoff_games_sum = np.zeros(n, dtype=float)
    playoff_home_games_sum = np.zeros(n, dtype=float)
    playoff_away_games_sum = np.zeros(n, dtype=float)
    champ_count = np.zeros(n, dtype=float)

    rng = np.random.RandomState(seed)

    for _ in range(int(n_sims)):
        wins, elos_end = simulate_regular_season(
            home_idx,
            away_idx,
            elos_init,
            k=k,
            home_advantage=home_advantage,
            rng=rng,
        )

        standings = rank_teams(teams, wins, elos_end)
        top8 = standings.head(8).copy()
        seed_order = [team_to_idx[t] for t in top8["team"].tolist()]

        seeds_by_team = np.full(n, 999, dtype=int)
        for s_i, idx in enumerate(seed_order, start=1):
            seeds_by_team[idx] = s_i

        made_po, po_w, po_g, po_hg, po_ag, champ = simulate_playoffs(
            seed_order,
            seeds_by_team,
            elos_end.copy(),
            k=k,
            home_advantage=home_advantage,
            playoff_params=playoff_params,
            rng=rng,
        )

        reg_wins_sum += wins
        reg_wins_sq += wins ** 2
        made_playoffs_sum += made_po
        playoff_wins_sum += po_w
        playoff_games_sum += po_g
        playoff_home_games_sum += po_hg
        playoff_away_games_sum += po_ag
        champ_count[champ] += 1.0

    mean_wins = reg_wins_sum / n_sims
    var_wins = (reg_wins_sq / n_sims) - (mean_wins ** 2)
    std_wins = np.sqrt(np.maximum(var_wins, 0.0))

    out = pd.DataFrame(
        {
            "team": teams,
            # pandas==0.23 + newer numpy breaks scalar broadcasting; keep same-length arrays.
            "games_per_team_sim": [games_per_team_sim] * n,
            "mean_wins_sim": mean_wins,
            "std_wins_sim": std_wins,
            "mean_win_pct": mean_wins / games_per_team_sim,
            "playoff_prob": made_playoffs_sum / n_sims,
            "mean_playoff_wins": playoff_wins_sum / n_sims,
            "mean_playoff_games": playoff_games_sum / n_sims,
            "mean_playoff_home_games": playoff_home_games_sum / n_sims,
            "mean_playoff_away_games": playoff_away_games_sum / n_sims,
            "champion_prob": champ_count / n_sims,
        }
    )
    return out.sort_values("mean_win_pct", ascending=False).reset_index(drop=True)


def build_roster(avg_salary_usd: float, *, roster_size: int = 12) -> List[Dict[str, float]]:
    return [{"salary": float(avg_salary_usd), "on_roster": 1, "equity_subsidy": 0.0} for _ in range(int(roster_size))]


def estimate_ebitda(
    revenue: float,
    costs: Dict[str, float],
    *,
    tax_rate: float,
) -> float:
    tax = tax_rate * float(revenue)
    ga_ex_tax = float(costs["ga_cost"]) - tax
    return float(revenue) - float(costs["salary_cost"]) - float(costs["venue_cost"]) - float(costs["sports_ops_cost"]) - ga_ex_tax


def fit_terminal_value_params(
    brand_b0: pd.DataFrame,
    *,
    avg_salary_by_team: pd.Series,
    default_mu: float = 10.0,
    default_mu_brand: float = 3.0e7,
) -> TerminalValueParams:
    valuation = load_team_valuation_2024()
    revenue = load_team_revenue_2024()
    if valuation.empty or revenue.empty:
        return TerminalValueParams(mu_ebitda=default_mu, mu_brand=default_mu_brand, intercept=0.0)

    b0 = brand_b0.set_index("team")["B0"]
    m = pd.concat([valuation.rename("V"), revenue.rename("R"), b0.rename("B0")], axis=1, sort=False).dropna()
    if len(m) < 6:
        return TerminalValueParams(mu_ebitda=default_mu, mu_brand=default_mu_brand, intercept=0.0)

    tax_rate = GeneralAdminParameters().tax_rate
    ebitda = []
    for team, row in m.iterrows():
        avg_salary = float(avg_salary_by_team.get(team, float(avg_salary_by_team.mean()) if len(avg_salary_by_team) else 120000.0))
        roster = build_roster(avg_salary)

        # Use team "avg_attendance" as a proxy for home attendance. Home games assumed 20.
        avg_att = float(brand_b0.set_index("team").get("avg_attendance", pd.Series()).get(team, np.nan))
        if not np.isfinite(avg_att):
            avg_att = float(brand_b0["avg_attendance"].median()) if "avg_attendance" in brand_b0 else 10000.0

        total_att = avg_att * 20.0
        costs = calculate_total_cost(
            roster=roster,
            total_attendance=total_att,
            num_home_games=20,
            revenue=float(row["R"]),
            debt_balance=0.0,
            marketing_spend=max(300000.0, min(0.02 * float(row["R"]), 1500000.0)),
            num_away_games=20,
        )
        ebitda.append(estimate_ebitda(float(row["R"]), costs, tax_rate=tax_rate))

    m = m.copy()
    m["EBITDA"] = ebitda
    m = m.replace([np.inf, -np.inf], np.nan).dropna(subset=["EBITDA"])
    if len(m) < 6:
        return TerminalValueParams(mu_ebitda=default_mu, mu_brand=default_mu_brand, intercept=0.0)

    X = np.column_stack([m["EBITDA"].values, m["B0"].values, np.ones(len(m))])
    y = m["V"].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    mu, mu_b, intercept = float(beta[0]), float(beta[1]), float(beta[2])

    if not np.isfinite(mu) or mu <= 0:
        mu = default_mu
    if not np.isfinite(mu_b) or mu_b <= 0:
        mu_b = default_mu_brand
    if not np.isfinite(intercept):
        intercept = 0.0

    return TerminalValueParams(mu_ebitda=mu, mu_brand=mu_b, intercept=intercept)


def estimate_financials_and_value(
    mc: pd.DataFrame,
    brand_b0: pd.DataFrame,
    *,
    avg_salary_by_team: pd.Series,
    season_params: SeasonParams,
    playoff_params: PlayoffParams,
    terminal_params: TerminalValueParams,
) -> pd.DataFrame:
    rev_params = RevenueParameters()
    tax_rate = GeneralAdminParameters().tax_rate

    b = brand_b0.set_index("team")
    ticket_usd = pd.to_numeric(b["avg_ticket_price_usd"], errors="coerce")
    league_ticket = float(ticket_usd.mean()) if ticket_usd.notna().any() else float(rev_params.p0)
    ticket_index = (ticket_usd / league_ticket).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    ticket_shrink = 0.25  # keep secondary-market prices from exploding team revenue
    ticket_mult = (1.0 + ticket_shrink * (ticket_index - 1.0)).clip(0.80, 1.20).astype(float)
    capacity = pd.to_numeric(b["arena_capacity"], errors="coerce").fillna(10000.0)
    brand = pd.to_numeric(b["B0"], errors="coerce").fillna(1.0)

    star_raw = pd.to_numeric(b.get("ig_followers", pd.Series(index=b.index, dtype=float)), errors="coerce").fillna(0.0)
    star_01 = min_max_scale(np.log1p(star_raw))
    star_power = 0.2 + 1.0 * star_01  # map to ~[0.2, 1.2]

    # League-average opponent profile (for revenue expectations)
    opp_strength = float(mc["mean_win_pct"].mean())
    opp_star = float(star_power.mean()) if len(star_power) else 0.6

    revenue_scale = fit_revenue_scale_2024(
        ticket_mult=ticket_mult,
        star_power=star_power,
        brand=brand,
        capacity=capacity,
        season_params=season_params,
        opp_strength=opp_strength,
        opp_star=opp_star,
    )

    out_rows = []
    for _, r in mc.iterrows():
        team = str(r["team"])
        win_pct = float(r["mean_win_pct"])

        g_home = float(season_params.home_games_per_team)
        g_away = float(season_params.games_per_team - season_params.home_games_per_team)

        # Regular season revenue (deterministic expectation)
        games_df = pd.DataFrame(
            {
                "is_home": [True] * int(g_home),
                "opponent_strength": [opp_strength] * int(g_home),
                "opponent_star_power": [opp_star] * int(g_home),
                "rivalry_factor": [0.15] * int(g_home),
                "game_importance": [0.50] * int(g_home),
            }
        )
        reg = calculate_period_revenue(
            games=games_df,
            ticket_price=float(ticket_mult.get(team, 1.0)),
            team_strength=win_pct,
            team_star_power=float(star_power.get(team, opp_star)),
            brand_value=float(brand.get(team, 1.0)),
            capacity=float(capacity.get(team, 10000.0)),
            stage_multiplier=1.0,
            total_league_revenue=50_000_000.0,
            num_teams=len(mc),
            add_noise=False,
        )
        reg_total_revenue = float(reg["total_revenue"]) * revenue_scale

        # Playoffs: only incremental ticket+merch from expected home games
        po_home_games = float(r["mean_playoff_home_games"])
        if po_home_games > 0:
            po_game = pd.DataFrame(
                {
                    "is_home": [True],
                    "opponent_strength": [opp_strength],
                    "opponent_star_power": [opp_star],
                    "rivalry_factor": [playoff_params.rivalry_factor],
                    "game_importance": [playoff_params.importance],
                }
            )
            po = calculate_period_revenue(
                games=po_game,
                ticket_price=float(ticket_mult.get(team, 1.0)) * playoff_params.ticket_multiplier,
                team_strength=win_pct,
                team_star_power=float(star_power.get(team, opp_star)),
                brand_value=float(brand.get(team, 1.0)),
                capacity=float(capacity.get(team, 10000.0)),
                stage_multiplier=playoff_params.stage_multiplier,
                total_league_revenue=0.0,
                num_teams=len(mc),
                add_noise=False,
            )
            po_ticket = float(po["ticket_revenue"]) * po_home_games * revenue_scale
            po_merch = float(po["merchandise_revenue"]) * po_home_games * revenue_scale
            po_att = float(po["total_attendance"]) * po_home_games
        else:
            po_ticket = 0.0
            po_merch = 0.0
            po_att = 0.0

        total_revenue = reg_total_revenue + po_ticket + po_merch
        total_attendance = float(reg["total_attendance"]) + po_att

        avg_salary = float(avg_salary_by_team.get(team, float(avg_salary_by_team.mean()) if len(avg_salary_by_team) else 120000.0))
        roster = build_roster(avg_salary)

        po_away_games = float(r["mean_playoff_away_games"])
        costs = calculate_total_cost(
            roster=roster,
            total_attendance=total_attendance,
            num_home_games=g_home + po_home_games,
            revenue=total_revenue,
            debt_balance=0.0,
            marketing_spend=max(300000.0, min(0.02 * total_revenue, 1500000.0)),
            num_away_games=g_away + po_away_games,
        )

        profit = float(total_revenue) - float(costs["total_cost"])
        ebitda = estimate_ebitda(total_revenue, costs, tax_rate=tax_rate)

        marketing_intensity = float(max(0.0, min(10.0, (0.02 * total_revenue) / 100000.0)))
        wins_scaled = win_pct * float(season_params.games_per_team)
        total_wins = wins_scaled + float(r["mean_playoff_wins"])
        b_end = update_brand(
            float(brand.get(team, 1.0)),
            total_wins,
            marketing_intensity,
            float(star_power.get(team, opp_star)),
            BrandParameters(),
        )

        v_end = terminal_params.intercept + terminal_params.mu_ebitda * ebitda + terminal_params.mu_brand * b_end

        out_rows.append(
            {
                "team": team,
                "revenue_scale": float(revenue_scale),
                "mean_win_pct": win_pct,
                "playoff_prob": float(r["playoff_prob"]),
                "champion_prob": float(r["champion_prob"]),
                "B0": float(brand.get(team, 1.0)),
                "B_end": float(b_end),
                "regular_revenue_usd": float(reg_total_revenue),
                "playoff_ticket_revenue_usd": float(po_ticket),
                "playoff_merch_revenue_usd": float(po_merch),
                "total_revenue_usd": float(total_revenue),
                "total_cost_usd": float(costs["total_cost"]),
                "profit_usd": float(profit),
                "ebitda_usd": float(ebitda),
                "terminal_value_usd": float(v_end),
            }
        )

    out = pd.DataFrame(out_rows)
    return out.sort_values("terminal_value_usd", ascending=False).reset_index(drop=True)


def fit_revenue_scale_2024(
    *,
    ticket_mult: pd.Series,
    star_power: pd.Series,
    brand: pd.Series,
    capacity: pd.Series,
    season_params: SeasonParams,
    opp_strength: float,
    opp_star: float,
) -> float:
    """
    Scale the revenue model to roughly match 2024 team revenue levels.

    Reason: income.py uses a "market ticket price" baseline; in practice, not all of that
    secondary-market price is captured by teams. A single multiplicative scale keeps the
    model usable while staying anchored to real 2024 revenue magnitudes.
    """
    revenue_2024 = load_team_revenue_2024()
    if revenue_2024.empty:
        return 1.0

    g_home = int(season_params.home_games_per_team)
    games_df = pd.DataFrame(
        {
            "is_home": [True] * g_home,
            "opponent_strength": [float(opp_strength)] * g_home,
            "opponent_star_power": [float(opp_star)] * g_home,
            "rivalry_factor": [0.15] * g_home,
            "game_importance": [0.50] * g_home,
        }
    )

    preds = []
    actuals = []
    for team, actual in revenue_2024.items():
        if team not in ticket_mult.index:
            continue

        reg = calculate_period_revenue(
            games=games_df,
            ticket_price=float(ticket_mult.get(team, 1.0)),
            team_strength=0.5,
            team_star_power=float(star_power.get(team, opp_star)),
            brand_value=float(brand.get(team, 1.0)),
            capacity=float(capacity.get(team, 10000.0)),
            stage_multiplier=1.0,
            total_league_revenue=50_000_000.0,
            num_teams=max(int(len(revenue_2024)), 1),
            add_noise=False,
        )

        pred = float(reg["total_revenue"])
        if pred > 0 and np.isfinite(pred) and np.isfinite(float(actual)):
            preds.append(pred)
            actuals.append(float(actual))

    if not preds or sum(preds) <= 0:
        return 1.0
    return float(sum(actuals) / sum(preds))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="After-regular-season: playoffs + terminal value (Elo Monte Carlo).")
    parser.add_argument("--brand-b0", type=str, default=str(CONFIG_DIR / "brand_b0.csv"))
    parser.add_argument("--elo-ratings", type=str, default=str(ELO_DIR / "elo_final_ratings.csv"))
    parser.add_argument("--elo-config", type=str, default=str(CONFIG_DIR / "elo_config.csv"))
    parser.add_argument("--n-sims", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=str(FINANCIAL_DIR / "after_regulate_summary.csv"))
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--mu-ebitda", type=float, default=None, help="Override EBITDA multiple (mu)")
    parser.add_argument("--mu-brand", type=float, default=None, help="Override brand coefficient (mu_B)")
    parser.add_argument("--mu-intercept", type=float, default=None, help="Override valuation intercept")
    args = parser.parse_args(argv)

    brand_b0 = load_brand_b0(args.brand_b0)
    teams = brand_b0["team"].astype(str).str.strip().tolist()
    teams = [t for t in teams if t]

    elo_cfg = load_elo_config(args.elo_config)
    base_elo = float(elo_cfg.get("base_elo", 1500.0))
    k = float(elo_cfg.get("k", 20.0))
    home_advantage = float(elo_cfg.get("home_advantage", 0.0))

    elo_by_team = load_elo_ratings(args.elo_ratings, teams, base_elo=base_elo)
    avg_salary_by_team = load_avg_salary_by_team_2025()

    playoff_params = PlayoffParams()
    mc = run_monte_carlo_season_and_playoffs(
        teams,
        elo_by_team=elo_by_team,
        k=k,
        home_advantage=home_advantage,
        n_sims=int(args.n_sims),
        seed=int(args.seed),
        playoff_params=playoff_params,
    )

    terminal_fit = fit_terminal_value_params(brand_b0, avg_salary_by_team=avg_salary_by_team)
    terminal_params = TerminalValueParams(
        mu_ebitda=float(args.mu_ebitda) if args.mu_ebitda is not None else terminal_fit.mu_ebitda,
        mu_brand=float(args.mu_brand) if args.mu_brand is not None else terminal_fit.mu_brand,
        intercept=float(args.mu_intercept) if args.mu_intercept is not None else terminal_fit.intercept,
    )

    season_params = SeasonParams()
    summary = estimate_financials_and_value(
        mc,
        brand_b0,
        avg_salary_by_team=avg_salary_by_team,
        season_params=season_params,
        playoff_params=playoff_params,
        terminal_params=terminal_params,
    )

    val_2024 = load_team_valuation_2024()
    if not val_2024.empty:
        m = summary.set_index("team")[["terminal_value_usd"]].join(val_2024.rename("valuation_2024_usd")).dropna()
        if len(m) >= 6:
            spearman = float(m["terminal_value_usd"].rank(ascending=False).corr(m["valuation_2024_usd"].rank(ascending=False)))
            print("Spearman(rank_terminal_value, rank_valuation_2024): %.3f" % spearman)

    print("")
    print("Terminal params used:")
    print("  mu_ebitda=%.3f, mu_brand=%.3e, intercept=%.3e" % (terminal_params.mu_ebitda, terminal_params.mu_brand, terminal_params.intercept))
    print("")
    print(summary[["team", "mean_win_pct", "playoff_prob", "champion_prob", "profit_usd", "terminal_value_usd"]].head(12).to_string(index=False))

    if not args.no_write:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.out, index=False)
        mc.to_csv(Path(args.out).with_name("after_regulate_mc_metrics.csv"), index=False)
        print("")
        print(f"Wrote: {args.out}")
        print(f"Wrote: {Path(args.out).with_name('after_regulate_mc_metrics.csv')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
