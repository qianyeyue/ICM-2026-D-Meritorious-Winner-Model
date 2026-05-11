"""
Elo model built only from `project/data/wnba_gamelogs_2015_2025.csv`.

The source CSV is *team-level* game logs (each real game appears twice, once for
each team). For Elo fitting we deduplicate by keeping only rows where `Home==1`
(the home-team perspective), which yields one row per game:

    home_team = Team
    away_team = Opp
    home_win  = (W/L == "W")

Win probability (home team):
    p = 1 / (1 + 10^(-delta/400))
    delta = S_home - S_away + H + C * I

This module focuses on:
- Predicting per-game win probability
- Updating Elo game-by-game
- Aggregating predicted win% per team/season
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
PROJECT_DATA_DIR = ROOT_DIR / "project" / "data"
RAW_DIR = PROJECT_DATA_DIR / "raw"
ELO_OUTPUT_DIR = PROJECT_DATA_DIR / "processed" / "elo"
CONFIG_DIR = PROJECT_DATA_DIR / "processed" / "config"


def home_advantage_from_p(p_home: float) -> float:
    """
    Calibrate H so that when S_home == S_away, p_home ~= p_home.

    p = 1 / (1 + 10^(-H/400))  ->  H = 400 * log10(p/(1-p))
    """
    if not (0.0 < p_home < 1.0):
        raise ValueError("p_home must be in (0, 1)")
    return float(400.0 * np.log10(p_home / (1.0 - p_home)))


def _win_flag(wl: str) -> int:
    wl = str(wl).strip().upper()
    if wl == "W":
        return 1
    if wl == "L":
        return 0
    raise ValueError(f"Unexpected W/L value: {wl!r}")


def load_team_gamelogs(csv_path: str | Path) -> pd.DataFrame:
    """Load the raw team-level game log CSV."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        # Common IDE pitfall: running from `project/model` makes relative defaults fail.
        # Try resolving relative to repo root and `project/data`.
        candidates = [
            csv_path,
            ROOT_DIR / csv_path,
            PROJECT_DATA_DIR / csv_path.name,
        ]
        for cand in candidates:
            if cand.is_file():
                csv_path = cand
                break
        else:
            tried = ", ".join(str(c) for c in candidates)
            raise FileNotFoundError(f"CSV not found. Tried: {tried}")
    df = pd.read_csv(csv_path)

    required = {"Season", "Team", "Date", "Home", "Opp", "W/L", "Tm_Pts", "Opp_Pts"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return df


def build_unique_games_from_gamelogs(team_logs: pd.DataFrame) -> pd.DataFrame:
    """
    Convert team-level game logs into unique games (one row per real game).

    Strategy: keep only home-team rows (`Home == 1`).
    """
    df = team_logs.copy()

    # pandas==0.23.x (older) does not support nullable integer dtype "Int64".
    df["Season"] = pd.to_numeric(df["Season"], errors="coerce")
    df["Home"] = pd.to_numeric(df["Home"], errors="coerce")
    df = df[df["Home"] == 1].copy()

    # `infer_datetime_format` is deprecated in newer pandas; the default parser is sufficient here.
    df["date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["date"].notna()].copy()

    df["home_team"] = df["Team"].astype(str).str.strip()
    df["away_team"] = df["Opp"].astype(str).str.strip()
    df["home_score"] = pd.to_numeric(df["Tm_Pts"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["Opp_Pts"], errors="coerce")
    df["home_win"] = df["W/L"].map(_win_flag).astype(int)

    # Optional margin column (already from home-team perspective after filtering)
    if "win_margin" in df.columns:
        df["margin"] = pd.to_numeric(df["win_margin"], errors="coerce")
    else:
        df["margin"] = df["home_score"] - df["away_score"]

    games = df[
        ["Season", "date", "home_team", "away_team", "home_score", "away_score", "home_win", "margin"]
    ].rename(columns={"Season": "season"})

    games = games.dropna(subset=["season", "home_team", "away_team", "home_score", "away_score", "home_win"])
    games["season"] = games["season"].astype(int)
    games = games.sort_values(["season", "date", "home_team", "away_team"]).reset_index(drop=True)

    # De-dup safety: keep first exact match of the core fields
    games = games.drop_duplicates(
        subset=["season", "date", "home_team", "away_team", "home_score", "away_score"],
        keep="first",
    ).reset_index(drop=True)

    return games


@dataclass(frozen=True)
class EloConfig:
    k: float = 20.0
    base_elo: float = 1500.0
    season_carryover: float = 0.75  # 0=full reset, 1=full carryover
    home_advantage: float = 0.0  # H in Elo points

    # Dynamic feedback parameters (Task 1 enhancement)
    alpha_u: float = 50.0  # Investment effect: Elo points per ln(1+u_million) [INCREASED from 12.0 to 50.0]
    phi: float = 5.0  # Fatigue penalty: Elo points per unit fatigue


class EloModel:
    def __init__(self, config: EloConfig):
        if config.k <= 0:
            raise ValueError("k must be > 0")
        if not (0.0 <= config.season_carryover <= 1.0):
            raise ValueError("season_carryover must be in [0, 1]")
        self.config = config

        self.ratings_: Dict[str, float] = {}
        self.history_: Optional[pd.DataFrame] = None

    def _get_rating(self, team: str) -> float:
        return float(self.ratings_.get(team, self.config.base_elo))

    def _set_rating(self, team: str, rating: float) -> None:
        self.ratings_[team] = float(rating)

    def predict_home_win_prob(
        self,
        home_elo: float,
        away_elo: float,
        *,
        clutch: float = 0.0,
        importance: float = 0.0,
    ) -> float:
        """
        Home win probability using the user's formula.

        delta = S_home - S_away + H + C*I
        p = 1 / (1 + 10^(-delta/400))
        """
        delta = (home_elo - away_elo) + self.config.home_advantage + (clutch * importance)
        return float(1.0 / (1.0 + 10.0 ** (-delta / 400.0)))

    def update_elo_with_dynamics(
        self,
        current_elo: float,
        actual_result: float,
        predicted_prob: float,
        *,
        investment_usd: float = 0.0,
        fatigue: float = 0.0,
    ) -> float:
        """
        Update Elo with investment and fatigue effects.

        S_{g+1} = S_g + K(y_g - p_g) + α_u·ln(1+u_t) - φ·Fatigue_g

        Parameters
        ----------
        current_elo : float
            Current Elo rating
        actual_result : float
            Actual game result (1=win, 0=loss)
        predicted_prob : float
            Predicted win probability
        investment_usd : float, default=0.0
            Sports investment in USD (u_t)
        fatigue : float, default=0.0
            Fatigue index (Fatigue_g)

        Returns
        -------
        float
            Updated Elo rating
        """
        # Base Elo update
        base_update = self.config.k * (actual_result - predicted_prob)

        # Investment boost: α_u·ln(1+u_million)
        u_million = investment_usd / 1_000_000.0
        investment_boost = self.config.alpha_u * float(np.log1p(max(u_million, 0.0)))

        # Fatigue penalty: -φ·Fatigue_g
        fatigue_penalty = self.config.phi * max(fatigue, 0.0)

        new_elo = current_elo + base_update + investment_boost - fatigue_penalty
        return float(new_elo)

    def _maybe_roll_season(self, season: int, prev_season: Optional[int]) -> None:
        if prev_season is None or season == prev_season:
            return

        # Regress all existing teams back toward base_elo at season boundary.
        base = self.config.base_elo
        carry = self.config.season_carryover
        for team, elo in list(self.ratings_.items()):
            self.ratings_[team] = float(base + (elo - base) * carry)

    def fit(self, games: pd.DataFrame) -> pd.DataFrame:
        """
        Fit Elo sequentially (chronological) and store per-game predictions & ratings.

        Required columns in `games`:
        - season, date, home_team, away_team, home_win
        """
        required = {"season", "date", "home_team", "away_team", "home_win"}
        missing = required - set(games.columns)
        if missing:
            raise ValueError(f"games missing columns: {sorted(missing)}")

        games_sorted = games.sort_values(["season", "date", "home_team", "away_team"]).reset_index(drop=True)

        rows = []
        prev_season: Optional[int] = None

        for _, g in games_sorted.iterrows():
            season = int(g["season"])
            self._maybe_roll_season(season, prev_season)
            prev_season = season

            home = str(g["home_team"])
            away = str(g["away_team"])
            y = int(g["home_win"])

            home_pre = self._get_rating(home)
            away_pre = self._get_rating(away)

            p_home = self.predict_home_win_prob(home_pre, away_pre)

            k = self.config.k
            delta = k * (y - p_home)
            home_post = home_pre + delta
            away_post = away_pre - delta

            self._set_rating(home, home_post)
            self._set_rating(away, away_post)

            rows.append(
                {
                    "season": season,
                    "date": g["date"],
                    "home_team": home,
                    "away_team": away,
                    "home_win": y,
                    "p_home": p_home,
                    "home_elo_pre": home_pre,
                    "away_elo_pre": away_pre,
                    "home_elo_post": home_post,
                    "away_elo_post": away_post,
                }
            )

        history = pd.DataFrame(rows)
        self.history_ = history
        return history

    def team_season_summary(self, history: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Aggregate predicted win% and actual win% per team/season using per-game p_home.
        """
        if history is None:
            if self.history_ is None:
                raise ValueError("No history available; call fit() first.")
            history = self.history_

        # Build per-team rows for each game (home + away)
        home_rows = history[["season", "home_team", "home_win", "p_home"]].copy()
        home_rows = home_rows.rename(
            columns={"home_team": "team", "home_win": "win", "p_home": "p_win"}
        )

        away_rows = history[["season", "away_team", "home_win", "p_home"]].copy()
        away_rows["win"] = 1 - away_rows["home_win"]
        away_rows["p_win"] = 1.0 - away_rows["p_home"]
        away_rows = away_rows.rename(columns={"away_team": "team"})[["season", "team", "win", "p_win"]]

        team_games = pd.concat([home_rows, away_rows], ignore_index=True)

        # pandas==0.23.x does not support "named aggregation" syntax.
        agg = team_games.groupby(["season", "team"]).agg({"win": ["size", "sum"], "p_win": "sum"}).reset_index()
        agg.columns = ["season", "team", "games", "wins", "exp_wins"]
        agg["win_pct"] = agg["wins"] / agg["games"]
        agg["exp_win_pct"] = agg["exp_wins"] / agg["games"]

        # End-of-season Elo from history (last appearance per team in season)
        # We reconstruct from pre/post fields:
        end_parts = []
        for side, team_col, elo_col in [
            ("home", "home_team", "home_elo_post"),
            ("away", "away_team", "away_elo_post"),
        ]:
            part = history[["season", "date", team_col, elo_col]].copy()
            part = part.rename(columns={team_col: "team", elo_col: "elo"})
            end_parts.append(part)

        end_df = pd.concat(end_parts, ignore_index=True)
        end_df = end_df.sort_values(["season", "date"]).drop_duplicates(["season", "team"], keep="last")
        agg = agg.merge(end_df[["season", "team", "elo"]], on=["season", "team"], how="left")

        return agg.sort_values(["season", "exp_win_pct"], ascending=[True, False]).reset_index(drop=True)

    def simulate_games(
        self,
        games: pd.DataFrame,
        *,
        n_sims: int = 1000,
        seed: int = 42,
        reset_ratings: bool = True,
    ) -> pd.DataFrame:
        """
        Monte Carlo simulate outcomes + Elo updates game-by-game.

        Notes:
        - Uses current Elo ratings at the time of simulation (unless reset_ratings=True).
        - Returns one row per simulation with wins per team (wide format).
        """
        if n_sims <= 0:
            raise ValueError("n_sims must be > 0")

        games_sorted = games.sort_values(["season", "date", "home_team", "away_team"]).reset_index(drop=True)
        teams = sorted(set(games_sorted["home_team"]).union(set(games_sorted["away_team"])))

        rng = np.random.RandomState(seed)
        sim_rows = []

        base_ratings = dict(self.ratings_)

        for sim in range(n_sims):
            if reset_ratings:
                self.ratings_ = dict(base_ratings)

            wins = {t: 0 for t in teams}
            prev_season: Optional[int] = None

            for _, g in games_sorted.iterrows():
                season = int(g["season"])
                self._maybe_roll_season(season, prev_season)
                prev_season = season

                home = str(g["home_team"])
                away = str(g["away_team"])

                home_pre = self._get_rating(home)
                away_pre = self._get_rating(away)
                p_home = self.predict_home_win_prob(home_pre, away_pre)

                y = int(rng.rand() < p_home)
                wins[home] += y
                wins[away] += 1 - y

                delta = self.config.k * (y - p_home)
                self._set_rating(home, home_pre + delta)
                self._set_rating(away, away_pre - delta)

            row = {"sim": sim}
            row.update(wins)
            sim_rows.append(row)

        if reset_ratings:
            self.ratings_ = base_ratings

        return pd.DataFrame(sim_rows)


def _default_input_path() -> Path:
    # Try new location first, then legacy
    new_path = RAW_DIR / "wnba_gamelogs_2015_2025.csv"
    if new_path.is_file():
        return new_path
    legacy = PROJECT_DATA_DIR / "wnba_gamelogs_2015_2025.csv"
    if legacy.is_file():
        return legacy
    return new_path


def _default_output_dir() -> Path:
    return ELO_OUTPUT_DIR


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fit Elo from WNBA team game logs (2015-2025).")

    parser.add_argument("--input", type=str, default=str(_default_input_path()), help="Input CSV path")
    parser.add_argument("--k", type=float, default=20.0, help="Elo K-factor")
    parser.add_argument("--home-win", type=float, default=0.56, help="Target home win prob when equal strength")
    parser.add_argument(
        "--season-carryover",
        type=float,
        default=0.75,
        help="Season carryover in [0,1] (0=reset, 1=carry fully)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(_default_output_dir()),
        help="Directory to write outputs (csv)",
    )
    parser.add_argument("--no-write", action="store_true", help="Do not write output CSV files")

    args = parser.parse_args(list(argv) if argv is not None else None)

    home_adv = home_advantage_from_p(args.home_win)
    cfg = EloConfig(
        k=args.k,
        base_elo=1500.0,
        season_carryover=args.season_carryover,
        home_advantage=home_adv,
    )

    raw = load_team_gamelogs(args.input)
    games = build_unique_games_from_gamelogs(raw)

    model = EloModel(cfg)
    history = model.fit(games)
    summary = model.team_season_summary(history)

    out_dir = Path(args.out_dir)
    if not args.no_write:
        out_dir.mkdir(parents=True, exist_ok=True)
        history.to_csv(out_dir / "elo_game_history.csv", index=False)
        summary.to_csv(out_dir / "elo_team_season_summary.csv", index=False)
        pd.DataFrame(sorted(model.ratings_.items()), columns=["team", "elo"]).to_csv(
            out_dir / "elo_final_ratings.csv", index=False
        )

        # Save configuration to config directory
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        config_df = pd.DataFrame([{
            "k": cfg.k,
            "base_elo": cfg.base_elo,
            "season_carryover": cfg.season_carryover,
            "home_advantage": cfg.home_advantage,
            "home_win_prob": args.home_win
        }])
        config_df.to_csv(CONFIG_DIR / "elo_config.csv", index=False)

    # Quick print: latest season ranking by Elo and expected win%
    latest_season = int(summary["season"].max()) if not summary.empty else None
    if latest_season is not None:
        latest = summary[summary["season"] == latest_season].copy()
        latest = latest.sort_values("elo", ascending=False).head(12)
        print(f"Home advantage H ~= {home_adv:.1f} (target p_home={args.home_win})")
        print(f"Latest season (season={latest_season}) top teams by end-of-season Elo:")
        print(latest[["team", "elo", "win_pct", "exp_win_pct"]].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
