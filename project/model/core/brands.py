"""
Brand state model (B_t) for WNBA teams.

This module provides:
1) B_0 construction via Entropy Weight Method + TOPSIS (cross-sectional index).
2) A simple slow-moving brand dynamics model:

    B_{t+1} = rho_B * B_t + eta_W * W_t + eta_m * log(1 + m_t) + eta_star * Star_t

Notes
-----
- The functions accept caller-provided team-level indicator tables.
- A default indicator builder is included using the CSVs already in this repo.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .utils import resolve_path, min_max_scale


ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT_DIR / "project" / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CONFIG_DIR = PROCESSED_DIR / "config"
CLEAN_DIR = PROCESSED_DIR / "clean"
OTHER_CLEAN_DIR = PROCESSED_DIR / "other_clean"
ELO_DIR = PROCESSED_DIR / "elo"


TEAM_NAME_TO_CODE: Dict[str, str] = {
    "Atlanta Dream": "ATL",
    "Chicago Sky": "CHI",
    "Connecticut Sun": "CON",
    "Dallas Wings": "DAL",
    "Indiana Fever": "IND",
    "Las Vegas Aces": "LVA",
    "Los Angeles Sparks": "LAS",
    "Minnesota Lynx": "MIN",
    "New York Liberty": "NYL",
    "Phoenix Mercury": "PHO",
    "Seattle Storm": "SEA",
    "Washington Mystics": "WAS",
    "Golden State Valkyries": "GSV",
    "Toronto Tempo": "TOR",
}

# Historical franchise codes used in this repo's Elo data -> map to modern team.
FRANCHISE_ALIASES: Dict[str, str] = {
    "SAS": "LVA",  # San Antonio Stars -> Las Vegas Aces
    "TUL": "DAL",  # Tulsa Shock -> Dallas Wings
}


DEFAULT_EXPERT_WEIGHTS: Dict[str, float] = {
    # Brand fundamentals (more stable, closer to valuation logic):
    # - revenue is the strongest monetization proxy
    # - followers + attendance reflect attention + local demand
    # - capacity is a structural ceiling
    # - secondary market ticket price is noisy/volatile, so keep it small
    "ig_followers": 0.20,
    "avg_attendance": 0.20,
    "avg_ticket_price_usd": 0.10,
    "team_revenue_usd": 0.40,
    "arena_capacity": 0.10,
}


# Removed: _resolve_path
# Now using resolve_path from utils.py


def _team_to_code(team: object) -> Optional[str]:
    if team is None:
        return None
    s = str(team).strip()
    if not s:
        return None

    if s in TEAM_NAME_TO_CODE:
        return TEAM_NAME_TO_CODE[s]

    # Stadium rows look like: "Target Center (Minnesota Lynx)".
    if "(" in s and ")" in s:
        inner = s[s.find("(") + 1 : s.rfind(")")].strip()
        if inner in TEAM_NAME_TO_CODE:
            return TEAM_NAME_TO_CODE[inner]

    # Already a code?
    if len(s) in (2, 3, 4) and s.isupper():
        return s

    return None


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _orient_indicators(
    df: pd.DataFrame,
    *,
    positive_cols: Sequence[str],
    negative_cols: Sequence[str],
) -> pd.DataFrame:
    out = df.copy()

    for c in list(positive_cols) + list(negative_cols):
        out[c] = _numeric(out[c])
        if out[c].isna().any():
            out[c] = out[c].fillna(float(out[c].median()))

    for c in negative_cols:
        mx = float(out[c].max())
        out[c] = mx - out[c]

    return out


def entropy_weights(x01: pd.DataFrame) -> pd.Series:
    """
    Entropy weight method.

    Input must be non-negative, typically scaled to [0,1].
    Returns weights that sum to 1.
    """
    if x01.empty:
        raise ValueError("Empty indicator table")
    if len(x01) < 2:
        return pd.Series([1.0] * x01.shape[1], index=x01.columns)

    x = x01.clip(lower=0.0).astype(float)
    col_sums = x.sum(axis=0).replace(0.0, np.nan)
    p = x.div(col_sums, axis=1).fillna(0.0)

    n = float(len(x))
    k = 1.0 / np.log(n)
    p_safe = p.replace(0.0, np.nan)
    e = (-k) * (p_safe * np.log(p_safe)).sum(axis=0, skipna=True)
    d = (1.0 - e).clip(lower=0.0)

    if float(d.sum()) == 0.0:
        return pd.Series([1.0 / x.shape[1]] * x.shape[1], index=x.columns)
    return (d / d.sum()).astype(float)


def topsis_closeness(x01: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """
    TOPSIS closeness score C in [0,1].

    Uses weighted min-max normalized indicators (x01) and entropy weights.
    """
    x = x01.fillna(0.0).astype(float)
    w = weights.reindex(x.columns).fillna(0.0).astype(float)

    v = x.mul(w, axis=1)
    ideal_best = v.max(axis=0)
    ideal_worst = v.min(axis=0)

    s_plus = np.sqrt(((v - ideal_best) ** 2).sum(axis=1))
    s_minus = np.sqrt(((v - ideal_worst) ** 2).sum(axis=1))

    denom = (s_plus + s_minus).replace(0.0, np.nan)
    c = (s_minus / denom).fillna(0.5)
    return c.clip(lower=0.0, upper=1.0)


def _normalize_weight_preset(preset: Dict[str, float], indicators: Sequence[str]) -> pd.Series:
    w = pd.Series({c: float(preset.get(c, 0.0)) for c in indicators}, index=indicators, dtype=float)
    s = float(w.sum())
    if s <= 0.0:
        return pd.Series([1.0 / len(indicators)] * len(indicators), index=indicators, dtype=float)
    return (w / s).astype(float)


def _cap_single_weight(w: pd.Series, col: str, cap: Optional[float]) -> pd.Series:
    if cap is None or col not in w.index:
        return w
    cap = float(cap)
    if cap <= 0.0 or cap >= 1.0:
        return w

    w = w.astype(float).copy()
    if float(w[col]) <= cap:
        return w

    w[col] = cap
    other = w.drop(index=[col])
    if len(other) == 0:
        return w

    other_sum = float(other.sum())
    if other_sum <= 0.0:
        w[other.index] = (1.0 - cap) / float(len(other))
        return w

    w[other.index] = other / other_sum * (1.0 - cap)
    return w


def build_brand_index_entropy_topsis(
    df: pd.DataFrame,
    *,
    indicator_cols: Sequence[str],
    positive_cols: Optional[Sequence[str]] = None,
    negative_cols: Optional[Sequence[str]] = None,
    scale_mean: Optional[float] = 1.0,
    weight_method: str = "hybrid",
    expert_weights: Optional[Dict[str, float]] = None,
    entropy_alpha: float = 0.10,
    max_ticket_weight: Optional[float] = 0.15,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build brand index from team indicators using Entropy Weight + TOPSIS.

    Returns:
      - result dataframe with columns: team, closeness_C, B0, ...raw indicators
      - entropy weights series (index=indicator)
    """
    if "team" not in df.columns:
        raise ValueError("df must contain a 'team' column with team codes")

    indicators = list(indicator_cols)
    if not indicators:
        raise ValueError("indicator_cols is empty")

    pos = list(positive_cols) if positive_cols is not None else indicators
    neg = list(negative_cols) if negative_cols is not None else []

    d = df[["team"] + indicators].copy()
    d = d.dropna(subset=["team"]).copy()
    for c in indicators:
        d[c] = _numeric(d[c])

    d = _orient_indicators(d, positive_cols=pos, negative_cols=neg)

    # Min-max scale each indicator column
    x01 = d[indicators].copy()
    for c in indicators:
        x01[c] = min_max_scale(d[c])

    method = (weight_method or "").strip().lower()
    if method not in {"entropy", "expert", "hybrid"}:
        raise ValueError("weight_method must be one of: entropy, expert, hybrid")

    w_entropy = entropy_weights(x01)
    w_expert = _normalize_weight_preset(expert_weights or DEFAULT_EXPERT_WEIGHTS, indicators)

    if method == "entropy":
        w = w_entropy
    elif method == "expert":
        w = w_expert
    else:
        a = float(entropy_alpha)
        a = 0.0 if a < 0.0 else 1.0 if a > 1.0 else a
        w = a * w_entropy + (1.0 - a) * w_expert
        w = w / float(w.sum()) if float(w.sum()) > 0 else w_expert

    w = _cap_single_weight(w, "avg_ticket_price_usd", max_ticket_weight)
    c = topsis_closeness(x01, w)

    out = d.copy()
    out["closeness_C"] = c.values

    if scale_mean is None:
        out["B0"] = out["closeness_C"]
    else:
        mean_c = float(out["closeness_C"].mean()) if len(out) else 1.0
        out["B0"] = (out["closeness_C"] / mean_c) * float(scale_mean) if mean_c > 0 else float(scale_mean)

    return out.sort_values("B0", ascending=False).reset_index(drop=True), w.sort_values(ascending=False)


def _load_team_valuations_2024() -> pd.Series:
    path = OTHER_CLEAN_DIR / "value-of-teams-in-the-wnba-2024_clean.csv"
    if not path.is_file():
        return pd.Series(dtype=float)

    df = pd.read_csv(path)
    if not {"Team", "TeamValueUSD"}.issubset(df.columns):
        return pd.Series(dtype=float)

    df["team"] = df["Team"].map(_team_to_code)
    df["value_usd"] = _numeric(df["TeamValueUSD"])
    out = df.dropna(subset=["team", "value_usd"]).set_index("team")["value_usd"]
    return out.astype(float)


def expand_brand_to_teams(
    brand_by_team: pd.Series,
    teams: Iterable[str],
    *,
    aliases: Optional[Dict[str, str]] = None,
    default: Optional[float] = None,
) -> pd.Series:
    aliases = aliases or {}
    out = {}
    base_mean = float(brand_by_team.mean()) if default is None else float(default)

    for t in teams:
        t = str(t).strip()
        if t in brand_by_team.index:
            out[t] = float(brand_by_team.loc[t])
            continue
        if t in aliases and aliases[t] in brand_by_team.index:
            out[t] = float(brand_by_team.loc[aliases[t]])
            continue
        out[t] = base_mean

    return pd.Series(out, name=brand_by_team.name)


@dataclass
class BrandParameters:
    rho_B: float = 0.90
    eta_W: float = 0.02
    eta_m: float = 0.05
    eta_star: float = 0.10


def update_brand(
    B_t: float,
    W_t: float,
    m_t: float,
    Star_t: float,
    params: BrandParameters,
) -> float:
    return (
        params.rho_B * float(B_t)
        + params.eta_W * float(W_t)
        + params.eta_m * float(np.log1p(max(float(m_t), 0.0)))
        + params.eta_star * float(Star_t)
    )


def roll_brand_states(
    season_df: pd.DataFrame,
    *,
    b0_by_team: Dict[str, float] | pd.Series,
    params: BrandParameters = BrandParameters(),
    team_col: str = "team",
    season_col: str = "season",
    wins_col: str = "W",
    marketing_col: str = "m",
    star_col: str = "Star",
) -> pd.DataFrame:
    """
    Roll forward B_t for each team across seasons.

    season_df must contain at least: team, season, W, m, Star.
    """
    d = season_df.copy()
    d[team_col] = d[team_col].astype(str).str.strip()
    d[season_col] = _numeric(d[season_col])
    d[wins_col] = _numeric(d[wins_col])
    d[marketing_col] = _numeric(d[marketing_col])
    d[star_col] = _numeric(d[star_col])

    b0 = pd.Series(b0_by_team, dtype=float)
    d = d.sort_values([team_col, season_col]).reset_index(drop=True)

    out_rows = []
    for team, g in d.groupby(team_col, sort=False):
        B = float(b0.get(team, float(b0.mean()) if len(b0) else 1.0))
        for _, r in g.iterrows():
            out_rows.append(
                {
                    team_col: team,
                    season_col: int(r[season_col]) if pd.notna(r[season_col]) else None,
                    "B_t": B,
                    wins_col: float(r[wins_col]) if pd.notna(r[wins_col]) else 0.0,
                    marketing_col: float(r[marketing_col]) if pd.notna(r[marketing_col]) else 0.0,
                    star_col: float(r[star_col]) if pd.notna(r[star_col]) else 0.0,
                }
            )
            B = update_brand(B, r[wins_col], r[marketing_col], r[star_col], params)

    return pd.DataFrame(out_rows)


def load_default_brand_indicators(
    *,
    year: int = 2025,
    ticket_price_year: int = 2024,
    revenue_year: int = 2024,
    team_value_year: int = 2024,
    capacity_year: int = 2025,
) -> pd.DataFrame:
    """
    Build a cross-sectional indicator table from repo CSVs (best-effort).

    The resulting table uses the repo's team codes (ATL, CHI, ...).
    """
    parts: List[pd.DataFrame] = []

    ig_path = OTHER_CLEAN_DIR / "wnba-instagram-followers-2024-2025-by-team_clean.csv"
    if ig_path.is_file():
        ig = pd.read_csv(ig_path)
        ig = ig[ig["Year"] == year].copy()
        ig["team"] = ig["Team"].map(_team_to_code)
        ig["ig_followers"] = _numeric(ig["IGFollowers"])
        parts.append(ig[["team", "ig_followers"]])

    att_path = CLEAN_DIR / "attendance_clean.csv"
    if att_path.is_file():
        att = pd.read_csv(att_path)
        att = att[att["Year"] == year].copy()
        att["team"] = att["Team"].map(_team_to_code)
        att["avg_attendance"] = _numeric(att["Average"])
        parts.append(att[["team", "avg_attendance"]])

    tp_path = OTHER_CLEAN_DIR / "average-ticket-price-wnba-games-2024_clean.csv"
    if tp_path.is_file():
        tp = pd.read_csv(tp_path)
        tp = tp[(tp["Year"] == ticket_price_year) & (tp["Team"] != "League average")].copy()
        tp["team"] = tp["Team"].map(_team_to_code)
        tp["avg_ticket_price_usd"] = _numeric(tp["AvgTicketPriceUSD"])
        parts.append(tp[["team", "avg_ticket_price_usd"]])

    rev_path = OTHER_CLEAN_DIR / "wnba-teams-with-the-highest-revenue-2024_clean.csv"
    if rev_path.is_file():
        rev = pd.read_csv(rev_path)
        rev = rev[rev["Year"] == revenue_year].copy()
        rev["team"] = rev["Team"].map(_team_to_code)
        rev["team_revenue_usd"] = _numeric(rev["TeamRevenueUSD"])
        parts.append(rev[["team", "team_revenue_usd"]])

    val_path = OTHER_CLEAN_DIR / "value-of-teams-in-the-wnba-2024_clean.csv"
    if val_path.is_file():
        val = pd.read_csv(val_path)
        val = val[val["Year"] == team_value_year].copy()
        val["team"] = val["Team"].map(_team_to_code)
        val["team_value_usd"] = _numeric(val["TeamValueUSD"])
        parts.append(val[["team", "team_value_usd"]])

    cap_path = OTHER_CLEAN_DIR / "wnba-stadiums-2025-by-capacity_clean.csv"
    if cap_path.is_file():
        cap = pd.read_csv(cap_path)
        cap = cap[(cap["Year"] == capacity_year) & (cap["Team"] != "League average")].copy()
        cap["team"] = cap["Team"].map(_team_to_code)
        cap["arena_capacity"] = _numeric(cap["ArenaCapacity"])
        parts.append(cap[["team", "arena_capacity"]])

    if not parts:
        raise FileNotFoundError("No default brand indicator CSVs found under project/data/processed/*")

    out = parts[0].copy()
    for p in parts[1:]:
        out = out.merge(p, on="team", how="outer")

    out = out.dropna(subset=["team"]).drop_duplicates(subset=["team"], keep="first")
    return out.reset_index(drop=True)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build B0 using entropy-weight TOPSIS.")
    parser.add_argument("--year", type=int, default=2025, help="Baseline year for IG/attendance indicators")
    parser.add_argument("--scale-mean", type=float, default=1.0, help="Scale brand index so mean(B0)=this value")
    parser.add_argument("--weight-method", type=str, default="hybrid", choices=["entropy", "expert", "hybrid"])
    parser.add_argument("--entropy-alpha", type=float, default=0.10, help="Only for hybrid: share of entropy weights")
    parser.add_argument("--max-ticket-weight", type=float, default=0.15, help="Cap avg_ticket_price_usd weight")
    parser.add_argument("--out", type=str, default=str(CONFIG_DIR / "brand_b0.csv"))
    parser.add_argument("--weights-out", type=str, default=str(CONFIG_DIR / "brand_weights.csv"))
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    indicators = load_default_brand_indicators(year=args.year)
    # Avoid using valuation itself as an input indicator to prevent "self-explaining"
    # when later comparing brand vs valuation.
    indicator_cols = [c for c in indicators.columns if c not in {"team", "team_value_usd"}]

    df_b0, w = build_brand_index_entropy_topsis(
        indicators,
        indicator_cols=indicator_cols,
        scale_mean=float(args.scale_mean),
        weight_method=args.weight_method,
        entropy_alpha=float(args.entropy_alpha),
        max_ticket_weight=float(args.max_ticket_weight) if args.max_ticket_weight is not None else None,
    )

    # Optional: expand to all teams present in Elo file (and fill historical aliases).
    elo_path = ELO_DIR / "elo_final_ratings.csv"
    if not elo_path.is_file():
        elo_path = PROCESSED_DIR / "elo_final_ratings.csv"  # legacy fallback
    if elo_path.is_file():
        elo = pd.read_csv(elo_path)
        teams = [str(t).strip() for t in elo["team"].tolist()]
        b0_full = expand_brand_to_teams(df_b0.set_index("team")["B0"], teams, aliases=FRANCHISE_ALIASES)
        df_b0 = df_b0.set_index("team")
        df_b0 = df_b0.reindex(teams).reset_index()
        df_b0["B0"] = b0_full.values
        df_b0 = df_b0[~df_b0["team"].isin(FRANCHISE_ALIASES.keys())].reset_index(drop=True)

    if not args.no_write:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df_b0.to_csv(args.out, index=False)

        Path(args.weights_out).parent.mkdir(parents=True, exist_ok=True)
        w.reset_index().rename(columns={"index": "indicator", 0: "weight"}).to_csv(args.weights_out, index=False)

    print("Indicators used:", indicator_cols)
    print("Top weights:")
    print(w.head(8).to_string())

    values_2024 = _load_team_valuations_2024()
    if not values_2024.empty:
        m = df_b0.set_index("team")[["B0"]].join(values_2024.rename("valuation_2024_usd")).dropna()
        if len(m) >= 3:
            s = float(m["B0"].rank(ascending=False).corr(m["valuation_2024_usd"].rank(ascending=False)))
            print("")
            print("Spearman(rank_B0, rank_valuation_2024): %.3f" % s)
    print("")
    print(df_b0[["team", "B0"]].head(12).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
