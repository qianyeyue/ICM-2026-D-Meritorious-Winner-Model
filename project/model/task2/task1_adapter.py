"""
Task2 -> Task1 adapter.

Task2 outputs a season initial state in "Task2 units" (S_0/B_0/Cash_0).
Task1 (MPC simulation) needs (elo/B/cash).

This module provides a lightweight, explicit mapping so Task1 can start from
the roster decision in Task2.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[3]
PROCESSED_DIR = ROOT_DIR / "project" / "data" / "processed"
DEFAULT_PCV_PATH = PROCESSED_DIR / "players" / "players_pcv.csv"


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if v == v else float(default)  # NaN guard
    except Exception:
        return float(default)


def _to_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return int(default)


def load_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def brand_impact_mean(*, pcv_path: str | Path = DEFAULT_PCV_PATH) -> float:
    p = Path(pcv_path)
    if not p.exists():
        return 4.6
    try:
        df = pd.read_csv(p)
    except Exception:
        return 4.6
    if "brand_impact" not in df.columns:
        return 4.6
    s = pd.to_numeric(df["brand_impact"], errors="coerce")
    m = float(s.mean()) if s.notna().any() else 4.6
    return m if m > 0 else 4.6


def _elo_std_from_processed(*, elo_path: str | Path = PROCESSED_DIR / "elo" / "elo_final_ratings.csv") -> float:
    p = Path(elo_path)
    if not p.exists():
        return 100.0
    try:
        df = pd.read_csv(p)
    except Exception:
        return 100.0
    if "elo" not in df.columns:
        return 100.0
    s = pd.to_numeric(df["elo"], errors="coerce")
    std = float(s.std()) if s.notna().any() else 100.0
    return std if std > 0 else 100.0


def map_task2_state_to_task1_initial_state(
    task2_state: Dict[str, Any],
    *,
    base_elo: float = 1500.0,
    elo_per_s_point: Optional[float] = None,
    initial_cash: float = 5_000_000.0,
    pcv_path: str | Path = DEFAULT_PCV_PATH,
    roster_size_default: int = 12,
    elo_clip: Optional[tuple[float, float]] = (1200.0, 1800.0),
) -> Dict[str, float]:
    """
    Map Task2 season initial state fields to Task1 init fields.

Task2 fields (expected):
  - S_0 in [0,100], centered at 50
  - B_0 as team brand index (TOPSIS, mean~1) OR (legacy) roster brand_impact sum
  - Cash_0 as cash delta (e.g. -transfer_fee). For legacy outputs, we infer delta.

    Returns:
      { "elo": ..., "B": ..., "cash": ..., "avg_salary_usd": ... }
    """
    s0 = _to_float(task2_state.get("S_0"), 50.0)
    b0 = _to_float(task2_state.get("B_0"), 0.0)
    cash0_raw = _to_float(task2_state.get("Cash_0"), 0.0)
    total_transfer_fee = _to_float(task2_state.get("total_transfer_fee"), 0.0)

    roster_size = _to_int(task2_state.get("roster_size"), roster_size_default)
    roster_size = max(roster_size, 1)

    if elo_per_s_point is None:
        elo_per_s_point = _elo_std_from_processed() / 10.0

    elo = float(base_elo) + (float(s0) - 50.0) * float(elo_per_s_point)
    if elo_clip is not None:
        elo = float(max(float(elo_clip[0]), min(float(elo_clip[1]), elo)))

    # Brand:
    # - New Task2 output: B_0 is TOPSIS team-level brand index (aligned with brands.py, mean~1)
    # - Legacy Task2 output: B_0 is roster brand_impact sum (large number), so we scale it
    looks_like_team_b0 = any(k in task2_state for k in ("B_roster_sum_0", "cap_space_0", "transfer_budget_remaining_0"))
    if looks_like_team_b0:
        B = float(b0) if float(b0) > 0 else 1.0
    elif 0.0 < float(b0) < 10.0:
        B = float(b0)
    else:
        mean_b = brand_impact_mean(pcv_path=pcv_path)
        baseline_sum = float(mean_b) * float(roster_size)
        B = float(b0) / baseline_sum if baseline_sum > 0 else 1.0
        if B <= 0.0:
            B = 1.0

    # Cash: Task1 tracks actual cash balance. Salary is paid during the season in cost.py,
    # so here we only treat transfer_fee (and other one-off costs if added later) as
    # an initial cash delta.
    if float(cash0_raw) > 0.0 and float(total_transfer_fee) > 0.0:
        # Legacy state used "cap space" style Cash_0; convert to cash delta.
        cash_delta = -float(total_transfer_fee)
    else:
        cash_delta = float(cash0_raw) if float(cash0_raw) <= 0.0 else 0.0

    cash = float(initial_cash) + float(cash_delta)

    total_salary = _to_float(task2_state.get("total_salary"), 0.0)
    avg_salary = float(total_salary) / float(roster_size) if total_salary > 0 else 0.0

    return {
        "elo": float(elo),
        "B": float(B),
        "cash": float(cash),
        "avg_salary_usd": float(avg_salary),
        "roster_size": float(roster_size),
    }


def export_task1_initial_state(
    task2_state: Dict[str, Any],
    *,
    output_path: str | Path,
    base_elo: float = 1500.0,
    elo_per_s_point: Optional[float] = None,
    initial_cash: float = 5_000_000.0,
    pcv_path: str | Path = DEFAULT_PCV_PATH,
    roster_size_default: int = 12,
) -> Dict[str, Any]:
    elo_per_s_point_used = float(elo_per_s_point) if elo_per_s_point is not None else float(_elo_std_from_processed() / 10.0)
    mapped = map_task2_state_to_task1_initial_state(
        task2_state,
        base_elo=base_elo,
        elo_per_s_point=elo_per_s_point_used,
        initial_cash=initial_cash,
        pcv_path=pcv_path,
        roster_size_default=roster_size_default,
    )

    out = {
        "team": str(task2_state.get("team")) if task2_state.get("team") is not None else None,
        "elo": float(mapped["elo"]),
        "B": float(mapped["B"]),
        "cash": float(mapped["cash"]),
        "avg_salary_usd": float(mapped["avg_salary_usd"]),
        "roster_size": float(mapped["roster_size"]),
        "source_task2": {
            "S_0": _to_float(task2_state.get("S_0"), 50.0),
            "B_0": _to_float(task2_state.get("B_0"), 0.0),
            "Cash_0": _to_float(task2_state.get("Cash_0"), 0.0),
            "cap_space_0": _to_float(task2_state.get("cap_space_0"), 0.0),
            "total_salary": _to_float(task2_state.get("total_salary"), 0.0),
            "total_transfer_fee": _to_float(task2_state.get("total_transfer_fee"), 0.0),
            "B_roster_sum_0": _to_float(task2_state.get("B_roster_sum_0"), 0.0),
        },
        "mapping_params": {
            "base_elo": float(base_elo),
            "elo_per_s_point": float(elo_per_s_point_used),
            "initial_cash": float(initial_cash),
            "brand_impact_mean": float(brand_impact_mean(pcv_path=pcv_path)),
        },
    }

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out
