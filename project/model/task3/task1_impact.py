"""
Export Task3 expansion impacts as Task1 exogenous shocks.

This bridges Task3 (expansion analysis) -> Task1 (MPC simulation) by producing a
small JSON file with:
  - revenue_sharing_ratio
  - competition_intensity
  - travel_fatigue_delta
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from .expansion_model import ExpansionConfig
from .scenarios import calculate_expansion_scenarios


def build_task1_exogenous_shock(
    *,
    team_code: str,
    expansion_city: str,
    config: Optional[ExpansionConfig] = None,
) -> Dict[str, Any]:
    team = str(team_code).strip().upper()
    city = str(expansion_city).strip()
    if not team:
        raise ValueError("team_code is required")
    if not city:
        raise ValueError("expansion_city is required")

    cfg = config or ExpansionConfig()

    results = calculate_expansion_scenarios(
        expansion_cities=[city],
        representative_teams=[team],
        config=cfg,
    )

    rev = results["revenue_sharing"]
    att = results["attendance_impact"]
    travel = results["travel_burden"]

    if rev.empty:
        raise ValueError(f"No revenue_sharing results for city={city!r}")
    if att.empty:
        raise ValueError(f"No attendance_impact results for team={team!r}, city={city!r}")
    if travel.empty:
        raise ValueError(f"No travel_burden results for team={team!r}, city={city!r}")

    rev_row = rev.iloc[0]
    att_row = att.iloc[0]
    travel_row = travel.iloc[0]

    revenue_sharing_ratio = float(rev_row["mean_ratio"])
    competition_intensity = float(att_row["competition_intensity"])
    mean_attendance_ratio = float(att_row["mean_attendance_ratio"])
    change_km = float(travel_row["change_km"])

    travel_fatigue_delta = float(cfg.travel_fatigue_per_1000km) * (float(change_km) / 1000.0)

    return {
        "schema_version": 1,
        "team_code": team,
        "expansion_city": city,
        "revenue_sharing_ratio": revenue_sharing_ratio,
        "competition_intensity": competition_intensity,
        "competition_elasticity": float(cfg.competition_elasticity),
        "travel_fatigue_delta": travel_fatigue_delta,
        "source": {
            "task3": {
                "attendance_mean_ratio": mean_attendance_ratio,
                "travel_change_km": change_km,
            }
        },
    }


def export_task1_exogenous_shock(
    *,
    team_code: str,
    expansion_city: str,
    output_path: str | Path,
    config: Optional[ExpansionConfig] = None,
) -> Dict[str, Any]:
    out = build_task1_exogenous_shock(
        team_code=team_code,
        expansion_city=expansion_city,
        config=config,
    )

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Task3 -> Task1: export exogenous shock JSON.")
    parser.add_argument("--team", required=True, help="Team code (Task1 codes), e.g. LVA/NYL/IND")
    parser.add_argument("--expansion-city", required=True, help="Expansion city, e.g. Toronto/Bay Area/Portland")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: project/model/task3/outputs/task3_to_task1_<team>_<city>.json)",
    )
    args = parser.parse_args(argv)

    team = str(args.team).strip().upper()
    city = str(args.expansion_city).strip()
    out_path = args.output
    if not out_path:
        safe_city = "".join([c if c.isalnum() else "_" for c in city]).strip("_")
        out_path = str(Path(__file__).parent / "outputs" / f"task3_to_task1_{team}_{safe_city}.json")

    export_task1_exogenous_shock(team_code=team, expansion_city=city, output_path=out_path)
    print(f"[OK] Wrote Task3->Task1 shock file: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
