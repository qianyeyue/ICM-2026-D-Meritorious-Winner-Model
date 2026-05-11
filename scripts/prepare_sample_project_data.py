"""Create small local data files for smoke-testing the public model code."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "project" / "data"
PROCESSED = DATA / "processed"
TASK1_INPUT = ROOT / "project" / "model" / "task1_input"


TEAMS = [
    ("LVA", "Las Vegas Aces", 1.12, 12000, 45.0, 720000, 1600),
    ("NYL", "New York Liberty", 1.08, 12500, 42.0, 680000, 1585),
    ("SEA", "Seattle Storm", 0.98, 11000, 36.0, 520000, 1510),
    ("CON", "Connecticut Sun", 0.94, 9800, 32.0, 430000, 1505),
]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def build_brand_and_elo() -> None:
    write_csv(
        PROCESSED / "config" / "brand_b0.csv",
        [
            {
                "team": code,
                "B0": b0,
                "arena_capacity": capacity,
                "avg_ticket_price_usd": ticket,
                "ig_followers": followers,
            }
            for code, _, b0, capacity, ticket, followers, _ in TEAMS
        ],
    )
    write_csv(
        PROCESSED / "elo" / "elo_final_ratings.csv",
        [{"team": code, "elo": elo} for code, _, _, _, _, _, elo in TEAMS],
    )
    write_csv(
        PROCESSED / "config" / "elo_config.csv",
        [{
            "base_elo": 1500,
            "home_advantage": 24,
            "k": 20,
            "season_carryover": 0.75,
            "home_win_prob": 0.535,
        }],
    )


def build_task3_data() -> None:
    write_csv(
        PROCESSED / "other_clean" / "wnba-teams-with-the-highest-revenue-2024_clean.csv",
        [
            {"Team": name, "Year": 2024, "TeamRevenueUSD": revenue}
            for _, name, _, _, _, _, revenue in [
                ("LVA", "Las Vegas Aces", 0, 0, 0, 0, 18_000_000),
                ("NYL", "New York Liberty", 0, 0, 0, 0, 17_000_000),
                ("SEA", "Seattle Storm", 0, 0, 0, 0, 14_000_000),
                ("CON", "Connecticut Sun", 0, 0, 0, 0, 12_000_000),
            ]
        ],
    )
    write_csv(
        PROCESSED / "other_clean" / "wnba-average-regular-season-attendance-1997-2025_clean.csv",
        [{"Year": 2024, "AvgAttendance": 9300}, {"Year": 2025, "AvgAttendance": 9800}],
    )
    write_csv(
        PROCESSED / "other_clean" / "average-player-salary-in-the-wnba-by-team-2025_clean.csv",
        [
            {"Team": name, "Year": 2025, "AvgSalaryUSD": 120000 + i * 5000}
            for i, (_, name, *_rest) in enumerate(TEAMS)
        ],
    )

    games = []
    game_id = 1
    for home, away in [("LVA", "NYL"), ("SEA", "CON"), ("NYL", "SEA"), ("CON", "LVA")]:
        games.append({
            "Season": 2024,
            "Team": home,
            "Date": f"2024-06-{game_id:02d}",
            "Home": 1,
            "Opp": away,
            "W/L": "W" if game_id % 2 else "L",
        })
        games.append({
            "Season": 2024,
            "Team": away,
            "Date": f"2024-06-{game_id:02d}",
            "Home": 0,
            "Opp": home,
            "W/L": "L" if game_id % 2 else "W",
        })
        game_id += 1
    write_csv(PROCESSED / "clean" / "wnba_gamelogs_clean.csv", games)


def build_task2_inputs() -> None:
    TASK1_INPUT.mkdir(parents=True, exist_ok=True)
    players = []
    inputs = []
    positions = ["G", "G", "G", "G", "F", "F", "F", "F", "C", "C", "C", "G"]
    for idx, pos in enumerate(positions, start=1):
        salary = 82000 + idx * 4500
        players.append({
            "Player_ID": idx,
            "Player_Name": f"Sample Player {idx:02d}",
            "Position": pos,
            "Age": 22 + (idx % 9),
            "Experience": idx % 6,
            "Salary": salary,
            "Commercial_Value_PCA": 0.45 + idx * 0.035,
            "Court_Impact": 0.50 + idx * 0.030,
            "Injury_Risk": 0.08 + (idx % 4) * 0.025,
            "Team": "LVA" if idx <= 5 else "FA",
            "PPG": 7 + idx,
            "RPG": 2 + idx * 0.2,
            "APG": 1 + idx * 0.15,
            "MPG": 12 + idx,
            "FG_Percent": 0.40 + idx * 0.005,
            "TS_Percent": 0.48 + idx * 0.004,
        })
        inputs.append({
            "offseason_year": 2025,
            "player_id": idx,
            "mu0_perf": 0.45 + idx * 0.03,
            "mu1_perf": 0.48 + idx * 0.03,
            "sigma_perf2": 0.02,
            "mu0_pop": 0.40 + idx * 0.025,
            "mu1_pop": 0.42 + idx * 0.025,
            "inj_score": 0.08 + (idx % 4) * 0.025,
        })

    write_csv(TASK1_INPUT / "candidates_new.csv", players)
    write_csv(TASK1_INPUT / "player_inputs_new.csv", inputs)
    write_csv(
        TASK1_INPUT / "position_targets_new.csv",
        [{
            "offseason_year": 2025,
            "team_id": "LVA",
            "target_G": 4,
            "target_F": 4,
            "target_C": 3,
            "penalty_pos": 50000,
        }],
    )


def main() -> None:
    build_brand_and_elo()
    build_task3_data()
    build_task2_inputs()
    print(f"Wrote smoke-test data under {DATA}")
    print(f"Wrote Task 2 sample inputs under {TASK1_INPUT}")


if __name__ == "__main__":
    main()
