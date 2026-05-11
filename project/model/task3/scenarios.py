"""
Calculate expansion scenarios for different cities and teams.
"""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

from .expansion_model import ExpansionModel, ExpansionConfig
from .data_loader import (
    load_wnba_expansion_data,
    get_team_market_size,
    estimate_expansion_city_revenue,
    estimate_expansion_city_attendance,
)


def calculate_expansion_scenarios(
    expansion_cities: List[str] = None,
    representative_teams: List[str] = None,
    config: ExpansionConfig = None,
    include_talent_dilution: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Calculate expansion impact scenarios for multiple cities and teams.

    Args:
        expansion_cities: List of expansion city names (default: Toronto, Bay Area, Portland)
        representative_teams: List of team abbreviations to analyze (default: NYL, CON, SEA)
        config: ExpansionConfig instance
        include_talent_dilution: Whether to include Channel 4 (talent dilution)

    Returns:
        Dictionary with DataFrames:
        - revenue_sharing: Revenue sharing impact by city
        - attendance_impact: Attendance impact by team and city
        - playoff_probability: Playoff probability changes
        - travel_burden: Travel burden changes
        - talent_dilution: Talent dilution impact (if include_talent_dilution=True)
    """
    if expansion_cities is None:
        expansion_cities = ["Toronto", "Bay Area", "Portland"]

    if representative_teams is None:
        representative_teams = ["NYL", "CON", "SEA"]

    if config is None:
        config = ExpansionConfig()

    # Load data
    data = load_wnba_expansion_data()
    team_locations = data['team_locations']
    expansion_locs = data['expansion_cities']
    revenue_df = data['team_revenue']

    model = ExpansionModel(config)

    # 1. Revenue Sharing Channel
    revenue_sharing_results = []

    for city in expansion_cities:
        # Estimate league growth based on city market potential
        city_revenue = estimate_expansion_city_revenue(city)
        total_current_revenue = revenue_df['TeamRevenueUSD'].sum()

        # League growth = new team revenue / current total
        implied_growth = city_revenue / total_current_revenue

        # Adjust growth estimates based on city characteristics
        growth_scenarios = {
            "Toronto": {"mean": 0.14, "std": 0.06},      # High growth potential
            "Bay Area": {"mean": 0.10, "std": 0.05},     # Moderate-high
            "Portland": {"mean": 0.07, "std": 0.04},     # Moderate
        }

        scenario = growth_scenarios.get(city, {"mean": 0.10, "std": 0.05})

        result = model.simulate_revenue_sharing(
            n_teams_before=12,
            n_teams_after=13,
            growth_mean=scenario["mean"],
            growth_std=scenario["std"]
        )

        revenue_sharing_results.append({
            "expansion_city": city,
            "league_growth_mean": scenario["mean"],
            "league_growth_std": scenario["std"],
            "mean_ratio": result["mean_ratio"],
            "ci_lower": result["ci_lower"],
            "ci_upper": result["ci_upper"],
            "prob_increase": result["prob_increase"],
            "breakeven_growth": result["breakeven_growth"]
        })

    revenue_sharing_df = pd.DataFrame(revenue_sharing_results)

    # 2. Attendance/Competition Channel
    attendance_results = []

    for team_abbr in representative_teams:
        if team_abbr not in team_locations:
            continue

        team_loc = team_locations[team_abbr]
        team_data = revenue_df[revenue_df['team_abbr'] == team_abbr]

        if team_data.empty:
            continue

        base_attendance = team_data['estimated_attendance'].values[0]
        market_size = get_team_market_size(team_abbr, revenue_df)

        for city in expansion_cities:
            if city not in expansion_locs:
                continue

            new_loc = expansion_locs[city]

            result = model.simulate_attendance_impact(
                team_location=team_loc,
                new_team_location=new_loc,
                base_attendance=base_attendance,
                attendance_std=base_attendance * 0.1  # 10% uncertainty
            )

            attendance_results.append({
                "team": team_abbr,
                "market_size": market_size,
                "expansion_city": city,
                "distance_km": result["distance_km"],
                "competition_intensity": result["competition_intensity"],
                "is_rivalry": result["is_rivalry"],
                "mean_attendance_ratio": result["mean_ratio"],
                "ci_lower_ratio": result["ci_lower_ratio"],
                "ci_upper_ratio": result["ci_upper_ratio"],
                "prob_increase_attendance": result["prob_increase_attendance"],
                "mean_delta_revenue_m": result["mean_delta_revenue_m"],
                "ci_lower_revenue_m": result["ci_lower_revenue_m"],
                "ci_upper_revenue_m": result["ci_upper_revenue_m"],
                "prob_increase_revenue": result["prob_increase_revenue"]
            })

    attendance_df = pd.DataFrame(attendance_results)

    # 3. Playoff Probability Changes
    playoff_results = []

    for n_new_teams in [1, 2]:
        result = model.calculate_playoff_probability_change(
            n_teams_before=12,
            n_teams_after=12 + n_new_teams,
            n_playoff_spots=8
        )

        playoff_results.append({
            "n_teams_before": 12,
            "n_teams_after": 12 + n_new_teams,
            "n_playoff_spots": 8,
            "prob_before": result["prob_before"],
            "prob_after": result["prob_after"],
            "absolute_change": result["absolute_change"],
            "relative_change": result["relative_change"]
        })

    playoff_df = pd.DataFrame(playoff_results)

    # 4. Travel Burden Changes
    travel_results = []

    for team_abbr in representative_teams:
        if team_abbr not in team_locations:
            continue

        team_loc = team_locations[team_abbr]
        other_locs = [loc for abbr, loc in team_locations.items() if abbr != team_abbr]

        # Before expansion
        result_before = model.calculate_travel_burden_change(
            team_location=team_loc,
            existing_team_locations=other_locs,
            new_team_location=None
        )

        for city in expansion_cities:
            if city not in expansion_locs:
                continue

            new_loc = expansion_locs[city]

            result_after = model.calculate_travel_burden_change(
                team_location=team_loc,
                existing_team_locations=other_locs,
                new_team_location=new_loc
            )

            travel_results.append({
                "team": team_abbr,
                "expansion_city": city,
                "avg_distance_before_km": result_before["avg_distance_before_km"],
                "avg_distance_after_km": result_after["avg_distance_after_km"],
                "change_km": result_after["change_km"],
                "relative_change": result_after["relative_change"]
            })

    travel_df = pd.DataFrame(travel_results)

    # 5. Talent Dilution Channel (Channel 4)
    talent_dilution_results = []

    if include_talent_dilution:
        # Estimate team Elo ratings (simplified - in practice would come from data)
        # Using market size as proxy: large market teams tend to be stronger
        team_elo_estimates = {
            "NYL": 1580,  # New York - strong team
            "CON": 1520,  # Connecticut - average
            "SEA": 1550,  # Seattle - above average
            "LVA": 1620,  # Las Vegas - very strong (paper's focal team)
            "IND": 1500,  # Indiana - average
            "CHI": 1530,  # Chicago - above average
        }

        for team_abbr in representative_teams:
            if team_abbr not in team_elo_estimates:
                # Default to league average
                team_elo = 1500
            else:
                team_elo = team_elo_estimates[team_abbr]

            # Calculate for different expansion scenarios
            for n_new_teams in [1, 2]:
                result = model.simulate_talent_dilution(
                    team_elo=team_elo,
                    roster_depth=12,
                    expansion_draft_protection=6,
                    n_new_teams=n_new_teams
                )

                talent_dilution_results.append({
                    "team": team_abbr,
                    "team_elo": team_elo,
                    "n_new_teams": n_new_teams,
                    "mean_delta_wins": result["mean_delta_wins"],
                    "ci_lower_wins": result["ci_lower_wins"],
                    "ci_upper_wins": result["ci_upper_wins"],
                    "prob_negative_impact": result["prob_negative_impact"],
                    "mean_delta_elo": result["mean_delta_elo"],
                    "ci_lower_elo": result["ci_lower_elo"],
                    "ci_upper_elo": result["ci_upper_elo"],
                    "prob_no_loss": result["scenario_distribution"]["no_loss"],
                    "prob_rotation_loss": result["scenario_distribution"]["rotation_loss"],
                    "prob_key_loss": result["scenario_distribution"]["key_loss"]
                })

        talent_dilution_df = pd.DataFrame(talent_dilution_results)
    else:
        talent_dilution_df = pd.DataFrame()

    results = {
        "revenue_sharing": revenue_sharing_df,
        "attendance_impact": attendance_df,
        "playoff_probability": playoff_df,
        "travel_burden": travel_df
    }

    if include_talent_dilution:
        results["talent_dilution"] = talent_dilution_df

    return results


def format_results_for_paper(results: Dict[str, pd.DataFrame]) -> Dict[str, str]:
    """
    Format results as LaTeX tables for paper.

    Args:
        results: Dictionary of result DataFrames

    Returns:
        Dictionary of LaTeX table strings
    """
    tables = {}

    # 1. Revenue Sharing Table
    rev_df = results["revenue_sharing"].copy()
    rev_df["ci_range"] = rev_df.apply(
        lambda x: f"[{x['ci_lower']:.3f}, {x['ci_upper']:.3f}]",
        axis=1
    )

    latex_rev = rev_df[[
        "expansion_city",
        "league_growth_mean",
        "mean_ratio",
        "ci_range",
        "prob_increase"
    ]].to_latex(
        index=False,
        float_format="%.3f",
        caption="Revenue Sharing Impact by Expansion City",
        label="tab:revenue_sharing"
    )
    tables["revenue_sharing"] = latex_rev

    # 2. Attendance Impact Table
    att_df = results["attendance_impact"].copy()
    att_df["attendance_ci"] = att_df.apply(
        lambda x: f"[{x['ci_lower_ratio']:.3f}, {x['ci_upper_ratio']:.3f}]",
        axis=1
    )
    att_df["revenue_ci"] = att_df.apply(
        lambda x: f"[{x['ci_lower_revenue_m']:.2f}, {x['ci_upper_revenue_m']:.2f}]",
        axis=1
    )

    latex_att = att_df[[
        "team",
        "expansion_city",
        "mean_attendance_ratio",
        "attendance_ci",
        "prob_increase_attendance",
        "mean_delta_revenue_m",
        "revenue_ci",
        "prob_increase_revenue"
    ]].to_latex(
        index=False,
        float_format="%.3f",
        caption="Attendance and Revenue Impact by Team and Expansion City",
        label="tab:attendance_impact"
    )
    tables["attendance_impact"] = latex_att

    # 3. Playoff Probability Table
    playoff_df = results["playoff_probability"].copy()

    latex_playoff = playoff_df.to_latex(
        index=False,
        float_format="%.3f",
        caption="Playoff Probability Structural Changes",
        label="tab:playoff_probability"
    )
    tables["playoff_probability"] = latex_playoff

    return tables


def generate_summary_statistics(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Generate summary statistics across all scenarios.

    Args:
        results: Dictionary of result DataFrames

    Returns:
        Summary statistics DataFrame
    """
    summary_rows = []

    # Revenue sharing summary
    rev_df = results["revenue_sharing"]
    summary_rows.append({
        "metric": "Revenue Sharing - Mean Ratio",
        "min": rev_df["mean_ratio"].min(),
        "max": rev_df["mean_ratio"].max(),
        "mean": rev_df["mean_ratio"].mean(),
        "std": rev_df["mean_ratio"].std()
    })

    summary_rows.append({
        "metric": "Revenue Sharing - P(Increase)",
        "min": rev_df["prob_increase"].min(),
        "max": rev_df["prob_increase"].max(),
        "mean": rev_df["prob_increase"].mean(),
        "std": rev_df["prob_increase"].std()
    })

    # Attendance impact summary
    att_df = results["attendance_impact"]
    summary_rows.append({
        "metric": "Attendance Ratio",
        "min": att_df["mean_attendance_ratio"].min(),
        "max": att_df["mean_attendance_ratio"].max(),
        "mean": att_df["mean_attendance_ratio"].mean(),
        "std": att_df["mean_attendance_ratio"].std()
    })

    summary_rows.append({
        "metric": "Revenue Impact ($M)",
        "min": att_df["mean_delta_revenue_m"].min(),
        "max": att_df["mean_delta_revenue_m"].max(),
        "mean": att_df["mean_delta_revenue_m"].mean(),
        "std": att_df["mean_delta_revenue_m"].std()
    })

    # Travel burden summary
    travel_df = results["travel_burden"]
    summary_rows.append({
        "metric": "Travel Distance Change (km)",
        "min": travel_df["change_km"].min(),
        "max": travel_df["change_km"].max(),
        "mean": travel_df["change_km"].mean(),
        "std": travel_df["change_km"].std()
    })

    return pd.DataFrame(summary_rows)
