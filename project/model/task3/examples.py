"""
Example usage and testing script for Task 3: League Expansion Impact Analysis.

Small examples for custom expansion scenarios.
"""

from pathlib import Path
import pandas as pd
import numpy as np

from .expansion_model import ExpansionModel, ExpansionConfig
from .data_loader import load_wnba_expansion_data, WNBA_TEAM_LOCATIONS, EXPANSION_CITIES
from .scenarios import calculate_expansion_scenarios


def example_1_basic_revenue_sharing():
    """Example 1: Calculate revenue sharing impact for a single scenario."""
    print("\n" + "="*80)
    print("Example 1: Basic Revenue Sharing Analysis")
    print("="*80 + "\n")

    config = ExpansionConfig()
    model = ExpansionModel(config)

    # Scenario: Toronto expansion with 14% league growth
    result = model.simulate_revenue_sharing(
        n_teams_before=12,
        n_teams_after=13,
        growth_mean=0.14,
        growth_std=0.06,
        n_sims=10000
    )

    print("Toronto Expansion Scenario:")
    print(f"  Expected Div'/Div: {result['mean_ratio']:.4f}")
    print(f"  90% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
    print(f"  P(Div' > Div): {result['prob_increase']:.2%}")
    print(f"  Breakeven growth: {result['breakeven_growth']:.2%}")


def example_2_distance_competition():
    """Example 2: Calculate competition intensity between teams."""
    print("\n" + "="*80)
    print("Example 2: Distance-Based Competition Analysis")
    print("="*80 + "\n")

    config = ExpansionConfig()
    model = ExpansionModel(config)

    # Calculate competition between NYL and potential Toronto team
    nyl_loc = WNBA_TEAM_LOCATIONS["NYL"]
    toronto_loc = EXPANSION_CITIES["Toronto"]

    comp_intensity = model.calculate_competition_intensity(nyl_loc, toronto_loc)
    distance = model._haversine_distance(nyl_loc, toronto_loc)

    print(f"New York Liberty vs Toronto:")
    print(f"  Distance: {distance:.1f} km")
    print(f"  Competition Intensity: {comp_intensity:.4f}")
    print(f"  Is Rivalry: {distance < config.rivalry_distance_threshold}")


def example_3_attendance_impact():
    """Example 3: Simulate attendance impact for a specific team."""
    print("\n" + "="*80)
    print("Example 3: Attendance Impact Simulation")
    print("="*80 + "\n")

    config = ExpansionConfig()
    model = ExpansionModel(config)

    # Load real data
    data = load_wnba_expansion_data()
    revenue_df = data['team_revenue']

    # Get Seattle Storm data
    sea_data = revenue_df[revenue_df['team_abbr'] == 'SEA']
    sea_attendance = sea_data['estimated_attendance'].values[0]
    sea_loc = WNBA_TEAM_LOCATIONS['SEA']

    # Test Portland expansion
    portland_loc = EXPANSION_CITIES['Portland']

    result = model.simulate_attendance_impact(
        team_location=sea_loc,
        new_team_location=portland_loc,
        base_attendance=sea_attendance,
        attendance_std=sea_attendance * 0.1
    )

    print(f"Seattle Storm - Portland Expansion:")
    print(f"  Distance: {result['distance_km']:.1f} km")
    print(f"  Is Rivalry: {result['is_rivalry']}")
    print(f"  E[Q'/Q]: {result['mean_ratio']:.4f}")
    print(f"  90% CI: [{result['ci_lower_ratio']:.4f}, {result['ci_upper_ratio']:.4f}]")
    print(f"  E[螖Revenue]: ${result['mean_delta_revenue_m']:.2f}M")
    print(f"  P(Revenue Increase): {result['prob_increase_revenue']:.2%}")


def example_4_playoff_probability():
    """Example 4: Calculate playoff probability changes."""
    print("\n" + "="*80)
    print("Example 4: Playoff Probability Structural Changes")
    print("="*80 + "\n")

    config = ExpansionConfig()
    model = ExpansionModel(config)

    scenarios = [
        (12, 13, "Add 1 team"),
        (12, 14, "Add 2 teams"),
        (12, 16, "Add 4 teams"),
    ]

    for n_before, n_after, desc in scenarios:
        result = model.calculate_playoff_probability_change(
            n_teams_before=n_before,
            n_teams_after=n_after,
            n_playoff_spots=8
        )

        print(f"{desc} ({n_before} ->{n_after} teams):")
        print(f"  Before: {result['prob_before']:.2%}")
        print(f"  After: {result['prob_after']:.2%}")
        print(f"  Change: {result['relative_change']:+.2%}")
        print()


def example_5_custom_scenario():
    """Example 5: Custom expansion scenario with multiple cities."""
    print("\n" + "="*80)
    print("Example 5: Custom Multi-City Expansion Scenario")
    print("="*80 + "\n")

    # Custom configuration
    config = ExpansionConfig(
        kappa=1000.0,  # Larger distance decay
        competition_elasticity=-0.20,  # Stronger competition effect
        rivalry_boost=0.30,  # Larger rivalry boost
        n_simulations=5000  # Fewer simulations for speed
    )

    print("Custom Configuration:")
    print(f"  Distance decay (魏): {config.kappa} km")
    print(f"  Competition elasticity: {config.competition_elasticity}")
    print(f"  Rivalry boost: {config.rivalry_boost}")
    print(f"  Simulations: {config.n_simulations}")
    print()

    # Calculate scenarios
    results = calculate_expansion_scenarios(
        expansion_cities=["Toronto", "Nashville"],
        representative_teams=["NYL", "ATL"],
        config=config
    )

    print("Revenue Sharing Results:")
    print(results['revenue_sharing'][['expansion_city', 'mean_ratio', 'prob_increase']])
    print()

    print("Attendance Impact Results:")
    print(results['attendance_impact'][['team', 'expansion_city', 'mean_delta_revenue_m', 'prob_increase_revenue']])


def example_6_sensitivity_analysis():
    """Example 6: Sensitivity analysis on distance decay parameter."""
    print("\n" + "="*80)
    print("Example 6: Sensitivity Analysis - Distance Decay Parameter")
    print("="*80 + "\n")

    # Test different kappa values
    kappa_values = [500, 800, 1200, 2000]
    nyl_loc = WNBA_TEAM_LOCATIONS["NYL"]
    toronto_loc = EXPANSION_CITIES["Toronto"]

    print("Competition Intensity vs Distance Decay (NYL-Toronto):")
    print(f"Distance: {ExpansionModel(ExpansionConfig())._haversine_distance(nyl_loc, toronto_loc):.1f} km\n")

    for kappa in kappa_values:
        config = ExpansionConfig(kappa=kappa)
        model = ExpansionModel(config)
        comp = model.calculate_competition_intensity(nyl_loc, toronto_loc)
        print(f"  魏 = {kappa:4d} km ->CompInc = {comp:.4f}")


def example_7_all_teams_comparison():
    """Example 7: Compare impact across all teams for one expansion city."""
    print("\n" + "="*80)
    print("Example 7: All Teams Impact - Toronto Expansion")
    print("="*80 + "\n")

    config = ExpansionConfig()
    model = ExpansionModel(config)
    data = load_wnba_expansion_data()
    revenue_df = data['team_revenue']
    toronto_loc = EXPANSION_CITIES['Toronto']

    results = []

    for team_abbr, team_loc in WNBA_TEAM_LOCATIONS.items():
        team_data = revenue_df[revenue_df['team_abbr'] == team_abbr]
        if team_data.empty:
            continue

        base_att = team_data['estimated_attendance'].values[0]

        result = model.simulate_attendance_impact(
            team_location=team_loc,
            new_team_location=toronto_loc,
            base_attendance=base_att,
            attendance_std=base_att * 0.1,
            n_sims=1000  # Fewer sims for speed
        )

        results.append({
            'team': team_abbr,
            'distance_km': result['distance_km'],
            'delta_revenue_m': result['mean_delta_revenue_m'],
            'prob_increase': result['prob_increase_revenue']
        })

    df = pd.DataFrame(results).sort_values('delta_revenue_m', ascending=False)
    print(df.to_string(index=False))


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("TASK 3: EXPANSION IMPACT ANALYSIS - USAGE EXAMPLES")
    print("="*80)

    try:
        example_1_basic_revenue_sharing()
        example_2_distance_competition()
        example_3_attendance_impact()
        example_4_playoff_probability()
        example_5_custom_scenario()
        example_6_sensitivity_analysis()
        example_7_all_teams_comparison()

        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n[ERROR] Example failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

