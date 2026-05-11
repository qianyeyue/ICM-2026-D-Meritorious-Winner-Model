"""
Main execution script for Task 3: League Expansion Impact Analysis.

Usage:
    python -m project.model.task3.main
"""

from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np
from datetime import datetime

from .expansion_model import ExpansionModel, ExpansionConfig
from .scenarios import (
    calculate_expansion_scenarios,
    format_results_for_paper,
    generate_summary_statistics
)
from .visualization import create_all_visualizations
from .data_loader import load_wnba_expansion_data


def print_section_header(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_dataframe_summary(df: pd.DataFrame, title: str, max_rows: int = 20):
    """Print DataFrame with title."""
    print(f"\n{title}")
    print("-" * 80)
    if len(df) > max_rows:
        print(df.head(max_rows))
        print(f"\n... ({len(df) - max_rows} more rows)")
    else:
        print(df)
    print()


def save_results(results: Dict[str, pd.DataFrame], output_dir: Path):
    """Save all results to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, df in results.items():
        output_path = output_dir / f"{name}.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved {name} to {output_path}")


def generate_markdown_report(
    results: Dict[str, pd.DataFrame],
    summary_stats: pd.DataFrame,
    output_path: Path
):
    """Generate markdown report with results."""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Task 3: League Expansion Impact Analysis (WNBA)\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Executive Summary\n\n")
        f.write("This analysis quantifies the impact of WNBA expansion through three channels:\n\n")
        f.write("1. **Revenue Sharing Channel**: Div_t 鈭?1/N_t\n")
        f.write("2. **Schedule/Travel Channel**: Fatigue ->Win rate/Injury ->Profit/Risk\n")
        f.write("3. **Market Competition Channel**: Distance decay model for local competition\n\n")

        f.write("## Summary Statistics\n\n")
        f.write(summary_stats.to_string(index=False))
        f.write("\n\n")

        f.write("## 1. Revenue Sharing Impact\n\n")
        f.write("### Formula\n\n")
        f.write("$$\\frac{Div'}{Div} = \\frac{(1+g_L) \\cdot N_{before}}{N_{after}}$$\n\n")
        f.write("Where:\n")
        f.write("- $g_L$ = League revenue growth rate\n")
        f.write("- $N_{before}$ = 12 teams\n")
        f.write("- $N_{after}$ = 13 teams\n\n")
        f.write("### Results\n\n")
        f.write(results['revenue_sharing'].to_string(index=False))
        f.write("\n\n")

        f.write("**Key Findings:**\n")
        rev_df = results['revenue_sharing']
        for _, row in rev_df.iterrows():
            city = row['expansion_city']
            mean_ratio = row['mean_ratio']
            prob = row['prob_increase']
            breakeven = row['breakeven_growth']

            f.write(f"- **{city}**: E[Div'/Div] = {mean_ratio:.3f}, ")
            f.write(f"P(increase) = {prob:.2%}, ")
            f.write(f"breakeven growth = {breakeven:.2%}\n")
        f.write("\n")

        f.write("## 2. Attendance & Revenue Impact\n\n")
        f.write("### Distance Decay Model\n\n")
        f.write("$$CompInc = \\exp(-dist/\\kappa), \\quad \\kappa = 800 \\text{ km}$$\n\n")
        f.write("### Results by Team and Expansion City\n\n")

        att_df = results['attendance_impact']

        # Group by team
        for team in att_df['team'].unique():
            team_data = att_df[att_df['team'] == team]
            market_size = team_data['market_size'].iloc[0]

            f.write(f"#### {team} ({market_size.upper()} market)\n\n")

            display_cols = [
                'expansion_city', 'distance_km', 'mean_attendance_ratio',
                'prob_increase_attendance', 'mean_delta_revenue_m', 'prob_increase_revenue'
            ]
            f.write(team_data[display_cols].to_string(index=False))
            f.write("\n\n")

        f.write("## 3. Playoff Probability Changes\n\n")
        f.write("### Structural Impact\n\n")
        f.write("With 8 playoff spots fixed:\n\n")
        f.write(results['playoff_probability'].to_string(index=False))
        f.write("\n\n")

        playoff_df = results['playoff_probability']
        for _, row in playoff_df.iterrows():
            n_after = row['n_teams_after']
            prob_before = row['prob_before']
            prob_after = row['prob_after']
            rel_change = row['relative_change']

            f.write(f"- **{n_after} teams**: {prob_before:.1%} ->{prob_after:.1%} ")
            f.write(f"({rel_change:+.1%} change)\n")
        f.write("\n")

        f.write("## 4. Travel Burden Changes\n\n")
        f.write("### Average Travel Distance Impact\n\n")

        travel_df = results['travel_burden']

        # Group by team
        for team in travel_df['team'].unique():
            team_data = travel_df[travel_df['team'] == team]

            f.write(f"#### {team}\n\n")
            f.write(team_data.to_string(index=False))
            f.write("\n\n")

        f.write("## Methodology\n\n")
        f.write("### Monte Carlo Simulation\n\n")
        f.write("- **Simulations**: 10,000 per scenario\n")
        f.write("- **Confidence Intervals**: 90% (5th to 95th percentile)\n")
        f.write("- **Random Seed**: 42 (reproducible)\n\n")

        f.write("### Model Parameters\n\n")
        config = ExpansionConfig()
        f.write(f"- Distance decay constant (魏): {config.kappa} km\n")
        f.write(f"- Competition elasticity: {config.competition_elasticity}\n")
        f.write(f"- Rivalry boost: {config.rivalry_boost}\n")
        f.write(f"- Rivalry distance threshold: {config.rivalry_distance_threshold} km\n")
        f.write(f"- Revenue per fan: ${config.revenue_per_fan}\n")
        f.write(f"- Home games per season: {config.home_games}\n\n")

        f.write("## Conclusions\n\n")
        f.write("1. **Revenue Sharing**: Expansion is net positive if league growth exceeds ~8.3%\n")
        f.write("2. **Market Competition**: Varies by distance; nearby teams face more competition\n")
        f.write("3. **Playoff Access**: Becomes structurally harder (66.7% ->61.5% for 13 teams)\n")
        f.write("4. **Travel Burden**: Generally increases, especially for geographically isolated teams\n\n")

        f.write("## Visualizations\n\n")
        f.write("See the `outputs/` directory for:\n")
        f.write("- `revenue_sharing_impact.png`\n")
        f.write("- `attendance_impact_heatmap.png`\n")
        f.write("- `playoff_probability_change.png`\n")
        f.write("- `travel_burden_change.png`\n")
        f.write("- `distance_decay_curve.png`\n")


def main():
    """Main execution function."""

    print_section_header("TASK 3: LEAGUE EXPANSION IMPACT ANALYSIS (WNBA)")

    # Setup output directory
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}\n")

    # Load data
    print_section_header("1. Loading WNBA Data")

    try:
        data = load_wnba_expansion_data()
        print(f"[OK] Loaded {len(data['team_locations'])} team locations")
        print(f"[OK] Loaded {len(data['expansion_cities'])} expansion city candidates")
        print(f"[OK] Loaded revenue data for {len(data['team_revenue'])} teams")
        print(f"[OK] Loaded attendance history: {len(data['attendance_history'])} years")

        print("\nCurrent WNBA Teams:")
        for abbr, loc in sorted(data['team_locations'].items()):
            print(f"  {abbr}: {loc}")

        print("\nExpansion City Candidates:")
        for city, loc in sorted(data['expansion_cities'].items()):
            print(f"  {city}: {loc}")

    except Exception as e:
        print(f"[ERROR] Error loading data: {e}")
        return 1

    # Calculate scenarios
    print_section_header("2. Calculating Expansion Scenarios")

    expansion_cities = ["Toronto", "Bay Area", "Portland"]
    representative_teams = ["NYL", "CON", "SEA"]

    print(f"Expansion cities: {', '.join(expansion_cities)}")
    print(f"Representative teams: {', '.join(representative_teams)}")
    print("\nRunning Monte Carlo simulations (10,000 iterations per scenario)...")

    try:
        results = calculate_expansion_scenarios(
            expansion_cities=expansion_cities,
            representative_teams=representative_teams
        )
        print("[OK] Scenarios calculated successfully")

    except Exception as e:
        print(f"[ERROR] Error calculating scenarios: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Display results
    print_section_header("3. Results Summary")

    print_dataframe_summary(
        results['revenue_sharing'],
        "Revenue Sharing Impact"
    )

    print_dataframe_summary(
        results['attendance_impact'],
        "Attendance & Revenue Impact"
    )

    print_dataframe_summary(
        results['playoff_probability'],
        "Playoff Probability Changes"
    )

    print_dataframe_summary(
        results['travel_burden'],
        "Travel Burden Changes"
    )

    # Generate summary statistics
    print_section_header("4. Summary Statistics")

    summary_stats = generate_summary_statistics(results)
    print(summary_stats.to_string(index=False))

    # Save results
    print_section_header("5. Saving Results")

    save_results(results, output_dir)

    # Save summary statistics
    summary_path = output_dir / "summary_statistics.csv"
    summary_stats.to_csv(summary_path, index=False)
    print(f"Saved summary statistics to {summary_path}")

    # Generate markdown report
    report_path = output_dir / "expansion_analysis_report.md"
    generate_markdown_report(results, summary_stats, report_path)
    print(f"Saved markdown report to {report_path}")

    # Create visualizations
    print_section_header("6. Creating Visualizations")

    try:
        create_all_visualizations(results, output_dir)
        print("[OK] All visualizations created successfully")

    except Exception as e:
        print(f"[ERROR] Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()

    # Generate LaTeX tables (optional)
    print_section_header("7. Generating LaTeX Tables")

    try:
        latex_tables = format_results_for_paper(results)

        latex_dir = output_dir / "latex"
        latex_dir.mkdir(exist_ok=True)

        for name, latex_str in latex_tables.items():
            latex_path = latex_dir / f"{name}.tex"
            with open(latex_path, 'w', encoding='utf-8') as f:
                f.write(latex_str)
            print(f"Saved LaTeX table: {latex_path}")

    except Exception as e:
        print(f"[ERROR] Error generating LaTeX tables: {e}")

    # Final summary
    print_section_header("ANALYSIS COMPLETE")

    print("Key Findings:")
    print()

    # Revenue sharing
    rev_df = results['revenue_sharing']
    print("1. Revenue Sharing Impact:")
    for _, row in rev_df.iterrows():
        city = row['expansion_city']
        mean_ratio = row['mean_ratio']
        prob = row['prob_increase']
        print(f"   - {city}: E[Div'/Div] = {mean_ratio:.3f}, P(increase) = {prob:.1%}")
    print()

    # Attendance impact
    att_df = results['attendance_impact']
    print("2. Attendance Impact (selected examples):")
    for team in representative_teams:
        team_data = att_df[att_df['team'] == team]
        best_city = team_data.loc[team_data['mean_delta_revenue_m'].idxmax()]
        print(f"   - {team}: Best with {best_city['expansion_city']} "
              f"(+${best_city['mean_delta_revenue_m']:.2f}M, "
              f"P={best_city['prob_increase_revenue']:.1%})")
    print()

    # Playoff probability
    playoff_df = results['playoff_probability']
    print("3. Playoff Probability:")
    for _, row in playoff_df.iterrows():
        print(f"   - {row['n_teams_after']} teams: "
              f"{row['prob_before']:.1%} ->{row['prob_after']:.1%} "
              f"({row['relative_change']:+.1%})")
    print()

    print(f"\nAll results saved to: {output_dir}")
    print(f"Report available at: {report_path}")
    print()

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

