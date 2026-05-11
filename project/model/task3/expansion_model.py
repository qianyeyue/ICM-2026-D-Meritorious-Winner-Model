"""
Core expansion impact model with three channels:
1. Revenue sharing channel
2. Schedule/travel/fatigue channel
3. Market competition/rivalry channel
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


@dataclass
class ExpansionConfig:
    """Configuration for expansion impact model."""

    # Revenue sharing parameters
    base_league_revenue: float = 200e6  # $200M baseline
    revenue_growth_mean: float = 0.10  # 10% mean growth
    revenue_growth_std: float = 0.05   # 5% std

    # Distance decay parameter (km)
    kappa: float = 800.0  # Distance decay constant

    # Market competition parameters
    base_attendance: float = 6500  # Average attendance
    competition_elasticity: float = -0.15  # Negative impact from competition
    rivalry_boost: float = 0.25  # Positive boost for rivalry games
    rivalry_distance_threshold: float = 500.0  # km for rivalry

    # Schedule/fatigue parameters
    games_per_season: int = 40
    home_games: int = 20
    fatigue_per_b2b: float = 0.05  # Fatigue increase per back-to-back
    travel_fatigue_per_1000km: float = 0.02

    # Financial parameters
    revenue_per_fan: float = 50.0  # $ per attendee
    playoff_revenue_bonus: float = 5e6  # $5M for making playoffs

    # Monte Carlo parameters
    n_simulations: int = 10000
    random_seed: int = 42


class ExpansionModel:
    """Model league expansion impact through multiple channels."""

    def __init__(self, config: ExpansionConfig):
        self.config = config
        self.rng = np.random.RandomState(config.random_seed)

    def calculate_revenue_sharing_impact(
        self,
        n_teams_before: int,
        n_teams_after: int,
        league_growth_rate: float
    ) -> Tuple[float, float]:
        """
        Calculate revenue sharing impact.

        Div' / Div = (1 + g_L) * N_before / N_after

        Returns:
            (expected_ratio, probability_increase)
        """
        ratio = (1 + league_growth_rate) * n_teams_before / n_teams_after
        return ratio, float(ratio > 1.0)

    def simulate_revenue_sharing(
        self,
        n_teams_before: int,
        n_teams_after: int,
        growth_mean: float = None,
        growth_std: float = None,
        n_sims: int = None
    ) -> Dict[str, float]:
        """
        Monte Carlo simulation of revenue sharing impact.

        Returns:
            Dictionary with E[Div'/Div], 90% CI, P(Div'>Div)
        """
        if growth_mean is None:
            growth_mean = self.config.revenue_growth_mean
        if growth_std is None:
            growth_std = self.config.revenue_growth_std
        if n_sims is None:
            n_sims = self.config.n_simulations

        # Sample league growth rates
        growth_rates = self.rng.normal(growth_mean, growth_std, n_sims)

        # Calculate ratios
        ratios = (1 + growth_rates) * n_teams_before / n_teams_after

        # Statistics
        mean_ratio = float(np.mean(ratios))
        ci_lower = float(np.percentile(ratios, 5))
        ci_upper = float(np.percentile(ratios, 95))
        prob_increase = float(np.mean(ratios > 1.0))

        return {
            "mean_ratio": mean_ratio,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "prob_increase": prob_increase,
            "breakeven_growth": float(n_teams_after / n_teams_before - 1)
        }

    def calculate_competition_intensity(
        self,
        team_location: Tuple[float, float],
        new_team_location: Tuple[float, float]
    ) -> float:
        """
        Calculate competition intensity using distance decay.

        CompInc = exp(-dist/κ)

        Args:
            team_location: (lat, lon) of existing team
            new_team_location: (lat, lon) of new team

        Returns:
            Competition intensity in [0, 1]
        """
        dist_km = self._haversine_distance(team_location, new_team_location)
        comp_inc = np.exp(-dist_km / self.config.kappa)
        return float(comp_inc)

    def _haversine_distance(
        self,
        loc1: Tuple[float, float],
        loc2: Tuple[float, float]
    ) -> float:
        """Calculate great circle distance in km."""
        lat1, lon1 = np.radians(loc1)
        lat2, lon2 = np.radians(loc2)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        return 6371.0 * c  # Earth radius in km

    def calculate_attendance_impact(
        self,
        base_attendance: float,
        competition_intensity: float,
        is_rivalry: bool = False
    ) -> Tuple[float, float]:
        """
        Calculate attendance impact from new team.

        For non-rivalry games:
            Q' / Q = 1 + elasticity * CompInc

        For rivalry games:
            Q' / Q = 1 + rivalry_boost

        Returns:
            (attendance_ratio, expected_delta_revenue)
        """
        if is_rivalry:
            ratio = 1.0 + self.config.rivalry_boost
        else:
            # Competition generally increases interest (positive elasticity for expansion)
            # but can be negative for very close teams
            ratio = 1.0 - self.config.competition_elasticity * competition_intensity

        delta_attendance = base_attendance * (ratio - 1.0) * self.config.home_games
        delta_revenue = delta_attendance * self.config.revenue_per_fan

        return float(ratio), float(delta_revenue)

    def simulate_attendance_impact(
        self,
        team_location: Tuple[float, float],
        new_team_location: Tuple[float, float],
        base_attendance: float,
        attendance_std: float = 500.0,
        n_sims: int = None
    ) -> Dict[str, float]:
        """
        Monte Carlo simulation of attendance impact.

        Returns:
            Dictionary with E[Q'/Q], 90% CI, P(Q'>Q), E[Δπ], CI, P(Δπ>0)
        """
        if n_sims is None:
            n_sims = self.config.n_simulations

        # Calculate competition intensity
        comp_int = self.calculate_competition_intensity(team_location, new_team_location)
        dist_km = self._haversine_distance(team_location, new_team_location)
        is_rivalry = dist_km < self.config.rivalry_distance_threshold

        # Sample base attendance with uncertainty
        base_att_samples = self.rng.normal(base_attendance, attendance_std, n_sims)
        base_att_samples = np.maximum(base_att_samples, 1000)  # Floor at 1000

        # Calculate ratios and revenue impacts
        ratios = np.zeros(n_sims)
        delta_revenues = np.zeros(n_sims)

        for i in range(n_sims):
            ratio, delta_rev = self.calculate_attendance_impact(
                base_att_samples[i],
                comp_int,
                is_rivalry
            )
            ratios[i] = ratio
            delta_revenues[i] = delta_rev / 1e6  # Convert to millions

        return {
            "mean_ratio": float(np.mean(ratios)),
            "ci_lower_ratio": float(np.percentile(ratios, 5)),
            "ci_upper_ratio": float(np.percentile(ratios, 95)),
            "prob_increase_attendance": float(np.mean(ratios > 1.0)),
            "mean_delta_revenue_m": float(np.mean(delta_revenues)),
            "ci_lower_revenue_m": float(np.percentile(delta_revenues, 5)),
            "ci_upper_revenue_m": float(np.percentile(delta_revenues, 95)),
            "prob_increase_revenue": float(np.mean(delta_revenues > 0.0)),
            "distance_km": dist_km,
            "competition_intensity": comp_int,
            "is_rivalry": is_rivalry
        }

    def calculate_playoff_probability_change(
        self,
        n_teams_before: int,
        n_teams_after: int,
        n_playoff_spots: int = 8
    ) -> Dict[str, float]:
        """
        Calculate structural change in playoff probability.

        Returns:
            Dictionary with probabilities before/after and change
        """
        prob_before = n_playoff_spots / n_teams_before
        prob_after = n_playoff_spots / n_teams_after

        return {
            "prob_before": prob_before,
            "prob_after": prob_after,
            "absolute_change": prob_after - prob_before,
            "relative_change": (prob_after - prob_before) / prob_before
        }

    def calculate_travel_burden_change(
        self,
        team_location: Tuple[float, float],
        existing_team_locations: List[Tuple[float, float]],
        new_team_location: Tuple[float, float] = None
    ) -> Dict[str, float]:
        """
        Calculate change in average travel distance.

        Returns:
            Dictionary with avg distance before/after and change
        """
        # Calculate current average distance
        distances_before = [
            self._haversine_distance(team_location, other_loc)
            for other_loc in existing_team_locations
            if other_loc != team_location
        ]
        avg_before = float(np.mean(distances_before)) if distances_before else 0.0

        if new_team_location is None:
            return {
                "avg_distance_before_km": avg_before,
                "avg_distance_after_km": avg_before,
                "change_km": 0.0
            }

        # Calculate new average distance
        all_locations = existing_team_locations + [new_team_location]
        distances_after = [
            self._haversine_distance(team_location, other_loc)
            for other_loc in all_locations
            if other_loc != team_location
        ]
        avg_after = float(np.mean(distances_after)) if distances_after else 0.0

        return {
            "avg_distance_before_km": avg_before,
            "avg_distance_after_km": avg_after,
            "change_km": avg_after - avg_before,
            "relative_change": (avg_after - avg_before) / avg_before if avg_before > 0 else 0.0
        }

    def calculate_talent_dilution_impact(
        self,
        team_elo: float,
        roster_depth: int = 12,
        expansion_draft_protection: int = 6,
        n_new_teams: int = 1
    ) -> Dict[str, float]:
        """
        Calculate talent dilution impact from expansion draft.

        Channel 4: Expansion draft causes roster depth loss.
        Paper shows: ΔS^talent ∈ {0, -25, -50} Elo points

        Args:
            team_elo: Current team Elo rating
            roster_depth: Total roster size
            expansion_draft_protection: Number of protected players
            n_new_teams: Number of expansion teams

        Returns:
            Dictionary with:
            - expected_delta_elo: Expected Elo change
            - expected_delta_wins: Expected wins change (over 40 games)
            - prob_no_loss: Probability no key player lost
            - prob_rotation_loss: Probability rotation player lost
            - prob_key_loss: Probability key rotation player lost
            - scenarios: Dict of scenario outcomes
        """
        # Number of exposed players
        exposed_players = roster_depth - expansion_draft_protection

        # Probability model based on paper's LVA analysis
        # Higher Elo teams more likely to lose quality players
        elo_factor = (team_elo - 1500) / 100  # Normalize around league average

        # Base probabilities (calibrated to paper's -1.06 wins result)
        prob_no_loss = max(0.1, 0.4 - 0.05 * elo_factor * n_new_teams)
        prob_key_loss = min(0.4, 0.15 + 0.05 * elo_factor * n_new_teams)
        prob_rotation_loss = 1.0 - prob_no_loss - prob_key_loss

        # Elo impact scenarios (from paper)
        elo_scenarios = {
            "no_loss": 0,
            "rotation_loss": -25,
            "key_loss": -50
        }

        # Expected Elo change
        expected_delta_elo = (
            prob_no_loss * elo_scenarios["no_loss"] +
            prob_rotation_loss * elo_scenarios["rotation_loss"] +
            prob_key_loss * elo_scenarios["key_loss"]
        )

        # Convert Elo to wins (approx: 25 Elo ≈ 0.5 wins over 40 games)
        # Paper shows -1.06 wins for LVA with 2 expansion teams
        wins_per_elo = 0.02  # 0.5 wins / 25 Elo
        expected_delta_wins = expected_delta_elo * wins_per_elo

        # Win probability impact (for playoff calculations)
        # Elo difference of 100 ≈ 64% vs 36% win probability
        # So delta_elo affects win rate by approximately delta_elo/400
        win_rate_change = expected_delta_elo / 400

        return {
            "expected_delta_elo": float(expected_delta_elo),
            "expected_delta_wins": float(expected_delta_wins),
            "win_rate_change": float(win_rate_change),
            "prob_no_loss": float(prob_no_loss),
            "prob_rotation_loss": float(prob_rotation_loss),
            "prob_key_loss": float(prob_key_loss),
            "scenarios": {
                "no_loss": {
                    "delta_elo": float(elo_scenarios["no_loss"]),
                    "delta_wins": 0.0,
                    "probability": float(prob_no_loss)
                },
                "rotation_loss": {
                    "delta_elo": float(elo_scenarios["rotation_loss"]),
                    "delta_wins": float(elo_scenarios["rotation_loss"] * wins_per_elo),
                    "probability": float(prob_rotation_loss)
                },
                "key_loss": {
                    "delta_elo": float(elo_scenarios["key_loss"]),
                    "delta_wins": float(elo_scenarios["key_loss"] * wins_per_elo),
                    "probability": float(prob_key_loss)
                }
            }
        }

    def simulate_talent_dilution(
        self,
        team_elo: float,
        roster_depth: int = 12,
        expansion_draft_protection: int = 6,
        n_new_teams: int = 1,
        n_sims: int = None
    ) -> Dict[str, float]:
        """
        Monte Carlo simulation of talent dilution impact.

        Returns:
            Dictionary with E[ΔW], 90% CI, P(ΔW < 0), scenario distribution
        """
        if n_sims is None:
            n_sims = self.config.n_simulations

        # Get probabilities
        impact = self.calculate_talent_dilution_impact(
            team_elo, roster_depth, expansion_draft_protection, n_new_teams
        )

        # Sample scenarios
        scenarios = ["no_loss", "rotation_loss", "key_loss"]
        probs = [
            impact["prob_no_loss"],
            impact["prob_rotation_loss"],
            impact["prob_key_loss"]
        ]

        # Draw samples
        scenario_samples = self.rng.choice(scenarios, size=n_sims, p=probs)

        # Calculate wins impact for each sample
        delta_wins = np.zeros(n_sims)
        delta_elo = np.zeros(n_sims)

        for i, scenario in enumerate(scenario_samples):
            delta_wins[i] = impact["scenarios"][scenario]["delta_wins"]
            delta_elo[i] = impact["scenarios"][scenario]["delta_elo"]

        return {
            "mean_delta_wins": float(np.mean(delta_wins)),
            "ci_lower_wins": float(np.percentile(delta_wins, 5)),
            "ci_upper_wins": float(np.percentile(delta_wins, 95)),
            "prob_negative_impact": float(np.mean(delta_wins < 0)),
            "mean_delta_elo": float(np.mean(delta_elo)),
            "ci_lower_elo": float(np.percentile(delta_elo, 5)),
            "ci_upper_elo": float(np.percentile(delta_elo, 95)),
            "scenario_distribution": {
                "no_loss": float(np.mean(scenario_samples == "no_loss")),
                "rotation_loss": float(np.mean(scenario_samples == "rotation_loss")),
                "key_loss": float(np.mean(scenario_samples == "key_loss"))
            }
        }
