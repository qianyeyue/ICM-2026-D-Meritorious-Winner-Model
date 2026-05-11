"""
Unified system configuration for WNBA model.

This module provides a single source of truth for all model parameters,
consolidating configurations from multiple modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .brands import BrandParameters
from .cost import (
    CashFlowParameters,
    FinancingParameters,
    GeneralAdminParameters,
    SalaryCostParameters,
    SportsOperationsParameters,
    VenueCostParameters,
)
from .elo import EloConfig
from .fatigue import FatigueParameters, InjuryRiskParameters
from .income import DemandParameters, RevenueParameters


@dataclass
class SystemConfig:
    """
    Unified system configuration.

    Consolidates all model parameters into a single configuration object,
    providing a single source of truth for the entire system.

    Parameters
    ----------
    elo : EloConfig
        Elo rating system configuration
    demand : DemandParameters
        Demand model parameters
    revenue : RevenueParameters
        Revenue model parameters
    salary : SalaryCostParameters
        Salary cost parameters
    venue : VenueCostParameters
        Venue cost parameters
    ga : GeneralAdminParameters
        General & administrative cost parameters
    sports : SportsOperationsParameters
        Sports operations cost parameters
    financing : FinancingParameters
        Financing cost parameters
    cash : CashFlowParameters
        Cash flow parameters
    fatigue : FatigueParameters
        Fatigue model parameters
    injury : InjuryRiskParameters
        Injury risk model parameters
    brand : BrandParameters
        Brand dynamics parameters

    Examples
    --------
    >>> config = SystemConfig.default()
    >>> config.elo.k
    20.0
    >>> config.demand.beta_0
    10.22

    >>> config = SystemConfig.from_files(Path("project/data/processed"))
    >>> config.elo.home_advantage
    24.0
    """

    elo: EloConfig = field(default_factory=EloConfig)
    demand: DemandParameters = field(default_factory=DemandParameters)
    revenue: RevenueParameters = field(default_factory=RevenueParameters)
    salary: SalaryCostParameters = field(default_factory=SalaryCostParameters)
    venue: VenueCostParameters = field(default_factory=VenueCostParameters)
    ga: GeneralAdminParameters = field(default_factory=GeneralAdminParameters)
    sports: SportsOperationsParameters = field(default_factory=SportsOperationsParameters)
    financing: FinancingParameters = field(default_factory=FinancingParameters)
    cash: CashFlowParameters = field(default_factory=CashFlowParameters)
    fatigue: FatigueParameters = field(default_factory=FatigueParameters)
    injury: InjuryRiskParameters = field(default_factory=InjuryRiskParameters)
    brand: BrandParameters = field(default_factory=BrandParameters)

    @classmethod
    def default(cls) -> SystemConfig:
        """
        Create default configuration with standard parameters.

        Returns
        -------
        SystemConfig
            Configuration with default values

        Examples
        --------
        >>> config = SystemConfig.default()
        >>> config.elo.k
        20.0
        """
        return cls()

    @classmethod
    def from_files(cls, config_dir: str | Path) -> SystemConfig:
        """
        Load configuration from files.

        Currently loads Elo config from CSV. Other parameters use defaults.
        Future versions will support loading all parameters from files.

        Parameters
        ----------
        config_dir : str | Path
            Directory containing configuration files

        Returns
        -------
        SystemConfig
            Configuration loaded from files

        Examples
        --------
        >>> config = SystemConfig.from_files("project/data/processed")
        >>> config.elo.home_advantage
        24.0
        """
        from .data_loader import DataLoader

        config_dir = Path(config_dir)
        loader = DataLoader()

        # Load Elo config
        elo_config_path = config_dir / "elo_config.csv"
        if elo_config_path.is_file():
            elo_dict = loader.load_elo_config(elo_config_path)
            elo = EloConfig(
                k=float(elo_dict.get("k", 20.0)),
                base_elo=float(elo_dict.get("base_elo", 1500.0)),
                season_carryover=float(elo_dict.get("season_carryover", 0.75)),
                home_advantage=float(elo_dict.get("home_advantage", 0.0)),
            )
        else:
            elo = EloConfig()

        # Other parameters use defaults for now
        # Future: load from additional config files
        return cls(
            elo=elo,
            demand=DemandParameters(),
            revenue=RevenueParameters(),
            salary=SalaryCostParameters(),
            venue=VenueCostParameters(),
            ga=GeneralAdminParameters(),
            sports=SportsOperationsParameters(),
            financing=FinancingParameters(),
            cash=CashFlowParameters(),
            fatigue=FatigueParameters(),
            injury=InjuryRiskParameters(),
            brand=BrandParameters(),
        )

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        """
        Convert configuration to nested dictionary.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Nested dictionary with module -> parameter -> value

        Examples
        --------
        >>> config = SystemConfig.default()
        >>> config_dict = config.to_dict()
        >>> config_dict["elo"]["k"]
        20.0
        """
        return {
            "elo": {
                "k": self.elo.k,
                "base_elo": self.elo.base_elo,
                "season_carryover": self.elo.season_carryover,
                "home_advantage": self.elo.home_advantage,
                "alpha_u": self.elo.alpha_u,
                "phi": self.elo.phi,
            },
            "demand": {
                "beta_0": self.demand.beta_0,
                "epsilon": self.demand.epsilon,
                "beta_S": self.demand.beta_S,
                "beta_star": self.demand.beta_star,
                "beta_cal": self.demand.beta_cal,
                "a1": self.demand.a1,
                "a2": self.demand.a2,
                "a3": self.demand.a3,
                "a4": self.demand.a4,
                "sigma_eta": self.demand.sigma_eta,
            },
            "revenue": {
                "p0": self.revenue.p0,
                "kappa_1": self.revenue.kappa_1,
                "kappa_2": self.revenue.kappa_2,
                "delta_0": self.revenue.delta_0,
                "delta_B": self.revenue.delta_B,
                "delta_star": self.revenue.delta_star,
                "theta_rev": self.revenue.theta_rev,
            },
            "salary": {
                "roster_size": self.salary.roster_size,
                "min_salary": self.salary.min_salary,
                "max_salary": self.salary.max_salary,
                "salary_cap": self.salary.salary_cap,
            },
            "venue": {
                "fixed_venue_cost": self.venue.fixed_venue_cost,
                "variable_cost_per_attendee": self.venue.variable_cost_per_attendee,
            },
            "ga": {
                "fixed_ga_cost": self.ga.fixed_ga_cost,
                "tax_rate": self.ga.tax_rate,
            },
            "sports": {
                "base_sports_ops_cost": self.sports.base_sports_ops_cost,
                "travel_cost_per_game": self.sports.travel_cost_per_game,
            },
            "financing": {
                "interest_rate": self.financing.interest_rate,
                "initial_debt": self.financing.initial_debt,
            },
            "cash": {
                "initial_cash": self.cash.initial_cash,
                "min_cash_balance": self.cash.min_cash_balance,
                "annual_capex": self.cash.annual_capex,
            },
            "fatigue": {
                "w1": self.fatigue.w1,
                "w2": self.fatigue.w2,
                "w3": self.fatigue.w3,
                "w4": self.fatigue.w4,
                "r0": self.fatigue.r0,
                "L": self.fatigue.L,
                "max_fatigue": self.fatigue.max_fatigue,
            },
            "injury": {
                "theta_0": self.injury.theta_0,
                "theta_1": self.injury.theta_1,
                "theta_2": self.injury.theta_2,
                "theta_3": self.injury.theta_3,
                "theta_med": self.injury.theta_med,
            },
            "brand": {
                "rho_B": self.brand.rho_B,
                "eta_W": self.brand.eta_W,
                "eta_m": self.brand.eta_m,
                "eta_star": self.brand.eta_star,
            },
        }

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert configuration to DataFrame for easy viewing.

        Returns
        -------
        pd.DataFrame
            Configuration as DataFrame with columns: module, parameter, value

        Examples
        --------
        >>> config = SystemConfig.default()
        >>> df = config.to_dataframe()
        >>> df[df["module"] == "elo"]
        """
        config_dict = self.to_dict()
        rows = []

        for module, params in config_dict.items():
            for param, value in params.items():
                rows.append({
                    "module": module,
                    "parameter": param,
                    "value": value,
                })

        return pd.DataFrame(rows)

    def save_to_csv(self, output_path: str | Path) -> None:
        """
        Save configuration to CSV file.

        Parameters
        ----------
        output_path : str | Path
            Output CSV file path

        Examples
        --------
        >>> config = SystemConfig.default()
        >>> config.save_to_csv("system_config.csv")
        """
        df = self.to_dataframe()
        df.to_csv(output_path, index=False)

    @classmethod
    def from_csv(cls, csv_path: str | Path) -> SystemConfig:
        """
        Load configuration from CSV file.

        Parameters
        ----------
        csv_path : str | Path
            CSV file path with columns: module, parameter, value

        Returns
        -------
        SystemConfig
            Configuration loaded from CSV

        Examples
        --------
        >>> config = SystemConfig.from_csv("system_config.csv")
        """
        df = pd.read_csv(csv_path)

        # Group by module
        config_dict = {}
        for module in df["module"].unique():
            module_df = df[df["module"] == module]
            config_dict[module] = dict(zip(module_df["parameter"], module_df["value"]))

        # Create config objects
        elo = EloConfig(
            k=float(config_dict.get("elo", {}).get("k", 20.0)),
            base_elo=float(config_dict.get("elo", {}).get("base_elo", 1500.0)),
            season_carryover=float(config_dict.get("elo", {}).get("season_carryover", 0.75)),
            home_advantage=float(config_dict.get("elo", {}).get("home_advantage", 0.0)),
            alpha_u=float(config_dict.get("elo", {}).get("alpha_u", 12.0)),
            phi=float(config_dict.get("elo", {}).get("phi", 5.0)),
        )

        demand = DemandParameters(
            beta_0=float(config_dict.get("demand", {}).get("beta_0", 10.22)),
            epsilon=float(config_dict.get("demand", {}).get("epsilon", 0.35)),
            beta_S=float(config_dict.get("demand", {}).get("beta_S", 0.20)),
            beta_star=float(config_dict.get("demand", {}).get("beta_star", 0.50)),
            beta_cal=float(config_dict.get("demand", {}).get("beta_cal", 0.0)),
            a1=float(config_dict.get("demand", {}).get("a1", 0.10)),
            a2=float(config_dict.get("demand", {}).get("a2", 0.20)),
            a3=float(config_dict.get("demand", {}).get("a3", 0.35)),
            a4=float(config_dict.get("demand", {}).get("a4", 0.25)),
            sigma_eta=float(config_dict.get("demand", {}).get("sigma_eta", 0.25)),
        )

        # Other parameters use defaults for brevity
        # Full implementation would load all parameters

        return cls(
            elo=elo,
            demand=demand,
            revenue=RevenueParameters(),
            salary=SalaryCostParameters(),
            venue=VenueCostParameters(),
            ga=GeneralAdminParameters(),
            sports=SportsOperationsParameters(),
            financing=FinancingParameters(),
            cash=CashFlowParameters(),
            fatigue=FatigueParameters(),
            injury=InjuryRiskParameters(),
            brand=BrandParameters(),
        )

    def __repr__(self) -> str:
        """String representation showing key parameters."""
        return (
            f"SystemConfig(\n"
            f"  elo: k={self.elo.k}, base={self.elo.base_elo}, H={self.elo.home_advantage}\n"
            f"  demand: β₀={self.demand.beta_0}, ε={self.demand.epsilon}\n"
            f"  revenue: p₀=${self.revenue.p0}\n"
            f"  salary: cap=${self.salary.salary_cap:,.0f}\n"
            f"  fatigue: φ={self.elo.phi}\n"
            f")"
        )


# Convenience function for quick loading
def load_system_config(config_dir: Optional[str | Path] = None) -> SystemConfig:
    """
    Quick load system configuration.

    Parameters
    ----------
    config_dir : str | Path, optional
        Configuration directory (defaults to project/data/processed)

    Returns
    -------
    SystemConfig
        Loaded configuration

    Examples
    --------
    >>> config = load_system_config()
    >>> config.elo.k
    20.0
    """
    if config_dir is None:
        root_dir = Path(__file__).resolve().parents[3]
        config_dir = root_dir / "project" / "data" / "processed"

    return SystemConfig.from_files(config_dir)
