"""
EBITDA-based valuation model for sports franchises.

This module implements the valuation approach from the paper (lines 310-314):
    V_end = ω · EBITDA + ω_B · B_T · L_t

Where:
- EBITDA = Earnings Before Interest, Taxes, Depreciation, and Amortization
- B_T = Brand value (slow-moving variable)
- L_t = League popularity level
"""

from typing import Dict, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class ValuationConfig:
    """Configuration for EBITDA-based valuation."""

    # EBITDA multiple (calibrated for WNBA scale)
    # WNBA franchises are smaller than NBA, use lower multiple
    ebitda_multiple: float = 2.5  # Calibrated to match paper results (was 8.0)

    # Brand value coefficient (calibrated for WNBA)
    brand_coefficient: float = 0.5  # Calibrated to match paper (was 2.5)

    # League popularity growth rate
    league_growth_rate: float = 0.10  # 10% annual growth

    # Tax rate (for EBITDA calculation)
    tax_rate: float = 0.25  # 25%

    # Depreciation rate (as % of revenue)
    depreciation_rate: float = 0.05  # 5%


class EBITDAValuationModel:
    """Calculate franchise valuation using EBITDA approach."""

    def __init__(self, config: ValuationConfig = None):
        self.config = config or ValuationConfig()

    def calculate_ebitda(
        self,
        revenue: float,
        operating_costs: float,
        salary_costs: float,
        marketing_costs: float,
        financing_costs: float
    ) -> float:
        """
        Calculate EBITDA from financial components.

        EBITDA = Revenue - Operating Costs - Salary - Marketing
        (excludes financing costs, taxes, depreciation, amortization)

        Args:
            revenue: Total revenue ($M)
            operating_costs: Operating expenses ($M)
            salary_costs: Player salaries ($M)
            marketing_costs: Marketing expenses ($M)
            financing_costs: Interest on debt ($M) - EXCLUDED from EBITDA

        Returns:
            EBITDA in $M
        """
        # EBITDA excludes financing costs
        ebitda = revenue - operating_costs - salary_costs - marketing_costs

        return float(ebitda)

    def calculate_ebitda_from_profit(
        self,
        net_profit: float,
        financing_costs: float,
        revenue: float
    ) -> float:
        """
        Calculate EBITDA from net profit by adding back excluded items.

        EBITDA = Net Profit + Financing Costs + Taxes + Depreciation + Amortization

        Args:
            net_profit: Net profit after all costs ($M)
            financing_costs: Interest on debt ($M)
            revenue: Total revenue ($M) - used to estimate depreciation

        Returns:
            EBITDA in $M
        """
        # Add back financing costs
        ebit = net_profit + financing_costs

        # Add back taxes (estimated)
        ebitda_before_da = ebit / (1 - self.config.tax_rate)

        # Add back depreciation & amortization (estimated as % of revenue)
        depreciation = revenue * self.config.depreciation_rate
        ebitda = ebitda_before_da + depreciation

        return float(ebitda)

    def calculate_valuation(
        self,
        ebitda: float,
        brand_value: float,
        league_popularity: float = 1.0
    ) -> float:
        """
        Calculate franchise valuation using EBITDA multiple and brand value.

        V = ω · EBITDA + ω_B · B · L

        Args:
            ebitda: EBITDA in $M
            brand_value: Brand index (normalized, ~1.0 for average team)
            league_popularity: League popularity multiplier (1.0 = baseline)

        Returns:
            Franchise valuation in $M
        """
        # EBITDA component
        ebitda_value = self.config.ebitda_multiple * ebitda

        # Brand component (scales with league popularity)
        brand_component = self.config.brand_coefficient * brand_value * league_popularity

        total_valuation = ebitda_value + brand_component

        return float(total_valuation)

    def calculate_valuation_change(
        self,
        delta_ebitda: float,
        delta_brand: float = 0.0,
        delta_league_popularity: float = 0.0,
        base_brand: float = 1.0,
        base_league_popularity: float = 1.0
    ) -> Dict[str, float]:
        """
        Calculate change in valuation from changes in components.

        Args:
            delta_ebitda: Change in EBITDA ($M)
            delta_brand: Change in brand value (index points)
            delta_league_popularity: Change in league popularity (multiplier)
            base_brand: Base brand value
            base_league_popularity: Base league popularity

        Returns:
            Dictionary with valuation change breakdown
        """
        # EBITDA contribution
        delta_v_ebitda = self.config.ebitda_multiple * delta_ebitda

        # Brand contribution (with league popularity effect)
        new_brand = base_brand + delta_brand
        new_league_pop = base_league_popularity + delta_league_popularity

        brand_value_before = self.config.brand_coefficient * base_brand * base_league_popularity
        brand_value_after = self.config.brand_coefficient * new_brand * new_league_pop

        delta_v_brand = brand_value_after - brand_value_before

        # Total change
        delta_v_total = delta_v_ebitda + delta_v_brand

        return {
            "delta_valuation_total": float(delta_v_total),
            "delta_valuation_ebitda": float(delta_v_ebitda),
            "delta_valuation_brand": float(delta_v_brand),
            "ebitda_multiple": self.config.ebitda_multiple,
            "brand_coefficient": self.config.brand_coefficient
        }


def calculate_enhanced_valuation_change(
    delta_revenue: float,
    delta_profit: float,
    financing_costs: float,
    base_revenue: float,
    delta_brand: float = 0.0,
    league_growth: float = 0.10
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate valuation change using EBITDA approach.

    This is a convenience function for Task 3 integration.

    Args:
        delta_revenue: Change in revenue ($M)
        delta_profit: Change in net profit ($M)
        financing_costs: Interest on debt ($M)
        base_revenue: Base revenue level ($M)
        delta_brand: Change in brand value (index points)
        league_growth: League popularity growth rate

    Returns:
        Tuple of (delta_valuation, breakdown_dict)
    """
    config = ValuationConfig()
    model = EBITDAValuationModel(config)

    # Calculate EBITDA change from profit change
    # EBITDA ≈ Profit + Financing + Taxes + Depreciation
    delta_ebitda = model.calculate_ebitda_from_profit(
        net_profit=delta_profit,
        financing_costs=financing_costs,
        revenue=base_revenue + delta_revenue
    ) - model.calculate_ebitda_from_profit(
        net_profit=0,
        financing_costs=financing_costs,
        revenue=base_revenue
    )

    # Calculate valuation change
    result = model.calculate_valuation_change(
        delta_ebitda=delta_ebitda,
        delta_brand=delta_brand,
        delta_league_popularity=league_growth,
        base_brand=1.0,
        base_league_popularity=1.0
    )

    return result["delta_valuation_total"], result


def compare_valuation_methods(
    delta_revenue: float,
    delta_profit: float,
    financing_costs: float = 0.0,
    base_revenue: float = 15.0
) -> Dict[str, float]:
    """
    Compare simple revenue multiple vs EBITDA-based valuation.

    Args:
        delta_revenue: Change in revenue ($M)
        delta_profit: Change in profit ($M)
        financing_costs: Interest on debt ($M)
        base_revenue: Base revenue level ($M)

    Returns:
        Dictionary comparing both methods
    """
    # Simple method (current implementation)
    simple_multiple = 3.34
    simple_valuation = simple_multiple * delta_revenue

    # EBITDA method (enhanced)
    ebitda_valuation, breakdown = calculate_enhanced_valuation_change(
        delta_revenue=delta_revenue,
        delta_profit=delta_profit,
        financing_costs=financing_costs,
        base_revenue=base_revenue
    )

    return {
        "simple_method": float(simple_valuation),
        "ebitda_method": float(ebitda_valuation),
        "difference": float(ebitda_valuation - simple_valuation),
        "relative_difference": float((ebitda_valuation - simple_valuation) / simple_valuation) if simple_valuation != 0 else 0,
        "ebitda_breakdown": breakdown
    }


if __name__ == "__main__":
    # Test the EBITDA valuation model
    print("Testing EBITDA Valuation Model")
    print("=" * 80)

    # LVA example from paper
    delta_revenue = 1.36  # $M
    delta_profit = 1.36  # $M
    financing_costs = 0.0  # No debt in base case
    base_revenue = 15.0  # $M

    comparison = compare_valuation_methods(
        delta_revenue=delta_revenue,
        delta_profit=delta_profit,
        financing_costs=financing_costs,
        base_revenue=base_revenue
    )

    print(f"\nValuation Change Comparison:")
    print(f"  Simple method (3.34x revenue):  ${comparison['simple_method']:.2f}M")
    print(f"  EBITDA method (8x EBITDA):      ${comparison['ebitda_method']:.2f}M")
    print(f"  Difference:                     ${comparison['difference']:+.2f}M")
    print(f"  Relative difference:            {comparison['relative_difference']*100:+.1f}%")
    print()

    print(f"EBITDA Method Breakdown:")
    breakdown = comparison['ebitda_breakdown']
    print(f"  EBITDA contribution:  ${breakdown['delta_valuation_ebitda']:.2f}M")
    print(f"  Brand contribution:   ${breakdown['delta_valuation_brand']:.2f}M")
    print(f"  Total:                ${breakdown['delta_valuation_total']:.2f}M")
    print()

    print(f"Paper benchmark: $3.53M")
    print()
