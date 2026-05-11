# Task 4: Dynamic Ticket Pricing

Task 4 adds ticket pricing to the Task 1 model. It works with a price multiplier
for each period, a season-ticket term, and an attendance-based home advantage.

## Main Files

```text
task4_dynamic_pricing.py        demand and pricing functions
integrated_pricing_model.py     combined pricing model
mpc_integration.py              connection to Task 1
run_task4_example.py            runnable example
task4_full_simulation.py        larger simulation script
```

## Basic Use

```python
from project.model.task4.task4_dynamic_pricing import (
    DynamicPricingParameters,
    calculate_single_game_demand,
)

params = DynamicPricingParameters()
demand = calculate_single_game_demand(
    tau=1.0,
    S_t=1600,
    Star_t=50,
    S_opp=1550,
    league_pop=1.0,
    period=0,
    params=params,
)
```

With the default calibration this baseline example is about 11,000 seats. The
capacity cap is still applied, but ordinary price checks should not all collapse
to 12,000.

For Task 1 integration, read `mpc_integration.py`.
