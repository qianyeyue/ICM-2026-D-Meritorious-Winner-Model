# Task 3: Expansion Shock

Task 3 estimates how a league expansion scenario changes travel, market
competition, revenue sharing, and playoff probability. The public interface is
`task1_impact.py`, which exports a JSON shock that Task 1 can read.

## Main Files

```text
task1_impact.py          export Task 3 output for Task 1
lva_analysis.py          scenario calculation
expansion_model.py       market and travel terms
playoff_simulator.py     playoff probability simulation
ebitda_valuation.py      valuation helper
scenarios.py             reusable scenario definitions
```

Generated figures and older dashboard scripts are not part of the public
release.

## Run

```bash
python -m project.model.task3.task1_impact --team LVA --expansion-city Toronto
```

The command writes a JSON file under `project/model/task3/outputs/` when output
directories are available.

## Notes

Some functions expect full WNBA data files. The sample files in this repo are
only enough to show the expected schema.
