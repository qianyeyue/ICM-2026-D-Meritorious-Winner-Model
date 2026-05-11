# Task 5: Injury and Risk

Task 5 models player injury risk and its effect on wins, revenue, and the Task 1
objective. The code is split into small pieces so each part can be tested on its
own.

## Main Files

```text
injury_model.py          injury probability and expected absence
markov_volatility.py     volatility state model
network_model.py         roster dependency graph
mitigation_strategies.py strategy cost and effect estimates
impact_analysis.py       combines the parts into a Task 5 result
data_utils.py            data loading and calibration helpers
```

## Run

```bash
python -m project.model.task5.impact_analysis
```

Full runs need the real roster, schedule, and player-level data files. They are
not included in this repo.
