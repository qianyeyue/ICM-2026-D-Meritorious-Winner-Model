# Model Map

Task 1 is the backbone:

```text
state input -> optimizer -> controls -> simulation -> state update -> reports
```

Readable entry point:

```text
project/model/task1/pipeline.py
```

Numerical implementation:

```text
project/model/simulation/mpc_simulation.py
```

## Task Interfaces

| Task | Role | Main code | Output |
| --- | --- | --- | --- |
| Task 1 | Dynamic optimizer | `task1/pipeline.py` | `period_log`, `summary` |
| Task 2 | Initial state | `task2/pipeline.py` | `data/processed/task2/task1_initial_state.json` |
| Task 3 | Expansion shock | `task3/task1_impact.py` | `task3_to_task1_<team>_<city>.json` |
| Task 4 | Pricing extension | `task4/mpc_integration.py` | dynamic ticket price and attendance terms |
| Task 5 | Injury/risk extension | `task5/impact_analysis.py` | win, revenue, and risk deltas |

## Reading Order

1. `task1/README.md`
2. `task1/pipeline.py`
3. `simulation/mpc_simulation.py`
4. `task2/task1_adapter.py`
5. `task3/task1_impact.py`
6. `task4/mpc_integration.py`
7. `task5/impact_analysis.py`

## Examples

```bash
python -m project.model.task1.pipeline --team LVA --periods 3 --n-mc 10
python -m project.model.task2.pipeline
python -m project.model.task3.task1_impact --team LVA --expansion-city Toronto
```
