# Task 1 Backbone

Task 1 is the main dynamic franchise model. It owns the state, controls, and
objective function:

```text
state    X_t = (S_t, B_t, Star_t, Cash_t, D_t)
controls a_t = (u_t, m_t, tau_t, d_t)
objective J = profit + win utility + playoff value + terminal value - risk
```

The original implementation remains in:

```text
project/model/simulation/mpc_simulation.py
```

This folder adds a clearer public entry point:

```bash
python -m project.model.task1.pipeline --team LVA --periods 3 --n-mc 10
```

## How Other Tasks Attach

Task 2 provides an initial state:

```bash
python -m project.model.task1.pipeline --task2-state project/data/processed/task2/task1_initial_state.json
```

Task 3 provides an expansion shock:

```bash
python -m project.model.task3.task1_impact --team LVA --expansion-city Toronto
python -m project.model.task1.pipeline --task3-impact project/model/task3/outputs/task3_to_task1_LVA_Toronto.json
```

Task 4 extends pricing and revenue through `project/model/task4/`.

Task 5 extends injury, volatility, and CVaR-style risk through
`project/model/task5/`.
