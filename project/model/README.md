# Model Code

Task 1 is the main model. It reads a state, chooses controls, runs the
simulation, and writes period-level results.

```text
Task 1  dynamic franchise model
Task 2  roster and initial state
Task 3  expansion shock
Task 4  ticket pricing
Task 5  injury and risk
```

Start with:

```text
MODEL_MAP.md
task1/README.md
task1/pipeline.py
```

Install dependencies from the repo root:

```bash
pip install -r requirements.txt
```

Useful commands:

```bash
python -m project.model.model_registry
python -m project.model.task1.pipeline --team LVA --periods 3 --n-mc 10
python -m project.model.task3.task1_impact --team LVA --expansion-city Toronto
```
