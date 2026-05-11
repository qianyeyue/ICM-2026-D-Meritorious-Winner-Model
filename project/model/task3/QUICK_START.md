# Task 3 Quick Start

Run the expansion shock exporter:

```bash
python -m project.model.task3.task1_impact --team LVA --expansion-city Toronto
```

Feed the exported file into Task 1:

```bash
python -m project.model.task1.pipeline --task3-impact project/model/task3/outputs/task3_to_task1_LVA_Toronto.json
```

For code reading, start with `task1_impact.py`, then `lva_analysis.py`.
