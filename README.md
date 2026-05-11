# WNBA Franchise Optimization Model

This repo keeps the code for a WNBA franchise decision model. The main loop is
in Task 1. The other task folders add roster input, expansion shocks, ticket
pricing, and injury risk.

## Layout

```text
project/model/
  core/          shared model code
  simulation/    simulation and MPC routines
  task1/         main model entry point
  task2/         roster and initial-state pipeline
  task3/         expansion shock analysis
  task4/         ticket pricing extension
  task5/         injury and risk extension
  tests/         regression and formula tests
scripts/
  preprocessing/ data-cleaning scripts
sample_data/     small synthetic examples
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

## Data

Only small synthetic files are kept in `sample_data/`. For a full run, prepare a
separate data folder:

```text
data/
  raw/
  external/
  processed/
```

Point the preprocessing scripts to it with `WNBA_DATA_DIR`.

Windows:

```bash
set WNBA_DATA_DIR=C:\path\to\data
```

macOS/Linux:

```bash
export WNBA_DATA_DIR=/path/to/data
```

Keep raw contest files, licensed reports, personal notes, generated figures,
and large processed datasets out of Git.

## Examples

Create local smoke-test data first:

```bash
python scripts/prepare_sample_project_data.py
```

Show the model map:

```bash
python -m project.model.model_registry
```

Run the Task 1 model:

```bash
python -m project.model.task1.pipeline --team LVA --periods 3 --n-mc 10
```

Export a Task 3 expansion shock:

```bash
python -m project.model.task3.task1_impact --team LVA --expansion-city Toronto
```

Some commands need real WNBA data. The files in `sample_data/` are just schema
examples; `prepare_sample_project_data.py` creates a small local dataset for
checking that the model entry points run.

