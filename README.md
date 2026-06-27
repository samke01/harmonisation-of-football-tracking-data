# Harmonisation of Football Tracking Data

This repository contains the code used to harmonise football tracking and event
data from DFL and SkillCorner and to evaluate whether the resulting common
dataset is useful for downstream machine-learning analyses.

The project has two main parts:

- `src/harmonization`: provider-specific parsers, coordinate and timestamp
  normalisation, kinematic processing, event normalisation, and export to a
  common data format.
- `src/evaluation`: source-shift diagnostics and downstream evaluation tasks
  used to assess how well the harmonised data supports cross-provider analysis.

The code is intended to accompany the paper and to make the preprocessing and
evaluation workflow reproducible for readers who have access to the required raw
data.

## Repository Layout

```text
.
|-- src/
|   |-- harmonization/      # DFL/SkillCorner ingestion, CDF export, kinematics
|   |-- evaluation/         # diagnostics, datasets, models, downstream tasks
|   `-- config.py           # settings loaded from .env
|-- scripts/                # command-line entrypoints for audits/evaluations
|-- tests/                  # focused regression tests
|-- pyproject.toml          # package metadata and dependencies
`-- uv.lock                 # locked dependency versions
```

## Setup

The project uses `uv` for dependency management.

```bash
uv sync
```

Configuration is loaded through `src/config.py` using `pydantic-settings`.
Create a local `.env` file and point `DATA_PATH` to the directory that contains
the raw provider data and where generated harmonised outputs should be written:

```bash
DATA_PATH=/path/to/your/dataset/
```

See `.env.example` for the expected variable name.

## Harmonisation Workflow

Run the tracking harmonisation pipeline:

```bash
uv run python -m src.harmonization.tracking
```

Run event normalisation:

```bash
uv run python -m src.harmonization.events
```

Validate kinematic fields and consistency checks:

```bash
uv run python scripts/validate_kinematics.py
```

The harmonisation code normalises provider-specific coordinates, timestamps,
periods, player and team metadata, play direction, visibility, ball status, and
event action labels into a shared representation. Tracking exports include CDF
JSONL records and nested match metadata.

## Evaluation Workflow

Run the direct source-shift diagnostics:

```bash
uv run python scripts/run_harmonization_evaluation.py
```

Run the final downstream evaluation suite:

```bash
uv run python scripts/run_final_evaluation.py --task all
```

Individual downstream task families can also be selected:

```bash
uv run python scripts/run_final_evaluation.py --task player-aggregate
uv run python scripts/run_final_evaluation.py --task kinematic-regression
uv run python scripts/run_final_evaluation.py --task ball-status
uv run python scripts/run_final_evaluation.py --task pass-success-event
uv run python scripts/run_final_evaluation.py --task pass-success-tracking
```

Run CDF audit checks:

```bash
uv run python scripts/run_cdf_audit.py
```

## Testing

Run the full test suite:

```bash
uv run pytest -q
```

The tests focus on the parts of the workflow where regressions are most likely
to affect the paper results: tracking export, event export, kinematic
processing, and train-only correction behaviour.

## Notes

Raw provider data, generated datasets, trained models, plots, and reports are
not stored in this repository. They are expected to live under `DATA_PATH` or in
local output directories ignored by Git.
