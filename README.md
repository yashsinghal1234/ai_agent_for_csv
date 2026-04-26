---
# Hugging Face Spaces metadata for deployment and card display.
title: CSV Cleaning OpenEnv
emoji: "🧹"
# Colors used by the Space card.
colorFrom: blue
colorTo: green
# Docker image entrypoint for the app.
sdk: docker
app_file: api/main.py
pinned: false
---
# CSV Cleaning OpenEnv

A real-world OpenEnv environment that simulates data cleaning on messy CSV files. Agents must fix dates, normalize categorical labels, and remove numeric outliers via the standard `reset()`, `step()`, and `state()` API.

<p align="center">
  <img src="https://img.shields.io/badge/OpenEnv-ready-0ea5e9?style=for-the-badge" alt="OpenEnv ready badge" />
  <img src="https://img.shields.io/badge/Docker-supported-2496ed?style=for-the-badge" alt="Docker supported badge" />
  <img src="https://img.shields.io/badge/CSV%20Cleaning-automation-16a34a?style=for-the-badge" alt="CSV cleaning badge" />
</p>

## What this environment models

This project packages a small but realistic CSV-cleaning workflow. Each episode exposes a table with inconsistent dates, messy category labels, and occasional outliers. The agent improves the dataset by applying targeted cleaning actions until the task is solved or the step budget runs out.

## Highlights

- deterministic task scoring for repeatable experiments
- clean OpenEnv-style `reset`, `step`, and `state` interactions
- baseline policy and inference entry point included
- Docker-ready deployment for Hugging Face Spaces or local runs

The repository includes:

- a FastAPI/OpenEnv service in `api/`
- a baseline agent in `agent/`
- an inference entry point in `inference.py`
- task metadata in `openenv.yaml`
- sample data and helper assets under `data/` and `CSV_Cleaning_OpenEnv/`

## Contributors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/yashsinghal1234">
        <img src="https://github.com/yashsinghal1234.png?size=160" width="120" height="120" alt="yashsinghal1234" />
      </a>
      <br />
      <strong>yashsinghal1234</strong>
      <br />
      Owner
    </td>
    <td align="center">
      <a href="https://github.com/Amit1557">
        <img src="https://github.com/Amit1557.png?size=160" width="120" height="120" alt="Amit1557" />
      </a>
      <br />
      <strong>Amit1557</strong>
      <br />
      Collaborator
    </td>
  </tr>
</table>


## OpenEnv compliance
---

- Typed Pydantic models for `Observation`, `Action`, and `Reward`.
- `reset`, `step`, and `state` endpoints.
- `openenv.yaml` metadata and task definitions.

## Tasks

| --- | --- | --- | --- |
| `csv_easy` | easy | Fix date formats and normalize categories | 10 |
| `csv_medium` | medium | Fix dates, normalize categories, remove outliers | 12 |
| `csv_hard` | hard | Full cleaning with fewer allowed steps | 8 |

Each task has a deterministic grader that scores progress in the range 0.0–1.0 based on issues resolved.

## Action space

```json
{
  "type": "fix_date" | "standardize_category" | "remove_outlier" | "noop",
  "row": 0,
  "col": "date"
}
```

## Observation space

```json
{
  "preview": {"col": ["values", "..."]},
  "issues": [{"type": "date_issue", "row": 0, "col": "date"}],
  "stats": {"row_count": 5, "issue_count": 9, "columns": ["id", "date", "category", "value"]},
  "task_id": "csv_medium",
  "step": 0,
  "max_steps": 12
}
```

## Setup

```bash
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 7860
```

## Quick start

1. Install dependencies with `pip install -r requirements.txt`.
2. Launch the API with `uvicorn api.main:app --host 0.0.0.0 --port 7860`.
3. Run `python agent/baseline_agent.py` to see the baseline cleaner in action.
4. Run `python inference.py` to try the scripted inference loop.

## Repository layout

- `api/`: OpenEnv-compatible API server
- `agent/`: baseline policy and helper logic
- `inference.py`: script for running the inference loop
- `data/`: local datasets and fixtures
- `CSV_Cleaning_OpenEnv/`: task assets used by the environment

### Environment variables

Required for inference or LLM agents:

- `API_BASE_URL` (OpenAI-compatible base URL)
- `MODEL_NAME` (model identifier)
- `HF_TOKEN` (token for the API)

Optional:

- `OPENENV_BASE_URL` (defaults to `http://127.0.0.1:7860`)
- `USE_LLM` (`1` to enable LLM policy in `inference.py`)
- `SEED` (integer seed, default `7`)

## Baseline agent

```bash
python agent/baseline_agent.py
```

Expected baseline scores (approx):

- `csv_easy`: ~1.0
- `csv_medium`: ~1.0
- `csv_hard`: ~0.88

## Inference script

```bash
python inference.py
```

This emits structured logs with `[START]`, `[STEP]`, and `[END]` tokens. Set `USE_LLM=1` to use the LLM policy.

## What makes it useful

The project is designed as a small sandbox for agent evaluation and CSV cleanup workflows. It can be extended with new tasks, custom reward logic, additional cleaning actions, or alternative agent policies without changing the core API shape.

Possible next additions:

- a web UI for previewing dataset fixes
- richer validation and scoring rules
- more task variants with different difficulty levels
- downloadable cleaned CSV outputs

## Docker

```bash
docker build -t csv-cleaning-openenv .
docker run -p 7860:7860 csv-cleaning-openenv
```

## HF Spaces

This repository is ready for a containerized Space. Ensure the required environment variables are configured in the Space settings.
