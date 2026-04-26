---
title: CSV Cleaning OpenEnv
emoji: "🧹"
colorFrom: blue
colorTo: green
sdk: docker
app_file: api/main.py
pinned: false
---
# CSV Cleaning OpenEnv

A real-world OpenEnv environment that simulates data cleaning on messy CSV files. Agents must fix dates, normalize categorical labels, and remove numeric outliers via the standard `reset()`, `step()`, and `state()` API.

## What this environment models

This project packages a small but realistic CSV-cleaning workflow. Each episode exposes a table with inconsistent dates, messy category labels, and occasional outliers. The agent improves the dataset by applying targeted cleaning actions until the task is solved or the step budget runs out.

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

## Docker

```bash
docker build -t csv-cleaning-openenv .
docker run -p 7860:7860 csv-cleaning-openenv
```

## HF Spaces

This repository is ready for a containerized Space. Ensure the required environment variables are configured in the Space settings.
