from typing import Optional

from fastapi import FastAPI, HTTPException

from env.environment import CSVEnvironment
from env.models import Action, Observation, ResetRequest, StepResult, TaskListResponse
from env.tasks import DEFAULT_TASK_ID, TASKS, list_tasks

app = FastAPI(title="CSV Cleaning OpenEnv")
env = CSVEnvironment("data/dirty.csv")


@app.get("/tasks", response_model=TaskListResponse)
def tasks() -> TaskListResponse:
    return TaskListResponse(tasks=list_tasks(), default_task_id=DEFAULT_TASK_ID)


@app.post("/reset", response_model=Observation)
def reset(request: ResetRequest) -> Observation:
    task_id = request.task_id or DEFAULT_TASK_ID
    if task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{task_id}'.")
    return env.reset(task_id=task_id, seed=request.seed)


@app.get("/reset", response_model=Observation)
def reset_get(task_id: Optional[str] = None, seed: Optional[int] = None) -> Observation:
    if task_id and task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{task_id}'.")
    return env.reset(task_id=task_id, seed=seed)


@app.post("/step", response_model=StepResult)
def step(action: Action) -> StepResult:
    return env.step(action)


@app.get("/state", response_model=Observation)
def state() -> Observation:
    return env.state()