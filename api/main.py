from typing import Optional

from fastapi import FastAPI, HTTPException

from env.environment import CSVEnvironment
from env.models import Action, Observation, ResetRequest, StepResult, TaskListResponse

from fastapi.responses import HTMLResponse
from env.tasks import DEFAULT_TASK_ID, TASKS, list_tasks


app = FastAPI(title="CSV Cleaning OpenEnv")
env = CSVEnvironment("data/dirty.csv")


# Simple web UI at root
@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
    <head><title>CSV Cleaning OpenEnv</title></head>
    <body style='font-family:sans-serif;max-width:600px;margin:40px auto;'>
        <h1>🧹 CSV Cleaning OpenEnv</h1>
        <p>This Space provides an OpenEnv-compliant API for CSV cleaning tasks.</p>
        <ul>
            <li>Use <a href='/docs'>/docs</a> for interactive API documentation.</li>
            <li>Endpoints: <code>/reset</code>, <code>/step</code>, <code>/state</code>, <code>/tasks</code></li>
        </ul>
        <p>See the <a href='https://github.com/singhalyash/CSV_Cleaning_OpenEnv' target='_blank'>project README</a> for more details.</p>
    </body>
    </html>
    """


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