from typing import Optional

from fastapi import FastAPI, HTTPException, Body

from env.environment import CSVEnvironment
from env.models import Action, Observation, ResetRequest, StepResult, TaskListResponse

from fastapi.responses import HTMLResponse
from fastapi import UploadFile, File, Request
import shutil
from env.tasks import DEFAULT_TASK_ID, TASKS, list_tasks



app = FastAPI(title="CSV Cleaning OpenEnv")
env = CSVEnvironment("data/dirty.csv")

# Helper to reload environment
def reload_env(csv_path: str = "data/dirty.csv"):
    global env
    env = CSVEnvironment(csv_path)


# Simple web UI at root with upload form
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
        <form action="/upload" method="post" enctype="multipart/form-data" style="margin-top:2em;">
            <label for="file">Upload your own CSV file:</label><br>
            <input type="file" id="file" name="file" accept=".csv">
            <button type="submit">Upload & Use</button>
        </form>
        <p>See the <a href='https://github.com/singhalyash/CSV_Cleaning_OpenEnv' target='_blank'>project README</a> for more details.</p>
    </body>
    </html>
    """


# CSV upload endpoint
@app.post("/upload", response_class=HTMLResponse)
async def upload_csv(request: Request, file: UploadFile = File(...)):
    # Save uploaded file as data/dirty.csv
    with open("data/dirty.csv", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    reload_env("data/dirty.csv")
    # Show confirmation page
    return """
    <html>
    <head><title>CSV Uploaded</title></head>
    <body style='font-family:sans-serif;max-width:600px;margin:40px auto;'>
        <h1>✅ CSV Uploaded!</h1>
        <p>Your CSV file has been uploaded and will be used for the environment.</p>
        <a href="/">Back to Home</a>
    </body>
    </html>
    """


@app.get("/tasks", response_model=TaskListResponse)
def tasks() -> TaskListResponse:
    return TaskListResponse(tasks=list_tasks(), default_task_id=DEFAULT_TASK_ID)



# Accept optional body for OpenEnv compliance
@app.post("/reset", response_model=Observation)
def reset(request: Optional[ResetRequest] = Body(None)) -> Observation:
    task_id = request.task_id if request and request.task_id else DEFAULT_TASK_ID
    seed = request.seed if request else None
    if task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{task_id}'.")
    return env.reset(task_id=task_id, seed=seed)


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