from typing import Dict, List, Literal

from pydantic import BaseModel


class TaskSpec(BaseModel):
    task_id: str
    name: str
    difficulty: Literal["easy", "medium", "hard"]
    description: str
    max_steps: int
    issue_types: List[str]
    outlier_multiplier: float
    preview_rows: int = 5


TASKS: Dict[str, TaskSpec] = {
    "csv_easy": TaskSpec(
        task_id="csv_easy",
        name="CSV Cleaning - Dates and Categories",
        difficulty="easy",
        description="Fix date formats and normalize category labels.",
        max_steps=10,
        issue_types=["date_issue", "category_issue"],
        outlier_multiplier=3.0,
    ),
    "csv_medium": TaskSpec(
        task_id="csv_medium",
        name="CSV Cleaning - Full Pass",
        difficulty="medium",
        description="Fix dates, normalize categories, and remove value outliers.",
        max_steps=12,
        issue_types=["date_issue", "category_issue", "outlier"],
        outlier_multiplier=2.5,
    ),
    "csv_hard": TaskSpec(
        task_id="csv_hard",
        name="CSV Cleaning - Tight Budget",
        difficulty="hard",
        description="Full cleaning with fewer allowed steps.",
        max_steps=8,
        issue_types=["date_issue", "category_issue", "outlier"],
        outlier_multiplier=2.0,
    ),
}

DEFAULT_TASK_ID = "csv_easy"


def list_tasks() -> List[dict]:
    return [task.dict() for task in TASKS.values()]
