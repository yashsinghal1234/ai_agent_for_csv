from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field


class Issue(BaseModel):
    type: Literal["date_issue", "category_issue", "outlier"]
    row: int
    col: Optional[str] = None
    detail: Optional[str] = None


class Observation(BaseModel):
    preview: Dict[str, List[Any]]
    issues: List[Issue]
    stats: Dict[str, Any]
    task_id: str
    step: int
    max_steps: int


class Action(BaseModel):
    type: Literal["fix_date", "standardize_category", "remove_outlier", "noop"]
    row: Optional[int] = None
    col: Optional[str] = None


class Reward(BaseModel):
    value: float = Field(gt=0.0, lt=1.0)
    components: Dict[str, float] = Field(default_factory=dict)


class StepResult(BaseModel):
    observation: Observation
    reward: float = Field(gt=0.0, lt=1.0)
    done: bool
    info: Dict[str, Any]


class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    seed: Optional[int] = None


class TaskListResponse(BaseModel):
    tasks: List[Dict[str, Any]]
    default_task_id: str
