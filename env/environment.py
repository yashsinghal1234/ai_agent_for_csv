import re
from typing import Optional

import numpy as np
import pandas as pd

from env.models import Action, Issue, Observation, Reward, StepResult
from env.tasks import DEFAULT_TASK_ID, TASKS, TaskSpec


class CSVEnvironment:
    def __init__(self, file_path: str, tasks: Optional[dict] = None) -> None:
        self.tasks = tasks or TASKS
        self.original = pd.read_csv(file_path)
        self.df = self.original.copy()
        self.task: TaskSpec = self.tasks[DEFAULT_TASK_ID]
        self.step_count = 0
        self.done = False
        self._baseline_issue_count = 0
        self.reset(task_id=self.task.task_id)

    def reset(self, task_id: Optional[str] = None, seed: Optional[int] = None) -> Observation:
        if task_id:
            if task_id not in self.tasks:
                raise ValueError(f"Unknown task_id '{task_id}'.")
            self.task = self.tasks[task_id]

        if seed is not None:
            np.random.seed(seed)

        self.df = self.original.copy()
        self.df.reset_index(drop=True, inplace=True)
        self.step_count = 0
        self.done = False
        self._baseline_issue_count = self._count_issues()
        return self._get_observation()

    def state(self) -> Observation:
        return self._get_observation()

    def step(self, action: Action) -> StepResult:
        if self.done:
            return StepResult(
                observation=self._get_observation(),
                reward=0.0,
                done=True,
                info={
                    "task_id": self.task.task_id,
                    "score": self._grade(),
                    "reason": "episode_done",
                    "reward_detail": Reward(value=0.0, components={"progress": 0.0, "penalty": 0.0}).dict(),
                },
            )

        before_score = self._grade()
        valid_action = self._apply_action(action)
        after_score = self._grade()

        progress = max(0.0, after_score - before_score)
        penalty = 0.1 if not valid_action else 0.0
        reward_value = max(0.0, min(1.0, progress - penalty))

        self.step_count += 1
        self.done = after_score >= 0.999 or self.step_count >= self.task.max_steps

        reward_detail = Reward(value=reward_value, components={"progress": progress, "penalty": penalty})
        info = {
            "task_id": self.task.task_id,
            "score": after_score,
            "progress": progress,
            "penalty": penalty,
            "invalid_action": not valid_action,
            "steps_remaining": max(0, self.task.max_steps - self.step_count),
            "reward_detail": reward_detail.dict(),
        }

        return StepResult(
            observation=self._get_observation(),
            reward=reward_value,
            done=self.done,
            info=info,
        )

    def _get_observation(self) -> Observation:
        issues = self.detect_issues()
        stats = {
            "row_count": int(len(self.df)),
            "issue_count": int(len(issues)),
            "columns": list(self.df.columns),
        }

        preview = self.df.head(self.task.preview_rows).to_dict(orient="list")
        return Observation(
            preview=preview,
            issues=issues,
            stats=stats,
            task_id=self.task.task_id,
            step=self.step_count,
            max_steps=self.task.max_steps,
        )

    def _count_issues(self) -> int:
        return len(self.detect_issues())

    def _grade(self) -> float:
        if self._baseline_issue_count <= 0:
            return 1.0

        remaining = self._count_issues()
        score = 1.0 - (remaining / float(self._baseline_issue_count))
        return float(max(0.0, min(1.0, score)))

    def detect_issues(self) -> list:
        issues = []
        issue_types = set(self.task.issue_types)

        numeric_means = {}
        if "outlier" in issue_types:
            for col in self.df.columns:
                if np.issubdtype(self.df[col].dtype, np.number):
                    numeric_means[col] = float(self.df[col].mean())

        for col in self.df.columns:
            col_lower = col.lower()
            values = list(self.df[col])

            for i, val in enumerate(values):
                if "date" in col_lower and "date_issue" in issue_types:
                    if self._is_date_issue(val):
                        issues.append(Issue(type="date_issue", row=i, col=col))

                if "category" in col_lower and "category_issue" in issue_types:
                    if isinstance(val, str) and val != val.strip().lower():
                        issues.append(Issue(type="category_issue", row=i, col=col))

                if "outlier" in issue_types and col in numeric_means:
                    mean_val = numeric_means[col]
                    threshold = mean_val * self.task.outlier_multiplier
                    try:
                        if float(val) > threshold:
                            issues.append(Issue(type="outlier", row=i, col=col))
                    except (TypeError, ValueError):
                        continue

        return issues

    def _is_date_issue(self, value: object) -> bool:
        normalized = self._normalize_date(value)
        if not normalized:
            return True
        return str(value) != normalized

    def _normalize_date(self, value: object) -> Optional[str]:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None

        text = str(value).strip()
        for dayfirst in (False, True):
            try:
                parsed = pd.to_datetime(text, errors="raise", dayfirst=dayfirst)
                return parsed.strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                continue

        match = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", text)
        if match:
            year, part1, part2 = match.groups()
            try:
                parsed = pd.to_datetime(f"{year}-{part2}-{part1}", errors="raise")
                return parsed.strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                return None

        return None

    def _apply_action(self, action: Action) -> bool:
        if action.type == "noop":
            return True

        if action.type in {"fix_date", "standardize_category"}:
            if action.row is None or action.col is None:
                return False
            if not self._row_col_valid(action.row, action.col):
                return False

        if action.type == "fix_date":
            value = self.df.at[action.row, action.col]
            normalized = self._normalize_date(value)
            if not normalized:
                return False
            self.df.at[action.row, action.col] = normalized
            return True

        if action.type == "standardize_category":
            value = self.df.at[action.row, action.col]
            if isinstance(value, str):
                self.df.at[action.row, action.col] = value.strip().lower()
                return True
            return False

        if action.type == "remove_outlier":
            if action.row is None:
                return False
            if not self._row_valid(action.row):
                return False
            self.df = self.df.drop(index=action.row).reset_index(drop=True)
            return True

        return False

    def _row_col_valid(self, row: int, col: str) -> bool:
        return self._row_valid(row) and col in self.df.columns

    def _row_valid(self, row: int) -> bool:
        return isinstance(row, int) and 0 <= row < len(self.df)