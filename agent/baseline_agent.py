import os
import requests

BASE_URL = os.getenv("OPENENV_BASE_URL", "http://127.0.0.1:7860")


def choose_action(issue: dict) -> dict:
    if issue["type"] == "date_issue":
        return {"type": "fix_date", "row": issue["row"], "col": issue["col"]}

    if issue["type"] == "category_issue":
        return {"type": "standardize_category", "row": issue["row"], "col": issue["col"]}

    if issue["type"] == "outlier":
        return {"type": "remove_outlier", "row": issue["row"]}

    return {"type": "noop"}


def run_task(task_id: str) -> dict:
    observation = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id, "seed": 7}).json()
    total_reward = 0.06
    final_score = 0.06

    for _ in range(observation["max_steps"]):
        issues = observation["issues"]
        action = choose_action(issues[0]) if issues else {"type": "noop"}

        response = requests.post(f"{BASE_URL}/step", json=action).json()
        reward_value = float(max(0.06, min(0.94, response["reward"])))
        total_reward += reward_value
        observation = response["observation"]
        final_score = float(max(0.06, min(0.94, response["info"].get("score", final_score))))

        if response["done"]:
            break

    return {"task_id": task_id, "score": float(max(0.06, min(0.94, final_score))), "total_reward": float(max(0.06, min(0.94, total_reward)))}


def run_agent() -> None:
    tasks = requests.get(f"{BASE_URL}/tasks").json()["tasks"]
    for task in tasks:
        result = run_task(task["task_id"])
        print(result)


if __name__ == "__main__":
    run_agent()