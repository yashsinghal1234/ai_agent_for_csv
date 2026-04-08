import json
import os
from typing import Dict

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_URL = os.getenv("OPENENV_BASE_URL", "http://127.0.0.1:7860")
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
USE_LLM = os.getenv("USE_LLM", "0") == "1"
SEED = int(os.getenv("SEED", "7"))


def log_event(tag: str, payload: Dict) -> None:
    print(f"[{tag}] {json.dumps(payload, separators=(',', ':'))}", flush=True)


def choose_action_from_issue(issue: dict) -> dict:
    if issue["type"] == "date_issue":
        return {"type": "fix_date", "row": issue["row"], "col": issue["col"]}

    if issue["type"] == "category_issue":
        return {"type": "standardize_category", "row": issue["row"], "col": issue["col"]}

    if issue["type"] == "outlier":
        return {"type": "remove_outlier", "row": issue["row"]}

    return {"type": "noop"}


def build_llm_client() -> OpenAI:
    if not API_BASE_URL or not HF_TOKEN:
        raise RuntimeError("API_BASE_URL and HF_TOKEN must be set when USE_LLM=1.")
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def llm_action(client: OpenAI, observation: dict) -> dict:
    prompt = (
        "You are a data cleaning agent.\n\n"
        f"Current issues:\n{observation['issues']}\n\n"
        "Choose ONE action in JSON format.\n"
        "Example: {\"type\": \"fix_date\", \"row\": 1, \"col\": \"date\"}\n\n"
        "Valid actions: fix_date, standardize_category, remove_outlier, noop.\n"
        "Only return JSON."
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        top_p=1,
    )

    content = (response.choices[0].message.content or "").strip()
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {"type": "noop"}

    try:
        return json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return {"type": "noop"}


def run_task(task_id: str, client: OpenAI = None) -> Dict:
    observation = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id, "seed": SEED}).json()
    episode_id = f"{task_id}-{SEED}"
    log_event("START", {"task_id": task_id, "episode_id": episode_id, "seed": SEED})

    total_reward = 0.0
    final_score = 0.0
    steps = 0

    for step in range(observation["max_steps"]):
        if observation["issues"]:
            if client:
                action = llm_action(client, observation)
            else:
                action = choose_action_from_issue(observation["issues"][0])
        else:
            action = {"type": "noop"}

        response = requests.post(f"{BASE_URL}/step", json=action).json()
        reward_value = response["reward"]
        total_reward += reward_value
        final_score = response["info"].get("score", final_score)
        steps = step + 1

        log_event(
            "STEP",
            {
                "task_id": task_id,
                "episode_id": episode_id,
                "step": step,
                "action": action,
                "reward": reward_value,
                "done": response["done"],
                "score": final_score,
            },
        )

        observation = response["observation"]
        if response["done"]:
            break

    log_event(
        "END",
        {
            "task_id": task_id,
            "episode_id": episode_id,
            "steps": steps,
            "total_reward": total_reward,
            "final_score": final_score,
        },
    )

    return {"task_id": task_id, "score": final_score, "total_reward": total_reward, "steps": steps}


def main() -> None:
    tasks = requests.get(f"{BASE_URL}/tasks").json()["tasks"]
    client = build_llm_client() if USE_LLM else None

    for task in tasks:
        run_task(task["task_id"], client=client)


if __name__ == "__main__":
    main()
