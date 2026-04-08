import json
import os
import sys
import time
from typing import Dict, Optional
import requests
from openai import OpenAI

# ── Environment variables (as required by hackathon) ──────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-api-base-url>")
MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model-name>")
HF_TOKEN = os.getenv("HF_TOKEN")

BASE_URL = os.environ.get("OPENENV_BASE_URL", "https://singhalyash-csv-cleaning-openenv.hf.space").rstrip("/")
SEED = int(os.environ.get("SEED", "7"))

# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_post(url: str, payload: dict, retries: int = 3) -> Optional[dict]:
    for attempt in range(retries):
        try:
            resp = requests.post(url, json=payload, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            print(f"[WARN] POST {url} returned {resp.status_code}: {resp.text}", flush=True)
        except Exception as e:
            print(f"[WARN] POST {url} attempt {attempt+1} failed: {e}", flush=True)
            time.sleep(2)
    return None


def safe_get(url: str, retries: int = 3) -> Optional[dict]:
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            print(f"[WARN] GET {url} returned {resp.status_code}: {resp.text}", flush=True)
        except Exception as e:
            print(f"[WARN] GET {url} attempt {attempt+1} failed: {e}", flush=True)
            time.sleep(2)
    return None

# ── Agent logic ───────────────────────────────────────────────────────────────

def choose_action_from_issue(issue: dict) -> dict:
    if issue["type"] == "date_issue":
        return {"type": "fix_date", "row": issue["row"], "col": issue.get("col")}
    if issue["type"] == "category_issue":
        return {"type": "standardize_category", "row": issue["row"], "col": issue.get("col")}
    if issue["type"] == "outlier":
        return {"type": "remove_outlier", "row": issue["row"]}
    return {"type": "noop"}


def build_llm_client() -> OpenAI:
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def llm_action(client: OpenAI, observation: dict) -> dict:
    try:
        prompt = (
            "You are a data cleaning agent.\n\n"
            f"Current issues:\n{observation['issues']}\n\n"
            "Choose ONE action in JSON format.\n"
            "Example: {\"type\": \"fix_date\", \"row\": 1, \"col\": \"date\"}\n\n"
            "Valid actions: fix_date, standardize_category, remove_outlier, noop.\n"
            "Only return JSON, nothing else."
        )
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            top_p=1,
            max_tokens=100,
        )
        content = (response.choices[0].message.content or "").strip()
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {"type": "noop"}
        return json.loads(content[start: end + 1])
    except Exception as e:
        print(f"[WARN] LLM call failed: {e}. Falling back to rule-based.", flush=True)
        if observation.get("issues"):
            return choose_action_from_issue(observation["issues"][0])
        return {"type": "noop"}

# ── Task runner ───────────────────────────────────────────────────────────────

def run_task(task_id: str, client: OpenAI = None) -> Dict:
    observation = safe_post(f"{BASE_URL}/reset", {"task_id": task_id, "seed": SEED})
    if observation is None:
        print(f"[ERROR] /reset failed for task {task_id}", flush=True)
        return {"task_id": task_id, "score": 0.01, "total_reward": 0.01, "steps": 0}

    episode_id = f"{task_id}-{SEED}"
    print(f"[START] task={task_id} env={task_id} model={MODEL_NAME}", flush=True)

    rewards = []
    final_score = 0.01
    steps = 0
    done = False

    for step in range(observation.get("max_steps", 10)):
        issues = observation.get("issues", [])
        if issues:
            action = llm_action(client, observation) if client else choose_action_from_issue(issues[0])
        else:
            action = {"type": "noop"}

        response = safe_post(f"{BASE_URL}/step", action)
        if response is None:
            print(f"[WARN] /step failed at step {step}, sending noop.", flush=True)
            response = safe_post(f"{BASE_URL}/step", {"type": "noop"})
            if response is None:
                break

        raw_reward = response.get("reward", 0.0)
        reward_value = min(0.99, max(0.01, round(float(raw_reward), 2)))
        raw_score = response.get("info", {}).get("score", final_score)
        final_score = min(0.99, max(0.01, round(float(raw_score), 2)))
        done = response.get("done", False)
        steps = step + 1
        rewards.append(reward_value)

        action_str = json.dumps(action)
        print(f"[STEP] step={steps} action={action_str} reward={reward_value:.2f} done={str(done).lower()} error=null", flush=True)

        observation = response.get("observation", observation)
        if done:
            break

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(done).lower()} steps={steps} score={final_score:.2f} rewards={rewards_str}", flush=True)

    return {"task_id": task_id, "score": final_score, "total_reward": sum(rewards), "steps": steps}

# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    result = safe_get(f"{BASE_URL}/tasks")
    if result is None:
        print("[ERROR] Could not fetch tasks from environment.", flush=True)
        sys.exit(1)

    tasks = result.get("tasks", [])
    client = build_llm_client()
    for task in tasks:
        run_task(task["task_id"], client=client)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] Unhandled exception: {e}", flush=True)
        sys.exit(1)