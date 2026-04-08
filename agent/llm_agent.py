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


def _require_llm_config() -> OpenAI:
    if not API_BASE_URL or not HF_TOKEN:
        raise RuntimeError("API_BASE_URL and HF_TOKEN must be set for LLM mode.")
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def _extract_json(text: str) -> Dict:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {"type": "noop"}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {"type": "noop"}


def get_llm_action(state: dict) -> dict:
    client = _require_llm_config()
    prompt = (
        "You are a data cleaning agent.\n\n"
        f"Current issues:\n{state['issues']}\n\n"
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

    action_text = response.choices[0].message.content or ""
    return _extract_json(action_text)


def run_agent() -> None:
    observation = requests.post(f"{BASE_URL}/reset", json={"task_id": "csv_medium", "seed": 7}).json()

    for step in range(observation["max_steps"]):
        if not observation["issues"]:
            print("All issues fixed!")
            break

        action = get_llm_action(observation)
        print(f"Step {step}: {action}")

        response = requests.post(f"{BASE_URL}/step", json=action).json()
        reward_value = float(max(0.06, min(0.94, response["reward"])))
        print("Reward:", reward_value)
        observation = response["observation"]

        if response["done"]:
            break


if __name__ == "__main__":
    run_agent()