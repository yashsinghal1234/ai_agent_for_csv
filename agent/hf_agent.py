import os
import requests
import json

def get_hf_action(prompt, api_url, hf_token):
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {"inputs": prompt}
    response = requests.post(api_url, headers=headers, json=payload)
    try:
        result = response.json()
        # HF returns a list of dicts with 'generated_text' or similar
        if isinstance(result, list) and 'generated_text' in result[0]:
            return result[0]['generated_text']
        elif isinstance(result, dict) and 'generated_text' in result:
            return result['generated_text']
        elif isinstance(result, list) and 'text' in result[0]:
            return result[0]['text']
        return str(result)
    except Exception as e:
        return f"[ERROR] {e}"

def choose_action_from_issues(issues):
    if not issues:
        return {"type": "noop"}
    issue = issues[0]
    if issue["type"] == "date_issue":
        return {"type": "fix_date", "row": issue["row"], "col": issue["col"]}
    if issue["type"] == "category_issue":
        return {"type": "standardize_category", "row": issue["row"], "col": issue["col"]}
    if issue["type"] == "outlier":
        return {"type": "remove_outlier", "row": issue["row"]}
    return {"type": "noop"}

def main():
    BASE_URL = os.getenv("OPENENV_BASE_URL", "http://127.0.0.1:7860")
    HF_TOKEN = os.getenv("HF_TOKEN")
    MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-2-7b-chat-hf")
    API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
    SEED = int(os.getenv("SEED", "7"))

    tasks = requests.get(f"{BASE_URL}/tasks").json()["tasks"]
    for task in tasks:
        observation = requests.post(f"{BASE_URL}/reset", json={"task_id": task["task_id"], "seed": SEED}).json()
        print(f"[START] {{'task_id': '{task['task_id']}', 'seed': {SEED}}}")
        for step in range(observation["max_steps"]):
            issues = observation["issues"]
            prompt = f"You are a data cleaning agent. Current issues: {issues}. Choose ONE action in JSON format. Example: {{'type': 'fix_date', 'row': 1, 'col': 'date'}}. Valid actions: fix_date, standardize_category, remove_outlier, noop. Only return JSON."
            action_text = get_hf_action(prompt, API_URL, HF_TOKEN)
            try:
                action = json.loads(action_text)
            except Exception:
                action = choose_action_from_issues(issues)
            response = requests.post(f"{BASE_URL}/step", json=action).json()
            print(f"[STEP] {{'action': {action}, 'reward': {response['reward']}, 'done': {response['done']}}}")
            observation = response["observation"]
            if response["done"]:
                break
        print(f"[END] {{'task_id': '{task['task_id']}'}}")

if __name__ == "__main__":
    main()
