import yaml
from crewai import Task

def load_tasks(yaml_path: str, agents: dict, context: dict) -> dict:
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    tasks = {}
    for key, spec in cfg.items():
        agent_key = spec["agent"]
        agent = agents[agent_key]
        description = spec["description"]
        expected_output = spec.get("expected_output", "")

    prompt = (
        f"Context: {context}\n\n"
        f"Task: {description}\n\n"
        f"Expected Output JSON Schema: {expected_output}"
    )

    tasks[key] = Task(
        description = prompt,
        agent = agent,
        expected_output = expected_output,
    )
    return tasks

