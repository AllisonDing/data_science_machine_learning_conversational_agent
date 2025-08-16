import yaml
from pathlib import Path
from crewai import Agent
from .llm import get_llm

def load_agents(yaml_agent: str) -> dict:
    yaml_path = Path(yaml_agent)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Agent YAML not found: {yaml_path}")
    
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    nvidia_llm = get_llm()

    agents = {}
    for key, spec in cfg.items():
        agents[key] = Agent(
            role = spec.get("role"),
            goal = spec.get("goal"),
            backstory = spec.get("backstory"),
            verbose = spec.get("verbose", True),
            allow_delegation = False,
            llm = nvidia_llm,
        )
    return agents 