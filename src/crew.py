from crewai import Crew, Process 
from .agents import load_agents
from .tasks import load_tasks 
from .tools.gpu_tools import load_data, basic_eda, build_feature_pipeline, split_data, train_candidates 

def build_crew(agents_yaml: str, tasks_yaml: str, data_path: str, target: str) -> Crew:

    agents = load_agents(agents_yaml)
    context = {"data_path": data_path, "target": target}
    tasks = load_tasks(tasks_yaml, agents, context)

    crew = Crew(
        agents = list(agents.values()),
        tasks = list(tasks.values()),
        process = Process.sequential,
        verbose = True,
    )
    return crew

def run_pipeline(data_path, target):
    df, _ = load_data(data_path, target)
    eda = basic_eda(df, target)
    pipeline, X_pd, y_np, _, _ = build_feature_pipeline(df, target)
    X_train, X_test, y_train, y_test = split_data(X_pd, y_np)
    candidates = train_candidates(X_train, X_test, y_train, y_test, pipeline)
    best = candidates[0]

    md = [
        "# ML Run Report",
        f"**Data:** {data_path}",
        f"**Target:** {target}",
        "## EDA",
        f"Nulls: {eda['null_report']}",
        f"Class balance: {eda['class_balance']}",
        "## Candidates",
    ]
    for c in candidates:
        md.append(f"- {c['name']}: acc = {c['metrics']['accuracy']:.4f}, f1 = {c['metrics']['f1']:.4f}")
    md.append("\n## Champion")
    md.append(f"**{best['name']}** with f1 = {best['metrics']['f1']:.4f}, acc = {best['metrics']['accuracy']:.4f}")
    report_md = "\n".join(md)

    return {
        "eda": eda,
        "candidates": candidates,
        "champion": {"name": best["name"], "metrics": best["metrics"]},
        "report_md": report_md,
        "artifact": best["artifact"],
    }
