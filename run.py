from pathlib import Path 
from src.crew import build_crew, run_pipeline

if __name__ == "__main__":

    DATA_PATH = "data/Titanic-Dataset-1.csv"
    TARGET = "Survived"

    agents_yaml = Path("config/agents.yaml").resolve()
    tasks_yaml = Path("config/tasks.yaml").resolve()

    crew = build_crew(str(agents_yaml), str(tasks_yaml), DATA_PATH, TARGET)
    crew.kickoff()

    result = run_pipeline(DATA_PATH, TARGET)

    out_path = Path("report.md")
    out_path.write_text(result["report_md"], encoding = "utf-8")
    print(f"\nSaved report to {out_path.resolve()}\n")

    import joblib
    joblib.dump(result["artifact"], "champion_model.joblib")
    print("Saved champion model to champion_model.joblib")

