# src/chat_agent.py
from __future__ import annotations

import json, re, sys, types
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from pathlib import Path

import pandas as pd
from joblib import dump

# Your modules
from src.tools import gpu_tools
from src import llm as llm_module

# -----------------------
# Helpers
# -----------------------
def _infer_target_from_df(df: pd.DataFrame) -> Optional[str]:
    for cand in ["Survived", "target", "label", "y", "class", "Outcome"]:
        if cand in df.columns:
            return cand
    for c in df.columns:
        try:
            u = df[c].dropna().nunique()
            if u in (2, 3):
                return c
        except Exception:
            continue
    return None

def _intent_router(user_message: str):
    """
    Lightweight fallback when the LLM doesn't emit tool_calls.
    Returns (tool_name, args) or (None, None).
    """
    m = user_message.lower().strip()

    if "set" in m and "dataset" in m:
        tgt = re.search(r"target\s*=?\s*([A-Za-z0-9_]+)", user_message)
        path = re.search(r"(?:to|=)\s*[`'\"]?([^\s`'\"\n]+)", user_message)

        if path:
            args = {"path": path.group(1).strip()}
            if tgt:
                args["target"] = tgt.group(1)
            return "set_dataset", args
        return None, None

    if "describe" in m or "schema" in m:
        return "describe", {}

    if "preview" in m or "head" in m or "show top" in m:
        n = 5
        mnum = re.search(r"(?:preview|head|top)\s+(\d+)", m)
        if mnum:
            try:
                n = max(1, min(50, int(mnum.group(1))))
            except Exception:
                n = 5
        return "preview", {"n": n}

    if "train" in m:
        # We keep one training tool; gpu_tools.train_candidates runs RF/LogReg/SVC internally.
        tgt = re.search(r"target\s*=?\s*([A-Za-z0-9_]+)", user_message)
        args = {}
        if tgt:
            args["target"] = tgt.group(1)
        return "train_classification", args
    
    # --- HPO routing ---
    if "hpo" in m or "tune" in m or "optuna" in m:
        # defaults
        args = {"n_trials": 30, "scoring": "f1_weighted"}
        # n_trials parser (e.g., "hpo logistic 50" or "n_trials=40")
        m_trials = re.search(r"(?:n[_\- ]?trials\s*=?\s*|hpo\s+\w+\s+)(\d+)", user_message, re.I)
        if m_trials:
            try:
                args["n_trials"] = max(1, min(200, int(m_trials.group(1))))
            except Exception:
                pass
        # scoring parser (e.g., "scoring=accuracy")
        m_scoring = re.search(r"scoring\s*=\s*([A-Za-z0-9_]+)", user_message)
        if m_scoring:
            args["scoring"] = m_scoring.group(1)

        if "logistic" in m or "logreg" in m or "lr" in m:
            return "hpo_logistic", args
        if "svc" in m or "svm" in m:
            return "hpo_svc", args
        if "rf" in m or "random forest" in m:
            return "hpo_random_forest", args

    return None, None

def _ensure_cupy_stub():
    """If cupy is missing, stub just enough to satisfy gpu_tools.train_candidates()."""
    if "cupy" in sys.modules:
        return
    try:
        import cupy  # noqa: F401
    except Exception:
        fake = types.SimpleNamespace(
            get_default_memory_pool=lambda: types.SimpleNamespace(free_bytes=lambda: None)
        )
        sys.modules["cupy"] = fake

def _get_llm_client():
    # must return an object exposing .chat(messages=..., tools=...)
    return llm_module.create_client()

def _clean_response(raw_output: str) -> str:
    """Strip leaked <think>...</think> and ensure valid JSON parsing when possible."""
    # Remove leaked reasoning blocks
    cleaned = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "answer" in parsed:
            return parsed["answer"]
    except Exception:
        pass
    return cleaned

# -----------------------
# Agent
# -----------------------
@dataclass
class ChatAgent:
    """
    Conversational wrapper exposing a handful of tools that internally call gpu_tools.
    """
    memory: List[dict] = field(default_factory=list)

    # runtime state
    df: Optional[pd.DataFrame] = None
    target: Optional[str] = None
    champion_path: Path = Path("champion_model.joblib")

    def __post_init__(self):
        _ensure_cupy_stub()
        self.client = _get_llm_client()

        # Map canonical tool names -> wrapper methods
        self.tools_map: Dict[str, Callable[..., str]] = {
            "set_dataset": self._tool_set_dataset,
            "describe": self._tool_describe,
            "preview": self._tool_preview,
            "train_classification": self._tool_train_classification,
            "hpo_logistic": self._tool_hpo_logistic,
            "hpo_svc": self._tool_hpo_svc,
            "hpo_random_forest": self._tool_hpo_random_forest,
        }

        # Keep the prompt simple and tool-first; no chain-of-thought
        self.system_prompt = (
            "You are a machine-learning assistant. "
            "Always call a tool when the user asks to operate on data (set dataset, describe, preview, train). "
            "Return concise results only.\n\n"
            "Tools:\n"
            "- set_dataset(path, target?)\n"
            "- describe()\n"
            "- preview(n)\n"
            "- train_classification(target?)\n"
            "- hpo_logistic(n_trials?, scoring?)\n"
            "- hpo_svc(n_trials?, scoring?)\n"
            "- hpo_random_forest(n_trials?, scoring?)\n"
            "Ask a short follow-up only if a required argument is missing."
        )

    # ---- Tool specs (what the LLM sees) ----
    def _tool_specs(self) -> List[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "set_dataset",
                    "description": "Load CSV/Parquet and set target (optional if inferable).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "target": {"type": "string"},
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "describe",
                    "description": "Rows/cols/dtypes/target + EDA (nulls, class balance).",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "preview",
                    "description": "Show top n rows (1-50).",
                    "parameters": {
                        "type": "object",
                        "properties": {"n": {"type": "integer", "default": 5, "minimum": 1, "maximum": 50}},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "train_classification",
                    "description": "Build features, split, train candidates (LogReg/RF/SVC). Save best pipeline.",
                    "parameters": {
                        "type": "object",
                        "properties": {"target": {"type": "string"}},
                    },
                },
            },
            # append these dicts to the returned list
            {
                "type": "function",
                "function": {
                    "name": "hpo_logistic",
                    "description": "Hyperparameter optimization for Logistic Regression via Optuna.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "n_trials": {"type": "integer", "default": 30, "minimum": 1, "maximum": 200},
                            "scoring":  {"type": "string",  "default": "f1_weighted"}
                        }
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "hpo_svc",
                    "description": "Hyperparameter optimization for SVC via Optuna.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "n_trials": {"type": "integer", "default": 30, "minimum": 1, "maximum": 200},
                            "scoring":  {"type": "string",  "default": "f1_weighted"}
                        }
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "hpo_random_forest",
                    "description": "Hyperparameter optimization for RandomForest via Optuna.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "n_trials": {"type": "integer", "default": 30, "minimum": 1, "maximum": 200},
                            "scoring":  {"type": "string",  "default": "f1_weighted"}
                        }
                    },
                },
            },
        ]

    # ---- Tool implementations (wrappers over gpu_tools) ----
    def _tool_set_dataset(self, path: str, target: Optional[str] = None) -> str:
        p = Path(path)
        if not p.exists():
            return f"File not found: {path}"

        if target is None:
            # Try to infer from a peek
            try:
                peek = pd.read_parquet(p) if p.suffix.lower() in (".parquet", ".pq") else pd.read_csv(p, nrows=2000)
                target = _infer_target_from_df(peek)
            except Exception:
                target = None

        if not target:
            return ("Target column is unknown. Specify it like: "
                    "`set the dataset to <path> target=Survived`.")

        df, t = gpu_tools.load_data(data_path=str(p), target=target)
        self.df, self.target = df, t

        return json.dumps({
            "loaded": p.name,
            "shape": [int(df.shape[0]), int(df.shape[1])],
            "target": t
        }, indent=2)

    def _tool_describe(self) -> str:
        if self.df is None:
            return "No dataset loaded. Use set_dataset(path, target?) first."
        eda = gpu_tools.basic_eda(self.df, self.target)
        info = {
            "rows": int(self.df.shape[0]),
            "cols": int(self.df.shape[1]),
            "columns": {c: str(self.df[c].dtype) for c in self.df.columns},
            "target": self.target,
            "eda": eda,
        }
        return json.dumps(info, indent=2)

    def _tool_preview(self, n: int = 5) -> str:
        if self.df is None:
            return "No dataset loaded."
        # avoid dependency on tabulate; fallback to string if markdown not available
        try:
            return self.df.head(n).to_markdown(index=False)
        except Exception:
            return self.df.head(n).to_string(index=False)

    def _tool_train_classification(self, target: Optional[str] = None) -> str:
        if self.df is None:
            return "No dataset loaded. Use set_dataset(path, target?) first."
        if target and target != self.target:
            if target not in self.df.columns:
                return f"Target '{target}' not in columns."
            self.target = target

        pre, X_pd, y_np, *_ = gpu_tools.build_feature_pipeline(self.df, self.target)
        X_tr, X_te, y_tr, y_te = gpu_tools.split_data(X_pd, y_np)
        results = gpu_tools.train_candidates(X_tr, X_te, y_tr, y_te, pre)

        # Save best pipeline (index 0 after sorting inside train_candidates)
        best_path = "(no artifact)"
        if results and results[0].get("artifact") is not None:
            try:
                dump(results[0]["artifact"], self.champion_path)
                best_path = str(self.champion_path)
            except Exception as e:
                best_path = f"(save failed: {e})"

        # Return concise JSON summary
        payload = {
            "target": self.target,
            "results": [
                {"model": r["name"], "train_accuracy": r["metrics"]["train_accuracy"], "test_accuracy": r["metrics"]["test_accuracy"], "f1_weighted": r["metrics"]["f1_weighted"]}
                for r in results
            ],
            "saved_champion_to": best_path,
        }
        return json.dumps(payload, indent=2)
    
    def _eval_on_test(self, pipe, X_te, y_te):
        from sklearn.metrics import accuracy_score, f1_score
        preds = pipe.predict(X_te)
        return float(accuracy_score(y_te, preds)), float(f1_score(y_te, preds, average="weighted"))

    def _run_hpo_and_summarize(self, hpo_fn, n_trials: int, scoring: str) -> str:
        if self.df is None:
            return "No dataset loaded. Use set_dataset(path, target?) first."
        pre, X_pd, y_np, *_ = gpu_tools.build_feature_pipeline(self.df, self.target)
        X_tr, X_te, y_tr, y_te = gpu_tools.split_data(X_pd, y_np)

        res = hpo_fn(X_tr, y_tr, preprocessor=pre, n_trials=n_trials, scoring=scoring, random_state=42)
        test_acc, test_f1 = self._eval_on_test(res["artifact"], X_te, y_te)
        res["metrics"]["test_accuracy"] = test_acc
        res["metrics"]["f1_weighted"] = test_f1

        # Optionally save the champion
        best_path = "(no artifact)"
        try:
            dump(res["artifact"], self.champion_path)
            best_path = str(self.champion_path)
        except Exception as e:
            best_path = f"(save failed: {e})"

        payload = {
            "target": self.target,
            "results": [{
                "model": res["name"],
                "train_accuracy": res["metrics"]["train_accuracy"],
                "test_accuracy": res["metrics"]["test_accuracy"],
                "f1_weighted": res["metrics"]["f1_weighted"],
                "best_params": res.get("best_params", {})
            }],
            "saved_champion_to": best_path,
        }
        return json.dumps(payload, indent=2)

    def _tool_hpo_logistic(self, n_trials: int = 30, scoring: str = "f1_weighted") -> str:
        return self._run_hpo_and_summarize(gpu_tools.hpo_logistic, n_trials, scoring)

    def _tool_hpo_svc(self, n_trials: int = 30, scoring: str = "f1_weighted") -> str:
        return self._run_hpo_and_summarize(gpu_tools.hpo_svc, n_trials, scoring)

    def _tool_hpo_random_forest(self, n_trials: int = 30, scoring: str = "f1_weighted") -> str:
        return self._run_hpo_and_summarize(gpu_tools.hpo_random_forest, n_trials, scoring)


    # ---- Plumbing ----
    def _client_chat(self, messages, tools=None):
        return self.client.chat(messages=messages, tools=tools)

    def _call_tool(self, fn_name: str, args: Dict[str, Any]) -> str:
        fn = self.tools_map.get(fn_name)
        if not fn:
            return f"ERROR: Unknown tool '{fn_name}'."
        try:
            return fn(**(args or {}))
        except Exception as e:
            return f"ERROR in {fn_name}: {e}"

    def chat(self, user_message: str) -> str:
        if not self.memory:
            self.memory.append({"role": "system", "content": self.system_prompt})
        self.memory.append({"role": "user", "content": user_message})

        tool_specs = self._tool_specs()
        first = self._client_chat(self.memory, tools=tool_specs)

        # If the model returns a tool call, run it; else use fallback router
        message = first["choices"][0].get("message") if isinstance(first, dict) else first.choices[0].message
        tool_calls = (getattr(message, "tool_calls", None) or message.get("tool_calls") or [])

        if tool_calls:
            call = tool_calls[0]
            fn_name = call["function"]["name"]
            try:
                args = json.loads(call["function"].get("arguments", "{}") or "{}")
            except Exception:
                args = {}
            tool_result = self._call_tool(fn_name, args)
            # Feed tool result back for a natural-language wrap-up (optional)
            self.memory.append({"role": "assistant", "tool_calls": [call], "content": None})
            self.memory.append({"role": "tool", "tool_call_id": call.get("id", "tool_0"), "name": fn_name, "content": tool_result})
            second = self._client_chat(self.memory, tools=tool_specs)
            # NL summary from the model:
            final_msg = second["choices"][0]["message"]["content"] if isinstance(second, dict) else second.choices[0].message.content
            return _clean_response(final_msg or tool_result)

        # Fallback: route ourselves
        fn_name, args = _intent_router(user_message)
        if fn_name:
            return self._call_tool(fn_name, args or {}) or "Done."

        # Otherwise return the model text
        return (getattr(message, "content", None) or message.get("content", "") or
                "I can do that. Try: set the dataset path first (optionally specify target=...).")
