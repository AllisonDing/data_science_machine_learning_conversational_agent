from __future__ import annotations
import os
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import optuna

def _try_free_cupy():
    try:
        import cupy
        cupy.get_default_memory_pool().free_bytes()
    except Exception:
        pass

def to_float32_fn(X):
    return X.astype(np.float32)

def load_data(data_path: str, target: str) -> Tuple[Any, str]:
    """Load CSV/Parquet to cuDF (if available) else pandas DataFrame."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at {data_path}")
    
    if data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    
    else:
        df = pd.read_csv(data_path)

    if target not in df.columns:
        raise ValueError(f"Target '{target}' not in columns: {list(df.columns)}")

    return df, target 

def basic_eda(df: Any, target: str) -> Dict[str, Any]:
    nulls = df.isnull().sum().to_dict()
    y_counts = df[target].value_counts().to_dict()
    notes = []
    if any(v > 0 for v in nulls.values()):
        notes.append("Missing values detected; will impute.")
    if len(y_counts) > 10:
        notes.append("Target appears to have many classes; consider stratification.")
    return {"null_report": nulls, "class_balance": y_counts, "notes": " ".join(notes)}

def build_feature_pipeline(df: Any, target: str) -> Tuple[Pipeline, np.ndarray, np.ndarray, list, list]:
    X = df.drop(columns = [target])
    y = df[target]

    num_cols = [c for c, t in zip(X.columns, X.dtypes) if t.kind in ("i", "f")]
    cat_cols = [c for c in X.columns if c not in num_cols]
    X_pd = X 
    y_np = y.values

    numeric_tf = Pipeline(steps = [
        ("imputer", SimpleImputer(strategy = "median")),
        ("scaler", StandardScaler()),
        ("to32", FunctionTransformer(to_float32_fn, accept_sparse=True)),
    ])

    categorical_tf = Pipeline(steps = [
        ("imputer", SimpleImputer(strategy = "most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown = "ignore", sparse_output = True, dtype = np.float32)),
    ])

    preprocessor = ColumnTransformer(
        transformers = [
            ("num", numeric_tf, num_cols),
            ("cat", categorical_tf, cat_cols),
        ]
    )

    pipeline = Pipeline(steps = [("pre", preprocessor)])

    return pipeline, X_pd, y_np, num_cols, cat_cols

def split_data(X_pd, y_np, test_size = 0.2, random_state = 42):
    X_train, X_test, y_train, y_test = train_test_split(X_pd, y_np, test_size = test_size, random_state = random_state, stratify = y_np)

    return X_train, X_test, y_train, y_test

def train_candidates(X_train, X_test, y_train, y_test, preprocessor: Pipeline) -> list:
    results = []

    to_float32 = FunctionTransformer(to_float32_fn, accept_sparse=True)

    def _evaluate(model, name: str):
        pipe = Pipeline([("pre", preprocessor), ("to32", to_float32), ("model", model)])
        
        import cupy
        cupy.get_default_memory_pool().free_bytes()
        
        pipe.fit(X_train, y_train)
        train_acc = float(pipe.score(X_train, y_train))
        preds = pipe.predict(X_test)
        acc = float(accuracy_score(y_test, preds))
        f1 = float(f1_score(y_test, preds, average = "weighted"))
        return {"name": name, "metrics": {"train_accuracy": train_acc, "test_accuracy": acc, "f1_weighted": f1}, "artifact": pipe}

    results.append(_evaluate(LogisticRegression(max_iter = 200), "(cuML) Logistic Regression"))
    results.append(_evaluate(RandomForestClassifier(n_estimators = 200, max_depth = 16), "(cuML) Random Forest"))
    results.append(_evaluate(SVC(), "(cuML) SVC"))

    results.sort(key = lambda x: (x["metrics"]["f1_weighted"], x["metrics"]["train_accuracy"], x["metrics"]["test_accuracy"]), reverse = True)
    return results 

def hpo_logistic(
    X_train, y_train, preprocessor: Pipeline,
    n_trials: int = 30, scoring: str = "f1_weighted", random_state: int = 42
):
    """
    Simple HPO for LogisticRegression.
    Search space: C, class_weight, max_iter (solver='lbfgs', penalty='l2').
    Returns a dict shaped like your train_candidates entries.
    """
    to32 = FunctionTransformer(to_float32_fn, accept_sparse=True)

    def objective(trial: optuna.Trial) -> float:
        C = trial.suggest_float("C", 1e-3, 1e2, log=True)
        class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
        max_iter = trial.suggest_int("max_iter", 200, 800, step=200)

        model = LogisticRegression(
            solver="lbfgs", penalty="l2",
            C=C, class_weight=class_weight,
            max_iter=max_iter, n_jobs=-1, random_state=random_state
        )

        pipe = Pipeline([("pre", preprocessor), ("to32", to32), ("model", model)])
        _try_free_cupy()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        scores = cross_val_score(pipe, X_train, y_train, scoring=scoring, cv=cv)
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=random_state),
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Build best pipeline and fit on full train
    best = study.best_trial.params
    best_model = LogisticRegression(
        solver="lbfgs", penalty="l2",
        C=best["C"], class_weight=best["class_weight"],
        max_iter=best["max_iter"], n_jobs=-1, random_state=random_state
    )
    best_pipe = Pipeline([("pre", preprocessor), ("to32", to32), ("model", best_model)])
    best_pipe.fit(X_train, y_train)

    return {
        "name": "Logistic Regression HPO",
        "metrics": {
            "train_accuracy": float(best_pipe.score(X_train, y_train)),
            "test_accuracy": None,                 
            "f1_weighted": float(study.best_value),    
        },
        "artifact": best_pipe,
        "best_params": best,
    }

def hpo_svc(
    X_train, y_train, preprocessor: Pipeline,
    n_trials: int = 30, scoring: str = "f1_weighted", random_state: int = 42
):
    """
    Simple HPO for SVC.
    Search space: kernel (linear or rbf), C, gamma (rbf only), class_weight.
    """
    to32 = FunctionTransformer(to_float32_fn, accept_sparse=True)

    def objective(trial: optuna.Trial) -> float:
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
        C = trial.suggest_float("C", 1e-3, 1e2, log=True)
        class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])

        svc_kwargs = dict(kernel=kernel, C=C, class_weight=class_weight, cache_size=512)
        if kernel == "rbf":
            svc_kwargs["gamma"] = trial.suggest_float("gamma", 1e-4, 1e0, log=True)

        model = SVC(**svc_kwargs)

        pipe = Pipeline([("pre", preprocessor), ("to32", to32), ("model", model)])
        _try_free_cupy()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        scores = cross_val_score(pipe, X_train, y_train, scoring=scoring, cv=cv)
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=random_state),
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial.params
    svc_kwargs = dict(kernel=best["kernel"], C=best["C"], class_weight=best["class_weight"], cache_size=512)
    if best["kernel"] == "rbf":
        svc_kwargs["gamma"] = best["gamma"]

    best_model = SVC(**svc_kwargs)
    best_pipe = Pipeline([("pre", preprocessor), ("to32", to32), ("model", best_model)])
    best_pipe.fit(X_train, y_train)

    return {
        "name": "(HPO) SVC (simple)",
        "metrics": {
            "train_accuracy": float(best_pipe.score(X_train, y_train)),
            "test_accuracy": None,                 
            "f1_weighted": float(study.best_value),   
        },
        "artifact": best_pipe,
        "best_params": best,
    }

def hpo_random_forest(
    X_train, y_train, preprocessor: Pipeline,
    n_trials: int = 30, scoring: str = "f1_weighted", random_state: int = 42
):
    """
    Simple HPO for RandomForestClassifier.
    Search space: n_estimators, max_depth, max_features.
    """
    to32 = FunctionTransformer(to_float32_fn, accept_sparse=True)

    def objective(trial: optuna.Trial) -> float:
        n_estimators = trial.suggest_int("n_estimators", 100, 500, step=100)
        max_depth = trial.suggest_int("max_depth", 4, 32)
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            n_jobs=-1,
            random_state=random_state,
        )

        pipe = Pipeline([("pre", preprocessor), ("to32", to32), ("model", model)])
        _try_free_cupy()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        scores = cross_val_score(pipe, X_train, y_train, scoring=scoring, cv=cv)
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=random_state),
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial.params
    best_model = RandomForestClassifier(
        n_estimators=best["n_estimators"],
        max_depth=best["max_depth"],
        max_features=best["max_features"],
        n_jobs=-1,
        random_state=random_state,
    )
    best_pipe = Pipeline([("pre", preprocessor), ("to32", to32), ("model", best_model)])
    best_pipe.fit(X_train, y_train)

    return {
        "name": "(HPO) Random Forest (simple)",
        "metrics": {
            "train_accuracy": float(best_pipe.score(X_train, y_train)),
            "test_accuracy": None,                 
            "f1_weighted": float(study.best_value), 
        },
        "artifact": best_pipe,
        "best_params": best,
    }