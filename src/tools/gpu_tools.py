from __future__ import annotations
import os
import warnings
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

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
        preds = pipe.predict(X_test)
        acc = float(accuracy_score(y_test, preds))
        f1 = float(f1_score(y_test, preds, average = "weighted"))
        return {"name": name, "metrics": {"accuracy": acc, "f1": f1}, "artifact": pipe}

    results.append(_evaluate(LogisticRegression(max_iter = 200), "(cuML) Logistic Regression"))
    results.append(_evaluate(RandomForestClassifier(n_estimators = 200, max_depth = 16), "(cuML) Random Forest"))
    results.append(_evaluate(SVC(), "(cuML) SVC"))

    results.sort(key = lambda x: (x["metrics"]["f1"], x["metrics"]["accuracy"]), reverse = True)
    return results 


