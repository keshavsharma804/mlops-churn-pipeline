import os
import sys
from datetime import datetime, timezone

import json
import mlflow
import glob
import re
import shap
import joblib

import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# allow running as script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils.io import load_config, save_model  # noqa: E402


# -------------------------------------------------------------------------
# Load dataset
# -------------------------------------------------------------------------
def load_data(config: dict):
    df = pd.read_csv(config["data"]["raw_path"])
    target = config["data"]["target_column"]

    X = df.drop(columns=[target])
    y = df[target]
    return X, y


# -------------------------------------------------------------------------
# SHAP saving logic (fixed)
# -------------------------------------------------------------------------
def save_shap_explainer(model, X_sample):
    os.makedirs("models", exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)  # compute once

    joblib.dump(
        {"explainer": explainer, "sample": X_sample, "shap_values": shap_values},
        "models/shap_explainer.joblib"
    )

    print("✅ SHAP explainer saved to models/shap_explainer.joblib")


def load_shap_explainer():
    explainer = joblib.load("models/shap_explainer.joblib")
    print("SHAP explainer loaded instantly")
    return explainer


# -------------------------------------------------------------------------
# Preprocessing + model pipeline
# -------------------------------------------------------------------------
def build_pipeline(X, config):
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])

    model = RandomForestClassifier(**config["model"]["params"])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline


# -------------------------------------------------------------------------
# Save Training Stats
# -------------------------------------------------------------------------
def save_training_stats(X, config):
    from pathlib import Path
    stats_dir = Path("monitoring")
    stats_dir.mkdir(parents=True, exist_ok=True)

    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    stats = {"numeric": {}, "categorical": {}}

    for col in numeric_cols:
        stats["numeric"][col] = {
            "mean": float(X[col].mean()),
            "std": float(X[col].std())
        }

    for col in categorical_cols:
        value_counts = X[col].value_counts(normalize=True).to_dict()
        stats["categorical"][col] = {
            str(k): float(v) for k, v in value_counts.items()
        }

    stats_path = stats_dir / "training_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"📊 Saved training stats to {stats_path}")


# -------------------------------------------------------------------------
# Main training script
# -------------------------------------------------------------------------
def main():
    config = load_config()

    mlflow.set_experiment(config.get("experiment_name", "churn_mlops_pipeline"))

    X, y = load_data(config)
    save_training_stats(X, config)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"],
        stratify=y
    )

    pipeline = build_pipeline(X, config)

    with mlflow.start_run(run_name=f"train_{datetime.now(timezone.utc).isoformat()}"):

        mlflow.log_params(config["model"]["params"])

        # train
        pipeline.fit(X_train, y_train)

        # -----------------------------------------------------------------
        # SHAP S AVING (Corrected)
        # -----------------------------------------------------------------
        tree_model = pipeline.named_steps["model"]
        save_shap_explainer(tree_model, X_train.sample(200))
        mlflow.log_artifact("models/shap_explainer.joblib")

        # -----------------------------------------------------------------
        # Metrics
        # -----------------------------------------------------------------
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        print("Accuracy:", acc)
        print("F1 Score:", f1)

        # -----------------------------------------------------------------
        # Save Model (Versioned)
        # -----------------------------------------------------------------
        model_dir = config["paths"]["model_output_dir"]
        os.makedirs(model_dir, exist_ok=True)

        existing = glob.glob(os.path.join(model_dir, "churn_pipeline_v*.joblib"))
        versions = [int(re.findall(r"v(\d+)", p)[0]) for p in existing] if existing else []
        new_version = max(versions) + 1 if versions else 1

        model_filename = f"churn_pipeline_v{new_version}.joblib"
        model_path = save_model(pipeline, model_dir, model_filename)

        mlflow.log_artifact(model_path)
        print(f"🚀 Saved model version: {model_filename}")
        print(f"📦 Saved pipeline at: {model_path}")


if __name__ == "__main__":
    main()
