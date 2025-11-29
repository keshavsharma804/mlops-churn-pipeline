import os
import yaml
import joblib


def load_config(path: str = "src/config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_model(model, output_dir: str, filename: str = "model.joblib") -> str:
    ensure_dir(output_dir)
    full_path = os.path.join(output_dir, filename)
    joblib.dump(model, full_path)
    return full_path
