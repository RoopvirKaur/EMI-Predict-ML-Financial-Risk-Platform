import mlflow
import mlflow.sklearn
import mlflow.xgboost
import os
import joblib
from pathlib import Path
from typing import Dict, Any, Optional

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, MLRUNS_DIR, MODELS_DIR

def setup_mlflow():
    """Configures local MLflow tracking server and experiment."""
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

def log_model_run(
    run_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    model_obj: Any,
    artifacts: Optional[Dict[str, str]] = None,
    register_model_name: Optional[str] = None
):
    """
    Logs parameters, metrics, model artifacts, and plot images to MLflow.
    """
    setup_mlflow()
    with mlflow.start_run(run_name=run_name):
        # Filter scalar params
        scalar_params = {}
        for k, v in params.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                scalar_params[k] = v
            else:
                scalar_params[k] = str(v)

        mlflow.log_params(scalar_params)
        mlflow.log_metrics(metrics)
        
        # Log artifacts (plot images, reports) if provided
        if artifacts:
            for art_name, art_path in artifacts.items():
                if art_path and os.path.exists(art_path):
                    mlflow.log_artifact(art_path, artifact_path="plots")

        # Log model artifact
        model_name_lower = str(type(model_obj)).lower()
        try:
            if "xgb" in model_name_lower:
                mlflow.xgboost.log_model(model_obj, artifact_path="model", registered_model_name=register_model_name)
            else:
                mlflow.sklearn.log_model(model_obj, artifact_path="model", registered_model_name=register_model_name)
        except Exception as e:
            # Fallback for file-based tracking store if model registry is restricted
            if "xgb" in model_name_lower:
                mlflow.xgboost.log_model(model_obj, artifact_path="model")
            else:
                mlflow.sklearn.log_model(model_obj, artifact_path="model")

        print(f"  [MLflow] Logged run '{run_name}'. Metrics: {metrics}")

def save_best_local_model(model_obj: Any, filepath: Path):
    """Saves best model object locally."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_obj, filepath)
    print(f"  Saved local model artifact to {filepath}")
