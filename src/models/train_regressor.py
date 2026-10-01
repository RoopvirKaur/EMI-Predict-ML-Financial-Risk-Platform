import time
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Tuple

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    TRAIN_DATASET_PATH,
    VAL_DATASET_PATH,
    TEST_DATASET_PATH,
    PREPROCESSOR_PATH,
    BEST_REGRESSOR_PATH,
    MODELS_DIR,
    REGRESSION_TARGET,
    RANDOM_SEED
)
from src.preprocessing.feature_engineering import FullPreprocessingPipeline, FinancialRatioTransformer
from src.models.mlflow_utils import log_model_run, save_best_local_model

# Register classes in sys.modules['__main__'] to prevent pickle attribute lookup errors
import sys
main_mod = sys.modules['__main__']
setattr(main_mod, 'FinancialRatioTransformer', FinancialRatioTransformer)
setattr(main_mod, 'FullPreprocessingPipeline', FullPreprocessingPipeline)

def plot_and_save_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path, title: str):
    """Plots and saves Actual vs Predicted scatter plot."""
    sample_idx = np.random.choice(len(y_true), size=min(3000, len(y_true)), replace=False)
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true[sample_idx], y_pred[sample_idx], alpha=0.3, color="#2980B9", edgecolors="none", s=20)
    
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([500, max_val], [500, max_val], "r--", label="Ideal Perfect Fit (y=x)")

    plt.title(f"Actual vs Predicted Max Safe EMI - {title}", fontsize=12, fontweight="bold")
    plt.xlabel("Actual Max Monthly EMI (INR)")
    plt.ylabel("Predicted Max Monthly EMI (INR)")
    plt.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_and_save_residuals(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path, title: str):
    """Plots and saves Residuals distribution plot."""
    residuals = y_true - y_pred
    sample_idx = np.random.choice(len(residuals), size=min(3000, len(residuals)), replace=False)
    
    plt.figure(figsize=(7, 6))
    plt.scatter(y_pred[sample_idx], residuals[sample_idx], alpha=0.3, color="#8E44AD", edgecolors="none", s=20)
    plt.axhline(0, color="r", linestyle="--")

    plt.title(f"Residuals Plot - {title}", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Max Monthly EMI (INR)")
    plt.ylabel("Residual Error (Actual - Predicted)")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_and_save_feature_importance(feature_names: list, importances: np.ndarray, save_path: Path, title: str):
    """Plots and saves top 15 feature importances for regressor."""
    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    fi_df = fi_df.sort_values("Importance", ascending=False).head(15)

    plt.figure(figsize=(9, 6))
    sns.barplot(data=fi_df, x="Importance", y="Feature", palette="magma")
    plt.title(f"Top 15 Regressor Feature Importances - {title}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def train_and_evaluate_regressors() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    print("\n==================================================================")
    print("Phase 5: Training Regression Subsystem (max_monthly_emi)")
    print("==================================================================")
    start_time = time.time()

    # 1. Load Parquet Data Splits & Preprocessor
    train_df = pd.read_parquet(TRAIN_DATASET_PATH)
    val_df = pd.read_parquet(VAL_DATASET_PATH)
    test_df = pd.read_parquet(TEST_DATASET_PATH)

    pipeline = FullPreprocessingPipeline.load(PREPROCESSOR_PATH)
    
    from src.config import NUMERICAL_RAW_FEATURES, CATEGORICAL_NOMINAL_FEATURES, CATEGORICAL_ORDINAL_FEATURES, DERIVED_RATIO_FEATURES
    input_cols = NUMERICAL_RAW_FEATURES + CATEGORICAL_NOMINAL_FEATURES + CATEGORICAL_ORDINAL_FEATURES + DERIVED_RATIO_FEATURES

    X_train = pipeline.transform(train_df[input_cols])
    X_val = pipeline.transform(val_df[input_cols])
    X_test = pipeline.transform(test_df[input_cols])

    feature_names = pipeline.get_feature_names_out()

    y_train = train_df[REGRESSION_TARGET].values.astype(np.float64)
    y_val = val_df[REGRESSION_TARGET].values.astype(np.float64)
    y_test = test_df[REGRESSION_TARGET].values.astype(np.float64)

    # 2. Define 3 Regression Models
    models = {
        "Linear_Regression": LinearRegression(n_jobs=-1),
        "Random_Forest_Regressor": RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),
        "XGBoost_Regressor": XGBRegressor(
            n_estimators=150,
            max_depth=8,
            learning_rate=0.1,
            tree_method="hist",
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
    }

    val_results = {}
    best_model_name = None
    best_rmse = float("inf")
    best_model_obj = None

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        t0 = time.time()
        
        model.fit(X_train, y_train)
        fit_dur = time.time() - t0

        # Predict Validation Set
        preds = model.predict(X_val)
        preds = np.clip(preds, 500, None)  # Ensure non-negative EMI >= 500

        rmse = root_mean_squared_error(y_val, preds)
        mae = mean_absolute_error(y_val, preds)
        r2 = r2_score(y_val, preds)

        metrics = {
            "val_rmse": float(rmse),
            "val_mae": float(mae),
            "val_r2_score": float(r2),
            "training_time_sec": float(fit_dur)
        }

        print(f"  RMSE:          INR {rmse:.2f}")
        print(f"  MAE:           INR {mae:.2f}")
        print(f"  R2 Score:      {r2:.4f}")
        print(f"  Training Time: {fit_dur:.2f}s")

        # Log to MLflow
        log_model_run(
            run_name=f"Regressor_{name}",
            params=model.get_params() if hasattr(model, "get_params") else {},
            metrics=metrics,
            model_obj=model
        )

        val_results[name] = {"model": model, "metrics": metrics, "preds": preds}

        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            best_model_obj = model

    print(f"\nSelected Best Regression Model: {best_model_name} (Val RMSE: INR {best_rmse:.2f})")

    # 3. Save Selected Best Model Locally
    save_best_local_model(best_model_obj, BEST_REGRESSOR_PATH)

    # Generate Feature Importance for Best Regressor if available
    fi_path = MODELS_DIR / "regression_feature_importance.png"
    if hasattr(best_model_obj, "feature_importances_"):
        plot_and_save_feature_importance(feature_names, best_model_obj.feature_importances_, fi_path, best_model_name)

    # 4. Final Evaluation ONCE on Test Set
    print("\n--- Final Test Set Evaluation for Selected Regressor ---")
    test_preds = best_model_obj.predict(X_test)
    test_preds = np.clip(test_preds, 500, None)

    test_rmse = root_mean_squared_error(y_test, test_preds)
    test_mae = mean_absolute_error(y_test, test_preds)
    test_r2 = r2_score(y_test, test_preds)

    act_vs_pred_path = MODELS_DIR / "regression_actual_vs_predicted.png"
    residuals_path = MODELS_DIR / "regression_residuals.png"

    plot_and_save_actual_vs_predicted(y_test, test_preds, act_vs_pred_path, f"{best_model_name} (Test Set)")
    plot_and_save_residuals(y_test, test_preds, residuals_path, f"{best_model_name} (Test Set)")

    print(f"Test Set Performance ({best_model_name}):")
    print(f"  - Test RMSE: INR {test_rmse:.2f}")
    print(f"  - Test MAE:  INR {test_mae:.2f}")
    print(f"  - Test R2:   {test_r2:.4f}")

    test_metrics = {
        "test_rmse": float(test_rmse),
        "test_mae": float(test_mae),
        "test_r2_score": float(test_r2),
        "total_regression_time_sec": float(time.time() - start_time)
    }

    # Register Best Model in MLflow
    log_model_run(
        run_name=f"Best_Regressor_{best_model_name}",
        params=best_model_obj.get_params() if hasattr(best_model_obj, "get_params") else {},
        metrics=test_metrics,
        model_obj=best_model_obj,
        artifacts={
            "actual_vs_predicted": str(act_vs_pred_path),
            "residuals_plot": str(residuals_path),
            "feature_importance": str(fi_path) if fi_path.exists() else ""
        },
        register_model_name="EMIPredict_Best_Regressor"
    )

    return val_results, test_metrics

if __name__ == "__main__":
    train_and_evaluate_regressors()
