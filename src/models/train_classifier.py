import time
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    TRAIN_DATASET_PATH,
    VAL_DATASET_PATH,
    TEST_DATASET_PATH,
    PREPROCESSOR_PATH,
    BEST_CLASSIFIER_PATH,
    MODELS_DIR,
    CLASSIFICATION_TARGET,
    ELIGIBILITY_CLASSES,
    RANDOM_SEED
)
from src.preprocessing.feature_engineering import FullPreprocessingPipeline, FinancialRatioTransformer
from src.models.mlflow_utils import log_model_run, save_best_local_model

# Register classes in sys.modules['__main__'] to prevent pickle attribute lookup errors
import sys
main_mod = sys.modules['__main__']
setattr(main_mod, 'FinancialRatioTransformer', FinancialRatioTransformer)
setattr(main_mod, 'FullPreprocessingPipeline', FullPreprocessingPipeline)

def plot_and_save_confusion_matrix(cm: np.ndarray, classes: list, save_path: Path, title: str):
    """Plots and saves confusion matrix heatmap."""
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix - {title}", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_and_save_feature_importance(feature_names: list, importances: np.ndarray, save_path: Path, title: str):
    """Plots and saves top 15 feature importances."""
    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    fi_df = fi_df.sort_values("Importance", ascending=False).head(15)

    plt.figure(figsize=(9, 6))
    sns.barplot(data=fi_df, x="Importance", y="Feature", palette="viridis")
    plt.title(f"Top 15 Feature Importances - {title}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def train_and_evaluate_classifiers() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    print("\n==================================================================")
    print("Phase 5: Training Classification Subsystem (emi_eligibility)")
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

    # Encode Targets
    label_encoder = LabelEncoder()
    label_encoder.fit(ELIGIBILITY_CLASSES)
    
    y_train = label_encoder.transform(train_df[CLASSIFICATION_TARGET])
    y_val = label_encoder.transform(val_df[CLASSIFICATION_TARGET])
    y_test = label_encoder.transform(test_df[CLASSIFICATION_TARGET])

    high_risk_idx = list(label_encoder.classes_).index("High_Risk")

    # Save Label Encoder
    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.pkl")

    # Calculate sample weights for XGBoost
    xgb_sample_weights = compute_sample_weight("balanced", y_train)

    # 2. Define 3 Classification Models
    models = {
        "Logistic_Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_SEED
        ),
        "Random_Forest_Classifier": RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),
        "XGBoost_Classifier": XGBClassifier(
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
    best_macro_f1 = -1.0
    best_model_obj = None

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        t0 = time.time()
        
        if name == "XGBoost_Classifier":
            model.fit(X_train, y_train, sample_weight=xgb_sample_weights)
        else:
            model.fit(X_train, y_train)
            
        fit_dur = time.time() - t0

        # Predict Validation Set
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)

        acc = accuracy_score(y_val, preds)
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_val, preds, average="macro")
        prec_wt, rec_wt, f1_wt, _ = precision_recall_fscore_support(y_val, preds, average="weighted")
        auc = roc_auc_score(y_val, probs, multi_class="ovr", average="macro")

        # Class-specific metrics for High_Risk
        prec_per_class, rec_per_class, f1_per_class, _ = precision_recall_fscore_support(y_val, preds, average=None)
        high_risk_rec = float(rec_per_class[high_risk_idx])
        high_risk_f1 = float(f1_per_class[high_risk_idx])

        cm = confusion_matrix(y_val, preds)
        cm_path = MODELS_DIR / f"cm_val_{name}.png"
        plot_and_save_confusion_matrix(cm, list(label_encoder.classes_), cm_path, f"{name} (Val)")

        metrics = {
            "val_accuracy": float(acc),
            "val_precision_macro": float(prec_macro),
            "val_recall_macro": float(rec_macro),
            "val_f1_macro": float(f1_macro),
            "val_f1_weighted": float(f1_wt),
            "val_high_risk_recall": high_risk_rec,
            "val_high_risk_f1": high_risk_f1,
            "val_roc_auc_macro": float(auc),
            "training_time_sec": float(fit_dur)
        }

        print(f"  Accuracy:          {acc:.4f}")
        print(f"  Macro F1:          {f1_macro:.4f}")
        print(f"  Weighted F1:       {f1_wt:.4f}")
        print(f"  High_Risk Recall:  {high_risk_rec:.4f}")
        print(f"  High_Risk F1:      {high_risk_f1:.4f}")
        print(f"  ROC-AUC Macro:     {auc:.4f}")
        print(f"  Training Time:     {fit_dur:.2f}s")

        # Log to MLflow
        log_model_run(
            run_name=f"Classifier_{name}",
            params=model.get_params() if hasattr(model, "get_params") else {},
            metrics=metrics,
            model_obj=model,
            artifacts={"confusion_matrix": str(cm_path)}
        )

        val_results[name] = {"model": model, "metrics": metrics, "cm": cm}

        # Model selection: Primary key Macro F1 + High_Risk Recall
        selection_score = f1_macro + 0.5 * high_risk_rec
        if selection_score > best_macro_f1:
            best_macro_f1 = selection_score
            best_model_name = name
            best_model_obj = model

    print(f"\nSelected Best Classification Model: {best_model_name}")

    # 3. Save Selected Best Model Locally
    save_best_local_model(best_model_obj, BEST_CLASSIFIER_PATH)

    # Generate Feature Importance for Best Model if available
    fi_path = MODELS_DIR / "classification_feature_importance.png"
    if hasattr(best_model_obj, "feature_importances_"):
        plot_and_save_feature_importance(feature_names, best_model_obj.feature_importances_, fi_path, best_model_name)

    # 4. Final Evaluation ONCE on Test Set
    print("\n--- Final Test Set Evaluation for Selected Classifier ---")
    test_preds = best_model_obj.predict(X_test)
    test_probs = best_model_obj.predict_proba(X_test)

    test_acc = accuracy_score(y_test, test_preds)
    t_prec_macro, t_rec_macro, t_f1_macro, _ = precision_recall_fscore_support(y_test, test_preds, average="macro")
    t_prec_wt, t_rec_wt, t_f1_wt, _ = precision_recall_fscore_support(y_test, test_preds, average="weighted")
    t_auc = roc_auc_score(y_test, test_probs, multi_class="ovr", average="macro")

    t_prec_per_class, t_rec_per_class, t_f1_per_class, _ = precision_recall_fscore_support(y_test, test_preds, average=None)
    t_hr_rec = float(t_rec_per_class[high_risk_idx])
    t_hr_f1 = float(t_f1_per_class[high_risk_idx])

    test_cm = confusion_matrix(y_test, test_preds)
    test_cm_path = MODELS_DIR / "classification_confusion_matrix.png"
    plot_and_save_confusion_matrix(test_cm, list(label_encoder.classes_), test_cm_path, f"Best Classifier {best_model_name} (Test Set)")

    test_report_str = classification_report(y_test, test_preds, target_names=list(label_encoder.classes_))
    print(f"\nTest Classification Report ({best_model_name}):\n{test_report_str}")

    test_metrics = {
        "test_accuracy": float(test_acc),
        "test_precision_macro": float(t_prec_macro),
        "test_recall_macro": float(t_rec_macro),
        "test_f1_macro": float(t_f1_macro),
        "test_f1_weighted": float(t_f1_wt),
        "test_high_risk_recall": t_hr_rec,
        "test_high_risk_f1": t_hr_f1,
        "test_roc_auc_macro": float(t_auc),
        "total_classification_time_sec": float(time.time() - start_time)
    }

    # Register Best Model in MLflow
    log_model_run(
        run_name=f"Best_Classifier_{best_model_name}",
        params=best_model_obj.get_params() if hasattr(best_model_obj, "get_params") else {},
        metrics=test_metrics,
        model_obj=best_model_obj,
        artifacts={
            "test_confusion_matrix": str(test_cm_path),
            "feature_importance": str(fi_path) if fi_path.exists() else ""
        },
        register_model_name="EMIPredict_Best_Classifier"
    )

    return val_results, test_metrics

if __name__ == "__main__":
    train_and_evaluate_classifiers()
