import os
from pathlib import Path

# Set environment variable to allow local file store fallback if needed
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"
MLRUNS_DIR = BASE_DIR / "mlruns"
SQLITE_DB_PATH = BASE_DIR / "mlflow.db"

# Raw Dataset Path (Actual Dataset)
RAW_DATASET_PATH = DATASET_DIR / "emi_prediction_dataset.csv"

# Processed Data Paths
TRAIN_DATASET_PATH = PROCESSED_DATA_DIR / "train.parquet"
VAL_DATASET_PATH = PROCESSED_DATA_DIR / "val.parquet"
TEST_DATASET_PATH = PROCESSED_DATA_DIR / "test.parquet"

# Model Paths
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
BEST_CLASSIFIER_PATH = MODELS_DIR / "classifier_best.pkl"
BEST_REGRESSOR_PATH = MODELS_DIR / "regressor_best.pkl"

# Random Seed
RANDOM_SEED = 42

# Target Definitions
CLASSIFICATION_TARGET = "emi_eligibility"
REGRESSION_TARGET = "max_monthly_emi"
ELIGIBILITY_CLASSES = ["Eligible", "High_Risk", "Not_Eligible"]

# Feature Definitions
CATEGORICAL_NOMINAL_FEATURES = [
    "gender",
    "marital_status",
    "employment_type",
    "company_type",
    "house_type",
    "existing_loans",
    "emi_scenario"
]

CATEGORICAL_ORDINAL_FEATURES = ["education"]
EDUCATION_ORDER = ["High School", "Graduate", "Post Graduate", "Professional"]

NUMERICAL_RAW_FEATURES = [
    "age",
    "monthly_salary",
    "years_of_employment",
    "monthly_rent",
    "family_size",
    "dependents",
    "school_fees",
    "college_fees",
    "travel_expenses",
    "groceries_utilities",
    "other_monthly_expenses",
    "current_emi_amount",
    "credit_score",
    "bank_balance",
    "emergency_fund",
    "requested_amount",
    "requested_tenure"
]

DERIVED_RATIO_FEATURES = [
    "dti_ratio",
    "expense_ratio",
    "disposable_income",
    "affordability_index",
    "liquidity_reserve_ratio"
]

# MLflow Experiment Configuration
MLFLOW_EXPERIMENT_NAME = "EMI_Predict_Risk_Assessment"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{SQLITE_DB_PATH.as_posix()}")
