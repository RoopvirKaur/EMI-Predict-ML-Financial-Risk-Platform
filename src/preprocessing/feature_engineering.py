import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from typing import Tuple, Dict, Any, List

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    TRAIN_DATASET_PATH,
    VAL_DATASET_PATH,
    TEST_DATASET_PATH,
    PREPROCESSOR_PATH,
    CATEGORICAL_NOMINAL_FEATURES,
    CATEGORICAL_ORDINAL_FEATURES,
    EDUCATION_ORDER,
    NUMERICAL_RAW_FEATURES,
    DERIVED_RATIO_FEATURES,
    CLASSIFICATION_TARGET,
    REGRESSION_TARGET
)

class FinancialRatioTransformer(BaseEstimator, TransformerMixin):
    """
    Computes 5 derived financial ratios without data leakage or division-by-zero errors.
    """
    def __init__(self, eps: float = 1e-5):
        self.eps = eps

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        
        salary = df["monthly_salary"].astype(np.float64)
        rent = df["monthly_rent"].astype(np.float64)
        school = df["school_fees"].astype(np.float64)
        college = df["college_fees"].astype(np.float64)
        travel = df["travel_expenses"].astype(np.float64)
        groceries = df["groceries_utilities"].astype(np.float64)
        other = df["other_monthly_expenses"].astype(np.float64)
        current_emi = df["current_emi_amount"].astype(np.float64)
        bank_bal = df["bank_balance"].astype(np.float64)
        emergency = df["emergency_fund"].astype(np.float64)
        req_amt = df["requested_amount"].astype(np.float64)
        req_tenure = df["requested_tenure"].astype(np.float64).clip(lower=1.0)

        # Total monthly living & obligation expenses
        total_expenses = rent + school + college + travel + groceries + other

        # 1. Debt-to-Income (DTI) Ratio
        df["dti_ratio"] = (current_emi / (salary + self.eps)).astype(np.float32)

        # 2. Total Expense Ratio
        df["expense_ratio"] = (total_expenses / (salary + self.eps)).astype(np.float32)

        # 3. Disposable Income
        df["disposable_income"] = (salary - (total_expenses + current_emi)).astype(np.float32)

        # 4. EMI Affordability Index
        requested_monthly_emi = req_amt / req_tenure
        df["affordability_index"] = (df["disposable_income"] / (requested_monthly_emi + self.eps)).astype(np.float32)

        # 5. Liquidity Reserve Ratio
        total_liquid_reserves = bank_bal + emergency
        df["liquidity_reserve_ratio"] = (total_liquid_reserves / (req_amt + self.eps)).astype(np.float32)

        # Sanity Guardrail against Inf/NaN
        for col in DERIVED_RATIO_FEATURES:
            df[col] = np.nan_to_num(df[col], nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)

        return df

def build_column_transformer() -> ColumnTransformer:
    """Constructs ColumnTransformer for numerical scaling and categorical encoding."""
    all_numerical = NUMERICAL_RAW_FEATURES + DERIVED_RATIO_FEATURES

    numerical_pipe = Pipeline([
        ("scaler", RobustScaler())
    ])

    ordinal_pipe = Pipeline([
        ("ordinal_enc", OrdinalEncoder(
            categories=[EDUCATION_ORDER],
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
    ])

    nominal_pipe = Pipeline([
        ("onehot_enc", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numerical_pipe, all_numerical),
            ("ord", ordinal_pipe, CATEGORICAL_ORDINAL_FEATURES),
            ("nom", nominal_pipe, CATEGORICAL_NOMINAL_FEATURES)
        ],
        remainder="drop"
    )

class FullPreprocessingPipeline(BaseEstimator, TransformerMixin):
    """
    Full reproducible pipeline wrapping FinancialRatioTransformer + ColumnTransformer.
    Fitted STRICTLY on Training Data.
    """
    def __init__(self):
        self.ratio_transformer = FinancialRatioTransformer()
        self.column_transformer = build_column_transformer()

    def fit(self, X: pd.DataFrame, y=None):
        X_ratio = self.ratio_transformer.transform(X)
        self.column_transformer.fit(X_ratio)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_ratio = self.ratio_transformer.transform(X)
        return self.column_transformer.transform(X_ratio)

    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        X_ratio = self.ratio_transformer.transform(X)
        return self.column_transformer.fit_transform(X_ratio)

    def get_feature_names_out(self) -> List[str]:
        return self.column_transformer.get_feature_names_out()

    def save(self, filepath: Path = PREPROCESSOR_PATH):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        print(f"Saved fitted preprocessing pipeline to {filepath}")

    @staticmethod
    def load(filepath: Path = PREPROCESSOR_PATH) -> "FullPreprocessingPipeline":
        if not filepath.exists():
            raise FileNotFoundError(f"Preprocessor not found at {filepath}")
        
        main_mod = sys.modules.get("__main__")
        if main_mod is not None:
            if not hasattr(main_mod, "FinancialRatioTransformer"):
                setattr(main_mod, "FinancialRatioTransformer", FinancialRatioTransformer)
            if not hasattr(main_mod, "FullPreprocessingPipeline"):
                setattr(main_mod, "FullPreprocessingPipeline", FullPreprocessingPipeline)
                
        return joblib.load(filepath)

def run_phase_3():
    """Main execution entry point for Phase 3 Feature Engineering."""
    print("Loading Parquet data splits created in Phase 2...")
    train_df = pd.read_parquet(TRAIN_DATASET_PATH)
    val_df = pd.read_parquet(VAL_DATASET_PATH)
    test_df = pd.read_parquet(TEST_DATASET_PATH)

    # 1. Transform raw DataFrames to include derived financial ratios
    ratio_tf = FinancialRatioTransformer()
    train_feat = ratio_tf.transform(train_df)
    val_feat = ratio_tf.transform(val_df)
    test_feat = ratio_tf.transform(test_df)

    # Save augmented feature DataFrames back to Parquet
    train_feat.to_parquet(TRAIN_DATASET_PATH, index=False)
    val_feat.to_parquet(VAL_DATASET_PATH, index=False)
    test_feat.to_parquet(TEST_DATASET_PATH, index=False)
    print("Augmented Parquet datasets with 5 new financial ratios.")

    # 2. Fit FullPreprocessingPipeline STRICTLY on Training Split
    print("\nFitting FullPreprocessingPipeline strictly on Training set...")
    input_cols = NUMERICAL_RAW_FEATURES + CATEGORICAL_NOMINAL_FEATURES + CATEGORICAL_ORDINAL_FEATURES + DERIVED_RATIO_FEATURES
    
    pipeline = FullPreprocessingPipeline()
    X_train_trans = pipeline.fit_transform(train_feat[input_cols])
    X_val_trans = pipeline.transform(val_feat[input_cols])
    X_test_trans = pipeline.transform(test_feat[input_cols])

    # 3. Save Fitted Pipeline
    pipeline.save(PREPROCESSOR_PATH)

    # Audit & Verification Reporting
    print(f"\nMatrix Shapes After Full Preprocessing:")
    print(f"  - Train Feature Matrix: {X_train_trans.shape}")
    print(f"  - Val Feature Matrix:   {X_val_trans.shape}")
    print(f"  - Test Feature Matrix:  {X_test_trans.shape}")

    return train_feat, val_feat, test_feat, X_train_trans, X_val_trans, X_test_trans

if __name__ == "__main__":
    run_phase_3()
