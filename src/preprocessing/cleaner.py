import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    RAW_DATASET_PATH,
    TRAIN_DATASET_PATH,
    VAL_DATASET_PATH,
    TEST_DATASET_PATH,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    CLASSIFICATION_TARGET,
    REGRESSION_TARGET,
    CATEGORICAL_NOMINAL_FEATURES,
    CATEGORICAL_ORDINAL_FEATURES,
    NUMERICAL_RAW_FEATURES,
    ELIGIBILITY_CLASSES
)

GENDER_MAP = {
    'Female': 'Female', 'female': 'Female', 'F': 'Female', 'FEMALE': 'Female',
    'Male': 'Male', 'male': 'Male', 'M': 'Male', 'MALE': 'Male'
}

OPTIMIZED_DTYPES = {
    "age": "int8",
    "gender": "category",
    "marital_status": "category",
    "education": "category",
    "monthly_salary": "float32",
    "employment_type": "category",
    "years_of_employment": "float32",
    "company_type": "category",
    "house_type": "category",
    "monthly_rent": "float32",
    "family_size": "int8",
    "dependents": "int8",
    "school_fees": "float32",
    "college_fees": "float32",
    "travel_expenses": "float32",
    "groceries_utilities": "float32",
    "other_monthly_expenses": "float32",
    "existing_loans": "category",
    "current_emi_amount": "float32",
    "credit_score": "int16",
    "bank_balance": "float32",
    "emergency_fund": "float32",
    "emi_scenario": "category",
    "requested_amount": "float32",
    "requested_tenure": "int16",
    "max_monthly_emi": "float32",
    "emi_eligibility": "category"
}

def load_raw_data(data_path: Path = RAW_DATASET_PATH) -> pd.DataFrame:
    """Loads actual dataset CSV with low_memory=False to prevent DtypeWarning."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {data_path}")
    
    print(f"Loading raw dataset from {data_path}...")
    df = pd.read_csv(data_path, low_memory=False)
    df.columns = df.columns.str.strip()
    print(f"Loaded dataset: {df.shape[0]:,} records, {df.shape[1]} columns.")
    return df

def clean_data_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs initial row-level cleaning:
    - Numeric type conversions (errors='coerce')
    - Gender standardization
    - Categorical string whitespace stripping
    - Duplicate row removal
    - Range validation and clip limits
    - Target column validation
    """
    df = df.copy()
    initial_count = len(df)

    # 1. Numeric Type Conversions
    for col in NUMERICAL_RAW_FEATURES + [REGRESSION_TARGET]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 2. Gender Standardization
    if 'gender' in df.columns:
        df['gender'] = df['gender'].astype(str).str.strip().map(GENDER_MAP)

    # 3. Categorical Whitespace Stripping
    cat_cols = CATEGORICAL_NOMINAL_FEATURES + CATEGORICAL_ORDINAL_FEATURES + [CLASSIFICATION_TARGET]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # 4. Duplicate Removal
    df = df.drop_duplicates().reset_index(drop=True)
    dedup_count = len(df)
    if dedup_count < initial_count:
        print(f"Removed {initial_count - dedup_count:,} duplicate records.")

    # 5. Target Validation
    valid_targets = set(ELIGIBILITY_CLASSES)
    df = df[df[CLASSIFICATION_TARGET].isin(valid_targets)].copy()
    df = df[df[REGRESSION_TARGET].notnull() & (df[REGRESSION_TARGET] >= 500)].copy()

    # 6. Range Validation Safeguards
    if 'age' in df.columns:
        df['age'] = df['age'].clip(25, 60)
    if 'credit_score' in df.columns:
        df['credit_score'] = df['credit_score'].clip(300, 850)
    if 'monthly_salary' in df.columns:
        df['monthly_salary'] = df['monthly_salary'].clip(lower=0)
    if 'requested_amount' in df.columns:
        df['requested_amount'] = df['requested_amount'].clip(lower=1000)
    if 'requested_tenure' in df.columns:
        df['requested_tenure'] = df['requested_tenure'].clip(lower=1)

    print(f"Cleaned dataset: {len(df):,} valid records remaining.")
    return df

def split_and_impute_without_leakage(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Prevents Data Leakage:
    1. Stratified split into Train (70%), Val (15%), Test (15%) on emi_eligibility.
    2. Imputation statistics (medians for numerical, mode for categorical) computed STRICTLY ON TRAIN SPLIT.
    3. Train statistics applied to Val and Test splits.
    """
    # 1. Stratified Splitting
    stratify_target = df[CLASSIFICATION_TARGET]

    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        random_state=RANDOM_SEED,
        stratify=stratify_target
    )

    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        random_state=RANDOM_SEED,
        stratify=temp_df[CLASSIFICATION_TARGET]
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    # 2. Compute Imputation Parameters STRICTLY on Train Set
    imputation_params = {}
    
    # Numerical medians from train_df
    for col in NUMERICAL_RAW_FEATURES:
        if train_df[col].isnull().sum() > 0:
            median_val = train_df[col].median()
            imputation_params[col] = float(median_val)

    # Categorical modes from train_df
    for col in CATEGORICAL_NOMINAL_FEATURES + CATEGORICAL_ORDINAL_FEATURES:
        if train_df[col].isnull().sum() > 0:
            mode_val = train_df[col].mode()[0]
            imputation_params[col] = str(mode_val)

    print(f"\nImputation Parameters computed (strictly on Training split):")
    for k, v in imputation_params.items():
        print(f"  - {k}: {v}")

    # 3. Apply Imputation Parameters to Train, Val, and Test
    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        for col, fill_val in imputation_params.items():
            split_df[col] = split_df[col].fillna(fill_val)

    # 4. Cast Data Types for Memory Efficiency
    for split_df in [train_df, val_df, test_df]:
        for col, dtype in OPTIMIZED_DTYPES.items():
            if col in split_df.columns:
                if dtype == "category":
                    split_df[col] = split_df[col].astype("category")
                else:
                    split_df[col] = split_df[col].astype(dtype)

    # 5. Export to Parquet
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(TRAIN_DATASET_PATH, index=False)
    val_df.to_parquet(VAL_DATASET_PATH, index=False)
    test_df.to_parquet(TEST_DATASET_PATH, index=False)

    print(f"\nData successfully saved to Parquet format in {PROCESSED_DATA_DIR}:")
    print(f"  - Train Split: {len(train_df):,} records ({len(train_df)/len(df):.1%})")
    print(f"  - Val Split:   {len(val_df):,} records ({len(val_df)/len(df):.1%})")
    print(f"  - Test Split:  {len(test_df):,} records ({len(test_df)/len(df):.1%})")

    # 6. Report Class Imbalance Distribution Across Splits
    print("\nClass Distribution Across Splits (emi_eligibility):")
    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        counts = split[CLASSIFICATION_TARGET].value_counts()
        props = split[CLASSIFICATION_TARGET].value_counts(normalize=True) * 100
        print(f"  {name} Set:")
        for cls in ELIGIBILITY_CLASSES:
            cnt = counts.get(cls, 0)
            prp = props.get(cls, 0.0)
            print(f"    - {cls:12s}: {cnt:7,} ({prp:5.2f}%)")

    return train_df, val_df, test_df, imputation_params

def run_phase_2():
    """Main execution function for Phase 2 Data Cleaning & Quality Pipeline."""
    df_raw = load_raw_data()
    df_clean = clean_data_structure(df_raw)
    train_df, val_df, test_df, params = split_and_impute_without_leakage(df_clean)
    return train_df, val_df, test_df

if __name__ == "__main__":
    run_phase_2()
