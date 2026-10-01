import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PREPROCESSOR_PATH,
    NUMERICAL_RAW_FEATURES,
    CATEGORICAL_NOMINAL_FEATURES,
    CATEGORICAL_ORDINAL_FEATURES,
    DERIVED_RATIO_FEATURES
)
from src.preprocessing.feature_engineering import FinancialRatioTransformer, FullPreprocessingPipeline

# Register classes into __main__ for unpickling compatibility
main_mod = sys.modules.get("__main__")
if main_mod is not None:
    setattr(main_mod, "FinancialRatioTransformer", FinancialRatioTransformer)
    setattr(main_mod, "FullPreprocessingPipeline", FullPreprocessingPipeline)


def create_sample_raw_dataframe(
    salary=100000.0,
    rent=15000.0,
    school=5000.0,
    college=2000.0,
    travel=3000.0,
    groceries=10000.0,
    other=5000.0,
    current_emi=10000.0,
    bank_bal=200000.0,
    emergency=100000.0,
    req_amt=500000.0,
    req_tenure=50.0
) -> pd.DataFrame:
    """Helper to construct a valid raw input DataFrame with specified financial values."""
    return pd.DataFrame([{
        "age": 32,
        "gender": "Male",
        "marital_status": "Single",
        "education": "Graduate",
        "employment_type": "Salaried",
        "company_type": "MNC",
        "house_type": "Rented",
        "existing_loans": "Yes",
        "emi_scenario": "Scenario_1",
        "monthly_salary": float(salary),
        "years_of_employment": 6.0,
        "monthly_rent": float(rent),
        "family_size": 3,
        "dependents": 1,
        "school_fees": float(school),
        "college_fees": float(college),
        "travel_expenses": float(travel),
        "groceries_utilities": float(groceries),
        "other_monthly_expenses": float(other),
        "current_emi_amount": float(current_emi),
        "credit_score": 750,
        "bank_balance": float(bank_bal),
        "emergency_fund": float(emergency),
        "requested_amount": float(req_amt),
        "requested_tenure": float(req_tenure)
    }])


def test_dti_ratio_calculation():
    """Test 1: Debt-to-Income (DTI) Ratio calculation."""
    df = create_sample_raw_dataframe(salary=100000.0, current_emi=10000.0)
    transformer = FinancialRatioTransformer()
    transformed_df = transformer.transform(df)

    expected_dti = 10000.0 / (100000.0 + 1e-5)
    actual_dti = transformed_df["dti_ratio"].iloc[0]
    assert np.isclose(actual_dti, expected_dti, rtol=1e-4)


def test_expense_ratio_calculation():
    """Test 2: Total Expense Ratio calculation."""
    df = create_sample_raw_dataframe(
        salary=100000.0,
        rent=15000.0,
        school=5000.0,
        college=2000.0,
        travel=3000.0,
        groceries=10000.0,
        other=5000.0
    )
    transformer = FinancialRatioTransformer()
    transformed_df = transformer.transform(df)

    total_expenses = 15000.0 + 5000.0 + 2000.0 + 3000.0 + 10000.0 + 5000.0  # 40000.0
    expected_expense_ratio = 40000.0 / (100000.0 + 1e-5)
    actual_expense_ratio = transformed_df["expense_ratio"].iloc[0]
    assert np.isclose(actual_expense_ratio, expected_expense_ratio, rtol=1e-4)


def test_disposable_income_calculation():
    """Test 3: Disposable Income calculation."""
    df = create_sample_raw_dataframe(
        salary=100000.0,
        rent=15000.0,
        school=5000.0,
        college=2000.0,
        travel=3000.0,
        groceries=10000.0,
        other=5000.0,
        current_emi=10000.0
    )
    transformer = FinancialRatioTransformer()
    transformed_df = transformer.transform(df)

    total_expenses = 40000.0
    expected_disposable = 100000.0 - (total_expenses + 10000.0)  # 50000.0
    actual_disposable = transformed_df["disposable_income"].iloc[0]
    assert np.isclose(actual_disposable, expected_disposable, rtol=1e-4)


def test_affordability_index_calculation():
    """Test 4: EMI Affordability Index calculation."""
    df = create_sample_raw_dataframe(
        salary=100000.0,
        rent=15000.0,
        school=5000.0,
        college=2000.0,
        travel=3000.0,
        groceries=10000.0,
        other=5000.0,
        current_emi=10000.0,
        req_amt=500000.0,
        req_tenure=50.0
    )
    transformer = FinancialRatioTransformer()
    transformed_df = transformer.transform(df)

    disposable_income = 50000.0
    requested_monthly_emi = 500000.0 / 50.0  # 10000.0
    expected_affordability = disposable_income / (requested_monthly_emi + 1e-5)  # ~5.0
    actual_affordability = transformed_df["affordability_index"].iloc[0]
    assert np.isclose(actual_affordability, expected_affordability, rtol=1e-4)


def test_liquidity_reserve_ratio_calculation():
    """Test 5: Liquidity Reserve Ratio calculation."""
    df = create_sample_raw_dataframe(
        bank_bal=200000.0,
        emergency=100000.0,
        req_amt=500000.0
    )
    transformer = FinancialRatioTransformer()
    transformed_df = transformer.transform(df)

    liquid_reserves = 200000.0 + 100000.0  # 300000.0
    expected_liquidity_ratio = 300000.0 / (500000.0 + 1e-5)  # ~0.6
    actual_liquidity_ratio = transformed_df["liquidity_reserve_ratio"].iloc[0]
    assert np.isclose(actual_liquidity_ratio, expected_liquidity_ratio, rtol=1e-4)


def test_division_by_zero_handling():
    """Test division-by-zero/near-zero handling with 0 salary, 0 requested amount, and 0 tenure."""
    df_zero = create_sample_raw_dataframe(
        salary=0.0,
        current_emi=0.0,
        rent=0.0,
        school=0.0,
        college=0.0,
        travel=0.0,
        groceries=0.0,
        other=0.0,
        bank_bal=0.0,
        emergency=0.0,
        req_amt=0.0,
        req_tenure=0.0
    )

    transformer = FinancialRatioTransformer()
    transformed_df = transformer.transform(df_zero)

    for ratio_col in DERIVED_RATIO_FEATURES:
        col_values = transformed_df[ratio_col]
        assert not col_values.isna().any(), f"NaN value found in {ratio_col}"
        assert not np.isinf(col_values).any(), f"Inf value found in {ratio_col}"
        assert np.isfinite(col_values).all(), f"Non-finite value found in {ratio_col}"


def test_full_preprocessing_pipeline_transform():
    """Test that the full preprocessor pipeline can transform valid input data without errors."""
    assert PREPROCESSOR_PATH.exists(), f"Preprocessor file missing at {PREPROCESSOR_PATH}"
    pipeline = FullPreprocessingPipeline.load(PREPROCESSOR_PATH)

    df_raw = create_sample_raw_dataframe()
    transformer = FinancialRatioTransformer()
    df_augmented = transformer.transform(df_raw)

    input_cols = (
        NUMERICAL_RAW_FEATURES +
        CATEGORICAL_NOMINAL_FEATURES +
        CATEGORICAL_ORDINAL_FEATURES +
        DERIVED_RATIO_FEATURES
    )

    transformed_matrix = pipeline.transform(df_augmented[input_cols])

    assert isinstance(transformed_matrix, np.ndarray), "Output is not a numpy ndarray"
    assert transformed_matrix.shape[0] == 1, f"Expected 1 sample row, got {transformed_matrix.shape[0]}"
    assert transformed_matrix.shape[1] > 0, "Feature matrix has 0 columns"
    assert not np.isnan(transformed_matrix).any(), "Transformed feature matrix contains NaNs"
    assert np.isfinite(transformed_matrix).all(), "Transformed feature matrix contains non-finite values"
