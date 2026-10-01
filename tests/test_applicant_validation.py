import sys
import pytest
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PREPROCESSOR_PATH,
    BEST_CLASSIFIER_PATH,
    BEST_REGRESSOR_PATH,
    MODELS_DIR,
    NUMERICAL_RAW_FEATURES,
    CATEGORICAL_NOMINAL_FEATURES,
    CATEGORICAL_ORDINAL_FEATURES,
    DERIVED_RATIO_FEATURES
)
from src.preprocessing.feature_engineering import FinancialRatioTransformer, FullPreprocessingPipeline

# Register classes into __main__ module for unpickling compatibility
main_mod = sys.modules.get("__main__")
if main_mod is not None:
    setattr(main_mod, "FinancialRatioTransformer", FinancialRatioTransformer)
    setattr(main_mod, "FullPreprocessingPipeline", FullPreprocessingPipeline)


def validate_employment_salary(employment_type: str, monthly_salary: float) -> tuple[bool, str]:
    if monthly_salary < 0:
        return False, "Monthly salary cannot be negative."
    if employment_type != "Unemployed" and monthly_salary <= 0:
        return False, "Monthly salary must be greater than ₹0 for employed applicants."
    return True, ""


def evaluate_test_record(record_dict: dict) -> dict:
    preprocessor = FullPreprocessingPipeline.load(PREPROCESSOR_PATH)
    classifier = joblib.load(BEST_CLASSIFIER_PATH)
    regressor = joblib.load(BEST_REGRESSOR_PATH)
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.pkl")

    df_single = pd.DataFrame([record_dict])
    ratio_tf = FinancialRatioTransformer()
    df_ratios = ratio_tf.transform(df_single)

    input_cols = (
        NUMERICAL_RAW_FEATURES +
        CATEGORICAL_NOMINAL_FEATURES +
        CATEGORICAL_ORDINAL_FEATURES +
        DERIVED_RATIO_FEATURES
    )
    X_trans = preprocessor.transform(df_ratios[input_cols])

    clf_pred = classifier.predict(X_trans)[0]
    tier_name = label_encoder.classes_[clf_pred]

    reg_pred = regressor.predict(X_trans)[0]
    max_safe_emi = float(np.clip(reg_pred, 500.0, None))

    record_dict["dti_ratio"] = round(float(df_ratios["dti_ratio"].iloc[0]), 4)
    record_dict["disposable_income"] = round(float(df_ratios["disposable_income"].iloc[0]), 2)
    record_dict["affordability_index"] = round(float(df_ratios["affordability_index"].iloc[0]), 2)
    record_dict["emi_eligibility"] = tier_name
    record_dict["max_monthly_emi"] = round(max_safe_emi, 2)

    return record_dict


def test_user_provided_applicant_id_preserved():
    """Test that a user-provided Applicant ID such as TEST-1001 is strictly preserved."""
    test_id = "TEST-1001"
    raw_record = {
        "applicant_id": test_id,
        "age": 30,
        "gender": "Male",
        "marital_status": "Single",
        "education": "Graduate",
        "employment_type": "Salaried",
        "company_type": "MNC",
        "years_of_employment": 4.0,
        "house_type": "Rented",
        "family_size": 2,
        "dependents": 0,
        "monthly_salary": 75000.0,
        "monthly_rent": 15000.0,
        "school_fees": 0.0,
        "college_fees": 0.0,
        "travel_expenses": 3000.0,
        "groceries_utilities": 8000.0,
        "other_monthly_expenses": 4000.0,
        "current_emi_amount": 5000.0,
        "credit_score": 750,
        "bank_balance": 250000.0,
        "emergency_fund": 100000.0,
        "requested_amount": 300000.0,
        "requested_tenure": 36,
        "existing_loans": "Yes",
        "emi_scenario": "Scenario_1"
    }

    evaluated = evaluate_test_record(raw_record)
    assert evaluated["applicant_id"] == "TEST-1001", f"Expected Applicant ID 'TEST-1001', got {evaluated['applicant_id']}"

    # Verify DataFrame concatenation preserves the exact ID
    df = pd.DataFrame([evaluated])
    assert df.iloc[0]["applicant_id"] == "TEST-1001", "DataFrame conversion modified the Applicant ID"


def test_unemployed_applicant_zero_salary_accepted():
    """Test that an unemployed applicant with monthly_salary = 0 is accepted as valid."""
    is_valid, msg = validate_employment_salary("Unemployed", 0.0)
    assert is_valid is True, f"Validation failed for unemployed applicant with ₹0 salary: {msg}"

    unemployed_record = {
        "applicant_id": "TEST-UNEMP-001",
        "age": 28,
        "gender": "Female",
        "marital_status": "Single",
        "education": "High School",
        "employment_type": "Unemployed",
        "company_type": "Other",
        "years_of_employment": 0.0,
        "house_type": "Parental",
        "family_size": 3,
        "dependents": 0,
        "monthly_salary": 0.0,
        "monthly_rent": 0.0,
        "school_fees": 0.0,
        "college_fees": 0.0,
        "travel_expenses": 1000.0,
        "groceries_utilities": 4000.0,
        "other_monthly_expenses": 2000.0,
        "current_emi_amount": 0.0,
        "credit_score": 620,
        "bank_balance": 15000.0,
        "emergency_fund": 5000.0,
        "requested_amount": 50000.0,
        "requested_tenure": 12,
        "existing_loans": "No",
        "emi_scenario": "Scenario_1"
    }

    evaluated = evaluate_test_record(unemployed_record)
    assert evaluated["monthly_salary"] == 0.0, f"Expected monthly_salary 0.0, got {evaluated['monthly_salary']}"
    assert "emi_eligibility" in evaluated, "Evaluation failed to predict eligibility"


def test_negative_salary_rejected():
    """Test that negative salary is rejected for both unemployed and employed applicants."""
    is_valid_unemp, msg1 = validate_employment_salary("Unemployed", -5000.0)
    assert is_valid_unemp is False, "Negative salary should be rejected for Unemployed applicants"
    assert "negative" in msg1.lower()

    is_valid_sal, msg2 = validate_employment_salary("Salaried", -10000.0)
    assert is_valid_sal is False, "Negative salary should be rejected for Salaried applicants"
    assert "negative" in msg2.lower()

    # Employed with 0 salary should also be rejected
    is_valid_sal_zero, msg3 = validate_employment_salary("Salaried", 0.0)
    assert is_valid_sal_zero is False, "₹0 salary should be rejected for Salaried applicants"
