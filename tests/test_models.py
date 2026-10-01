import sys
import pytest
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure project root is in sys.path
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
    DERIVED_RATIO_FEATURES,
    ELIGIBILITY_CLASSES
)
from src.preprocessing.feature_engineering import FinancialRatioTransformer, FullPreprocessingPipeline

# Register classes into __main__ module to ensure joblib unpickling compatibility
main_mod = sys.modules.get("__main__")
if main_mod is not None:
    setattr(main_mod, "FinancialRatioTransformer", FinancialRatioTransformer)
    setattr(main_mod, "FullPreprocessingPipeline", FullPreprocessingPipeline)


def create_valid_test_sample() -> pd.DataFrame:
    """Constructs a single valid borrower profile DataFrame for model inference testing."""
    return pd.DataFrame([{
        "age": 35,
        "gender": "Female",
        "marital_status": "Married",
        "education": "Post Graduate",
        "employment_type": "Salaried",
        "company_type": "MNC",
        "house_type": "Owned",
        "existing_loans": "No",
        "emi_scenario": "Scenario_1",
        "monthly_salary": 85000.0,
        "years_of_employment": 7.5,
        "monthly_rent": 0.0,
        "family_size": 4,
        "dependents": 2,
        "school_fees": 6000.0,
        "college_fees": 0.0,
        "travel_expenses": 5000.0,
        "groceries_utilities": 12000.0,
        "other_monthly_expenses": 4000.0,
        "current_emi_amount": 0.0,
        "credit_score": 780,
        "bank_balance": 350000.0,
        "emergency_fund": 150000.0,
        "requested_amount": 400000.0,
        "requested_tenure": 36.0
    }])


def get_transformed_sample_matrix(sample_df: pd.DataFrame) -> np.ndarray:
    """Preprocesses a raw sample DataFrame into a transformed feature matrix ready for model inference."""
    preprocessor = FullPreprocessingPipeline.load(PREPROCESSOR_PATH)
    ratio_tf = FinancialRatioTransformer()
    sample_augmented = ratio_tf.transform(sample_df)

    input_cols = (
        NUMERICAL_RAW_FEATURES +
        CATEGORICAL_NOMINAL_FEATURES +
        CATEGORICAL_ORDINAL_FEATURES +
        DERIVED_RATIO_FEATURES
    )
    return preprocessor.transform(sample_augmented[input_cols])


def test_classifier_prediction_on_valid_sample():
    """Test that the loaded classifier can make valid predictions on a sample input."""
    assert BEST_CLASSIFIER_PATH.exists(), f"Classifier model missing at {BEST_CLASSIFIER_PATH}"
    classifier = joblib.load(BEST_CLASSIFIER_PATH)

    sample_df = create_valid_test_sample()
    X_trans = get_transformed_sample_matrix(sample_df)

    preds = classifier.predict(X_trans)
    assert len(preds) == 1, f"Expected 1 prediction output, got {len(preds)}"

    # Test probability output
    probs = classifier.predict_proba(X_trans)
    assert probs.shape[0] == 1, "Probability array shape mismatch"
    assert np.isclose(np.sum(probs[0]), 1.0, rtol=1e-3), "Predicted probabilities do not sum to 1.0"


def test_classifier_output_belongs_to_expected_classes():
    """Test that classifier predictions correspond to expected EMI eligibility classes."""
    classifier = joblib.load(BEST_CLASSIFIER_PATH)
    label_encoder_path = MODELS_DIR / "label_encoder.pkl"
    assert label_encoder_path.exists(), f"Label encoder missing at {label_encoder_path}"
    label_encoder = joblib.load(label_encoder_path)

    sample_df = create_valid_test_sample()
    X_trans = get_transformed_sample_matrix(sample_df)

    raw_pred = classifier.predict(X_trans)[0]

    # Verify integer index bounds or class name string
    if isinstance(raw_pred, (int, np.integer)):
        assert 0 <= raw_pred < len(ELIGIBILITY_CLASSES), f"Predicted class index {raw_pred} out of bounds"
        decoded_class = label_encoder.inverse_transform([raw_pred])[0]
    else:
        decoded_class = str(raw_pred)

    assert decoded_class in ELIGIBILITY_CLASSES, f"Decoded class '{decoded_class}' not in expected classes {ELIGIBILITY_CLASSES}"


def test_regressor_returns_numeric_prediction():
    """Test that the loaded regressor returns a valid numeric prediction."""
    assert BEST_REGRESSOR_PATH.exists(), f"Regressor model missing at {BEST_REGRESSOR_PATH}"
    regressor = joblib.load(BEST_REGRESSOR_PATH)

    sample_df = create_valid_test_sample()
    X_trans = get_transformed_sample_matrix(sample_df)

    reg_pred = regressor.predict(X_trans)
    assert len(reg_pred) == 1, f"Expected 1 regression prediction output, got {len(reg_pred)}"

    pred_value = float(reg_pred[0])
    assert isinstance(pred_value, float), f"Predicted value is not a float: {type(pred_value)}"
    assert np.isfinite(pred_value), f"Predicted value is not finite: {pred_value}"


def test_regressor_minimum_safe_emi_boundary():
    """Test that predicted maximum monthly EMI satisfies the application's minimum safe EMI boundary (>= 500 INR)."""
    MINIMUM_SAFE_EMI_BOUNDARY = 500.0  # Defined as minimum safe EMI lower bound in application & regressor contract

    regressor = joblib.load(BEST_REGRESSOR_PATH)

    sample_df = create_valid_test_sample()
    X_trans = get_transformed_sample_matrix(sample_df)

    raw_reg_pred = float(regressor.predict(X_trans)[0])
    max_safe_emi = float(np.clip(raw_reg_pred, MINIMUM_SAFE_EMI_BOUNDARY, None))

    assert max_safe_emi >= MINIMUM_SAFE_EMI_BOUNDARY, (
        f"Predicted max safe EMI {max_safe_emi} is below the application's minimum safe boundary of {MINIMUM_SAFE_EMI_BOUNDARY}"
    )
