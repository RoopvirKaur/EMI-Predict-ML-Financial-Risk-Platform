import streamlit as st
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure root project directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    PREPROCESSOR_PATH,
    BEST_CLASSIFIER_PATH,
    BEST_REGRESSOR_PATH,
    MODELS_DIR,
    NUMERICAL_RAW_FEATURES,
    CATEGORICAL_NOMINAL_FEATURES,
    CATEGORICAL_ORDINAL_FEATURES,
    DERIVED_RATIO_FEATURES,
    EDUCATION_ORDER
)
from src.preprocessing.feature_engineering import FullPreprocessingPipeline, FinancialRatioTransformer

st.set_page_config(
    page_title="Realtime Risk Predictor - EMIPredict AI",
    page_icon="🔮",
    layout="wide"
)

# Custom Styling for Stepper & Results
st.markdown("""
<style>
    /* Workflow Stepper Banner */
    .stepper-container {
        background: linear-gradient(90deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .step-item {
        flex: 1;
        text-align: center;
        padding: 0.5rem;
    }
    .step-num {
        display: inline-block;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background-color: #38BDF8;
        color: #0F172A;
        font-weight: 800;
        font-size: 0.9rem;
        line-height: 28px;
        margin-right: 0.4rem;
    }
    .step-text {
        color: #F8FAFC;
        font-size: 0.95rem;
        font-weight: 600;
    }

    /* Section Cards */
    .stage-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .stage-title {
        color: #38BDF8;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Stage 2 Submit Button Prominence Match */
    div[data-testid="stFormSubmitButton"] > button,
    div[data-testid="stFormSubmitButton"] > button p,
    .stButton > button,
    .stButton > button p {
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.01em !important;
        padding-top: 0.6rem !important;
        padding-bottom: 0.6rem !important;
    }

    /* Prediction Result Cards */
    .pred-card-eligible {
        background-color: rgba(46, 204, 113, 0.1);
        border: 2px solid #2ECC71;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .pred-card-high-risk {
        background-color: rgba(241, 196, 15, 0.1);
        border: 2px solid #F1C40F;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .pred-card-not-eligible {
        background-color: rgba(231, 76, 60, 0.1);
        border: 2px solid #E74C3C;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .status-badge {
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        margin-top: 0.5rem;
    }
    .ratio-card {
        background-color: #0F172A;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔮 Real-Time Borrower Risk & Safe EMI Predictor")
st.caption("Multi-Task inference engine evaluating applicant risk tier and predicting maximum sustainable monthly EMI.")

# Cached Model Artifact Loading
@st.cache_resource
def load_all_model_artifacts():
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Preprocessor pkl missing at {PREPROCESSOR_PATH}")
    if not BEST_CLASSIFIER_PATH.exists():
        raise FileNotFoundError(f"Classifier pkl missing at {BEST_CLASSIFIER_PATH}")
    if not BEST_REGRESSOR_PATH.exists():
        raise FileNotFoundError(f"Regressor pkl missing at {BEST_REGRESSOR_PATH}")
    
    label_encoder_path = MODELS_DIR / "label_encoder.pkl"
    if not label_encoder_path.exists():
        raise FileNotFoundError(f"Label encoder pkl missing at {label_encoder_path}")

    # Register class references into __main__ module to guarantee pickle unpickling compatibility
    main_mod = sys.modules.get("__main__")
    if main_mod is not None:
        setattr(main_mod, "FinancialRatioTransformer", FinancialRatioTransformer)
        setattr(main_mod, "FullPreprocessingPipeline", FullPreprocessingPipeline)

    preprocessor = FullPreprocessingPipeline.load(PREPROCESSOR_PATH)
    classifier = joblib.load(BEST_CLASSIFIER_PATH)
    regressor = joblib.load(BEST_REGRESSOR_PATH)
    label_encoder = joblib.load(label_encoder_path)

    return preprocessor, classifier, regressor, label_encoder

try:
    preprocessor, classifier, regressor, label_encoder = load_all_model_artifacts()
except Exception as e:
    st.error(f"Failed to load trained model artifacts: {e}")
    st.stop()

# Visual Workflow Stepper Bar
st.markdown("""
<div class="stepper-container">
    <div class="step-item">
        <span class="step-num">1</span>
        <span class="step-text">Fill Borrower Profile</span>
    </div>
    <div style="color: #475569; font-weight: bold;">➔</div>
    <div class="step-item">
        <span class="step-num">2</span>
        <span class="step-text">Click Evaluate Button</span>
    </div>
    <div style="color: #475569; font-weight: bold;">➔</div>
    <div class="step-item">
        <span class="step-num">3</span>
        <span class="step-text">View Risk & Safe EMI</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Quick Presets for Demo Testing
st.markdown("### ⚡ Quick Profile Presets")
st.caption("Click any preset below to pre-fill the form with benchmark borrower financial parameters:")
preset_col1, preset_col2, preset_col3 = st.columns(3)

if "preset" not in st.session_state:
    st.session_state.preset = "Standard Salaried"

with preset_col1:
    if st.button("🟢 Load Low Risk Profile", width="stretch"):
        st.session_state.preset = "Low Risk"
        st.session_state.has_evaluated = True

with preset_col2:
    if st.button("🟡 Load Borderline / High Risk Profile", width="stretch"):
        st.session_state.preset = "High Risk"
        st.session_state.has_evaluated = True

with preset_col3:
    if st.button("🔴 Load Debt-Stressed / Overleveraged Profile", width="stretch"):
        st.session_state.preset = "Not Eligible"
        st.session_state.has_evaluated = True

# Set default values based on preset
defaults = {
    "Low Risk": {
        "monthly_salary": 85000.0, "credit_score": 780, "bank_balance": 450000.0, "emergency_fund": 200000.0,
        "requested_amount": 300000.0, "requested_tenure": 36, "current_emi_amount": 5000.0, "monthly_rent": 12000.0,
        "school_fees": 3000.0, "college_fees": 0.0, "travel_expenses": 4000.0, "groceries_utilities": 10000.0,
        "other_monthly_expenses": 5000.0, "years_of_employment": 8.0, "employment_type": "Salaried", "company_type": "MNC"
    },
    "High Risk": {
        "monthly_salary": 38000.0, "credit_score": 640, "bank_balance": 50000.0, "emergency_fund": 15000.0,
        "requested_amount": 400000.0, "requested_tenure": 24, "current_emi_amount": 12000.0, "monthly_rent": 8000.0,
        "school_fees": 4000.0, "college_fees": 2000.0, "travel_expenses": 3500.0, "groceries_utilities": 8000.0,
        "other_monthly_expenses": 4000.0, "years_of_employment": 2.5, "employment_type": "Self-Employed", "company_type": "Private"
    },
    "Not Eligible": {
        "monthly_salary": 25000.0, "credit_score": 540, "bank_balance": 8000.0, "emergency_fund": 2000.0,
        "requested_amount": 500000.0, "requested_tenure": 12, "current_emi_amount": 14000.0, "monthly_rent": 7000.0,
        "school_fees": 3000.0, "college_fees": 1500.0, "travel_expenses": 3000.0, "groceries_utilities": 7000.0,
        "other_monthly_expenses": 3500.0, "years_of_employment": 1.0, "employment_type": "Freelancer", "company_type": "Other"
    }
}.get(st.session_state.preset, {
    "monthly_salary": 60000.0, "credit_score": 710, "bank_balance": 180000.0, "emergency_fund": 60000.0,
    "requested_amount": 250000.0, "requested_tenure": 36, "current_emi_amount": 8000.0, "monthly_rent": 10000.0,
    "school_fees": 2000.0, "college_fees": 0.0, "travel_expenses": 4000.0, "groceries_utilities": 9000.0,
    "other_monthly_expenses": 4000.0, "years_of_employment": 5.0, "employment_type": "Salaried", "company_type": "Private"
})

st.markdown("<br>", unsafe_allow_html=True)

# STAGE 1: APPLICANT INPUT FORM
st.markdown("""
<div class="stage-title">
    📝 STAGE 1: Enter Borrower Information & Loan Parameters
</div>
""", unsafe_allow_html=True)

with st.form("applicant_prediction_form"):
    col1, col2, col3 = st.columns(3)
    
    # Section 1: Demographics & Employment
    with col1:
        st.markdown("##### 👤 Demographics & Employment")
        age = st.number_input("Age", min_value=18, max_value=75, value=32, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married"])
        education = st.selectbox("Education Level", EDUCATION_ORDER, index=1)
        employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Freelancer", "Unemployed"],
                                      index=["Salaried", "Self-Employed", "Freelancer", "Unemployed"].index(defaults["employment_type"]))
        company_type = st.selectbox("Company Type", ["MNC", "Private", "Government", "Startup", "Other"],
                                    index=["MNC", "Private", "Government", "Startup", "Other"].index(defaults["company_type"]))
        years_of_employment = st.number_input("Years of Employment", min_value=0.0, max_value=50.0, value=float(defaults["years_of_employment"]), step=0.5)
        house_type = st.selectbox("House Type", ["Rented", "Owned", "Parental"])
        family_size = st.number_input("Family Size", min_value=1, max_value=12, value=3, step=1)
        dependents = st.number_input("Dependents", min_value=0, max_value=10, value=1, step=1)

    # Section 2: Income & Wealth Reserves
    with col2:
        st.markdown("##### 💰 Income, Credit & Reserves")
        monthly_salary = st.number_input("Monthly Gross Salary (INR ₹)", min_value=0.0, max_value=2000000.0, value=float(defaults["monthly_salary"]), step=1000.0)
        credit_score = st.slider("Credit Score (CIBIL / Experian)", min_value=300, max_value=850, value=int(defaults["credit_score"]), step=5)
        bank_balance = st.number_input("Current Bank Balance (INR ₹)", min_value=0.0, max_value=10000000.0, value=float(defaults["bank_balance"]), step=5000.0)
        emergency_fund = st.number_input("Emergency Reserve Fund (INR ₹)", min_value=0.0, max_value=5000000.0, value=float(defaults["emergency_fund"]), step=5000.0)
        existing_loans = st.selectbox("Existing Active Loans", ["Yes", "No"], index=0 if defaults["current_emi_amount"] > 0 else 1)
        current_emi_amount = st.number_input("Current Existing Monthly EMI (INR ₹)", min_value=0.0, max_value=500000.0, value=float(defaults["current_emi_amount"]), step=500.0)

    # Section 3: Monthly Expenses & Loan Request
    with col3:
        st.markdown("##### 🧾 Expenses & Requested Loan")
        monthly_rent = st.number_input("Monthly Rent (INR ₹)", min_value=0.0, max_value=200000.0, value=float(defaults["monthly_rent"]), step=500.0)
        school_fees = st.number_input("School Fees (INR ₹)", min_value=0.0, max_value=100000.0, value=float(defaults["school_fees"]), step=500.0)
        college_fees = st.number_input("College Fees (INR ₹)", min_value=0.0, max_value=100000.0, value=float(defaults["college_fees"]), step=500.0)
        travel_expenses = st.number_input("Travel Expenses (INR ₹)", min_value=0.0, max_value=100000.0, value=float(defaults["travel_expenses"]), step=500.0)
        groceries_utilities = st.number_input("Groceries & Utilities (INR ₹)", min_value=0.0, max_value=100000.0, value=float(defaults["groceries_utilities"]), step=500.0)
        other_monthly_expenses = st.number_input("Other Monthly Expenses (INR ₹)", min_value=0.0, max_value=100000.0, value=float(defaults["other_monthly_expenses"]), step=500.0)
        
        st.markdown("---")
        requested_amount = st.number_input("Requested Loan Amount (INR ₹)", min_value=5000.0, max_value=10000000.0, value=float(defaults["requested_amount"]), step=10000.0)
        requested_tenure = st.number_input("Requested Tenure (Months)", min_value=1, max_value=360, value=int(defaults["requested_tenure"]), step=1)
        emi_scenario = st.selectbox("EMI Scenario Category", ["Scenario_1", "Scenario_2", "Scenario_3", "Scenario_4", "Scenario_5"], index=0)

    st.markdown("<br>", unsafe_allow_html=True)
    # STAGE 2: EVALUATE BUTTON TRANSITION
    submit_btn = st.form_submit_button("🚀 STAGE 2: Evaluate Borrower Risk & Max Safe EMI", width="stretch")

if submit_btn:
    if monthly_salary < 0:
        st.error("⚠️ Monthly salary cannot be negative.")
        st.session_state.has_evaluated = False
    elif employment_type != "Unemployed" and monthly_salary <= 0:
        st.error("⚠️ Monthly salary must be greater than ₹0 for employed applicants.")
        st.session_state.has_evaluated = False
    else:
        st.session_state.has_evaluated = True

# STAGE 3: RESULTS SECTION
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="stage-title">
    🎯 STAGE 3: Underwriting Assessment & Financial Risk Results
</div>
""", unsafe_allow_html=True)

# Check if user has triggered evaluation or preset
if getattr(st.session_state, "has_evaluated", True):
    # 1. Input DataFrame Construction
    input_data = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "marital_status": marital_status,
        "education": education,
        "employment_type": employment_type,
        "company_type": company_type,
        "house_type": house_type,
        "existing_loans": existing_loans,
        "emi_scenario": emi_scenario,
        "monthly_salary": monthly_salary,
        "years_of_employment": years_of_employment,
        "monthly_rent": monthly_rent,
        "family_size": family_size,
        "dependents": dependents,
        "school_fees": school_fees,
        "college_fees": college_fees,
        "travel_expenses": travel_expenses,
        "groceries_utilities": groceries_utilities,
        "other_monthly_expenses": other_monthly_expenses,
        "current_emi_amount": current_emi_amount,
        "credit_score": credit_score,
        "bank_balance": bank_balance,
        "emergency_fund": emergency_fund,
        "requested_amount": requested_amount,
        "requested_tenure": requested_tenure
    }])

    # Input Validation Guardrail
    if monthly_salary < 0:
        st.error("⚠️ Monthly salary cannot be negative.")
        st.stop()
    elif employment_type != "Unemployed" and monthly_salary <= 0:
        st.error("⚠️ Monthly salary must be greater than zero for employed applicants.")
        st.stop()
    if requested_tenure < 1:
        st.error("⚠️ Requested tenure must be at least 1 month.")
        st.stop()

    # 2. Pipeline Feature Engineering & Transformation
    ratio_tf = FinancialRatioTransformer()
    input_with_ratios = ratio_tf.transform(input_data)
    
    input_cols = NUMERICAL_RAW_FEATURES + CATEGORICAL_NOMINAL_FEATURES + CATEGORICAL_ORDINAL_FEATURES + DERIVED_RATIO_FEATURES
    X_trans = preprocessor.transform(input_with_ratios[input_cols])

    # 3. Model Predictions
    clf_pred_idx = classifier.predict(X_trans)[0]
    clf_probs = classifier.predict_proba(X_trans)[0]
    
    classes_list = list(label_encoder.classes_)
    predicted_tier = classes_list[clf_pred_idx]
    
    reg_pred_emi = regressor.predict(X_trans)[0]
    max_safe_emi = float(np.clip(reg_pred_emi, 500.0, None))
    
    requested_monthly_emi = float(requested_amount / max(requested_tenure, 1))

    # Derived Ratios for display
    dti_val = float(input_with_ratios["dti_ratio"].iloc[0])
    exp_ratio_val = float(input_with_ratios["expense_ratio"].iloc[0])
    disp_income_val = float(input_with_ratios["disposable_income"].iloc[0])
    affordability_val = float(input_with_ratios["affordability_index"].iloc[0])
    liquidity_val = float(input_with_ratios["liquidity_reserve_ratio"].iloc[0])

    res_col1, res_col2 = st.columns([1.2, 1])

    with res_col1:
        # Prediction Card styling based on tier
        if predicted_tier == "Eligible":
            st.markdown(f"""
            <div class="pred-card-eligible">
                <div style="color: #2ECC71; font-weight: 700; font-size: 1.1rem;">CLASSIFICATION TARGET</div>
                <div class="status-badge" style="color: #2ECC71;">✅ ELIGIBLE</div>
                <div style="color: #CBD5E1; margin-top: 0.5rem; font-size: 0.95rem;">
                    Borrower meets credit score and cashflow requirements for immediate loan approval.
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif predicted_tier == "High_Risk":
            st.markdown(f"""
            <div class="pred-card-high-risk">
                <div style="color: #F1C40F; font-weight: 700; font-size: 1.1rem;">CLASSIFICATION TARGET</div>
                <div class="status-badge" style="color: #F1C40F;">⚠️ HIGH RISK</div>
                <div style="color: #CBD5E1; margin-top: 0.5rem; font-size: 0.95rem;">
                    Cautionary tier. High DTI or tight liquidity reserves require risk mitigation or lowered EMI cap.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="pred-card-not-eligible">
                <div style="color: #E74C3C; font-weight: 700; font-size: 1.1rem;">CLASSIFICATION TARGET</div>
                <div class="status-badge" style="color: #E74C3C;">❌ NOT ELIGIBLE</div>
                <div style="color: #CBD5E1; margin-top: 0.5rem; font-size: 0.95rem;">
                    High probability of default. Disposable cashflow deficit or severe debt burden detected.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### 📊 Risk Tier Probabilities")
        for cls_name, prob in zip(classes_list, clf_probs):
            st.write(f"**{cls_name}**: `{prob * 100:.1f}%`")
            st.progress(float(prob))

    with res_col2:
        st.markdown("##### 💰 Safe EMI Capacity vs Requested EMI")
        
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Predicted Max Safe EMI", f"₹{max_safe_emi:,.2f} / mo")
        m_col2.metric("Requested Monthly EMI", f"₹{requested_monthly_emi:,.2f} / mo")

        if requested_monthly_emi <= max_safe_emi:
            st.success(f"✅ Requested monthly EMI (₹{requested_monthly_emi:,.2f}) is **WITHIN** the predicted maximum safe limit (₹{max_safe_emi:,.2f}).")
        else:
            st.warning(f"⚠️ Requested monthly EMI (₹{requested_monthly_emi:,.2f}) **EXCEEDS** maximum safe EMI capacity by **₹{requested_monthly_emi - max_safe_emi:,.2f}**. Consider extending tenure to at least {int(np.ceil(requested_amount / max_safe_emi))} months.")

        st.markdown("##### 📐 Derived Financial Ratios")
        
        st.markdown(f"""
        <div class="ratio-card">
            <b>Debt-to-Income (DTI) Ratio:</b> <span style="float:right; font-weight:bold;">{dti_val * 100:.1f}%</span><br>
            <small style="color:#94A3B8;">Existing monthly EMI / Gross salary</small>
        </div>
        <div class="ratio-card">
            <b>Total Expense Ratio:</b> <span style="float:right; font-weight:bold;">{exp_ratio_val * 100:.1f}%</span><br>
            <small style="color:#94A3B8;">Living expenses / Gross salary</small>
        </div>
        <div class="ratio-card">
            <b>Monthly Disposable Income:</b> <span style="float:right; font-weight:bold;">₹{disp_income_val:,.2f}</span><br>
            <small style="color:#94A3B8;">Salary - (Living Expenses + Existing EMI)</small>
        </div>
        <div class="ratio-card">
            <b>EMI Affordability Index:</b> <span style="float:right; font-weight:bold;">{affordability_val:.2f}x</span><br>
            <small style="color:#94A3B8;">Disposable Income / Requested Monthly EMI</small>
        </div>
        <div class="ratio-card">
            <b>Liquidity Reserve Ratio:</b> <span style="float:right; font-weight:bold;">{liquidity_val * 100:.1f}%</span><br>
            <small style="color:#94A3B8;">Total liquid reserves / Requested loan amount</small>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("👇 Enter or adjust the borrower parameters in Stage 1 above and click **'🚀 STAGE 2: Evaluate Borrower Risk & Max Safe EMI'** to generate underwriting results.")
