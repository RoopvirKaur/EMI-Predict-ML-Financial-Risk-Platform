import streamlit as st
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Ensure root project directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    RAW_DATASET_PATH,
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
    page_title="Data Management CRUD - EMIPredict AI",
    page_icon="⚙️",
    layout="wide"
)

st.markdown("""
<style>
    .crud-banner {
        background-color: #1E293B;
        border-left: 4px solid #38BDF8;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚙️ Applicant Data Management (Isolated Session CRUD)")
st.caption("Perform Create, Read, Update, and Delete operations on applicant records with live model inference.")

st.markdown("""
<div class="crud-banner">
    🔒 <b>Data Safety Isolation Notice:</b> All CRUD operations occur strictly inside application session memory.
    The primary training dataset (<code>dataset/emi_prediction_dataset.csv</code>) remains untouched and read-only.
</div>
""", unsafe_allow_html=True)

# Helper function to load model artifacts for auto-evaluation
@st.cache_resource
def get_inference_models():
    main_mod = sys.modules.get("__main__")
    if main_mod is not None:
        setattr(main_mod, "FinancialRatioTransformer", FinancialRatioTransformer)
        setattr(main_mod, "FullPreprocessingPipeline", FullPreprocessingPipeline)

    preprocessor = FullPreprocessingPipeline.load(PREPROCESSOR_PATH)
    classifier = joblib.load(BEST_CLASSIFIER_PATH)
    regressor = joblib.load(BEST_REGRESSOR_PATH)
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.pkl")
    return preprocessor, classifier, regressor, label_encoder

try:
    preprocessor, classifier, regressor, label_encoder = get_inference_models()
except Exception as e:
    st.error(f"Failed loading inference pipeline for CRUD auto-evaluation: {e}")
    st.stop()

def evaluate_record(record_dict: dict) -> dict:
    """Runs preprocessor, classifier, and regressor on a single record dictionary."""
    df_single = pd.DataFrame([record_dict])
    
    # Financial Ratios
    ratio_tf = FinancialRatioTransformer()
    df_ratios = ratio_tf.transform(df_single)
    
    input_cols = NUMERICAL_RAW_FEATURES + CATEGORICAL_NOMINAL_FEATURES + CATEGORICAL_ORDINAL_FEATURES + DERIVED_RATIO_FEATURES
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

# Initialize Session Data
if "crud_df" not in st.session_state:
    if RAW_DATASET_PATH.exists():
        raw_df = pd.read_csv(RAW_DATASET_PATH, low_memory=False).head(20)
        # Convert numeric columns stored as strings in raw CSV to proper numeric types for Arrow display compatibility
        non_numeric_cols = [
            "applicant_id", "gender", "marital_status", "education",
            "employment_type", "company_type", "house_type",
            "existing_loans", "emi_scenario", "emi_eligibility"
        ]
        for col in raw_df.columns:
            if col not in non_numeric_cols:
                raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce')
        # Ensure Applicant ID exists
        if "applicant_id" not in raw_df.columns:
            raw_df.insert(0, "applicant_id", [f"APP-{1000 + i}" for i in range(len(raw_df))])
        
        evaluated_list = []
        for _, row in raw_df.iterrows():
            row_dict = row.to_dict()
            eval_dict = evaluate_record(row_dict)
            eval_dict["applicant_id"] = str(row_dict["applicant_id"])
            evaluated_list.append(eval_dict)
        st.session_state.crud_df = pd.DataFrame(evaluated_list)
    else:
        # Fallback sample dataset if CSV missing
        sample_records = [
            {
                "applicant_id": "APP-1001", "age": 30, "gender": "Male", "marital_status": "Single",
                "education": "Graduate", "employment_type": "Salaried", "company_type": "MNC",
                "years_of_employment": 5.0, "house_type": "Rented", "family_size": 2, "dependents": 0,
                "monthly_salary": 75000.0, "monthly_rent": 15000.0, "school_fees": 0.0, "college_fees": 0.0,
                "travel_expenses": 3000.0, "groceries_utilities": 8000.0, "other_monthly_expenses": 4000.0,
                "current_emi_amount": 5000.0, "credit_score": 750, "bank_balance": 250000.0, "emergency_fund": 100000.0,
                "requested_amount": 300000.0, "requested_tenure": 36, "existing_loans": "Yes", "emi_scenario": "Scenario_1"
            }
        ]
        evaluated = [evaluate_record(r) for r in sample_records]
        st.session_state.crud_df = pd.DataFrame(evaluated)

df = st.session_state.crud_df

# Summary Metrics Bar
m1, m2, m3, m4 = st.columns(4)
m1.metric("Active Session Records", f"{len(df)} Records")

if "emi_eligibility" in df.columns:
    m2.metric("Eligible Count", f"{(df['emi_eligibility'] == 'Eligible').sum()}")
    m3.metric("High Risk Count", f"{(df['emi_eligibility'] == 'High_Risk').sum()}")
    m4.metric("Not Eligible Count", f"{(df['emi_eligibility'] == 'Not_Eligible').sum()}")
else:
    m2.metric("Eligible Count", "N/A")
    m3.metric("High Risk Count", "N/A")
    m4.metric("Not Eligible Count", "N/A")

st.markdown("<br>", unsafe_allow_html=True)

crud_tab1, crud_tab2, crud_tab3, crud_tab4, crud_tab5 = st.tabs([
    "📖 Read Records",
    "➕ Create New Applicant",
    "✏️ Update Record",
    "🗑️ Delete Record",
    "📥 Reset & Export Data"
])

# Tab 1: READ
with crud_tab1:
    st.markdown('<div id="read-records-section"></div>', unsafe_allow_html=True)
    st.subheader("📖 Applicant Record Inspector")
    
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        search_id = st.text_input("🔍 Search by Applicant ID", value="")
    with col_f2:
        filter_status = st.multiselect("Filter by Eligibility Status", options=df["emi_eligibility"].unique() if "emi_eligibility" in df.columns else [], default=[])
    with col_f3:
        filter_emp = st.multiselect("Filter by Employment Type", options=df["employment_type"].unique() if "employment_type" in df.columns else [], default=[])

    filtered_df = df.copy()
    if search_id:
        filtered_df = filtered_df[filtered_df["applicant_id"].str.contains(search_id, case=False, na=False)]
    if filter_status:
        filtered_df = filtered_df[filtered_df["emi_eligibility"].isin(filter_status)]
    if filter_emp:
        filtered_df = filtered_df[filtered_df["employment_type"].isin(filter_emp)]

    st.dataframe(filtered_df, width="stretch", height=400)

# Tab 2: CREATE
with crud_tab2:
    st.subheader("➕ Create New Applicant Record")
    if "create_notice" in st.session_state and st.session_state.create_notice:
        st.success(st.session_state.create_notice)
    st.caption("Submitting this form auto-computes financial ratios and runs real-time ML risk predictions.")

    with st.form("create_applicant_form"):
        c1, c2, c3 = st.columns(3)
        
        with c1:
            default_id = f"APP-{1000 + len(df)}"
            new_id = st.text_input("Applicant ID", value=default_id, help="Enter a unique Applicant ID (e.g. TEST-1001)")
            new_age = st.number_input("Age", 18, 75, 30)
            new_gender = st.selectbox("Gender", ["Male", "Female"])
            new_marital = st.selectbox("Marital Status", ["Single", "Married"])
            new_edu = st.selectbox("Education", EDUCATION_ORDER)
            new_emp = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Freelancer", "Unemployed"])
            new_comp = st.selectbox("Company Type", ["MNC", "Private", "Government", "Startup", "Other"])
            new_yoe = st.number_input("Years of Employment", 0.0, 50.0, 4.0)

        with c2:
            new_salary = st.number_input("Monthly Salary (INR ₹)", min_value=0.0, max_value=2000000.0, value=65000.0, step=1000.0)
            new_cibil = st.slider("Credit Score", 300, 850, 720)
            new_bank = st.number_input("Bank Balance (INR ₹)", 0.0, 10000000.0, 150000.0, 5000.0)
            new_emergency = st.number_input("Emergency Fund (INR ₹)", 0.0, 5000000.0, 50000.0, 5000.0)
            new_exist_loans = st.selectbox("Existing Loans", ["Yes", "No"])
            new_curr_emi = st.number_input("Current EMI Amount (INR ₹)", 0.0, 500000.0, 6000.0, 500.0)
            new_rent = st.number_input("Monthly Rent (INR ₹)", 0.0, 200000.0, 12000.0, 500.0)

        with c3:
            new_school = st.number_input("School Fees (INR ₹)", 0.0, 100000.0, 2000.0, 500.0)
            new_college = st.number_input("College Fees (INR ₹)", 0.0, 100000.0, 0.0, 500.0)
            new_travel = st.number_input("Travel Expenses (INR ₹)", 0.0, 100000.0, 3000.0, 500.0)
            new_groc = st.number_input("Groceries & Utilities (INR ₹)", 0.0, 100000.0, 8000.0, 500.0)
            new_other = st.number_input("Other Expenses (INR ₹)", 0.0, 100000.0, 4000.0, 500.0)
            new_req_amt = st.number_input("Requested Amount (INR ₹)", 5000.0, 10000000.0, 200000.0, 10000.0)
            new_req_ten = st.number_input("Requested Tenure (Months)", 1, 360, 24)
            new_scen = st.selectbox("EMI Scenario", ["Scenario_1", "Scenario_2", "Scenario_3", "Scenario_4", "Scenario_5"])
            new_house = st.selectbox("House Type", ["Rented", "Owned", "Parental"])
            new_fam = st.number_input("Family Size", 1, 10, 3)
            new_dep = st.number_input("Dependents", 0, 8, 1)

        create_btn = st.form_submit_button("➕ Add Applicant to Session Table", width="stretch")

    if create_btn:
        clean_id = new_id.strip() if new_id else ""
        if not clean_id:
            st.error("⚠️ Applicant ID cannot be empty.")
        elif clean_id in df["applicant_id"].astype(str).values:
            st.error(f"⚠️ Applicant ID '{clean_id}' already exists. Please enter a unique Applicant ID.")
        elif new_salary < 0:
            st.error("⚠️ Monthly salary cannot be negative.")
        elif new_emp != "Unemployed" and new_salary <= 0:
            st.error("⚠️ Monthly salary must be greater than ₹0 for employed applicants.")
        else:
            rec = {
                "applicant_id": clean_id, "age": new_age, "gender": new_gender, "marital_status": new_marital,
                "education": new_edu, "employment_type": new_emp, "company_type": new_comp, "years_of_employment": new_yoe,
                "house_type": new_house, "family_size": new_fam, "dependents": new_dep, "monthly_salary": float(new_salary),
                "monthly_rent": float(new_rent), "school_fees": float(new_school), "college_fees": float(new_college),
                "travel_expenses": float(new_travel), "groceries_utilities": float(new_groc),
                "other_monthly_expenses": float(new_other), "current_emi_amount": float(new_curr_emi),
                "credit_score": int(new_cibil), "bank_balance": float(new_bank), "emergency_fund": float(new_emergency),
                "requested_amount": float(new_req_amt), "requested_tenure": int(new_req_ten),
                "existing_loans": new_exist_loans, "emi_scenario": new_scen
            }
            
            evaluated_rec = evaluate_record(rec)
            evaluated_rec["applicant_id"] = clean_id
            new_row_df = pd.DataFrame([evaluated_rec])
            cols = [c for c in st.session_state.crud_df.columns if c in new_row_df.columns] + [c for c in new_row_df.columns if c not in st.session_state.crud_df.columns]
            new_row_df = new_row_df[cols]
            st.session_state.crud_df = pd.concat([new_row_df, st.session_state.crud_df], ignore_index=True)
            st.session_state.create_notice = f"Applicant {clean_id} created successfully."
            st.session_state.pop("update_notice", None)
            st.session_state.pop("delete_notice", None)
            st.session_state.scroll_to_read = True
            st.rerun()

# Tab 3: UPDATE
with crud_tab3:
    st.subheader("✏️ Update Applicant Record")
    if "update_notice" in st.session_state and st.session_state.update_notice:
        st.success(st.session_state.update_notice)
        
    if len(df) == 0:
        st.info("No records available to update.")
    else:
        select_id = st.selectbox("Select Applicant ID to Update", options=df["applicant_id"].tolist())
        target_row = df[df["applicant_id"] == select_id].iloc[0].to_dict()

        with st.form("update_applicant_form"):
            u_col1, u_col2 = st.columns(2)
            with u_col1:
                u_salary = st.number_input("Monthly Salary (INR ₹)", min_value=0.0, max_value=2000000.0, value=float(target_row.get("monthly_salary", 50000.0)), step=1000.0)
                u_cibil = st.slider("Credit Score", 300, 850, int(target_row.get("credit_score", 700)))
                u_curr_emi = st.number_input("Current EMI Amount (INR ₹)", 0.0, 500000.0, float(target_row.get("current_emi_amount", 5000.0)), 500.0)
                u_bank = st.number_input("Bank Balance (INR ₹)", 0.0, 10000000.0, float(target_row.get("bank_balance", 100000.0)), 5000.0)

            with u_col2:
                u_req_amt = st.number_input("Requested Loan Amount (INR ₹)", 5000.0, 10000000.0, float(target_row.get("requested_amount", 200000.0)), 10000.0)
                u_req_ten = st.number_input("Requested Tenure (Months)", 1, 360, int(target_row.get("requested_tenure", 24)))
                u_rent = st.number_input("Monthly Rent (INR ₹)", 0.0, 200000.0, float(target_row.get("monthly_rent", 10000.0)), 500.0)
                emp_options = ["Salaried", "Self-Employed", "Freelancer", "Unemployed"]
                curr_emp = target_row.get("employment_type", "Salaried")
                emp_idx = emp_options.index(curr_emp) if curr_emp in emp_options else 0
                u_emp = st.selectbox("Employment Type", emp_options, index=emp_idx)

            update_btn = st.form_submit_button("✏️ Save & Re-Evaluate Prediction", width="stretch")

        if update_btn:
            if u_salary < 0:
                st.error("⚠️ Monthly salary cannot be negative.")
            elif u_emp != "Unemployed" and u_salary <= 0:
                st.error("⚠️ Monthly salary must be greater than ₹0 for employed applicants.")
            else:
                updated_dict = target_row.copy()
                updated_dict["applicant_id"] = select_id
                updated_dict["monthly_salary"] = u_salary
                updated_dict["credit_score"] = u_cibil
                updated_dict["current_emi_amount"] = u_curr_emi
                updated_dict["bank_balance"] = u_bank
                updated_dict["requested_amount"] = u_req_amt
                updated_dict["requested_tenure"] = u_req_ten
                updated_dict["monthly_rent"] = u_rent
                updated_dict["employment_type"] = u_emp

                re_evaluated = evaluate_record(updated_dict)
                re_evaluated["applicant_id"] = select_id
                
                # Replace row safely using column alignment
                idx = df[df["applicant_id"] == select_id].index[0]
                re_eval_series = pd.Series(re_evaluated)[st.session_state.crud_df.columns]
                st.session_state.crud_df.iloc[idx] = re_eval_series
                st.session_state.update_notice = f"Applicant {select_id} updated successfully."
                st.session_state.pop("create_notice", None)
                st.session_state.pop("delete_notice", None)
                st.session_state.scroll_to_read = True
                st.rerun()

# Tab 4: DELETE
with crud_tab4:
    st.subheader("🗑️ Delete Applicant Record")
    if "delete_notice" in st.session_state and st.session_state.delete_notice:
        st.success(st.session_state.delete_notice)
        
    if len(df) == 0:
        st.info("No records available to delete.")
    else:
        del_id = st.selectbox("Select Applicant ID to Delete", options=df["applicant_id"].tolist(), key="del_select")
        confirm_del = st.checkbox(f"I confirm deletion of applicant record {del_id}")
        
        if st.button("🔴 Permanently Remove Record from Session", disabled=not confirm_del):
            st.session_state.crud_df = st.session_state.crud_df[st.session_state.crud_df["applicant_id"] != del_id]
            st.session_state.delete_notice = f"Applicant {del_id} deleted successfully."
            st.session_state.pop("create_notice", None)
            st.session_state.pop("update_notice", None)
            st.session_state.scroll_to_read = True
            st.rerun()

# Tab 5: RESET & EXPORT
with crud_tab5:
    st.subheader("📥 Data Export & Reset Controls")
    
    col_e1, col_e2 = st.columns(2)
    
    with col_e1:
        st.markdown("##### 📥 Export Current Session Data")
        csv_data = st.session_state.crud_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Download Session Dataset (CSV)",
            data=csv_data,
            file_name="session_applicant_crud_records.csv",
            mime="text/csv",
            width="stretch"
        )

    with col_e2:
        st.markdown("##### 🔄 Reset Session Table")
        if st.button("🔄 Reload Default Sample Dataset", width="stretch"):
            del st.session_state.crud_df
            st.success("Session state reset to original sample dataset.")
            st.rerun()

if st.session_state.get("scroll_to_read", False):
    st.session_state.scroll_to_read = False
    import streamlit.components.v1 as components
    components.html(
        """
        <script>
            setTimeout(function() {
                const tabs = window.parent.document.querySelectorAll('button[role="tab"], [data-baseweb="tab"], [data-testid="stTab"]');
                if (tabs && tabs.length > 0) {
                    tabs[0].click();
                }
                const anchor = window.parent.document.getElementById('read-records-section');
                if (anchor) {
                    anchor.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            }, 100);
        </script>
        """,
        height=0,
        width=0
    )

