import streamlit as st
import sys
import pandas as pd
from pathlib import Path

# Ensure root project directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.preprocessing.eda import (
    load_processed_train_data,
    plot_target_distribution,
    plot_scenario_distribution,
    plot_salary_and_credit_distribution,
    plot_credit_score_distribution,
    plot_existing_emi_and_dti,
    plot_disposable_income_distribution,
    plot_affordability_and_liquidity,
    plot_correlation_matrix,
    plot_scenario_comparisons,
    plot_outlier_analysis,
    generate_eda_report
)

st.set_page_config(
    page_title="Overview & EDA - EMIPredict AI",
    page_icon="📊",
    layout="wide"
)

# Page Styling
st.markdown("""
<style>
    /* Default / Dark Theme Styles (Restored & Unchanged for Dark Mode) */
    .section-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .insight-title {
        color: #38BDF8;
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 0.5rem;
    }
    .insight-text {
        color: #CBD5E1;
        font-size: 0.92rem;
        line-height: 1.5;
    }

    /* Light Mode Readability Fixes (Targeting Streamlit Light Theme Active State Only) */
    [data-theme="light"] .section-card,
    .stApp[data-theme="light"] .section-card,
    html[data-theme="light"] .section-card,
    body[data-theme="light"] .section-card,
    [data-testid="stAppViewContainer"][data-theme="light"] .section-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    [data-theme="light"] .insight-title,
    .stApp[data-theme="light"] .insight-title,
    html[data-theme="light"] .insight-title,
    body[data-theme="light"] .insight-title,
    [data-testid="stAppViewContainer"][data-theme="light"] .insight-title {
        color: #0284C7;
    }

    [data-theme="light"] .insight-text,
    .stApp[data-theme="light"] .insight-text,
    html[data-theme="light"] .insight-text,
    body[data-theme="light"] .insight-text,
    [data-testid="stAppViewContainer"][data-theme="light"] .insight-text {
        color: #334155;
    }

    [data-theme="light"] .insight-text code,
    .stApp[data-theme="light"] .insight-text code,
    html[data-theme="light"] .insight-text code,
    body[data-theme="light"] .insight-text code,
    [data-testid="stAppViewContainer"][data-theme="light"] .insight-text code {
        background-color: #F1F5F9;
        color: #0284C7;
        border: 1px solid #CBD5E1;
    }

    [data-theme="light"] [data-testid="stCaptionContainer"],
    [data-theme="light"] .stCaption,
    .stApp[data-theme="light"] [data-testid="stCaptionContainer"],
    .stApp[data-theme="light"] .stCaption,
    html[data-theme="light"] [data-testid="stCaptionContainer"],
    html[data-theme="light"] .stCaption,
    body[data-theme="light"] [data-testid="stCaptionContainer"],
    body[data-theme="light"] .stCaption {
        color: #475569 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Dataset Overview & Exploratory Data Analysis")
st.caption("Interactive analysis of training data (80,000 borrowers) reusing Phase 4 Plotly analytics functions.")

# Cache dataset loading
@st.cache_data
def get_train_data():
    return load_processed_train_data()

try:
    df_train = get_train_data()
    eda_report = generate_eda_report(df_train)
except Exception as e:
    st.error(f"Error loading training data for EDA: {e}")
    st.stop()

# Key Data Summary Metrics
m1, m2, m3, m4 = st.columns(4)

eligible_pct = eda_report["target_distribution"].get("Eligible", {}).get("pct", 0)
high_risk_pct = eda_report["target_distribution"].get("High_Risk", {}).get("pct", 0)
not_eligible_pct = eda_report["target_distribution"].get("Not_Eligible", {}).get("pct", 0)
mean_salary = df_train["monthly_salary"].mean()

m1.metric("Total EDA Sample Count", f"{len(df_train):,} Records")
m2.metric("Eligible Applicants", f"{eligible_pct}%", delta="Low Risk", delta_color="normal")
m3.metric("High Risk Applicants", f"{high_risk_pct}%", delta="Cautionary", delta_color="off")
m4.metric("Not Eligible Applicants", f"{not_eligible_pct}%", delta="Declined", delta_color="inverse")

st.markdown("<br>", unsafe_allow_html=True)

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Risk & Product Distribution",
    "💰 Income & Debt Profile",
    "⚖️ Affordability & Liquidity Ratios",
    "📋 Business Insights & Summary"
])

# Tab 1: Risk & Product Distribution
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        fig_target = plot_target_distribution(df_train)
        st.plotly_chart(fig_target, width="stretch")
        st.markdown("""
        <div class="insight-title">💡 Risk Distribution Insight</div>
        <div class="insight-text">
            The target variable <code>emi_eligibility</code> is divided into three tiers: Eligible (~55%), High_Risk (~25%), and Not_Eligible (~20%).
            The classification pipeline accounts for class imbalance using cost-sensitive sample weighting.
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        fig_scenario = plot_scenario_distribution(df_train)
        st.plotly_chart(fig_scenario, width="stretch")
        st.markdown("""
        <div class="insight-title">💡 Product Category Risk Breakdown</div>
        <div class="insight-text">
            Compounding EMI product categories show distinct risk profiles. Micro-loans and high-tenure personal loans exhibit higher proportions of High_Risk applicants compared to short-term retail consumer EMIs.
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Tab 2: Income & Debt Profile
with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        fig_salary = plot_salary_and_credit_distribution(df_train)
        st.plotly_chart(fig_salary, width="stretch")
        st.markdown("""
        <div class="insight-title">💡 Salary Threshold Analysis</div>
        <div class="insight-text">
            Eligible applicants consistently exhibit higher median monthly salaries (₹45,000+), while Not_Eligible applicants suffer from constrained gross income relative to fixed living obligations.
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        fig_credit = plot_credit_score_distribution(df_train)
        st.plotly_chart(fig_credit, width="stretch")
        st.markdown("""
        <div class="insight-title">💡 Credit Score Thresholds</div>
        <div class="insight-text">
            Credit scores below 600 strongly align with Not_Eligible classifications. Scores between 600–700 frequently trigger High_Risk classification requiring reduced loan principal or higher liquidity reserves.
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        fig_dti = plot_existing_emi_and_dti(df_train)
        st.plotly_chart(fig_dti, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        fig_disp = plot_disposable_income_distribution(df_train)
        st.plotly_chart(fig_disp, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

# Tab 3: Affordability & Liquidity Ratios
with tab3:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    fig_afford = plot_affordability_and_liquidity(df_train)
    st.plotly_chart(fig_afford, width="stretch")
    st.markdown("""
    <div class="insight-title">💡 Affordability Index vs Liquidity Reserve Ratio</div>
    <div class="insight-text">
        Scatter plot mapping monthly cashflow coverage against liquid asset cushion. Applicants in the upper-right quadrant (Affordability Index > 1.5 & Liquidity Reserve > 0.5) display strong repayment capacity.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        fig_corr = plot_correlation_matrix(df_train)
        st.plotly_chart(fig_corr, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        fig_scen_comp = plot_scenario_comparisons(df_train)
        st.plotly_chart(fig_scen_comp, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    fig_outlier = plot_outlier_analysis(df_train)
    st.plotly_chart(fig_outlier, width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)

# Tab 4: Business Insights & Summary
with tab4:
    st.subheader("📌 Key Statistical Findings & Credit Underwriting Guidance")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("""
        ### 1. Primary Risk Determinants
        - **Disposable Income & DTI**: The strongest predictors of default risk are Debt-to-Income (DTI > 40%) and negative disposable income after living obligations.
        - **Liquidity Buffer**: Emergency fund and bank balance reserves provide critical protection against income shocks. Applicants with liquidity reserves < 10% of requested principal present high vulnerability.
        - **Existing Obligations**: High existing monthly EMI commitments significantly diminish safe new EMI capacity regardless of gross salary.
        """)
        
    with col_b:
        st.markdown("""
        ### 2. Loan Structuring Recommendations
        - **High_Risk Applicants**: Require either an extended loan tenure to lower requested monthly EMI below the predicted max safe EMI limit, or a co-applicant guarantee.
        - **Not_Eligible Applicants**: Primary rejection triggers include severe DTI overload (>50%), credit score < 580, or disposable income deficit.
        - **Safe EMI Estimation**: The XGBoost Regression model reliably limits EMI caps to safe disposable income headroom.
        """)

    st.markdown("### 📊 Mean Financial Ratios by Risk Tier")
    ratios_summary = pd.DataFrame(eda_report["ratio_by_class"]).T
    ratios_summary.columns = [c.replace("_", " ").title() for c in ratios_summary.columns]
    st.dataframe(ratios_summary.style.format("{:.4f}"), width="stretch")
