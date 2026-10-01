import streamlit as st
import sys
from pathlib import Path

# Ensure root project directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    MLFLOW_EXPERIMENT_NAME,
    CLASSIFICATION_TARGET,
    REGRESSION_TARGET,
    ELIGIBILITY_CLASSES,
    DERIVED_RATIO_FEATURES
)

# Page Configuration
st.set_page_config(
    page_title="EMIPredict AI - Financial Risk Platform",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Custom CSS Styling with Light/Dark Theme Support & Equal Height Cards
st.markdown("""
<style>
    /* CSS Variables for Theme Awareness (Default: Dark Theme) */
    :root {
        --hero-bg: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        --hero-border: #334155;
        --hero-title-grad: linear-gradient(90deg, #38BDF8 0%, #818CF8 100%);
        --hero-subtitle: #94A3B8;
        
        --card-bg: #1E293B;
        --card-border: #334155;
        --card-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        
        --metric-title-color: #94A3B8;
        --metric-value-color: #F8FAFC;
        --metric-desc-color: #64748B;
        
        --badge-bg: #0F172A;
        --badge-border: #334155;
        --badge-text: #38BDF8;
        
        --nav-title-color: #F8FAFC;
        --nav-desc-color: #94A3B8;
    }

    /* Light Theme Overrides (System Preference & Streamlit Light Theme) */
    @media (prefers-color-scheme: light) {
        :root {
            --hero-bg: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%);
            --hero-border: #CBD5E1;
            --hero-title-grad: linear-gradient(90deg, #0284C7 0%, #4F46E5 100%);
            --hero-subtitle: #475569;
            
            --card-bg: #FFFFFF;
            --card-border: #E2E8F0;
            --card-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            
            --metric-title-color: #64748B;
            --metric-value-color: #0F172A;
            --metric-desc-color: #475569;
            
            --badge-bg: #F1F5F9;
            --badge-border: #CBD5E1;
            --badge-text: #0284C7;
            
            --nav-title-color: #0F172A;
            --nav-desc-color: #475569;
        }
    }

    /* Streamlit Light Theme Active Selector */
    [data-theme="light"],
    .stApp[data-theme="light"],
    html[data-theme="light"],
    body[data-theme="light"],
    [data-testid="stAppViewContainer"][data-theme="light"] {
        --hero-bg: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%) !important;
        --hero-border: #CBD5E1 !important;
        --hero-title-grad: linear-gradient(90deg, #0284C7 0%, #4F46E5 100%) !important;
        --hero-subtitle: #475569 !important;
        
        --card-bg: #FFFFFF !important;
        --card-border: #E2E8F0 !important;
        --card-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important;
        
        --metric-title-color: #64748B !important;
        --metric-value-color: #0F172A !important;
        --metric-desc-color: #475569 !important;
        
        --badge-bg: #F1F5F9 !important;
        --badge-border: #CBD5E1 !important;
        --badge-text: #0284C7 !important;
        
        --nav-title-color: #0F172A !important;
        --nav-desc-color: #475569 !important;
    }

    /* Streamlit Dark Theme Active Selector */
    [data-theme="dark"],
    .stApp[data-theme="dark"],
    html[data-theme="dark"],
    body[data-theme="dark"],
    [data-testid="stAppViewContainer"][data-theme="dark"] {
        --hero-bg: linear-gradient(135deg, #1E293B 0%, #0F172A 100%) !important;
        --hero-border: #334155 !important;
        --hero-title-grad: linear-gradient(90deg, #38BDF8 0%, #818CF8 100%) !important;
        --hero-subtitle: #94A3B8 !important;
        
        --card-bg: #1E293B !important;
        --card-border: #334155 !important;
        --card-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2) !important;
        
        --metric-title-color: #94A3B8 !important;
        --metric-value-color: #F8FAFC !important;
        --metric-desc-color: #64748B !important;
        
        --badge-bg: #0F172A !important;
        --badge-border: #334155 !important;
        --badge-text: #38BDF8 !important;
        
        --nav-title-color: #F8FAFC !important;
        --nav-desc-color: #94A3B8 !important;
    }

    /* Global Container Adjustments */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Column Flex Alignment & Equal Height Layout */
    [data-testid="stHorizontalBlock"] {
        align-items: stretch !important;
    }
    [data-testid="stColumn"] {
        display: flex !important;
        flex-direction: column !important;
        flex: 1 1 0px !important;
    }
    [data-testid="stColumn"] > div,
    [data-testid="stColumn"] [data-testid="stVerticalBlockBorderWrapper"],
    [data-testid="stColumn"] [data-testid="stVerticalBlock"],
    [data-testid="stColumn"] [data-testid="stElementContainer"],
    [data-testid="stColumn"] [data-testid="stMarkdown"],
    [data-testid="stColumn"] [data-testid="stMarkdownContainer"],
    [data-testid="stColumn"] [data-testid="stMarkdownContainer"] > p,
    [data-testid="stColumn"] [data-testid="stMarkdownContainer"] > div {
        height: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        flex: 1 1 auto !important;
        margin: 0 !important;
    }

    /* Hero Header Banner */
    .hero-card {
        background: var(--hero-bg);
        border: 1px solid var(--hero-border);
        border-radius: 16px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        box-shadow: var(--card-shadow);
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: var(--hero-title-grad);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        color: var(--hero-subtitle);
        font-size: 1.15rem;
        font-weight: 400;
        line-height: 1.6;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: var(--card-bg);
        border: 1px solid var(--card-border);
        border-left-width: 4px;
        border-left-style: solid;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin: 0 !important;
        box-shadow: var(--card-shadow);
        height: 100% !important;
        min-height: 100% !important;
        box-sizing: border-box !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        flex: 1 1 auto !important;
    }
    .metric-title {
        color: var(--metric-title-color);
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .metric-value {
        color: var(--metric-value-color);
        font-size: 1.45rem;
        font-weight: 700;
        margin-top: 0.25rem;
        line-height: 1.3;
        word-break: break-word;
    }
    .metric-desc {
        color: var(--metric-desc-color);
        font-size: 0.8rem;
        margin-top: 0.2rem;
    }

    /* Feature Badge */
    .ratio-badge {
        display: inline-block;
        background: var(--badge-bg);
        border: 1px solid var(--badge-border);
        color: var(--badge-text);
        padding: 0.35rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
    }

    /* Navigation Quick Cards */
    .nav-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0 !important;
        height: 100% !important;
        min-height: 100% !important;
        box-sizing: border-box !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: flex-start !important;
        flex: 1 1 auto !important;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .nav-card:hover {
        border-color: #38BDF8;
        transform: translateY(-2px);
    }
    .nav-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--nav-title-color);
        margin-bottom: 0.5rem;
    }
    .nav-desc {
        color: var(--nav-desc-color);
        font-size: 0.9rem;
        line-height: 1.5;
        flex: 1 1 auto !important;
    }
</style>
""", unsafe_allow_html=True)

# Hero Header
st.markdown("""
<div class="hero-card">
    <div class="hero-title">EMIPredict AI</div>
    <div class="hero-subtitle">
        Multi-Task Financial Risk Intelligence & Safe Monthly EMI Estimation Platform.<br>
        Powered by XGBoost & Scikit-Learn pipelines trained on 100,000 borrower financial profiles.
    </div>
</div>
""", unsafe_allow_html=True)

# High-Level Summary Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card" style="border-left-color: #38BDF8;">
        <div class="metric-title">Dataset Size</div>
        <div class="metric-value">100,000 Records</div>
        <div class="metric-desc">80k Train / 10k Val / 10k Test</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card" style="border-left-color: #818CF8;">
        <div class="metric-title">Classifier Model</div>
        <div class="metric-value">XGBoost (Tier Risk)</div>
        <div class="metric-desc">Eligible | High Risk | Not Eligible</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card" style="border-left-color: #34D399;">
        <div class="metric-title">Regressor Model</div>
        <div class="metric-value">XGBoost (Max EMI)</div>
        <div class="metric-desc">Predicts Safe Monthly EMI Cap (₹)</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card" style="border-left-color: #F43F5E;">
        <div class="metric-title">MLflow Tracking</div>
        <div class="metric-value">Active Suite</div>
        <div class="metric-desc">SQLite: EMI_Predict_Risk_Assessment</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Platform Purpose & Architecture Section
st.subheader("🎯 Purpose & Intelligence Architecture")

col_a, col_b = st.columns([1.2, 1])

with col_a:
    st.markdown("""
    **EMIPredict AI** helps financial institutions, retail loan officers, and borrowers make **data-driven credit underwriting decisions**.
    
    The system simultaneously solves two core financial modeling tasks:
    1. **Multi-Class Risk Tier Classification**: Predicts whether a borrower is **Eligible**, **High Risk**, or **Not Eligible** based on complete cashflow analysis.
    2. **Max Safe EMI Capacity Regression**: Predicts the maximum sustainable monthly installment amount (INR ₹) without causing debt distress or default risk.

    #### 🧮 Derived Financial Ratio Engine
    The underlying preprocessing pipeline calculates 5 key financial safety metrics without division-by-zero risk:
    """)
    
    st.markdown("""
    <span class="ratio-badge">1. Debt-to-Income (DTI)</span>
    <span class="ratio-badge">2. Total Expense Ratio</span>
    <span class="ratio-badge">3. Disposable Income (INR ₹)</span>
    <span class="ratio-badge">4. Affordability Index</span>
    <span class="ratio-badge">5. Liquidity Reserve Ratio</span>
    """, unsafe_allow_html=True)

with col_b:
    with st.expander("📌 Key Feature Breakdown", expanded=True):
        st.markdown("""
        - **Demographics & Profile**: Age, Gender, Marital Status, Education, Family Size, Dependents.
        - **Employment & Income**: Employment Type, Company Type, Years of Employment, Monthly Salary.
        - **Monthly Living Obligations**: Rent, School/College Fees, Travel, Groceries/Utilities, Other Expenses.
        - **Credit & Wealth Reserves**: Bank Balance, Emergency Fund, Credit Score (300-850), Existing EMI Amount.
        - **Loan Product Scenario**: Requested Amount, Tenure, EMI Product Category.
        """)

st.markdown("---")

# Quick Navigation Section
st.subheader("🗺️ Platform Navigation")

n1, n2, n3, n4 = st.columns(4)

with n1:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-title">📊 1. Overview & EDA</div>
        <div class="nav-desc">
            Explore interactive Plotly visualizations of target distributions, loan product scenarios, affordability scatter plots, and correlation heatmaps.
        </div>
    </div>
    """, unsafe_allow_html=True)

with n2:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-title">🔮 2. Risk Predictor</div>
        <div class="nav-desc">
            Input borrower parameters in the real-time form to evaluate risk eligibility tier, maximum safe EMI capacity, and financial safety warnings.
        </div>
    </div>
    """, unsafe_allow_html=True)

with n3:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-title">🧪 3. MLflow Dashboard</div>
        <div class="nav-desc">
            Inspect model comparison leaderboards, validation & test set performance metrics, confusion matrices, and MLflow experiment runs.
        </div>
    </div>
    """, unsafe_allow_html=True)

with n4:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-title">⚙️ 4. Data Management</div>
        <div class="nav-desc">
            Manages applicant records efficiently and securely using an isolated interactive session table supporting Create, Read, Update, Delete, and CSV export.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("EMIPredict AI Risk Assessment Engine | Built with Streamlit, Plotly, Scikit-Learn & MLflow")
