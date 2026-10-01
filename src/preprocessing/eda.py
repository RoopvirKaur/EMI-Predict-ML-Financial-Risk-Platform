import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, Any, List

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import TRAIN_DATASET_PATH, DERIVED_RATIO_FEATURES, CLASSIFICATION_TARGET, REGRESSION_TARGET

COLOR_MAP = {
    "Eligible": "#2ECC71",
    "High_Risk": "#F1C40F",
    "Not_Eligible": "#E74C3C"
}

def load_processed_train_data(parquet_path: Path = TRAIN_DATASET_PATH) -> pd.DataFrame:
    """Loads feature-engineered training split Parquet file."""
    if not parquet_path.exists():
        raise FileNotFoundError(f"Training Parquet file not found at {parquet_path}")
    return pd.read_parquet(parquet_path)

def plot_target_distribution(df: pd.DataFrame) -> go.Figure:
    """
    1. Target/Class Distribution Visualization
    Business Insight: Reveals overall loan approval rates and quantifies class imbalance.
    """
    counts = df[CLASSIFICATION_TARGET].value_counts().reset_index()
    counts.columns = ["Eligibility", "Count"]
    counts["Percentage"] = (counts["Count"] / len(df) * 100).round(2)

    fig = px.bar(
        counts,
        x="Eligibility",
        y="Count",
        color="Eligibility",
        color_discrete_map=COLOR_MAP,
        text=counts.apply(lambda r: f"{r['Count']:,}<br>({r['Percentage']}%)", axis=1),
        title="<b>1. Classification Target Distribution (emi_eligibility)</b>"
    )
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Eligibility Tier",
        yaxis_title="Record Count",
        showlegend=False
    )
    return fig

def plot_scenario_distribution(df: pd.DataFrame) -> go.Figure:
    """
    2. EMI Scenario Distribution & Risk Breakdown
    Business Insight: Compares risk profiles across 5 distinct lending product categories.
    """
    grouped = df.groupby(["emi_scenario", CLASSIFICATION_TARGET], observed=True).size().reset_index(name="Count")
    
    fig = px.bar(
        grouped,
        x="emi_scenario",
        y="Count",
        color=CLASSIFICATION_TARGET,
        barmode="group",
        color_discrete_map=COLOR_MAP,
        title="<b>2. Eligibility Risk Breakdown Across 5 EMI Scenarios</b>"
    )
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="EMI Scenario Category",
        yaxis_title="Number of Applicants"
    )
    return fig

def plot_salary_and_credit_distribution(df: pd.DataFrame) -> go.Figure:
    """
    3. Monthly Salary & Credit Score Distributions
    Business Insight: Demonstrates income thresholds and credit score cutoffs for loan qualification.
    """
    sample_df = df.sample(n=min(5000, len(df)), random_state=42)
    
    fig = px.box(
        sample_df,
        x=CLASSIFICATION_TARGET,
        y="monthly_salary",
        color=CLASSIFICATION_TARGET,
        color_discrete_map=COLOR_MAP,
        points="outliers",
        title="<b>3. Monthly Salary Distribution by Eligibility Tier</b>"
    )
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Eligibility Tier",
        yaxis_title="Monthly Salary (INR ₹)",
        showlegend=False
    )
    return fig

def plot_credit_score_distribution(df: pd.DataFrame) -> go.Figure:
    """
    4. Credit Score Histogram / Box Plot
    Business Insight: Shows how creditworthiness score dictates approval versus high-risk pricing.
    """
    fig = px.histogram(
        df,
        x="credit_score",
        color=CLASSIFICATION_TARGET,
        barmode="overlay",
        color_discrete_map=COLOR_MAP,
        nbins=40,
        title="<b>4. Credit Score Distribution Across Risk Tiers</b>"
    )
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Credit Score (300 - 850)",
        yaxis_title="Applicant Count"
    )
    return fig

def plot_existing_emi_and_dti(df: pd.DataFrame) -> go.Figure:
    """
    5. Existing EMI & Debt-to-Income (DTI) Ratio Distribution
    Business Insight: Measures how existing debt burdens constrain new loan capacity.
    """
    sample_df = df.sample(n=min(5000, len(df)), random_state=42)
    fig = px.box(
        sample_df,
        x=CLASSIFICATION_TARGET,
        y="dti_ratio",
        color=CLASSIFICATION_TARGET,
        color_discrete_map=COLOR_MAP,
        title="<b>5. Debt-to-Income (DTI) Ratio Distribution by Eligibility Tier</b>"
    )
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Eligibility Tier",
        yaxis_title="DTI Ratio (Existing Debt / Salary)",
        showlegend=False
    )
    return fig

def plot_disposable_income_distribution(df: pd.DataFrame) -> go.Figure:
    """
    6. Disposable Income Distribution
    Business Insight: Identifies net cashflow surpluses versus deficits across applicants.
    """
    sample_df = df.sample(n=min(5000, len(df)), random_state=42)
    fig = px.violin(
        sample_df,
        y="disposable_income",
        x=CLASSIFICATION_TARGET,
        color=CLASSIFICATION_TARGET,
        color_discrete_map=COLOR_MAP,
        box=True,
        points=False,
        title="<b>6. Disposable Income Distribution by Eligibility Tier</b>"
    )
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Eligibility Tier",
        yaxis_title="Disposable Income (INR ₹)"
    )
    return fig

def plot_affordability_and_liquidity(df: pd.DataFrame) -> go.Figure:
    """
    7. EMI Affordability Index vs Liquidity Reserve Ratio
    Business Insight: Correlates monthly cashflow coverage with liquid asset safety nets.
    """
    sample_df = df.sample(n=min(3000, len(df)), random_state=42)
    fig = px.scatter(
        sample_df,
        x="affordability_index",
        y="liquidity_reserve_ratio",
        color=CLASSIFICATION_TARGET,
        color_discrete_map=COLOR_MAP,
        hover_data=["monthly_salary", "requested_amount", "credit_score"],
        title="<b>7. EMI Affordability Index vs. Liquidity Reserve Ratio</b>"
    )
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Affordability Index (Disposable Income / Monthly EMI)",
        yaxis_title="Liquidity Reserve Ratio (Reserves / Loan Principal)"
    )
    return fig

def plot_correlation_matrix(df: pd.DataFrame) -> go.Figure:
    """
    8. Numerical Correlation Analysis Heatmap
    Business Insight: Uncovers key feature dependencies and feature importance drivers.
    """
    cols = [
        "monthly_salary", "credit_score", "bank_balance", "requested_amount",
        "dti_ratio", "expense_ratio", "disposable_income", "affordability_index",
        "liquidity_reserve_ratio", REGRESSION_TARGET
    ]
    existing = [c for c in cols if c in df.columns]
    corr = df[existing].corr().round(2)

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="<b>8. Correlation Matrix of Numerical Features & Derived Ratios</b>"
    )
    fig.update_layout(template="plotly_dark")
    return fig

def plot_scenario_comparisons(df: pd.DataFrame) -> go.Figure:
    """
    9. Loan Product Scenario Comparisons: Requested Amount vs Max Monthly EMI
    Business Insight: Highlights structural difference between micro-loans and high-value asset loans.
    """
    grouped = df.groupby("emi_scenario", observed=True)[["requested_amount", REGRESSION_TARGET, "monthly_salary"]].mean().reset_index()
    
    fig = px.bar(
        grouped,
        x="emi_scenario",
        y=["requested_amount", REGRESSION_TARGET],
        barmode="group",
        title="<b>9. Average Requested Loan Principal vs. Predicted Safe EMI by Scenario</b>",
        labels={"value": "Amount (INR ₹)", "variable": "Metric"}
    )
    fig.update_layout(template="plotly_dark", xaxis_title="EMI Scenario")
    return fig

def plot_outlier_analysis(df: pd.DataFrame) -> go.Figure:
    """
    10. Outlier Distribution Analysis
    Business Insight: Identifies extreme income & wealth outliers versus low-income applicants.
    """
    sample_df = df.sample(n=min(5000, len(df)), random_state=42)
    fig = px.box(
        sample_df,
        y=["monthly_salary", "bank_balance", "emergency_fund"],
        title="<b>10. Financial Outlier Analysis Across Income and Reserves</b>"
    )
    fig.update_layout(template="plotly_dark", yaxis_title="INR ₹ Value")
    return fig

def generate_eda_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Generates structured numerical findings from the dataset."""
    findings = {}
    
    # Target counts
    target_counts = df[CLASSIFICATION_TARGET].value_counts()
    target_props = (df[CLASSIFICATION_TARGET].value_counts(normalize=True) * 100).round(2)
    findings["target_distribution"] = {cls: {"count": int(target_counts[cls]), "pct": float(target_props[cls])} for cls in target_counts.index}

    # Mean ratios by class
    ratio_by_class = df.groupby(CLASSIFICATION_TARGET, observed=True)[DERIVED_RATIO_FEATURES].mean().round(4).to_dict()
    findings["ratio_by_class"] = ratio_by_class

    # Scenario stats
    scenario_stats = df.groupby("emi_scenario", observed=True)[["requested_amount", "requested_tenure", REGRESSION_TARGET]].mean().round(2).to_dict()
    findings["scenario_stats"] = scenario_stats

    return findings

if __name__ == "__main__":
    df_train = load_processed_train_data()
    print(f"Loaded training dataset for EDA: {df_train.shape[0]:,} rows, {df_train.shape[1]} columns.")
    report = generate_eda_report(df_train)
    
    print("\n--- Key Statistical Findings ---")
    print("1. Target Distribution:")
    for cls, stats in report["target_distribution"].items():
        print(f"   - {cls:12s}: {stats['count']:,} ({stats['pct']}%)")

    print("\n2. Mean Financial Ratios by Risk Class:")
    ratios_df = pd.DataFrame(report["ratio_by_class"])
    print(ratios_df)
