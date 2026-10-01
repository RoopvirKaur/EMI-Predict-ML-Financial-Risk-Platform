# EMIPredict AI Platform

An ML-based financial risk assessment and EMI affordability platform
that predicts borrower eligibility tiers and estimates maximum safe
monthly EMI limits using Scikit-Learn, XGBoost, MLflow, and Streamlit.

------------------------------------------------------------------------

## 1. Project Overview

**EMIPredict AI Platform** assesses borrower creditworthiness and
estimates sustainable repayment capacity to assist credit underwriting
decision-making.

The platform addresses two predictive tasks:

1.  **Borrower Risk Categorization (Classification):** Categorizes
    borrowers into risk tiers --- `Eligible`, `High_Risk`, or
    `Not_Eligible`.
2.  **Maximum Safe Monthly EMI Capacity (Regression):** Estimates the
    maximum safe monthly installment limit (in INR ₹) a borrower can
    afford.

------------------------------------------------------------------------

## 2. Problem Statement

Retail credit underwriting requires balancing loan approval rates
against default risks. Standard underwriting rules relying solely on
credit scores or basic debt-to-income caps often fail to capture an
applicant's complete cashflow picture.

This platform addresses:

-   **Cashflow-Based Affordability:** Combining gross income, fixed
    living expenses, and existing loan EMIs into a net disposable
    cashflow model.
-   **Risk Stratification:** Identifying borderline applicants
    (`High_Risk`) who may qualify with modified loan terms
    vs. non-viable applicants (`Not_Eligible`).
-   **Safe EMI Estimation:** Recommending individualized monthly EMI
    limits to prevent over-leveraging.

------------------------------------------------------------------------

## 3. Key Features

-   **Multi-Task Machine Learning:** Risk tier classification
    (`emi_eligibility`) and maximum safe EMI regression
    (`max_monthly_emi`).
-   **Derived Financial Safety Ratios:** Calculates 5 financial safety
    metrics --- DTI, Total Expense Ratio, Disposable Income,
    Affordability Index, and Liquidity Reserve Ratio.
-   **Real-Time Risk Predictor:** Interactive input form with presets
    (`Low Risk`, `High Risk`, `Not Eligible`) and instant prediction
    outputs.
-   **Exploratory Data Analysis:** Interactive Plotly visualizations for
    distributions, risk breakdowns, scatter plots, and correlation
    heatmaps.
-   **MLflow Model Tracking:** SQLite-backed experiment tracking for
    parameters, metrics, artifacts, and model versions
    (`EMI_Predict_Risk_Assessment`).
-   **Model Leaderboard:** Comparison of candidate models (Logistic
    Regression, Linear Regression, Random Forest, XGBoost).
-   **Session-Based Data Management (CRUD):** Create, Read, Update,
    Delete applicant records in memory with instant model re-evaluation
    and CSV export (`session_applicant_crud_records.csv`).
-   **Validation Rules:** Form validation handling zero-salary rules for
    unemployed applicants, negative salary rejection, tenure bounds, and
    ID uniqueness.

------------------------------------------------------------------------

## 4. Dataset

The project was developed using a raw dataset containing **404,800
records and 27 raw feature columns**.

The original dataset is **not included in this GitHub repository because
of its large file size**.

The application expects the raw dataset at:

``` text
dataset/emi_prediction_dataset.csv
```

If you have the dataset locally, place it at the path above before
running the project.

### Data Splits

The dataset is divided into:

-   **Training Set (70%):** 283,360 records
-   **Validation Set (15%):** 60,720 records
-   **Test Set (15%):** 60,720 records

The processed Parquet files are generated locally and are **not included
in this repository** because they are generated data artifacts.

### Feature Schema

  Feature Name               Type              Description
  -------------------------- ----------------- ------------------------------------
  `age`                      Numerical (int)   Borrower age
  `gender`                   Categorical       Gender
  `marital_status`           Categorical       Marital status
  `education`                Categorical       Education level
  `employment_type`          Categorical       Employment category
  `years_of_employment`      Numerical         Work experience in years
  `company_type`             Categorical       Employer category
  `house_type`               Categorical       Housing arrangement
  `family_size`              Numerical         Number of family members
  `dependents`               Numerical         Number of dependent family members
  `monthly_salary`           Numerical         Gross monthly salary
  `monthly_rent`             Numerical         Monthly rent
  `school_fees`              Numerical         Monthly school fees
  `college_fees`             Numerical         Monthly college fees
  `travel_expenses`          Numerical         Monthly travel costs
  `groceries_utilities`      Numerical         Monthly groceries and utilities
  `other_monthly_expenses`   Numerical         Other monthly expenses
  `existing_loans`           Categorical       Existing active loans
  `current_emi_amount`       Numerical         Current active monthly EMI
  `credit_score`             Numerical         Credit score
  `bank_balance`             Numerical         Liquid bank balance
  `emergency_fund`           Numerical         Emergency savings
  `requested_amount`         Numerical         Requested loan principal
  `requested_tenure`         Numerical         Requested loan term in months
  `emi_scenario`             Categorical       EMI scenario category

> **Note:** The raw CSV dataset remains unmodified on disk. CRUD
> operations operate strictly on isolated in-memory session records.

------------------------------------------------------------------------

## 5. Data Preprocessing

Implemented in `src/preprocessing/cleaner.py`:

1.  **Cleaning & Normalization:** Numeric coercion, gender string
    harmonization, whitespace stripping, and duplicate removal.
2.  **Validation & Boundaries:** Target filtering (`emi_eligibility`
    valid classes, `max_monthly_emi >= ₹500`) and feature range
    clipping.
3.  **Data Leakage Prevention:** Stratified splitting (70% Train / 15%
    Val / 15% Test). Imputation parameters (medians for numerical, modes
    for categorical) are calculated strictly on the training split.
4.  **Optimization:** Memory optimization using Parquet format under
    `data/processed/`.

------------------------------------------------------------------------

## 6. Feature Engineering

Implemented in `src/preprocessing/feature_engineering.py` using
`FinancialRatioTransformer` with `epsilon = 1e-5`.

### Financial Safety Metrics

1.  **Debt-to-Income Ratio (DTI)**

``` text
DTI Ratio = current_emi_amount / (monthly_salary + epsilon)
```

2.  **Total Expense Ratio**

``` text
Expense Ratio =
(monthly_rent + school_fees + college_fees + travel_expenses
+ groceries_utilities + other_monthly_expenses)
 / (monthly_salary + epsilon)
```

3.  **Disposable Income**

``` text
Disposable Income =
monthly_salary - (total_living_expenses + current_emi_amount)
```

4.  **EMI Affordability Index**

``` text
Affordability Index =
Disposable Income / (requested_amount / requested_tenure + epsilon)
```

5.  **Liquidity Reserve Ratio**

``` text
Liquidity Reserve Ratio =
(bank_balance + emergency_fund) / (requested_amount + epsilon)
```

The `FullPreprocessingPipeline` combines ratio computation with
`ColumnTransformer` (`RobustScaler`, `OrdinalEncoder`, `OneHotEncoder`)
fitted on training data and saved to `models/preprocessor.pkl`.

------------------------------------------------------------------------

## 7. Exploratory Data Analysis

Implemented in `src/preprocessing/eda.py` and visualised in:

``` text
src/app/pages/1_📊_Overview_&_EDA.py
```

The EDA includes:

-   Target distribution chart
-   Risk breakdown across 5 EMI scenario categories
-   Salary and credit score threshold distributions
-   DTI ratio and Disposable Income distributions
-   Affordability Index vs. Liquidity Reserve Ratio scatter plot
-   Feature correlation matrix heatmap
-   Financial outlier analysis

------------------------------------------------------------------------

## 8. Machine Learning Models

Implemented in:

``` text
src/models/train_classifier.py
src/models/train_regressor.py
```

### Candidate Models

**Classification (`emi_eligibility`)** - Logistic Regression
(balanced) - Random Forest Classifier - XGBoost Classifier

**Regression (`max_monthly_emi`)** - Linear Regression - Random Forest
Regressor - XGBoost Regressor

### Model Selection

Models are evaluated on the validation split. The best models are
selected based on:

-   Macro F1 / High-Risk recall for classification
-   Lowest RMSE for regression

The selected models are saved as:

``` text
models/classifier_best.pkl
models/regressor_best.pkl
```

------------------------------------------------------------------------

## 9. Model Performance

The models were tested on the held-out test dataset containing **60,720
records**.

### Classification Target --- `emi_eligibility`

**Best Model:** XGBoost Classifier

  Metric                   Test Value
  ---------------------- ------------
  **Test Accuracy**        **96.83%**
  **High-Risk Recall**     **92.07%**
  **Macro F1-Score**       **89.36%**

### Regression Target --- `max_monthly_emi`

**Best Model:** XGBoost Regressor

  Metric                 Test Value
  ------------------- -------------
  **Test RMSE**         **₹734.04**
  **Test MAE**          **₹258.20**
  **Test R² Score**      **0.9910**

------------------------------------------------------------------------

## 10. MLflow / MLOps Integration

MLflow integration is managed in:

``` text
src/models/mlflow_utils.py
```

The project uses the experiment:

``` text
EMI_Predict_Risk_Assessment
```

During development, MLflow was used for:

-   Hyperparameter tracking
-   Metric tracking
-   Plot artifacts
-   Model versions
-   Model registration

The local MLflow tracking database and `mlruns/` artifacts are **not
included in this GitHub repository**. The MLflow integration code
remains available in the source files.

------------------------------------------------------------------------

## 11. Streamlit Application

The application is organized into **5 pages**:

1.  **Home**

    ``` text
    src/app/🏠_Home.py
    ```

    Overview of platform architecture, metrics, and navigation.

2.  **Overview & EDA**

    ``` text
    src/app/pages/1_📊_Overview_&_EDA.py
    ```

    Interactive Plotly analytics across multiple sections.

3.  **Realtime Risk Predictor**

    ``` text
    src/app/pages/2_🔮_Realtime_Risk_Predictor.py
    ```

    Borrower input form with presets, risk tier prediction, maximum safe
    EMI estimation, and financial ratio cards.

4.  **MLflow Model Dashboard**

    ``` text
    src/app/pages/3_🧪_MLflow_Model_Dashboard.py
    ```

    Model comparison leaderboards, MLflow experiment metrics, diagnostic
    plots, and registry information.

5.  **Data Management CRUD**

    ``` text
    src/app/pages/4_⚙️_Data_Management_CRUD.py
    ```

    In-memory CRUD operations, applicant filtering, model re-evaluation,
    and CSV export.

------------------------------------------------------------------------

## 12. Project Structure

``` text
EMI Predict/
├──config.toml
├── data/
│   └── processed/
    ├── test.parquet
│   ├── train.parquet
│   ├── val.parquet

├── dataset/
│   └── README.md
├── docs/
│   ├── architecture.md
│   ├── context.md
│   └── problemStatement.txt
├── models/
│   ├── classification_feature_importance.png
│   ├── classifier_best.pkl
│   ├── cm_val_Logistic_Regression.png
│   ├── cm_val_Random_Forest_Classifier.png
│   ├── cm_val_XGBoost_Classifier.png
│   ├── label_encoder.pkl
│   ├── preprocessor.pkl
│   ├── regression_actual_vs_predicted.png
│   ├── regression_feature_importance.png
│   ├── regression_residuals.png
│   └── regressor_best.pkl
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── app/
│   │   ├── 🏠_Home.py
│   │   └── pages/
│   │       ├── 1_📊_Overview_&_EDA.py
│   │       ├── 2_🔮_Realtime_Risk_Predictor.py
│   │       ├── 3_🧪_MLflow_Model_Dashboard.py
│   │       └── 4_⚙️_Data_Management_CRUD.py
│   ├── models/
│   │   ├── mlflow_utils.py
│   │   ├── train_classifier.py
│   │   └── train_regressor.py
│   └── preprocessing/
│       ├── cleaner.py
│       ├── eda.py
│       └── feature_engineering.py
└── tests/
    ├── test_applicant_validation.py
    ├── test_models.py
    └── test_preprocessing.py
```

### Files intentionally excluded from the repository

The following generated or large files are not included:

-   `dataset/emi_prediction_dataset.csv`
-   `data/processed/train.parquet`
-   `data/processed/val.parquet`
-   `data/processed/test.parquet`
-   `mlflow.db`
-   `mlruns/`

The source code and documentation required to understand and run the
project are included.

------------------------------------------------------------------------

## 13. Technologies Used

-   **Language:** Python (\>= 3.10)
-   **Data Processing:** Pandas, NumPy, PyArrow
-   **Machine Learning:** Scikit-Learn, XGBoost, Joblib
-   **MLOps:** MLflow, SQLite
-   **Web Interface:** Streamlit
-   **Visualization:** Plotly, Matplotlib, Seaborn
-   **Testing:** Pytest

------------------------------------------------------------------------

## 14. Installation

### 1. Clone the repository

``` powershell
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd "EMI Predict"
```

### 2. Create a virtual environment

``` powershell
python -m venv .venv
```

### 3. Activate the virtual environment

**PowerShell:**

``` powershell
.\.venv\Scripts\Activate.ps1
```

**Command Prompt:**

``` cmd
.\.venv\Scripts\activate.bat
```

### 4. Install dependencies

``` powershell
pip install -r requirements.txt
```

### 5. Add the dataset

Place the original dataset at:

``` text
dataset/emi_prediction_dataset.csv
```

The dataset is not included in the repository because of its large file
size.

------------------------------------------------------------------------

## 15. How to Run

Launch the Streamlit application from the project root:

``` powershell
streamlit run src/app/🏠_Home.py
```

Then access the application in your browser at:

``` text
http://localhost:8501
```

------------------------------------------------------------------------

## 16. Testing

Run the automated test suite using Pytest:

``` powershell
pytest tests/
```

The project test suite covers:

-   Applicant validation
-   Salary validation and unemployed zero-salary rules
-   Applicant ID preservation
-   Classifier prediction bounds
-   Regressor numeric outputs
-   Minimum safe EMI boundary
-   Financial ratio formulas
-   Zero-division handling
-   Preprocessor loading

------------------------------------------------------------------------

## 17. Results / Business Value

-   **Decision Support:** Provides standardized scoring to assist manual
    underwriting workflows.
-   **Default Risk Management:** High-risk recall of 92.07% helps flag
    vulnerable applicants prior to loan approval.
-   **Repayment Capping:** Safe EMI capacity regression with an R² score
    of 0.9910 helps estimate affordable repayment limits.
-   **Underwriting Guardrail:** Serves as a decision-support system to
    complement institutional credit policies.

------------------------------------------------------------------------

## 18. Validation & Error Handling

-   **Income Validation:** Unemployed applicants accept ₹0 salary.
    Employed applicants require salary \> 0. Negative salaries are
    rejected.
-   **Tenure Boundary:** Requested tenure must be at least 1 month.
-   **Applicant ID Integrity:** CRUD operations require unique,
    non-empty Applicant IDs.
-   **Zero-Division Safeguards:** Ratio calculations use
    `epsilon = 1e-5` to prevent divide-by-zero errors.

------------------------------------------------------------------------

## 19. Deployment

Deployment URL will be added after final deployment.

------------------------------------------------------------------------

## 20. Future Improvements

-   **Cloud Deployment:** Containerized deployment via Docker /
    Streamlit Cloud.
-   **Model Monitoring:** Data drift monitoring and automated
    re-training pipelines.
-   **Authentication:** Role-based access control for underwriters and
    auditors.
-   **Model Explainability:** Integration of SHAP value attributions for
    prediction explanations.
