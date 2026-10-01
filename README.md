# EMIPredict AI Platform

An ML-based financial risk assessment and EMI affordability platform that predicts borrower eligibility tiers and estimates maximum safe monthly EMI limits using Scikit-Learn, XGBoost, MLflow, and Streamlit.

---

## 1. Project Overview

**EMIPredict AI Platform** assesses borrower creditworthiness and estimates sustainable repayment capacity to assist credit underwriting decision-making.

The platform addresses two predictive tasks:
1. **Borrower Risk Categorization (Classification)**: Categorizes borrowers into risk tiers—`Eligible`, `High_Risk`, or `Not_Eligible`.
2. **Maximum Safe Monthly EMI Capacity (Regression)**: Estimates the maximum safe monthly installment limit (in INR ₹) a borrower can afford.

---

## 2. Problem Statement

Retail credit underwriting requires balancing loan approval rates against default risks. Standard underwriting rules relying solely on credit scores or basic debt-to-income caps often fail to capture an applicant's complete cashflow picture.

This platform addresses:
- **Cashflow-Based Affordability**: Combining gross income, fixed living expenses, and existing loan EMIs into a net disposable cashflow model.
- **Risk Stratification**: Identifying borderline applicants (`High_Risk`) who may qualify with modified loan terms vs. non-viable applicants (`Not_Eligible`).
- **Safe EMI Estimation**: Recommending individualized monthly EMI limits to prevent over-leveraging.

---

## 3. Key Features

- **Multi-Task Machine Learning**: Risk tier classification (`emi_eligibility`) and maximum safe EMI regression (`max_monthly_emi`).
- **Derived Financial Safety Ratios**: Calculates 5 financial safety metrics (DTI, Total Expense Ratio, Disposable Income, Affordability Index, Liquidity Reserve Ratio).
- **Real-Time Risk Predictor**: Interactive input form with presets (`Low Risk`, `High Risk`, `Not Eligible`) and instant prediction outputs.
- **Exploratory Data Analysis**: Interactive Plotly visualizations for distributions, risk breakdowns, scatter plots, and correlation heatmaps.
- **MLflow Model Tracking**: SQLite-backed experiment tracking for parameters, metrics, artifacts, and model versions (`EMI_Predict_Risk_Assessment`).
- **Model Leaderboard**: Comparison of candidate models (Logistic Regression, Linear Regression, Random Forest, XGBoost).
- **Session-Based Data Management (CRUD)**: Create, Read, Update, Delete applicant records in memory with instant model re-evaluation and CSV export (`session_applicant_crud_records.csv`).
- **Validation Rules**: Form validation handling zero-salary rules for unemployed applicants, negative salary rejection, tenure bounds, and ID uniqueness.

---

## 4. Dataset

- **Raw Dataset**: `dataset/emi_prediction_dataset.csv` (404,800 records, 27 raw feature columns).
- **Data Splits**: Saved as Parquet files under `data/processed/`:
  - **Training Set (70%)**: 283,360 records (`train.parquet`)
  - **Validation Set (15%)**: 60,720 records (`val.parquet`)
  - **Test Set (15%)**: 60,720 records (`test.parquet`)

### Feature Schema

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `age` | Numerical (int) | Borrower age (clipped range: 25–60) |
| `gender` | Categorical (Nominal) | Gender (`Male`, `Female`) |
| `marital_status` | Categorical (Nominal) | Marital status (`Single`, `Married`) |
| `education` | Categorical (Ordinal) | Education level (`High School` < `Graduate` < `Post Graduate` < `Professional`) |
| `employment_type` | Categorical (Nominal) | Employment category (`Salaried`, `Self-Employed`, `Freelancer`, `Unemployed`) |
| `years_of_employment` | Numerical (float) | Work experience in years |
| `company_type` | Categorical (Nominal) | Employer category (`MNC`, `Private`, `Government`, `Startup`, `Other`) |
| `house_type` | Categorical (Nominal) | Housing arrangement (`Rented`, `Owned`, `Parental`) |
| `family_size` | Numerical (int) | Number of family members |
| `dependents` | Numerical (int) | Number of dependent family members |
| `monthly_salary` | Numerical (float) | Gross monthly salary (INR ₹) |
| `monthly_rent` | Numerical (float) | Monthly rent (INR ₹) |
| `school_fees` | Numerical (float) | Monthly school fees (INR ₹) |
| `college_fees` | Numerical (float) | Monthly college fees (INR ₹) |
| `travel_expenses` | Numerical (float) | Monthly travel costs (INR ₹) |
| `groceries_utilities` | Numerical (float) | Monthly groceries and utilities (INR ₹) |
| `other_monthly_expenses`| Numerical (float) | Other monthly expenses (INR ₹) |
| `existing_loans` | Categorical (Nominal) | Existing active loans (`Yes`, `No`) |
| `current_emi_amount` | Numerical (float) | Current active monthly EMI (INR ₹) |
| `credit_score` | Numerical (int) | Credit score (clipped range: 300–850) |
| `bank_balance` | Numerical (float) | Liquid bank balance (INR ₹) |
| `emergency_fund` | Numerical (float) | Emergency savings (INR ₹) |
| `requested_amount` | Numerical (float) | Requested loan principal (INR ₹) |
| `requested_tenure` | Numerical (int) | Requested loan term in months |
| `emi_scenario` | Categorical (Nominal) | EMI scenario category (`Scenario_1` to `Scenario_5`) |

*Note: The raw CSV dataset remains unmodified on disk. CRUD operations operate strictly on isolated in-memory session records.*

---

## 5. Data Preprocessing

Implemented in `src/preprocessing/cleaner.py`:
1. **Cleaning & Normalization**: Numeric coercion, gender string harmonization, whitespace stripping, and duplicate removal.
2. **Validation & Boundaries**: Target filtering (`emi_eligibility` valid classes, `max_monthly_emi` $\ge$ ₹500) and feature range clipping.
3. **Data Leakage Prevention**: Stratified splitting (70% Train / 15% Val / 15% Test). Imputation parameters (medians for numerical, modes for categorical) are calculated **strictly on the training split**.
4. **Optimization**: Memory optimization using Parquet format under `data/processed/`.

---

## 6. Feature Engineering

Implemented in `src/preprocessing/feature_engineering.py` (`FinancialRatioTransformer` with $\epsilon = 1 \times 10^{-5}$):

1. **Debt-to-Income Ratio (DTI)**:
   $$\text{DTI Ratio} = \frac{\text{current\_emi\_amount}}{\text{monthly\_salary} + \epsilon}$$

2. **Total Expense Ratio**:
   $$\text{Expense Ratio} = \frac{\text{monthly\_rent} + \text{school\_fees} + \text{college\_fees} + \text{travel\_expenses} + \text{groceries\_utilities} + \text{other\_monthly\_expenses}}{\text{monthly\_salary} + \epsilon}$$

3. **Disposable Income**:
   $$\text{Disposable Income} = \text{monthly\_salary} - (\text{total\_living\_expenses} + \text{current\_emi\_amount})$$

4. **EMI Affordability Index**:
   $$\text{Affordability Index} = \frac{\text{Disposable Income}}{\frac{\text{requested\_amount}}{\text{requested\_tenure}} + \epsilon}$$

5. **Liquidity Reserve Ratio**:
   $$\text{Liquidity Reserve Ratio} = \frac{\text{bank\_balance} + \text{emergency\_fund}}{\text{requested\_amount} + \epsilon}$$

The `FullPreprocessingPipeline` combines ratio computation with `ColumnTransformer` (`RobustScaler`, `OrdinalEncoder`, `OneHotEncoder`) fitted on training data and saved to `models/preprocessor.pkl`.

---

## 7. Exploratory Data Analysis

Implemented in `src/preprocessing/eda.py` and visualised in `src/app/pages/1_📊_Overview_&_EDA.py`:
- Target distribution chart (`Eligible` ~55%, `High_Risk` ~25%, `Not_Eligible` ~20%).
- Risk breakdown across 5 EMI scenario categories.
- Salary and credit score threshold distributions.
- DTI ratio and Disposable Income distributions.
- Affordability Index vs. Liquidity Reserve Ratio scatter plot.
- Feature correlation matrix heatmap.
- Financial outlier analysis.

---

## 8. Machine Learning Models

Implemented in `src/models/train_classifier.py` and `src/models/train_regressor.py`.

### Candidate Models
- **Classification (`emi_eligibility`)**: Logistic Regression (balanced), Random Forest Classifier, XGBoost Classifier.
- **Regression (`max_monthly_emi`)**: Linear Regression, Random Forest Regressor, XGBoost Regressor.

### Model Selection
Models are evaluated on the validation split (`val.parquet`). Best models are selected based on Macro F1 / High-Risk recall (classification) and lowest RMSE (regression), then saved to `models/classifier_best.pkl` and `models/regressor_best.pkl`.

---

## 9. Model Performance

Tested on the held-out test dataset (60,720 records):

### Classification Target (`emi_eligibility`)
- **Best Model**: XGBoost Classifier

| Metric | Test Value |
| :--- | :--- |
| **Test Accuracy** | **96.83%** |
| **High-Risk Recall** | **92.07%** |
| **Macro F1-Score** | **89.36%** |

### Regression Target (`max_monthly_emi`)
- **Best Model**: XGBoost Regressor

| Metric | Test Value |
| :--- | :--- |
| **Test RMSE** | **₹734.04** |
| **Test MAE** | **₹258.20** |
| **Test $R^2$ Score** | **0.9910** |

---

## 10. MLflow / MLOps Integration

Managed in `src/models/mlflow_utils.py`:
- **Experiment Name**: `EMI_Predict_Risk_Assessment`
- **Tracking Store**: SQLite database at `mlflow.db` (`sqlite:///mlflow.db`) with artifacts in `mlruns/`.
- **Tracked Artifacts**: Hyperparameters, metrics (accuracy, recall, F1, ROC-AUC, RMSE, MAE, $R^2$), plot artifacts (confusion matrices, feature importance charts, actual vs. predicted scatter plots, residuals plots), and registered models (`EMIPredict_Best_Classifier`, `EMIPredict_Best_Regressor`).

---

## 11. Streamlit Application

The application (`src/app/`) consists of 4 pages:

1. **Home (`src/app/🏠_Home.py`)**: Overview of platform architecture, metrics, and navigation.
2. **Overview & EDA (`src/app/pages/1_📊_Overview_&_EDA.py`)**: Interactive Plotly analytics across 4 tabs.
3. **Realtime Risk Predictor (`src/app/pages/2_🔮_Realtime_Risk_Predictor.py`)**: Borrower input form with presets (`Low Risk`, `High Risk`, `Not Eligible`), risk tier prediction, max safe EMI estimation, and financial ratio cards.
4. **MLflow Model Dashboard (`src/app/pages/3_🧪_MLflow_Model_Dashboard.py`)**: Model comparison leaderboards, live MLflow experiment metrics, diagnostic plots, and registry information.
5. **Data Management CRUD (`src/app/pages/4_⚙️_Data_Management_CRUD.py`)**: In-memory CRUD operations (Create, Read, Update, Delete), applicant filtering, model re-evaluation, and CSV export.

---

## 12. Project Structure

```text
EMI Predict/
├── .streamlit/
│   └── config.toml
├── data/
│   └── processed/
│       ├── train.parquet
│       ├── val.parquet
│       └── test.parquet
├── dataset/
│   └── emi_prediction_dataset.csv
├── docs/
│   ├── architecture.md
│   ├── context.md
│   └── problemStatement.txt
├── mlflow.db
├── mlruns/
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

---

## 13. Technologies Used

- **Language**: Python (>= 3.10)
- **Data Processing**: Pandas, NumPy, PyArrow
- **Machine Learning**: Scikit-Learn, XGBoost, Joblib
- **MLOps**: MLflow, SQLite
- **Web Interface**: Streamlit
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Testing**: Pytest

---

## 14. Installation

1. Open PowerShell / terminal in the project root:
   ```powershell
   cd "D:\EMI Predict"
   ```

2. Create a virtual environment:
   ```powershell
   python -m venv .venv
   ```

3. Activate the virtual environment:
   - **PowerShell**:
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - **Command Prompt**:
     ```cmd
     .\.venv\Scripts\activate.bat
     ```

4. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

---

## 15. How to Run

Launch the Streamlit application:

```powershell
streamlit run src/app/🏠_Home.py
```

Access the app in your browser at `http://localhost:8501`.

---

## 16. Testing

Run the automated test suite using Pytest:

```powershell
pytest tests/
```

- **Test Results**: 14 passed
- **Modules Covered**:
  - `tests/test_applicant_validation.py`: Salary validation, unemployed zero-salary rules, Applicant ID preservation.
  - `tests/test_models.py`: Classifier prediction bounds, regressor numeric outputs, minimum safe EMI boundary.
  - `tests/test_preprocessing.py`: Financial ratio formulas, zero-division handling, preprocessor loading.

---

## 17. Results / Business Value

- **Decision Support**: Provides standardized scoring to assist manual underwriting workflows.
- **Default Risk Management**: High-risk recall of 92.07% helps flag vulnerable applicants prior to loan approval.
- **Repayment Capping**: Safe EMI capacity regression ($R^2 = 0.9910$) prevents over-leveraging.
- **Underwriting Guardrail**: Serves as a decision-support system to complement institutional credit policies.

---

## 18. Validation & Error Handling

- **Income Validation**: Unemployed applicants accept ₹0 salary. Employed applicants require salary $> 0$. Negative salaries are rejected.
- **Tenure Boundary**: Requested tenure must be $\ge 1$ month.
- **Applicant ID Integrity**: CRUD operations require unique, non-empty Applicant IDs.
- **Zero-Division Safeguards**: Ratio calculations use $\epsilon = 1 \times 10^{-5}$ to prevent divide-by-zero errors.

---


## 19. Future Improvements

- **Cloud Deployment**: Containerized deployment via Docker / Streamlit Cloud.
- **Model Monitoring**: Data drift monitoring and automated re-training pipelines.
- **Authentication**: Role-based access control for underwriters and auditors.
- **Model Explainability**: Integration of SHAP value attributions for prediction explanations.
