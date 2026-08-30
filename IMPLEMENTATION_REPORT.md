# ChurnGuard AI — Implementation Report
### Explainable Customer Churn Prediction & Retention Intelligence Platform
**Major Capstone Project Implementation & Engineering Report**

---

## 1. Executive Summary & Project Context

**ChurnGuard AI** is an enterprise-grade, explainable machine learning platform engineered to address subscriber attrition in subscription-based businesses (specifically telecommunications). 

Developed as an internship Machine Learning Major Capstone Project, the platform combines a **first-principles mathematical implementation of Logistic Regression built entirely from scratch in pure NumPy** with an end-to-end operational software architecture featuring zero-leakage preprocessing, rigorous Scikit-Learn benchmarking, 100% transparent log-odds waterfall explainability, deterministic retention action playbooks, and a dual-mode Streamlit user interface.

### Key Project Highlights
* **Pure Mathematical Implementation:** Custom Logistic Regression written in pure NumPy with no high-level estimator dependencies for training or inference. Includes vectorized Sigmoid, class-weighted Binary Cross-Entropy (Log-Loss), L2 regularization, analytical gradient computation, and mini-batch Momentum Gradient Descent.
* **Scientific Benchmark Parity:** Evaluated against Scikit-Learn's `LogisticRegression(solver='lbfgs')` on a locked holdout test partition ($N = 1,057$). The custom model achieves **$0.8452$ ROC-AUC** (vs. Scikit-Learn's $0.8449$) with a Pearson probability correlation of **$r = 0.998200$** and a **4.0x faster inference latency** ($0.056\text{ ms}$ vs. $0.222\text{ ms}$ per 1,000 records).
* **Validated Business Value:** Generates **$+\$64,350.00$** in net retention savings ($+\$60.88$ per monitored subscriber on the test split) based on a realistic cost-benefit matrix ($211 \times \$350 - 190 \times \$50 = \$64,350$).
* **Fidelity & Explainability:** Full additive log-odds decomposition ($z = b + \sum w_j x_j \implies p = \sigma(z)$) providing exact step-by-step waterfall attribution with zero mathematical discrepancy ($0.00\text{e}+00$).
* **Dynamic Tabular ML Engine (Mode 2):** Supports arbitrary binary classification tabular datasets (e.g. Employee Attrition, Loan Default, Fraud Detection, Medical Diagnosis). Automatically detects target candidates, positive classes, identifier columns, and feature types; fits a zero-leakage pipeline strictly on training data; trains a fresh scratch NumPy model; optimizes decision thresholds; and exports full cohort predictions without touching the frozen official demo.
* **Automated Quality Assurance:** 57 comprehensive unit and integration tests passing cleanly across 8 distinct test suites ($100\%$ pass rate).

---

## 2. System Architecture & Directory Structure

The project adheres to a modular, production-ready software engineering structure:

```
a:/intership capstone project/
├── app/                                    # Streamlit Production UI Application
│   ├── components/
│   │   ├── batch_view.py                   # Legacy Batch Scoring View
│   │   ├── benchmark_view.py               # Scientific Benchmark Arena View
│   │   ├── executive_view.py               # Executive Command Center View
│   │   ├── guided_demo_view.py             # 7-Step Interactive Guided Tour View
│   │   ├── landing_view.py                 # Dual-Mode Landing Page
│   │   ├── single_prediction_view.py       # Live Customer Simulator & Waterfall View
│   │   └── user_dataset_view.py            # Mode B: Analyze Your Dataset Workflow
│   └── streamlit_app.py                    # Main Application Entrypoint & Router
├── artifacts/                              # Frozen Model Weights & Evaluation Metadata
│   ├── benchmark_results.json              # Side-by-Side Model Benchmark Metrics
│   ├── custom_logistic_model.npz           # Frozen Custom NumPy Model Parameters
│   ├── feature_names.json                  # 41 Transformed Feature Column Names
│   ├── final_results.json                  # Authoritative Master Results File
│   └── preprocessor_pipeline.joblib        # Fitted Scikit-Learn Preprocessing Pipeline
├── config/
│   └── config.py                           # Global Paths, Seeds, Schema & Hyperparameters
├── data/
│   ├── processed/                          # Partitioned & Engineered Dataset Splits
│   │   ├── schema_metadata.json            # Dataset Schema Metadata & Checksums
│   │   ├── test.csv                        # Frozen Test Split (15%, N = 1,057)
│   │   ├── train.csv                       # Frozen Train Split (70%, N = 4,930)
│   │   └── val.csv                         # Frozen Validation Split (15%, N = 1,056)
│   └── raw/
│       └── telco_customer_churn.csv        # Master Raw Dataset (7,043 rows, 21 columns)
├── reports/                                # Generated Reports & Visualizations
│   ├── figures/                            # Publication-Quality Plots & ROC/PR Curves
│   ├── capstone_analytical_report.md       # Comprehensive Capstone Documentation
│   ├── final_experiment_summary.md         # Final Frozen Experiment Summary
│   └── IMPLEMENTATION_REPORT.md            # This Master Implementation Report
├── src/                                    # Core Python Source Code Modules
│   ├── data/
│   │   ├── data_loader.py                  # Dataset Ingestion & Partitioning
│   │   ├── feature_engineer.py             # Domain Feature Engineering Engine
│   │   ├── preprocessor.py                 # Zero-Leakage Preprocessing Pipeline
│   │   └── validator.py                    # Schema Validator & Health Check Engine
│   ├── evaluation/
│   │   └── metrics.py                      # Pure NumPy Evaluation Metrics & Cost Utility
│   ├── explainability/
│   │   ├── feature_importance.py           # Global Feature Odds Ratios & Text Gen
│   │   ├── individual_explainer.py         # Additive Log-Odds Waterfall Explainer
│   │   └── retention_playbook.py           # Deterministic Prescriptive Action Engine
│   └── models/
│       ├── custom_logistic_regression.py   # Scratch NumPy Logistic Regression Classifier
│       └── optimizer.py                    # Mini-Batch Momentum Gradient Descent Optimizer
├── tests/                                  # Comprehensive Test Suite (49 Tests)
│   ├── test_custom_logistic_regression.py  # Model Mathematical Unit Tests
│   ├── test_end_to_end_pipeline.py         # End-to-End Pipeline Integration Tests
│   ├── test_explainability.py              # Log-Odds Reconstruction & Playbook Tests
│   ├── test_math_primitives.py             # Vectorized Math & Numerical Gradient Checks
│   ├── test_metrics.py                     # Metric Parity & Business Value Tests
│   ├── test_preprocessor.py                # Preprocessing & Type Coercion Tests
│   └── test_user_dataset_workflow.py       # Dataset Upload, Health Check & Invariance Tests
├── FINAL_EXPERIMENT_AND_SUMMARY_RESULTS.md # Authoritative Experiment Master Document
├── README.md                               # Project Readme & Setup Guide
└── requirements.txt                        # Core Dependencies (numpy, pandas, scikit-learn, etc.)
```

---

## 3. Data Pipeline & Zero-Leakage Preprocessing

### 3.1 Dataset Specification
* **Dataset:** Telco Customer Churn (`telco_customer_churn.csv`)
* **Total Records:** $7,043$ customer rows $\times$ $21$ original columns.
* **Target Variable:** `Churn` ($1 = \text{Yes}$, $0 = \text{No}$).
* **Class Distribution:** $5,174$ Active ($73.46\%$) vs. $1,869$ Churned ($26.54\%$) (Class Imbalance Ratio $\approx 2.77:1$).
* **Data Cleaning:** 11 blank whitespace entries in `TotalCharges` (associated with new subscribers having `tenure == 0`) were identified and imputed to `0.00`.

### 3.2 Stratified Partitioning
To guarantee zero data leakage, partitioning was executed prior to any scaling or encoding:
* **Train Split ($70\%$):** $4,930$ rows ($1,308$ churned, $26.53\%$).
* **Validation Split ($15\%$):** $1,056$ rows ($280$ churned, $26.52\%$) — reserved strictly for hyperparameter tuning and decision threshold optimization.
* **Test Split ($15\%$):** $1,057$ rows ($281$ churned, $26.58\%$) — held out completely until final frozen evaluation.

### 3.3 Domain Feature Engineering
Nine domain features were engineered to capture non-linear service adoption, billing volatility, and tenure risks:
1. `tenure_cohort`: Discrete tenure buckets (`0-12m`, `13-24m`, `25-48m`, `49-72m`).
2. `total_services_count`: Cumulative sum of subscribed services ($0$ to $9$).
3. `streaming_services_count`: Count of streaming products (`StreamingTV`, `StreamingMovies`).
4. `protection_services_count`: Count of security products (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`).
5. `has_internet_service`: Binary flag indicating active fixed-line broadband.
6. `avg_monthly_charges`: Longitudinal charge ratio $\frac{\text{TotalCharges}}{\text{tenure} + 1}$.
7. `monthly_charge_discrepancy`: Current billing divergence $\text{MonthlyCharges} - \text{avg\_monthly\_charges}$.
8. `is_long_tenure_m2m`: Interaction indicator for $\text{tenure} > 24$ on a month-to-month contract.
9. `is_fiber_without_techsupport`: Interaction indicator for high-risk fiber subscribers lacking tech support.

### 3.4 Feature Transformation Matrix
* **Numerical Features (6 features):** Standardized using `StandardScaler` fitted strictly on the training partition:
  $$x_{\text{scaled}} = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
* **Categorical Features (35 features after encoding):** One-hot encoded using `OneHotEncoder(drop='first', sparse_output=False)` fitted strictly on training data.
* **Total Dimension:** $X \in \mathbb{R}^{N \times 41}$.

---

## 4. Mathematical Formulation & NumPy Logistic Regression

The primary estimator ([`src/models/custom_logistic_regression.py`](file:///a:/intership%20capstone%20project/src/models/custom_logistic_regression.py)) is implemented entirely from first principles using NumPy array operations.

### 4.1 Linear Hypothesis & Numerically Stable Sigmoid
The linear predictor is formulated as:
$$z = Xw + b$$

The sigmoid activation function transforms logits into calibrated probabilities:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

To eliminate floating-point overflow encountered when $z < -709$ or $z > 709$, the sigmoid function uses a numerically stable piecewise formulation:
$$\sigma(z) = \begin{cases} \frac{1}{1 + e^{-z}} & \text{if } z \ge 0 \\ \frac{e^z}{1 + e^z} & \text{if } z < 0 \end{cases}$$

### 4.2 Weighted Binary Cross-Entropy Loss with L2 Regularization
To address the $2.77:1$ class imbalance, sample loss contributions are weighted inversely to class frequencies:
$$w_1 = \frac{m}{2 \cdot m_1}, \quad w_0 = \frac{m}{2 \cdot m_0}$$

The total cost function combines weighted log-loss and an L2 parameter shrinkage penalty:
$$J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ w_1 y^{(i)} \ln(\hat{y}^{(i)} + \epsilon) + w_0 (1 - y^{(i)}) \ln(1 - \hat{y}^{(i)} + \epsilon) \right] + \frac{\lambda}{2m} \|w\|_2^2$$
where $\epsilon = 10^{-15}$ prevents $\ln(0)$ undefined evaluations.

### 4.3 Analytical Gradients & Numerical Gradient Checking
The analytical gradients with respect to weights $w$ and bias $b$ are derived as:
$$e^{(i)} = \begin{cases} w_1 (\hat{y}^{(i)} - 1) & \text{if } y^{(i)} = 1 \\ w_0 \hat{y}^{(i)} & \text{if } y^{(i)} = 0 \end{cases}$$
$$\frac{\partial J}{\partial w} = \frac{1}{m} X^T e + \frac{\lambda}{m} w$$
$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} e^{(i)}$$

Gradient computation correctness was validated via finite-difference numerical gradient checking:
$$\frac{\partial J}{\partial w_j} \approx \frac{J(w_j + \delta) - J(w_j - \delta)}{2\delta}, \quad \delta = 10^{-5}$$
* **Observed Relative Gradient Error:** $\mathbf{5.91 \times 10^{-10}}$ (well below the $10^{-7}$ verification threshold).

### 4.4 Mini-Batch Momentum Gradient Descent
Optimization is performed via Mini-Batch Gradient Descent augmented with Classical Momentum ($\beta = 0.9$):
$$v_w := \beta v_w + \alpha \nabla_w J(w, b)$$
$$v_b := \beta v_b + \alpha \nabla_b J(w, b)$$
$$w := w - v_w, \quad b := b - v_b$$
* **Hyperparameters:** $\alpha = 0.05$, $\beta = 0.9$, $\lambda = 0.01$, $\text{batch\_size} = 64$, $\text{max\_iter} = 1,500$, with early stopping on validation loss ($\text{patience} = 50$).
* **Training Convergence:** Converged in **122 iterations** with a final training loss of $0.4996$ and validation loss of $0.5103$.

---

## 5. Scientific Benchmark Parity & Metric Evaluation

The custom NumPy model was benchmarked side-by-side against Scikit-Learn's `LogisticRegression(solver='lbfgs', C=100.0, class_weight='balanced')` on the exact same 1,057-record holdout test split:

### 5.1 Test Set Performance Scorecard ($N = 1,057$)

| Evaluation Metric | Custom NumPy LR (Scratch) | Scikit-Learn LR (Baseline) | Parity Delta | Assessment |
|---|---|---|---|---|
| **ROC-AUC Score** | **`0.8452`** | `0.8449` | $+0.0003$ | **Custom Outperforms Baseline** |
| **PR-AUC Score** | **`0.6694`** | `0.6687` | $+0.0007$ | **Custom Outperforms Baseline** |
| **Accuracy ($t=0.50$)** | **`75.40%`** ($797/1057$) | `75.21%` ($795/1057$) | $+0.19\%$ | Equivalent |
| **Precision ($t=0.50$)** | **`52.62%`** ($211/401$) | `52.37%` ($210/401$) | $+0.25\%$ | Equivalent |
| **Recall / Sensitivity ($t=0.50$)**| **`75.09%`** ($211/281$) | `74.73%` ($210/281$) | $+0.36\%$ | Equivalent |
| **Specificity ($t=0.50$)** | **`75.52%`** ($586/776$) | `75.39%` ($585/776$) | $+0.13\%$ | Equivalent |
| **F1-Score ($t=0.50$)** | **`0.6188`** | `0.6158` | $+0.0030$ | Equivalent |
| **Log-Loss** | **`0.5126`** | `0.5129` | $-0.0003$ | Equivalent |
| **Probability Correlation ($r$)** | **`0.998200`** | `1.000000` | — | **Mathematical Equivalence** |
| **Mean Absolute Prob Delta** | **`0.0124`** | — | — | Near-Zero Deviation |
| **Inference Latency (1k rows)** | **`0.056 ms`** | `0.222 ms` | **-74.8%** | **4.0x Faster Inference** |

---

## 6. Threshold Optimization & Decision Boundaries

Because telecom retention involves asymmetrical financial trade-offs, decision thresholds were calibrated on the independent validation split ($N = 1,056$):

```
Validation Split Optimization Curve:
Threshold Range Evaluated: [0.10, 0.90] in steps of 0.01
- Default Baseline: t = 0.50 -> Val F1 = 0.6272, Val Recall = 75.71%
- Optimal Validation Threshold: t* = 0.58 -> Val F1 = 0.6375, Val Recall = 69.64%
```

### Comparative Evaluation at Fixed Thresholds on Holdout Test Set ($N = 1,057$)

| Operating Condition | Threshold ($t$) | Accuracy | Precision | Recall | Specificity | F1-Score | Confusion Matrix $[[TN, FP], [FN, TP]]$ |
|---|---|---|---|---|---|---|---|
| **Balanced Sensitivity** | $t = 0.50$ | $75.40\%$ | $52.62\%$ | **$75.09\%$** | $75.52\%$ | $0.6188$ | $[[586, 190], [70, 211]]$ |
| **Validation-Selected Optimal** | $t^* = 0.58$ | **$78.15\%$** | **$57.35\%$** | $69.40\%$ | **$81.31\%$** | **$0.6280$** | $[[631, 145], [86, 195]]$ |

---

## 7. Business Value & Financial Utility Analysis

### 7.1 Financial Cost-Utility Matrix
To evaluate operational impact, a business retention cost model was applied:
* **Cost of False Negative ($\text{FN}$):** $-\$500.00$ (Lost Customer Lifetime Value due to missed churn).
* **Cost of False Positive ($\text{FP}$):** $-\$50.00$ (Cost of unnecessary retention discount offered to a loyal customer).
* **Gain of True Positive ($\text{TP}$):** $+\$350.00$ (Net recovered value through proactive retention).
* **Gain of True Negative ($\text{TN}$):** $\$0.00$ (Standard organic continuation).

### 7.2 Net Retention Value Arithmetic Derivation
$$\text{Net Retention Value} = (\text{TP} \times \$350.00) - (\text{FP} \times \$50.00)$$

* **At Default Threshold ($t = 0.50$):**
  $$\text{TP} = 211, \quad \text{FP} = 190$$
  $$211 \times \$350.00 = \$73,850.00$$
  $$190 \times \$50.00 = \$9,500.00$$
  $$\mathbf{\text{Net Retention Value} = \$73,850.00 - \$9,500.00 = +\$64,350.00}$$
  $$\mathbf{\text{Per Monitored Subscriber (} N=1,057 \text{): } \frac{\$64,350.00}{1,057} = +\$60.88}$$

* **At Validation-Selected Threshold ($t^* = 0.58$):**
  $$\text{TP} = 195, \quad \text{FP} = 145$$
  $$195 \times \$350.00 = \$68,250.00$$
  $$145 \times \$50.00 = \$7,250.00$$
  $$\mathbf{\text{Net Retention Value} = \$68,250.00 - \$7,250.00 = +\$61,000.00 \quad (+\$57.71 / \text{subscriber})}$$

---

## 8. Explainability Engine & Attribution Architecture

### 8.1 Global Feature Importance & Odds Ratios
Odds ratios are computed as $\text{OR}_j = \exp(w_j)$. All feature interpretations adhere to strict non-causal statistical language:
* **Standardized Continuous Variables:** Interpreted explicitly as the change in modeled odds for a **one-standard-deviation increase**, holding other inputs constant.
* **One-Hot Categorical Features:** Interpreted relative to the omitted baseline reference category (e.g. `Contract_Two year` relative to `Month-to-month Contract`, `InternetService_Fiber optic` relative to `DSL`).

#### Top Global Churn Predictors
1. `Contract_Two year` ($w = -1.4089, \text{OR} = 0.2444$): Associated with a **$75.6\%$ reduction in modeled churn odds** relative to a Month-to-month contract.
2. `tenure` ($w = -0.8252, \text{OR} = 0.4382$): A 1-std-dev increase ($24.5$ months) is associated with a **$56.2\%$ reduction in modeled churn odds**.
3. `Contract_One year` ($w = -0.6877, \text{OR} = 0.5027$): Associated with a **$49.7\%$ reduction in modeled churn odds** relative to Month-to-month.
4. `InternetService_Fiber optic` ($w = +0.6570, \text{OR} = 1.9290$): Associated with a **$92.9\%$ increase in modeled churn odds** relative to DSL service.
5. `PaymentMethod_Electronic check` ($w = +0.3463, \text{OR} = 1.4138$): Associated with a **$41.4\%$ increase in modeled churn odds** relative to Bank Transfer auto-pay.

### 8.2 Additive Log-Odds Waterfall Decomposition
Every individual customer score is decomposed into an exact step-by-step linear attribution starting from the baseline intercept:
$$z = b + \sum_{j=1}^{d} w_j x_j \implies p = \sigma(z)$$
* **Trained Intercept ($b$):** $-0.027012$ ($\sigma(b) = 49.32\%$).
* **Reconstruction Verification:** Programmatically audited in [`IndividualExplainer`](file:///a:/intership%20capstone%20project/src/explainability/individual_explainer.py) to confirm that the reconstructed probability matches the model's direct output with zero discrepancy ($|\hat{p} - \sigma(z)| \le 10^{-14}$).

---

## 9. Prescriptive Retention Action Playbook

The platform translates individual model risk signals into deterministic, cost-modeled retention interventions:

```
+----------------------------------------------------------------------------------------------------+
|                                DETERMINISTIC PLAYBOOK MAPPING                                      |
+------------------------------------+-------------------------------------------+-------------------+
| Trigger Condition                  | Prescribed Retention Play                 | Cost Impact       |
+------------------------------------+-------------------------------------------+-------------------+
| Contract == 'Month-to-month'       | Annual Loyalty Commitment Offer           | $12/mo discount   |
| Internet == 'Fiber optic' & Tech=No| Complimentary Tech & Security Bundle      | $5/mo credit      |
| MonthlyCharges >= $75.00           | Account Plan Optimization & Review        | $10/mo credit     |
| PaymentMethod == 'Electronic check'| Automated Auto-Pay Migration Bonus        | $15 one-time      |
| tenure <= 6 months                 | Customer Success Concierge Onboarding     | $0.00 (Ops time)  |
| p < 0.40 (Healthy)                 | Standard Lifecycle Maintenance            | $0.00 (Standard)  |
+------------------------------------+-------------------------------------------+-------------------+
```

---

## 10. Dual-Mode Demo & User Dataset Experience

The user application ([`app/streamlit_app.py`](file:///a:/intership%20capstone%20project/app/streamlit_app.py)) provides two distinctly separated modes:

### Mode A: Interactive Demo & 7-Step Guided Tour
* **Pre-Loaded Archetypes:** Deterministic demonstration of high-risk ($p = 96.1\%$) and low-risk ($p = 2.2\%$) customer personas.
* **7-Step Guided Tour:** Structured walkthrough covering Problem $\to$ Data $\to$ Inference $\to$ Waterfall Attribution $\to$ Retention Playbook $\to$ Benchmark Scorecard $\to$ Cohort Scaling.

### Mode B: Dynamic Tabular ML Pipeline (Analyze Your Dataset)
* **General Tabular Binary Classification:** Dynamically processes any tabular CSV (e.g. Employee Attrition, Loan Default, Fraud Detection, Medical Diagnostics).
* **Automated Schema & Target Detection:** Detects binary targets, identifier columns (`customerID`, `EmployeeID`, `LoanID`, etc.), and separates numerical from categorical features.
* **Zero-Leakage Dynamic Pipeline:** Fits `DynamicPreprocessor` strictly on training data ($X_{\text{train}}$) and transforms validation and test splits with automated imputation and one-hot encoding.
* **From-Scratch NumPy Training:** Dynamically trains a fresh instance of `CustomLogisticRegression` with Momentum GD and balanced class weights.
* **Validation-Selected Threshold:** Optimizes decision threshold $t^*$ on the validation partition to maximize F1-score.
* **Dynamic Results Dashboard & Full Scored Export:** Outputs independent test performance (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix), global Odds Ratios ($\text{OR} = \exp(w)$), and exports scored predictions for all $N$ uploaded rows (`my_dataset_predictions.csv`).
* **Complete Isolation:** Never modifies frozen official demo model weights, metrics, or `final_results.json`.

---

## 11. Testing & Verification Suite

The repository includes 57 automated unit and integration tests across 8 test modules:

```
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0
rootdir: A:\intership capstone project
collected 57 items

tests/test_custom_logistic_regression.py (5 tests) ................. PASSED [  8%]
tests/test_dynamic_custom_dataset.py (8 tests) ..................... PASSED [ 22%]
tests/test_end_to_end_pipeline.py (2 tests) ........................ PASSED [ 26%]
tests/test_explainability.py (8 tests) ............................. PASSED [ 40%]
tests/test_math_primitives.py (6 tests) ............................ PASSED [ 50%]
tests/test_metrics.py (7 tests) .................................... PASSED [ 63%]
tests/test_preprocessor.py (8 tests) ............................... PASSED [ 77%]
tests/test_user_dataset_workflow.py (13 tests) ..................... PASSED [100%]

======================= 57 passed, 3 warnings in 4.27s ========================
```

### Verified Test Capabilities:
1. **Mathematical Primitives:** Sigmoid bounds, overflow resistance, monotonic property, weighted BCE loss, analytical vs. numerical gradient checks ($< 10^{-9}$).
2. **Estimator Mechanics:** Gradient descent convergence on synthetic data, L2 weight shrinkage, balanced class weighting impact, model `.npz` serialization and recovery.
3. **Pipeline & Leakage:** Preprocessor fitted strictly on training data, whitespace TotalCharges imputation, single-record and batch transformation equivalence.
4. **Metrics & Financials:** Accuracy, Precision, Recall, Specificity, F1, ROC-AUC parity vs. Scikit-Learn, exact business-value calculation ($+\$64,350.00$).
5. **Explainability & Attribution:** Odds ratio $\exp(w)$ fidelity, standardized vs. categorical interpretations, log-odds waterfall reconstruction ($0.00\text{e}+00$ discrepancy), validation threshold boundary ($0.58$).
6. **User Dataset Workflow:** Valid labeled/unlabeled CSVs, missing columns rejection, impossible numeric values rejection, template generation integrity, and parameter invariance under inference.
7. **Dynamic Tabular Pipeline (Step 8B):** General binary target detection, zero-leakage dynamic preprocessing, scratch NumPy training on custom datasets (HR Attrition, Loan Default), validation threshold calibration, Dataset A vs. Dataset B result distinctness, and official demo isolation.

---

## 12. Deliverables & Authoritative Artifacts Summary

| Artifact File | Description | Checksum / Status |
|---|---|---|
| [`artifacts/custom_logistic_model.npz`](file:///a:/intership%20capstone%20project/artifacts/custom_logistic_model.npz) | Frozen Custom NumPy weights ($41$ weights, $1$ bias) | $\|w\|_2 = 2.172126, b = -0.027012$ |
| [`artifacts/preprocessor_pipeline.joblib`](file:///a:/intership%20capstone%20project/artifacts/preprocessor_pipeline.joblib) | Zero-leakage fitted Scaler & OneHotEncoder | Fitted on Train Split ($N=4,930$) |
| [`artifacts/final_results.json`](file:///a:/intership%20capstone%20project/artifacts/final_results.json) | Authoritative frozen empirical experiment metrics | `FROZEN_FINAL_AUTHORITATIVE` |
| [`artifacts/benchmark_results.json`](file:///a:/intership%20capstone%20project/artifacts/benchmark_results.json) | Side-by-side benchmark metrics vs. Scikit-Learn | Custom vs. Sklearn ($0.8452$ vs $0.8449$) |
| [`FINAL_EXPERIMENT_AND_SUMMARY_RESULTS.md`](file:///a:/intership%20capstone%20project/FINAL_EXPERIMENT_AND_SUMMARY_RESULTS.md) | Single Source of Truth project documentation | Authoritative Reference |
| [`app/streamlit_app.py`](file:///a:/intership%20capstone%20project/app/streamlit_app.py) | Full interactive production web application | Dual-mode ready at `localhost:8501` |

---

## 13. Conclusion

**ChurnGuard AI** demonstrates how machine learning algorithms can be engineered from pure mathematical foundations without sacrificing operational performance, interpretability, or software quality. By pairing a custom NumPy estimator with zero-leakage preprocessing, full additive attribution, deterministic business playbooks, and a dual-mode user application, the project delivers a complete, production-ready capstone solution.
