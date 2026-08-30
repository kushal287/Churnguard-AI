# 🛡️ ChurnGuard AI — Final Empirical Experiment & Summary Results

**Project Title:** ChurnGuard AI — Explainable Customer Churn Prediction & Retention Intelligence Platform  
**Project Context:** Internship Machine Learning Major Capstone Project  
**Implementation Standard:** 100% Pure NumPy Custom Logistic Regression (Primary Model) vs. Scikit-Learn (Comparative Benchmark Only)  
**Experiment Status:** **FROZEN & AUTHORITATIVELY PERSISTED**  
**Single Source of Truth:** [`artifacts/final_results.json`](file:///a:/intership%20capstone%20project/artifacts/final_results.json)  
**Execution Timestamp:** 2026-08-25T00:05:27 UTC  
**Environment:** Python 3.11.9, NumPy 2.4.4, Scikit-Learn 1.9.0, Pandas 3.0.5, SciPy 1.17.1, Streamlit 1.62.0 (Windows 10/11)  

---

## 1. Executive Summary & Verification Scorecard

**ChurnGuard AI** is an explainable machine learning platform engineered to detect subscriber attrition risks and prescribe targeted retention actions before churn occurs. Built from scratch without high-level estimator dependencies, the system satisfies all mathematical, engineering, explainability, and deployment requirements of the Major Capstone Project.

| Evaluation Dimension | Scope of Verification | Actual Empirical Measurement | Status |
|---|---|---|:---:|
| **Custom NumPy Classifier** | Pure NumPy Logistic Regression (Sigmoid, Weighted BCE, Gradient Descent) | Zero library estimator calls in primary model; analytical gradients verified with **$8.24 \times 10^{-10}$** relative error. | **PASS** |
| **Data Hygiene & Integrity** | 7,043 customer records, 0 NaNs, 11 whitespace `TotalCharges` imputed | Scaler & encoder fitted strictly on 70% train split; test data transformation remains **100% invariant**. | **PASS** |
| **Feature Engineering** | 9 domain behavioral & contractual features | Expanded raw 21 columns to **41 total transformed dimensions** with zero leakage. | **PASS** |
| **Class Imbalance Handling** | Balanced inverse frequency sample weights ($w_0, w_1$) | Increased minority churn recall on holdout test set from **48.04% to 75.09%**. | **PASS** |
| **Holdout Test Performance** | Untouched test split evaluation ($N=1,057$) | **ROC-AUC: `0.8452`**, **PR-AUC: `0.6694`**, **Recall: `75.09%`**, **F1: `0.6188`** ($t=0.50$). | **PASS** |
| **Validation Threshold Tuning** | Validation-only threshold optimization ($N=1,056$) | Optimal validation threshold **$t^* = 0.58$** achieves **`78.15%` Test Accuracy** & **`0.6280` Test F1**. | **PASS** |
| **Scikit-Learn Benchmark** | Fair comparative benchmark on identical matrices | Prediction probability Pearson correlation: **`r = 0.9982`**; Custom NumPy inference **4.0x faster**. | **PASS** |
| **Explainability Parity** | Global Odds Ratios & Customer Log-Odds Waterfall | Direct model probability matches reconstructed log-odds to **`0.00e+00` error**. | **PASS** |
| **Automated Test Suite** | Pytest unit, integration, and regression tests | **28 / 28 Tests Passed (100%)** in 6.05 seconds. | **PASS** |
| **Interactive Platform** | 5-Page Streamlit Production Intelligence Platform | Executive View, Live Simulator, Benchmark Arena, Batch Scoring, and Report View fully operational. | **PASS** |

---

## 2. Dataset Specification & Provenance

* **Dataset Identifier:** Canonical IBM Telco Customer Churn Dataset
* **Raw File Location:** [`data/raw/telco_customer_churn.csv`](file:///a:/intership%20capstone%20project/data/raw/telco_customer_churn.csv)
* **File Size:** 970,457 bytes
* **SHA-256 Checksum:** `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`
* **Total Records ($N$):** **7,043** subscriber accounts
* **Total Columns:** **21** (1 unique identifier `customerID`, 1 binary target `Churn`, 19 customer attributes)
* **Target Distribution (`Churn`):**
  * `No` (Retained): **5,174** customers (**73.46%**)
  * `Yes` (Churned): **1,869** customers (**26.54%**)
  * Class Imbalance Ratio: $2.768 : 1$
* **Data Hygiene & Preprocessing Actions:**
  * Native `NaN` values across all columns: **0**
  * Whitespace strings: Exactly **11 records** in `TotalCharges` contained whitespace `' '`. All 11 records have `tenure == 0` (new subscribers who joined in the current billing month). Coerced deterministically to float `0.0`.

---

## 3. Data Partitioning Protocol (Zero Data Leakage)

A strict, stratified 3-way partition was applied using a fixed seed (`random_state = 42`):

$$\text{Total Dataset } (N = 7,043) = \text{Train } (70.0\%) + \text{Validation } (15.0\%) + \text{Holdout Test } (15.0\%)$$

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Canonical Dataset (7,043 rows)                     │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ Stratified Split (Seed 42)
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
│  Train Partition │       │   Val Partition  │       │  Test Partition  │
│  4,930 rows (70%)│       │  1,056 rows (15%)│       │  1,057 rows (15%)│
│  Churn: 26.53%   │       │  Churn: 26.52%   │       │  Churn: 26.58%   │
└────────┬─────────┘       └────────┬─────────┘       └────────┬─────────┘
         │                          │                          │
         ▼                          ▼                          ▼
  Fit Preprocessor          Monitor Early Stopping      Single Final Unbiased
  Fit Custom NumPy LR       Tune Decision Threshold        Model Evaluation
```

* **Customer ID Overlap:** Train $\cap$ Val = **0**, Train $\cap$ Test = **0**, Val $\cap$ Test = **0**.
* **Zero Leakage Verification:** Preprocessing parameters (StandardScaler `mean_`, `scale_` and OneHotEncoder categories) are fitted strictly on `train_df`. Transformations of validation and test sets never modify preprocessor state.

---

## 4. Domain Feature Engineering

[`src/features/engineering.py`](file:///a:/intership%20capstone%20project/src/features/engineering.py) constructs 9 domain-specific behavioral, contractual, and service features:

1. **`tenure_cohort`:** Categorical lifecycle stages binned into `0-12m` (New), `13-24m` (Early), `25-48m` (Established), and `49-72m` (Loyal).
2. **`monthly_to_total_ratio`:** $\frac{\text{MonthlyCharges}}{\text{TotalCharges} + 1.0}$ (Captures early billing intensity and recent bill shock).
3. **`monthly_charge_discrepancy`:** $\text{MonthlyCharges} - \frac{\text{TotalCharges}}{\text{tenure} + 1.0}$ (Detects rate inflation relative to historical average).
4. **`total_services_count`:** Count of active services active across 8 add-on products (Range: $0–8$).
5. **`protection_services_count`:** Sum of security and support products (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`).
6. **`streaming_services_count`:** Sum of entertainment products (`StreamingTV`, `StreamingMovies`).
7. **`has_internet`:** Binary indicator for active DSL or Fiber Optic internet connection.
8. **`is_solo_senior`:** Interaction flag for senior citizens living without partners and dependents.
9. **`high_risk_fiber_m2m`:** Critical risk interaction flag (`Contract == 'Month-to-month'` and `InternetService == 'Fiber optic'`).

**Final Encoded Feature Matrix Dimension ($d$):** **41 features** ($11 \text{ numerical features} + 30 \text{ one-hot encoded categories}$).

---

## 5. Custom Logistic Regression Mathematical Architecture

The primary classifier [`CustomLogisticRegression`](file:///a:/intership%20capstone%20project/src/models/custom_logistic_regression.py) is implemented strictly using NumPy matrix operations:

### A. Piecewise Numerically Stable Sigmoid
$$\sigma(z) = \begin{cases} \frac{1}{1 + e^{-z}} & \text{if } z \ge 0 \\ \frac{e^z}{1 + e^z} & \text{if } z < 0 \end{cases} \quad \text{where } z = \text{clip}(Xw + b, -500, 500)$$
* *Verification:* $\sigma(-1000) = 7.12 \times 10^{-218}$, $\sigma(0) = 0.500000$, $\sigma(1000) = 1.000000$. Eliminates NaN and Inf underflow/overflow risks.

### B. Weighted Binary Cross-Entropy Loss with $L_2$ Regularization
$$J(w, b) = -\frac{1}{m} \sum_{i=1}^m v_i \left[ y_i \ln(p_i + \epsilon) + (1 - y_i) \ln(1 - p_i + \epsilon) \right] + \frac{\lambda}{2m} \sum_{j=1}^d w_j^2$$
where $\epsilon = 10^{-15}$, $\lambda = 0.01$, and $v_i$ denotes sample weights derived from inverse class frequencies:
$$w_0 = \frac{N}{2 N_0} = \frac{4930}{2 \times 3622} \approx 0.6806, \quad w_1 = \frac{N}{2 N_1} = \frac{4930}{2 \times 1308} \approx 1.8846$$

### C. Analytical Gradients & Finite-Difference Verification
$$\nabla_w J = \frac{1}{m} X^T \left( v \odot (\hat{y} - y) \right) + \frac{\lambda}{m} w, \quad \nabla_b J = \frac{1}{m} \sum_{i=1}^m v_i (\hat{y}_i - y_i)$$
* *Finite-Difference Gradient Check ($\epsilon = 1.0 \times 10^{-6}$):*
  * Weight Relative Error: **`8.2404e-10`** (Passed $< 10^{-5}$)
  * Bias Relative Error: **`6.8467e-09`** (Passed $< 10^{-5}$)

### D. Mini-Batch Momentum Optimizer & Early Stopping
* Mini-batch size: $B = 64$, Momentum: $\beta = 0.90$, Learning rate: $\alpha = 0.05$.
* Terminated at **Epoch 175** with early stopping parameter rollback to **Epoch 125** ($\text{Best Val Loss} = 0.47928$, $\text{Final Train Loss} = 0.47640$).

---

## 6. Authoritative Evaluation Protocol & Test Set Benchmark

### A. Strict Evaluation Flow & Threshold Discipline
To ensure zero data leakage and unbiased evaluation, the experimental workflow strictly follows three distinct phases:
1. **TRAINING PHASE ($N = 4,930$, 70% split):** Preprocessor parameters are fitted and Custom NumPy Logistic Regression weights are optimized.
2. **VALIDATION PHASE ($N = 1,056$, 15% split):** Early stopping monitors loss to prevent overfitting. Decision threshold sweep $t \in [0.01, 0.99]$ is conducted strictly on validation data to select optimal threshold **$t^* = 0.5800$** ($\text{Validation } F_1 = \mathbf{0.6415}$).
3. **HOLD-OUT TEST PHASE ($N = 1,057$, 15% split):** The trained model and locked thresholds ($t=0.50$ and validation-selected $t^*=0.58$) are evaluated once on untouched test data. The test partition was never used for threshold selection.

---

### B. Performance Scorecard: Custom NumPy vs. Scikit-Learn Baseline (Holdout Test Set: $N = 1,057$)

| Performance Metric | Custom NumPy LR (Primary Model) | Scikit-Learn LR (Baseline Benchmark) | Empirical Delta ($\Delta$) | Significance / Dataset Role |
|---|:---:|:---:|:---:|---|
| **Test ROC-AUC Score** | **`0.8452`** | **`0.8449`** | **`+0.0003`** | Test Split ($N=1,057$) Discrimination |
| **Test PR-AUC (Average Precision)** | **`0.6694`** | **`0.6687`** | **`+0.0007`** | Test Split ($N=1,057$) Precision-Recall Area |
| **Test Accuracy at Default $t=0.50$** | **`75.40%`** | **`74.74%`** | **`+0.66%`** | Test Split ($N=1,057$) |
| **Test Recall / Sensitivity at $t=0.50$** | **`75.09%`** ($211/281$) | **`76.16%`** ($214/281$) | `-1.07%` | Test Split ($N=1,057$) |
| **Test Precision at $t=0.50$** | **`52.62%`** | **`51.69%`** | **`+0.93%`** | Test Split ($N=1,057$) |
| **Test Specificity / TNR at $t=0.50$** | **`75.52%`** ($586/776$) | **`74.23%`** ($576/776$) | **`+1.29%`** | Test Split ($N=1,057$) |
| **Test F1-Score at Default $t=0.50$** | **`0.6188`** | **`0.6158`** | **`+0.0030`** | Test Split ($N=1,057$) |
| **Validation F1-Score at Optimal $t^*=0.58$** | **`0.6415`** | — | — | **Validation Split Only ($N=1,056$)** |
| **Test F1-Score at Validation-Selected $t^*=0.58$** | **`0.6280`** | **`0.6286`** | `-0.0006` | **Locked Test Evaluation ($N=1,057$)** |
| **Test Prediction Probability Correlation ($r$)** | **`0.998200`** | `1.000000` | **Mathematical Parity** | Holdout Test Set Output Alignment |
| **Inference Latency (per 1,000 samples)** | **`0.056 ms`** | **`0.222 ms`** | **`4.0x Faster`** | Vectorized Pure NumPy Evaluation |
| **Training Duration (175 Epochs)** | **`522.0 ms`** | **`35.2 ms`** | `+486.8 ms` | Python Gradient Descent vs C++ L-BFGS |

---

### C. Test Set Confusion Matrix Evidence (Holdout Test Split: $N = 1,057$)

#### 1. Custom Model at Default Threshold ($t = 0.50$ on Test Set):
$$\begin{bmatrix} \text{True Negatives (Retained Correct)} = \mathbf{586} & \text{False Positives (False Alarms)} = \mathbf{190} \\ \text{False Negatives (Missed Churners)} = \mathbf{70} & \text{True Positives (Caught Churners)} = \mathbf{211} \end{bmatrix}$$

#### 2. Custom Model at Validation-Selected Optimal Threshold ($t^* = 0.58$ on Test Set):
$$\begin{bmatrix} \text{True Negatives (Retained Correct)} = \mathbf{631} & \text{False Positives (False Alarms)} = \mathbf{145} \\ \text{False Negatives (Missed Churners)} = \mathbf{86} & \text{True Positives (Caught Churners)} = \mathbf{195} \end{bmatrix}$$

#### 3. Scikit-Learn Baseline Model ($t = 0.50$ on Test Set):
$$\begin{bmatrix} \text{True Negatives (Retained Correct)} = \mathbf{576} & \text{False Positives (False Alarms)} = \mathbf{200} \\ \text{False Negatives (Missed Churners)} = \mathbf{67} & \text{True Positives (Caught Churners)} = \mathbf{214} \end{bmatrix}$$

---

## 7. Dual-Layer Explainability & Retention Playbook

### A. Global Odds Ratios ($\text{OR} = e^{w_j}$)
* **Top 5 Churn Risk Drivers:**
  1. `TotalCharges` (Standardized): $\text{OR} = 1.6567$ ($+65.7\%$ higher churn odds per unit increase)
  2. `PaperlessBilling_Yes`: $\text{OR} = 1.4604$ ($+46.0\%$ higher churn odds)
  3. `PaymentMethod_Electronic check`: $\text{OR} = 1.4138$ ($+41.4\%$ higher churn odds)
  4. `monthly_charge_discrepancy`: $\text{OR} = 1.3771$ ($+37.7\%$ higher churn odds)
  5. `high_risk_fiber_m2m`: $\text{OR} = 1.3032$ ($+30.3\%$ higher churn odds)

* **Top 5 Retention Anchors (Protective Features):**
  1. `Contract_Two year`: $\text{OR} = 0.2444$ (**$-75.6\%$ reduction in churn odds**)
  2. `tenure` (Standardized): $\text{OR} = 0.4255$ (**$-57.5\%$ reduction in churn odds**)
  3. `Contract_One year`: $\text{OR} = 0.5283$ (**$-47.2\%$ reduction in churn odds**)
  4. `PhoneService_Yes`: $\text{OR} = 0.6746$ (**$-32.5\%$ reduction in churn odds**)
  5. `tenure_cohort_13-24m`: $\text{OR} = 0.7322$ (**$-26.8\%$ reduction in churn odds**)

### B. Single-Customer Mathematical Waterfall Reconstruction
$$\ln\left(\frac{p}{1-p}\right) = z = b + \sum_{j=1}^d w_j x_j \implies p = \sigma(z)$$
* Tested on Test Customer 0: Direct Probability $= 0.54828953$, Reconstructed Probability $= 0.54828953$, Discrepancy $= \mathbf{0.00 \times 10^{-16}}$ (**Exact Match**).

### C. Prescriptive Retention Action Playbook
1. **Annual Loyalty Lock-in:** For Month-to-Month customers with $>50\%$ risk $\rightarrow$ Offer 15% discount for a 1-year contract.
2. **TechSupport Concierge:** For Fiber optic subscribers without support $\rightarrow$ 3 months complimentary 24/7 dedicated support.
3. **Auto-Pay Migration:** For Electronic check users $\rightarrow$ One-time $\$15$ credit for switching to ACH/Credit Card auto-pay.
4. **Early-Tenure Check-in:** For tenure $\le 6$ months in critical risk $\rightarrow$ Proactive success manager outreach.

---

## 8. Illustrative Business Cost-Utility Model

> [!NOTE]
> All financial figures represent **illustrative business simulation assumptions** for capstone ROI modeling.

* **Assumed Customer Lifetime Value (LTV):** $\$500.00$
* **Assumed Cost of Missed Churn (False Negative):** $-\$500.00$
* **Assumed Cost of Unnecessary Incentive (False Positive):** $-\$50.00$
* **Assumed Net Recovered Value per Caught Churner (True Positive):** $+\$350.00$
* **Net Financial Impact on Holdout Test Cohort ($N = 1,057$ at Default $t = 0.50$):**
  $$\text{Net Retention Value} = (\text{TP} \times \text{Net Value per Retention}) - (\text{FP} \times \text{Cost of Intervention})$$
  $$\text{Net Retention Value} = (211 \times \$350) - (190 \times \$50) = \$73,850 - \$9,500 = \mathbf{+\$64,350.00}$$
  (Yielding $+\$60.88$ net value per customer monitored: $\$64,350 / 1,057 \approx \$60.88$).

---

## 9. Automated Testing & Verification Summary

Executed `pytest -v`:
* **Total Test Cases:** **28 / 28 Passed (100%) in 6.05 seconds**.
* **Test Suite Breakdown:**
  * [`tests/test_math_primitives.py`](file:///a:/intership%20capstone%20project/tests/test_math_primitives.py): Sigmoid bounds $[-1000, 1000]$, BCE loss boundaries, analytical vs numerical gradient checking ($10^{-10}$ error).
  * [`tests/test_custom_logistic_regression.py`](file:///a:/intership%20capstone%20project/tests/test_custom_logistic_regression.py): Convergence, L2 shrinkage, class weights recall boost, model `.npz` serialization recovery.
  * [`tests/test_preprocessor.py`](file:///a:/intership%20capstone%20project/tests/test_preprocessor.py): Zero-leakage scaler isolation, missing values, CSV integer type coercion, unknown categories handling, single vs batch transform equivalence.
  * [`tests/test_metrics.py`](file:///a:/intership%20capstone%20project/tests/test_metrics.py): Exact parity of custom NumPy metrics vs `sklearn.metrics`.
  * [`tests/test_end_to_end_pipeline.py`](file:///a:/intership%20capstone%20project/tests/test_end_to_end_pipeline.py): Master training pipeline execution, figure generation, and single-customer explainability inference.

---

## 10. Reproducibility & Deterministic Verification

Two full independent pipeline training cycles executed with fixed `random_state = 42`:
* **Max Weight Discrepancy ($\max |w_1 - w_2|$):** **`0.00e+00`**
* **Bias Discrepancy ($|b_1 - b_2|$):** **`0.00e+00`**
* **Validation-Selected Optimal Threshold:** **`0.5800` vs. `0.5800`**
* **Test $F_1$-Score at Default Threshold ($t=0.50$):** **`0.61876833` vs. `0.61876833`**
* **Test $F_1$-Score at Validation-Selected Threshold ($t^*=0.58$):** **`0.6280` vs. `0.6280`**
* **Result:** **100% Bitwise Deterministic Reproducibility**.

---

### Final Project Deliverables Map

* **Authoritative Results Payload:** [`artifacts/final_results.json`](file:///a:/intership%20capstone%20project/artifacts/final_results.json)
* **Trained Custom Model Weights:** [`artifacts/custom_logistic_model.npz`](file:///a:/intership%20capstone%20project/artifacts/custom_logistic_model.npz)
* **Serialized Preprocessing Pipeline:** [`artifacts/preprocessor_pipeline.joblib`](file:///a:/intership%20capstone%20project/artifacts/preprocessor_pipeline.joblib)
* **Comprehensive Analytical Report:** [`reports/capstone_analytical_report.md`](file:///a:/intership%20capstone%20project/reports/capstone_analytical_report.md)
* **High-Resolution Figures:** [`reports/figures/`](file:///a:/intership%20capstone%20project/reports/figures/)
* **Streamlit Web Application:** [`app/streamlit_app.py`](file:///a:/intership%20capstone%20project/app/streamlit_app.py) (`streamlit run app/streamlit_app.py`)
