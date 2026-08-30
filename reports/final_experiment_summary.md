# ChurnGuard AI — Final Empirical Experiment Summary

**Project Title:** ChurnGuard AI — Explainable Customer Churn Prediction & Retention Intelligence Platform  
**Document Status:** FROZEN & AUTHORITATIVE  
**Single Source of Truth File:** [`artifacts/final_results.json`](file:///a:/intership%20capstone%20project/artifacts/final_results.json)  
**Execution Timestamp:** 2026-08-25T00:05:27 UTC  
**Environment:** Python 3.11.9, NumPy 2.4.4, Scikit-Learn 1.9.0, Pandas 3.0.5, SciPy 1.17.1 (Windows 10/11)  

---

## 1. Dataset Specification

* **Filename:** `telco_customer_churn.csv` ([`data/raw/telco_customer_churn.csv`](file:///a:/intership%20capstone%20project/data/raw/telco_customer_churn.csv))
* **File Size:** 970,457 bytes
* **SHA-256 Checksum:** `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`
* **Total Records ($N$):** **7,043** subscriber records
* **Total Columns:** **21** (1 unique ID column, 1 target column, 19 raw feature columns)
* **Target Distribution (`Churn`):**
  * `No` (Retained): **5,174** customers (**73.46%**)
  * `Yes` (Churned): **1,869** customers (**26.54%**)
  * Imbalance Ratio: $2.768 : 1$
* **Data Hygiene & Missing Value Handling:**
  * Native `NaN` values across all columns: **0**
  * Whitespace strings: Exactly **11 records** in `TotalCharges` contained whitespace `' '`. All 11 records have `tenure == 0` (new subscribers prior to first billing cycle). Sanitized deterministically to float `0.0`.

---

## 2. Experimental Protocol & Data Partitioning

A strict, zero-data-leakage 3-way stratified partition was applied:

$$\text{Total Dataset } (N=7,043) = \text{Train } (70.0\%) + \text{Validation } (15.0\%) + \text{Holdout Test } (15.0\%)$$

| Partition | Sample Count | % of Total | Retained (`No`) | Churned (`Yes`) | Churn Rate | Role in Protocol |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **Training Split** | **4,930** | 70.00% | 3,622 | 1,308 | **26.5314%** | Preprocessor fitting & model optimization |
| **Validation Split** | **1,056** | 14.99% | 776 | 280 | **26.5152%** | Early stopping & decision threshold selection |
| **Holdout Test Split** | **1,057** | 15.01% | 776 | 281 | **26.5847%** | Single final unbiased evaluation |

* **Zero Leakage Guarantee:** Customer ID overlap between all sets is **0**. Preprocessor scalers and encoders are fitted strictly on `train_df`. Validation/Test splits are only transformed.

---

## 3. Preprocessing Pipeline

Implemented in [`src/data/preprocessor.py`](file:///a:/intership%20capstone%20project/src/data/preprocessor.py):
* **Continuous Scaling:** 11 continuous features scaled via `StandardScaler(with_mean=True, with_std=True)` fitted on training data.
* **Categorical Encoding:** 30 dummy columns generated via `OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)`.
* **Type Normalization:** Centralized `_prepare_dataframe()` enforces string casting on all categorical columns and numeric float conversion with training median imputation fallback on numerical columns.
* **Transformed Matrix Dimensions:**
  * $X_{\text{train}}$: **(4930, 41)**
  * $X_{\text{val}}$: **(1056, 41)**
  * $X_{\text{test}}$: **(1057, 41)**

---

## 4. Feature Engineering

9 domain-specific behavioral, contractual, and service features engineered in [`src/features/engineering.py`](file:///a:/intership%20capstone%20project/src/features/engineering.py):

1. `tenure_cohort`: Binned lifecycle stages (`0-12m`, `13-24m`, `25-48m`, `49-72m`).
2. `monthly_to_total_ratio`: $\frac{\text{MonthlyCharges}}{\text{TotalCharges} + 1.0}$ (billing intensity indicator).
3. `monthly_charge_discrepancy`: $\text{MonthlyCharges} - \frac{\text{TotalCharges}}{\text{tenure} + 1.0}$ (bill shock / recent price increase).
4. `total_services_count`: Count of active subscription add-ons (Range: $0–8$).
5. `protection_services_count`: Count of security, backup, device protection, tech support add-ons (Range: $0–4$).
6. `streaming_services_count`: Count of entertainment add-ons (Range: $0–2$).
7. `has_internet`: Boolean indicator for active DSL/Fiber optic internet service.
8. `is_solo_senior`: Interaction flag for senior citizen with no partner and no dependents.
9. `high_risk_fiber_m2m`: Critical risk interaction flag (`Contract == 'Month-to-month'` & `InternetService == 'Fiber optic'`).

---

## 5. Custom Logistic Regression Model Configuration

Implemented in pure NumPy in [`src/models/custom_logistic_regression.py`](file:///a:/intership%20capstone%20project/src/models/custom_logistic_regression.py):
* **Class Weighting:** $w_0 = \frac{4930}{2 \times 3622} \approx 0.6806$, $w_1 = \frac{4930}{2 \times 1308} \approx 1.8846$.
* **Learning Rate ($\alpha$):** `0.05`
* **Mini-Batch Size ($B$):** `64` samples
* **Momentum ($\beta$):** `0.90`
* **$L_2$ Regularization ($\lambda$):** `0.01`
* **Early Stopping:** Monitored validation loss with `patience = 50` epochs and parameter rollback.
* **Training Telemetry:**
  * Total Epochs Executed: **175**
  * Best Validation Loss: **`0.47928`** achieved at **Epoch 125**
  * Final Train Loss: **`0.47640`**
  * Final Bias: $b = -0.0270$
  * Weight Vector $L_2$ Norm: $\|w\|_2 = 2.1721$

---

## 6. Threshold Selection Protocol

* **Search Method:** Grid search $t \in [0.01, 0.99]$ with step size $0.01$ evaluated **strictly on the validation split** ($N=1,056$).
* **Optimization Objective:** Maximize Validation $F_1$-score.
* **Validation Results:** Optimal threshold found at **$t^* = 0.5800$** (Validation $F_1 = \mathbf{0.6415}$, Precision $= 55.64\%$, Recall $= 75.71\%$).

---

## 7. Final Holdout Test Set Evaluation Results

Evaluated on the **untouched holdout test partition** ($N=1,057$, 281 actual churners, 776 retained):

### A. Performance at Default Threshold ($t = 0.50$)

| Evaluation Metric | Custom NumPy Model | Scikit-Learn Benchmark | Empirical Delta ($\Delta$) |
|---|:---:|:---:|:---:|
| **ROC-AUC Score** | **`0.8452`** | **`0.8449`** | **`+0.0003`** |
| **PR-AUC (Average Precision)** | **`0.6694`** | **`0.6687`** | **`+0.0007`** |
| **Accuracy** | **`75.40%`** | **`74.74%`** | **`+0.66%`** |
| **Recall / Sensitivity** | **`75.09%`** ($211/281$) | **`76.16%`** ($214/281$) | `-1.07%` |
| **Precision** | **`52.62%`** | **`51.69%`** | **`+0.93%`** |
| **Specificity (TNR)** | **`75.52%`** ($586/776$) | **`74.23%`** ($576/776$) | **`+1.29%`** |
| **F1-Score** | **`0.6188`** | **`0.6158`** | **`+0.0030`** |
| **Inference Latency (per 1k)** | **`0.056 ms`** | **`0.222 ms`** | **`4.0x Faster`** |

$$\text{Custom Confusion Matrix } (t=0.50) = \begin{bmatrix} \text{TN}=586 & \text{FP}=190 \\ \text{FN}=70 & \text{TP}=211 \end{bmatrix}$$

$$\text{Scikit-Learn Confusion Matrix } (t=0.50) = \begin{bmatrix} \text{TN}=576 & \text{FP}=200 \\ \text{FN}=67 & \text{TP}=214 \end{bmatrix}$$

---

### B. Performance at Validation-Selected Optimal Threshold ($t^* = 0.58$)

| Evaluation Metric | Custom NumPy Model ($t^* = 0.58$) | Delta vs. $t=0.50$ |
|---|:---:|:---:|
| **Accuracy** | **`78.15%`** | $+2.75\%$ |
| **Precision** | **`57.35%`** | $+4.73\%$ |
| **Recall / Sensitivity** | **`69.40%`** ($195/281$) | $-5.69\%$ |
| **Specificity (TNR)** | **`81.31%`** ($631/776$) | $+5.79\%$ |
| **F1-Score** | **`0.6280`** | **`+0.0092`** |

$$\text{Custom Confusion Matrix } (t^*=0.58) = \begin{bmatrix} \text{TN}=631 & \text{FP}=145 \\ \text{FN}=86 & \text{TP}=195 \end{bmatrix}$$

---

## 8. Mathematical Fidelity & Equivalence

* **Prediction Probability Pearson Correlation:** **$r = 0.998200$**
* **Weight Vector Cosine Similarity:** **$0.5585$**
* **Weight Pearson Correlation:** **$0.5479$**
* **Single-Customer Log-Odds Waterfall Reconstruction Discrepancy ($|\hat{p} - \sigma(b + w^T x)|$):** **`0.00e+00` (Exact Match)**.

---

## 9. Illustrative Business Value & Retention Cost Model

> [!NOTE]
> All financial valuations represent **illustrative business simulation assumptions** for capstone ROI modeling.

* **Assumed Customer Lifetime Value (LTV):** $\$500.00$
* **Assumed Cost of Missed Churn (False Negative):** $-\$500.00$
* **Assumed Cost of Retention Intervention (False Positive):** $-\$50.00$
* **Assumed Net Recovered Value per Caught Churner (True Positive):** $+\$350.00$
* **Net Retention Value on Holdout Test Cohort ($N=1,057$ at Default $t=0.50$):**
  $$\text{Net Retention Value} = (\text{TP} \times \text{Net Value per Retention}) - (\text{FP} \times \text{Cost of Intervention})$$
  $$\text{Net Retention Value} = (211 \times \$350) - (190 \times \$50) = \$73,850 - \$9,500 = \mathbf{+\$64,350.00}$$
  (Yielding $+\$60.88$ net value per customer monitored: $\$64,350 / 1,057 \approx \$60.88$).

---

## 10. Known Limitations

1. **Static Dataset Snapshot:** IBM Telco dataset reflects fixed historical customer profiles without temporal drift telemetry.
2. **First-Order Optimization Speed vs C++:** Pure Python/NumPy first-order gradient descent requires 522ms for 175 epochs compared to 35ms for compiled C++ L-BFGS in Scikit-Learn.
3. **Illustrative Financial Multipliers:** Financial figures depend on specific telco billing structure and customer acquisition costs.

---

## 11. Reproducibility & Deterministic Audit

* **Random Seed:** `42` enforced across DataLoader, Preprocessor, Optimizer, and Scikit-Learn.
* **Deterministic Verification:** Independent re-execution produced **`0.00e+00`** weight discrepancy and identical metric outputs.
* **Validation-Selected Optimal Threshold:** **`0.5800` vs. `0.5800`**
* **Test $F_1$-Score at Default Threshold ($t=0.50$):** **`0.61876833` vs. `0.61876833`**
* **Test $F_1$-Score at Validation-Selected Threshold ($t^*=0.58$):** **`0.6280` vs. `0.6280`**
* **Test Suite:** **28 / 28 Tests Passed (100%) in 6.05s** via `pytest -v`.
