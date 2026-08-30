# 🛡️ ChurnGuard AI — Explainable Customer Churn Prediction & Retention Intelligence Platform

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/numpy-pure%20math-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-light%20enterprise%20ui-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/tests-60%2F60%20passing-success.svg)](https://pytest.org/)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.8452%20(Parity)-brightgreen.svg)](https://github.com/kushal287/Churnguard-AI)
[![Mathematical Equivalence](https://img.shields.io/badge/Pearson%20r-0.9982-blueviolet.svg)](https://github.com/kushal287/Churnguard-AI)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**An enterprise-grade, explainable machine learning platform engineered from first principles in pure NumPy.**  
*Zero Data Leakage Preprocessing • Scikit-Learn Parity Audit • Exact Log-Odds Waterfall Explainability • Dual-Mode Production UI*

[Overview](#-overview) • [Key Features](#-key-features) • [Mathematical Formulation](#-mathematical-formulation) • [Benchmark Parity](#-scientific-benchmark-parity) • [Dual-Mode Architecture](#-dual-mode-architecture) • [Getting Started](#-getting-started) • [Repository Structure](#-system-architecture)

</div>

---

## 🌟 Overview

**ChurnGuard AI** is an end-to-end, production-ready machine learning intelligence system designed to predict customer attrition risk in advance and prescribe targeted, cost-effective retention actions before service cancellation.

Developed as a **Major Capstone Machine Learning Project**, the platform combines:
1. A **first-principles mathematical implementation of Logistic Regression** built entirely from scratch in pure NumPy (featuring analytical gradient derivation, weighted binary cross-entropy, L2 weight decay, and mini-batch momentum gradient descent).
2. A **strict zero-leakage preprocessing and feature engineering pipeline** fitted strictly on training data partitions.
3. A **scientific side-by-side benchmark** against Scikit-Learn's L-BFGS optimizer, proving mathematical parity ($0.8452$ vs $0.8449$ ROC-AUC, $r = 0.998200$) with a **4x faster inference speedup**.
4. **100% transparent log-odds waterfall attribution** ($z = b + \sum w_j x_j \implies p = \sigma(z)$) with zero mathematical discrepancy ($0.00\text{e}+00$).
5. A **modern Light Enterprise AI analytics dashboard** with dual operating modes: a guided interactive demonstration on canonical Telco data (Mode A) and a dynamic custom tabular ML engine for general binary classification datasets (Mode B).

---

## ✨ Key Features

| Capability | Technical Description | Business & Engineering Value |
|---|---|---|
| **Pure NumPy ML Engine** | Logistic Regression implemented from scratch with Sigmoid activation, weighted log loss, and momentum GD. | Complete transparency, zero black-box dependencies for model training or inference. |
| **Zero Data Leakage Pipeline** | `StandardScaler` and `OneHotEncoder` parameters fitted strictly on training splits ($70\%$). Imputation and transformations strictly isolated. | Ensures scientific integrity, reproducible evaluations, and true generalization. |
| **Audited Benchmark Parity** | Evaluated on holdout test partition ($N = 1,057$) side-by-side against Scikit-Learn `LogisticRegression(solver='lbfgs')`. | **$0.8452$ vs $0.8449$ ROC-AUC**, $r = 0.9982$ probability correlation, and **4.0x faster latency** ($0.056\text{ ms}$ vs $0.222\text{ ms}$ per 1k). |
| **Quantified Financial Impact** | Cost-utility matrix optimizing threshold $t^* = 0.58$ against false positives ($-\$50$) and true positives ($+\$350$). | Delivers **$+\$64,350.00$** in net retention savings ($+\$60.88$ per monitored subscriber on the test split). |
| **Exact Log-Odds Explainability** | Direct additive decomposition of logits into feature-level risk drivers and protective factors. | Complete compliance, zero approximation discrepancy, fully auditable decisions. |
| **Dynamic Tabular Engine (Mode B)** | Ingests arbitrary tabular CSVs (HR Attrition, Loan Default, Fraud, Medical Diagnosis). Detects schemas, trains fresh NumPy model, and exports predictions. | Generalizes first-principles ML architecture to any real-world tabular binary classification problem. |
| **Comprehensive Test Suite** | 60 automated unit and integration tests passing cleanly across 9 test suites (`pytest -v`). | Guaranteed regression prevention, mathematical gradient verification, and model isolation. |

---

## 📐 Mathematical Formulation

### 1. Vectorized Hypothesis & Sigmoid Activation
Given input feature matrix $\mathbf{X} \in \mathbb{R}^{N \times d}$, weight vector $\mathbf{w} \in \mathbb{R}^d$, and bias scalar $b \in \mathbb{R}$:

$$\mathbf{z} = \mathbf{X}\mathbf{w} + b$$

$$\hat{\mathbf{y}} = \sigma(\mathbf{z}) = \frac{1}{1 + e^{-\mathbf{z}}}$$

*Numerically stable implementation using piecewise clipping:*
$$\sigma(z) = \begin{cases} \frac{1}{1 + e^{-z}} & \text{if } z \ge 0 \\ \frac{e^z}{1 + e^z} & \text{if } z < 0 \end{cases}$$

### 2. Weighted Binary Cross-Entropy Loss with L2 Regularization
To address class imbalance (minority churn rate $\approx 26.5\%$), class weights $w_0$ and $w_1$ are computed via inverse class frequency:

$$w_k = \frac{N}{2 \cdot N_k}, \quad k \in \{0, 1\}$$

The regularized objective function is defined as:

$$\mathcal{L}(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^N \Big[ w_1 y_i \ln(\hat{y}_i + \epsilon) + w_0 (1 - y_i) \ln(1 - \hat{y}_i + \epsilon) \Big] + \frac{\lambda}{2} \|\mathbf{w}\|_2^2$$

### 3. Analytical Gradient Derivation
The exact parameter gradients with respect to $\mathbf{w}$ and $b$ are derived analytically:

$$\mathbf{e}_i = \hat{y}_i - y_i$$

$$s_i = w_1 y_i + w_0 (1 - y_i)$$

$$\nabla_{\mathbf{w}} \mathcal{L} = \frac{1}{N} \mathbf{X}^T (\mathbf{s} \odot \mathbf{e}) + \lambda \mathbf{w}$$

$$\nabla_b \mathcal{L} = \frac{1}{N} \sum_{i=1}^N s_i e_i$$

### 4. Mini-Batch Momentum Gradient Descent
Parameters are updated iteratively with momentum velocity $\mathbf{v}$ and momentum coefficient $\beta = 0.9$:

$$\mathbf{v}_{\mathbf{w}}^{(t)} = \beta \mathbf{v}_{\mathbf{w}}^{(t-1)} + \alpha \nabla_{\mathbf{w}} \mathcal{L}$$

$$\mathbf{v}_b^{(t)} = \beta \mathbf{v}_b^{(t-1)} + \alpha \nabla_b \mathcal{L}$$

$$\mathbf{w}^{(t)} = \mathbf{w}^{(t-1)} - \mathbf{v}_{\mathbf{w}}^{(t)}, \quad b^{(t)} = b^{(t-1)} - \mathbf{v}_b^{(t)}$$

---

## ⚔️ Scientific Benchmark Parity

Audited side-by-side on the **holdout test split ($N = 1,057$)** with zero metric fabrication:

| Evaluation Metric / Attribute | Custom NumPy LR (Primary Model) | Scikit-Learn LR (Baseline Benchmark) | Parity Delta / Observation |
|---|:---:|:---:|:---:|
| **Underlying Solver** | Mini-Batch Momentum GD (1st Order) | Quasi-Newton L-BFGS (2nd Order) | Pure NumPy vs C-optimizations |
| **ROC-AUC Score** | **0.8452** | **0.8449** | **+0.0003** (Statistically Equivalent) |
| **PR-AUC (Average Precision)** | **0.6694** | **0.6687** | **+0.0007** (Robust on Imbalance) |
| **Accuracy (Default $t = 0.50$)** | **75.40%** | **75.40%** | **0.00%** (Exact Match) |
| **Recall / Sensitivity ($t = 0.50$)** | **75.09%** | **75.09%** | **0.00%** (Exact Match) |
| **Precision ($t = 0.50$)** | **52.62%** | **52.62%** | **0.00%** (Exact Match) |
| **F1-Score ($t = 0.50$)** | **0.6188** | **0.6188** | **0.0000** (Exact Match) |
| **Optimal F1 ($t^* = 0.58$)** | **0.6280** | **0.6280** | **0.0000** (Validation-Tuned) |
| **Probability Correlation ($r$)** | **0.998200** | **1.000000** | **Pearson $r \to 1.0$ Mathematical Parity** |
| **Inference Latency (per 1k records)** | **0.056 ms** | **0.222 ms** | **4.0x Faster Inference** |
| **Test Cohort Net Retention Savings** | **+$64,350.00** | **+$64,350.00** | **Exact Financial Equivalence** |

---

## 🧬 Exact Log-Odds Waterfall Explainability

Unlike opaque black-box models or perturbation-based Shapley approximations, ChurnGuard AI provides **mathematically exact attribution**:

$$z = b + \sum_{j=1}^d w_j x_j \implies \hat{p} = \frac{1}{1 + e^{-z}}$$

```
[Base Prior Logit (b = -0.0270)]  ────────> Implied Base Probability: 49.3%
  + Month-to-Month Contract (+1.4087)  ───> Cumulative Logit: +1.3817 (Implied: 79.9%)
  + Fiber Optic Internet (+0.8421)  ──────> Cumulative Logit: +2.2238 (Implied: 90.2%)
  + No Online Tech Support (+0.6120) ─────> Cumulative Logit: +2.8358 (Implied: 94.5%)
  + Low Tenure (< 6 mos) (+0.3618)  ──────> Cumulative Logit: +3.1976 (Implied: 96.1%)
-----------------------------------------------------------------------------------------
Final Model Prediction: 96.1% Churn Risk  |  Discrepancy vs Reconstructed: 0.00e+00
```

Every risk driver is mapped deterministically to a **Prescriptive Retention Playbook** that provides operational actions, expected churn impact, and financial cost estimates.

---

## 🚀 Dual-Mode Architecture

The platform provides two fully decoupled user experiences:

### 🎭 Mode A — Interactive Demo & Simulator (Canonical Telco Cohort)
- **Frozen Canonical Artifacts:** Locks the official model weights, validation threshold ($0.58$), and preprocessor pipeline.
- **Persona Archetypes:** Instant pre-loaded profiles (High-Risk Month-to-Month Fiber Optic, Established 2-Year Contract, Early-Tenure Senior Citizen).
- **Interactive Simulator:** Dynamic form sliders with real-time logit recalculation, probability progress bars, and waterfall tables.
- **7-Step Guided Tour:** Executive walkthrough demonstrating business problem formulation, inference, attribution, playbooks, and benchmark parity.

### 📁 Mode B — Analyze Your Dataset (Dynamic Tabular ML Engine)
- **Automated Schema Detection:** Identifies target candidates, positive classes, identifier columns, and categorical/numerical feature roles.
- **Zero-Leakage Dynamic Pipeline:** Imputes missing values, encodes categories, and standardizes numerical columns strictly on training data.
- **From-Scratch Training:** Fits a fresh pure NumPy Logistic Regression model on the uploaded CSV with validation-tuned decision boundary.
- **Cohort Export:** Generates full-dataset probability scores, risk tiers, and downloadable prediction CSVs.
- **Strict Isolation:** Completely isolated from official demo artifacts to prevent data or metric pollution.

---

## 💻 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/kushal287/Churnguard-AI.git
cd Churnguard-AI
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run Automated Tests
Verify mathematical primitives, zero leakage, model convergence, and benchmark parity:
```bash
pytest -v
```
*All 60 tests pass with 100% success rate.*

### 4. Retrain Model & Generate Artifacts (Optional)
To execute the end-to-end training pipeline and regenerate all figures and metrics:
```bash
python -m src.pipeline.train_pipeline
```

### 5. Launch the Streamlit Web Application
```bash
streamlit run app/streamlit_app.py
```
Open your browser at **`http://localhost:8501`**.

---

## 🏛️ System Architecture

```
Churnguard-AI/
├── app/                                    # Streamlit Web Application
│   ├── assets/
│   │   └── logo.png                        # Official Branded Logo Emblem
│   ├── components/
│   │   ├── benchmark_view.py               # Scientific Benchmark Arena View
│   │   ├── executive_view.py               # Executive Command Center View
│   │   ├── guided_demo_view.py             # 7-Step Interactive Guided Tour View
│   │   ├── landing_view.py                 # Dual-Mode Landing Page
│   │   ├── single_prediction_view.py       # Live Simulator & Log-Odds Waterfall View
│   │   └── user_dataset_view.py            # Mode B: Dynamic Tabular ML Engine
│   └── streamlit_app.py                    # Main App Controller & Light Enterprise CSS
├── artifacts/                              # Authoritative Frozen Model Artifacts
│   ├── benchmark_results.json              # Side-by-Side Model Benchmark Metrics
│   ├── custom_logistic_model.npz           # Frozen Custom NumPy Model Parameters
│   ├── feature_names.json                  # 41 Transformed Feature Column Names
│   ├── final_results.json                  # Single Authoritative Results Master File
│   └── preprocessor_pipeline.joblib        # Fitted Zero-Leakage Preprocessor
├── config/
│   └── config.py                           # Global Paths, Seeds, Schemas & Hyperparameters
├── data/
│   ├── processed/                          # Partitioned & Engineered Dataset Splits
│   │   ├── schema_metadata.json            # Dataset Schema Metadata & Checksums
│   │   ├── test.csv                        # Frozen Test Split (15%, N = 1,057)
│   │   ├── train.csv                       # Frozen Train Split (70%, N = 4,930)
│   │   └── val.csv                         # Frozen Validation Split (15%, N = 1,056)
│   └── raw/
│       └── telco_customer_churn.csv        # Canonical Raw Dataset (7,043 rows, 21 columns)
├── reports/                                # Documentation & Evaluation Figures
│   ├── figures/                            # Publication-Quality Evaluation Plots
│   │   ├── confusion_matrices.png          # Side-by-Side Confusion Matrix Plots
│   │   ├── eda_churn_distribution.png      # Class Distribution Plot
│   │   ├── eda_correlation_heatmap.png     # Correlation Matrix Heatmap
│   │   ├── feature_importance.png          # Top Learned Feature Odds Ratios
│   │   └── roc_pr_curves_comparison.png    # Overlaid ROC & PR Curves
│   ├── CAPSTONE_MASTER_PROJECT_REPORT.md   # Consolidated Master 3-Part Capstone Document
│   ├── capstone_analytical_report.md       # Detailed Analytical Report
│   ├── final_experiment_summary.md         # Final Frozen Experiment Summary
│   └── IMPLEMENTATION_REPORT.md            # Comprehensive Implementation Report
├── src/                                    # Core Python Source Modules
│   ├── data/
│   │   ├── dynamic_pipeline.py             # Dynamic ML Pipeline for General Tabular Data
│   │   ├── loader.py                       # Dataset Ingestion & Partitioning
│   │   ├── preprocessor.py                 # Zero-Leakage Preprocessing Engine
│   │   └── validator.py                    # Schema Validator & Health Check Engine
│   ├── evaluation/
│   │   ├── benchmark.py                    # Scikit-Learn Benchmark & Comparison Engine
│   │   ├── metrics.py                      # Pure NumPy Evaluation Metrics & Cost Utility
│   │   └── plots.py                        # High-Resolution Visualization Generator
│   ├── explainability/
│   │   ├── dynamic_explainer.py            # Dynamic Log-Odds Explainer for Custom Data
│   │   ├── feature_importance.py           # Global Feature Odds Ratios Engine
│   │   ├── individual_explainer.py         # Additive Log-Odds Waterfall Explainer
│   │   └── retention_playbook.py           # Deterministic Prescriptive Action Engine
│   ├── features/
│   │   └── engineering.py                  # Domain Feature Engineering Engine
│   ├── models/
│   │   ├── custom_logistic_regression.py   # Scratch NumPy Logistic Regression Model
│   │   └── optimizer.py                    # Mini-Batch Momentum GD Optimizer
│   ├── pipeline/
│   │   └── train_pipeline.py              # End-to-End Pipeline Orchestrator
│   └── utils/
│       └── integrity.py                    # Artifact Integrity & Checksum Verification
├── tests/                                  # Full Automated Test Suite (60 Tests)
│   ├── test_custom_logistic_regression.py  # Model Mathematical Unit Tests
│   ├── test_dynamic_custom_dataset.py      # Dynamic ML Pipeline & Custom Dataset Tests
│   ├── test_end_to_end_pipeline.py         # End-to-End Pipeline Integration Tests
│   ├── test_explainability.py              # Log-Odds Reconstruction & Playbook Tests
│   ├── test_integrity.py                   # Artifact Integrity & Checksum Tests
│   ├── test_math_primitives.py             # Vectorized Math & Numerical Gradient Checks
│   ├── test_metrics.py                     # Metric Parity & Business Value Tests
│   ├── test_preprocessor.py                # Preprocessing & Type Coercion Tests
│   └── test_user_dataset_workflow.py       # Dataset Upload, Health Check & Invariance Tests
├── .gitignore                              # Git Ignore Configuration
├── CAPSTONE_MASTER_PROJECT_REPORT.md       # Consolidated Master Capstone Technical Report
├── FINAL_EXPERIMENT_AND_SUMMARY_RESULTS.md # Single Source of Truth
├── IMPLEMENTATION_REPORT.md                # Master Implementation Report
├── README.md                               # Project Readme & Overview
└── requirements.txt                        # Python Dependencies
```

---

## 📊 Evaluation & Experimentation Summary

For in-depth mathematical proofs, experiment logs, parameter sensitivities, threshold selection sweeps, and regulatory compliance audits, refer to the master technical report:
👉 **[`CAPSTONE_MASTER_PROJECT_REPORT.md`](CAPSTONE_MASTER_PROJECT_REPORT.md)**

---

## 📄 License & Attribution

This project is developed as an academic and professional machine learning capstone. Licensed under the [MIT License](LICENSE).

Developed with ❤️ by **[Kushal](https://github.com/kushal287)**.
