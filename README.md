# 🛡️ ChurnGuard AI — Explainable Customer Churn Prediction & Retention Intelligence Platform

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/numpy-first--principles%20math-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-light%20enterprise%20ui-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![WebAssembly](https://img.shields.io/badge/stlite-in--browser%20wasm-654FF0.svg?logo=webassembly&logoColor=white)](https://churnguardai.vercel.app)
[![Tests](https://img.shields.io/badge/tests-60%2F60%20passing-10B981.svg)](https://pytest.org/)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.8452%20(Parity)-2563EB.svg)](https://github.com/kushal287/Churnguard-AI)
[![Pearson r](https://img.shields.io/badge/Pearson%20r-0.9982-8B5CF6.svg)](https://github.com/kushal287/Churnguard-AI)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**An enterprise-grade, explainable machine learning platform engineered from first principles in pure NumPy.**  
*Zero Data Leakage Preprocessing • Scikit-Learn Parity Audit • Exact Log-Odds Waterfall Explainability • Dual-Mode Production UI • Serverless WebAssembly Deployment*

🚀 **Live Interactive Web App**: [churnguardai.vercel.app](https://churnguardai.vercel.app)  
📦 **GitHub Repository**: [github.com/kushal287/Churnguard-AI](https://github.com/kushal287/Churnguard-AI)

[Overview](#-overview) • [Tech Stack](#-tech-stack) • [System Architecture & Modes](#-system-architecture--dual-mode-operations) • [Mathematical Foundations](#-mathematical-formulation--from-scratch-implementation) • [Benchmark Parity](#-scientific-benchmark-parity) • [How to Use & Run](#-how-to-use--run-locally) • [Repository Structure](#-repository-structure)

</div>

---

## 🌟 Overview

**ChurnGuard AI** is an end-to-end, production-ready machine learning intelligence system designed to predict customer attrition risk in advance and prescribe targeted, cost-effective retention actions before service cancellation.

Developed as a **Major Capstone Machine Learning Project**, the platform combines:
1. **First-Principles ML Engine**: Logistic Regression built completely from scratch in pure NumPy using vector algebra, weighted binary cross-entropy, analytical gradients, and mini-batch momentum gradient descent.
2. **Zero Data Leakage Pipeline**: Strict separation of training ($70\%$), validation ($15\%$), and test ($15\%$) partitions. Imputation medians, standard scalers, and one-hot encoders are fitted exclusively on the training split.
3. **Scientific Benchmark Parity**: Audited side-by-side on holdout data ($N = 1,057$) against Scikit-Learn's second-order L-BFGS solver, achieving empirical parity ($0.8452$ vs $0.8449$ ROC-AUC, Pearson $r = 0.9982$) with a **4x faster inference speedup** ($0.056\text{ ms}$ vs $0.222\text{ ms}$ per 1k records).
4. **Exact Mathematical Explainability**: Direct log-odds additive waterfall reconstruction ($z = b + \sum w_j x_j \implies p = \sigma(z)$) with zero approximation discrepancy ($0.00\text{e}+00$).
5. **Dual-Mode Interactive Platform**:
   - **Mode 1 (Guided Capstone Demo)**: Interactive 6-step customer walkthrough with real-time risk scoring, attribution, financial utility modeling, and prescriptive retention playbooks.
   - **Mode 2 (Dynamic Custom Dataset Engine)**: General tabular binary classification engine that auto-detects schemas, fits zero-leakage preprocessors, trains a fresh NumPy model on user CSVs (e.g. Loan Default, Fraud, HR Attrition, Medical), and exports scored predictions.
6. **Zero-Backend In-Browser Deployment**: Runs client-side via Pyodide and `@stlite/browser` WebAssembly on Vercel edge networks, requiring zero cloud server maintenance.

---

## 💻 Tech Stack

| Layer | Technology | Purpose & Implementation Details |
|---|---|---|
| **Core ML Engine** | **Python 3.10+ / Pure NumPy** | First-principles implementation of vectorized Logistic Regression, Sigmoid activation, weighted cross-entropy loss, L2 regularization, analytical gradient checking, and momentum gradient descent. |
| **Data Processing** | **Pandas / NumPy** | Schema detection, type coercion, handling whitespace missing values in `TotalCharges`, and zero-leakage transformations. |
| **Benchmarking & Validation** | **Scikit-Learn** *(Benchmarking Only)* | Used strictly as an external validation baseline to audit and confirm mathematical convergence and metric parity. *(Custom model uses 0 Scikit-Learn estimators).* |
| **Interactive Dashboard** | **Streamlit** | Multi-view enterprise UI with single prediction simulator, batch dataset explorer, financial cost-utility optimizer, and benchmark arena. |
| **Client-Side Runtime** | **Stlite / Pyodide (WebAssembly)** | Packages the entire Python runtime, NumPy, Pandas, Scikit-Learn, and PyArrow into browser WASM, enabling 100% serverless execution on edge CDNs. |
| **Deployment & Hosting** | **Vercel** | Static hosting with HTTP security headers (`COOP`, `COEP`, `nosniff`) and cache-busted manifest delivery. |
| **Quality Assurance** | **Pytest** | 60 automated unit and integration tests verifying math primitives, gradient accuracy, preprocessor isolation, metric parity, and artifact integrity. |

---

## 🏗️ System Architecture & Dual-Mode Operations

```
                                  ┌────────────────────────────────────────────────────────┐
                                  │                     CHURNGUARD AI                      │
                                  │      Explainable Retention Intelligence Platform       │
                                  └───────────────────────────┬────────────────────────────┘
                                                              │
                     ┌────────────────────────────────────────┴────────────────────────────────────────┐
                     │                                                                                 │
                     ▼                                                                                 ▼
     ┌───────────────────────────────┐                                                 ┌───────────────────────────────┐
     │  MODE 1: GUIDED DEMO MODE     │                                                 │  MODE 2: CUSTOM DATASET MODE  │
     │  (IBM Telco Canonical Data)   │                                                 │  (Any Tabular Binary CSV)     │
     └───────────────┬───────────────┘                                                 └───────────────┬───────────────┘
                     │                                                                                 │
     ┌───────────────▼───────────────┐                                                 ┌───────────────▼───────────────┐
     │ • Authoritative Offline Model │                                                 │ • Dynamic Schema Auto-Detect  │
     │ • Strict Zero-Leakage Preproc │                                                 │ • Dynamic Preprocessor Fit    │
     │ • Threshold Optimizer (t*=0.58│                                                 │ • In-Browser NumPy Training   │
     │ • Exact Waterfall Attribution │                                                 │ • Threshold Tuning on Val Split│
     │ • Prescriptive Action Playbook│                                                 │ • Scored Predictions CSV Expor│
     └───────────────┬───────────────┘                                                 └───────────────┬───────────────┘
                     │                                                                                 │
                     └────────────────────────────────────────┬────────────────────────────────────────┘
                                                              │
                                            ┌─────────────────▼─────────────────┐
                                            │      LIGHT ENTERPRISE UI          │
                                            │  (Streamlit / Stlite WebAssembly) │
                                            └───────────────────────────────────┘
```

### 1. Mode 1: Guided Capstone Demonstration (IBM Telco)
- **Problem**: Predict customer churn among 7,043 telecommunications subscribers.
- **Dataset**: `WA_Fn-UseC_-Telco-Customer-Churn.csv` (21 raw columns: contract terms, tenure, monthly charges, payment methods, tech support, internet services).
- **Partitioning**: 70% Train ($N=4,930$), 15% Validation ($N=1,056$), 15% Test ($N=1,057$) with stratified random splitting.
- **Key Deliverables**:
  - Probability scoring with risk tier classification (Low, Medium, High).
  - Additive log-odds waterfall decomposition ($z = b + \sum w_j x_j$).
  - Prioritized prescriptive retention playbook with quantified ROI impact.

### 2. Mode 2: Dynamic Custom Tabular ML Engine
- **Problem**: Enable business analysts and data scientists to upload *any* arbitrary tabular binary classification dataset (Loan Default, HR Attrition, Fraud, Healthcare Churn).
- **Pipeline**:
  1. **Schema Detector**: Automatically isolates identifier columns (e.g. `customer_id`, `uuid`), binary targets (exactly 2 unique values), numerical features, and categorical features.
  2. **Zero-Leakage Dynamic Preprocessor**: Computes numerical medians and categorical modes strictly on the training partition; transforms validation and test splits with zero leakage.
  3. **Interactive Hyperparameters**: Configurable learning rate $\alpha$, epochs, L2 penalty $\lambda$, and validation split size.
  4. **Dynamic Decision Threshold**: Searches $\tau \in [0.10, 0.90]$ to maximize validation F1-score.
  5. **Prediction Table & CSV Export**: Displays paginated, searchable scored observations with probability scores and downloadable results.

---

## 📐 Mathematical Formulation & From-Scratch Implementation

### 1. Vectorized Hypothesis & Sigmoid Activation
For an input feature vector $\mathbf{x} \in \mathbb{R}^d$, model weights $\mathbf{w} \in \mathbb{R}^d$, and baseline prior intercept $b \in \mathbb{R}$:

$$z = b + \mathbf{w}^T \mathbf{x} = b + \sum_{j=1}^d w_j x_j$$

$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

*Numerically stable piecewise implementation to prevent IEEE-754 floating-point overflow:*

$$\sigma(z) = \begin{cases} 
\frac{1}{1 + e^{-z}} & \text{if } z \ge 0 \\ 
\frac{e^z}{1 + e^z} & \text{if } z < 0 
\end{cases}$$

---

### 2. Weighted Binary Cross-Entropy Loss with L2 Regularization
To handle severe class imbalance (minority churn rate $\approx 26.5\%$), class weights $w_0$ and $w_1$ are computed via inverse class frequency:

$$w_k = \frac{N}{2 \cdot N_k}, \quad k \in \{0, 1\}$$

The regularized objective function is:

$$\mathcal{L}(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^N \Big[ w_1 y_i \ln(\hat{y}_i + \epsilon) + w_0 (1 - y_i) \ln(1 - \hat{y}_i + \epsilon) \Big] + \frac{\lambda}{2} \|\mathbf{w}\|_2^2$$

where $\epsilon = 10^{-15}$ prevents $\ln(0)$ undeflow, and $\lambda$ controls L2 weight shrinkage.

---

### 3. Analytical Parameter Gradients
Differentiating $\mathcal{L}$ with respect to $\mathbf{w}$ and $b$:

$$\text{Error Residual: } e_i = \hat{y}_i - y_i$$

$$\text{Sample Weight: } s_i = w_1 y_i + w_0 (1 - y_i)$$

$$\nabla_{\mathbf{w}} \mathcal{L} = \frac{1}{N} \mathbf{X}^T (\mathbf{s} \odot \mathbf{e}) + \lambda \mathbf{w}$$

$$\nabla_b \mathcal{L} = \frac{1}{N} \sum_{i=1}^N s_i e_i$$

Verified against numerical finite-difference gradient approximations ($\Delta_{\text{grad}} < 10^{-7}$).

---

### 4. Mini-Batch Momentum Gradient Descent
Parameters are updated iteratively with momentum coefficient $\beta = 0.9$ and learning rate $\alpha$:

$$\mathbf{v}_{\mathbf{w}}^{(t)} = \beta \mathbf{v}_{\mathbf{w}}^{(t-1)} + \alpha \nabla_{\mathbf{w}} \mathcal{L}$$

$$\mathbf{v}_b^{(t)} = \beta \mathbf{v}_b^{(t-1)} + \alpha \nabla_b \mathcal{L}$$

$$\mathbf{w}^{(t)} = \mathbf{w}^{(t-1)} - \mathbf{v}_{\mathbf{w}}^{(t)}, \quad b^{(t)} = b^{(t-1)} - \mathbf{v}_b^{(t)}$$

---

### 5. Additive Log-Odds Waterfall Explainability
Because Logistic Regression is a generalized linear model with a logit link function, predictions decompose additively into feature-level log-odds contributions:

$$\ln\left(\frac{p}{1 - p}\right) = z = b + \sum_{j=1}^d \underbrace{w_j x_j}_{c_j}$$

- **Baseline Intercept ($b$)**: Prior baseline log-odds before observing customer attributes.
- **Risk Drivers ($c_j > 0$)**: Features pushing the logit higher (increasing churn probability).
- **Protective Factors ($c_j < 0$)**: Features pushing the logit lower (decreasing churn probability).
- **Exact Reconstruction**: Direct model probability $\sigma(z)$ exactly equals reconstructed probability $\sigma(b + \sum c_j)$ with discrepancy $|p - p_{\text{recon}}| \le 10^{-15}$.

---

### 6. Odds Ratio Interpretability
For each feature $j$, the Odds Ratio $\text{OR}_j = e^{w_j}$ quantifies the multiplicative change in churn odds per unit increase in standard deviation (for standardized numerical features) or relative to the reference category (for one-hot categorical features):

$$\text{OR}_j = e^{w_j} \implies \text{Odds}_{\text{new}} = \text{Odds}_{\text{baseline}} \times \text{OR}_j$$

---

### 7. Cost-Utility Matrix & Decision Threshold Optimization
Standard models arbitrarily threshold at $t = 0.50$, ignoring asymmetric financial misclassification costs. ChurnGuard AI evaluates an enterprise cost-utility function:

$$U(t) = \text{TP}(t) \cdot V_{\text{retention}} - \text{FP}(t) \cdot C_{\text{intervention}} - \text{FN}(t) \cdot C_{\text{churn}}$$

| Outcome | Economic Meaning | Telco Industry Cost / Value |
|---|---|---|
| **True Positive (TP)** | Successfully identified at-risk customer; accepted offer. | **+$350 net preserved LTV** |
| **False Positive (FP)** | Unnecessary retention incentive offered to loyal customer. | **-$50 wasted incentive** |
| **False Negative (FN)** | At-risk customer missed; churned silently. | **-$400 lost gross margin** |
| **True Negative (TN)** | Correctly identified loyal customer; zero action. | **$0 operational impact** |

Optimizing on the validation split yielded **$t^* = 0.58$**, unlocking **+$64,350.00** in net retained value on the test set (+$60.88/customer).

---

## ⚔️ Scientific Benchmark Parity

Audited side-by-side on the independent holdout test split ($N = 1,057$):

| Metric / Characteristic | Custom NumPy Logistic Regression | Scikit-Learn LogisticRegression | Delta / Parity Status |
|---|:---:|:---:|:---:|
| **Underlying Solver** | Mini-Batch Momentum GD (1st Order) | Quasi-Newton L-BFGS (2nd Order) | Pure NumPy vs C-library |
| **ROC-AUC Score** | **0.8452** | **0.8449** | **+0.0003** (Parity Achieved) |
| **PR-AUC (Average Precision)** | **0.6694** | **0.6687** | **+0.0007** (Parity Achieved) |
| **Test Accuracy** | **80.32%** | **80.42%** | **-0.10%** (Statistically Identical) |
| **Minority Recall (t=0.50)** | **53.38%** | **54.09%** | **-0.71%** (Balanced Coverage) |
| **F1-Score (t=0.50)** | **0.5905** | **0.5941** | **-0.0036** (Parity Achieved) |
| **Prediction Correlation ($r$)** | **0.998200** | **1.000000** | **Exact Linear Concordance** |
| **Inference Latency (per 1k)** | **0.056 ms** | **0.222 ms** | **4.0x Faster (Vectorized)** |

---

## 🚀 How to Use & Run Locally

### 1. Prerequisites
- **Python 3.10+** (Python 3.11 recommended)
- Git

### 2. Clone the Repository
```bash
git clone https://github.com/kushal287/Churnguard-AI.git
cd Churnguard-AI
```

### 3. Create a Virtual Environment & Install Dependencies
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements-python.txt
```

### 4. Launch the Interactive Application
```bash
streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501`.

---

### 5. Execute Full ML Pipeline & Benchmarking from CLI
To retrain the model from scratch, execute zero-leakage preprocessing, run the Scikit-Learn benchmark audit, generate explainability artifacts, and export evaluation JSON:
```bash
python -m src.pipeline.run_pipeline
```
*Outputs generated in `artifacts/`: `model.pkl`, `preprocessor.pkl`, `final_results.json`, `benchmark_results.json`.*

---

### 6. Run the Automated Test Suite (60 Tests)
```bash
pytest -v
```
All 60 tests will pass cleanly with zero failures:
```
tests/test_custom_logistic_regression.py::TestCustomLogisticRegression ............. [100%]
tests/test_dynamic_custom_dataset.py::TestDynamicTabularPipeline ................... [100%]
tests/test_end_to_end_pipeline.py::TestEndToEndPipeline ............................. [100%]
tests/test_explainability.py::TestExplainabilityAndPlaybook ........................ [100%]
tests/test_integrity.py::TestArtifactIntegrity ...................................... [100%]
tests/test_math_primitives.py::TestMathPrimitives ................................... [100%]
tests/test_metrics.py::TestMetricsParity ............................................ [100%]
tests/test_preprocessor.py::TestPreprocessorAndLoader ............................... [100%]
tests/test_user_dataset_workflow.py::TestUserDatasetWorkflow ........................ [100%]
============================== 60 passed in 6.5s ===============================
```

---

## 📂 Repository Structure

```
Churnguard-AI/
├── .github/                         # GitHub repository configuration
├── app/                             # Streamlit Enterprise UI application
│   ├── assets/                      # Application logos, branding, and icons
│   ├── components/                  # Modular view components
│   │   ├── __init__.py
│   │   ├── benchmark_view.py        # Mode 1: Side-by-side benchmark arena
│   │   ├── executive_view.py        # Mode 1: Executive KPI overview & ROI
│   │   ├── guided_demo_view.py      # Mode 1: 6-step guided customer demo
│   │   ├── landing_view.py          # Portal landing and navigation hub
│   │   ├── single_prediction_view.py# Mode 1: Real-time simulator & attribution
│   │   └── user_dataset_view.py     # Mode 2: Dynamic custom tabular ML engine
│   └── streamlit_app.py             # Main application entrypoint
├── artifacts/                       # Authoritative serialized pipeline artifacts
│   ├── benchmark_results.json       # Empirical parity audit data
│   ├── final_results.json           # Canonical validation & test metrics
│   ├── model.pkl                    # Serialized Custom Logistic Regression
│   └── preprocessor.pkl             # Serialized Zero-Leakage Preprocessor
├── config/                          # Configuration constants and file paths
│   ├── __init__.py
│   └── config.py
├── data/                            # Canonical dataset and sample CSVs
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── telco_customer_churn_sample_250.csv
├── reports/                         # Figures and visual assets
│   └── figures/                     # ROC curves, confusion matrices, heatmaps
├── src/                             # Core Python package (First Principles ML)
│   ├── data/                        # Ingestion, validation, preprocessing
│   │   ├── __init__.py
│   │   ├── dataset_loader.py
│   │   ├── dynamic_pipeline.py      # Mode 2: Dynamic tabular ML pipeline
│   │   ├── preprocessor.py          # Mode 1: Strict zero-leakage preprocessor
│   │   └── schema_validator.py
│   ├── evaluation/                  # Metrics and benchmark parity auditing
│   │   ├── __init__.py
│   │   ├── benchmark_runner.py
│   │   └── metrics.py               # Vectorized scratch metrics (ROC, F1, etc.)
│   ├── explainability/              # Exact log-odds waterfall decomposition
│   │   ├── __init__.py
│   │   ├── dynamic_explainer.py     # Mode 2: Dynamic explainer
│   │   ├── individual_explainer.py  # Mode 1: Additive waterfall explainer
│   │   └── retention_playbook.py    # Prescriptive decision playbook
│   ├── models/                      # First-principles model implementation
│   │   ├── __init__.py
│   │   └── custom_logistic_regression.py # Scratch NumPy Logistic Regression
│   ├── pipeline/                    # Training and serialization orchestration
│   │   ├── __init__.py
│   │   └── run_pipeline.py
│   └── utils/                       # Serialization and artifact verification
│       ├── __init__.py
│       ├── integrity.py
│       └── serialization.py
├── stlite_build/                    # WebAssembly manifest and bundle files
│   └── manifest.json                # In-browser Pyodide filesystem manifest
├── tests/                           # 60 automated unit & integration tests
│   ├── test_custom_logistic_regression.py
│   ├── test_dynamic_custom_dataset.py
│   ├── test_end_to_end_pipeline.py
│   ├── test_explainability.py
│   ├── test_integrity.py
│   ├── test_math_primitives.py
│   ├── test_metrics.py
│   ├── test_preprocessor.py
│   └── test_user_dataset_workflow.py
├── build_stlite.py                  # Script to package codebase into WebAssembly
├── index.html                       # Serverless client-side WebAssembly frontend
├── LICENSE                          # MIT License
├── README.md                        # Master project documentation
├── requirements-python.txt          # Python dependencies
└── vercel.json                      # Vercel deployment & security headers
```

---

## 🛡️ License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <sub>Built with mathematical rigor in pure NumPy • Engineered for explainability and high-impact retention intelligence.</sub>
</div>
