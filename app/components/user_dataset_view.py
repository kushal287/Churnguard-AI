"""
Analyze Your Dataset Component for ChurnGuard AI (Mode 2 — Custom Tabular Analysis).
Supports general tabular binary classification datasets (Customer Churn, Fraud, Loan Default,
Employee Attrition, Medical Diagnosis, etc.) using a fully dynamic, zero-leakage training
and inference pipeline powered by the project's scratch NumPy Logistic Regression model.
"""

# Pyodide / WebAssembly environment polyfill for PyArrow RecordBatch compatibility
try:
    import pyarrow as pa
    if not hasattr(pa, "RecordBatch"):
        try:
            import pyarrow.lib as _palib
            if hasattr(_palib, "RecordBatch"):
                pa.RecordBatch = _palib.RecordBatch
            else:
                class _RecordBatchStub:
                    @classmethod
                    def from_pandas(cls, *args, **kwargs):
                        return None
                    @classmethod
                    def from_arrays(cls, *args, **kwargs):
                        return None
                pa.RecordBatch = _RecordBatchStub
        except Exception:
            class _RecordBatchStub:
                @classmethod
                def from_pandas(cls, *args, **kwargs):
                    return None
                @classmethod
                def from_arrays(cls, *args, **kwargs):
                    return None
            pa.RecordBatch = _RecordBatchStub
    if not hasattr(pa, "Table"):
        try:
            import pyarrow.lib as _palib
            if hasattr(_palib, "Table"):
                pa.Table = _palib.Table
            else:
                class _TableStub:
                    @classmethod
                    def from_pandas(cls, *args, **kwargs):
                        return None
                pa.Table = _TableStub
        except Exception:
            class _TableStub:
                @classmethod
                def from_pandas(cls, *args, **kwargs):
                    return None
            pa.Table = _TableStub
except Exception:
    pass

from typing import Any, Dict, List, Optional
import io
import numpy as np
import pandas as pd
import streamlit as st

from src.data.dynamic_pipeline import DynamicPipelineTrainer, DynamicSchemaDetector
from src.explainability.dynamic_explainer import DynamicIndividualExplainer


def _generate_sample_template_csv() -> str:
    """Generate a realistic sample CSV template for binary classification."""
    records = [
        {"record_id": "REC-001", "age": 24, "monthly_income": 35000, "tenure_months": 4, "support_calls": 5, "contract_type": "Monthly", "payment_method": "Electronic", "premium_member": "No", "target": 1},
        {"record_id": "REC-002", "age": 42, "monthly_income": 72000, "tenure_months": 38, "support_calls": 1, "contract_type": "Annual", "payment_method": "Credit Card", "premium_member": "Yes", "target": 0},
        {"record_id": "REC-003", "age": 31, "monthly_income": 48000, "tenure_months": 12, "support_calls": 3, "contract_type": "Monthly", "payment_method": "Electronic", "premium_member": "No", "target": 1},
        {"record_id": "REC-004", "age": 55, "monthly_income": 95000, "tenure_months": 60, "support_calls": 0, "contract_type": "Annual", "payment_method": "Bank Transfer", "premium_member": "Yes", "target": 0},
        {"record_id": "REC-005", "age": 28, "monthly_income": 41000, "tenure_months": 7, "support_calls": 4, "contract_type": "Monthly", "payment_method": "Electronic", "premium_member": "No", "target": 1},
        {"record_id": "REC-006", "age": 38, "monthly_income": 63000, "tenure_months": 24, "support_calls": 2, "contract_type": "Bi-Annual", "payment_method": "Credit Card", "premium_member": "No", "target": 0},
        {"record_id": "REC-007", "age": 45, "monthly_income": 82000, "tenure_months": 48, "support_calls": 1, "contract_type": "Annual", "payment_method": "Bank Transfer", "premium_member": "Yes", "target": 0},
        {"record_id": "REC-008", "age": 22, "monthly_income": 28000, "tenure_months": 2, "support_calls": 6, "contract_type": "Monthly", "payment_method": "Electronic", "premium_member": "No", "target": 1},
        {"record_id": "REC-009", "age": 50, "monthly_income": 88000, "tenure_months": 55, "support_calls": 0, "contract_type": "Annual", "payment_method": "Credit Card", "premium_member": "Yes", "target": 0},
        {"record_id": "REC-010", "age": 33, "monthly_income": 52000, "tenure_months": 15, "support_calls": 3, "contract_type": "Monthly", "payment_method": "Electronic", "premium_member": "No", "target": 0},
        {"record_id": "REC-011", "age": 27, "monthly_income": 37000, "tenure_months": 5, "support_calls": 5, "contract_type": "Monthly", "payment_method": "Electronic", "premium_member": "No", "target": 1},
        {"record_id": "REC-012", "age": 60, "monthly_income": 105000, "tenure_months": 72, "support_calls": 0, "contract_type": "Annual", "payment_method": "Bank Transfer", "premium_member": "Yes", "target": 0},
        {"record_id": "REC-013", "age": 35, "monthly_income": 56000, "tenure_months": 20, "support_calls": 2, "contract_type": "Bi-Annual", "payment_method": "Credit Card", "premium_member": "No", "target": 0},
        {"record_id": "REC-014", "age": 29, "monthly_income": 39000, "tenure_months": 3, "support_calls": 7, "contract_type": "Monthly", "payment_method": "Electronic", "premium_member": "No", "target": 1},
        {"record_id": "REC-015", "age": 47, "monthly_income": 78000, "tenure_months": 42, "support_calls": 1, "contract_type": "Annual", "payment_method": "Credit Card", "premium_member": "Yes", "target": 0},
        {"record_id": "REC-016", "age": 26, "monthly_income": 33000, "tenure_months": 6, "support_calls": 4, "contract_type": "Monthly", "payment_method": "Electronic", "premium_member": "No", "target": 1},
        {"record_id": "REC-017", "age": 40, "monthly_income": 67000, "tenure_months": 30, "support_calls": 1, "contract_type": "Bi-Annual", "payment_method": "Bank Transfer", "premium_member": "Yes", "target": 0},
        {"record_id": "REC-018", "age": 52, "monthly_income": 91000, "tenure_months": 58, "support_calls": 0, "contract_type": "Annual", "payment_method": "Credit Card", "premium_member": "Yes", "target": 0},
        {"record_id": "REC-019", "age": 23, "monthly_income": 30000, "tenure_months": 1, "support_calls": 8, "contract_type": "Monthly", "payment_method": "Electronic", "premium_member": "No", "target": 1},
        {"record_id": "REC-020", "age": 36, "monthly_income": 59000, "tenure_months": 22, "support_calls": 2, "contract_type": "Bi-Annual", "payment_method": "Credit Card", "premium_member": "No", "target": 0},
        {"record_id": "REC-021", "age": 44, "monthly_income": 76000, "tenure_months": 36, "support_calls": 1, "contract_type": "Annual", "payment_method": "Bank Transfer", "premium_member": "Yes", "target": 0},
        {"record_id": "REC-022", "age": 30, "monthly_income": 44000, "tenure_months": 9, "support_calls": 3, "contract_type": "Monthly", "payment_method": "Electronic", "premium_member": "No", "target": 1},
        {"record_id": "REC-023", "age": 57, "monthly_income": 98000, "tenure_months": 65, "support_calls": 0, "contract_type": "Annual", "payment_method": "Bank Transfer", "premium_member": "Yes", "target": 0},
        {"record_id": "REC-024", "age": 25, "monthly_income": 36000, "tenure_months": 4, "support_calls": 6, "contract_type": "Monthly", "payment_method": "Electronic", "premium_member": "No", "target": 1},
        {"record_id": "REC-025", "age": 39, "monthly_income": 65000, "tenure_months": 28, "support_calls": 1, "contract_type": "Bi-Annual", "payment_method": "Credit Card", "premium_member": "No", "target": 0},
        {"record_id": "REC-026", "age": 48, "monthly_income": 84000, "tenure_months": 50, "support_calls": 0, "contract_type": "Annual", "payment_method": "Credit Card", "premium_member": "Yes", "target": 0},
        {"record_id": "REC-027", "age": 21, "monthly_income": 26000, "tenure_months": 1, "support_calls": 9, "contract_type": "Monthly", "payment_method": "Electronic", "premium_member": "No", "target": 1},
        {"record_id": "REC-028", "age": 34, "monthly_income": 54000, "tenure_months": 18, "support_calls": 2, "contract_type": "Monthly", "payment_method": "Credit Card", "premium_member": "No", "target": 0},
        {"record_id": "REC-029", "age": 46, "monthly_income": 80000, "tenure_months": 45, "support_calls": 1, "contract_type": "Annual", "payment_method": "Bank Transfer", "premium_member": "Yes", "target": 0},
        {"record_id": "REC-030", "age": 32, "monthly_income": 47000, "tenure_months": 10, "support_calls": 4, "contract_type": "Monthly", "payment_method": "Electronic", "premium_member": "No", "target": 1},
    ]
    df = pd.DataFrame(records)
    return df.to_csv(index=False)


def render_user_dataset_view():
    """Render the dynamic custom dataset upload, training, and analysis workspace."""
    st.markdown("## 📁 Analyze Your Dataset (Dynamic Tabular ML Engine)")
    st.markdown(
        "Upload any tabular dataset for **binary classification** (e.g., Customer Churn, Employee Attrition, "
        "Loan Default, Fraud Detection). The platform will dynamically detect your schema, fit a zero-leakage "
        "preprocessing pipeline, train a fresh **Custom NumPy Logistic Regression model from scratch**, "
        "optimize the decision threshold, and generate mathematically faithful predictions and log-odds explanations."
    )

    # 1. Expected Dataset Format & Scope Guidelines
    with st.expander("📋 Expected Dataset Format & Requirements", expanded=False):
        fmt_col1, fmt_col2 = st.columns(2)
        with fmt_col1:
            st.markdown(
                """
                #### ✅ Supported
                | Requirement | Accepted |
                |---|---|
                | **File Format** | CSV (UTF-8, comma-separated) |
                | **Problem Type** | Binary Classification |
                | **Target Column** | Exactly 2 distinct values |
                | **Target Examples** | `Yes`/`No`, `1`/`0`, `Default`/`Fully Paid`, `Attrited`/`Retained` |
                | **Features** | Numerical and/or categorical columns |
                | **Identifier Column** | Optional (auto-detected and excluded) |
                | **Minimum Rows** | ≥ 15 rows (≥ 30 recommended) |
                """
            )
        with fmt_col2:
            st.markdown(
                """
                #### 🚫 Not Currently Supported
                | Data Type | Status |
                |---|---|
                | Images / Audio / Video | ✕ Not supported |
                | Free-form text / NLP | ✕ Not supported |
                | Continuous Regression targets | ✕ Rejected |
                | Multiclass targets (≥ 3 classes) | ✕ Rejected |
                | Time-series sequential forecasting | ✕ Not supported |
                """
            )

        st.markdown("---")
        st.markdown("#### 📝 Example CSV Schema")
        st.markdown(
            """
            Each row = one observation. Each column = one feature. One column = binary target.

            | record_id | age | monthly_income | tenure_months | support_calls | contract_type | payment_method | target |
            |---|---|---|---|---|---|---|---|
            | REC-001 | 24 | 35000 | 4 | 5 | Monthly | Electronic | 1 |
            | REC-002 | 42 | 72000 | 38 | 1 | Annual | Credit Card | 0 |
            | REC-003 | 31 | 48000 | 12 | 3 | Monthly | Electronic | 1 |

            - **Identifier columns** (e.g. `record_id`, `customerID`) are auto-detected and excluded from features.
            - **Categorical values** should be consistent (e.g. always `Monthly`, not `monthly` / `MONTHLY`).
            - **Numerical columns** must contain valid numbers.
            """
        )

    st.markdown("---")

    # 2. Download Sample Template
    st.markdown("### 📥 Download Sample Dataset Template")
    st.caption("Download a ready-to-use sample CSV that is directly compatible with this engine. You can also modify it with your own data.")
    dl_col1, dl_col2 = st.columns([1, 2])
    with dl_col1:
        template_csv = _generate_sample_template_csv()
        st.download_button(
            label="📥 Download Sample CSV Template (30 rows)",
            data=template_csv,
            file_name="churnguard_sample_template.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl_col2:
        st.info(
            "💡 **What should my CSV look like?** Each row is one observation. Include numerical features "
            "(age, income, tenure), categorical features (contract_type, payment_method), and one binary "
            "target column (0/1 or Yes/No). Identifiers like `record_id` are auto-excluded."
        )

    st.markdown("---")

    # 3. File Upload Area & Sample Cohort Selectors
    st.markdown("### 📤 Step 1: Ingest Your Dataset")

    col_up, col_samp = st.columns([2, 1])

    with col_up:
        uploaded_file = st.file_uploader(
            "Upload a tabular CSV file:",
            type=["csv"],
            help="Upload any tabular CSV file with a binary classification target.",
            key="custom_dataset_file_uploader",
        )

    with col_samp:
        st.markdown("##### 🧪 Or Try a Sample Dataset:")
        sample_choice = st.selectbox(
            "Load demonstration tabular dataset:",
            options=["None (Use Uploaded File)", "Customer Churn (Telco)", "Employee Attrition (HR)", "Loan Default (Credit)"],
            index=0,
            key="custom_sample_dataset_select",
        )

    df_raw: Optional[pd.DataFrame] = None
    dataset_name = "custom_dataset.csv"

    # Handle dataset loading
    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            dataset_name = uploaded_file.name
            st.session_state["active_custom_df"] = df_raw
            st.session_state["active_custom_name"] = dataset_name
        except Exception as e:
            st.error(f"Failed to parse uploaded CSV: {str(e)}. Please ensure the file is a valid UTF-8 CSV.")
            return
    elif sample_choice != "None (Use Uploaded File)":
        if sample_choice == "Customer Churn (Telco)":
            from config.config import RAW_DATA_PATH
            if RAW_DATA_PATH.exists():
                df_raw = pd.read_csv(RAW_DATA_PATH).sample(n=250, random_state=42).reset_index(drop=True)
                dataset_name = "telco_customer_churn_sample_250.csv"
        elif sample_choice == "Employee Attrition (HR)":
            # Generate realistic synthetic HR dataset
            np.random.seed(42)
            n_hr = 200
            hr_records = []
            for i in range(1, n_hr + 1):
                overtime = "Yes" if i % 3 == 0 else "No"
                dist = int(np.random.randint(1, 30))
                satisfaction = int(np.random.randint(1, 5))
                monthly_inc = int(np.random.randint(2500, 15000))
                tenure_yrs = int(np.random.randint(1, 15))
                is_attrited = "Yes" if (overtime == "Yes" and satisfaction <= 2) or (dist > 20 and monthly_inc < 4000) else ("Yes" if i % 6 == 0 else "No")
                hr_records.append({
                    "EmployeeID": f"EMP-{i:04d}",
                    "Age": int(np.random.randint(22, 60)),
                    "Department": np.random.choice(["Sales", "R&D", "Human Resources"]),
                    "DistanceFromHome": dist,
                    "Education": np.random.choice(["Bachelor", "Master", "Doctor", "College"]),
                    "JobSatisfaction": satisfaction,
                    "MonthlyIncome": monthly_inc,
                    "OverTime": overtime,
                    "YearsAtCompany": tenure_yrs,
                    "Attrition": is_attrited,
                })
            df_raw = pd.DataFrame(hr_records)
            dataset_name = "employee_attrition_sample.csv"
        elif sample_choice == "Loan Default (Credit)":
            # Generate realistic synthetic credit dataset
            np.random.seed(101)
            n_loan = 180
            loan_records = []
            for i in range(1, n_loan + 1):
                credit_score = int(np.random.randint(550, 820))
                dti = round(float(np.random.uniform(0.1, 0.6)), 2)
                loan_amt = int(np.random.randint(5000, 40000))
                income = int(np.random.randint(30000, 150000))
                has_delinq = "Yes" if i % 5 == 0 else "No"
                is_default = "Default" if (credit_score < 620 and dti > 0.4) or has_delinq == "Yes" else ("Default" if i % 7 == 0 else "Fully Paid")
                loan_records.append({
                    "LoanID": f"LOAN-{i:05d}",
                    "LoanAmount": loan_amt,
                    "AnnualIncome": income,
                    "CreditScore": credit_score,
                    "DebtToIncome": dti,
                    "HomeOwnership": np.random.choice(["RENT", "MORTGAGE", "OWN"]),
                    "EmploymentYears": int(np.random.randint(1, 12)),
                    "PriorDelinquency": has_delinq,
                    "LoanStatus": is_default,
                })
            df_raw = pd.DataFrame(loan_records)
            dataset_name = "loan_default_credit_sample.csv"

        st.session_state["active_custom_df"] = df_raw
        st.session_state["active_custom_name"] = dataset_name
    elif "active_custom_df" in st.session_state:
        df_raw = st.session_state["active_custom_df"]
        dataset_name = st.session_state.get("active_custom_name", "custom_dataset.csv")

    if df_raw is None:
        st.info("👆 Please upload a CSV file or select a sample dataset above to proceed.")
        return

    # 4. Dataset Health Panel
    st.markdown("### 🩺 Dataset Health Check")
    health_items = []
    health_warnings = []
    health_errors = []

    # Basic checks
    health_items.append(f"✅ File loaded successfully: `{dataset_name}`")
    health_items.append(f"✅ {len(df_raw):,} rows detected")
    health_items.append(f"✅ {len(df_raw.columns):,} columns detected")

    # Check for numerical and categorical features
    num_count = sum(1 for c in df_raw.columns if pd.api.types.is_numeric_dtype(df_raw[c]))
    cat_count = len(df_raw.columns) - num_count
    if num_count > 0:
        health_items.append(f"✅ {num_count} numerical feature(s) detected")
    if cat_count > 0:
        health_items.append(f"✅ {cat_count} categorical/text column(s) detected")

    # Check for binary target candidates
    schema_info = DynamicSchemaDetector.detect_schema(df_raw)
    target_candidates = schema_info["target_candidates"]
    if target_candidates:
        health_items.append(f"✅ Binary target candidate(s) detected: `{'`, `'.join(target_candidates)}`")
    else:
        health_errors.append("❌ **No binary target detected.** No column has exactly 2 unique non-null values. Upload a binary classification dataset.")

    # Check for missing values
    total_missing = int(df_raw.isnull().sum().sum())
    if total_missing > 0:
        pct_missing = total_missing / (len(df_raw) * len(df_raw.columns))
        health_warnings.append(f"⚠️ {total_missing:,} missing value(s) detected ({pct_missing:.1%} of cells). They will be imputed automatically.")

    # Check for class imbalance
    if target_candidates:
        first_target = target_candidates[0]
        vc = df_raw[first_target].value_counts(normalize=True)
        minority_pct = vc.min()
        if minority_pct < 0.15:
            health_warnings.append(f"⚠️ Class imbalance detected in `{first_target}`: minority class = {minority_pct:.1%}. Balanced class weights will be applied.")

    # Check for identifier candidates
    if schema_info["identifier_candidates"]:
        health_warnings.append(f"⚠️ Possible identifier column(s) detected: `{'`, `'.join(schema_info['identifier_candidates'])}`. They will be excluded from features.")

    # Row count check
    if len(df_raw) < 15:
        health_errors.append(f"❌ **Too few rows.** Dataset has {len(df_raw)} rows. A minimum of 15 rows is required.")
    elif len(df_raw) < 30:
        health_warnings.append(f"⚠️ Small dataset ({len(df_raw)} rows). Results may exhibit higher variance. 30+ rows recommended.")

    # Render health panel
    health_md = "\n".join(health_items)
    if health_warnings:
        health_md += "\n\n" + "\n".join(health_warnings)
    if health_errors:
        health_md += "\n\n" + "\n".join(health_errors)

    if health_errors:
        st.error(health_md)
        return
    elif health_warnings:
        st.warning(health_md)
    else:
        st.success(health_md)

    st.markdown("---")

    # 5. Dataset Preview (Actual Uploaded DataFrame)
    st.markdown("### 🔍 Step 2: Ingested Dataset Overview")
    mem_size_kb = df_raw.memory_usage(deep=True).sum() / 1024.0

    ov_c1, ov_c2, ov_c3, ov_c4 = st.columns(4)
    with ov_c1:
        st.metric("Dataset File", dataset_name)
    with ov_c2:
        st.metric("Total Rows", f"{len(df_raw):,}")
    with ov_c3:
        st.metric("Total Columns", f"{len(df_raw.columns):,}")
    with ov_c4:
        st.metric("Memory Footprint", f"{mem_size_kb:.1f} KB")

    st.markdown(f"**First 10 Rows of `{dataset_name}`:**")
    st.table(df_raw.head(10))

    st.markdown("---")

    # 6. Schema Detection & Interactive Target Selection
    st.markdown("### 🎯 Step 3: Target & Feature Configuration")

    # Target Column Selector
    t_col1, t_col2 = st.columns([1.5, 1.5])

    with t_col1:
        if not target_candidates:
            st.error(
                "❌ **No Binary Target Detected:** None of the columns contain exactly 2 unique non-null values. "
                "Please select a column to inspect its class distribution or upload a binary classification dataset."
            )
            target_selection = st.selectbox("Select Target Column Manually:", options=list(df_raw.columns))
        else:
            default_idx = 0
            # Heuristic for common target names
            common_target_names = ["churn", "target", "label", "attrition", "default", "status", "fraud", "outcome"]
            for idx, c in enumerate(target_candidates):
                if any(t_name in c.lower() for t_name in common_target_names):
                    default_idx = idx
                    break

            target_selection = st.selectbox(
                "Select Binary Target Column:",
                options=target_candidates,
                index=default_idx,
                help="The column containing the binary outcome to predict.",
            )

    # Positive Class Selector
    positive_class = None
    if target_selection:
        target_series = df_raw[target_selection].dropna()
        unique_classes = list(target_series.unique())

        if len(unique_classes) != 2:
            st.error(
                f"❌ Column `{target_selection}` has {len(unique_classes)} unique values ({unique_classes[:6]}). "
                f"Binary classification requires exactly 2 classes."
            )
            return

        with t_col2:
            # Smart default positive class (e.g. Yes, 1, Attrited, Default, Positive)
            pos_default_idx = 0
            for p_idx, u_val in enumerate(unique_classes):
                if str(u_val).lower().strip() in ["yes", "1", "true", "positive", "default", "attrition", "attrited", "fraud", "churn"]:
                    pos_default_idx = p_idx
                    break

            positive_class = st.selectbox(
                f"Select Positive Class for `{target_selection}`:",
                options=unique_classes,
                index=pos_default_idx,
                help="The event of interest (e.g. Churn=Yes, Default=1). Precision and Recall will evaluate this class.",
            )

        # Show target distribution
        n_pos = int(np.sum(target_series.astype(str).str.strip().str.lower() == str(positive_class).strip().lower()))
        n_neg = len(target_series) - n_pos
        st.caption(
            f"📊 Target Distribution: **Positive Class (`{positive_class}`)** = {n_pos:,} ({n_pos/len(target_series):.1%}) | "
            f"**Negative Class** = {n_neg:,} ({n_neg/len(target_series):.1%})"
        )

    # Identifier Column Exclusion
    st.markdown("##### 🔑 Identifier & Excluded Columns")
    detected_ids = schema_info["identifier_candidates"]
    excluded_ids = st.multiselect(
        "Exclude unique identifiers or metadata columns from predictive features:",
        options=[c for c in df_raw.columns if c != target_selection],
        default=detected_ids,
        help="Identifiers should be excluded to prevent the model from memorizing IDs.",
    )

    # Display Detected Feature Roles
    active_features = [c for c in df_raw.columns if c != target_selection and c not in excluded_ids]
    num_detected = [c for c in active_features if pd.api.types.is_numeric_dtype(df_raw[c]) and df_raw[c].nunique() > 2]
    cat_detected = [c for c in active_features if c not in num_detected]

    st.markdown(
        f"**Active Predictive Features ({len(active_features)}):** "
        f"`{len(num_detected)} Numerical` | `{len(cat_detected)} Categorical`"
    )

    st.markdown("---")

    # 7. Advanced Training Hyperparameters (Collapsible)
    with st.expander("⚙️ Advanced Training Settings (Optional)", expanded=False):
        st.markdown("Customize optimization hyperparameters for this custom dataset:")
        hp_c1, hp_c2, hp_c3, hp_c4 = st.columns(4)
        with hp_c1:
            lr_val = st.number_input("Learning Rate (α)", value=0.05, min_value=0.001, max_value=1.0, step=0.01)
        with hp_c2:
            epochs_val = st.number_input("Max Iterations", value=1000, min_value=100, max_value=5000, step=100)
        with hp_c3:
            l2_val = st.number_input("L2 Regularization (λ)", value=0.01, min_value=0.0, max_value=1.0, step=0.005)
        with hp_c4:
            val_pct = st.slider("Validation Split %", min_value=10, max_value=25, value=15, step=1)

    # 8. Execute Dynamic Training & Evaluation
    st.markdown("### 🚀 Step 4: Execute Model Training & Inference")
    st.markdown(
        f"Click below to fit zero-leakage preprocessing, train the **Custom NumPy Logistic Regression model** "
        f"on `{dataset_name}`, optimize the threshold on validation data, and generate full dataset predictions."
    )

    btn_col1, btn_col2 = st.columns([3, 1])
    with btn_col1:
        train_clicked = st.button("🚀 Train Custom NumPy Model & Analyze", type="primary", use_container_width=True)
    with btn_col2:
        clear_clicked = st.button("🗑️ Clear Dataset & Results", use_container_width=True)

    if clear_clicked:
        for key in [
            "custom_training_results",
            "custom_results_dataset_name",
            "active_custom_df",
            "active_custom_name",
            "custom_sample_dataset_select",
        ]:
            st.session_state.pop(key, None)
        st.rerun()

    if train_clicked:
        with st.spinner("Executing dynamic ML pipeline with pure NumPy..."):
            progress_bar = st.progress(0, text="Preparing data...")
            try:
                progress_bar.progress(10, text="Validating target and splitting data...")
                custom_hp = {
                    "learning_rate": lr_val,
                    "max_iter": int(epochs_val),
                    "l2_lambda": l2_val,
                    "val_size": float(val_pct) / 100.0,
                }

                progress_bar.progress(30, text="Training Custom NumPy Logistic Regression...")
                results = DynamicPipelineTrainer.run_training_pipeline(
                    df=df_raw,
                    target_col=target_selection,
                    positive_class=positive_class,
                    id_cols=excluded_ids,
                    val_size=float(val_pct) / 100.0,
                    test_size=0.15,
                    random_seed=42,
                    hyperparams=custom_hp,
                )

                progress_bar.progress(80, text="Generating predictions and explanations...")
                st.session_state["custom_training_results"] = results
                st.session_state["custom_results_dataset_name"] = dataset_name
                progress_bar.progress(100, text="Analysis complete!")
                st.success(f"🎉 Successfully trained Custom NumPy model on `{dataset_name}` in {results['metadata']['iterations_trained']} iterations!")
            except Exception as e:
                progress_bar.empty()
                st.error(
                    f"❌ **Training pipeline failed.**\n\n"
                    f"**What happened:** {str(e)}\n\n"
                    f"**How to fix:** Ensure your CSV has a valid binary target (exactly 2 classes), "
                    f"at least 15 rows with ≥ 2 instances per class, and at least one numerical or categorical feature column."
                )
                return

    # 9. Render Custom Results Dashboard
    if "custom_training_results" in st.session_state and st.session_state.get("custom_results_dataset_name") == dataset_name:
        res = st.session_state["custom_training_results"]
        meta = res["metadata"]
        test_m = res["test_metrics"]
        opt_m = test_m["at_optimal_threshold"]
        preds = res["predictions"]

        # Derive dynamic label from target column name
        target_label = str(meta.get("target_col", "Prediction")).replace("_", " ").title()
        pos_label = str(meta.get("positive_class", "Positive"))

        st.markdown("---")
        st.markdown(f"## 📊 {target_label} Analysis Results Dashboard")
        st.caption(f"Results generated genuinely from `{dataset_name}` using scratch NumPy Logistic Regression. Official capstone demo metrics remain unaffected.")

        # Metric Summary Header
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        with m_col1:
            st.metric("Total Observations", f"{meta['total_rows']:,}")
        with m_col2:
            st.metric("Active Features", f"{meta['feature_count']:,}")
        with m_col3:
            st.metric(f"`{pos_label}` Event Rate", f"{meta['positive_rate']:.1%}")
        with m_col4:
            st.metric("Selected Threshold", f"{meta['selected_threshold']:.2f}")
        with m_col5:
            st.metric("Validation F1", f"{meta['val_f1_at_threshold']:.4f}")

        # Test Set Performance Scorecard
        st.markdown("### 🏆 Independent Test Set Performance Scorecard")
        st.caption(f"Evaluated on {meta['test_rows']:,} holdout test observations not seen during training:")

        sc1, sc2, sc3, sc4, sc5 = st.columns(5)
        with sc1:
            st.metric("ROC-AUC Score", f"{test_m['roc_auc']:.4f}")
        with sc2:
            st.metric("PR-AUC Score", f"{test_m['pr_auc']:.4f}")
        with sc3:
            st.metric(f"Accuracy (t={meta['selected_threshold']})", f"{opt_m['accuracy']:.1%}")
        with sc4:
            st.metric(f"Recall (t={meta['selected_threshold']})", f"{opt_m['recall']:.1%}")
        with sc5:
            st.metric(f"F1-Score (t={meta['selected_threshold']})", f"{opt_m['f1']:.4f}")

        # Confusion Matrix
        cm = opt_m["confusion_matrix"]
        st.markdown(
            f"""
            <div style="background: #F0F9FF; border: 1px solid #BAE6FD; border-radius: 8px; padding: 14px 18px; font-size: 13.5px; margin-top: 8px; color: #0369A1;">
                <strong style="color: #0284C7;">Test Confusion Matrix:</strong> True Negatives = <code style="color: #0369A1; background: #E0F2FE; padding: 2px 6px; border-radius: 4px;">{cm[0][0]}</code> | 
                False Positives = <code style="color: #D97706; background: #FEF3C7; padding: 2px 6px; border-radius: 4px;">{cm[0][1]}</code> | False Negatives = <code style="color: #DC2626; background: #FEE2E2; padding: 2px 6px; border-radius: 4px;">{cm[1][0]}</code> | 
                True Positives = <code style="color: #16A34A; background: #DCFCE7; padding: 2px 6px; border-radius: 4px;">{cm[1][1]}</code>.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Global Feature Importance
        st.markdown("### 🧬 Global Feature Importance & Odds Ratios")
        st.markdown("Key predictive drivers learned by the Custom NumPy Logistic Regression model:")

        fi_col1, fi_col2 = st.columns(2)
        with fi_col1:
            st.markdown(f"##### 🔺 Top Features Pushing TOWARD `{pos_label}`")
            df_pos = pd.DataFrame(res["feature_importance"]["top_positive"])
            if not df_pos.empty:
                st.table(df_pos[["Feature", "Weight (Log-Odds)", "Odds Ratio"]])
            else:
                st.info("No positive coefficients found.")

        with fi_col2:
            st.markdown(f"##### 🔻 Top Features Pushing AWAY from `{pos_label}`")
            df_neg = pd.DataFrame(res["feature_importance"]["top_negative"])
            if not df_neg.empty:
                st.table(df_neg[["Feature", "Weight (Log-Odds)", "Odds Ratio"]])
            else:
                st.info("No negative coefficients found.")

        st.markdown("---")

        # Full Scored Observations Table (For ALL N uploaded rows)
        st.markdown(f"### 📋 Full Scored Predictions Table ({len(df_raw):,} Observations)")
        st.caption("Predictions generated for every single row in the uploaded dataset:")

        df_scored = df_raw.copy()
        prob_col_name = f"{target_label} Probability"
        df_scored[prob_col_name] = np.round(preds["probabilities"], 4)
        thresh_val = float(meta["selected_threshold"])
        tier_conditions = [
            preds["probabilities"] >= thresh_val,
            preds["probabilities"] >= (thresh_val * 0.6),
        ]
        tier_choices = [f"High {target_label}", f"Medium {target_label}"]
        tier_col_name = f"{target_label} Tier"
        df_scored[tier_col_name] = np.select(tier_conditions, tier_choices, default=f"Low {target_label}")

        # Filters
        f_col1, f_col2 = st.columns([1, 2])
        with f_col1:
            all_tiers = [f"High {target_label}", f"Medium {target_label}", f"Low {target_label}"]
            tier_filter = st.selectbox(
                f"Filter by {target_label} Tier:",
                options=["All Observations"] + all_tiers,
                index=0,
            )
        with f_col2:
            search_txt = st.text_input("Search observations:", placeholder="Search by text value...")

        df_table = df_scored.copy()
        if tier_filter != "All Observations":
            df_table = df_table[df_table[tier_col_name] == tier_filter]
        if search_txt:
            mask = df_table.astype(str).apply(lambda row: row.str.contains(search_txt, case=False).any(), axis=1)
            df_table = df_table[mask]

        st.dataframe(df_table.astype(str), use_container_width=True)

        # Export Predictions CSV
        export_csv_data = df_scored.to_csv(index=False)
        export_filename = f"{dataset_name.rsplit('.', 1)[0]}_predictions.csv"

        st.download_button(
            label=f"📥 Download Scored Predictions CSV ({len(df_scored):,} rows)",
            data=export_csv_data,
            file_name=export_filename,
            mime="text/csv",
            type="primary",
            help="Download the complete scored dataset with model probabilities and predictions.",
        )
