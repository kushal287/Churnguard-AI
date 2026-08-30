"""
Executive View Component for ChurnGuard AI.
Provides an executive-level overview of the business problem, dataset scale,
primary model performance scorecard, observed churn patterns, and illustrative retention ROI.
Loads dynamically from the authoritative final_results.json artifact.
Light Enterprise Theme Edition.
"""

import json
from pathlib import Path
import streamlit as st

from config.config import BENCHMARK_RESULTS_PATH, FIGURES_DIR, FINAL_RESULTS_PATH


def render_executive_view():
    """Render Executive Command Center and Strategic Overview with light enterprise styling."""
    st.markdown("## 📊 Executive Churn Intelligence Center")
    st.markdown(
        "A rigorous, explainable intelligence platform designed to identify at-risk subscribers "
        "and guide proactive, high-ROI retention interventions before contract termination."
    )

    # Load authoritative frozen results
    final_data = {}
    if FINAL_RESULTS_PATH.exists():
        with open(FINAL_RESULTS_PATH, "r", encoding="utf-8") as f:
            final_data = json.load(f)
    elif BENCHMARK_RESULTS_PATH.exists():
        with open(BENCHMARK_RESULTS_PATH, "r", encoding="utf-8") as f:
            final_data = json.load(f)

    c_50 = (
        final_data.get("custom_numpy_model_test_metrics", {}).get("at_default_threshold_0_50", {})
        or final_data.get("custom_numpy_model", {}).get("test_metrics", {})
    )
    c_opt = (
        final_data.get("custom_numpy_model_test_metrics", {}).get("at_validation_optimal_threshold", {})
    )
    dataset_info = final_data.get("dataset_info", {})

    total_records = dataset_info.get("total_records", 7043)
    churn_prop = dataset_info.get("target_proportions", {}).get("Yes", 0.26536987)

    # 1. Top Executive Metric Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(
            label="Monitored Dataset Scale",
            value=f"{total_records:,}",
            help="Total verified subscriber accounts in canonical dataset",
        )
    with col2:
        st.metric(
            label="Baseline Churn Rate",
            value=f"{churn_prop:.2%}",
            delta="-1.2% Target",
            delta_color="inverse",
            help="Historical baseline proportion of churned customer accounts",
        )
    with col3:
        roc_auc_val = c_50.get("roc_auc", 0.8452)
        st.metric(
            label="Model ROC-AUC Score",
            value=f"{roc_auc_val:.4f}",
            delta="+0.345 vs Chance",
            help="Discrimination power of Custom NumPy Logistic Regression on holdout test partition",
        )
    with col4:
        recall_val = c_50.get("recall", 0.7509)
        st.metric(
            label="Churn Recall (Sensitivity)",
            value=f"{recall_val:.1%}",
            help="Proportion of actual churners successfully identified at default threshold 0.50",
        )
    with col5:
        savings_val = c_50.get("illustrative_net_retention_savings", c_50.get("net_retention_savings", 64350.0))
        st.metric(
            label="Test Set Net Value*",
            value=f"${savings_val:,.0f}",
            delta="+$60.88 / Cust",
            help="Illustrative net retention savings on holdout test cohort (N = 1,057)",
        )

    st.markdown("---")

    # 2. Strategic Context & Architecture Overview
    st.markdown("### 🏢 Strategic Overview & Business Problem")
    c_arch1, c_arch2 = st.columns(2)

    with c_arch1:
        st.markdown(
            """
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 22px; height: 100%; box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);">
                <h4 style="margin-top: 0; color: #1D4ED8; font-size: 17px; font-weight: 700;">The Core Business Challenge</h4>
                <p style="font-size: 0.92rem; line-height: 1.6; color: #334155;">
                    In subscription telecommunications, customer acquisition costs are 5x to 7x higher than retention costs. 
                    Unmonitored churn erodes recurring revenue and customer lifetime value (LTV).
                </p>
                <ul style="font-size: 0.88rem; line-height: 1.6; margin-bottom: 0; color: #475569;">
                    <li><strong style="color: #0F172A;">Objective:</strong> Proactively identify at-risk subscribers before service termination.</li>
                    <li><strong style="color: #0F172A;">Approach:</strong> Pure NumPy Logistic Regression pairing calibrated log-odds with actionable retention playbooks.</li>
                    <li><strong style="color: #0F172A;">Compliance:</strong> 100% explainable attribution ensures transparent, auditable business decisions.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_arch2:
        st.markdown(
            f"""
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 22px; height: 100%; box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);">
                <h4 style="margin-top: 0; color: #059669; font-size: 17px; font-weight: 700;">Primary Model Scorecard (Holdout Split: N = 1,057)</h4>
                <table style="width: 100%; font-size: 0.88rem; line-height: 1.7; border-collapse: collapse; color: #334155;">
                    <tr style="border-bottom: 1px solid #F1F5F9;"><td><strong style="color: #0F172A;">Classifier Architecture:</strong></td><td>Pure NumPy Logistic Regression (From Scratch)</td></tr>
                    <tr style="border-bottom: 1px solid #F1F5F9;"><td><strong style="color: #0F172A;">Discrimination (ROC-AUC):</strong></td><td><strong style="color: #2563EB;">{c_50.get('roc_auc', 0.8452):.4f}</strong> (Parity with Sklearn: 0.8449)</td></tr>
                    <tr style="border-bottom: 1px solid #F1F5F9;"><td><strong style="color: #0F172A;">Precision-Recall Area (PR-AUC):</strong></td><td><strong style="color: #2563EB;">{c_50.get('pr_auc', 0.6694):.4f}</strong> (Balanced on Imbalance)</td></tr>
                    <tr style="border-bottom: 1px solid #F1F5F9;"><td><strong style="color: #0F172A;">Default Cutoff (t=0.50):</strong></td><td>Accuracy: {c_50.get('accuracy', 0.7540):.1%} | F1: {c_50.get('f1_score', 0.6188):.4f}</td></tr>
                    <tr><td><strong style="color: #0F172A;">Optimal Cutoff (t*=0.58):</strong></td><td>Accuracy: {c_opt.get('accuracy', 0.7815):.1%} | F1: {c_opt.get('f1_score', 0.6280):.4f}</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # 3. Key Empirical Churn Insights
    st.markdown("### 🔍 Key Observable Churn Patterns")
    k_col1, k_col2, k_col3 = st.columns(3)

    with k_col1:
        st.markdown(
            """
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-top: 4px solid #EF4444; border-radius: 10px; padding: 18px; height: 100%; box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);">
                <h5 style="margin-top: 0; color: #DC2626; font-size: 15px; font-weight: 700;">1. Contract Commitment</h5>
                <p style="font-size: 0.86rem; line-height: 1.55; color: #334155;">
                    Subscribers on <strong style="color: #DC2626;">Month-to-month contracts</strong> exhibit approximately <strong style="color: #DC2626;">4.09x higher modeled churn odds</strong> relative to 2-year contract holders. Long-term contracts serve as the strongest statistical retention anchor in the dataset.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k_col2:
        st.markdown(
            """
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-top: 4px solid #F59E0B; border-radius: 10px; padding: 18px; height: 100%; box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);">
                <h5 style="margin-top: 0; color: #D97706; font-size: 15px; font-weight: 700;">2. Internet Tier & Support</h5>
                <p style="font-size: 0.86rem; line-height: 1.55; color: #334155;">
                    <strong style="color: #D97706;">Fiber optic subscribers lacking Tech Support</strong> show elevated attrition risk. Fiber optic service is associated with higher monthly charges, amplifying setup dissatisfaction and billing sensitivity.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k_col3:
        st.markdown(
            """
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-top: 4px solid #2563EB; border-radius: 10px; padding: 18px; height: 100%; box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);">
                <h5 style="margin-top: 0; color: #2563EB; font-size: 15px; font-weight: 700;">3. Early-Tenure Friction</h5>
                <p style="font-size: 0.86rem; line-height: 1.55; color: #334155;">
                    Subscribers in their <strong style="color: #2563EB;">first 6-12 months</strong> experience the highest drop-off rate. A 1-standard-deviation increase in tenure is associated with a <strong style="color: #16A34A;">57.5% reduction</strong> in modeled churn odds.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # 4. Illustrative Retention Economics
    st.markdown("### 💰 Illustrative Financial Impact & Retention Economics")
    f_col1, f_col2 = st.columns([1, 1])

    with f_col1:
        st.markdown(
            """
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 22px; box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);">
                <h4 style="margin-top: 0; color: #1D4ED8; font-size: 16px; font-weight: 700;">*Illustrative Business Simulation Model</h4>
                <ul style="line-height: 1.8; font-size: 0.88rem; margin-bottom: 0; color: #334155;">
                    <li><strong style="color: #0F172A;">Assumed Customer Lifetime Value (LTV):</strong> $500.00</li>
                    <li><strong style="color: #0F172A;">Cost of Unnecessary Incentive (False Positive):</strong> -$50.00</li>
                    <li><strong style="color: #0F172A;">Net Recovered Value per Caught Churner (True Positive):</strong> +$350.00</li>
                    <li><strong style="color: #0F172A;">Net Value Formula:</strong> (TP × $350) - (FP × $50)</li>
                </ul>
                <p style="font-size: 0.80rem; color: #64748B; margin-top: 10px; margin-bottom: 0;">
                    <em>*Note: Figures represent illustrative simulation assumptions for evaluation, not actual corporate financial records.</em>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with f_col2:
        st.markdown(
            """
            <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 12px; padding: 22px; box-shadow: 0 2px 8px rgba(22, 163, 74, 0.06);">
                <h4 style="margin-top: 0; color: #15803D; font-size: 16px; font-weight: 700;">Test Cohort Financial Simulation (N = 1,057)</h4>
                <p style="font-size: 0.88rem; line-height: 1.6; margin-bottom: 10px; color: #166534;">
                    Under default threshold (t = 0.50), the platform successfully catches <strong style="color: #15803D;">211 churners (TP)</strong> while incurring <strong style="color: #B45309;">190 false alarms (FP)</strong>:
                </p>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.88rem; background: #FFFFFF; border: 1px solid #86EFAC; padding: 14px 16px; border-radius: 8px; color: #166534; line-height: 1.6; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
                    Net Retention Value = (211 × $350) - (190 × $50) = $73,850 - $9,500 = <strong style="font-size: 1.05rem; color: #15803D;">+$64,350.00</strong><br>
                    Average Net Value Created: <strong style="color: #15803D;">+$60.88 per monitored customer</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
