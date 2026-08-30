"""
Single Customer Churn Predictor & Simulator Component for ChurnGuard AI.
Provides real-time interactive risk scoring, exact linear log-odds waterfall explainability,
and prescriptive retention action playbooks locked to the validation threshold (0.58).
Light Enterprise Theme Edition.
"""

from typing import Any, Dict
import numpy as np
import pandas as pd
import streamlit as st

from config.config import CUSTOM_MODEL_PATH, PREPROCESSOR_PATH
from src.data.preprocessor import DataPreprocessor
from src.explainability.individual_explainer import IndividualExplainer
from src.explainability.retention_playbook import RetentionPlaybook
from src.models.custom_logistic_regression import CustomLogisticRegression


@st.cache_resource
def load_models_and_explainer():
    """Load cached model, preprocessor, and explainer instances."""
    if not CUSTOM_MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        return None, None, None, None
    model = CustomLogisticRegression.load(CUSTOM_MODEL_PATH)
    preprocessor = DataPreprocessor.load(PREPROCESSOR_PATH)
    explainer = IndividualExplainer(
        feature_names=preprocessor.get_feature_names(),
        weights=model.weights,
        bias=model.bias,
    )
    playbook = RetentionPlaybook(threshold=0.58)
    return model, preprocessor, explainer, playbook


def render_single_prediction_view():
    """Render Single Customer Prediction & Retention Intelligence Simulator with light enterprise styling."""
    st.markdown("## 🔍 Live Customer Churn Risk Simulator & Explainability")
    st.markdown(
        "Evaluate individual customer profiles with **100% mathematical transparency**. "
        "Every prediction decomposes into exact additive log-odds contributions from the underlying Logistic Regression model."
    )

    loaded = load_models_and_explainer()
    if loaded[0] is None:
        st.error("Model artifacts not found. Please execute the training pipeline first.")
        return

    model, preprocessor, explainer, playbook = loaded

    # Persona Presets & Reset Control
    st.markdown("### 🎭 Quick-Load Persona Archetypes")

    preset_options = [
        "Custom (Manual Input)",
        "High-Risk Archetype: Month-to-Month Fiber Optic Subscriber (New Customer)",
        "Low-Risk Archetype: Long-Term 2-Year Contract Customer (Established)",
        "Moderate-Risk Archetype: Senior Citizen on Electronic Check (Early Tenure)",
    ]

    if "pending_preset" in st.session_state and st.session_state["pending_preset"]:
        st.session_state["selected_preset"] = st.session_state["pending_preset"]
        st.session_state["pending_preset"] = None

    if "selected_preset" not in st.session_state or st.session_state["selected_preset"] not in preset_options:
        st.session_state["selected_preset"] = preset_options[0]

    current_preset_idx = preset_options.index(st.session_state["selected_preset"])

    col_preset, col_reset = st.columns([3, 1])
    with col_preset:
        preset = st.selectbox(
            "Select a pre-configured customer profile to auto-fill the inputs:",
            options=preset_options,
            index=current_preset_idx,
        )
        st.session_state["selected_preset"] = preset

    with col_reset:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Reset Profile", use_container_width=True, help="Reset customer profile inputs to baseline defaults"):
            st.session_state["pending_preset"] = "Custom (Manual Input)"
            st.rerun()

    # Default values based on preset
    if preset == "High-Risk Archetype: Month-to-Month Fiber Optic Subscriber (New Customer)":
        d_gender, d_senior, d_partner, d_dep = "Female", "0", "No", "No"
        d_tenure, d_phone, d_multi = 2, "Yes", "Yes"
        d_internet, d_sec, d_backup = "Fiber optic", "No", "No"
        d_dev, d_tech, d_tv, d_mov = "No", "No", "Yes", "Yes"
        d_contract, d_paperless, d_pay = "Month-to-month", "Yes", "Electronic check"
        d_monthly, d_total = 98.50, 197.00
    elif preset == "Low-Risk Archetype: Long-Term 2-Year Contract Customer (Established)":
        d_gender, d_senior, d_partner, d_dep = "Male", "0", "Yes", "Yes"
        d_tenure, d_phone, d_multi = 65, "Yes", "Yes"
        d_internet, d_sec, d_backup = "DSL", "Yes", "Yes"
        d_dev, d_tech, d_tv, d_mov = "Yes", "Yes", "No", "No"
        d_contract, d_paperless, d_pay = "Two year", "No", "Credit card (automatic)"
        d_monthly, d_total = 64.20, 4173.00
    elif preset == "Moderate-Risk Archetype: Senior Citizen on Electronic Check (Early Tenure)":
        d_gender, d_senior, d_partner, d_dep = "Female", "1", "No", "No"
        d_tenure, d_phone, d_multi = 14, "Yes", "No"
        d_internet, d_sec, d_backup = "DSL", "No", "No"
        d_dev, d_tech, d_tv, d_mov = "No", "No", "No", "No"
        d_contract, d_paperless, d_pay = "Month-to-month", "Yes", "Electronic check"
        d_monthly, d_total = 45.30, 634.20
    else:
        d_gender, d_senior, d_partner, d_dep = "Female", "0", "No", "No"
        d_tenure, d_phone, d_multi = 12, "Yes", "No"
        d_internet, d_sec, d_backup = "Fiber optic", "No", "No"
        d_dev, d_tech, d_tv, d_mov = "No", "No", "No", "No"
        d_contract, d_paperless, d_pay = "Month-to-month", "Yes", "Electronic check"
        d_monthly, d_total = 70.00, 840.00

    # Input Form Layout
    with st.expander("📝 Customer Account & Service Configurations", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 👤 Demographics & Identity")
            gender = st.selectbox("Gender", ["Female", "Male"], index=0 if d_gender == "Female" else 1)
            senior = st.selectbox("Senior Citizen", ["0", "1"], index=0 if d_senior == "0" else 1, help="0: No, 1: Yes (Age 65+)")
            partner = st.selectbox("Partner", ["No", "Yes"], index=0 if d_partner == "No" else 1)
            dependents = st.selectbox("Dependents", ["No", "Yes"], index=0 if d_dep == "No" else 1)
            tenure = st.slider("Tenure (Months Subscribed)", min_value=0, max_value=72, value=int(d_tenure))

        with col2:
            st.markdown("##### 🌐 Subscribed Services")
            phone = st.selectbox("Phone Service", ["Yes", "No"], index=0 if d_phone == "Yes" else 1)
            multi = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], index=["No", "Yes", "No phone service"].index(d_multi))
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], index=["DSL", "Fiber optic", "No"].index(d_internet))
            sec = st.selectbox("Online Security", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(d_sec))
            backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(d_backup))
            dev = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(d_dev))
            tech = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(d_tech))
            tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(d_tv))
            mov = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(d_mov))

        with col3:
            st.markdown("##### 💳 Contract & Billing")
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(d_contract))
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=0 if d_paperless == "Yes" else 1)
            payment = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(d_pay),
            )
            monthly = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=float(d_monthly), step=1.0)
            calc_total = float(d_total) if d_total > 0 else (monthly * max(tenure, 1))
            total = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=calc_total, step=10.0)

    # Build customer record
    customer_record = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multi,
        "InternetService": internet,
        "OnlineSecurity": sec,
        "OnlineBackup": backup,
        "DeviceProtection": dev,
        "TechSupport": tech,
        "StreamingTV": tv,
        "StreamingMovies": mov,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }

    # Transform and Predict via Custom NumPy Model
    X_vec = preprocessor.transform_single_record(customer_record)
    proba = model.predict_proba(X_vec)
    churn_prob = float(proba[0, 1])
    retention_prob = 1.0 - churn_prob

    # Risk Explanation and Classification
    explanation = explainer.explain_instance(X_vec, top_n=5, raw_customer_dict=customer_record)
    risk_info = playbook.classify_risk(churn_prob)
    recommendations = playbook.generate_recommendations(customer_record, churn_prob, explanation["risk_drivers"])

    st.markdown("---")

    # Section 7: Live Customer Risk Summary
    st.markdown("### 🎯 Live Customer Risk Summary")

    p_col1, p_col2, p_col3 = st.columns([1.2, 1, 1])

    with p_col1:
        badge_color = risk_info["badge_color"]
        # Determine background tint for light theme
        if churn_prob >= 0.58:
            bg_tint = "#FEF2F2"
            text_head = "#991B1B"
            desc_color = "#7F1D1D"
        elif churn_prob >= 0.35:
            bg_tint = "#FFFBEB"
            text_head = "#92400E"
            desc_color = "#78350F"
        else:
            bg_tint = "#F0FDF4"
            text_head = "#166534"
            desc_color = "#14532D"

        st.markdown(
            f"""
            <div style="background: {bg_tint}; border: 1.5px solid {badge_color}; border-radius: 12px; padding: 22px; box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);">
                <span style="font-size: 11px; font-weight: 700; color: {text_head}; text-transform: uppercase; letter-spacing: 0.5px;">Decision Threshold: 0.58 (Validation-Selected)</span>
                <h2 style="color: {text_head}; margin: 6px 0 10px 0; font-size: 22px; font-weight: 800;">{risk_info['tier']}</h2>
                <div style="font-size: 34px; font-weight: 800; color: #0F172A; font-family: 'Inter', sans-serif;">{churn_prob:.1%} <span style="font-size: 14px; font-weight: normal; color: #64748B;">Modeled Churn Risk</span></div>
                <p style="margin-top: 10px; margin-bottom: 0; color: {desc_color}; font-size: 13.5px; line-height: 1.5;">{risk_info['status_text']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with p_col2:
        st.markdown("##### Probability Breakdown")
        st.progress(churn_prob, text=f"Modeled Churn Risk: {churn_prob:.1%}")
        st.progress(retention_prob, text=f"Modeled Retention Probability: {retention_prob:.1%}")
        st.markdown(
            f"""
            <div style="font-size: 12.5px; color: #334155; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px; margin-top: 8px; font-family: 'JetBrains Mono', monospace; line-height: 1.6; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
                <div><strong style="color: #64748B;">Model Intercept (b):</strong> {explanation['bias_weight']:+.4f} (Base prior: {explanation['base_probability']:.1%})</div>
                <div><strong style="color: #64748B;">Total Linear Logit (z):</strong> {explanation['total_log_odds']:+.4f}</div>
                <div><strong style="color: #2563EB;">Formula:</strong> p = 1 / (1 + exp(-z))</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with p_col3:
        st.markdown("##### Illustrative Financial Simulation*")
        expected_attrition_cost = churn_prob * 500.0
        st.metric(
            label="Illustrative Attrition Exposure*",
            value=f"${expected_attrition_cost:.2f}",
            help="Modeled Probability × $500 Assumed LTV",
        )
        potential_recovery = 350.0 if churn_prob >= 0.58 else 0.0
        st.metric(
            label="Potential Recovery Value*",
            value=f"${potential_recovery:.2f}",
            help="Assumed net recovered value if targeted retention offer prevents churn",
        )
        st.caption("*Illustrative business simulation assumptions.")

    st.markdown("---")

    # Risk Drivers and Protective Factors
    st.markdown("### 🧬 Mathematical Log-Odds Attribution")
    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        st.markdown("#### 🔴 Top Risk Drivers (Pushes Churn Risk UP)")
        if explanation["risk_drivers"]:
            df_drivers = pd.DataFrame(explanation["risk_drivers"])
            df_drivers.columns = ["Model Feature", "Transformed Input (x)", "Model Weight (w)", "Log-Odds Contribution (+z)"]
            st.dataframe(
                df_drivers.style.format({
                    "Transformed Input (x)": "{:.3f}",
                    "Model Weight (w)": "{:+.4f}",
                    "Log-Odds Contribution (+z)": "+{:.4f}",
                }),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No substantial positive risk drivers identified for this customer profile.")

    with exp_col2:
        st.markdown("#### 🟢 Top Protective Factors (Pushes Churn Risk DOWN)")
        if explanation["protective_factors"]:
            df_anchors = pd.DataFrame(explanation["protective_factors"])
            df_anchors.columns = ["Model Feature", "Transformed Input (x)", "Model Weight (w)", "Log-Odds Contribution (-z)"]
            st.dataframe(
                df_anchors.style.format({
                    "Transformed Input (x)": "{:.3f}",
                    "Model Weight (w)": "{:+.4f}",
                    "Log-Odds Contribution (-z)": "{:.4f}",
                }),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No substantial protective factors present for this customer profile.")

    st.markdown("---")

    # Step-by-Step Waterfall Decomposition Table
    st.markdown("### 📊 Cumulative Log-Odds Waterfall Reconstruction")
    st.markdown(
        "Linear reconstruction demonstrating how the model calculates the final prediction from the prior baseline intercept:"
    )

    df_waterfall = pd.DataFrame(explanation["waterfall_steps"])
    st.dataframe(
        df_waterfall[["step", "feature", "feature_value", "log_odds_delta", "cumulative_log_odds", "implied_churn_probability", "direction"]].rename(
            columns={
                "step": "Waterfall Step",
                "feature": "Feature Description",
                "feature_value": "Transformed Value",
                "log_odds_delta": "Contribution (Δz)",
                "cumulative_log_odds": "Cumulative Logit (z)",
                "implied_churn_probability": "Implied Probability σ(z)",
                "direction": "Impact Direction",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    recon = explanation["mathematical_reconstruction"]
    st.markdown(
        f"""
        <div style="background: #F0F9FF; border: 1px solid #BAE6FD; border-radius: 8px; padding: 14px 18px; font-size: 13.5px; margin-top: 10px; color: #0369A1;">
            <strong style="color: #0284C7;">✓ Mathematical Verification:</strong> Direct Model Output: <code style="color: #0369A1; background: #E0F2FE; padding: 2px 6px; border-radius: 4px;">{recon['direct_probability']:.6f}</code> | 
            Reconstructed Output σ(b + Σw_j x_j): <code style="color: #0369A1; background: #E0F2FE; padding: 2px 6px; border-radius: 4px;">{recon['reconstructed_probability']:.6f}</code> | 
            Discrepancy: <code style="color: #16A34A; background: #DCFCE7; padding: 2px 6px; border-radius: 4px;">{recon['discrepancy']:.2e}</code> (Exact Equivalence).
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Prescriptive Retention Action Playbook
    st.markdown("### 📋 Prescriptive Retention Action Playbook")
    st.markdown("Prioritized, deterministic retention interventions tailored to this customer's modeled risk factors:")

    for idx, rec in enumerate(recommendations, 1):
        if rec["priority"] == "HIGH":
            border_col = "#DC2626"
            badge_bg = "#FEF2F2"
            badge_color = "#991B1B"
            p_badge = "🔴 HIGH"
        elif rec["priority"] == "MEDIUM":
            border_col = "#D97706"
            badge_bg = "#FFFBEB"
            badge_color = "#92400E"
            p_badge = "🟠 MEDIUM"
        else:
            border_col = "#16A34A"
            badge_bg = "#F0FDF4"
            badge_color = "#166534"
            p_badge = "🟢 LOW"

        with st.container():
            st.markdown(
                f"""
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 5px solid {border_col}; border-radius: 10px; padding: 18px; margin-bottom: 14px; box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-weight: 700; font-size: 16px; color: #0F172A;">{idx}. {rec['action']}</span>
                        <span style="font-size: 11.5px; font-weight: 700; background: {badge_bg}; color: {badge_color}; border: 1px solid {border_col}40; padding: 3px 10px; border-radius: 16px;">{p_badge} PRIORITY</span>
                    </div>
                    <p style="margin: 4px 0 8px 0; color: #334155; font-size: 14px; line-height: 1.5;"><strong style="color: #0F172A;">Action:</strong> {rec['description']}</p>
                    <div style="font-size: 13px; color: #64748B; margin-bottom: 8px;"><strong style="color: #334155;">Model Signal Rationale:</strong> {rec['rationale']}</div>
                    <div style="font-size: 12.5px; color: #64748B; display: flex; gap: 20px; flex-wrap: wrap;">
                        <span><strong style="color: #334155;">Expected Impact:</strong> {rec['expected_impact']}</span>
                        <span><strong style="color: #334155;">Illustrative Cost:</strong> {rec['financial_cost']}</span>
                        <span><strong style="color: #334155;">Category:</strong> {rec['category']}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
