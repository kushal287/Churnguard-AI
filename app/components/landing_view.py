"""
Landing / Home View Component for ChurnGuard AI.
Provides enterprise project introduction, architecture highlights,
and two prominent entry points: Mode A (Interactive Demo) and Mode B (Analyze Your Dataset).
Light Enterprise Theme Edition.
"""

from typing import Callable
import streamlit as st


def render_landing_view(navigate_to: Callable[[str], None]):
    """Render the main landing and product introduction page with light enterprise styling."""
    # Product Narrative Intro Card
    st.markdown(
        """
        <div style="background: #FFFFFF; border-left: 4px solid #2563EB; border-top: 1px solid #E2E8F0; border-right: 1px solid #E2E8F0; border-bottom: 1px solid #E2E8F0; border-radius: 12px; padding: 22px 26px; margin-bottom: 26px; box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
                <span style="font-size: 20px;">🛡️</span>
                <h3 style="margin: 0; color: #1E40AF; font-size: 19px; font-weight: 700; letter-spacing: -0.3px;">Enterprise Retention Intelligence Platform</h3>
            </div>
            <p style="font-size: 14.5px; line-height: 1.65; margin: 0; color: #334155;">
                An enterprise-grade, explainable machine learning platform engineered from first principles in pure NumPy. 
                Identify customer churn risks in advance, unpack transparent log-odds waterfall attribution, and deploy 
                data-driven prescriptive retention interventions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Two Main Product Entry Points
    st.markdown("### 🚀 Select an Experience Mode")
    col_demo, col_user = st.columns(2)

    with col_demo:
        st.markdown(
            """
            <div style="background: #FFFFFF; border: 1.5px solid #93C5FD; border-radius: 14px; padding: 24px; height: 100%; display: flex; flex-direction: column; justify-content: space-between; box-shadow: 0 6px 20px rgba(37, 99, 235, 0.06); transition: all 0.2s ease;">
                <div>
                    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
                        <span style="font-size: 26px; background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 10px; padding: 4px 8px;">🎭</span>
                        <div>
                            <h3 style="margin: 0; color: #1D4ED8; font-size: 19px; font-weight: 700;">Mode A — Interactive Demo</h3>
                            <span style="font-size: 11px; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Frozen Canonical Model</span>
                        </div>
                    </div>
                    <p style="font-size: 13.5px; color: #334155; line-height: 1.6; margin-bottom: 16px;">
                        Explore a guided, deterministic demonstration powered by the official frozen Telco model. 
                        Test realistic customer archetypes, inspect mathematical log-odds waterfalls, and review prescriptive retention playbooks.
                    </p>
                    <ul style="font-size: 13px; color: #475569; line-height: 1.7; padding-left: 20px; margin-bottom: 20px;">
                        <li><strong style="color: #0F172A;">Pre-loaded customer personas:</strong> High, Medium, Low risk archetypes</li>
                        <li><strong style="color: #0F172A;">Real-time feature adjustments:</strong> Dynamic logit &amp; probability calculation</li>
                        <li><strong style="color: #0F172A;">Exact log-odds waterfall:</strong> Zero-discrepancy additive attribution</li>
                        <li><strong style="color: #0F172A;">Guided Tour:</strong> 7-step executive evaluation walkthrough</li>
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("▶️ Launch Interactive Demo", use_container_width=True, type="primary", key="btn_landing_mode_a"):
            navigate_to("🎭 Interactive Demo & Simulator")

    with col_user:
        st.markdown(
            """
            <div style="background: #FFFFFF; border: 1.5px solid #A7F3D0; border-radius: 14px; padding: 24px; height: 100%; display: flex; flex-direction: column; justify-content: space-between; box-shadow: 0 6px 20px rgba(16, 185, 129, 0.06); transition: all 0.2s ease;">
                <div>
                    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
                        <span style="font-size: 26px; background: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 10px; padding: 4px 8px;">📁</span>
                        <div>
                            <h3 style="margin: 0; color: #065F46; font-size: 19px; font-weight: 700;">Mode B — Analyze Your Dataset</h3>
                            <span style="font-size: 11px; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Dynamic Tabular ML Engine</span>
                        </div>
                    </div>
                    <p style="font-size: 13.5px; color: #334155; line-height: 1.6; margin-bottom: 16px;">
                        Upload any tabular CSV for binary classification (Churn, Fraud, Attrition, Loan Default, Medical, etc.). 
                        The platform dynamically detects your schema, trains a fresh NumPy model from scratch, and generates scored predictions.
                    </p>
                    <ul style="font-size: 13px; color: #475569; line-height: 1.7; padding-left: 20px; margin-bottom: 20px;">
                        <li><strong style="color: #0F172A;">Automated schema detection:</strong> Automatic target &amp; identifier candidate parsing</li>
                        <li><strong style="color: #0F172A;">Zero-leakage pipeline:</strong> StandardScaler &amp; OneHotEncoder fitted strictly on train</li>
                        <li><strong style="color: #0F172A;">Threshold optimization:</strong> Dynamic validation-tuned decision boundary</li>
                        <li><strong style="color: #0F172A;">Cohort export:</strong> Full scored predictions CSV download</li>
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("📤 Upload & Analyze Your Dataset", use_container_width=True, key="btn_landing_mode_b"):
            navigate_to("📁 Analyze Your Dataset")

    st.markdown("---")

    # Platform Pillars & Architectural Guarantees
    st.markdown("### ⚙️ Core Technical Pillars & Architecture")
    p1, p2, p3, p4 = st.columns(4)

    with p1:
        st.markdown(
            """
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-top: 3px solid #2563EB; border-radius: 10px; padding: 18px; height: 100%; box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03);">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    <span style="font-size: 18px;">⚙️</span>
                    <h5 style="color: #1E40AF; margin: 0; font-size: 15px; font-weight: 700;">1. Pure NumPy Math</h5>
                </div>
                <p style="font-size: 12.5px; color: #475569; line-height: 1.55; margin-bottom: 0;">
                    Primary Logistic Regression built 100% from scratch using vectorized NumPy algebra (Sigmoid, Weighted BCE, Momentum GD).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with p2:
        st.markdown(
            """
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-top: 3px solid #16A34A; border-radius: 10px; padding: 18px; height: 100%; box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03);">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    <span style="font-size: 18px;">⚔️</span>
                    <h5 style="color: #166534; margin: 0; font-size: 15px; font-weight: 700;">2. Scientific Parity</h5>
                </div>
                <p style="font-size: 12.5px; color: #475569; line-height: 1.55; margin-bottom: 0;">
                    Audited against Scikit-Learn with near-identical ROC-AUC (0.8452 vs 0.8449), r=0.9982 probability correlation, and 4x faster inference.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with p3:
        st.markdown(
            """
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-top: 3px solid #D97706; border-radius: 10px; padding: 18px; height: 100%; box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03);">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    <span style="font-size: 18px;">🧬</span>
                    <h5 style="color: #B45309; margin: 0; font-size: 15px; font-weight: 700;">3. 100% Explainable</h5>
                </div>
                <p style="font-size: 12.5px; color: #475569; line-height: 1.55; margin-bottom: 0;">
                    Every customer prediction decomposes into exact linear log-odds contributions (z = b + Σw_j x_j) with zero black-box opacity.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with p4:
        st.markdown(
            """
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-top: 3px solid #7C3AED; border-radius: 10px; padding: 18px; height: 100%; box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03);">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    <span style="font-size: 18px;">📋</span>
                    <h5 style="color: #6D28D9; margin: 0; font-size: 15px; font-weight: 700;">4. Actionable Playbooks</h5>
                </div>
                <p style="font-size: 12.5px; color: #475569; line-height: 1.55; margin-bottom: 0;">
                    Deterministic business rule engine maps identified risk factors to prioritized, cost-modeled retention interventions.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
