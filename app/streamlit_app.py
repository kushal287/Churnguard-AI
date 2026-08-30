"""
ChurnGuard AI — Explainable Customer Churn Prediction & Retention Intelligence Platform.
Main Streamlit Application Entrypoint with Dual-Mode Architecture (Demo Mode vs User Dataset Mode).
Light Enterprise Theme Edition.
"""

import base64
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st

from app.components.benchmark_view import render_benchmark_view
from app.components.executive_view import render_executive_view
from app.components.guided_demo_view import render_guided_demo_view
from app.components.landing_view import render_landing_view
from app.components.single_prediction_view import render_single_prediction_view
from app.components.user_dataset_view import render_user_dataset_view
from src.utils.integrity import verify_artifact_integrity

from PIL import Image

LOGO_PATH = BASE_DIR / "app" / "assets" / "logo.png"


def get_logo_base64() -> str:
    """Return base64-encoded string of the official logo."""
    if LOGO_PATH.exists():
        with open(LOGO_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""


# Set Streamlit Page Configuration with Branded Logo Favicon
app_icon = Image.open(LOGO_PATH) if LOGO_PATH.exists() else "🛡️"
st.set_page_config(
    page_title="ChurnGuard AI — Explainable Retention Intelligence",
    page_icon=app_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Light Enterprise AI Analytics Design System (Vanilla CSS)
logo_b64 = get_logo_base64()

# Inject branded favicon via HTML head tags (f-string needed for base64 variable)
st.markdown(
    f"""
    <head>
        <link rel="icon" type="image/png" href="data:image/png;base64,{logo_b64}">
        <link rel="shortcut icon" type="image/png" href="data:image/png;base64,{logo_b64}">
        <link rel="apple-touch-icon" href="data:image/png;base64,{logo_b64}">
    </head>
    """,
    unsafe_allow_html=True,
)

# CSS Design System (regular string — no f-string to avoid CSS brace conflicts)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Core Typography & Light Theme Canvas */
    html, body, .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        background-color: #F8FAFC !important;
        color: #0F172A !important;
        letter-spacing: -0.01em;
    }
    
    /* Ensure icon fonts are never overridden by Inter font */
    [data-testid="stIconMaterial"], [class*="material-icons"], [class*="material-symbols"], i, [data-testid="stExpanderToggleIcon"] {
        font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons', sans-serif !important;
    }

    code, pre, .mono-font {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Clean subtle light background mesh */
    .stApp {
        background-image: 
            radial-gradient(circle at 10% 10%, rgba(37, 99, 235, 0.035) 0%, transparent 40%),
            radial-gradient(circle at 90% 90%, rgba(6, 182, 212, 0.025) 0%, transparent 40%),
            radial-gradient(rgba(15, 23, 42, 0.02) 1px, transparent 1px) !important;
        background-size: 100% 100%, 100% 100%, 24px 24px !important;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(5px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulseGlowGreen {
        0%, 100% { box-shadow: 0 0 0 0 rgba(22, 163, 74, 0.6); }
        50% { box-shadow: 0 0 0 5px rgba(22, 163, 74, 0); }
    }
    
    /* Main Top Header Banner - Clean Light Enterprise */
    .main-header {
        background: linear-gradient(135deg, #FFFFFF 0%, #F0F7FF 55%, #E0F2FE 100%);
        border: 1px solid #BAE6FD;
        padding: 22px 28px;
        border-radius: 14px;
        color: #0F172A;
        margin-bottom: 24px;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.06), 0 1px 3px rgba(0, 0, 0, 0.03);
        display: flex;
        align-items: center;
        gap: 22px;
        animation: fadeIn 0.25s ease-out;
    }
    
    .main-header h1 {
        font-size: 28px !important;
        font-weight: 800 !important;
        margin: 0 !important;
        letter-spacing: -0.5px !important;
        color: #0F172A !important;
    }
    
    .main-header p {
        font-size: 14.5px !important;
        margin: 4px 0 0 0 !important;
        color: #334155 !important;
        font-weight: 400;
    }
    
    .header-logo-container {
        width: 76px;
        height: 76px;
        border-radius: 14px;
        background: #FFFFFF;
        border: 1.5px solid #93C5FD;
        padding: 6px;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.12);
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .header-logo-container:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.2);
    }
    
    .header-logo-img {
        width: 100%;
        height: 100%;
        object-fit: contain;
    }
    
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11.5px;
        font-weight: 600;
        background: #FFFFFF;
        border: 1px solid #BFDBFE;
        color: #1D4ED8;
        margin-top: 8px;
        margin-right: 6px;
        box-shadow: 0 1px 3px rgba(37, 99, 235, 0.08);
    }
    
    /* Sidebar Customization - Clean Light Panel */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    
    .sidebar-logo-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 16px 12px;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        text-align: center;
        transition: all 0.2s ease;
    }
    
    .sidebar-logo-card:hover {
        border-color: #93C5FD;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.08);
    }
    
    .sidebar-status-pill {
        display: flex;
        align-items: center;
        gap: 8px;
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        border-radius: 20px;
        padding: 4px 10px;
        font-size: 11px;
        font-weight: 600;
        color: #065F46;
        margin-top: 10px;
        width: fit-content;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #16A34A;
        animation: pulseGlowGreen 2s infinite;
    }
    
    /* Metric Cards - Clean White */
    div[data-testid="stMetric"] {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    div[data-testid="stMetric"]:hover {
        border-color: #93C5FD !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.08) !important;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 26px !important;
        font-weight: 800 !important;
        color: #0F172A !important;
        letter-spacing: -0.5px !important;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 12px !important;
        font-weight: 600 !important;
        color: #64748B !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Buttons - Professional Light Theme */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border: 1px solid #1D4ED8 !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.25) !important;
        padding: 8px 20px !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
        background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%) !important;
    }
    
    div.stButton > button:not([kind="primary"]) {
        background: #FFFFFF !important;
        color: #1E293B !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
        font-weight: 500 !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    div.stButton > button:not([kind="primary"]):hover {
        background: #F8FAFC !important;
        border-color: #93C5FD !important;
        color: #0F172A !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06) !important;
    }
    
    /* Expanders & Cards */
    div[data-testid="stExpander"] {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03) !important;
        margin-bottom: 16px !important;
    }
    
    /* Tabs */
    button[data-baseweb="tab"] {
        color: #64748B !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 10px 18px !important;
        transition: all 0.2s ease !important;
    }
    
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #2563EB !important;
        border-bottom-color: #2563EB !important;
    }
    
    /* Progress Bars */
    div[data-testid="stProgress"] > div > div > div > div {
        background: linear-gradient(90deg, #2563EB 0%, #06B6D4 100%) !important;
        border-radius: 6px !important;
    }
    
    /* Dataframe / Table styling */
    div[data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        background-color: #FFFFFF !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02) !important;
    }
    
    hr {
        border-color: #E2E8F0 !important;
        margin: 24px 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    """Main application lifecycle controller and dynamic router."""
    # Startup Artifact Integrity Verification
    artifacts_ok, artifact_issues = verify_artifact_integrity()
    if not artifacts_ok:
        st.error(
            "⚠️ **Authoritative ML Artifact Integrity Warning**\n\n"
            "One or more required frozen experiment artifacts are missing or unreadable:\n"
            + "\n".join([f"- {issue}" for issue in artifact_issues])
            + "\n\n**Resolution:** Please run `python -m src.pipeline.train_pipeline` from the project root "
            "to generate the frozen experiment artifacts, then reload the application."
        )

    nav_options = [
        "🏠 Home & Overview",
        "🎭 Interactive Demo & Simulator",
        "📁 Analyze Your Dataset",
        "📊 Executive Command Center",
        "⚔️ Model Benchmark Arena",
    ]

    # Handle pending programmatic navigation safely before rendering widgets
    if "pending_navigation" in st.session_state and st.session_state["pending_navigation"]:
        st.session_state["current_page"] = st.session_state["pending_navigation"]
        st.session_state["pending_navigation"] = None

    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "🏠 Home & Overview"

    def navigate_to(page_name: str):
        """Programmatically switch active page and trigger clean rerun."""
        st.session_state["pending_navigation"] = page_name
        st.rerun()

    # Header Banner with Official Logo in Clean Light Container
    logo_b64 = get_logo_base64()
    if logo_b64:
        st.markdown(
            f"""
            <div class="main-header">
                <div class="header-logo-container">
                    <img src="data:image/png;base64,{logo_b64}" class="header-logo-img" alt="ChurnGuard AI Logo" />
                </div>
                <div style="flex-grow: 1;">
                    <h1>ChurnGuard AI</h1>
                    <p>Explainable Customer Churn Prediction & Retention Intelligence Platform</p>
                    <span class="badge-pill">⚙️ Pure NumPy Logistic Regression (From Scratch)</span>
                    <span class="badge-pill">⚔️ Scikit-Learn Benchmark Audited (0.8452 vs 0.8449 ROC-AUC)</span>
                    <span class="badge-pill">🧬 100% Explainable Log-Odds Waterfall</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="main-header">
                <div style="flex-grow: 1;">
                    <h1>🛡️ ChurnGuard AI</h1>
                    <p>Explainable Customer Churn Prediction & Retention Intelligence Platform</p>
                    <span class="badge-pill">⚙️ Pure NumPy Logistic Regression (From Scratch)</span>
                    <span class="badge-pill">⚔️ Scikit-Learn Benchmark Audited (0.8452 vs 0.8449 ROC-AUC)</span>
                    <span class="badge-pill">🧬 100% Explainable Log-Odds Waterfall</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Sidebar Navigation with Clean White Logo Card & System Health Indicator
    if LOGO_PATH.exists():
        st.sidebar.markdown(
            f"""
            <div class="sidebar-logo-card">
                <div style="width: 120px; height: 120px; margin: 0 auto 10px auto; border-radius: 14px; background: #FFFFFF; border: 1.5px solid #E2E8F0; padding: 8px; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04); display: flex; align-items: center; justify-content: center;">
                    <img src="data:image/png;base64,{logo_b64}" style="width: 100%; height: 100%; object-fit: contain;" alt="Logo" />
                </div>
                <div style="font-size: 16px; font-weight: 800; color: #0F172A; letter-spacing: -0.3px;">ChurnGuard AI</div>
                <div style="font-size: 12px; color: #64748B; margin-top: 2px;">Retention Intelligence Platform</div>
                <div style="display: flex; justify-content: center; margin-top: 8px;">
                    <div class="sidebar-status-pill">
                        <div class="status-dot"></div>
                        <span>All Systems Operational</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown("# 🛡️ ChurnGuard AI")
        st.sidebar.markdown(
            """
            <div class="sidebar-status-pill">
                <div class="status-dot"></div>
                <span>All Systems Operational</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    current_idx = (
        nav_options.index(st.session_state["current_page"])
        if st.session_state["current_page"] in nav_options
        else 0
    )

    nav_selection = st.sidebar.radio(
        "Navigation Menu",
        options=nav_options,
        index=current_idx,
    )

    if nav_selection != st.session_state["current_page"]:
        st.session_state["current_page"] = nav_selection
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📌 Architecture Summary")
    st.sidebar.markdown(
        """
        - **Primary Model:** Custom NumPy LR
        - **Loss:** Weighted Binary Cross-Entropy
        - **Optimizer:** Mini-Batch Momentum GD
        - **Regularization:** L2 ($\lambda=0.01$)
        - **Class Weights:** Balanced ($w_0, w_1$)
        - **Decision Threshold:** $t^* = 0.58$
        - **Benchmark:** Scikit-Learn L-BFGS
        - **Zero Leakage:** Train Split Only
        """
    )
    st.sidebar.markdown("---")

    # Global Session Reset Control in Sidebar
    with st.sidebar.expander("🛠️ Session Controls", expanded=False):
        st.caption("Reset application session state to defaults:")
        if st.button("🔄 Reset Entire Session", use_container_width=True, key="sidebar_reset_btn"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.session_state["current_page"] = "🏠 Home & Overview"
            st.rerun()

    st.sidebar.caption("© 2026 ChurnGuard AI. All rights reserved.")

    # Page Routing
    active_page = st.session_state["current_page"]

    if active_page == "🏠 Home & Overview":
        render_landing_view(navigate_to=navigate_to)
    elif active_page == "🎭 Interactive Demo & Simulator":
        demo_mode_tab1, demo_mode_tab2 = st.tabs(["🔍 Live Customer Simulator", "🧭 7-Step Guided Tour"])
        with demo_mode_tab1:
            render_single_prediction_view()
        with demo_mode_tab2:
            render_guided_demo_view()
    elif active_page == "📁 Analyze Your Dataset":
        render_user_dataset_view()
    elif active_page == "📊 Executive Command Center":
        render_executive_view()
    elif active_page == "⚔️ Model Benchmark Arena":
        render_benchmark_view()


if __name__ == "__main__":
    main()
