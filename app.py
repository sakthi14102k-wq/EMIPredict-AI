import streamlit as st

st.set_page_config(
    page_title="EMIPredict AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Sidebar Navigation ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1.5rem 0 2rem;'>
        <div style='font-size:2.5rem;'>💳</div>       
        <div style='font-size:1.2rem; font-weight:700; color:#f1f5f9; letter-spacing:-0.5px;'>EMIPredict AI</div>
        <div style='font-size:0.75rem; color:#64748b; margin-top:4px;'>Financial Risk Assessment</div>
    </div>
    """, unsafe_allow_html=True)


    page = st.selectbox(
        "Navigation",
        ["🏠  Home", "🔮  EMI Predictor", "📊  EDA Dashboard", "📈  Admin Tracker"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#475569; padding: 0.5rem;'>
        <b style='color:#94a3b8;'>Dataset</b><br>
        400,000 records · 22 features<br><br>
        <b style='color:#94a3b8;'>Models</b><br>
        4 Classifiers · 4 Regressors<br><br>
        <b style='color:#94a3b8;'>Tracking</b><br>
        MLflow 3.10.1
    </div>
    """, unsafe_allow_html=True)

# ── Page routing ───────────────────────────────────────────────

if page == "🏠  Home":
    from pages import home
    home.show()

elif page == "🔮  EMI Predictor":
    from pages import predictor
    predictor.show()

elif page == "📊  EDA Dashboard":
    from pages import eda
    eda.show()

elif page == "📈  Admin Tracker":
    from pages import admin_page
    admin_page.show()
    