import streamlit as st

def show():
    
    st.set_page_config(
        page_title="EMIPredict AI",
        layout="wide"
    )

    # ---- Small Design Style ----
    st.markdown("""
    <style>
    .hero {
        padding:20px;
        border-radius:12px;
        background: #f5f7fb;
        border:1px solid #e6e9f2;
        margin-bottom:20px;
    }
    .card {
        padding:12px;
        border-radius:10px;
        background:white;
        border:1px solid #eee;
        margin-bottom:8px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ---- Hero Section ----
    st.markdown("""
    <div class="hero">
        <h2>💳 EMIPredict AI</h2>
        <p>
        Intelligent Financial Risk Assessment Platform that predicts 
        EMI eligibility and maximum monthly EMI capacity using Machine Learning.
        </p>
        <b>FinTech ML Platform</b><br>
        📊 400K Records | 🤖 8 ML Models | 📈 MLflow Tracked | ☁️ Streamlit Cloud
    </div>
    """, unsafe_allow_html=True)

    # ---- Stats ----
    st.subheader("📊 Project Statistics")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Training Records", "400,000")
    c2.metric("Input Features", "22")
    c3.metric("Best Accuracy", "91.4%")
    c4.metric("Best RMSE", "1,842")

    st.divider()

    # ---- ML Pipeline ----
    left, right = st.columns(2)

    with left:
        st.subheader("🔄 ML Pipeline")

        pipeline_steps = [
            "1️⃣ Data Preprocessing – Clean and scale 400K records",
            "2️⃣ Feature Engineering – Create financial ratio features",
            "3️⃣ Classification – Predict EMI eligibility",
            "4️⃣ Regression – Predict maximum EMI amount",
            "5️⃣ MLflow Tracking – Track experiments",
            "6️⃣ Streamlit App – Interactive interface"
        ]

        for step in pipeline_steps:
            st.markdown(f"<div class='card'>{step}</div>", unsafe_allow_html=True)

    # ---- EMI Scenarios ----
    with right:
        st.subheader("🏷️ EMI Scenarios")

        scenarios = [
            ("🛒 E-commerce"),
            ("🏠 Home Appliances"),
            ("🚗 Vehicle"),
            ("💰 Personal Loan"),
            ("🎓 Education"),
        ]

        for name in scenarios:
            st.markdown(
                f"<div class='card'><b>{name}</b><br>",
                unsafe_allow_html=True
            )

    st.divider()

    # ---- Models ----
    st.subheader("🤖 Trained Models")

    mc1, mc2 = st.columns(2)

    with mc1:
        st.markdown("### 📊 Classification Models")

        clf_models = [
            ("Logistic Regression", "84%", "0.843"),
            ("Decision Tree", "86%", "0.861"),
            ("Random Forest", "90%", "0.899"),
            ("XGBoost", "91%", "0.913"),
        ]

        for name, acc, f1 in clf_models:
            st.markdown(
                f"<div class='card'><b>{name}</b><br>Accuracy: {acc} | F1: {f1}</div>",
                unsafe_allow_html=True
            )
    with mc2:
        st.markdown("### 📐 Regression Models")

        reg_models = [
            ("Linear Regression", "4200","0.81"),
            ("Decision Tree", "2800","0.88"),
            ("Random Forest", "2100","0.92"),
            ("XGBoost", "1842","0.94"),
        ]

        for name, rmse,r2 in reg_models:
            st.markdown(
                f"<div class='card'><b>{name}</b><br>RMSE: {rmse} | R²: {r2}</div>",
                unsafe_allow_html=True
            )