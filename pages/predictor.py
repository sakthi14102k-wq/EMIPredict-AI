import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Load models ────────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = os.path.dirname(os.path.dirname(__file__))
    model_dir = os.path.join(base, "model")
    try:
        preprocessor   = joblib.load(os.path.join(model_dir, "preprocessor.pkl"))
        clf             = joblib.load(os.path.join(model_dir, "best_classifier.pkl"))
        reg             = joblib.load(os.path.join(model_dir, "best_regressor.pkl"))
        target_encoders = joblib.load(os.path.join(model_dir, "target_encoders.pkl"))
        return preprocessor, clf, reg, target_encoders, None
    except Exception as e:
        return None, None, None, None, str(e)

def show():
    st.markdown("""
    <div style='margin-bottom:2rem;'>
        <h2 style='font-size:1.8rem; font-weight:700; color:#0f172a; margin:0;'>🔮 EMI Predictor</h2>
        <p style='color:#64748b; margin-top:6px;'>Fill in the applicant's financial details to get an instant EMI eligibility prediction.</p>
    </div>
    """, unsafe_allow_html=True)

    preprocessor, clf, reg, target_encoders, err = load_models()

    if err:
        st.error(f"⚠️ Could not load models: `{err}`")
        st.info("Make sure you have run `feature.py` and `train_model.py` first, then restart the app.")
        st.code("""
# Run these in order:
python feature.py
python train_model.py
streamlit run app.py
        """)
        return

    # ── Input Form ─────────────────────────────────────────────
    st.markdown("#### 👤 Personal Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        age         = st.number_input("Age", 21, 65, 30)
        gender      = st.selectbox("Gender", ["Male", "Female"])
        employment  = st.selectbox("Employment Type", ["Salaried", "Self-employed", "Business"])
        marital_status  = st.selectbox("Marital Status",  ["Single", "Married", "Divorced", "Widowed"])
        family_size     = st.number_input("Family Size",   1, 15, 3)
        education       = st.selectbox("Education",        ["Graduate", "Post Graduate", "Professional", "High School"])
    with col2:
        house_type      = st.selectbox("House Type",       ["Rented", "Own", "Family", "Mortgaged"])
        company_type    = st.selectbox("Company Type",     ["MNC", "Large Indian", "Mid-size", "Startup", "Small", "Government"])
        years_of_employment = st.number_input("Years of Employment", 0, 40, 5)
        existing_loans  = st.selectbox("Existing Loans",   ["No", "Yes"])
        emi_scenario    = st.selectbox("EMI Scenario",     ["E-commerce", "Home Appliances", "Vehicle", "Personal Loan", "Education"])
        dependents      = st.number_input("Dependents", 0, 6, 1)
    
    with col3:
        monthly_salary  = st.number_input("Monthly Salary (₹)", 10000, 500000, 50000, step=1000)
        bank_balance    = st.number_input("Bank Balance (₹)", 0, 5000000, 100000, step=5000)
        credit_score    = st.number_input("Credit Score", 300, 850, 700)
        emergency_fund  = st.number_input("Emergency Fund (₹)", 0, 1000000, 50000, step=5000)
        current_emi     = st.number_input("Current EMI Amount (₹)", 0, 100000, 5000, step=500)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 💸 Expense Details")
    ex1, ex2, ex3 = st.columns(3)
    with ex1:
        monthly_rent  = st.number_input("Monthly Rent (₹)",    0, 100000, 8000, step=500)
        groceries     = st.number_input("Groceries & Utilities (₹)", 0, 50000, 5000, step=500)
    with ex2:
        school_fees   = st.number_input("School Fees (₹)",     0, 50000, 2000, step=500)
        college_fees  = st.number_input("College Fees (₹)",    0, 100000, 0, step=500)
    with ex3:
        travel_exp    = st.number_input("Travel Expenses (₹)", 0, 50000, 3000, step=500)
        other_exp     = st.number_input("Other Expenses (₹)",  0, 50000, 2000, step=500)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🏦 Loan Request")
    l1, l2, l3 = st.columns(3)
    with l1:
        loan_category   = st.selectbox("Loan Category", ["E-commerce", "Home Appliances", "Vehicle", "Personal Loan", "Education"])
    with l2:
        requested_amount = st.number_input("Requested Amount (₹)", 10000, 1500000, 200000, step=5000)
    with l3:
        requested_tenure = st.number_input("Requested Tenure (months)", 3, 84, 24)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Predict Button ─────────────────────────────────────────
    predict_col, _ = st.columns([1, 3])
    with predict_col:
        predict_btn = st.button("⚡ Predict Now", use_container_width=True)

    if predict_btn:
        # Build raw input dict (matching original dataset columns)
        input_data = {
            'age':                     age,
            'gender':                  gender,
            'employment_type':         employment,
            'dependents':              dependents,
            'monthly_salary':          monthly_salary,
            'bank_balance':            bank_balance,
            'credit_score':            credit_score,
            'emergency_fund':          emergency_fund,
            'current_emi_amount':      current_emi,
            'monthly_rent':            monthly_rent,
            'groceries_utilities':     groceries,
            'school_fees':             school_fees,
            'college_fees':            college_fees,
            'travel_expenses':         travel_exp,
            'other_monthly_expenses':  other_exp,
            'loan_category':           loan_category,
            'requested_amount':        requested_amount,
            'requested_tenure':        requested_tenure,
            'marital_status':         marital_status,
            'family_size':            family_size,
            'education':              education,
            'house_type':             house_type,
            'company_type':           company_type,
            'years_of_employment':    years_of_employment,
            'existing_loans':         existing_loans,
            'emi_scenario':           emi_scenario,
        }
        df_input = pd.DataFrame([input_data])

        # ── Apply same feature engineering ─────────────────────
        df_input['total_expenses'] = (
            df_input['college_fees'] + df_input['monthly_rent'] +
            df_input['school_fees']  + df_input['travel_expenses'] +
            df_input['groceries_utilities'] + df_input['other_monthly_expenses']
        )
        df_input['disposable_income']   = df_input['monthly_salary'] - df_input['total_expenses']
        df_input['debt_to_income']      = df_input['current_emi_amount'] / (df_input['monthly_salary'] + 1e-6)
        df_input['expense_ratio']       = df_input['total_expenses'] / (df_input['monthly_salary'] + 1e-6)
        df_input['estimated_new_emi']   = df_input['requested_amount'] / (df_input['requested_tenure'] + 1e-6)
        df_input['emi_to_income_ratio'] = (df_input['current_emi_amount'] + df_input['estimated_new_emi']) / (df_input['monthly_salary'] + 1e-6)
        df_input['savings_rate']        = df_input['bank_balance'] / (df_input['monthly_salary'] + 1e-6)
        df_input['emergency_coverage']  = df_input['emergency_fund'] / (df_input['total_expenses'] + 1)
        df_input['salary_per_dependent']= df_input['monthly_salary'] / (df_input['dependents'] + 1)

        cs = df_input['credit_score'].values[0]
        df_input['credit_risk_band'] = 3 if cs < 550 else (2 if cs < 650 else (1 if cs < 750 else 0))

        # ── Transform using saved pipeline ─────────────────────
        pipeline      = preprocessor['pipeline']
        feature_names = preprocessor['feature_names']

        try:
            X_transformed = pipeline.transform(df_input)
            X_df = pd.DataFrame(X_transformed, columns=feature_names)

            # Predict
            clf_pred    = clf.predict(X_df)[0]
            clf_proba   = clf.predict_proba(X_df)[0]
            reg_pred    = reg.predict(X_df)[0]

            # Decode label
            if 'emi_eligibility' in target_encoders:
                le = target_encoders['emi_eligibility']
                eligibility_label = le.inverse_transform([clf_pred])[0]
            else:
                label_map = {0: 'Eligible', 1: 'High_Risk', 2: 'Not_Eligible'}
                eligibility_label = label_map.get(clf_pred, str(clf_pred))

            # ── Results Display ─────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("### 📋 Prediction Results")

            r1, r2, r3 = st.columns(3)

            # Badge color
            if 'Eligible' in eligibility_label and 'Not' not in eligibility_label:
                badge_color = "#16a34a"; bg_color = "#dcfce7"; icon = "✅"
            elif 'High' in eligibility_label:
                badge_color = "#ca8a04"; bg_color = "#fef9c3"; icon = "⚠️"
            else:
                badge_color = "#dc2626"; bg_color = "#fee2e2"; icon = "❌"

            with r1:
                st.markdown(f"""
                <div class='emi-card' style='text-align:center; border-top:4px solid {badge_color};'>
                    <div style='font-size:2.5rem;'>{icon}</div>
                    <div style='font-size:0.8rem; color:#64748b; margin:8px 0 4px;'>EMI ELIGIBILITY</div>
                    <div style='font-size:1.2rem; font-weight:700; color:{badge_color};
                                background:{bg_color}; padding:8px 16px; border-radius:20px;
                                display:inline-block;'>{eligibility_label}</div>
                </div>
                """, unsafe_allow_html=True)

            with r2:
                st.markdown(f"""
                <div class='emi-card' style='text-align:center; border-top:4px solid #3b82f6;'>
                    <div style='font-size:2.5rem;'>💰</div>
                    <div style='font-size:0.8rem; color:#64748b; margin:8px 0 4px;'>MAX MONTHLY EMI</div>
                    <div style='font-size:1.6rem; font-weight:700; color:#3b82f6;
                                font-family:"JetBrains Mono",monospace;'>
                        ₹{reg_pred:,.0f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with r3:
                conf = max(clf_proba) * 100
                st.markdown(f"""
                <div class='emi-card' style='text-align:center; border-top:4px solid #8b5cf6;'>
                    <div style='font-size:2.5rem;'>🎯</div>
                    <div style='font-size:0.8rem; color:#64748b; margin:8px 0 4px;'>CONFIDENCE</div>
                    <div style='font-size:1.6rem; font-weight:700; color:#8b5cf6;
                                font-family:"JetBrains Mono",monospace;'>
                        {conf:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Key financial metrics ───────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📊 Financial Summary")
            fm1, fm2, fm3, fm4 = st.columns(4)

            total_exp       = input_data['monthly_rent'] + input_data['groceries_utilities'] + \
                              input_data['school_fees']  + input_data['college_fees'] + \
                              input_data['travel_expenses'] + input_data['other_monthly_expenses']
            disposable      = monthly_salary - total_exp
            dti             = (current_emi / (monthly_salary + 1e-6)) * 100
            new_emi_est     = requested_amount / (requested_tenure + 1e-6)

            fm1.metric("Disposable Income", f"₹{disposable:,.0f}")
            fm2.metric("Debt-to-Income",    f"{dti:.1f}%",
                       delta="Good" if dti < 40 else "High", delta_color="normal" if dti < 40 else "inverse")
            fm3.metric("Est. New EMI",      f"₹{new_emi_est:,.0f}")
            fm4.metric("Emergency Coverage", f"{(emergency_fund/(total_exp+1)):.1f} months")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info("Make sure your input columns match what was used during training.")
