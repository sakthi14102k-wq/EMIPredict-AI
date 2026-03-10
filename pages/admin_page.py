import streamlit as st
from pathlib import Path
import pandas as pd
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

def show():
    st.subheader("🛠 Admin Panel – Loan Applications")

    ADMIN_DATA_PATH = BASE_DIR / "admin_applications.csv"


    # Load or Create Dataset

    if ADMIN_DATA_PATH.exists():
        admin_df = pd.read_csv(ADMIN_DATA_PATH)
    else:
        admin_df = pd.DataFrame(
            columns=[
                "application_id",
                "age",
                "monthly_salary",
                "credit_score",
                "requested_amount",
                "requested_tenure",
                "emi_eligibility",
                "status",
            ]
        )


    # VIEW APPLICATIONS

    st.markdown("### 📄 Existing Applications")

    if admin_df.empty:
        st.info("No applications available.")
    else:
        st.dataframe(admin_df, use_container_width=True)


    # CREATE APPLICATION

    st.markdown("### ➕ Add New Application")

    with st.form("add_application", clear_on_submit=True):

        c1, c2 = st.columns(2)

        with c1:
            app_id = st.text_input("Application ID")
            age = st.number_input("Age", 18, 80, 30)
            salary = st.number_input("Monthly Salary", 0, 500000, 60000)

        with c2:
            credit = st.number_input("Credit Score", 300, 900, 720)
            loan_amt = st.number_input("Requested Amount", 0, 5000000, 500000)
            tenure = st.number_input("Tenure (Months)", 1, 240, 24)

        submitted = st.form_submit_button("➕ Create Application")

        if submitted:

            if app_id.strip() == "":
                st.error("Application ID cannot be empty")

            else:
                new_row = {
                    "application_id": app_id,
                    "age": age,
                    "monthly_salary": salary,
                    "credit_score": credit,
                    "requested_amount": loan_amt,
                    "requested_tenure": tenure,
                    "emi_eligibility": "Pending",
                    "status": "New",
                }

                admin_df = pd.concat(
                    [admin_df, pd.DataFrame([new_row])],
                    ignore_index=True,
                )

                admin_df.to_csv(ADMIN_DATA_PATH, index=False)

                st.success("✅ Application Added Successfully")


    # UPDATE APPLICATION

    st.markdown("### ✏ Update Application")

    valid_admin_df = admin_df.dropna(subset=["application_id"])

    if valid_admin_df.empty:
        st.info("ℹ No valid applications available to update.")

    else:

        selected_id = st.selectbox(
            "Select Application ID",
            valid_admin_df["application_id"].astype(str),
        )

        record = valid_admin_df[
            valid_admin_df["application_id"].astype(str) == str(selected_id)
        ].iloc[0]

        col1, col2 = st.columns(2)

        with col1:
            new_salary = st.number_input(
                "Update Monthly Salary",
                0,
                500000,
                int(record["monthly_salary"]),
            )

        with col2:
            new_credit = st.number_input(
                "Update Credit Score",
                300,
                900,
                int(record["credit_score"]),
            )

        if st.button("💾 Update Application"):

            admin_df.loc[
                admin_df["application_id"].astype(str) == str(selected_id),
                ["monthly_salary", "credit_score"],
            ] = [new_salary, new_credit]

            admin_df.to_csv(ADMIN_DATA_PATH, index=False)

            st.success("✅ Application Updated Successfully")


    # DELETE APPLICATION

    st.markdown("### 🗑 Delete Application")

    if not admin_df.empty:

        delete_id = st.selectbox(
            "Select Application to Delete",
            admin_df["application_id"].astype(str),
            key="delete_id",
        )

        if st.button("❌ Delete Application"):

            admin_df = admin_df[
                admin_df["application_id"].astype(str) != str(delete_id)
            ]

            admin_df.to_csv(ADMIN_DATA_PATH, index=False)

            st.warning("⚠ Application Deleted")

    else:
        st.info("No applications available to delete.")


    # EXPORT DATA

    st.markdown("### ⬇ Export Data")

    st.download_button(
        label="Download Admin Data",
        data=admin_df.to_csv(index=False),
        file_name="admin_applications.csv",
        mime="text/csv",
    )