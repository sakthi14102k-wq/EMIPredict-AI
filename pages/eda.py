import sys
from pathlib import Path

# PATH SETUP
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import streamlit as st
import pandas as pd



# DATA LOADER

@st.cache_data
def load_data():
    try:
        data_path = BASE_DIR / "data" / "emi_prediction_dataset_eda.csv"
        df = pd.read_csv(data_path)
        return df, None
    except Exception as e:
        return None, str(e)



# PAGE UI

def show():

    st.markdown(
        """
        <div style='margin-bottom:2rem;'>
            <h2 style='font-size:1.8rem; font-weight:700; color:#0f172a; margin:0;'>
                📊 EDA Dashboard
            </h2>
            <p style='color:#64748b; margin-top:6px;'>
                Explore the processed dataset — distributions, class balance,
                and feature correlations.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    df, err = load_data()

    if err:
        st.warning(f"⚠️ Dataset could not be loaded: `{err}`")
        st.info("Make sure the file exists: data/emi_prediction_dataset_eda.csv")
        return


    
    # DATASET OVERVIEW
   
    st.markdown("### 🗂 Dataset Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Rows", f"{len(df):,}")
    c2.metric("Total Columns", f"{len(df.columns)}")
    c3.metric("Missing Values", f"{df.isnull().sum().sum()}")

    st.markdown("<br>", unsafe_allow_html=True)


    
    # TABS
   
    tab1, tab2 = st.tabs(["📈 Target Distribution", "🔢 Data Sample"])


    
    # TARGET DISTRIBUTION

    with tab1:

        if "emi_eligibility" in df.columns:

            st.markdown("#### EMI Eligibility Class Distribution")

            counts = df["emi_eligibility"].value_counts()

            st.bar_chart(counts)

        else:
            st.warning("Column 'emi_eligibility' not found in dataset")

        # EMI statistics
        if "max_monthly_emi" in df.columns:

            st.markdown("#### Max Monthly EMI Distribution")

            emi_col = df["max_monthly_emi"]

            e1, e2, e3, e4 = st.columns(4)

            e1.metric("Mean", f"{emi_col.mean():,.0f}")
            e2.metric("Median", f"{emi_col.median():,.0f}")
            e3.metric("Min", f"{emi_col.min():,.0f}")
            e4.metric("Max", f"{emi_col.max():,.0f}")



    # DATA SAMPLE

    with tab2:

        st.markdown("#### Sample Data")

        st.dataframe(
            df.head(100),
            use_container_width=True
        )

        st.caption(
            f"Showing 100 of {len(df):,} rows · {len(df.columns)} columns"
        )