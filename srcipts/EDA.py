# ══════════════════════════════════════════════════════════════
#  Pipeline:
#    1.  Load Dataset
#    2.  Data Overview & Info
#    3.  Data Cleaning  (age, gender, numeric types)
#    4.  Missing Value Analysis & Imputation
#    5.  Outlier Detection  (Boxplots — Before)
#    6.  Outlier Clipping   (IQR method)
#    7.  Outlier Verification (Boxplots — After)
#    8.  Target Variable Analysis  (Classification + Regression)
#    9.  Correlation Analysis
#   10.  Feature vs Target Plots
#   11.  Save Cleaned Dataset
# ══════════════════════════════════════════════════════════════

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── Plot style ─────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.dpi"    : 120,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})

# Paths
INPUT_PATH  = r"C:\Users\SAKTHI\Desktop\myproject\EMI Predict AI\data\emi_prediction_dataset.csv"
OUTPUT_PATH = r"C:\Users\SAKTHI\Desktop\myproject\EMI Predict AI\data\emi_prediction_dataset_eda.csv"



#  STEP 1 — Load Dataset


print("=" * 60)
print("  EMIPredict AI — EDA Pipeline")
print("=" * 60)

print("\n[1] Loading dataset...")
df = pd.read_csv(INPUT_PATH)
print(f"    Shape  : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"    Columns: {list(df.columns)}")



#  STEP 2 — Data Overview


print("[2] Data Overview")
data_info = pd.DataFrame({
    'Data_column'            : df.columns,
    'Total_Rows'             : len(df),
    'Data_count'             : df.count().values,
    'Data_null'              : df.isnull().sum().values,
    'Data_missing_percentage': (df.isnull().sum() / len(df) * 100).values,
    'Data_type'              : df.dtypes.values,
    'sample_unique_values'   : [df[col].unique()[:5].tolist() for col in df.columns],
}).reset_index(drop=True)

print(data_info.to_string(index=False))

print(f"\n  Head (5 rows):")
print(df.head().to_string())

print(f"\n  Describe (numeric):")
print(df.describe().round(2).to_string())



#  STEP 3 — Data Cleaning

print("\n[3] Data Cleaning")
print("─" * 60)

# Age: extract numeric digits only 

df['age'] = df['age'].astype(str).str.extract(r'(\d+)')

# Gender: standardize all variants → Male / Female 

gender_map = {
    "female": "Female", "Female": "Female",
    "F"     : "Female", "FEMALE": "Female",
    "male"  : "Male",   "Male"  : "Male",
    "M"     : "Male",   "MALE"  : "Male",
}
df['gender'] = df['gender'].replace(gender_map)


# Convert numeric columns to proper dtype

numeric_cols = [
    'age', 'monthly_salary', 'years_of_employment', 'monthly_rent',
    'family_size', 'dependents', 'school_fees', 'college_fees',
    'travel_expenses', 'groceries_utilities', 'other_monthly_expenses',
    'current_emi_amount', 'credit_score', 'bank_balance',
    'emergency_fund', 'requested_amount', 'requested_tenure', 'max_monthly_emi'
]
for col in numeric_cols:
    before = df[col].dtype
    df[col] = pd.to_numeric(df[col], errors='coerce')




#  STEP 4 — Missing Value Analysis & Imputation

total_rows = df.shape[0]

# Print missing % for key columns
key_cols = ['monthly_rent', 'bank_balance', 'credit_score', 'education', 'emergency_fund']
for col in key_cols:
    pct = (df[col].isnull().sum() / total_rows) * 100

# Visualise missing values
missing = df.isnull().sum().sort_values(ascending=False)
missing = missing[missing > 0]

if len(missing) > 0:
    fig, ax = plt.subplots(figsize=(10, 4))
    missing.plot(kind='bar', ax=ax, color='#ef4444', edgecolor='white')
    ax.set_title("Missing Values per Column", fontsize=13, fontweight='bold')
    ax.set_xlabel("Column")
    ax.set_ylabel("Count")
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 10,
                f'{int(bar.get_height())}',
                ha='center', va='bottom', fontsize=9)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

#Impute: Numerical → median, Categorical → mode

print("\n  Imputing missing values...")

numerical_to_fill = ['monthly_rent', 'bank_balance', 'credit_score', 'emergency_fund', 'monthly_salary']
for col in numerical_to_fill:
    median_val = df[col].median()
    filled     = df[col].isnull().sum()
    df[col].fillna(median_val, inplace=True)

cat_to_fill = ['education']
for col in cat_to_fill:
    mode_val = df[col].mode().iloc[0]
    filled   = df[col].isnull().sum()
    df[col]  = df[col].fillna(mode_val)

print(f"\n  Remaining nulls after imputation: {df.isnull().sum().sum()}")



#  STEP 5 — Outlier Detection (Boxplots BEFORE Clipping)

numerical_all = [
    'age', 'monthly_salary', 'years_of_employment', 'monthly_rent',
    'family_size', 'dependents', 'school_fees', 'college_fees',
    'travel_expenses', 'groceries_utilities', 'other_monthly_expenses',
    'current_emi_amount', 'credit_score', 'bank_balance',
    'emergency_fund', 'requested_amount', 'requested_tenure', 'max_monthly_emi'
]

n_cols = 4
n_rows = math.ceil(len(numerical_all) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
fig.suptitle("Boxplots — BEFORE Outlier Clipping", fontsize=15, fontweight='bold', y=1.01)
axes = axes.flatten()

for i, col in enumerate(numerical_all):
    sns.boxplot(y=df[col], ax=axes[i], color='#f87171', linewidth=1.2)
    axes[i].set_title(col, fontsize=10)
    axes[i].set_ylabel("")

# Hide empty subplots
for j in range(len(numerical_all), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()



#  STEP 6 — Outlier Clipping (IQR method)


print("\n Outlier Clipping — IQR Method")

def clip_outliers(df: pd.DataFrame, col: str) -> pd.DataFrame:

    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower=lower, upper=upper)
    return df


# Only clip these two
cols_to_clip = ['credit_score', 'requested_tenure']
for col in cols_to_clip:
    df = clip_outliers(df, col)



#  STEP 7 — Outlier Verification (Boxplots AFTER Clipping)


print("\n Outlier Verification — Boxplots AFTER Clipping")

fig, axes = plt.subplots(1, len(cols_to_clip), figsize=(6 * len(cols_to_clip), 5))
if len(cols_to_clip) == 1:
    axes = [axes]

for ax, col in zip(axes, cols_to_clip):
    sns.boxplot(y=df[col], ax=ax, color='#4ade80', linewidth=1.2)
    ax.set_title(f"{col} (After Clipping)", fontsize=11, fontweight='bold')
    ax.set_ylabel("")

plt.suptitle("Boxplots — AFTER Outlier Clipping", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()


#  STEP 8 — Target Variable Analysis

print(" Target Variable Analysis")


fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Target Variables", fontsize=13, fontweight='bold')

# Classification target: emi_eligibility 

emi_counts = df["emi_eligibility"].value_counts()

print("\nemi_eligibility distribution:")
axes[0].bar(
    emi_counts.index.astype(str),
    emi_counts.values
)

axes[0].set_title("Class Distribution: EMI Eligibility", fontweight="bold")
axes[0].set_xlabel("Eligibility Status")
axes[0].set_ylabel("Number of Applicants")
axes[0].tick_params(axis="x", rotation=0)

# ── Regression target: max_monthly_emi 

print("\nmax_monthly_emi statistics:")

sns.histplot(
    df["max_monthly_emi"],
    kde=True,
    bins=40,
    ax=axes[1]
)

axes[1].set_title("Distribution: Max Monthly EMI", fontweight="bold")
axes[1].set_xlabel("Max Monthly EMI (₹)")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.show()



#  STEP 9 — Correlation Analysis


print("\n[9] Correlation Analysis")

# ── Full correlation heatmap 
num_df  = df.select_dtypes(include='number')
corr    = num_df.corr()

plt.figure(figsize=(18, 14))
sns.heatmap(
    corr, annot=True, fmt=".2f",
    cmap="RdBu_r", center=0
)
plt.title("Feature Correlation Heatmap", fontsize=14, fontweight='bold', pad=15)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.show()


#  STEP 10 — Feature vs Target Plots

print("\n[10] Feature vs Target Plots")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Key Feature Relationships", fontsize=13, fontweight='bold')

#  Salary vs Max Monthly EMI (scatter)
axes[0].scatter(
    df['monthly_salary'], df['max_monthly_emi']
)
axes[0].set_title("Monthly Salary vs Max Monthly EMI")
axes[0].set_xlabel("Monthly Salary (₹)")
axes[0].set_ylabel("Max Monthly EMI (₹)")

# Credit Score vs EMI Eligibility (boxplot)
df_sorted = df.copy()
df_sorted['emi_eligibility'] = df_sorted['emi_eligibility'].astype(str)
sns.boxplot(
    data=df_sorted,
    x='emi_eligibility',
    y='credit_score'
)
axes[1].set_title("Credit Score vs EMI Eligibility")
axes[1].set_xlabel("EMI Eligibility")
axes[1].set_ylabel("Credit Score")

plt.tight_layout()
plt.show()

# Pairplot for key variables
key_vars = ['monthly_salary', 'credit_score', 'bank_balance',
            'current_emi_amount', 'max_monthly_emi', 'emi_eligibility']

available = [c for c in key_vars if c in df.columns]
sample_df = df[available].sample(n=min(3000, len(df)), random_state=42)
sample_df['emi_eligibility'] = sample_df['emi_eligibility'].astype(str)

pair = sns.pairplot(
    sample_df,
    hue='emi_eligibility',
    diag_kind='kde',
    plot_kws={'alpha': 0.3, 's': 10}
)
pair.fig.suptitle("Pairplot — Key Features (3K sample)", y=1.02, fontsize=12, fontweight='bold')
plt.show()



#  STEP 11 — Save Cleaned Dataset


print("\n[11] Saving cleaned dataset...")
df.to_csv(OUTPUT_PATH, index=False)
print(f"     ✅ Saved to: {OUTPUT_PATH}")
