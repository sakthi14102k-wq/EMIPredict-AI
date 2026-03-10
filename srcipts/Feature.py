# Data handling
import pandas as pd
import numpy as np
import os
import joblib

# Preprocessing tools
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class FeatureEngineer:

    def __init__(self, df: pd.DataFrame):
        print("=" * 55)
        print("  FeatureEngineer Initialized")
        print("=" * 55)
        print(f"  Input shape : {df.shape[0]:,} rows × {df.shape[1]} columns")

        self.df            = df.copy()
        self.pipeline      = None
        self.feature_names_ = None
        self.target_encoders = {}

    # 
    # STEP 1 — Financial Feature Creation
    # 

    def create_financial_ratios(self):

        print("\n  Creating financial features...")

        #  Aggregate all expense columns 
        self.df['total_expenses'] = (
            self.df['college_fees'] +
            self.df['monthly_rent'] +
            self.df['school_fees'] +
            self.df['travel_expenses'] +
            self.df['groceries_utilities'] +
            self.df['other_monthly_expenses']
        )

        #  Disposable income 
        # Money left after all costs — most predictive for max EMI
        self.df['disposable_income'] = (
            self.df['monthly_salary'] - self.df['total_expenses']
        )

        #  Debt-to-income ratio 
        # Industry standard metric. DTI > 50% = very risky
        
        self.df['debt_to_income'] = (
            self.df['current_emi_amount'] / (self.df['monthly_salary'] + 1e-6)
        )

        #  Expense ratio 
        # % of income spent. > 80% = almost no EMI capacity
        
        self.df['expense_ratio'] = (
            self.df['total_expenses'] / (self.df['monthly_salary'] + 1e-6)
        )

        #  Estimated new EMI 
        # Converts loan request into monthly cost estimate
        
        self.df['estimated_new_emi'] = (
            self.df['requested_amount'] / (self.df['requested_tenure'] + 1e-6)
        )

        #  EMI-to-income ratio 
        # Forward-looking total EMI burden after new loan
        
        self.df['emi_to_income_ratio'] = (
            (self.df['current_emi_amount'] + self.df['estimated_new_emi'])
            / (self.df['monthly_salary'] + 1e-6)
        )

        #  Savings rate 
        # Months of salary saved — financial buffer indicator

        self.df['savings_rate'] = (
            self.df['bank_balance'] / (self.df['monthly_salary'] + 1e-6)
        )

        #  Emergency coverage 
        # Months of expenses covered if income stops

        self.df['emergency_coverage'] = (
            self.df['emergency_fund'] / (self.df['total_expenses'] + 1)
        )

        #  Salary per dependent
        # Income normalised by financial responsibility

        self.df['salary_per_dependent'] = (
            self.df['monthly_salary'] / (self.df['dependents'] + 1)
        )

        #  Credit risk band 

        conditions = [
            self.df['credit_score'] < 550,   # → 3 (Poor)
            self.df['credit_score'] < 650,   # → 2 (Fair)
            self.df['credit_score'] < 750,   # → 1 (Good)
        ]
        choices = [3, 2, 1]
        self.df['credit_risk_band'] = np.select(conditions, choices, default=0)

        # Validate new features 
        new_cols = [
            'total_expenses', 'disposable_income', 'debt_to_income',
            'expense_ratio', 'estimated_new_emi', 'emi_to_income_ratio',
            'savings_rate', 'emergency_coverage', 'salary_per_dependent',
            'credit_risk_band'
        ]

        nan_count = self.df[new_cols].isnull().sum().sum()
        inf_count = np.isinf(
            self.df[new_cols].select_dtypes(include='number')
        ).sum().sum()

        if nan_count > 0 or inf_count > 0:
            self.df[new_cols] = self.df[new_cols].replace([np.inf, -np.inf], np.nan)
            self.df[new_cols] = self.df[new_cols].fillna(self.df[new_cols].median())
            print(" NaN values fixed")

        return self


    # STEP 2 — Encoding

    def encode_categories(self, target=None):
        print("\n  Encoding categorical numeric features")

        if target is None:
            target = []

        # Separate features and targets
        X = self.df.drop(columns=target, errors='ignore')
        y = self.df[target].copy() if target else None

        # Detect feature types
        cat_features = X.select_dtypes(include=['object']).columns.tolist()
        num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

        print(f"  Categorical features : {len(cat_features)} → {cat_features}")
        print(f"  Numeric features     : {len(num_features)}")

        # Build preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features),
                ('num', StandardScaler(), num_features),
            ],
            remainder='drop'
        )

        self.pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
        transformed   = self.pipeline.fit_transform(X)

        # Recover feature names after OneHot expansion
        if len(cat_features) > 0:
            cat_names = (
                self.pipeline.named_steps['preprocessor']
                .named_transformers_['cat']
                .get_feature_names_out(cat_features)
            )
        else:
            cat_names = []

        self.feature_names_ = list(cat_names) + num_features

        transformed_df = pd.DataFrame(
            transformed,
            columns=self.feature_names_,
            index=X.index
        )

        # Encode target columns
        if y is not None:
            self.target_encoders = {}

            for col in y.columns:
                if y[col].dtype == 'object':

                    # fill NaN before LabelEncoder
                    if y[col].isnull().sum() > 0:
                        y[col] = y[col].fillna(y[col].mode()[0])

                    le = LabelEncoder()
                    y[col] = le.fit_transform(y[col])
                    self.target_encoders[col] = le

        print(f"Encoding complete")

        return transformed_df, y

    # STEP 3 — Run Full Pipeline

    def run_full_pipeline(self, target_cols=None):

        self.create_financial_ratios()

        return self.encode_categories(target_cols)


if __name__ == "__main__":

    input_path = r"C:\Users\SAKTHI\Desktop\myproject\EMI Predict AI\data\emi_prediction_dataset_eda.csv"
    output_dir = r"C:\Users\SAKTHI\Desktop\myproject\EMI Predict AI\model"
    data_dir   = r"C:\Users\SAKTHI\Desktop\myproject\EMI Predict AI\data"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_dir,   exist_ok=True)

    # Load raw dataset
    print(f"\n  Loading dataset from:\n  {input_path}")
    df = pd.read_csv(input_path)

    # Run pipeline
    TARGET_COLS = ['emi_eligibility', 'max_monthly_emi']
    fe = FeatureEngineer(df)
    X, y = fe.run_full_pipeline(target_cols=TARGET_COLS)

    # Save pipeline + feature names
    preprocessor_artifact = {
        'pipeline'     : fe.pipeline,
        'feature_names': fe.feature_names_
    }
    joblib.dump(
        preprocessor_artifact,
        os.path.join(output_dir, 'preprocessor.pkl')
    )
    print("Saved-preprocessor.pkl")

    # Save target_encoders 
    joblib.dump(
        fe.target_encoders,
        os.path.join(output_dir, 'target_encoders.pkl')
    )
    print("Saved-target_encoders.pkl")

    #  Save processed datasets 
    feature_out = os.path.join(data_dir, 'feature_dataset.csv')
    target_out  = os.path.join(data_dir, 'target_dataset.csv')

    X.to_csv(feature_out, index=False)
    y.to_csv(target_out,  index=False)

    print(f"   Saved: feature_dataset.csv  → {X.shape}")
    print(f"  Saved: target_dataset.csv   → {y.shape}")
    print("  Feature engineering complete!")
