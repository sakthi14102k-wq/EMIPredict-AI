# ══════════════════════════════════════════════════════════════
#  EMIPredict AI — Model Training Pipeline
#  File: train_model.py
#  Fixes applied:
#    T-BUG-1 : Added average='weighted' to ALL 3 metric functions
#    T-BUG-2 : Fixed roc_auc_score for multiclass (removed [:,1])
#    T-BUG-3 : Changed eval_metric='logloss' → 'mlogloss'
#    T-BUG-4 : Added stratify= to train_test_split
#    T-WARN-1: Added MAPE to regression metrics
#    T-WARN-2: Added mlflow.register_model() for best models
#    T-WARN-3: Added class_weight='balanced' to DecisionTree
#    T-INFO-1: Added print_summary() comparison table
#    T-INFO-2: clf_results now stores all metrics, not just f1
# ══════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Regression Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

# Gradient Boosting
from xgboost import XGBClassifier, XGBRegressor

# MLflow
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


class ModelTrain:
    """
    Full ML training pipeline for EMIPredict AI.

    Trains 4 Classification models and 4 Regression models,
    logs everything to MLflow, selects and registers best models,
    then saves them as .pkl files.

    Usage:
        trainer = ModelTrain(
            feature_path='data/feature_dataset.csv',
            target_path='data/target_dataset.csv'
        )
        trainer.run()
    """

    def __init__(self, feature_path, target_path):

        print("  EMIPredict AI — Model Training Pipeline")

        # Load Data
        print("\n  Loading data...")
        self.X = pd.read_csv(feature_path)
        self.y = pd.read_csv(target_path)
        print(f"  Features shape : {self.X.shape[0]:,} × {self.X.shape[1]}")
        print(f"  Targets shape  : {self.y.shape}")

        # Output directory
        
        self.output_dir = "model"
        os.makedirs(self.output_dir, exist_ok=True)

        # TEST/TRAIN DATA

        print("\nSplitting data (80% train / 20% test)")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.2,
            random_state=42,
            stratify=self.y['emi_eligibility']   
        )
        print(f"  Train rows : {self.X_train.shape[0]:,}")
        print(f"  Test rows  : {self.X_test.shape[0]:,}")

        # MLflow setup 

        MLFLOW_DIR = r"C:\Users\SAKTHI\Desktop\myproject\EMI Predict AI\mlruns"
        mlflow.set_tracking_uri(f"file:///{MLFLOW_DIR}")
        mlflow.set_experiment("EMI_Prediction")
        print(f" MLflow saving to: {MLFLOW_DIR}")

        # Results storage
        self.clf_results      = {}   
        self.reg_results      = {}
        self.best_clf_model   = None
        self.best_reg_model   = None
        self.best_clf_run_id  = None
        self.best_reg_run_id  = None


    # CLASSIFICATION TRAINING


    def train_classification_models(self):
        """
        Trains 4 classifiers, evaluates with weighted multiclass metrics,
        logs all runs to MLflow, then selects the best by F1 score.
        """

        print("  🤖 Training Classification Models")
        print("  Target: emi_eligibility  (3 classes — MULTICLASS)")


        y_train = self.y_train['emi_eligibility']
        y_test  = self.y_test['emi_eligibility']

        print(f"\n  Class distribution in test set:")
        print(y_test.value_counts().to_string())

        models = {
            'LogisticRegression': LogisticRegression(
                max_iter=2000,
                class_weight='balanced',
                random_state=42
            ),
            'DecisionTreeClassifier': DecisionTreeClassifier(
                random_state=42,
                class_weight='balanced'
            ),
            'RandomForestClassifier': RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            ),
            'XGBoostClassifier': XGBClassifier(
                n_estimators=200,
                eval_metric='mlogloss',
                random_state=42
                
            ),
        }

        for name, model in models.items():

            print(f"\n  ▶ Training {name}...")

            with mlflow.start_run(run_name=f'CLF_{name}') as run:

                # Train
                model.fit(self.X_train, y_train)
                y_pred  = model.predict(self.X_test)

                #  AUC
                y_proba = model.predict_proba(self.X_test)

                # metrics
                acc  = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec  = recall_score(y_test, y_pred,    average='weighted', zero_division=0)
                f1   = f1_score(y_test, y_pred,        average='weighted', zero_division=0)

                # AUC
                auc = roc_auc_score(
                    y_test, y_proba,
                    multi_class='ovr',
                    average='weighted'
                )

                # Log params + metrics to MLflow
                mlflow.log_params(model.get_params())
                mlflow.log_metrics({
                    'accuracy' : acc,
                    'precision': prec,
                    'recall'   : rec,
                    'f1'       : f1,
                    'roc_auc'  : auc,
                })
                mlflow.sklearn.log_model(model, f'clf_{name}')

                # FIX T-INFO-2: store ALL metrics, not just f1
                self.clf_results[name] = {
                    'model'    : model,
                    'run_id'   : run.info.run_id,
                    'accuracy' : acc,
                    'precision': prec,
                    'recall'   : rec,
                    'f1'       : f1,
                    'roc_auc'  : auc,
                }

                print(f"     Acc={acc:.4f}  Prec={prec:.4f}  "
                      f"Rec={rec:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

        # Select best by F1

        best_name = max(self.clf_results, key=lambda k: self.clf_results[k]['f1'])
        best      = self.clf_results[best_name]

        self.best_clf_model  = best['model']
        self.best_clf_run_id = best['run_id']

        print(f"\n  🏆 Best Classifier : {best_name}")
        print(f"     F1={best['f1']:.4f}  AUC={best['roc_auc']:.4f}")

        # Save best model as .pkl
        clf_path = os.path.join(self.output_dir, 'best_classifier.pkl')
        joblib.dump(self.best_clf_model, clf_path)
        print(f"  💾 Saved → {clf_path}")

        # FIX T-WARN-2: Register best model in MLflow Model Registry
        try:
            mlflow.register_model(
                model_uri=f"runs:/{self.best_clf_run_id}/clf_{best_name}",
                name='EMI_BestClassifier'
            )
            print(f"  📋 Registered in MLflow Model Registry → 'EMI_BestClassifier'")
        except Exception as e:
            print(f"  ⚠️  MLflow registry skipped: {e}")

        return self

    
    # REGRESSION TRAINING
   
    def train_regression_models(self):
        """
        Trains 4 regressors, evaluates with RMSE/MAE/R²/MAPE,
        logs all runs to MLflow, then selects the best by RMSE.
        """

        print("   Training Regression Models")
        print("  Target: max_monthly_emi  (continuous, range ₹500–₹50,000)")

        y_train = self.y_train['max_monthly_emi']
        y_test  = self.y_test['max_monthly_emi']

        print(f"\n  Target stats:")
        print(f"  Mean={y_test.mean():.0f}  Std={y_test.std():.0f}  "
              f"Min={y_test.min():.0f}  Max={y_test.max():.0f}")

        models = {
            'LinearRegression': LinearRegression(),
            'DecisionTreeRegressor': DecisionTreeRegressor(random_state=42),
            'RandomForestRegressor': RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            ),
            'XGBoostRegressor': XGBRegressor(
                n_estimators=200,
                random_state=42
            ),
        }

        for name, model in models.items():

            print(f"\n  ▶ Training {name}...")

            with mlflow.start_run(run_name=f'REG_{name}') as run:

                # Train
                model.fit(self.X_train, y_train)
                y_pred = model.predict(self.X_test)

                # Metrics
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae  = mean_absolute_error(y_test, y_pred)
                r2   = r2_score(y_test, y_pred)

                
                mape = np.mean(
                    np.abs((y_test - y_pred) / (y_test + 1e-6))
                ) * 100

                # Log to MLflow
                mlflow.log_params(model.get_params())
                mlflow.log_metrics({
                    'rmse': rmse,
                    'mae' : mae,
                    'r2'  : r2,
                    'mape': mape,
                })
                mlflow.sklearn.log_model(model, f'reg_{name}')

                self.reg_results[name] = {
                    'model' : model,
                    'run_id': run.info.run_id,
                    'rmse'  : rmse,
                    'mae'   : mae,
                    'r2'    : r2,
                    'mape'  : mape,
                }

                print(f"     RMSE={rmse:.1f}  MAE={mae:.1f}  "
                      f"R²={r2:.4f}  MAPE={mape:.2f}%")

        # Select best by RMSE (lower is better)
        best_name = min(self.reg_results, key=lambda k: self.reg_results[k]['rmse'])
        best      = self.reg_results[best_name]

        self.best_reg_model  = best['model']
        self.best_reg_run_id = best['run_id']

        print(f"\n Best Regressor : {best_name}")
        print(f" RMSE={best['rmse']:.1f}  R²={best['r2']:.4f}  MAPE={best['mape']:.2f}%")

        # Save best model as .pkl
        reg_path = os.path.join(self.output_dir, 'best_regressor.pkl')
        joblib.dump(self.best_reg_model, reg_path)
        print(f"  💾 Saved → {reg_path}")

        # Register best model in MLflow Model Registry
        try:
            mlflow.register_model(
                model_uri=f"runs:/{self.best_reg_run_id}/reg_{best_name}",
                name='EMI_BestRegressor'
            )
            print(f"   Registered in MLflow Model Registry → 'EMI_BestRegressor'")
        except Exception as e:
            print(f"   MLflow registry skipped: {e}")

        return self


    # SUMMARY TABLE 


    def print_summary(self):
        """Prints a side-by-side comparison table of all trained models."""

        print("  RESULTS SUMMARY")

        # Classification table
        print("\n Classification Models  (emi_eligibility — 3 classes)")
        print(f"  {'Model':<28} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8}")

        for name, r in self.clf_results.items():
            marker = "best" if r['model'] is self.best_clf_model else ""
            print(
                f"  {name:<28} "
                f"{r['accuracy']:>9.4f} "
                f"{r['precision']:>10.4f} "
                f"{r['recall']:>8.4f} "
                f"{r['f1']:>8.4f} "
                f"{r['roc_auc']:>8.4f}"
                f"{marker}"
            )

        #Regression table 
        print(f"\nRegression Models  (max_monthly_emi — continuous)")
        print(f"  {'Model':<28} {'RMSE (₹)':>10} {'MAE (₹)':>9} {'R²':>8} {'MAPE %':>8}")

        for name, r in self.reg_results.items():
            marker = " best" if r['model'] is self.best_reg_model else ""
            print(
                f"  {name:<28} "
                f"{r['rmse']:>10.1f} "
                f"{r['mae']:>9.1f} "
                f"{r['r2']:>8.4f} "
                f"{r['mape']:>8.2f}"
                f"{marker}"
            )

        print("\n   = Best selected model\n")
        print("   Best models saved:")
        print(f"     model/best_classifier.pkl")
        print(f"     model/best_regressor.pkl")
        print("   MLflow Model Registry:")
        print(f"     EMI_BestClassifier")
        print(f"     EMI_BestRegressor")
        print("\n    Next step: build Streamlit app using these .pkl files")


    # RUN FULL PIPELINE

    def run(self):
        """Run complete pipeline: classify → regress → summarise."""
        self.train_classification_models()
        self.train_regression_models()
        self.print_summary()



#  MAIN


if __name__ == "__main__":

    trainer = ModelTrain(
        feature_path='data/feature_dataset.csv',
        target_path='data/target_dataset.csv'
    )

    trainer.run()