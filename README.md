# EMIPredict-AI

# EMIPredict AI - Intelligent Financial Risk Assessment Platform


An end-to-end FinTech machine learning platform that dynamically automates financial risk assessment. Built with a unified **Dual-Engine ML architecture**, the system processes a massive dataset of 400,000 credit records to evaluate risk using concurrent **multi-class classification** (EMI Eligibility) and **continuous regression** (Maximum Safe EMI Capacity) pipelines[cite: 1, 4, 8].

---

## 🎯 Domain & Problem Statement

### The FinTech Challenge
Modern retail credit systems often struggle to protect consumers and financial institutions from systemic defaults due to inadequate forward-looking risk modeling and poor personal financial planning. Traditional static credit scoring fails to look deeper into an individual's real-time disposable margins across different loan scenarios.

### The EMIPredict AI Solution
This project bridges the gap by delivering a secure, highly scalable, data-driven platform that provides deep insight into borrowing capacity across five distinct lending scenarios: **E-commerce, Home Appliances, Vehicle, Personal, and Education loans**

#### **Core Platform Deliverables:**
* **Dual ML Engines:** Real-time generation of discrete eligibility decisions along with explicit, continuous debt limits.
* **Industrial-Scale Processing:** Engineering complex financial insights over a massive pool of **400,000 multi-featured profiles**.
* **Rigorous MLOps Integration:** Experiment management, hyperparameter tracking, and structured versioning with MLflow
* **Production Cloud App:** A multi-page interactive dashboard with self-contained preprocessing layers and built-in CRUD data management

---

## 🏗️ Data Flow and Architecture

The platform separates data concerns through a strict five-tier decoupling logic to guarantee zero data leakage between model training pipelines and downstream operational instances[cite: 5, 7].

```text
Dataset (400K Records) 
        ↓
Data Quality Assessment & Preprocessing (EDA.py)
        ↓  
Feature Engineering & Transformation Pipeline (Feature.py)
        ↓
Dual ML Model Training & Parallel Experimentation (train_models.py)
        ↓
MLflow Tracking Server & Versioned Model Registry (mlflow UI)
        ↓
Multi-Page Production UI Layer (app.py, home.py, predictor.py, eda.py, admin_page.py)
        ↓
Streamlit Cloud Deployment with Automated CI/CD
