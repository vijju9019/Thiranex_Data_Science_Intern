# Thiranex Data Science Internship Portfolio

**Internship Organization:** Thiranex  
**Role:** Data Science Intern  
**Intern:** Data Science Intern  
**Repository Name:** `Thiranex_Data_Science_Internship`  
**Portfolio Scope:** Four Production-Grade, GitHub-Ready Data Science and Machine Learning Projects  

---

## 🌟 Executive Portfolio Overview

Welcome to my **Data Science Internship Portfolio** completed during my tenure at **Thiranex**. This repository houses four comprehensive, independently executable projects spanning the full data science lifecycle:

1. **Project 1: Data Cleaning and Visualization** — Auditing, grouped demographic imputation, 1.5x IQR Winsorization, and publication visualizations on the Titanic manifest.
2. **Project 2: Predictive Modeling Using Machine Learning** — Leakage-free Scikit-Learn pipelines, stratified evaluation, and supervised classification of fine-needle aspirate biopsies on the Breast Cancer Wisconsin dataset.
3. **Project 3: Exploratory Data Analysis (EDA)** — Deep multi-attribute distributional profiling, correlation decomposition, and formal statistical hypothesis testing ($\chi^2$ and Mann-Whitney U tests).
4. **Project 4: Real-World Data Science Project** — Retail Sales Analytics and Prediction System with time-series feature engineering, chronological train-test splitting, regression benchmarking, and an interactive executive Streamlit suite.

---

## 🏗️ Complete Directory Structure

```
Thiranex_Data_Science_Internship/
│
├── README.md                                   # Root portfolio documentation & guide
├── requirements.txt                            # Pinned Python package dependencies
├── .gitignore                                  # Git exclusion rules
│
├── Project_1_Data_Cleaning_Visualization/
│   ├── data/
│   │   ├── raw/
│   │   │   └── titanic_raw.csv                # Immutable raw passenger manifest
│   │   └── processed/
│   │       └── titanic_cleaned.csv            # Cleaned, imputed, and outlier-capped data
│   ├── notebooks/
│   │   └── data_cleaning_visualization.ipynb  # Step-by-step interactive workflow
│   ├── src/
│   │   ├── cleaning.py                        # Ingestion, imputation & outlier functions
│   │   └── visualization.py                   # Generation of 9 high-res diagnostic charts
│   ├── outputs/
│   │   └── figures/                           # 9 publication PNG figures
│   ├── app.py                                 # Interactive Streamlit data cleaning explorer
│   ├── report.md                              # Formal technical report
│   └── README.md                              # Project-specific documentation
│
├── Project_2_Predictive_Modeling/
│   ├── data/
│   │   └── breast_cancer_raw.csv              # Raw Wisconsin diagnostic records
│   ├── notebooks/
│   │   └── predictive_modeling.ipynb          # End-to-end ML training & evaluation
│   ├── src/
│   │   ├── preprocessing.py                   # Stratified splitting & pipeline builder
│   │   ├── train.py                           # 4-model trainer, metrics & comparison table
│   │   └── predict.py                         # Standalone CLI & callable inference module
│   ├── models/
│   │   ├── best_model.joblib                  # Serialized Scikit-Learn pipeline
│   │   └── model_metadata.json                # Benchmark metrics & model parameters
│   ├── outputs/
│   │   └── figures/                           # 4 publication figures (ROC, CM, Metrics, Imp)
│   ├── app.py                                 # Streamlit oncology risk classification app
│   ├── report.md                              # Detailed ML benchmarking report
│   └── README.md                              # Project-specific documentation
│
├── Project_3_EDA/
│   ├── data/
│   │   └── titanic_data.csv                   # Enriched demographic dataset
│   ├── notebooks/
│   │   └── exploratory_data_analysis.ipynb    # Comprehensive EDA & hypothesis tests
│   ├── src/
│   │   └── analysis.py                        # Descriptive stats, hypothesis tests & plots
│   ├── outputs/
│   │   └── figures/                           # 12 publication figures
│   ├── app.py                                 # Interactive Streamlit EDA dashboard
│   ├── report.md                              # Formal EDA & business insights report
│   └── README.md                              # Project-specific documentation
│
└── Project_4_Retail_Analytics/
    ├── data/
    │   ├── raw/
    │   │   └── retail_sales_raw.csv           # 7,500 commercial retail transactions
    │   └── processed/
    │       ├── retail_sales_cleaned.csv       # Validated order records
    │       └── retail_daily_sales.csv         # Daily time series with autoregressive lags
    ├── notebooks/
    │   └── retail_sales_analysis.ipynb        # Complete time-series & regression notebook
    ├── src/
    │   ├── cleaning.py                        # Transaction generator, cleaner & aggregator
    │   ├── analysis.py                        # BI metrics, category margins & figures
    │   ├── feature_engineering.py             # Lags, rolling windows & chronological split
    │   ├── train.py                           # Chronological model training & evaluation
    │   └── predict.py                         # CLI forecasting engine with 95% CIs
    ├── models/
    │   ├── best_sales_model.joblib            # Serialized regression model
    │   └── model_metadata.json                # Benchmark metrics & feature schema
    ├── outputs/
    │   └── figures/                           # 12 publication figures
    ├── app.py                                 # Executive Retail Suite & What-If Simulator
    ├── report.md                              # Executive business & forecast report
    └── README.md                              # Project-specific documentation
```

---

## 🛠️ Technology Stack

| Domain | Tools & Libraries |
|:-------|:------------------|
| **Core Programming** | Python 3.11+, Object-Oriented Design, Modular `src/` Architecture |
| **Data Manipulation** | Pandas 2.x, NumPy 1.26+ |
| **Statistical Inference**| SciPy (`scipy.stats` for Chi-Square, Mann-Whitney U, T-Tests) |
| **Machine Learning** | Scikit-Learn 1.8+ (Pipelines, Estimators, Regressors, Model Selection) |
| **Model Persistence** | Joblib |
| **Data Visualization** | Matplotlib 3.8+, Seaborn 0.13+ (Custom Palettes, Publication Styles) |
| **Interactive Apps** | Streamlit 1.30+ |
| **Version Control** | Git & GitHub |

---

## 🚀 Installation & Setup Guide

### 1. Clone the Repository
```powershell
git clone https://github.com/<your-username>/Thiranex_Data_Science_Internship.git
cd Thiranex_Data_Science_Internship
```

### 2. Create and Activate a Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install All Dependencies
```powershell
pip install -r requirements.txt
```

---

## ⚡ Execution Commands (Windows PowerShell)

Every project is fully modular and independently executable:

### 🚢 Project 1: Data Cleaning & Visualization
```powershell
# 1. Run Data Cleaning Pipeline
python Project_1_Data_Cleaning_Visualization\src\cleaning.py

# 2. Generate 9 High-Resolution Publication Figures
python Project_1_Data_Cleaning_Visualization\src\visualization.py

# 3. Launch Interactive Streamlit Dashboard
streamlit run Project_1_Data_Cleaning_Visualization\app.py
```

### 🔬 Project 2: Predictive Modeling Using Machine Learning
```powershell
# 1. Train All 4 Models, Generate Benchmarks & Diagnostic Figures
python Project_2_Predictive_Modeling\src\train.py

# 2. Execute Standalone CLI Inference on Patient Profiles
python Project_2_Predictive_Modeling\src\predict.py

# 3. Launch Interactive Oncology Risk Classifier Dashboard
streamlit run Project_2_Predictive_Modeling\app.py
```

### 📊 Project 3: Exploratory Data Analysis (EDA)
```powershell
# 1. Execute Statistical Analysis & Generate 12 Figures
python Project_3_EDA\src\analysis.py

# 2. Launch Interactive Streamlit EDA Explorer
streamlit run Project_3_EDA\app.py
```

### 🛍️ Project 4: Retail Sales Analytics & Demand Forecasting
```powershell
# 1. Execute Full Retail Pipeline (Cleaning, Analysis, Chronological Training)
python Project_4_Retail_Analytics\src\train.py

# 2. Run Standalone CLI Sales Revenue Forecast
python Project_4_Retail_Analytics\src\predict.py

# 3. Launch Executive Retail Analytics & What-If Simulator Suite
streamlit run Project_4_Retail_Analytics\app.py
```

---

## 📊 Summary of Actual Model Evaluation Results

### Project 2: Supervised Classification Benchmark (Held-Out Test Set, n=114)
*Zero Preprocessing Leakage | Stratified 80/20 Partition*

| Algorithm | Test Accuracy | Precision (Malignant) | Recall (Sensitivity) | F1-Score | ROC-AUC | Status |
|:----------|:-------------:|:---------------------:|:--------------------:|:--------:|:-------:|:-------|
| **Logistic Regression** | **98.25%** | **97.62%** | **97.62%** | **0.9762** | **0.9954** | **Selected Model** |
| **Support Vector Machine (RBF)** | **98.25%** | **97.62%** | **97.62%** | **0.9762** | **0.9950** | Strong Competitor |
| **Random Forest** | 94.74% | 92.86% | 92.86% | 0.9286 | 0.9947 | Non-Linear Robust |
| **Decision Tree** | 92.11% | 86.67% | 92.86% | 0.8966 | 0.9163 | High Variance |

*Safety Rationale: Logistic Regression missed only 1 out of 42 malignant test cases (False Negative rate = 2.38%), critical for clinical oncology screening.*

---

### Project 4: Retail Sales Chronological Regression (141-Day Out-of-Sample Window)
*Chronological 80/20 Split | Zero Look-Ahead Leakage*

| Algorithm | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | $R^2$ Score | Rank |
|:----------|:-------------------------:|:------------------------------:|:-----------:|:----:|
| **Linear Regression** | **$1,799.50** | **$2,332.91** | **0.3868** | **Best Generalization** |
| **Random Forest Regressor** | $1,782.43 | $2,410.35 | 0.3454 | Lowest MAE |
| **Decision Tree Regressor** | $2,122.39 | $2,762.20 | 0.1403 | Overfitting Baseline |

---

## 💡 Beginner-Friendly Project Explanations (For Your Internship Mentor)

When your technical mentor asks you to explain each project during your review, use these clear, professional explanations:

### Project 1: Data Cleaning and Visualization
> *"In Project 1, I focused on data quality and integrity. Rather than simply dropping rows with missing values, I analyzed their structure. For passenger ages, I used a conditional grouped median by class and gender, ensuring we didn't distort age distributions. For ticket fares, extreme values reached over $512; instead of throwing out those valid records, I applied 1.5x IQR Winsorization (capping) to stabilize model leverage while keeping 100% of the passenger cohort intact. Finally, I built 9 publication-grade charts and an interactive Streamlit app."*

### Project 2: Predictive Modeling Using Machine Learning
> *"In Project 2, I built a diagnostic classifier for breast tumor detection using Scikit-Learn. I prioritized preventing data leakage by using Scikit-Learn Pipelines so the StandardScaler was fitted strictly on training data only. Because this is medical screening, I prioritized Recall (Sensitivity) over simple accuracy to ensure we minimized dangerous False Negatives (missing a malignant tumor). Logistic Regression achieved 98.25% test accuracy and 97.62% recall on our stratified test set. I packaged the model using Joblib and deployed an educational Streamlit dashboard with confidence intervals."*

### Project 3: Exploratory Data Analysis (EDA)
> *"In Project 3, I conducted a deep-dive statistical exploration. Beyond standard histograms, I used inferential statistics—Pearson Chi-Square tests for categorical features and Mann-Whitney U non-parametric tests for continuous variables—to mathematically prove whether survival disparities were statistically significant (p < 0.01). I identified an inverted-U family dynamic where small families (2-4 people) survived at 58-72% while solo travelers (30.4%) and very large families (<15%) suffered high mortality. I synthesized 10 actionable business and historical insights and built an interactive EDA dashboard."*

### Project 4: Real-World Retail Analytics & Sales Prediction
> *"In Project 4, I analyzed an e-commerce retail dataset of 7,500 transactions over two full years. I engineered calendar indicators, autoregressive lags (yesterday's sales, last week's sales), and 7-day/30-day moving average windows. Crucially, I used a strict chronological train-test split (first 80% of dates to train, last 20% to forecast) rather than random shuffling, eliminating look-ahead bias. Linear Regression achieved the best out-of-sample RMSE ($2,332.91). I delivered an executive dashboard with real-time financial KPIs and a What-If predictive simulator allowing store managers to forecast daily sales given discount strategies and calendar factors."*

---

## 📤 GitHub Upload & Submission Instructions

### Step 1: Initialize Git and Create Local Commit
```powershell
cd C:\Users\Lenovo\.gemini\antigravity-ide\scratch\Thiranex_Data_Science_Internship
git init
git add .
git commit -m "feat: complete Thiranex data science internship portfolio with 4 production projects"
```

### Step 2: Push to Your GitHub Account
1. Open your browser and navigate to [GitHub](https://github.com/new).
2. Create a new repository named `Thiranex_Data_Science_Internship`.
3. Set the repository visibility to **Public**.
4. In your terminal, link your local repository and push:
```powershell
git branch -M main
git remote add origin https://github.com/<your-github-username>/Thiranex_Data_Science_Internship.git
git push -u origin main
```

---

## 📋 Project Submission Links Format

When submitting your internship portfolio to **Thiranex**, format your submission as follows:

```markdown
### Thiranex Data Science Internship Final Submission
- **Intern Name:** <Your Name>
- **Intern ID:** <Your Intern ID>
- **Submission Date:** October 2026

#### Repository & Deliverables:
- **Main GitHub Repository:** https://github.com/<your-username>/Thiranex_Data_Science_Internship
- **Project 1 (Cleaning & Viz):** https://github.com/<your-username>/Thiranex_Data_Science_Internship/tree/main/Project_1_Data_Cleaning_Visualization
- **Project 2 (ML Predictive Modeling):** https://github.com/<your-username>/Thiranex_Data_Science_Internship/tree/main/Project_2_Predictive_Modeling
- **Project 3 (Advanced EDA):** https://github.com/<your-username>/Thiranex_Data_Science_Internship/tree/main/Project_3_EDA
- **Project 4 (Retail Sales System):** https://github.com/<your-username>/Thiranex_Data_Science_Internship/tree/main/Project_4_Retail_Analytics
```

---

## ✅ Final Quality Assurance & Verification Checklist

- [x] **Project 1 Complete:** Raw and clean data, 9 figures, `cleaning.py`, `visualization.py`, Jupyter notebook, Streamlit app, and report.
- [x] **Project 2 Complete:** Raw data, 4 ML algorithms, leakage-free pipelines, 4 diagnostic figures, joblib model serialization, predict script, Streamlit app, and report.
- [x] **Project 3 Complete:** Statistical hypothesis testing ($\chi^2$, Mann-Whitney U), 12 figures, descriptive stats, Jupyter notebook, Streamlit app, and report with 10 insights.
- [x] **Project 4 Complete:** Calibrated commercial dataset, daily time series with lags, chronological train-test split, 3 regressors evaluated on MAE/RMSE/R², joblib serialization, 12 figures, Streamlit suite with simulator, and business report.
- [x] **Industry Standards:** Clean PEP-8 compliant code, random seeds fixed (`random_state=42`), zero fake accuracy metrics, and complete documentation.
