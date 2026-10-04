# Project 2: Predictive Modeling Using Machine Learning

Part of the **Thiranex Data Science Internship Portfolio**

---

## 📌 Project Overview
This project delivers a complete supervised machine learning pipeline on the **Breast Cancer Wisconsin (Diagnostic)** dataset. It rigorously emphasizes:
- Leakage-free preprocessing using Scikit-Learn `Pipeline`.
- Stratified train-test partitioning.
- Clinical metric prioritization (**Recall / Sensitivity** on malignant cases to minimize False Negatives).
- Model comparison across 4 algorithmic paradigms: Logistic Regression, Decision Tree, Random Forest, and Support Vector Machine.
- Joblib serialization and Streamlit clinical simulation dashboard.

> ⚠️ **Educational Advisory:** This project is strictly for educational purposes and is not a clinical medical diagnostic tool.

---

## 📂 Directory Structure
```
Project_2_Predictive_Modeling/
├── data/
│   └── breast_cancer_raw.csv
├── notebooks/
│   └── predictive_modeling.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
├── models/
│   ├── best_model.joblib
│   └── model_metadata.json
├── outputs/
│   └── figures/
│       ├── 01_model_comparison_metrics.png
│       ├── 02_confusion_matrices.png
│       ├── 03_roc_curves.png
│       └── 04_feature_importance.png
├── app.py
├── report.md
└── README.md
```

---

## 📊 Real Experimental Results (Held-Out Test Set, n=114)

| Algorithm | Accuracy | Precision (Malignant) | Recall (Sensitivity) | F1-Score | ROC-AUC |
|:----------|:--------:|:---------------------:|:--------------------:|:--------:|:-------:|
| **Logistic Regression** | **98.25%** | **97.62%** | **97.62%** | **0.9762** | **0.9954** |
| **Support Vector Machine** | **98.25%** | **97.62%** | **97.62%** | **0.9762** | **0.9950** |
| **Random Forest** | 94.74% | 92.86% | 92.86% | 0.9286 | 0.9947 |
| **Decision Tree** | 92.11% | 86.67% | 92.86% | 0.8966 | 0.9163 |

---

## 🚀 Execution Instructions

### 1. Train and Benchmark All Models
```powershell
python Project_2_Predictive_Modeling\src\train.py
```
*Generates figures in `outputs/figures/`, serializes model to `models/best_model.joblib`, and writes `models/model_metadata.json`.*

### 2. Run Standalone CLI Inference
```powershell
python Project_2_Predictive_Modeling\src\predict.py
```
*Runs automated diagnostic test cases on sample patient profiles.*

### 3. Launch Interactive Streamlit Dashboard
```powershell
streamlit run Project_2_Predictive_Modeling\app.py
```
*Provides real-time patient presets, slider controls, confidence scores, and benchmark charts.*
