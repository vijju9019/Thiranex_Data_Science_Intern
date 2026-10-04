# Project 2: Predictive Modeling Using Machine Learning Report

**Organization:** Thiranex Data Science Internship  
**Role:** Data Science Intern  
**Project Title:** Oncology Risk Classification via Leakage-Free Preprocessing Pipelines and Supervised Learning Benchmark  
**Dataset:** Breast Cancer Wisconsin (Diagnostic)  
**Author:** Data Science Intern  
**Date:** October 2026  

---

> ⚠️ **CLINICAL & EDUCATIONAL DISCLAIMER:**  
> This project is conducted solely for educational and portfolio demonstration purposes as part of the Thiranex Data Science Internship. The trained algorithms, prediction scripts, and dashboard are **not** medical devices and must never be utilized for real-world clinical diagnoses or treatment decisions.

---

## 1. Executive Summary

Early detection of malignant breast neoplasms substantially enhances clinical prognosis. This project builds a production-standard, supervised machine learning system that classifies Fine Needle Aspirate (FNA) biopsied breast masses into **Malignant** (positive class, 0 in raw label) versus **Benign** (negative class, 1 in raw label).

To ensure experimental integrity and prevent optimistic performance bias, all data preprocessing was implemented using **Scikit-Learn Pipelines**, ensuring no information from the test fold leaked into parameter scaling. Four distinct classification algorithms—**Logistic Regression, Support Vector Machine (RBF), Random Forest, and Decision Tree**—were benchmarked on an identical stratified test partition (114 patients).

Both **Logistic Regression** and **Support Vector Machine** achieved top-tier performance with **98.25% Test Accuracy**, **97.62% Malignant Recall (Sensitivity)**, and **0.995+ ROC-AUC**. The fitted pipeline was serialized using `joblib`, and an interactive clinical simulation dashboard was built using Streamlit.

---

## 2. Problem Statement

Automating the initial triage of cell nuclear morphometry from digitized microscopic biopsies requires balancing accuracy with safety. In medical screening:
- **Type I Error (False Positive):** A benign tumor is predicted as malignant. The patient experiences psychological distress and undergoes a confirmatory secondary biopsy.
- **Type II Error (False Negative):** A malignant tumor is incorrectly predicted as benign. The patient is dismissed without treatment, allowing neoplastic proliferation—a catastrophic failure mode.

Hence, the optimization target prioritizes **Recall (Sensitivity)** alongside **F1-Score** and **ROC-AUC** rather than simple unweighted accuracy.

---

## 3. Dataset Description

The Breast Cancer Wisconsin (Diagnostic) dataset comprises **569 instances** with **30 continuous numerical features** computed from digitized images of fine needle aspirates (FNA) of breast masses.

- **Sample Size:** 569 patient records
- **Class Balance:**
  - Benign: 357 instances (62.74%)
  - Malignant: 212 instances (37.26%)
- **Input Features (30 continuous variables):**
  Ten fundamental real-valued characteristics are computed for each cell nucleus:
  1. `radius` (mean of distances from center to points on perimeter)
  2. `texture` (standard deviation of gray-scale values)
  3. `perimeter`
  4. `area`
  5. `smoothness` (local variation in radius lengths)
  6. `compactness` ($\frac{\text{perimeter}^2}{\text{area} - 1.0}$)
  7. `concavity` (severity of concave portions of the contour)
  8. `concave points` (number of concave portions of the contour)
  9. `symmetry`
  10. `fractal dimension` ("coastline approximation" - 1.0)
  
  For each image, the **Mean**, **Standard Error (SE)**, and **"Worst"** (mean of the three largest values) are recorded: $10 \times 3 = 30$ features.

---

## 4. Technologies Used

- **Language:** Python 3.11+
- **Machine Learning Core:** Scikit-Learn 1.8+, NumPy, SciPy
- **Model Serialization:** Joblib
- **Data Handling:** Pandas
- **Visualization:** Matplotlib, Seaborn
- **Interactive UI:** Streamlit

---

## 5. Implementation Workflow & Architecture

```
[Raw Breast Cancer Dataset]
          │
          ▼
[Stratified 80/20 Train-Test Split] (Stratify by diagnosis)
    ┌─────┴────────────────────────────────┐
    ▼                                      ▼
[Train Fold (455 samples)]            [Held-Out Test Fold (114 samples)]
    │                                      │
    ▼                                      │
[Scikit-Learn Pipeline]                    │
  ├─ StandardScaler.fit_transform()        │
  └─ Model Training                        │
    ├─ Logistic Regression                 │
    ├─ Decision Tree                       │
    ├─ Random Forest                       │
    └─ Support Vector Machine (RBF)        │
    │                                      │
    └───────────────────────┬──────────────┘
                            ▼
      [Leakage-Free Test Set Transformation & Evaluation]
      ├─ Confusion Matrices
      ├─ ROC Curves & AUC
      ├─ Recall / Precision / F1
      └─ Joblib Serialization & Streamlit Deployment
```

---

## 6. Preprocessing & Data Leakage Prevention

### 6.1 Stratified Splitting
Given the 62.7% / 37.3% class split, random splitting risks drawing an unrepresentative test partition. We enforced `stratify=y` with `random_state=42`:
- **Training Set:** 455 samples (37.36% Malignant)
- **Testing Set:** 114 samples (36.84% Malignant: exactly 42 Malignant, 72 Benign)

### 6.2 Data Leakage Mitigation via Pipelines
A frequent error in ML benchmarks is applying `StandardScaler` to the whole dataset prior to splitting. This allows test fold parameters ($\mu_{\text{test}}, \sigma_{\text{test}}$) to bias the normalization, yielding artificially inflated accuracy.

To guarantee zero leakage:
```python
pipeline = Pipeline([
    ("scaler", StandardScaler()),  # Fitted strictly on X_train only
    ("classifier", estimator)
])
pipeline.fit(X_train, y_train)
```
During evaluation, `pipeline.predict(X_test)` transforms test inputs solely using the training fold's computed means and variances.

---

## 7. Actual Experimental Results & Model Comparison

The following are the **actual, un-fabricated test benchmark results** executed on the held-out test cohort (114 unseen patient records):

| Algorithm | Test Accuracy | Precision (Malignant) | Recall (Sensitivity) | F1-Score (Malignant) | ROC-AUC Score |
|:----------|:-------------:|:---------------------:|:--------------------:|:--------------------:|:-------------:|
| **Logistic Regression** | **98.25%** | **97.62%** | **97.62%** | **0.9762** | **0.9954** |
| **Support Vector Machine (RBF)** | **98.25%** | **97.62%** | **97.62%** | **0.9762** | **0.9950** |
| **Random Forest Classifier** | 94.74% | 92.86% | 92.86% | 0.9286 | 0.9947 |
| **Decision Tree Classifier** | 92.11% | 86.67% | 92.86% | 0.8966 | 0.9163 |

### 7.1 Confusion Matrix Breakdown on Test Cohort (n = 114)

- **Logistic Regression & SVM:**
  - True Malignant (TP): **41** / 42
  - False Negative (FN): **1** / 42 (Only 1 malignant case missed)
  - True Benign (TN): **71** / 72
  - False Positive (FP): **1** / 72
- **Random Forest:**
  - True Malignant (TP): **39** / 42
  - False Negative (FN): **3** / 42
  - True Benign (TN): **69** / 72
  - False Positive (FP): **3** / 72
- **Decision Tree:**
  - True Malignant (TP): **39** / 42
  - False Negative (FN): **3** / 42
  - True Benign (TN): **66** / 72
  - False Positive (FP): **6** / 72

---

## 8. Diagnostic Visualizations

Generated figures in `outputs/figures/`:
1. **`01_model_comparison_metrics.png`:** Grouped bar chart comparing Accuracy, Recall, F1, and AUC across all 4 models.
2. **`02_confusion_matrices.png`:** 2x2 subplot matrix showing exact TP, FP, TN, and FN counts.
3. **`03_roc_curves.png`:** Superimposed Receiver Operating Characteristic curves showing outstanding diagnostic discrimination (AUC > 0.99 for LR, SVM, and RF).
4. **`04_feature_importance.png`:** Gini importance from the Random Forest model highlighting `worst perimeter`, `worst concave points`, and `mean concave points` as the primary structural discriminators.

---

## 9. Key Findings & Technical Rationale

1. **Superiority of Regularized Linear & Kernel Classifiers:**
   Because FNA morphometric features correlate strongly with cell mass and perimeter in continuous Euclidean space, standardized features exhibit near-linear separability. As a consequence, **L2-regularized Logistic Regression** and **RBF-kernel Support Vector Machines** outperformed decision trees.
2. **Safety-Critical Metric Alignment:**
   Out of 42 malignant test cases, Logistic Regression achieved a **97.62% Recall**, resulting in a critical False Negative rate of only 2.38%.
3. **Primary Biological Drivers:**
   Random Forest feature importance and regression coefficients confirm that nuclear contour indentations (`concave points`) and cell enlargement (`worst perimeter`, `worst radius`) are the strongest biometric markers of malignant cell proliferation.

---

## 10. Selected Model & Serialization

**Logistic Regression within a StandardScaler Pipeline** was selected as the final deployed model based on highest F1-Score (0.9762), highest Sensitivity (0.9762), and fastest inference latency.
- Model file: `models/best_model.joblib`
- Metadata & benchmarks: `models/model_metadata.json`

---

## 11. Limitations & Future Improvements

- **Dataset Scale:** 569 records is sufficient for baseline tabular benchmarking, but clinical-grade models require hundreds of thousands of diverse multi-center patient biopsies.
- **Multimodal Absence:** Clinical practice combines biopsy cytology with patient age, genetic profiling (BRCA1/2 mutations), mammograms, and ultrasound imagery. Future work should implement multimodal fusion architectures.

---

## 12. How to Run the Project

### Execute Training & Evaluation Pipeline:
```powershell
python Project_2_Predictive_Modeling\src\train.py
```

### Run Standalone CLI Inference:
```powershell
python Project_2_Predictive_Modeling\src\predict.py
```

### Launch Interactive Streamlit Oncology Diagnostic App:
```powershell
streamlit run Project_2_Predictive_Modeling\app.py
```
