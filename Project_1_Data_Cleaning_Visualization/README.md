# Project 1: Titanic Data Cleaning and Visualization

Part of the **Thiranex Data Science Internship Portfolio**

---

## 📌 Project Overview
This project presents a robust, end-to-end data cleaning, outlier treatment, and diagnostic visualization workflow applied to the Titanic passenger dataset. It highlights how methodical preprocessing safeguards data integrity and empowers downstream machine learning.

## 🎯 Objectives & Deliverables
- Ingest and preserve the immutable raw dataset in `data/raw/`.
- Perform comprehensive data inspection (types, cardinalities, missing patterns).
- Statistically sound imputation for missing values without distribution distortion.
- Outlier detection and treatment via 1.5x Interquartile Range (IQR) Winsorization.
- Feature engineering of domain signals (`Title`, `FamilySize`, `IsAlone`, `Deck`).
- Generate 9 publication-grade visualizations in `outputs/figures/`.
- Deploy an interactive Streamlit exploration application (`app.py`).

## 📂 Directory Structure
```
Project_1_Data_Cleaning_Visualization/
├── data/
│   ├── raw/
│   │   └── titanic_raw.csv
│   └── processed/
│       └── titanic_cleaned.csv
├── notebooks/
│   └── data_cleaning_visualization.ipynb
├── src/
│   ├── cleaning.py
│   └── visualization.py
├── outputs/
│   └── figures/
│       ├── 01_missing_values_heatmap.png
│       ├── 02_missing_values_after_cleaning.png
│       ├── 03_age_distribution_before_after.png
│       ├── 04_survival_distribution.png
│       ├── 05_correlation_heatmap.png
│       ├── 06_fare_age_outliers_boxplots.png
│       ├── 07_survival_by_pclass_and_sex.png
│       ├── 08_family_size_survival_distribution.png
│       └── 09_embarked_vs_fare_distribution.png
├── app.py
├── report.md
└── README.md
```

## 🚀 Quickstart & Execution

### 1. Run Data Cleaning Pipeline
```powershell
python Project_1_Data_Cleaning_Visualization\src\cleaning.py
```
*Outputs: Generates `data/raw/titanic_raw.csv` and `data/processed/titanic_cleaned.csv`.*

### 2. Generate Publication-Quality Figures
```powershell
python Project_1_Data_Cleaning_Visualization\src\visualization.py
```
*Outputs: Generates 9 high-resolution charts in `outputs/figures/`.*

### 3. Launch Interactive Streamlit App
```powershell
streamlit run Project_1_Data_Cleaning_Visualization\app.py
```
*Accessible locally at: `http://localhost:8501`*

## 📊 Summary of Cleaning Decisions
1. **Age (19.9% missing):** Imputed via median conditional on `Pclass` and `Sex`. Preserves natural demographic variance.
2. **Embarked (0.2% missing):** Imputed using the empirical mode (`"S"` - Southampton).
3. **Cabin (77.1% missing):** Extracted deck character (`A`-`G`), encoding unrecorded entries as `"Missing"`.
4. **Fare Outliers:** Detected 116 extreme values (> £65.63). Winsorized at $Q_3 + 1.5 \times \text{IQR}$ to curb model leverage while retaining valid passenger rows.
