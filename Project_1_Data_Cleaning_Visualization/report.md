# Project 1: Data Cleaning and Visualization Report

**Organization:** Thiranex Data Science Internship  
**Role:** Data Science Intern  
**Project Title:** End-to-End Data Cleaning, Outlier Remediation, and Diagnostic Visualization of the Titanic Passenger Manifest  
**Author:** Data Science Intern  
**Date:** October 2026  

---

## 1. Executive Summary

Data quality is the foundational prerequisite of reliable statistical inference and predictive machine learning. This project implements an industrial-grade data cleaning, validation, and diagnostic visualization workflow on the historical Titanic passenger dataset (891 passenger records, 12 attributes).

Key achievements include:
- Designed an automated ingestion and schema validation pipeline preserving original raw data in `data/raw/`.
- Performed statistical imputation for missing numerical (`Age`) and categorical (`Embarked`, `Cabin`) features without artificial distribution distortion.
- Handled extreme skewness and ticket fare anomalies up to $512.33 using **1.5x IQR Winsorization (capping)**, eliminating extreme leverage without dropping legitimate passenger records.
- Engineered predictive domain signals: extracted honorific titles (`Title`), quantified total traveling units (`FamilySize`), travel solitude (`IsAlone`), and physical vessel level (`Deck`).
- Rendered 9 publication-grade visualizations documenting missingness patterns, demographic survival disparities, family survival dynamics, and multivariate relationships.
- Deployed an interactive Streamlit application (`app.py`) for dynamic data filtering and cohort exploration.

---

## 2. Problem Statement

Raw datasets extracted from transactional logs or historical records frequently suffer from missingness, non-standard text entries, structural redundancies, and extreme outliers. Direct application of statistical models or machine learning algorithms to such uncurated data yields biased estimators, elevated variance, and invalid inferences. 

The objective of this project is to construct a modular, reproducible Python pipeline that systematically audits, transforms, cleans, and enriches the dataset while documenting every decision according to best data science practices.

---

## 3. Dataset Description

The Titanic manifest tracks passenger demographic, ticketing, and survival status during the vessel's maiden transatlantic voyage in April 1912.

| Feature Name | Data Type | Description | Raw Null Count | Null % |
|:-------------|:----------|:------------|:---------------|:-------|
| `PassengerId`| Integer   | Unique passenger identifier | 0 | 0.00% |
| `Survived`   | Integer   | Target variable (0 = Perished, 1 = Survived) | 0 | 0.00% |
| `Pclass`     | Integer   | Ticket class (1 = 1st Class, 2 = 2nd Class, 3 = 3rd Class) | 0 | 0.00% |
| `Name`       | String    | Passenger full legal name with honorific title | 0 | 0.00% |
| `Sex`        | String    | Passenger biological sex (`male`, `female`) | 0 | 0.00% |
| `Age`        | Float     | Passenger chronological age in years | 177 | 19.87% |
| `SibSp`      | Integer   | Number of siblings / spouses aboard | 0 | 0.00% |
| `Parch`      | Integer   | Number of parents / children aboard | 0 | 0.00% |
| `Ticket`     | String    | Ticket serial identifier number | 0 | 0.00% |
| `Fare`       | Float     | Passenger ticket fare in British Pounds (£) | 0 | 0.00% |
| `Cabin`      | String    | Cabin designation number | 687 | 77.10% |
| `Embarked`   | String    | Port of embarkation (`C` = Cherbourg, `Q` = Queenstown, `S` = Southampton) | 2 | 0.22% |

---

## 4. Technologies Used

- **Programming Language:** Python 3.11+
- **Data Manipulation:** Pandas 2.x, NumPy 1.26+
- **Statistical Computations:** SciPy
- **Data Visualization:** Matplotlib 3.8+, Seaborn 0.13+
- **Interactive Web Framework:** Streamlit 1.30+
- **Interactive Prototyping:** Jupyter Notebook (`.ipynb`)
- **Version Control:** Git & GitHub

---

## 5. Implementation Steps & Workflow Architecture

The project execution strictly adhered to a sequential, reproducible pipeline:

```
[Raw Data Ingestion] ──> [Schema & Null Audit] ──> [Duplicate Check]
         │
         ▼
[Grouped Median Imputation & Mode Fill] ──> [Deck Extraction]
         │
         ▼
[1.5x IQR Outlier Winsorization] ──> [Domain Feature Engineering]
         │
         ▼
[Processed Data Serialization] ──> [9 Publication Figures] ──> [Streamlit Dashboard]
```

1. **Ingestion & Cache Validation:** Retrieved raw data from remote repository, persisting an immutable snapshot in `data/raw/titanic_raw.csv`.
2. **Schema & Null Audit:** Extracted data types, missing percentage metrics, and cardinality distributions.
3. **Imputation:** Executed grouped conditional median imputation on `Age`, mode imputation on `Embarked`, and structural deck parsing on `Cabin`.
4. **Outlier Mitigation:** Formulated interquartile boundaries for `Fare` and applied upper-bound clipping.
5. **Feature Transformation:** Extracted social and behavioral indicators (`Title`, `FamilySize`, `IsAlone`, `AgeGroup`).
6. **Persistence:** Exported the verified clean table to `data/processed/titanic_cleaned.csv`.
7. **Visualization & Dashboard:** Synthesized 9 analytical figures in `outputs/figures/` and developed `app.py`.

---

## 6. Data Cleaning & Preprocessing Methodology

### 6.1 Duplicate Record Analysis
Initial examination confirmed that `df.duplicated().sum() == 0`. Every row corresponds to a distinct `PassengerId`.

### 6.2 Missing Value Remediation

#### Age (177 missing, 19.87%)
- **Rationale for Grouped Median:** Replacing missing ages with the global mean or median introduces artificial variance collapse and ignores strong demographic differences. For example, third-class males were significantly younger than first-class males.
- **Formulation:**
  $$\hat{\text{Age}}_{i} = \text{Median}(\text{Age} \mid \text{Pclass}_i, \text{Sex}_i)$$
- **Result:** Median age for 1st-class females was 35.0, 1st-class males 40.0, 3rd-class females 21.5, and 3rd-class males 25.0. Imputing conditionally preserved the demographic profile.

#### Embarked (2 missing, 0.22%)
- Missing values accounted for <0.25% of records.
- Imputed using the empirical mode: `"S"` (Southampton), which accounted for over 72.4% of total passenger boardings.

#### Cabin (687 missing, 77.10%)
- Deleting rows with missing cabins would discard 77% of the dataset.
- Deleting the column outright discards high-value spatial data: cabin letters correspond directly to vessel decks (`A` through `G`), which dictate physical proximity to lifeboats.
- **Solution:** Extracted the leading character representing Deck (`A`, `B`, `C`, `D`, `E`, `F`, `G`, `T`). Missing values were formally categorized as `"Missing"`, converting missingness itself into an informative categorical signal.

### 6.3 Outlier Detection & Treatment via IQR
- **Inspection:** Ticket fares exhibited severe right-skewness (skewness > 4.7), with minimum fare £0.00, median £14.45, 75th percentile £31.00, and max values reaching £512.33.
- **Mathematical Form:**
  $$\text{IQR} = Q_3 - Q_1 = 31.00 - 7.91 = 23.09$$
  $$\text{Upper Threshold} = Q_3 + 1.5 \times \text{IQR} = 31.00 + 1.5 \times 23.09 = 65.63$$
- **Treatment (Winsorization):** Rather than truncating 116 passenger records (13.02% of data), fares exceeding £65.63 were capped at £65.63. This retained complete passenger records while safeguarding subsequent regression and distance-based algorithms from extreme leverage points.

---

## 7. Feature Engineering Summary

| Engineered Feature | Calculation / Mapping | Business / Analytical Hypothesis |
|:-------------------|:----------------------|:---------------------------------|
| `Title` | Extracted from `Name` (`Mr`, `Mrs`, `Miss`, `Master`, `Officer`, `Royalty`) | Captures social standing, marital status, and age seniority |
| `FamilySize` | `SibSp + Parch + 1` | Distinguishes solo passengers from cohesive familial units |
| `IsAlone` | Binary: `1` if `FamilySize == 1`, else `0` | Isolates solo travel risks during emergency evacuation |
| `Deck` | First character of `Cabin` or `"Missing"` | Approximates physical vertical positioning on the ship |
| `AgeGroup` | Binned: `Child`, `Adolescent`, `Young_Adult`, `Middle_Aged`, `Senior` | Accommodates non-linear survival prioritization by life stage |

---

## 8. Visualizations & Analytical Interpretations

All high-resolution figures are generated and stored in `outputs/figures/`:

1. **`01_missing_values_heatmap.png`:** Highlights severe missingness in `Cabin` (77%) and moderate sparseness in `Age` (20%), justifying structured imputation.
2. **`02_missing_values_after_cleaning.png`:** Validates 100% data integrity with 0 remaining nulls across all active feature vectors.
3. **`03_age_distribution_before_after.png`:** Overlays KDE densities before and after grouped median imputation, confirming smooth distribution without artificial spikes.
4. **`04_survival_distribution.png`:** Documents baseline class imbalance: 549 passengers perished (61.6%), while 342 survived (38.4%).
5. **`05_correlation_heatmap.png`:** Quantifies negative correlation between `Pclass` and `Fare` (-0.55) and survival (+0.26 with Fare, -0.34 with Pclass).
6. **`06_fare_age_outliers_boxplots.png`:** Contrasts raw fare outliers up to $512 against the stabilized capped distribution at $65.63.
7. **`07_survival_by_pclass_and_sex.png`:** Unpacks survival disparity: 1st class females (96.8%) vs 3rd class males (13.5%).
8. **`08_family_size_survival_distribution.png`:** Reveals the inverted-U curve of survival: solo travelers had 30.4% survival, small families (3-4 members) peaked at 57.8%-72.4%, and large families (7+) collapsed below 15%.
9. **`09_embarked_vs_fare_distribution.png`:** Demonstrates Cherbourg passengers paid significantly higher median fares due to a greater proportion of 1st-class ticket holders.

---

## 9. Key Findings & Insights

1. **Maritime Protocol Dominance:** Gender was the single most decisive determinant of survival. Female passengers had a **74.2%** survival rate versus **18.9%** for male passengers.
2. **Socio-Economic Stratification:** Survival was heavily gated by cabin tier: **62.9%** for 1st Class, **47.3%** for 2nd Class, and **24.2%** for 3rd Class.
3. **Family Size Non-Linearity:** Traveling with 1 to 3 family members substantially enhanced survival odds (assistance during evacuation), whereas large families (5+) faced logistical failure in reaching lifeboats together.
4. **Age Dynamics:** Children under 12 achieved a **57.4%** survival rate across all classes, reflecting prioritizing youth alongside females.
5. **Port Socio-Economics:** Boarding port was strongly linked to ticket wealth: Cherbourg passengers had an average fare nearly double that of Queenstown passengers.

---

## 10. Limitations

- **Historical Record Gaps:** 77% of cabin entries are missing due to historical ticketing documentation limitations. While deck extraction provides signal, it contains an unobserved survival bias (survivors were more likely to report cabin numbers).
- **Lifeboat Allocation Unobserved:** Detailed assignments of individual passengers to specific port or starboard lifeboats are not present in this tabular manifest.

---

## 11. Conclusion & Recommendations

The structured cleaning pipeline transformed a messy, incomplete passenger manifest into a clean, leakage-free dataset. By combining grouped demographic imputation, 1.5x IQR Winsorization, and domain feature engineering, we maximized analytical signal while eliminating distribution distortion.

---

## 12. How to Run the Project

### Execute Cleaning & Visualization Pipeline:
```powershell
python Project_1_Data_Cleaning_Visualization\src\cleaning.py
python Project_1_Data_Cleaning_Visualization\src\visualization.py
```

### Launch Interactive Streamlit Dashboard:
```powershell
streamlit run Project_1_Data_Cleaning_Visualization\app.py
```
