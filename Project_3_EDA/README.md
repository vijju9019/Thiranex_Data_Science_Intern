# Project 3: Exploratory Data Analysis (EDA) & Statistical Inference

Part of the **Thiranex Data Science Internship Portfolio**

---

## 📌 Project Overview
This project delivers a comprehensive, multi-dimensional Exploratory Data Analysis (EDA) examining the sociological, financial, and demographic dynamics of the Titanic passenger cohort.

It incorporates **both descriptive visualizations and formal inferential hypothesis testing** (Chi-Square and Mann-Whitney U tests) to scientifically substantiate patterns of survival.

---

## 🎯 Key Deliverables
- Complete statistical profiling: Mean, Std, Median, IQR, Skewness, Kurtosis.
- Hypothesis testing engine computing exact p-values for all features.
- 12 publication-grade figures saved in `outputs/figures/`.
- 10 structured, actionable historical and business insights.
- Fully executable Jupyter Notebook (`notebooks/exploratory_data_analysis.ipynb`).
- Interactive Streamlit exploratory application (`app.py`).

---

## 📂 Directory Structure
```
Project_3_EDA/
├── data/
│   └── titanic_data.csv
├── notebooks/
│   └── exploratory_data_analysis.ipynb
├── src/
│   └── analysis.py
├── outputs/
│   └── figures/
│       ├── 01_univariate_numerical_distributions.png
│       ├── 02_categorical_frequency_counts.png
│       ├── 03_survival_by_gender_class_bar.png
│       ├── 04_fare_distribution_by_class_boxplot.png
│       ├── 05_violin_plots_age_pclass_survival.png
│       ├── 06_age_vs_fare_multivariate_scatter.png
│       ├── 07_correlation_matrices_pearson_spearman.png
│       ├── 08_family_travel_dynamics_analysis.png
│       ├── 09_embarkation_port_socioeconomics.png
│       ├── 10_pairplot_key_features.png
│       ├── 11_survival_by_title_and_age_group.png
│       └── 12_statistical_hypothesis_tests_summary.png
├── app.py
├── report.md
└── README.md
```

---

## 🚀 Execution Instructions

### 1. Run Complete Statistical EDA Pipeline
```powershell
python Project_3_EDA\src\analysis.py
```
*Generates all 12 publication figures in `outputs/figures/` and outputs statistical test summaries to the console.*

### 2. Launch Interactive Streamlit EDA Explorer
```powershell
streamlit run Project_3_EDA\app.py
```
*Accessible locally at `http://localhost:8501` featuring interactive distributions, bivariate plots, hypothesis calculators, and insight cards.*

---

## 🔬 Summary of Key Statistical Insights
1. **Gender Primacy:** Female survival was 74.2% vs 18.9% for males ($\chi^2 = 260.7, p < 10^{-50}$).
2. **Socio-Economic Egress Privilege:** 1st Class (62.9%) > 2nd Class (47.3%) > 3rd Class (24.2%) ($\chi^2 = 102.9, p < 10^{-22}$).
3. **Double-Privilege Effect:** 1st-class females achieved 96.8% survival; 3rd-class males suffered 86.5% mortality.
4. **Inverted-U Family Dynamics:** Peak survival occurred in families of 3-4 (58-72%), collapsing for solo travelers (30.4%) and large families 7+ (<15%).
5. **Solitary Vulnerability:** Solo passengers accounted for 60.3% of travelers and 69.6% of casualties ($\chi^2 = 36.0, p < 10^{-8}$).
