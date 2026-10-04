# Project 3: Exploratory Data Analysis (EDA) Report

**Organization:** Thiranex Data Science Internship  
**Role:** Data Science Intern  
**Project Title:** In-Depth Exploratory Data Analysis, Multivariate Behavioral Interaction, and Hypothesis Testing on Titanic Disaster Demographics  
**Author:** Data Science Intern  
**Date:** October 2026  

---

## 1. Executive Summary

Exploratory Data Analysis (EDA) provides the analytical bedrock for discovering hidden distributions, structural anomalies, and causal correlations in tabular datasets. This project executes an industrial-standard exploratory study of 891 passenger records across 12 sociological, financial, and demographic dimensions.

Beyond basic summary metrics, this report integrates **inferential hypothesis testing (Pearson Chi-Square and Mann-Whitney U non-parametric tests)** to mathematically validate whether observed disparities in survival are genuine population phenomena or stochastic noise. Twelve publication-grade figures were generated in `outputs/figures/`, an interactive Streamlit analytical explorer was created (`app.py`), and ten actionable business/historical findings were formulated.

---

## 2. Problem Statement

Historical datasets frequently exhibit multi-collinearity, confounding variables, and non-linear interactions. A superficial univariate analysis fails to explain the underlying mechanisms governing passenger outcomes. The objective of this project is to apply exploratory statistics, distribution decomposition, and multivariate visualization to answer:
1. What was the exact mathematical hierarchy of survival determinants (gender vs socio-economic status vs family structure)?
2. Are ticket fare and socio-economic tier linearly related or power-law distributed?
3. What role did family size and social isolation play in crisis evacuation dynamics?

---

## 3. Dataset Description

The dataset captures passenger manifestations, cabin ticketing records, and demographic profiles:

| Feature Name | Type | Description | Descriptive Summary |
|:-------------|:-----|:------------|:--------------------|
| `Survived`   | Binary Nominal | 0 = Perished, 1 = Survived | 38.38% overall survival rate |
| `Pclass`     | Ordinal | Socio-economic Ticket Tier | 1st (24.2%), 2nd (20.7%), 3rd (55.1%) |
| `Sex`        | Nominal | Biological Gender | 64.76% Male, 35.24% Female |
| `Age`        | Continuous Ratio | Chronological Age in Years | Mean: 29.11, Median: 26.0, Range: 0.42 to 80.0 |
| `SibSp`      | Discrete Count | Siblings & Spouses aboard | Range: 0 to 8, 76.5% traveled with 0 siblings |
| `Parch`      | Discrete Count | Parents & Children aboard | Range: 0 to 6, 76.1% traveled with 0 parents/children |
| `Fare`       | Continuous Ratio | Ticket Price (£ / $) | Mean: $32.20, Median: $14.45, Skew: 4.79 (Raw) |
| `Embarked`   | Nominal | Port of Embarkation | Southampton (72.4%), Cherbourg (18.9%), Queenstown (8.7%) |
| `FamilySize` | Discrete Count | Total Traveling Unit (`SibSp+Parch+1`) | Solo (60.3%), Small 2-4 (34.5%), Large 5+ (5.2%) |
| `IsAlone`    | Binary Nominal | 1 = Solitary, 0 = Family Accompanied | 60.27% Solo Travelers |
| `Title`      | Nominal | Honorific Title from Name | `Mr` (58.0%), `Miss` (20.4%), `Mrs` (14.0%), `Master` (4.5%) |

---

## 4. Technologies Used

- **Core Analysis:** Python 3.11+, Pandas 2.x, NumPy 1.26+
- **Statistical Testing:** SciPy (`scipy.stats.chi2_contingency`, `scipy.stats.mannwhitneyu`)
- **Data Visualization:** Matplotlib 3.8+, Seaborn 0.13+
- **Interactive Application:** Streamlit 1.30+
- **Interactive Prototyping:** Jupyter Notebook (`.ipynb`)

---

## 5. Statistical Hypothesis Testing Methodology & Results

To substantiate visual patterns, we formulated formal null hypotheses ($H_0$) and conducted significance tests at $\alpha = 0.01$:

| Feature | Variable Type | Test Utilized | Null Hypothesis ($H_0$) | Test Statistic | p-value | Significance ($\alpha=0.01$) |
|:--------|:--------------|:--------------|:------------------------|:--------------:|:-------:|:-----------------------------:|
| `Sex` | Categorical | Pearson $\chi^2$ Test | Survival is independent of biological sex | $\chi^2 = 260.72$ | $1.20 \times 10^{-58}$ | **Statistically Significant** |
| `Title` | Categorical | Pearson $\chi^2$ Test | Survival is independent of honorific title | $\chi^2 = 290.68$ | $8.11 \times 10^{-60}$ | **Statistically Significant** |
| `Pclass` | Categorical | Pearson $\chi^2$ Test | Survival is independent of ticket tier | $\chi^2 = 102.89$ | $4.55 \times 10^{-23}$ | **Statistically Significant** |
| `Fare` | Numerical | Mann-Whitney U | Fare distribution is identical between groups | $U = 58207.5$ | $1.17 \times 10^{-21}$ | **Statistically Significant** |
| `IsAlone`| Categorical | Pearson $\chi^2$ Test | Traveling alone does not alter survival odds | $\chi^2 = 36.00$ | $1.97 \times 10^{-9}$ | **Statistically Significant** |
| `FamilySize`| Numerical | Mann-Whitney U | Family sizes are identical between groups | $U = 77659.0$ | $7.97 \times 10^{-7}$ | **Statistically Significant** |
| `Embarked` | Categorical | Pearson $\chi^2$ Test | Boarding port has no association with survival | $\chi^2 = 25.96$ | $2.30 \times 10^{-6}$ | **Statistically Significant** |
| `Age` | Numerical | Mann-Whitney U | Overall age distribution is identical | $U = 98171.5$ | $0.250$ | *Not Significant Globally* |

> **Key Inferential Insight on Age:**  
> Notice that `Age` is not globally significant ($p = 0.25$) when tested as a single monotonic continuous variable because child survival (>57%) and adult male mortality (<19%) offset each other. However, when partitioned conditionally by gender and socio-economic class, age interactions become profound.

---

## 6. Detailed Visualization Explanations (12 Publication Figures)

All figures were rendered at 300 DPI and stored in `outputs/figures/`:

1. **`01_univariate_numerical_distributions.png`:** Highlights positive skewness in `Fare` (skewness = 4.79) and `FamilySize`, confirming that parametric assumptions of normality are violated and non-parametric tests are required.
2. **`02_categorical_frequency_counts.png`:** Shows demographic baselines: 3rd Class passengers constituted over 55% of the ship, males represented nearly 65%, and Southampton accounted for over 72% of all boardings.
3. **`03_survival_by_gender_class_bar.png`:** Documents the stark intersection of class and gender: 1st Class females survived at 96.8%, 2nd Class females at 92.1%, 3rd Class females at 50.0%, 1st Class males at 36.9%, 2nd Class males at 15.7%, and 3rd Class males at 13.5%.
4. **`04_fare_distribution_by_class_boxplot.png`:** Shows that 1st class fares exhibit heavy variance, while 2nd and 3rd class fares are tightly compressed between £7 and £26.
5. **`05_violin_plots_age_pclass_survival.png`:** The split violins reveal that among survivors in 2nd and 3rd classes, the lower tail (children aged 0-10) is prominently expanded, visually proving child prioritization.
6. **`06_age_vs_fare_multivariate_scatter.png`:** Multi-attribute scatter demonstrates that green points (survivors) cluster heavily in the upper fare tier regardless of age, and along the young age axis for lower fares.
7. **`07_correlation_matrices_pearson_spearman.png`:** Demonstrates that Spearman rank correlation between `Pclass` and `Fare` (-0.69) is much stronger than Pearson linear correlation (-0.55), proving exponential ticket pricing tiers.
8. **`08_family_travel_dynamics_analysis.png`:** Visualizes the classic inverted-U family curve: peak survival at family size 3 (57.8%) and size 4 (72.4%), collapsing to 0% for family sizes of 8 and 11.
9. **`09_embarkation_port_socioeconomics.png`:** Explains why Cherbourg had higher aggregate survival (55.4%): Cherbourg had a 50.6% 1st-class cohort compared to only 19.7% in Southampton.
10. **`10_pairplot_key_features.png`:** Pairwise KDE and scatter matrices demonstrating non-linear boundaries separating survivors and non-survivors across multidimensional space.
11. **`11_survival_by_title_and_age_group.png`:** Illustrates title hierarchy: `Mrs` (79.2%), `Miss` (69.8%), `Master` (57.5%), `Officer` (27.8%), and `Mr` (15.7%).
12. **`12_statistical_hypothesis_tests_summary.png`:** Visual summary bar chart on logarithmic scale ranking features by their Chi-Square and Mann-Whitney test statistics.

---

## 7. Top 10 Executive Business & Historical Insights

### Insight 1: Absolute Primacy of the Gender Protocol
- **Finding:** Biological sex was the single most powerful predictor ($\chi^2 = 260.72, p = 1.2 \times 10^{-58}$). Females achieved a **74.2%** survival rate versus **18.9%** for males.
- **Implication:** Maritime emergency evacuation adhered strictly to the "women and children first" doctrine, overriding all other social considerations.

### Insight 2: Structural Class Stratification
- **Finding:** Passenger class created an inescapable survival gradient ($\chi^2 = 102.89, p = 4.5 \times 10^{-23}$): 1st Class (62.9%), 2nd Class (47.3%), and 3rd Class (24.2%).
- **Implication:** Physical cabin location on upper decks versus steerage deep in the hull directly governed escape time before bulkhead closure.

### Insight 3: The Compounded Privilege of 1st-Class Females
- **Finding:** 1st-class females achieved **96.8% survival** (only 3 perished out of 94). By contrast, 3rd-class males suffered **86.5% mortality** (only 47 survived out of 347).
- **Implication:** Demographic intersectionality produces orders-of-magnitude differences in risk outcomes.

### Insight 4: The Inverted-U Family Dynamic
- **Finding:** Survival probability followed a non-linear parabolic curve with respect to family size. Solo travelers had 30.4% survival. Families of 2, 3, and 4 members peaked at 55.2%, 57.8%, and 72.4% survival. Families of 5+ plummeted below 20%.
- **Implication:** Small groups provide mutual aid and physical assistance, whereas large groups suffer from communication bottlenecks and refuse to board lifeboats unless all members are accounted for.

### Insight 5: Vulnerability of Solitary Travelers
- **Finding:** Solo passengers (`IsAlone = 1`) constituted **60.3%** of all passengers and accounted for **69.6%** of all fatalities ($\chi^2 = 36.00, p = 1.97 \times 10^{-9}$).
- **Implication:** Social isolation in emergency operations significantly increases mortality risk due to lack of shared situational awareness.

### Insight 6: Youth Prioritization in 2nd Class
- **Finding:** Male children under age 14 traveling under the title "Master" achieved a **100% survival rate in 2nd Class** (10 out of 10) and 80% in 1st Class, compared to only 8.3% for adult males in 2nd Class.
- **Implication:** Age triage was enforced strictly, prioritizing young boys over adult men regardless of wealth.

### Insight 7: Purchasing Power and Ticket Fare Elasticity
- **Finding:** Non-parametric Mann-Whitney testing verified that survivors paid significantly higher fares ($U = 58207.5, p = 1.17 \times 10^{-21}$).
- **Implication:** Financial investment in travel correlated directly with safety infrastructure and proximity to lifeboat decks.

### Insight 8: Embarkation Port as a Socio-Economic Confounder
- **Finding:** Cherbourg passengers had a 55.4% survival rate versus 33.7% for Southampton. Port was not a causal driver; Cherbourg was a luxury French boarding hub where 50.6% of passengers purchased 1st-class tickets.
- **Implication:** Disentangling confounding geographic covariates is critical before drawing causal conclusions.

### Insight 9: Title Outperforms Raw Chronological Age
- **Finding:** Honorific title ($\chi^2 = 290.68$) provided greater discriminatory power than continuous age. Titles implicitly combined gender, marital status, and age maturity into a single cohesive social signal.
- **Implication:** Domain-driven categorical parsing often captures richer non-linear signals than raw numerical variables.

### Insight 10: Non-Linearity in Ticket Pricing
- **Finding:** Spearman rank correlation between Class and Fare was **-0.69**, whereas Pearson correlation was only **-0.55**.
- **Implication:** First-class luxury suites exhibited exponential price increases rather than linear increments, confirming power-law wealth dynamics.

---

## 8. Limitations & Analytical Constraints

- **Lack of Temporal Evacuation Logs:** The manifest does not record exact passenger time-of-arrival at boat stations or the sequence in which lifeboats were lowered.
- **Survival Record Reporting Bias:** Cabin numbers were more diligently recorded for surviving passengers who could be interviewed after rescue.

---

## 9. Conclusion

This project demonstrated how exploratory data analysis bridges raw metrics and domain intelligence. By combining distribution profiling, cross-tabulated visualizations, and rigorous hypothesis testing, we proved that evacuation survival was governed by a strict hierarchy of **gender doctrine**, **physical deck location (class)**, and **family coordination logistics**.

---

## 10. How to Run the Project

### Execute Complete EDA Script & Generate 12 Figures:
```powershell
python Project_3_EDA\src\analysis.py
```

### Launch Interactive Streamlit EDA Application:
```powershell
streamlit run Project_3_EDA\app.py
```
