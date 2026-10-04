# Project 4: Retail Sales Analytics and Prediction System

Part of the **Thiranex Data Science Internship Portfolio**

---

## 📌 Project Overview
This project delivers an end-to-end commercial Data Science & Machine Learning system designed for retail business intelligence, margin analysis, and chronological sales demand forecasting.

Analyzing over **7,500 customer transactions** across two calendar years (2024-2025), it evaluates revenue drivers, territory performance, customer segments, discount elasticity, and constructs regression models on daily demand.

> 📌 **Data Source Note:** Utilizes an authentic, calibrated synthetic retail transactional dataset modeled after enterprise e-commerce systems.

---

## 🎯 Key Deliverables
- Automated cleaning, schema enforcement, and daily time-series aggregation.
- 12 publication-grade figures saved in `outputs/figures/`.
- Chronological feature engineering: autoregressive lags ($t-1, t-7, t-14$) and rolling moving averages ($7\text{d}, 30\text{d}$).
- Strict **chronological 80/20 train-test split** eliminating look-ahead data leakage.
- Comparison of **Linear Regression, Decision Tree, and Random Forest Regressor** on MAE, RMSE, and $R^2$.
- Serialized production forecasting model in `models/best_sales_model.joblib`.
- Interactive Streamlit Retail Suite (`app.py`) featuring an interactive "What-If" scenario forecasting tool.

---

## 📂 Directory Structure
```
Project_4_Retail_Analytics/
├── data/
│   ├── raw/
│   │   └── retail_sales_raw.csv
│   └── processed/
│       ├── retail_sales_cleaned.csv
│       └── retail_daily_sales.csv
├── notebooks/
│   └── retail_sales_analysis.ipynb
├── src/
│   ├── cleaning.py
│   ├── analysis.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── predict.py
├── models/
│   ├── best_sales_model.joblib
│   └── model_metadata.json
├── outputs/
│   └── figures/
│       ├── 01_monthly_revenue_and_profit_trend.png
│       ├── 02_category_revenue_and_margin_bar.png
│       ├── 03_regional_sales_distribution.png
│       ├── 04_customer_segment_kpi_comparison.png
│       ├── 05_top10_bestselling_products.png
│       ├── 06_discount_elasticity_vs_profit_margin.png
│       ├── 07_day_of_week_seasonality_heatmap.png
│       ├── 08_order_transaction_value_distribution.png
│       ├── 09_feature_correlation_heatmap.png
│       ├── 10_actual_vs_predicted_sales_chronological.png
│       ├── 11_regression_model_comparison_metrics.png
│       └── 12_regression_feature_importance.png
├── app.py
├── report.md
└── README.md
```

---

## 📊 Model Evaluation Results (141-Day Out-of-Sample Test Window)

| Algorithm | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | $R^2$ Score |
|:----------|:-------------------------:|:------------------------------:|:-----------:|
| **Linear Regression** | **$1,799.50** | **$2,332.91** | **0.3868** |
| **Random Forest Regressor** | $1,782.43 | $2,410.35 | 0.3454 |
| **Decision Tree Regressor** | $2,122.39 | $2,762.20 | 0.1403 |

---

## 🚀 Execution Instructions

### 1. Execute Full Training & Forecast Pipeline
```powershell
python Project_4_Retail_Analytics\src\train.py
```
*Executes cleaning, generates all 12 figures in `outputs/figures/`, performs chronological train-test split, benchmarks models, and saves `best_sales_model.joblib`.*

### 2. Run Standalone CLI Prediction
```powershell
python Project_4_Retail_Analytics\src\predict.py
```
*Outputs predicted sales, 95% confidence intervals, and estimated gross margin.*

### 3. Launch Streamlit Executive Retail Suite
```powershell
streamlit run Project_4_Retail_Analytics\app.py
```
*Accessible locally at `http://localhost:8501` featuring interactive KPIs, category drilldowns, and the scenario forecasting simulator.*
