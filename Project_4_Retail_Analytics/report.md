# Project 4: Retail Sales Analytics and Prediction System Report

**Organization:** Thiranex Data Science Internship  
**Role:** Data Science Intern  
**Project Title:** End-to-End Enterprise Retail Sales Analytics, Profitability Optimization, and Chronological Demand Forecasting  
**Domain:** Retail & E-Commerce Business Analytics  
**Author:** Data Science Intern  
**Date:** October 2026  

---

## 1. Executive Summary

In commercial retail and omni-channel e-commerce, accurate demand prediction directly dictates inventory holding costs, stockout mitigation, promotional pricing elasticity, and quarterly working capital. This capstone project implements an industrial-grade analytics and regression forecasting engine tracking **7,500 customer transactions** over two full calendar years (January 2024 through December 2025).

Core deliverables include:
- Ingestion, schema validation, and financial calculations (Revenue, COGS, Profit, Margin %) preserving raw data in `data/raw/`.
- Exploratory commercial business intelligence across four product categories, four geographic territories, and three customer segments.
- Feature engineering of calendar signals, autoregressive lags ($t-1, t-7, t-14$), and moving average windows ($7\text{d}, 30\text{d}$).
- Strict **chronological train-test splitting (80% historical training, 20% out-of-sample forecast)** to completely eliminate look-ahead data leakage.
- Benchmarking of **Linear Regression, Decision Tree Regressor, and Random Forest Regressor** using MAE, MSE, RMSE, and $R^2$.
- Generation of 12 publication-grade figures in `outputs/figures/`.
- Deployment of an executive Streamlit analytics suite (`app.py`) featuring an interactive "What-If" scenario forecasting simulator.

---

## 2. Problem Statement

Retail executives frequently encounter two conflicting inventory risks:
1. **Under-Stocking (Stockouts):** Lost revenue opportunities, impaired customer lifetime value (LTV), and brand abandonment.
2. **Over-Stocking (Inventory Bloat):** High warehouse holding costs, capital lockup, and distressed end-of-season markdown discounting.

Traditional static spreadsheets fail to incorporate day-of-week seasonality, promotional discount erosion, and autoregressive sales momentum. The objective of this project is to build an automated machine learning regression system that forecasts daily revenue and guides pricing and inventory allocation.

---

## 3. Dataset Description

> 📌 **Data Provenance Documentation:**  
> This project utilizes an authentic, calibrated synthetic retail transactional dataset generated using NumPy and Pandas, carefully modeled after enterprise e-commerce transactional systems.

### Transaction Schema (7,500 Order Records):
- `Order ID`: Alphanumeric order primary key (`ORD-20240001` through `ORD-20247500`)
- `Order Date`: Daily timestamp spanning 2024-01-01 to 2025-12-31
- `Customer ID`: Unique customer identifier (`CUST-1000` to `CUST-2500`)
- `Customer Segment`: `Consumer` (50%), `Corporate` (30%), `Home Office` (20%)
- `Region`: `North` (28%), `South` (22%), `East` (30%), `West` (20%)
- `Product Category`: `Electronics` (35%), `Furniture` (25%), `Office Supplies` (22%), `Apparel` (18%)
- `Product Name`: 23 catalog items spanning consumer electronics, office furniture, supplies, and apparel
- `Quantity`: Discrete order quantity (1 to 15 units)
- `Unit Price`: Base item price ($18.00 to $650.00)
- `Unit Cost`: Cost of Goods Sold (COGS)
- `Discount`: Applied promotional rate (0% to 25%)
- `Revenue`: Net transaction price: $\text{Quantity} \times \text{Unit Price} \times (1 - \text{Discount})$
- `Cost`: Total COGS: $\text{Quantity} \times \text{Unit Cost}$
- `Profit`: Net margin: $\text{Revenue} - \text{Cost}$
- `Payment Method`: `Credit Card` (55%), `Debit Card` (20%), `PayPal` (18%), `Bank Transfer` (7%)

---

## 4. Technologies Used

- **Programming Language:** Python 3.11+
- **Data Engineering & Manipulation:** Pandas 2.x, NumPy 1.26+
- **Machine Learning Regressors:** Scikit-Learn 1.8+ (`LinearRegression`, `DecisionTreeRegressor`, `RandomForestRegressor`)
- **Model Serialization:** Joblib
- **Visualization:** Matplotlib 3.8+, Seaborn 0.13+
- **Interactive Web Suite:** Streamlit 1.30+

---

## 5. Feature Engineering & Chronological Splitting Methodology

### 5.1 Time-Series Feature Engineering
Daily aggregated metrics ($N = 701$ continuous trading days after lag trimming) were enriched with three categories of features:
1. **Calendar Signals:** `DayOfWeek` (0-6), `DayOfMonth` (1-31), `Month` (1-12), `Quarter` (1-4), `IsWeekend` (0/1), `DayOfYear` (1-365).
2. **Autoregressive Lags:**
   - $\text{Revenue}_{t-1}$: Yesterday's sales (immediate momentum)
   - $\text{Revenue}_{t-7}$: Same day of the prior week (weekly seasonality anchor)
   - $\text{Revenue}_{t-14}$: Two-week prior baseline
3. **Rolling Window Aggregations:**
   - $\text{Revenue\_Roll7\_Mean}$: 7-day moving average
   - $\text{Revenue\_Roll7\_Std}$: 7-day short-term sales volatility
   - $\text{Revenue\_Roll30\_Mean}$: 30-day monthly macro trend
4. **Exogenous Operational Indicators:** `Avg_Discount`, `Avg_Unit_Price`.

### 5.2 Strict Chronological Splitting (Zero Look-Ahead Bias)
Standard randomized cross-validation is **invalid** for time series because training on future dates to predict past dates causes severe look-ahead leakage.
- **Training Set (Earliest 80%):** 560 calendar days (January 31, 2024 to August 12, 2025)
- **Out-of-Sample Test Set (Latest 20%):** 141 calendar days (August 13, 2025 to December 31, 2025)

---

## 6. Model Evaluation & Benchmark Results

All algorithms were trained on the identical 560-day historical window and evaluated on the held-out 141-day future forecast window:

| Algorithm | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | $R^2$ Score | Execution Rationale |
|:----------|:-------------------------:|:------------------------------:|:-----------:|:--------------------|
| **Linear Regression** | **$1,799.50** | **$2,332.91** | **0.3868** | **Best Out-of-Sample R² & RMSE** (Stable linear trend capture) |
| **Random Forest Regressor** | $1,782.43 | $2,410.35 | 0.3454 | Lowest MAE; captures non-linear weekend and holiday spikes |
| **Decision Tree Regressor** | $2,122.39 | $2,762.20 | 0.1403 | Susceptible to high variance on continuous moving averages |

### 6.1 Evaluation Discussion
- **Linear Regression** achieved the lowest overall RMSE ($2,332.91) and highest $R^2$ (0.3868). Because daily sales fluctuations are driven by strong continuous moving averages ($\text{Roll7\_Mean}$) and linear weekend lifts, a regularized linear model avoids the overfitting risks inherent in single decision trees.
- **Random Forest** achieved the lowest absolute error (MAE $1,782.43), making it highly effective for estimating typical day-to-day demand while occasionally under-predicting extreme holiday spikes.

---

## 7. Diagnostic Visualizations (12 Publication Figures)

1. **`01_monthly_revenue_and_profit_trend.png`:** Line trajectory documenting sustained revenue growth and significant holiday quarter surges in Q4.
2. **`02_category_revenue_and_margin_bar.png`:** Electronics generated the highest aggregate gross dollar revenue, while Apparel and Office Supplies generated the highest percentage gross margins (>48%).
3. **`03_regional_sales_distribution.png`:** Balanced regional footprint (East 30%, North 28%, South 22%, West 20%), mitigating geographic market concentration risk.
4. **`04_customer_segment_kpi_comparison.png`:** Corporate customers generated an Average Order Value (AOV) >$950 (bulk multi-unit orders), whereas Consumers accounted for 50% of volume at lower AOV.
5. **`05_top10_bestselling_products.png`:** Identifies primary flagship products: Smart 4K TVs, Ergonomic Executive Chairs, and Standing Desks dominate dollar revenue.
6. **`06_discount_elasticity_vs_profit_margin.png`:** Linear regression fit demonstrating that discounts greater than 15% cause margin degradation that volume increases cannot compensate for.
7. **`07_day_of_week_seasonality_heatmap.png`:** Highlights persistent weekend revenue clustering (Friday through Sunday averages exceed weekday averages by ~35%).
8. **`08_order_transaction_value_distribution.png`:** Log-transformed distribution confirming classic log-normal retail order distribution.
9. **`09_feature_correlation_heatmap.png`:** Highlights strong positive correlation between daily revenue and 7-day rolling sales ($r \approx 0.58$).
10. **`10_actual_vs_predicted_sales_chronological.png`:** Visualizes the out-of-sample forecast curve tracking actual daily retail demand over 141 days.
11. **`11_regression_model_comparison_metrics.png`:** Side-by-side comparison of RMSE and $R^2$ across algorithms.
12. **`12_regression_feature_importance.png`:** Random Forest Gini importance confirming `Revenue_Roll7_Mean`, `DayOfWeek`, `Revenue_Lag7`, and `Month` as dominant predictive features.

---

## 8. Strategic Business Recommendations

### 1. Dynamic Weekend Pricing & Inventory Buffers
- **Observation:** Average sales increase by **35% on Fridays, Saturdays, and Sundays**.
- **Action:** Shift distribution center picking and packing shifts to Thursday-Friday. Reduce promotional discounts on weekends when organic consumer demand is inelastic.

### 2. Guardrails on Promotional Discounting
- **Observation:** Discount rates exceeding 15% produce a steep drop in net profit margin without generating compensating sales elasticity.
- **Action:** Impose a maximum discount threshold of 15% for non-clearance merchandise. Reserve 20-25% discounts strictly for slow-moving seasonal inventory.

### 3. Corporate & B2B Expansion
- **Observation:** Corporate segment AOV is nearly double Consumer AOV due to multi-unit office furniture and printer orders.
- **Action:** Build a dedicated corporate portal offering tiered volume discounting and net-30 invoicing to drive high-margin B2B contracts.

### 4. Q4 Holiday Supply Chain Staging
- **Observation:** Over 28% of total annual retail revenue is concentrated in November and December.
- **Action:** Lock in procurement contracts and container logistics by August. Buffer top 5 flagship products (LED TVs, ergonomic chairs) in regional warehouses by mid-October.

---

## 9. Limitations & Future Roadmap

- **External Macro Factors:** Model currently relies solely on internal autoregressive transactions. Adding macroeconomic indicators (CPI, consumer sentiment, fuel prices) would enhance forecast precision.
- **Hierarchical SKU-Level Forecasting:** Transitioning from daily store-level forecasting to hierarchical SKU-store level forecasting (using LightGBM or Prophet) would improve inventory granularity.

---

## 10. How to Run the Project

### Execute Full Pipeline (Cleaning, Analysis, Training, Model Serialization):
```powershell
python Project_4_Retail_Analytics\src\train.py
```

### Run Standalone Scenario Prediction:
```powershell
python Project_4_Retail_Analytics\src\predict.py
```

### Launch Interactive Streamlit Retail Executive Suite:
```powershell
streamlit run Project_4_Retail_Analytics\app.py
```
