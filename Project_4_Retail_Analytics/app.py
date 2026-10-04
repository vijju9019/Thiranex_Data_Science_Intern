"""
Project 4: Real-World Data Science Project
Retail Sales Analytics and Prediction System
Streamlit Application: app.py
Author: Data Science Intern (Thiranex)
Description:
    Executive retail analytics dashboard and sales forecasting tool.
    Provides real-time KPI metrics, time-series revenue trends,
    category/regional drill-downs, and a What-If predictive simulation engine.
"""

import os
import sys
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure src modules can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

from predict import RetailSalesForecaster

# Page config
st.set_page_config(
    page_title="Retail Sales Analytics & Forecast Suite | Thiranex Internship",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .kpi-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #2563EB;
    }
    .kpi-lbl {
        font-size: 0.8rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
    }
    .pred-card {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border: 2px solid #3B82F6;
        border-radius: 12px;
        padding: 22px;
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_forecaster():
    return RetailSalesForecaster()


@st.cache_data
def load_retail_datasets():
    clean_path = os.path.join(current_dir, "data", "processed", "retail_sales_cleaned.csv")
    daily_path = os.path.join(current_dir, "data", "processed", "retail_daily_sales.csv")
    meta_path = os.path.join(current_dir, "models", "model_metadata.json")

    if not os.path.exists(clean_path) or not os.path.exists(daily_path):
        from src.train import run_full_retail_pipeline
        run_full_retail_pipeline()

    df_clean = pd.read_csv(clean_path)
    df_daily = pd.read_csv(daily_path)
    df_clean["Order Date"] = pd.to_datetime(df_clean["Order Date"])
    df_daily["Order Date"] = pd.to_datetime(df_daily["Order Date"])

    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)

    return df_clean, df_daily, meta


forecaster = get_forecaster()
df_clean, df_daily, metadata = load_retail_datasets()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1441986300917-64674bd600d8?auto=format&fit=crop&w=400&q=80", use_container_width=True)
    st.markdown("### 🛍️ Analytics Filters")
    st.caption("**Thiranex Internship** | Project 4: Retail Analytics")
    st.divider()

    # Region Filter
    all_regions = sorted(df_clean["Region"].unique())
    sel_regions = st.multiselect("Territory / Region", all_regions, default=all_regions)

    # Category Filter
    all_cats = sorted(df_clean["Product Category"].unique())
    sel_cats = st.multiselect("Product Category", all_cats, default=all_cats)

    # Segment Filter
    all_segs = ["All"] + sorted(df_clean["Customer Segment"].unique().tolist())
    sel_seg = st.selectbox("Customer Segment", all_segs, index=0)

    # Date Range Filter
    min_date = df_clean["Order Date"].min().date()
    max_date = df_clean["Order Date"].max().date()
    sel_dates = st.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

    st.divider()
    st.caption("Forecasting Engine: Linear Regression & Random Forest")

# Filter logic
filtered_df = df_clean.copy()
if sel_regions:
    filtered_df = filtered_df[filtered_df["Region"].isin(sel_regions)]
if sel_cats:
    filtered_df = filtered_df[filtered_df["Product Category"].isin(sel_cats)]
if sel_seg != "All":
    filtered_df = filtered_df[filtered_df["Customer Segment"] == sel_seg]
if len(sel_dates) == 2:
    filtered_df = filtered_df[(filtered_df["Order Date"].dt.date >= sel_dates[0]) &
                             (filtered_df["Order Date"].dt.date <= sel_dates[1])]

# ----------------- MAIN DISPLAY -----------------
st.markdown('<div class="main-header">🛍️ Retail Sales Analytics & Predictive System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Data Science Internship Project 4 | Business Intelligence, Time-Series Analytics & Demand Forecasting</div>', unsafe_allow_html=True)

# Executive KPI Row
k1, k2, k3, k4, k5 = st.columns(5)
total_rev = filtered_df["Revenue"].sum()
total_orders = len(filtered_df)
aov = total_rev / total_orders if total_orders > 0 else 0
total_profit = filtered_df["Profit"].sum()
margin_pct = (total_profit / total_rev * 100) if total_rev > 0 else 0
avg_disc = (filtered_df["Discount"].mean() * 100) if total_orders > 0 else 0

with k1:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-lbl">Total Net Revenue</div><div class="kpi-val">${total_rev / 1e6:.2f}M</div></div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-lbl">Total Orders Placed</div><div class="kpi-val">{total_orders:,}</div></div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-lbl">Average Order Value</div><div class="kpi-val">${aov:,.2f}</div></div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-lbl">Gross Profit Margin</div><div class="kpi-val" style="color:#10B981;">{margin_pct:.1f}%</div></div>""", unsafe_allow_html=True)
with k5:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-lbl">Avg Promo Discount</div><div class="kpi-val" style="color:#7C3AED;">{avg_disc:.1f}%</div></div>""", unsafe_allow_html=True)

st.write("")

# Main Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Revenue Trajectory & Trends",
    "🏢 Category & Territorial Drill-Down",
    "🏆 Product & Segment Leaderboard",
    "🔮 What-If Sales Prediction Tool",
    "📊 ML Regression Benchmarks"
])

# TAB 1: Trends
with tab1:
    st.subheader("Monthly Financial Trajectory")
    monthly = filtered_df.groupby("Year_Month").agg(
        Revenue=("Revenue", "sum"),
        Cost=("Cost", "sum"),
        Profit=("Profit", "sum")
    ).reset_index()

    fig_m, ax_m = plt.subplots(figsize=(12, 4.5))
    ax_m.plot(monthly["Year_Month"], monthly["Revenue"] / 1000, marker="o", color="#2563EB", linewidth=2.5, label="Revenue ($K)")
    ax_m.plot(monthly["Year_Month"], monthly["Profit"] / 1000, marker="s", color="#10B981", linewidth=2.2, label="Net Profit ($K)")
    ax_m.plot(monthly["Year_Month"], monthly["Cost"] / 1000, marker="^", color="#EF4444", linewidth=1.8, linestyle="--", label="Cost ($K)")
    ax_m.set_ylabel("Amount ($ in Thousands)")
    ax_m.set_xlabel("Month")
    plt.xticks(rotation=45)
    ax_m.legend(loc="upper left")
    st.pyplot(fig_m)
    plt.close(fig_m)

    st.markdown("##### Weekly Revenue Seasonality")
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_df = filtered_df.groupby("Day_of_Week")["Revenue"].mean().reindex(days_order).reset_index()

    fig_w, ax_w = plt.subplots(figsize=(10, 3.8))
    sns.barplot(data=dow_df, x="Day_of_Week", y="Revenue", palette="Blues_d", ax=ax_w, edgecolor="black")
    ax_w.set_ylabel("Average Daily Revenue ($)")
    ax_w.set_xlabel("Day of Week")
    for p in ax_w.patches:
        h = p.get_height()
        ax_w.annotate(f"${h:,.0f}", (p.get_x() + p.get_width() / 2., h / 2),
                      ha="center", va="center", color="white", weight="bold")
    st.pyplot(fig_w)
    plt.close(fig_w)

# TAB 2: Category & Regional
with tab2:
    st.subheader("Category Economics & Territorial Performance")
    c_col1, c_col2 = st.columns(2)

    with c_col1:
        st.markdown("##### Revenue & Margin by Category")
        cat_df = filtered_df.groupby("Product Category").agg(
            Revenue=("Revenue", "sum"),
            Profit=("Profit", "sum")
        ).reset_index()
        cat_df["Margin"] = (cat_df["Profit"] / cat_df["Revenue"]) * 100

        fig_c, ax_c = plt.subplots(figsize=(6, 4))
        sns.barplot(data=cat_df, x="Product Category", y="Revenue", palette="Blues_r", ax=ax_c, edgecolor="black")
        ax_c.set_ylabel("Total Revenue ($)")
        plt.xticks(rotation=15)
        st.pyplot(fig_c)
        plt.close(fig_c)

    with c_col2:
        st.markdown("##### Regional Territorial Contribution")
        reg_df = filtered_df.groupby("Region")["Revenue"].sum().reset_index()
        fig_r, ax_r = plt.subplots(figsize=(6, 4))
        ax_r.pie(reg_df["Revenue"], labels=reg_df["Region"], autopct="%1.1f%%",
                 colors=["#3B82F6", "#10B981", "#8B5CF6", "#F59E0B"], startangle=140,
                 wedgeprops=dict(width=0.4, edgecolor="black"))
        st.pyplot(fig_r)
        plt.close(fig_r)

# TAB 3: Products & Segments
with tab3:
    st.subheader("Product Performance & Customer Segmentation")
    p_col1, p_col2 = st.columns([1.3, 1])

    with p_col1:
        st.markdown("##### Top 10 Bestselling Products by Revenue")
        top_prod = filtered_df.groupby("Product Name").agg(
            Revenue=("Revenue", "sum"),
            Units=("Quantity", "sum")
        ).reset_index().sort_values(by="Revenue", ascending=False).head(10)

        fig_tp, ax_tp = plt.subplots(figsize=(7, 5))
        sns.barplot(data=top_prod, y="Product Name", x="Revenue", palette="Blues_r", ax=ax_tp, edgecolor="black")
        ax_tp.set_xlabel("Revenue ($)")
        st.pyplot(fig_tp)
        plt.close(fig_tp)

    with p_col2:
        st.markdown("##### Customer Segment Economics")
        seg_df = filtered_df.groupby("Customer Segment").agg(
            Revenue=("Revenue", "sum"),
            Orders=("Order ID", "count")
        ).reset_index()
        seg_df["AOV"] = (seg_df["Revenue"] / seg_df["Orders"]).round(2)
        st.dataframe(seg_df, use_container_width=True)

        st.info("""
        **Segment Takeaways:**
        - **Consumer:** Drives highest volume (50% of orders), lower AOV.
        - **Corporate:** Highest AOV due to multi-unit bulk orders (chairs, desks, printers).
        - **Home Office:** Stable, mid-tier margin and high repeat purchase frequency.
        """)

# TAB 4: Prediction Tool
with tab4:
    st.subheader("🔮 Interactive Sales Prediction & Scenario Simulator")
    st.markdown("""
    Test future business scenarios using the serialized machine learning model.
    Adjust promotional discount strategies, seasonal calendar factors, and recent rolling sales momentum.
    """)

    sim_col1, sim_col2 = st.columns(2)

    with sim_col1:
        st.markdown("##### 1. Forecast Calendar Parameters")
        target_month = st.selectbox("Target Month", list(range(1, 13)), index=10, format_func=lambda m: pd.to_datetime(f"2026-{m}-01").strftime("%B"))
        target_day = st.slider("Day of Month", 1, 31, 15)
        target_dow = st.selectbox("Day of Week", list(range(7)), index=5, format_func=lambda d: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][d])
        is_weekend = 1 if target_dow in [5, 6] else 0
        quarter = (target_month - 1) // 3 + 1
        day_of_year = (target_month - 1) * 30 + target_day

    with sim_col2:
        st.markdown("##### 2. Operational & Momentum Inputs")
        planned_discount = st.slider("Planned Average Promo Discount Rate", 0.0, 0.35, 0.10, 0.01)
        recent_7d_avg = st.slider("Recent 7-Day Rolling Revenue Average ($)", 1000.0, 15000.0, 5200.0, 100.0)
        recent_30d_avg = st.slider("Recent 30-Day Moving Average ($)", 1000.0, 15000.0, 4800.0, 100.0)
        avg_unit_price = st.slider("Expected Portfolio Unit Price ($)", 20.0, 500.0, 125.0, 5.0)

    # Build feature record
    sim_input = {
        "DayOfWeek": target_dow,
        "DayOfMonth": target_day,
        "Month": target_month,
        "Quarter": quarter,
        "IsWeekend": is_weekend,
        "DayOfYear": day_of_year,
        "Revenue_Lag1": recent_7d_avg * 1.02,
        "Revenue_Lag7": recent_7d_avg * 0.98,
        "Revenue_Lag14": recent_7d_avg * 0.95,
        "Revenue_Roll7_Mean": recent_7d_avg,
        "Revenue_Roll7_Std": recent_7d_avg * 0.20,
        "Revenue_Roll30_Mean": recent_30d_avg,
        "Avg_Discount": planned_discount,
        "Avg_Unit_Price": avg_unit_price
    }

    pred_res = forecaster.predict(sim_input)
    p_rev = pred_res["predicted_daily_revenue"]
    ci = pred_res["confidence_interval_95"]
    exp_profit = pred_res["expected_gross_profit_est"]

    st.markdown(f"""
    <div class="pred-card">
        <div style="font-size: 0.95rem; font-weight: 600; color: #1E40AF; text-transform: uppercase;">
            Predicted Daily Revenue Forecast
        </div>
        <div style="font-size: 2.8rem; font-weight: 800; color: #1D4ED8; margin: 4px 0;">
            ${p_rev:,.2f}
        </div>
        <div style="font-size: 1.05rem; color: #1E3A8A; font-weight: 500;">
            95% Confidence Interval: <strong>${ci['lower_bound']:,.2f}</strong> — <strong>${ci['upper_bound']:,.2f}</strong>
        </div>
        <div style="font-size: 0.9rem; color: #059669; font-weight: 600; margin-top: 6px;">
            Estimated Gross Profit (~38% margin): <strong>${exp_profit:,.2f}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

# TAB 5: Model Benchmarks
with tab5:
    st.subheader("Machine Learning Regression Benchmark & Evaluation")
    st.markdown("""
    Models were trained on an 80% chronological training window and evaluated on an unseen **future out-of-sample period (141 days)**.
    Chronological splitting prevents look-ahead bias and simulates true deployment forecasting conditions.
    """)

    all_mods = metadata.get("all_models_summary", [])
    if all_mods:
        df_bench = pd.DataFrame(all_mods)
        st.dataframe(df_bench.style.highlight_min(subset=["RMSE ($)", "MAE ($)"], color="#D1FAE5").highlight_max(subset=["R² Score"], color="#D1FAE5"), use_container_width=True)
    else:
        st.info("Run `python src/train.py` to regenerate the live regression benchmark table.")

    col_g1, col_g2 = st.columns(2)
    fig_dir = os.path.join(current_dir, "outputs", "figures")
    f_ts = os.path.join(fig_dir, "10_actual_vs_predicted_sales_chronological.png")
    f_bar = os.path.join(fig_dir, "11_regression_model_comparison_metrics.png")

    with col_g1:
        if os.path.exists(f_ts):
            st.image(f_ts, caption="Chronological Out-of-Sample Sales Forecast vs Actuals", use_container_width=True)
    with col_g2:
        if os.path.exists(f_bar):
            st.image(f_bar, caption="Regression Model Error Comparison (RMSE & R²)", use_container_width=True)
