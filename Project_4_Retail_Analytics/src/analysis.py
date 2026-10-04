"""
Project 4: Real-World Data Science Project
Retail Sales Analytics and Prediction System
Module: analysis.py
Author: Data Science Intern (Thiranex)
Description:
    Exploratory and diagnostic retail analytics engine.
    Analyzes revenue, margins, customer segments, regional performance,
    discount elasticity, and generates 12 publication-grade figures.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Apply publication-grade styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.family": "sans-serif",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10
})

PRIMARY_COLOR = "#2563EB"
SECONDARY_COLOR = "#7C3AED"


def get_paths():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return {
        "processed_csv": os.path.join(base_dir, "data", "processed", "retail_sales_cleaned.csv"),
        "daily_csv": os.path.join(base_dir, "data", "processed", "retail_daily_sales.csv"),
        "figures_dir": os.path.join(base_dir, "outputs", "figures")
    }


def save_fig(fig, filename: str, figures_dir: str):
    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, filename)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure: {out_path}")


def generate_exploratory_visualizations(df_clean: pd.DataFrame, df_daily: pd.DataFrame, figures_dir: str):
    """Generate figures 1 through 9 for exploratory retail analysis."""
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Monthly Revenue & Profit Trend
    monthly = df_clean.groupby("Year_Month").agg(
        Revenue=("Revenue", "sum"),
        Cost=("Cost", "sum"),
        Profit=("Profit", "sum")
    ).reset_index()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly["Year_Month"], monthly["Revenue"] / 1000, marker="o", color=PRIMARY_COLOR, linewidth=2.5, label="Revenue ($K)")
    ax.plot(monthly["Year_Month"], monthly["Profit"] / 1000, marker="s", color="#10B981", linewidth=2.2, label="Net Profit ($K)")
    ax.plot(monthly["Year_Month"], monthly["Cost"] / 1000, marker="^", color="#EF4444", linewidth=1.8, linestyle="--", label="COGS / Cost ($K)")
    ax.set_title("Monthly Retail Revenue, Cost, and Profit Trajectory (2024 - 2025)", fontsize=13, weight="bold", pad=12)
    ax.set_ylabel("Amount ($ in Thousands)", fontsize=11)
    ax.set_xlabel("Calendar Month", fontsize=11)
    plt.xticks(rotation=45)
    ax.legend(loc="upper left")
    save_fig(fig, "01_monthly_revenue_and_profit_trend.png", figures_dir)

    # 2. Category-wise Revenue and Profit Margin %
    cat_summary = df_clean.groupby("Product Category").agg(
        Total_Revenue=("Revenue", "sum"),
        Total_Profit=("Profit", "sum")
    ).reset_index()
    cat_summary["Margin_%"] = (cat_summary["Total_Profit"] / cat_summary["Total_Revenue"]) * 100
    cat_summary = cat_summary.sort_values(by="Total_Revenue", ascending=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=cat_summary, x="Product Category", y="Total_Revenue", palette="Blues_r", ax=ax1, edgecolor="black")
    ax1.set_title("Total Revenue by Product Category", weight="bold")
    ax1.set_ylabel("Revenue ($)")
    for p in ax1.patches:
        h = p.get_height()
        ax1.annotate(f"${h / 1000:,.1f}K", (p.get_x() + p.get_width() / 2., h / 2),
                     ha="center", va="center", color="white", weight="bold")

    sns.barplot(data=cat_summary, x="Product Category", y="Margin_%", palette="Purples_r", ax=ax2, edgecolor="black")
    ax2.set_title("Gross Profit Margin % by Category", weight="bold")
    ax2.set_ylabel("Profit Margin (%)")
    for p in ax2.patches:
        h = p.get_height()
        ax2.annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2., h / 2),
                     ha="center", va="center", color="white", weight="bold")
    save_fig(fig, "02_category_revenue_and_margin_bar.png", figures_dir)

    # 3. Regional Sales Distribution
    reg_summary = df_clean.groupby("Region").agg(Revenue=("Revenue", "sum"), Orders=("Order ID", "count")).reset_index()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    sns.barplot(data=reg_summary, x="Region", y="Revenue", palette="viridis", ax=ax1, edgecolor="black")
    ax1.set_title("Revenue by Geographic Territory", weight="bold")
    ax1.set_ylabel("Revenue ($)")
    for p in ax1.patches:
        h = p.get_height()
        ax1.annotate(f"${h / 1000:,.1f}K", (p.get_x() + p.get_width() / 2., h / 2),
                     ha="center", va="center", color="white", weight="bold")

    ax2.pie(reg_summary["Revenue"], labels=reg_summary["Region"], autopct="%1.1f%%",
            colors=["#3B82F6", "#10B981", "#8B5CF6", "#F59E0B"], startangle=140,
            wedgeprops=dict(width=0.4, edgecolor="black"))
    ax2.set_title("Regional Revenue Contribution %", weight="bold")
    save_fig(fig, "03_regional_sales_distribution.png", figures_dir)

    # 4. Customer Segment Performance
    seg_summary = df_clean.groupby("Customer Segment").agg(
        Revenue=("Revenue", "sum"),
        Orders=("Order ID", "count"),
        Units=("Quantity", "sum")
    ).reset_index()
    seg_summary["AOV"] = seg_summary["Revenue"] / seg_summary["Orders"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    sns.barplot(data=seg_summary, x="Customer Segment", y="Revenue", palette="Blues_d", ax=ax1, edgecolor="black")
    ax1.set_title("Total Revenue by Customer Segment", weight="bold")
    ax1.set_ylabel("Revenue ($)")
    for p in ax1.patches:
        h = p.get_height()
        ax1.annotate(f"${h / 1000:,.1f}K", (p.get_x() + p.get_width() / 2., h / 2),
                     ha="center", va="center", color="white", weight="bold")

    sns.barplot(data=seg_summary, x="Customer Segment", y="AOV", palette="Purples_d", ax=ax2, edgecolor="black")
    ax2.set_title("Average Order Value (AOV) by Segment", weight="bold")
    ax2.set_ylabel("AOV ($ per Order)")
    for p in ax2.patches:
        h = p.get_height()
        ax2.annotate(f"${h:,.2f}", (p.get_x() + p.get_width() / 2., h / 2),
                     ha="center", va="center", color="white", weight="bold")
    save_fig(fig, "04_customer_segment_kpi_comparison.png", figures_dir)

    # 5. Top 10 Bestselling Products
    top_prod = df_clean.groupby("Product Name").agg(
        Revenue=("Revenue", "sum"),
        Units=("Quantity", "sum")
    ).reset_index().sort_values(by="Revenue", ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top_prod, y="Product Name", x="Revenue", palette="Blues_r", ax=ax, edgecolor="black")
    ax.set_title("Top 10 Revenue-Generating Retail Products", weight="bold", fontsize=13, pad=12)
    ax.set_xlabel("Total Sales Revenue ($)")
    for p in ax.patches:
        w = p.get_width()
        ax.annotate(f" ${w:,.0f}", (w, p.get_y() + p.get_height() / 2.),
                    ha="left", va="center", fontsize=9, weight="bold")
    save_fig(fig, "05_top10_bestselling_products.png", figures_dir)

    # 6. Discount Elasticity vs Profit Margin
    sample_sub = df_clean.sample(min(1500, len(df_clean)), random_state=42)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.regplot(data=sample_sub, x="Discount", y="Profit Margin %", scatter_kws={"alpha": 0.3, "color": PRIMARY_COLOR},
                line_kws={"color": "#DC2626", "linewidth": 2.5}, ax=ax)
    ax.set_title("Discount Depth Elasticity vs Net Profit Margin %", weight="bold", fontsize=13, pad=12)
    ax.set_xlabel("Discount Rate (e.g., 0.15 = 15%)")
    ax.set_ylabel("Profit Margin (%)")
    save_fig(fig, "06_discount_elasticity_vs_profit_margin.png", figures_dir)

    # 7. Day of Week Seasonality Heatmap
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot_dow = df_clean.pivot_table(index="Day_of_Week", columns="Month_Name", values="Revenue", aggfunc="mean")
    pivot_dow = pivot_dow.reindex(days_order)
    months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot_dow = pivot_dow[[m for m in months_order if m in pivot_dow.columns]]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.heatmap(pivot_dow, cmap="YlGnBu", annot=True, fmt=",.0f", cbar_kws={"label": "Avg Order Revenue ($)"}, ax=ax)
    ax.set_title("Seasonal Heatmap: Average Order Revenue by Day of Week vs Month", weight="bold", fontsize=13, pad=12)
    save_fig(fig, "07_day_of_week_seasonality_heatmap.png", figures_dir)

    # 8. Order Value Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(df_clean["Revenue"], kde=True, color="#2563EB", bins=30, ax=ax1)
    ax1.set_title("Order Value Distribution (Raw)", weight="bold")
    ax1.set_xlabel("Revenue ($)")

    sns.histplot(np.log1p(df_clean["Revenue"]), kde=True, color="#7C3AED", bins=30, ax=ax2)
    ax2.set_title("Log-Transformed Order Value Distribution", weight="bold")
    ax2.set_xlabel("Log(Revenue + 1)")
    save_fig(fig, "08_order_transaction_value_distribution.png", figures_dir)

    # 9. Correlation Matrix
    corr_cols = ["Revenue", "Quantity", "Unit Price", "Discount", "Profit", "Profit Margin %"]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_clean[corr_cols].corr(), annot=True, fmt=".2f", cmap="vlag", vmin=-0.8, vmax=0.8, ax=ax)
    ax.set_title("Correlation Matrix of Financial & Transactional Variables", weight="bold", fontsize=13, pad=12)
    save_fig(fig, "09_feature_correlation_heatmap.png", figures_dir)

    logger.info(f"Generated exploratory figures 1-9 in: {figures_dir}")
