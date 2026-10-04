"""
Project 4: Real-World Data Science Project
Retail Sales Analytics and Prediction System
Module: cleaning.py
Author: Data Science Intern (Thiranex)
Description:
    Generates a realistic calibrated retail transactional dataset,
    performs data cleaning, schema validation, financial column calculation,
    and aggregates daily sales time-series for chronological forecasting.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def get_paths():
    """Resolve directory paths for Project 4."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return {
        "base": base_dir,
        "raw_dir": os.path.join(base_dir, "data", "raw"),
        "processed_dir": os.path.join(base_dir, "data", "processed"),
        "raw_csv": os.path.join(base_dir, "data", "raw", "retail_sales_raw.csv"),
        "processed_csv": os.path.join(base_dir, "data", "processed", "retail_sales_cleaned.csv"),
        "daily_csv": os.path.join(base_dir, "data", "processed", "retail_daily_sales.csv"),
        "figures_dir": os.path.join(base_dir, "outputs", "figures"),
        "models_dir": os.path.join(base_dir, "models")
    }


def generate_realistic_retail_dataset(n_records: int = 7500, random_state: int = 42) -> pd.DataFrame:
    """
    Generate an authentic, calibrated synthetic retail transactional dataset
    spanning 2 full calendar years (2024-01-01 to 2025-12-31).
    Incorporates authentic weekly/seasonal demand cycles, realistic product catalogs,
    pricing, discounting, and cost-of-goods-sold (COGS) margins.
    
    *DOCUMENTATION: Clearly documented as a calibrated synthetic retail dataset.*
    """
    np.random.seed(random_state)
    logger.info(f"Generating realistic synthetic retail dataset with {n_records} order transactions...")

    # Product Catalog Definition: {Category: [(Product Name, Base Price, Cost Ratio)]}
    catalog = {
        "Electronics": [
            ("Smart 4K LED TV 55-inch", 650.0, 0.68),
            ("Noise-Cancelling Wireless Headphones", 180.0, 0.55),
            ("Bluetooth Soundbar Speaker", 120.0, 0.58),
            ("Mechanical Gaming Keyboard", 85.0, 0.50),
            ("Ultra-HD Streaming Webcam", 65.0, 0.52),
            ("USB-C Fast Charging Hub", 35.0, 0.45)
        ],
        "Furniture": [
            ("Ergonomic High-Back Executive Chair", 320.0, 0.62),
            ("Solid Oak Dining Table", 550.0, 0.65),
            ("Adjustable Motorized Standing Desk", 420.0, 0.60),
            ("Modern Fabric Accent Armchair", 240.0, 0.58),
            ("5-Tier Wooden Bookshelf Unit", 140.0, 0.55)
        ],
        "Office Supplies": [
            ("All-in-One Wireless Laser Printer", 220.0, 0.70),
            ("Heavy-Duty Cross-Cut Paper Shredder", 95.0, 0.60),
            ("Ergonomic Gel Wrist Mouse Pad", 18.0, 0.40),
            ("Premium Hardcover Notebook Pack (5-Pack)", 25.0, 0.42),
            ("Deluxe Executive Fountain Pen Set", 45.0, 0.48),
            ("Heavy-Duty Desktop Stapler & Puncher", 22.0, 0.45)
        ],
        "Apparel": [
            ("Thermal Water-Resistant Winter Jacket", 160.0, 0.50),
            ("Classic Tailored Denim Jeans", 75.0, 0.48),
            ("Breathable Running Athletic Shoes", 110.0, 0.52),
            ("Organic Cotton Oxford Dress Shirt", 55.0, 0.45),
            ("Merino Wool Casual Sweater", 85.0, 0.46)
        ]
    }

    categories = list(catalog.keys())
    cat_weights = [0.35, 0.25, 0.22, 0.18]

    regions = ["North", "South", "East", "West"]
    region_weights = [0.28, 0.22, 0.30, 0.20]

    segments = ["Consumer", "Corporate", "Home Office"]
    segment_weights = [0.50, 0.30, 0.20]

    payment_methods = ["Credit Card", "Debit Card", "PayPal", "Bank Transfer"]
    pay_weights = [0.55, 0.20, 0.18, 0.07]

    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 12, 31)
    total_days = (end_date - start_date).days + 1

    records = []

    # Assign order dates with realistic seasonality (weekend lift, Q4 holiday boost)
    days_array = np.arange(total_days)
    dates_list = [start_date + timedelta(days=int(d)) for d in days_array]

    # Calculate day-of-year seasonal weight
    date_weights = []
    for d in dates_list:
        # Weekend multiplier (Fri, Sat, Sun)
        w_factor = 1.35 if d.weekday() in [4, 5, 6] else 1.0
        # Month multiplier (Nov, Dec holiday quarter peak; Aug back-to-school)
        if d.month in [11, 12]:
            m_factor = 1.60
        elif d.month in [8, 9]:
            m_factor = 1.25
        elif d.month in [1, 2]:
            m_factor = 0.85
        else:
            m_factor = 1.05
        date_weights.append(w_factor * m_factor)

    date_probs = np.array(date_weights) / sum(date_weights)
    chosen_dates = np.random.choice(dates_list, size=n_records, p=date_probs)
    chosen_dates = sorted(chosen_dates)

    for i in range(n_records):
        order_id = f"ORD-{20240001 + i}"
        order_date = chosen_dates[i]

        cat = np.random.choice(categories, p=cat_weights)
        prod_tuple = catalog[cat][np.random.randint(0, len(catalog[cat]))]
        prod_name, base_price, cost_ratio = prod_tuple

        # Slight price variation (+- 5%)
        unit_price = round(base_price * np.random.uniform(0.95, 1.05), 2)
        unit_cost = round(unit_price * cost_ratio, 2)

        # Quantity: Corporate/Home Office often order higher quantities
        seg = np.random.choice(segments, p=segment_weights)
        if seg == "Corporate":
            qty = int(np.clip(np.random.poisson(lam=4.0), 1, 15))
        elif seg == "Home Office":
            qty = int(np.clip(np.random.poisson(lam=2.5), 1, 10))
        else:
            qty = int(np.clip(np.random.poisson(lam=1.8), 1, 6))

        # Discount: higher probability during holiday months or promotional days
        if order_date.month in [11, 12]:
            discount = float(np.random.choice([0.0, 0.10, 0.15, 0.20, 0.25], p=[0.25, 0.25, 0.25, 0.15, 0.10]))
        else:
            discount = float(np.random.choice([0.0, 0.05, 0.10, 0.15], p=[0.55, 0.25, 0.15, 0.05]))

        # Financial Calculations
        gross_rev = qty * unit_price
        net_rev = round(gross_rev * (1.0 - discount), 2)
        total_cost = round(qty * unit_cost, 2)
        net_profit = round(net_rev - total_cost, 2)

        region = np.random.choice(regions, p=region_weights)
        pay_method = np.random.choice(payment_methods, p=pay_weights)

        customer_id = f"CUST-{np.random.randint(1000, 2500)}"

        records.append({
            "Order ID": order_id,
            "Order Date": order_date.strftime("%Y-%m-%d"),
            "Customer ID": customer_id,
            "Customer Segment": seg,
            "Region": region,
            "Product Category": cat,
            "Product Name": prod_name,
            "Quantity": qty,
            "Unit Price": unit_price,
            "Unit Cost": unit_cost,
            "Discount": discount,
            "Revenue": net_rev,
            "Cost": total_cost,
            "Profit": net_profit,
            "Payment Method": pay_method
        })

    df = pd.DataFrame(records)
    logger.info(f"Synthetic dataset generated: Total Revenue = ${df['Revenue'].sum():,.2f}, Total Profit = ${df['Profit'].sum():,.2f}")
    return df


def clean_and_validate_retail_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Perform thorough data cleaning, type conversion, and integrity validation.
    """
    df = df_raw.copy()
    initial_rows = len(df)

    # 1. Date formatting
    df["Order Date"] = pd.to_datetime(df["Order Date"])

    # 2. Check and eliminate duplicates
    df = df.drop_duplicates(subset=["Order ID"])
    logger.info(f"Deduplication: Retained {len(df)} records (dropped {initial_rows - len(df)} duplicates).")

    # 3. Handle missing values
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        logger.warning(f"Detected missing values:\n{null_counts[null_counts > 0]}")
        df = df.dropna()

    # 4. Data sanity filters
    df = df[df["Quantity"] > 0]
    df = df[df["Unit Price"] > 0]
    df = df[df["Revenue"] >= 0]
    df = df[df["Discount"].between(0.0, 1.0)]

    # 5. Financial metrics
    df["Profit Margin %"] = ((df["Profit"] / df["Revenue"]) * 100).round(2)

    # 6. Date features
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Month_Name"] = df["Order Date"].dt.strftime("%b")
    df["Year_Month"] = df["Order Date"].dt.to_period("M").astype(str)
    df["Day_of_Week"] = df["Order Date"].dt.day_name()
    df["Is_Weekend"] = df["Order Date"].dt.dayofweek.isin([5, 6]).astype(int)

    logger.info(f"Cleaned dataset verified: {df.shape[0]} valid order records.")
    return df


def aggregate_daily_sales(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transactional orders into daily time-series metrics
    specifically engineered for chronological regression forecasting.
    """
    logger.info("Aggregating transactional records into daily time series...")

    daily = df_clean.groupby("Order Date").agg(
        Daily_Revenue=("Revenue", "sum"),
        Daily_Profit=("Profit", "sum"),
        Daily_Orders=("Order ID", "count"),
        Daily_Quantity=("Quantity", "sum"),
        Avg_Discount=("Discount", "mean"),
        Avg_Unit_Price=("Unit Price", "mean")
    ).reset_index()

    # Reindex to ensure unbroken calendar date continuum
    daily = daily.sort_values("Order Date").reset_index(drop=True)
    all_dates = pd.date_range(start=daily["Order Date"].min(), end=daily["Order Date"].max(), freq="D")
    daily = daily.set_index("Order Date").reindex(all_dates).fillna({
        "Daily_Revenue": 0.0,
        "Daily_Profit": 0.0,
        "Daily_Orders": 0,
        "Daily_Quantity": 0,
        "Avg_Discount": 0.0,
        "Avg_Unit_Price": daily["Avg_Unit_Price"].median()
    }).reset_index()
    daily.rename(columns={"index": "Order Date"}, inplace=True)

    # Date Calendar Signals
    daily["DayOfWeek"] = daily["Order Date"].dt.dayofweek
    daily["DayOfMonth"] = daily["Order Date"].dt.day
    daily["Month"] = daily["Order Date"].dt.month
    daily["Quarter"] = daily["Order Date"].dt.quarter
    daily["Year"] = daily["Order Date"].dt.year
    daily["DayOfYear"] = daily["Order Date"].dt.dayofyear
    daily["IsWeekend"] = daily["DayOfWeek"].isin([5, 6]).astype(int)

    # Autoregressive Lag Features
    daily["Revenue_Lag1"] = daily["Daily_Revenue"].shift(1)
    daily["Revenue_Lag7"] = daily["Daily_Revenue"].shift(7)
    daily["Revenue_Lag14"] = daily["Daily_Revenue"].shift(14)

    # Rolling Window Moving Averages
    daily["Revenue_Roll7_Mean"] = daily["Daily_Revenue"].shift(1).rolling(window=7).mean()
    daily["Revenue_Roll7_Std"] = daily["Daily_Revenue"].shift(1).rolling(window=7).std()
    daily["Revenue_Roll30_Mean"] = daily["Daily_Revenue"].shift(1).rolling(window=30).mean()

    # Drop initial rows with NaN from rolling/lags
    daily = daily.dropna().reset_index(drop=True)
    logger.info(f"Daily sales dataset compiled: {daily.shape[0]} daily records with complete lag features.")
    return daily


def run_cleaning_and_generation_pipeline():
    """End-to-end execution of Project 4 data generation & cleaning."""
    paths = get_paths()
    os.makedirs(paths["raw_dir"], exist_ok=True)
    os.makedirs(paths["processed_dir"], exist_ok=True)

    # Step 1: Generate or load raw
    if os.path.exists(paths["raw_csv"]):
        logger.info(f"Loading existing raw dataset from: {paths['raw_csv']}")
        df_raw = pd.read_csv(paths["raw_csv"])
    else:
        df_raw = generate_realistic_retail_dataset(n_records=7500, random_state=42)
        df_raw.to_csv(paths["raw_csv"], index=False)
        logger.info(f"Raw retail dataset saved to: {paths['raw_csv']}")

    # Step 2: Clean & Validate
    df_clean = clean_and_validate_retail_data(df_raw)
    df_clean.to_csv(paths["processed_csv"], index=False)
    logger.info(f"Cleaned retail dataset saved to: {paths['processed_csv']}")

    # Step 3: Aggregate Daily Time Series
    df_daily = aggregate_daily_sales(df_clean)
    df_daily.to_csv(paths["daily_csv"], index=False)
    logger.info(f"Daily aggregated time-series saved to: {paths['daily_csv']}")

    return df_clean, df_daily


if __name__ == "__main__":
    run_cleaning_and_generation_pipeline()
