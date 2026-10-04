"""
Project 4: Real-World Data Science Project
Retail Sales Analytics and Prediction System
Module: feature_engineering.py
Author: Data Science Intern (Thiranex)
Description:
    Chronological feature engineering, calendar transformations,
    rolling autoregressive metrics, and chronological train-test splitting
    to prevent look-ahead data leakage in sales forecasting.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def prepare_features_for_modeling(df_daily: pd.DataFrame):
    """
    Select engineered features for supervised regression modeling.
    
    Target:
        Daily_Revenue (float)
        
    Predictors:
        - Calendar indicators: DayOfWeek, DayOfMonth, Month, Quarter, IsWeekend, DayOfYear
        - Autoregressive lags: Revenue_Lag1, Revenue_Lag7, Revenue_Lag14
        - Moving trend windows: Revenue_Roll7_Mean, Revenue_Roll7_Std, Revenue_Roll30_Mean
        - Operational drivers: Avg_Discount, Avg_Unit_Price
    """
    feature_cols = [
        "DayOfWeek", "DayOfMonth", "Month", "Quarter", "IsWeekend", "DayOfYear",
        "Revenue_Lag1", "Revenue_Lag7", "Revenue_Lag14",
        "Revenue_Roll7_Mean", "Revenue_Roll7_Std", "Revenue_Roll30_Mean",
        "Avg_Discount", "Avg_Unit_Price"
    ]

    target_col = "Daily_Revenue"

    # Ensure sorting chronologically
    df_sorted = df_daily.sort_values("Order Date").reset_index(drop=True)
    X = df_sorted[feature_cols]
    y = df_sorted[target_col]
    dates = df_sorted["Order Date"]

    return X, y, dates, feature_cols


def chronological_train_test_split(X: pd.DataFrame, y: pd.Series, dates: pd.Series, train_ratio: float = 0.80):
    """
    Perform strict chronological train-test split.
    DO NOT shuffle. The earliest 80% of dates form the training set;
    the final 20% chronologically form the unseen out-of-sample forecast set.
    """
    n_samples = len(X)
    split_idx = int(n_samples * train_ratio)

    X_train = X.iloc[:split_idx].copy()
    y_train = y.iloc[:split_idx].copy()
    dates_train = dates.iloc[:split_idx]

    X_test = X.iloc[split_idx:].copy()
    y_test = y.iloc[split_idx:].copy()
    dates_test = dates.iloc[split_idx:]

    logger.info(
        f"Chronological Split: Total={n_samples} days. "
        f"Train={len(X_train)} days ({dates_train.min().strftime('%Y-%m-%d')} to {dates_train.max().strftime('%Y-%m-%d')}) | "
        f"Test={len(X_test)} days ({dates_test.min().strftime('%Y-%m-%d')} to {dates_test.max().strftime('%Y-%m-%d')})"
    )

    return X_train, X_test, y_train, y_test, dates_train, dates_test
