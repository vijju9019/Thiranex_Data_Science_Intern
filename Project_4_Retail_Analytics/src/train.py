"""
Project 4: Real-World Data Science Project
Retail Sales Analytics and Prediction System
Module: train.py
Author: Data Science Intern (Thiranex)
Description:
    Trains, validates, benchmarks, and serializes retail sales regression models:
    - Linear Regression
    - Decision Tree Regressor
    - Random Forest Regressor
    Utilizes strict chronological train-test splitting to eliminate look-ahead bias.
"""

import os
import sys
import json
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Relative path support
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from cleaning import get_paths, run_cleaning_and_generation_pipeline
from feature_engineering import prepare_features_for_modeling, chronological_train_test_split
from analysis import generate_exploratory_visualizations, save_fig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
PRIMARY_COLOR = "#2563EB"


def train_and_evaluate_regressors(X_train, X_test, y_train, y_test, dates_test, feature_cols, figures_dir: str):
    """Train 3 regression algorithms, benchmark performance, and plot diagnostic figures."""
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, min_samples_split=4, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    }

    results = {}
    fitted_models = {}

    logger.info("--- RETAIL REGRESSION BENCHMARKING COMMENCED ---")

    for name, reg in models.items():
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        results[name] = {
            "MAE": round(mae, 2),
            "MSE": round(mse, 2),
            "RMSE": round(rmse, 2),
            "R2_Score": round(r2, 4),
            "y_pred": y_pred.tolist()
        }
        fitted_models[name] = reg

        logger.info(f"{name} -> MAE: ${mae:,.2f} | RMSE: ${rmse:,.2f} | R²: {r2:.4f}")

    # Build comparison table
    df_comp = pd.DataFrame([
        {"Algorithm": k, "MAE ($)": v["MAE"], "RMSE ($)": v["RMSE"], "R² Score": v["R2_Score"]}
        for k, v in results.items()
    ]).sort_values(by="R² Score", ascending=False).reset_index(drop=True)

    logger.info(f"\n================ REGRESSION BENCHMARK TABLE ================\n{df_comp.to_string(index=False)}")

    # 10. Actual vs Predicted Chronological Time Series (Best Model)
    best_algo = df_comp.iloc[0]["Algorithm"]
    best_pred = np.array(results[best_algo]["y_pred"])

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(dates_test, y_test.values, label="Actual Daily Sales ($)", color="#1E293B", linewidth=2.0, alpha=0.9)
    ax.plot(dates_test, best_pred, label=f"Predicted ({best_algo})", color="#2563EB", linewidth=2.0, linestyle="--")
    ax.fill_between(dates_test, y_test.values, best_pred, color="#93C5FD", alpha=0.3, label="Forecast Residual / Error")
    ax.set_title(f"Out-of-Sample Chronological Sales Forecast vs Actuals ({best_algo})", weight="bold", fontsize=13, pad=12)
    ax.set_ylabel("Daily Sales Revenue ($)", fontsize=11)
    ax.set_xlabel("Date (Chronological Test Horizon)", fontsize=11)
    plt.xticks(rotation=30)
    ax.legend(loc="upper left")
    save_fig(fig, "10_actual_vs_predicted_sales_chronological.png", figures_dir)

    # 11. Regression Metric Comparison Bar Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    sns.barplot(data=df_comp, x="Algorithm", y="RMSE ($)", palette="Reds_r", ax=ax1, edgecolor="black")
    ax1.set_title("Root Mean Squared Error (RMSE - Lower is Better)", weight="bold")
    ax1.set_ylabel("RMSE ($)")
    for p in ax1.patches:
        h = p.get_height()
        ax1.annotate(f"${h:,.1f}", (p.get_x() + p.get_width() / 2., h / 2),
                     ha="center", va="center", color="white", weight="bold")

    sns.barplot(data=df_comp, x="Algorithm", y="R² Score", palette="Greens_r", ax=ax2, edgecolor="black")
    ax2.set_title("Coefficient of Determination (R² - Higher is Better)", weight="bold")
    ax2.set_ylabel("R² Score")
    ax2.set_ylim(0, 1.05)
    for p in ax2.patches:
        h = p.get_height()
        ax2.annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2., h / 2),
                     ha="center", va="center", color="white", weight="bold")
    save_fig(fig, "11_regression_model_comparison_metrics.png", figures_dir)

    # 12. Feature Importance (Random Forest Regressor)
    rf_reg = fitted_models["Random Forest"]
    feat_imp = pd.DataFrame({"Feature": feature_cols, "Importance": rf_reg.feature_importances_})
    feat_imp = feat_imp.sort_values(by="Importance", ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=feat_imp, x="Importance", y="Feature", palette="Blues_r", ax=ax, edgecolor="black")
    ax.set_title("Top 10 Drivers of Daily Retail Revenue (Random Forest)", weight="bold", fontsize=13, pad=12)
    ax.set_xlabel("Relative Predictive Importance")
    save_fig(fig, "12_regression_feature_importance.png", figures_dir)

    return results, fitted_models, df_comp


def save_best_regression_model(fitted_models: dict, df_comp: pd.DataFrame, feature_cols: list, results: dict):
    """Serialize best regression pipeline and write comprehensive metadata."""
    paths = get_paths()
    os.makedirs(paths["models_dir"], exist_ok=True)

    best_name = df_comp.iloc[0]["Algorithm"]
    best_model = fitted_models[best_name]

    model_path = os.path.join(paths["models_dir"], "best_sales_model.joblib")
    joblib.dump(best_model, model_path)
    logger.info(f"Serialized best regression model ('{best_name}') to: {model_path}")

    meta_path = os.path.join(paths["models_dir"], "model_metadata.json")
    metadata = {
        "best_model_name": best_name,
        "selected_metrics": {
            "MAE": results[best_name]["MAE"],
            "RMSE": results[best_name]["RMSE"],
            "R2_Score": results[best_name]["R2_Score"]
        },
        "all_models_summary": df_comp.to_dict(orient="records"),
        "feature_names": feature_cols,
        "target": "Daily_Revenue",
        "split_method": "Chronological 80/20 Out-of-Sample",
        "horizon": "Daily Retail Revenue Forecasting"
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Model metadata serialized to: {meta_path}")

    return best_name


def run_full_retail_pipeline():
    """Execute end-to-end Project 4 pipeline."""
    paths = get_paths()

    # Step 1: Cleaning & Generation
    df_clean, df_daily = run_cleaning_and_generation_pipeline()

    # Step 2: Exploratory Visualizations (Figures 1-9)
    generate_exploratory_visualizations(df_clean, df_daily, paths["figures_dir"])

    # Step 3: Feature Engineering & Chronological Split
    X, y, dates, feature_cols = prepare_features_for_modeling(df_daily)
    X_train, X_test, y_train, y_test, dates_train, dates_test = chronological_train_test_split(X, y, dates, train_ratio=0.80)

    # Step 4: Model Training, Evaluation & Figures 10-12
    results, fitted_models, df_comp = train_and_evaluate_regressors(
        X_train, X_test, y_train, y_test, dates_test, feature_cols, paths["figures_dir"]
    )

    # Step 5: Save Model & Metadata
    best_name = save_best_regression_model(fitted_models, df_comp, feature_cols, results)
    logger.info(f"Project 4 Retail Pipeline Completed! Best Model: {best_name}")


if __name__ == "__main__":
    run_full_retail_pipeline()
