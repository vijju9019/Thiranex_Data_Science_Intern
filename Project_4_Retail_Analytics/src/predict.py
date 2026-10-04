"""
Project 4: Real-World Data Science Project
Retail Sales Analytics and Prediction System
Module: predict.py
Author: Data Science Intern (Thiranex)
Description:
    Standalone prediction module and CLI forecasting engine for retail daily revenue.
    Loads the serialized model and outputs expected daily sales revenue and margin estimates.
"""

import os
import sys
import json
import logging
import joblib
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def get_paths():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return {
        "model": os.path.join(base_dir, "models", "best_sales_model.joblib"),
        "metadata": os.path.join(base_dir, "models", "model_metadata.json"),
        "daily_csv": os.path.join(base_dir, "data", "processed", "retail_daily_sales.csv")
    }


class RetailSalesForecaster:
    """Production inference engine for daily retail sales prediction."""

    def __init__(self, model_path: str = None, metadata_path: str = None):
        paths = get_paths()
        self.model_path = model_path or paths["model"]
        self.metadata_path = metadata_path or paths["metadata"]

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model artifact not found at {self.model_path}. Execute src/train.py first.")

        logger.info(f"Loading trained sales model from: {self.model_path}")
        self.model = joblib.load(self.model_path)

        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                self.metadata = json.load(f)
            self.feature_names = self.metadata.get("feature_names", [])
            self.best_model_name = self.metadata.get("best_model_name", "Regression Model")
            self.rmse = self.metadata.get("selected_metrics", {}).get("RMSE", 500.0)
        else:
            self.feature_names = []
            self.best_model_name = "Trained Regressor"
            self.rmse = 500.0

    def predict(self, input_data) -> dict:
        """
        Predict daily sales revenue given feature dictionary or DataFrame.
        """
        if isinstance(input_data, dict):
            df_in = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            df_in = input_data.copy()
        else:
            raise TypeError("Input must be a dict or pandas DataFrame.")

        if self.feature_names:
            missing_cols = [c for c in self.feature_names if c not in df_in.columns]
            if missing_cols:
                raise ValueError(f"Input is missing required features: {missing_cols}")
            df_in = df_in[self.feature_names]

        preds = self.model.predict(df_in)

        results = []
        for p in preds:
            pred_val = float(max(0.0, p))
            # 95% Confidence Interval based on validation RMSE
            ci_lower = max(0.0, pred_val - 1.96 * self.rmse)
            ci_upper = pred_val + 1.96 * self.rmse

            results.append({
                "predicted_daily_revenue": round(pred_val, 2),
                "confidence_interval_95": {
                    "lower_bound": round(ci_lower, 2),
                    "upper_bound": round(ci_upper, 2)
                },
                "model_used": self.best_model_name,
                "expected_gross_profit_est": round(pred_val * 0.38, 2)  # Benchmark ~38% historical margin
            })

        return results[0] if len(results) == 1 else results


def run_cli_test():
    """Execute prediction on recent real historical scenario."""
    paths = get_paths()
    forecaster = RetailSalesForecaster()

    if os.path.exists(paths["daily_csv"]):
        df_daily = pd.read_csv(paths["daily_csv"])
        feature_cols = forecaster.feature_names

        # Take last available day as input scenario
        sample_row = df_daily.iloc[-1][feature_cols].to_dict()
        actual_rev = df_daily.iloc[-1]["Daily_Revenue"]

        res = forecaster.predict(sample_row)

        print("\n================ RETAIL REVENUE FORECAST TEST ================")
        print(f"Model Employed:           {res['model_used']}")
        print(f"Actual Daily Revenue:     ${actual_rev:,.2f}")
        print(f"Predicted Daily Revenue:  ${res['predicted_daily_revenue']:,.2f}")
        print(f"95% Confidence Interval:  ${res['confidence_interval_95']['lower_bound']:,.2f} - ${res['confidence_interval_95']['upper_bound']:,.2f}")
        print(f"Expected Gross Profit:    ${res['expected_gross_profit_est']:,.2f}")
        print("=============================================================\n")
    else:
        print("Daily sales data not found. Run cleaning.py and train.py first.")


if __name__ == "__main__":
    run_cli_test()
