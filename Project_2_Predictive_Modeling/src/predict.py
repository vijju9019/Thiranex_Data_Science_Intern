"""
Project 2: Predictive Modeling Using Machine Learning
Module: predict.py
Author: Data Science Intern (Thiranex)
Description:
    Standalone inference module and CLI prediction script for breast tumor classification.
    Loads the serialized joblib pipeline and outputs class label, probabilities, and confidence.
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
    """Resolve paths to model artifacts."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return {
        "model": os.path.join(base_dir, "models", "best_model.joblib"),
        "metadata": os.path.join(base_dir, "models", "model_metadata.json"),
        "raw_csv": os.path.join(base_dir, "data", "breast_cancer_raw.csv")
    }


class BreastCancerPredictor:
    """Production-grade inference handler for Breast Cancer prediction."""

    def __init__(self, model_path: str = None, metadata_path: str = None):
        paths = get_paths()
        self.model_path = model_path or paths["model"]
        self.metadata_path = metadata_path or paths["metadata"]

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. Please execute src/train.py first."
            )

        logger.info(f"Loading trained pipeline from: {self.model_path}")
        self.pipeline = joblib.load(self.model_path)

        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                self.metadata = json.load(f)
            self.feature_names = self.metadata.get("feature_names", [])
            self.best_model_name = self.metadata.get("best_model_name", "Classifier")
        else:
            self.metadata = {}
            self.feature_names = []
            self.best_model_name = "Trained Classifier"

    def predict(self, input_data) -> dict:
        """
        Execute prediction on single record or batch.
        
        Input:
            input_data: dict of {feature_name: value} or pd.DataFrame
        
        Returns:
            dict containing:
              - prediction_code: 0 (Malignant) or 1 (Benign)
              - diagnosis: 'Malignant' or 'Benign'
              - prob_malignant: float [0, 1]
              - prob_benign: float [0, 1]
              - confidence: float [0.5, 1.0]
              - advisory: Educational disclaimer
        """
        if isinstance(input_data, dict):
            df_input = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            df_input = input_data.copy()
        else:
            raise TypeError("input_data must be a dict or pandas DataFrame.")

        # Ensure correct column alignment
        if self.feature_names:
            missing_cols = [col for col in self.feature_names if col not in df_input.columns]
            if missing_cols:
                raise ValueError(f"Input is missing required features: {missing_cols}")
            df_input = df_input[self.feature_names]

        # Inference
        pred_codes = self.pipeline.predict(df_input)
        probs = self.pipeline.predict_proba(df_input)  # [P(0: Malignant), P(1: Benign)]

        results = []
        for code, prob in zip(pred_codes, probs):
            p_mal = float(prob[0])
            p_ben = float(prob[1])
            diag = "Malignant" if code == 0 else "Benign"
            conf = max(p_mal, p_ben)

            results.append({
                "prediction_code": int(code),
                "diagnosis": diag,
                "probability_malignant": round(p_mal, 4),
                "probability_benign": round(p_ben, 4),
                "confidence_score": round(conf, 4),
                "model_used": self.best_model_name,
                "advisory": "Educational machine learning prediction. Not a certified clinical medical tool."
            })

        return results[0] if len(results) == 1 else results


def run_cli_test():
    """Test inference on sample patient cases from raw dataset."""
    paths = get_paths()
    predictor = BreastCancerPredictor()

    if os.path.exists(paths["raw_csv"]):
        df = pd.read_csv(paths["raw_csv"])
        feature_cols = [c for c in df.columns if c not in ["target", "diagnosis"]]

        # Sample 1: Actual Malignant
        sample_mal = df[df["target"] == 0].iloc[0][feature_cols].to_dict()
        pred_mal = predictor.predict(sample_mal)

        print("\n================ TEST CASE 1: ACTUAL MALIGNANT ================")
        print(f"Predicted Diagnosis: {pred_mal['diagnosis']}")
        print(f"Probability Malignant: {pred_mal['probability_malignant'] * 100:.2f}%")
        print(f"Probability Benign:    {pred_mal['probability_benign'] * 100:.2f}%")
        print(f"Confidence Score:      {pred_mal['confidence_score'] * 100:.2f}%")

        # Sample 2: Actual Benign
        sample_ben = df[df["target"] == 1].iloc[0][feature_cols].to_dict()
        pred_ben = predictor.predict(sample_ben)

        print("\n================ TEST CASE 2: ACTUAL BENIGN ================")
        print(f"Predicted Diagnosis: {pred_ben['diagnosis']}")
        print(f"Probability Malignant: {pred_ben['probability_malignant'] * 100:.2f}%")
        print(f"Probability Benign:    {pred_ben['probability_benign'] * 100:.2f}%")
        print(f"Confidence Score:      {pred_ben['confidence_score'] * 100:.2f}%")
        print("=============================================================")
        print(f"Advisory: {pred_mal['advisory']}\n")
    else:
        print("Raw dataset not found. Run preprocessing.py and train.py first.")


if __name__ == "__main__":
    run_cli_test()
