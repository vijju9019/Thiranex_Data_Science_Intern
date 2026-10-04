"""
Project 2: Predictive Modeling Using Machine Learning
Module: preprocessing.py
Author: Data Science Intern (Thiranex)
Description:
    Data ingestion, stratified splitting, and leakage-free pipeline creation
    for the Breast Cancer Wisconsin (Diagnostic) dataset.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def get_project_paths():
    """Resolve absolute paths for Project 2 directory structure."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return {
        "base": base_dir,
        "data_dir": os.path.join(base_dir, "data"),
        "raw_csv": os.path.join(base_dir, "data", "breast_cancer_raw.csv"),
        "models_dir": os.path.join(base_dir, "models"),
        "best_model": os.path.join(base_dir, "models", "best_model.joblib"),
        "metadata_json": os.path.join(base_dir, "models", "model_metadata.json"),
        "figures_dir": os.path.join(base_dir, "outputs", "figures")
    }


def load_and_export_raw_data(save_path: str = None) -> pd.DataFrame:
    """
    Load Breast Cancer Wisconsin dataset from Scikit-Learn and persist
    a raw CSV in data/ for reproducibility and auditability.
    
    Target encoding:
      - 0: Malignant
      - 1: Benign
    """
    paths = get_project_paths()
    save_path = save_path or paths["raw_csv"]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cancer = load_breast_cancer()
    df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
    df["target"] = cancer.target
    df["diagnosis"] = df["target"].map({0: "Malignant", 1: "Benign"})

    df.to_csv(save_path, index=False)
    logger.info(f"Loaded Breast Cancer dataset ({df.shape[0]} rows, {df.shape[1]} cols).")
    logger.info(f"Target breakdown: {df['diagnosis'].value_counts().to_dict()}")
    logger.info(f"Raw dataset successfully saved to: {save_path}")

    return df


def split_features_target(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Perform stratified train-test splitting to maintain target proportion
    and prevent class distribution skew between train and test folds.
    
    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    feature_names = [c for c in df.columns if c not in ["target", "diagnosis"]]
    X = df[feature_names]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    logger.info(f"Stratified Split completed: Train={X_train.shape[0]} samples, Test={X_test.shape[0]} samples.")
    logger.info(f"Train Malignant Ratio: {(y_train == 0).mean() * 100:.2f}%")
    logger.info(f"Test Malignant Ratio: {(y_test == 0).mean() * 100:.2f}%")

    return X_train, X_test, y_train, y_test, feature_names


def create_pipeline(estimator) -> Pipeline:
    """
    Create a leakage-free Scikit-learn Pipeline encapsulating:
    1. StandardScaler (fitted ONLY on training folds)
    2. Specified classification algorithm
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", estimator)
    ])


if __name__ == "__main__":
    df = load_and_export_raw_data()
    X_train, X_test, y_train, y_test, features = split_features_target(df)
    print("Preprocessing verification successful. Feature count:", len(features))
