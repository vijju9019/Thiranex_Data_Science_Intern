"""
Project 1: Data Cleaning and Visualization
Module: cleaning.py
Author: Data Science Intern (Thiranex)
Description:
    Industrial-grade data cleaning, validation, outlier treatment,
    and feature engineering pipeline for the Titanic dataset.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

RAW_DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"


def get_project_paths():
    """Resolve absolute paths for Project 1 directory structure."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return {
        "base": base_dir,
        "raw_dir": os.path.join(base_dir, "data", "raw"),
        "processed_dir": os.path.join(base_dir, "data", "processed"),
        "raw_csv": os.path.join(base_dir, "data", "raw", "titanic_raw.csv"),
        "processed_csv": os.path.join(base_dir, "data", "processed", "titanic_cleaned.csv"),
        "figures_dir": os.path.join(base_dir, "outputs", "figures")
    }


def load_and_cache_raw_data(url: str = RAW_DATA_URL, save_path: str = None) -> pd.DataFrame:
    """
    Load raw Titanic dataset from remote URL or local cache and ensure
    it is saved intact to data/raw/ without modification.
    """
    paths = get_project_paths()
    save_path = save_path or paths["raw_csv"]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        logger.info(f"Loading cached raw dataset from: {save_path}")
        df = pd.read_csv(save_path)
    else:
        logger.info(f"Downloading raw dataset from: {url}")
        df = pd.read_csv(url)
        df.to_csv(save_path, index=False)
        logger.info(f"Raw dataset successfully saved to: {save_path}")

    return df


def inspect_dataset(df: pd.DataFrame) -> dict:
    """
    Perform comprehensive initial dataset inspection:
    shape, data types, missing value counts, and summary statistics.
    """
    logger.info("--- INITIAL DATASET INSPECTION ---")
    logger.info(f"Dataset Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")

    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_summary = pd.DataFrame({
        "Missing_Count": missing,
        "Missing_Percentage": missing_pct.round(2)
    })
    missing_summary = missing_summary[missing_summary["Missing_Count"] > 0]

    logger.info(f"Columns with Missing Values:\n{missing_summary}")

    duplicates = df.duplicated().sum()
    logger.info(f"Duplicate Rows Count: {duplicates}")

    return {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "missing_summary": missing_summary,
        "duplicates": duplicates,
        "describe_num": df.describe().to_dict(),
    }


def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and eliminate duplicate records if any exist.
    """
    initial_len = len(df)
    df_clean = df.drop_duplicates()
    dropped = initial_len - len(df_clean)
    if dropped > 0:
        logger.info(f"Removed {dropped} duplicate records.")
    else:
        logger.info("No duplicate records detected in the dataset.")
    return df_clean


def extract_title(name: str) -> str:
    """
    Extract honorific title from passenger name string.
    Maps rare titles to standardized categories.
    """
    if not isinstance(name, str):
        return "Unknown"
    # Extract string between comma and period
    try:
        title = name.split(",")[1].split(".")[0].strip()
    except IndexError:
        return "Unknown"

    title_mapping = {
        "Mr": "Mr",
        "Miss": "Miss",
        "Mrs": "Mrs",
        "Master": "Master",
        "Dr": "Officer",
        "Rev": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Mlle": "Miss",
        "Mme": "Mrs",
        "Ms": "Miss",
        "Capt": "Officer",
        "Lady": "Royalty",
        "Countess": "Royalty",
        "Sir": "Royalty",
        "Jonkheer": "Royalty",
        "Don": "Royalty",
        "Dona": "Royalty"
    }
    return title_mapping.get(title, "Other")


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform statistically rigorous missing value imputation:
    1. Age: Imputed by median grouped by Pclass and Sex. Median is robust
       against the positive skewness of age distributions.
    2. Embarked: Imputed using the statistical mode ('S' - Southampton).
    3. Cabin: Over 77% missing. Extracted Deck category ('A'-'G', 'T')
       and encoded unrecorded values as 'Missing' Deck to avoid data loss.
    """
    df = df.copy()

    # 1. Feature extraction prior to age imputation
    if "Title" not in df.columns and "Name" in df.columns:
        df["Title"] = df["Name"].apply(extract_title)

    # 2. Age Imputation: Grouped median by Pclass and Sex
    logger.info("Imputing missing Age values using median grouped by Pclass and Sex...")
    age_medians = df.groupby(["Pclass", "Sex"])["Age"].transform("median")
    df["Age_Original"] = df["Age"]  # Retain for before/after comparison
    df["Age"] = df["Age"].fillna(age_medians)

    # Fallback if any remain
    if df["Age"].isnull().any():
        df["Age"] = df["Age"].fillna(df["Age"].median())

    # 3. Embarked Imputation: Mode
    embarked_mode = df["Embarked"].mode()[0]
    logger.info(f"Imputing missing Embarked values with mode: '{embarked_mode}'")
    df["Embarked"] = df["Embarked"].fillna(embarked_mode)

    # 4. Cabin Handling: Extract Deck level
    logger.info("Extracting Deck feature from Cabin and encoding missing values as 'Missing'...")
    df["Deck"] = df["Cabin"].apply(lambda x: str(x)[0] if pd.notnull(x) else "Missing")

    # 5. Fare: Impute any missing fare with median of that Pclass
    if df["Fare"].isnull().any():
        fare_median = df.groupby("Pclass")["Fare"].transform("median")
        df["Fare"] = df["Fare"].fillna(fare_median)

    logger.info("Missing value imputation completed.")
    return df


def treat_outliers_iqr(df: pd.DataFrame, column: str = "Fare", factor: float = 1.5) -> pd.DataFrame:
    """
    Detect and treat extreme outliers using the Interquartile Range (IQR) method.
    Outliers are capped (Winsorized) at the upper threshold (Q3 + 1.5 * IQR)
    rather than dropped, preserving historical passenger records without
    allowing extreme luxury fares to distort statistical and ML models.
    """
    df = df.copy()
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = max(0.0, q1 - factor * iqr)
    upper_bound = q3 + factor * iqr

    outlier_count = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
    logger.info(
        f"Outlier Treatment for '{column}': Q1={q1:.2f}, Q3={q3:.2f}, "
        f"IQR={iqr:.2f}, Upper Bound={upper_bound:.2f}. "
        f"Detected {outlier_count} outliers ({outlier_count / len(df) * 100:.2f}%)."
    )

    df[f"{column}_Raw"] = df[column]  # Retain raw for comparison
    # Apply capping
    df[f"{column}_Capped"] = df[column].clip(lower=lower_bound, upper=upper_bound)
    df[column] = df[f"{column}_Capped"]

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct domain-specific features:
    - FamilySize: Siblings + Spouses + Parents + Children + Self
    - IsAlone: Binary flag for solo travelers
    - FareBin: Binned fare categories
    - AgeGroup: Binned age categories
    - Format appropriate categorical and integer types
    """
    df = df.copy()

    # Family Size
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Age Group Categorization
    age_bins = [0, 12, 18, 35, 60, 120]
    age_labels = ["Child", "Adolescent", "Young_Adult", "Middle_Aged", "Senior"]
    df["AgeGroup"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels, right=False)

    # Fare Bins (Quartiles)
    df["FareBin"] = pd.qcut(df["Fare_Raw"] if "Fare_Raw" in df.columns else df["Fare"],
                            q=4, labels=["Low", "Medium_Low", "Medium_High", "High"])

    # Ensure consistent data types
    df["Survived"] = df["Survived"].astype(int)
    df["Pclass"] = df["Pclass"].astype(int)
    df["Sex"] = df["Sex"].astype("category")
    df["Embarked"] = df["Embarked"].astype("category")

    return df


def run_cleaning_pipeline() -> pd.DataFrame:
    """
    End-to-end execution of the data cleaning and processing pipeline.
    Saves clean dataset to data/processed/titanic_cleaned.csv.
    """
    paths = get_project_paths()
    logger.info("Starting Project 1 Data Cleaning Pipeline...")

    # Step 1: Load raw data
    df_raw = load_and_cache_raw_data(save_path=paths["raw_csv"])

    # Step 2: Initial inspection
    inspect_dataset(df_raw)

    # Step 3: Handle duplicates
    df_clean = handle_duplicates(df_raw)

    # Step 4: Impute missing values
    df_clean = impute_missing_values(df_clean)

    # Step 5: Treat outliers
    df_clean = treat_outliers_iqr(df_clean, column="Fare", factor=1.5)

    # Step 6: Feature engineering & typing
    df_clean = engineer_features(df_clean)

    # Step 7: Final verification
    remaining_nulls = df_clean.isnull().sum().sum()
    logger.info(f"Data Cleaning Completed. Remaining null values: {remaining_nulls}")

    # Step 8: Save processed dataset
    os.makedirs(paths["processed_dir"], exist_ok=True)
    df_clean.to_csv(paths["processed_csv"], index=False)
    logger.info(f"Cleaned dataset successfully saved to: {paths['processed_csv']}")

    return df_clean


if __name__ == "__main__":
    run_cleaning_pipeline()
