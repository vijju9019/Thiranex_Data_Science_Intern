"""
Project 2: Predictive Modeling Using Machine Learning
Module: train.py
Author: Data Science Intern (Thiranex)
Description:
    Trains, validates, evaluates, and compares 4 classification models:
    - Logistic Regression
    - Decision Tree Classifier
    - Random Forest Classifier
    - Support Vector Machine (SVM)
    Generates diagnostic figures and serializes the best performing model.
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

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report
)

# Relative import support
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from preprocessing import (
    get_project_paths,
    load_and_export_raw_data,
    split_features_target,
    create_pipeline
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Visual styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
PRIMARY_COLOR = "#2563EB"
SECONDARY_COLOR = "#7C3AED"


def train_and_evaluate_all_models(X_train, X_test, y_train, y_test, feature_names):
    """
    Train 4 classifiers within preprocessing pipelines and collect actual metrics.
    
    Target mapping:
      0 = Malignant (positive class of interest in oncology)
      1 = Benign
    
    We evaluate recall/precision both for Malignant detection (pos_label=0)
    and overall weighted/macro metrics.
    """
    models = {
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_split=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        "Support Vector Machine": SVC(C=1.0, kernel="rbf", probability=True, random_state=42)
    }

    results = {}
    fitted_pipelines = {}

    logger.info("--- MODEL TRAINING & EVALUATION COMMENCED ---")

    for name, estimator in models.items():
        logger.info(f"Training {name} with Standard Scaler pipeline...")
        pipe = create_pipeline(estimator)
        pipe.fit(X_train, y_train)

        # Predictions
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)  # [P(0: Malignant), P(1: Benign)]
        prob_malignant = y_prob[:, 0]

        # Calculate metrics (Malignant as positive class: pos_label=0)
        acc = accuracy_score(y_test, y_pred)
        prec_mal = precision_score(y_test, y_pred, pos_label=0)
        rec_mal = recall_score(y_test, y_pred, pos_label=0)
        f1_mal = f1_score(y_test, y_pred, pos_label=0)
        # Invert prob so higher score predicts class 0 for roc_auc if binary target is 0/1
        auc = roc_auc_score(y_test == 0, prob_malignant)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "Accuracy": acc,
            "Precision (Malignant)": prec_mal,
            "Recall (Sensitivity)": rec_mal,
            "F1-Score": f1_mal,
            "ROC-AUC Score": auc,
            "Confusion_Matrix": cm.tolist(),
            "y_pred": y_pred.tolist(),
            "prob_malignant": prob_malignant.tolist()
        }
        fitted_pipelines[name] = pipe

        logger.info(
            f"{name} -> Accuracy: {acc:.4f} | Recall (Mal): {rec_mal:.4f} | "
            f"Precision (Mal): {prec_mal:.4f} | F1: {f1_mal:.4f} | AUC: {auc:.4f}"
        )

    return results, fitted_pipelines


def generate_comparison_table(results: dict) -> pd.DataFrame:
    """Format and display model benchmark comparison table."""
    rows = []
    for model_name, m in results.items():
        rows.append({
            "Algorithm": model_name,
            "Accuracy": round(m["Accuracy"], 4),
            "Precision (Mal)": round(m["Precision (Malignant)"], 4),
            "Recall (Sensitivity)": round(m["Recall (Sensitivity)"], 4),
            "F1-Score": round(m["F1-Score"], 4),
            "ROC-AUC": round(m["ROC-AUC Score"], 4)
        })

    df_comp = pd.DataFrame(rows).sort_values(by="F1-Score", ascending=False).reset_index(drop=True)
    logger.info(f"\n================ MODEL BENCHMARK TABLE ================\n{df_comp.to_string(index=False)}")
    return df_comp


def plot_and_save_figures(results: dict, fitted_pipelines: dict, X_test, y_test, feature_names, figures_dir: str):
    """Generate 4 publication-grade figures documenting model evaluation."""
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Model Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(11, 5))
    metrics_to_plot = ["Accuracy", "Recall (Sensitivity)", "F1-Score", "ROC-AUC Score"]
    model_names = list(results.keys())

    x = np.arange(len(model_names))
    width = 0.20

    palette = ["#2563EB", "#7C3AED", "#10B981", "#F59E0B"]

    for i, metric in enumerate(metrics_to_plot):
        vals = [results[m][metric] for m in model_names]
        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=metric, color=palette[i], edgecolor="black", linewidth=0.5)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, weight="bold")

    ax.set_ylabel("Metric Score", fontsize=11, weight="bold")
    ax.set_title("Machine Learning Benchmark Comparison (Breast Cancer Diagnostic)", fontsize=13, weight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=10, weight="bold")
    ax.set_ylim(0.85, 1.05)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "01_model_comparison_metrics.png"), dpi=300)
    plt.close(fig)

    # 2. Confusion Matrices (2x2 Grid)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()

    for i, (name, m) in enumerate(results.items()):
        cm = np.array(m["Confusion_Matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axes[i],
            xticklabels=["Malignant (0)", "Benign (1)"],
            yticklabels=["Malignant (0)", "Benign (1)"]
        )
        axes[i].set_title(f"{name}\nAcc: {m['Accuracy']:.3f} | Recall: {m['Recall (Sensitivity)']:.3f}", fontsize=11, weight="bold")
        axes[i].set_xlabel("Predicted Diagnosis")
        axes[i].set_ylabel("True Diagnosis")

    fig.suptitle("Confusion Matrices Across All Classifiers", fontsize=14, weight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "02_confusion_matrices.png"), dpi=300)
    plt.close(fig)

    # 3. Combined ROC Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2563EB", "#EF4444", "#10B981", "#7C3AED"]

    for i, (name, m) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_test == 0, m["prob_malignant"])
        ax.plot(fpr, tpr, label=f"{name} (AUC = {m['ROC-AUC Score']:.3f})", color=colors[i], linewidth=2.2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess (AUC = 0.50)")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, weight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, weight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, weight="bold", pad=12)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "03_roc_curves.png"), dpi=300)
    plt.close(fig)

    # 4. Feature Importance (Random Forest)
    rf_estimator = fitted_pipelines["Random Forest"].named_steps["classifier"]
    importances = rf_estimator.feature_importances_
    feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    feat_df = feat_df.sort_values(by="Importance", ascending=False).head(12)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.barplot(data=feat_df, x="Importance", y="Feature", palette="viridis", ax=ax, edgecolor="black", linewidth=0.5)
    ax.set_title("Top 12 Predictive Features (Random Forest Importance)", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Relative Gini Importance", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "04_feature_importance.png"), dpi=300)
    plt.close(fig)

    logger.info(f"Saved all 4 evaluation figures in: {figures_dir}")


def save_best_model_and_metadata(results: dict, fitted_pipelines: dict, feature_names, df_comp: pd.DataFrame):
    """
    Select best model based on primary metric:
    In oncology screening, Recall on malignant class and F1 are paramount to minimize False Negatives.
    """
    paths = get_project_paths()
    os.makedirs(paths["models_dir"], exist_ok=True)

    # Rank by F1 then Recall
    best_model_name = df_comp.iloc[0]["Algorithm"]
    best_pipeline = fitted_pipelines[best_model_name]

    # Save model pipeline with joblib
    joblib.dump(best_pipeline, paths["best_model"])
    logger.info(f"Best Model ('{best_model_name}') serialized to: {paths['best_model']}")

    # Save serializable metadata
    metadata = {
        "best_model_name": best_model_name,
        "selected_metrics": {
            "Accuracy": float(results[best_model_name]["Accuracy"]),
            "Recall_Sensitivity": float(results[best_model_name]["Recall (Sensitivity)"]),
            "Precision_Malignant": float(results[best_model_name]["Precision (Malignant)"]),
            "F1_Score": float(results[best_model_name]["F1-Score"]),
            "ROC_AUC": float(results[best_model_name]["ROC-AUC Score"])
        },
        "all_models_summary": df_comp.to_dict(orient="records"),
        "feature_names": feature_names,
        "classes": ["Malignant (0)", "Benign (1)"],
        "advisory": "Educational machine learning portfolio project. Not intended for clinical or diagnostic healthcare decisions."
    }

    with open(paths["metadata_json"], "w") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Model metadata serialized to: {paths['metadata_json']}")

    return best_model_name


def run_training_pipeline():
    """Execute end-to-end Project 2 pipeline."""
    paths = get_project_paths()

    # Step 1: Load data
    df = load_and_export_raw_data()

    # Step 2: Stratified Split
    X_train, X_test, y_train, y_test, feature_names = split_features_target(df)

    # Step 3: Train and evaluate all 4 algorithms
    results, fitted_pipelines = train_and_evaluate_all_models(X_train, X_test, y_train, y_test, feature_names)

    # Step 4: Comparison Table
    df_comp = generate_comparison_table(results)

    # Step 5: Generate and save figures
    plot_and_save_figures(results, fitted_pipelines, X_test, y_test, feature_names, paths["figures_dir"])

    # Step 6: Serialize best model and metadata
    best_name = save_best_model_and_metadata(results, fitted_pipelines, feature_names, df_comp)

    logger.info(f"Project 2 Training Pipeline Completed Successfully! Selected Model: {best_name}")


if __name__ == "__main__":
    run_training_pipeline()
