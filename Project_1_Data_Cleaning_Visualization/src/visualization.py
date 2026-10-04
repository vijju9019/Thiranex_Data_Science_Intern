"""
Project 1: Data Cleaning and Visualization
Module: visualization.py
Author: Data Science Intern (Thiranex)
Description:
    Industrial-grade exploratory and diagnostic visualizations for Project 1,
    saving high-resolution publication-quality PNG charts.
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
    "figure.titlesize": 14,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.autolayout": False
})

PRIMARY_COLOR = "#2563EB"    # Professional Royal Blue
SECONDARY_COLOR = "#7C3AED"  # Royal Purple
ACCENT_GREEN = "#059669"     # Emerald Green
ACCENT_RED = "#DC2626"       # Crimson Red
PALETTE_SURVIVAL = ["#EF4444", "#10B981"]  # Red = Perished, Green = Survived


def get_paths():
    """Resolve paths for data and output figures."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return {
        "raw_csv": os.path.join(base_dir, "data", "raw", "titanic_raw.csv"),
        "processed_csv": os.path.join(base_dir, "data", "processed", "titanic_cleaned.csv"),
        "figures_dir": os.path.join(base_dir, "outputs", "figures")
    }


def save_figure(fig, filename: str, figures_dir: str):
    """Save figure with high DPI and tight layout."""
    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, filename)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure: {out_path}")


def plot_missing_values_heatmap_raw(df_raw: pd.DataFrame, figures_dir: str):
    """1. Missing Values Heatmap on Raw Data."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        df_raw.isnull(),
        cbar=True,
        cmap="viridis",
        yticklabels=False,
        ax=ax
    )
    ax.set_title("Missing Values Pattern in Raw Titanic Dataset", fontsize=14, weight="bold", pad=15)
    ax.set_xlabel("Columns", fontsize=12)
    save_figure(fig, "01_missing_values_heatmap.png", figures_dir)


def plot_missing_values_after_cleaning(df_clean: pd.DataFrame, figures_dir: str):
    """2. Verification Heatmap after cleaning (Zero Missing Values)."""
    fig, ax = plt.subplots(figsize=(10, 3))
    # Select key modeling columns
    cols_to_check = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "IsAlone"]
    sns.heatmap(
        df_clean[cols_to_check].isnull(),
        cbar=False,
        cmap="Blues",
        yticklabels=False,
        ax=ax
    )
    ax.set_title("Post-Cleaning Verification: Complete Data Integrity (0 Missing Values)", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Cleaned Feature Columns", fontsize=11)
    save_figure(fig, "02_missing_values_after_cleaning.png", figures_dir)


def plot_age_distribution_before_after(df_clean: pd.DataFrame, figures_dir: str):
    """3. Age Distribution Before vs After Imputation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Before Imputation (raw Age with NaNs omitted)
    sns.histplot(df_clean["Age_Original"].dropna(), kde=True, color="#4B5563", bins=30, ax=ax1, stat="density")
    ax1.set_title("Age Distribution: Raw (n=714, Missing=177)", fontsize=12, weight="bold")
    ax1.set_xlabel("Passenger Age (Years)")
    ax1.set_ylabel("Density")
    ax1.axvline(df_clean["Age_Original"].median(), color="red", linestyle="--", label=f"Median: {df_clean['Age_Original'].median():.1f}")
    ax1.legend()

    # After Grouped Imputation
    sns.histplot(df_clean["Age"], kde=True, color=PRIMARY_COLOR, bins=30, ax=ax2, stat="density")
    ax2.set_title("Age Distribution: Post-Grouped Imputation (n=891)", fontsize=12, weight="bold")
    ax2.set_xlabel("Passenger Age (Years)")
    ax2.set_ylabel("Density")
    ax2.axvline(df_clean["Age"].median(), color="green", linestyle="--", label=f"Median: {df_clean['Age'].median():.1f}")
    ax2.legend()

    fig.suptitle("Impact of Grouped Median Imputation on Age Distribution", fontsize=14, weight="bold", y=1.02)
    save_figure(fig, "03_age_distribution_before_after.png", figures_dir)


def plot_survival_distribution(df_clean: pd.DataFrame, figures_dir: str):
    """4. Survival Distribution: Count & Percentage."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    counts = df_clean["Survived"].value_counts().sort_index()
    labels = ["Perished (0)", "Survived (1)"]

    # Bar Count Plot
    bars = ax1.bar(labels, counts.values, color=PALETTE_SURVIVAL, width=0.55, edgecolor="black", linewidth=0.8)
    for bar in bars:
        h = bar.get_height()
        pct = (h / len(df_clean)) * 100
        ax1.annotate(f"{h}\n({pct:.1f}%)", xy=(bar.get_x() + bar.get_width() / 2, h / 2),
                     xytext=(0, 0), textcoords="offset points", ha="center", va="center",
                     fontsize=11, weight="bold", color="white")
    ax1.set_title("Passenger Count by Survival Outcome", fontsize=12, weight="bold")
    ax1.set_ylabel("Number of Passengers")

    # Donut Chart
    wedges, texts, autotexts = ax2.pie(
        counts.values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=PALETTE_SURVIVAL,
        wedgeprops=dict(width=0.4, edgecolor="black")
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_weight("bold")
    ax2.set_title("Survival Proportion Breakdown", fontsize=12, weight="bold")

    fig.suptitle("Overall Titanic Survival Distribution", fontsize=14, weight="bold", y=1.02)
    save_figure(fig, "04_survival_distribution.png", figures_dir)


def plot_correlation_heatmap(df_clean: pd.DataFrame, figures_dir: str):
    """5. Correlation Heatmap of Numerical Features."""
    fig, ax = plt.subplots(figsize=(9, 7))
    num_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]
    corr = df_clean[num_cols].corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=0.8,
        vmin=-0.8,
        center=0,
        square=True,
        linewidths=0.5,
        annot=True,
        fmt=".2f",
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    ax.set_title("Correlation Matrix of Numerical Passenger Features", fontsize=13, weight="bold", pad=15)
    save_figure(fig, "05_correlation_heatmap.png", figures_dir)


def plot_outliers_boxplots(df_clean: pd.DataFrame, figures_dir: str):
    """6. Boxplots: Outlier Detection and Capping Effect on Fare & Age."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Raw Fare
    sns.boxplot(y=df_clean["Fare_Raw"], ax=axes[0], color="#FCA5A5")
    axes[0].set_title("Raw Fare (Extreme Outliers up to $512)", fontsize=11, weight="bold")
    axes[0].set_ylabel("Fare ($)")

    # Capped Fare
    sns.boxplot(y=df_clean["Fare"], ax=axes[1], color="#93C5FD")
    axes[1].set_title("Fare After 1.5x IQR Capping (Winsorized)", fontsize=11, weight="bold")
    axes[1].set_ylabel("Capped Fare ($)")

    # Age Boxplot
    sns.boxplot(y=df_clean["Age"], ax=axes[2], color="#A7F3D0")
    axes[2].set_title("Age Distribution (Cleaned)", fontsize=11, weight="bold")
    axes[2].set_ylabel("Age (Years)")

    fig.suptitle("Outlier Detection & Treatment via Interquartile Range (IQR)", fontsize=14, weight="bold", y=1.02)
    save_figure(fig, "06_fare_age_outliers_boxplots.png", figures_dir)


def plot_survival_by_pclass_and_sex(df_clean: pd.DataFrame, figures_dir: str):
    """7. Comparative Survival Rates across Socio-economic Class and Gender."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Survival Rate by Sex
    sns.barplot(
        data=df_clean,
        x="Sex",
        y="Survived",
        hue="Sex",
        palette=["#3B82F6", "#EC4899"],
        legend=False,
        errorbar=None,
        ax=ax1,
        edgecolor="black"
    )
    for p in ax1.patches:
        ax1.annotate(f"{p.get_height() * 100:.1f}%",
                     (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                     ha="center", va="center", fontsize=11, weight="bold", color="white")
    ax1.set_title("Survival Rate by Gender", fontsize=12, weight="bold")
    ax1.set_ylabel("Survival Rate")
    ax1.set_ylim(0, 1.0)

    # Survival Rate by Pclass and Sex
    sns.barplot(
        data=df_clean,
        x="Pclass",
        y="Survived",
        hue="Sex",
        palette=["#3B82F6", "#EC4899"],
        errorbar=None,
        ax=ax2,
        edgecolor="black"
    )
    for p in ax2.patches:
        h = p.get_height()
        if h > 0:
            ax2.annotate(f"{h * 100:.1f}%",
                         (p.get_x() + p.get_width() / 2., h / 2),
                         ha="center", va="center", fontsize=10, weight="bold", color="white")
    ax2.set_title("Survival Rate by Socio-economic Class & Gender", fontsize=12, weight="bold")
    ax2.set_xlabel("Passenger Class (1st, 2nd, 3rd)")
    ax2.set_ylabel("Survival Rate")
    ax2.set_ylim(0, 1.0)
    ax2.legend(title="Gender", loc="upper right")

    fig.suptitle("Demographic and Socio-Economic Disparities in Survival", fontsize=14, weight="bold", y=1.02)
    save_figure(fig, "07_survival_by_pclass_and_sex.png", figures_dir)


def plot_family_size_survival(df_clean: pd.DataFrame, figures_dir: str):
    """8. Family Dynamics: Survival Rate across Family Size and IsAlone."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Family Size vs Survival Rate
    sns.barplot(
        data=df_clean,
        x="FamilySize",
        y="Survived",
        color=PRIMARY_COLOR,
        errorbar=None,
        ax=ax1,
        edgecolor="black"
    )
    for p in ax1.patches:
        h = p.get_height()
        ax1.annotate(f"{h * 100:.1f}%",
                     (p.get_x() + p.get_width() / 2., max(h / 2, 0.05)),
                     ha="center", va="center", fontsize=10, weight="bold", color="white" if h > 0.15 else "black")
    ax1.set_title("Survival Rate by Family Size (SibSp + Parch + 1)", fontsize=12, weight="bold")
    ax1.set_ylabel("Survival Rate")
    ax1.set_xlabel("Family Size")
    ax1.set_ylim(0, 0.85)

    # IsAlone vs Survival
    sns.barplot(
        data=df_clean,
        x="IsAlone",
        y="Survived",
        hue="IsAlone",
        palette=[SECONDARY_COLOR, "#64748B"],
        legend=False,
        errorbar=None,
        ax=ax2,
        edgecolor="black"
    )
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["With Family (0)", "Traveling Alone (1)"])
    for p in ax2.patches:
        h = p.get_height()
        ax2.annotate(f"{h * 100:.1f}%",
                     (p.get_x() + p.get_width() / 2., h / 2),
                     ha="center", va="center", fontsize=11, weight="bold", color="white")
    ax2.set_title("Survival Rate: Traveling Alone vs With Family", fontsize=12, weight="bold")
    ax2.set_ylabel("Survival Rate")
    ax2.set_ylim(0, 0.85)

    fig.suptitle("Impact of Family Travel Dynamics on Passenger Survival", fontsize=14, weight="bold", y=1.02)
    save_figure(fig, "08_family_size_survival_distribution.png", figures_dir)


def plot_embarked_vs_fare(df_clean: pd.DataFrame, figures_dir: str):
    """9. Port of Embarkation vs Ticket Fare and Class."""
    fig, ax = plt.subplots(figsize=(10, 5))
    port_map = {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}
    df_clean_plot = df_clean.copy()
    df_clean_plot["PortName"] = df_clean_plot["Embarked"].map(port_map)

    sns.boxplot(
        data=df_clean_plot,
        x="PortName",
        y="Fare",
        hue="Pclass",
        palette="Purples",
        ax=ax
    )
    ax.set_title("Ticket Fare Distribution Across Embarkation Ports by Class", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Port of Embarkation")
    ax.set_ylabel("Fare ($)")
    ax.legend(title="Passenger Class", loc="upper right")
    save_figure(fig, "09_embarked_vs_fare_distribution.png", figures_dir)


def generate_all_visualizations():
    """Execute complete visualization pipeline."""
    paths = get_paths()
    logger.info("Starting Project 1 Visualization Pipeline...")

    if not os.path.exists(paths["raw_csv"]):
        raise FileNotFoundError(f"Raw CSV not found at: {paths['raw_csv']}. Run cleaning.py first.")
    if not os.path.exists(paths["processed_csv"]):
        raise FileNotFoundError(f"Cleaned CSV not found at: {paths['processed_csv']}. Run cleaning.py first.")

    df_raw = pd.read_csv(paths["raw_csv"])
    df_clean = pd.read_csv(paths["processed_csv"])

    fig_dir = paths["figures_dir"]
    plot_missing_values_heatmap_raw(df_raw, fig_dir)
    plot_missing_values_after_cleaning(df_clean, fig_dir)
    plot_age_distribution_before_after(df_clean, fig_dir)
    plot_survival_distribution(df_clean, fig_dir)
    plot_correlation_heatmap(df_clean, fig_dir)
    plot_outliers_boxplots(df_clean, fig_dir)
    plot_survival_by_pclass_and_sex(df_clean, fig_dir)
    plot_family_size_survival(df_clean, fig_dir)
    plot_embarked_vs_fare(df_clean, fig_dir)

    logger.info(f"Successfully generated all 9 high-resolution figures in: {fig_dir}")


if __name__ == "__main__":
    generate_all_visualizations()
