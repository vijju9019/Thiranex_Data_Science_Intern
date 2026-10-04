"""
Project 3: Exploratory Data Analysis (EDA)
Module: analysis.py
Author: Data Science Intern (Thiranex)
Description:
    Industrial-grade exploratory data analysis engine.
    Performs univariate, bivariate, multivariate, correlation,
    and statistical hypothesis testing (Chi-Square & T-Tests),
    saving 12 publication-quality diagnostic visualizations.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Apply publication-grade aesthetics
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
PALETTE_SURVIVAL = ["#EF4444", "#10B981"]  # 0: Perished, 1: Survived


def get_paths():
    """Resolve paths for Project 3."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return {
        "base": base_dir,
        "data_csv": os.path.join(base_dir, "data", "titanic_data.csv"),
        "figures_dir": os.path.join(base_dir, "outputs", "figures")
    }


def load_dataset() -> pd.DataFrame:
    """Load and format the EDA dataset."""
    paths = get_paths()
    if not os.path.exists(paths["data_csv"]):
        raise FileNotFoundError(f"Dataset not found at: {paths['data_csv']}")
    df = pd.read_csv(paths["data_csv"])
    logger.info(f"Loaded dataset for EDA: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def compute_descriptive_statistics(df: pd.DataFrame) -> dict:
    """
    Compute comprehensive parametric and non-parametric statistics:
    Mean, Std, Median, IQR, Skewness, Kurtosis.
    """
    num_cols = ["Age", "Fare", "FamilySize", "SibSp", "Parch"]
    stats_dict = {}

    for col in num_cols:
        series = df[col].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        stats_dict[col] = {
            "Count": len(series),
            "Mean": round(series.mean(), 2),
            "Std": round(series.std(), 2),
            "Median": round(series.median(), 2),
            "IQR": round(q3 - q1, 2),
            "Min": round(series.min(), 2),
            "Max": round(series.max(), 2),
            "Skewness": round(series.skew(), 2),
            "Kurtosis": round(series.kurtosis(), 2)
        }

    df_stats = pd.DataFrame(stats_dict).T
    logger.info(f"\n--- DESCRIPTIVE NUMERICAL STATISTICS ---\n{df_stats}")
    return df_stats


def perform_hypothesis_testing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Conduct rigorous statistical hypothesis testing:
    1. Chi-Square Test of Independence for categorical features against Survival.
    2. Mann-Whitney U test & Two-Sample T-test for numerical features between survivors & non-survivors.
    """
    test_results = []

    # Categorical Tests (Chi-Square)
    cat_vars = ["Sex", "Pclass", "Embarked", "IsAlone", "Title"]
    for var in cat_vars:
        contingency_table = pd.crosstab(df[var], df["Survived"])
        chi2, p_val, dof, _ = stats.chi2_contingency(contingency_table)
        test_results.append({
            "Feature": var,
            "Variable Type": "Categorical",
            "Statistical Test": "Chi-Square Independence",
            "Test Statistic": round(chi2, 3),
            "p-value": f"{p_val:.2e}",
            "Statistically Significant (alpha=0.01)": "YES (p < 0.01)" if p_val < 0.01 else "NO"
        })

    # Numerical Tests (Mann-Whitney U & T-Test)
    num_vars = ["Fare", "Age", "FamilySize"]
    survived_0 = df[df["Survived"] == 0]
    survived_1 = df[df["Survived"] == 1]

    for var in num_vars:
        s0 = survived_0[var].dropna()
        s1 = survived_1[var].dropna()
        u_stat, p_val = stats.mannwhitneyu(s0, s1, alternative="two-sided")
        test_results.append({
            "Feature": var,
            "Variable Type": "Numerical",
            "Statistical Test": "Mann-Whitney U Test",
            "Test Statistic": round(u_stat, 3),
            "p-value": f"{p_val:.2e}",
            "Statistically Significant (alpha=0.01)": "YES (p < 0.01)" if p_val < 0.01 else "NO"
        })

    df_hyp = pd.DataFrame(test_results)
    logger.info(f"\n--- STATISTICAL HYPOTHESIS TESTING RESULTS ---\n{df_hyp.to_string(index=False)}")
    return df_hyp


def save_fig(fig, filename: str, figures_dir: str):
    """Save plot with high quality."""
    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, filename)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure: {out_path}")


def generate_all_eda_visualizations(df: pd.DataFrame, figures_dir: str):
    """Generate and save all 12 publication-grade figures."""
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Univariate Numerical Distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    num_vars = ["Age", "Fare", "FamilySize"]
    colors = ["#2563EB", "#7C3AED", "#059669"]
    for i, var in enumerate(num_vars):
        sns.histplot(df[var], kde=True, color=colors[i], bins=25, ax=axes[i], stat="density")
        axes[i].set_title(f"Univariate Distribution: {var}", weight="bold")
        axes[i].set_xlabel(var)
        axes[i].set_ylabel("Density")
    save_fig(fig, "01_univariate_numerical_distributions.png", figures_dir)

    # 2. Categorical Frequency Counts
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    cat_vars = ["Pclass", "Sex", "Embarked", "Title"]
    for i, var in enumerate(cat_vars):
        sns.countplot(
            data=df,
            x=var,
            hue=var,
            palette="Blues_r",
            legend=False,
            ax=axes[i],
            edgecolor="black"
        )
        axes[i].set_title(f"Frequency Count: {var}", weight="bold")
        for p in axes[i].patches:
            h = p.get_height()
            if h > 0:
                axes[i].annotate(f"{int(h)} ({h / len(df) * 100:.1f}%)",
                                 (p.get_x() + p.get_width() / 2., h / 2),
                                 ha="center", va="center", color="white", weight="bold", fontsize=9)
    save_fig(fig, "02_categorical_frequency_counts.png", figures_dir)

    # 3. Survival Rate by Gender & Class (Grouped Bar)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=df,
        x="Pclass",
        y="Survived",
        hue="Sex",
        palette=["#3B82F6", "#EC4899"],
        errorbar=None,
        ax=ax,
        edgecolor="black"
    )
    ax.set_title("Survival Rate across Socio-Economic Class & Gender", weight="bold", fontsize=13, pad=12)
    ax.set_ylabel("Survival Probability")
    ax.set_xlabel("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)")
    ax.set_ylim(0, 1.05)
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h * 100:.1f}%",
                        (p.get_x() + p.get_width() / 2., h / 2),
                        ha="center", va="center", color="white", weight="bold")
    ax.legend(title="Gender", loc="upper right")
    save_fig(fig, "03_survival_by_gender_class_bar.png", figures_dir)

    # 4. Fare Distribution Across Classes (Box & Strip Plot)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.boxplot(data=df, x="Pclass", y="Fare", palette="Purples", ax=ax, showmeans=True)
    sns.stripplot(data=df, x="Pclass", y="Fare", color="black", alpha=0.15, jitter=0.2, size=3, ax=ax)
    ax.set_title("Ticket Fare Distribution across Passenger Classes (with Outlier Capping)", weight="bold", fontsize=13)
    ax.set_xlabel("Passenger Class")
    ax.set_ylabel("Fare (£ / $)")
    save_fig(fig, "04_fare_distribution_by_class_boxplot.png", figures_dir)

    # 5. Violin Plots: Age vs Class Split by Survival
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.violinplot(
        data=df,
        x="Pclass",
        y="Age",
        hue="Survived",
        split=True,
        palette=PALETTE_SURVIVAL,
        inner="quartile",
        ax=ax
    )
    ax.set_title("Age Distribution by Class Split by Survival Outcome", weight="bold", fontsize=13)
    ax.set_xlabel("Passenger Class")
    ax.set_ylabel("Age (Years)")
    ax.legend(title="Outcome", labels=["Perished (0)", "Survived (1)"], loc="upper right")
    save_fig(fig, "05_violin_plots_age_pclass_survival.png", figures_dir)

    # 6. Multivariate Scatter: Age vs Fare by Survival
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=df,
        x="Age",
        y="Fare",
        hue="Survived",
        style="Sex",
        palette=PALETTE_SURVIVAL,
        alpha=0.8,
        s=60,
        ax=ax
    )
    ax.set_title("Multivariate Interaction: Age vs Fare Segmented by Survival & Gender", weight="bold", fontsize=13)
    ax.set_xlabel("Passenger Age (Years)")
    ax.set_ylabel("Fare ($)")
    save_fig(fig, "06_age_vs_fare_multivariate_scatter.png", figures_dir)

    # 7. Pearson vs Spearman Correlation Matrix
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    num_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]

    corr_p = df[num_cols].corr(method="pearson")
    corr_s = df[num_cols].corr(method="spearman")
    mask = np.triu(np.ones_like(corr_p, dtype=bool))

    sns.heatmap(corr_p, mask=mask, annot=True, fmt=".2f", cmap="vlag", vmin=-0.8, vmax=0.8, ax=ax1, cbar=False)
    ax1.set_title("Pearson Correlation (Linear Relationships)", weight="bold")

    sns.heatmap(corr_s, mask=mask, annot=True, fmt=".2f", cmap="vlag", vmin=-0.8, vmax=0.8, ax=ax2)
    ax2.set_title("Spearman Correlation (Monotonic Rank Relationships)", weight="bold")

    save_fig(fig, "07_correlation_matrices_pearson_spearman.png", figures_dir)

    # 8. Family Travel Dynamics Analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=df, x="FamilySize", y="Survived", color=PRIMARY_COLOR, errorbar=None, ax=ax1, edgecolor="black")
    ax1.set_title("Survival Rate by Family Size (SibSp + Parch + 1)", weight="bold")
    ax1.set_ylabel("Survival Rate")
    for p in ax1.patches:
        h = p.get_height()
        ax1.annotate(f"{h * 100:.1f}%", (p.get_x() + p.get_width() / 2., max(h / 2, 0.05)),
                     ha="center", va="center", color="white" if h > 0.15 else "black", weight="bold")

    sns.barplot(
        data=df,
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
    ax2.set_title("Survival Rate: Traveling Alone vs With Family", weight="bold")
    ax2.set_ylabel("Survival Rate")
    for p in ax2.patches:
        h = p.get_height()
        ax2.annotate(f"{h * 100:.1f}%", (p.get_x() + p.get_width() / 2., h / 2),
                     ha="center", va="center", color="white", weight="bold")
    save_fig(fig, "08_family_travel_dynamics_analysis.png", figures_dir)

    # 9. Port of Embarkation Socioeconomics
    fig, ax = plt.subplots(figsize=(9, 5))
    port_map = {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}
    df_port = df.copy()
    df_port["Port"] = df_port["Embarked"].map(port_map)
    sns.barplot(data=df_port, x="Port", y="Survived", hue="Pclass", palette="Blues", errorbar=None, ax=ax, edgecolor="black")
    ax.set_title("Survival Disparities across Embarkation Ports by Class", weight="bold", fontsize=13)
    ax.set_ylabel("Survival Rate")
    ax.legend(title="Passenger Class", loc="upper right")
    save_fig(fig, "09_embarkation_port_socioeconomics.png", figures_dir)

    # 10. Pairplot of Key Numerical Features
    pairplot_cols = ["Age", "Fare", "FamilySize", "Survived"]
    pair_fig = sns.pairplot(
        df[pairplot_cols],
        hue="Survived",
        palette=PALETTE_SURVIVAL,
        diag_kind="kde",
        plot_kws={"alpha": 0.6, "s": 35}
    )
    pair_fig.fig.suptitle("Pairwise Relationships of Key Numerical Features by Survival", weight="bold", y=1.02)
    save_fig(pair_fig.fig, "10_pairplot_key_features.png", figures_dir)

    # 11. Demographic Age Groups & Titles Survival
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x="Title", y="Survived", palette="Set2", errorbar=None, ax=ax, edgecolor="black")
    ax.set_title("Survival Probability across Honorific Titles", weight="bold", fontsize=13)
    ax.set_ylabel("Survival Probability")
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(f"{h * 100:.1f}%", (p.get_x() + p.get_width() / 2., h / 2),
                    ha="center", va="center", color="white" if h > 0.1 else "black", weight="bold")
    save_fig(fig, "11_survival_by_title_and_age_group.png", figures_dir)

    # 12. Statistical Hypothesis Tests Summary Visual
    hyp_df = perform_hypothesis_testing(df)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=hyp_df, x="Feature", y="Test Statistic", hue="Variable Type", palette=["#2563EB", "#7C3AED"], ax=ax)
    ax.set_title("Statistical Significance Profile (Chi-Square & Mann-Whitney U Statistics)", weight="bold", fontsize=13)
    ax.set_ylabel("Test Statistic Value (log scale)")
    ax.set_yscale("log")
    save_fig(fig, "12_statistical_hypothesis_tests_summary.png", figures_dir)

    logger.info(f"Successfully created all 12 EDA publication figures in: {figures_dir}")


def run_eda_pipeline():
    """Execute complete EDA analytical pipeline."""
    paths = get_paths()
    df = load_dataset()
    compute_descriptive_statistics(df)
    generate_all_eda_visualizations(df, paths["figures_dir"])


if __name__ == "__main__":
    run_eda_pipeline()
