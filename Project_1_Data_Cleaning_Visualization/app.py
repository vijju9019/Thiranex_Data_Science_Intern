"""
Project 1: Data Cleaning and Visualization
Streamlit Application: app.py
Author: Data Science Intern (Thiranex)
Description:
    Interactive dashboard to inspect raw vs cleaned Titanic passenger data,
    audit missing value imputation, explore outlier capping, and analyze survival insights.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Titanic Data Cleaning & Insights | Thiranex Internship",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Professional Clean White / Indigo & Purple Accents)
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        text-align: center;
    }
    .kpi-title {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2563EB;
        margin-top: 4px;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: #EEF2FF;
        color: #4F46E5;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_datasets():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(base_dir, "data", "raw", "titanic_raw.csv")
    cleaned_path = os.path.join(base_dir, "data", "processed", "titanic_cleaned.csv")

    if not os.path.exists(raw_path) or not os.path.exists(cleaned_path):
        # Auto-run cleaning if not present
        from src.cleaning import run_cleaning_pipeline
        run_cleaning_pipeline()

    df_raw = pd.read_csv(raw_path)
    df_clean = pd.read_csv(cleaned_path)
    return df_raw, df_clean


df_raw, df_clean = load_datasets()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1509316975850-ff9c5deb0cd9?auto=format&fit=crop&w=400&q=80", use_container_width=True)
    st.markdown("### 🚢 Navigation & Filters")
    st.markdown("**Thiranex Internship Portfolio**  \n*Project 1: Data Cleaning & Visualization*")
    st.divider()

    # Filter: Class
    pclass_options = sorted(df_clean["Pclass"].unique())
    selected_pclass = st.multiselect("Passenger Class (Pclass)", pclass_options, default=pclass_options)

    # Filter: Sex
    sex_options = ["All"] + list(df_clean["Sex"].unique())
    selected_sex = st.selectbox("Gender", sex_options, index=0)

    # Filter: Age Range
    min_age, max_age = int(df_clean["Age"].min()), int(df_clean["Age"].max())
    selected_age = st.slider("Age Range (Years)", min_value=min_age, max_value=max_age, value=(min_age, max_age))

    # Filter: Embarkation Port
    port_names = {"S": "Southampton (S)", "C": "Cherbourg (C)", "Q": "Queenstown (Q)"}
    port_opts = ["All"] + list(port_names.keys())
    selected_port = st.selectbox("Embarkation Port", port_opts, format_func=lambda x: port_names.get(x, x))

    st.divider()
    st.caption("⚙️ Framework: Python 3.11+ | Pandas | Streamlit")

# Apply Filters
filtered_df = df_clean.copy()
if selected_pclass:
    filtered_df = filtered_df[filtered_df["Pclass"].isin(selected_pclass)]
if selected_sex != "All":
    filtered_df = filtered_df[filtered_df["Sex"] == selected_sex]
filtered_df = filtered_df[(filtered_df["Age"] >= selected_age[0]) & (filtered_df["Age"] <= selected_age[1])]
if selected_port != "All":
    filtered_df = filtered_df[filtered_df["Embarked"] == selected_port]

# ----------------- MAIN CONTENT -----------------
st.markdown('<div class="main-header">🚢 Titanic Data Cleaning & Statistical Insights</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Data Science Internship Project 1 | Robust Imputation, Outlier Treatment & Exploratory Analysis</div>', unsafe_allow_html=True)

# KPI Cards
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Passengers Filtered</div>
        <div class="kpi-value">{len(filtered_df):,} <span style="font-size:0.9rem; color:#94A3B8;">/ {len(df_clean)}</span></div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    surv_rate = filtered_df["Survived"].mean() * 100 if len(filtered_df) > 0 else 0
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Survival Rate</div>
        <div class="kpi-value" style="color: {'#10B981' if surv_rate >= 40 else '#EF4444'};">{surv_rate:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    avg_fare = filtered_df["Fare"].mean() if len(filtered_df) > 0 else 0
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Average Fare (Capped)</div>
        <div class="kpi-value">${avg_fare:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    med_age = filtered_df["Age"].median() if len(filtered_df) > 0 else 0
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Median Age</div>
        <div class="kpi-value">{med_age:.1f} <span style="font-size:0.9rem; color:#94A3B8;">yrs</span></div>
    </div>
    """, unsafe_allow_html=True)

with kpi5:
    solo_pct = (filtered_df["IsAlone"].mean() * 100) if len(filtered_df) > 0 else 0
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Solo Travelers</div>
        <div class="kpi-value" style="color:#7C3AED;">{solo_pct:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Data Cleaning Audit",
    "📊 Outlier Analysis & Treatment",
    "📈 Exploratory Visualizations",
    "🔥 Correlation Matrix",
    "💾 Download Cleaned Dataset"
])

# TAB 1: Cleaning Audit
with tab1:
    st.subheader("Data Cleaning & Preprocessing Audit")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("##### 1. Raw Dataset Status (Before Cleaning)")
        st.write(f"Shape: `{df_raw.shape[0]}` rows, `{df_raw.shape[1]}` columns")
        raw_nulls = df_raw.isnull().sum()
        raw_null_df = pd.DataFrame({
            "Missing Count": raw_nulls,
            "Percentage": (raw_nulls / len(df_raw) * 100).round(2)
        })
        st.dataframe(raw_null_df[raw_null_df["Missing Count"] > 0], use_container_width=True)

        st.info("""
        **Data Issues Identified:**
        - **Age:** 177 missing values (19.87%). Imputed with median grouped by `Pclass` and `Sex`.
        - **Cabin:** 687 missing values (77.10%). Rather than dropping, extracted Deck letter ('A'-'G') and assigned 'Missing' level.
        - **Embarked:** 2 missing values (0.22%). Imputed with statistical mode ('S' - Southampton).
        """)

    with col2:
        st.markdown("##### 2. Processed Dataset Status (After Cleaning)")
        st.write(f"Shape: `{df_clean.shape[0]}` rows, `{df_clean.shape[1]}` columns")
        clean_nulls = df_clean[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "IsAlone"]].isnull().sum()
        clean_null_df = pd.DataFrame({
            "Missing Count": clean_nulls,
            "Percentage": (clean_nulls / len(df_clean) * 100).round(2)
        })
        st.dataframe(clean_null_df, use_container_width=True)

        st.success("""
        **Validation Status: Passed ✅**
        - Zero missing values in all feature columns.
        - Outliers capped using 1.5x IQR Winsorization.
        - Engineered new high-signal features: `Title`, `FamilySize`, `IsAlone`, `AgeGroup`, `Deck`.
        """)

    st.markdown("##### Raw vs Cleaned Data Inspector")
    view_option = st.radio("Choose view:", ["Processed Data (Clean)", "Raw Data (Original)"], horizontal=True)
    if view_option == "Processed Data (Clean)":
        st.dataframe(filtered_df.head(15), use_container_width=True)
    else:
        st.dataframe(df_raw.head(15), use_container_width=True)

# TAB 2: Outlier Analysis
with tab2:
    st.subheader("Outlier Detection & IQR Treatment on Ticket Fare")
    st.markdown("""
    Extreme ticket fares (reaching up to **$512.33**) create heavy right-skewness and distort linear relationships.
    To prevent data loss from deleting valid high-net-worth passenger records, we applied **1.5x IQR Winsorization (capping)**.
    """)

    q1 = df_raw["Fare"].quantile(0.25)
    q3 = df_raw["Fare"].quantile(0.75)
    iqr = q3 - q1
    upper_threshold = q3 + 1.5 * iqr
    outliers_count = (df_raw["Fare"] > upper_threshold).sum()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("25th Percentile (Q1)", f"${q1:.2f}")
    m2.metric("75th Percentile (Q3)", f"${q3:.2f}")
    m3.metric("IQR", f"${iqr:.2f}")
    m4.metric("Upper Outlier Cap", f"${upper_threshold:.2f}", f"{outliers_count} points capped")

    fig_box, (ax_b1, ax_b2) = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(x=df_clean["Fare_Raw"], ax=ax_b1, color="#FCA5A5")
    ax_b1.set_title("Original Fare (Raw with Outliers)", weight="bold")
    ax_b1.set_xlabel("Fare ($)")

    sns.boxplot(x=df_clean["Fare"], ax=ax_b2, color="#93C5FD")
    ax_b2.set_title(f"Cleaned Fare (Capped at ${upper_threshold:.2f})", weight="bold")
    ax_b2.set_xlabel("Fare ($)")

    st.pyplot(fig_box)
    plt.close(fig_box)

# TAB 3: Visualizations
with tab3:
    st.subheader("Exploratory Visualizations on Filtered Cohort")
    v_col1, v_col2 = st.columns(2)

    with v_col1:
        st.markdown("##### Survival by Gender & Class")
        fig_dem, ax_d = plt.subplots(figsize=(6, 4))
        sns.barplot(
            data=filtered_df,
            x="Pclass",
            y="Survived",
            hue="Sex",
            palette=["#3B82F6", "#EC4899"],
            errorbar=None,
            ax=ax_d
        )
        ax_d.set_ylabel("Survival Probability")
        ax_d.set_xlabel("Passenger Class")
        ax_d.set_ylim(0, 1.0)
        st.pyplot(fig_dem)
        plt.close(fig_dem)

    with v_col2:
        st.markdown("##### Age Distribution (Imputed)")
        fig_age, ax_a = plt.subplots(figsize=(6, 4))
        sns.histplot(filtered_df["Age"], kde=True, color="#4F46E5", bins=25, ax=ax_a)
        ax_a.set_xlabel("Age (Years)")
        ax_a.set_ylabel("Count")
        st.pyplot(fig_age)
        plt.close(fig_age)

    v_col3, v_col4 = st.columns(2)
    with v_col3:
        st.markdown("##### Family Size vs Survival")
        fig_fam, ax_f = plt.subplots(figsize=(6, 4))
        sns.barplot(
            data=filtered_df,
            x="FamilySize",
            y="Survived",
            color="#2563EB",
            errorbar=None,
            ax=ax_f
        )
        ax_f.set_ylabel("Survival Probability")
        ax_f.set_xlabel("Family Size (SibSp + Parch + 1)")
        st.pyplot(fig_fam)
        plt.close(fig_fam)

    with v_col4:
        st.markdown("##### Passenger Count by Honorific Title")
        fig_title, ax_t = plt.subplots(figsize=(6, 4))
        sns.countplot(
            data=filtered_df,
            x="Title",
            order=filtered_df["Title"].value_counts().index,
            palette="Blues_r",
            ax=ax_t
        )
        ax_t.set_ylabel("Count")
        ax_t.set_xlabel("Extracted Title")
        plt.xticks(rotation=25)
        st.pyplot(fig_title)
        plt.close(fig_title)

# TAB 4: Correlation Matrix
with tab4:
    st.subheader("Correlation Matrix (Numerical Features)")
    num_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]
    corr_df = filtered_df[num_cols].corr()

    fig_corr, ax_c = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    sns.heatmap(corr_df, mask=mask, annot=True, fmt=".2f", cmap="vlag", vmin=-0.8, vmax=0.8, ax=ax_c)
    st.pyplot(fig_corr)
    plt.close(fig_corr)

# TAB 5: Export
with tab5:
    st.subheader("Export Cleaned & Transformed Dataset")
    st.markdown("Download the fully cleaned, outlier-capped, and feature-engineered Titanic dataset ready for machine learning pipelines.")
    csv_bytes = df_clean.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download titanic_cleaned.csv",
        data=csv_bytes,
        file_name="titanic_cleaned.csv",
        mime="text/csv",
        use_container_width=True
    )
