"""
Project 2: Predictive Modeling Using Machine Learning
Streamlit Application: app.py
Author: Data Science Intern (Thiranex)
Description:
    Interactive oncology risk classification app powered by a Scikit-Learn pipeline.
    Demonstrates model evaluation, confusion matrix analytics, ROC curves,
    and real-time inference on FNA cell nuclear measurements.
"""

import os
import sys
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure src modules can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

from predict import BreastCancerPredictor

# Page Configuration
st.set_page_config(
    page_title="Breast Tumor Diagnostic Classifier | Thiranex Internship",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Clean Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    .disclaimer-banner {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 12px 18px;
        border-radius: 6px;
        color: #92400E;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 20px;
    }
    .result-card-malignant {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 2px solid #EF4444;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-bottom: 20px;
    }
    .result-card-benign {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border: 2px solid #10B981;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-bottom: 20px;
    }
    .result-title {
        font-size: 1.0rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .result-outcome {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 6px 0;
    }
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .kpi-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #2563EB;
    }
    .kpi-lbl {
        font-size: 0.8rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_predictor():
    return BreastCancerPredictor()


@st.cache_data
def get_dataset_and_metadata():
    raw_path = os.path.join(current_dir, "data", "breast_cancer_raw.csv")
    meta_path = os.path.join(current_dir, "models", "model_metadata.json")

    df = pd.read_csv(raw_path) if os.path.exists(raw_path) else None
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
    return df, meta


predictor = get_predictor()
df_cancer, metadata = get_dataset_and_metadata()
feature_cols = metadata.get("feature_names", [c for c in df_cancer.columns if c not in ["target", "diagnosis"]])

# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1579684385127-1ef15d508118?auto=format&fit=crop&w=400&q=80", use_container_width=True)
    st.markdown("### 🔬 Patient Case Profiles")
    st.caption("**Thiranex Data Science Internship** | Project 2")

    profile_choice = st.radio(
        "Load Clinical Profile Preset:",
        ["Typical Malignant Case", "Typical Benign Case", "Custom Manual Entry", "Random Test Case"],
        index=0
    )

    # Preset logic
    if df_cancer is not None:
        mal_mean = df_cancer[df_cancer["target"] == 0][feature_cols].median().to_dict()
        ben_mean = df_cancer[df_cancer["target"] == 1][feature_cols].median().to_dict()
    else:
        mal_mean = {f: 1.0 for f in feature_cols}
        ben_mean = {f: 0.5 for f in feature_cols}

    if profile_choice == "Typical Malignant Case":
        active_features = mal_mean.copy()
    elif profile_choice == "Typical Benign Case":
        active_features = ben_mean.copy()
    elif profile_choice == "Random Test Case":
        rand_idx = np.random.randint(0, len(df_cancer))
        active_features = df_cancer.iloc[rand_idx][feature_cols].to_dict()
    else:
        active_features = ben_mean.copy()

    st.divider()
    st.markdown("#### Fine-Needle Aspirate (FNA) Sliders")
    st.caption("Adjust primary nuclear morphometry parameters:")

    # Core interactive sliders for key top-predictive features
    c1_val = st.slider("Mean Radius (µm)", 6.0, 30.0, float(active_features.get("mean radius", 14.0)), 0.1)
    c2_val = st.slider("Mean Texture (grayscale)", 9.0, 40.0, float(active_features.get("mean texture", 19.0)), 0.1)
    c3_val = st.slider("Mean Perimeter (µm)", 40.0, 200.0, float(active_features.get("mean perimeter", 90.0)), 0.5)
    c4_val = st.slider("Mean Area (µm²)", 140.0, 2600.0, float(active_features.get("mean area", 650.0)), 5.0)
    c5_val = st.slider("Mean Concave Points", 0.0, 0.25, float(active_features.get("mean concave points", 0.05)), 0.005)
    c6_val = st.slider("Worst Radius (µm)", 7.0, 38.0, float(active_features.get("worst radius", 16.0)), 0.1)
    c7_val = st.slider("Worst Perimeter (µm)", 50.0, 260.0, float(active_features.get("worst perimeter", 107.0)), 0.5)
    c8_val = st.slider("Worst Concave Points", 0.0, 0.35, float(active_features.get("worst concave points", 0.1)), 0.005)

    # Update dictionary
    active_features["mean radius"] = c1_val
    active_features["mean texture"] = c2_val
    active_features["mean perimeter"] = c3_val
    active_features["mean area"] = c4_val
    active_features["mean concave points"] = c5_val
    active_features["worst radius"] = c6_val
    active_features["worst perimeter"] = c7_val
    active_features["worst concave points"] = c8_val

# ----------------- MAIN CONTENT -----------------
st.markdown('<div class="main-header">🔬 Breast Cancer Diagnostic ML System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Data Science Internship Project 2 | Leakage-Free Preprocessing, Multi-Classifier Benchmarking & Inference</div>', unsafe_allow_html=True)

# Mandatory Educational Disclaimer Banner
st.markdown("""
<div class="disclaimer-banner">
    ⚠️ <strong>EDUCATIONAL PROJECT ADVISORY:</strong> This application is developed strictly as an educational machine learning portfolio project for academic and skill demonstration purposes. It is <strong>NOT</strong> a certified medical device, clinical diagnostic tool, or a substitute for professional medical counsel.
</div>
""", unsafe_allow_html=True)

# Run Inference
prediction_res = predictor.predict(active_features)
diag = prediction_res["diagnosis"]
p_mal = prediction_res["probability_malignant"]
p_ben = prediction_res["probability_benign"]
conf = prediction_res["confidence_score"]
best_model_name = prediction_res["model_used"]

# Real-Time Diagnostic Banner
if diag == "Malignant":
    st.markdown(f"""
    <div class="result-card-malignant">
        <div class="result-title" style="color: #991B1B;">Diagnostic Prediction Result</div>
        <div class="result-outcome" style="color: #DC2626;">MALIGNANT (High Risk)</div>
        <div style="font-size: 1.1rem; color: #7F1D1D; font-weight: 500;">
            Estimated Malignancy Probability: <strong>{p_mal * 100:.1f}%</strong> | Confidence: <strong>{conf * 100:.1f}%</strong>
        </div>
        <div style="font-size: 0.85rem; color: #991B1B; margin-top: 6px;">
            Pipeline Model: {best_model_name} with Standard Scaler
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="result-card-benign">
        <div class="result-title" style="color: #065F46;">Diagnostic Prediction Result</div>
        <div class="result-outcome" style="color: #059669;">BENIGN (Low Risk)</div>
        <div style="font-size: 1.1rem; color: #064E3B; font-weight: 500;">
            Estimated Benign Probability: <strong>{p_ben * 100:.1f}%</strong> | Confidence: <strong>{conf * 100:.1f}%</strong>
        </div>
        <div style="font-size: 0.85rem; color: #065F46; margin-top: 6px;">
            Pipeline Model: {best_model_name} with Standard Scaler
        </div>
    </div>
    """, unsafe_allow_html=True)

# KPI Metrics Row
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-lbl">Model Accuracy</div><div class="kpi-val">{metadata.get('selected_metrics', {}).get('Accuracy', 0.9825)*100:.1f}%</div></div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-lbl">Recall / Sensitivity</div><div class="kpi-val" style="color:#10B981;">{metadata.get('selected_metrics', {}).get('Recall_Sensitivity', 0.9762)*100:.1f}%</div></div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-lbl">F1-Score</div><div class="kpi-val" style="color:#7C3AED;">{metadata.get('selected_metrics', {}).get('F1_Score', 0.9762):.4f}</div></div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-lbl">ROC-AUC Score</div><div class="kpi-val" style="color:#F59E0B;">{metadata.get('selected_metrics', {}).get('ROC_AUC', 0.9954):.4f}</div></div>""", unsafe_allow_html=True)

st.write("")

# Main Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Model Comparison & Benchmarks",
    "📈 Evaluation Figures & Diagnostics",
    "🎯 Feature Importance & Biological Drivers",
    "📑 Wisconsin Dataset Inspector"
])

# TAB 1: Model Benchmarks
with tab1:
    st.subheader("Supervised Classification Benchmark Comparison")
    st.markdown("""
    All models were trained strictly using **stratified 80/20 train-test splits** and encapsulated within **StandardScaler preprocessing pipelines** to prevent data leakage.
    In medical diagnostic tasks, **Recall (Sensitivity)** on malignant cases is the primary optimization objective to minimize hazardous False Negatives.
    """)

    all_models_summary = metadata.get("all_models_summary", [])
    if all_models_summary:
        df_bench = pd.DataFrame(all_models_summary)
        st.dataframe(
            df_bench.style.highlight_max(subset=["Accuracy", "Recall (Sensitivity)", "F1-Score", "ROC-AUC"], color="#D1FAE5"),
            use_container_width=True
        )
    else:
        st.info("Run `python src/train.py` to regenerate the live model summary table.")

    st.markdown("""
    ##### 💡 Metric Interpretation
    - **Logistic Regression & SVM:** Tied for highest test accuracy (**98.25%**) and highest sensitivity (**97.62%**), with near-perfect ROC-AUC (**0.995+**).
    - **Random Forest:** Achieved strong generalizability (**94.74%** accuracy) with robust non-linear boundaries.
    - **Decision Tree:** Single tree had lower test accuracy (**92.11%**) and higher False Positives due to variance on continuous features.
    """)

# TAB 2: Figures
with tab2:
    st.subheader("Diagnostic Figures & Evaluation Plots")
    col_f1, col_f2 = st.columns(2)

    fig_dir = os.path.join(current_dir, "outputs", "figures")
    f_comp = os.path.join(fig_dir, "01_model_comparison_metrics.png")
    f_cm = os.path.join(fig_dir, "02_confusion_matrices.png")
    f_roc = os.path.join(fig_dir, "03_roc_curves.png")
    f_feat = os.path.join(fig_dir, "04_feature_importance.png")

    with col_f1:
        if os.path.exists(f_comp):
            st.image(f_comp, caption="1. Algorithmic Metric Comparison", use_container_width=True)
        if os.path.exists(f_roc):
            st.image(f_roc, caption="3. Receiver Operating Characteristic (ROC) Curves", use_container_width=True)

    with col_f2:
        if os.path.exists(f_cm):
            st.image(f_cm, caption="2. Confusion Matrices (Held-out Test Cohort)", use_container_width=True)
        if os.path.exists(f_feat):
            st.image(f_feat, caption="4. Random Forest Feature Importances", use_container_width=True)

# TAB 3: Feature Importance
with tab3:
    st.subheader("Cellular Morphometry & Biological Drivers of Malignancy")
    st.markdown("""
    The features are derived from digitized images of Fine Needle Aspirates (FNA) describing characteristics of the cell nuclei present in the biopsy.
    """)

    b1, b2 = st.columns(2)
    with b1:
        st.info("""
        **Top Predictive Indicators:**
        1. **Worst Perimeter & Worst Radius:** Malignant tumors display severe cellular hypertrophy and irregular enlargement.
        2. **Worst Concave Points & Mean Concavity:** Malignant nuclei exhibit significant indentations, notches, and boundary irregularities.
        3. **Mean Texture:** Variations in chromatin distribution cause granular, uneven grayscale shading.
        """)
    with b2:
        # Comparison of current active sample vs cohort medians
        comp_df = pd.DataFrame({
            "Measurement": ["Mean Radius", "Mean Texture", "Mean Perimeter", "Worst Radius", "Worst Concave Points"],
            "Current Input": [active_features.get("mean radius"), active_features.get("mean texture"), active_features.get("mean perimeter"), active_features.get("worst radius"), active_features.get("worst concave points")],
            "Typical Benign Median": [ben_mean.get("mean radius"), ben_mean.get("mean texture"), ben_mean.get("mean perimeter"), ben_mean.get("worst radius"), ben_mean.get("worst concave points")],
            "Typical Malignant Median": [mal_mean.get("mean radius"), mal_mean.get("mean texture"), mal_mean.get("mean perimeter"), mal_mean.get("worst radius"), mal_mean.get("worst concave points")]
        })
        st.dataframe(comp_df, use_container_width=True)

# TAB 4: Dataset Inspector
with tab4:
    st.subheader("Breast Cancer Wisconsin (Diagnostic) Dataset Inspector")
    if df_cancer is not None:
        st.write(f"Total Cohort Size: `{df_cancer.shape[0]}` patients, `{df_cancer.shape[1]}` features")
        diag_filter = st.selectbox("Filter by True Diagnosis:", ["All Cases", "Malignant Only", "Benign Only"])
        if diag_filter == "Malignant Only":
            st.dataframe(df_cancer[df_cancer["target"] == 0].head(25), use_container_width=True)
        elif diag_filter == "Benign Only":
            st.dataframe(df_cancer[df_cancer["target"] == 1].head(25), use_container_width=True)
        else:
            st.dataframe(df_cancer.head(25), use_container_width=True)
    else:
        st.warning("Raw dataset file not found.")
