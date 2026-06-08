import json
import urllib.parse
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Transaction Risk Engine",
    page_icon="🛡️",
    layout="wide",
)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Overview", 
    "Transaction Explorer", 
    "Model Metrics",
    "Explainability & Monitoring"
])


def page_overview():
    st.title("🛡️ Transaction Risk Engine")
    st.markdown("""
    Welcome to the **Transaction Risk Engine** dashboard.
    
    This system scores online transactions in real time and returns a fraud probability, 
    risk score, approve/review/block decision, and human-readable explanations using 
    behavioral, velocity, and entity-relationship features.
    
    ### Architecture
    - **Frontend:** Streamlit Dashboard (You are here)
    - **Backend:** FastAPI Inference Engine
    - **Model:** LightGBM Gradient Boosting
    - **Features:** High-cardinality frequency encoding, proxy IDs, base numerics.
    """)


def page_transaction_explorer():
    st.title("🔍 Transaction Explorer")
    st.markdown("Select a sample transaction and send it to the real-time inference API for scoring.")

    # Load samples
    sample_path = Path("data/sample/sample_transactions.json")
    if not sample_path.exists():
        st.error("Sample transactions not found. Please run scripts/extract_samples.py")
        return

    with open(sample_path, "r") as f:
        samples = json.load(f)

    # Sidebar selection
    transaction_ids = [str(s.get("TransactionID", "Unknown")) for s in samples]
    selected_id_str = st.sidebar.selectbox("Select TransactionID", transaction_ids)

    selected_sample = None
    for s in samples:
        if str(s.get("TransactionID")) == selected_id_str:
            selected_sample = s
            break

    if selected_sample:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Raw Transaction Payload")
            st.json(selected_sample)

        with col2:
            st.subheader("API Risk Assessment")
            if st.button("Score Transaction 🚀", type="primary"):
                with st.spinner("Calling API..."):
                    try:
                        import math
                        def clean_nans(obj):
                            if isinstance(obj, dict):
                                return {k: clean_nans(v) for k, v in obj.items()}
                            elif isinstance(obj, list):
                                return [clean_nans(v) for v in obj]
                            elif isinstance(obj, float) and math.isnan(obj):
                                return None
                            return obj
                            
                        clean_sample = clean_nans(selected_sample)
                        response = requests.post(f"{API_URL}/predict", json=clean_sample)
                        
                        if response.status_code == 200:
                            data = response.json()
                            
                            score = data["risk_score"]
                            decision = data["decision"]
                            
                            # Determine color
                            color = "green"
                            if decision == "BLOCK":
                                color = "red"
                            elif decision == "REVIEW":
                                color = "orange"
                                
                            st.markdown(f"### Score: <span style='color:{color}'>{score} / 100</span>", unsafe_allow_html=True)
                            st.markdown(f"### Decision: <span style='color:{color}'>{decision}</span>", unsafe_allow_html=True)
                            
                            st.markdown("#### Top Risk Signals")
                            for reason in data.get("top_reasons", []):
                                st.markdown(f"- {reason}")
                                
                        else:
                            st.error(f"API Error: {response.text}")
                    except requests.exceptions.ConnectionError:
                        st.error("Could not connect to API. Is the FastAPI server running on port 8000?")


def page_model_metrics():
    st.title("📊 Model Metrics")
    st.markdown("Performance of the baseline champion model on the unseen validation time-split.")
    
    try:
        response = requests.get(f"{API_URL}/metrics")
        if response.status_code == 200:
            metrics_data = response.json()
            if "lgbm" in metrics_data:
                lgbm_metrics = metrics_data["lgbm"]
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("PR-AUC", f"{lgbm_metrics.get('pr_auc', 0):.4f}")
                col2.metric("ROC-AUC", f"{lgbm_metrics.get('roc_auc', 0):.4f}")
                col3.metric("Recall @ Top 1%", f"{lgbm_metrics.get('recall_at_top_1_pct', 0):.4f}")
                col4.metric("Precision @ Top 1%", f"{lgbm_metrics.get('precision_at_top_1_pct', 0):.4f}")
    except:
        st.warning("Could not fetch metrics from API. Ensure it is running.")
        
    st.subheader("Evaluation Curves")
    col1, col2 = st.columns(2)
    
    pr_path = Path("reports/lgbm_pr_curve.png")
    roc_path = Path("reports/lgbm_roc_curve.png")
    
    with col1:
        if pr_path.exists():
            st.image(str(pr_path), caption="LightGBM Precision-Recall Curve", use_container_width=True)
            
    with col2:
        if roc_path.exists():
            st.image(str(roc_path), caption="LightGBM ROC Curve", use_container_width=True)


def page_explainability_monitoring():
    st.title("🧠 Explainability & Monitoring")
    
    tab1, tab2 = st.tabs(["Global Explainability", "Production Drift"])
    
    with tab1:
        st.subheader("Global Feature Importance (SHAP)")
        st.markdown("These plots demonstrate how the LightGBM model globally interprets features to assign risk.")
        
        col1, col2 = st.columns(2)
        summary_path = Path("reports/shap_summary.png")
        bar_path = Path("reports/shap_bar.png")
        
        with col1:
            if summary_path.exists():
                st.image(str(summary_path), caption="SHAP Summary Dot Plot", use_container_width=True)
            else:
                st.warning("SHAP summary plot not found. Run `python -m transaction_risk_engine.explain.global_shap`")
                
        with col2:
            if bar_path.exists():
                st.image(str(bar_path), caption="SHAP Mean Absolute Importance", use_container_width=True)
                
    with tab2:
        st.subheader("Simulated Production Drift")
        st.markdown("Monitoring covariate shift between Training (Reference) and Testing (Current) datasets.")
        
        report_path = Path("reports/drift_report.json")
        if not report_path.exists():
            st.warning("Drift report not found. Run `python -m transaction_risk_engine.monitoring.drift`")
            return
            
        with open(report_path, "r") as f:
            drift_data = json.load(f)
            
        pred_drift = drift_data.get("prediction_drift", {})
        st.markdown("### Model Score Drift (PSI)")
        
        status = pred_drift.get("status", "Unknown")
        color = "green" if status == "Stable" else ("orange" if "Moderate" in status else "red")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Population Stability Index (PSI)", f"{pred_drift.get('psi', 0):.4f}")
        c2.markdown(f"**Status:** <span style='color:{color}; font-size:1.5rem'>{status}</span>", unsafe_allow_html=True)
        c3.metric("Current Reference Size", f"{pred_drift.get('current_size', 0):,}")
        
        st.markdown("---")
        st.markdown("### Feature Drift (Kolmogorov-Smirnov Test)")
        
        features_drift = drift_data.get("feature_drift_ks", {})
        if features_drift:
            # Convert dict to df for display
            df_drift = pd.DataFrame.from_dict(features_drift, orient="index")
            df_drift.index.name = "Feature"
            df_drift.reset_index(inplace=True)
            
            # Format
            def highlight_drift(val):
                return 'color: red' if val else 'color: green'
                
            st.dataframe(df_drift.style.map(highlight_drift, subset=['drift_detected']), use_container_width=True)


if page == "Overview":
    page_overview()
elif page == "Transaction Explorer":
    page_transaction_explorer()
elif page == "Model Metrics":
    page_model_metrics()
elif page == "Explainability & Monitoring":
    page_explainability_monitoring()
