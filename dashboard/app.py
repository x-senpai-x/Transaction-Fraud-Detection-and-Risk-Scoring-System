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
page = st.sidebar.radio("Go to", ["Overview", "Transaction Explorer", "Model Metrics"])


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
                        response = requests.post(f"{API_URL}/predict", json=selected_sample)
                        
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


if page == "Overview":
    page_overview()
elif page == "Transaction Explorer":
    page_transaction_explorer()
elif page == "Model Metrics":
    page_model_metrics()
