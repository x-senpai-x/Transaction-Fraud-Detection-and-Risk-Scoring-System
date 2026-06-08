# Transaction Risk Engine Architecture

This document provides a comprehensive architectural overview of the Transaction Risk Engine. The system is designed to be a realistic, production-ready, and leakage-safe machine learning pipeline that ingests raw tabular data, trains gradient-boosting models, serves predictions in real-time, and monitors for covariate shift.

---

## 1. High-Level System Flow

The architecture is broken down into five distinct phases that form a complete ML lifecycle:

1. **Data Ingestion (`data/load.py`)**: Loads raw CSVs, standardizes schemas, creates proxy entity IDs, and stores the interim data securely as Parquet.
2. **Feature Engineering (`features/build.py`)**: Applies strict chronological time-based splitting to prevent temporal leakage. Extracts missingness flags, log-transformed amounts, and high-cardinality frequency encodings.
3. **Model Training (`models/train.py`)**: Trains both a heavily imputed/scaled Logistic Regression baseline and a LightGBM champion model. Evaluates purely on PR-AUC and Recall@Top-K.
4. **Real-Time Inference (`api/main.py`)**: A highly scalable FastAPI service that mirrors the feature engineering pipeline dynamically in memory to score raw incoming JSON payloads.
5. **Explainability & Monitoring (`explain/`, `monitoring/`)**: Evaluates global feature logic via SHAP and tracks production drift via PSI and KS statistics.

---

## 2. Component Architecture

### A. Data & Feature Layer
- **Parquet Storage**: We utilize `.parquet` instead of `.csv` for all interim and processed files to enforce strict datatype retention (preventing Pandas from dynamically re-casting strings to floats) and to significantly reduce memory/disk overhead.
- **Proxy IDs**: Instead of pure graph networks, we use deterministic entity resolution concatenating hardware, email, and address strings to create robust `_uid` features.
- **Frequency Encoding**: High-cardinality strings (like `card1` or `device_uid`) are encoded by their training-set frequency. These frequency maps are explicitly saved to `models/frequency_maps.json` to ensure the API uses the exact same dictionary during inference, preventing leakage.
- **Time-Based Split**: Random `train_test_split` is strictly banned. The engine orders transactions chronologically, using the first 80% of time for training, the next 10% for validation (early stopping), and the final 10% as an unseen test holdout.

### B. Modeling Layer
- **Evaluation Philosophy**: Accuracy is mathematically useless when the fraud rate is only ~3.5%. The engine explicitly tracks:
  - `PR-AUC` (Precision-Recall Area Under Curve): To balance catching fraud against punishing legitimate users.
  - `Recall @ Top 1%`: Measuring operational efficiency (how much fraud is caught if a human analyst team can only review 1% of total volume).
- **LightGBM Gradient Boosting**: Tree-based models are used as the champion due to their native handling of sparse missing values (NaNs) without requiring heavy imputation, which is critical for highly sparse transactional data.

### C. Serving Layer (FastAPI)
The API (`api/main.py`) is designed as a stateless microservice:
- **Pydantic Validation**: Uses `TransactionRequest` with `model_config(extra="allow")` to rigidly validate required keys while flexibly accepting the massive ~400 column payload.
- **Dynamic Imputation**: The `inference.py` class runs the exact same `build.py` functions (e.g., `add_proxy_ids`) in memory on a single dictionary, forcing alignment against `models/feature_metadata.json` to guarantee dimensional stability before pinging `model.predict_proba()`.
- **Risk Score Scaling**: The API natively scales the raw, heavily imbalanced base-rate probabilities into a normalized `0-100` Human-Readable Risk Score.

### D. Explainability Layer (SHAP)
- **Global Interpretability**: `global_shap.py` calculates the global matrix to understand systemic patterns (e.g., "Are email domains more important than transaction amounts?").
- **Local Interpretability**: `shap_explainer.py` uses `shap.TreeExplainer` on the fly inside the API to provide human-readable English strings explaining exactly why a *specific* transaction was blocked.

### E. Monitoring Layer (Drift Detection)
To simulate a production alert system, `drift.py` loads the Train split (Reference) and Test split (Current) to calculate:
- **Population Stability Index (PSI)**: Bins the predicted probabilities into 10 deciles to monitor if the model's overall confidence output distribution is drastically shifting over time.
- **Kolmogorov-Smirnov (KS) Test**: Tracks severe covariate distribution shifts in the top 5 raw numeric features.

---

## 3. Directory Structure

```text
transaction-risk-engine/
├── api/                        # FastAPI application and Pydantic schemas
├── config/                     # YAML configuration for parameters
├── dashboard/                  # Streamlit frontend user interface
├── data/
│   ├── raw/                    # Original raw CSVs (ignored by git)
│   ├── interim/                # Joined Parquet files
│   ├── processed/              # Chronologically split feature matrices
│   └── sample/                 # Subset JSON payloads for the dashboard
├── docs/                       # Project scope and roadmaps
├── models/                     # Serialized champion models and JSON maps
├── reports/                    # Extracted PR/ROC curve PNGs, metrics, and drift reports
├── scripts/                    # Helper scripts (e.g., sample extractors)
├── src/transaction_risk_engine/
│   ├── data/                   # Extraction and splitting logic
│   ├── explain/                # SHAP interpretability logic
│   ├── features/               # Feature generation and encoding logic
│   ├── models/                 # Training orchestrators and wrappers
│   └── monitoring/             # PSI and drift logic
└── tests/                      # Pytest unit tests for all components
```
