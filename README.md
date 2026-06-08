# Transaction Risk Engine

Transaction Risk Engine is a resume-oriented ML engineering project inspired by modern payment fraud systems. The baseline goal is not to clone Stripe/Visa infrastructure. The goal is to build a clean, demoable system that proves you can handle:

- messy tabular financial transaction data,
- leakage-safe feature engineering,
- imbalanced ML evaluation,
- production-style inference,
- explainability,
- monitoring and model iteration.

## One-line project explanation

> Transaction Risk Engine scores online transactions in real time and returns a fraud probability, risk score, approve/review/block decision, and human-readable explanation using behavioral, velocity, and entity-relationship features.

## What this is not

This is not a generic credit default project. Credit default predicts whether a customer will fail to repay later. Transaction Risk Engine predicts whether a specific transaction happening now looks suspicious based on transaction behavior and entity relationships such as card, device, email domain, address, and recent activity.

## Baseline first, advanced later

### Resume-ready baseline

Build this first:

1. IEEE-CIS data loading and cleaning.
2. Time-based train/validation/test split.
3. Frequency encoding and basic transaction features.
4. Leakage-aware historical aggregations.
5. LightGBM or XGBoost model.
6. PR-AUC, ROC-AUC, Recall@Top-K, and threshold report.
7. FastAPI `/predict` endpoint.
8. Streamlit dashboard with model metrics and transaction explorer.
9. SHAP explanations for individual transactions.
10. Clean README, architecture diagram, and resume bullets.

### Advanced extensions

Add only after the baseline works:

1. Velocity features over trailing windows.
2. Entity graph features: card-device-email-address relationships.
3. Drift monitoring using PSI/KS tests.
4. Champion vs challenger model comparison.
5. Replay simulator for pseudo-real-time scoring.
6. Node2Vec or GraphSAGE experiment.

## Current status

Phase 0, Phase 1 (Data Ingestion), Phase 2 (Baseline Features & Splits), and Phase 3 (Baseline Model Training) are complete.

Next milestone: Phase 4 API and Dashboard.

## Setup

Use Python 3.11+.

Recommended Conda environment:

```bash
conda env create -f environment.yml
conda activate transaction-risk-engine
```

Standard `venv` setup:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Validate the Phase 0 setup:

```bash
python -m pytest
python -c "from transaction_risk_engine.config import load_config; print(load_config()['project']['name'])"
```

## Data

Download the IEEE-CIS Fraud Detection files and place them here:

```txt
data/raw/train_transaction.csv
data/raw/train_identity.csv
```

Do not commit raw Kaggle data, generated feature matrices, model artifacts, or reports. The repository keeps those directories with `.gitkeep` files only.

## Planned Milestone Commands

These commands become available as each milestone is implemented:

```bash
python -m transaction_risk_engine.data.load --config config/config.yaml
python -m transaction_risk_engine.features.build --config config/config.yaml
python -m transaction_risk_engine.models.train --config config/config.yaml
uvicorn api.main:app --reload
streamlit run dashboard/app.py
```

## Suggested final repository structure

```txt
transaction-risk-engine/
├── README.md
├── AGENTS.md
├── CLAUDE.md
├── CODEX.md
├── ANTIGRAVITY.md
├── pyproject.toml
├── requirements.txt
├── requirements.baseline.txt
├── environment.yml
├── .env.example
├── config/
│   └── config.yaml
├── docs/
│   ├── PROJECT_SCOPE.md
│   ├── IMPLEMENTATION_ROADMAP.md
│   ├── DATASET_AND_LEAKAGE_RULES.md
│   ├── ML_CONCEPTS_TARGETED.md
│   ├── SYSTEM_ARCHITECTURE.md
│   ├── RESUME_POSITIONING.md
│   ├── DATA_VALIDATION_REPORT.md
│   ├── PHASE_0_READINESS_REVIEW.md
│   ├── PHASE_1_IMPLEMENTATION_PLAN.md
│   └── reference/
│       └── real_time_fraud_detection_system_design.md
├── prompts/
├── agents/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── sample/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_debugging.ipynb
│   └── 03_model_analysis.ipynb
├── src/transaction_risk_engine/
│   ├── config.py
│   ├── data/
│   │   ├── load.py
│   │   ├── schema.py
│   │   └── split.py
│   ├── features/
│   │   ├── base.py
│   │   ├── frequency.py
│   │   ├── historical.py
│   │   ├── velocity.py
│   │   └── graph.py
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── threshold.py
│   │   └── inference.py
│   ├── explain/
│   │   └── shap_explainer.py
│   ├── monitoring/
│   │   ├── drift.py
│   │   └── champion_challenger.py
│   └── utils/
├── api/
│   ├── main.py
│   ├── schemas.py
│   └── routes.py
├── dashboard/
│   └── app.py
├── models/
├── reports/
├── tests/
└── scripts/
```

## Tool workflow recommendation

Use one main coding agent at a time. Do not let multiple agents edit the same files simultaneously.

Recommended workflow:

1. Main coding agent, currently Codex: repo-wide implementation and refactoring, one milestone at a time.
2. Optional secondary agent: isolated review, tests, debugging, or UI exploration only.
3. Antigravity: optional UI/browser/end-to-end testing and exploratory agent work.
4. You: architecture decisions, scope control, metric validation, final README/resume positioning.

The rule: agents write code; you approve architecture and numbers.
